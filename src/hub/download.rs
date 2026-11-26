//! HTTP side of `palito pull`: Hugging Face metadata, resumable downloads and
//! sha256 verification.
//!
//! The token (`HF_TOKEN`, else the file `hf auth login` writes) goes only to
//! the Hugging Face host; reqwest drops `Authorization` on the redirect to the
//! CDN. A token that Hugging Face rejects is retried once without it, so a
//! stale login never blocks a public model.

use super::{
    DEFAULT_ENDPOINT, exit,
    progress::Progress,
    store::{lock_name, partial_name},
};
use crate::cli_ui::{fail, human_size};
use palito::{Error, Result};
use reqwest::{
    StatusCode,
    blocking::{Client, RequestBuilder, Response},
    header,
    redirect::Policy,
};
use sha2::{Digest, Sha256};
use std::{
    fs::{self, File, OpenOptions},
    io::{Read, Write},
    path::{Path, PathBuf},
    time::Duration,
};

const CONNECT_TIMEOUT: Duration = Duration::from_secs(15);
const REGISTRY_TIMEOUT: Duration = Duration::from_secs(3);
const RETRIES: u32 = 3;
const CHUNK: usize = 1 << 20;
/// Free space kept on top of the download itself.
const DISK_MARGIN: u64 = 512 << 20;

/// `PALITO_OFFLINE=1` or `HF_HUB_OFFLINE=1`.
pub(crate) fn offline() -> bool {
    ["PALITO_OFFLINE", "HF_HUB_OFFLINE"]
        .iter()
        .any(|name| std::env::var(name).is_ok_and(|v| !v.is_empty() && v != "0"))
}

fn endpoint() -> String {
    std::env::var("HF_ENDPOINT")
        .ok()
        .filter(|v| !v.is_empty())
        .unwrap_or_else(|| DEFAULT_ENDPOINT.into())
        .trim_end_matches('/')
        .to_owned()
}

fn host_of(url: &str) -> &str {
    url.split("://").nth(1).unwrap_or(url).split('/').next().unwrap_or("")
}

fn token() -> Option<String> {
    if let Ok(token) = std::env::var("HF_TOKEN") {
        return Some(token).filter(|t| !t.trim().is_empty()).map(|t| t.trim().to_owned());
    }
    let home = std::env::var_os("HF_HOME")
        .filter(|v| !v.is_empty())
        .map(PathBuf::from)
        .or_else(|| std::env::home_dir().map(|h| h.join(".cache").join("huggingface")))?;
    let token = fs::read_to_string(home.join("token")).ok()?;
    Some(token.trim().to_owned()).filter(|t| !t.is_empty())
}

fn network_error(error: &reqwest::Error) -> Error {
    let mut detail = error.to_string();
    let mut source = std::error::Error::source(error);
    while let Some(inner) = source {
        detail = inner.to_string();
        source = inner.source();
    }
    let host = error.url().map_or_else(|| host_of(&endpoint()).to_owned(), |u| {
        u.host_str().unwrap_or_default().to_owned()
    });
    Error::Invalid(format!(
        "could not reach {host}: {detail}\n  \
         hint: check your connection or proxy (HTTPS_PROXY); downloaded models work offline, see 'palito list'"
    ))
}

fn ensure_online() -> Result<()> {
    if offline() {
        return Err(Error::Invalid(
            "offline mode is on (PALITO_OFFLINE or HF_HUB_OFFLINE), so nothing can be downloaded\n  \
             hint: unset it to download, or use a model from 'palito list'"
                .into(),
        ));
    }
    Ok(())
}

/// Small text file (the registry), with a short timeout.
pub(crate) fn fetch_text(url: &str) -> Result<String> {
    let client = Client::builder()
        .timeout(REGISTRY_TIMEOUT)
        .user_agent(user_agent())
        .build()
        .map_err(|e| network_error(&e))?;
    let response = client.get(url).send().map_err(|e| network_error(&e))?;
    if !response.status().is_success() {
        return Err(Error::Invalid(format!("{url}: HTTP {}", response.status())));
    }
    response.text().map_err(|e| network_error(&e))
}

fn user_agent() -> String {
    format!("palito/{}", env!("CARGO_PKG_VERSION"))
}

/// Hugging Face calls with optional auth and the stale-token fallback.
pub(crate) struct Hub {
    endpoint: String,
    token: Option<String>,
    /// Follows redirects (downloads).
    client: Client,
    /// Stops at the first response (HEAD metadata lives on the redirect).
    no_redirect: Client,
}

/// A file on Hugging Face, from a HEAD on its resolve URL.
#[derive(Debug)]
pub(crate) struct Remote {
    pub(crate) url: String,
    pub(crate) commit: Option<String>,
    pub(crate) size: Option<u64>,
    /// LFS files carry their sha256 as the linked ETag.
    pub(crate) sha256: Option<String>,
}

impl Hub {
    pub(crate) fn new() -> Result<Self> {
        ensure_online()?;
        let build = |policy| {
            Client::builder()
                .connect_timeout(CONNECT_TIMEOUT)
                .timeout(None)
                .redirect(policy)
                .user_agent(user_agent())
                .build()
                .map_err(|e| network_error(&e))
        };
        Ok(Self {
            endpoint: endpoint(),
            token: token(),
            client: build(Policy::limited(10))?,
            no_redirect: build(Policy::none())?,
        })
    }

    fn send(&self, make: impl Fn(Option<&str>) -> RequestBuilder) -> Result<Response> {
        let token = self.token.as_deref();
        let response = make(token).send().map_err(|e| network_error(&e))?;
        // A rejected token must not hide a public file.
        if token.is_some() && response.status() == StatusCode::UNAUTHORIZED {
            return make(None).send().map_err(|e| network_error(&e));
        }
        Ok(response)
    }

    fn authorized(&self, request: RequestBuilder, token: Option<&str>) -> RequestBuilder {
        match token {
            Some(token) => request.bearer_auth(token),
            None => request,
        }
    }

    fn check_status(&self, response: &Response, what: &str, repo: &str) -> Result<()> {
        let status = response.status();
        if status.is_success() || status.is_redirection() {
            return Ok(());
        }
        let page = format!("{}/{repo}", self.endpoint);
        Err(match status {
            StatusCode::NOT_FOUND | StatusCode::UNAUTHORIZED => fail(
                exit::NOT_FOUND,
                format!(
                    "{what} not found on Hugging Face (or it is private or gated)\n  \
                     hint: names are case-sensitive: {page}\n  \
                     hint: for gated repos, accept the terms on the website and log in with 'hf auth login' (or set HF_TOKEN)"
                ),
            ),
            StatusCode::FORBIDDEN => fail(
                exit::AUTH,
                format!(
                    "access denied to {what}\n  hint: accept the model's terms at {page}, then try again"
                ),
            ),
            _ => Error::Invalid(format!(
                "Hugging Face answered HTTP {status} for {what}\n  hint: try again in a moment"
            )),
        })
    }

    /// Files of a repo at a revision.
    pub(crate) fn list_files(&self, repo: &str, revision: &str) -> Result<Vec<String>> {
        let url = format!("{}/api/models/{repo}/revision/{revision}", self.endpoint);
        let response = self.send(|t| self.authorized(self.client.get(&url), t))?;
        self.check_status(&response, &format!("repo {repo}"), repo)?;
        let body: serde_json::Value = response.json().map_err(|e| network_error(&e))?;
        Ok(body["siblings"]
            .as_array()
            .map(|files| {
                files.iter().filter_map(|f| f["rfilename"].as_str().map(str::to_owned)).collect()
            })
            .unwrap_or_default())
    }

    pub(crate) fn head(&self, repo: &str, revision: &str, file: &str) -> Result<Remote> {
        let url = format!("{}/{repo}/resolve/{revision}/{file}", self.endpoint);
        let response = self.send(|t| self.authorized(self.no_redirect.head(&url), t))?;
        self.check_status(&response, &format!("{repo}:{file}"), repo)?;
        let headers = response.headers();
        let text = |name: &str| {
            headers.get(name).and_then(|v| v.to_str().ok()).map(|v| v.trim_matches('"').to_owned())
        };
        let size = text("x-linked-size")
            .or_else(|| (!response.status().is_redirection()).then(|| text("content-length")).flatten())
            .and_then(|v| v.parse().ok());
        let sha256 = text("x-linked-etag")
            .filter(|v| v.len() == 64 && v.bytes().all(|b| b.is_ascii_hexdigit()))
            .map(|v| v.to_ascii_lowercase());
        Ok(Remote { url, commit: text("x-repo-commit"), size, sha256 })
    }

    /// Downloads `remote` into `dir/file`, resuming a `.part` left by an
    /// earlier run. Returns the file's sha256.
    pub(crate) fn download(
        &self,
        remote: &Remote,
        dir: &Path,
        file: &str,
        expected_sha: Option<&str>,
        progress: &mut Progress,
    ) -> Result<String> {
        fs::create_dir_all(dir)?;
        let _lock = Lock::acquire(&dir.join(lock_name(file)), file)?;
        let part = dir.join(partial_name(file));
        let mut have = fs::metadata(&part).map_or(0, |m| m.len());
        if remote.size.is_some_and(|size| have > size) {
            fs::remove_file(&part)?;
            have = 0;
        }
        if let Some(size) = remote.size {
            check_space(dir, size - have)?;
        }
        let mut hasher = Sha256::new();
        if have > 0 {
            progress.message(&format!("Checking partial download ({})...", human_size(have)));
            hash_into(&part, &mut hasher)?;
        }
        progress.start(remote.size, have);
        let mut attempt = 0;
        loop {
            match self.fetch_into(remote, &part, &mut have, &mut hasher, progress) {
                Ok(()) => break,
                Err(Fetch::Fatal(error)) => return Err(error),
                Err(Fetch::Retry(error)) if attempt >= RETRIES => {
                    progress.abandon();
                    let total = remote.size.map_or(String::new(), |s| format!(" of {}", human_size(s)));
                    return Err(Error::Invalid(format!(
                        "download of {file} stopped at {}{total}: {error}\n  \
                         hint: run the same command again to resume from {}",
                        human_size(have),
                        human_size(have)
                    )));
                }
                Err(Fetch::Retry(error)) => {
                    attempt += 1;
                    let wait = Duration::from_secs(1 << (attempt - 1));
                    progress.suspend(|| {
                        crate::cli_ui::warn(&format!(
                            "{error} at {}, retrying ({attempt}/{RETRIES}) in {}s...",
                            human_size(have),
                            wait.as_secs()
                        ))
                    });
                    std::thread::sleep(wait);
                }
            }
        }
        progress.finish();
        let digest = hex(&hasher.finalize());
        let wanted = expected_sha.or(remote.sha256.as_deref());
        if let Some(wanted) = wanted.filter(|w| !w.eq_ignore_ascii_case(&digest)) {
            let _ = fs::remove_file(&part);
            return Err(fail(
                exit::CHECKSUM,
                format!(
                    "checksum mismatch for {file}; the download was deleted\n  \
                     expected sha256 {wanted}\n  \
                     got      sha256 {digest}\n  \
                     hint: run the same command again; if it fails again, report it at https://github.com/natanloterio/palito/issues"
                ),
            ));
        }
        // Windows flushes only handles opened for writing (FlushFileBuffers).
        OpenOptions::new().write(true).open(&part)?.sync_all()?;
        fs::rename(&part, dir.join(file))?;
        Ok(digest)
    }

    fn fetch_into(
        &self,
        remote: &Remote,
        part: &Path,
        have: &mut u64,
        hasher: &mut Sha256,
        progress: &mut Progress,
    ) -> std::result::Result<(), Fetch> {
        if remote.size.is_some_and(|size| *have == size) && part.exists() {
            return Ok(());
        }
        let start = *have;
        let response = self
            .send(|t| {
                let request = self.authorized(self.client.get(&remote.url), t);
                if start > 0 { request.header(header::RANGE, format!("bytes={start}-")) } else { request }
            })
            .map_err(Fetch::Retry)?;
        let status = response.status();
        if status.is_server_error() {
            return Err(Fetch::Retry(Error::Invalid(format!("HTTP {status}"))));
        }
        if !status.is_success() {
            return Err(Fetch::Fatal(Error::Invalid(format!(
                "Hugging Face answered HTTP {status} while downloading"
            ))));
        }
        let mut out = OpenOptions::new().create(true).append(true).open(part).map_err(io_fatal)?;
        if start > 0 && status != StatusCode::PARTIAL_CONTENT {
            // The server ignored Range: start over.
            out.set_len(0).map_err(io_fatal)?;
            *have = 0;
            *hasher = Sha256::new();
            progress.start(remote.size, 0);
        }
        let mut reader = response;
        let mut buffer = vec![0u8; CHUNK];
        loop {
            let read = match reader.read(&mut buffer) {
                Ok(0) => break,
                Ok(n) => n,
                Err(error) => return Err(Fetch::Retry(Error::Invalid(format!("connection lost ({error})")))),
            };
            out.write_all(&buffer[..read]).map_err(io_fatal)?;
            hasher.update(&buffer[..read]);
            *have += read as u64;
            progress.set(*have);
        }
        match remote.size {
            Some(size) if *have < size => Err(Fetch::Retry(Error::Invalid("connection closed early".into()))),
            _ => Ok(()),
        }
    }
}

enum Fetch {
    Retry(Error),
    Fatal(Error),
}

fn io_fatal(error: std::io::Error) -> Fetch {
    Fetch::Fatal(error.into())
}

fn hash_into(path: &Path, hasher: &mut Sha256) -> Result<()> {
    let mut file = File::open(path)?;
    let mut buffer = vec![0u8; CHUNK];
    loop {
        let read = file.read(&mut buffer)?;
        if read == 0 {
            return Ok(());
        }
        hasher.update(&buffer[..read]);
    }
}

/// sha256 of a file on disk (`pull --verify`).
pub(crate) fn sha256_file(path: &Path) -> Result<String> {
    let mut hasher = Sha256::new();
    hash_into(path, &mut hasher)?;
    Ok(hex(&hasher.finalize()))
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// Bytes needed on top of the download, as the free-space rule.
pub(crate) fn space_needed(remaining: u64) -> u64 {
    remaining + DISK_MARGIN
}

fn check_space(dir: &Path, remaining: u64) -> Result<()> {
    let Ok(free) = fs4::available_space(dir) else { return Ok(()) };
    if free >= space_needed(remaining) {
        return Ok(());
    }
    Err(fail(
        exit::DISK_SPACE,
        format!(
            "not enough space in {}: {} needed, {} free\n  \
             hint: free up space, or download to another disk: PALITO_MODELS=/mnt/big/palito palito pull ...",
            dir.display(),
            human_size(remaining),
            human_size(free)
        ),
    ))
}

/// One download per file at a time; a lock left by a dead process is taken over.
struct Lock(PathBuf);

impl Lock {
    fn acquire(path: &Path, file: &str) -> Result<Self> {
        for _ in 0..2 {
            match OpenOptions::new().write(true).create_new(true).open(path) {
                Ok(mut handle) => {
                    write!(handle, "{}", std::process::id())?;
                    return Ok(Self(path.to_owned()));
                }
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
                    let pid = fs::read_to_string(path).ok().and_then(|p| p.trim().parse::<u32>().ok());
                    if pid.is_some_and(process_alive) {
                        return Err(Error::Invalid(format!(
                            "another palito (pid {}) is already downloading {file}\n  \
                             hint: wait for it to finish, or stop it and run this again",
                            pid.unwrap_or_default()
                        )));
                    }
                    fs::remove_file(path)?;
                }
                Err(error) => return Err(error.into()),
            }
        }
        Err(Error::Invalid(format!("could not lock {}", path.display())))
    }
}

impl Drop for Lock {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.0);
    }
}

#[cfg(target_os = "linux")]
fn process_alive(pid: u32) -> bool {
    Path::new(&format!("/proc/{pid}")).exists()
}

/// Without /proc a lock is trusted; a stale one is removed by hand.
#[cfg(not(target_os = "linux"))]
fn process_alive(_pid: u32) -> bool {
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hosts_and_space_rule() {
        assert_eq!(host_of("https://huggingface.co/a/b"), "huggingface.co");
        assert_eq!(host_of("http://127.0.0.1:9/x"), "127.0.0.1:9");
        assert_eq!(space_needed(1 << 30), (1 << 30) + (512 << 20));
    }

    #[test]
    fn lock_is_exclusive_and_released() {
        let dir = super::super::store::tests::temp_dir("lock");
        let path = dir.join(".m.lock");
        let lock = Lock::acquire(&path, "m").unwrap();
        assert!(Lock::acquire(&path, "m").is_err());
        drop(lock);
        assert!(!path.exists());
        fs::write(&path, "999999999").unwrap(); // dead pid
        let _taken = Lock::acquire(&path, "m").unwrap();
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn hashes_files() {
        let dir = super::super::store::tests::temp_dir("hash");
        let path = dir.join("f");
        fs::write(&path, b"abc").unwrap();
        assert_eq!(
            sha256_file(&path).unwrap(),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        fs::remove_dir_all(dir).unwrap();
    }
}
