//! `palito pull/list/rm` and `--model <name>` against a local fake Hugging
//! Face server (HEAD metadata on a redirect, CDN with Range, model API).
#![cfg(feature = "hub")]

use std::{
    fs,
    io::{BufRead, BufReader, Read, Write},
    net::{TcpListener, TcpStream},
    path::{Path, PathBuf},
    process::{Command, Output, Stdio},
    sync::{Arc, Mutex},
    thread,
};

const BODY_LEN: usize = 300_000;

fn body() -> Vec<u8> {
    let mut data = b"PALITO1\0".to_vec();
    data.extend((0..BODY_LEN - 8).map(|i| (i % 251) as u8));
    data
}

fn sha256(data: &[u8]) -> String {
    // Shell out to keep the test free of extra dev-dependencies.
    let dir = temp("sha");
    let path = dir.join("f");
    fs::write(&path, data).unwrap();
    let out = Command::new("sha256sum").arg(&path).output().unwrap();
    String::from_utf8(out.stdout).unwrap().split_whitespace().next().unwrap().to_owned()
}

fn temp(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("palito-pull-{tag}-{}-{:?}", std::process::id(), thread::current().id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    dir
}

/// Fake hub serving repo `u/r` with `m.palito`, `LICENSE` and a registry.
struct Server {
    base: String,
    ranges: Arc<Mutex<Vec<String>>>,
}

fn respond(stream: &mut TcpStream, status: &str, headers: &[(String, String)], body: &[u8], send_body: bool) {
    let mut head = format!("HTTP/1.1 {status}\r\nConnection: close\r\nContent-Length: {}\r\n", body.len());
    for (k, v) in headers {
        head.push_str(&format!("{k}: {v}\r\n"));
    }
    head.push_str("\r\n");
    let _ = stream.write_all(head.as_bytes());
    if send_body {
        let _ = stream.write_all(body);
    }
}

fn serve(registry_sha: String) -> Server {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let base = format!("http://{}", listener.local_addr().unwrap());
    let ranges = Arc::new(Mutex::new(Vec::new()));
    let seen = ranges.clone();
    let data = body();
    let sha = sha256(&data);
    let registry = format!(
        r#"{{"schema_version":1,"models":[{{"name":"tiny","aliases":["tiny-alias"],"description":"test","repo":"u/r","revision":"main","file":"m.palito","size":{BODY_LEN},"sha256":"{registry_sha}","extra_files":["LICENSE"],"license":"MIT","notice":"Built with Tests."}}]}}"#
    );
    thread::spawn(move || {
        for stream in listener.incoming() {
            let Ok(mut stream) = stream else { continue };
            let mut reader = BufReader::new(stream.try_clone().unwrap());
            let mut line = String::new();
            reader.read_line(&mut line).unwrap();
            let mut parts = line.split_whitespace();
            let (method, path) = (parts.next().unwrap_or("").to_owned(), parts.next().unwrap_or("").to_owned());
            let mut range = None;
            let mut bad_token = false;
            loop {
                let mut header = String::new();
                if reader.read_line(&mut header).unwrap() == 0 || header == "\r\n" {
                    break;
                }
                if header.to_ascii_lowercase().starts_with("authorization: bearer bad") {
                    bad_token = true;
                }
                if let Some(v) = header.to_ascii_lowercase().strip_prefix("range:") {
                    range = Some(v.trim().to_owned());
                }
            }
            let head = method == "HEAD";
            let send = |s: &mut TcpStream, status: &str, h: Vec<(String, String)>, b: &[u8]| {
                respond(s, status, &h, b, !head)
            };
            if bad_token {
                send(&mut stream, "401 Unauthorized", vec![], b"invalid token");
                continue;
            }
            match path.as_str() {
                "/u/r/resolve/main/m.palito" | "/u/r/resolve/abc123/m.palito" => send(&mut stream, "302 Found", vec![
                    ("Location".into(), "/cdn/m.palito".into()),
                    ("x-repo-commit".into(), "0123456789abcdef0123456789abcdef01234567".into()),
                    ("x-linked-size".into(), BODY_LEN.to_string()),
                    ("x-linked-etag".into(), format!("\"{sha}\"")),
                ], b""),
                "/u/r/resolve/main/LICENSE" => send(&mut stream, "200 OK", vec![], b"MIT license text"),
                "/cdn/m.palito" => {
                    seen.lock().unwrap().push(range.clone().unwrap_or_default());
                    match range.and_then(|r| r.strip_prefix("bytes=").and_then(|r| r.trim_end_matches('-').parse::<usize>().ok())) {
                        Some(start) => send(&mut stream, "206 Partial Content", vec![
                            ("Content-Range".into(), format!("bytes {start}-{}/{BODY_LEN}", BODY_LEN - 1)),
                        ], &data[start..]),
                        None => send(&mut stream, "200 OK", vec![], &data),
                    }
                }
                "/api/models/u/r/revision/main" => send(&mut stream, "200 OK", vec![],
                    br#"{"siblings":[{"rfilename":"README.md"},{"rfilename":"m.palito"}]}"#),
                "/registry.json" => send(&mut stream, "200 OK", vec![], registry.as_bytes()),
                _ => send(&mut stream, "404 Not Found", vec![], b"not found"),
            }
            let mut sink = [0u8; 1];
            let _ = stream.flush();
            let _ = stream.read(&mut sink[..0]);
        }
    });
    Server { base, ranges }
}

struct Env {
    models: PathBuf,
    cache: PathBuf,
    server: Server,
}

impl Env {
    fn new(tag: &str, registry_sha: Option<&str>) -> Self {
        let root = temp(tag);
        let sha = registry_sha.map_or_else(|| sha256(&body()), str::to_owned);
        Self { models: root.join("models"), cache: root.join("cache"), server: serve(sha) }
    }

    fn run(&self, args: &[&str]) -> Output {
        self.run_with_token(args, "")
    }

    fn run_with_token(&self, args: &[&str], token: &str) -> Output {
        Command::new(env!("CARGO_BIN_EXE_palito"))
            .args(args)
            .env("PALITO_MODELS", &self.models)
            .env("XDG_CACHE_HOME", &self.cache)
            .env("HF_ENDPOINT", &self.server.base)
            .env("PALITO_REGISTRY_URL", format!("{}/registry.json", self.server.base))
            .env("HF_TOKEN", token)
            .env_remove("PALITO_OFFLINE")
            .env_remove("HF_HUB_OFFLINE")
            .env_remove("PALITO_MODEL")
            .stdin(Stdio::null())
            .output()
            .unwrap()
    }

    fn model_path(&self) -> PathBuf {
        self.models.join("u").join("r").join("m.palito")
    }
}

fn text(bytes: &[u8]) -> String {
    String::from_utf8_lossy(bytes).into_owned()
}

#[test]
fn pull_by_name_downloads_verifies_and_lists() {
    let env = Env::new("name", None);
    let out = env.run(&["pull", "tiny"]);
    assert!(out.status.success(), "{}", text(&out.stderr));
    assert_eq!(text(&out.stdout).trim(), env.model_path().display().to_string());
    assert_eq!(fs::read(env.model_path()).unwrap(), body());
    let stderr = text(&out.stderr);
    assert!(stderr.contains("sha256") && stderr.contains("ok"), "{stderr}");
    assert!(stderr.contains("Built with Tests."), "notice printed: {stderr}");
    assert!(env.model_path().with_file_name("LICENSE").is_file(), "{stderr}");

    let again = env.run(&["pull", "tiny-alias"]);
    assert!(again.status.success());
    assert!(text(&again.stderr).contains("is up to date"));

    let list = env.run(&["list"]);
    assert!(text(&list.stdout).contains("tiny"), "{}", text(&list.stdout));
    let json: serde_json::Value = serde_json::from_slice(&env.run(&["list", "--json"]).stdout).unwrap();
    assert_eq!(json[0]["name"], "tiny");

    let available = env.run(&["list", "--available"]);
    assert!(text(&available.stdout).contains("[downloaded]"));
}

#[test]
fn pull_resumes_a_partial_download() {
    let env = Env::new("resume", None);
    let dir = env.model_path().parent().unwrap().to_owned();
    fs::create_dir_all(&dir).unwrap();
    fs::write(dir.join(".m.palito.part"), &body()[..100_000]).unwrap();
    let out = env.run(&["pull", "u/r"]);
    assert!(out.status.success(), "{}", text(&out.stderr));
    assert_eq!(fs::read(env.model_path()).unwrap(), body());
    assert_eq!(env.server.ranges.lock().unwrap().as_slice(), ["bytes=100000-"]);
    assert!(!dir.join(".m.palito.part").exists());
}

#[test]
fn checksum_mismatch_deletes_the_download_and_exits_4() {
    let env = Env::new("sha", Some(&"0".repeat(64)));
    let out = env.run(&["pull", "tiny"]);
    assert_eq!(out.status.code(), Some(4), "{}", text(&out.stderr));
    assert!(text(&out.stderr).contains("checksum mismatch"));
    assert!(!env.model_path().exists());
    assert!(!env.model_path().with_file_name(".m.palito.part").exists());
}

#[test]
fn unknown_models_and_repos_exit_3_with_hints() {
    let env = Env::new("missing", None);
    let out = env.run(&["pull", "tinyy"]);
    assert_eq!(out.status.code(), Some(3));
    assert!(text(&out.stderr).contains("did you mean 'tiny'"), "{}", text(&out.stderr));
    let out = env.run(&["pull", "u/nope:m.palito"]);
    assert_eq!(out.status.code(), Some(3), "{}", text(&out.stderr));
    assert!(text(&out.stderr).contains("not found on Hugging Face"));
}

#[test]
fn model_names_resolve_for_commands_and_rm_cleans_up() {
    let env = Env::new("resolve", None);
    let out = env.run(&["tokenize", "--model", "tiny", "--prompt", "hi"]);
    assert_eq!(out.status.code(), Some(3), "{}", text(&out.stderr));
    assert!(text(&out.stderr).contains("palito pull tiny"));
    // --pull downloads without asking, then the command reaches the model loader.
    let out = env.run(&["generate", "--model", "tiny", "--pull", "--prompt", "hi"]);
    assert!(env.model_path().is_file(), "{}", text(&out.stderr));

    let refuse = env.run(&["rm", "tiny"]);
    assert_eq!(refuse.status.code(), Some(1));
    assert!(text(&refuse.stderr).contains("--yes"));
    let out = env.run(&["rm", "tiny", "--yes"]);
    assert!(out.status.success(), "{}", text(&out.stderr));
    assert!(!env.models.join("u").exists());
    assert!(text(&env.run(&["list"]).stderr).contains("No models"));
}

#[test]
fn offline_mode_refuses_to_download() {
    let env = Env::new("offline", None);
    let out = Command::new(env!("CARGO_BIN_EXE_palito"))
        .args(["pull", "u/r"])
        .env("PALITO_MODELS", &env.models)
        .env("XDG_CACHE_HOME", &env.cache)
        .env("HF_ENDPOINT", &env.server.base)
        .env("PALITO_OFFLINE", "1")
        .output()
        .unwrap();
    assert_eq!(out.status.code(), Some(1));
    assert!(text(&out.stderr).contains("offline mode"));
    assert!(!Path::new(&env.models).join("u").exists());
}

#[test]
fn a_rejected_token_does_not_block_public_models() {
    let env = Env::new("token", None);
    let out = env.run_with_token(&["pull", "tiny"], "bad");
    assert!(out.status.success(), "{}", text(&out.stderr));
    assert_eq!(fs::read(env.model_path()).unwrap(), body());
}
