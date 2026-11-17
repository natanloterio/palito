use super::{CompileOptions, Result, compile_ptx_with_opts, failure};
use cudarc::nvrtc::Ptx;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    fs::{self, OpenOptions},
    io::{ErrorKind, Write},
    path::{Path, PathBuf},
    sync::atomic::{AtomicU64, Ordering},
};

static SERIAL: AtomicU64 = AtomicU64::new(0);

fn digest(bytes: impl AsRef<[u8]>) -> String {
    format!("{:x}", Sha256::digest(bytes.as_ref()))
}

fn key(source: &str, options: &CompileOptions, version: (i32, i32)) -> String {
    digest(format!(
        "rust-cuda-ptx-cache-v1\n{version:?}\n{options:?}\n{}\n{}",
        digest(source),
        digest(include_str!("../../Cargo.lock")),
    ))
}

#[derive(Serialize, Deserialize)]
struct Entry {
    key: String,
    checksum: String,
    ptx: String,
}

pub(super) fn compile(
    directory: Option<&Path>,
    source: &str,
    options: CompileOptions,
) -> Result<Ptx> {
    let Some(directory) = directory else {
        return compile_ptx_with_opts(source, options).map_err(failure);
    };
    let (mut major, mut minor) = (0i32, 0i32);
    // Both pointers address initialized, writable integers for the duration of the call.
    unsafe { cudarc::nvrtc::sys::nvrtcVersion(&mut major, &mut minor) }
        .result()
        .map_err(failure)?;
    let key = key(source, &options, (major, minor));
    cached(directory, &key, || {
        compile_ptx_with_opts(source, options)
            .map(|ptx| ptx.to_src())
            .map_err(failure)
    })
}

fn cached(directory: &Path, key: &str, compile: impl FnOnce() -> Result<String>) -> Result<Ptx> {
    let path = directory.join(format!("{key}.json"));
    let error =
        |e: &dyn std::fmt::Display| failure(format!("kernel cache {}: {e}", path.display()));
    match fs::read(&path) {
        Ok(bytes) => {
            let entry: Entry = serde_json::from_slice(&bytes).map_err(|e| error(&e))?;
            if entry.key != key
                || entry.ptx.is_empty()
                || entry.ptx.contains('\0')
                || entry.checksum != digest(&entry.ptx)
            {
                return Err(error(
                    &"invalid identity or checksum; remove this entry to rebuild",
                ));
            }
            return Ok(Ptx::from_src(entry.ptx));
        }
        Err(e) if e.kind() == ErrorKind::NotFound => {}
        Err(e) => return Err(error(&e)),
    }
    fs::create_dir_all(directory).map_err(|e| error(&e))?;
    let ptx = compile()?;
    if ptx.is_empty() || ptx.contains('\0') {
        return Err(error(&"compiler produced invalid PTX"));
    }
    let entry = Entry {
        key: key.to_owned(),
        checksum: digest(&ptx),
        ptx,
    };
    let bytes = serde_json::to_vec(&entry).map_err(|e| error(&e))?;
    let temporary_path = directory.join(format!(
        ".{key}.{}.{}.tmp",
        std::process::id(),
        SERIAL.fetch_add(1, Ordering::Relaxed),
    ));
    let mut options = OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    let mut file = options.open(&temporary_path).map_err(|e| error(&e))?;
    let temporary = Temporary(temporary_path);
    file.write_all(&bytes).map_err(|e| error(&e))?;
    file.sync_all().map_err(|e| error(&e))?;
    drop(file);
    fs::rename(&temporary.0, &path).map_err(|e| error(&e))?;
    Ok(Ptx::from_src(entry.ptx))
}

struct Temporary(PathBuf);

impl Drop for Temporary {
    fn drop(&mut self) {
        if let Err(e) = fs::remove_file(&self.0) {
            if e.kind() != ErrorKind::NotFound {
                eprintln!("CUDA kernel cache cleanup {}: {e}", self.0.display());
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_covers_sources_options_architecture_compiler_and_lockfile() {
        let options = CompileOptions {
            fmad: Some(false),
            options: vec!["--gpu-architecture=compute_120".into()],
            ..Default::default()
        };
        let original = key("kernel A", &options, (13, 0));
        assert_ne!(original, key("kernel B", &options, (13, 0)));
        assert_ne!(original, key("kernel A", &options, (13, 1)));
        let mut changed = options.clone();
        changed.fmad = Some(true);
        assert_ne!(original, key("kernel A", &changed, (13, 0)));
        changed = options.clone();
        changed.options = vec!["--gpu-architecture=compute_90".into()];
        assert_ne!(original, key("kernel A", &changed, (13, 0)));
    }

    #[test]
    fn cache_miss_hit_corruption_and_compile_failure_are_explicit() {
        let directory = std::env::temp_dir().join(format!(
            "rust-cuda-cache-{}-{}",
            std::process::id(),
            SERIAL.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&directory).unwrap();
        let key = digest("test");
        let path = directory.join(format!("{key}.json"));
        assert_eq!(
            cached(&directory, &key, || Ok("ptx".into()))
                .unwrap()
                .to_src(),
            "ptx"
        );
        assert_eq!(
            cached(&directory, &key, || panic!("cache hit recompiled"))
                .unwrap()
                .to_src(),
            "ptx"
        );
        let mut entry: Entry = serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
        entry.ptx.push_str("corrupt");
        fs::write(&path, serde_json::to_vec(&entry).unwrap()).unwrap();
        assert!(
            cached(&directory, &key, || panic!(
                "corruption silently recompiled"
            ))
            .is_err()
        );
        fs::remove_file(&path).unwrap();
        assert!(cached(&directory, &key, || Err(failure("compile failed"))).is_err());
        assert_eq!(fs::read_dir(&directory).unwrap().count(), 0);
        fs::remove_dir(&directory).unwrap();
    }
}
