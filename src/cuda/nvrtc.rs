//! Finds the CUDA libraries before cudarc's lazy loader needs them.
//!
//! crates.io cannot carry NVRTC, so a Rust program built with the `cuda` feature
//! may only have it through an installed Palito bundle.
use std::env;
use std::ffi::OsStr;
use std::fs;
use std::path::PathBuf;

const CUDA: &str = "libcuda.so.1";
const NVRTC: &str = "libnvrtc.so.13";
const RELEASES_URL: &str = "https://github.com/natanloterio/palito/releases/latest";

/// NVRTC of the first `palito` command on `path` that belongs to a bundle.
fn bundled_nvrtc(path: Option<&OsStr>) -> Option<PathBuf> {
    env::split_paths(path?).find_map(|dir| {
        let command = fs::canonicalize(dir.join("palito")).ok()?;
        let library = command.parent()?.join("lib").join(NVRTC);
        library.is_file().then_some(library)
    })
}

fn missing_nvrtc_message(cause: &str) -> String {
    format!(
        "{NVRTC} not found ({cause}). Install the Palito bundle from {RELEASES_URL} and put its \
         `palito` command on PATH, or set LD_LIBRARY_PATH to CUDA 13 libraries"
    )
}

/// Reports missing libraries as errors, because cudarc's loader would panic.
/// A bundled NVRTC stays loaded so that cudarc's lookup by name resolves to it.
pub(crate) fn ensure_libraries() -> Result<(), String> {
    // These are fixed CUDA libraries, not user-supplied executable paths.
    unsafe { libloading::Library::new(CUDA) }
        .map_err(|e| format!("{CUDA}: {e}; install an NVIDIA driver with CUDA 13 support"))?;
    let Err(by_name) = (unsafe { libloading::Library::new(NVRTC) }) else {
        return Ok(());
    };
    let path = env::var_os("PATH");
    let bundled = bundled_nvrtc(path.as_deref())
        .ok_or_else(|| missing_nvrtc_message(&by_name.to_string()))?;
    // The path comes from the user's own PATH, the same trust as running `palito`.
    let library = unsafe { libloading::Library::new(&bundled) }
        .map_err(|e| format!("{}: {e}", bundled.display()))?;
    std::mem::forget(library);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::OsString;
    use std::fs;
    use std::os::unix::fs::symlink;
    use std::path::{Path, PathBuf};

    struct TempDir(PathBuf);

    impl TempDir {
        fn new(name: &str) -> Self {
            let path = std::env::temp_dir().join(format!("palito-nvrtc-{}-{name}", std::process::id()));
            let _ = fs::remove_dir_all(&path);
            fs::create_dir_all(&path).unwrap();
            Self(path)
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    fn bundle(root: &Path) -> PathBuf {
        let dir = root.join("palito-1.0.0");
        fs::create_dir_all(dir.join("lib")).unwrap();
        fs::write(dir.join("palito"), "").unwrap();
        fs::write(dir.join("lib").join(NVRTC), "").unwrap();
        dir
    }

    #[test]
    fn finds_nvrtc_through_the_palito_symlink_on_path() {
        let temp = TempDir::new("symlink");
        let dir = bundle(&temp.0);
        let bin = temp.0.join("bin");
        fs::create_dir(&bin).unwrap();
        symlink(dir.join("palito"), bin.join("palito")).unwrap();
        let found = bundled_nvrtc(Some(&std::env::join_paths([&bin]).unwrap())).unwrap();
        assert_eq!(found, fs::canonicalize(dir).unwrap().join("lib").join(NVRTC));
    }

    #[test]
    fn skips_path_entries_without_a_bundle() {
        let temp = TempDir::new("skip");
        let dir = bundle(&temp.0);
        let empty = temp.0.join("empty");
        fs::create_dir(&empty).unwrap();
        let path = std::env::join_paths([empty, dir.clone()]).unwrap();
        assert!(bundled_nvrtc(Some(&path)).unwrap().starts_with(fs::canonicalize(dir).unwrap()));
    }

    #[test]
    fn a_palito_without_nvrtc_is_not_a_bundle() {
        let temp = TempDir::new("no-lib");
        fs::write(temp.0.join("palito"), "").unwrap();
        assert_eq!(bundled_nvrtc(Some(&OsString::from(&temp.0))), None);
    }

    #[test]
    fn no_path_finds_nothing() {
        assert_eq!(bundled_nvrtc(None), None);
    }

    #[test]
    fn missing_nvrtc_error_says_how_to_install_it() {
        let message = missing_nvrtc_message("not found");
        assert!(message.contains("github.com/natanloterio/palito/releases"));
        assert!(message.contains("LD_LIBRARY_PATH"));
    }
}
