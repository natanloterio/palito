//! Downloaded models on disk.
//!
//! ```text
//! $PALITO_MODELS (default ~/.local/share/palito/models)/
//!   loterio/Qwen3-8B-palito/
//!     qwen3-8b.palito
//!     qwen3-8b.palito.json   manifest, written last: its presence marks a complete download
//!     LICENSE  NOTICE
//!     .qwen3-8b.palito.part  only while downloading
//! ```

use super::{ModelRef, registry::Entry};
use palito::Result;
use serde::{Deserialize, Serialize};
use std::{
    fs,
    path::{Path, PathBuf},
};

const MANIFEST_SUFFIX: &str = ".json";

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
pub(crate) struct Manifest {
    /// Registry name, or the file stem for repos outside the registry.
    pub(crate) name: String,
    #[serde(default)]
    pub(crate) aliases: Vec<String>,
    pub(crate) repo: String,
    pub(crate) revision: String,
    pub(crate) file: String,
    pub(crate) size: u64,
    #[serde(default)]
    pub(crate) sha256: Option<String>,
    #[serde(default)]
    pub(crate) license: String,
    #[serde(default)]
    pub(crate) notice: Option<String>,
    #[serde(default)]
    pub(crate) pulled_at: u64,
}

impl Manifest {
    pub(crate) fn from_entry(entry: &Entry) -> Self {
        Self {
            name: entry.name.clone(),
            aliases: entry.aliases.clone(),
            repo: entry.repo.clone(),
            revision: entry.revision.clone(),
            file: entry.file.clone(),
            size: entry.size,
            sha256: Some(entry.sha256.clone()),
            license: entry.license.clone(),
            notice: entry.notice.clone(),
            pulled_at: 0,
        }
    }

    fn matches(&self, model: &ModelRef) -> bool {
        match model {
            ModelRef::Name(name) => {
                self.name == *name || self.aliases.contains(name) || stem(&self.file) == name
            }
            ModelRef::Repo { repo, file } => {
                self.repo == *repo && file.as_ref().is_none_or(|f| *f == self.file)
            }
        }
    }
}

fn stem(file: &str) -> &str {
    file.rsplit_once('.').map_or(file, |(stem, _)| stem)
}

/// A downloaded model: its manifest and the model file.
#[derive(Clone, Debug)]
pub(crate) struct Installed {
    pub(crate) manifest: Manifest,
    pub(crate) path: PathBuf,
}

#[derive(Clone, Debug)]
pub(crate) struct Store {
    root: PathBuf,
}

impl Store {
    /// `$PALITO_MODELS`, else `$XDG_DATA_HOME/palito/models`, else
    /// `~/.local/share/palito/models` (next to the installed versions).
    pub(crate) fn from_env() -> Self {
        let env_dir = |name| std::env::var_os(name).filter(|v| !v.is_empty()).map(PathBuf::from);
        let root = env_dir("PALITO_MODELS").unwrap_or_else(|| {
            env_dir("XDG_DATA_HOME")
                .or_else(|| std::env::home_dir().map(|home| home.join(".local").join("share")))
                .unwrap_or_else(|| PathBuf::from("."))
                .join("palito")
                .join("models")
        });
        Self { root }
    }

    #[cfg(test)]
    pub(crate) fn at(root: &Path) -> Self {
        Self { root: root.to_owned() }
    }

    pub(crate) fn root(&self) -> &Path {
        &self.root
    }

    /// Folder of one repo; `repo` was validated by `ModelRef`/`Registry`.
    pub(crate) fn repo_dir(&self, repo: &str) -> PathBuf {
        repo.split('/').fold(self.root.clone(), |dir, part| dir.join(part))
    }

    pub(crate) fn manifest_path(&self, repo: &str, file: &str) -> PathBuf {
        self.repo_dir(repo).join(format!("{file}{MANIFEST_SUFFIX}"))
    }

    /// Every complete download, sorted by name.
    pub(crate) fn installed(&self) -> Vec<Installed> {
        let mut found = Vec::new();
        for user in read_dirs(&self.root) {
            for repo in read_dirs(&user) {
                let Ok(entries) = fs::read_dir(&repo) else { continue };
                for entry in entries.flatten() {
                    if let Some(installed) = load(&entry.path()) {
                        found.push(installed);
                    }
                }
            }
        }
        found.sort_by(|a, b| a.manifest.name.cmp(&b.manifest.name));
        found
    }

    /// Downloaded models matching `model` (several only for `user/repo`
    /// without a file when the repo has more than one download).
    pub(crate) fn find(&self, model: &ModelRef) -> Vec<Installed> {
        self.installed().into_iter().filter(|i| i.manifest.matches(model)).collect()
    }

    pub(crate) fn write_manifest(&self, manifest: &Manifest) -> Result<()> {
        let path = self.manifest_path(&manifest.repo, &manifest.file);
        let temp = path.with_extension("json.tmp");
        fs::write(&temp, serde_json::to_vec_pretty(manifest)?)?;
        fs::rename(&temp, &path)?;
        Ok(())
    }

    /// Deletes the model, its manifest and leftovers; the repo folder goes
    /// too when nothing else was downloaded from it. Returns bytes freed.
    pub(crate) fn remove(&self, installed: &Installed) -> Result<u64> {
        let manifest = &installed.manifest;
        let dir = self.repo_dir(&manifest.repo);
        let size = fs::metadata(&installed.path).map_or(0, |m| m.len());
        fs::remove_file(self.manifest_path(&manifest.repo, &manifest.file))?;
        for path in [
            installed.path.clone(),
            dir.join(partial_name(&manifest.file)),
            dir.join(lock_name(&manifest.file)),
        ] {
            match fs::remove_file(&path) {
                Ok(()) => {}
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
                Err(error) => return Err(error.into()),
            }
        }
        let others = self.installed().iter().any(|i| i.manifest.repo == manifest.repo);
        if !others {
            fs::remove_dir_all(&dir)?;
            if let Some(user) = dir.parent() {
                let _ = fs::remove_dir(user); // only succeeds when empty
            }
        }
        Ok(size)
    }
}

pub(crate) fn partial_name(file: &str) -> String {
    format!(".{file}.part")
}

pub(crate) fn lock_name(file: &str) -> String {
    format!(".{file}.lock")
}

fn read_dirs(dir: &Path) -> Vec<PathBuf> {
    let Ok(entries) = fs::read_dir(dir) else { return Vec::new() };
    let mut dirs: Vec<PathBuf> =
        entries.flatten().map(|e| e.path()).filter(|p| p.is_dir()).collect();
    dirs.sort();
    dirs
}

fn load(manifest_path: &Path) -> Option<Installed> {
    let name = manifest_path.file_name()?.to_str()?;
    let file = name.strip_suffix(MANIFEST_SUFFIX)?;
    if file.starts_with('.') {
        return None;
    }
    let manifest: Manifest = serde_json::from_slice(&fs::read(manifest_path).ok()?).ok()?;
    let path = manifest_path.with_file_name(file);
    (manifest.file == file && path.is_file()).then_some(Installed { manifest, path })
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    pub(crate) fn temp_dir(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "palito-hub-{tag}-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    pub(crate) fn manifest(name: &str, repo: &str) -> Manifest {
        Manifest {
            name: name.into(),
            aliases: vec![format!("alt-{name}")],
            repo: repo.into(),
            revision: "main".into(),
            file: format!("{name}.palito"),
            size: 5,
            sha256: None,
            license: "MIT".into(),
            notice: None,
            pulled_at: 0,
        }
    }

    fn install(store: &Store, manifest: &Manifest) {
        let dir = store.repo_dir(&manifest.repo);
        fs::create_dir_all(&dir).unwrap();
        fs::write(dir.join(&manifest.file), b"PLT1x").unwrap();
        store.write_manifest(manifest).unwrap();
    }

    #[test]
    fn finds_downloads_by_name_alias_stem_and_repo() {
        let root = temp_dir("find");
        let store = Store::at(&root);
        install(&store, &manifest("qwen3-8b", "u/qwen"));
        install(&store, &manifest("llama", "u/llama"));
        let names = |m: &str| -> Vec<String> {
            store.find(&ModelRef::parse(m).unwrap()).into_iter().map(|i| i.manifest.name).collect()
        };
        assert_eq!(names("qwen3-8b"), ["qwen3-8b"]);
        assert_eq!(names("alt-llama"), ["llama"]);
        assert_eq!(names("u/qwen"), ["qwen3-8b"]);
        assert_eq!(names("u/qwen:qwen3-8b.palito"), ["qwen3-8b"]);
        assert!(names("u/qwen:other.palito").is_empty());
        assert!(names("missing").is_empty());
        assert_eq!(store.installed().len(), 2);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn incomplete_downloads_are_invisible() {
        let root = temp_dir("partial");
        let store = Store::at(&root);
        let dir = store.repo_dir("u/r");
        fs::create_dir_all(&dir).unwrap();
        fs::write(dir.join(partial_name("m.palito")), b"half").unwrap();
        fs::write(dir.join("m.palito"), b"no manifest yet").unwrap();
        assert!(store.installed().is_empty());
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn remove_deletes_files_and_empty_folders() {
        let root = temp_dir("remove");
        let store = Store::at(&root);
        install(&store, &manifest("a", "u/r"));
        install(&store, &manifest("b", "u/r"));
        let a = store.find(&ModelRef::parse("a").unwrap()).remove(0);
        assert_eq!(store.remove(&a).unwrap(), 5);
        assert!(store.repo_dir("u/r").is_dir(), "b still lives there");
        let b = store.find(&ModelRef::parse("b").unwrap()).remove(0);
        store.remove(&b).unwrap();
        assert!(!root.join("u").exists());
        fs::remove_dir_all(root).unwrap();
    }
}
