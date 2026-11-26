//! Short model names (`qwen3-8b`) mapped to pinned Hugging Face files.
//!
//! The source of truth is `registry/models.json` in the repository. The binary
//! embeds a copy, so names work offline; `pull` and `list --available` refresh
//! it from GitHub (cached for a day) so new models need no release.

use super::valid_segment;
use palito::{Error, Result};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

pub(crate) const SCHEMA_VERSION: u32 = 1;
const EMBEDDED: &str = include_str!("../../registry/models.json");
#[cfg(feature = "hub")]
const DEFAULT_URL: &str =
    "https://raw.githubusercontent.com/natanloterio/palito/master/registry/models.json";
#[cfg(feature = "hub")]
const CACHE_MAX_AGE: std::time::Duration = std::time::Duration::from_secs(24 * 3600);

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
pub(crate) struct Entry {
    pub(crate) name: String,
    #[serde(default)]
    pub(crate) aliases: Vec<String>,
    #[serde(default)]
    pub(crate) description: String,
    pub(crate) repo: String,
    pub(crate) revision: String,
    pub(crate) file: String,
    pub(crate) size: u64,
    pub(crate) sha256: String,
    #[serde(default)]
    pub(crate) extra_files: Vec<String>,
    #[serde(default)]
    pub(crate) license: String,
    #[serde(default)]
    pub(crate) notice: Option<String>,
    #[serde(default)]
    pub(crate) min_palito: Option<String>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct Registry {
    pub(crate) schema_version: u32,
    pub(crate) models: Vec<Entry>,
}

impl Entry {
    pub(crate) fn matches(&self, name: &str) -> bool {
        self.name == name || self.aliases.iter().any(|a| a == name)
    }

    /// Whether this palito build is new enough for the file.
    pub(crate) fn supported(&self) -> bool {
        self.min_palito
            .as_deref()
            .is_none_or(|min| version_tuple(env!("CARGO_PKG_VERSION")) >= version_tuple(min))
    }

    fn validate(&self) -> bool {
        let repo_ok = self.repo.split('/').count() == 2 && self.repo.split('/').all(valid_segment);
        let hex_ok = self.sha256.len() == 64 && self.sha256.bytes().all(|b| b.is_ascii_hexdigit());
        repo_ok
            && hex_ok
            && valid_segment(&self.name)
            && valid_segment(&self.file)
            && valid_segment(&self.revision)
            && self.aliases.iter().all(|a| valid_segment(a))
            && self.extra_files.iter().all(|f| valid_segment(f))
    }
}

fn version_tuple(text: &str) -> (u64, u64, u64) {
    let mut parts = text.split(['.', '-']).map(|p| p.parse::<u64>().unwrap_or(0));
    (
        parts.next().unwrap_or(0),
        parts.next().unwrap_or(0),
        parts.next().unwrap_or(0),
    )
}

impl Registry {
    /// Parses a registry, dropping entries with unsafe names or hashes.
    pub(crate) fn parse(text: &str) -> Result<Self> {
        let mut registry: Self = serde_json::from_str(text)
            .map_err(|error| Error::Invalid(format!("model registry is not valid JSON: {error}")))?;
        if registry.schema_version > SCHEMA_VERSION {
            return Err(Error::Invalid(format!(
                "model registry schema {} is newer than this palito supports ({SCHEMA_VERSION})",
                registry.schema_version
            )));
        }
        registry.models.retain(Entry::validate);
        Ok(registry)
    }

    pub(crate) fn embedded() -> Self {
        Self::parse(EMBEDDED).expect("registry/models.json is valid")
    }

    /// Cached registry if it parses, else the embedded one. Never touches the network.
    pub(crate) fn local() -> Self {
        cache_path()
            .and_then(|path| std::fs::read_to_string(path).ok())
            .and_then(|text| Self::parse(&text).ok())
            .unwrap_or_else(Self::embedded)
    }

    pub(crate) fn find(&self, name: &str) -> Option<&Entry> {
        self.models.iter().find(|entry| entry.matches(name))
    }

    /// The closest known name, for "did you mean".
    pub(crate) fn suggest(&self, name: &str) -> Option<&str> {
        self.models
            .iter()
            .flat_map(|e| std::iter::once(&e.name).chain(&e.aliases))
            .map(|known| (crate::cli_ui::edit_distance(name, known), known))
            .filter(|(distance, known)| *distance <= 3.max(known.len() / 3))
            .min_by_key(|(distance, _)| *distance)
            .map(|(_, known)| known.as_str())
    }
}

/// `$XDG_CACHE_HOME/palito/registry.json`.
fn cache_path() -> Option<PathBuf> {
    let base = std::env::var_os("XDG_CACHE_HOME")
        .filter(|v| !v.is_empty())
        .map(PathBuf::from)
        .or_else(|| std::env::home_dir().map(|home| home.join(".cache")))?;
    Some(base.join("palito").join("registry.json"))
}

/// Fresh cache, else a network fetch (written to the cache), else the local copy.
/// Network problems never fail the command: the registry is a convenience.
#[cfg(feature = "hub")]
pub(crate) fn refreshed(force: bool) -> Registry {
    let path = cache_path();
    let fresh = path
        .as_ref()
        .and_then(|p| std::fs::metadata(p).ok())
        .and_then(|m| m.modified().ok())
        .and_then(|t| t.elapsed().ok())
        .is_some_and(|age| age < CACHE_MAX_AGE);
    if (fresh && !force) || super::download::offline() {
        return Registry::local();
    }
    let url = std::env::var("PALITO_REGISTRY_URL").unwrap_or_else(|_| DEFAULT_URL.into());
    match super::download::fetch_text(&url) {
        Ok(text) if Registry::parse(&text).is_ok() => {
            if let Some(path) = path {
                let _ = std::fs::create_dir_all(path.parent().unwrap_or(&path));
                let _ = std::fs::write(&path, &text);
            }
            Registry::parse(&text).unwrap_or_else(|_| Registry::local())
        }
        _ => Registry::local(),
    }
}

#[cfg(not(feature = "hub"))]
pub(crate) fn refreshed(_force: bool) -> Registry {
    Registry::local()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entry(name: &str) -> String {
        format!(
            r#"{{"name":"{name}","aliases":["alt-{name}"],"repo":"u/r","revision":"main","file":"{name}.palito","size":1,"sha256":"{}"}}"#,
            "a".repeat(64)
        )
    }

    #[test]
    fn embedded_registry_is_valid_and_complete() {
        let registry = Registry::embedded();
        assert!(registry.find("qwen3-8b").is_some());
        assert!(registry.find("llama-3.1-8b").is_some());
        let raw: serde_json::Value = serde_json::from_str(EMBEDDED).unwrap();
        assert_eq!(raw["models"].as_array().unwrap().len(), registry.models.len(), "an entry failed validation");
    }

    #[test]
    fn finds_by_alias_and_suggests_typos() {
        let text = format!(r#"{{"schema_version":1,"models":[{}]}}"#, entry("qwen3-8b"));
        let registry = Registry::parse(&text).unwrap();
        assert_eq!(registry.find("alt-qwen3-8b").unwrap().name, "qwen3-8b");
        assert_eq!(registry.suggest("qwen-8b"), Some("qwen3-8b"));
        assert_eq!(registry.suggest("totally-different-thing"), None);
    }

    #[test]
    fn drops_unsafe_entries_and_rejects_future_schemas() {
        let bad = entry("ok").replace("u/r", "../etc");
        let text = format!(r#"{{"schema_version":1,"models":[{bad},{}]}}"#, entry("good"));
        let registry = Registry::parse(&text).unwrap();
        assert_eq!(registry.models.len(), 1);
        assert!(Registry::parse(r#"{"schema_version":99,"models":[]}"#).is_err());
        assert!(Registry::parse("not json").is_err());
    }

    #[test]
    fn min_version_compares_numerically() {
        assert!(version_tuple("0.15.0") > version_tuple("0.9.9"));
        assert!(version_tuple("1.0.0-rc1") >= version_tuple("1.0.0"));
    }
}
