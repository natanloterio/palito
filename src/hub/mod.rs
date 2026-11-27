//! Ready-made models from Hugging Face: `palito pull`, `palito list`,
//! `palito rm`, and resolving `--model qwen3-8b` to a downloaded file.
//!
//! Layout: a registry of short names (compiled in, refreshed from GitHub),
//! a local store of downloaded files with one JSON manifest each, and an HTTP
//! downloader with resume and sha256 verification (feature `hub`).

pub(crate) mod commands;
#[cfg(feature = "hub")]
mod download;
#[cfg(feature = "hub")]
mod progress;
pub(crate) mod registry;
pub(crate) mod select;
pub(crate) mod store;

use palito::Error;
use std::fmt;

/// Exit codes for model commands (see `cli_ui::fail`).
pub(crate) mod exit {
    pub(crate) const NOT_FOUND: i32 = 3;
    pub(crate) const CHECKSUM: i32 = 4;
    pub(crate) const AUTH: i32 = 5;
    pub(crate) const DISK_SPACE: i32 = 6;
}

/// Default Hugging Face endpoint; `HF_ENDPOINT` overrides it.
pub(crate) const DEFAULT_ENDPOINT: &str = "https://huggingface.co";

/// What the user typed after `pull`, `rm` or `--model`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum ModelRef {
    /// A registry name or alias, e.g. `qwen3-8b`.
    Name(String),
    /// `user/repo`, optionally with `:file`.
    Repo { repo: String, file: Option<String> },
}

impl fmt::Display for ModelRef {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Name(name) => f.write_str(name),
            Self::Repo { repo, file: None } => f.write_str(repo),
            Self::Repo { repo, file: Some(file) } => write!(f, "{repo}:{file}"),
        }
    }
}

/// One path segment that is safe to use as a directory or file name.
pub(crate) fn valid_segment(text: &str) -> bool {
    let mut chars = text.chars();
    let first_ok = chars.next().is_some_and(|c| c.is_ascii_alphanumeric());
    first_ok
        && text.len() <= 128
        && text.chars().all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-'))
        && !text.contains("..")
}

fn invalid(text: &str) -> Error {
    Error::Invalid(format!(
        "'{text}' is not a model name or Hugging Face repo\n  \
         hint: use a name from 'palito list --available', user/repo or user/repo:file.palito"
    ))
}

impl ModelRef {
    pub(crate) fn parse(text: &str) -> palito::Result<Self> {
        // Ollama's hf.co/user/repo and pasted Hugging Face URLs.
        let stripped = ["https://huggingface.co/", "http://huggingface.co/", "huggingface.co/", "hf.co/"]
            .iter()
            .find_map(|prefix| text.strip_prefix(prefix))
            .map(|rest| rest.trim_end_matches('/'));
        if let Some(rest) = stripped {
            return match Self::parse(rest)? {
                repo @ Self::Repo { .. } => Ok(repo),
                Self::Name(_) => Err(invalid(text)),
            };
        }
        let (repo, file) = match text.split_once(':') {
            Some((repo, file)) => (repo, Some(file)),
            None => (text, None),
        };
        let parts: Vec<&str> = repo.split('/').collect();
        let segments_ok = parts.iter().all(|p| valid_segment(p));
        match (parts.len(), file) {
            (1, None) if segments_ok => Ok(Self::Name(text.to_owned())),
            (2, file) if segments_ok && file.is_none_or(valid_segment) => Ok(Self::Repo {
                repo: repo.to_owned(),
                file: file.map(str::to_owned),
            }),
            _ => Err(invalid(text)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_names_repos_and_files() {
        assert_eq!(ModelRef::parse("qwen3-8b").unwrap(), ModelRef::Name("qwen3-8b".into()));
        assert_eq!(
            ModelRef::parse("loterio/Qwen3-8B-palito").unwrap(),
            ModelRef::Repo { repo: "loterio/Qwen3-8B-palito".into(), file: None }
        );
        assert_eq!(
            ModelRef::parse("a/b:c.palito").unwrap(),
            ModelRef::Repo { repo: "a/b".into(), file: Some("c.palito".into()) }
        );
    }

    #[test]
    fn accepts_hugging_face_prefixes_and_quant_tags() {
        let want = ModelRef::Repo { repo: "bartowski/Qwen3-8B-GGUF".into(), file: Some("Q4_K_M".into()) };
        for text in [
            "bartowski/Qwen3-8B-GGUF:Q4_K_M",
            "hf.co/bartowski/Qwen3-8B-GGUF:Q4_K_M",
            "huggingface.co/bartowski/Qwen3-8B-GGUF:Q4_K_M",
        ] {
            assert_eq!(ModelRef::parse(text).unwrap(), want, "{text}");
        }
        assert_eq!(
            ModelRef::parse("https://huggingface.co/bartowski/Qwen3-8B-GGUF/").unwrap(),
            ModelRef::Repo { repo: "bartowski/Qwen3-8B-GGUF".into(), file: None }
        );
        assert!(ModelRef::parse("hf.co/qwen3-8b").is_err(), "a prefix needs user/repo");
    }

    #[test]
    fn rejects_paths_that_escape_the_store() {
        for bad in ["../x", "a/../b", "a/b/c", "/abs", "a/b:../c", "a/b:", "", ".hidden", "x:y", "a b"] {
            assert!(ModelRef::parse(bad).is_err(), "{bad} should be rejected");
        }
    }

    #[test]
    fn displays_like_input() {
        for text in ["qwen3-8b", "a/b", "a/b:c.palito"] {
            assert_eq!(ModelRef::parse(text).unwrap().to_string(), text);
        }
    }
}
