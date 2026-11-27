//! Which file of a Hugging Face repo to download for `user/repo[:tag]`.
//!
//! The tag after `:` is a file name (`m.gguf`) or, like llama.cpp `-hf` and
//! Ollama, a quantization (`Q4_K_M`), matched against the end of GGUF file
//! names. Without a tag: the repo's only `.palito` file, else its only GGUF,
//! else its `Q4_K_M` GGUF (llama.cpp's default).

use super::exit;
use crate::cli_ui::fail;
use palito::{Error, Result};

/// Quantization picked when a repo has several GGUF files and no tag is given.
pub(crate) const DEFAULT_QUANT: &str = "Q4_K_M";

/// `m.gguf`, `m.palito`: a file name rather than a quantization tag.
pub(crate) fn is_model_file(name: &str) -> bool {
    let lower = name.to_ascii_lowercase();
    lower.ends_with(".gguf") || lower.ends_with(".palito")
}

/// `Qwen3-8B-Q4_K_M.gguf` and `model.q4_k_m.gguf` match `q4_k_m`, but
/// `Qwen3-8B-IQ4_XS.gguf` does not match `Q4_XS`.
pub(crate) fn matches_tag(file: &str, tag: &str) -> bool {
    let lower = file.to_ascii_lowercase();
    let Some(stem) = lower.strip_suffix(".gguf") else { return false };
    let tag = tag.to_ascii_lowercase();
    stem == tag || ["-", ".", "_"].iter().any(|sep| stem.ends_with(&format!("{sep}{tag}")))
}

/// The quantization a GGUF file name ends with, for listing choices.
fn quant_tag(file: &str) -> &str {
    let stem = &file[..file.len() - ".gguf".len()];
    stem.rsplit(['-', '.']).next().unwrap_or(stem)
}

/// `-00001-of-00003.gguf`: one part of a split GGUF.
fn is_split_part(file: &str) -> bool {
    let lower = file.to_ascii_lowercase();
    let Some(stem) = lower.strip_suffix(".gguf") else { return false };
    let parts: Vec<&str> = stem.rsplitn(4, '-').collect();
    matches!(parts.as_slice(), [total, "of", part, _]
        if total.len() == 5 && part.len() == 5
            && total.bytes().all(|b| b.is_ascii_digit()) && part.bytes().all(|b| b.is_ascii_digit()))
}

/// Single-file GGUF models at the top of the repo (no projectors, no parts).
fn ggufs(files: &[String]) -> Vec<&String> {
    files
        .iter()
        .filter(|f| {
            f.to_ascii_lowercase().ends_with(".gguf")
                && !f.contains('/')
                && !f.to_ascii_lowercase().contains("mmproj")
                && !is_split_part(f)
        })
        .collect()
}

/// One line of tags and one command, e.g.
/// "Q4_K_M, Q8_0, IQ4_XS\n  palito pull user/repo:Q4_K_M".
fn choices(repo: &str, files: &[&String]) -> String {
    let tags: Vec<&str> = files.iter().map(|f| quant_tag(f)).collect();
    let example = tags.iter().find(|t| t.eq_ignore_ascii_case(DEFAULT_QUANT)).or(tags.first()).copied().unwrap_or(DEFAULT_QUANT);
    format!(" {}\n  palito pull {repo}:{example}", tags.join(", "))
}

/// Picks the file to download from `files` (the repo's file list).
pub(crate) fn pick_file(repo: &str, files: &[String], tag: Option<&str>) -> Result<String> {
    let ggufs = ggufs(files);
    if let Some(tag) = tag {
        let matching: Vec<&String> = ggufs.iter().copied().filter(|f| matches_tag(f, tag)).collect();
        return match matching.as_slice() {
            [one] => Ok((*one).clone()),
            [] => {
                let parts = files.iter().filter(|f| is_split_part(f) && quant_matches_part(f, tag)).count();
                if parts > 0 {
                    return Err(Error::Invalid(format!(
                        "{repo}:{tag} is a GGUF split into {parts} files, which palito cannot load yet\n  \
                         hint: pick a single-file quantization:{}",
                        choices(repo, &ggufs)
                    )));
                }
                let hint = if ggufs.is_empty() {
                    format!("{repo} has no single-file GGUF models")
                } else {
                    format!("available quantizations:{}", choices(repo, &ggufs))
                };
                Err(fail(exit::NOT_FOUND, format!("{repo} has no {tag} GGUF file\n  hint: {hint}")))
            }
            many => Err(Error::Invalid(format!(
                "{repo}:{tag} matches {} files; name one:{}",
                many.len(),
                many.iter().map(|f| format!("\n  palito pull {repo}:{f}")).collect::<String>()
            ))),
        };
    }
    let palitos: Vec<&String> = files.iter().filter(|f| f.ends_with(".palito") && !f.contains('/')).collect();
    match palitos.as_slice() {
        [one] => return Ok((*one).clone()),
        [] => {}
        many => {
            return Err(Error::Invalid(format!(
                "{repo} has {} .palito files; choose one:{}",
                many.len(),
                many.iter().map(|f| format!("\n  palito pull {repo}:{f}")).collect::<String>()
            )));
        }
    }
    match ggufs.as_slice() {
        [one] => Ok((*one).clone()),
        [] => Err(fail(
            exit::NOT_FOUND,
            format!(
                "{repo} has no .palito or single-file .gguf model\n  \
                 hint: name a file explicitly: palito pull {repo}:<file>.gguf"
            ),
        )),
        many => match many.iter().filter(|f| matches_tag(f, DEFAULT_QUANT)).collect::<Vec<_>>().as_slice() {
            [one] => Ok((**one).clone()),
            _ => Err(Error::Invalid(format!(
                "{repo} has {} GGUF files and none is {DEFAULT_QUANT}; choose a quantization:{}",
                many.len(),
                choices(repo, many)
            ))),
        },
    }
}

/// `Q4_K_M` for `m-Q4_K_M-00001-of-00002.gguf`.
fn quant_matches_part(file: &str, tag: &str) -> bool {
    let lower = file.to_ascii_lowercase();
    lower.contains(&format!("-{}-", tag.to_ascii_lowercase())) || lower.contains(&format!(".{}-", tag.to_ascii_lowercase()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cli_ui;

    fn files(names: &[&str]) -> Vec<String> {
        names.iter().map(|s| s.to_string()).collect()
    }

    const BARTOWSKI: &[&str] = &[
        "README.md",
        "Qwen3-8B-Q8_0.gguf",
        "Qwen3-8B-Q4_K_M.gguf",
        "Qwen3-8B-Q4_K_S.gguf",
        "Qwen3-8B-IQ4_XS.gguf",
        "Qwen3-8B-Q4_K.gguf",
        "mmproj-Qwen3-8B-f16.gguf",
        "Qwen3-8B-BF16/Qwen3-8B-BF16-00001-of-00002.gguf",
    ];

    #[test]
    fn tags_match_the_end_of_gguf_names() {
        assert!(matches_tag("Qwen3-8B-Q4_K_M.gguf", "q4_k_m"));
        assert!(matches_tag("llama-2-7b.Q4_K_M.gguf", "Q4_K_M"));
        assert!(matches_tag("Q8_0.gguf", "Q8_0"));
        assert!(!matches_tag("Qwen3-8B-IQ4_XS.gguf", "Q4_XS"));
        assert!(!matches_tag("Qwen3-8B-Q4_K_M.gguf", "Q4_K"));
        assert!(!matches_tag("Qwen3-8B-Q4_K_M.palito", "Q4_K_M"));
        assert!(is_model_file("x.GGUF") && is_model_file("x.palito") && !is_model_file("Q4_K_M"));
    }

    #[test]
    fn a_tag_picks_one_quantization() {
        let files = files(BARTOWSKI);
        assert_eq!(pick_file("b/q", &files, Some("Q8_0")).unwrap(), "Qwen3-8B-Q8_0.gguf");
        assert_eq!(pick_file("b/q", &files, Some("q4_k")).unwrap(), "Qwen3-8B-Q4_K.gguf");
        assert_eq!(pick_file("b/q", &files, Some("IQ4_XS")).unwrap(), "Qwen3-8B-IQ4_XS.gguf");
        let missing = pick_file("b/q", &files, Some("Q2_K")).unwrap_err();
        let text = cli_ui::message(&missing);
        assert!(text.contains("b/q has no Q2_K GGUF file"), "{text}");
        assert!(text.contains("palito pull b/q:Q4_K_M") && !text.contains("mmproj"), "{text}");
    }

    #[test]
    fn no_tag_prefers_palito_then_the_only_gguf_then_q4_k_m() {
        assert_eq!(pick_file("u/r", &files(&["m.palito", "m-Q8_0.gguf"]), None).unwrap(), "m.palito");
        assert_eq!(pick_file("u/r", &files(&["README.md", "m-Q8_0.gguf"]), None).unwrap(), "m-Q8_0.gguf");
        assert_eq!(pick_file("b/q", &files(BARTOWSKI), None).unwrap(), "Qwen3-8B-Q4_K_M.gguf");
        let many = cli_ui::message(&pick_file("u/r", &files(&["m-Q8_0.gguf", "m-Q6_K.gguf"]), None).unwrap_err());
        assert!(many.contains("choose a quantization: Q8_0, Q6_K") && many.contains("palito pull u/r:Q8_0"), "{many}");
        assert!(pick_file("u/r", &files(&["README.md"]), None).is_err());
        let two = cli_ui::message(&pick_file("u/r", &files(&["a.palito", "b.palito"]), None).unwrap_err());
        assert!(two.contains("palito pull u/r:a.palito"));
    }

    #[test]
    fn split_ggufs_are_refused_with_alternatives() {
        let repo = files(&["m-Q8_0-00001-of-00002.gguf", "m-Q8_0-00002-of-00002.gguf", "m-Q4_K_M.gguf"]);
        assert!(is_split_part("m-Q8_0-00001-of-00002.gguf") && !is_split_part("m-Q8_0.gguf"));
        let text = cli_ui::message(&pick_file("u/r", &repo, Some("Q8_0")).unwrap_err());
        assert!(text.contains("split into 2 files") && text.contains("palito pull u/r:Q4_K_M"), "{text}");
    }
}
