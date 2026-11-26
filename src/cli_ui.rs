//! CLI-only presentation helpers: error rendering, color policy, file inputs
//! that name their path in errors, `-` for stdin, and human-readable sizes.

use clap::ColorChoice;
use palito::{Error, Result};
use std::{
    ffi::OsString,
    fs,
    io::{IsTerminal, Read},
    path::Path,
    sync::OnceLock,
};

/// Path argument meaning "read standard input".
const STDIN_PATH: &str = "-";
const GGUF_MAGIC: &[u8; 4] = b"GGUF";
const MODEL_HINT: &str =
    "pass a .gguf or .palito file with --model (or set PALITO_MODEL)";

static COLOR: OnceLock<bool> = OnceLock::new();

/// `--color` from the raw arguments, before clap parses (clap needs it to style
/// its own errors).
pub(crate) fn color_choice(args: &[OsString]) -> ColorChoice {
    let mut iter = args.iter().filter_map(|a| a.to_str());
    while let Some(arg) = iter.next() {
        let value = match arg.strip_prefix("--color") {
            Some("") => iter.next(),
            Some(rest) => rest.strip_prefix('='),
            None => continue,
        };
        return match value {
            Some("always") => ColorChoice::Always,
            Some("never") => ColorChoice::Never,
            _ => ColorChoice::Auto,
        };
    }
    ColorChoice::Auto
}

/// Decides once whether stderr gets ANSI styling.
pub(crate) fn init_color(choice: ColorChoice) {
    let enabled = match choice {
        ColorChoice::Always => true,
        ColorChoice::Never => false,
        ColorChoice::Auto => {
            std::env::var_os("NO_COLOR").is_none_or(|v| v.is_empty())
                && std::env::var("TERM").map_or(true, |t| t != "dumb")
                && std::io::stderr().is_terminal()
        }
    };
    let _ = COLOR.set(enabled);
}

/// Whether stderr output may use ANSI styling.
pub(crate) fn color_enabled() -> bool {
    *COLOR.get().unwrap_or(&false)
}

fn paint(text: &str, code: &str) -> String {
    if color_enabled() {
        format!("\x1b[{code}m{text}\x1b[0m")
    } else {
        text.to_owned()
    }
}

/// The user-facing text of an error, without library-internal prefixes.
pub(crate) fn message(error: &Error) -> String {
    match error {
        Error::Invalid(text) => text.clone(),
        other => other.to_string(),
    }
}

/// Prints `error: ...` to stderr; continuation lines starting with `hint:`
/// are styled as hints.
pub(crate) fn report(error: &Error) {
    let text = message(error);
    let mut lines = text.lines();
    eprintln!("{} {}", paint("error:", "1;31"), lines.next().unwrap_or_default());
    for line in lines {
        match line.trim_start().strip_prefix("hint:") {
            Some(rest) => eprintln!("  {}{rest}", paint("hint:", "1;36")),
            None => eprintln!("{line}"),
        }
    }
}

/// A warning on stderr, styled like `report`.
pub(crate) fn warn(text: &str) {
    eprintln!("{} {text}", paint("warning:", "1;33"));
}

fn file_error(path: &Path, error: std::io::Error) -> Error {
    match error.kind() {
        std::io::ErrorKind::NotFound => {
            Error::Invalid(format!("file not found: {}", path.display()))
        }
        _ => Error::Invalid(format!("{}: {error}", path.display())),
    }
}

/// Reads a file (or stdin for `-`), naming the path in any error.
pub(crate) fn read_bytes(path: &Path) -> Result<Vec<u8>> {
    if path.as_os_str() == STDIN_PATH {
        let mut bytes = Vec::new();
        std::io::stdin().lock().read_to_end(&mut bytes)?;
        return Ok(bytes);
    }
    fs::read(path).map_err(|error| file_error(path, error))
}

/// Reads UTF-8 text from a file (or stdin for `-`).
pub(crate) fn read_text(path: &Path) -> Result<String> {
    String::from_utf8(read_bytes(path)?)
        .map_err(|_| Error::Invalid(format!("{}: not valid UTF-8 text", path.display())))
}

/// Parses a JSON array of token IDs from a file (or stdin for `-`).
pub(crate) fn read_token_ids(path: &Path) -> Result<Vec<i32>> {
    serde_json::from_slice(&read_bytes(path)?).map_err(|error| {
        Error::Invalid(format!(
            "{}: expected a JSON array of token IDs ({error})",
            path.display()
        ))
    })
}

/// Fails early, with the path and a hint, when `path` is not a model file.
pub(crate) fn check_model(path: &Path) -> Result<()> {
    if path.is_dir() {
        return Err(Error::Invalid(format!(
            "{} is a directory, not a model file\n  hint: {MODEL_HINT}",
            path.display()
        )));
    }
    let mut header = [0u8; 4];
    let mut file = fs::File::open(path).map_err(|error| match error.kind() {
        std::io::ErrorKind::NotFound => Error::Invalid(format!(
            "model file not found: {}\n  hint: {MODEL_HINT}",
            path.display()
        )),
        _ => file_error(path, error),
    })?;
    let readable = file.read_exact(&mut header).is_ok();
    if readable && (&header == GGUF_MAGIC || palito::plt1::is_plt1_file(path)) {
        return Ok(());
    }
    Err(Error::Invalid(format!(
        "{} is not a GGUF or .palito model (unrecognized file header)\n  hint: {MODEL_HINT}",
        path.display()
    )))
}

const SIZE_UNITS: &[(&str, u64)] = &[
    ("tib", 1 << 40),
    ("gib", 1 << 30),
    ("mib", 1 << 20),
    ("kib", 1 << 10),
    ("tb", 1_000_000_000_000),
    ("gb", 1_000_000_000),
    ("mb", 1_000_000),
    ("kb", 1_000),
    ("t", 1 << 40),
    ("g", 1 << 30),
    ("m", 1 << 20),
    ("k", 1 << 10),
    ("b", 1),
];

/// Parses a byte size: plain bytes (`1048576`) or a unit (`512MiB`, `8GB`,
/// `1.5G`; a bare letter means the binary unit).
pub(crate) fn parse_size(text: &str) -> std::result::Result<u64, String> {
    let lower = text.trim().to_ascii_lowercase();
    let (number, scale) = SIZE_UNITS
        .iter()
        .find_map(|(unit, scale)| lower.strip_suffix(unit).map(|n| (n.trim(), *scale)))
        .unwrap_or((lower.as_str(), 1));
    if let Ok(value) = number.parse::<u64>() {
        return value
            .checked_mul(scale)
            .ok_or_else(|| format!("{text} is too large"));
    }
    match number.parse::<f64>() {
        Ok(value) if value.is_finite() && value >= 0.0 => Ok((value * scale as f64) as u64),
        _ => Err(format!(
            "expected a size such as 1073741824, 512MiB or 8GB, got '{text}'"
        )),
    }
}

/// `4.9 GB`-style size for messages.
pub(crate) fn human_size(bytes: u64) -> String {
    const UNITS: [&str; 4] = ["KB", "MB", "GB", "TB"];
    let mut value = bytes as f64;
    let mut unit = "B";
    for next in UNITS {
        if value < 1000.0 {
            break;
        }
        value /= 1000.0;
        unit = next;
    }
    if unit == "B" {
        format!("{bytes} B")
    } else {
        format!("{value:.1} {unit}")
    }
}

/// Edit distance, for "did you mean" suggestions.
pub(crate) fn edit_distance(a: &str, b: &str) -> usize {
    let b: Vec<char> = b.chars().collect();
    let mut row: Vec<usize> = (0..=b.len()).collect();
    for (i, ca) in a.chars().enumerate() {
        let mut previous = row[0];
        row[0] = i + 1;
        for (j, cb) in b.iter().enumerate() {
            let substitute = previous + usize::from(ca != *cb);
            previous = row[j + 1];
            row[j + 1] = substitute.min(row[j] + 1).min(previous + 1);
        }
    }
    row[b.len()]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sizes_accept_units() {
        assert_eq!(parse_size("1048576"), Ok(1 << 20));
        assert_eq!(parse_size("512MiB"), Ok(512 << 20));
        assert_eq!(parse_size("8GB"), Ok(8_000_000_000));
        assert_eq!(parse_size("8g"), Ok(8 << 30));
        assert_eq!(parse_size("1.5GiB"), Ok(3 << 29));
        assert!(parse_size("lots").is_err());
        assert!(parse_size("-1").is_err());
    }

    #[test]
    fn human_sizes() {
        assert_eq!(human_size(512), "512 B");
        assert_eq!(human_size(4_900_000_000), "4.9 GB");
    }

    #[test]
    fn color_flag_is_found_before_parsing() {
        let args = |a: &[&str]| a.iter().map(OsString::from).collect::<Vec<_>>();
        assert_eq!(color_choice(&args(&["palito", "--color", "never"])), ColorChoice::Never);
        assert_eq!(color_choice(&args(&["palito", "chat", "--color=always"])), ColorChoice::Always);
        assert_eq!(color_choice(&args(&["palito", "chat"])), ColorChoice::Auto);
    }

    #[test]
    fn edit_distance_counts_edits() {
        assert_eq!(edit_distance("exti", "exit"), 2);
        assert_eq!(edit_distance("rest", "reset"), 1);
        assert_eq!(edit_distance("same", "same"), 0);
    }

    #[test]
    fn missing_and_non_model_files_name_the_path() {
        let missing = check_model(Path::new("/nope/x.gguf")).unwrap_err();
        assert!(message(&missing).starts_with("model file not found: /nope/x.gguf"));
        let readme = check_model(Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/Cargo.toml"))).unwrap_err();
        assert!(message(&readme).contains("is not a GGUF or .palito model"));
    }
}
