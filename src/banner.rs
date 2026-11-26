//! The README logo as terminal art, printed when `palito serve` starts on a
//! terminal: the llama in green sunglasses on a chip with speed lines, the
//! slanted wordmark with an orange dot on the i, and the tagline.
//!
//! Lines use a tiny markup: `{g}` green, `{o}` orange, `{i}` italic, `{w}` reset.

use crate::cli_ui;

/// Left column (the llama) is padded to this width so the wordmark lines up.
const LLAMA_WIDTH: usize = 30;

const LLAMA: [&str; 11] = [
    "        ┬  ┬  ┬  ┬  ┬",
    "     ┌──┴──┴──┴──┴──┴──┐",
    "    ─┤   /\\    /\\      ├─",
    "    ─┤  /  \\__/  \\     ├─",
    " {g}━━━━{w}┤ |  {g}▄▄▄▄▄▄▄▄▄▄{w}   ├─",
    "{g}━━━━━{w}┤ |   {g}▀██▀▀▀██▀{w} ) ├─",
    " {g}━━━━{w}┤ |          ◡ )  ├─",
    "{g}━━━━━{w}┤  \\  ,,,,,__-'   ├─",
    "    ─┤   |'''''|       ├─",
    "     └──┬──┬──┬──┬──┬──┘",
    "        ┴  ┴  ┴  ┴  ┴",
];

/// Figlet "slant", italic like the logo's wordmark.
const WORDMARK: [&str; 11] = [
    "",
    "",
    "    ____        ___ __",
    "   / __ \\____ _/ {o}(_){w} /_____",
    "  / /_/ / __ `/ / / __/ __ \\",
    " / ____/ /_/ / / / /_/ /_/ /",
    "/_/    \\__,_/_/_/\\__/\\____/",
    "",
    "",
    "{i}Fast local LLM inference{w}",
    "{o}━━━━━━━━━━━━━━━━━━━━━━━━{w}",
];

const GREEN: &str = "\x1b[1;32m";
const ORANGE: &str = "\x1b[1;38;5;208m";
const ITALIC: &str = "\x1b[3m";
const RESET: &str = "\x1b[0m";

fn render(markup: &str, color: bool) -> String {
    let codes = if color { [GREEN, ORANGE, ITALIC, RESET] } else { [""; 4] };
    markup
        .replace("{g}", codes[0])
        .replace("{o}", codes[1])
        .replace("{i}", codes[2])
        .replace("{w}", codes[3])
}

fn visible_width(markup: &str) -> usize {
    render(markup, false).chars().count()
}

/// The banner as lines ready to print.
pub(crate) fn lines(color: bool) -> Vec<String> {
    LLAMA
        .iter()
        .zip(WORDMARK)
        .map(|(llama, word)| {
            let pad = " ".repeat(LLAMA_WIDTH.saturating_sub(visible_width(llama)));
            let line = format!("{}{pad}{}", render(llama, color), render(word, color));
            line.trim_end().to_owned()
        })
        .collect()
}

/// Prints the banner to stderr when it is a terminal wide enough for it.
pub(crate) fn print() {
    use std::io::IsTerminal;
    let wide_enough = std::env::var("COLUMNS")
        .ok()
        .and_then(|columns| columns.parse::<usize>().ok())
        .is_none_or(|columns| columns >= LLAMA_WIDTH + 26);
    if !std::io::stderr().is_terminal() || !wide_enough {
        return;
    }
    eprintln!();
    for line in lines(cli_ui::color_enabled()) {
        eprintln!("{line}");
    }
    eprintln!();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn banner_fits_80_columns_and_strips_markup() {
        for line in lines(false) {
            assert!(line.chars().count() <= 80, "{line}");
            assert!(!line.contains('{'), "{line}");
        }
        // `cargo test banner -- --nocapture` shows it.
        println!("{}", lines(true).join("\n"));
    }

    #[test]
    fn llama_column_fits_its_width() {
        for line in LLAMA {
            assert!(visible_width(line) < LLAMA_WIDTH, "{line}");
        }
    }
}
