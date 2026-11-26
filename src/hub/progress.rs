//! Download progress on stderr: a bar on a terminal, a line every 10% (or
//! 30 s) otherwise, nothing with `--quiet`.

use crate::cli_ui::human_size;
use indicatif::{ProgressBar, ProgressDrawTarget, ProgressStyle};
use std::{
    io::IsTerminal,
    time::{Duration, Instant},
};

const LINE_INTERVAL: Duration = Duration::from_secs(30);
const NARROW_COLUMNS: u16 = 60;

pub(crate) struct Progress {
    label: String,
    quiet: bool,
    bar: Option<ProgressBar>,
    total: Option<u64>,
    started: Instant,
    start_bytes: u64,
    pos: u64,
    last_tenth: u64,
    last_line: Instant,
}

impl Progress {
    pub(crate) fn new(label: &str, quiet: bool) -> Self {
        Self {
            label: label.to_owned(),
            quiet,
            bar: None,
            total: None,
            started: Instant::now(),
            start_bytes: 0,
            pos: 0,
            last_tenth: 0,
            last_line: Instant::now(),
        }
    }

    fn interactive(&self) -> bool {
        !self.quiet && std::io::stderr().is_terminal()
    }

    pub(crate) fn message(&self, text: &str) {
        if !self.quiet {
            eprintln!("  {text}");
        }
    }

    pub(crate) fn start(&mut self, total: Option<u64>, pos: u64) {
        self.total = total;
        self.pos = pos;
        self.start_bytes = pos;
        self.started = Instant::now();
        self.last_tenth = tenth(pos, total);
        if let Some(bar) = self.bar.take() {
            bar.finish_and_clear();
        }
        if !self.interactive() {
            return;
        }
        let bar = match total {
            Some(total) => ProgressBar::with_draw_target(Some(total), ProgressDrawTarget::stderr()),
            None => ProgressBar::with_draw_target(None, ProgressDrawTarget::stderr()),
        };
        bar.set_style(style(total.is_some()));
        bar.set_prefix(self.label.clone());
        bar.set_position(pos);
        bar.enable_steady_tick(Duration::from_millis(250));
        self.bar = Some(bar);
    }

    pub(crate) fn set(&mut self, pos: u64) {
        self.pos = pos;
        if let Some(bar) = &self.bar {
            bar.set_position(pos);
            return;
        }
        if self.quiet {
            return;
        }
        let now_tenth = tenth(pos, self.total);
        if now_tenth > self.last_tenth || self.last_line.elapsed() >= LINE_INTERVAL {
            self.last_tenth = now_tenth;
            self.last_line = Instant::now();
            let total = self.total.map_or(String::new(), |t| format!(" of {}", human_size(t)));
            let percent = self.total.map_or(String::new(), |t| format!("{}% ", pos * 100 / t.max(1)));
            eprintln!("{}: {percent}({}{total}, {}/s)", self.label, human_size(pos), human_size(self.rate()));
        }
    }

    fn rate(&self) -> u64 {
        let seconds = self.started.elapsed().as_secs_f64().max(0.001);
        ((self.pos - self.start_bytes) as f64 / seconds) as u64
    }

    /// Runs `f` (a warning) without tearing the bar.
    pub(crate) fn suspend(&self, f: impl FnOnce()) {
        match &self.bar {
            Some(bar) => bar.suspend(f),
            None => f(),
        }
    }

    pub(crate) fn abandon(&mut self) {
        if let Some(bar) = self.bar.take() {
            bar.abandon();
        }
    }

    pub(crate) fn finish(&mut self) {
        if let Some(bar) = self.bar.take() {
            bar.finish_and_clear();
        }
        if !self.quiet {
            eprintln!(
                "  {}  {}  done in {} ({}/s)",
                self.label,
                human_size(self.pos),
                duration(self.started.elapsed()),
                human_size(self.rate())
            );
        }
    }
}

fn tenth(pos: u64, total: Option<u64>) -> u64 {
    total.map_or(0, |t| pos * 10 / t.max(1))
}

fn duration(elapsed: Duration) -> String {
    let seconds = elapsed.as_secs();
    if seconds >= 60 { format!("{}m{:02}s", seconds / 60, seconds % 60) } else { format!("{:.1}s", elapsed.as_secs_f64()) }
}

fn style(known_total: bool) -> ProgressStyle {
    let utf8 = std::env::var("LANG").or_else(|_| std::env::var("LC_ALL")).is_ok_and(|v| v.to_ascii_uppercase().contains("UTF-8"))
        && std::env::var("TERM").map_or(true, |t| t != "dumb");
    let narrow = console_columns().is_some_and(|c| c < NARROW_COLUMNS);
    let template = match (known_total, narrow) {
        (false, _) => "  {prefix}  {bytes}  {bytes_per_sec}",
        (true, true) => "  {percent}%  {bytes}/{total_bytes}  ETA {eta}",
        (true, false) => "  {prefix}  [{bar:26}]  {bytes}/{total_bytes}  {bytes_per_sec}  ETA {eta}",
    };
    ProgressStyle::with_template(template)
        .expect("valid progress template")
        .progress_chars(if utf8 { "█░" } else { "#-" })
}

fn console_columns() -> Option<u16> {
    std::env::var("COLUMNS").ok().and_then(|c| c.parse().ok())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn durations_and_tenths() {
        assert_eq!(duration(Duration::from_secs(74)), "1m14s");
        assert_eq!(duration(Duration::from_millis(2500)), "2.5s");
        assert_eq!(tenth(55, Some(100)), 5);
        assert_eq!(tenth(5, None), 0);
    }
}
