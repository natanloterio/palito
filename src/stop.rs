//! Stop sequences: generation ends as soon as the text contains one of them,
//! and neither the stop text nor anything after it is returned.
//!
//! Streamed text is held back while it could still be the start of a stop
//! sequence, so a client never receives part of one.

use crate::{Error, Result};

pub const MAX_STOP_SEQUENCES: usize = 16;
pub const MAX_STOP_BYTES: usize = 256;

pub fn validate(stops: &[String]) -> Result<()> {
    if stops.len() > MAX_STOP_SEQUENCES {
        return Err(Error::Invalid(format!(
            "at most {MAX_STOP_SEQUENCES} stop sequences are supported"
        )));
    }
    if stops.iter().any(|s| s.is_empty() || s.len() > MAX_STOP_BYTES) {
        return Err(Error::Invalid(format!(
            "stop sequences must be 1 to {MAX_STOP_BYTES} bytes"
        )));
    }
    Ok(())
}

/// Index of the earliest stop sequence in `text`.
pub fn find(text: &str, stops: &[String]) -> Option<usize> {
    stops.iter().filter_map(|stop| text.find(stop.as_str())).min()
}

pub struct StopFilter<'a> {
    stops: &'a [String],
    /// Text received but not yet released.
    held: String,
    stopped: bool,
}

impl<'a> StopFilter<'a> {
    pub fn new(stops: &'a [String]) -> Self {
        Self {
            stops,
            held: String::new(),
            stopped: false,
        }
    }

    pub fn stopped(&self) -> bool {
        self.stopped
    }

    /// Adds generated text; returns the text that is safe to release. After a
    /// stop sequence appears, `stopped` is true and nothing more is released.
    pub fn push(&mut self, piece: &str) -> String {
        if self.stopped {
            return String::new();
        }
        self.held.push_str(piece);
        if let Some(index) = find(&self.held, self.stops) {
            self.stopped = true;
            self.held.truncate(index);
            return std::mem::take(&mut self.held);
        }
        let keep = self.partial_suffix();
        let release = self.held.len() - keep;
        let rest = self.held.split_off(release);
        std::mem::replace(&mut self.held, rest)
    }

    /// The end of generation: held text can no longer become a stop sequence.
    pub fn finish(&mut self) -> String {
        std::mem::take(&mut self.held)
    }

    /// Length of the longest suffix of `held` that starts some stop sequence.
    fn partial_suffix(&self) -> usize {
        let held = self.held.as_str();
        let longest = self.stops.iter().map(String::len).max().unwrap_or(0);
        (1..longest.min(held.len() + 1))
            .rev()
            .filter(|&n| held.is_char_boundary(held.len() - n))
            .find(|&n| {
                let tail = &held[held.len() - n..];
                self.stops.iter().any(|stop| stop.starts_with(tail))
            })
            .unwrap_or(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(stops: &[&str], pieces: &[&str]) -> (Vec<String>, bool, String) {
        let stops: Vec<String> = stops.iter().map(|s| s.to_string()).collect();
        let mut filter = StopFilter::new(&stops);
        let mut out: Vec<String> = pieces.iter().map(|p| filter.push(p)).collect();
        let stopped = filter.stopped();
        out.push(filter.finish());
        let joined = out.concat();
        (out, stopped, joined)
    }

    #[test]
    fn stops_across_pieces_without_leaking_the_stop_text() {
        let (out, stopped, text) = run(&["\nUser:"], &["Hello", " there\n", "Us", "er: hi"]);
        assert!(stopped);
        assert_eq!(text, "Hello there");
        // "\n" and "Us" are held until they turn into the stop sequence.
        assert_eq!(out[1], " there");
        assert_eq!(out[2], "");
    }

    #[test]
    fn releases_held_text_that_does_not_become_a_stop() {
        let (out, stopped, text) = run(&["END"], &["the E", "ND", "ing"]);
        assert!(stopped);
        assert_eq!(text, "the ");
        let (_, stopped, text) = run(&["END"], &["the E", "nd"]);
        assert!(!stopped);
        assert_eq!(text, "the End");
        assert_eq!(out[0], "the ");
    }

    #[test]
    fn earliest_stop_wins_and_multibyte_text_is_safe() {
        let (_, _, text) = run(&["ção", "x"], &["informa", "ção x"]);
        assert_eq!(text, "informa");
        assert_eq!(find("a, b. c", &["c".into(), ".".into()]), Some(4));
        let (out, stopped, _) = run(&["é!"], &["café", " ok"]);
        assert!(!stopped);
        assert_eq!(out[0], "caf");
    }

    #[test]
    fn validates_count_and_length() {
        assert!(validate(&["".into()]).is_err());
        assert!(validate(&vec!["a".into(); 17]).is_err());
        assert!(validate(&["x".repeat(257)]).is_err());
        assert!(validate(&["\n\n".into()]).is_ok());
    }
}
