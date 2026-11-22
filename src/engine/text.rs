//! Text sinks for streamed generation.

/// Receives streamed text; `write` returns false to stop generation.
pub(super) trait TextSink {
    fn write(&mut self, text: &str) -> bool;
    fn end(&mut self) {}
}

impl<F: FnMut(&str) -> bool> TextSink for F {
    fn write(&mut self, text: &str) -> bool {
        self(text)
    }
}

/// Streams chat text without leading whitespace and holds trailing whitespace
/// until more text follows, so the streamed reply equals the trimmed one.
pub(super) struct TrimmedText<F> {
    inner: F,
    started: bool,
    pending: String,
}

impl<F: FnMut(&str) -> bool> TrimmedText<F> {
    pub(super) fn new(inner: F) -> Self {
        Self {
            inner,
            started: false,
            pending: String::new(),
        }
    }
}

impl<F: FnMut(&str) -> bool> TextSink for TrimmedText<F> {
    fn write(&mut self, text: &str) -> bool {
        let text = if self.started {
            text
        } else {
            text.trim_start()
        };
        if text.is_empty() {
            return true;
        }
        self.started = true;
        let body = text.trim_end();
        let keep_going = if body.is_empty() {
            true
        } else {
            let piece = std::mem::take(&mut self.pending) + body;
            (self.inner)(&piece)
        };
        self.pending.push_str(&text[body.len()..]);
        keep_going
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trimmed_text_streams_the_trimmed_reply() {
        let mut out = String::new();
        {
            let mut sink = TrimmedText::new(|piece: &str| {
                out.push_str(piece);
                true
            });
            for piece in ["\n\n", "  Hello", ", world", "  ", "\n", "again", " \n"] {
                assert!(sink.write(piece));
            }
        }
        assert_eq!(out, "Hello, world  \nagain");
    }

    #[test]
    fn trimmed_text_reports_stop() {
        let mut sink = TrimmedText::new(|_: &str| false);
        assert!(sink.write(" "));
        assert!(!sink.write("text"));
    }
}
