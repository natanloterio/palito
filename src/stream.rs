//! Incremental detokenization for streamed generation.
//!
//! Text is emitted as soon as it is stable: a token whose bytes end in the
//! middle of a UTF-8 character is held until the character completes. The
//! concatenated output equals `detokenize` of the whole sequence.

use crate::Result;

/// Tokens kept before the unemitted ones so a tokenizer that treats the start
/// of a sequence specially (e.g. SentencePiece's leading space) sees context.
const CONTEXT_TOKENS: usize = 4;

pub struct TextStream {
    tokens: Vec<i32>,
    /// Tokens whose text has been emitted.
    stable: usize,
    emitted: String,
}

impl Default for TextStream {
    fn default() -> Self {
        Self::new()
    }
}

impl TextStream {
    pub fn new() -> Self {
        Self {
            tokens: Vec::new(),
            stable: 0,
            emitted: String::new(),
        }
    }

    /// Adds a token; returns the text that became stable (possibly empty).
    pub fn push(
        &mut self,
        token: i32,
        detokenize: impl Fn(&[i32]) -> Result<String>,
    ) -> Result<String> {
        self.tokens.push(token);
        let start = self.stable.saturating_sub(CONTEXT_TOKENS);
        let prefix = detokenize(&self.tokens[start..self.stable])?;
        let text = detokenize(&self.tokens[start..])?;
        let new = match text.strip_prefix(prefix.as_str()) {
            Some(new) => new.to_owned(),
            // The window decoded differently from the emitted text: fall back to
            // the whole sequence, which is always consistent.
            None => match detokenize(&self.tokens)?.strip_prefix(self.emitted.as_str()) {
                Some(new) => new.to_owned(),
                None => return Ok(String::new()),
            },
        };
        if new.ends_with('\u{FFFD}') {
            return Ok(String::new());
        }
        self.stable = self.tokens.len();
        self.emitted.push_str(&new);
        Ok(new)
    }

    /// Text not yet emitted (held partial characters) once generation ends.
    pub fn finish(&mut self, detokenize: impl Fn(&[i32]) -> Result<String>) -> Result<String> {
        let text = detokenize(&self.tokens)?;
        let rest = text
            .strip_prefix(self.emitted.as_str())
            .unwrap_or_default()
            .to_owned();
        self.stable = self.tokens.len();
        self.emitted = text;
        Ok(rest)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Each token is one byte; byte 0xFF marks a leading space dropped at the
    /// start of a sequence (SentencePiece-like).
    fn decode(tokens: &[i32]) -> Result<String> {
        let mut bytes: Vec<u8> = Vec::new();
        for (i, &token) in tokens.iter().enumerate() {
            match token {
                0xFF if i == 0 => {}
                0xFF => bytes.push(b' '),
                byte => bytes.push(byte as u8),
            }
        }
        Ok(String::from_utf8_lossy(&bytes).into_owned())
    }

    #[test]
    fn streamed_text_equals_whole_decode_and_holds_partial_characters() {
        let text = "Olá, coração 🌍";
        let mut tokens: Vec<i32> = Vec::new();
        for word in text.split(' ') {
            if !tokens.is_empty() {
                tokens.push(0xFF);
            }
            tokens.extend(word.bytes().map(i32::from));
        }
        let mut stream = TextStream::new();
        let mut out = String::new();
        for &token in &tokens {
            let piece = stream.push(token, decode).unwrap();
            assert!(!piece.contains('\u{FFFD}'), "partial character emitted");
            out.push_str(&piece);
        }
        out.push_str(&stream.finish(decode).unwrap());
        assert_eq!(out, decode(&tokens).unwrap());
        assert_eq!(out, text);
    }

    #[test]
    fn invalid_bytes_are_flushed_at_finish() {
        let tokens = [b'a' as i32, 0xC3];
        let mut stream = TextStream::new();
        let mut out = String::new();
        for token in tokens {
            out.push_str(&stream.push(token, decode).unwrap());
        }
        assert_eq!(out, "a");
        out.push_str(&stream.finish(decode).unwrap());
        assert_eq!(out, decode(&tokens).unwrap());
    }
}
