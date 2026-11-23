//! Turns generated tokens, with their roles, into text and tool call events.
//!
//! Works on bytes because a character can be split across tokens. The call
//! shape is fixed by the grammar: `{"name": <string>, "arguments": <value>}`.

use super::TokenRole;
use std::str;

const NAME_PREFIX: &[u8] = b"{\"name\": ";
const ARGUMENTS_KEY: &[u8] = b", \"arguments\": ";

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ToolEvent {
    Text(String),
    CallStart { index: usize, name: String },
    CallArguments { index: usize, text: String },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ToolCall {
    pub name: String,
    /// JSON text of the arguments; partial when `complete` is false.
    pub arguments: String,
    pub complete: bool,
}

/// The call being generated.
#[derive(Default)]
struct ActiveCall {
    bytes: Vec<u8>,
    index: Option<usize>,
    /// First byte after the name string.
    after_name: usize,
    arguments_start: Option<usize>,
    /// Scan position inside an object or array argument.
    scan: usize,
    depth: usize,
    in_string: bool,
    escaped: bool,
    /// One past the closing bracket of the arguments.
    arguments_end: Option<usize>,
    emitted: usize,
}

#[derive(Default)]
pub struct ToolStream {
    held_text: Vec<u8>,
    calls: Vec<ToolCall>,
    active: Option<ActiveCall>,
}

impl ToolStream {
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds one token's bytes, with its role from the `ToolGate`.
    pub fn push(&mut self, role: TokenRole, piece: &[u8]) -> Vec<ToolEvent> {
        let mut events = Vec::new();
        match role {
            TokenRole::Text => self.push_text(piece, &mut events),
            TokenRole::CallOpen => {
                self.flush_text(&mut events);
                self.close_call(false, &mut events);
                self.active = Some(ActiveCall::default());
            }
            TokenRole::CallBody => {
                if let Some(active) = &mut self.active {
                    active.bytes.extend_from_slice(piece);
                    active.advance(&mut self.calls, &mut events);
                }
            }
            TokenRole::CallClose => self.close_call(true, &mut events),
        }
        events
    }

    /// Ends the reply: releases held text and keeps a truncated call.
    pub fn finish(&mut self) -> Vec<ToolEvent> {
        let mut events = Vec::new();
        self.flush_text(&mut events);
        self.close_call(false, &mut events);
        events
    }

    pub fn calls(&self) -> &[ToolCall] {
        &self.calls
    }

    pub fn in_call(&self) -> bool {
        self.active.is_some()
    }

    fn push_text(&mut self, piece: &[u8], events: &mut Vec<ToolEvent>) {
        // Between calls the gate only allows whitespace, which is not content.
        if !self.calls.is_empty() && piece.iter().all(u8::is_ascii_whitespace) {
            return;
        }
        self.held_text.extend_from_slice(piece);
        let valid = valid_prefix(&self.held_text);
        if valid > 0 {
            let rest = self.held_text.split_off(valid);
            let text = String::from_utf8_lossy(&self.held_text).into_owned();
            self.held_text = rest;
            events.push(ToolEvent::Text(text));
        }
    }

    fn flush_text(&mut self, events: &mut Vec<ToolEvent>) {
        if !self.held_text.is_empty() {
            let text = String::from_utf8_lossy(&self.held_text).into_owned();
            self.held_text.clear();
            events.push(ToolEvent::Text(text));
        }
    }

    fn close_call(&mut self, closed: bool, events: &mut Vec<ToolEvent>) {
        if let Some(mut active) = self.active.take() {
            active.finish(closed, &mut self.calls, events);
        }
    }
}

impl ActiveCall {
    fn advance(&mut self, calls: &mut Vec<ToolCall>, events: &mut Vec<ToolEvent>) {
        if self.index.is_none() && !self.read_name(calls, events) {
            return;
        }
        if self.arguments_start.is_none() && !self.find_arguments() {
            return;
        }
        if self.is_container() {
            self.scan_container();
            let limit = self.arguments_end.unwrap_or(self.scan);
            self.emit_valid(limit, calls, events);
        }
    }

    fn read_name(&mut self, calls: &mut Vec<ToolCall>, events: &mut Vec<ToolEvent>) -> bool {
        let Some(start) = self.bytes.iter().position(|b| !b.is_ascii_whitespace()) else {
            return false;
        };
        let Some(text) = self.bytes[start..].strip_prefix(NAME_PREFIX) else {
            return false;
        };
        let Some(length) = string_length(text) else {
            return false;
        };
        let Ok(name) = serde_json::from_slice::<String>(&text[..length]) else {
            return false;
        };
        self.after_name = start + NAME_PREFIX.len() + length;
        let index = calls.len();
        self.index = Some(index);
        calls.push(ToolCall { name: name.clone(), arguments: String::new(), complete: false });
        events.push(ToolEvent::CallStart { index, name });
        true
    }

    fn find_arguments(&mut self) -> bool {
        let Some(rest) = self.bytes[self.after_name..].strip_prefix(ARGUMENTS_KEY) else {
            return false;
        };
        let Some(skip) = rest.iter().position(|b| !b.is_ascii_whitespace()) else {
            return false;
        };
        let start = self.after_name + ARGUMENTS_KEY.len() + skip;
        self.arguments_start = Some(start);
        self.scan = start;
        self.emitted = start;
        true
    }

    fn is_container(&self) -> bool {
        self.arguments_start
            .and_then(|start| self.bytes.get(start))
            .is_some_and(|byte| matches!(byte, b'{' | b'['))
    }

    fn scan_container(&mut self) {
        while self.arguments_end.is_none() && self.scan < self.bytes.len() {
            let byte = self.bytes[self.scan];
            self.scan += 1;
            if self.in_string {
                match (self.escaped, byte) {
                    (true, _) => self.escaped = false,
                    (false, b'\\') => self.escaped = true,
                    (false, b'"') => self.in_string = false,
                    _ => {}
                }
                continue;
            }
            match byte {
                b'"' => self.in_string = true,
                b'{' | b'[' => self.depth += 1,
                b'}' | b']' => {
                    self.depth = self.depth.saturating_sub(1);
                    if self.depth == 0 {
                        self.arguments_end = Some(self.scan);
                    }
                }
                _ => {}
            }
        }
    }

    /// Emits up to `limit`, stopping before a character that is still split.
    fn emit_valid(&mut self, limit: usize, calls: &mut [ToolCall], events: &mut Vec<ToolEvent>) {
        let end = self.emitted + valid_prefix(&self.bytes[self.emitted..limit]);
        if end > self.emitted {
            let text = String::from_utf8_lossy(&self.bytes[self.emitted..end]).into_owned();
            self.emitted = end;
            self.send(text, calls, events);
        }
    }

    fn send(&self, text: String, calls: &mut [ToolCall], events: &mut Vec<ToolEvent>) {
        if let Some(index) = self.index {
            calls[index].arguments.push_str(&text);
            events.push(ToolEvent::CallArguments { index, text });
        }
    }

    fn finish(&mut self, closed: bool, calls: &mut Vec<ToolCall>, events: &mut Vec<ToolEvent>) {
        self.advance(calls, events);
        let Some(index) = self.index else {
            return;
        };
        if self.arguments_start.is_some() {
            let end = self.final_end().max(self.emitted);
            if end > self.emitted {
                let text = String::from_utf8_lossy(&self.bytes[self.emitted..end]).into_owned();
                self.emitted = end;
                self.send(text, calls, events);
            }
        }
        calls[index].complete = closed;
    }

    /// End of the arguments, without the closing brace of the envelope.
    fn final_end(&self) -> usize {
        if let Some(end) = self.arguments_end {
            return end;
        }
        if self.is_container() {
            return self.bytes.len();
        }
        let start = self.arguments_start.unwrap_or(self.bytes.len());
        let value = self.bytes[start..].trim_ascii_end();
        let value = value.strip_suffix(b"}").unwrap_or(value).trim_ascii_end();
        start + value.len()
    }
}

/// Length of the JSON string at the start of `text`, quotes included.
fn string_length(text: &[u8]) -> Option<usize> {
    if text.first() != Some(&b'"') {
        return None;
    }
    let mut escaped = false;
    for (index, &byte) in text.iter().enumerate().skip(1) {
        match (escaped, byte) {
            (true, _) => escaped = false,
            (false, b'\\') => escaped = true,
            (false, b'"') => return Some(index + 1),
            _ => {}
        }
    }
    None
}

/// How many leading bytes can be released: all but a split final character.
fn valid_prefix(bytes: &[u8]) -> usize {
    match str::from_utf8(bytes) {
        Ok(_) => bytes.len(),
        Err(error) if error.error_len().is_none() => error.valid_up_to(),
        Err(_) => bytes.len(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TEXT: TokenRole = TokenRole::Text;
    const WHOLE: &str = "\n{\"name\": \"w\", \"arguments\": {\"t\": \"</tool_call> ç 😊\", \"n\": {\"a\": [1]}}}\n";
    const ARGUMENTS: &str = "{\"t\": \"</tool_call> ç 😊\", \"n\": {\"a\": [1]}}";

    fn body(stream: &mut ToolStream, text: &str, cut: usize, events: &mut Vec<ToolEvent>) {
        for chunk in text.as_bytes().chunks(cut) {
            events.extend(stream.push(TokenRole::CallBody, chunk));
        }
    }

    fn run_cut(prefix: &[(TokenRole, &[u8])], whole: &str, cut: usize) -> (Vec<ToolEvent>, Vec<ToolCall>) {
        let mut stream = ToolStream::new();
        let mut events = Vec::new();
        for (role, piece) in prefix {
            events.extend(stream.push(*role, piece));
        }
        events.extend(stream.push(TokenRole::CallOpen, b"<tool_call>"));
        body(&mut stream, whole, cut, &mut events);
        events.extend(stream.push(TokenRole::CallClose, b"</tool_call>"));
        events.extend(stream.finish());
        (events, stream.calls().to_vec())
    }

    fn run_truncated(text: &str) -> (Vec<ToolEvent>, Vec<ToolCall>) {
        let mut stream = ToolStream::new();
        let mut events = stream.push(TokenRole::CallOpen, b"<tool_call>");
        body(&mut stream, text, 3, &mut events);
        events.extend(stream.finish());
        (events, stream.calls().to_vec())
    }

    fn run_two_calls() -> (Vec<ToolEvent>, Vec<ToolCall>) {
        let mut stream = ToolStream::new();
        let mut events = Vec::new();
        for (name, separator) in [("a", b"\n" as &[u8]), ("b", b"")] {
            events.extend(stream.push(TokenRole::CallOpen, b"<tool_call>"));
            let call = format!("\n{{\"name\": \"{name}\", \"arguments\": {{}}}}\n");
            body(&mut stream, &call, 2, &mut events);
            events.extend(stream.push(TokenRole::CallClose, b"</tool_call>"));
            events.extend(stream.push(TEXT, separator));
        }
        events.extend(stream.finish());
        (events, stream.calls().to_vec())
    }

    fn concat_arguments(events: &[ToolEvent], wanted: usize) -> String {
        events
            .iter()
            .filter_map(|event| match event {
                ToolEvent::CallArguments { index, text } if *index == wanted => Some(text.as_str()),
                _ => None,
            })
            .collect()
    }

    fn concat_text(events: &[ToolEvent]) -> String {
        events
            .iter()
            .filter_map(|event| match event {
                ToolEvent::Text(text) => Some(text.as_str()),
                _ => None,
            })
            .collect()
    }

    fn first_call_event(events: &[ToolEvent]) -> ToolEvent {
        events
            .iter()
            .find(|event| !matches!(event, ToolEvent::Text(_)))
            .cloned()
            .unwrap()
    }

    #[test]
    fn argument_deltas_match_for_every_cut() {
        for cut in 1..12 {
            let (events, calls) = run_cut(&[(TEXT, b"Vou ver.")], WHOLE, cut);
            assert_eq!(calls[0].name, "w");
            assert_eq!(calls[0].arguments, ARGUMENTS);
            assert!(calls[0].complete);
            assert_eq!(concat_arguments(&events, 0), calls[0].arguments, "cut {cut}");
            assert_eq!(first_call_event(&events), ToolEvent::CallStart { index: 0, name: "w".into() });
            assert_eq!(concat_text(&events), "Vou ver.");
        }
    }

    #[test]
    fn names_with_escapes_are_decoded() {
        let whole = "{\"name\": \"a\\\"b\\\\ç\", \"arguments\": {}}";
        for cut in 1..6 {
            let (_, calls) = run_cut(&[], whole, cut);
            assert_eq!((calls[0].name.as_str(), calls[0].arguments.as_str()), ("a\"b\\ç", "{}"));
        }
    }

    #[test]
    fn text_split_inside_a_character_is_not_corrupted() {
        let bytes = "ç😊".as_bytes();
        let mut stream = ToolStream::new();
        let mut events = Vec::new();
        for byte in bytes {
            events.extend(stream.push(TEXT, std::slice::from_ref(byte)));
        }
        events.extend(stream.finish());
        assert_eq!(concat_text(&events), "ç😊");
    }

    #[test]
    fn non_object_arguments_are_returned_without_the_envelope() {
        let (events, calls) = run_cut(&[], "{\"name\": \"s\", \"arguments\": \"x}\"}", 2);
        assert_eq!(calls[0].arguments, "\"x}\"");
        assert_eq!(concat_arguments(&events, 0), "\"x}\"");
    }

    #[test]
    fn markers_never_reach_text_and_parallel_calls_get_indexes() {
        let (events, calls) = run_two_calls();
        assert_eq!(calls.len(), 2);
        assert!(concat_text(&events).is_empty());
        assert!(events.contains(&ToolEvent::CallStart { index: 1, name: "b".into() }));
    }

    #[test]
    fn truncated_call_keeps_partial_arguments_and_truncated_name_is_dropped() {
        let (events, calls) = run_truncated("{\"name\": \"w\", \"arguments\": {\"t\": \"par");
        assert_eq!((calls[0].arguments.as_str(), calls[0].complete), ("{\"t\": \"par", false));
        assert_eq!(concat_arguments(&events, 0), "{\"t\": \"par");
        let (events, calls) = run_truncated("{\"name\": \"w");
        assert!(calls.is_empty() && !events.iter().any(|e| matches!(e, ToolEvent::CallStart { .. })));
    }

    #[test]
    fn truncation_after_the_arguments_keeps_only_the_arguments() {
        let (_, calls) = run_truncated("{\"name\": \"w\", \"arguments\": {\"a\": 1}}");
        assert_eq!((calls[0].arguments.as_str(), calls[0].complete), ("{\"a\": 1}", false));
    }
}
