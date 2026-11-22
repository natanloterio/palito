//! Splits `<think>…</think>` reasoning (Qwen3, DeepSeek-R1 style) from the
//! answer, for the `reasoning_content` field llama.cpp and vLLM return.
//!
//! Works on streamed pieces: a tag split across pieces is held until it is
//! complete. Whitespace around the reasoning and at the start of the answer is
//! dropped. The result does not depend on where the pieces are cut.

const OPEN: &str = "<think>";
const CLOSE: &str = "</think>";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Part {
    Reasoning,
    Content,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum State {
    /// Before any text: an opening tag may follow.
    Start,
    Reasoning,
    Content,
}

pub struct ReasoningSplitter {
    state: State,
    held: String,
    reasoning_started: bool,
    /// Trailing reasoning whitespace, released only if more reasoning follows.
    reasoning_space: String,
    content_started: bool,
}

impl ReasoningSplitter {
    /// `opened`: the prompt already ends with `<think>`, so generation starts
    /// inside the reasoning.
    pub fn new(opened: bool) -> Self {
        Self {
            state: if opened { State::Reasoning } else { State::Start },
            held: String::new(),
            reasoning_started: false,
            reasoning_space: String::new(),
            content_started: false,
        }
    }

    pub fn push(&mut self, piece: &str) -> Vec<(Part, String)> {
        self.held.push_str(piece);
        let mut out = Vec::new();
        self.drain(&mut out, false);
        out
    }

    pub fn finish(&mut self) -> Vec<(Part, String)> {
        let mut out = Vec::new();
        self.drain(&mut out, true);
        out
    }

    fn drain(&mut self, out: &mut Vec<(Part, String)>, last: bool) {
        loop {
            match self.state {
                State::Start => {
                    let text = self.held.trim_start();
                    if let Some(rest) = text.strip_prefix(OPEN) {
                        self.held = rest.to_owned();
                        self.state = State::Reasoning;
                    } else if !last && OPEN.starts_with(text) {
                        return;
                    } else {
                        self.state = State::Content;
                    }
                }
                State::Reasoning => {
                    if let Some(index) = self.held.find(CLOSE) {
                        let reasoning = self.held[..index].to_owned();
                        self.held.drain(..index + CLOSE.len());
                        self.reasoning(out, &reasoning);
                        self.reasoning_space.clear();
                        self.state = State::Content;
                        continue;
                    }
                    let keep = if last { 0 } else { partial_close(&self.held) };
                    let rest = self.held.split_off(self.held.len() - keep);
                    let reasoning = std::mem::replace(&mut self.held, rest);
                    self.reasoning(out, &reasoning);
                    return;
                }
                State::Content => {
                    let text = std::mem::take(&mut self.held);
                    let text = if self.content_started {
                        text.as_str()
                    } else {
                        text.trim_start()
                    };
                    if !text.is_empty() {
                        self.content_started = true;
                        out.push((Part::Content, text.to_owned()));
                    }
                    return;
                }
            }
        }
    }

    fn reasoning(&mut self, out: &mut Vec<(Part, String)>, text: &str) {
        let text = if self.reasoning_started {
            text
        } else {
            text.trim_start()
        };
        if text.is_empty() {
            return;
        }
        self.reasoning_started = true;
        let body = text.trim_end();
        if !body.is_empty() {
            let piece = std::mem::take(&mut self.reasoning_space) + body;
            out.push((Part::Reasoning, piece));
        }
        self.reasoning_space.push_str(&text[body.len()..]);
    }
}

/// Length of the longest suffix of `text` that starts `</think>`.
fn partial_close(text: &str) -> usize {
    (1..CLOSE.len())
        .rev()
        .find(|&n| text.ends_with(&CLOSE[..n]))
        .unwrap_or(0)
}

/// (reasoning, answer) of a complete reply.
pub fn split(text: &str, opened: bool) -> (String, String) {
    let mut splitter = ReasoningSplitter::new(opened);
    let mut parts = splitter.push(text);
    parts.extend(splitter.finish());
    let mut reasoning = String::new();
    let mut content = String::new();
    for (part, piece) in parts {
        match part {
            Part::Reasoning => reasoning.push_str(&piece),
            Part::Content => content.push_str(&piece),
        }
    }
    (reasoning, content)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn streamed(text: &str, opened: bool, cut: usize) -> (String, String) {
        let mut splitter = ReasoningSplitter::new(opened);
        let mut parts = Vec::new();
        let mut start = 0;
        while start < text.len() {
            let mut end = (start + cut).min(text.len());
            while !text.is_char_boundary(end) {
                end += 1;
            }
            parts.extend(splitter.push(&text[start..end]));
            start = end;
        }
        parts.extend(splitter.finish());
        let pick = |want: Part| {
            parts
                .iter()
                .filter(|(part, _)| *part == want)
                .map(|(_, piece)| piece.as_str())
                .collect::<String>()
        };
        (pick(Part::Reasoning), pick(Part::Content))
    }

    #[test]
    fn splits_qwen3_reasoning_from_the_answer() {
        let reply = "<think>\nThe user greets.\n\nReply briefly. </think>\n\nOlá! Tudo bem? 😊";
        let expected = ("The user greets.\n\nReply briefly.".to_owned(), "Olá! Tudo bem? 😊".to_owned());
        assert_eq!(split(reply, false), expected);
        for cut in 1..12 {
            assert_eq!(streamed(reply, false, cut), expected, "cut {cut}");
        }
    }

    #[test]
    fn empty_reasoning_plain_answers_and_open_prompts() {
        assert_eq!(split("<think>\n\n</think>\n\nHi", false), (String::new(), "Hi".into()));
        assert_eq!(split("Just text <think> here", false), (String::new(), "Just text <think> here".into()));
        assert_eq!(split("weighing it</think>Answer", true), ("weighing it".into(), "Answer".into()));
        assert_eq!(streamed("weighing it</think>Answer", true, 3), ("weighing it".into(), "Answer".into()));
    }

    #[test]
    fn truncated_reasoning_and_partial_tags() {
        // max_tokens hit inside the reasoning: everything is reasoning.
        assert_eq!(split("<think>still going </th", false), ("still going </th".into(), String::new()));
        assert_eq!(split("<thi", false), (String::new(), "<thi".into()));
        for cut in 1..6 {
            assert_eq!(streamed("<think>a</think>b", false, cut), ("a".into(), "b".into()));
        }
    }
}
