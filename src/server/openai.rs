//! OpenAI-compatible request and response shapes (the subset Palito serves).
//!
//! Sampling fields such as `temperature` or `top_p` are accepted and ignored:
//! Palito decodes greedily. Unsupported behavior (`n` > 1, `stop` sequences)
//! is rejected rather than silently ignored.

use palito::{chat::Message, engine::FinishReason};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::{
    sync::atomic::{AtomicU64, Ordering},
    time::{SystemTime, UNIX_EPOCH},
};

#[derive(Deserialize)]
pub struct ChatRequest {
    pub messages: Vec<ChatMessage>,
    #[serde(default)]
    pub stream: bool,
    pub max_tokens: Option<u32>,
    pub max_completion_tokens: Option<u32>,
    pub stream_options: Option<StreamOptions>,
    /// llama.cpp/vLLM convention: `{"enable_thinking": false}` for Qwen3.
    pub chat_template_kwargs: Option<TemplateKwargs>,
    pub n: Option<u32>,
    pub stop: Option<Value>,
}

#[derive(Deserialize)]
pub struct CompletionRequest {
    pub prompt: Value,
    #[serde(default)]
    pub stream: bool,
    pub max_tokens: Option<u32>,
    pub stream_options: Option<StreamOptions>,
    pub n: Option<u32>,
    pub stop: Option<Value>,
}

#[derive(Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: Option<Value>,
}

#[derive(Deserialize, Default)]
pub struct StreamOptions {
    #[serde(default)]
    pub include_usage: bool,
}

#[derive(Deserialize)]
pub struct TemplateKwargs {
    pub enable_thinking: Option<bool>,
}

/// OpenAI's default for /v1/completions.
pub const COMPLETION_DEFAULT_MAX_TOKENS: u32 = 16;

impl ChatRequest {
    pub fn messages(&self) -> Result<Vec<Message>, String> {
        if self.messages.is_empty() {
            return Err("messages must not be empty".into());
        }
        self.messages
            .iter()
            .map(|m| Ok(Message::new(&m.role, content_text(m.content.as_ref())?)))
            .collect()
    }

    pub fn max_tokens(&self) -> Option<u32> {
        self.max_completion_tokens.or(self.max_tokens)
    }

    pub fn enable_thinking(&self) -> bool {
        self.chat_template_kwargs
            .as_ref()
            .and_then(|kwargs| kwargs.enable_thinking)
            .unwrap_or(true)
    }

    pub fn check_supported(&self) -> Result<(), String> {
        check_supported(self.n, self.stop.as_ref())
    }
}

impl CompletionRequest {
    pub fn prompt(&self) -> Result<String, String> {
        match &self.prompt {
            Value::String(text) => Ok(text.clone()),
            Value::Array(items) if items.len() == 1 && items[0].is_string() => {
                Ok(items[0].as_str().unwrap_or_default().to_owned())
            }
            _ => Err("prompt must be a string (or an array with one string)".into()),
        }
    }

    pub fn check_supported(&self) -> Result<(), String> {
        check_supported(self.n, self.stop.as_ref())
    }
}

fn check_supported(n: Option<u32>, stop: Option<&Value>) -> Result<(), String> {
    if n.is_some_and(|n| n != 1) {
        return Err("only n = 1 is supported".into());
    }
    let empty = match stop {
        None | Some(Value::Null) => true,
        Some(Value::String(text)) => text.is_empty(),
        Some(Value::Array(items)) => items.is_empty(),
        Some(_) => false,
    };
    if empty {
        Ok(())
    } else {
        Err("stop sequences are not supported".into())
    }
}

/// Message content: a string, or text parts `[{"type": "text", "text": ...}]`.
fn content_text(content: Option<&Value>) -> Result<String, String> {
    match content {
        None | Some(Value::Null) => Ok(String::new()),
        Some(Value::String(text)) => Ok(text.clone()),
        Some(Value::Array(parts)) => parts
            .iter()
            .map(|part| match (part.get("type"), part.get("text")) {
                (Some(Value::String(kind)), Some(Value::String(text))) if kind == "text" => {
                    Ok(text.as_str())
                }
                _ => Err("only text content parts are supported".to_owned()),
            })
            .collect::<Result<Vec<_>, _>>()
            .map(|texts| texts.concat()),
        Some(_) => Err("message content must be a string or text parts".into()),
    }
}

pub fn finish_reason(reason: FinishReason) -> &'static str {
    match reason {
        FinishReason::Stop | FinishReason::Cancelled => "stop",
        FinishReason::Length | FinishReason::ContextFull => "length",
    }
}

#[derive(Serialize, Clone, Copy)]
pub struct Usage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
}

impl Usage {
    pub fn new(prompt_tokens: usize, completion_tokens: usize) -> Self {
        Self {
            prompt_tokens,
            completion_tokens,
            total_tokens: prompt_tokens + completion_tokens,
        }
    }
}

pub fn unix_seconds() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |elapsed| elapsed.as_secs())
}

/// Unique per process run: start time plus a counter.
pub fn response_id(prefix: &str) -> String {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let count = COUNTER.fetch_add(1, Ordering::Relaxed);
    let start = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |elapsed| elapsed.as_nanos() as u64);
    format!("{prefix}-{start:x}{count:x}")
}

pub fn error_body(message: &str, kind: &str) -> Value {
    json!({"error": {"message": message, "type": kind, "param": null, "code": null}})
}

#[cfg(test)]
mod tests {
    use super::*;

    fn chat(body: Value) -> ChatRequest {
        serde_json::from_value(body).unwrap()
    }

    #[test]
    fn parses_string_and_part_content_and_options() {
        let request = chat(json!({
            "model": "anything",
            "messages": [
                {"role": "system", "content": "Be brief."},
                {"role": "user", "content": [{"type": "text", "text": "Hi"}, {"type": "text", "text": "!"}]}
            ],
            "temperature": 0.7,
            "max_tokens": 10,
            "max_completion_tokens": 5,
            "chat_template_kwargs": {"enable_thinking": false},
            "stop": null
        }));
        let messages = request.messages().unwrap();
        assert_eq!(messages[1], Message::user("Hi!"));
        assert_eq!(request.max_tokens(), Some(5));
        assert!(!request.enable_thinking());
        assert!(request.check_supported().is_ok());
    }

    #[test]
    fn rejects_unsupported_requests() {
        let base = json!({"messages": [{"role": "user", "content": "x"}]});
        let with = |key: &str, value: Value| {
            let mut body = base.clone();
            body[key] = value;
            chat(body)
        };
        assert!(with("n", json!(2)).check_supported().is_err());
        assert!(with("stop", json!(["\n"])).check_supported().is_err());
        assert!(with("stop", json!([])).check_supported().is_ok());
        let image = json!([{"type": "image_url", "image_url": {"url": "x"}}]);
        let mut body = base.clone();
        body["messages"][0]["content"] = image;
        assert!(chat(body).messages().is_err());
        assert!(chat(json!({"messages": []})).messages().is_err());
        let prompt: CompletionRequest = serde_json::from_value(json!({"prompt": ["a", "b"]})).unwrap();
        assert!(prompt.prompt().is_err());
    }
}
