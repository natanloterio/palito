//! OpenAI-compatible request and response shapes (the subset Palito serves).
//!
//! Supported generation fields: `max_tokens`/`max_completion_tokens`,
//! `temperature`, `top_p`, `top_k` (llama.cpp extension), `seed`, `stop` and
//! `response_format` (`text`, `json_object`, `json_schema`).
//! Omitted sampling fields take the server defaults (greedy unless `palito
//! serve` was started with `--temperature`). Unsupported behavior (`n` > 1) is
//! rejected rather than silently ignored; other unknown fields are ignored.

use palito::{chat::Message, engine::FinishReason, grammar::JsonSchema, sampling::Sampling};
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
    /// llama.cpp convention: "none" keeps `<think>` reasoning inside `content`;
    /// anything else returns it as `reasoning_content`.
    pub reasoning_format: Option<String>,
    pub n: Option<u32>,
    pub stop: Option<Value>,
    pub response_format: Option<Value>,
    #[serde(flatten)]
    pub sampling: SamplingFields,
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
    pub response_format: Option<Value>,
    #[serde(flatten)]
    pub sampling: SamplingFields,
}

#[derive(Deserialize, Default)]
pub struct SamplingFields {
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub top_k: Option<u32>,
    pub seed: Option<u64>,
}

impl SamplingFields {
    /// Request settings over the server defaults, validated.
    pub fn resolve(&self, defaults: &Sampling) -> Result<Sampling, String> {
        let sampling = Sampling {
            temperature: self.temperature.unwrap_or(defaults.temperature),
            top_p: self.top_p.unwrap_or(defaults.top_p),
            top_k: self.top_k.unwrap_or(defaults.top_k),
            seed: self.seed.or(defaults.seed),
        };
        sampling.validate().map_err(|error| error.to_string())?;
        Ok(sampling)
    }
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
        check_supported(self.n)
    }

    pub fn stop(&self) -> Result<Vec<String>, String> {
        stop_sequences(self.stop.as_ref())
    }

    pub fn json_schema(&self) -> Result<Option<JsonSchema>, String> {
        json_schema(self.response_format.as_ref())
    }

    pub fn split_reasoning(&self) -> bool {
        self.reasoning_format.as_deref() != Some("none")
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
        check_supported(self.n)
    }

    pub fn stop(&self) -> Result<Vec<String>, String> {
        stop_sequences(self.stop.as_ref())
    }

    pub fn json_schema(&self) -> Result<Option<JsonSchema>, String> {
        json_schema(self.response_format.as_ref())
    }
}

/// `response_format`: `{"type": "text"}` (or null), `{"type": "json_object"}`,
/// or `{"type": "json_schema", "json_schema": {"name": ..., "schema": {...}}}`.
/// `name` and `strict` are accepted and ignored: output always follows the
/// schema.
fn json_schema(format: Option<&Value>) -> Result<Option<JsonSchema>, String> {
    let Some(format) = format.filter(|format| !format.is_null()) else {
        return Ok(None);
    };
    let kind = format
        .get("type")
        .and_then(Value::as_str)
        .ok_or("response_format must be an object with a string \"type\"")?;
    match kind {
        "text" => Ok(None),
        "json_object" => Ok(Some(JsonSchema::any_object())),
        "json_schema" => {
            let schema = format
                .get("json_schema")
                .and_then(|spec| spec.get("schema"))
                .ok_or("response_format json_schema needs json_schema.schema")?;
            JsonSchema::new(schema.clone())
                .map(Some)
                .map_err(|error| error.to_string())
        }
        other => Err(format!(
            "unsupported response_format type {other:?} (text, json_object or json_schema)"
        )),
    }
}

fn check_supported(n: Option<u32>) -> Result<(), String> {
    if n.is_some_and(|n| n != 1) {
        return Err("only n = 1 is supported".into());
    }
    Ok(())
}

/// `stop`: null, a string, or an array of strings (empty strings are ignored).
fn stop_sequences(stop: Option<&Value>) -> Result<Vec<String>, String> {
    let stops = match stop {
        None | Some(Value::Null) => Vec::new(),
        Some(Value::String(text)) => vec![text.clone()],
        Some(Value::Array(items)) => items
            .iter()
            .map(|item| item.as_str().map(str::to_owned))
            .collect::<Option<Vec<_>>>()
            .ok_or("stop must be a string or an array of strings")?,
        Some(_) => return Err("stop must be a string or an array of strings".into()),
    };
    let stops: Vec<String> = stops.into_iter().filter(|s| !s.is_empty()).collect();
    palito::stop::validate(&stops).map_err(|error| error.to_string())?;
    Ok(stops)
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
        assert_eq!(with("stop", json!(["\n", ""])).stop().unwrap(), vec!["\n"]);
        assert_eq!(with("stop", json!("END")).stop().unwrap(), vec!["END"]);
        assert!(with("stop", json!([1])).stop().is_err());
        assert!(with("stop", json!(vec!["x"; 17])).stop().is_err());
        let defaults = Sampling { temperature: 0.6, top_p: 0.95, top_k: 20, seed: None };
        let custom = with("temperature", json!(0.2)).sampling.resolve(&defaults).unwrap();
        assert_eq!((custom.temperature, custom.top_p, custom.top_k), (0.2, 0.95, 20));
        assert!(with("top_p", json!(0)).sampling.resolve(&defaults).is_err());
        assert!(with("reasoning_format", json!("none")).split_reasoning() == false);
        assert!(chat(base.clone()).split_reasoning());
        let image = json!([{"type": "image_url", "image_url": {"url": "x"}}]);
        let mut body = base.clone();
        body["messages"][0]["content"] = image;
        assert!(chat(body).messages().is_err());
        assert!(chat(json!({"messages": []})).messages().is_err());
        let prompt: CompletionRequest = serde_json::from_value(json!({"prompt": ["a", "b"]})).unwrap();
        assert!(prompt.prompt().is_err());
    }

    #[test]
    fn parses_response_format() {
        let format = |value: Value| {
            chat(json!({"messages": [], "response_format": value})).json_schema()
        };
        assert_eq!(chat(json!({"messages": []})).json_schema(), Ok(None));
        assert_eq!(format(json!({"type": "text"})), Ok(None));
        assert_eq!(format(json!({"type": "json_object"})), Ok(Some(JsonSchema::any_object())));
        let schema = json!({"type": "object", "properties": {"a": {"type": "integer"}}});
        let parsed = format(json!({
            "type": "json_schema",
            "json_schema": {"name": "thing", "strict": true, "schema": schema}
        }))
        .unwrap()
        .unwrap();
        assert_eq!(parsed.schema(), &schema);
        for (bad, needle) in [
            (json!("json"), "string \"type\""),
            (json!({"type": "xml"}), "unsupported response_format"),
            (json!({"type": "json_schema", "json_schema": {"name": "x"}}), "json_schema.schema"),
            (
                json!({"type": "json_schema", "json_schema": {"schema": {"type": "array", "uniqueItems": true}}}),
                "uniqueItems",
            ),
        ] {
            let error = format(bad.clone()).unwrap_err();
            assert!(error.contains(needle), "{bad}: {error}");
        }
    }
}
