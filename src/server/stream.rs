//! Server-sent events in the OpenAI streaming format.
//!
//! Chat: a first chunk with `delta.role`, one chunk per piece of text, a final
//! chunk with `finish_reason`, an optional usage chunk, then `data: [DONE]`.
//! Completions use the same order with `text` instead of `delta`. Chat
//! reasoning goes in `delta.reasoning_content` (llama.cpp/vLLM convention).

use super::{
    openai::{self, Usage},
    worker::Event,
};
use axum::response::{
    IntoResponse, Response,
    sse::{Event as SseEvent, KeepAlive, Sse},
};
use futures_util::stream::{self, StreamExt};
use serde_json::{Value, json};
use std::{convert::Infallible, sync::Arc};
use tokio::sync::mpsc::UnboundedReceiver;

#[derive(Clone, Copy)]
pub enum Kind {
    Chat,
    Text,
}

impl Kind {
    pub fn id_prefix(self) -> &'static str {
        match self {
            Kind::Chat => "chatcmpl",
            Kind::Text => "cmpl",
        }
    }

    pub fn object(self) -> &'static str {
        match self {
            Kind::Chat => "chat.completion",
            Kind::Text => "text_completion",
        }
    }

    fn chunk_object(self) -> &'static str {
        match self {
            Kind::Chat => "chat.completion.chunk",
            Kind::Text => "text_completion",
        }
    }
}

struct Chunks {
    kind: Kind,
    id: String,
    model: Arc<str>,
    created: u64,
}

impl Chunks {
    fn chunk(&self, text: Option<(&str, bool)>, role: bool, finish: Option<&str>) -> Value {
        let choice = match self.kind {
            Kind::Chat => {
                let mut delta = json!({});
                if role {
                    delta["role"] = json!("assistant");
                }
                if let Some((text, reasoning)) = text {
                    let field = if reasoning { "reasoning_content" } else { "content" };
                    delta[field] = json!(text);
                }
                json!({"index": 0, "delta": delta, "finish_reason": finish, "logprobs": null})
            }
            Kind::Text => json!({
                "index": 0,
                "text": text.map_or("", |(text, _)| text),
                "finish_reason": finish,
                "logprobs": null,
            }),
        };
        self.envelope(json!([choice]))
    }

    fn envelope(&self, choices: Value) -> Value {
        json!({
            "id": self.id,
            "object": self.kind.chunk_object(),
            "created": self.created,
            "model": &*self.model,
            "choices": choices,
        })
    }
}

enum Phase {
    Start,
    Body,
    End,
}

pub fn sse(
    kind: Kind,
    id: String,
    model: Arc<str>,
    events: UnboundedReceiver<Event>,
    include_usage: bool,
) -> Response {
    let chunks = Chunks {
        kind,
        id,
        model,
        created: openai::unix_seconds(),
    };
    let state = (chunks, events, Phase::Start);
    let body = stream::unfold(state, move |(chunks, mut events, phase)| async move {
        let out: Vec<Value> = match phase {
            Phase::End => return None,
            Phase::Start => {
                let first = match chunks.kind {
                    Kind::Chat => vec![chunks.chunk(Some(("", false)), true, None)],
                    Kind::Text => vec![],
                };
                return Some((data(first), (chunks, events, Phase::Body)));
            }
            Phase::Body => match events.recv().await {
                Some(Event::Text { reasoning, text }) => {
                    let piece = vec![chunks.chunk(Some((&text, reasoning)), false, None)];
                    return Some((data(piece), (chunks, events, Phase::Body)));
                }
                Some(Event::Done(Ok(reply))) => {
                    let completion = reply.completion;
                    super::log(chunks.kind, &completion);
                    let finish = openai::finish_reason(completion.finish_reason);
                    let mut out = vec![chunks.chunk(None, false, Some(finish))];
                    if include_usage {
                        let usage = Usage::new(completion.prompt_tokens, completion.generated_tokens);
                        let mut last = chunks.envelope(json!([]));
                        last["usage"] = json!(usage);
                        out.push(last);
                    }
                    out
                }
                Some(Event::Done(Err(message))) => {
                    vec![openai::error_body(&message, "server_error")]
                }
                None => vec![openai::error_body("engine stopped", "server_error")],
            },
        };
        let mut events_out = data(out);
        events_out.push(Ok(SseEvent::default().data("[DONE]")));
        Some((events_out, (chunks, events, Phase::End)))
    })
    .flat_map(stream::iter);
    Sse::new(body)
        .keep_alive(KeepAlive::default())
        .into_response()
}

fn data(values: Vec<Value>) -> Vec<Result<SseEvent, Infallible>> {
    values
        .into_iter()
        .map(|value| Ok(SseEvent::default().data(value.to_string())))
        .collect()
}
