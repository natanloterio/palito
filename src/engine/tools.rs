//! Chat with tool calling. Separate from `Engine::generate`, so requests
//! without tools keep their decode path.

use super::{Completion, Engine, FinishReason, GenerationOptions, TokenClock, decode_rate};
use crate::{
    Error, GenerationMode, Result,
    chat::{Message, ToolFormat},
    grammar::{Constraint, LogitConstraint},
    sampling::Sampler,
    stop::{self, StopFilter},
    tools::{
        TokenRole, ToolCall, ToolEvent, ToolGate, ToolMode, ToolSpec, ToolStream, ToolTokens,
        call_grammar::call_grammar,
    },
};
use std::cell::Cell;

const CALL_OPEN: &str = "<tool_call>";
const CALL_CLOSE: &str = "</tool_call>";
const THINK_OPEN: &str = "<think>";
const THINK_CLOSE: &str = "</think>";

#[derive(Clone, Debug)]
pub struct ToolCompletion {
    /// `text` holds the reply without tool call markup.
    pub completion: Completion,
    pub calls: Vec<ToolCall>,
}

/// Publishes each token's role to the token callback, which runs while the
/// executor holds the gate.
struct RoleGate<'a> {
    gate: ToolGate,
    role: &'a Cell<TokenRole>,
}

impl LogitConstraint for RoleGate<'_> {
    fn sample(
        &mut self,
        logits: &[f32],
        pick: &mut dyn FnMut(&[f32]) -> Result<i32>,
    ) -> Result<i32> {
        let token = self.gate.sample(logits, pick)?;
        self.role.set(self.gate.last_role());
        Ok(token)
    }
}

/// Applies stop sequences to text and forwards events to the caller.
struct EventSink<'s, F> {
    stream: ToolStream,
    /// None when stop sequences do not apply (a call is required).
    stops: Option<StopFilter<'s>>,
    on_event: F,
    text: String,
    cancelled: bool,
    stopped: bool,
}

impl<F: FnMut(ToolEvent) -> bool> EventSink<'_, F> {
    fn token(&mut self, role: TokenRole, piece: &[u8]) -> Result<()> {
        let events = self.stream.push(role, piece);
        self.deliver(events)
    }

    fn finish(&mut self) -> Result<()> {
        let events = self.stream.finish();
        self.deliver(events)?;
        self.release_held()
    }

    fn deliver(&mut self, events: Vec<ToolEvent>) -> Result<()> {
        for event in events {
            match event {
                ToolEvent::Text(text) => self.text_piece(text)?,
                other => {
                    self.release_held()?;
                    self.send(other)?;
                }
            }
        }
        Ok(())
    }

    fn text_piece(&mut self, text: String) -> Result<()> {
        let Some(stops) = &mut self.stops else {
            return self.send_text(text);
        };
        let released = stops.push(&text);
        let stopped = stops.stopped();
        if !released.is_empty() {
            self.send_text(released)?;
        }
        if stopped {
            self.stopped = true;
            return Err(Error::Cancelled);
        }
        Ok(())
    }

    /// Text held as a possible stop sequence, before a call or the end.
    fn release_held(&mut self) -> Result<()> {
        let held = self.stops.as_mut().map(StopFilter::finish).unwrap_or_default();
        if held.is_empty() { Ok(()) } else { self.send_text(held) }
    }

    fn send_text(&mut self, text: String) -> Result<()> {
        self.text.push_str(&text);
        self.send(ToolEvent::Text(text))
    }

    fn send(&mut self, event: ToolEvent) -> Result<()> {
        if (self.on_event)(event) {
            Ok(())
        } else {
            self.cancelled = true;
            Err(Error::Cancelled)
        }
    }
}

impl Engine {
    /// The model's tool call format, when Palito can generate it.
    pub fn tool_format(&self) -> Option<ToolFormat> {
        let format = self.template.as_ref()?.tool_format()?;
        self.single_token(CALL_OPEN)?;
        self.single_token(CALL_CLOSE)?;
        Some(format)
    }

    pub fn supports_tools(&self) -> bool {
        self.tool_format().is_some()
    }

    /// The prompt `chat_tools` would evaluate.
    pub fn render_chat_tools(
        &self,
        messages: &[Message],
        spec: &ToolSpec,
        enable_thinking: bool,
    ) -> Result<String> {
        let template = self.template.as_ref().ok_or_else(|| {
            Error::Invalid("model has no chat template; use plain completion".into())
        })?;
        template.render_tools(messages, spec.template_tools(), enable_thinking)
    }

    /// Generates an assistant reply that may call the tools in `spec`.
    /// `on_event` gets text and calls as they are generated; false stops.
    pub fn chat_tools(
        &mut self,
        messages: &[Message],
        spec: &ToolSpec,
        options: &GenerationOptions,
        on_event: impl FnMut(ToolEvent) -> bool,
    ) -> Result<ToolCompletion> {
        if options.json_schema.is_some() {
            return Err(Error::Invalid("tools cannot be combined with JSON output".into()));
        }
        stop::validate(&options.stop)?;
        let mut sampler = Sampler::new(options.sampling)?;
        let prompt = self.render_chat_tools(messages, spec, options.enable_thinking)?;
        let tokens = self.executor.tokenize(&prompt)?;
        let (budget, room) = self.reply_budget(tokens.len(), options.max_tokens)?;
        let role = Cell::new(TokenRole::Text);
        let mut gate = RoleGate { gate: self.tool_gate(spec, &prompt)?, role: &role };
        for warning in &spec.warnings {
            eprintln!("warning: {warning}");
        }
        let applies_stops = matches!(spec.mode, ToolMode::Auto | ToolMode::Forbidden);
        let mut sink = EventSink {
            stream: ToolStream::new(),
            stops: applies_stops.then(|| StopFilter::new(&options.stop)),
            on_event,
            text: String::new(),
            cancelled: false,
            stopped: false,
        };
        let mut clock = TokenClock::new();
        let mut generated = Vec::new();
        let tokenizer = &self.tokenizer;
        let result = self.executor.generate_constrained(
            &tokens,
            budget,
            GenerationMode::UntilEog,
            &mut sampler,
            Some(&mut gate as &mut dyn LogitConstraint),
            |token| {
                clock.tick();
                generated.push(token);
                let piece = usize::try_from(token).ok().and_then(|id| tokenizer.token_bytes(id));
                sink.token(role.get(), piece.unwrap_or_default())
            },
        );
        let sample = match result {
            Ok(sample) => Some(sample),
            Err(Error::Cancelled) => None,
            Err(error) => return Err(error),
        };
        if sample.is_some() {
            match sink.finish() {
                Ok(()) | Err(Error::Cancelled) => {}
                Err(error) => return Err(error),
            }
        }
        let ended = generated.last().is_some_and(|&t| self.executor.is_eog(t));
        let finish_reason = if sink.cancelled {
            FinishReason::Cancelled
        } else if sink.stopped || ended {
            FinishReason::Stop
        } else if budget < room {
            FinishReason::Length
        } else {
            FinishReason::ContextFull
        };
        let (ttft_ns, tokens_per_second) = match &sample {
            Some(sample) => (sample.ttft_ns, decode_rate(sample)),
            None => clock.stats(),
        };
        Ok(ToolCompletion {
            completion: Completion {
                text: sink.text.trim().to_owned(),
                finish_reason,
                prompt_tokens: tokens.len(),
                cached_tokens: sample.as_ref().map_or(0, |s| s.cached_prompt_tokens),
                generated_tokens: generated.len(),
                ttft_ns,
                tokens_per_second,
            },
            calls: sink.stream.calls().to_vec(),
        })
    }

    fn reply_budget(&self, prompt: usize, max_tokens: Option<u32>) -> Result<(u32, u32)> {
        let context = self.context_tokens() as usize;
        if prompt == 0 || prompt >= context {
            return Err(Error::Invalid(format!(
                "prompt of {prompt} tokens does not fit in the {context}-token context"
            )));
        }
        let room = (context + 1 - prompt) as u32;
        let budget = max_tokens.map_or(room, |max| max.min(room));
        if budget == 0 {
            return Err(Error::Invalid("max_tokens must be positive".into()));
        }
        Ok((budget, room))
    }

    fn tool_gate(&mut self, spec: &ToolSpec, prompt: &str) -> Result<ToolGate> {
        let tokens = self.tool_tokens()?;
        let call = self.call_constraint(spec)?;
        let opens_reasoning = prompt.trim_end().ends_with(THINK_OPEN);
        Ok(ToolGate::new(tokens, call, spec.mode.clone(), spec.parallel, opens_reasoning))
    }

    fn call_constraint(&mut self, spec: &ToolSpec) -> Result<Option<Constraint>> {
        if spec.mode == ToolMode::Forbidden {
            return Ok(None);
        }
        let grammar = call_grammar(&spec.defs, &spec.mode);
        self.json_decoder()?.lark_constraint(&grammar).map(Some)
    }

    fn tool_tokens(&mut self) -> Result<ToolTokens> {
        if self.tool_tokens.is_none() {
            self.tool_tokens = Some(self.find_tool_tokens()?);
        }
        self.tool_tokens
            .clone()
            .ok_or_else(|| Error::Backend("tool tokens missing".into()))
    }

    fn find_tool_tokens(&self) -> Result<ToolTokens> {
        let required = |text: &str| {
            self.single_token(text)
                .ok_or_else(|| Error::Invalid(format!("model has no single {text} token")))
        };
        let vocab = self.executor.info().vocab_tokens as usize;
        Ok(ToolTokens {
            call_open: required(CALL_OPEN)?,
            call_close: required(CALL_CLOSE)?,
            think_open: self.single_token(THINK_OPEN),
            think_close: self.single_token(THINK_CLOSE),
            eog: (0..vocab as i32).filter(|&id| self.executor.is_eog(id)).collect(),
            whitespace: (0..vocab)
                .map(|id| {
                    self.tokenizer
                        .token_bytes(id)
                        .is_some_and(|b| !b.is_empty() && b.iter().all(u8::is_ascii_whitespace))
                })
                .collect(),
        })
    }

    /// The ID of `text` when it is one vocabulary token (BOS/EOS aside).
    fn single_token(&self, text: &str) -> Option<i32> {
        let added = self.executor.tokenize("").ok()?;
        let ids = self.executor.tokenize(text).ok()?;
        let mut own = ids.into_iter().filter(|id| !added.contains(id));
        let token = own.next()?;
        own.next().is_none().then_some(token)
    }
}
