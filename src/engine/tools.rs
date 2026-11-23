//! Chat with tool calling. Separate from `Engine::generate`, so requests
//! without tools keep their decode path.

use super::{
    Completion, Engine, GenerationOptions, ReplyOutput,
    output::ToolOutput,
    run_stats,
};
use crate::{
    Error, GenerationMode, Result,
    backend::Backend,
    chat::{Message, ToolFormat},
    grammar::{Constraint, LogitConstraint},
    sampling::Sampler,
    stop,
    tools::{
        TokenRole, ToolCall, ToolEvent, ToolGate, ToolMode, ToolSpec, ToolTokens,
        call_grammar::call_grammar,
    },
};
use std::{cell::Cell, rc::Rc};

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
pub(super) struct RoleGate {
    gate: ToolGate,
    role: Rc<Cell<TokenRole>>,
}

impl LogitConstraint for RoleGate {
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

/// A validated request with tools.
pub(super) struct ToolRequest {
    pub tokens: Vec<i32>,
    pub budget: u32,
    pub room: u32,
    pub sampler: Sampler,
    pub gate: RoleGate,
    pub role: Rc<Cell<TokenRole>>,
    pub applies_stops: bool,
}

impl<B: Backend + ?Sized> Engine<B> {
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
        let ToolRequest { tokens, budget, room, mut sampler, mut gate, role, applies_stops } =
            self.tool_request(messages, spec, options)?;
        let plan = self.reply_plan(tokens.len(), budget, room);
        let stops = applies_stops.then_some(options.stop.as_slice());
        let mut output = ToolOutput::new(plan, role, stops, on_event);
        let result = self.executor.generate_constrained(
            &tokens,
            budget,
            GenerationMode::UntilEog,
            &mut sampler,
            Some(&mut gate as &mut dyn LogitConstraint),
            |token| output.token(token),
        );
        let sample = match result {
            Ok(sample) => Some(sample),
            Err(Error::Cancelled) => None,
            Err(error) => return Err(error),
        };
        if sample.is_some() {
            output.drain()?;
        }
        output.complete(sample.as_ref().map(run_stats))
    }

    pub(super) fn tool_request(
        &mut self,
        messages: &[Message],
        spec: &ToolSpec,
        options: &GenerationOptions,
    ) -> Result<ToolRequest> {
        if options.json_schema.is_some() {
            return Err(Error::Invalid("tools cannot be combined with JSON output".into()));
        }
        stop::validate(&options.stop)?;
        let sampler = Sampler::new(options.sampling)?;
        let prompt = self.render_chat_tools(messages, spec, options.enable_thinking)?;
        let tokens = self.executor.tokenize(&prompt)?;
        let (budget, room) = self.reply_budget(tokens.len(), options.max_tokens)?;
        let role = Rc::new(Cell::new(TokenRole::Text));
        let gate = RoleGate { gate: self.tool_gate(spec, &prompt)?, role: role.clone() };
        for warning in &spec.warnings {
            eprintln!("warning: {warning}");
        }
        let applies_stops = matches!(spec.mode, ToolMode::Auto | ToolMode::Forbidden);
        Ok(ToolRequest { tokens, budget, room, sampler, gate, role, applies_stops })
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
