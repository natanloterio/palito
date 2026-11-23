//! `Engine` over a backend with several slots: requests are prepared here and
//! then advanced in turns by the server's scheduler.

use super::{
    Engine, EngineOptions, GenerationOptions, TextRequest,
    open,
    output::{ReplyOutput, TextOutput, ToolOutput},
    text::{TextSink, TrimmedText},
    tools::ToolRequest,
};
use crate::{
    Result,
    chat::Message,
    grammar::LogitConstraint,
    slots::{SlotBackend, SlotRun},
    tools::{ToolEvent, ToolSpec},
};
use std::path::Path;

pub type SlotEngine = Engine<dyn SlotBackend>;

impl SlotEngine {
    pub fn open_slots(path: &Path, options: &EngineOptions, slots: u32) -> Result<Self> {
        let opened = open::open_slots(path, options, slots)?;
        let mut engine = Self::wrap(
            path,
            opened.backend,
            opened.config,
            opened.profile,
            options.chat_template.as_deref(),
        )?;
        engine.notices = opened.notices;
        Ok(engine)
    }

    pub fn slots(&self) -> usize {
        self.executor.backend.slot_count()
    }

    pub fn backend_mut(&mut self) -> &mut dyn SlotBackend {
        &mut *self.executor.backend
    }

    pub fn prepare_chat(
        &mut self,
        messages: &[Message],
        options: &GenerationOptions,
        on_text: Box<dyn FnMut(&str) -> bool>,
    ) -> Result<SlotRun> {
        let prompt = self.render_chat(messages, options.thinking())?;
        self.prepare_text(&prompt, options, TrimmedText::new(on_text), true)
    }

    pub fn prepare_complete(
        &mut self,
        prompt: &str,
        options: &GenerationOptions,
        on_text: Box<dyn FnMut(&str) -> bool>,
    ) -> Result<SlotRun> {
        self.prepare_text(prompt, options, on_text, false)
    }

    pub fn prepare_chat_tools(
        &mut self,
        messages: &[Message],
        spec: &ToolSpec,
        options: &GenerationOptions,
        on_event: Box<dyn FnMut(ToolEvent) -> bool>,
    ) -> Result<SlotRun> {
        let ToolRequest { tokens, budget, room, sampler, gate, role, applies_stops } =
            self.tool_request(messages, spec, options)?;
        let plan = self.reply_plan(tokens.len(), budget, room);
        let stops = applies_stops.then_some(options.stop.as_slice());
        let output: Box<dyn ReplyOutput> = Box::new(ToolOutput::new(plan, role, stops, on_event));
        let gate: Box<dyn LogitConstraint> = Box::new(gate);
        Ok(SlotRun::new(tokens, budget, sampler, Some(gate), output, self.eog.clone()))
    }

    fn prepare_text(
        &mut self,
        prompt: &str,
        options: &GenerationOptions,
        sink: impl TextSink + 'static,
        trim: bool,
    ) -> Result<SlotRun> {
        let TextRequest { tokens, budget, room, sampler, constraint } =
            self.text_request(prompt, options)?;
        let plan = self.reply_plan(tokens.len(), budget, room);
        let output = TextOutput::new(plan, &options.stop, sink);
        let output: Box<dyn ReplyOutput> =
            if trim { Box::new(output.trimmed()) } else { Box::new(output) };
        let constraint = constraint.map(|c| Box::new(c) as Box<dyn LogitConstraint>);
        Ok(SlotRun::new(tokens, budget, sampler, constraint, output, self.eog.clone()))
    }
}
