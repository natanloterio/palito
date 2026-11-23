//! Turns generated tokens into a reply: streamed text or tool events, stop
//! sequences and the finish reason. Shared by the blocking generation of
//! `Engine` and the stepped generation of server slots.

use super::{Completion, FinishReason, TokenClock, ToolCompletion, text::TextSink};
use crate::{
    Error, Result,
    stop::{self, StopFilter},
    stream::TextStream,
    tokenizer::Tokenizer,
    tools::{TokenRole, ToolEvent, ToolStream},
};
use std::{cell::Cell, rc::Rc, sync::Arc};

/// Timing of a generation that ran to its end.
#[derive(Clone, Copy, Debug)]
pub(crate) struct RunStats {
    pub cached_tokens: usize,
    pub ttft_ns: u64,
    pub tokens_per_second: f64,
}

#[derive(Clone, Debug)]
pub enum SlotReply {
    Text(Completion),
    Tools(ToolCompletion),
}

pub(crate) trait ReplyOutput {
    /// Err(Cancelled) ends the generation, by a stop sequence or the client.
    fn token(&mut self, token: i32) -> Result<()>;
    /// Releases held text. Called only when generation ended without Cancelled.
    fn drain(&mut self) -> Result<()>;
    /// `stats` is None when generation was cancelled or stopped early.
    fn finish(self: Box<Self>, stats: Option<RunStats>) -> Result<SlotReply>;
}

/// What every output needs to know about its request.
#[derive(Clone)]
pub(super) struct ReplyPlan {
    pub tokenizer: Arc<Tokenizer>,
    pub eog: Arc<[i32]>,
    pub prompt_tokens: usize,
    pub budget: u32,
    pub room: u32,
}

impl ReplyPlan {
    fn finish_reason(&self, generated: &[i32], cancelled: bool, stopped: bool) -> FinishReason {
        let ended = generated.last().is_some_and(|token| self.eog.contains(token));
        if cancelled {
            FinishReason::Cancelled
        } else if stopped || ended {
            FinishReason::Stop
        } else if self.budget < self.room {
            FinishReason::Length
        } else {
            FinishReason::ContextFull
        }
    }

    fn completion(
        &self,
        text: String,
        finish_reason: FinishReason,
        generated: usize,
        stats: Option<RunStats>,
        clock: &TokenClock,
    ) -> Completion {
        let (ttft_ns, tokens_per_second) = match &stats {
            Some(stats) => (stats.ttft_ns, stats.tokens_per_second),
            None => clock.stats(),
        };
        Completion {
            text,
            finish_reason,
            prompt_tokens: self.prompt_tokens,
            cached_tokens: stats.map_or(0, |stats| stats.cached_tokens),
            generated_tokens: generated,
            ttft_ns,
            tokens_per_second,
        }
    }
}

/// Plain text replies.
pub(super) struct TextOutput<S: TextSink> {
    plan: ReplyPlan,
    stop_list: Vec<String>,
    stream: TextStream,
    stops: StopFilter,
    sink: S,
    clock: TokenClock,
    generated: Vec<i32>,
    cancelled: bool,
}

impl<S: TextSink> TextOutput<S> {
    pub(super) fn new(plan: ReplyPlan, stops: &[String], sink: S) -> Self {
        Self {
            plan,
            stop_list: stops.to_vec(),
            stream: TextStream::new(),
            stops: StopFilter::new(stops),
            sink,
            clock: TokenClock::new(),
            generated: Vec::new(),
            cancelled: false,
        }
    }

    pub(super) fn complete(mut self, stats: Option<RunStats>) -> Result<Completion> {
        self.sink.end();
        let mut text = self.plan.tokenizer.detokenize(&self.generated)?;
        if let Some(index) = stop::find(&text, &self.stop_list) {
            text.truncate(index);
        }
        let reason =
            self.plan.finish_reason(&self.generated, self.cancelled, self.stops.stopped());
        Ok(self.plan.completion(text, reason, self.generated.len(), stats, &self.clock))
    }
}

impl<S: TextSink> ReplyOutput for TextOutput<S> {
    fn token(&mut self, token: i32) -> Result<()> {
        self.clock.tick();
        self.generated.push(token);
        let tokenizer = &self.plan.tokenizer;
        let text = self.stream.push(token, |ids: &[i32]| tokenizer.detokenize(ids))?;
        let piece = self.stops.push(&text);
        if !piece.is_empty() && !self.sink.write(&piece) {
            self.cancelled = true;
            return Err(Error::Cancelled);
        }
        if self.stops.stopped() {
            return Err(Error::Cancelled);
        }
        Ok(())
    }

    fn drain(&mut self) -> Result<()> {
        let tokenizer = &self.plan.tokenizer;
        let text = self.stream.finish(|ids: &[i32]| tokenizer.detokenize(ids))?;
        let mut rest = self.stops.push(&text);
        if !self.stops.stopped() {
            rest += &self.stops.finish();
        }
        if !rest.is_empty() {
            self.sink.write(&rest);
        }
        Ok(())
    }

    fn finish(self: Box<Self>, stats: Option<RunStats>) -> Result<SlotReply> {
        (*self).complete(stats).map(SlotReply::Text)
    }
}

/// Replies that may call tools. Applies stop sequences to text.
pub(super) struct ToolOutput<F> {
    plan: ReplyPlan,
    role: Rc<Cell<TokenRole>>,
    stream: ToolStream,
    /// None when stop sequences do not apply (a call is required).
    stops: Option<StopFilter>,
    on_event: F,
    text: String,
    clock: TokenClock,
    generated: Vec<i32>,
    cancelled: bool,
    stopped: bool,
}

impl<F: FnMut(ToolEvent) -> bool> ToolOutput<F> {
    pub(super) fn new(
        plan: ReplyPlan,
        role: Rc<Cell<TokenRole>>,
        stops: Option<&[String]>,
        on_event: F,
    ) -> Self {
        Self {
            plan,
            role,
            stream: ToolStream::new(),
            stops: stops.map(StopFilter::new),
            on_event,
            text: String::new(),
            clock: TokenClock::new(),
            generated: Vec::new(),
            cancelled: false,
            stopped: false,
        }
    }

    pub(super) fn complete(self, stats: Option<RunStats>) -> Result<ToolCompletion> {
        let reason = self.plan.finish_reason(&self.generated, self.cancelled, self.stopped);
        let text = self.text.trim().to_owned();
        let completion =
            self.plan.completion(text, reason, self.generated.len(), stats, &self.clock);
        Ok(ToolCompletion { completion, calls: self.stream.calls().to_vec() })
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

impl<F: FnMut(ToolEvent) -> bool> ReplyOutput for ToolOutput<F> {
    fn token(&mut self, token: i32) -> Result<()> {
        self.clock.tick();
        self.generated.push(token);
        let piece = usize::try_from(token)
            .ok()
            .and_then(|id| self.plan.tokenizer.token_bytes(id));
        let events = self.stream.push(self.role.get(), piece.unwrap_or_default());
        self.deliver(events)
    }

    fn drain(&mut self) -> Result<()> {
        let events = self.stream.finish();
        let delivered = self.deliver(events).and_then(|()| self.release_held());
        match delivered {
            Ok(()) | Err(Error::Cancelled) => Ok(()),
            Err(error) => Err(error),
        }
    }

    fn finish(self: Box<Self>, stats: Option<RunStats>) -> Result<SlotReply> {
        (*self).complete(stats).map(SlotReply::Tools)
    }
}
