//! One request generating in steps, so slots can take turns on the device.
//! Follows the rules of `Executor::generate_streaming` and
//! `generate_constrained`, one turn at a time.

use super::{PROMPT_TURN_TOKENS, SOLO_BURST_TOKENS, SlotBackend};
use crate::{
    Error, GenerationMode, Result,
    engine::{ReplyOutput, RunStats, SlotReply},
    grammar::LogitConstraint,
    sampling::Sampler,
};
use std::{sync::Arc, time::Instant};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Advance {
    Working,
    Finished,
}

enum Phase {
    /// `next` is None until the prefix cache was consulted.
    Prompt { next: Option<usize> },
    Generate,
    Done,
}

pub struct SlotRun {
    prompt: Vec<i32>,
    budget: usize,
    sampler: Sampler,
    constraint: Option<Box<dyn LogitConstraint>>,
    output: Box<dyn ReplyOutput>,
    eog: Arc<[i32]>,
    phase: Phase,
    generated: usize,
    last: i32,
    cancelled: bool,
    cached: usize,
    start: Instant,
    ttft_ns: u64,
    decode_start: Option<Instant>,
    decode_ns: u64,
}

impl SlotRun {
    pub(crate) fn new(
        prompt: Vec<i32>,
        budget: u32,
        sampler: Sampler,
        constraint: Option<Box<dyn LogitConstraint>>,
        output: Box<dyn ReplyOutput>,
        eog: Arc<[i32]>,
    ) -> Self {
        Self {
            prompt,
            budget: budget as usize,
            sampler,
            constraint,
            output,
            eog,
            phase: Phase::Prompt { next: None },
            generated: 0,
            last: 0,
            cancelled: false,
            cached: 0,
            start: Instant::now(),
            ttft_ns: 0,
            decode_start: None,
            decode_ns: 0,
        }
    }

    pub fn prompt(&self) -> &[i32] {
        &self.prompt
    }

    /// One turn on the slot already selected. `alone` allows a short burst.
    pub fn advance(&mut self, backend: &mut dyn SlotBackend, alone: bool) -> Result<Advance> {
        match self.phase {
            Phase::Prompt { next } => self.prompt_turn(backend, next),
            Phase::Generate => self.generate_turn(backend, alone),
            Phase::Done => Ok(Advance::Finished),
        }
    }

    pub fn into_reply(self) -> Result<SlotReply> {
        let stats = (!self.cancelled).then(|| RunStats {
            cached_tokens: self.cached,
            ttft_ns: self.ttft_ns,
            tokens_per_second: if self.decode_ns == 0 {
                0.0
            } else {
                (self.generated - 1) as f64 * 1e9 / self.decode_ns as f64
            },
        });
        self.output.finish(stats)
    }

    /// Greedy without a constraint needs no host logits.
    fn token_only(&self) -> bool {
        self.sampler.is_greedy() && self.constraint.is_none()
    }

    fn prompt_turn(
        &mut self,
        backend: &mut dyn SlotBackend,
        next: Option<usize>,
    ) -> Result<Advance> {
        let next = match next {
            Some(next) => next,
            None => self.restore(backend)?,
        };
        let turn = PROMPT_TURN_TOKENS.min(backend.info().batch_tokens) as usize;
        // Blocks aligned to position 0, so a restored prefix evaluates the same blocks.
        let end = ((next / turn + 1) * turn).min(self.prompt.len());
        let last = end == self.prompt.len();
        if last && !self.token_only() {
            backend.evaluate(&self.prompt[next..end], next as u32)?;
        } else {
            backend.evaluate_for_sampling(&self.prompt[next..end], next as u32)?;
        }
        if !last {
            self.phase = Phase::Prompt { next: Some(end) };
            return Ok(Advance::Working);
        }
        backend.store_prefix(&self.prompt)?;
        let first = if self.token_only() {
            backend.sample_greedy()?
        } else {
            self.draw(backend)?
        };
        self.ttft_ns = self.start.elapsed().as_nanos() as u64;
        self.decode_start = Some(Instant::now());
        self.phase = Phase::Generate;
        self.deliver(first)?;
        self.after_tokens()
    }

    fn restore(&mut self, backend: &mut dyn SlotBackend) -> Result<usize> {
        backend.reset()?;
        let restored = backend.restore_prefix(&self.prompt)?;
        if restored > self.prompt.len() {
            return Err(Error::Backend("prefix restoration exceeds prompt length".into()));
        }
        self.cached = restored;
        Ok(restored)
    }

    fn generate_turn(&mut self, backend: &mut dyn SlotBackend, alone: bool) -> Result<Advance> {
        let position = (self.prompt.len() + self.generated - 1) as u32;
        if self.token_only() {
            backend.evaluate_for_sampling(&[self.last], position)?;
            self.sample_burst(backend, alone)?;
        } else {
            backend.evaluate(&[self.last], position)?;
            let token = self.draw(backend)?;
            self.deliver(token)?;
        }
        self.after_tokens()
    }

    fn sample_burst(&mut self, backend: &mut dyn SlotBackend, alone: bool) -> Result<()> {
        let remaining = self.budget - self.generated;
        let maximum = if alone { remaining.min(SOLO_BURST_TOKENS as usize) } else { 1 };
        let vocabulary = backend.info().vocab_tokens;
        let (output, generated, last) = (&mut self.output, &mut self.generated, &mut self.last);
        let result = backend.sample_tokens_streaming(
            maximum,
            GenerationMode::UntilEog,
            &mut |token| {
                *generated += 1;
                *last = token;
                output.token(token)
            },
        );
        match result {
            Ok(sampled)
                if sampled.is_empty()
                    || sampled.len() > maximum
                    || sampled.iter().any(|&t| t < 0 || t as u32 >= vocabulary) =>
            {
                Err(Error::Backend("invalid sampled token batch".into()))
            }
            Ok(_) => Ok(()),
            Err(Error::Cancelled) => {
                self.cancelled = true;
                Ok(())
            }
            Err(error) => Err(error),
        }
    }

    fn draw(&mut self, backend: &dyn SlotBackend) -> Result<i32> {
        let logits = backend.logits()?;
        if logits.len() != backend.info().vocab_tokens as usize {
            return Err(Error::Backend("logits do not cover the vocabulary".into()));
        }
        let sampler = &mut self.sampler;
        match &mut self.constraint {
            None => sampler.sample(logits),
            Some(constraint) => constraint.sample(logits, &mut |masked| sampler.sample(masked)),
        }
    }

    fn deliver(&mut self, token: i32) -> Result<()> {
        self.generated += 1;
        self.last = token;
        match self.output.token(token) {
            Err(Error::Cancelled) => {
                self.cancelled = true;
                Ok(())
            }
            other => other,
        }
    }

    fn after_tokens(&mut self) -> Result<Advance> {
        if !self.cancelled
            && self.generated < self.budget
            && !self.eog.contains(&self.last)
        {
            return Ok(Advance::Working);
        }
        if !self.cancelled {
            self.output.drain()?;
        }
        if self.generated > 1 {
            self.decode_ns = self
                .decode_start
                .map_or(0, |start| start.elapsed().as_nanos() as u64);
        }
        self.phase = Phase::Done;
        Ok(Advance::Finished)
    }
}

#[cfg(test)]
mod tests;
