use super::*;
use crate::{
    Config, Error, Executor, GenerationMode,
    backend::{Backend, Capabilities, ModelInfo},
    engine::{Completion, FinishReason, ReplyOutput, RunStats, SlotReply},
    sampling::{Sampler, Sampling},
};
use std::{cell::RefCell, rc::Rc, sync::Arc};

const EOG: i32 = 1;
const VOCABULARY: usize = 32;

/// Deterministic backend: the next token depends only on the position.
struct FakeSlots {
    info: ModelInfo,
    position: u32,
    restore: usize,
    eog_at: Option<u32>,
    evaluated: Vec<(usize, u32)>,
    sample_maximums: Vec<usize>,
    logits: Vec<f32>,
}

impl FakeSlots {
    fn new() -> Self {
        Self {
            info: ModelInfo {
                backend: "test".into(),
                revision: "test".into(),
                architecture: "test".into(),
                tensor_types: vec![],
                weight_bytes: 0,
                parameters: 0,
                context_tokens: 4096,
                batch_tokens: 512,
                vocab_tokens: VOCABULARY as u32,
                capabilities: Capabilities { token_step: true, chunked_prefill: true },
            },
            position: 0,
            restore: 0,
            eog_at: None,
            evaluated: vec![],
            sample_maximums: vec![],
            logits: vec![0.0; VOCABULARY],
        }
    }

    fn token_at(&self, position: u32) -> i32 {
        if self.eog_at == Some(position) { EOG } else { (position % 29) as i32 + 2 }
    }

    fn evaluated_prompt(&self) -> Vec<(usize, u32)> {
        self.evaluated.iter().copied().filter(|&(count, _)| count > 1).collect()
    }
}

impl Backend for FakeSlots {
    fn info(&self) -> &ModelInfo {
        &self.info
    }
    fn logits(&self) -> Result<&[f32]> {
        Ok(&self.logits)
    }
    fn tokenize(&self, _: &str) -> Result<Vec<i32>> {
        Ok(vec![2])
    }
    fn detokenize(&self, _: &[i32]) -> Result<String> {
        Ok(String::new())
    }
    fn reset(&mut self) -> Result<()> {
        self.position = 0;
        Ok(())
    }
    fn evaluate(&mut self, tokens: &[i32], position: u32) -> Result<()> {
        assert_eq!(position, self.position, "evaluation out of order");
        self.evaluated.push((tokens.len(), position));
        self.position += tokens.len() as u32;
        let next = self.token_at(self.position) as usize;
        self.logits = vec![0.0; VOCABULARY];
        self.logits[next] = 1.0;
        Ok(())
    }
    fn sample_greedy(&mut self) -> Result<i32> {
        Ok(self.token_at(self.position))
    }
    fn is_eog(&self, token: i32) -> bool {
        token == EOG
    }
    fn restore_prefix(&mut self, _: &[i32]) -> Result<usize> {
        self.position = self.restore as u32;
        Ok(self.restore)
    }
    fn sample_tokens_streaming(
        &mut self,
        maximum: usize,
        mode: GenerationMode,
        sink: &mut dyn FnMut(i32) -> Result<()>,
    ) -> Result<Vec<i32>> {
        self.sample_maximums.push(maximum);
        let mut tokens = vec![];
        loop {
            let token = self.token_at(self.position);
            tokens.push(token);
            sink(token)?;
            if tokens.len() == maximum || (mode == GenerationMode::UntilEog && token == EOG) {
                return Ok(tokens);
            }
            self.position += 1;
        }
    }
}

impl SlotBackend for FakeSlots {
    fn slot_count(&self) -> usize {
        1
    }
    fn select_slot(&mut self, _: usize) -> Result<()> {
        Ok(())
    }
    fn restorable(&self, _: usize, _: &[i32]) -> usize {
        self.restore
    }
    fn abandon_slot(&mut self, _: usize) -> Result<()> {
        Ok(())
    }
    fn device_alive(&mut self) -> Result<()> {
        Ok(())
    }
}

#[derive(Clone, Default)]
struct Record {
    tokens: Rc<RefCell<Vec<i32>>>,
    drained: Rc<RefCell<bool>>,
}

struct RecordOutput {
    record: Record,
    cancel_at: Option<usize>,
}

impl ReplyOutput for RecordOutput {
    fn token(&mut self, token: i32) -> Result<()> {
        self.record.tokens.borrow_mut().push(token);
        if self.cancel_at == Some(self.record.tokens.borrow().len()) {
            return Err(Error::Cancelled);
        }
        Ok(())
    }
    fn drain(&mut self) -> Result<()> {
        *self.record.drained.borrow_mut() = true;
        Ok(())
    }
    fn finish(self: Box<Self>, stats: Option<RunStats>) -> Result<SlotReply> {
        Ok(SlotReply::Text(Completion {
            text: String::new(),
            finish_reason: FinishReason::Stop,
            prompt_tokens: 0,
            cached_tokens: stats.map_or(0, |stats| stats.cached_tokens),
            generated_tokens: self.record.tokens.borrow().len(),
            ttft_ns: 0,
            tokens_per_second: 0.0,
        }))
    }
}

fn prompt_of(count: usize) -> Vec<i32> {
    (0..count).map(|i| (i % 29) as i32 + 2).collect()
}

fn run_with(
    prompt: Vec<i32>,
    budget: u32,
    sampling: Sampling,
    cancel_at: Option<usize>,
) -> (SlotRun, Record) {
    let record = Record::default();
    let output = RecordOutput { record: record.clone(), cancel_at };
    let eog: Arc<[i32]> = Arc::from([EOG]);
    let run = SlotRun::new(prompt, budget, Sampler::new(sampling).unwrap(), None, Box::new(output), eog);
    (run, record)
}

fn setup(prompt: usize, budget: u32) -> (FakeSlots, SlotRun, Record) {
    let (run, record) = run_with(prompt_of(prompt), budget, Sampling::GREEDY, None);
    (FakeSlots::new(), run, record)
}

fn drive(fake: &mut FakeSlots, run: &mut SlotRun, alone: bool) {
    for _ in 0..1000 {
        if matches!(run.advance(fake, alone).unwrap(), Advance::Finished) {
            return;
        }
    }
    panic!("run did not finish");
}

/// Evaluates the prompt, leaving the run about to generate.
fn ready_to_generate(budget: u32) -> (FakeSlots, SlotRun, Record) {
    let (mut fake, mut run, record) = setup(4, budget);
    assert!(matches!(run.advance(&mut fake, false).unwrap(), Advance::Working));
    (fake, run, record)
}

#[test]
fn prompt_is_evaluated_in_512_token_turns_from_zero() {
    let (mut fake, mut run, _) = setup(1300, 2);
    for _ in 0..3 {
        run.advance(&mut fake, false).unwrap();
    }
    assert_eq!(fake.evaluated_prompt(), [(512, 0), (512, 512), (276, 1024)]);
}

#[test]
fn exact_multiple_of_512_has_no_empty_turn() {
    let (mut fake, mut run, _) = setup(1024, 2);
    run.advance(&mut fake, false).unwrap();
    run.advance(&mut fake, false).unwrap();
    assert_eq!(fake.evaluated_prompt(), [(512, 0), (512, 512)]);
    assert_eq!(run.generated, 1, "the last prompt turn draws the first token");
}

#[test]
fn restored_prefix_keeps_the_alignment() {
    let (mut cold, mut cold_run, cold_record) = setup(1300, 6);
    drive(&mut cold, &mut cold_run, false);
    let (mut fake, mut run, record) = setup(1300, 6);
    fake.restore = 512;
    drive(&mut fake, &mut run, false);
    assert_eq!(fake.evaluated_prompt(), [(512, 512), (276, 1024)]);
    assert_eq!(*record.tokens.borrow(), *cold_record.tokens.borrow());
}

#[test]
fn generates_one_token_per_turn_when_shared() {
    let (mut fake, mut run, _) = ready_to_generate(16);
    run.advance(&mut fake, false).unwrap();
    assert_eq!(fake.sample_maximums, [1]);
}

#[test]
fn bursts_at_most_8_tokens_when_alone() {
    let (mut fake, mut run, _) = ready_to_generate(32);
    run.advance(&mut fake, true).unwrap();
    assert_eq!(fake.sample_maximums, [8]);
}

#[test]
fn burst_never_exceeds_the_budget() {
    let (mut fake, mut run, _) = ready_to_generate(4);
    run.advance(&mut fake, true).unwrap();
    assert_eq!(fake.sample_maximums, [3]);
}

#[test]
fn sampling_uses_host_logits_one_token_at_a_time() {
    let sampling = Sampling { temperature: 0.7, seed: Some(7), ..Sampling::GREEDY };
    let (mut run, record) = run_with(prompt_of(4), 5, sampling, None);
    let mut fake = FakeSlots::new();
    drive(&mut fake, &mut run, true);
    assert!(fake.sample_maximums.is_empty());
    assert_eq!(record.tokens.borrow().len(), 5);
}

#[test]
fn stops_at_end_of_generation_and_at_budget() {
    let (mut fake, mut run, record) = setup(4, 100);
    fake.eog_at = Some(7);
    drive(&mut fake, &mut run, false);
    assert_eq!(*record.tokens.borrow().last().unwrap(), EOG);
    assert!(*record.drained.borrow());
    let (mut fake, mut run, record) = setup(4, 3);
    drive(&mut fake, &mut run, false);
    assert_eq!(record.tokens.borrow().len(), 3);
}

#[test]
fn cancelled_output_finishes_without_error() {
    let (mut run, record) = run_with(prompt_of(4), 20, Sampling::GREEDY, Some(3));
    let mut fake = FakeSlots::new();
    drive(&mut fake, &mut run, false);
    assert_eq!(record.tokens.borrow().len(), 3);
    assert!(!*record.drained.borrow());
    assert!(run.into_reply().is_ok());
}

#[test]
fn stepped_path_matches_the_executor() {
    let config = Config { context_tokens: 4096, batch_tokens: 512, prefill_chunk: 512, threads: 1 };
    let prompt = prompt_of(600);
    for alone in [false, true] {
        let mut executor = Executor::new(FakeSlots::new(), config.clone()).unwrap();
        let greedy = executor.generate(&prompt, 20, GenerationMode::UntilEog).unwrap();
        let (mut run, record) = run_with(prompt.clone(), 20, Sampling::GREEDY, None);
        drive(&mut FakeSlots::new(), &mut run, alone);
        assert_eq!(*record.tokens.borrow(), greedy.generated_tokens);

        let sampling = Sampling { temperature: 0.8, seed: Some(11), ..Sampling::GREEDY };
        let mut sampler = Sampler::new(sampling.clone()).unwrap();
        let mut executor = Executor::new(FakeSlots::new(), config.clone()).unwrap();
        let sampled = executor
            .generate_sampled(&prompt, 20, GenerationMode::UntilEog, &mut sampler, |_| Ok(()))
            .unwrap();
        let (mut run, record) = run_with(prompt.clone(), 20, sampling, None);
        drive(&mut FakeSlots::new(), &mut run, alone);
        assert_eq!(*record.tokens.borrow(), sampled.generated_tokens);
    }
}
