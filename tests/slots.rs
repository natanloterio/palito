//! Server slots on a real model. Needs a GGUF in `PALITO_SLOT_MODEL`, or
//! models/compat-qwen3/Qwen3-0.6B-Q4_K_M.gguf.
#![cfg(feature = "cuda")]

use palito::{
    Config, Error, GenerationMode,
    backend::Backend,
    chat::Message,
    cuda::{CudaBackend, CudaMatmul, CudaOptions, open_slot_backend},
    engine::{
        Completion, Device, Engine, EngineOptions, GenerationOptions, SlotEngine, SlotReply,
    },
    slots::{Advance, SlotBackend, SlotRun},
};
use std::path::PathBuf;

fn model() -> PathBuf {
    std::env::var_os("PALITO_SLOT_MODEL").map_or_else(
        || PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("models/compat-qwen3/Qwen3-0.6B-Q4_K_M.gguf"),
        PathBuf::from,
    )
}

fn config() -> Config {
    Config {
        context_tokens: 1024,
        batch_tokens: 512,
        prefill_chunk: 512,
        threads: 4,
    }
}

/// `PALITO_SLOT_PROFILE` picks the matmul profile (e.g. int8-fast for a MoE model).
fn profile() -> CudaMatmul {
    std::env::var("PALITO_SLOT_PROFILE").map_or(CudaMatmul::F32Tiled, |name| {
        clap::ValueEnum::from_str(&name, true).expect("unknown matmul profile")
    })
}

fn options(slots: u32) -> CudaOptions {
    CudaOptions {
        matmul: profile(),
        decode_graph: true,
        prefix_cache: true,
        slots,
        ..CudaOptions::default()
    }
}

fn open(slots: u32) -> Box<dyn SlotBackend> {
    open_slot_backend(&model(), &config(), options(slots)).unwrap()
}

/// A greedy sequence advanced one token at a time.
struct Sequence {
    slot: usize,
    prompt: usize,
    tokens: Vec<i32>,
}

fn begin(backend: &mut dyn SlotBackend, slot: usize, prompt: &[i32]) -> Sequence {
    backend.select_slot(slot).unwrap();
    backend.reset().unwrap();
    let restored = backend.restore_prefix(prompt).unwrap();
    backend
        .evaluate_for_sampling(&prompt[restored..], restored as u32)
        .unwrap();
    backend.store_prefix(prompt).unwrap();
    let first = backend.sample_greedy().unwrap();
    Sequence { slot, prompt: prompt.len(), tokens: vec![first] }
}

fn advance(backend: &mut dyn SlotBackend, sequence: &mut Sequence) {
    backend.select_slot(sequence.slot).unwrap();
    let position = sequence.prompt + sequence.tokens.len() - 1;
    backend
        .evaluate_for_sampling(&[*sequence.tokens.last().unwrap()], position as u32)
        .unwrap();
    let sampled = backend
        .sample_tokens_streaming(1, GenerationMode::FixedLength, &mut |_| Ok(()))
        .unwrap();
    sequence.tokens.extend(sampled);
}

fn run_alone(backend: &mut dyn SlotBackend, slot: usize, prompt: &[i32], count: usize) -> Vec<i32> {
    let mut sequence = begin(backend, slot, prompt);
    while sequence.tokens.len() < count {
        advance(backend, &mut sequence);
    }
    sequence.tokens
}

fn prompt(backend: &dyn SlotBackend, text: &str) -> Vec<i32> {
    backend.tokenize(text).unwrap()
}

const FRANCE: &str = "The capital of France is";
const RAIN: &str = "Write a short haiku about the rain:";

#[test]
#[ignore = "requires a GPU and PALITO_SLOT_MODEL"]
fn interleaved_slots_match_isolated_runs() {
    let mut backend = open(2);
    let (a, b) = (prompt(&*backend, FRANCE), prompt(&*backend, RAIN));
    let alone_a = run_alone(&mut *backend, 0, &a, 32);
    let alone_b = run_alone(&mut *backend, 1, &b, 32);
    backend.abandon_slot(0).unwrap();
    backend.abandon_slot(1).unwrap();
    let mut mixed_a = begin(&mut *backend, 0, &a);
    let mut mixed_b = begin(&mut *backend, 1, &b);
    while mixed_a.tokens.len() < 32 {
        advance(&mut *backend, &mut mixed_a);
        advance(&mut *backend, &mut mixed_b);
    }
    assert_eq!(mixed_a.tokens, alone_a);
    assert_eq!(mixed_b.tokens, alone_b);
}

#[test]
#[ignore = "requires a GPU and PALITO_SLOT_MODEL"]
fn switching_after_an_interrupted_burst_keeps_the_other_slot() {
    let mut backend = open(2);
    let (a, b) = (prompt(&*backend, FRANCE), prompt(&*backend, RAIN));
    let alone_b = run_alone(&mut *backend, 1, &b, 16);
    backend.abandon_slot(1).unwrap();
    let mut other = begin(&mut *backend, 1, &b);
    while other.tokens.len() < 4 {
        advance(&mut *backend, &mut other);
    }
    let cancelled = begin(&mut *backend, 0, &a);
    backend
        .evaluate_for_sampling(&[cancelled.tokens[0]], a.len() as u32)
        .unwrap();
    let mut seen = 0;
    let result = backend.sample_tokens_streaming(8, GenerationMode::FixedLength, &mut |_| {
        seen += 1;
        if seen == 2 { Err(Error::Cancelled) } else { Ok(()) }
    });
    assert!(matches!(result, Err(Error::Cancelled)));
    while other.tokens.len() < 16 {
        advance(&mut *backend, &mut other);
    }
    assert_eq!(other.tokens, alone_b);
}

#[test]
#[ignore = "requires a GPU and PALITO_SLOT_MODEL"]
fn abandoning_a_slot_clears_its_failure_for_the_next() {
    let mut backend = CudaBackend::open(&model(), &config(), options(2)).unwrap();
    let b = prompt(&backend, RAIN);
    let alone_b = run_alone(&mut backend, 1, &b, 8);
    backend.abandon_slot(1).unwrap();
    backend.select_slot(0).unwrap();
    backend.poison_status_for_test().unwrap();
    backend.abandon_slot(0).unwrap();
    assert_eq!(run_alone(&mut backend, 1, &b, 8), alone_b);
}

#[test]
#[ignore = "requires a GPU and PALITO_SLOT_MODEL"]
fn reservation_counts_every_slot() {
    let one = open(1).memory_usage().unwrap().numerical_reservation_bytes;
    let two = open(2).memory_usage().unwrap().numerical_reservation_bytes;
    assert!(one < two);
}

fn engine_options(context: Option<u32>) -> EngineOptions {
    EngineOptions { context_tokens: context, ..EngineOptions::default() }
}

fn text_sink() -> Box<dyn FnMut(&str) -> bool> {
    Box::new(|_| true)
}

fn finish(engine: &mut SlotEngine, mut run: SlotRun) -> Completion {
    loop {
        match run.advance(engine.backend_mut(), true).unwrap() {
            Advance::Working => {}
            Advance::Finished => break,
        }
    }
    match run.into_reply().unwrap() {
        SlotReply::Text(completion) => completion,
        SlotReply::Tools(_) => panic!("expected a text reply"),
    }
}

#[test]
fn cpu_refuses_several_slots() {
    let options = EngineOptions { device: Device::Cpu, ..EngineOptions::default() };
    let error = SlotEngine::open_slots(&model(), &options, 2).err().unwrap();
    assert!(error.to_string().contains("--parallel"));
}

#[test]
#[ignore = "requires a GPU and PALITO_SLOT_MODEL"]
fn context_is_split_between_slots() {
    let engine = SlotEngine::open_slots(&model(), &engine_options(Some(4096)), 4).unwrap();
    assert_eq!(engine.context_tokens(), 1024);
    assert_eq!(engine.slots(), 4);
}

#[test]
#[ignore = "requires a GPU and PALITO_SLOT_MODEL"]
fn prompt_over_the_slot_context_is_refused() {
    let mut engine = SlotEngine::open_slots(&model(), &engine_options(Some(1024)), 4).unwrap();
    let options = GenerationOptions::default();
    let error = engine
        .prepare_complete(&"word ".repeat(600), &options, text_sink())
        .err()
        .unwrap();
    assert!(error.to_string().contains("256-token context"), "{error}");
}

#[test]
#[ignore = "requires a GPU and PALITO_SLOT_MODEL"]
fn stepped_chat_matches_engine_chat_for_short_prompts() {
    let messages = [Message::user("Name three colors.")];
    let options = GenerationOptions {
        max_tokens: Some(48),
        stop: vec!["blue".into()],
        enable_thinking: false,
        ..GenerationOptions::default()
    };
    let expected = {
        let mut engine = Engine::open(&model(), &engine_options(Some(1024))).unwrap();
        engine.chat(&messages, &options, |_| true).unwrap()
    };
    let mut engine = SlotEngine::open_slots(&model(), &engine_options(Some(2048)), 2).unwrap();
    engine.backend_mut().select_slot(0).unwrap();
    let run = engine.prepare_chat(&messages, &options, text_sink()).unwrap();
    let completion = finish(&mut engine, run);
    assert_eq!(completion.text, expected.text);
    assert_eq!(completion.finish_reason, expected.finish_reason);
    assert_eq!(completion.generated_tokens, expected.generated_tokens);
}
