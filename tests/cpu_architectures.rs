//! The default CPU device must run every layout the CPU reference path supports.
//! Set PALITO_TEST_MODELS to the models directory; PALITO_TINYLLAMA_GGUF adds a
//! legacy SentencePiece conversion without `tokenizer.ggml.pre`.

use palito::{
    Config,
    backend::Backend,
    engine::{Device, Engine, EngineOptions, GenerationOptions},
    reference::ReferenceBackend,
};
use std::{
    io::Write,
    path::{Path, PathBuf},
    process::{Command, Stdio},
};

const GREEDY_TOKENS: usize = 16;
const PROMPT: &str = "The capital of France is";

fn models() -> PathBuf {
    std::env::var_os("PALITO_TEST_MODELS").map_or_else(
        || Path::new(env!("CARGO_MANIFEST_DIR")).join("models"),
        PathBuf::from,
    )
}

fn assert_engine_generates(path: &Path) {
    let options = EngineOptions {
        device: Device::Cpu,
        context_tokens: Some(512),
        ..EngineOptions::default()
    };
    let mut engine = Engine::open(path, &options).unwrap();
    assert_eq!(engine.profile(), "cpu");
    let generation = GenerationOptions {
        max_tokens: Some(4),
        ..GenerationOptions::default()
    };
    let reply = engine.complete(PROMPT, &generation, |_| true).unwrap();
    assert!(!reply.text.is_empty());
}

/// Prints a line that scripts can compare with another engine's greedy output.
fn print_greedy(name: &str, path: &Path) {
    let config = Config {
        context_tokens: 512,
        batch_tokens: 128,
        prefill_chunk: 64,
        threads: 8,
    };
    let mut backend = ReferenceBackend::open(path, &config).unwrap();
    let prompt = backend.tokenize(PROMPT).unwrap();
    backend.evaluate(&prompt, 0).unwrap();
    let mut generated = Vec::with_capacity(GREEDY_TOKENS);
    for step in 0..GREEDY_TOKENS {
        let token = backend.sample_greedy().unwrap();
        generated.push(token);
        backend.evaluate(&[token], (prompt.len() + step) as u32).unwrap();
    }
    println!("GREEDY {name} {prompt:?} {generated:?}");
}

fn assert_cli_generates(path: &Path) {
    let mut child = Command::new(env!("CARGO_BIN_EXE_palito"))
        .args(["chat", "--backend", "cpu", "--context", "512", "--tokens", "4", "--model"])
        .arg(path)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    child
        .stdin
        .take()
        .unwrap()
        .write_all(b"What is the capital of France?\n/exit\n")
        .unwrap();
    let output = child.wait_with_output().unwrap();
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(output.status.success(), "{stderr}");
    assert!(!output.stdout.is_empty());
}

fn check(name: &str, path: &Path) {
    assert_engine_generates(path);
    assert_cli_generates(path);
    print_greedy(name, path);
}

#[test]
#[ignore = "requires compat-qwen25 Q4_K_M"]
fn qwen2_runs_on_the_cpu_device() {
    let path = models().join("compat-qwen25/qwen2.5-0.5b-instruct-q4_k_m.gguf");
    check("qwen25", &path);
}

#[test]
#[ignore = "requires compat-llama32 Q8_0"]
fn llama_with_rope_freqs_runs_on_the_cpu_device() {
    let path = models().join("compat-llama32/Llama-3.2-1B-Instruct-Q8_0.gguf");
    check("llama32", &path);
}

#[test]
#[ignore = "requires PALITO_TINYLLAMA_GGUF"]
fn legacy_sentencepiece_llama_runs_on_the_cpu_device() {
    let path = PathBuf::from(std::env::var_os("PALITO_TINYLLAMA_GGUF").unwrap());
    check("tinyllama", &path);
}

#[test]
#[ignore = "requires local compat GGUFs under models/"]
fn cpu_fast_path_matches_reference_across_shared_architectures() {
    use palito::cpu::{CpuBackend, has_fast_path};

    let config = Config {
        context_tokens: 512,
        batch_tokens: 128,
        prefill_chunk: 64,
        threads: 4,
    };
    let cases = [
        "SmolLM2-135M-Instruct-Q4_K_M.gguf",
        "compat-qwen25/qwen2.5-0.5b-instruct-q4_k_m.gguf",
        "compat-qwen3/Qwen3-0.6B-Q4_K_M.gguf",
        "compat-llama32/Llama-3.2-1B-Instruct-Q8_0.gguf",
        "compat-phi3/Phi-3-mini-4k-instruct-Q4_K_M.gguf",
        "compat-olmo2/OLMo-2-0425-1B-Q4_K_M.gguf",
        "compat-gemma3/gemma-3-270m-it-Q8_0.gguf",
    ];
    for rel in cases {
        let path = models().join(rel);
        assert!(has_fast_path(&path).unwrap(), "{rel} should use fast CPU path");
        let mut fast = CpuBackend::open(&path, &config).unwrap();
        let mut reference = ReferenceBackend::open(&path, &config).unwrap();
        let prompt = fast.tokenize(PROMPT).unwrap();
        fast.evaluate(&prompt, 0).unwrap();
        reference.evaluate(&prompt, 0).unwrap();
        for step in 0..8 {
            let f_tok = fast.sample_greedy().unwrap();
            let r_tok = reference.sample_greedy().unwrap();
            assert_eq!(f_tok, r_tok, "{rel} diverged at step {step}");
            let pos = (prompt.len() + step) as u32;
            fast.evaluate(&[f_tok], pos).unwrap();
            reference.evaluate(&[r_tok], pos).unwrap();
        }
    }
}
