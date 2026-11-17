#![cfg(feature = "llama-cpp")]

use local_inference::{Config, Executor, GenerationMode, backend::Backend, llama::LlamaBackend};
use serde_json::Value;
use std::{path::PathBuf, process::Command};

fn model_path() -> PathBuf {
    PathBuf::from(std::env::var_os("LI_TEST_MODEL").expect("set LI_TEST_MODEL to a real GGUF"))
}

fn config(chunk: u32) -> Config {
    Config {
        context_tokens: 512,
        batch_tokens: 128,
        prefill_chunk: chunk,
        threads: 4,
    }
}

#[test]
#[ignore = "requires LI_TEST_MODEL; run with --ignored --test-threads=1"]
fn real_model_matches_cpp_across_chunks_and_request_resets() {
    let prompt_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/prompt.txt");
    let prompt = std::fs::read_to_string(&prompt_path).unwrap();
    let mut previous = None;
    for chunk in [1, 7, 128] {
        let config = config(chunk);
        let backend = LlamaBackend::open(&model_path(), &config).unwrap();
        let mut executor = Executor::new(backend, config).unwrap();
        let tokens = executor.tokenize(&prompt).unwrap();
        let output = Command::new(concat!(env!("LI_NATIVE_BIN"), "/li-reference"))
            .arg(model_path())
            .arg(&prompt_path)
            .args(["512", "128", &chunk.to_string(), "4", "8", "1", "2", "v1"])
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        let reference: Value = serde_json::from_slice(&output.stdout).unwrap();
        assert_eq!(reference["prompt_tokens"], tokens.len());
        for sample in reference["samples"].as_array().unwrap() {
            let actual = executor
                .generate(&tokens, 8, GenerationMode::FixedLength)
                .unwrap();
            assert_eq!(
                serde_json::to_value(&actual.generated_tokens).unwrap(),
                sample["generated_tokens"]
            );
            if let Some(previous) = &previous {
                assert_eq!(&actual.generated_tokens, previous);
            }
            previous = Some(actual.generated_tokens);
        }
    }
}

#[test]
#[ignore = "requires LI_TEST_MODEL; run with --ignored --test-threads=1"]
fn native_reset_invalidates_logits_and_recovers_after_bad_position() {
    let mut backend = LlamaBackend::open(&model_path(), &config(7)).unwrap();
    let tokens = backend.tokenize("A small test").unwrap();
    backend.evaluate(&tokens, 0).unwrap();
    let first = backend.sample_greedy().unwrap();
    assert!(backend.evaluate(&tokens, 0).is_err());
    backend.reset().unwrap();
    assert!(backend.sample_greedy().is_err());
    backend.evaluate(&tokens, 0).unwrap();
    assert_eq!(backend.sample_greedy().unwrap(), first);
}

#[test]
fn vocabulary_only_gguf_is_not_accepted_as_an_inference_model() {
    let vocabulary = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../llama.cpp/models/ggml-vocab-llama-spm.gguf");
    let error = match LlamaBackend::open(&vocabulary, &config(7)) {
        Ok(_) => panic!("a vocabulary fixture is not a model"),
        Err(error) => error,
    };
    assert!(error.to_string().contains("backend failure"));
}
