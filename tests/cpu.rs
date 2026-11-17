use local_inference::{
    Config, Error, Executor, GenerationMode,
    backend::Backend,
    cpu::{CpuBackend, CpuOptions},
};
use std::{fs, path::PathBuf};

fn string(bytes: &mut Vec<u8>, value: &str) {
    bytes.extend_from_slice(&(value.len() as u64).to_le_bytes());
    bytes.extend_from_slice(value.as_bytes());
}

fn tiny_model() -> Vec<u8> {
    let integers = [
        ("llama.embedding_length", 4u32),
        ("llama.attention.head_count", 2),
        ("llama.attention.head_count_kv", 1),
        ("llama.block_count", 1),
        ("llama.feed_forward_length", 4),
        ("llama.context_length", 8),
        ("llama.rope.dimension_count", 2),
        ("tokenizer.ggml.eos_token_id", 3),
    ];
    let mut bytes = b"GGUF".to_vec();
    bytes.extend_from_slice(&3u32.to_le_bytes());
    bytes.extend_from_slice(&11u64.to_le_bytes());
    bytes.extend_from_slice(&11u64.to_le_bytes());
    string(&mut bytes, "general.architecture");
    bytes.extend_from_slice(&8u32.to_le_bytes());
    string(&mut bytes, "llama");
    for (name, value) in integers {
        string(&mut bytes, name);
        bytes.extend_from_slice(&4u32.to_le_bytes());
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    for (name, value) in [
        ("llama.attention.layer_norm_rms_epsilon", 1e-5f32),
        ("llama.rope.freq_base", 10000.0),
    ] {
        string(&mut bytes, name);
        bytes.extend_from_slice(&6u32.to_le_bytes());
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    let tensors: Vec<(&str, Vec<u64>)> = vec![
        ("token_embd.weight", vec![4, 4]),
        ("output_norm.weight", vec![4]),
        ("blk.0.attn_norm.weight", vec![4]),
        ("blk.0.attn_q.weight", vec![4, 4]),
        ("blk.0.attn_k.weight", vec![4, 2]),
        ("blk.0.attn_v.weight", vec![4, 2]),
        ("blk.0.attn_output.weight", vec![4, 4]),
        ("blk.0.ffn_norm.weight", vec![4]),
        ("blk.0.ffn_gate.weight", vec![4, 4]),
        ("blk.0.ffn_up.weight", vec![4, 4]),
        ("blk.0.ffn_down.weight", vec![4, 4]),
    ];
    let mut data = Vec::new();
    for (name, dims) in tensors {
        while data.len() % 32 != 0 {
            data.push(0);
        }
        string(&mut bytes, name);
        bytes.extend_from_slice(&(dims.len() as u32).to_le_bytes());
        for dim in &dims {
            bytes.extend_from_slice(&dim.to_le_bytes());
        }
        bytes.extend_from_slice(&0u32.to_le_bytes());
        bytes.extend_from_slice(&(data.len() as u64).to_le_bytes());
        for index in 0..dims.iter().product::<u64>() {
            let value = if dims.len() == 1 {
                1.0
            } else {
                (index as f32 - 7.0) / 32.0
            };
            data.extend_from_slice(&value.to_le_bytes());
        }
    }
    while bytes.len() % 32 != 0 {
        bytes.push(0);
    }
    bytes.extend(data);
    bytes
}

struct Fixture(PathBuf);

impl Fixture {
    fn new(name: &str, bytes: &[u8]) -> Self {
        let path =
            std::env::temp_dir().join(format!("rust-gguf-{}-{name}.gguf", std::process::id()));
        use std::io::Write;
        fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)
            .unwrap()
            .write_all(bytes)
            .unwrap();
        Self(path)
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        fs::remove_file(&self.0).unwrap();
    }
}

fn config(chunk: u32) -> Config {
    Config {
        context_tokens: 8,
        batch_tokens: 3,
        prefill_chunk: chunk,
        threads: 2,
    }
}

#[test]
fn chunked_prefill_and_reset_preserve_logits_and_hide_stale_kv() {
    let fixture = Fixture::new("reset", &tiny_model());
    let mut backend = CpuBackend::open(&fixture.0, &config(3)).unwrap();
    assert!(backend.sample_greedy().is_err());
    backend.evaluate(&[0, 1, 2], 0).unwrap();
    let expected = backend.logits().unwrap().to_vec();
    backend.evaluate(&[3, 0, 1], 3).unwrap();
    backend.reset().unwrap();
    assert!(backend.sample_greedy().is_err());
    for (position, token) in [0, 1, 2].into_iter().enumerate() {
        backend.evaluate(&[token], position as u32).unwrap();
    }
    assert_eq!(backend.logits().unwrap(), expected);
    let mut executor = Executor::new(backend, config(1)).unwrap();
    let first = executor
        .generate(&[0, 1, 2], 4, GenerationMode::FixedLength)
        .unwrap();
    let second = executor
        .generate(&[0, 1, 2], 4, GenerationMode::FixedLength)
        .unwrap();
    assert_eq!(first.generated_tokens, second.generated_tokens);
}

#[test]
fn invalid_evaluations_do_not_advance_the_cache_and_exact_capacity_is_usable() {
    let fixture = Fixture::new("admission", &tiny_model());
    let mut backend = CpuBackend::open(&fixture.0, &config(3)).unwrap();
    backend.evaluate(&[0, 1], 0).unwrap();
    for (tokens, position) in [
        (&[][..], 2),
        (&[-1][..], 2),
        (&[4][..], 2),
        (&[1][..], 0),
        (&[1; 4][..], 2),
    ] {
        assert!(matches!(
            backend.evaluate(tokens, position),
            Err(Error::Invalid(_))
        ));
    }
    backend.evaluate(&[0, 1, 2], 2).unwrap();
    backend.evaluate(&[0, 1, 2], 5).unwrap();
    assert!(backend.evaluate(&[1], 8).is_err());
    assert!(
        backend
            .logits()
            .unwrap()
            .iter()
            .all(|value| value.is_finite())
    );
}

#[test]
fn rejects_overlapping_tensor_data_and_unsupported_quantization() {
    let original = tiny_model();
    let name = b"blk.0.ffn_down.weight";
    let start = original
        .windows(name.len())
        .position(|part| part == name)
        .unwrap()
        + name.len();
    let kind_offset = start + 4 + 2 * 8;
    for (name, offset, replacement, expected) in [
        (
            "overlap",
            kind_offset + 4,
            vec![0; 8],
            "overlapping tensor data",
        ),
        (
            "unsupported-type",
            kind_offset,
            2u32.to_le_bytes().to_vec(),
            "unsupported tensor type",
        ),
    ] {
        let mut bytes = original.clone();
        bytes[offset..offset + replacement.len()].copy_from_slice(&replacement);
        let fixture = Fixture::new(name, &bytes);
        let error = match CpuBackend::open(&fixture.0, &config(3)) {
            Ok(_) => panic!("invalid tensor accepted: {name}"),
            Err(error) => error,
        };
        assert!(error.to_string().contains(expected), "{error}");
    }
}

#[test]
fn rejects_truncated_tensor_payload_before_inference() {
    let mut bytes = tiny_model();
    bytes.pop();
    let fixture = Fixture::new("truncated", &bytes);
    let error = match CpuBackend::open(&fixture.0, &config(3)) {
        Ok(_) => panic!("truncated tensor data accepted"),
        Err(error) => error,
    };
    assert!(error.to_string().contains("beyond file"));
}

#[test]
#[ignore = "requires LI_TEST_MODEL; run in release mode with --ignored"]
fn real_quantized_model_preserves_logits_across_chunks_threads_and_reset() {
    let model = PathBuf::from(std::env::var_os("LI_TEST_MODEL").expect("set LI_TEST_MODEL"));
    let mut reference = None;
    let mut packed_bytes = None;
    for (expand_weights, sequential_prefill, threads, chunk) in [
        (false, true, 1, 1),
        (false, false, 4, 3),
        (true, true, 1, 1),
        (true, false, 4, 3),
    ] {
        let config = Config {
            threads,
            ..config(chunk)
        };
        let mut backend = CpuBackend::open_with_options(
            &model,
            &config,
            CpuOptions {
                expand_weights,
                sequential_prefill,
                ..CpuOptions::default()
            },
        )
        .unwrap();
        let allocated = backend.memory_usage().unwrap().weights_allocated_bytes;
        if expand_weights {
            assert!(packed_bytes.unwrap() < allocated / 3);
        } else {
            packed_bytes = Some(allocated);
        }
        for _ in 0..2 {
            backend.reset().unwrap();
            for (i, tokens) in [1, 2, 3, 4, 5].chunks(chunk as usize).enumerate() {
                backend
                    .evaluate(tokens, (i * chunk as usize) as u32)
                    .unwrap();
            }
            let logits = backend.logits().unwrap().to_vec();
            if let Some(expected) = &reference {
                assert_eq!(&logits, expected);
            }
            reference = Some(logits);
        }
    }
}

#[test]
fn numerical_reservations_admit_exact_budget_and_reset_keeps_allocations() {
    let fixture = Fixture::new("memory", &tiny_model());
    let mut backend = CpuBackend::open(&fixture.0, &config(3)).unwrap();
    let before = backend.memory_usage().unwrap();
    backend.evaluate(&[0, 1, 2], 0).unwrap();
    assert!(backend.memory_usage().unwrap().kv_live_bytes > 0);
    backend.reset().unwrap();
    let after = backend.memory_usage().unwrap();
    assert_eq!(after.kv_live_bytes, 0);
    assert_eq!(after.kv_allocated_bytes, before.kv_allocated_bytes);
    assert_eq!(
        after.workspace_allocated_bytes,
        before.workspace_allocated_bytes
    );
    for (limit, admitted) in [
        (before.numerical_reservation_bytes, true),
        (before.numerical_reservation_bytes - 1, false),
    ] {
        let result = CpuBackend::open_with_options(
            &fixture.0,
            &config(3),
            CpuOptions {
                memory_limit_bytes: Some(limit),
                ..CpuOptions::default()
            },
        );
        assert_eq!(result.is_ok(), admitted);
        if let Err(error) = result {
            assert!(error.to_string().contains("exceeds memory limit"));
        }
    }
}

#[test]
fn transactions_rollback_logits_and_kv_branches_and_reject_stale_or_foreign_handles() {
    let fixture = Fixture::new("transactions", &tiny_model());
    let mut backend = CpuBackend::open(&fixture.0, &config(3)).unwrap();
    let mut other = CpuBackend::open(&fixture.0, &config(3)).unwrap();
    backend.evaluate(&[0, 1], 0).unwrap();
    other.evaluate(&[0, 1], 0).unwrap();
    let expected = backend.logits().unwrap().to_vec();
    let transaction = backend.begin_transaction().unwrap();
    assert!(backend.begin_transaction().is_err());
    backend.evaluate(&[2, 3, 0], 2).unwrap();
    backend.rollback_transaction(transaction).unwrap();
    assert_eq!(backend.logits().unwrap(), expected);
    backend.evaluate(&[1, 0], 2).unwrap();
    other.evaluate(&[1, 0], 2).unwrap();
    assert_eq!(backend.logits().unwrap(), other.logits().unwrap());
    let transaction = backend.begin_transaction().unwrap();
    backend.evaluate(&[2], 4).unwrap();
    backend.commit_transaction(transaction).unwrap();
    assert!(backend.evaluate(&[2], 4).is_err());
    let stale = backend.begin_transaction().unwrap();
    backend.reset().unwrap();
    backend.evaluate(&[3], 0).unwrap();
    assert!(backend.rollback_transaction(stale).is_err());
    let foreign = other.begin_transaction().unwrap();
    assert!(backend.rollback_transaction(foreign).is_err());
    assert_eq!(backend.memory_usage().unwrap().kv_live_bytes, 16);
}

#[test]
fn bounded_prefix_reuse_matches_cold_execution_and_never_uses_mismatched_tokens() {
    let fixture = Fixture::new("prefix", &tiny_model());
    for (budget, cached) in [(75, 0), (76, 3)] {
        let backend = CpuBackend::open_with_options(
            &fixture.0,
            &config(3),
            CpuOptions {
                prefix_cache_bytes: budget,
                ..CpuOptions::default()
            },
        )
        .unwrap();
        let mut executor = Executor::new(backend, config(3)).unwrap();
        let cold = executor
            .generate(&[0, 1, 2], 3, GenerationMode::FixedLength)
            .unwrap();
        assert_eq!(cold.cached_prompt_tokens, 0);
        let warm = executor
            .generate(&[0, 1, 2], 3, GenerationMode::FixedLength)
            .unwrap();
        assert_eq!(warm.cached_prompt_tokens, cached);
        assert_eq!(warm.generated_tokens, cold.generated_tokens);
        assert!(executor.memory_usage().unwrap().prefix_allocated_bytes <= budget);
        let longer = executor
            .generate(&[0, 1, 2, 3], 3, GenerationMode::FixedLength)
            .unwrap();
        assert_eq!(longer.cached_prompt_tokens, cached);
        let mut reference =
            Executor::new(CpuBackend::open(&fixture.0, &config(3)).unwrap(), config(3)).unwrap();
        assert_eq!(
            longer.generated_tokens,
            reference
                .generate(&[0, 1, 2, 3], 3, GenerationMode::FixedLength)
                .unwrap()
                .generated_tokens
        );
        let miss = executor
            .generate(&[1, 2, 0], 3, GenerationMode::FixedLength)
            .unwrap();
        assert_eq!(miss.cached_prompt_tokens, 0);
        assert_eq!(
            miss.generated_tokens,
            reference
                .generate(&[1, 2, 0], 3, GenerationMode::FixedLength)
                .unwrap()
                .generated_tokens
        );
    }
}

#[test]
fn draft_verification_handles_full_rejection_partial_acceptance_and_full_acceptance() {
    let fixture = Fixture::new("verify-draft", &tiny_model());
    let mut reference =
        Executor::new(CpuBackend::open(&fixture.0, &config(3)).unwrap(), config(3)).unwrap();
    let correct = reference
        .generate(&[0, 1], 4, GenerationMode::FixedLength)
        .unwrap()
        .generated_tokens;
    let mut backend = CpuBackend::open_with_options(
        &fixture.0,
        &config(3),
        CpuOptions {
            speculative_window: 3,
            ..CpuOptions::default()
        },
    )
    .unwrap();
    let mut target = CpuBackend::open(&fixture.0, &config(3)).unwrap();
    for accepted in [0, 1, 3] {
        backend.reset().unwrap();
        target.reset().unwrap();
        backend.evaluate(&[0, 1], 0).unwrap();
        target.evaluate(&[0, 1], 0).unwrap();
        let mut draft = correct[..3].to_vec();
        if accepted < 3 {
            draft[accepted] = (draft[accepted] + 1) % 4;
        }
        assert_eq!(backend.verify_draft(&draft).unwrap(), accepted);
        if accepted > 0 {
            target.evaluate(&correct[..accepted], 2).unwrap();
        }
        assert_eq!(backend.logits().unwrap(), target.logits().unwrap());
        backend
            .evaluate(&[correct[accepted]], (2 + accepted) as u32)
            .unwrap();
        target
            .evaluate(&[correct[accepted]], (2 + accepted) as u32)
            .unwrap();
        assert_eq!(backend.logits().unwrap(), target.logits().unwrap());
    }
}

#[test]
fn speculative_generation_matches_greedy_at_exact_capacity_and_eog_boundaries() {
    let fixture = Fixture::new("speculative", &tiny_model());
    for mode in [GenerationMode::FixedLength, GenerationMode::UntilEog] {
        for count in [1, 4, 6] {
            let mut reference =
                Executor::new(CpuBackend::open(&fixture.0, &config(3)).unwrap(), config(3))
                    .unwrap();
            let backend = CpuBackend::open_with_options(
                &fixture.0,
                &config(3),
                CpuOptions {
                    speculative_window: 3,
                    prefix_cache_bytes: 256,
                    ..CpuOptions::default()
                },
            )
            .unwrap();
            let mut executor = Executor::new(backend, config(3)).unwrap();
            let expected = reference.generate(&[0, 0, 0], count, mode).unwrap();
            for _ in 0..2 {
                let actual = executor.generate(&[0, 0, 0], count, mode).unwrap();
                assert_eq!(actual.generated_tokens, expected.generated_tokens);
                assert_eq!(actual.decode_steps, expected.decode_steps);
                if mode == GenerationMode::FixedLength && count == 6 {
                    let stats = actual.speculation.unwrap();
                    assert!(stats.verification_batches > 0 && stats.accepted_tokens > 0);
                    assert!(stats.accepted_tokens <= stats.proposed_tokens);
                }
            }
        }
    }
}
