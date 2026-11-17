#![cfg(feature = "cuda")]
use local_inference::{
    Config, Executor, GenerationMode,
    backend::Backend,
    cuda::{CudaBackend, CudaOptions, reference::ReferenceBackend},
};
use std::{fs, path::PathBuf};

fn string(bytes: &mut Vec<u8>, value: &str) {
    bytes.extend_from_slice(&(value.len() as u64).to_le_bytes());
    bytes.extend_from_slice(value.as_bytes());
}

fn tiny_moe(used: u32) -> Vec<u8> {
    tiny_moe_weights(used, false)
}

fn tiny_moe_weights(used: u32, overflow_first_token: bool) -> Vec<u8> {
    tiny_weights(used, overflow_first_token, false)
}

fn tiny_weights(used: u32, overflow_first_token: bool, dense: bool) -> Vec<u8> {
    let layers = if dense && !overflow_first_token { 2 } else { 1 };
    tiny_weights_shape(
        used,
        overflow_first_token,
        dense,
        layers,
        if dense { 160 } else { 8 },
    )
}

fn tiny_weights_shape(
    used: u32,
    overflow_first_token: bool,
    dense: bool,
    layers: u32,
    context: u32,
) -> Vec<u8> {
    let integers = if dense {
        vec![
            ("embedding_length", 4),
            ("attention.head_count", 2),
            ("attention.head_count_kv", 1),
            ("block_count", layers),
            ("feed_forward_length", 4),
            ("context_length", context),
            ("rope.dimension_count", 2),
        ]
    } else {
        vec![
            ("embedding_length", 4),
            ("attention.head_count", 2),
            ("attention.head_count_kv", 1),
            ("attention.key_length", 4),
            ("attention.value_length", 4),
            ("block_count", layers),
            ("expert_feed_forward_length", 4),
            ("expert_count", 3),
            ("expert_used_count", used),
            ("context_length", context),
        ]
    };
    let mut tensors: Vec<(String, Vec<u64>)> = vec![
        ("token_embd.weight".to_owned(), vec![4, 4]),
        ("output_norm.weight".to_owned(), vec![4]),
        ("output.weight".to_owned(), vec![4, 4]),
    ];
    for layer in 0..layers {
        let q = if dense { 4 } else { 8 };
        let kv = if dense { 2 } else { 4 };
        for (name, dims) in [
            ("attn_norm.weight", vec![4]),
            ("attn_q.weight", vec![4, q]),
            ("attn_k.weight", vec![4, kv]),
            ("attn_v.weight", vec![4, kv]),
            ("attn_output.weight", vec![q, 4]),
        ] {
            tensors.push((format!("blk.{layer}.{name}"), dims));
        }
        if !dense {
            for name in ["attn_q_norm.weight", "attn_k_norm.weight"] {
                tensors.push((format!("blk.{layer}.{name}"), vec![4]));
            }
        }
        tensors.push((format!("blk.{layer}.ffn_norm.weight"), vec![4]));
        if dense {
            for name in ["ffn_gate.weight", "ffn_up.weight", "ffn_down.weight"] {
                tensors.push((format!("blk.{layer}.{name}"), vec![4, 4]));
            }
        } else {
            tensors.push((format!("blk.{layer}.ffn_gate_inp.weight"), vec![4, 3]));
            for name in [
                "ffn_gate_exps.weight",
                "ffn_up_exps.weight",
                "ffn_down_exps.weight",
            ] {
                tensors.push((format!("blk.{layer}.{name}"), vec![4, 4, 3]));
            }
        }
    }
    let arch = if dense { "llama" } else { "qwen3moe" };
    let mut bytes = b"GGUF".to_vec();
    bytes.extend_from_slice(&3u32.to_le_bytes());
    bytes.extend_from_slice(&(tensors.len() as u64).to_le_bytes());
    bytes.extend_from_slice(&(integers.len() as u64 + 6).to_le_bytes());
    string(&mut bytes, "general.architecture");
    bytes.extend_from_slice(&8u32.to_le_bytes());
    string(&mut bytes, arch);
    for (name, value) in integers {
        string(&mut bytes, &format!("{arch}.{name}"));
        bytes.extend_from_slice(&4u32.to_le_bytes());
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    string(&mut bytes, "tokenizer.ggml.eos_token_id");
    bytes.extend_from_slice(&4u32.to_le_bytes());
    bytes.extend_from_slice(&3u32.to_le_bytes());
    string(&mut bytes, "tokenizer.ggml.tokens");
    bytes.extend_from_slice(&9u32.to_le_bytes());
    bytes.extend_from_slice(&8u32.to_le_bytes());
    bytes.extend_from_slice(&4u64.to_le_bytes());
    for token in ["a", "b", "<|endoftext|>", "<|im_end|>"] {
        string(&mut bytes, token);
    }
    string(&mut bytes, "tokenizer.ggml.token_type");
    bytes.extend_from_slice(&9u32.to_le_bytes());
    bytes.extend_from_slice(&4u32.to_le_bytes());
    bytes.extend_from_slice(&4u64.to_le_bytes());
    for kind in [1u32, 1, 3, 3] {
        bytes.extend_from_slice(&kind.to_le_bytes());
    }
    for (name, value) in [
        ("attention.layer_norm_rms_epsilon", 1e-6f32),
        ("rope.freq_base", 1000000.0),
    ] {
        string(&mut bytes, &format!("{arch}.{name}"));
        bytes.extend_from_slice(&6u32.to_le_bytes());
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    let mut data = Vec::new();
    for (t, (name, dims)) in tensors.into_iter().enumerate() {
        while data.len() % 32 != 0 {
            data.push(0);
        }
        string(&mut bytes, &name);
        bytes.extend_from_slice(&(dims.len() as u32).to_le_bytes());
        for dim in &dims {
            bytes.extend_from_slice(&dim.to_le_bytes());
        }
        bytes.extend_from_slice(&0u32.to_le_bytes());
        bytes.extend_from_slice(&(data.len() as u64).to_le_bytes());
        for i in 0..dims.iter().product::<u64>() {
            let value = if dims.len() == 1 {
                1.0
            } else if overflow_first_token {
                if name == "token_embd.weight" {
                    if i == 0 || i == 5 { 1.0 } else { 0.0 }
                } else if (name.ends_with("ffn_gate_exps.weight")
                    || name.ends_with("ffn_up_exps.weight")
                    || name.ends_with("ffn_gate.weight")
                    || name.ends_with("ffn_up.weight"))
                    && i % 4 == 0
                {
                    f32::MAX
                } else {
                    0.0
                }
            } else {
                ((i * 17 + t as u64 * 5) % 31) as f32 / 64.0 - 0.25
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
    fn new(name: &str, used: u32) -> Self {
        Self::from_bytes(name, &tiny_moe(used))
    }

    fn from_bytes(name: &str, bytes: &[u8]) -> Self {
        use std::io::Write;
        let path =
            std::env::temp_dir().join(format!("rust-cuda-{}-{name}.gguf", std::process::id()));
        fs::OpenOptions::new()
            .create_new(true)
            .write(true)
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
fn config() -> Config {
    Config {
        context_tokens: 8,
        batch_tokens: 3,
        prefill_chunk: 3,
        threads: 2,
    }
}

#[test]
#[ignore = "requires CUDA 13 and an idle GPU; run serially in release mode"]
fn captured_decode_matches_full_logits_token_only_generation_and_context_reset() {
    for dense in [false, true] {
        let fixture = Fixture::from_bytes(
            if dense { "graph-dense" } else { "graph-moe" },
            &tiny_weights_shape(if dense { 1 } else { 2 }, false, dense, 2, 32),
        );
        let config = Config {
            context_tokens: 32,
            batch_tokens: 7,
            prefill_chunk: 7,
            threads: 2,
        };
        let mut baseline = CudaBackend::open(&fixture.0, &config, CudaOptions::default()).unwrap();
        let mut captured = CudaBackend::open(
            &fixture.0,
            &config,
            CudaOptions {
                decode_graph: true,
                ..Default::default()
            },
        )
        .unwrap();
        let mut position = 0;
        for ids in [
            vec![0, 1, 0],
            vec![1],
            vec![0, 1, 0, 1, 1, 0, 1],
            vec![0],
            vec![1],
        ] {
            baseline.evaluate(&ids, position).unwrap();
            captured.evaluate(&ids, position).unwrap();
            assert_eq!(captured.logits().unwrap(), baseline.logits().unwrap());
            position += ids.len() as u32;
        }
        baseline.evaluate(&[0], position).unwrap();
        captured.evaluate_for_sampling(&[0], position).unwrap();
        assert!(captured.logits().is_err());
        assert_eq!(
            captured.sample_greedy().unwrap(),
            baseline.sample_greedy().unwrap()
        );
        assert!(captured.evaluate_for_sampling(&[4], position + 1).is_err());
        assert_eq!(
            captured.sample_greedy().unwrap(),
            baseline.sample_greedy().unwrap()
        );
        let mut baseline = Executor::new(baseline, config.clone()).unwrap();
        let mut captured = Executor::new(captured, config).unwrap();
        for mode in [GenerationMode::FixedLength, GenerationMode::UntilEog] {
            for prompt in [vec![0], vec![0, 1, 0, 1, 0, 1, 0, 1]] {
                let count = 33 - prompt.len() as u32;
                let expected = baseline.generate(&prompt, count, mode).unwrap();
                let actual = captured.generate(&prompt, count, mode).unwrap();
                assert_eq!(actual.generated_tokens, expected.generated_tokens);
                assert_eq!(actual.decode_steps, expected.decode_steps);
                assert_eq!(
                    captured.memory_usage().unwrap().kv_live_bytes,
                    baseline.memory_usage().unwrap().kv_live_bytes
                );
            }
        }
    }
}

#[test]
#[ignore = "requires CUDA 13 and an idle GPU; run serially in release mode"]
fn captured_decode_poisoning_requires_reset_and_rejects_unsupported_context() {
    let fixture = Fixture::from_bytes("graph-poison", &tiny_moe_weights(2, true));
    let mut backend = CudaBackend::open(
        &fixture.0,
        &config(),
        CudaOptions {
            decode_graph: true,
            ..Default::default()
        },
    )
    .unwrap();
    assert!(backend.evaluate_for_sampling(&[0], 0).is_err());
    assert!(backend.sample_greedy().is_err());
    assert!(backend.evaluate_for_sampling(&[1], 0).is_err());
    backend.reset().unwrap();
    backend.evaluate_for_sampling(&[1], 0).unwrap();
    assert!(backend.sample_greedy().is_ok());
    let unsupported = Fixture::from_bytes(
        "graph-unsupported",
        &tiny_weights_shape(2, false, false, 1, 2048),
    );
    let config = Config {
        context_tokens: 2048,
        ..config()
    };
    assert!(
        CudaBackend::open(
            &unsupported.0,
            &config,
            CudaOptions {
                decode_graph: true,
                ..Default::default()
            }
        )
        .is_err()
    );
}

#[test]
fn moe_cpu_oracle_preserves_causal_state_across_chunks_and_reset() {
    let fixture = Fixture::new("reference", 2);
    let mut reference = ReferenceBackend::open(&fixture.0, &config()).unwrap();
    assert!(reference.is_eog(2));
    assert!(reference.is_eog(3));
    assert!(!reference.is_eog(0));
    reference.evaluate(&[0, 1, 2], 0).unwrap();
    let expected = reference.logits().unwrap().to_vec();
    assert!(reference.evaluate(&[4], 3).is_err());
    assert_eq!(reference.logits().unwrap(), expected);
    reference.evaluate(&[3, 0, 1], 3).unwrap();
    reference.reset().unwrap();
    assert!(reference.logits().is_err());
    for (i, token) in [0, 1, 2].into_iter().enumerate() {
        reference.evaluate(&[token], i as u32).unwrap();
    }
    assert_eq!(reference.logits().unwrap(), expected);
    let mut executor = Executor::new(reference, config()).unwrap();
    let sample = executor
        .generate(&[0, 1, 2, 0, 1], 4, GenerationMode::FixedLength)
        .unwrap();
    assert_eq!(sample.generated_tokens.len(), 4);
    assert!(
        executor
            .generate(&[0, 1, 2, 0, 1], 5, GenerationMode::FixedLength)
            .is_err()
    );
}

#[test]
fn moe_loader_rejects_empty_or_excessive_top_k() {
    for used in [0, 4] {
        let fixture = Fixture::new(&format!("invalid-{used}"), used);
        assert!(ReferenceBackend::open(&fixture.0, &config()).is_err());
    }
}

#[test]
fn reference_trace_is_position_filtered_and_preserves_execution() {
    let fixture = Fixture::new("reference-trace", 2);
    let path = fixture.0.with_extension("jsonl");
    let mut reference = ReferenceBackend::open(&fixture.0, &config()).unwrap();
    let mut traced = ReferenceBackend::open(&fixture.0, &config()).unwrap();
    traced.trace_at(&path, 2).unwrap();
    assert!(traced.trace_at(&path, 2).is_err());
    reference.evaluate(&[0, 1, 2], 0).unwrap();
    traced.evaluate(&[0, 1, 2], 0).unwrap();
    assert_eq!(traced.logits().unwrap(), reference.logits().unwrap());
    let records: Vec<serde_json::Value> = fs::read_to_string(&path)
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect();
    assert!(!records.is_empty());
    assert!(
        records
            .iter()
            .all(|r| r["position"] == 2 && r["layer"] == 0)
    );
    let selected = records
        .iter()
        .find(|r| r["stage"] == "ffn_moe_topk")
        .unwrap();
    assert_eq!(selected["values"].as_array().unwrap().len(), 2);
    drop(traced);
    fs::remove_file(path).unwrap();
}

#[test]
#[ignore = "requires CUDA 13 NVRTC/driver; run serially in release mode"]
fn gpu_moe_matches_cpu_with_wider_queries_qk_norm_neox_and_reset() {
    let fixture = Fixture::new("gpu", 2);
    let mut gpu = CudaBackend::open(&fixture.0, &config(), CudaOptions::default()).unwrap();
    let mut reference = ReferenceBackend::open(&fixture.0, &config()).unwrap();
    for tokens in [&[0, 1, 2][..], &[2, 0][..]] {
        gpu.reset().unwrap();
        reference.reset().unwrap();
        gpu.evaluate(tokens, 0).unwrap();
        reference.evaluate(tokens, 0).unwrap();
        for (&a, &b) in gpu
            .logits()
            .unwrap()
            .iter()
            .zip(reference.logits().unwrap())
        {
            assert!((a - b).abs() < 1e-5, "{a} != {b}");
        }
        assert_eq!(
            gpu.sample_greedy().unwrap(),
            reference.sample_greedy().unwrap()
        );
        let expected = gpu.logits().unwrap().to_vec();
        assert!(gpu.evaluate(&[4], tokens.len() as u32).is_err());
        assert_eq!(gpu.logits().unwrap(), expected);
    }

    gpu.reset().unwrap();
    reference.reset().unwrap();
    let tokens = [0, 1, 2, 0, 1, 0, 1, 0];
    for (position, chunk) in tokens.chunks(3).enumerate() {
        gpu.evaluate(chunk, (position * 3) as u32).unwrap();
        reference.evaluate(chunk, (position * 3) as u32).unwrap();
        for (a, b) in gpu
            .logits()
            .unwrap()
            .iter()
            .zip(reference.logits().unwrap())
        {
            assert!((a - b).abs() < 1e-5, "nonzero-position batch: {a} != {b}");
        }
    }
    let expected = gpu.logits().unwrap().to_vec();
    gpu.reset().unwrap();
    for (position, token) in tokens.iter().enumerate() {
        gpu.evaluate(&[*token], position as u32).unwrap();
    }
    assert_eq!(gpu.logits().unwrap(), expected);
}

#[test]
#[ignore = "requires CUDA 13 NVRTC/driver; run serially in release mode"]
fn gpu_skipped_logits_cannot_hide_an_earlier_tokens_numerical_failure() {
    let fixture = Fixture::from_bytes("overflow", &tiny_moe_weights(2, true));
    let mut gpu = CudaBackend::open(&fixture.0, &config(), CudaOptions::default()).unwrap();
    gpu.evaluate(&[1], 0).unwrap();
    assert!(gpu.logits().unwrap().iter().all(|&x| x == 0.0));
    gpu.reset().unwrap();
    assert!(gpu.evaluate(&[0, 1], 0).is_err());
    assert!(gpu.logits().is_err());
    assert!(gpu.evaluate(&[1], 0).is_err());
    gpu.reset().unwrap();
    gpu.evaluate(&[1], 0).unwrap();
    assert!(gpu.logits().unwrap().iter().all(|&x| x == 0.0));
}

#[test]
#[ignore = "requires CUDA 13 NVRTC/driver; run serially in release mode"]
fn gpu_dense_prefill_preserves_causality_chunks_capacity_and_budget() {
    check_batched_prefill(true, 0);
}

#[test]
#[ignore = "requires CUDA 13 NVRTC/driver; run serially in release mode"]
fn gpu_grouped_moe_prefill_preserves_causality_chunks_capacity_and_budget() {
    for used in [1, 2, 3] {
        check_batched_prefill(false, used);
    }
}

fn check_batched_prefill(dense: bool, used: u32) {
    let fixture = Fixture::from_bytes(
        &format!("batched-{dense}-{used}"),
        &tiny_weights_shape(used, false, dense, 2, 160),
    );
    let config = Config {
        batch_tokens: 8,
        prefill_chunk: 8,
        ..config()
    };
    let mut gpu = CudaBackend::open(&fixture.0, &config, CudaOptions::default()).unwrap();
    let mut sequential = CudaBackend::open(
        &fixture.0,
        &Config {
            batch_tokens: 1,
            prefill_chunk: 1,
            ..config.clone()
        },
        CudaOptions::default(),
    )
    .unwrap();
    let mut reference = ReferenceBackend::open(&fixture.0, &config).unwrap();
    let tokens = [0, 1, 2, 0, 3, 1, 2, 0];
    for chunks in [&[8][..], &[1, 5, 2][..], &[3, 1, 4][..], &[4, 4][..]] {
        gpu.reset().unwrap();
        sequential.reset().unwrap();
        reference.reset().unwrap();
        let mut position = 0;
        for &count in chunks {
            let input = &tokens[position..position + count];
            gpu.evaluate(input, position as u32).unwrap();
            reference.evaluate(input, position as u32).unwrap();
            for (i, &token) in input.iter().enumerate() {
                sequential
                    .evaluate(&[token], (position + i) as u32)
                    .unwrap();
            }
            assert_eq!(gpu.logits().unwrap(), sequential.logits().unwrap());
            for (&a, &b) in gpu
                .logits()
                .unwrap()
                .iter()
                .zip(reference.logits().unwrap())
            {
                assert!((a - b).abs() < 1e-5, "dense GQA: {a} != {b}");
            }
            let expected = gpu.logits().unwrap().to_vec();
            assert!(gpu.evaluate(&[4], (position + count) as u32).is_err());
            assert_eq!(gpu.logits().unwrap(), expected);
            position += count;
        }
        assert!(gpu.evaluate(&[0], 8).is_err());
    }
    // Reset must hide the future KV written by the preceding full-context batch.
    gpu.reset().unwrap();
    sequential.reset().unwrap();
    gpu.evaluate(&[3, 2, 1], 0).unwrap();
    for (i, token) in [3, 2, 1].into_iter().enumerate() {
        sequential.evaluate(&[token], i as u32).unwrap();
    }
    assert_eq!(gpu.logits().unwrap(), sequential.logits().unwrap());
    let usage = gpu.memory_usage().unwrap();
    let small = sequential.memory_usage().unwrap();
    assert!(usage.workspace_allocated_bytes > small.workspace_allocated_bytes);
    assert_eq!(usage.kv_allocated_bytes, small.kv_allocated_bytes);
    for (budget, accepted) in [
        (usage.numerical_reservation_bytes, true),
        (usage.numerical_reservation_bytes - 1, false),
    ] {
        let result = CudaBackend::open(
            &fixture.0,
            &config,
            CudaOptions {
                memory_limit_bytes: budget,
                ..CudaOptions::default()
            },
        );
        assert_eq!(result.is_ok(), accepted, "dense workspace budget {budget}");
    }

    let large = Config {
        context_tokens: 160,
        batch_tokens: 160,
        prefill_chunk: 160,
        ..config
    };
    let mut gpu = CudaBackend::open(&fixture.0, &large, CudaOptions::default()).unwrap();
    let mut reference = ReferenceBackend::open(&fixture.0, &large).unwrap();
    let tokens: Vec<_> = (0..129).map(|i| i % 4).collect();
    gpu.evaluate(&tokens, 0).unwrap();
    reference.evaluate(&tokens, 0).unwrap();
    for (&a, &b) in gpu
        .logits()
        .unwrap()
        .iter()
        .zip(reference.logits().unwrap())
    {
        assert!(
            (a - b).abs() < 1e-5,
            "internal 128-token boundary: {a} != {b}"
        );
    }
    gpu.evaluate(&[1, 2, 3], 129).unwrap();
    reference.evaluate(&[1, 2, 3], 129).unwrap();
    for (&a, &b) in gpu
        .logits()
        .unwrap()
        .iter()
        .zip(reference.logits().unwrap())
    {
        assert!(
            (a - b).abs() < 1e-5,
            "nonzero internal boundary: {a} != {b}"
        );
    }
}

#[test]
#[ignore = "requires CUDA 13 NVRTC/driver; run serially in release mode"]
fn gpu_batched_attention_falls_back_above_fused_context_limit() {
    let fixture = Fixture::from_bytes(
        "attention-fallback",
        &tiny_weights_shape(0, false, true, 2, 1025),
    );
    let config = Config {
        context_tokens: 1025,
        batch_tokens: 8,
        prefill_chunk: 8,
        ..config()
    };
    let mut gpu = CudaBackend::open(&fixture.0, &config, CudaOptions::default()).unwrap();
    let mut reference = ReferenceBackend::open(&fixture.0, &config).unwrap();
    for (position, tokens) in [(0, &[0, 1, 2, 3][..]), (4, &[3, 1, 2][..])] {
        gpu.evaluate(tokens, position).unwrap();
        reference.evaluate(tokens, position).unwrap();
        for (&a, &b) in gpu
            .logits()
            .unwrap()
            .iter()
            .zip(reference.logits().unwrap())
        {
            assert!((a - b).abs() < 1e-5, "unfused fallback: {a} != {b}");
        }
    }
}

#[test]
#[ignore = "requires CUDA 13 NVRTC/driver; run serially in release mode"]
fn gpu_dense_prefill_numerical_failure_is_sticky_until_reset() {
    let fixture = Fixture::from_bytes("dense-overflow", &tiny_weights(0, true, true));
    let mut gpu = CudaBackend::open(&fixture.0, &config(), CudaOptions::default()).unwrap();
    gpu.evaluate(&[1, 1], 0).unwrap();
    assert!(gpu.logits().unwrap().iter().all(|&x| x == 0.0));
    gpu.reset().unwrap();
    assert!(gpu.evaluate(&[0, 1], 0).is_err());
    assert!(gpu.logits().is_err());
    assert!(gpu.evaluate(&[1, 1], 0).is_err());
    gpu.reset().unwrap();
    gpu.evaluate(&[1, 1], 0).unwrap();
    assert!(gpu.logits().unwrap().iter().all(|&x| x == 0.0));
}
