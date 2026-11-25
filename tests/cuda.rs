#![cfg(feature = "cuda")]
use palito::{
    Config, Executor, GenerationMode,
    backend::Backend,
    cuda::{CudaBackend, CudaMatmul, CudaOptions, reference::ReferenceBackend},
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
    tiny_weights_arch(
        if dense { "llama" } else { "qwen3moe" },
        used,
        overflow_first_token,
        layers,
        context,
        4,
    )
}

fn tiny_weights_arch(
    arch: &str,
    used: u32,
    overflow_first_token: bool,
    layers: u32,
    context: u32,
    value_length: u32,
) -> Vec<u8> {
    let dense = arch != "qwen3moe";
    let qk_norm = arch != "llama";
    let mut integers = if dense {
        vec![
            ("embedding_length", 4),
            ("attention.head_count", 2),
            ("attention.head_count_kv", 1),
            ("block_count", layers),
            ("feed_forward_length", 4),
            ("context_length", context),
            ("rope.dimension_count", if qk_norm { 4 } else { 2 }),
        ]
    } else {
        vec![
            ("embedding_length", 4),
            ("attention.head_count", 2),
            ("attention.head_count_kv", 1),
            ("block_count", layers),
            ("expert_feed_forward_length", 4),
            ("expert_count", 3),
            ("expert_used_count", used),
            ("context_length", context),
        ]
    };
    if qk_norm {
        integers.extend([
            ("attention.key_length", 4),
            ("attention.value_length", value_length),
        ]);
    }
    if arch == "gemma3" {
        integers.extend([
            ("attention.sliding_window", 2),
            ("attention.sliding_window_pattern", 2),
        ]);
    }
    let mut tensors: Vec<(String, Vec<u64>)> = vec![
        ("token_embd.weight".to_owned(), vec![4, 4]),
        ("output_norm.weight".to_owned(), vec![4]),
        ("output.weight".to_owned(), vec![4, 4]),
    ];
    for layer in 0..layers {
        let q = if qk_norm { 8 } else { 4 };
        let kv = if qk_norm { 4 } else { 2 };
        for (name, dims) in [
            ("attn_norm.weight", vec![4]),
            ("attn_q.weight", vec![4, q]),
            ("attn_k.weight", vec![4, kv]),
            ("attn_v.weight", vec![4, kv]),
            ("attn_output.weight", vec![q, 4]),
        ] {
            tensors.push((format!("blk.{layer}.{name}"), dims));
        }
        if qk_norm {
            for name in ["attn_q_norm.weight", "attn_k_norm.weight"] {
                tensors.push((format!("blk.{layer}.{name}"), vec![4]));
            }
            if arch == "gemma3" {
                for name in ["post_attention_norm.weight", "post_ffw_norm.weight"] {
                    tensors.push((format!("blk.{layer}.{name}"), vec![4]));
                }
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
    let tokens = if arch == "gemma3" {
        ["a", "b", "<end_of_turn>", "<eos>"]
    } else {
        ["a", "b", "<|endoftext|>", "<|im_end|>"]
    };
    for token in tokens {
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
    for (dense, matmul) in [
        (false, CudaMatmul::F32),
        (true, CudaMatmul::F32),
        (false, CudaMatmul::F32Tiled),
        (true, CudaMatmul::F32Tiled),
    ] {
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
        let mut baseline = CudaBackend::open(
            &fixture.0,
            &config,
            CudaOptions {
                matmul,
                ..Default::default()
            },
        )
        .unwrap();
        let mut captured = CudaBackend::open(
            &fixture.0,
            &config,
            CudaOptions {
                matmul,
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
fn dense_qwen3_cpu_handles_wide_queries_and_rejects_mismatched_heads() {
    let good = Fixture::from_bytes("qwen3", &tiny_weights_arch("qwen3", 0, false, 2, 8, 4));
    let bad = Fixture::from_bytes(
        "qwen3-bad-heads",
        &tiny_weights_arch("qwen3", 0, false, 2, 8, 2),
    );
    let mut backend = ReferenceBackend::open(&good.0, &config()).unwrap();
    assert_eq!(backend.info().architecture, "qwen3");
    assert!(backend.is_eog(2));
    assert!(backend.is_eog(3));
    assert!(!backend.is_eog(0));
    assert!(ReferenceBackend::open(&bad.0, &config()).is_err());
    backend.evaluate(&[0, 1, 0], 0).unwrap();
    let expected = backend.logits().unwrap().to_vec();
    backend.reset().unwrap();
    for (position, token) in [0, 1, 0].into_iter().enumerate() {
        backend.evaluate(&[token], position as u32).unwrap();
    }
    assert_eq!(backend.logits().unwrap(), expected);
}

#[test]
#[ignore = "requires CUDA 13 NVRTC/driver; run serially in release mode"]
fn gpu_dense_qwen3_matches_cpu_across_prefill_and_decode() {
    let fixture = Fixture::from_bytes("qwen3-gpu", &tiny_weights_arch("qwen3", 0, false, 2, 8, 4));
    let mut gpu = CudaBackend::open(&fixture.0, &config(), CudaOptions::default()).unwrap();
    let mut reference = ReferenceBackend::open(&fixture.0, &config()).unwrap();
    for (position, tokens) in [(0, &[0, 1, 0][..]), (3, &[1, 0][..]), (5, &[1][..])] {
        gpu.evaluate(tokens, position).unwrap();
        reference.evaluate(tokens, position).unwrap();
        let expected = reference.logits().unwrap();
        let actual = gpu.logits().unwrap();
        assert_eq!(actual.len(), expected.len());
        for (&a, &b) in actual.iter().zip(expected) {
            assert!((a - b).abs() < 1e-5, "{a} != {b}");
        }
        assert_eq!(
            gpu.sample_greedy().unwrap(),
            reference.sample_greedy().unwrap()
        );
    }
}

#[test]
fn gemma3_reference_masks_evicted_tokens_only_in_local_layers() {
    let local = Fixture::from_bytes(
        "gemma-local",
        &tiny_weights_arch("gemma3", 0, false, 1, 8, 4),
    );
    let mixed = Fixture::from_bytes(
        "gemma-mixed",
        &tiny_weights_arch("gemma3", 0, false, 2, 8, 4),
    );
    for (fixture, evicted) in [(&local, true), (&mixed, false)] {
        let mut backend = ReferenceBackend::open(&fixture.0, &config()).unwrap();
        assert!(backend.is_eog(2) && backend.is_eog(3));
        assert!(!backend.is_eog(0));
        backend.evaluate(&[0, 1, 0], 0).unwrap();
        let expected = backend.logits().unwrap().to_vec();
        backend.reset().unwrap();
        backend.evaluate(&[0, 1], 0).unwrap();
        backend.evaluate(&[0], 2).unwrap();
        assert_eq!(backend.logits().unwrap(), expected);
        backend.reset().unwrap();
        backend.evaluate(&[1, 1, 0], 0).unwrap();
        assert_eq!(backend.logits().unwrap() == expected, evicted);
    }
}

#[test]
#[ignore = "requires CUDA 13 NVRTC/driver; run serially in release mode"]
fn gpu_gemma3_matches_cpu_across_local_window_and_split_prefill() {
    let fixture = Fixture::from_bytes("gemma-gpu", &tiny_weights_arch("gemma3", 0, false, 2, 8, 4));
    let mut gpu = CudaBackend::open(&fixture.0, &config(), CudaOptions::default()).unwrap();
    let mut reference = ReferenceBackend::open(&fixture.0, &config()).unwrap();
    for (position, tokens) in [(0, &[0, 1, 0][..]), (3, &[1][..]), (4, &[0, 1][..])] {
        gpu.evaluate(tokens, position).unwrap();
        reference.evaluate(tokens, position).unwrap();
        let actual = gpu.logits().unwrap();
        let expected = reference.logits().unwrap();
        assert_eq!(actual.len(), expected.len());
        for (&a, &b) in actual.iter().zip(expected) {
            assert!((a - b).abs() < 1e-5, "position {position}: {a} != {b}");
        }
    }
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

#[test]
#[ignore = "requires CUDA 13 NVRTC/driver; run serially in release mode"]
fn gpu_fit_context_shrinks_to_the_largest_admitted_step() {
    let fixture = Fixture::from_bytes("fit-context", &tiny_weights_shape(2, false, false, 2, 4096));
    let config = Config {
        context_tokens: 4096,
        batch_tokens: 8,
        prefill_chunk: 8,
        ..config()
    };
    let full = CudaBackend::open(&fixture.0, &config, CudaOptions::default()).unwrap();
    let full_bytes = full.memory_usage().unwrap().numerical_reservation_bytes;
    let per_token = full.memory_usage().unwrap().kv_allocated_bytes / 4096;
    drop(full);
    // A budget that admits about half the context: fitted to a 256-token step below.
    let budget = full_bytes - per_token * 2000;
    let options = CudaOptions {
        memory_limit_bytes: budget,
        fit_context: true,
        ..CudaOptions::default()
    };
    let fitted = CudaBackend::open(&fixture.0, &config, options.clone()).unwrap();
    let context = fitted.info().context_tokens;
    assert!(context < 4096 && context % 256 == 0, "fitted {context}");
    assert!(fitted.memory_usage().unwrap().numerical_reservation_bytes <= budget);
    let next = Config { context_tokens: context + 256, ..config.clone() };
    let strict = CudaOptions { fit_context: false, ..options.clone() };
    assert!(CudaBackend::open(&fixture.0, &next, strict).is_err());
    // Without fitting the same budget refuses the full context.
    assert!(
        CudaBackend::open(&fixture.0, &config, CudaOptions { fit_context: false, ..options }).is_err()
    );
}

/// Prompt-sized prefill through the relaxed tensor-core profiles against the strict
/// f32-tiled profile on real models whose projections are not K-quants (Q8_0 and
/// F16 take dedicated tensor-core paths there). Relaxed gate of the CUDA policy:
/// same argmax, cosine >= 0.99999 and max_abs <= 0.15 for int8-prefill; int8-fast
/// (llama.cpp precision) only needs the same argmax and cosine >= 0.9999.
/// PALITO_PREFILL_MODELS: colon-separated GGUF paths.
#[test]
#[ignore = "requires CUDA 13, PALITO_PREFILL_MODELS and an idle GPU; run serially in release mode"]
fn relaxed_prefill_profiles_match_f32_tiled_on_dense_projection_types() {
    let models = std::env::var("PALITO_PREFILL_MODELS").expect("PALITO_PREFILL_MODELS");
    let config = Config {
        context_tokens: 512,
        batch_tokens: 512,
        prefill_chunk: 512,
        threads: 2,
    };
    // Deterministic ids spread over a Llama 3 vocabulary, 300 tokens in one chunk.
    let tokens: Vec<i32> = (0..300).map(|i| (i * 7919 + 13) % 30000 + 100).collect();
    let logits = |path: &str, matmul: CudaMatmul| {
        let options = CudaOptions { matmul, ..CudaOptions::default() };
        let mut gpu = CudaBackend::open(path.as_ref(), &config, options).unwrap();
        gpu.evaluate(&tokens, 0).unwrap();
        gpu.logits().unwrap().to_vec()
    };
    let argmax = |v: &[f32]| {
        v.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).unwrap().0
    };
    for path in models.split(':').filter(|p| !p.is_empty()) {
        let expected = logits(path, CudaMatmul::F32Tiled);
        for (matmul, min_cosine, max_abs_bound) in [
            (CudaMatmul::Int8Prefill, 0.99999, 0.15f32),
            (CudaMatmul::Int8Fast, 0.9999, f32::INFINITY),
        ] {
            let actual = logits(path, matmul);
            let (mut dot, mut aa, mut bb, mut max_abs) = (0.0f64, 0.0f64, 0.0f64, 0.0f32);
            for (&a, &b) in actual.iter().zip(&expected) {
                dot += f64::from(a) * f64::from(b);
                aa += f64::from(a) * f64::from(a);
                bb += f64::from(b) * f64::from(b);
                max_abs = max_abs.max((a - b).abs());
            }
            let cosine = dot / (aa.sqrt() * bb.sqrt());
            println!("{path} {matmul:?}: cosine {cosine:.7} max_abs {max_abs:.4}");
            assert_eq!(argmax(&actual), argmax(&expected), "{path} {matmul:?} argmax");
            assert!(cosine >= min_cosine, "{path} {matmul:?} cosine {cosine}");
            assert!(max_abs <= max_abs_bound, "{path} {matmul:?} max_abs {max_abs}");
        }
    }
}
