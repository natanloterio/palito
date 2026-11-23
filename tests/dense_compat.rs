#![cfg(feature = "cuda")]

use palito::{
    Config,
    backend::Backend,
    cpu::CpuBackend,
    cuda::{CudaBackend, CudaOptions, reference::ReferenceBackend},
};
use std::path::Path;

fn exercise(backend: &mut dyn Backend, eog: &[i32]) {
    let tokens = backend.tokenize("The capital of France is Paris.").unwrap();
    assert!(tokens.len() > 5);
    for &id in eog {
        assert!(backend.is_eog(id), "missing EOG {id}");
    }

    let prefill = &tokens[..tokens.len() - 1];
    backend.evaluate(prefill, 0).unwrap();
    assert_eq!(
        backend.logits().unwrap().len(),
        backend.info().vocab_tokens as usize
    );
    backend
        .evaluate(&tokens[prefill.len()..], prefill.len() as u32)
        .unwrap();
    let before = backend.logits().unwrap().to_vec();
    let greedy = backend.sample_greedy().unwrap();

    backend.reset().unwrap();
    assert!(backend.logits().is_err());
    assert!(backend.evaluate(&tokens[1..], 1).is_err());
    backend.reset().unwrap();
    // Prefill with different KV to catch stale visibility after a second reset.
    backend.evaluate(&tokens[..2], 0).unwrap();
    backend.reset().unwrap();
    backend.evaluate(prefill, 0).unwrap();
    backend
        .evaluate(&tokens[prefill.len()..], prefill.len() as u32)
        .unwrap();
    assert!(
        before
            .iter()
            .zip(backend.logits().unwrap())
            .all(|(a, b)| a.to_bits() == b.to_bits())
    );
    assert_eq!(greedy, backend.sample_greedy().unwrap());
}

fn run_fixture(env: &str, cuda: bool, eog: &[i32]) {
    run_at_context(env, cuda, eog, 512);
}

fn run_at_context(env: &str, cuda: bool, eog: &[i32], context_tokens: u32) {
    let model = std::env::var_os(env).expect("set the GGUF fixture path");
    let path = Path::new(&model);
    let config = Config {
        context_tokens,
        batch_tokens: 128,
        prefill_chunk: 8,
        threads: 4,
    };
    let mut backend: Box<dyn Backend> = if cuda {
        Box::new(CudaBackend::open(path, &config, CudaOptions::default()).unwrap())
    } else {
        Box::new(ReferenceBackend::open(path, &config).unwrap())
    };
    exercise(backend.as_mut(), eog);
}

#[test]
#[ignore = "requires LLAMA32_GGUF"]
fn llama32_reference_lifecycle() {
    run_fixture("LLAMA32_GGUF", false, &[128001, 128008, 128009]);
}

#[test]
#[ignore = "requires LLAMA32_GGUF and CUDA 13 NVRTC/driver"]
fn llama32_cuda_lifecycle() {
    run_fixture("LLAMA32_GGUF", true, &[128001, 128008, 128009]);
}

#[test]
#[ignore = "requires PHI3_GGUF"]
fn phi3_reference_lifecycle() {
    run_fixture("PHI3_GGUF", false, &[2, 32000, 32007]);
}

#[test]
#[ignore = "requires PHI3_GGUF and CUDA 13 NVRTC/driver"]
fn phi3_cuda_lifecycle() {
    run_fixture("PHI3_GGUF", true, &[2, 32000, 32007]);
}

#[test]
#[ignore = "requires PHI3_128K_GGUF"]
fn phi3_longrope_reference_lifecycle() {
    for context in [4096, 4097] {
        run_at_context("PHI3_128K_GGUF", false, &[2, 32000, 32007], context);
    }
}

#[test]
#[ignore = "requires PHI3_128K_GGUF and CUDA 13 NVRTC/driver"]
fn phi3_longrope_cuda_lifecycle() {
    for context in [4096, 4097] {
        run_at_context("PHI3_128K_GGUF", true, &[2, 32000, 32007], context);
    }
}

#[test]
#[ignore = "requires SMOLLM2_Q4_1_GGUF"]
fn smollm2_q4_1_cpu_lifecycle() {
    smollm2_cpu_lifecycle("SMOLLM2_Q4_1_GGUF");
}

fn smollm2_cpu_lifecycle(env: &str) {
    let path = std::env::var_os(env).expect("set the GGUF fixture path");
    let config = Config {
        context_tokens: 512,
        batch_tokens: 128,
        prefill_chunk: 8,
        threads: 4,
    };
    exercise(
        &mut CpuBackend::open(Path::new(&path), &config).unwrap(),
        &[2],
    );
    run_fixture(env, false, &[2]);
}

#[test]
#[ignore = "requires SMOLLM2_Q4_1_GGUF and CUDA 13 NVRTC/driver"]
fn smollm2_q4_1_cuda_lifecycle() {
    run_fixture("SMOLLM2_Q4_1_GGUF", true, &[2]);
}

#[test]
#[ignore = "requires SMOLLM2_Q5_1_GGUF"]
fn smollm2_q5_1_cpu_lifecycle() {
    smollm2_cpu_lifecycle("SMOLLM2_Q5_1_GGUF");
}

#[test]
#[ignore = "requires SMOLLM2_Q5_1_GGUF and CUDA 13 NVRTC/driver"]
fn smollm2_q5_1_cuda_lifecycle() {
    run_fixture("SMOLLM2_Q5_1_GGUF", true, &[2]);
}

#[test]
#[ignore = "requires STARCODER2_GGUF"]
fn starcoder2_reference_lifecycle() {
    run_fixture("STARCODER2_GGUF", false, &[0, 4]);
}

#[test]
#[ignore = "requires STARCODER2_GGUF and CUDA 13 NVRTC/driver"]
fn starcoder2_cuda_lifecycle() {
    run_fixture("STARCODER2_GGUF", true, &[0, 4]);
}

#[test]
#[ignore = "requires GPT2_GGUF"]
fn gpt2_reference_lifecycle() {
    run_fixture("GPT2_GGUF", false, &[50256]);
}

#[test]
#[ignore = "requires GPT2_GGUF and CUDA 13 NVRTC/driver"]
fn gpt2_cuda_lifecycle() {
    run_fixture("GPT2_GGUF", true, &[50256]);
}

#[test]
#[ignore = "requires BLOOM_GGUF"]
fn bloom_reference_lifecycle() {
    run_fixture("BLOOM_GGUF", false, &[2]);
}

#[test]
#[ignore = "requires BLOOM_GGUF and CUDA 13 NVRTC/driver"]
fn bloom_cuda_lifecycle() {
    run_fixture("BLOOM_GGUF", true, &[2]);
}

#[test]
#[ignore = "requires PYTHIA_GGUF"]
fn pythia_reference_lifecycle() {
    run_fixture("PYTHIA_GGUF", false, &[0]);
}

#[test]
#[ignore = "requires PYTHIA_GGUF and CUDA 13 NVRTC/driver"]
fn pythia_cuda_lifecycle() {
    run_fixture("PYTHIA_GGUF", true, &[0]);
}

#[test]
#[ignore = "requires OLMO_GGUF"]
fn olmo_reference_lifecycle() {
    run_fixture("OLMO_GGUF", false, &[50279]);
}

#[test]
#[ignore = "requires OLMO_GGUF and CUDA 13 NVRTC/driver"]
fn olmo_cuda_lifecycle() {
    run_fixture("OLMO_GGUF", true, &[50279]);
}

#[test]
#[ignore = "requires OLMO2_GGUF"]
fn olmo2_reference_lifecycle() {
    run_fixture("OLMO2_GGUF", false, &[100257, 100265]);
}

#[test]
#[ignore = "requires OLMO2_GGUF and CUDA 13 NVRTC/driver"]
fn olmo2_cuda_lifecycle() {
    run_fixture("OLMO2_GGUF", true, &[100257, 100265]);
}

#[test]
#[ignore = "requires OLMOE_GGUF"]
fn olmoe_reference_lifecycle() {
    run_fixture("OLMOE_GGUF", false, &[50279]);
}

#[test]
#[ignore = "requires OLMOE_GGUF and CUDA 13 NVRTC/driver"]
fn olmoe_cuda_lifecycle() {
    run_fixture("OLMOE_GGUF", true, &[50279]);
}

#[test]
#[ignore = "requires QWEN35_GGUF"]
fn qwen35_reference_lifecycle() {
    qwen35_lifecycle(false);
}

#[test]
#[ignore = "requires QWEN35_GGUF and CUDA 13 NVRTC/driver"]
fn qwen35_cuda_lifecycle() {
    qwen35_lifecycle(true);
}

fn qwen35_lifecycle(cuda: bool) {
    let path = std::env::var_os("QWEN35_GGUF").expect("set the GGUF fixture path");
    let config = Config {
        context_tokens: 32,
        batch_tokens: if cuda { 32 } else { 16 },
        prefill_chunk: if cuda { 17 } else { 8 },
        threads: 4,
    };
    let path = Path::new(&path);
    let mut backend: Box<dyn Backend> = if cuda {
        let options = CudaOptions {
            decode_graph: true,
            ..CudaOptions::default()
        };
        assert!(
            palito::cuda::open_backend(path, &config, options)
                .err()
                .is_some_and(|e| e.to_string().contains("without a decode graph"))
        );
        palito::cuda::open_backend(path, &config, CudaOptions::default()).unwrap()
    } else {
        Box::new(ReferenceBackend::open(path, &config).unwrap())
    };
    exercise(backend.as_mut(), &[248044, 248046, 248063, 248064, 248065]);
    exercise_recurrent_chunks(backend.as_mut());
    if cuda {
        let phrase = backend.tokenize("A completely different prefix.").unwrap();
        let tokens: Vec<_> = phrase.into_iter().cycle().take(32).collect();
        backend.reset().unwrap();
        backend.evaluate(&tokens[..31], 0).unwrap();
        backend.evaluate(&tokens[31..], 31).unwrap();
        let expected = backend.logits().unwrap().to_vec();
        backend.reset().unwrap();
        for (i, token) in tokens.iter().enumerate() {
            backend
                .evaluate(std::slice::from_ref(token), i as u32)
                .unwrap();
        }
        assert!(
            backend
                .logits()
                .unwrap()
                .iter()
                .zip(expected)
                .all(|(a, b)| a.to_bits() == b.to_bits()),
            "DeltaNet panel and fused attention changed the continuation logits"
        );
    }
}

#[test]
#[ignore = "requires MAMBA_GGUF"]
fn mamba_reference_lifecycle() {
    let path = std::env::var_os("MAMBA_GGUF").expect("set the GGUF fixture path");
    let config = Config {
        context_tokens: 32,
        batch_tokens: 16,
        prefill_chunk: 8,
        threads: 4,
    };
    let mut backend = ReferenceBackend::open(Path::new(&path), &config).unwrap();
    exercise(&mut backend, &[0]);
    exercise_recurrent_chunks(&mut backend);
}

fn exercise_recurrent_chunks(backend: &mut dyn Backend) {
    let tokens = backend
        .tokenize("The capital of France is Paris. It is located on the Seine.")
        .unwrap();
    assert!(tokens.len() > 8);
    let expected = {
        backend.reset().unwrap();
        backend.evaluate(&tokens, 0).unwrap();
        backend.logits().unwrap().to_vec()
    };
    for chunk in [1, 4] {
        backend.reset().unwrap();
        let unrelated = backend.tokenize("A completely different prefix.").unwrap();
        backend.evaluate(&unrelated, 0).unwrap();
        backend.reset().unwrap();
        for (i, part) in tokens.chunks(chunk).enumerate() {
            backend.evaluate(part, (i * chunk) as u32).unwrap();
        }
        assert!(
            backend
                .logits()
                .unwrap()
                .iter()
                .zip(&expected)
                .all(|(actual, expected)| actual.to_bits() == expected.to_bits()),
            "recurrent state or logits changed after reset/replay with chunk {chunk}"
        );
    }
}

#[test]
#[ignore = "requires SMOLLM2_IQ4_NL_GGUF"]
fn smollm2_iq4_nl_cpu_lifecycle() {
    smollm2_cpu_lifecycle("SMOLLM2_IQ4_NL_GGUF");
}

#[test]
#[ignore = "requires SMOLLM2_IQ4_NL_GGUF and CUDA 13 NVRTC/driver"]
fn smollm2_iq4_nl_cuda_lifecycle() {
    run_fixture("SMOLLM2_IQ4_NL_GGUF", true, &[2]);
}

#[test]
#[ignore = "requires QWEN3_Q2_K_GGUF"]
fn qwen3_q2_k_reference_lifecycle() {
    run_fixture("QWEN3_Q2_K_GGUF", false, &[151643, 151645]);
}

#[test]
#[ignore = "requires QWEN3_Q2_K_GGUF and CUDA 13 NVRTC/driver"]
fn qwen3_q2_k_cuda_lifecycle() {
    run_fixture("QWEN3_Q2_K_GGUF", true, &[151643, 151645]);
}

#[test]
#[ignore = "requires QWEN25_GGUF"]
fn qwen25_reference_lifecycle() {
    run_fixture("QWEN25_GGUF", false, &[151643, 151645]);
}

#[test]
#[ignore = "requires QWEN25_GGUF and CUDA 13 NVRTC/driver"]
fn qwen25_cuda_lifecycle() {
    run_fixture("QWEN25_GGUF", true, &[151643, 151645]);
}

#[test]
#[ignore = "requires QWEN3_BF16_GGUF"]
fn qwen3_bf16_reference_lifecycle() {
    run_fixture("QWEN3_BF16_GGUF", false, &[151643, 151645]);
}

#[test]
#[ignore = "requires QWEN3_BF16_GGUF and CUDA 13 NVRTC/driver"]
fn qwen3_bf16_cuda_lifecycle() {
    run_fixture("QWEN3_BF16_GGUF", true, &[151643, 151645]);
}

#[test]
#[ignore = "requires GEMMA2_GGUF"]
fn gemma2_reference_lifecycle() {
    run_fixture("GEMMA2_GGUF", false, &[1, 107]);
}

#[test]
#[ignore = "requires GEMMA2_GGUF and CUDA 13 NVRTC/driver"]
fn gemma2_cuda_lifecycle() {
    run_fixture("GEMMA2_GGUF", true, &[1, 107]);
}

#[test]
#[ignore = "requires local compat GGUFs under models/ and CUDA 13 NVRTC/driver"]
fn fast_path_architectures_match_f32_and_select_int8_fast() {
    use palito::{
        GenerationMode,
        cuda::CudaMatmul,
        engine::{Engine, EngineOptions, SlotEngine},
    };

    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let models = [
        "models/compat-qwen25/qwen2.5-0.5b-instruct-q4_k_m.gguf",
        "models/compat-llama32/Llama-3.2-1B-Instruct-Q8_0.gguf",
        "models/compat-phi3/Phi-3-mini-4k-instruct-Q4_K_M.gguf",
        "models/Phi-3-mini-128k-instruct.Q4_K_M.gguf",
        "models/compat-olmo2/OLMo-2-0425-1B-Q4_K_M.gguf",
        "models/compat-olmoe/olmoe-1b-7b-0924-instruct-q4_k_m.gguf",
        "models/compat-gemma2/gemma-2-2b-it-Q8_0.gguf",
        "models/compat-gemma3/gemma-3-270m-it-Q8_0.gguf",
    ];
    let config = Config {
        context_tokens: 512,
        batch_tokens: 256,
        prefill_chunk: 256,
        threads: 4,
    };
    let sample_tokens = |path: &Path, matmul: CudaMatmul, decode_graph: bool| -> Vec<i32> {
        let mut backend = CudaBackend::open(
            path,
            &config,
            CudaOptions {
                matmul,
                decode_graph,
                ..CudaOptions::default()
            },
        )
        .unwrap_or_else(|e| panic!("{} ({matmul:?}, graph={decode_graph}): {e}", path.display()));
        let prompt = backend.tokenize("The capital of France is").unwrap();
        backend.evaluate_for_sampling(&prompt, 0).unwrap();
        let mut tokens = Vec::new();
        while tokens.len() < 32 {
            let chunk = backend
                .sample_tokens(32 - tokens.len(), GenerationMode::FixedLength)
                .unwrap();
            tokens.extend(chunk);
            if tokens.len() < 32 {
                let pos = (prompt.len() + tokens.len() - 1) as u32;
                backend
                    .evaluate_for_sampling(&[*tokens.last().unwrap()], pos)
                    .unwrap();
            }
        }
        tokens
    };

    for rel in models {
        let path = root.join(rel);
        let engine = Engine::open(
            &path,
            &EngineOptions {
                context_tokens: Some(512),
                ..EngineOptions::default()
            },
        )
        .unwrap_or_else(|e| panic!("Engine::open({rel}): {e}"));
        assert_eq!(engine.profile(), "int8-fast", "{rel} fell back from int8-fast");
        assert!(engine.notices().is_empty(), "{rel} notices: {:?}", engine.notices());
        drop(engine);

        let f32_tokens = sample_tokens(&path, CudaMatmul::F32, false);
        let tiled_tokens = sample_tokens(&path, CudaMatmul::F32Tiled, true);
        assert_eq!(tiled_tokens, f32_tokens, "{rel}: f32-tiled + decode_graph diverged from f32");

        let int8_tokens = sample_tokens(&path, CudaMatmul::Int8Fast, true);
        assert_eq!(int8_tokens.len(), 32, "{rel}: int8-fast short output");
        assert_eq!(
            int8_tokens[0], f32_tokens[0],
            "{rel}: int8-fast first token diverged from f32"
        );
    }

    // Verify multi-slot fast path opens cleanly on newly unlocked architectures.
    for rel in [
        "models/compat-qwen25/qwen2.5-0.5b-instruct-q4_k_m.gguf",
        "models/compat-gemma3/gemma-3-270m-it-Q8_0.gguf",
    ] {
        let path = root.join(rel);
        let engine = SlotEngine::open_slots(
            &path,
            &EngineOptions {
                context_tokens: Some(1024),
                ..EngineOptions::default()
            },
            2,
        )
        .unwrap_or_else(|e| panic!("SlotEngine::open_slots({rel}): {e}"));
        assert_eq!(engine.profile(), "int8-fast", "{rel} multi-slot fell back");
        assert_eq!(engine.slots(), 2);
    }
}
