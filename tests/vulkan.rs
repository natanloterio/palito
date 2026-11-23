//! Native Vulkan compute backend tests.
#![cfg(feature = "vulkan")]

use palito::{
    Config, GenerationMode,
    backend::Backend,
    cpu::{CpuBackend, CpuOptions},
    engine::{Device, Engine, EngineOptions, GenerationOptions},
    vulkan::{VulkanBackend, VulkanOptions},
};
use std::path::Path;

fn config() -> Config {
    Config {
        context_tokens: 512,
        batch_tokens: 128,
        prefill_chunk: 64,
        threads: 4,
    }
}

#[test]
#[ignore = "requires a Vulkan GPU and models/SmolLM2-135M-Instruct-Q4_K_M.gguf"]
fn vulkan_matches_cpu_tokens_and_resets_cleanly() {
    let model = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("models/SmolLM2-135M-Instruct-Q4_K_M.gguf");
    let cfg = config();
    let mut cpu = CpuBackend::open_with_options(&model, &cfg, CpuOptions::default()).unwrap();
    let mut vk = VulkanBackend::open(&model, &cfg, VulkanOptions::default()).unwrap();
    assert!(vk.info().backend.starts_with("vulkan"));
    assert!(vk.memory_usage().unwrap().weights_allocated_bytes > 0);

    let prompt = vk.tokenize("The capital of France is").unwrap();
    cpu.evaluate(&prompt, 0).unwrap();
    vk.evaluate(&prompt, 0).unwrap();

    let mut cpu_tokens = Vec::new();
    let mut vk_tokens = Vec::new();
    for step in 0..16 {
        let ct = cpu.sample_greedy().unwrap();
        let vt = vk.sample_greedy().unwrap();
        cpu_tokens.push(ct);
        vk_tokens.push(vt);
        let pos = (prompt.len() + step) as u32;
        cpu.evaluate(&[ct], pos).unwrap();
        vk.evaluate(&[vt], pos).unwrap();
    }
    assert_eq!(vk_tokens, cpu_tokens);

    // Reset must discard state and reproduce the exact first token.
    vk.reset().unwrap();
    assert!(vk.logits().is_err());
    vk.evaluate(&prompt, 0).unwrap();
    assert_eq!(vk.sample_greedy().unwrap(), vk_tokens[0]);
}

#[test]
#[ignore = "requires a Vulkan GPU and local compat GGUFs under models/"]
fn vulkan_engine_runs_modern_architectures() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let models = [
        "models/compat-qwen3/Qwen3-0.6B-Q4_K_M.gguf",
        "models/compat-qwen25/qwen2.5-0.5b-instruct-q4_k_m.gguf",
        "models/compat-llama32/Llama-3.2-1B-Instruct-Q8_0.gguf",
        "models/compat-phi3/Phi-3-mini-4k-instruct-Q4_K_M.gguf",
        "models/compat-olmo2/OLMo-2-0425-1B-Q4_K_M.gguf",
        "models/compat-olmoe/olmoe-1b-7b-0924-instruct-q4_k_m.gguf",
        "models/compat-gemma2/gemma-2-2b-it-Q8_0.gguf",
        "models/compat-gemma3/gemma-3-270m-it-Q8_0.gguf",
    ];
    let gen_opts = GenerationOptions {
        max_tokens: Some(8),
        ..GenerationOptions::default()
    };
    for rel in models {
        let path = root.join(rel);
        let mut engine = Engine::open(
            &path,
            &EngineOptions {
                device: Device::Vulkan,
                context_tokens: Some(512),
                ..EngineOptions::default()
            },
        )
        .unwrap_or_else(|e| panic!("Vulkan Engine::open({rel}): {e}"));
        assert_eq!(engine.profile(), "vulkan");
        let reply = engine
            .complete("The capital of France is", &gen_opts, |_| true)
            .unwrap();
        assert_eq!(reply.generated_tokens, 8, "{rel}: expected 8 tokens");
        assert!(!reply.text.is_empty(), "{rel}: empty reply");
    }
    let _ = GenerationMode::FixedLength;
}
