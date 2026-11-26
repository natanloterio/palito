//! Native Apple Metal compute backend tests (macOS Apple Silicon).
#![cfg(all(feature = "metal", target_os = "macos"))]

use palito::{
    Config,
    backend::Backend,
    cpu::{CpuBackend, CpuOptions},
    metal::{MetalBackend, MetalOptions},
};
use std::path::Path;

#[test]
#[ignore = "requires macOS Apple Silicon and models/SmolLM2-135M-Instruct-Q4_K_M.gguf"]
fn metal_matches_cpu_tokens_and_resets_cleanly() {
    let model = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("models/SmolLM2-135M-Instruct-Q4_K_M.gguf");
    let cfg = Config {
        context_tokens: 512,
        batch_tokens: 128,
        prefill_chunk: 64,
        threads: 4,
    };
    let mut cpu = CpuBackend::open_with_options(&model, &cfg, CpuOptions::default()).unwrap();
    let mut mtl = MetalBackend::open(&model, &cfg, MetalOptions::default()).unwrap();
    assert!(mtl.info().backend.starts_with("metal"));
    assert!(mtl.memory_usage().unwrap().weights_allocated_bytes > 0);

    let prompt = mtl.tokenize("The capital of France is").unwrap();
    cpu.evaluate(&prompt, 0).unwrap();
    mtl.evaluate(&prompt, 0).unwrap();

    let mut cpu_tokens = Vec::new();
    let mut mtl_tokens = Vec::new();
    for step in 0..16 {
        let ct = cpu.sample_greedy().unwrap();
        let mt = mtl.sample_greedy().unwrap();
        cpu_tokens.push(ct);
        mtl_tokens.push(mt);
        let pos = (prompt.len() + step) as u32;
        cpu.evaluate(&[ct], pos).unwrap();
        mtl.evaluate(&[mt], pos).unwrap();
    }
    assert_eq!(mtl_tokens, cpu_tokens);

    mtl.reset().unwrap();
    assert!(mtl.logits().is_err());
    mtl.evaluate(&prompt, 0).unwrap();
    assert_eq!(mtl.sample_greedy().unwrap(), mtl_tokens[0]);
}

/// `.palito` (PLT1) weights have only CUDA and CPU kernels: Metal must refuse
/// them before touching the GPU instead of running the bundle's GGUF skeleton.
#[test]
fn metal_refuses_palito_models() {
    let path = std::env::temp_dir().join(format!("palito-mtl-plt1-{}.palito", std::process::id()));
    std::fs::write(&path, b"PALITO1\0not a real bundle").unwrap();
    let cfg = Config { context_tokens: 512, batch_tokens: 128, prefill_chunk: 64, threads: 4 };
    let result = MetalBackend::open(&path, &cfg, MetalOptions::default());
    let _ = std::fs::remove_file(&path);
    let Err(error) = result else { panic!("Metal opened a .palito model") };
    let text = error.to_string();
    assert!(text.contains(".palito") && text.contains("--backend cpu"), "{text}");
}
