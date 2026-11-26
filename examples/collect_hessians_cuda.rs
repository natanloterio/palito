//! Thin CLI wrapper around `palito::qtip::hessian_cuda::collect_hessians_cuda`.

use clap::Parser;
use palito::{
    Result,
    cuda::CudaMatmul,
    qtip::hessian_cuda::{CudaHessianOptions, collect_hessians_cuda},
};
use std::path::PathBuf;

#[derive(Parser)]
struct Args {
    /// Path to the GGUF model.
    #[arg(long)]
    model: PathBuf,
    /// Path to calibration text (e.g. `models/wikitext-2-raw/wiki.train.raw`).
    #[arg(long)]
    text: Option<PathBuf>,
    /// Output directory for `manifest.json` and `.f32` Hessian files.
    #[arg(long)]
    output: PathBuf,
    /// Number of calibration chunks to process.
    #[arg(long, default_value_t = 128)]
    chunks: usize,
    /// Tokens per calibration chunk.
    #[arg(long, default_value_t = 512)]
    context: u32,
    /// Number of layers to accumulate per forward pass (0 = all layers in one pass).
    #[arg(long, default_value_t = 16)]
    layers_per_pass: usize,
    /// Skip collecting per-expert `ffn_down_exps.f32` (saves RAM/disk/time).
    #[arg(long)]
    skip_down_exps: bool,
    /// Blend per-expert `ffn_down_exps` Hessians toward scaled identity when
    /// an expert observes fewer than this many tokens.
    #[arg(long, default_value_t = 32)]
    min_expert_tokens: u64,
    /// CUDA matmul profile.
    #[arg(long, value_enum, default_value_t = CudaMatmul::F32)]
    matmul: CudaMatmul,
    /// CPU threads for tokenization / backend setup.
    #[arg(long, default_value_t = 8)]
    threads: u32,
}

fn main() -> Result<()> {
    let args = Args::parse();
    collect_hessians_cuda(&CudaHessianOptions {
        model: args.model,
        text: args.text,
        output: args.output,
        chunks: args.chunks,
        context: args.context,
        layers_per_pass: args.layers_per_pass,
        skip_down_exps: args.skip_down_exps,
        min_expert_tokens: args.min_expert_tokens,
        matmul: args.matmul,
        threads: args.threads,
    })
}
