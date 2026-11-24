//! Writes a quality-probe GGUF: an F16 model whose attention and MLP matrices
//! hold QTIP-style quantized-then-reconstructed weights.
//!
//!     cargo run --release --features qtip-research --example qtip-quantize -- \
//!         --input model-f16.gguf --output model-qtip.gguf --method trellis
//!
//! `--method trellis-ldlq` needs `--hessians`, a directory written by
//! `scripts/qtip/collect_hessians.py`. With `--hessians`, every method also
//! reports the proxy loss tr(E H E^T) / tr(W H W^T).
use clap::{Parser, ValueEnum};
use palito::qtip::{
    hessian::Hessians,
    model::{Method, quantize_gguf},
};
use std::{path::PathBuf, time::Instant};

#[derive(Clone, Copy, ValueEnum)]
enum MethodArg {
    Trellis,
    TrellisLdlq,
    Scalar,
    RotationOnly,
    Noise,
}

#[derive(Parser)]
struct Args {
    #[arg(long)]
    input: PathBuf,
    #[arg(long)]
    output: PathBuf,
    #[arg(long, value_enum, default_value_t = MethodArg::Trellis)]
    method: MethodArg,
    #[arg(long, default_value_t = 12)]
    state_bits: u32,
    #[arg(long, default_value_t = 2)]
    symbol_bits: u32,
    /// Relative noise energy for --method noise.
    #[arg(long, default_value_t = 0.073)]
    noise: f32,
    /// Directory of input Hessians (manifest.json and .f32 files).
    #[arg(long)]
    hessians: Option<PathBuf>,
    /// LDLQ damping, relative to the mean of diag(H).
    #[arg(long, default_value_t = 0.01)]
    damping: f32,
    /// Block-diagonal Hadamard of this size (power of two); default: whole rows/columns.
    #[arg(long)]
    hadamard_block: Option<usize>,
    #[arg(long, default_value_t = 0x5EED)]
    seed: u64,
}

fn main() -> palito::Result<()> {
    let args = Args::parse();
    let method = match args.method {
        MethodArg::Trellis => Method::Trellis {
            state_bits: args.state_bits,
            symbol_bits: args.symbol_bits,
        },
        MethodArg::TrellisLdlq => Method::TrellisLdlq {
            state_bits: args.state_bits,
            symbol_bits: args.symbol_bits,
            damping: args.damping,
        },
        MethodArg::Scalar => Method::Scalar2Bit,
        MethodArg::RotationOnly => Method::RotationOnly,
        MethodArg::Noise => Method::Noise {
            relative: args.noise,
        },
    };
    let started = Instant::now();
    let hessians = args.hessians.as_deref().map(Hessians::open).transpose()?;
    let reports = quantize_gguf(
        &args.input,
        &args.output,
        method,
        args.seed,
        hessians.as_ref(),
        args.hadamard_block,
        |report, done, total| {
            let proxy = report
                .proxy_error
                .map_or(String::new(), |p| format!(", proxy {p:.4}"));
            eprintln!(
                "[{done}/{total}] {} relative error {:.4}{proxy} ({:.0}s)",
                report.name,
                report.relative_error,
                started.elapsed().as_secs_f64()
            );
        },
    )?;
    let count = reports.len() as f64;
    let mean = reports.iter().map(|r| r.relative_error).sum::<f64>() / count;
    let proxy = reports
        .iter()
        .map(|r| r.proxy_error)
        .sum::<Option<f64>>()
        .map(|p| p / count);
    let proxy = proxy.map_or(String::new(), |p| format!(", mean proxy {p:.4}"));
    println!(
        "{} tensors, mean relative error {mean:.4}{proxy}",
        reports.len()
    );
    Ok(())
}
