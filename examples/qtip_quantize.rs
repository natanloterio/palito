//! Writes a quality-probe GGUF: an F16 model whose attention and MLP matrices
//! hold QTIP-style quantized-then-reconstructed weights.
//!
//!     cargo run --release --features qtip-research --example qtip-quantize -- \
//!         --input model-f16.gguf --output model-qtip.gguf --method trellis
use clap::{Parser, ValueEnum};
use palito::qtip::model::{Method, quantize_gguf};
use std::{path::PathBuf, time::Instant};

#[derive(Clone, Copy, ValueEnum)]
enum MethodArg {
    Trellis,
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
        MethodArg::Scalar => Method::Scalar2Bit,
        MethodArg::RotationOnly => Method::RotationOnly,
        MethodArg::Noise => Method::Noise { relative: args.noise },
    };
    let started = Instant::now();
    let reports = quantize_gguf(&args.input, &args.output, method, args.seed, |report, done, total| {
        eprintln!(
            "[{done}/{total}] {} relative error {:.4} ({:.0}s)",
            report.name,
            report.relative_error,
            started.elapsed().as_secs_f64()
        );
    })?;
    let mean = reports.iter().map(|r| r.relative_error).sum::<f64>() / reports.len() as f64;
    println!("{} tensors, mean relative error {mean:.4}", reports.len());
    Ok(())
}
