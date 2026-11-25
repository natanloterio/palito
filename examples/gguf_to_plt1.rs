//! Converts an F16 GGUF model (or first N layers) to Palito's GPU-native
//! `PLT1` (`.palito`) format with offline RMSNorm + residual RHT folding and
//! MMA-swizzled QTIP HYB (`Q=5` zero-SMEM warp-register LUT) BlockLDLQ tiles.

#[cfg(not(feature = "qtip-research"))]
fn main() {
    eprintln!("gguf-to-plt1 requires --features qtip-research");
    std::process::exit(2);
}

#[cfg(feature = "qtip-research")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use clap::Parser;
    use palito::qtip::{
        hessian::Hessians,
        plt1_export::{Plt1ExportOptions, export_gguf_to_plt1},
    };
    use std::{path::PathBuf, time::Instant};

    #[derive(Parser)]
    struct Args {
        #[arg(long)]
        input: PathBuf,
        #[arg(long)]
        output: PathBuf,
        #[arg(long)]
        hessians: PathBuf,
        #[arg(long, default_value_t = 12)]
        state_bits: u32,
        /// Bits per pair (3=1.5bpw, 4=2.0bpw, 5=2.5bpw, 6=3.0bpw).
        #[arg(long, default_value_t = 4)]
        pair_bits: u32,
        /// LUT index bits Q (5=32 entries warp-reg, 6..9=SMEM).
        #[arg(long, default_value_t = 5)]
        lut_bits: u32,
        #[arg(long, default_value_t = 0.01)]
        damping: f32,
        #[arg(long)]
        max_layers: Option<usize>,
        #[arg(long)]
        no_fold_rht: bool,
    }

    let args = Args::parse();
    let hessians = Hessians::open(&args.hessians)?;
    let opts = Plt1ExportOptions {
        state_bits: args.state_bits,
        pair_bits: args.pair_bits,
        lut_bits: args.lut_bits,
        damping: args.damping,
        fold_residual_rht: !args.no_fold_rht,
        max_layers: args.max_layers,
        ..Default::default()
    };

    let t0 = Instant::now();
    let (reports, index) = export_gguf_to_plt1(&args.input, &args.output, &hessians, &opts, |r, i, n| {
        println!(
            "[{i:3}/{n:3}] {:30} rel={:.4} proxy={:.5}",
            r.name,
            r.relative_error,
            r.proxy_error.unwrap_or(0.0)
        );
    })?;
    let mean_rel = reports.iter().map(|r| r.relative_error).sum::<f64>() / reports.len() as f64;
    let mean_proxy = reports
        .iter()
        .filter_map(|r| r.proxy_error)
        .sum::<f64>()
        / reports.len() as f64;
    let file_size = std::fs::metadata(&args.output)?.len();
    println!(
        "Exported {} ({:.2} MB, {} tensors, target_sm={}, flags=0x{:x}) in {:.1}s | mean rel={:.4}, mean proxy={:.5}",
        args.output.display(),
        file_size as f64 / 1e6,
        index.tensors.len(),
        index.header.target_sm,
        index.header.flags,
        t0.elapsed().as_secs_f64(),
        mean_rel,
        mean_proxy
    );
    Ok(())
}
