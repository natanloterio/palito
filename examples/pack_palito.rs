//! Bundles a base GGUF (tokenizer, metadata, embeddings, LM head, norms),
//! a PLT1 HYB Q=5 layers file, and an optional PLT1 EAGLE-3 draft head into
//! ONE self-contained `.palito` file that `palito chat`, `palito serve`, and
//! `palito generate` load directly for 1,000+ tok/s out of the box!

#[cfg(not(feature = "qtip-research"))]
fn main() {
    eprintln!("pack-palito requires --features qtip-research");
    std::process::exit(2);
}

#[cfg(feature = "qtip-research")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use clap::Parser;
    use std::path::PathBuf;

    #[derive(Parser)]
    struct Args {
        /// Base GGUF providing tokenizer, metadata, embed_tokens, output.weight, norms.
        #[arg(long)]
        base_gguf: PathBuf,
        /// PLT1 (.palito) file containing the quantized HYB Q=5 layer projections.
        #[arg(long)]
        plt1_layers: PathBuf,
        /// Optional PLT1 (.palito) file containing the EAGLE-3 draft head.
        #[arg(long)]
        eagle3: Option<PathBuf>,
        /// Output self-contained `.palito` bundle path.
        #[arg(long)]
        output: PathBuf,
    }

    let args = Args::parse();
    let report = palito::qtip::plt1_export::pack_palito_bundle(
        &args.base_gguf,
        &args.plt1_layers,
        args.eagle3.as_deref(),
        &args.output,
    )?;
    println!(
        "Skeleton GGUF (tokenizer + metadata + embed + lm_head + norms): {:.2} MB",
        report.skeleton_bytes as f64 / 1e6
    );
    if report.eagle3_tensors > 0 {
        println!("Embedded EAGLE-3 draft head ({} tensors)", report.eagle3_tensors);
    }
    println!(
        "Wrote self-contained bundle: {} ({:.2} MB)",
        args.output.display(),
        report.total_bytes as f64 / 1e6
    );
    Ok(())
}
