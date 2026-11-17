use clap::Parser;
use local_inference::{Config, Result, backend::Backend, cuda::reference::ReferenceBackend};
use std::{fs, path::PathBuf};

#[derive(Parser)]
struct Args {
    #[arg(long)]
    model: PathBuf,
    #[arg(long)]
    input: PathBuf,
    #[arg(long)]
    output: PathBuf,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let ids: Vec<i32> = serde_json::from_slice(&fs::read(args.input)?)?;
    if ids.is_empty() || ids.len() > 1024 {
        return Err(local_inference::Error::Invalid(
            "trace needs 1..=1024 tokens".into(),
        ));
    }
    let config = Config {
        context_tokens: 1024,
        batch_tokens: 128,
        prefill_chunk: 128,
        threads: 4,
    };
    let mut backend = ReferenceBackend::open(&args.model, &config)?;
    backend.trace_at(&args.output, ids.len() - 1)?;
    for (chunk, values) in ids.chunks(128).enumerate() {
        backend.evaluate(values, (chunk * 128) as u32)?;
    }
    println!("{}", serde_json::json!({"logits": backend.logits()?}));
    Ok(())
}
