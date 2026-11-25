//! Perplexity as llama.cpp's `llama-perplexity` computes it: the text is cut
//! into `--context`-token chunks, each starting with BOS, and only the second
//! half of each chunk is scored. With `--prefill`, the unscored first half
//! is one batched prefill call, so the scores also measure the prefill path
//! (its KV cache), not only decode.
//!
//!     cargo run --release --features cuda --example perplexity -- \
//!         --model model.gguf --text wiki.test.raw --chunks 40
use clap::Parser;
use palito::{Config, Error, Result, backend::Backend};
use std::{fs, path::PathBuf, time::Instant};

#[derive(Parser)]
struct Args {
    #[arg(long)]
    model: PathBuf,
    #[arg(long)]
    text: PathBuf,
    #[arg(long, default_value_t = 512)]
    context: u32,
    /// At most this many chunks (0: all).
    #[arg(long, default_value_t = 0)]
    chunks: usize,
    /// CPU reference backend instead of CUDA.
    #[arg(long)]
    cpu: bool,
    #[arg(long, default_value_t = 16)]
    threads: u32,
    /// Evaluate the unscored first half of each chunk in one prefill call.
    #[arg(long)]
    prefill: bool,
    /// CUDA matrix profile (f32, f32-tiled, ...).
    #[cfg(feature = "cuda")]
    #[arg(long, default_value = "f32")]
    cuda_matmul: palito::cuda::CudaMatmul,
}

fn open(args: &Args) -> Result<Box<dyn Backend>> {
    let config = Config {
        context_tokens: args.context,
        batch_tokens: args.context,
        prefill_chunk: args.context,
        threads: args.threads,
    };
    if args.cpu {
        return Ok(Box::new(palito::reference::ReferenceBackend::open(&args.model, &config)?));
    }
    #[cfg(feature = "cuda")]
    return palito::cuda::open_backend(
        &args.model,
        &config,
        palito::cuda::CudaOptions {
            matmul: args.cuda_matmul,
            ..Default::default()
        },
    );
    #[cfg(not(feature = "cuda"))]
    Err(Error::Invalid("built without CUDA; pass --cpu".into()))
}

/// -log p(target) from raw logits, in f64.
fn negative_log_likelihood(logits: &[f32], target: i32) -> f64 {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let sum: f64 = logits.iter().map(|&l| f64::from(l - max).exp()).sum();
    sum.ln() - f64::from(logits[target as usize] - max)
}

fn main() -> Result<()> {
    let args = Args::parse();
    let mut backend = open(&args)?;
    let bos = *backend
        .tokenize("")?
        .first()
        .ok_or_else(|| Error::Invalid("model adds no BOS token".into()))?;
    let text = fs::read_to_string(&args.text)?;
    let mut tokens = backend.tokenize(&text)?;
    if tokens.first() == Some(&bos) {
        tokens.remove(0);
    }
    let context = args.context as usize;
    let available = tokens.len() / context;
    let chunks = if args.chunks == 0 { available } else { args.chunks.min(available) };
    if chunks == 0 {
        return Err(Error::Invalid("text is shorter than one chunk".into()));
    }
    eprintln!(
        "{} on {}: {} tokens, {chunks} chunks of {context}",
        backend.info().backend,
        args.model.display(),
        tokens.len()
    );
    let first_scored = context / 2;
    let (mut total, mut count) = (0.0f64, 0usize);
    let started = Instant::now();
    for chunk in 0..chunks {
        let mut window = tokens[chunk * context..(chunk + 1) * context].to_vec();
        window[0] = bos;
        backend.reset()?;
        let start = if args.prefill {
            backend.evaluate(&window[..first_scored], 0)?;
            first_scored
        } else {
            0
        };
        for (position, &token) in window.iter().enumerate().take(context - 1).skip(start) {
            backend.evaluate(&[token], position as u32)?;
            if position >= first_scored {
                total += negative_log_likelihood(backend.logits()?, window[position + 1]);
                count += 1;
            }
        }
        eprintln!(
            "[{}/{chunks}] ppl {:.4} ({:.0}s)",
            chunk + 1,
            (total / count as f64).exp(),
            started.elapsed().as_secs_f64()
        );
    }
    println!("perplexity {:.4} over {count} tokens", (total / count as f64).exp());
    Ok(())
}
