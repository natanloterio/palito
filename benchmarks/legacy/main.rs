use clap::{Args, Parser, Subcommand};
use palito::{Config, Executor, GenerationMode, Result, llama::LlamaBackend};
use serde_json::json;
use std::{fs, path::PathBuf, time::Instant};

#[derive(Parser)]
#[command(version, about)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Print paths to the native reference and official llama-bench.
    Paths,
    /// Run fixed-length, greedy, warm-cache benchmark samples.
    Bench {
        #[command(flatten)]
        model: ModelArgs,
        #[arg(long)]
        prompt_file: PathBuf,
        #[arg(long, default_value_t = 1, value_parser = clap::value_parser!(u32).range(1..))]
        warmups: u32,
        #[arg(long, default_value_t = 3, value_parser = clap::value_parser!(u32).range(1..))]
        repetitions: u32,
    },
    /// Generate greedy text, stopping at an end-of-generation token.
    Run {
        #[command(flatten)]
        model: ModelArgs,
        #[arg(long)]
        prompt: String,
    },
}

#[derive(Args)]
struct ModelArgs {
    #[arg(long)]
    model: PathBuf,
    #[arg(long, default_value_t = 512)]
    context: u32,
    #[arg(long, default_value_t = 128)]
    batch: u32,
    #[arg(long, default_value_t = 64)]
    prefill_chunk: u32,
    #[arg(long, default_value_t = 4)]
    threads: u32,
    #[arg(long, default_value_t = 32)]
    tokens: u32,
}

impl ModelArgs {
    fn config(&self) -> Config {
        Config {
            context_tokens: self.context,
            batch_tokens: self.batch,
            prefill_chunk: self.prefill_chunk,
            threads: self.threads,
        }
    }
}

fn execute() -> Result<()> {
    match Cli::parse().command {
        Command::Paths => println!(
            "{}",
            json!({
                "native_reference": concat!(env!("LI_NATIVE_BIN"), "/li-reference"),
                "llama_bench": concat!(env!("LI_NATIVE_BIN"), "/llama-bench"),
                "llama_revision": env!("LI_LLAMA_REVISION"),
                "build_profile": env!("LI_BUILD_PROFILE"),
            })
        ),
        Command::Bench {
            model,
            prompt_file,
            warmups,
            repetitions,
        } => {
            let prompt = fs::read_to_string(prompt_file)?;
            let config = model.config();
            let start = Instant::now();
            let backend = LlamaBackend::open(&model.model, &config)?;
            let load_ns = start.elapsed().as_nanos();
            let mut executor = Executor::new(backend, config.clone())?;
            let tokens = executor.tokenize(&prompt)?;
            for _ in 0..warmups {
                executor.generate(&tokens, model.tokens, GenerationMode::FixedLength)?;
            }
            let mut samples = Vec::new();
            for _ in 0..repetitions {
                samples.push(executor.generate(
                    &tokens,
                    model.tokens,
                    GenerationMode::FixedLength,
                )?);
            }
            println!(
                "{}",
                serde_json::to_string(&json!({
                    "schema_version": 1, "runner": "rust", "load_ns": load_ns,
                    "model": executor.info(), "config": config,
                    "prompt_tokens": tokens.len(), "prompt_token_ids": tokens, "samples": samples,
                }))?
            );
        }
        Command::Run { model, prompt } => {
            let config = model.config();
            let backend = LlamaBackend::open(&model.model, &config)?;
            let mut executor = Executor::new(backend, config)?;
            let tokens = executor.tokenize(&prompt)?;
            let sample = executor.generate(&tokens, model.tokens, GenerationMode::UntilEog)?;
            println!("{}", executor.detokenize(&sample.generated_tokens)?);
        }
    }
    Ok(())
}

fn main() {
    if let Err(error) = execute() {
        eprintln!("palito: {error}");
        std::process::exit(1);
    }
}
