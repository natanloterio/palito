use clap::{Parser, Subcommand};
use local_inference::{
    Error, GenerationMode, Result,
    backend::{CompletionRequest, GenerationInput, RequestBackend},
    ktransformers::{API_SOURCE_REVISION, KTransformersBackend, KTransformersConfig},
};
use serde_json::json;
use std::{
    fs::{self, OpenOptions},
    io::Write,
    path::PathBuf,
    time::Duration,
};

#[derive(Parser)]
#[command(
    version,
    about = "Request-level adapter for a local KTransformers/SGLang server"
)]
struct Cli {
    #[arg(long, default_value = "http://127.0.0.1:30000")]
    endpoint: String,
    /// Expected --served-model-name of the already running server.
    #[arg(long)]
    model: String,
    /// Expected --kt-method; no method or weight conversion is selected implicitly.
    #[arg(long)]
    kt_method: String,
    #[arg(long, default_value_t = 300, value_parser = clap::value_parser!(u64).range(1..))]
    timeout_seconds: u64,
    #[arg(long)]
    allow_expert_deferral: bool,
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    Info,
    Run {
        #[arg(long)]
        prompt: String,
        #[arg(long, default_value_t = 32)]
        tokens: u32,
    },
    /// Fixed-length request benchmark, not an isolated prefill/decode benchmark.
    Bench {
        #[arg(
            long,
            required_unless_present = "input_tokens_file",
            conflicts_with = "input_tokens_file"
        )]
        prompt_file: Option<PathBuf>,
        #[arg(
            long,
            required_unless_present = "prompt_file",
            conflicts_with = "prompt_file"
        )]
        input_tokens_file: Option<PathBuf>,
        #[arg(long, default_value_t = 32)]
        tokens: u32,
        #[arg(long, default_value_t = 1, value_parser = clap::value_parser!(u32).range(1..))]
        warmups: u32,
        #[arg(long, default_value_t = 3, value_parser = clap::value_parser!(u32).range(1..))]
        repetitions: u32,
        #[arg(long)]
        output: PathBuf,
    },
}

fn execute() -> Result<()> {
    let args = Cli::parse();
    let mut backend = KTransformersBackend::connect(KTransformersConfig {
        endpoint: args.endpoint,
        expected_model: args.model,
        expected_method: args.kt_method,
        timeout: Duration::from_secs(args.timeout_seconds),
        allow_expert_deferral: args.allow_expert_deferral,
    })?;
    match args.command {
        Command::Info => println!("{}", serde_json::to_string_pretty(backend.info())?),
        Command::Run { prompt, tokens } => {
            let completion = backend.generate(&CompletionRequest {
                input: GenerationInput::Text(prompt),
                max_tokens: tokens,
                mode: GenerationMode::UntilEog,
            })?;
            println!("{}", completion.text);
        }
        Command::Bench {
            prompt_file,
            input_tokens_file,
            tokens,
            warmups,
            repetitions,
            output,
        } => {
            if output.exists() {
                return Err(Error::Invalid(
                    "report exists; choose a new --output".into(),
                ));
            }
            if backend.info().prefix_cache_enabled {
                return Err(Error::Invalid("benchmark requires server --disable-radix-cache; cache is never flushed by this client".into()));
            }
            let request = CompletionRequest {
                input: match (prompt_file, input_tokens_file) {
                    (Some(path), None) => GenerationInput::Text(fs::read_to_string(path)?),
                    (None, Some(path)) => {
                        GenerationInput::TokenIds(serde_json::from_slice(&fs::read(path)?)?)
                    }
                    _ => {
                        return Err(Error::Invalid(
                            "provide exactly one prompt or input-token file".into(),
                        ));
                    }
                },
                max_tokens: tokens,
                mode: GenerationMode::FixedLength,
            };
            let mut reference = None;
            let mut prompt_count = None;
            let mut samples = Vec::new();
            for index in 0..u64::from(warmups) + u64::from(repetitions) {
                let sample = backend.generate(&request)?;
                if sample.cached_tokens != 0
                    || reference
                        .as_ref()
                        .is_some_and(|previous| previous != &sample.generated_tokens)
                    || prompt_count.is_some_and(|count| count != sample.prompt_tokens)
                {
                    return Err(Error::Backend("benchmark received cached work, changed tokenization, or non-repeatable greedy output".into()));
                }
                reference = Some(sample.generated_tokens.clone());
                prompt_count = Some(sample.prompt_tokens);
                if index >= u64::from(warmups) {
                    samples.push(sample);
                }
            }
            let report = json!({
                "schema_version": 1, "runner": "rust-ktransformers-http",
                "api_source_revision": API_SOURCE_REVISION, "model": backend.info(),
                "timing_scope": "HTTP request + server queue/tokenization/generation + response parsing; no phase/TTFT split",
                "sampling": "greedy, fixed-length including EOG; no client chat template",
                "input": request.input,
                "warmups": warmups, "requested_tokens": tokens, "samples": samples,
            });
            if let Some(parent) = output.parent().filter(|path| !path.as_os_str().is_empty()) {
                fs::create_dir_all(parent)?;
            }
            let mut file = OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&output)?;
            serde_json::to_writer_pretty(&mut file, &report)?;
            file.write_all(b"\n")?;
            file.sync_all()?;
            println!("Saved request-level benchmark: {}", output.display());
        }
    }
    Ok(())
}

fn main() {
    if let Err(error) = execute() {
        eprintln!("ktransformers: {error}");
        std::process::exit(1);
    }
}
