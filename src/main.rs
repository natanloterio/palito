use clap::{ArgGroup, Args, Parser, Subcommand, ValueEnum};
use local_inference::{
    Config, Executor, GenerationMode, Result,
    backend::Backend,
    cpu::{CpuBackend, CpuOptions},
    tokenizer::Tokenizer,
};
use serde_json::json;
use std::{fs, path::PathBuf, time::Instant};

#[derive(Parser)]
#[command(
    version,
    about = "Independent Rust GGUF inference with optional CUDA and KTransformers adapters"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Whole-request KTransformers service; not Rust token-step/KV execution.
    #[cfg(feature = "ktransformers")]
    Remote {
        #[command(flatten)]
        args: RemoteArgs,
    },
    /// Greedy completion, stopping at EOG. No chat template is applied.
    Generate {
        #[command(flatten)]
        model: ModelArgs,
    },
    /// Tokenize text offline without loading model weights.
    Tokenize {
        #[arg(long)]
        model: PathBuf,
        #[command(flatten)]
        text: TextArgs,
    },
    /// Decode IDs offline, omitting control tokens.
    Detokenize {
        #[arg(long)]
        model: PathBuf,
        #[arg(long)]
        input_tokens_file: PathBuf,
    },
    /// Fixed-length generation with per-phase timing and full state reset per sample.
    Bench {
        #[command(flatten)]
        model: ModelArgs,
        #[arg(long, default_value_t = 1)]
        warmups: u32,
        #[arg(long, default_value_t = 3, value_parser = clap::value_parser!(u32).range(1..))]
        repetitions: u32,
    },
    /// Evaluate input IDs and emit F32 next-token logits for differential validation.
    Logits {
        #[command(flatten)]
        model: ModelArgs,
    },
}

#[derive(Clone, Copy, PartialEq, Eq, ValueEnum)]
enum LocalBackend {
    Cpu,
    #[cfg(feature = "cuda")]
    Cuda,
    /// Slow independent packed CPU oracle for the CUDA model layouts.
    #[cfg(feature = "cuda")]
    Reference,
}

#[derive(Args)]
#[command(group(ArgGroup::new("input").required(true).multiple(false)
    .args(["input_tokens_file", "prompt", "prompt_file"])))]
struct ModelArgs {
    #[arg(long, value_enum, default_value_t = LocalBackend::Cpu)]
    backend: LocalBackend,
    #[arg(long)]
    model: PathBuf,
    #[arg(long)]
    input_tokens_file: Option<PathBuf>,
    #[arg(long, allow_hyphen_values = true)]
    prompt: Option<String>,
    #[arg(long)]
    prompt_file: Option<PathBuf>,
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
    /// Expand weights for differential validation rather than retaining packed GGUF blocks.
    #[arg(long)]
    f32_weights: bool,
    /// Evaluate prefill one token at a time for differential validation.
    #[arg(long)]
    sequential_prefill: bool,
    /// Bound numerical reservations, not process RSS or transient model loading.
    #[arg(long)]
    memory_limit_bytes: Option<u64>,
    /// Opt-in, one-entry immutable prefix cache with a numerical payload budget.
    #[arg(long, default_value_t = 0)]
    prefix_cache_bytes: u64,
    /// Opt-in n-gram draft verification; zero disables speculation.
    #[arg(long, default_value_t = 0)]
    speculative_window: u32,
    /// Disable runtime-dispatched AVX2 specialization for validation.
    #[arg(long)]
    scalar_kernels: bool,
    #[cfg(feature = "cuda")]
    #[arg(long, requires = "backend")]
    cuda_device: Option<usize>,
    /// Device-buffer limit; excludes driver/JIT overhead and host model loading.
    #[cfg(feature = "cuda")]
    #[arg(long)]
    gpu_memory_limit_bytes: Option<u64>,
    /// Opt-in private directory for persistent CUDA kernel compilation cache.
    #[cfg(feature = "cuda")]
    #[arg(long)]
    cuda_kernel_cache: Option<PathBuf>,
    /// Prompt projection mode; tiled modes change only experts, default preserves F32 arithmetic.
    #[cfg(feature = "cuda")]
    #[arg(long, value_enum, default_value_t = local_inference::cuda::CudaMatmul::F32)]
    cuda_matmul: local_inference::cuda::CudaMatmul,
    /// Opt-in captured decode with token-only greedy output; context <=1024.
    #[cfg(feature = "cuda")]
    #[arg(long)]
    cuda_decode_graph: bool,
}

#[derive(Args)]
#[group(required = true, multiple = false)]
struct TextArgs {
    #[arg(long, allow_hyphen_values = true)]
    prompt: Option<String>,
    #[arg(long)]
    prompt_file: Option<PathBuf>,
}

#[cfg(feature = "ktransformers")]
#[derive(Args)]
#[command(group(ArgGroup::new("remote_input").required(true).multiple(false)
    .args(["input_tokens_file", "prompt", "prompt_file"])))]
struct RemoteArgs {
    #[arg(long, default_value = "http://127.0.0.1:30000")]
    endpoint: String,
    /// Expected --served-model-name, not a local GGUF filename.
    #[arg(long)]
    model: String,
    #[arg(long)]
    kt_method: String,
    #[arg(long, default_value_t = 300, value_parser = clap::value_parser!(u64).range(1..))]
    timeout_seconds: u64,
    #[arg(long)]
    allow_expert_deferral: bool,
    #[arg(long)]
    input_tokens_file: Option<PathBuf>,
    #[arg(long, allow_hyphen_values = true)]
    prompt: Option<String>,
    #[arg(long)]
    prompt_file: Option<PathBuf>,
    #[arg(long, default_value_t = 32)]
    tokens: u32,
    #[arg(long)]
    fixed_length: bool,
}

#[cfg(feature = "ktransformers")]
fn remote(args: RemoteArgs) -> Result<()> {
    use local_inference::{
        backend::{CompletionRequest, GenerationInput, RequestBackend},
        ktransformers::{KTransformersBackend, KTransformersConfig},
    };
    let input = match args.input_tokens_file {
        Some(path) => GenerationInput::TokenIds(serde_json::from_slice(&fs::read(path)?)?),
        None => GenerationInput::Text(text_input(&args.prompt, &args.prompt_file)?),
    };
    let mut backend = KTransformersBackend::connect(KTransformersConfig {
        endpoint: args.endpoint,
        expected_model: args.model,
        expected_method: args.kt_method,
        timeout: std::time::Duration::from_secs(args.timeout_seconds),
        allow_expert_deferral: args.allow_expert_deferral,
    })?;
    let completion = backend.generate(&CompletionRequest {
        input,
        max_tokens: args.tokens,
        mode: if args.fixed_length {
            GenerationMode::FixedLength
        } else {
            GenerationMode::UntilEog
        },
    })?;
    println!(
        "{}",
        serde_json::to_string(&json!({
            "runner": "rust-ktransformers-http", "model": backend.info(), "completion": completion,
            "execution_scope": "SGLang/KT owns token steps, experts and KV; elapsed includes HTTP and scheduling",
        }))?
    );
    Ok(())
}

fn text_input(prompt: &Option<String>, file: &Option<PathBuf>) -> Result<String> {
    match (prompt, file) {
        (Some(text), None) => Ok(text.clone()),
        (None, Some(path)) => Ok(fs::read_to_string(path)?),
        _ => Err(local_inference::Error::Invalid(
            "supply exactly one of --prompt and --prompt-file".into(),
        )),
    }
}

impl ModelArgs {
    fn options(&self) -> CpuOptions {
        CpuOptions {
            expand_weights: self.f32_weights,
            sequential_prefill: self.sequential_prefill,
            memory_limit_bytes: self.memory_limit_bytes,
            prefix_cache_bytes: self.prefix_cache_bytes,
            speculative_window: self.speculative_window,
            scalar_kernels: self.scalar_kernels,
        }
    }

    fn open(&self) -> Result<Box<dyn Backend>> {
        #[cfg(feature = "cuda")]
        {
            if self.backend != LocalBackend::Cuda
                && (self.cuda_device.is_some()
                    || self.gpu_memory_limit_bytes.is_some()
                    || self.cuda_kernel_cache.is_some()
                    || self.cuda_matmul != local_inference::cuda::CudaMatmul::F32
                    || self.cuda_decode_graph)
            {
                return Err(local_inference::Error::Invalid(
                    "CUDA options require --backend cuda".into(),
                ));
            }
            if self.backend != LocalBackend::Cpu
                && (self.f32_weights
                    || self.sequential_prefill
                    || self.memory_limit_bytes.is_some()
                    || self.prefix_cache_bytes != 0
                    || self.speculative_window != 0
                    || self.scalar_kernels)
            {
                return Err(local_inference::Error::Invalid(
                    "CPU policy options require --backend cpu".into(),
                ));
            }
            match self.backend {
                LocalBackend::Cuda => {
                    return Ok(Box::new(local_inference::cuda::CudaBackend::open(
                        &self.model,
                        &self.config(),
                        self.cuda_options(),
                    )?));
                }
                LocalBackend::Reference => {
                    return Ok(Box::new(
                        local_inference::cuda::reference::ReferenceBackend::open(
                            &self.model,
                            &self.config(),
                        )?,
                    ));
                }
                LocalBackend::Cpu => {}
            }
        }
        Ok(Box::new(CpuBackend::open_with_options(
            &self.model,
            &self.config(),
            self.options(),
        )?))
    }

    #[cfg(feature = "cuda")]
    fn cuda_options(&self) -> local_inference::cuda::CudaOptions {
        let mut options = local_inference::cuda::CudaOptions::default();
        if let Some(device) = self.cuda_device {
            options.device = device;
        }
        if let Some(limit) = self.gpu_memory_limit_bytes {
            options.memory_limit_bytes = limit;
        }
        options.kernel_cache_dir = self.cuda_kernel_cache.clone();
        options.matmul = self.cuda_matmul;
        options.decode_graph = self.cuda_decode_graph;
        options
    }

    fn execution(&self, _info: &local_inference::backend::ModelInfo) -> serde_json::Value {
        #[cfg(feature = "cuda")]
        match self.backend {
            LocalBackend::Cuda => {
                return json!({
                    "backend": "cuda", "options": self.cuda_options(),
                    "expert_residency": "all weights GPU-resident",
                    "prefill": if self.batch > 1 {
                        "layer-wise batched (internal capacity <=512)"
                    } else { "sequential token steps" },
                    "kv_precision": "f32",
                });
            }
            LocalBackend::Reference => {
                return json!({"backend": "cpu-reference", "prefill": "sequential token steps"});
            }
            LocalBackend::Cpu => {}
        }
        json!(self.options())
    }

    fn config(&self) -> Config {
        Config {
            context_tokens: self.context,
            batch_tokens: self.batch,
            prefill_chunk: self.prefill_chunk,
            threads: self.threads,
        }
    }

    fn input(&self, backend: &dyn Backend) -> Result<Vec<i32>> {
        match &self.input_tokens_file {
            Some(path) => Ok(serde_json::from_slice(&fs::read(path)?)?),
            None => backend.tokenize(&text_input(&self.prompt, &self.prompt_file)?),
        }
    }
}

fn execute() -> Result<()> {
    match Cli::parse().command {
        #[cfg(feature = "ktransformers")]
        Command::Remote { args } => remote(args)?,
        Command::Tokenize { model, text } => {
            let tokenizer = Tokenizer::open(&model)?;
            let input = text_input(&text.prompt, &text.prompt_file)?;
            let tokens = tokenizer.tokenize(&input)?;
            println!(
                "{}",
                serde_json::to_string(&json!({
                    "prompt_token_ids": tokens,
                    "decoded_text": tokenizer.detokenize(&tokens)?,
                }))?
            );
        }
        Command::Detokenize {
            model,
            input_tokens_file,
        } => {
            let tokenizer = Tokenizer::open(&model)?;
            let tokens: Vec<i32> = serde_json::from_slice(&fs::read(input_tokens_file)?)?;
            println!(
                "{}",
                serde_json::to_string(&json!({"text": tokenizer.detokenize(&tokens)?}))?
            );
        }
        Command::Generate { model } => {
            let config = model.config();
            let start = Instant::now();
            let backend = model.open()?;
            let load_ns = start.elapsed().as_nanos();
            let input = model.input(backend.as_ref())?;
            let mut executor = Executor::new(backend, config)?;
            // Check text support before any numerical execution, including ID inputs.
            executor.detokenize(&[])?;
            let sample = executor.generate(&input, model.tokens, GenerationMode::UntilEog)?;
            println!(
                "{}",
                serde_json::to_string(&json!({
                    "runner": "rust-independent",
                    "load_ns": load_ns,
                    "prompt_token_ids": input,
                    "text": executor.detokenize(&sample.generated_tokens)?,
                    "sample": sample,
                    "memory": executor.memory_usage(),
                }))?
            );
        }
        Command::Bench {
            model,
            warmups,
            repetitions,
        } => {
            let config = model.config();
            let start = Instant::now();
            let backend = model.open()?;
            let load_ns = start.elapsed().as_nanos();
            let input = model.input(backend.as_ref())?;
            let mut executor = Executor::new(backend, config.clone())?;
            for _ in 0..warmups {
                executor.generate(&input, model.tokens, GenerationMode::FixedLength)?;
            }
            let mut samples = Vec::new();
            for _ in 0..repetitions {
                samples.push(executor.generate(
                    &input,
                    model.tokens,
                    GenerationMode::FixedLength,
                )?);
            }
            println!(
                "{}",
                serde_json::to_string(&json!({
                    "schema_version": 1, "runner": "rust-independent",
                    "load_ns": load_ns, "model": executor.info(), "config": config,
                    "execution": model.execution(executor.info()),
                    "memory": executor.memory_usage(),
                    "expanded_weight_bytes": model.f32_weights.then_some(executor.info().parameters * 4),
                    "prompt_tokens": input.len(), "prompt_token_ids": input, "samples": samples,
                }))?
            );
        }
        Command::Logits { model } => {
            let config = model.config();
            config.validate()?;
            let mut backend = model.open()?;
            let input = model.input(backend.as_ref())?;
            if input.is_empty() || input.len() > config.context_tokens as usize {
                return Err(local_inference::Error::Invalid(
                    "invalid prompt length".into(),
                ));
            }
            for (i, chunk) in input.chunks(config.prefill_chunk as usize).enumerate() {
                backend.evaluate(chunk, (i * config.prefill_chunk as usize) as u32)?;
            }
            println!(
                "{}",
                serde_json::to_string(&json!({
                    "model": backend.info(), "prompt_token_ids": input, "logits": backend.logits()?,
                }))?
            );
        }
    }
    Ok(())
}

fn main() {
    if let Err(error) = execute() {
        eprintln!("local-inference: {error}");
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn model_commands_require_exactly_one_input_source() {
        for command in ["generate", "bench", "logits"] {
            let args = ["local-inference", command, "--model", "unused.gguf"];
            assert!(Cli::try_parse_from(args).is_err());
            for source in ["--input-tokens-file", "--prompt", "--prompt-file"] {
                assert!(
                    Cli::try_parse_from([args.as_slice(), &[source, "input"]].concat()).is_ok()
                );
                for other in ["--input-tokens-file", "--prompt", "--prompt-file"] {
                    if source != other {
                        assert!(
                            Cli::try_parse_from(
                                [args.as_slice(), &[source, "input", other, "input"]].concat()
                            )
                            .is_err()
                        );
                    }
                }
            }
        }
        let args = ["local-inference", "tokenize", "--model", "unused.gguf"];
        assert!(Cli::try_parse_from(args).is_err());
        assert!(Cli::try_parse_from([args.as_slice(), &["--prompt", ""]].concat()).is_ok());
        assert!(
            Cli::try_parse_from(
                [
                    args.as_slice(),
                    &["--prompt", "text", "--prompt-file", "input"]
                ]
                .concat()
            )
            .is_err()
        );
    }
}
