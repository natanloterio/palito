use clap::{ArgGroup, Args, Parser, Subcommand, ValueEnum};
use palito::{
    Config, Executor, GenerationMode, Result,
    backend::Backend,
    cpu::{CpuBackend, CpuOptions},
    tokenizer::Tokenizer,
};
use serde_json::json;
use std::{ffi::OsString, fs, io::Write, path::PathBuf, time::Instant};

mod chat_repl;
mod server;

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
    /// Interactive multi-turn chat using the model's GGUF chat template (greedy).
    Chat {
        #[command(flatten)]
        args: chat_repl::ChatArgs,
    },
    /// OpenAI-compatible HTTP server (/v1/chat/completions, /v1/completions, streaming).
    Serve {
        #[command(flatten)]
        args: server::ServeArgs,
    },
    /// Greedy completion, stopping at EOG. No chat template is applied.
    Generate {
        #[command(flatten)]
        model: ModelArgs,
        /// Print the completion as plain text token by token instead of one JSON
        /// object at the end.
        #[arg(long)]
        stream: bool,
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
    #[cfg(feature = "vulkan")]
    Vulkan,
    /// Slow independent packed CPU oracle for the CUDA model layouts.
    #[cfg(feature = "cuda")]
    Reference,
}

#[derive(Args)]
#[command(group(ArgGroup::new("input").required(true).multiple(false)
    .args(["input_tokens_file", "prompt", "prompt_file"])))]
struct ModelArgs {
    #[command(flatten)]
    runtime: RuntimeArgs,
    #[arg(long)]
    input_tokens_file: Option<PathBuf>,
    #[arg(long, allow_hyphen_values = true)]
    prompt: Option<String>,
    #[arg(long)]
    prompt_file: Option<PathBuf>,
}

impl std::ops::Deref for ModelArgs {
    type Target = RuntimeArgs;
    fn deref(&self) -> &RuntimeArgs {
        &self.runtime
    }
}

/// Read only by the CUDA backend, so a global setting never breaks CPU runs.
#[cfg(feature = "cuda")]
const KERNEL_CACHE_ENV: &str = "PALITO_CUDA_KERNEL_CACHE";
const DEFAULT_GENERATED_TOKENS: u32 = 32;
const DEFAULT_CONTEXT_TOKENS: u32 = 512;
/// Largest CUDA context (CudaBackend::open).
#[cfg(feature = "cuda")]
const CUDA_CONTEXT_MAX: u32 = 65535;
const DEFAULT_BATCH_TOKENS: u32 = 128;
const DEFAULT_PREFILL_CHUNK: u32 = 64;

/// Backend, model and execution-plan options shared by every local model command.
#[derive(Args)]
pub(crate) struct RuntimeArgs {
    #[arg(long, value_enum, env = "PALITO_BACKEND", default_value_t = LocalBackend::Cpu)]
    backend: LocalBackend,
    #[arg(long, env = "PALITO_MODEL")]
    pub(crate) model: PathBuf,
    /// Context tokens [default: 512; chat: the model's trained context (GGUF
    /// context_length; CUDA: at most 65535, shrunk to the largest that fits in free
    /// device memory)].
    #[arg(long)]
    context: Option<u32>,
    /// Prefill batch capacity [default: 128; chat on CUDA: 4096, capped by
    /// --context].
    #[arg(long)]
    batch: Option<u32>,
    /// Tokens per prefill call [default: 64; chat on CUDA: 4096, capped by
    /// --context and --batch].
    #[arg(long)]
    prefill_chunk: Option<u32>,
    #[arg(long, default_value_t = 4)]
    threads: u32,
    /// Maximum generated tokens [default: 32; chat: until end of turn or a full
    /// context].
    #[arg(long)]
    tokens: Option<u32>,
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
    /// Opt-in private directory for persistent CUDA kernel compilation cache
    /// [fallback when omitted with --backend cuda: $PALITO_CUDA_KERNEL_CACHE].
    #[cfg(feature = "cuda")]
    #[arg(long)]
    cuda_kernel_cache: Option<PathBuf>,
    /// CUDA matrix profile [default: f32; chat: int8-fast, f32-tiled below sm_80]. f32 keeps its existing
    /// kernels and arithmetic. f32-tiled selects measured F32 prefill/decode kernels
    /// without activation quantization. int8-prefill keeps f32-tiled decode and runs
    /// prefill on tensor cores (INT8 projections, FP16 hi/lo attention): faster long
    /// prompts, approximate logits (same top token on the quality gates).
    /// int8-fast is int8-prefill at llama.cpp precision (one 8-bit plane per 32
    /// activations, single FP16 attention, TF32 router): fastest prefill, logits about
    /// 1 off F32 and generated text can differ. residual-int8-research is opt-in approximate prefill, not quality-admitted.
    #[cfg(feature = "cuda")]
    #[arg(long, value_enum)]
    cuda_matmul: Option<palito::cuda::CudaMatmul>,
    /// Captured decode with token-only greedy output (f32: context <= 1024). On by
    /// default for chat with f32-tiled, int8-prefill and int8-fast.
    #[cfg(feature = "cuda")]
    #[arg(long)]
    cuda_decode_graph: bool,
    /// Disable the decode graph chat enables by default.
    #[cfg(feature = "cuda")]
    #[arg(long, conflicts_with = "cuda_decode_graph")]
    no_cuda_decode_graph: bool,
    /// Re-evaluate the whole conversation every turn instead of reusing the
    /// previous prompt's KV cache (chat with --backend cuda).
    #[cfg(feature = "cuda")]
    #[arg(long)]
    no_prefix_cache: bool,
    /// --context defaulted to the model's trained context (chat): CUDA may shrink it
    /// to fit device memory.
    #[arg(skip)]
    model_context: bool,
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
    use palito::{
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
        _ => Err(palito::Error::Invalid(
            "supply exactly one of --prompt and --prompt-file".into(),
        )),
    }
}

impl RuntimeArgs {
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

    pub(crate) fn open(&self) -> Result<Box<dyn Backend>> {
        #[cfg(feature = "cuda")]
        {
            if self.backend != LocalBackend::Cuda
                && (self.cuda_device.is_some()
                    || self.gpu_memory_limit_bytes.is_some()
                    || self.cuda_kernel_cache.is_some()
                    || self
                        .cuda_matmul
                        .is_some_and(|m| m != palito::cuda::CudaMatmul::F32)
                    || self.cuda_decode_graph
                    || self.no_cuda_decode_graph
                    || self.no_prefix_cache)
            {
                return Err(palito::Error::Invalid(
                    "CUDA options require --backend cuda".into(),
                ));
            }
            self.check_cpu_policy()?;
            match self.backend {
                LocalBackend::Cuda => {
                    return palito::cuda::open_backend(
                        &self.model,
                        &self.config(),
                        self.cuda_options(),
                    );
                }
                LocalBackend::Reference => {
                    return Ok(Box::new(
                        palito::cuda::reference::ReferenceBackend::open(
                            &self.model,
                            &self.config(),
                        )?,
                    ));
                }
                #[cfg(feature = "vulkan")]
                LocalBackend::Vulkan => {}
                LocalBackend::Cpu => {}
            }
        }
        #[cfg(feature = "vulkan")]
        if self.backend == LocalBackend::Vulkan {
            self.check_cpu_policy()?;
            let options = palito::vulkan::VulkanOptions::default();
            return palito::vulkan::open_backend(&self.model, &self.config(), options);
        }
        Ok(Box::new(CpuBackend::open_with_options(
            &self.model,
            &self.config(),
            self.options(),
        )?))
    }

    #[cfg(feature = "cuda")]
    fn cuda_options(&self) -> palito::cuda::CudaOptions {
        let mut options = palito::cuda::CudaOptions::default();
        if let Some(device) = self.cuda_device {
            options.device = device;
        }
        if let Some(limit) = self.gpu_memory_limit_bytes {
            options.memory_limit_bytes = limit;
        }
        options.kernel_cache_dir = self.cuda_kernel_cache.clone().or_else(|| {
            std::env::var_os(KERNEL_CACHE_ENV)
                .filter(|dir| !dir.is_empty())
                .map(PathBuf::from)
        });
        options.matmul = self.cuda_matmul();
        options.decode_graph = self.cuda_decode_graph;
        options.fit_context = self.model_context;
        options
    }

    #[cfg(feature = "cuda")]
    fn cuda_matmul(&self) -> palito::cuda::CudaMatmul {
        self.cuda_matmul.unwrap_or_default()
    }

    #[cfg(any(feature = "cuda", feature = "vulkan"))]
    fn check_cpu_policy(&self) -> Result<()> {
        if self.backend != LocalBackend::Cpu
            && (self.f32_weights
                || self.sequential_prefill
                || self.memory_limit_bytes.is_some()
                || self.prefix_cache_bytes != 0
                || self.speculative_window != 0
                || self.scalar_kernels)
        {
            return Err(palito::Error::Invalid(
                "CPU policy options require --backend cpu".into(),
            ));
        }
        Ok(())
    }

    /// Engine options for `--backend cuda` (chat and serve defaults: model
    /// context, fastest profile that opens, decode graph, 4096-token prefill).
    #[cfg(feature = "cuda")]
    pub(crate) fn cuda_engine_options(&self) -> Result<palito::engine::EngineOptions> {
        self.check_cpu_policy()?;
        Ok(palito::engine::EngineOptions {
            device: palito::engine::Device::Cuda,
            context_tokens: self.context,
            threads: self.threads,
            batch_tokens: self.batch,
            prefill_chunk: self.prefill_chunk,
            gpu_device: self.cuda_device.unwrap_or(0),
            gpu_memory_limit_bytes: self.gpu_memory_limit_bytes,
            kernel_cache_dir: self.cuda_kernel_cache.clone(),
            cuda_matmul: self.cuda_matmul,
            decode_graph: if self.cuda_decode_graph {
                Some(true)
            } else if self.no_cuda_decode_graph {
                Some(false)
            } else {
                None
            },
            chat_template: None,
            prefix_cache: !self.no_prefix_cache,
        })
    }

    /// True when the backend is CUDA (chat then opens through `Engine`).
    pub(crate) fn is_cuda(&self) -> bool {
        #[cfg(feature = "cuda")]
        {
            self.backend == LocalBackend::Cuda
        }
        #[cfg(not(feature = "cuda"))]
        false
    }

    /// "cpu" or "cpu-reference".
    pub(crate) fn cpu_profile(&self) -> &'static str {
        #[cfg(feature = "cuda")]
        if self.backend == LocalBackend::Reference {
            return "cpu-reference";
        }
        "cpu"
    }

    pub(crate) fn tokens(&self) -> u32 {
        self.tokens.unwrap_or(DEFAULT_GENERATED_TOKENS)
    }

    pub(crate) fn context(&self) -> u32 {
        self.context.unwrap_or(DEFAULT_CONTEXT_TOKENS)
    }

    /// Explicit --tokens for chat replies; None generates until end of turn or a full
    /// context.
    pub(crate) fn chat_tokens(&self) -> Option<u32> {
        self.tokens
    }

    /// Without --context, use the model's trained context (CUDA: at most
    /// CUDA_CONTEXT_MAX, fitted to device memory when opening).
    pub(crate) fn use_model_context(&mut self) -> Result<()> {
        if self.context.is_none() {
            let trained = palito::model_context_tokens(&self.model)?;
            #[cfg(feature = "cuda")]
            let cap = if self.backend == LocalBackend::Cuda {
                CUDA_CONTEXT_MAX
            } else {
                u32::MAX
            };
            #[cfg(not(feature = "cuda"))]
            let cap = u32::MAX;
            self.context = Some(trained.min(cap));
            self.model_context = true;
        }
        Ok(())
    }

    fn batch(&self) -> u32 {
        self.batch.unwrap_or(DEFAULT_BATCH_TOKENS)
    }

    fn prefill_chunk(&self) -> u32 {
        self.prefill_chunk.unwrap_or(DEFAULT_PREFILL_CHUNK)
    }

    fn execution(&self, _info: &palito::backend::ModelInfo) -> serde_json::Value {
        #[cfg(feature = "cuda")]
        match self.backend {
            LocalBackend::Cuda => {
                return json!({
                    "backend": "cuda", "options": self.cuda_options(),
                    "expert_residency": "all weights GPU-resident",
                    "prefill": if self.batch() > 1 {
                        "layer-wise batched (internal capacity <=4096)"
                    } else { "sequential token steps" },
                    "kv_precision": "f32",
                });
            }
            LocalBackend::Reference => {
                return json!({"backend": "cpu-reference", "prefill": "sequential token steps"});
            }
            #[cfg(feature = "vulkan")]
            LocalBackend::Vulkan => {}
            LocalBackend::Cpu => {}
        }
        #[cfg(feature = "vulkan")]
        if self.backend == LocalBackend::Vulkan {
            return json!({
                "backend": "vulkan",
                "options": palito::vulkan::VulkanOptions::default(),
                "expert_residency": "all weights GPU-resident",
                "prefill": "sequential token steps",
                "kv_precision": "f32",
            });
        }
        json!(self.options())
    }

    pub(crate) fn config(&self) -> Config {
        Config {
            context_tokens: self.context(),
            batch_tokens: self.batch(),
            prefill_chunk: self.prefill_chunk(),
            threads: self.threads,
        }
    }
}

impl ModelArgs {
    fn input(&self, backend: &dyn Backend) -> Result<Vec<i32>> {
        match &self.input_tokens_file {
            Some(path) => Ok(serde_json::from_slice(&fs::read(path)?)?),
            None => backend.tokenize(&text_input(&self.prompt, &self.prompt_file)?),
        }
    }
}

/// Rewrites llama-server's `-np` for `serve`: clap has no two-letter short flags.
fn normalize_args(args: impl Iterator<Item = OsString>) -> Vec<OsString> {
    let args: Vec<OsString> = args.collect();
    if args.get(1).is_none_or(|command| command != "serve") {
        return args;
    }
    args.into_iter()
        .map(|arg| match arg.to_str() {
            Some("-np") => "--parallel".into(),
            Some(text) => match text.strip_prefix("-np=") {
                Some(value) => format!("--parallel={value}").into(),
                None => arg,
            },
            None => arg,
        })
        .collect()
}

fn execute() -> Result<()> {
    match Cli::parse_from(normalize_args(std::env::args_os())).command {
        #[cfg(feature = "ktransformers")]
        Command::Remote { args } => remote(args)?,
        Command::Chat { args } => chat_repl::run(args)?,
        Command::Serve { args } => server::run(args)?,
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
        Command::Generate { model, stream: true } => {
            let config = model.config();
            let backend = model.open()?;
            let input = model.input(backend.as_ref())?;
            let tokenizer = Tokenizer::open(&model.model)?;
            let detokenize = |tokens: &[i32]| tokenizer.detokenize(tokens);
            let mut executor = Executor::new(backend, config)?;
            executor.detokenize(&[])?;
            let mut text = palito::stream::TextStream::new();
            let mut stdout = std::io::stdout().lock();
            let result = executor.generate_streaming(
                &input,
                model.tokens(),
                GenerationMode::UntilEog,
                |token| {
                    stdout.write_all(text.push(token, detokenize)?.as_bytes())?;
                    Ok(stdout.flush()?)
                },
            );
            writeln!(stdout, "{}", text.finish(detokenize)?)?;
            result?;
        }
        Command::Generate {
            model,
            stream: false,
        } => {
            let config = model.config();
            let start = Instant::now();
            let backend = model.open()?;
            let load_ns = start.elapsed().as_nanos();
            let input = model.input(backend.as_ref())?;
            let mut executor = Executor::new(backend, config)?;
            // Check text support before any numerical execution, including ID inputs.
            executor.detokenize(&[])?;
            let sample = executor.generate(&input, model.tokens(), GenerationMode::UntilEog)?;
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
                executor.generate(&input, model.tokens(), GenerationMode::FixedLength)?;
            }
            let mut samples = Vec::new();
            for _ in 0..repetitions {
                samples.push(executor.generate(
                    &input,
                    model.tokens(),
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
                return Err(palito::Error::Invalid(
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
        eprintln!("palito: {error}");
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn np_is_an_alias_of_parallel() {
        let parsed = |a: &[&str]| normalize_args(a.iter().map(OsString::from));
        assert_eq!(
            parsed(&["palito", "serve", "-np", "4"]),
            ["palito", "serve", "--parallel", "4"]
        );
        assert_eq!(
            parsed(&["palito", "serve", "-np=4"]),
            ["palito", "serve", "--parallel=4"]
        );
        assert_eq!(
            parsed(&["palito", "chat", "-np", "4"]),
            ["palito", "chat", "-np", "4"]
        );
    }

    #[test]
    fn parallel_is_limited_to_64() {
        let base = ["palito", "serve", "--model", "m.gguf", "--parallel"];
        assert!(Cli::try_parse_from([&base[..], &["64"]].concat()).is_ok());
        assert!(Cli::try_parse_from([&base[..], &["0"]].concat()).is_err());
        assert!(Cli::try_parse_from([&base[..], &["65"]].concat()).is_err());
    }

    #[test]
    fn model_commands_require_exactly_one_input_source() {
        for command in ["generate", "bench", "logits"] {
            let args = ["palito", command, "--model", "unused.gguf"];
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
        let args = ["palito", "tokenize", "--model", "unused.gguf"];
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
