use clap::{ArgGroup, Args, CommandFactory, FromArgMatches, Parser, Subcommand, ValueHint};
use palito::{
    Executor, GenerationMode, Result,
    backend::Backend,
    tokenizer::Tokenizer,
};
use serde_json::json;
use std::{ffi::OsString, io::Write, path::PathBuf, time::Instant};

mod banner;
mod chat_repl;
mod cli_ui;
mod convert;
mod help_text;
mod runtime_args;
mod server;

use help_text::*;
pub(crate) use runtime_args::RuntimeArgs;

#[derive(Parser)]
#[command(
    version,
    about = "Fast local LLM inference in Rust (.palito GPU-native format & GGUF) with CUDA, Vulkan, and Metal",
    after_help = ROOT_EXAMPLES
)]
struct Cli {
    /// When to color output: auto, always or never (NO_COLOR also disables it).
    #[arg(long, global = true, value_name = "WHEN", default_value = "auto", hide_short_help = true)]
    color: clap::ColorChoice,
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
    /// Interactive multi-turn chat using the model's chat template (supports .palito & GGUF).
    #[command(after_help = CHAT_EXAMPLES)]
    Chat {
        #[command(flatten)]
        args: chat_repl::ChatArgs,
    },
    /// OpenAI-compatible HTTP server (/v1/chat/completions, /v1/completions, streaming).
    #[command(after_help = SERVE_EXAMPLES)]
    Serve {
        #[command(flatten)]
        args: server::ServeArgs,
    },
    /// Convert a GGUF model into a single self-contained `.palito` (`PLT1`) file.
    #[command(after_help = CONVERT_EXAMPLES)]
    Convert {
        #[command(flatten)]
        args: convert::ConvertArgs,
    },
    /// Greedy completion, stopping at EOG. No chat template is applied.
    #[command(after_help = GENERATE_EXAMPLES)]
    #[command(override_usage = "palito generate --model <FILE> --prompt <TEXT> [OPTIONS]\n       palito generate --model <FILE> --prompt-file <FILE> [OPTIONS]\n       palito generate --model <FILE> --input-tokens-file <FILE> [OPTIONS]")]
    Generate {
        #[command(flatten)]
        model: ModelArgs,
        /// Print the completion as plain text token by token.
        #[arg(long, conflicts_with = "json", help_heading = "Output")]
        stream: bool,
        /// Print one JSON object with the completion, timings and memory at the end
        /// (the current default; pass it explicitly in scripts).
        #[arg(long, help_heading = "Output")]
        json: bool,
    },
    /// Tokenize text offline without loading model weights.
    #[command(after_help = TOKENIZE_EXAMPLES)]
    Tokenize {
        /// Model whose tokenizer to use (.gguf or .palito).
        #[arg(long, env = "PALITO_MODEL", value_name = "FILE", value_hint = ValueHint::FilePath)]
        model: PathBuf,
        #[command(flatten)]
        text: TextArgs,
    },
    /// Decode IDs offline, omitting control tokens.
    #[command(after_help = DETOKENIZE_EXAMPLES)]
    #[command(group(ArgGroup::new("ids").required(true).multiple(false)
        .args(["input_tokens_file", "tokens"])))]
    Detokenize {
        /// Model whose tokenizer to use (.gguf or .palito).
        #[arg(long, env = "PALITO_MODEL", value_name = "FILE", value_hint = ValueHint::FilePath)]
        model: PathBuf,
        /// JSON array of token IDs; `-` reads standard input.
        #[arg(long, value_name = "FILE", value_hint = ValueHint::FilePath)]
        input_tokens_file: Option<PathBuf>,
        /// Comma-separated token IDs, e.g. 9906,1917.
        #[arg(long, value_delimiter = ',', value_name = "IDS")]
        tokens: Vec<i32>,
    },
    /// Fixed-length generation with per-phase timing and full state reset per sample.
    #[command(after_help = BENCH_EXAMPLES)]
    #[command(override_usage = "palito bench --model <FILE> --prompt <TEXT> [OPTIONS]\n       palito bench --model <FILE> --prompt-file <FILE> [OPTIONS]\n       palito bench --model <FILE> --input-tokens-file <FILE> [OPTIONS]")]
    Bench {
        #[command(flatten)]
        model: ModelArgs,
        /// Untimed runs before measuring.
        #[arg(long, default_value_t = 1, help_heading = "Benchmark")]
        warmups: u32,
        /// Timed runs reported as samples.
        #[arg(long, default_value_t = 3, value_parser = clap::value_parser!(u32).range(1..), help_heading = "Benchmark")]
        repetitions: u32,
    },
    /// Evaluate input IDs and emit F32 next-token logits for differential validation.
    #[command(after_help = LOGITS_EXAMPLES)]
    #[command(override_usage = "palito logits --model <FILE> --prompt <TEXT> [OPTIONS]\n       palito logits --model <FILE> --prompt-file <FILE> [OPTIONS]\n       palito logits --model <FILE> --input-tokens-file <FILE> [OPTIONS]")]
    #[command(mut_arg("tokens", |arg| arg.hide(true)))]
    #[cfg_attr(feature = "cuda", command(mut_arg("no_prefix_cache", |arg| arg.hide(true))))]
    Logits {
        #[command(flatten)]
        model: ModelArgs,
    },
    /// Print a shell completion script (bash, zsh, fish, powershell, elvish).
    #[command(after_help = "Examples:\n  palito completions bash > ~/.local/share/bash-completion/completions/palito\n  palito completions zsh > \"${fpath[1]}/_palito\"\n  palito completions fish > ~/.config/fish/completions/palito.fish")]
    Completions {
        shell: clap_complete::Shell,
    },
}

#[derive(Args)]
#[command(group(ArgGroup::new("input").required(true).multiple(false)
    .args(["input_tokens_file", "prompt", "prompt_file"])))]
struct ModelArgs {
    #[command(flatten)]
    runtime: RuntimeArgs,
    /// JSON array of prompt token IDs; `-` reads standard input.
    #[arg(long, value_name = "FILE", value_hint = ValueHint::FilePath, help_heading = "Input")]
    input_tokens_file: Option<PathBuf>,
    /// Prompt text.
    #[arg(long, allow_hyphen_values = true, value_name = "TEXT", help_heading = "Input")]
    prompt: Option<String>,
    /// Read the prompt from a file; `-` reads standard input.
    #[arg(long, value_name = "FILE", value_hint = ValueHint::FilePath, help_heading = "Input")]
    prompt_file: Option<PathBuf>,
}

impl std::ops::Deref for ModelArgs {
    type Target = RuntimeArgs;
    fn deref(&self) -> &RuntimeArgs {
        &self.runtime
    }
}

#[derive(Args)]
#[group(required = true, multiple = false)]
struct TextArgs {
    /// Text to tokenize.
    #[arg(long, allow_hyphen_values = true, value_name = "TEXT")]
    prompt: Option<String>,
    /// Read the text from a file; `-` reads standard input.
    #[arg(long, value_name = "FILE", value_hint = ValueHint::FilePath)]
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
        Some(path) => GenerationInput::TokenIds(cli_ui::read_token_ids(&path)?),
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
        (None, Some(path)) => cli_ui::read_text(path),
        _ => Err(palito::Error::Invalid(
            "supply exactly one of --prompt and --prompt-file".into(),
        )),
    }
}

impl ModelArgs {
    fn input(&self, backend: &dyn Backend) -> Result<Vec<i32>> {
        match &self.input_tokens_file {
            Some(path) => cli_ui::read_token_ids(path),
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

/// `--version` lists the backends and features compiled in, since they change
/// which options exist.
fn long_version() -> String {
    let backends: Vec<&str> = [
        ("cpu", true),
        ("cuda", cfg!(feature = "cuda")),
        ("vulkan", cfg!(feature = "vulkan")),
        ("metal", cfg!(all(feature = "metal", target_os = "macos"))),
    ]
    .into_iter()
    .filter_map(|(name, built)| built.then_some(name))
    .collect();
    let features: Vec<&str> = [
        ("qtip-research", cfg!(feature = "qtip-research")),
        ("ktransformers", cfg!(feature = "ktransformers")),
    ]
    .into_iter()
    .filter_map(|(name, built)| built.then_some(name))
    .collect();
    let features = if features.is_empty() {
        String::new()
    } else {
        format!("; features: {}", features.join(", "))
    };
    format!("{} (backends: {}{features})", env!("CARGO_PKG_VERSION"), backends.join(", "))
}

fn parse_cli() -> Cli {
    let args = normalize_args(std::env::args_os());
    let color = cli_ui::color_choice(&args);
    cli_ui::init_color(color);
    let matches = Cli::command()
        .color(color)
        .long_version(&*Box::leak(long_version().into_boxed_str()))
        .get_matches_from(args);
    Cli::from_arg_matches(&matches).unwrap_or_else(|error| error.exit())
}

fn execute() -> Result<()> {
    match parse_cli().command {
        Command::Completions { shell } => {
            clap_complete::generate(shell, &mut Cli::command(), "palito", &mut std::io::stdout());
        }
        #[cfg(feature = "ktransformers")]
        Command::Remote { args } => remote(args)?,
        Command::Chat { args } => chat_repl::run(args)?,
        Command::Serve { args } => server::run(args)?,
        Command::Convert { args } => convert::run(args)?,
        Command::Tokenize { model, text } => {
            cli_ui::check_model(&model)?;
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
            tokens,
        } => {
            cli_ui::check_model(&model)?;
            let tokenizer = Tokenizer::open(&model)?;
            let tokens = match input_tokens_file {
                Some(path) => cli_ui::read_token_ids(&path)?,
                None => tokens,
            };
            println!(
                "{}",
                serde_json::to_string(&json!({"text": tokenizer.detokenize(&tokens)?}))?
            );
        }
        Command::Generate {
            model,
            stream: true,
            ..
        } => {
            model.check_model()?;
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
            ..
        } => {
            model.check_model()?;
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
            model.check_model()?;
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
            model.check_model()?;
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
        cli_ui::report(&error);
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
    fn json_and_stream_are_exclusive() {
        let base = ["palito", "generate", "--model", "m.gguf", "--prompt", "x"];
        assert!(Cli::try_parse_from([&base[..], &["--json"]].concat()).is_ok());
        assert!(Cli::try_parse_from([&base[..], &["--json", "--stream"]].concat()).is_err());
    }

    #[test]
    fn sizes_accept_units_and_short_aliases() {
        let base = ["palito", "bench", "--model", "m.gguf", "--prompt", "x"];
        let parsed = Cli::try_parse_from([&base[..], &["--memory-limit", "2GiB"]].concat()).unwrap();
        let Command::Bench { model, .. } = parsed.command else { unreachable!() };
        assert_eq!(model.memory_limit_bytes, Some(2 << 30));
        assert!(Cli::try_parse_from([&base[..], &["--prefix-cache-bytes", "lots"]].concat()).is_err());
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_device_is_an_alias_of_gpu_device() {
        let base = ["palito", "chat", "--model", "m.gguf", "--backend", "cuda"];
        for flag in ["--gpu-device", "--cuda-device"] {
            assert!(Cli::try_parse_from([&base[..], &[flag, "1"]].concat()).is_ok());
        }
    }

    #[test]
    fn detokenize_takes_a_file_or_inline_ids() {
        let base = ["palito", "detokenize", "--model", "m.gguf"];
        assert!(Cli::try_parse_from(base).is_err());
        assert!(Cli::try_parse_from([&base[..], &["--tokens", "1,2,3"]].concat()).is_ok());
        assert!(Cli::try_parse_from([&base[..], &["--input-tokens-file", "-"]].concat()).is_ok());
        assert!(
            Cli::try_parse_from([&base[..], &["--tokens", "1", "--input-tokens-file", "-"]].concat())
                .is_err()
        );
    }

    #[test]
    fn negative_temperature_reaches_validation() {
        let args = ["palito", "chat", "--model", "m.gguf", "--temperature", "-1"];
        assert!(Cli::try_parse_from(args).is_ok());
    }

    #[test]
    fn cli_definition_is_consistent() {
        Cli::command().debug_assert();
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
