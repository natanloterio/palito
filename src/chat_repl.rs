//! Interactive multi-turn chat on top of `palito::engine::Engine`.
//!
//! Each turn re-renders the whole conversation with the model's chat template
//! and runs a fresh generation (the engine resets state per request). Opening,
//! CUDA profile fallbacks, sampling and stop sequences are the engine's.

use crate::RuntimeArgs;
use clap::Args;
use palito::{
    Result,
    chat::{Message, fit_history},
    engine::{Completion, Engine, FinishReason, GenerationOptions},
    sampling::Sampling,
};
use std::{
    fs,
    io::{BufRead, IsTerminal, Write},
    path::PathBuf,
    time::Instant,
};

/// Chat history keeps this much context free for the reply (Qwen3-style
/// reasoning before the answer); the reply itself may use all remaining context.
const CHAT_REPLY_RESERVE: u32 = 2048;

#[derive(Args)]
pub(crate) struct ChatArgs {
    #[command(flatten)]
    runtime: RuntimeArgs,
    /// System message placed before the conversation.
    #[arg(long, allow_hyphen_values = true)]
    system: Option<String>,
    /// Jinja template replacing the GGUF tokenizer.chat_template.
    #[arg(long)]
    chat_template_file: Option<PathBuf>,
    /// Disable reasoning for templates that support it (Qwen3 `enable_thinking`).
    #[arg(long)]
    no_thinking: bool,
    /// Print per-reply timing to stderr.
    #[arg(long)]
    stats: bool,
    /// Print each reply only once it is complete instead of token by token.
    #[arg(long)]
    no_stream: bool,
    #[command(flatten)]
    sampling: SamplingArgs,
    /// End the reply at this text (not printed); repeat for several.
    #[arg(long, allow_hyphen_values = true)]
    stop: Vec<String>,
}

/// Sampling flags shared by `chat` and `serve`.
#[derive(Args, Clone)]
pub(crate) struct SamplingArgs {
    /// 0 is greedy (deterministic). Qwen3 suggests 0.6 with reasoning, 0.7 without.
    #[arg(long, default_value_t = 0.0)]
    pub(crate) temperature: f32,
    /// Nucleus sampling mass in (0, 1]; 1 disables it.
    #[arg(long, default_value_t = 1.0)]
    pub(crate) top_p: f32,
    /// Sample among the k most likely tokens; 0 disables it.
    #[arg(long, default_value_t = 0)]
    pub(crate) top_k: u32,
    /// Seed for reproducible sampling [default: a new one each reply].
    #[arg(long)]
    pub(crate) seed: Option<u64>,
}

impl SamplingArgs {
    pub(crate) fn sampling(&self) -> Sampling {
        Sampling {
            temperature: self.temperature,
            top_p: self.top_p,
            top_k: self.top_k,
            seed: self.seed,
        }
    }
}

struct Session {
    engine: Engine,
    options: GenerationOptions,
    initial: Vec<Message>,
    history: Vec<Message>,
    context: usize,
    /// Context kept free for the reply when dropping old turns.
    reserve: u32,
}

pub(crate) fn run(mut args: ChatArgs) -> Result<()> {
    let start = Instant::now();
    let sampling = args.sampling.sampling();
    sampling.validate()?;
    palito::stop::validate(&args.stop)?;
    let mut session = Session::open(&mut args)?;
    let interactive = std::io::stdin().is_terminal();
    if interactive {
        let limit = match session.options.max_tokens {
            Some(tokens) => format!("up to {tokens} tokens per reply"),
            None => "replies run to the end of turn or a full context".into(),
        };
        let decoding = if sampling.is_greedy() {
            "Greedy decoding".to_owned()
        } else {
            format!("Sampling at temperature {}", sampling.temperature)
        };
        eprintln!(
            "Model loaded in {:.1}s. Context {} tokens; {limit}. Commands: /reset, /exit. {decoding}.",
            start.elapsed().as_secs_f64(),
            session.context
        );
    }
    let stdin = std::io::stdin();
    let mut lines = stdin.lock().lines();
    loop {
        if interactive {
            eprint!("> ");
            std::io::stderr().flush()?;
        }
        let Some(line) = lines.next() else { break };
        let line = line?;
        match line.trim() {
            "" => continue,
            "/exit" | "/quit" => break,
            "/reset" => {
                session.history = session.initial.clone();
                eprintln!("(conversation cleared)");
            }
            text => match session.reply(text, !args.no_stream) {
                Ok(completion) => {
                    if args.no_stream {
                        println!("{}", completion.text);
                    }
                    if args.stats {
                        print_stats(&completion);
                    }
                }
                Err(error) => eprintln!("palito: {error}"),
            },
        }
    }
    Ok(())
}

impl Session {
    fn open(args: &mut ChatArgs) -> Result<Self> {
        let template = match &args.chat_template_file {
            Some(path) => Some(fs::read_to_string(path)?),
            None => None,
        };
        let runtime = &mut args.runtime;
        let engine = if runtime.is_cuda() {
            open_cuda(runtime, template)?
        } else {
            runtime.use_model_context()?;
            let backend = runtime.open()?;
            let config = runtime.config();
            Engine::from_backend(
                &runtime.model,
                backend,
                config,
                runtime.cpu_profile(),
                template.as_deref(),
            )?
        };
        for notice in engine.notices() {
            eprintln!("({notice})");
        }
        if !engine.has_chat_template() {
            return Err(palito::Error::Invalid(
                "model has no chat template; pass --chat-template-file or use generate".into(),
            ));
        }
        let context = engine.context_tokens();
        let max_tokens = runtime.chat_tokens();
        let initial: Vec<_> = args.system.iter().map(Message::system).collect();
        Ok(Self {
            engine,
            options: GenerationOptions {
                max_tokens,
                enable_thinking: !args.no_thinking,
                sampling: args.sampling.sampling(),
                stop: args.stop.clone(),
                json_schema: None,
            },
            history: initial.clone(),
            initial,
            context: context as usize,
            reserve: max_tokens.unwrap_or(CHAT_REPLY_RESERVE.min((context / 2).max(1))),
        })
    }

    /// Generates a reply; history only changes when generation succeeds.
    /// With `stream`, prints the reply to stdout as it is generated and ends it
    /// with a newline.
    fn reply(&mut self, text: &str, stream: bool) -> Result<Completion> {
        let candidate = [self.history.as_slice(), &[Message::user(text)]].concat();
        let reserved = self.reserve.saturating_sub(1) as usize;
        let fitted = match fit_history(&candidate, |messages| {
            Ok(self.prompt_tokens(messages)? + reserved <= self.context)
        }) {
            Ok(fitted) => fitted,
            // Without --tokens the reserve only guides dropping old turns: a message
            // that fits alone keeps whatever context is left for the reply.
            Err(_) if self.options.max_tokens.is_none() => {
                let latest: Vec<Message> = candidate
                    .iter()
                    .filter(|m| m.role == "system")
                    .chain(candidate.last())
                    .cloned()
                    .collect();
                let tokens = self.prompt_tokens(&latest)?;
                if tokens >= self.context {
                    return Err(palito::Error::Invalid(format!(
                        "message of {tokens} tokens does not fit in the {}-token context",
                        self.context
                    )));
                }
                latest
            }
            Err(error) => return Err(error),
        };
        if fitted.len() < candidate.len() {
            eprintln!("(oldest turns dropped to fit --context {})", self.context);
        }
        let completion = if stream {
            let mut stdout = std::io::stdout().lock();
            let mut failed = None;
            let result = self.engine.chat(&fitted, &self.options, |piece| {
                match stdout.write_all(piece.as_bytes()).and_then(|()| stdout.flush()) {
                    Ok(()) => true,
                    Err(error) => {
                        failed = Some(error);
                        false
                    }
                }
            });
            // End the line even when generation failed part way.
            writeln!(stdout)?;
            stdout.flush()?;
            if let Some(error) = failed {
                return Err(error.into());
            }
            result?
        } else {
            self.engine.chat(&fitted, &self.options, |_| true)?
        };
        match completion.finish_reason {
            FinishReason::Length => {
                eprintln!("(reply stopped at --tokens {})", completion.generated_tokens)
            }
            FinishReason::ContextFull => eprintln!(
                "(reply stopped: the {}-token context is full; /reset clears the conversation)",
                self.context
            ),
            FinishReason::Stop | FinishReason::Cancelled => {}
        }
        self.history = [fitted, vec![Message::assistant(completion.text.clone())]].concat();
        Ok(completion)
    }

    fn prompt_tokens(&self, messages: &[Message]) -> Result<usize> {
        let prompt = self
            .engine
            .render_chat(messages, self.options.thinking())?;
        Ok(self.engine.tokenize(&prompt)?.len())
    }
}

#[cfg(feature = "cuda")]
fn open_cuda(runtime: &RuntimeArgs, template: Option<String>) -> Result<Engine> {
    let options = palito::engine::EngineOptions {
        chat_template: template,
        ..runtime.cuda_engine_options()?
    };
    Engine::open(&runtime.model, &options)
}

#[cfg(not(feature = "cuda"))]
fn open_cuda(_: &RuntimeArgs, _: Option<String>) -> Result<Engine> {
    unreachable!("CUDA backend without the cuda feature")
}

fn print_stats(completion: &Completion) {
    eprintln!(
        "[{} tokens · first token {:.0} ms · {:.1} tok/s]",
        completion.generated_tokens,
        completion.ttft_ns as f64 / 1e6,
        completion.tokens_per_second
    );
}
