//! Interactive multi-turn chat on top of `palito::engine::Engine`.
//!
//! Each turn re-renders the whole conversation with the model's chat template
//! and runs a fresh generation (the engine resets state per request). Opening,
//! CUDA profile fallbacks, sampling and stop sequences are the engine's.

use crate::{RuntimeArgs, cli_ui};
use clap::{Args, ValueHint};
use palito::{
    Result,
    chat::{Message, fit_history},
    engine::{Completion, Engine, FinishReason, GenerationOptions},
    sampling::{DEFAULT_REPEAT_LAST_N, Sampling},
};
use std::{
    io::{BufRead, IsTerminal, Write},
    path::PathBuf,
    sync::atomic::{AtomicBool, Ordering},
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
    #[arg(long, allow_hyphen_values = true, value_name = "TEXT")]
    system: Option<String>,
    /// Jinja template replacing the GGUF tokenizer.chat_template.
    #[arg(long, value_name = "FILE", value_hint = ValueHint::FilePath)]
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
    #[arg(long, allow_hyphen_values = true, value_name = "TEXT")]
    stop: Vec<String>,
}

/// Sampling flags shared by `chat` and `serve`.
#[derive(Args, Clone)]
#[command(next_help_heading = "Sampling")]
pub(crate) struct SamplingArgs {
    /// 0 is greedy (deterministic). Qwen3 suggests 0.6 with reasoning, 0.7 without.
    #[arg(long, default_value_t = 0.0, allow_hyphen_values = true)]
    pub(crate) temperature: f32,
    /// Nucleus sampling mass in (0, 1]; 1 disables it.
    #[arg(long, default_value_t = 1.0)]
    pub(crate) top_p: f32,
    /// Sample among the k most likely tokens; 0 disables it.
    #[arg(long, default_value_t = 0)]
    pub(crate) top_k: u32,
    /// Drop tokens less likely than this fraction of the most likely one; 0 disables it.
    #[arg(long, default_value_t = 0.0)]
    pub(crate) min_p: f32,
    /// Penalize repeating any of the last --repeat-last-n tokens (prompt included); 1 disables it.
    #[arg(long, default_value_t = 1.0)]
    pub(crate) repeat_penalty: f32,
    /// Tokens the penalties look back at; 0 disables the penalties.
    #[arg(long, default_value_t = DEFAULT_REPEAT_LAST_N)]
    pub(crate) repeat_last_n: u32,
    /// Subtracted from the logit of every token in the window; 0 disables it.
    #[arg(long, default_value_t = 0.0, allow_hyphen_values = true)]
    pub(crate) presence_penalty: f32,
    /// Subtracted from a token's logit per occurrence in the window; 0 disables it.
    #[arg(long, default_value_t = 0.0, allow_hyphen_values = true)]
    pub(crate) frequency_penalty: f32,
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
            min_p: self.min_p,
            repeat_penalty: self.repeat_penalty,
            repeat_last_n: self.repeat_last_n,
            presence_penalty: self.presence_penalty,
            frequency_penalty: self.frequency_penalty,
            logit_bias: Vec::new(),
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

/// Chat commands typed at the prompt.
const COMMANDS: &[(&str, &str)] = &[
    ("/help", "show these commands"),
    ("/reset", "clear the conversation"),
    ("/exit", "quit (also /quit or Ctrl-D)"),
];

/// Set by Ctrl-C while a reply is generating; the stream callback then stops.
static CANCEL: AtomicBool = AtomicBool::new(false);
static GENERATING: AtomicBool = AtomicBool::new(false);

/// One line typed at the prompt.
#[derive(Debug, PartialEq, Eq)]
enum Input<'a> {
    Empty,
    Exit,
    Reset,
    Help,
    /// `/word` that is not a command.
    Unknown(&'a str),
    /// Text for the model; `//text` sends `/text`.
    Text(&'a str),
}

fn classify(line: &str) -> Input<'_> {
    let line = line.trim();
    if line.is_empty() {
        return Input::Empty;
    }
    if let Some(literal) = line.strip_prefix("//") {
        return Input::Text(&line[line.len() - literal.len() - 1..]);
    }
    let is_command = line.strip_prefix('/').is_some_and(|word| {
        !word.is_empty() && word.chars().all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
    });
    if !is_command {
        return Input::Text(line);
    }
    match line {
        "/exit" | "/quit" => Input::Exit,
        "/reset" => Input::Reset,
        "/help" | "/?" => Input::Help,
        other => Input::Unknown(other),
    }
}

fn suggest(command: &str) -> Option<&'static str> {
    COMMANDS
        .iter()
        .map(|(name, _)| (cli_ui::edit_distance(command, name), *name))
        .filter(|(distance, _)| *distance <= 2)
        .min()
        .map(|(_, name)| name)
}

fn print_help() {
    for (name, what) in COMMANDS {
        eprintln!("  {name:<8} {what}");
    }
    eprintln!("  //text   send text that starts with '/'");
    eprintln!("  Ctrl-C   stop the current reply (at the prompt: quit)");
}

/// Ctrl-C stops the reply being generated; at the prompt it quits.
fn install_interrupt_handler() {
    let installed = ctrlc::set_handler(|| {
        if GENERATING.load(Ordering::SeqCst) {
            CANCEL.store(true, Ordering::SeqCst);
        } else {
            eprintln!();
            std::process::exit(130);
        }
    });
    if let Err(error) = installed {
        cli_ui::warn(&format!("Ctrl-C will quit instead of stopping replies ({error})"));
    }
}

pub(crate) fn run(mut args: ChatArgs) -> Result<()> {
    args.runtime.check_model()?;
    let start = Instant::now();
    let sampling = args.sampling.sampling();
    sampling.validate()?;
    palito::stop::validate(&args.stop)?;
    let interactive = std::io::stdin().is_terminal();
    if std::io::stderr().is_terminal() {
        let size = std::fs::metadata(&args.runtime.model).map_or(0, |m| m.len());
        eprintln!(
            "Loading {} ({}) on {}...",
            args.runtime.model.display(),
            cli_ui::human_size(size),
            args.runtime.backend_name()
        );
    }
    let mut session = Session::open(&mut args)?;
    if interactive {
        install_interrupt_handler();
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
            "Model loaded in {:.1}s. Context {} tokens; {limit}. {decoding}. Type /help for commands.",
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
        match classify(&line) {
            Input::Empty => continue,
            Input::Exit => break,
            Input::Reset => {
                session.history = session.initial.clone();
                eprintln!("(conversation cleared)");
            }
            Input::Help => print_help(),
            Input::Unknown(command) => match suggest(command) {
                Some(name) => eprintln!("unknown command '{command}' (did you mean {name}?); type /help"),
                None => eprintln!("unknown command '{command}'; type /help, or //text to send it to the model"),
            },
            Input::Text(text) => match session.reply(text, !args.no_stream) {
                Ok(None) => {}
                Ok(Some(completion)) => {
                    if args.no_stream {
                        println!("{}", completion.text);
                    }
                    if args.stats {
                        print_stats(&completion);
                    }
                }
                // Scripts piping into chat must see the failure in the exit code.
                Err(error) if !interactive => return Err(error),
                Err(error) => cli_ui::report(&error),
            },
        }
    }
    Ok(())
}

impl Session {
    fn open(args: &mut ChatArgs) -> Result<Self> {
        let template = match &args.chat_template_file {
            Some(path) => Some(cli_ui::read_text(path)?),
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
    /// with a newline. None when Ctrl-C stopped the reply.
    fn reply(&mut self, text: &str, stream: bool) -> Result<Option<Completion>> {
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
        CANCEL.store(false, Ordering::SeqCst);
        GENERATING.store(true, Ordering::SeqCst);
        let completion = self.generate(&fitted, stream);
        GENERATING.store(false, Ordering::SeqCst);
        let completion = match completion {
            Err(palito::Error::Cancelled) => None,
            other => Some(other?),
        };
        let Some(completion) = completion.filter(|c| c.finish_reason != FinishReason::Cancelled) else {
            eprintln!("(reply stopped; the conversation is unchanged. Ctrl-C at the prompt or /exit quits)");
            return Ok(None);
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
        Ok(Some(completion))
    }

    /// Runs one generation, stopping early when Ctrl-C sets `CANCEL`.
    fn generate(&mut self, fitted: &[Message], stream: bool) -> Result<Completion> {
        let cancelled = || CANCEL.load(Ordering::SeqCst);
        if stream {
            let mut stdout = std::io::stdout().lock();
            let mut failed = None;
            let result = self.engine.chat(fitted, &self.options, |piece| {
                match stdout.write_all(piece.as_bytes()).and_then(|()| stdout.flush()) {
                    Ok(()) => !cancelled(),
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
            result
        } else {
            self.engine.chat(fitted, &self.options, |_| !cancelled())
        }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slash_words_are_commands_and_other_lines_are_text() {
        assert_eq!(classify("  "), Input::Empty);
        assert_eq!(classify("/exit"), Input::Exit);
        assert_eq!(classify("/quit"), Input::Exit);
        assert_eq!(classify("/reset"), Input::Reset);
        assert_eq!(classify("/help"), Input::Help);
        assert_eq!(classify("/exti"), Input::Unknown("/exti"));
        assert_eq!(classify("hello"), Input::Text("hello"));
        assert_eq!(classify("/usr/bin is a path"), Input::Text("/usr/bin is a path"));
        assert_eq!(classify("//help"), Input::Text("/help"));
    }

    #[test]
    fn close_typos_get_a_suggestion() {
        assert_eq!(suggest("/exti"), Some("/exit"));
        assert_eq!(suggest("/rest"), Some("/reset"));
        assert_eq!(suggest("/hlep"), Some("/help"));
        assert_eq!(suggest("/translate"), None);
    }
}
