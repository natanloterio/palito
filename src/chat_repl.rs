//! Interactive multi-turn chat on top of the greedy `Executor`.
//!
//! Each turn re-renders the whole conversation with the model's chat template
//! and runs a fresh greedy generation (the executor resets state per request).

use crate::RuntimeArgs;
use clap::Args;
use local_inference::{
    Executor, GenerationMode, Result, Sample,
    backend::Backend,
    chat::{ChatTemplate, Message, fit_history},
};
use std::{
    fs,
    io::{BufRead, IsTerminal, Write},
    path::PathBuf,
    time::Instant,
};

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
}

struct Session {
    executor: Executor<Box<dyn Backend>>,
    template: ChatTemplate,
    initial: Vec<Message>,
    history: Vec<Message>,
    context: usize,
    /// Explicit --tokens; None generates until end of turn or a full context.
    max_tokens: Option<u32>,
    /// Context kept free for the reply when dropping old turns.
    reserve: u32,
}

pub(crate) fn run(mut args: ChatArgs) -> Result<()> {
    let start = Instant::now();
    args.runtime.use_fast_cuda_defaults();
    args.runtime.use_model_context()?;
    let mut session = Session::open(&mut args)?;
    let interactive = std::io::stdin().is_terminal();
    if interactive {
        let limit = match session.max_tokens {
            Some(tokens) => format!("up to {tokens} tokens per reply"),
            None => "replies run to the end of turn or a full context".into(),
        };
        eprintln!(
            "Model loaded in {:.1}s. Context {} tokens; {limit}. Commands: /reset, /exit. Greedy decoding.",
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
            text => match session.reply(text) {
                Ok((reply, sample)) => {
                    println!("{reply}");
                    if args.stats {
                        print_stats(&sample);
                    }
                }
                Err(error) => eprintln!("local-inference: {error}"),
            },
        }
    }
    Ok(())
}

impl Session {
    fn open(args: &mut ChatArgs) -> Result<Self> {
        let override_source = match &args.chat_template_file {
            Some(path) => Some(fs::read_to_string(path)?),
            None => None,
        };
        let backend = open_backend(&mut args.runtime)?;
        let requested = args.runtime.context();
        let opened = backend.info().context_tokens;
        args.runtime.adopt_context(opened);
        if opened < requested {
            eprintln!(
                "(context {opened} tokens: {requested} do not fit in device memory)"
            );
        }
        let runtime = &args.runtime;
        let tokenizer_adds_bos = !backend.tokenize("")?.is_empty();
        let template = ChatTemplate::from_model(
            &runtime.model,
            override_source.as_deref(),
            tokenizer_adds_bos,
        )?
        .with_thinking(!args.no_thinking);
        let executor = Executor::new(backend, runtime.config())?;
        executor.detokenize(&[])?;
        let initial: Vec<_> = args.system.iter().map(Message::system).collect();
        Ok(Self {
            executor,
            template,
            history: initial.clone(),
            initial,
            context: runtime.context() as usize,
            max_tokens: runtime.chat_tokens(),
            reserve: runtime.chat_reserve(),
        })
    }

    /// Generates a reply; history only changes when generation succeeds.
    fn reply(&mut self, text: &str) -> Result<(String, Sample)> {
        let candidate = [self.history.as_slice(), &[Message::user(text)]].concat();
        let reserved = self.reserve.saturating_sub(1) as usize;
        let mut prompt = Vec::new();
        let fitted = match fit_history(&candidate, |messages| {
            prompt = self.prompt_tokens(messages)?;
            Ok(prompt.len() + reserved <= self.context)
        }) {
            Ok(fitted) => fitted,
            // Without --tokens the reserve only guides dropping old turns: a message
            // that fits alone keeps whatever context is left for the reply.
            Err(_) if self.max_tokens.is_none() => {
                let latest: Vec<Message> = candidate
                    .iter()
                    .filter(|m| m.role == "system")
                    .chain(candidate.last())
                    .cloned()
                    .collect();
                prompt = self.prompt_tokens(&latest)?;
                if prompt.len() >= self.context {
                    return Err(local_inference::Error::Invalid(format!(
                        "message of {} tokens does not fit in the {}-token context",
                        prompt.len(),
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
        // All remaining context (prompt + generated - 1 <= context), or --tokens.
        let room = (self.context + 1).saturating_sub(prompt.len()) as u32;
        let budget = self.max_tokens.map_or(room, |tokens| tokens.min(room));
        let sample = self
            .executor
            .generate(&prompt, budget, GenerationMode::UntilEog)?;
        let ended = sample
            .generated_tokens
            .last()
            .is_some_and(|&token| self.executor.is_eog(token));
        if !ended && sample.generated_tokens.len() as u32 >= budget {
            if budget < room {
                eprintln!("(reply stopped at --tokens {budget})");
            } else {
                eprintln!(
                    "(reply stopped: the {}-token context is full; /reset clears the conversation)",
                    self.context
                );
            }
        }
        let reply = self
            .executor
            .detokenize(&sample.generated_tokens)?
            .trim()
            .to_owned();
        self.history = [fitted, vec![Message::assistant(reply.clone())]].concat();
        Ok((reply, sample))
    }

    fn prompt_tokens(&self, messages: &[Message]) -> Result<Vec<i32>> {
        self.executor
            .tokenize(&self.template.render(messages, true)?)
    }
}

/// Opens the backend; if only the implicit decode graph fails (unsupported
/// architecture or shape), retries without it instead of refusing to chat.
fn open_backend(runtime: &mut RuntimeArgs) -> Result<Box<dyn Backend>> {
    match runtime.open() {
        #[cfg(feature = "cuda")]
        Err(error) if runtime.fall_back_from_implicit_int8() => {
            eprintln!("(int8-fast unavailable: {error}; using f32-tiled)");
            open_with_graph_fallback(runtime)
        }
        #[cfg(feature = "cuda")]
        Err(error) if runtime.implicit_decode_graph() => {
            eprintln!("(decode graph unavailable: {error}; continuing without it)");
            runtime.disable_implicit_decode_graph();
            runtime.open()
        }
        result => result,
    }
}

#[cfg(feature = "cuda")]
fn open_with_graph_fallback(runtime: &mut RuntimeArgs) -> Result<Box<dyn Backend>> {
    let result = match runtime.open() {
        Err(_) if runtime.implicit_decode_graph() => {
            runtime.disable_implicit_decode_graph();
            runtime.open()
        }
        result => result,
    };
    match result {
        Err(error) if runtime.fall_back_to_reference() => {
            eprintln!("(f32-tiled unavailable: {error}; using the f32 profile)");
            runtime.open()
        }
        result => result,
    }
}

fn print_stats(sample: &Sample) {
    let decode_seconds = sample.decode_ns as f64 / 1e9;
    let rate = if decode_seconds > 0.0 {
        sample.decode_steps as f64 / decode_seconds
    } else {
        0.0
    };
    eprintln!(
        "[{} tokens · first token {:.0} ms · {:.1} tok/s]",
        sample.generated_tokens.len(),
        sample.ttft_ns as f64 / 1e6,
        rate
    );
}
