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
    max_tokens: u32,
}

pub(crate) fn run(args: ChatArgs) -> Result<()> {
    let start = Instant::now();
    let mut session = Session::open(&args)?;
    let interactive = std::io::stdin().is_terminal();
    if interactive {
        eprintln!(
            "Model loaded in {:.1}s. Commands: /reset, /exit. Greedy decoding, up to {} tokens per reply.",
            start.elapsed().as_secs_f64(),
            session.max_tokens
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
    fn open(args: &ChatArgs) -> Result<Self> {
        let runtime = &args.runtime;
        let override_source = match &args.chat_template_file {
            Some(path) => Some(fs::read_to_string(path)?),
            None => None,
        };
        let backend = runtime.open()?;
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
            context: runtime.context as usize,
            max_tokens: runtime.tokens,
        })
    }

    /// Generates a reply; history only changes when generation succeeds.
    fn reply(&mut self, text: &str) -> Result<(String, Sample)> {
        let candidate = [self.history.as_slice(), &[Message::user(text)]].concat();
        let reserved = self.max_tokens.saturating_sub(1) as usize;
        let mut prompt = Vec::new();
        let fitted = fit_history(&candidate, |messages| {
            prompt = self.prompt_tokens(messages)?;
            Ok(prompt.len() + reserved <= self.context)
        })?;
        if fitted.len() < candidate.len() {
            eprintln!("(oldest turns dropped to fit --context {})", self.context);
        }
        let sample = self
            .executor
            .generate(&prompt, self.max_tokens, GenerationMode::UntilEog)?;
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
