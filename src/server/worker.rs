//! The model lives on one worker thread; HTTP handlers queue jobs to it.
//!
//! The engine runs one request at a time (one sequence on the device), so jobs
//! wait in a bounded FIFO queue. Generated text is sent back as it is produced;
//! when the client goes away its receiver is dropped and generation stops.

use palito::{
    Error,
    chat::Message,
    engine::{Completion, Engine, EngineOptions, FinishReason, GenerationOptions},
};
use super::reasoning::{self, Part, ReasoningSplitter};
use std::{
    path::PathBuf,
    sync::mpsc::{SyncSender, TrySendError, sync_channel},
    thread,
};
use tokio::sync::mpsc::{UnboundedReceiver, UnboundedSender, unbounded_channel};

pub enum Prompt {
    Chat(Vec<Message>),
    Text(String),
}

pub enum Event {
    /// Generated text; `reasoning` marks `<think>` content when splitting.
    Text { reasoning: bool, text: String },
    Done(Result<Reply, String>),
}

pub struct Reply {
    pub completion: Completion,
    /// Reasoning split from the answer (None when not splitting or empty).
    pub reasoning: Option<String>,
    /// The answer (the whole text when not splitting).
    pub content: String,
}

struct Job {
    prompt: Prompt,
    options: GenerationOptions,
    split_reasoning: bool,
    events: UnboundedSender<Event>,
}

/// What the worker reports once the model is open.
#[derive(Clone, Debug)]
pub struct ModelReport {
    pub context_tokens: u32,
    pub profile: &'static str,
    pub has_chat_template: bool,
}

#[derive(Clone)]
pub struct Worker {
    jobs: SyncSender<Job>,
}

pub enum SubmitError {
    /// The queue is full; retry later.
    Busy,
    /// The worker thread is gone.
    Stopped,
}

impl Worker {
    /// Opens the model on a new thread; returns once it is ready (or failed).
    pub fn start(
        path: PathBuf,
        options: EngineOptions,
        queue: usize,
    ) -> palito::Result<(Self, ModelReport)> {
        let (jobs, inbox) = sync_channel::<Job>(queue);
        let (ready, opened) = sync_channel(1);
        thread::Builder::new()
            .name("palito-engine".into())
            .spawn(move || {
                let mut engine = match Engine::open(&path, &options) {
                    Ok(engine) => engine,
                    Err(error) => {
                        let _ = ready.send(Err(error));
                        return;
                    }
                };
                for notice in engine.notices() {
                    eprintln!("({notice})");
                }
                let _ = ready.send(Ok(ModelReport {
                    context_tokens: engine.context_tokens(),
                    profile: engine.profile(),
                    has_chat_template: engine.has_chat_template(),
                }));
                for job in inbox {
                    run(&mut engine, job);
                }
            })?;
        let report = opened
            .recv()
            .map_err(|_| Error::Backend("engine thread exited while opening".into()))??;
        Ok((Self { jobs }, report))
    }

    /// `split_reasoning` applies to chat prompts only.
    pub fn submit(
        &self,
        prompt: Prompt,
        options: GenerationOptions,
        split_reasoning: bool,
    ) -> Result<UnboundedReceiver<Event>, SubmitError> {
        let (events, receiver) = unbounded_channel();
        let job = Job {
            prompt,
            options,
            split_reasoning,
            events,
        };
        match self.jobs.try_send(job) {
            Ok(()) => Ok(receiver),
            Err(TrySendError::Full(_)) => Err(SubmitError::Busy),
            Err(TrySendError::Disconnected(_)) => Err(SubmitError::Stopped),
        }
    }
}

fn run(engine: &mut Engine, job: Job) {
    // A request whose client already left is skipped without touching the GPU.
    if job.events.is_closed() {
        return;
    }
    let events = &job.events;
    let send = |reasoning: bool, text: String| events.send(Event::Text { reasoning, text }).is_ok();
    let opened = match (&job.prompt, job.split_reasoning) {
        (Prompt::Chat(messages), true) => Some(opens_reasoning(engine, messages, &job.options)),
        _ => None,
    };
    let mut splitter = opened.map(ReasoningSplitter::new);
    let on_text = |piece: &str| match &mut splitter {
        None => send(false, piece.to_owned()),
        Some(splitter) => splitter
            .push(piece)
            .into_iter()
            .all(|(part, text)| send(part == Part::Reasoning, text)),
    };
    let result = match &job.prompt {
        Prompt::Chat(messages) => engine.chat(messages, &job.options, on_text),
        Prompt::Text(text) => engine.complete(text, &job.options, on_text),
    };
    if let Some(splitter) = &mut splitter {
        for (part, text) in splitter.finish() {
            send(part == Part::Reasoning, text);
        }
    }
    let reply = result.map_err(|e| e.to_string()).map(|completion| {
        if completion.finish_reason == FinishReason::Cancelled {
            eprintln!(
                "request cancelled by the client after {} tokens",
                completion.generated_tokens
            );
        }
        let (reasoning, content) = match opened {
            Some(opened) => {
                let (reasoning, content) = reasoning::split(&completion.text, opened);
                ((!reasoning.is_empty()).then_some(reasoning), content)
            }
            None => (None, completion.text.clone()),
        };
        Reply {
            completion,
            reasoning,
            content,
        }
    });
    let _ = job.events.send(Event::Done(reply));
}

/// Whether the rendered prompt already opens a `<think>` block.
fn opens_reasoning(engine: &Engine, messages: &[Message], options: &GenerationOptions) -> bool {
    engine
        .render_chat(messages, options.thinking())
        .is_ok_and(|prompt| prompt.trim_end().ends_with("<think>"))
}
