//! The model lives on one worker thread; HTTP handlers queue jobs to it.
//!
//! The engine runs one request at a time (one sequence on the device), so jobs
//! wait in a bounded FIFO queue. Generated text is sent back as it is produced;
//! when the client goes away its receiver is dropped and generation stops.

use palito::{
    Error,
    chat::Message,
    engine::{Completion, Engine, EngineOptions, GenerationOptions},
};
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
    Text(String),
    Done(Result<Completion, String>),
}

struct Job {
    prompt: Prompt,
    options: GenerationOptions,
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

    pub fn submit(
        &self,
        prompt: Prompt,
        options: GenerationOptions,
    ) -> Result<UnboundedReceiver<Event>, SubmitError> {
        let (events, receiver) = unbounded_channel();
        let job = Job {
            prompt,
            options,
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
    let send = |piece: &str| events.send(Event::Text(piece.to_owned())).is_ok();
    let result = match &job.prompt {
        Prompt::Chat(messages) => engine.chat(messages, &job.options, send),
        Prompt::Text(text) => engine.complete(text, &job.options, send),
    };
    if let Ok(completion) = &result {
        if completion.finish_reason == palito::engine::FinishReason::Cancelled {
            eprintln!(
                "request cancelled by the client after {} tokens",
                completion.generated_tokens
            );
        }
    }
    let _ = job.events.send(Event::Done(result.map_err(|e| e.to_string())));
}
