//! The model lives on one worker thread; HTTP handlers queue jobs to it.
//!
//! The engine runs one request at a time (one sequence on the device), so jobs
//! wait in a bounded FIFO queue. Generated text is sent back as it is produced;
//! when the client goes away its receiver is dropped and generation stops.

use palito::{
    Error,
    chat::{Message, ToolFormat},
    engine::{Completion, Engine, EngineOptions, FinishReason, GenerationOptions},
    tools::{ToolEvent, ToolSpec},
};
use super::{
    openai,
    reasoning::{self, Part, ReasoningSplitter},
};
use std::{
    path::PathBuf,
    sync::mpsc::{SyncSender, TrySendError, sync_channel},
    thread,
};
use tokio::sync::mpsc::{UnboundedReceiver, UnboundedSender, unbounded_channel};

pub enum Prompt {
    Chat(Vec<Message>),
    ChatTools(Vec<Message>, ToolSpec),
    Text(String),
}

pub enum Event {
    /// Generated text; `reasoning` marks `<think>` content when splitting.
    Text { reasoning: bool, text: String },
    CallStart { index: usize, id: String, name: String },
    CallArguments { index: usize, text: String },
    Done(Result<Reply, String>),
}

pub struct Reply {
    pub completion: Completion,
    /// Reasoning split from the answer (None when not splitting or empty).
    pub reasoning: Option<String>,
    /// The answer (the whole text when not splitting).
    pub content: String,
    pub tool_calls: Vec<ReplyCall>,
}

#[derive(Clone, Debug)]
pub struct ReplyCall {
    pub id: String,
    pub name: String,
    /// JSON text; partial when `complete` is false.
    pub arguments: String,
    pub complete: bool,
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
    /// None when the template has no tool call format Palito can generate.
    pub tool_format: Option<ToolFormat>,
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
                    tool_format: engine.tool_format(),
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
        Prompt::ChatTools(messages, spec) => return run_tools(engine, &job, messages, spec),
        Prompt::Text(text) => engine.complete(text, &job.options, on_text),
    };
    if let Some(splitter) = &mut splitter {
        for (part, text) in splitter.finish() {
            send(part == Part::Reasoning, text);
        }
    }
    let reply = result
        .map_err(|e| e.to_string())
        .map(|completion| build_reply(completion, opened, Vec::new()));
    let _ = job.events.send(Event::Done(reply));
}

fn run_tools(engine: &mut Engine, job: &Job, messages: &[Message], spec: &ToolSpec) {
    let events = &job.events;
    let opened = job.split_reasoning.then(|| {
        engine
            .render_chat_tools(messages, spec, job.options.enable_thinking)
            .is_ok_and(|prompt| prompt.trim_end().ends_with("<think>"))
    });
    let mut splitter = opened.map(ReasoningSplitter::new);
    let mut ids = Vec::new();
    let result = engine.chat_tools(messages, spec, &job.options, |event| match event {
        ToolEvent::Text(text) => send_text(events, &mut splitter, &text),
        ToolEvent::CallStart { index, name } => {
            // Text held by the splitter comes before the call.
            let flushed = flush_text(events, &mut splitter);
            let id = openai::response_id("call");
            ids.push(id.clone());
            flushed && events.send(Event::CallStart { index, id, name }).is_ok()
        }
        ToolEvent::CallArguments { index, text } => {
            events.send(Event::CallArguments { index, text }).is_ok()
        }
    });
    flush_text(events, &mut splitter);
    let reply = result.map_err(|e| e.to_string()).map(|done| {
        let calls = done.calls.into_iter().zip(ids).map(|(call, id)| ReplyCall {
            id,
            name: call.name,
            arguments: call.arguments,
            complete: call.complete,
        });
        build_reply(done.completion, opened, calls.collect())
    });
    let _ = events.send(Event::Done(reply));
}

fn send_text(
    events: &UnboundedSender<Event>,
    splitter: &mut Option<ReasoningSplitter>,
    piece: &str,
) -> bool {
    match splitter {
        None => send_part(events, Part::Content, piece.to_owned()),
        Some(splitter) => {
            splitter.push(piece).into_iter().all(|(part, text)| send_part(events, part, text))
        }
    }
}

fn flush_text(events: &UnboundedSender<Event>, splitter: &mut Option<ReasoningSplitter>) -> bool {
    splitter.as_mut().is_none_or(|splitter| {
        splitter.finish().into_iter().all(|(part, text)| send_part(events, part, text))
    })
}

fn send_part(events: &UnboundedSender<Event>, part: Part, text: String) -> bool {
    events.send(Event::Text { reasoning: part == Part::Reasoning, text }).is_ok()
}

fn build_reply(completion: Completion, opened: Option<bool>, tool_calls: Vec<ReplyCall>) -> Reply {
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
        tool_calls,
    }
}

/// Whether the rendered prompt already opens a `<think>` block.
fn opens_reasoning(engine: &Engine, messages: &[Message], options: &GenerationOptions) -> bool {
    engine
        .render_chat(messages, options.thinking())
        .is_ok_and(|prompt| prompt.trim_end().ends_with("<think>"))
}
