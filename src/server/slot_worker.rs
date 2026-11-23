//! The worker loop for `--parallel` above 1: one thread owns the device, takes
//! requests from the bounded queue while a slot is free, and gives each active
//! slot a turn. The queue holds only requests that are waiting.

use super::{
    openai,
    reasoning::ReasoningSplitter,
    worker::{
        Event, Job, Prompt, ReplyCall, build_reply, flush_text, opens_reasoning, send_text,
    },
};
use palito::{
    Result,
    engine::{SlotEngine, SlotReply},
    slots::{Advance, Round, Scheduler, SlotBackend, SlotRun, SlotTask},
    tools::ToolEvent,
};
use std::{
    cell::RefCell,
    rc::Rc,
    sync::mpsc::{Receiver, TryRecvError},
};
use tokio::sync::{OwnedSemaphorePermit, mpsc::UnboundedSender};

const DEVICE_LOST: &str = "the GPU stopped responding; restart the server";

/// Streaming state shared between a run's callback and its completion.
struct JobEvents {
    events: UnboundedSender<Event>,
    splitter: Option<ReasoningSplitter>,
    ids: Vec<String>,
}

struct ServerTask {
    run: SlotRun,
    shared: Rc<RefCell<JobEvents>>,
    events: UnboundedSender<Event>,
    opened: Option<bool>,
    permit: Option<OwnedSemaphorePermit>,
}

impl ServerTask {
    fn flush(&self) {
        let mut shared = self.shared.borrow_mut();
        let JobEvents { events, splitter, .. } = &mut *shared;
        flush_text(events, splitter);
    }
}

impl SlotTask for ServerTask {
    fn prompt(&self) -> &[i32] {
        self.run.prompt()
    }

    fn abandoned(&self) -> bool {
        self.events.is_closed()
    }

    fn advance(&mut self, backend: &mut dyn SlotBackend, alone: bool) -> Result<Advance> {
        self.run.advance(backend, alone)
    }

    fn complete(self) {
        let Self { run, shared, events, opened, permit } = self;
        // The reply's last text reaches the splitter here, so it is flushed after.
        let reply = run.into_reply();
        let ids = {
            let mut state = shared.borrow_mut();
            let JobEvents { events, splitter, ids } = &mut *state;
            flush_text(events, splitter);
            std::mem::take(ids)
        };
        let reply = reply.map_err(|error| error.to_string()).map(|reply| match reply {
            SlotReply::Text(completion) => build_reply(completion, opened, Vec::new()),
            SlotReply::Tools(done) => {
                let calls = done.calls.into_iter().zip(ids).map(|(call, id)| ReplyCall {
                    id,
                    name: call.name,
                    arguments: call.arguments,
                    complete: call.complete,
                });
                build_reply(done.completion, opened, calls.collect())
            }
        });
        drop(permit);
        let _ = events.send(Event::Done(reply));
    }

    fn fail(self, error: String) {
        self.flush();
        let Self { events, permit, .. } = self;
        drop(permit);
        let _ = events.send(Event::Done(Err(error)));
    }
}

fn text_callback(shared: &Rc<RefCell<JobEvents>>) -> Box<dyn FnMut(&str) -> bool> {
    let shared = shared.clone();
    Box::new(move |piece| {
        let mut state = shared.borrow_mut();
        let JobEvents { events, splitter, .. } = &mut *state;
        send_text(events, splitter, piece)
    })
}

fn tool_callback(shared: &Rc<RefCell<JobEvents>>) -> Box<dyn FnMut(ToolEvent) -> bool> {
    let shared = shared.clone();
    Box::new(move |event| {
        let mut state = shared.borrow_mut();
        let JobEvents { events, splitter, ids } = &mut *state;
        match event {
            ToolEvent::Text(text) => send_text(events, splitter, &text),
            ToolEvent::CallStart { index, name } => {
                // Text held by the splitter comes before the call.
                let flushed = flush_text(events, splitter);
                let id = openai::response_id("call");
                ids.push(id.clone());
                flushed && events.send(Event::CallStart { index, id, name }).is_ok()
            }
            ToolEvent::CallArguments { index, text } => {
                events.send(Event::CallArguments { index, text }).is_ok()
            }
        }
    })
}

/// Builds the request's run. A request error is answered here, without a slot.
fn prepare(engine: &mut SlotEngine, job: Job) -> Option<ServerTask> {
    let Job { prompt, options, split_reasoning, events, permit } = job;
    let opened = match (&prompt, split_reasoning) {
        (Prompt::Chat(messages), true) => Some(opens_reasoning(engine, messages, &options)),
        (Prompt::ChatTools(messages, spec), true) => Some(
            engine
                .render_chat_tools(messages, spec, options.enable_thinking)
                .is_ok_and(|text| text.trim_end().ends_with("<think>")),
        ),
        _ => None,
    };
    let shared = Rc::new(RefCell::new(JobEvents {
        events: events.clone(),
        splitter: opened.map(ReasoningSplitter::new),
        ids: Vec::new(),
    }));
    let run = match &prompt {
        Prompt::Chat(messages) => engine.prepare_chat(messages, &options, text_callback(&shared)),
        Prompt::ChatTools(messages, spec) => {
            engine.prepare_chat_tools(messages, spec, &options, tool_callback(&shared))
        }
        Prompt::Text(text) => engine.prepare_complete(text, &options, text_callback(&shared)),
    };
    match run {
        Ok(run) => Some(ServerTask { run, shared, events, opened, permit }),
        Err(error) => {
            drop(permit);
            let _ = events.send(Event::Done(Err(error.to_string())));
            None
        }
    }
}

enum Next {
    Job(Box<Job>),
    Empty,
    Closed,
}

/// The next waiting request whose client is still there.
fn next_job(inbox: &Receiver<Job>, block: bool) -> Next {
    loop {
        let job = if block {
            match inbox.recv() {
                Ok(job) => job,
                Err(_) => return Next::Closed,
            }
        } else {
            match inbox.try_recv() {
                Ok(job) => job,
                Err(TryRecvError::Empty) => return Next::Empty,
                Err(TryRecvError::Disconnected) => return Next::Closed,
            }
        };
        if !job.events.is_closed() {
            return Next::Job(Box::new(job));
        }
    }
}

pub(super) fn serve_slots(mut engine: SlotEngine, inbox: Receiver<Job>) {
    let mut scheduler = Scheduler::new(engine.slots());
    loop {
        while scheduler.has_free_slot() {
            match next_job(&inbox, scheduler.is_idle()) {
                Next::Job(job) => {
                    if let Some(task) = prepare(&mut engine, *job) {
                        scheduler.admit(engine.backend_mut(), task);
                    }
                }
                Next::Empty => break,
                Next::Closed if scheduler.is_idle() => return,
                Next::Closed => break,
            }
        }
        if scheduler.round(engine.backend_mut()) == Round::DeviceLost {
            scheduler.fail_all(DEVICE_LOST);
            for job in inbox {
                drop(job.permit);
                let _ = job.events.send(Event::Done(Err(DEVICE_LOST.into())));
            }
            return;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use palito::engine::GenerationOptions;
    use std::sync::mpsc::sync_channel;
    use tokio::sync::mpsc::{UnboundedReceiver, unbounded_channel};

    fn job(text: &str) -> (Job, UnboundedReceiver<Event>) {
        let (events, receiver) = unbounded_channel();
        let job = Job {
            prompt: Prompt::Text(text.into()),
            options: GenerationOptions::default(),
            split_reasoning: false,
            events,
            permit: None,
        };
        (job, receiver)
    }

    #[test]
    fn queued_request_of_a_gone_client_is_skipped() {
        let (sender, inbox) = sync_channel(2);
        let (gone, receiver) = job("gone");
        drop(receiver);
        let (live, _keep) = job("live");
        sender.send(gone).ok().unwrap();
        sender.send(live).ok().unwrap();
        match next_job(&inbox, false) {
            Next::Job(job) => {
                assert!(matches!(&job.prompt, Prompt::Text(text) if text == "live"));
            }
            _ => panic!("expected the live request"),
        }
        assert!(matches!(next_job(&inbox, false), Next::Empty));
        drop(sender);
        assert!(matches!(next_job(&inbox, true), Next::Closed));
    }
}
