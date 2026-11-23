use super::*;
use crate::{
    Error,
    backend::{Backend, Capabilities, ModelInfo},
};
use std::{cell::RefCell, rc::Rc};

#[derive(Default)]
struct FakeDevice {
    selected: Vec<usize>,
    abandoned: Vec<usize>,
    restorable: Vec<usize>,
    dead: bool,
}

struct FakeBackend {
    info: ModelInfo,
    device: FakeDevice,
    slots: usize,
}

fn backend(slots: usize) -> FakeBackend {
    FakeBackend {
        info: ModelInfo {
            backend: "test".into(),
            revision: "test".into(),
            architecture: "test".into(),
            tensor_types: vec![],
            weight_bytes: 0,
            parameters: 0,
            context_tokens: 64,
            batch_tokens: 8,
            vocab_tokens: 32,
            capabilities: Capabilities { token_step: true, chunked_prefill: true },
        },
        device: FakeDevice { restorable: vec![0; slots], ..FakeDevice::default() },
        slots,
    }
}

impl Backend for FakeBackend {
    fn info(&self) -> &ModelInfo {
        &self.info
    }
    fn tokenize(&self, _: &str) -> Result<Vec<i32>> {
        Ok(vec![])
    }
    fn detokenize(&self, _: &[i32]) -> Result<String> {
        Ok(String::new())
    }
    fn reset(&mut self) -> Result<()> {
        Ok(())
    }
    fn evaluate(&mut self, _: &[i32], _: u32) -> Result<()> {
        Ok(())
    }
    fn sample_greedy(&mut self) -> Result<i32> {
        Ok(0)
    }
    fn is_eog(&self, _: i32) -> bool {
        false
    }
}

impl SlotBackend for FakeBackend {
    fn slot_count(&self) -> usize {
        self.slots
    }
    fn select_slot(&mut self, slot: usize) -> Result<()> {
        self.device.selected.push(slot);
        Ok(())
    }
    fn restorable(&self, slot: usize, _: &[i32]) -> usize {
        self.device.restorable[slot]
    }
    fn abandon_slot(&mut self, slot: usize) -> Result<()> {
        self.device.abandoned.push(slot);
        Ok(())
    }
    fn device_alive(&mut self) -> Result<()> {
        if self.device.dead {
            Err(Error::Backend("device lost".into()))
        } else {
            Ok(())
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
enum Seen {
    Advanced(&'static str, bool),
    Completed(&'static str),
    Failed(&'static str),
}

type Log = Rc<RefCell<Vec<Seen>>>;

struct Task {
    name: &'static str,
    steps: usize,
    gone: Rc<RefCell<bool>>,
    error: Option<fn() -> Error>,
    log: Log,
}

impl Task {
    fn new(name: &'static str, steps: usize, log: &Log) -> Self {
        Self { name, steps, gone: Rc::default(), error: None, log: log.clone() }
    }
}

impl SlotTask for Task {
    fn prompt(&self) -> &[i32] {
        &[]
    }
    fn abandoned(&self) -> bool {
        *self.gone.borrow()
    }
    fn advance(&mut self, _: &mut dyn SlotBackend, alone: bool) -> Result<Advance> {
        self.log.borrow_mut().push(Seen::Advanced(self.name, alone));
        if let Some(error) = self.error {
            return Err(error());
        }
        self.steps -= 1;
        Ok(if self.steps == 0 { Advance::Finished } else { Advance::Working })
    }
    fn complete(self) {
        self.log.borrow_mut().push(Seen::Completed(self.name));
    }
    fn fail(self, _: String) {
        self.log.borrow_mut().push(Seen::Failed(self.name));
    }
}

fn outcomes(log: &Log) -> Vec<Seen> {
    log.borrow()
        .iter()
        .filter(|seen| !matches!(seen, Seen::Advanced(..)))
        .cloned()
        .collect()
}

#[test]
fn gives_one_turn_per_active_slot_in_order() {
    let log = Log::default();
    let mut device = backend(3);
    let mut scheduler = Scheduler::new(3);
    for name in ["a", "b", "c"] {
        scheduler.admit(&device, Task::new(name, 9, &log));
    }
    scheduler.round(&mut device);
    scheduler.round(&mut device);
    assert_eq!(device.device.selected, [0, 1, 2, 0, 1, 2]);
}

#[test]
fn short_task_finishes_before_a_long_one() {
    let log = Log::default();
    let mut device = backend(2);
    let mut scheduler = Scheduler::new(2);
    scheduler.admit(&device, Task::new("long", 100, &log));
    scheduler.admit(&device, Task::new("short", 4, &log));
    while !scheduler.is_idle() {
        scheduler.round(&mut device);
    }
    assert_eq!(outcomes(&log), [Seen::Completed("short"), Seen::Completed("long")]);
}

#[test]
fn lone_task_runs_with_alone_true() {
    let log = Log::default();
    let mut device = backend(2);
    let mut scheduler = Scheduler::new(2);
    scheduler.admit(&device, Task::new("a", 9, &log));
    scheduler.round(&mut device);
    scheduler.admit(&device, Task::new("b", 9, &log));
    scheduler.round(&mut device);
    assert_eq!(
        *log.borrow(),
        [
            Seen::Advanced("a", true),
            Seen::Advanced("a", false),
            Seen::Advanced("b", false)
        ]
    );
}

#[test]
fn admit_prefers_the_slot_with_the_longest_prefix() {
    let log = Log::default();
    let mut device = backend(3);
    device.device.restorable = vec![0, 512, 0];
    let mut scheduler = Scheduler::new(3);
    scheduler.admit(&device, Task::new("a", 9, &log));
    scheduler.round(&mut device);
    assert_eq!(device.device.selected, [1]);
}

#[test]
fn a_tie_goes_to_the_least_recently_used_slot() {
    let log = Log::default();
    let mut device = backend(2);
    let mut scheduler = Scheduler::new(2);
    scheduler.admit(&device, Task::new("first", 1, &log));
    scheduler.round(&mut device);
    device.device.selected.clear();
    scheduler.admit(&device, Task::new("second", 1, &log));
    scheduler.round(&mut device);
    assert_eq!(device.device.selected, [1]);
}

#[test]
fn abandoned_task_frees_its_slot_without_advancing() {
    let log = Log::default();
    let mut device = backend(1);
    let mut scheduler = Scheduler::new(1);
    let task = Task::new("a", 9, &log);
    let gone = task.gone.clone();
    scheduler.admit(&device, task);
    assert!(!scheduler.has_free_slot());
    *gone.borrow_mut() = true;
    scheduler.round(&mut device);
    assert!(scheduler.has_free_slot());
    assert!(log.borrow().is_empty());
    assert!(device.device.selected.is_empty());
    assert!(device.device.abandoned.is_empty(), "its prefix stays reusable");
}

#[test]
fn request_error_ends_only_that_task() {
    let log = Log::default();
    let mut device = backend(2);
    let mut scheduler = Scheduler::new(2);
    let mut bad = Task::new("bad", 9, &log);
    bad.error = Some(|| Error::Invalid("no token allowed".into()));
    scheduler.admit(&device, bad);
    scheduler.admit(&device, Task::new("good", 2, &log));
    assert!(matches!(scheduler.round(&mut device), Round::Ran));
    assert_eq!(device.device.abandoned, [0]);
    while !scheduler.is_idle() {
        scheduler.round(&mut device);
    }
    assert_eq!(outcomes(&log), [Seen::Failed("bad"), Seen::Completed("good")]);
}

#[test]
fn backend_error_with_a_live_device_keeps_the_others() {
    let log = Log::default();
    let mut device = backend(2);
    let mut scheduler = Scheduler::new(2);
    let mut bad = Task::new("bad", 9, &log);
    bad.error = Some(|| Error::Backend("nonfinite logits".into()));
    scheduler.admit(&device, bad);
    scheduler.admit(&device, Task::new("good", 1, &log));
    assert!(matches!(scheduler.round(&mut device), Round::Ran));
    assert_eq!(outcomes(&log), [Seen::Failed("bad"), Seen::Completed("good")]);
}

#[test]
fn lost_device_is_reported() {
    let log = Log::default();
    let mut device = backend(2);
    device.device.dead = true;
    let mut scheduler = Scheduler::new(2);
    let mut bad = Task::new("bad", 9, &log);
    bad.error = Some(|| Error::Backend("device lost".into()));
    scheduler.admit(&device, bad);
    scheduler.admit(&device, Task::new("other", 9, &log));
    assert!(matches!(scheduler.round(&mut device), Round::DeviceLost));
}

#[test]
fn fail_all_reaches_every_active_task() {
    let log = Log::default();
    let device = backend(2);
    let mut scheduler = Scheduler::new(2);
    scheduler.admit(&device, Task::new("a", 9, &log));
    scheduler.admit(&device, Task::new("b", 9, &log));
    scheduler.fail_all("device lost");
    assert!(scheduler.is_idle());
    assert_eq!(outcomes(&log), [Seen::Failed("a"), Seen::Failed("b")]);
}

#[test]
fn an_idle_scheduler_reports_idle() {
    let mut device = backend(1);
    let mut scheduler = Scheduler::<Task>::new(1);
    assert!(matches!(scheduler.round(&mut device), Round::Idle));
}
