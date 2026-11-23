//! Gives each active slot one turn at a time on the device.

use super::{Advance, SlotBackend, SlotCandidate, pick_slot};
use crate::Result;

/// A request occupying a slot.
pub trait SlotTask {
    fn prompt(&self) -> &[i32];
    /// The client is gone: the slot can be freed without more device work.
    fn abandoned(&self) -> bool;
    fn advance(&mut self, backend: &mut dyn SlotBackend, alone: bool) -> Result<Advance>;
    /// Delivers the reply to the client.
    fn complete(self);
    fn fail(self, error: String);
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Round {
    /// No slot is active.
    Idle,
    Ran,
    /// The device stopped responding; the remaining tasks are still in their slots.
    DeviceLost,
}

pub struct Scheduler<T: SlotTask> {
    tasks: Vec<Option<T>>,
    last_used: Vec<u64>,
    clock: u64,
}

impl<T: SlotTask> Scheduler<T> {
    pub fn new(slots: usize) -> Self {
        Self {
            tasks: (0..slots).map(|_| None).collect(),
            last_used: vec![0; slots],
            clock: 0,
        }
    }

    pub fn has_free_slot(&self) -> bool {
        self.tasks.iter().any(Option::is_none)
    }

    pub fn is_idle(&self) -> bool {
        self.tasks.iter().all(Option::is_none)
    }

    /// Puts `task` in the free slot that restores most of its prompt.
    pub fn admit(&mut self, backend: &dyn SlotBackend, task: T) {
        let free: Vec<SlotCandidate> = (0..self.tasks.len())
            .filter(|&slot| self.tasks[slot].is_none())
            .map(|slot| SlotCandidate {
                slot,
                restorable: backend.restorable(slot, task.prompt()),
                last_used: self.last_used[slot],
            })
            .collect();
        match pick_slot(&free) {
            Some(slot) => {
                self.touch(slot);
                self.tasks[slot] = Some(task);
            }
            None => task.fail("no free slot".into()),
        }
    }

    /// One turn for each active slot, in slot order.
    pub fn round(&mut self, backend: &mut dyn SlotBackend) -> Round {
        if self.is_idle() {
            return Round::Idle;
        }
        for slot in 0..self.tasks.len() {
            if !self.turn(backend, slot) {
                return Round::DeviceLost;
            }
        }
        Round::Ran
    }

    /// Ends every active task with `error`.
    pub fn fail_all(&mut self, error: &str) {
        for task in self.tasks.iter_mut().filter_map(Option::take) {
            task.fail(error.to_owned());
        }
    }

    /// Returns false when the device is lost.
    fn turn(&mut self, backend: &mut dyn SlotBackend, slot: usize) -> bool {
        let alone = self.tasks.iter().flatten().count() == 1;
        let Some(mut task) = self.tasks[slot].take() else {
            return true;
        };
        if task.abandoned() {
            // Its K/V stays valid as this slot's prefix cache.
            self.touch(slot);
            return true;
        }
        if let Err(error) = backend.select_slot(slot) {
            task.fail(error.to_string());
            return false;
        }
        match task.advance(backend, alone) {
            Ok(Advance::Working) => {
                self.tasks[slot] = Some(task);
                true
            }
            Ok(Advance::Finished) => {
                self.touch(slot);
                task.complete();
                true
            }
            Err(error) => {
                task.fail(error.to_string());
                self.touch(slot);
                backend.abandon_slot(slot).is_ok() && backend.device_alive().is_ok()
            }
        }
    }

    fn touch(&mut self, slot: usize) {
        self.clock += 1;
        self.last_used[slot] = self.clock;
    }
}

#[cfg(test)]
mod tests;
