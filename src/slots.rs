//! Server slots: several sequences taking turns on one device.

use crate::{Error, Result, backend::Backend};

pub const MAX_SLOTS: u32 = 64;
/// Most prompt tokens one slot evaluates in a turn.
pub const PROMPT_TURN_TOKENS: u32 = 512;
/// Most tokens a slot generates in a turn while no other slot is active.
pub const SOLO_BURST_TOKENS: u32 = 8;
const SLOT_CONTEXT_STEP: u32 = 256;

/// Context of each slot when `slots` share `total` tokens: the integer share,
/// rounded up to a multiple of 256. One slot keeps `total` as it is.
pub fn slot_context(total: u32, slots: u32) -> Result<u32> {
    if slots == 0 || slots > MAX_SLOTS {
        return Err(Error::Invalid(format!(
            "--parallel must be between 1 and {MAX_SLOTS}"
        )));
    }
    if slots == 1 {
        return Ok(total);
    }
    let share = total / slots;
    if share == 0 {
        return Err(Error::Invalid(format!(
            "context of {total} tokens is too small for --parallel {slots}"
        )));
    }
    Ok(share.next_multiple_of(SLOT_CONTEXT_STEP))
}

/// A backend holding several sequences, one active at a time. The `Backend`
/// methods act on the active slot.
pub trait SlotBackend: Backend {
    fn slot_count(&self) -> usize;
    /// Waits for the device to finish and makes `slot` the active one.
    fn select_slot(&mut self, slot: usize) -> Result<()>;
    /// Tokens of `prompt` the cache of `slot` would restore, without changing it.
    fn restorable(&self, slot: usize, prompt: &[i32]) -> usize;
    /// Drops the slot's prefix and clears any failure left on the device.
    fn abandon_slot(&mut self, slot: usize) -> Result<()>;
    /// Err when the device context no longer responds.
    fn device_alive(&mut self) -> Result<()>;
}

/// A free slot that could take a request.
#[derive(Clone, Copy)]
pub struct SlotCandidate {
    pub slot: usize,
    /// Prompt tokens its prefix cache would restore.
    pub restorable: usize,
    /// Larger is more recent.
    pub last_used: u64,
}

/// The slot restoring the most tokens; on a tie, the least recently used.
pub fn pick_slot(free: &[SlotCandidate]) -> Option<usize> {
    free.iter()
        .max_by_key(|c| (c.restorable, std::cmp::Reverse(c.last_used)))
        .map(|c| c.slot)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn one_slot_keeps_the_context() {
        assert_eq!(slot_context(1000, 1).unwrap(), 1000);
    }

    #[test]
    fn context_rounds_up_to_256() {
        assert_eq!(slot_context(4096, 4).unwrap(), 1024);
        // Integer division comes first.
        assert_eq!(slot_context(4097, 4).unwrap(), 1024);
        assert_eq!(slot_context(4100, 4).unwrap(), 1280);
    }

    #[test]
    fn context_smaller_than_the_slots_fails() {
        assert!(slot_context(3, 4).is_err());
    }

    #[test]
    fn picks_most_restorable_then_oldest() {
        let c = |slot, restorable, last_used| SlotCandidate {
            slot,
            restorable,
            last_used,
        };
        assert_eq!(pick_slot(&[c(0, 0, 1), c(1, 512, 9)]), Some(1));
        assert_eq!(pick_slot(&[c(0, 512, 9), c(1, 512, 2)]), Some(1));
        assert_eq!(pick_slot(&[]), None);
    }
}
