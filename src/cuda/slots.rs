//! Several sequences on one device. Weights, kernels and the workspace are
//! shared; each slot owns its K/V, logits, position, prefix cache and decode
//! graph. The active slot's state lives in the backend's own fields, so the
//! numerical code does not know slots exist.

use super::{CudaBackend, CudaMatmul, State, decode, failure, invalid, kv, prefix};
use crate::{Result, slots::SlotBackend};
use cudarc::driver::CudaStream;
use std::{mem::swap, sync::Arc};

/// The state of a slot that is not active.
pub(super) struct ParkedSlot {
    // First: its graph holds raw addresses of the buffers below.
    decode: Option<decode::DecodeState>,
    keys: Vec<kv::KvBuffer>,
    values: Vec<kv::KvBuffer>,
    logits: Vec<f32>,
    state: State,
    prefix: Option<prefix::PrefixCache>,
}

/// What to allocate for each slot beyond the first.
pub(super) struct SlotShape {
    pub slots: usize,
    pub layers: usize,
    pub kv_elements: usize,
    pub vocabulary: usize,
    pub matmul: CudaMatmul,
    pub decode_graph: bool,
    pub prefix_cache: bool,
}

/// One entry per slot; slot 0 starts active, so its entry is empty.
pub(super) fn park_slots(
    stream: &Arc<CudaStream>,
    shape: &SlotShape,
) -> Result<Vec<Option<ParkedSlot>>> {
    let mut parked = vec![];
    parked.push(None);
    for _ in 1..shape.slots {
        parked.push(Some(new_slot(stream, shape)?));
    }
    Ok(parked)
}

fn new_slot(stream: &Arc<CudaStream>, shape: &SlotShape) -> Result<ParkedSlot> {
    let buffers = || {
        (0..shape.layers)
            .map(|_| kv::KvBuffer::zeros(stream, shape.kv_elements, shape.matmul))
            .collect::<Result<Vec<_>>>()
    };
    Ok(ParkedSlot {
        decode: shape
            .decode_graph
            .then(|| decode::DecodeState::new(stream, shape.vocabulary))
            .transpose()?,
        keys: buffers()?,
        values: buffers()?,
        logits: vec![0.0; shape.vocabulary],
        state: State::default(),
        prefix: shape.prefix_cache.then(prefix::PrefixCache::default),
    })
}

impl CudaBackend {
    /// Waits for the device and makes `slot` the active one.
    fn activate(&mut self, slot: usize) -> Result<()> {
        if slot >= self.parked.len() {
            return Err(invalid("slot out of range"));
        }
        self.stream.synchronize().map_err(failure)?;
        if slot == self.active_slot {
            return Ok(());
        }
        let mut next = self.parked[slot]
            .take()
            .ok_or_else(|| failure("slot state missing"))?;
        swap(&mut self.decode, &mut next.decode);
        swap(&mut self.keys, &mut next.keys);
        swap(&mut self.values, &mut next.values);
        swap(&mut self.logits, &mut next.logits);
        swap(&mut self.state, &mut next.state);
        swap(&mut self.prefix, &mut next.prefix);
        self.parked[self.active_slot] = Some(next);
        self.active_slot = slot;
        Ok(())
    }

    /// Marks the shared status word as a failed kernel would.
    #[doc(hidden)]
    pub fn poison_status_for_test(&mut self) -> Result<()> {
        self.stream
            .memcpy_htod(&[1], &mut self.work.status.slice_mut(..1))
            .map_err(failure)?;
        self.stream.synchronize().map_err(failure)
    }
}

impl SlotBackend for CudaBackend {
    fn slot_count(&self) -> usize {
        self.parked.len()
    }

    fn select_slot(&mut self, slot: usize) -> Result<()> {
        self.activate(slot)?;
        // Slot 0's graph is captured at open, the others on first use.
        if self.decode.as_ref().is_some_and(|d| !d.has_graph()) {
            self.capture_decode_graph()?;
        }
        Ok(())
    }

    fn restorable(&self, slot: usize, prompt: &[i32]) -> usize {
        let cache = if slot == self.active_slot {
            self.prefix.as_ref()
        } else {
            self.parked
                .get(slot)
                .and_then(Option::as_ref)
                .and_then(|parked| parked.prefix.as_ref())
        };
        cache.map_or(0, |cache| cache.restorable(prompt))
    }

    fn abandon_slot(&mut self, slot: usize) -> Result<()> {
        self.activate(slot)?;
        crate::backend::Backend::reset(self)?;
        if let Some(prefix) = &mut self.prefix {
            *prefix = prefix::PrefixCache::default();
        }
        Ok(())
    }

    fn device_alive(&mut self) -> Result<()> {
        self.stream.synchronize().map_err(failure)
    }
}
