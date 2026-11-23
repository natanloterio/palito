//! Request state shared by the CPU reference and CUDA backends.
use crate::{Error, Result, backend::ModelInfo};

#[derive(Default)]
pub(crate) struct State {
    pub(crate) position: usize,
    pub(crate) valid: bool,
    pub(crate) failed: bool,
}

impl State {
    pub(crate) fn begin(&mut self, tokens: &[i32], position: u32, info: &ModelInfo) -> Result<()> {
        if self.failed {
            return Err(Error::Backend("failed request must be reset".into()));
        }
        if tokens.is_empty()
            || tokens.len() > info.batch_tokens as usize
            || position as usize != self.position
            || self
                .position
                .checked_add(tokens.len())
                .is_none_or(|n| n > info.context_tokens as usize)
            || tokens
                .iter()
                .any(|&id| id < 0 || id as u32 >= info.vocab_tokens)
        {
            return Err(Error::Invalid(
                "invalid IDs, batch, position, or context capacity".into(),
            ));
        }
        self.valid = false;
        // A numerical or driver error at any point leaves the request poisoned.
        self.failed = true;
        Ok(())
    }

    pub(crate) fn finish(&mut self, count: usize) {
        self.position += count;
        self.valid = true;
        self.failed = false;
    }

    pub(crate) fn logits<'a>(&self, logits: &'a [f32]) -> Result<&'a [f32]> {
        if !self.valid || self.failed {
            return Err(Error::Backend("no valid next-token logits".into()));
        }
        Ok(logits)
    }
}

pub(crate) fn greedy(logits: &[f32]) -> Result<i32> {
    if logits.is_empty() || logits.iter().any(|x| !x.is_finite()) {
        return Err(Error::Backend("invalid logits".into()));
    }
    let mut best = 0;
    for i in 1..logits.len() {
        if logits[i] > logits[best] {
            best = i;
        }
    }
    Ok(best as i32)
}
