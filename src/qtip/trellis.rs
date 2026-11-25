//! Bitshift trellis quantization with Viterbi encoding.
//!
//! A state is an `L`-bit window. Each weight after the first shifts `K` new
//! bits into the window, so consecutive states share `L - K` bits, and the
//! weight is reconstructed as `scale * one_mad(state)`. The encoder stores the
//! first state and then `K` bits per weight: `K + (L - K) / T` bits per weight
//! for a block of `T` weights (tail-biting, which removes the extra bits, is a
//! later step).

use super::codebook::one_mad;
use crate::{Error, Result};

/// Larger windows make the Viterbi tables too big for a prototype.
const MAX_STATE_BITS: u32 = 24;
const MAX_SYMBOL_BITS: u32 = 8;

/// An encoded sequence: the first state, then `K` new bits per later weight.
#[derive(Clone, Debug, PartialEq)]
pub struct Encoded {
    pub start: u32,
    pub symbols: Vec<u8>,
    /// Sum of squared errors of the chosen path.
    pub distortion: f64,
}

#[derive(Clone, Copy, Debug)]
pub struct Trellis {
    state_bits: u32,
    symbol_bits: u32,
    scale: f32,
}

impl Trellis {
    pub fn new(state_bits: u32, symbol_bits: u32, scale: f32) -> Result<Self> {
        if !(1..=MAX_SYMBOL_BITS).contains(&symbol_bits)
            || state_bits <= symbol_bits
            || state_bits > MAX_STATE_BITS
        {
            return Err(Error::Invalid(format!(
                "trellis needs 1 <= K <= {MAX_SYMBOL_BITS} and K < L <= {MAX_STATE_BITS}"
            )));
        }
        if !scale.is_finite() || scale <= 0.0 {
            return Err(Error::Invalid("trellis scale must be positive and finite".into()));
        }
        Ok(Self {
            state_bits,
            symbol_bits,
            scale,
        })
    }

    fn states(&self) -> usize {
        1 << self.state_bits
    }

    fn next_state(&self, state: u32, symbol: u8) -> u32 {
        let mask = (1u32 << self.state_bits) - 1;
        ((state << self.symbol_bits) | u32::from(symbol)) & mask
    }

    fn value(&self, state: u32) -> f32 {
        self.scale * one_mad(state)
    }

    /// The minimum-distortion path through the trellis (Viterbi search).
    pub fn quantize(&self, weights: &[f32]) -> Encoded {
        if weights.is_empty() {
            return Encoded {
                start: 0,
                symbols: Vec::new(),
                distortion: 0.0,
            };
        }
        let states = self.states();
        let values: Vec<f32> = (0..states as u32).map(|s| self.value(s)).collect();
        let error = |weight: f32, state: usize| f64::from(weight - values[state]).powi(2);
        let mut cost: Vec<f64> = (0..states).map(|s| error(weights[0], s)).collect();
        let mut next = vec![0.0; states];
        // A state's predecessors depend only on its tail (state >> K), so the
        // 2^K states sharing a tail share one minimum. back[t][tail] holds the
        // high bits that, shifted in front of the tail, give that predecessor.
        let tails = states >> self.symbol_bits;
        let fan = 1usize << self.symbol_bits;
        let mut back = vec![0u8; (weights.len() - 1) * tails];
        let mut tail_cost = vec![0.0; tails];
        let shift = self.state_bits - self.symbol_bits;
        for (step, &weight) in weights.iter().enumerate().skip(1) {
            let row = &mut back[(step - 1) * tails..step * tails];
            for (tail, (best, high_out)) in tail_cost.iter_mut().zip(row.iter_mut()).enumerate() {
                let (high, value) = (0..fan)
                    .map(|high| (high, cost[tail | (high << shift)]))
                    .min_by(|a, b| a.1.total_cmp(&b.1))
                    .expect("at least one predecessor");
                *best = value;
                *high_out = high as u8;
            }
            for (state, slot) in next.iter_mut().enumerate() {
                *slot = tail_cost[state >> self.symbol_bits] + error(weight, state);
            }
            std::mem::swap(&mut cost, &mut next);
        }
        let (mut state, distortion) = cost
            .iter()
            .copied()
            .enumerate()
            .min_by(|a, b| a.1.total_cmp(&b.1))
            .expect("at least one state");
        let symbol_mask = (1usize << self.symbol_bits) - 1;
        let mut symbols = vec![0u8; weights.len() - 1];
        for step in (1..weights.len()).rev() {
            symbols[step - 1] = (state & symbol_mask) as u8;
            let high = usize::from(back[(step - 1) * tails + (state >> self.symbol_bits)]);
            state = (state >> self.symbol_bits) | (high << shift);
        }
        Encoded {
            start: state as u32,
            symbols,
            distortion,
        }
    }

    pub fn decode(&self, encoded: &Encoded) -> Vec<f32> {
        let first = self.value(encoded.start);
        let rest = encoded.symbols.iter().scan(encoded.start, |state, &symbol| {
            *state = self.next_state(*state, symbol);
            Some(self.value(*state))
        });
        std::iter::once(first).chain(rest).collect()
    }
}
