//! Research prototype of QTIP-style trellis quantization (Tseng et al., 2024).
//!
//! Step 1 of the plan: a bitshift trellis with a procedural codebook, encoded
//! by Viterbi search and checked on synthetic Gaussian blocks against the
//! optimal 2-bit scalar quantizer. Incoherence processing (random Hadamard
//! transforms), tail-biting and GPU decoding come later.

pub mod codebook;
pub mod scalar;
pub mod trellis;

#[cfg(test)]
mod tests;
