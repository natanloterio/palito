//! Research prototype of QTIP-style trellis quantization (Tseng et al., 2024).
//!
//! Step 1: a bitshift trellis with a procedural codebook, encoded by Viterbi
//! search and checked on synthetic Gaussian blocks against the optimal 2-bit
//! scalar quantizer. Step 2: random Hadamard incoherence, checked on real
//! Llama layers. Tail-biting and GPU decoding come later.

pub mod codebook;
pub mod incoherence;
pub mod matrix;
pub mod scalar;
pub mod trellis;

#[cfg(test)]
mod layer_tests;
#[cfg(test)]
mod tests;
