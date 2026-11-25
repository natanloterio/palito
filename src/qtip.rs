//! Research prototype of QTIP-style trellis quantization (Tseng et al., 2024).
//!
//! Step 1: a bitshift trellis with a procedural codebook, encoded by Viterbi
//! search and checked on synthetic Gaussian blocks against the optimal 2-bit
//! scalar quantizer. Step 2: random Hadamard incoherence, checked on real
//! Llama layers. Step 3: whole-model GGUFs with reconstructed weights, for
//! perplexity. Step 4: BlockLDLQ with input Hessians collected by
//! `scripts/qtip/collect_hessians.py`. Tail-biting and GPU decoding come later.

pub mod codebook;
pub mod hessian;
pub mod incoherence;
pub mod ldlq;
pub mod matrix;
pub mod model;
pub mod packed;
pub mod scalar;
pub mod trellis;

#[cfg(test)]
mod layer_tests;
#[cfg(test)]
mod tests;
