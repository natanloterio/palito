//! Procedural codebooks: a trellis state maps to a value without a lookup table.

const MULTIPLIER: u32 = 34_038_481;
const INCREMENT: u32 = 76_625_530;
/// Mean of the sum of four uniform bytes.
const BYTE_SUM_MEAN: f32 = 510.0;
/// Standard deviation of that sum, so the code has about unit variance.
const BYTE_SUM_STD: f32 = 147.800_54;

/// QTIP's "1MAD" code: one multiply-add on the 32-bit state, then the sum of
/// its four bytes, which is roughly Gaussian by the central limit theorem.
/// Normalized to about zero mean and unit variance.
pub fn one_mad(state: u32) -> f32 {
    let mixed = state.wrapping_mul(MULTIPLIER).wrapping_add(INCREMENT);
    let byte_sum: u32 = mixed.to_le_bytes().iter().map(|&b| u32::from(b)).sum();
    (byte_sum as f32 - BYTE_SUM_MEAN) / BYTE_SUM_STD
}
