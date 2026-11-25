//! QTIP HYB (Hybrid LUT + Bitshift Trellis) with V=2 weights per step.
//!
//! Each 16x16 tile holds 256 weights = 128 pairs. A 16-bit trellis state (`L=16`)
//! maps to a pair of FP16 weights via a 512-entry (`Q=9`) positive-quadrant 2D
//! Gaussian codebook (`2 KB`) plus 2 sign bits extracted directly from the hash
//! (`dual sign-flip`), yielding 2,048 effective 2D Gaussian centroids.
//!
//! Because `V=2`, shifting `B in {3, 4, 5, 6}` bits per pair gives native
//! `1.5, 2.0, 2.5, 3.0` bits/weight with exact circular (tail-biting) tile sizes
//! of `48, 64, 80, 96` bytes (`12, 16, 20, 24` u32 words) — zero start-state
//! overhead and clean cache-line alignment.

use super::ldlq::TILE;
use crate::{Error, Result};
use half::f16;

pub const HYB_STATE_BITS: u32 = 16;
pub const HYB_LUT_ENTRIES: usize = 512;
pub const HYB_PAIRS_PER_TILE: usize = (TILE * TILE) / 2; // 128 pairs = 256 weights
/// Knuth 32-bit multiplicative hash constant (shared bit-for-bit with CUDA).
pub const HYB_HASH_MULT: u32 = 0x9E37_79B1;

/// Number of `u32` words per circular 16x16 HYB tile for `pair_bits` in {3, 4, 5, 6}.
pub const fn hyb_tile_words(pair_bits: u32) -> usize {
    (HYB_PAIRS_PER_TILE * pair_bits as usize) / 32
}

/// Positive-quadrant 2D Gaussian `half2` codebook of `2^q` entries (`q in 5..=9`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HybCodebook {
    pub q_bits: u32,
    pub lut_f16x2: Vec<u32>,
}

impl HybCodebook {
    /// Default Q=9 (512 entries, 2 KB) codebook.
    pub fn gaussian_default() -> Self {
        Self::gaussian_for_q(9)
    }

    /// Generates a deterministic, unit-variance 2D positive-quadrant Gaussian
    /// codebook of `2^q` entries using the R2 quasi-random low-discrepancy
    /// sequence mapped through the half-normal inverse CDF.
    pub fn gaussian_for_q(q_bits: u32) -> Self {
        assert!((5..=9).contains(&q_bits), "q_bits must be in 5..=9");
        let entries = 1usize << q_bits;
        // Plastic constant phi_2 ≈ 1.32471795724474602596 for R2 sequence.
        const A1: f64 = 0.754_877_666_246_692_7;
        const A2: f64 = 0.569_840_290_998_053_2;

        let mut pairs = vec![(0.0f64, 0.0f64); entries];
        let mut sum_sq = 0.0f64;
        for (i, slot) in pairs.iter_mut().enumerate() {
            let n = (i + 1) as f64;
            let u0 = (0.5 + n * A1).fract();
            let u1 = (0.5 + n * A2).fract();
            let p0 = 0.5 + 0.5 * (0.001 + 0.998 * u0);
            let p1 = 0.5 + 0.5 * (0.001 + 0.998 * u1);
            let x0 = norm_inv_cdf(p0);
            let x1 = norm_inv_cdf(p1);
            *slot = (x0, x1);
            sum_sq += x0 * x0 + x1 * x1;
        }
        let rms = (sum_sq / (2 * entries) as f64).sqrt();
        let mut lut = vec![0u32; entries];
        for (dst, &(x0, x1)) in lut.iter_mut().zip(&pairs) {
            let h0 = f16::from_f32((x0 / rms) as f32).to_bits() & 0x7FFF;
            let h1 = f16::from_f32((x1 / rms) as f32).to_bits() & 0x7FFF;
            *dst = u32::from(h0) | (u32::from(h1) << 16);
        }
        Self {
            q_bits,
            lut_f16x2: lut,
        }
    }

    /// Reconstructs the `HybCodebook` from raw `u32` words (e.g. loaded from PLT1).
    pub fn from_words(words: &[u32]) -> Result<Self> {
        let len = words.len();
        if !len.is_power_of_two() || !(32..=HYB_LUT_ENTRIES).contains(&len) {
            return Err(Error::Invalid(format!(
                "HYB codebook requires 32..={HYB_LUT_ENTRIES} power-of-two u32 entries, got {len}"
            )));
        }
        Ok(Self {
            q_bits: len.trailing_zeros(),
            lut_f16x2: words.to_vec(),
        })
    }

    /// Decodes a 16-bit trellis state into `(w0, w1)` bit-identically to CUDA.
    #[inline]
    pub fn decode_pair(&self, state: u32) -> (f32, f32) {
        let h = (state & 0xFFFF).wrapping_mul(HYB_HASH_MULT);
        let mask = (1u32 << self.q_bits) - 1;
        let idx = ((h >> 5) & mask) as usize;
        let raw = self.lut_f16x2[idx] ^ (h & 0x8000_8000);
        let w0 = f16::from_bits((raw & 0xFFFF) as u16).to_f32();
        let w1 = f16::from_bits((raw >> 16) as u16).to_f32();
        (w0, w1)
    }
}

/// Rational approximation to the standard normal inverse CDF (Acklam's algorithm),
/// accurate to ~1e-9 for p in (0, 1).
fn norm_inv_cdf(p: f64) -> f64 {
    const A: [f64; 6] = [
        -3.969_683_028_665_376e1,
        2.209_460_984_245_205e2,
        -2.759_285_104_469_687e2,
        1.383_577_518_672_690e2,
        -3.066_479_806_614_716e1,
        2.506_628_277_459_239e0,
    ];
    const B: [f64; 5] = [
        -5.447_609_879_822_406e1,
        1.615_858_368_580_409e2,
        -1.556_989_798_598_866e2,
        6.680_131_188_771_972e1,
        -1.328_068_155_288_572e1,
    ];
    const C: [f64; 6] = [
        -7.784_894_002_430_293e-3,
        -3.223_964_580_411_365e-1,
        -2.400_758_277_161_838e0,
        -2.549_732_539_343_734e0,
        4.374_664_141_464_968e0,
        2.938_163_982_698_783e0,
    ];
    const D: [f64; 4] = [
        7.784_695_709_041_462e-3,
        3.224_671_290_700_398e-1,
        2.445_134_137_142_996e0,
        3.754_408_661_907_416e0,
    ];
    const P_LOW: f64 = 0.02425;
    const P_HIGH: f64 = 1.0 - P_LOW;

    if p < P_LOW {
        let q = (-2.0 * p.ln()).sqrt();
        (((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    } else if p <= P_HIGH {
        let q = p - 0.5;
        let r = q * q;
        (((((A[0] * r + A[1]) * r + A[2]) * r + A[3]) * r + A[4]) * r + A[5]) * q
            / (((((B[0] * r + B[1]) * r + B[2]) * r + B[3]) * r + B[4]) * r + 1.0)
    } else {
        let q = (-2.0 * (1.0 - p).ln()).sqrt();
        -(((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    }
}

/// Extracts the `state_bits`-bit circular LSB-first state of pair `p` (`0..128`)
/// from a packed HYB tile of `W = hyb_tile_words(pair_bits)` words.
#[inline]
pub fn hyb_state_at_l(words: &[u32], p: usize, state_bits: u32, pair_bits: u32) -> u32 {
    let total_words = words.len();
    let bit = p * pair_bits as usize;
    let w0 = (bit / 32) % total_words;
    let w1 = (w0 + 1) % total_words;
    let shift = bit % 32;
    let combined = u64::from(words[w0]) | (u64::from(words[w1]) << 32);
    let mask = (1u64 << state_bits) - 1;
    ((combined >> shift) & mask) as u32
}

/// Extracts the 16-bit circular LSB-first state of pair `p` (`0..128`) from a
/// packed HYB tile of `W = hyb_tile_words(pair_bits)` words.
#[inline]
pub fn hyb_state_at(words: &[u32], p: usize, pair_bits: u32) -> u32 {
    hyb_state_at_l(words, p, HYB_STATE_BITS, pair_bits)
}

/// Permutes a row-major `16x16` tile (`256` weights) into exact
/// `mma.sync.aligned.m16n8k16` A-fragment order (128 pairs = 32 lanes * 4 pairs).
pub fn swizzle_tile_to_mma(tile_rm: &[f32]) -> [f32; TILE * TILE] {
    let mut out = [0.0f32; TILE * TILE];
    for lane in 0..32usize {
        let group_id = lane >> 2;
        let tid = lane & 3;
        let coords = [
            (group_id, 2 * tid),
            (group_id + 8, 2 * tid),
            (group_id, 2 * tid + 8),
            (group_id + 8, 2 * tid + 8),
        ];
        for (p, &(r, c)) in coords.iter().enumerate() {
            let dst = (lane * 4 + p) * 2;
            out[dst] = tile_rm[r * TILE + c];
            out[dst + 1] = tile_rm[r * TILE + c + 1];
        }
    }
    out
}

/// Decodes a 16x16 HYB tile stored in offline MMA A-fragment order
/// (`mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32`) into row-major `[16, 16]`.
pub fn decode_hyb_tile_mma_l(
    words: &[u32],
    state_bits: u32,
    pair_bits: u32,
    codebook: &HybCodebook,
    scale: f32,
) -> [f32; TILE * TILE] {
    let mut out = [0.0f32; TILE * TILE];
    for lane in 0..32usize {
        let group_id = lane >> 2;
        let tid = lane & 3;
        let coords = [
            (group_id, 2 * tid),
            (group_id + 8, 2 * tid),
            (group_id, 2 * tid + 8),
            (group_id + 8, 2 * tid + 8),
        ];
        for (p, &(r, c)) in coords.iter().enumerate() {
            let state = hyb_state_at_l(words, lane * 4 + p, state_bits, pair_bits);
            let (w0, w1) = codebook.decode_pair(state);
            out[r * TILE + c] = scale * w0;
            out[r * TILE + c + 1] = scale * w1;
        }
    }
    out
}

/// Decodes a 16x16 HYB tile stored in offline MMA A-fragment order (L=16).
pub fn decode_hyb_tile_mma(
    words: &[u32],
    pair_bits: u32,
    codebook: &HybCodebook,
    scale: f32,
) -> [f32; TILE * TILE] {
    decode_hyb_tile_mma_l(words, HYB_STATE_BITS, pair_bits, codebook, scale)
}

/// Decodes a packed matrix of MMA-swizzled HYB tiles into row-major `[rows, cols]`.
pub fn decode_hyb_matrix_mma(
    words: &[u32],
    rows: usize,
    cols: usize,
    pair_bits: u32,
    codebook: &HybCodebook,
    scale: f32,
) -> Vec<f32> {
    let wpt = hyb_tile_words(pair_bits);
    let col_tiles = cols / TILE;
    let mut out = vec![0.0f32; rows * cols];
    for (index, tile) in words.chunks_exact(wpt).enumerate() {
        let (group, block) = (index / col_tiles, index % col_tiles);
        let decoded = decode_hyb_tile_mma(tile, pair_bits, codebook, scale);
        for r in 0..TILE {
            let dst = (group * TILE + r) * cols + block * TILE;
            out[dst..dst + TILE].copy_from_slice(&decoded[r * TILE..(r + 1) * TILE]);
        }
    }
    out
}
/// Decodes a full 16x16 HYB tile (256 weights, row-major within tile).
pub fn decode_hyb_tile(
    words: &[u32],
    pair_bits: u32,
    codebook: &HybCodebook,
    scale: f32,
) -> [f32; TILE * TILE] {
    let mut out = [0.0f32; TILE * TILE];
    for p in 0..HYB_PAIRS_PER_TILE {
        let state = hyb_state_at(words, p, pair_bits);
        let (w0, w1) = codebook.decode_pair(state);
        out[2 * p] = scale * w0;
        out[2 * p + 1] = scale * w1;
    }
    out
}

/// Decodes a packed matrix of HYB tiles (tile order: `group * (cols/16) + block`).
pub fn decode_hyb_matrix(
    words: &[u32],
    rows: usize,
    cols: usize,
    pair_bits: u32,
    codebook: &HybCodebook,
    scale: f32,
) -> Vec<f32> {
    let wpt = hyb_tile_words(pair_bits);
    let col_tiles = cols / TILE;
    let mut out = vec![0.0f32; rows * cols];
    for (index, tile) in words.chunks_exact(wpt).enumerate() {
        let (group, block) = (index / col_tiles, index % col_tiles);
        let decoded = decode_hyb_tile(tile, pair_bits, codebook, scale);
        for r in 0..TILE {
            let dst = (group * TILE + r) * cols + block * TILE;
            out[dst..dst + TILE].copy_from_slice(&decoded[r * TILE..(r + 1) * TILE]);
        }
    }
    out
}

/// Precomputed table for fast repeated Viterbi quantization of HYB tiles
/// (`state_bits` in `8..=16`, `pair_bits` in `3..=6`).
pub struct HybQuantizer {
    pub state_bits: u32,
    pub pair_bits: u32,
    pub codebook: HybCodebook,
    vals: Vec<(f32, f32)>,
}

impl HybQuantizer {
    pub fn new(state_bits: u32, pair_bits: u32, codebook: HybCodebook) -> Result<Self> {
        if !(8..=HYB_STATE_BITS).contains(&state_bits) {
            return Err(Error::Invalid(format!(
                "HYB state_bits must be in 8..={HYB_STATE_BITS}, got {state_bits}"
            )));
        }
        if !(3..=6).contains(&pair_bits) || state_bits <= pair_bits {
            return Err(Error::Invalid(format!(
                "HYB pair_bits must be in 3..=6 and < state_bits, got {pair_bits}"
            )));
        }
        let states = 1usize << state_bits;
        let vals = (0..states)
            .map(|s| codebook.decode_pair(s as u32))
            .collect();
        Ok(Self {
            state_bits,
            pair_bits,
            codebook,
            vals,
        })
    }

    /// Quantizes a normalized `16x16` tile (already in pair order, e.g. via
    /// `swizzle_tile_to_mma`) using 2-pass circular tail-biting Viterbi.
    pub fn quantize_normalized(&self, weights: &[f32]) -> Result<Vec<u32>> {
        if weights.len() != TILE * TILE {
            return Err(Error::Invalid(format!(
                "HYB tile needs {} weights, got {}",
                TILE * TILE,
                weights.len()
            )));
        }
        let states = 1usize << self.state_bits;
        let fan = 1usize << self.pair_bits;
        let head_bits = self.state_bits - self.pair_bits;
        let heads = 1usize << head_bits;
        let head_mask = heads - 1;

        let run_viterbi = |pinned_wrap: Option<usize>| -> ([u32; HYB_PAIRS_PER_TILE], f32) {
            let mut cost: Vec<f32> = (0..states)
                .map(|s| {
                    if let Some(w) = pinned_wrap
                        && (s & head_mask) != w
                    {
                        return f32::INFINITY;
                    }
                    let (a, b) = self.vals[s];
                    let d0 = weights[0] - a;
                    let d1 = weights[1] - b;
                    d0 * d0 + d1 * d1
                })
                .collect();
            let mut next = vec![0.0f32; states];
            let mut head_cost = vec![0.0f32; heads];
            let mut back = vec![0u8; (HYB_PAIRS_PER_TILE - 1) * heads];

            for p in 1..HYB_PAIRS_PER_TILE {
                let w0 = weights[2 * p];
                let w1 = weights[2 * p + 1];
                let row = &mut back[(p - 1) * heads..p * heads];
                for (head, (best, low_out)) in head_cost.iter_mut().zip(row.iter_mut()).enumerate()
                {
                    let base = head << self.pair_bits;
                    let mut min_c = f32::INFINITY;
                    let mut min_low = 0u8;
                    for low in 0..fan {
                        let c = cost[base | low];
                        if c < min_c {
                            min_c = c;
                            min_low = low as u8;
                        }
                    }
                    *best = min_c;
                    *low_out = min_low;
                }
                for (s, slot) in next.iter_mut().enumerate() {
                    let (a, b) = self.vals[s];
                    let d0 = w0 - a;
                    let d1 = w1 - b;
                    *slot = head_cost[s & head_mask] + d0 * d0 + d1 * d1;
                }
                std::mem::swap(&mut cost, &mut next);
            }

            let mut best_s = 0usize;
            let mut best_c = f32::INFINITY;
            for (s, &c) in cost.iter().enumerate() {
                if let Some(w) = pinned_wrap
                    && (s >> self.pair_bits) != w
                {
                    continue;
                }
                if c < best_c {
                    best_c = c;
                    best_s = s;
                }
            }

            let mut st_out = [0u32; HYB_PAIRS_PER_TILE];
            let mut cur = best_s;
            for p in (1..HYB_PAIRS_PER_TILE).rev() {
                st_out[p] = cur as u32;
                let head = cur & head_mask;
                let low = usize::from(back[(p - 1) * heads + head]);
                cur = (head << self.pair_bits) | low;
            }
            st_out[0] = cur as u32;
            (st_out, best_c)
        };

        let (pass1_states, _) = run_viterbi(None);
        let wrap_bits = (pass1_states[HYB_PAIRS_PER_TILE - 1] as usize) >> self.pair_bits;
        let (st, _) = run_viterbi(Some(wrap_bits));

        let wpt = hyb_tile_words(self.pair_bits);
        let mut words = vec![0u32; wpt];
        let sym_mask = (1u32 << self.pair_bits) - 1;
        for (p, &s) in st.iter().enumerate() {
            let bits = s & sym_mask;
            let offset = p * self.pair_bits as usize;
            let w = offset / 32;
            let shift = offset % 32;
            words[w] |= bits << shift;
            if shift + self.pair_bits as usize > 32 {
                words[(w + 1) % wpt] |= bits >> (32 - shift);
            }
        }
        Ok(words)
    }
}

/// Viterbi quantizer for a 16x16 tile (`256` weights = `128` pairs) under the
/// `L=16, V=2` HYB bitshift trellis, packed into a circular `128 * pair_bits`-bit
/// buffer (`pair_bits` in {3, 4, 5, 6} -> 1.5, 2.0, 2.5, 3.0 bpw).
pub fn quantize_hyb_tile(
    weights: &[f32],
    pair_bits: u32,
    codebook: &HybCodebook,
    scale: f32,
) -> Result<Vec<u32>> {
    let q = HybQuantizer::new(HYB_STATE_BITS, pair_bits, codebook.clone())?;
    let norm: Vec<f32> = weights.iter().map(|&v| v / scale).collect();
    q.quantize_normalized(&norm)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_codebook_has_zero_mean_and_unit_variance() {
        let cb = HybCodebook::gaussian_default();
        let mut sum = 0.0f64;
        let mut sum_sq = 0.0f64;
        let count = (1usize << HYB_STATE_BITS) * 2;
        for s in 0..(1u32 << HYB_STATE_BITS) {
            let (a, b) = cb.decode_pair(s);
            sum += f64::from(a) + f64::from(b);
            sum_sq += f64::from(a * a) + f64::from(b * b);
        }
        let mean = sum / count as f64;
        let var = sum_sq / count as f64 - mean * mean;
        assert!(mean.abs() < 0.01, "mean {mean}");
        assert!((var - 1.0).abs() < 0.02, "var {var}");
    }

    #[test]
    fn hyb_tile_words_match_expected_cache_friendly_sizes() {
        assert_eq!(hyb_tile_words(3), 12); // 48 bytes (1.5 bpw)
        assert_eq!(hyb_tile_words(4), 16); // 64 bytes (2.0 bpw)
        assert_eq!(hyb_tile_words(5), 20); // 80 bytes (2.5 bpw)
        assert_eq!(hyb_tile_words(6), 24); // 96 bytes (3.0 bpw)
    }

    #[test]
    fn hyb_gaussian_mse_beats_scalar_2bit() {
        // Generate 8 synthetic N(0,1) tiles (2048 weights) via Box-Muller.
        let cb = HybCodebook::gaussian_default();
        let mut state = 0x1234_5678_9ABC_DEF0u64;
        let mut next_u = || -> f64 {
            state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = state;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            ((z ^ (z >> 31)) >> 11) as f64 / ((1u64 << 53) as f64)
        };
        let mut total_se_2b = 0.0f64;
        let mut total_se_3b = 0.0f64;
        let mut total_e = 0.0f64;
        for _ in 0..8 {
            let mut tile = [0.0f32; TILE * TILE];
            for chunk in tile.chunks_exact_mut(2) {
                let u1 = next_u().max(1e-12);
                let u2 = next_u();
                let r = (-2.0 * u1.ln()).sqrt();
                let theta = std::f64::consts::TAU * u2;
                chunk[0] = (r * theta.cos()) as f32;
                chunk[1] = (r * theta.sin()) as f32;
            }
            let w2 = quantize_hyb_tile(&tile, 4, &cb, 1.0).unwrap();
            let d2 = decode_hyb_tile(&w2, 4, &cb, 1.0);
            let w3 = quantize_hyb_tile(&tile, 6, &cb, 1.0).unwrap();
            let d3 = decode_hyb_tile(&w3, 6, &cb, 1.0);
            for i in 0..TILE * TILE {
                total_e += f64::from(tile[i] * tile[i]);
                total_se_2b += f64::from((tile[i] - d2[i]).powi(2));
                total_se_3b += f64::from((tile[i] - d3[i]).powi(2));
            }
        }
        let mse_2b = total_se_2b / total_e;
        let mse_3b = total_se_3b / total_e;
        // 2-bit scalar Lloyd-Max is 0.1175; HYB L16 V2 must beat it cleanly.
        assert!(mse_2b < 0.09, "HYB 2-bit MSE {mse_2b:.4} too high");
        assert!(mse_3b < 0.03, "HYB 3-bit MSE {mse_3b:.4} too high");
    }
}
