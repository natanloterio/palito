//! Incoherence processing: a two-sided random Hadamard transform (RHT).
//!
//! `W' = H_r S_r W S_c H_c`, with `S` random sign diagonals and `H` normalized
//! Walsh-Hadamard matrices. Both factors are orthogonal, so the transform
//! preserves the Frobenius norm (and hence squared quantization error) while
//! spreading outliers until the entries look Gaussian.

use crate::{Error, Result};
use rayon::prelude::*;
use std::ops::{Add, Mul, Sub};

/// In-place normalized fast Walsh-Hadamard transform. `values.len()` must be
/// a power of two. The normalized transform is its own inverse.
fn fwht<T>(values: &mut [T])
where
    T: Copy + Add<Output = T> + Sub<Output = T> + Mul<Output = T> + From<f32>,
{
    let n = values.len();
    let mut span = 1;
    while span < n {
        for start in (0..n).step_by(2 * span) {
            for i in start..start + span {
                let (a, b) = (values[i], values[i + span]);
                values[i] = a + b;
                values[i + span] = a - b;
            }
        }
        span *= 2;
    }
    let norm = T::from((n as f32).sqrt().recip());
    values.iter_mut().for_each(|v| *v = *v * norm);
}

/// `fwht` on consecutive chunks of `block` values: `I (x) H_block`.
fn blocked_fwht<T>(values: &mut [T], block: usize)
where
    T: Copy + Add<Output = T> + Sub<Output = T> + Mul<Output = T> + From<f32>,
{
    values.chunks_exact_mut(block).for_each(fwht);
}

/// Deterministic random signs (SplitMix64), so encoder and decoder agree on a seed.
fn signs(count: usize, seed: u64) -> Vec<f32> {
    let mut state = seed;
    (0..count)
        .map(|_| {
            state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = state;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            if (z ^ (z >> 31)) & 1 == 0 { 1.0 } else { -1.0 }
        })
        .collect()
}

/// A two-sided RHT for a row-major `rows x cols` matrix. The Hadamard factors
/// may be block-diagonal (`I (x) H_b`), which a GPU kernel can apply inside
/// its own tile of the product instead of in separate passes.
pub struct Incoherence {
    rows: usize,
    cols: usize,
    row_block: usize,
    col_block: usize,
    row_signs: Vec<f32>,
    col_signs: Vec<f32>,
}

impl Incoherence {
    pub fn new(rows: usize, cols: usize, seed: u64) -> Result<Self> {
        Self::with_block(rows, cols, seed, None)
    }

    /// `block`: Hadamard block size (a power of two of at least 2), capped at
    /// each dimension; `None` transforms whole rows and columns. Each
    /// dimension must be a power of two or a multiple of the block.
    pub fn with_block(rows: usize, cols: usize, seed: u64, block: Option<usize>) -> Result<Self> {
        if let Some(b) = block
            && (b < 2 || !b.is_power_of_two())
        {
            return Err(Error::Invalid(format!(
                "Hadamard block must be a power of two >= 2, got {b}"
            )));
        }
        let block = block.unwrap_or(usize::MAX);
        let (row_block, col_block) = (block.min(rows), block.min(cols));
        let tiles = |dim: usize, b: usize| b.is_power_of_two() && dim.is_multiple_of(b);
        if !tiles(rows, row_block) || !tiles(cols, col_block) {
            return Err(Error::Invalid(format!(
                "Hadamard incoherence needs power-of-two dimensions or multiples of \
                 the block, got {rows} x {cols}"
            )));
        }
        Ok(Self {
            rows,
            cols,
            row_block,
            col_block,
            row_signs: signs(rows, seed),
            col_signs: signs(cols, seed ^ 0xA5A5_A5A5_A5A5_A5A5),
        })
    }

    /// `S_r` as `+-1` values.
    pub fn row_signs(&self) -> &[f32] {
        &self.row_signs
    }

    /// `S_c` as `+-1` values.
    pub fn col_signs(&self) -> &[f32] {
        &self.col_signs
    }

    fn check(&self, matrix: &[f32]) {
        assert_eq!(matrix.len(), self.rows * self.cols, "matrix shape");
    }

    /// `M -> S_r M S_c`.
    fn apply_signs(&self, matrix: &mut [f32]) {
        for (row, &row_sign) in matrix.chunks_exact_mut(self.cols).zip(&self.row_signs) {
            row.iter_mut()
                .zip(&self.col_signs)
                .for_each(|(v, &col_sign)| *v *= row_sign * col_sign);
        }
    }

    /// `M -> H_r M H_c`. Left and right multiplications commute.
    fn apply_hadamards(&self, matrix: &mut [f32]) {
        matrix
            .chunks_exact_mut(self.cols)
            .for_each(|row| blocked_fwht(row, self.col_block));
        let mut column = vec![0.0; self.rows];
        for c in 0..self.cols {
            column
                .iter_mut()
                .enumerate()
                .for_each(|(r, v)| *v = matrix[r * self.cols + c]);
            blocked_fwht(&mut column, self.row_block);
            column
                .iter()
                .enumerate()
                .for_each(|(r, &v)| matrix[r * self.cols + c] = v);
        }
    }

    /// `W' = H_r S_r W S_c H_c`.
    pub fn forward(&self, matrix: &[f32]) -> Vec<f32> {
        self.check(matrix);
        let mut out = matrix.to_vec();
        self.apply_signs(&mut out);
        self.apply_hadamards(&mut out);
        out
    }

    /// The input Hessian seen by the rotated matrix. Inputs become
    /// `X' = X S_c H_c`, so `H' = H_c S_c H S_c H_c` (`cols x cols`).
    pub fn transform_hessian(&self, h: &[f64]) -> Vec<f64> {
        let n = self.cols;
        assert_eq!(h.len(), n * n, "Hessian shape");
        // Right-multiplies every row by S_c H_c.
        let right = |m: &mut [f64]| {
            m.par_chunks_mut(n).for_each(|row| {
                row.iter_mut()
                    .zip(&self.col_signs)
                    .for_each(|(v, &s)| *v *= f64::from(s));
                blocked_fwht(row, self.col_block);
            });
        };
        let mut a = h.to_vec();
        right(&mut a);
        // H' = H_c S_c A = (A^T S_c H_c)^T, and H' is symmetric.
        let mut t: Vec<f64> = (0..n * n).map(|e| a[(e % n) * n + e / n]).collect();
        right(&mut t);
        t
    }

    /// `W = S_r H_r W' H_c S_c`: undoes `forward` (both factors are symmetric
    /// and orthogonal, so each is its own inverse).
    pub fn inverse(&self, matrix: &[f32]) -> Vec<f32> {
        self.check(matrix);
        let mut out = matrix.to_vec();
        self.apply_hadamards(&mut out);
        self.apply_signs(&mut out);
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn matrix(rows: usize, cols: usize) -> Vec<f32> {
        (0..rows * cols)
            .map(|i| ((i * 37 % 101) as f32 - 50.0) / 25.0)
            .collect()
    }

    fn norm(values: &[f32]) -> f64 {
        values.iter().map(|&v| f64::from(v).powi(2)).sum()
    }

    #[test]
    fn hadamard_is_its_own_inverse_and_keeps_the_norm() {
        let x: Vec<f32> = (0..64).map(|i| (i as f32 * 0.7).sin()).collect();
        let mut y = x.clone();
        fwht(&mut y);
        assert!((norm(&x) - norm(&y)).abs() < 1e-4);
        fwht(&mut y);
        assert!(x.iter().zip(&y).all(|(a, b)| (a - b).abs() < 1e-5));
    }

    #[test]
    fn inverse_undoes_forward() {
        let w = matrix(16, 32);
        let rht = Incoherence::new(16, 32, 5).unwrap();
        let rotated = rht.forward(&w);
        assert!((norm(&w) - norm(&rotated)).abs() / norm(&w) < 1e-5);
        assert_ne!(rotated, w);
        let back = rht.inverse(&rotated);
        assert!(w.iter().zip(&back).all(|(a, b)| (a - b).abs() < 1e-4));
    }

    #[test]
    fn spreads_a_single_outlier() {
        let (rows, cols) = (64, 64);
        let mut w = vec![0.01f32; rows * cols];
        w[123] = 100.0;
        let rotated = Incoherence::new(rows, cols, 9).unwrap().forward(&w);
        let peak = rotated.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        let rms = (norm(&rotated) / rotated.len() as f64).sqrt() as f32;
        assert!(peak / rms < 1.2, "peak/rms {}", peak / rms);
    }

    #[test]
    fn transformed_hessian_keeps_the_proxy_loss() {
        let (rows, cols) = (8, 16);
        let e = matrix(rows, cols);
        let x: Vec<f64> = (0..cols * 40)
            .map(|i| ((i * 53 % 97) as f64 - 48.0) / 30.0)
            .collect();
        let mut h = vec![0.0; cols * cols];
        for sample in x.chunks_exact(cols) {
            for i in 0..cols {
                for j in 0..cols {
                    h[i * cols + j] += sample[i] * sample[j];
                }
            }
        }
        let loss = |e: &[f32], h: &[f64]| -> f64 {
            e.chunks_exact(cols)
                .map(|row| {
                    (0..cols)
                        .map(|i| {
                            (0..cols)
                                .map(|j| f64::from(row[i]) * h[i * cols + j] * f64::from(row[j]))
                                .sum::<f64>()
                        })
                        .sum::<f64>()
                })
                .sum()
        };
        let rht = Incoherence::new(rows, cols, 3).unwrap();
        let rotated_h = rht.transform_hessian(&h);
        let (before, after) = (loss(&e, &h), loss(&rht.forward(&e), &rotated_h));
        assert!(
            (before - after).abs() < 1e-4 * before,
            "{before} vs {after}"
        );
        assert!((0..cols).all(|i| {
            (0..cols).all(|j| (rotated_h[i * cols + j] - rotated_h[j * cols + i]).abs() < 1e-9)
        }));
    }

    #[test]
    fn blocked_transform_inverts_keeps_the_proxy_loss_and_stays_in_blocks() {
        let (rows, cols) = (8, 16);
        let w = matrix(rows, cols);
        let rht = Incoherence::with_block(rows, cols, 3, Some(4)).unwrap();
        let back = rht.inverse(&rht.forward(&w));
        assert!(w.iter().zip(&back).all(|(a, b)| (a - b).abs() < 1e-4));
        // A unit vector in column 5 only spreads within columns 4..8.
        let mut e = vec![0.0f64; cols * cols];
        e[5 * cols + 5] = 1.0;
        let h = rht.transform_hessian(&e);
        for i in 0..cols {
            for j in 0..cols {
                if h[i * cols + j] != 0.0 {
                    assert!((4..8).contains(&i) && (4..8).contains(&j), "({i}, {j})");
                }
            }
        }
        assert_ne!(
            rht.forward(&w),
            Incoherence::new(rows, cols, 3).unwrap().forward(&w)
        );
        assert!(Incoherence::with_block(rows, cols, 3, Some(3)).is_err());
        // A block of at least the dimension is the full transform.
        assert_eq!(
            Incoherence::with_block(rows, cols, 3, Some(64))
                .unwrap()
                .forward(&w),
            Incoherence::new(rows, cols, 3).unwrap().forward(&w)
        );
    }

    #[test]
    fn rejects_non_power_of_two_shapes() {
        assert!(Incoherence::new(12, 16, 1).is_err());
        assert!(Incoherence::new(16, 24, 1).is_err());
        // Blocks must tile each dimension exactly.
        assert!(Incoherence::with_block(16, 24, 1, Some(16)).is_err());
        assert!(Incoherence::with_block(12, 16, 1, Some(8)).is_err());
    }

    #[test]
    fn blocks_allow_dimensions_that_are_multiples_of_the_block() {
        // Like ffn_down of Llama 3.1 8B: 14336 = 56 x 256 columns.
        let (rows, cols) = (16, 24);
        let rht = Incoherence::with_block(rows, cols, 5, Some(8)).unwrap();
        let w = matrix(rows, cols);
        let rotated = rht.forward(&w);
        assert!((norm(&rotated) - norm(&w)).abs() < 1e-3 * norm(&w));
        let back = rht.inverse(&rotated);
        let err: f64 = w
            .iter()
            .zip(&back)
            .map(|(a, b)| f64::from(a - b).powi(2))
            .sum();
        assert!(err < 1e-8 * norm(&w), "round trip error {err}");
    }
}
