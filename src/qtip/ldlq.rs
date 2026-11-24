//! BlockLDLQ (QuIP/QTIP): Hessian-aware rounding with error feedback.
//!
//! The proxy loss of a reconstruction `W_hat` of a `rows x cols` matrix `W`
//! whose inputs have second moment `H` is `tr(E H E^T)`, `E = W_hat - W`.
//! Write `H = U D U^T` with `U` unit block-upper-triangular (identity
//! diagonal blocks of `TILE` columns) and `D` block-diagonal. Quantizing the
//! column blocks in order as
//!
//!     W_hat_k = Q(W_k + (W - W_hat)_{<k} U_{<k,k})
//!
//! makes the loss `sum_k tr(e_k D_k e_k^T)`, where `e_k` is only the error of
//! block `k`'s own quantizer: earlier errors are compensated, not added.
//! Each block is cut into `TILE x TILE` tiles, which are the 256-weight
//! sequences that the trellis encodes.

use crate::{Error, Result};
use rayon::prelude::*;

/// Tile side: a tile of `TILE x TILE` weights is one trellis sequence.
pub const TILE: usize = 16;
/// Column block of the blocked Cholesky.
const PANEL: usize = 64;

/// Dot product with independent accumulators, so it vectorizes.
fn dot<T: Copy + Into<f64>>(a: &[T], b: &[T]) -> f64 {
    let mut lanes = [0.0f64; 8];
    let (chunks_a, rest_a) = a.split_at(a.len() / 8 * 8);
    let (chunks_b, rest_b) = b.split_at(chunks_a.len());
    for (x, y) in chunks_a.chunks_exact(8).zip(chunks_b.chunks_exact(8)) {
        for lane in 0..8 {
            lanes[lane] += x[lane].into() * y[lane].into();
        }
    }
    let tail: f64 = rest_a
        .iter()
        .zip(rest_b)
        .map(|(&x, &y)| x.into() * y.into())
        .sum();
    lanes.iter().sum::<f64>() + tail
}

/// f32 dot product with f32 accumulators (the feedback hot loop).
fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
    let mut lanes = [0.0f32; 16];
    let (chunks_a, rest_a) = a.split_at(a.len() / 16 * 16);
    let (chunks_b, rest_b) = b.split_at(chunks_a.len());
    for (x, y) in chunks_a.chunks_exact(16).zip(chunks_b.chunks_exact(16)) {
        for lane in 0..16 {
            lanes[lane] += x[lane] * y[lane];
        }
    }
    let tail: f32 = rest_a.iter().zip(rest_b).map(|(x, y)| x * y).sum();
    lanes.iter().sum::<f32>() + tail
}

/// In-place blocked Cholesky of a row-major `n x n` SPD matrix: on return
/// the lower triangle holds `L` with `A = L L^T` (the upper part is stale).
fn cholesky_lower(a: &mut [f64], n: usize) -> Result<()> {
    for k0 in (0..n).step_by(PANEL) {
        let k1 = (k0 + PANEL).min(n);
        for j in k0..k1 {
            let pivot = a[j * n + j] - dot(&a[j * n + k0..j * n + j], &a[j * n + k0..j * n + j]);
            if pivot.is_nan() || pivot <= 0.0 {
                return Err(Error::Invalid(format!(
                    "Hessian is not positive definite (pivot {j})"
                )));
            }
            let pivot = pivot.sqrt();
            a[j * n + j] = pivot;
            for i in j + 1..k1 {
                let sum = dot(&a[i * n + k0..i * n + j], &a[j * n + k0..j * n + j]);
                a[i * n + j] = (a[i * n + j] - sum) / pivot;
            }
        }
        let (top, bottom) = a.split_at_mut(k1 * n);
        // Panel: L[i, k0..k1] = A[i, k0..k1] L_kk^{-T} for every row below.
        bottom.par_chunks_mut(n).for_each(|row| {
            for j in k0..k1 {
                let sum = dot(&row[k0..j], &top[j * n + k0..j * n + j]);
                row[j] = (row[j] - sum) / top[j * n + j];
            }
        });
        let width = k1 - k0;
        let panel: Vec<f64> = bottom
            .chunks_exact(n)
            .flat_map(|row| row[k0..k1].iter().copied())
            .collect();
        // Trailing update of the lower triangle: A[i, j] -= L[i, panel] . L[j, panel].
        bottom
            .par_chunks_mut(n)
            .enumerate()
            .for_each(|(local, row)| {
                let mine = &panel[local * width..(local + 1) * width];
                for other in 0..=local {
                    row[k1 + other] -= dot(mine, &panel[other * width..(other + 1) * width]);
                }
            });
    }
    Ok(())
}

/// Upper-triangular `R` (row-major) with `H = R R^T`: the Cholesky factor of
/// `H` with its index order reversed.
pub fn cholesky_upper(h: &[f64], n: usize) -> Result<Vec<f64>> {
    if h.len() != n * n {
        return Err(Error::Invalid(format!(
            "Hessian has {} entries, expected {n}^2",
            h.len()
        )));
    }
    let mut reversed: Vec<f64> = (0..n * n).map(|k| h[n * n - 1 - k]).collect();
    cholesky_lower(&mut reversed, n)?;
    let mut upper = vec![0.0; n * n];
    upper.par_chunks_mut(n).enumerate().for_each(|(i, row)| {
        for j in i..n {
            row[j] = reversed[(n - 1 - i) * n + (n - 1 - j)];
        }
    });
    Ok(upper)
}

/// Inverse of an upper-triangular `b x b` matrix (row-major).
fn invert_upper(r: &[f64], b: usize) -> Vec<f64> {
    let mut inverse = vec![0.0; b * b];
    for j in (0..b).rev() {
        inverse[j * b + j] = 1.0 / r[j * b + j];
        for i in (0..j).rev() {
            let sum: f64 = (i + 1..=j).map(|k| r[i * b + k] * inverse[k * b + j]).sum();
            inverse[i * b + j] = -sum / r[i * b + i];
        }
    }
    inverse
}

/// `U = R B^{-1}`, with `B` the block diagonal of `R` (blocks of `TILE`).
/// Then `H = U D U^T` with `D = B B^T` block-diagonal and identity diagonal
/// blocks in `U`.
pub fn block_feedback(r: &[f64], n: usize) -> Result<Vec<f64>> {
    if !n.is_multiple_of(TILE) || r.len() != n * n {
        return Err(Error::Invalid(format!(
            "LDLQ needs a multiple of {TILE} columns, got {n}"
        )));
    }
    let inverses: Vec<Vec<f64>> = (0..n / TILE)
        .map(|k| {
            let block: Vec<f64> = (0..TILE * TILE)
                .map(|e| r[(k * TILE + e / TILE) * n + k * TILE + e % TILE])
                .collect();
            invert_upper(&block, TILE)
        })
        .collect();
    let mut u = vec![0.0; n * n];
    u.par_chunks_mut(n).enumerate().for_each(|(i, row)| {
        for (k, inverse) in inverses.iter().enumerate().skip(i / TILE) {
            for c in 0..TILE {
                row[k * TILE + c] = (0..TILE)
                    .map(|p| r[i * n + k * TILE + p] * inverse[p * TILE + c])
                    .sum();
            }
        }
    });
    Ok(u)
}

/// BlockLDLQ of a row-major `rows x cols` matrix. `u` is the `cols x cols`
/// output of [`block_feedback`]; `quantize_tile` maps one row-major
/// `TILE x TILE` tile to its reconstruction.
pub fn ldlq(
    w: &[f32],
    rows: usize,
    cols: usize,
    u: &[f64],
    quantize_tile: impl Fn(&[f32]) -> Vec<f32> + Sync,
) -> Result<Vec<f32>> {
    if !rows.is_multiple_of(TILE)
        || !cols.is_multiple_of(TILE)
        || w.len() != rows * cols
        || u.len() != cols * cols
    {
        return Err(Error::Invalid(format!(
            "LDLQ needs {TILE}-aligned shapes, got {rows} x {cols}"
        )));
    }
    let mut w_hat = vec![0.0f32; rows * cols];
    // W - W_hat; columns at or after the current block are still zero.
    let mut error = vec![0.0f32; rows * cols];
    for start in (0..cols).step_by(TILE) {
        // coefficients[c * start + j] = U[j, start + c] for the earlier columns j.
        let coefficients: Vec<f32> = (0..TILE)
            .flat_map(|c| (0..start).map(move |j| u[j * cols + start + c] as f32))
            .collect();
        w_hat
            .par_chunks_mut(TILE * cols)
            .zip(error.par_chunks_mut(TILE * cols))
            .enumerate()
            .for_each(|(group, (hat_rows, error_rows))| {
                let mut tile = [0.0f32; TILE * TILE];
                for r in 0..TILE {
                    let row = (group * TILE + r) * cols;
                    let earlier = &error_rows[r * cols..r * cols + start];
                    for c in 0..TILE {
                        let feedback = dot_f32(earlier, &coefficients[c * start..(c + 1) * start]);
                        tile[r * TILE + c] = w[row + start + c] + feedback;
                    }
                }
                let quantized = quantize_tile(&tile);
                assert_eq!(quantized.len(), TILE * TILE, "tile quantizer output");
                for r in 0..TILE {
                    let row = (group * TILE + r) * cols;
                    for c in 0..TILE {
                        let q = quantized[r * TILE + c];
                        hat_rows[r * cols + start + c] = q;
                        error_rows[r * cols + start + c] = w[row + start + c] - q;
                    }
                }
            });
    }
    Ok(w_hat)
}

/// Relative proxy loss `tr(E H E^T) / tr(W H W^T)`.
pub fn proxy_error(w: &[f32], w_hat: &[f32], rows: usize, cols: usize, h: &[f64]) -> f64 {
    let quadratic = |v: &[f64]| -> f64 {
        h.chunks_exact(cols)
            .zip(v)
            .map(|(h_row, &vi)| vi * dot(h_row, v))
            .sum()
    };
    let (error, energy) = (0..rows)
        .into_par_iter()
        .map(|r| {
            let row = &w[r * cols..(r + 1) * cols];
            let hat = &w_hat[r * cols..(r + 1) * cols];
            let e: Vec<f64> = row
                .iter()
                .zip(hat)
                .map(|(&a, &b)| f64::from(b - a))
                .collect();
            let x: Vec<f64> = row.iter().map(|&a| f64::from(a)).collect();
            (quadratic(&e), quadratic(&x))
        })
        .reduce(|| (0.0, 0.0), |a, b| (a.0 + b.0, a.1 + b.1));
    error / energy
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic values in [-1, 1).
    fn uniform(count: usize, seed: u64) -> Vec<f64> {
        let mut state = seed;
        (0..count)
            .map(|_| {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((state >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
            })
            .collect()
    }

    /// `A A^T + damping I` with strongly correlated columns.
    fn spd(n: usize, seed: u64, damping: f64) -> Vec<f64> {
        let a = uniform(n * n, seed);
        let mut h = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..n {
                h[i * n + j] = (0..n).map(|k| a[i * n + k] * a[j * n + k]).sum::<f64>()
                    + 3.0
                        * (0..n).map(|k| a[i * n + k]).sum::<f64>()
                        * (0..n).map(|k| a[j * n + k]).sum::<f64>()
                        / n as f64;
            }
            h[i * n + i] += damping;
        }
        h
    }

    fn matmul(a: &[f64], b: &[f64], n: usize) -> Vec<f64> {
        let mut c = vec![0.0; n * n];
        for i in 0..n {
            for k in 0..n {
                for j in 0..n {
                    c[i * n + j] += a[i * n + k] * b[k * n + j];
                }
            }
        }
        c
    }

    fn transpose(a: &[f64], n: usize) -> Vec<f64> {
        (0..n * n).map(|e| a[(e % n) * n + e / n]).collect()
    }

    fn max_diff(a: &[f64], b: &[f64]) -> f64 {
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0, f64::max)
    }

    /// Rounds to a grid of step 0.5 (a stand-in for the trellis).
    fn round_tile(tile: &[f32]) -> Vec<f32> {
        tile.iter().map(|v| (v * 2.0).round() / 2.0).collect()
    }

    #[test]
    fn cholesky_upper_reconstructs_the_hessian() {
        // 150 is not a multiple of the panel, so the edge path runs too.
        let n = 150;
        let h = spd(n, 3, 0.5);
        let r = cholesky_upper(&h, n).unwrap();
        assert!(
            (0..n).all(|i| (0..i).all(|j| r[i * n + j] == 0.0)),
            "R is upper"
        );
        let product = matmul(&r, &transpose(&r, n), n);
        assert!(max_diff(&product, &h) < 1e-8 * max_diff(&h, &vec![0.0; n * n]));
    }

    #[test]
    fn cholesky_rejects_an_indefinite_matrix() {
        let mut h = vec![0.0; 4];
        h[0] = 1.0;
        h[3] = -1.0;
        assert!(cholesky_upper(&h, 2).is_err());
    }

    #[test]
    fn block_feedback_factors_the_hessian_with_block_diagonal_d() {
        let n = 4 * TILE;
        let h = spd(n, 5, 0.5);
        let r = cholesky_upper(&h, n).unwrap();
        let u = block_feedback(&r, n).unwrap();
        for i in 0..n {
            for j in 0..n {
                let expected = if i / TILE == j / TILE {
                    f64::from(u8::from(i == j))
                } else {
                    u[i * n + j]
                };
                assert!(
                    (u[i * n + j] - expected).abs() < 1e-10,
                    "diagonal block at {i},{j}"
                );
                if i / TILE > j / TILE {
                    assert_eq!(u[i * n + j], 0.0, "U is block-upper");
                }
            }
        }
        // D = B B^T with B the block diagonal of R.
        let block_r: Vec<f64> = (0..n * n)
            .map(|e| {
                if e / n / TILE == e % n / TILE {
                    r[e]
                } else {
                    0.0
                }
            })
            .collect();
        let d = matmul(&block_r, &transpose(&block_r, n), n);
        let rebuilt = matmul(&matmul(&u, &d, n), &transpose(&u, n), n);
        assert!(max_diff(&rebuilt, &h) < 1e-8 * max_diff(&h, &vec![0.0; n * n]));
    }

    #[test]
    fn identity_hessian_quantizes_each_tile_independently() {
        let (rows, cols) = (2 * TILE, 3 * TILE);
        let w: Vec<f32> = uniform(rows * cols, 7)
            .iter()
            .map(|&v| v as f32 * 2.0)
            .collect();
        let identity: Vec<f64> = (0..cols * cols)
            .map(|e| f64::from(u8::from(e / cols == e % cols)))
            .collect();
        let u = block_feedback(&cholesky_upper(&identity, cols).unwrap(), cols).unwrap();
        let got = ldlq(&w, rows, cols, &u, round_tile).unwrap();
        let direct: Vec<f32> = w.iter().map(|v| (v * 2.0).round() / 2.0).collect();
        assert_eq!(got, direct);
    }

    #[test]
    fn feedback_lowers_the_proxy_loss_on_a_correlated_hessian() {
        let (rows, cols) = (4 * TILE, 8 * TILE);
        let h = spd(cols, 11, 0.05);
        let w: Vec<f32> = uniform(rows * cols, 13)
            .iter()
            .map(|&v| v as f32 * 1.5)
            .collect();
        let identity: Vec<f64> = (0..cols * cols)
            .map(|e| f64::from(u8::from(e / cols == e % cols)))
            .collect();
        let plain = ldlq(
            &w,
            rows,
            cols,
            &block_feedback(&cholesky_upper(&identity, cols).unwrap(), cols).unwrap(),
            round_tile,
        )
        .unwrap();
        let u = block_feedback(&cholesky_upper(&h, cols).unwrap(), cols).unwrap();
        let fed_back = ldlq(&w, rows, cols, &u, round_tile).unwrap();
        let (before, after) = (
            proxy_error(&w, &plain, rows, cols, &h),
            proxy_error(&w, &fed_back, rows, cols, &h),
        );
        assert!(after < 0.7 * before, "proxy error {before} -> {after}");
        assert_eq!(proxy_error(&w, &w, rows, cols, &h), 0.0);
    }

    #[test]
    fn rejects_unaligned_shapes() {
        let u = vec![0.0; 17 * 17];
        assert!(ldlq(&vec![0.0; 17 * 16], 16, 17, &u, round_tile).is_err());
        assert!(block_feedback(&u, 17).is_err());
    }
}
