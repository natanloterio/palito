//! Whole-matrix quantization: normalize by the tensor RMS, then quantize
//! fixed-size blocks in parallel and rescale.

use super::{scalar::lloyd_max_2bit, trellis::Trellis};
use rayon::prelude::*;

/// Distortion statistics of a reconstruction.
#[derive(Clone, Copy, Debug)]
pub struct Distortion {
    /// `||W - W_hat||^2 / ||W||^2`.
    pub relative: f64,
}

pub fn distortion(original: &[f32], reconstructed: &[f32]) -> Distortion {
    let (error, energy) = original
        .par_iter()
        .zip(reconstructed)
        .map(|(&w, &q)| (f64::from(w - q).powi(2), f64::from(w).powi(2)))
        .reduce(|| (0.0, 0.0), |a, b| (a.0 + b.0, a.1 + b.1));
    Distortion {
        relative: error / energy,
    }
}

pub fn rms(values: &[f32]) -> f32 {
    let energy: f64 = values.par_iter().map(|&v| f64::from(v).powi(2)).sum();
    (energy / values.len() as f64).sqrt() as f32
}

/// Excess kurtosis: 0 for a Gaussian, large for heavy-tailed weights.
pub fn excess_kurtosis(values: &[f32]) -> f64 {
    let n = values.len() as f64;
    let mean = values.par_iter().map(|&v| f64::from(v)).sum::<f64>() / n;
    let (m2, m4) = values
        .par_iter()
        .map(|&v| {
            let d = f64::from(v) - mean;
            (d * d, d.powi(4))
        })
        .reduce(|| (0.0, 0.0), |a, b| (a.0 + b.0, a.1 + b.1));
    (m4 / n) / (m2 / n).powi(2) - 3.0
}

/// Trellis-quantizes `values` in blocks of `block` weights.
pub fn trellis_reconstruct(values: &[f32], trellis: &Trellis, block: usize) -> Vec<f32> {
    let sigma = rms(values);
    values
        .par_chunks(block)
        .flat_map_iter(|chunk| {
            let normalized: Vec<f32> = chunk.iter().map(|&v| v / sigma).collect();
            let encoded = trellis.quantize(&normalized);
            trellis.decode(&encoded).into_iter().map(move |v| v * sigma)
        })
        .collect()
}

/// The 2-bit Lloyd-Max baseline with the same per-tensor RMS scale.
pub fn scalar_reconstruct(values: &[f32]) -> Vec<f32> {
    let sigma = rms(values);
    values
        .par_iter()
        .map(|&v| lloyd_max_2bit(v / sigma) * sigma)
        .collect()
}
