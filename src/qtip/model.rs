//! Whole-model quantization: writes a copy of an F16 GGUF whose attention and
//! MLP matrices hold the quantized-then-reconstructed weights. Every backend
//! can then measure the quantization's effect on the model without a new
//! tensor type. The file keeps the F16 size; it is a quality probe, not a
//! compressed format.

use super::{
    hessian::Hessians,
    incoherence::Incoherence,
    ldlq::{block_feedback, cholesky_upper, ldlq, proxy_error},
    matrix::{distortion, rms, scalar_reconstruct, trellis_reconstruct},
    trellis::Trellis,
};
use crate::{
    Error, Result,
    gguf::{Gguf, Tensor},
};
use half::f16;
use std::{
    fs::{self, OpenOptions},
    io::{Seek, SeekFrom, Write},
    path::Path,
};

const F16_KIND: u32 = 1;
const BLOCK: usize = 256;
const QUANTIZED_SUFFIXES: [&str; 7] = [
    "attn_q.weight",
    "attn_k.weight",
    "attn_v.weight",
    "attn_output.weight",
    "ffn_gate.weight",
    "ffn_up.weight",
    "ffn_down.weight",
];

#[derive(Clone, Copy, Debug)]
pub enum Method {
    /// Random Hadamard incoherence, then a bitshift trellis.
    Trellis { state_bits: u32, symbol_bits: u32 },
    /// Random Hadamard incoherence, then BlockLDLQ with the trellis on
    /// 16 x 16 tiles. Needs Hessians; `damping` is relative to mean(diag H).
    TrellisLdlq {
        state_bits: u32,
        symbol_bits: u32,
        damping: f32,
    },
    /// Random Hadamard incoherence, then the 2-bit Lloyd-Max scalar quantizer.
    Scalar2Bit,
    /// Diagnostic: rotate and rotate back without quantizing.
    RotationOnly,
    /// Diagnostic: add Gaussian noise with this relative energy, no rotation.
    Noise { relative: f32 },
}

/// Deterministic N(0, 1) samples (SplitMix64 and Box-Muller).
fn gaussian(count: usize, seed: u64) -> Vec<f32> {
    let mut state = seed;
    let mut uniform = move || {
        state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        ((z ^ (z >> 31)) >> 11) as f64 / (1u64 << 53) as f64
    };
    (0..count)
        .map(|_| {
            let radius = (-2.0 * (1.0 - uniform()).ln()).sqrt();
            (radius * (std::f64::consts::TAU * uniform()).cos()) as f32
        })
        .collect()
}

#[derive(Clone, Debug)]
pub struct TensorReport {
    pub name: String,
    pub relative_error: f64,
    /// `tr(E H E^T) / tr(W H W^T)`, when Hessians were given.
    pub proxy_error: Option<f64>,
}

/// Default Hessian damping when only the proxy loss is reported.
const REPORT_DAMPING: f64 = 0.01;

/// Row-major `rows x cols` floats of a matrix (cols = GGUF dims[0]).
pub(crate) fn matrix(tensor: &Tensor) -> (Vec<f32>, usize, usize) {
    let (cols, rows) = (tensor.dims[0], tensor.dims[1]);
    let mut values = vec![0.0; rows * cols];
    for (r, row) in values.chunks_exact_mut(cols).enumerate() {
        tensor.row(r, row);
    }
    (values, rows, cols)
}

fn is_quantized(name: &str) -> bool {
    name.starts_with("blk.")
        && QUANTIZED_SUFFIXES
            .iter()
            .any(|suffix| name.ends_with(suffix))
}

/// BlockLDLQ in the rotated domain: `h` is the original-domain Hessian.
fn ldlq_reconstruct(
    rotated: &[f32],
    rows: usize,
    cols: usize,
    h: &[f64],
    rht: &Incoherence,
    trellis: &Trellis,
) -> Result<Vec<f32>> {
    let u = block_feedback(&cholesky_upper(&rht.transform_hessian(h), cols)?, cols)?;
    let sigma = rms(rotated);
    ldlq(rotated, rows, cols, &u, |tile| {
        let normalized: Vec<f32> = tile.iter().map(|&v| v / sigma).collect();
        trellis
            .decode(&trellis.quantize(&normalized))
            .into_iter()
            .map(|v| v * sigma)
            .collect()
    })
}

fn reconstruct(
    weights: &[f32],
    rows: usize,
    cols: usize,
    method: Method,
    seed: u64,
    h: Option<&[f64]>,
) -> Result<Vec<f32>> {
    if let Method::Noise { relative } = method {
        let sigma = super::matrix::rms(weights) * relative.sqrt();
        let noise = gaussian(weights.len(), seed);
        return Ok(weights
            .iter()
            .zip(noise)
            .map(|(w, n)| w + sigma * n)
            .collect());
    }
    let rht = Incoherence::new(rows, cols, seed)?;
    let rotated = rht.forward(weights);
    let quantized = match method {
        Method::Trellis {
            state_bits,
            symbol_bits,
        } => trellis_reconstruct(
            &rotated,
            &Trellis::new(state_bits, symbol_bits, 1.0)?,
            BLOCK,
        ),
        Method::TrellisLdlq {
            state_bits,
            symbol_bits,
            ..
        } => {
            let h = h.ok_or_else(|| Error::Invalid("LDLQ needs Hessians".into()))?;
            ldlq_reconstruct(
                &rotated,
                rows,
                cols,
                h,
                &rht,
                &Trellis::new(state_bits, symbol_bits, 1.0)?,
            )?
        }
        Method::Scalar2Bit => scalar_reconstruct(&rotated),
        Method::RotationOnly => rotated,
        Method::Noise { .. } => unreachable!("handled above"),
    };
    Ok(rht.inverse(&quantized))
}

/// Quantizes every attention and MLP matrix of `input` (all F16) into a new
/// file `output`. With `hessians`, each report also carries the proxy loss.
/// `progress` sees each tensor as it finishes.
pub fn quantize_gguf(
    input: &Path,
    output: &Path,
    method: Method,
    seed: u64,
    hessians: Option<&Hessians>,
    mut progress: impl FnMut(&TensorReport, usize, usize),
) -> Result<Vec<TensorReport>> {
    if output.exists() {
        return Err(Error::Invalid(format!(
            "{} already exists",
            output.display()
        )));
    }
    let model = Gguf::open(input, false)?;
    let mut names: Vec<&String> = model.tensors.keys().filter(|n| is_quantized(n)).collect();
    names.sort();
    if matches!(method, Method::TrellisLdlq { .. }) && hessians.is_none() {
        return Err(Error::Invalid(
            "--method trellis-ldlq needs --hessians".into(),
        ));
    }
    if names.is_empty() {
        return Err(Error::Invalid(
            "no attention or MLP matrices to quantize".into(),
        ));
    }
    for name in &names {
        if model.data_locations[*name].1 != F16_KIND || model.tensors[*name].dims.len() != 2 {
            return Err(Error::Invalid(format!("{name} must be an F16 matrix")));
        }
    }
    let partial = output.with_extension("partial");
    fs::copy(input, &partial)?;
    let mut file = OpenOptions::new().write(true).open(&partial)?;
    let mut reports = Vec::with_capacity(names.len());
    for (index, name) in names.iter().enumerate() {
        let (weights, rows, cols) = matrix(&model.tensors[*name]);
        let tensor_seed = seed ^ (index as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
        let damping = match method {
            Method::TrellisLdlq { damping, .. } => f64::from(damping),
            _ => REPORT_DAMPING,
        };
        let h = hessians
            .map(|set| set.load(name, cols, damping))
            .transpose()?;
        let reconstructed = reconstruct(&weights, rows, cols, method, tensor_seed, h.as_deref())?;
        let bytes: Vec<u8> = reconstructed
            .iter()
            .flat_map(|&v| f16::from_f32(v).to_bits().to_le_bytes())
            .collect();
        file.seek(SeekFrom::Start(model.data_locations[*name].0 as u64))?;
        file.write_all(&bytes)?;
        let report = TensorReport {
            name: (*name).clone(),
            relative_error: distortion(&weights, &reconstructed).relative,
            proxy_error: h
                .as_deref()
                .map(|h| proxy_error(&weights, &reconstructed, rows, cols, h)),
        };
        progress(&report, index + 1, names.len());
        reports.push(report);
    }
    file.sync_all()?;
    fs::rename(&partial, output)?;
    Ok(reports)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A minimal GGUF v3 with one F16 matrix per name and one F32 vector.
    fn write_gguf(path: &Path, matrices: &[(&str, usize, usize, Vec<f32>)]) {
        let string = |s: &str| [&(s.len() as u64).to_le_bytes()[..], s.as_bytes()].concat();
        let mut header = b"GGUF".to_vec();
        header.extend(3u32.to_le_bytes());
        header.extend(((matrices.len() + 1) as u64).to_le_bytes());
        header.extend(0u64.to_le_bytes());
        let mut data = Vec::new();
        for (name, rows, cols, values) in matrices {
            header.extend(string(name));
            header.extend(2u32.to_le_bytes());
            header.extend((*cols as u64).to_le_bytes());
            header.extend((*rows as u64).to_le_bytes());
            header.extend(F16_KIND.to_le_bytes());
            header.extend((data.len() as u64).to_le_bytes());
            data.extend(
                values
                    .iter()
                    .flat_map(|&v| f16::from_f32(v).to_bits().to_le_bytes()),
            );
            data.resize(data.len().div_ceil(32) * 32, 0);
        }
        header.extend(string("output_norm.weight"));
        header.extend(1u32.to_le_bytes());
        header.extend(8u64.to_le_bytes());
        header.extend(0u32.to_le_bytes());
        header.extend((data.len() as u64).to_le_bytes());
        data.extend((0..8).flat_map(|i| (i as f32).to_le_bytes()));
        header.resize(header.len().div_ceil(32) * 32, 0);
        fs::write(path, [header, data].concat()).unwrap();
    }

    fn values(count: usize) -> Vec<f32> {
        (0..count)
            .map(|i| ((i * 7919 % 1000) as f32 - 500.0) / 400.0)
            .collect()
    }

    #[test]
    fn rewrites_only_the_selected_matrices() {
        let dir = std::env::temp_dir().join(format!("palito-qtip-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let (input, output) = (dir.join("in.gguf"), dir.join("out.gguf"));
        let _ = fs::remove_file(&output);
        let q = values(32 * 64);
        let embedding = values(16 * 64);
        write_gguf(
            &input,
            &[
                ("blk.0.attn_q.weight", 32, 64, q.clone()),
                ("token_embd.weight", 16, 64, embedding),
            ],
        );
        let method = Method::Trellis {
            state_bits: 8,
            symbol_bits: 2,
        };
        let reports = quantize_gguf(&input, &output, method, 1, None, |_, _, _| {}).unwrap();
        assert_eq!(reports.len(), 1);
        assert!(reports[0].relative_error > 0.0 && reports[0].relative_error < 0.2);

        let before = Gguf::open(&input, false).unwrap();
        let after = Gguf::open(&output, false).unwrap();
        let unchanged =
            |name: &str| matrix(&before.tensors[name]).0 == matrix(&after.tensors[name]).0;
        assert!(unchanged("token_embd.weight"));
        assert_eq!(
            before.tensors["output_norm.weight"].floats(),
            after.tensors["output_norm.weight"].floats()
        );
        assert!(!unchanged("blk.0.attn_q.weight"));
        let written = matrix(&after.tensors["blk.0.attn_q.weight"]).0;
        assert!((distortion(&q, &written).relative - reports[0].relative_error).abs() < 1e-3);

        assert!(quantize_gguf(&input, &output, method, 1, None, |_, _, _| {}).is_err());
        fs::remove_dir_all(&dir).unwrap();
    }
}
