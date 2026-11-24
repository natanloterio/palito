//! Whole-model quantization: writes a copy of an F16 GGUF whose attention and
//! MLP matrices hold the quantized-then-reconstructed weights. Every backend
//! can then measure the quantization's effect on the model without a new
//! tensor type. The file keeps the F16 size; it is a quality probe, not a
//! compressed format.

use super::{
    hessian::Hessians,
    incoherence::Incoherence,
    ldlq::{block_feedback, cholesky_upper, ldlq, ldlq_with, proxy_error},
    matrix::{distortion, rms, scalar_reconstruct, trellis_reconstruct},
    packed::pack_tile,
    trellis::Trellis,
};
use crate::{
    Error, Result,
    gguf::{Directory, Gguf, QTIP3_COMPANION, QTIP3_KIND, Tensor, write_gguf},
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
    hadamard_block: Option<usize>,
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
    let rht = Incoherence::with_block(rows, cols, seed, hadamard_block)?;
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
    hadamard_block: Option<usize>,
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
        let reconstructed = reconstruct(
            &weights,
            rows,
            cols,
            method,
            tensor_seed,
            h.as_deref(),
            hadamard_block,
        )?;
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

/// Trellis bits of the QTIP3 tensor type (`gguf::QTIP3_KIND`).
const QTIP3_TRELLIS: (u32, u32) = (12, 3);
/// Hadamard block of the QTIP3 tensor type.
const QTIP3_BLOCK: usize = 256;

/// One matrix as QTIP3: the packed tiles (little-endian words, row-group
/// order), the companion `[sigma, row signs, column signs]`, and the dense
/// reconstruction for the report.
fn qtip3_matrix(
    weights: &[f32],
    rows: usize,
    cols: usize,
    seed: u64,
    h: &[f64],
) -> Result<(Vec<u8>, Vec<f32>, Vec<f32>)> {
    let (state_bits, symbol_bits) = QTIP3_TRELLIS;
    let rht = Incoherence::with_block(rows, cols, seed, Some(QTIP3_BLOCK))?;
    let rotated = rht.forward(weights);
    let trellis = Trellis::new(state_bits, symbol_bits, 1.0)?;
    let u = block_feedback(&cholesky_upper(&rht.transform_hessian(h), cols)?, cols)?;
    let sigma = rms(&rotated);
    let (quantized, tiles) = ldlq_with(&rotated, rows, cols, &u, |tile| {
        let normalized: Vec<f32> = tile.iter().map(|&v| v / sigma).collect();
        let encoded = trellis.quantize(&normalized);
        let words = pack_tile(&encoded, state_bits, symbol_bits);
        let values = trellis
            .decode(&encoded)
            .into_iter()
            .map(|v| v * sigma)
            .collect();
        (values, words)
    })?;
    let bytes = tiles
        .into_iter()
        .collect::<Result<Vec<_>>>()?
        .concat()
        .iter()
        .flat_map(|w| w.to_le_bytes())
        .collect();
    let companion = std::iter::once(sigma)
        .chain(rht.row_signs().iter().copied())
        .chain(rht.col_signs().iter().copied())
        .collect();
    Ok((bytes, companion, rht.inverse(&quantized)))
}

/// Writes `output` with every attention and MLP matrix of `input` (F16) as a
/// QTIP3 tensor (BlockLDLQ, L12 K3, 256-block Hadamards) plus its F32
/// companion vector; other tensors and all metadata are copied unchanged.
/// Only Palito's CUDA backend with `qtip-research` reads the result.
pub fn quantize_gguf_qtip3(
    input: &Path,
    output: &Path,
    seed: u64,
    hessians: &Hessians,
    damping: f32,
    mut progress: impl FnMut(&TensorReport, usize, usize),
) -> Result<Vec<TensorReport>> {
    if output.exists() {
        return Err(Error::Invalid(format!(
            "{} already exists",
            output.display()
        )));
    }
    let original = fs::read(input)?;
    let directory = Directory::read(&original)?;
    let model = Gguf::open(input, false)?;
    let mut names: Vec<&String> = model.tensors.keys().filter(|n| is_quantized(n)).collect();
    names.sort();
    if names.is_empty() {
        return Err(Error::Invalid(
            "no attention or MLP matrices to quantize".into(),
        ));
    }
    let mut packed = std::collections::HashMap::new();
    let mut reports = Vec::with_capacity(names.len());
    for (index, name) in names.iter().enumerate() {
        let tensor = &model.tensors[*name];
        if model.data_locations[*name].1 != F16_KIND || tensor.dims.len() != 2 {
            return Err(Error::Invalid(format!("{name} must be an F16 matrix")));
        }
        let (weights, rows, cols) = matrix(tensor);
        if rows % QTIP3_BLOCK != 0 || cols % QTIP3_BLOCK != 0 {
            return Err(Error::Invalid(format!(
                "{name}: QTIP3 needs 256-aligned shapes"
            )));
        }
        // Same per-tensor seed as quantize_gguf, so the probe and this file match.
        let tensor_seed = seed ^ (index as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
        let h = hessians.load(name, cols, f64::from(damping))?;
        let (bytes, companion, reconstructed) =
            qtip3_matrix(&weights, rows, cols, tensor_seed, &h)?;
        let report = TensorReport {
            name: (*name).clone(),
            relative_error: distortion(&weights, &reconstructed).relative,
            proxy_error: Some(proxy_error(&weights, &reconstructed, rows, cols, &h)),
        };
        progress(&report, index + 1, names.len());
        reports.push(report);
        let companion: Vec<u8> = companion.iter().flat_map(|v| v.to_le_bytes()).collect();
        packed.insert((*name).clone(), (bytes, companion));
    }
    let mut tensors: Vec<(String, Vec<usize>, u32, &[u8])> = Vec::new();
    for entry in &directory.tensors {
        if let Some((bytes, companion)) = packed.get(&entry.name) {
            tensors.push((entry.name.clone(), entry.dims.clone(), QTIP3_KIND, bytes));
            let length = companion.len() / 4;
            tensors.push((
                format!("{}{QTIP3_COMPANION}", entry.name),
                vec![length],
                0,
                companion,
            ));
        } else {
            tensors.push((
                entry.name.clone(),
                entry.dims.clone(),
                entry.kind,
                &original[entry.data.clone()],
            ));
        }
    }
    let partial = output.with_extension("partial");
    write_gguf(&partial, &original, &directory, &tensors)?;
    fs::rename(&partial, output)?;
    Ok(reports)
}

/// Writes `output`: `base` with each tensor in `names` replaced by the
/// tensor of the same name and shape from `donor` (any type `gguf` reads,
/// such as a llama.cpp Q6_K output head). Metadata comes from `base`.
pub fn splice_gguf(base: &Path, donor: &Path, names: &[String], output: &Path) -> Result<()> {
    if output.exists() {
        return Err(Error::Invalid(format!(
            "{} already exists",
            output.display()
        )));
    }
    let (base_bytes, donor_bytes) = (fs::read(base)?, fs::read(donor)?);
    let (base_dir, donor_dir) = (
        Directory::read(&base_bytes)?,
        Directory::read(&donor_bytes)?,
    );
    let mut tensors: Vec<(String, Vec<usize>, u32, &[u8])> = Vec::new();
    let mut replaced = 0;
    for entry in &base_dir.tensors {
        let source = if names.contains(&entry.name) {
            let found = donor_dir
                .tensors
                .iter()
                .find(|d| d.name == entry.name && d.dims == entry.dims)
                .ok_or_else(|| {
                    Error::Invalid(format!("{}: missing or reshaped in donor", entry.name))
                })?;
            replaced += 1;
            (found.kind, &donor_bytes[found.data.clone()])
        } else {
            (entry.kind, &base_bytes[entry.data.clone()])
        };
        tensors.push((entry.name.clone(), entry.dims.clone(), source.0, source.1));
    }
    if replaced != names.len() {
        return Err(Error::Invalid(
            "a spliced tensor is not in the base model".into(),
        ));
    }
    let partial = output.with_extension("partial");
    write_gguf(&partial, &base_bytes, &base_dir, &tensors)?;
    fs::rename(&partial, output)?;
    Ok(())
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
        let reports = quantize_gguf(&input, &output, method, 1, None, None, |_, _, _| {}).unwrap();
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

        assert!(quantize_gguf(&input, &output, method, 1, None, None, |_, _, _| {}).is_err());
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn splices_only_the_named_tensors() {
        let dir = std::env::temp_dir().join(format!("palito-splice-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let (base, donor, output) = (dir.join("a.gguf"), dir.join("b.gguf"), dir.join("c.gguf"));
        let (q, e) = (values(32 * 64), values(16 * 64));
        let other: Vec<f32> = e.iter().map(|v| v * 0.5).collect();
        write_gguf(
            &base,
            &[
                ("blk.0.attn_q.weight", 32, 64, q.clone()),
                ("token_embd.weight", 16, 64, e),
            ],
        );
        write_gguf(
            &donor,
            &[
                (
                    "blk.0.attn_q.weight",
                    32,
                    64,
                    values(32 * 64).iter().map(|v| -v).collect(),
                ),
                ("token_embd.weight", 16, 64, other),
            ],
        );
        let names = ["token_embd.weight".to_string()];
        splice_gguf(&base, &donor, &names, &output).unwrap();
        let (a, b, c) = (
            Gguf::open(&base, false).unwrap(),
            Gguf::open(&donor, false).unwrap(),
            Gguf::open(&output, false).unwrap(),
        );
        let same = |x: &Gguf, name: &str| matrix(&x.tensors[name]).0 == matrix(&c.tensors[name]).0;
        assert!(same(&b, "token_embd.weight") && !same(&a, "token_embd.weight"));
        assert!(same(&a, "blk.0.attn_q.weight"));
        // Existing output, and a name the base lacks, are refused.
        assert!(splice_gguf(&base, &donor, &names, &output).is_err());
        let missing = ["blk.9.attn_q.weight".to_string()];
        assert!(splice_gguf(&base, &donor, &missing, &dir.join("d.gguf")).is_err());
        assert!(!dir.join("d.gguf").exists());
        fs::remove_dir_all(&dir).unwrap();
    }
}
