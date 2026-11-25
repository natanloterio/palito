//! GGUF -> PLT1 (`.palito`) converter with offline RMSNorm + residual-stream
//! Hadamard folding (`FOLDED_RMSNORM | FOLDED_RESIDUAL_RHT`) and MMA-swizzled
//! QTIP HYB (`L=12/16, Q=5..9, V=2`) BlockLDLQ quantization.

use super::{
    hessian::Hessians,
    hyb::{HybCodebook, HybQuantizer, decode_hyb_tile_mma_l, swizzle_tile_to_mma},
    incoherence::Incoherence,
    ldlq::{block_feedback, cholesky_upper, ldlq_with, proxy_error},
    matrix::{distortion, rms},
    model::{TensorReport, matrix},
};
use crate::{
    Error, Result,
    gguf::Gguf,
    plt1::{
        Codec, PLT1_VERSION, Plt1Header, Plt1Index, Plt1Writer, SectionKind, TensorDesc, flags,
        tensor_flags,
    },
};
use std::{fs::File, io::BufWriter, path::Path};

const RHT_BLOCK: usize = 256;

#[derive(Clone, Debug)]
pub struct Plt1ExportOptions {
    pub state_bits: u32,
    pub pair_bits: u32,
    pub lut_bits: u32,
    pub damping: f32,
    pub seed: u64,
    pub fold_residual_rht: bool,
    /// Optional layer limit (e.g. `Some(1)` for fast 1-layer quality/size probe).
    pub max_layers: Option<usize>,
}

impl Default for Plt1ExportOptions {
    fn default() -> Self {
        Self {
            state_bits: 12,
            pair_bits: 4, // 2.0 bpw
            lut_bits: 5,  // Q=5 zero-SMEM warp-register LUT
            damping: 0.01,
            seed: 0x5041_4C49_544F_3031,
            fold_residual_rht: true,
            max_layers: None,
        }
    }
}

fn layer_index(name: &str) -> Option<usize> {
    let rest = name.strip_prefix("blk.")?;
    let (idx, _) = rest.split_once('.')?;
    idx.parse().ok()
}

fn is_residual_input_proj(name: &str) -> bool {
    name.ends_with("attn_q.weight")
        || name.ends_with("attn_k.weight")
        || name.ends_with("attn_v.weight")
        || name.ends_with("ffn_gate.weight")
        || name.ends_with("ffn_up.weight")
        || name.ends_with("ffn_gate_exps.weight")
        || name.ends_with("ffn_up_exps.weight")
}

fn is_residual_output_proj(name: &str) -> bool {
    name.ends_with("attn_output.weight")
        || name.ends_with("ffn_down.weight")
        || name.ends_with("ffn_down_exps.weight")
}

fn is_quantized_proj(name: &str) -> bool {
    name.starts_with("blk.") && (is_residual_input_proj(name) || is_residual_output_proj(name))
}

fn rmsnorm_name_for(proj: &str) -> Option<String> {
    let idx = layer_index(proj)?;
    if proj.ends_with("attn_q.weight")
        || proj.ends_with("attn_k.weight")
        || proj.ends_with("attn_v.weight")
    {
        Some(format!("blk.{idx}.attn_norm.weight"))
    } else if proj.ends_with("ffn_gate.weight")
        || proj.ends_with("ffn_up.weight")
        || proj.ends_with("ffn_gate_exps.weight")
        || proj.ends_with("ffn_up_exps.weight")
    {
        Some(format!("blk.{idx}.ffn_norm.weight"))
    } else {
        None
    }
}

/// Converts an F16 GGUF into a 128B-aligned `.palito` (`PLT1`) container with
/// MMA-swizzled HYB tiles, optional offline RMSNorm + residual-stream RHT
/// folding, and BlockLDLQ Hessian-guided quantization.
pub fn export_gguf_to_plt1(
    input: &Path,
    output: &Path,
    hessians: &Hessians,
    opts: &Plt1ExportOptions,
    mut progress: impl FnMut(&TensorReport, usize, usize),
) -> Result<(Vec<TensorReport>, Plt1Index)> {
    let model = Gguf::open(input, false)?;
    let codebook = HybCodebook::gaussian_for_q(opts.lut_bits);
    let quantizer = HybQuantizer::new(opts.state_bits, opts.pair_bits, codebook.clone())?;

    let mut names: Vec<String> = model
        .tensors
        .keys()
        .filter(|n| {
            is_quantized_proj(n)
                && opts
                    .max_layers
                    .is_none_or(|max| layer_index(n).is_some_and(|i| i < max))
        })
        .cloned()
        .collect();
    names.sort();
    if names.is_empty() {
        return Err(Error::Invalid("no projections selected for PLT1 export".into()));
    }

    // Infer model dimensions from the first layer.
    let d_model = model
        .tensors
        .get("blk.0.attn_q.weight")
        .map_or(0, |t| t.dims[0] as u32);
    let d_ff = model
        .tensors
        .get("blk.0.ffn_gate.weight")
        .or_else(|| model.tensors.get("blk.0.ffn_gate_exps.weight"))
        .map_or(0, |t| t.dims[1] as u32);
    let vocab_size = model
        .tensors
        .get("token_embd.weight")
        .map_or(0, |t| t.dims[1] as u32);
    let n_layers = names
        .iter()
        .filter_map(|n| layer_index(n))
        .max()
        .map_or(0, |m| (m + 1) as u32);

    let mut hdr_flags = flags::MMA_M16N8K16_SWIZZLED;
    if opts.fold_residual_rht {
        hdr_flags |= flags::FOLDED_RMSNORM | flags::FOLDED_RESIDUAL_RHT;
    }

    let header = Plt1Header {
        version: PLT1_VERSION,
        target_sm: 1200,
        flags: hdr_flags,
        num_sections: 0,
        section_dir_offset: 0,
        n_layers,
        d_model,
        d_ff,
        vocab_size,
    };

    let mut writer = Plt1Writer::new(header);
    let meta_json = format!(
        "{{\"source\":\"{}\",\"codec\":\"hyb\",\"state_bits\":{},\"pair_bits\":{},\"lut_bits\":{},\"folded_rht\":{}}}",
        input.file_name().and_then(|s| s.to_str()).unwrap_or("unknown"),
        opts.state_bits,
        opts.pair_bits,
        opts.lut_bits,
        opts.fold_residual_rht
    );
    writer.add_section(SectionKind::MetadataJson, 0, meta_json.into_bytes());

    let codec = if opts.lut_bits == 5 {
        Codec::QtipHybQ5
    } else {
        Codec::QtipHybSmem
    };

    // Shared global residual-stream sign diagonal S_res (d_model floats) when
    // fold_residual_rht is enabled: Q = H_256 * S_res cancels across all 32 layers.
    let r1_signs: Vec<f32> = Incoherence::with_block(
        d_model as usize,
        d_model as usize,
        opts.seed ^ 0x5231_5349_474E_5331,
        Some(RHT_BLOCK),
    )?
    .col_signs()
    .to_vec();
    if opts.fold_residual_rht {
        let r1_bytes: Vec<u8> = r1_signs.iter().flat_map(|v| v.to_le_bytes()).collect();
        writer.add_tensor(
            TensorDesc {
                name: "plt1.r1_signs".into(),
                rows: 1,
                cols: d_model,
                codec: Codec::F16, // raw f32 sign vector
                flags: 0,
                pair_bits: 0,
                lut_bits: 0,
                scale: 1.0,
                teal_thresholds: [0.0; 4],
                data_offset: 0,
                data_size: 0,
                lut_offset: 0,
            },
            r1_bytes,
            None,
        );
    }

    let mut reports = Vec::with_capacity(names.len());
    for (idx, name) in names.iter().enumerate() {
        let tensor = &model.tensors[name];
        if tensor.dims.len() == 3 {
            // 3D MoE expert tensor: GGUF dims = [cols, rows, n_experts].
            let cols = tensor.dims[0];
            let rows = tensor.dims[1];
            let n_experts = tensor.dims[2];
            if rows % RHT_BLOCK != 0 || cols % RHT_BLOCK != 0 {
                return Err(Error::Invalid(format!(
                    "{name}: PLT1 HYB requires {RHT_BLOCK}-aligned shapes, got {n_experts}x{rows}x{cols}"
                )));
            }
            let gamma = if opts.fold_residual_rht
                && let Some(norm_name) = rmsnorm_name_for(name)
                && let Some(norm_tensor) = model.tensors.get(&norm_name)
            {
                let mut g = vec![0.0f32; cols];
                norm_tensor.row(0, &mut g);
                Some(g)
            } else {
                None
            };
            let tensor_seed = opts.seed ^ (idx as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
            let base_rht = Incoherence::with_block(rows, cols, tensor_seed, Some(RHT_BLOCK))?;
            let rht = if opts.fold_residual_rht && is_residual_input_proj(name) {
                Incoherence::with_signs(
                    rows,
                    cols,
                    base_rht.row_signs().to_vec(),
                    r1_signs.clone(),
                    Some(RHT_BLOCK),
                )?
            } else if opts.fold_residual_rht && is_residual_output_proj(name) {
                Incoherence::with_signs(
                    rows,
                    cols,
                    r1_signs.clone(),
                    base_rht.col_signs().to_vec(),
                    Some(RHT_BLOCK),
                )?
            } else {
                base_rht
            };
            // Identity Hessian fallback for MoE experts (optional shared Hessian if present).
            let mut h_mat = vec![0.0f64; cols * cols];
            for i in 0..cols {
                h_mat[i * cols + i] = 1.0 + f64::from(opts.damping);
            }
            let u = block_feedback(&cholesky_upper(&rht.transform_hessian(&h_mat), cols)?, cols)?;

            let mut all_tile_bytes = Vec::new();
            let mut expert_scales = Vec::with_capacity(n_experts);
            let mut sum_rel = 0.0f64;
            let mut row_buf = vec![0.0f32; cols];
            for e in 0..n_experts {
                let mut w_e = vec![0.0f32; rows * cols];
                for r in 0..rows {
                    tensor.row(e * rows + r, &mut row_buf);
                    if let Some(g) = &gamma {
                        for (dst, (&v, &gv)) in
                            w_e[r * cols..(r + 1) * cols].iter_mut().zip(row_buf.iter().zip(g))
                        {
                            *dst = v * gv;
                        }
                    } else {
                        w_e[r * cols..(r + 1) * cols].copy_from_slice(&row_buf);
                    }
                }
                let rotated = rht.forward(&w_e);
                let sigma = rms(&rotated).max(1e-8);
                let (quantized, tiles) = ldlq_with(&rotated, rows, cols, &u, |tile| {
                    let swizzled = swizzle_tile_to_mma(tile);
                    let norm: Vec<f32> = swizzled.iter().map(|&v| v / sigma).collect();
                    let words = quantizer.quantize_normalized(&norm);
                    let decoded = match &words {
                        Ok(w) => decode_hyb_tile_mma_l(
                            w,
                            opts.state_bits,
                            opts.pair_bits,
                            &codebook,
                            sigma,
                        )
                        .to_vec(),
                        Err(_) => vec![0.0; tile.len()],
                    };
                    (decoded, words)
                })?;
                let reconstructed = rht.inverse(&quantized);
                sum_rel += distortion(&w_e, &reconstructed).relative;
                for w in tiles.into_iter().collect::<Result<Vec<_>>>()?.concat() {
                    all_tile_bytes.extend_from_slice(&w.to_le_bytes());
                }
                expert_scales.push(if opts.fold_residual_rht {
                    sigma / 16.0
                } else {
                    sigma / 256.0
                });
            }
            let report = TensorReport {
                name: name.clone(),
                relative_error: sum_rel / n_experts as f64,
                proxy_error: None,
            };
            progress(&report, idx + 1, names.len());
            reports.push(report);

            let mut tflags = tensor_flags::MMA_SWIZZLED;
            if opts.fold_residual_rht {
                if is_residual_input_proj(name) {
                    tflags |= tensor_flags::SKIP_IN_RHT;
                }
                if is_residual_output_proj(name) {
                    tflags |= tensor_flags::SKIP_OUT_RHT;
                }
            }
            writer.add_tensor(
                TensorDesc {
                    name: name.clone(),
                    rows: rows as u32,
                    cols: cols as u32,
                    codec,
                    flags: tflags,
                    pair_bits: opts.pair_bits as u8,
                    lut_bits: opts.lut_bits as u8,
                    scale: 1.0,
                    teal_thresholds: [0.0; 4],
                    data_offset: 0,
                    data_size: 0,
                    lut_offset: 0,
                },
                all_tile_bytes,
                Some(codebook.lut_f16x2.clone()),
            );
            let scales_bytes: Vec<u8> = expert_scales
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect();
            writer.add_tensor(
                TensorDesc {
                    name: format!("{name}.scales"),
                    rows: 1,
                    cols: n_experts as u32,
                    codec: Codec::F16,
                    flags: 0,
                    pair_bits: 0,
                    lut_bits: 0,
                    scale: 1.0,
                    teal_thresholds: [0.0; 4],
                    data_offset: 0,
                    data_size: 0,
                    lut_offset: 0,
                },
                scales_bytes,
                None,
            );
            let signs_bytes: Vec<u8> = rht
                .row_signs()
                .iter()
                .chain(rht.col_signs().iter())
                .flat_map(|v| v.to_le_bytes())
                .collect();
            writer.add_tensor(
                TensorDesc {
                    name: format!("{name}.signs"),
                    rows: 1,
                    cols: (rows + cols) as u32,
                    codec: Codec::F16,
                    flags: 0,
                    pair_bits: 0,
                    lut_bits: 0,
                    scale: 1.0,
                    teal_thresholds: [0.0; 4],
                    data_offset: 0,
                    data_size: 0,
                    lut_offset: 0,
                },
                signs_bytes,
                None,
            );
            continue;
        }
        let (mut weights, rows, cols) = matrix(tensor);
        if rows % RHT_BLOCK != 0 || cols % RHT_BLOCK != 0 {
            return Err(Error::Invalid(format!(
                "{name}: PLT1 HYB requires {RHT_BLOCK}-aligned shapes, got {rows}x{cols}"
            )));
        }

        // Offline RMSNorm folding for residual-input projections when enabled:
        // load undamped H first so scaling by diag(1/gamma) does not amplify
        // damping on tiny-gamma channels, then damp H' cleanly.
        let h = if opts.fold_residual_rht
            && let Some(norm_name) = rmsnorm_name_for(name)
            && let Some(norm_tensor) = model.tensors.get(&norm_name)
        {
            let mut h_raw = hessians.load(name, cols, 0.0)?;
            let mut gamma = vec![0.0f32; cols];
            norm_tensor.row(0, &mut gamma);
            for row in weights.chunks_exact_mut(cols) {
                for (w, &g) in row.iter_mut().zip(&gamma) {
                    *w *= g;
                }
            }
            let mut valid_diags = Vec::with_capacity(cols);
            let inv: Vec<f64> = gamma
                .iter()
                .enumerate()
                .map(|(i, &g)| {
                    if g.abs() >= 1e-2 {
                        let iv = 1.0 / f64::from(g);
                        valid_diags.push(h_raw[i * cols + i] * iv * iv);
                        iv
                    } else {
                        0.0
                    }
                })
                .collect();
            valid_diags.sort_by(|a, b| a.total_cmp(b));
            let floor_diag = if valid_diags.is_empty() {
                1.0
            } else {
                valid_diags[valid_diags.len() / 2].max(1e-6)
            };
            for r in 0..cols {
                let ir = inv[r];
                for c in 0..cols {
                    h_raw[r * cols + c] *= ir * inv[c];
                }
                if ir == 0.0 {
                    // Keep unit-norm diagonal weight on near-zero gamma channels so LDLQ
                    // does not dump quantization error onto them (x_unit still has ~1.0 RMS).
                    h_raw[r * cols + r] = floor_diag;
                }
            }
            let mean_diag = (0..cols).map(|i| h_raw[i * cols + i]).sum::<f64>() / cols as f64;
            let damp = f64::from(opts.damping) * mean_diag.max(1e-12);
            for i in 0..cols {
                h_raw[i * cols + i] += damp;
            }
            h_raw
        } else {
            hessians.load(name, cols, f64::from(opts.damping))?
        };

        let tensor_seed = opts.seed ^ (idx as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
        let base_rht = Incoherence::with_block(rows, cols, tensor_seed, Some(RHT_BLOCK))?;
        let rht = if opts.fold_residual_rht && is_residual_input_proj(name) {
            Incoherence::with_signs(
                rows,
                cols,
                base_rht.row_signs().to_vec(),
                r1_signs.clone(),
                Some(RHT_BLOCK),
            )?
        } else if opts.fold_residual_rht && is_residual_output_proj(name) {
            Incoherence::with_signs(
                rows,
                cols,
                r1_signs.clone(),
                base_rht.col_signs().to_vec(),
                Some(RHT_BLOCK),
            )?
        } else {
            base_rht
        };
        let rotated = rht.forward(&weights);
        let u = block_feedback(&cholesky_upper(&rht.transform_hessian(&h), cols)?, cols)?;
        let sigma = rms(&rotated);

        let (quantized, tiles) = ldlq_with(&rotated, rows, cols, &u, |tile| {
            let swizzled = swizzle_tile_to_mma(tile);
            let norm: Vec<f32> = swizzled.iter().map(|&v| v / sigma).collect();
            let words = quantizer.quantize_normalized(&norm);
            let decoded = match &words {
                Ok(w) => {
                    decode_hyb_tile_mma_l(w, opts.state_bits, opts.pair_bits, &codebook, sigma)
                        .to_vec()
                }
                Err(_) => vec![0.0; tile.len()],
            };
            (decoded, words)
        })?;

        let reconstructed = rht.inverse(&quantized);
        let report = TensorReport {
            name: name.clone(),
            relative_error: distortion(&weights, &reconstructed).relative,
            proxy_error: Some(proxy_error(&weights, &reconstructed, rows, cols, &h)),
        };
        progress(&report, idx + 1, names.len());
        reports.push(report);

        let tile_bytes: Vec<u8> = tiles
            .into_iter()
            .collect::<Result<Vec<_>>>()?
            .concat()
            .iter()
            .flat_map(|w| w.to_le_bytes())
            .collect();

        let mut tflags = tensor_flags::MMA_SWIZZLED;
        let gpu_scale = if opts.fold_residual_rht {
            if is_residual_input_proj(name) {
                tflags |= tensor_flags::SKIP_IN_RHT;
            }
            if is_residual_output_proj(name) {
                tflags |= tensor_flags::SKIP_OUT_RHT;
            }
            // Only one unnormalized H_256 runs online (factor 16).
            sigma / 16.0
        } else {
            // Both unnormalized H_256 run online (factor 256).
            sigma / 256.0
        };

        writer.add_tensor(
            TensorDesc {
                name: name.clone(),
                rows: rows as u32,
                cols: cols as u32,
                codec,
                flags: tflags,
                pair_bits: opts.pair_bits as u8,
                lut_bits: opts.lut_bits as u8,
                scale: gpu_scale,
                teal_thresholds: [0.0; 4],
                data_offset: 0,
                data_size: 0,
                lut_offset: 0,
            },
            tile_bytes,
            Some(codebook.lut_f16x2.clone()),
        );
        // Store [row_signs, col_signs] companion vector for the non-folded side.
        let signs_bytes: Vec<u8> = rht
            .row_signs()
            .iter()
            .chain(rht.col_signs().iter())
            .flat_map(|v| v.to_le_bytes())
            .collect();
        writer.add_tensor(
            TensorDesc {
                name: format!("{name}.signs"),
                rows: 1,
                cols: (rows + cols) as u32,
                codec: Codec::F16,
                flags: 0,
                pair_bits: 0,
                lut_bits: 0,
                scale: 1.0,
                teal_thresholds: [0.0; 4],
                data_offset: 0,
                data_size: 0,
                lut_offset: 0,
            },
            signs_bytes,
            None,
        );
    }

    let partial = output.with_extension("palito.partial");
    {
        let file = File::create(&partial)?;
        let mut bw = BufWriter::new(file);
        writer.write_to(&mut bw)?;
    }
    std::fs::rename(&partial, output)?;

    let mut verify_file = File::open(output)?;
    let index = Plt1Index::read_from(&mut verify_file)?;
    Ok((reports, index))
}

/// Writes a compact skeleton GGUF copying all metadata, tokenizer, embeddings,
/// LM head, and layer norms from `input` while omitting the 7 per-layer linear
/// projections (and any `.qtip` companion vectors) that live in `PLT1`'s `LayerSlabs`.
pub fn strip_projections_to_skeleton_gguf(input: &Path, output: &Path) -> Result<()> {
    let original = std::fs::read(input)?;
    let directory = crate::gguf::Directory::read(&original)?;
    let mut kept: Vec<(String, Vec<usize>, u32, &[u8])> = Vec::new();
    for entry in &directory.tensors {
        let base_name = entry
            .name
            .strip_suffix(crate::gguf::QTIP3_COMPANION)
            .unwrap_or(&entry.name);
        if is_quantized_proj(base_name) {
            continue;
        }
        kept.push((
            entry.name.clone(),
            entry.dims.clone(),
            entry.kind,
            &original[entry.data.clone()],
        ));
    }
    crate::gguf::write_gguf(output, &original, &directory, &kept)
}
