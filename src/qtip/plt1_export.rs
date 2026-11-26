//! GGUF -> PLT1 (`.palito`) converter with offline RMSNorm + residual-stream
//! Hadamard folding (`FOLDED_RMSNORM | FOLDED_RESIDUAL_RHT`) and MMA-swizzled
//! QTIP HYB (`L=12/16, Q=5..9, V=2`) BlockLDLQ quantization.

use super::{
    hessian::Hessians,
    hyb::{HybCodebook, HybQuantizer, decode_hyb_tile_mma_l, swizzle_tile_to_mma},
    incoherence::Incoherence,
    ldlq::{block_feedback, cholesky_upper, ldlq_with, ldlq_with_batch, proxy_error},
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

/// Folds `diag(1/gamma)` into an undamped input Hessian `h_raw` (with median
/// diagonal floor on near-zero `gamma` channels) and applies relative diagonal
/// damping.
fn fold_rmsnorm_hessian(h_raw: &mut [f64], gamma: &[f32], cols: usize, damping: f64) {
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
    let damp = damping * mean_diag.max(1e-12);
    for i in 0..cols {
        h_raw[i * cols + i] += damp;
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
    #[cfg(feature = "cuda")]
    let gpu_quantizer: Option<super::hyb_cuda::CudaHybQuantizer> =
        cudarc::driver::CudaContext::new(0).ok().and_then(|ctx| {
            super::hyb_cuda::CudaHybQuantizer::new(
                ctx,
                opts.state_bits,
                opts.pair_bits,
                codebook.clone(),
            )
            .ok()
        });
    let words_per_tile = (128 * opts.pair_bits as usize) / 32;
    let quantize_matrix = |rotated: &[f32],
                           rows: usize,
                           cols: usize,
                           u: &[f64],
                           sigma: f32|
     -> Result<(Vec<f32>, Vec<Vec<u32>>)> {
        #[cfg(feature = "cuda")]
        if let Some(gpu) = &gpu_quantizer {
            let inv_sigma = 1.0 / sigma;
            return ldlq_with_batch(rotated, rows, cols, u, |batch_tiles| {
                let mut swizzled_norm = Vec::with_capacity(batch_tiles.len() * 256);
                for tile in batch_tiles {
                    let swz = swizzle_tile_to_mma(tile);
                    swizzled_norm.extend(swz.iter().map(|&v| v * inv_sigma));
                }
                let (packed, mut decoded) = gpu.quantize_tiles_normalized(&swizzled_norm)?;
                for v in &mut decoded {
                    *v *= sigma;
                }
                let codes: Vec<Vec<u32>> = packed
                    .chunks_exact(words_per_tile)
                    .map(|c| c.to_vec())
                    .collect();
                Ok((decoded, codes))
            });
        }
        let (quantized, tiles) = ldlq_with(rotated, rows, cols, u, |tile| {
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
        let tiles = tiles.into_iter().collect::<Result<Vec<_>>>()?;
        Ok((quantized, tiles))
    };

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
            let has_per_expert_h = hessians.expert_count(name).is_some();
            let (shared_h, shared_u) = if has_per_expert_h {
                (None, None)
            } else if hessians.contains(name) {
                let h = if let Some(g) = &gamma {
                    let mut h_raw = hessians.load(name, cols, 0.0)?;
                    fold_rmsnorm_hessian(&mut h_raw, g, cols, f64::from(opts.damping));
                    h_raw
                } else {
                    hessians.load(name, cols, f64::from(opts.damping))?
                };
                let u = block_feedback(&cholesky_upper(&rht.transform_hessian(&h), cols)?, cols)?;
                (Some(h), Some(u))
            } else {
                // Identity Hessian fallback when no Hessian is provided for this MoE tensor.
                let mut h_mat = vec![0.0f64; cols * cols];
                for i in 0..cols {
                    h_mat[i * cols + i] = 1.0 + f64::from(opts.damping);
                }
                let u =
                    block_feedback(&cholesky_upper(&rht.transform_hessian(&h_mat), cols)?, cols)?;
                (None, Some(u))
            };

            let mut all_tile_bytes = Vec::new();
            let mut expert_scales = Vec::with_capacity(n_experts);
            let mut sum_rel = 0.0f64;
            let mut sum_proxy = 0.0f64;
            let mut has_proxy = false;
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
                let expert_h_owned = if has_per_expert_h {
                    let h = if let Some(g) = &gamma {
                        let mut h_raw = hessians.load_expert(name, e, n_experts, cols, 0.0)?;
                        fold_rmsnorm_hessian(&mut h_raw, g, cols, f64::from(opts.damping));
                        h_raw
                    } else {
                        hessians.load_expert(name, e, n_experts, cols, f64::from(opts.damping))?
                    };
                    Some(h)
                } else {
                    None
                };
                let expert_u_owned = match &expert_h_owned {
                    Some(h_e) => Some(block_feedback(
                        &cholesky_upper(&rht.transform_hessian(h_e), cols)?,
                        cols,
                    )?),
                    None => None,
                };
                let u_ref = expert_u_owned
                    .as_deref()
                    .or(shared_u.as_deref())
                    .expect("either per-expert or shared feedback matrix must exist");
                let rotated = rht.forward(&w_e);
                let sigma = rms(&rotated).max(1e-8);
                let (quantized, tiles) = quantize_matrix(&rotated, rows, cols, u_ref, sigma)?;
                let reconstructed = rht.inverse(&quantized);
                sum_rel += distortion(&w_e, &reconstructed).relative;
                if let Some(h_ref) = expert_h_owned.as_deref().or(shared_h.as_deref()) {
                    sum_proxy += proxy_error(&w_e, &reconstructed, rows, cols, h_ref);
                    has_proxy = true;
                }
                for w in tiles.concat() {
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
                proxy_error: has_proxy.then_some(sum_proxy / n_experts as f64),
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
        let has_h = hessians.contains(name);
        let h = if opts.fold_residual_rht
            && let Some(norm_name) = rmsnorm_name_for(name)
            && let Some(norm_tensor) = model.tensors.get(&norm_name)
        {
            let mut gamma = vec![0.0f32; cols];
            norm_tensor.row(0, &mut gamma);
            for row in weights.chunks_exact_mut(cols) {
                for (w, &g) in row.iter_mut().zip(&gamma) {
                    *w *= g;
                }
            }
            if has_h {
                let mut h_raw = hessians.load(name, cols, 0.0)?;
                fold_rmsnorm_hessian(&mut h_raw, &gamma, cols, f64::from(opts.damping));
                h_raw
            } else {
                let mut h_id = vec![0.0f64; cols * cols];
                for i in 0..cols {
                    h_id[i * cols + i] = 1.0 + f64::from(opts.damping);
                }
                h_id
            }
        } else if has_h {
            hessians.load(name, cols, f64::from(opts.damping))?
        } else {
            let mut h_id = vec![0.0f64; cols * cols];
            for i in 0..cols {
                h_id[i * cols + i] = 1.0 + f64::from(opts.damping);
            }
            h_id
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

        let (quantized, tiles) = quantize_matrix(&rotated, rows, cols, &u, sigma)?;

        let reconstructed = rht.inverse(&quantized);
        let report = TensorReport {
            name: name.clone(),
            relative_error: distortion(&weights, &reconstructed).relative,
            proxy_error: Some(proxy_error(&weights, &reconstructed, rows, cols, &h)),
        };
        progress(&report, idx + 1, names.len());
        reports.push(report);

        let tile_bytes: Vec<u8> = tiles
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

/// Summary returned by [`pack_palito_bundle`].
#[derive(Debug, Clone)]
pub struct BundleReport {
    pub skeleton_bytes: u64,
    pub eagle3_tensors: usize,
    pub total_bytes: u64,
}

/// Packs a base GGUF (stripped to tokenizer/metadata/embeddings/norms/LM head),
/// a `PLT1` layers file, and an optional `EAGLE-3` draft head into one
/// self-contained `.palito` file.
pub fn pack_palito_bundle(
    base_gguf: &Path,
    plt1_layers: &Path,
    eagle3: Option<&Path>,
    output: &Path,
) -> Result<BundleReport> {
    use std::io::{Read, Seek, SeekFrom};

    let skeleton_tmp = output.with_extension("skeleton.gguf.tmp");
    strip_projections_to_skeleton_gguf(base_gguf, &skeleton_tmp)?;
    let skeleton_bytes = std::fs::read(&skeleton_tmp)?;
    let _ = std::fs::remove_file(&skeleton_tmp);
    let skeleton_len = skeleton_bytes.len() as u64;

    let mut layers_file = File::open(plt1_layers)?;
    let layers_idx = Plt1Index::read_from(&mut layers_file)?;
    let mut header = layers_idx.header.clone();
    if eagle3.is_some() {
        header.flags |= flags::HAS_EAGLE3;
    }

    let mut writer = Plt1Writer::new(header);
    writer.add_section(SectionKind::BaseGguf, 0, skeleton_bytes);

    let read_slice = |f: &mut File, off: u64, size: u64| -> std::io::Result<Vec<u8>> {
        f.seek(SeekFrom::Start(off))?;
        let mut buf = vec![0u8; size as usize];
        f.read_exact(&mut buf)?;
        Ok(buf)
    };

    if let Some(meta) = layers_idx
        .sections
        .iter()
        .find(|s| s.kind == SectionKind::MetadataJson)
    {
        let data = read_slice(&mut layers_file, meta.offset, meta.size)?;
        writer.add_section(SectionKind::MetadataJson, meta.flags, data);
    }

    for desc in &layers_idx.tensors {
        let data = read_slice(&mut layers_file, desc.data_offset, desc.data_size)?;
        let lut = if desc.lut_offset != 0 && desc.lut_bits > 0 {
            let lut_bytes = read_slice(
                &mut layers_file,
                desc.lut_offset,
                (1u64 << desc.lut_bits) * 4,
            )?;
            Some(
                lut_bytes
                    .chunks_exact(4)
                    .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
                    .collect(),
            )
        } else {
            None
        };
        writer.add_tensor(desc.clone(), data, lut);
    }

    let mut eagle3_tensors = 0usize;
    if let Some(eagle_path) = eagle3 {
        let mut ef = File::open(eagle_path)?;
        let eidx = Plt1Index::read_from(&mut ef)?;
        let esec = eidx
            .sections
            .iter()
            .find(|s| s.kind == SectionKind::Eagle3SpecHead)
            .ok_or_else(|| {
                Error::Invalid(format!(
                    "eagle3 file {} has no Eagle3SpecHead section",
                    eagle_path.display()
                ))
            })?;
        let edata = read_slice(&mut ef, esec.offset, esec.size)?;
        writer.add_section(SectionKind::Eagle3SpecHead, esec.flags, edata);
        for desc in &eidx.tensors {
            let data = read_slice(&mut ef, desc.data_offset, desc.data_size)?;
            writer.add_tensor(desc.clone(), data, None);
        }
        eagle3_tensors = eidx.tensors.len();
    }

    let partial = output.with_extension("palito.partial");
    {
        let out_file = File::create(&partial)?;
        let mut bw = BufWriter::new(out_file);
        writer.write_to(&mut bw)?;
    }
    std::fs::rename(&partial, output)?;
    let total_bytes = std::fs::metadata(output)?.len();
    Ok(BundleReport {
        skeleton_bytes: skeleton_len,
        eagle3_tensors,
        total_bytes,
    })
}
