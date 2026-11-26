//! Collect input Hessians `H = X^T X / n` on CUDA using `CudaBackend`'s
//! streaming `CaptureObserver` hook (no PyTorch, no weight dequantization).
//!
//! Works for both dense models and 3D MoE models (e.g. `Qwen3-30B-A3B-Q4_K_M.gguf`):
//! - `blk.{l}.attn_qkv.f32` (`d_model x d_model`, shared for `attn_q/k/v`)
//! - `blk.{l}.attn_output.f32` (`attn_out_in x attn_out_in`, for `attn_output`)
//! - `blk.{l}.ffn_gate_up_exps.f32` (`d_model x d_model`, shared for `ffn_gate_exps/up_exps`)
//!   or `blk.{l}.ffn_gate_up.f32` for dense layers
//! - `blk.{l}.ffn_down_exps.f32` (packed `n_experts x d_ff x d_ff` f32, with
//!   identity blend when an expert sees fewer than `--min-expert-tokens` tokens)
//!   or `blk.{l}.ffn_down.f32` for dense layers
//! - `manifest.json` compatible with `palito::qtip::hessian::Hessians`.

use clap::Parser;
use palito::{
    Config, Error, Result,
    backend::Backend,
    cuda::{
        CudaBackend, CudaMatmul, CudaOptions,
        diagnostics::{CaptureObserver, LayerDims, ObserverOptions},
    },
};
use rayon::prelude::*;
use serde::Serialize;
use std::{
    collections::BTreeMap,
    fs::{self, File},
    io::{BufWriter, Write},
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
    time::Instant,
};

#[derive(Parser)]
struct Args {
    /// Path to the GGUF model.
    #[arg(long)]
    model: PathBuf,
    /// Path to calibration text (e.g. `models/wikitext-2-raw/wiki.train.raw`).
    #[arg(long)]
    text: PathBuf,
    /// Output directory for `manifest.json` and `.f32` Hessian files.
    #[arg(long)]
    output: PathBuf,
    /// Number of calibration chunks to process.
    #[arg(long, default_value_t = 128)]
    chunks: usize,
    /// Tokens per calibration chunk.
    #[arg(long, default_value_t = 512)]
    context: u32,
    /// Number of layers to accumulate per forward pass (0 = all layers in one pass).
    #[arg(long, default_value_t = 16)]
    layers_per_pass: usize,
    /// Skip collecting per-expert `ffn_down_exps.f32` (saves RAM/disk/time).
    #[arg(long)]
    skip_down_exps: bool,
    /// Blend per-expert `ffn_down_exps` Hessians toward scaled identity when
    /// an expert observes fewer than this many tokens.
    #[arg(long, default_value_t = 32)]
    min_expert_tokens: u64,
    /// CUDA matmul profile.
    #[arg(long, value_enum, default_value_t = CudaMatmul::F32)]
    matmul: CudaMatmul,
    /// CPU threads for tokenization / backend setup.
    #[arg(long, default_value_t = 8)]
    threads: u32,
}

#[derive(Serialize)]
struct ManifestEntry {
    file: String,
    dim: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    experts: Option<usize>,
}

#[derive(Serialize)]
struct Manifest {
    tokens: u64,
    tensors: BTreeMap<String, ManifestEntry>,
}

struct LayerAccum {
    spec: LayerDims,
    attn_qkv: Vec<f64>,
    attn_out: Vec<f64>,
    ffn_in: Vec<f64>,
    /// For dense layers: `ffn_down_dim * ffn_down_dim`.
    ffn_down_dense: Vec<f64>,
    /// For MoE layers (when enabled): `n_experts` matrices of `ffn_down_dim * ffn_down_dim`.
    ffn_down_exps: Vec<Vec<f64>>,
    ffn_down_exp_counts: Vec<u64>,
    /// Most recent routing IDs for this layer's current chunk (`tokens * used`).
    last_ids: Vec<i32>,
    last_used: usize,
}

impl LayerAccum {
    fn new(spec: LayerDims, collect_down_exps: bool) -> Self {
        let ffn_down_dense = if spec.n_experts == 0 {
            vec![0.0; spec.ffn_down_dim * spec.ffn_down_dim]
        } else {
            Vec::new()
        };
        let (ffn_down_exps, ffn_down_exp_counts) = if spec.n_experts > 0 && collect_down_exps {
            (
                (0..spec.n_experts)
                    .map(|_| vec![0.0; spec.ffn_down_dim * spec.ffn_down_dim])
                    .collect(),
                vec![0u64; spec.n_experts],
            )
        } else {
            (Vec::new(), Vec::new())
        };
        Self {
            attn_qkv: vec![0.0; spec.attn_qkv_dim * spec.attn_qkv_dim],
            attn_out: vec![0.0; spec.attn_out_dim * spec.attn_out_dim],
            ffn_in: vec![0.0; spec.ffn_in_dim * spec.ffn_in_dim],
            ffn_down_dense,
            ffn_down_exps,
            ffn_down_exp_counts,
            last_ids: Vec::new(),
            last_used: 0,
            spec,
        }
    }

    fn estimated_bytes(&self) -> usize {
        let f64_elems = self.attn_qkv.len()
            + self.attn_out.len()
            + self.ffn_in.len()
            + self.ffn_down_dense.len()
            + self.ffn_down_exps.iter().map(Vec::len).sum::<usize>();
        f64_elems * std::mem::size_of::<f64>()
    }
}

struct PassState {
    first_layer: usize,
    layers: Vec<LayerAccum>,
    collect_down_exps: bool,
}

struct HessianObserver {
    state: Arc<Mutex<PassState>>,
}

/// Accumulates the upper triangle (`j >= i`) of `X^T X` in f64, where `X` is
/// row-major `[tokens, dim]`.
fn accumulate_gram_upper(h: &mut [f64], x: &[f32], tokens: usize, dim: usize) {
    h.par_chunks_mut(dim).enumerate().for_each(|(i, h_row)| {
        let dst = &mut h_row[i..];
        for t in 0..tokens {
            let x_row = &x[t * dim..t * dim + dim];
            let xi = f64::from(x_row[i]);
            let src = &x_row[i..];
            for (d, &xj) in dst.iter_mut().zip(src.iter()) {
                *d += xi * f64::from(xj);
            }
        }
    });
}

impl CaptureObserver for HessianObserver {
    fn on_routing(
        &mut self,
        layer: usize,
        _position: usize,
        _tokens: usize,
        experts: usize,
        used: usize,
        ids: &[i32],
    ) -> Result<()> {
        let mut guard = self.state.lock().unwrap();
        if !guard.collect_down_exps {
            return Ok(());
        }
        let local = layer - guard.first_layer;
        let acc = &mut guard.layers[local];
        if acc.spec.n_experts != experts {
            return Err(Error::Invalid(format!(
                "layer {layer}: routing experts {experts} != GGUF experts {}",
                acc.spec.n_experts
            )));
        }
        acc.last_ids.clear();
        acc.last_ids.extend_from_slice(ids);
        acc.last_used = used;
        Ok(())
    }

    fn on_projection(
        &mut self,
        layer: usize,
        _position: usize,
        tokens: usize,
        stage: &'static str,
        width: usize,
        values: &[f32],
    ) -> Result<()> {
        let mut guard = self.state.lock().unwrap();
        let collect_down_exps = guard.collect_down_exps;
        let local = layer - guard.first_layer;
        let acc = &mut guard.layers[local];
        match stage {
            "attn_norm" => {
                if width != acc.spec.attn_qkv_dim {
                    return Err(Error::Invalid(format!(
                        "layer {layer}: attn_norm width {width} != expected {}",
                        acc.spec.attn_qkv_dim
                    )));
                }
                accumulate_gram_upper(&mut acc.attn_qkv, values, tokens, width);
            }
            "attn_attended" => {
                if width != acc.spec.attn_out_dim {
                    return Err(Error::Invalid(format!(
                        "layer {layer}: attn_attended width {width} != expected {}",
                        acc.spec.attn_out_dim
                    )));
                }
                accumulate_gram_upper(&mut acc.attn_out, values, tokens, width);
            }
            "ffn_norm" => {
                if width != acc.spec.ffn_in_dim {
                    return Err(Error::Invalid(format!(
                        "layer {layer}: ffn_norm width {width} != expected {}",
                        acc.spec.ffn_in_dim
                    )));
                }
                accumulate_gram_upper(&mut acc.ffn_in, values, tokens, width);
            }
            "ffn_down_inp" => {
                if acc.spec.n_experts == 0 {
                    if width != acc.spec.ffn_down_dim {
                        return Err(Error::Invalid(format!(
                            "layer {layer}: ffn_down_inp width {width} != expected {}",
                            acc.spec.ffn_down_dim
                        )));
                    }
                    accumulate_gram_upper(&mut acc.ffn_down_dense, values, tokens, width);
                } else if collect_down_exps {
                    let used = acc.last_used;
                    let hidden = acc.spec.ffn_down_dim;
                    if used == 0 || width != hidden * used || acc.last_ids.len() != tokens * used {
                        return Err(Error::Invalid(format!(
                            "layer {layer}: ffn_down_inp shape mismatch (width={width}, hidden={hidden}, used={used}, ids={})",
                            acc.last_ids.len()
                        )));
                    }
                    let n_experts = acc.spec.n_experts;
                    let mut buckets: Vec<Vec<usize>> = vec![Vec::new(); n_experts];
                    for (assignment, &eid) in acc.last_ids.iter().enumerate() {
                        if eid < 0 || eid as usize >= n_experts {
                            return Err(Error::Invalid(format!(
                                "layer {layer}: invalid expert id {eid}"
                            )));
                        }
                        buckets[eid as usize].push(assignment);
                    }
                    acc.ffn_down_exps
                        .par_iter_mut()
                        .zip(acc.ffn_down_exp_counts.par_iter_mut())
                        .zip(buckets.par_iter())
                        .for_each(|((h_e, count), assignments)| {
                            if assignments.is_empty() {
                                return;
                            }
                            *count += assignments.len() as u64;
                            for &assignment in assignments {
                                let x_row = &values[assignment * hidden..(assignment + 1) * hidden];
                                for i in 0..hidden {
                                    let xi = f64::from(x_row[i]);
                                    let dst = &mut h_e[i * hidden + i..i * hidden + hidden];
                                    let src = &x_row[i..hidden];
                                    for (d, &xj) in dst.iter_mut().zip(src.iter()) {
                                        *d += xi * f64::from(xj);
                                    }
                                }
                            }
                        });
                }
            }
            _ => {}
        }
        Ok(())
    }
}

/// Finalizes an upper-triangular `dim x dim` f64 sum matrix in place by
/// mirroring the upper triangle, dividing by `count`, and writing little-endian
/// f32 values to `writer`. If `min_tokens > 0 && count < min_tokens`, blends
/// toward scaled identity `scale * I` with weight `alpha = count / min_tokens`.
fn finalize_and_write_matrix(
    h: &mut [f64],
    dim: usize,
    count: u64,
    min_tokens: u64,
    writer: &mut BufWriter<File>,
) -> Result<()> {
    if count == 0 {
        h.fill(0.0);
        for i in 0..dim {
            h[i * dim + i] = 1.0;
        }
    } else {
        let inv_n = 1.0 / count as f64;
        for i in 0..dim {
            h[i * dim + i] *= inv_n;
            for j in (i + 1)..dim {
                let v = h[i * dim + j] * inv_n;
                h[i * dim + j] = v;
                h[j * dim + i] = v;
            }
        }
        let mean_diag = (0..dim).map(|i| h[i * dim + i]).sum::<f64>() / dim as f64;
        if !mean_diag.is_finite() || mean_diag <= 0.0 {
            h.fill(0.0);
            for i in 0..dim {
                h[i * dim + i] = 1.0;
            }
        } else if min_tokens > 0 && count < min_tokens {
            let alpha = count as f64 / min_tokens as f64;
            let blend_diag = (1.0 - alpha) * mean_diag;
            for v in h.iter_mut() {
                *v *= alpha;
            }
            for i in 0..dim {
                h[i * dim + i] += blend_diag;
            }
        }
    }

    let mut row_bytes = vec![0u8; dim * 4];
    for i in 0..dim {
        let row = &h[i * dim..(i + 1) * dim];
        for (dst, &val) in row_bytes.chunks_exact_mut(4).zip(row.iter()) {
            dst.copy_from_slice(&(val as f32).to_le_bytes());
        }
        writer.write_all(&row_bytes)?;
    }
    Ok(())
}

fn write_single_matrix(path: &Path, h: &mut [f64], dim: usize, count: u64) -> Result<()> {
    let mut writer = BufWriter::new(File::create(path)?);
    finalize_and_write_matrix(h, dim, count, 0, &mut writer)?;
    writer.flush()?;
    Ok(())
}

fn main() -> Result<()> {
    let args = Args::parse();
    if args.chunks == 0 || args.context < 2 {
        return Err(Error::Invalid(
            "--chunks must be >= 1 and --context must be >= 2".into(),
        ));
    }

    let collect_down_exps = !args.skip_down_exps;
    let prefill_chunk = args.context.min(512);
    if args.context % prefill_chunk == 1 {
        return Err(Error::Invalid(format!(
            "--context {} leaves a singleton tail for prefill chunk {prefill_chunk}",
            args.context
        )));
    }

    fs::create_dir_all(&args.output)?;

    let config = Config {
        context_tokens: args.context,
        batch_tokens: prefill_chunk,
        prefill_chunk,
        threads: args.threads,
    };
    config.validate()?;

    let t_open = Instant::now();
    let mut backend = CudaBackend::open(
        &args.model,
        &config,
        CudaOptions {
            matmul: args.matmul,
            decode_graph: false,
            ..CudaOptions::default()
        },
    )?;
    let layer_specs = backend.layer_dims();
    let n_layers = layer_specs.len();
    let layers_per_pass = if args.layers_per_pass == 0 {
        n_layers
    } else {
        args.layers_per_pass.min(n_layers)
    };
    eprintln!(
        "loaded {} ({n_layers} layers) in {:.1}s",
        args.model.display(),
        t_open.elapsed().as_secs_f64()
    );

    let bos = backend.tokenize("")?.first().copied();
    let text = fs::read_to_string(&args.text)?;
    let mut tokens = backend.tokenize(&text)?;
    if let Some(b) = bos
        && tokens.first() == Some(&b)
    {
        tokens.remove(0);
    }
    let context = args.context as usize;
    let available = tokens.len() / context;
    let chunks = args.chunks.min(available);
    if chunks == 0 {
        return Err(Error::Invalid(
            "calibration text is shorter than one context chunk".into(),
        ));
    }
    let total_tokens = (chunks * context) as u64;
    let n_passes = n_layers.div_ceil(layers_per_pass);
    eprintln!(
        "collecting over {chunks} chunks x {context} tokens = {total_tokens} tokens ({n_passes} pass(es), {layers_per_pass} layers/pass, down_exps={collect_down_exps})"
    );

    let mut manifest_tensors = BTreeMap::<String, ManifestEntry>::new();
    let mut total_blended_experts = 0usize;
    let mut total_zero_experts = 0usize;
    let mut total_expert_slots = 0usize;
    let mut peak_accum_bytes = 0usize;
    let t_start = Instant::now();

    let mut stages = vec![
        "attn_norm".to_string(),
        "attn_attended".to_string(),
        "ffn_norm".to_string(),
    ];
    let has_any_dense = layer_specs.iter().any(|s| s.n_experts == 0);
    let has_any_moe = layer_specs.iter().any(|s| s.n_experts > 0);
    if has_any_dense || (has_any_moe && collect_down_exps) {
        stages.push("ffn_down_inp".to_string());
    }

    for pass_idx in 0..n_passes {
        let start_layer = pass_idx * layers_per_pass;
        let end_layer = ((pass_idx + 1) * layers_per_pass).min(n_layers);
        let pass_layers: Vec<usize> = (start_layer..end_layer).collect();
        let accums: Vec<LayerAccum> = layer_specs[start_layer..end_layer]
            .iter()
            .cloned()
            .map(|s| LayerAccum::new(s, collect_down_exps))
            .collect();
        let pass_bytes: usize = accums.iter().map(LayerAccum::estimated_bytes).sum();
        peak_accum_bytes = peak_accum_bytes.max(pass_bytes);

        let pass_has_moe = layer_specs[start_layer..end_layer]
            .iter()
            .any(|s| s.n_experts > 0);
        let state = Arc::new(Mutex::new(PassState {
            first_layer: start_layer,
            layers: accums,
            collect_down_exps,
        }));

        let t_pass = Instant::now();
        for chunk in 0..chunks {
            let mut window = tokens[chunk * context..(chunk + 1) * context].to_vec();
            if let Some(b) = bos {
                window[0] = b;
            }
            backend.reset()?;
            backend.enable_capture_observer(
                ObserverOptions {
                    layers: pass_layers.clone(),
                    stages: stages.clone(),
                    routing: pass_has_moe && collect_down_exps,
                },
                Box::new(HessianObserver {
                    state: Arc::clone(&state),
                }),
            )?;
            let step = prefill_chunk as usize;
            for (sub_idx, sub) in window.chunks(step).enumerate() {
                backend.evaluate(sub, (sub_idx * step) as u32)?;
            }
            if (chunk + 1) % 8 == 0 || chunk + 1 == chunks {
                eprintln!(
                    "pass {}/{n_passes} layers {start_layer}..{end_layer} [{}/{chunks}] {:.1}s (accum {:.2} GiB)",
                    pass_idx + 1,
                    chunk + 1,
                    t_pass.elapsed().as_secs_f64(),
                    pass_bytes as f64 / (1024.0 * 1024.0 * 1024.0),
                );
            }
        }
        // Detach observer before taking exclusive ownership of pass accumulators.
        backend.reset()?;

        let mut guard = state.lock().unwrap();
        for acc in &mut guard.layers {
            let l = acc.spec.index;
            let qkv_file = format!("blk.{l}.attn_qkv.f32");
            write_single_matrix(
                &args.output.join(&qkv_file),
                &mut acc.attn_qkv,
                acc.spec.attn_qkv_dim,
                total_tokens,
            )?;
            for proj in ["attn_q", "attn_k", "attn_v"] {
                manifest_tensors.insert(
                    format!("blk.{l}.{proj}.weight"),
                    ManifestEntry {
                        file: qkv_file.clone(),
                        dim: acc.spec.attn_qkv_dim,
                        experts: None,
                    },
                );
            }

            let out_file = format!("blk.{l}.attn_output.f32");
            write_single_matrix(
                &args.output.join(&out_file),
                &mut acc.attn_out,
                acc.spec.attn_out_dim,
                total_tokens,
            )?;
            manifest_tensors.insert(
                format!("blk.{l}.attn_output.weight"),
                ManifestEntry {
                    file: out_file,
                    dim: acc.spec.attn_out_dim,
                    experts: None,
                },
            );

            if acc.spec.n_experts > 0 {
                let gu_file = format!("blk.{l}.ffn_gate_up_exps.f32");
                write_single_matrix(
                    &args.output.join(&gu_file),
                    &mut acc.ffn_in,
                    acc.spec.ffn_in_dim,
                    total_tokens,
                )?;
                for proj in ["ffn_gate_exps", "ffn_up_exps"] {
                    manifest_tensors.insert(
                        format!("blk.{l}.{proj}.weight"),
                        ManifestEntry {
                            file: gu_file.clone(),
                            dim: acc.spec.ffn_in_dim,
                            experts: None,
                        },
                    );
                }

                if collect_down_exps {
                    let down_file = format!("blk.{l}.ffn_down_exps.f32");
                    let mut writer = BufWriter::new(File::create(args.output.join(&down_file))?);
                    for (e, h_e) in acc.ffn_down_exps.iter_mut().enumerate() {
                        let count = acc.ffn_down_exp_counts[e];
                        total_expert_slots += 1;
                        if count == 0 {
                            total_zero_experts += 1;
                        } else if count < args.min_expert_tokens {
                            total_blended_experts += 1;
                        }
                        finalize_and_write_matrix(
                            h_e,
                            acc.spec.ffn_down_dim,
                            count,
                            args.min_expert_tokens,
                            &mut writer,
                        )?;
                    }
                    writer.flush()?;
                    manifest_tensors.insert(
                        format!("blk.{l}.ffn_down_exps.weight"),
                        ManifestEntry {
                            file: down_file,
                            dim: acc.spec.ffn_down_dim,
                            experts: Some(acc.spec.n_experts),
                        },
                    );
                }
            } else {
                let gu_file = format!("blk.{l}.ffn_gate_up.f32");
                write_single_matrix(
                    &args.output.join(&gu_file),
                    &mut acc.ffn_in,
                    acc.spec.ffn_in_dim,
                    total_tokens,
                )?;
                for proj in ["ffn_gate", "ffn_up"] {
                    manifest_tensors.insert(
                        format!("blk.{l}.{proj}.weight"),
                        ManifestEntry {
                            file: gu_file.clone(),
                            dim: acc.spec.ffn_in_dim,
                            experts: None,
                        },
                    );
                }

                let down_file = format!("blk.{l}.ffn_down.f32");
                write_single_matrix(
                    &args.output.join(&down_file),
                    &mut acc.ffn_down_dense,
                    acc.spec.ffn_down_dim,
                    total_tokens,
                )?;
                manifest_tensors.insert(
                    format!("blk.{l}.ffn_down.weight"),
                    ManifestEntry {
                        file: down_file,
                        dim: acc.spec.ffn_down_dim,
                        experts: None,
                    },
                );
            }
        }
    }

    let manifest = Manifest {
        tokens: total_tokens,
        tensors: manifest_tensors,
    };
    let manifest_path = args.output.join("manifest.json");
    fs::write(&manifest_path, serde_json::to_string_pretty(&manifest)?)?;

    eprintln!(
        "done in {:.1}s: {} ({} tensors, peak accum {:.2} GiB)",
        t_start.elapsed().as_secs_f64(),
        manifest_path.display(),
        manifest.tensors.len(),
        peak_accum_bytes as f64 / (1024.0 * 1024.0 * 1024.0),
    );
    if total_expert_slots > 0 {
        eprintln!(
            "MoE ffn_down_exps stats: {total_expert_slots} expert matrices, {total_blended_experts} blended (<{} tokens), {total_zero_experts} identity-fallback (0 tokens)",
            args.min_expert_tokens
        );
    }
    Ok(())
}
