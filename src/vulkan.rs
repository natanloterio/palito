//! Native Vulkan compute backend (`--features vulkan`, `--backend vulkan`).
//!
//! Uses `ash` (dynamic `libvulkan` loading, no build-time Vulkan SDK) and `naga`
//! (in-process WGSL -> SPIR-V compilation) to run GGUF models on AMD, Intel, and
//! NVIDIA GPUs across Linux and Windows.

mod device;
mod pipeline;

use crate::{
    Config, Error, Result,
    backend::{Backend, MemoryUsage, ModelInfo},
    gguf::{Storage, Tensor},
    transformer::{Weights, invalid},
};
use device::{Buffer, VkContext, failure};
use pipeline::{Pipelines, PushParams};
use serde::Serialize;
use std::path::Path;

#[derive(Clone, Debug, Serialize)]
pub struct VulkanOptions {
    pub device: usize,
    pub memory_limit_bytes: u64,
}

impl Default for VulkanOptions {
    fn default() -> Self {
        Self {
            device: 0,
            memory_limit_bytes: u64::MAX,
        }
    }
}

struct DeviceTensor {
    buf: Buffer,
    kind: u32,
    width: u32,
    rows: u32,
}

#[derive(Default)]
struct State {
    position: usize,
    valid: bool,
    failed: bool,
}

impl State {
    fn begin(&mut self, tokens: &[i32], position: u32, info: &ModelInfo) -> Result<()> {
        if self.failed {
            return Err(failure("failed request must be reset"));
        }
        if tokens.is_empty()
            || tokens.len() > info.batch_tokens as usize
            || position as usize != self.position
            || self
                .position
                .checked_add(tokens.len())
                .is_none_or(|n| n > info.context_tokens as usize)
            || tokens
                .iter()
                .any(|&id| id < 0 || id as u32 >= info.vocab_tokens)
        {
            return Err(invalid("invalid IDs, batch, position, or context capacity"));
        }
        self.valid = false;
        self.failed = true;
        Ok(())
    }

    fn finish(&mut self, count: usize) {
        self.position += count;
        self.valid = true;
        self.failed = false;
    }

    fn logits<'a>(&self, logits: &'a [f32]) -> Result<&'a [f32]> {
        if !self.valid || self.failed {
            return Err(failure("no valid next-token logits"));
        }
        Ok(logits)
    }
}

struct Workspace {
    x: Buffer,
    normalized: Buffer,
    q: Buffer,
    k: Buffer,
    v: Buffer,
    attended: Buffer,
    projected: Buffer,
    gate: Buffer,
    up: Buffer,
    down: Buffer,
    router: Buffer,
    ids: Buffer,
    token_ids: Buffer,
    token_ids_staging: Buffer,
    probabilities: Buffer,
    logits: Buffer,
    logits_staging: Buffer,
    batch_cap: usize,
}

pub struct VulkanBackend {
    ctx: VkContext,
    pipelines: Pipelines,
    weights: Weights<DeviceTensor>,
    keys: Vec<Buffer>,
    values: Vec<Buffer>,
    work: Workspace,
    logits: Vec<f32>,
    state: State,
    memory: MemoryUsage,
}

fn upload_tensor(ctx: &VkContext, tensor: Tensor) -> Result<DeviceTensor> {
    let width = u32::try_from(tensor.dims.first().copied().unwrap_or(0))
        .map_err(|_| invalid("tensor width overflow"))?;
    let rows = u32::try_from(tensor.dims.get(1).copied().unwrap_or(1))
        .map_err(|_| invalid("tensor rows overflow"))?;
    let (kind, bytes): (u32, &[u8]) = match &tensor.storage {
        Storage::F32(v) => (0, unsafe {
            std::slice::from_raw_parts(v.as_ptr().cast::<u8>(), std::mem::size_of_val(v.as_slice()))
        }),
        Storage::Packed { kind, bytes } => {
            if !matches!(*kind, 1 | 6 | 8 | 12 | 13 | 14) {
                return Err(invalid(format!(
                    "Vulkan backend does not yet support tensor kind {kind}"
                )));
            }
            (*kind, bytes.as_slice())
        }
    };
    let buf = ctx.upload_bytes(bytes)?;
    Ok(DeviceTensor {
        buf,
        kind,
        width,
        rows,
    })
}

fn greedy(logits: &[f32]) -> Result<i32> {
    if logits.is_empty() || logits.iter().any(|x| !x.is_finite()) {
        return Err(failure("invalid logits"));
    }
    let mut best = 0;
    for (i, &value) in logits.iter().enumerate().skip(1) {
        if value > logits[best] {
            best = i;
        }
    }
    Ok(best as i32)
}

pub fn open_backend(
    path: &Path,
    config: &Config,
    options: VulkanOptions,
) -> Result<Box<dyn Backend>> {
    Ok(Box::new(VulkanBackend::open(path, config, options)?))
}

impl VulkanBackend {
    pub fn open(path: &Path, config: &Config, options: VulkanOptions) -> Result<Self> {
        config.validate()?;
        if options.memory_limit_bytes == 0 || config.context_tokens > 65535 {
            return Err(invalid(
                "Vulkan requires a positive device budget and context <= 65535",
            ));
        }
        crate::plt1::reject_on_backend(path, "Vulkan")?;
        let ctx = VkContext::new(options.device)?;
        let mut raw = Weights::open(path, config)?;
        if raw.position_embedding.is_some()
            || raw.embedding_norm.is_some()
            || raw.output_norm_bias.is_some()
            || raw.layers.iter().any(|l| l.affine.is_some() || l.gate.is_none())
        {
            return Err(invalid(format!(
                "Vulkan backend does not yet support {} architecture",
                raw.spec.info.architecture
            )));
        }
        if raw.spec.dim > 256 {
            return Err(invalid(format!(
                "Vulkan backend requires head_dim <= 256, got {}",
                raw.spec.dim
            )));
        }
        raw.spec.info.backend = format!("vulkan ({})", ctx.device_name());
        let s = &raw.spec;
        let max_dispatches = (raw.layers.len() as u32) * 24 + 16;
        let pipelines = Pipelines::compile(ctx.inner.clone(), max_dispatches)?;

        let kv_elements = s.info.context_tokens as u64 * s.kv_heads as u64 * s.dim as u64;
        let kv_bytes_per_layer = kv_elements * 4;
        let kv_total_bytes = kv_bytes_per_layer * 2 * raw.layers.len() as u64;
        let mut weights_bytes = 0u64;
        let weights = raw.map(|t| {
            let dt = upload_tensor(&ctx, t)?;
            weights_bytes += dt.buf.size;
            Ok(dt)
        })?;

        let s = &weights.spec;
        let alloc_f32 = |n: usize| ctx.alloc_zeros((n as u64) * 4);
        let qw = s.heads * s.dim;
        let kw = s.kv_heads * s.dim;
        let batch_cap = if s.experts == 0 {
            (s.info.batch_tokens as usize).clamp(1, 64)
        } else {
            1
        };
        let work = Workspace {
            x: alloc_f32(s.width * batch_cap)?,
            normalized: alloc_f32(s.width * batch_cap)?,
            q: alloc_f32(qw * batch_cap)?,
            k: alloc_f32(kw * batch_cap)?,
            v: alloc_f32(kw * batch_cap)?,
            attended: alloc_f32(qw * batch_cap)?,
            projected: alloc_f32(s.width * batch_cap)?,
            gate: alloc_f32(s.used * s.hidden * batch_cap)?,
            up: alloc_f32(s.used * s.hidden * batch_cap)?,
            down: alloc_f32(s.used * s.width * batch_cap)?,
            router: alloc_f32(s.experts.max(1))?,
            ids: ctx.alloc_zeros((s.used.max(1) as u64) * 4)?,
            token_ids: ctx.alloc_zeros((batch_cap as u64) * 4)?,
            token_ids_staging: ctx.alloc_host_visible((batch_cap as u64) * 4)?,
            probabilities: alloc_f32(s.used.max(1))?,
            logits: alloc_f32(s.info.vocab_tokens as usize)?,
            logits_staging: ctx.alloc_host_visible((s.info.vocab_tokens as u64) * 4)?,
            batch_cap,
        };
        let workspace_bytes = work.x.size
            + work.normalized.size
            + work.q.size
            + work.k.size
            + work.v.size
            + work.attended.size
            + work.projected.size
            + work.gate.size
            + work.up.size
            + work.down.size
            + work.router.size
            + work.ids.size
            + work.probabilities.size;
        let logits_bytes = work.logits.size;

        let mut keys = Vec::with_capacity(weights.layers.len());
        let mut values = Vec::with_capacity(weights.layers.len());
        for _ in 0..weights.layers.len() {
            keys.push(ctx.alloc_zeros(kv_bytes_per_layer)?);
            values.push(ctx.alloc_zeros(kv_bytes_per_layer)?);
        }
        let reservation = weights_bytes + kv_total_bytes + workspace_bytes + logits_bytes;
        if reservation > options.memory_limit_bytes {
            return Err(Error::Invalid(format!(
                "Vulkan reservation ({reservation} B) exceeds limit ({} B)",
                options.memory_limit_bytes
            )));
        }
        let memory = MemoryUsage {
            allocation_domain: "vulkan",
            weights_allocated_bytes: weights_bytes,
            kv_allocated_bytes: kv_total_bytes,
            kv_live_bytes: 0,
            workspace_allocated_bytes: workspace_bytes,
            logits_allocated_bytes: logits_bytes,
            token_state_allocated_bytes: 0,
            prefix_capacity_bytes: 0,
            prefix_allocated_bytes: 0,
            worker_scratch_bound_bytes: 0,
            numerical_reservation_bytes: reservation,
        };
        let vocab = weights.spec.info.vocab_tokens as usize;
        Ok(Self {
            ctx,
            pipelines,
            weights,
            keys,
            values,
            work,
            logits: vec![0.0; vocab],
            state: State::default(),
            memory,
        })
    }

    fn evaluate_token(&mut self, token: i32, position: u32, compute_logits: bool) -> Result<()> {
        self.pipelines.reset_descriptors()?;
        let s = &self.weights.spec;
        let w = s.width as u32;
        let heads = s.heads as u32;
        let kv_heads = s.kv_heads as u32;
        let dim = s.dim as u32;
        let qw = heads * dim;
        let kw = kv_heads * dim;
        let used = s.used as u32;
        let experts = s.experts as u32;
        let neox = u32::from(s.neox);
        let rotary_dim = s.rotary_dim as u32;
        let olmo2 = s.info.architecture == "olmo2";
        let olmoe = s.info.architecture == "olmoe";
        let gemma = self.weights.gemma.as_ref();
        let dummy = &self.ctx.dummy;

        let mut dispatch_err: Option<Error> = None;
        let logits_bytes = (self.logits.len() * 4) as u64;
        self.ctx.submit_one_shot(|device, cmd| {
            let mut run = |name: &str, bufs: [&Buffer; 6], params: PushParams, grid: (u32, u32, u32)| {
                if dispatch_err.is_none()
                    && let Err(e) = self.pipelines.dispatch(cmd, name, bufs, params, grid)
                {
                    dispatch_err = Some(e);
                }
            };

            // 1. Token embedding
            run(
                "embedding",
                [&self.weights.embedding.buf, dummy, dummy, &self.work.x, dummy, dummy],
                PushParams { p: [token as u32, w, self.weights.embedding.kind, 0, 0, 0, 0, 0], f: [0.0; 4] },
                (w.div_ceil(256), 1, 1),
            );
            if gemma.is_some() {
                run(
                    "scale_vec",
                    [dummy, dummy, dummy, &self.work.x, dummy, dummy],
                    PushParams { p: [w, 0, 0, 0, 0, 0, 0, 0], f: [(s.width as f32).sqrt(), 0.0, 0.0, 0.0] },
                    (w.div_ceil(256), 1, 1),
                );
            }

            // 2. Transformer layers
            for (index, l) in self.weights.layers.iter().enumerate() {
                let gemma_attn = gemma.map(|g| g.attention(s, index));
                let rope_base = gemma_attn.map_or(s.rope_base, |(_, base, _)| base);

                // Attention input norm (olmo2 uses post-norm instead of pre-norm)
                let attn_in = if olmo2 {
                    &self.work.x
                } else if let Some(norm) = &l.norm {
                    run(
                        "rmsnorm",
                        [&norm.buf, &self.work.x, dummy, &self.work.normalized, dummy, dummy],
                        PushParams { p: [w, 0, 0, 0, 0, 0, 0, 0], f: [s.epsilon, 0.0, 0.0, 0.0] },
                        (1, 1, 1),
                    );
                    &self.work.normalized
                } else {
                    &self.work.x
                };

                // Q, K, V projections
                for (tensor, out) in [(&l.q, &self.work.q), (&l.k, &self.work.k), (&l.v, &self.work.v)] {
                    run(
                        "matvec",
                        [&tensor.buf, attn_in, dummy, out, dummy, &self.work.ids],
                        PushParams { p: [tensor.kind, tensor.width, tensor.rows, 0, 0, 0, 0, 0], f: [0.0; 4] },
                        (tensor.rows.div_ceil(8), 1, 1),
                    );
                }
                if let Some(biases) = &l.qkv_bias {
                    for (out, bias) in [(&self.work.q, &biases[0]), (&self.work.k, &biases[1]), (&self.work.v, &biases[2])] {
                        run(
                            "add_row_bias",
                            [&bias.buf, dummy, dummy, out, dummy, dummy],
                            PushParams { p: [bias.width, 0, 0, 0, 0, 0, 0, 0], f: [0.0; 4] },
                            (bias.width.div_ceil(256), 1, 1),
                        );
                    }
                }

                // Q/K norms + RoPE
                let factors_buf = self.weights.rope_factors.as_ref().map_or(dummy, |f| &f.buf);
                let has_factors = u32::from(self.weights.rope_factors.is_some());
                for (norm, vec_buf, size) in [(&l.q_norm, &self.work.q, qw), (&l.k_norm, &self.work.k, kw)] {
                    if let Some(norm) = norm {
                        let norm_w = if olmo2 || olmoe { size } else { dim };
                        run(
                            "head_norm",
                            [&norm.buf, dummy, dummy, vec_buf, dummy, dummy],
                            PushParams { p: [norm_w, 0, 0, 0, 0, 0, 0, 0], f: [s.epsilon, 0.0, 0.0, 0.0] },
                            (size / norm_w, 1, 1),
                        );
                    }
                    let pairs = (size / dim) * (rotary_dim / 2);
                    run(
                        "rotary",
                        [dummy, dummy, factors_buf, vec_buf, dummy, dummy],
                        PushParams {
                            p: [size, dim, position, neox, rotary_dim, has_factors, 0, 0],
                            f: [rope_base, s.rope_attn_factor, 0.0, 0.0],
                        },
                        (pairs.div_ceil(256), 1, 1),
                    );
                }

                // Cache K/V
                run(
                    "cache_kv",
                    [dummy, &self.work.k, &self.work.v, &self.keys[index], &self.values[index], dummy],
                    PushParams { p: [kw, position, 0, 0, 0, 0, 0, 0], f: [0.0; 4] },
                    (kw.div_ceil(256), 1, 1),
                );

                // Causal GQA/MHA Attention
                let (window, scale, softcap) = match gemma_attn {
                    Some((win, _, sc)) => (win as u32, sc, gemma.unwrap().attention_softcap),
                    None => (0u32, (s.dim as f32).sqrt().recip(), 0.0f32),
                };
                run(
                    "attention",
                    [&self.values[index], &self.work.q, &self.keys[index], &self.work.attended, dummy, dummy],
                    PushParams {
                        p: [dim, heads, kv_heads, position + 1, window, 0, 0, 0],
                        f: [scale, softcap, 0.0, 0.0],
                    },
                    (heads, 1, 1),
                );

                // Attention output projection
                run(
                    "matvec",
                    [&l.output.buf, &self.work.attended, dummy, &self.work.projected, dummy, &self.work.ids],
                    PushParams { p: [l.output.kind, l.output.width, l.output.rows, 0, 0, 0, 0, 0], f: [0.0; 4] },
                    (l.output.rows.div_ceil(8), 1, 1),
                );

                // Post-attention norm + residual
                if let Some([post_attn, _]) = &l.post_norms {
                    run(
                        "rmsnorm",
                        [&post_attn.buf, &self.work.projected, dummy, &self.work.normalized, dummy, dummy],
                        PushParams { p: [w, 0, 0, 0, 0, 0, 0, 0], f: [s.epsilon, 0.0, 0.0, 0.0] },
                        (1, 1, 1),
                    );
                    run(
                        "residual",
                        [dummy, &self.work.normalized, dummy, &self.work.x, dummy, dummy],
                        PushParams { p: [w, 0, 0, 0, 0, 0, 0, 0], f: [0.0; 4] },
                        (w.div_ceil(256), 1, 1),
                    );
                } else if olmo2 {
                    if let Some(norm) = &l.norm {
                        run(
                            "rmsnorm",
                            [&norm.buf, &self.work.projected, dummy, &self.work.normalized, dummy, dummy],
                            PushParams { p: [w, 0, 0, 0, 0, 0, 0, 0], f: [s.epsilon, 0.0, 0.0, 0.0] },
                            (1, 1, 1),
                        );
                        run(
                            "residual",
                            [dummy, &self.work.normalized, dummy, &self.work.x, dummy, dummy],
                            PushParams { p: [w, 0, 0, 0, 0, 0, 0, 0], f: [0.0; 4] },
                            (w.div_ceil(256), 1, 1),
                        );
                    }
                } else {
                    run(
                        "residual",
                        [dummy, &self.work.projected, dummy, &self.work.x, dummy, dummy],
                        PushParams { p: [w, 0, 0, 0, 0, 0, 0, 0], f: [0.0; 4] },
                        (w.div_ceil(256), 1, 1),
                    );
                }

                // FFN input norm
                let ffn_in = if olmo2 {
                    &self.work.x
                } else if let Some(ffn_norm) = &l.ffn_norm {
                    run(
                        "rmsnorm",
                        [&ffn_norm.buf, &self.work.x, dummy, &self.work.normalized, dummy, dummy],
                        PushParams { p: [w, 0, 0, 0, 0, 0, 0, 0], f: [s.epsilon, 0.0, 0.0, 0.0] },
                        (1, 1, 1),
                    );
                    &self.work.normalized
                } else {
                    &self.work.x
                };

                // Optional MoE router
                let moe = u32::from(l.router.is_some());
                if let Some(router_t) = &l.router {
                    run(
                        "matvec",
                        [&router_t.buf, ffn_in, dummy, &self.work.router, dummy, &self.work.ids],
                        PushParams { p: [router_t.kind, router_t.width, router_t.rows, 0, 0, 0, 0, 0], f: [0.0; 4] },
                        (router_t.rows.div_ceil(8), 1, 1),
                    );
                    run(
                        "router",
                        [dummy, &self.work.router, dummy, &self.work.probabilities, dummy, &self.work.ids],
                        PushParams { p: [experts, used, u32::from(olmoe), 0, 0, 0, 0, 0], f: [0.0; 4] },
                        (1, 1, 1),
                    );
                }

                // Gate & Up projections + SwiGLU / GeGLU
                let gate = l.gate.as_ref().unwrap();
                run(
                    "matvec",
                    [&gate.buf, ffn_in, dummy, &self.work.gate, dummy, &self.work.ids],
                    PushParams { p: [gate.kind, gate.width, gate.rows, moe, 0, 0, 0, 0], f: [0.0; 4] },
                    (gate.rows.div_ceil(8), used, 1),
                );
                run(
                    "matvec",
                    [&l.up.buf, ffn_in, dummy, &self.work.up, dummy, &self.work.ids],
                    PushParams { p: [l.up.kind, l.up.width, l.up.rows, moe, 0, 0, 0, 0], f: [0.0; 4] },
                    (l.up.rows.div_ceil(8), used, 1),
                );
                let act_count = (s.hidden * s.used) as u32;
                run(
                    "swiglu",
                    [dummy, &self.work.up, dummy, &self.work.gate, dummy, dummy],
                    PushParams { p: [act_count, u32::from(gemma.is_some()), 0, 0, 0, 0, 0, 0], f: [0.0; 4] },
                    (act_count.div_ceil(256), 1, 1),
                );

                // Down projection
                run(
                    "matvec",
                    [&l.down.buf, &self.work.gate, dummy, &self.work.down, dummy, &self.work.ids],
                    PushParams { p: [l.down.kind, l.down.width, l.down.rows, moe, 1, 0, 0, 0], f: [0.0; 4] },
                    (l.down.rows.div_ceil(8), used, 1),
                );

                // Combine experts / post-FFN norm + residual
                if moe != 0 {
                    run(
                        "combine_experts",
                        [dummy, &self.work.down, &self.work.probabilities, &self.work.x, dummy, dummy],
                        PushParams { p: [w, used, 0, 0, 0, 0, 0, 0], f: [0.0; 4] },
                        (w.div_ceil(256), 1, 1),
                    );
                } else if let Some([_, post_ffn]) = &l.post_norms {
                    run(
                        "rmsnorm",
                        [&post_ffn.buf, &self.work.down, dummy, &self.work.normalized, dummy, dummy],
                        PushParams { p: [w, 0, 0, 0, 0, 0, 0, 0], f: [s.epsilon, 0.0, 0.0, 0.0] },
                        (1, 1, 1),
                    );
                    run(
                        "residual",
                        [dummy, &self.work.normalized, dummy, &self.work.x, dummy, dummy],
                        PushParams { p: [w, 0, 0, 0, 0, 0, 0, 0], f: [0.0; 4] },
                        (w.div_ceil(256), 1, 1),
                    );
                } else if olmo2 {
                    if let Some(ffn_norm) = &l.ffn_norm {
                        run(
                            "rmsnorm",
                            [&ffn_norm.buf, &self.work.down, dummy, &self.work.normalized, dummy, dummy],
                            PushParams { p: [w, 0, 0, 0, 0, 0, 0, 0], f: [s.epsilon, 0.0, 0.0, 0.0] },
                            (1, 1, 1),
                        );
                        run(
                            "residual",
                            [dummy, &self.work.normalized, dummy, &self.work.x, dummy, dummy],
                            PushParams { p: [w, 0, 0, 0, 0, 0, 0, 0], f: [0.0; 4] },
                            (w.div_ceil(256), 1, 1),
                        );
                    }
                } else {
                    run(
                        "residual",
                        [dummy, &self.work.down, dummy, &self.work.x, dummy, dummy],
                        PushParams { p: [w, 0, 0, 0, 0, 0, 0, 0], f: [0.0; 4] },
                        (w.div_ceil(256), 1, 1),
                    );
                }
            }

            // 3. Final output norm + LM head
            if compute_logits {
                let final_in = if let Some(norm) = &self.weights.output_norm {
                    run(
                        "rmsnorm",
                        [&norm.buf, &self.work.x, dummy, &self.work.normalized, dummy, dummy],
                        PushParams { p: [w, 0, 0, 0, 0, 0, 0, 0], f: [s.epsilon, 0.0, 0.0, 0.0] },
                        (1, 1, 1),
                    );
                    &self.work.normalized
                } else {
                    &self.work.x
                };
                let head = self.weights.output.as_ref().unwrap_or(&self.weights.embedding);
                run(
                    "matvec",
                    [&head.buf, final_in, dummy, &self.work.logits, dummy, &self.work.ids],
                    PushParams { p: [head.kind, head.width, head.rows, 0, 0, 0, 0, 0], f: [0.0; 4] },
                    (head.rows.div_ceil(8), 1, 1),
                );
                if let Some(gemma) = gemma
                    && gemma.final_softcap > 0.0
                {
                    let vocab = head.rows;
                    run(
                        "softcap",
                        [dummy, dummy, dummy, &self.work.logits, dummy, dummy],
                        PushParams { p: [vocab, 0, 0, 0, 0, 0, 0, 0], f: [gemma.final_softcap, 0.0, 0.0, 0.0] },
                        (vocab.div_ceil(256), 1, 1),
                    );
                }
                unsafe {
                    let barrier = [ash::vk::MemoryBarrier::default()
                        .src_access_mask(ash::vk::AccessFlags::SHADER_WRITE)
                        .dst_access_mask(ash::vk::AccessFlags::TRANSFER_READ)];
                    device.cmd_pipeline_barrier(
                        cmd,
                        ash::vk::PipelineStageFlags::COMPUTE_SHADER,
                        ash::vk::PipelineStageFlags::TRANSFER,
                        ash::vk::DependencyFlags::empty(),
                        &barrier,
                        &[],
                        &[],
                    );
                    let region = [ash::vk::BufferCopy::default().size(logits_bytes)];
                    device.cmd_copy_buffer(
                        cmd,
                        self.work.logits.handle,
                        self.work.logits_staging.handle,
                        &region,
                    );
                }
            }
        })?;
        if let Some(err) = dispatch_err {
            return Err(err);
        }
        if compute_logits {
            self.ctx
                .read_mapped_f32(&self.work.logits_staging, &mut self.logits)?;
            if self.logits.iter().any(|x| !x.is_finite()) {
                return Err(failure("nonfinite logits; request poisoned"));
            }
        }
        Ok(())
    }
    fn evaluate_batch(&mut self, tokens: &[i32], position: u32, compute_logits: bool) -> Result<()> {
        let n = tokens.len() as u32;
        self.ctx.write_mapped_i32(&self.work.token_ids_staging, tokens)?;
        self.pipelines.reset_descriptors()?;
        let s = &self.weights.spec;
        let w = s.width as u32;
        let heads = s.heads as u32;
        let kv_heads = s.kv_heads as u32;
        let dim = s.dim as u32;
        let qw = heads * dim;
        let kw = kv_heads * dim;
        let neox = u32::from(s.neox);
        let rotary_dim = s.rotary_dim as u32;
        let olmo2 = s.info.architecture == "olmo2";
        let gemma = self.weights.gemma.as_ref();
        let dummy = &self.ctx.dummy;
        let token_bytes = (tokens.len() * 4) as u64;
        let logits_bytes = (self.logits.len() * 4) as u64;
        let mut dispatch_err: Option<Error> = None;

        self.ctx.submit_one_shot(|device, cmd| {
            // Copy batch token IDs into device-local buffer.
            unsafe {
                let region = [ash::vk::BufferCopy::default().size(token_bytes)];
                device.cmd_copy_buffer(
                    cmd,
                    self.work.token_ids_staging.handle,
                    self.work.token_ids.handle,
                    &region,
                );
                let barrier = [ash::vk::MemoryBarrier::default()
                    .src_access_mask(ash::vk::AccessFlags::TRANSFER_WRITE)
                    .dst_access_mask(ash::vk::AccessFlags::SHADER_READ)];
                device.cmd_pipeline_barrier(
                    cmd,
                    ash::vk::PipelineStageFlags::TRANSFER,
                    ash::vk::PipelineStageFlags::COMPUTE_SHADER,
                    ash::vk::DependencyFlags::empty(),
                    &barrier,
                    &[],
                    &[],
                );
            }

            let mut run = |name: &str, bufs: [&Buffer; 6], params: PushParams, grid: (u32, u32, u32)| {
                if dispatch_err.is_none()
                    && let Err(e) = self.pipelines.dispatch(cmd, name, bufs, params, grid)
                {
                    dispatch_err = Some(e);
                }
            };

            run(
                "embedding_batch",
                [&self.weights.embedding.buf, dummy, dummy, &self.work.x, dummy, &self.work.token_ids],
                PushParams { p: [w, self.weights.embedding.kind, 0, 0, 0, 0, 0, 0], f: [0.0; 4] },
                (w.div_ceil(256), n, 1),
            );
            if gemma.is_some() {
                let nw = n * w;
                run(
                    "scale_vec",
                    [dummy, dummy, dummy, &self.work.x, dummy, dummy],
                    PushParams { p: [nw, 0, 0, 0, 0, 0, 0, 0], f: [(s.width as f32).sqrt(), 0.0, 0.0, 0.0] },
                    (nw.div_ceil(256), 1, 1),
                );
            }

            let t_tiles = n.div_ceil(4);
            for (index, l) in self.weights.layers.iter().enumerate() {
                let gemma_attn = gemma.map(|g| g.attention(s, index));
                let rope_base = gemma_attn.map_or(s.rope_base, |(_, base, _)| base);

                let attn_in = if olmo2 {
                    &self.work.x
                } else if let Some(norm) = &l.norm {
                    run(
                        "rmsnorm_batch",
                        [&norm.buf, &self.work.x, dummy, &self.work.normalized, dummy, dummy],
                        PushParams { p: [w, 0, 0, 0, 0, 0, 0, 0], f: [s.epsilon, 0.0, 0.0, 0.0] },
                        (n, 1, 1),
                    );
                    &self.work.normalized
                } else {
                    &self.work.x
                };

                for (tensor, out) in [(&l.q, &self.work.q), (&l.k, &self.work.k), (&l.v, &self.work.v)] {
                    run(
                        "matmul_batch",
                        [&tensor.buf, attn_in, dummy, out, dummy, dummy],
                        PushParams { p: [tensor.kind, tensor.width, tensor.rows, n, 0, 0, 0, 0], f: [0.0; 4] },
                        (tensor.rows.div_ceil(8), t_tiles, 1),
                    );
                }
                if let Some(biases) = &l.qkv_bias {
                    for (out, bias) in [(&self.work.q, &biases[0]), (&self.work.k, &biases[1]), (&self.work.v, &biases[2])] {
                        run(
                            "add_row_bias_batch",
                            [&bias.buf, dummy, dummy, out, dummy, dummy],
                            PushParams { p: [bias.width, 0, 0, 0, 0, 0, 0, 0], f: [0.0; 4] },
                            (bias.width.div_ceil(256), n, 1),
                        );
                    }
                }

                let factors_buf = self.weights.rope_factors.as_ref().map_or(dummy, |f| &f.buf);
                let has_factors = u32::from(self.weights.rope_factors.is_some());
                for (norm, vec_buf, size) in [(&l.q_norm, &self.work.q, qw), (&l.k_norm, &self.work.k, kw)] {
                    if let Some(norm) = norm {
                        let norm_w = if olmo2 { size } else { dim };
                        run(
                            "head_norm_batch",
                            [&norm.buf, dummy, dummy, vec_buf, dummy, dummy],
                            PushParams { p: [norm_w, size, 0, 0, 0, 0, 0, 0], f: [s.epsilon, 0.0, 0.0, 0.0] },
                            (size / norm_w, n, 1),
                        );
                    }
                    let pairs = (size / dim) * (rotary_dim / 2);
                    run(
                        "rotary_batch",
                        [dummy, dummy, factors_buf, vec_buf, dummy, dummy],
                        PushParams {
                            p: [size, dim, position, neox, rotary_dim, has_factors, 0, 0],
                            f: [rope_base, s.rope_attn_factor, 0.0, 0.0],
                        },
                        (pairs.div_ceil(256), n, 1),
                    );
                }

                run(
                    "cache_kv_batch",
                    [dummy, &self.work.k, &self.work.v, &self.keys[index], &self.values[index], dummy],
                    PushParams { p: [kw, position, 0, 0, 0, 0, 0, 0], f: [0.0; 4] },
                    (kw.div_ceil(256), n, 1),
                );

                let (window, scale, softcap) = match gemma_attn {
                    Some((win, _, sc)) => (win as u32, sc, gemma.unwrap().attention_softcap),
                    None => (0u32, (s.dim as f32).sqrt().recip(), 0.0f32),
                };
                run(
                    "attention_batch",
                    [&self.values[index], &self.work.q, &self.keys[index], &self.work.attended, dummy, dummy],
                    PushParams {
                        p: [dim, heads, kv_heads, position + 1, window, 0, 0, 0],
                        f: [scale, softcap, 0.0, 0.0],
                    },
                    (heads, n, 1),
                );

                run(
                    "matmul_batch",
                    [&l.output.buf, &self.work.attended, dummy, &self.work.projected, dummy, dummy],
                    PushParams { p: [l.output.kind, l.output.width, l.output.rows, n, 0, 0, 0, 0], f: [0.0; 4] },
                    (l.output.rows.div_ceil(8), t_tiles, 1),
                );

                let nw = n * w;
                if let Some([post_attn, _]) = &l.post_norms {
                    run(
                        "rmsnorm_batch",
                        [&post_attn.buf, &self.work.projected, dummy, &self.work.normalized, dummy, dummy],
                        PushParams { p: [w, 0, 0, 0, 0, 0, 0, 0], f: [s.epsilon, 0.0, 0.0, 0.0] },
                        (n, 1, 1),
                    );
                    run(
                        "residual",
                        [dummy, &self.work.normalized, dummy, &self.work.x, dummy, dummy],
                        PushParams { p: [nw, 0, 0, 0, 0, 0, 0, 0], f: [0.0; 4] },
                        (nw.div_ceil(256), 1, 1),
                    );
                } else if olmo2 {
                    if let Some(norm) = &l.norm {
                        run(
                            "rmsnorm_batch",
                            [&norm.buf, &self.work.projected, dummy, &self.work.normalized, dummy, dummy],
                            PushParams { p: [w, 0, 0, 0, 0, 0, 0, 0], f: [s.epsilon, 0.0, 0.0, 0.0] },
                            (n, 1, 1),
                        );
                        run(
                            "residual",
                            [dummy, &self.work.normalized, dummy, &self.work.x, dummy, dummy],
                            PushParams { p: [nw, 0, 0, 0, 0, 0, 0, 0], f: [0.0; 4] },
                            (nw.div_ceil(256), 1, 1),
                        );
                    }
                } else {
                    run(
                        "residual",
                        [dummy, &self.work.projected, dummy, &self.work.x, dummy, dummy],
                        PushParams { p: [nw, 0, 0, 0, 0, 0, 0, 0], f: [0.0; 4] },
                        (nw.div_ceil(256), 1, 1),
                    );
                }

                let ffn_in = if olmo2 {
                    &self.work.x
                } else if let Some(ffn_norm) = &l.ffn_norm {
                    run(
                        "rmsnorm_batch",
                        [&ffn_norm.buf, &self.work.x, dummy, &self.work.normalized, dummy, dummy],
                        PushParams { p: [w, 0, 0, 0, 0, 0, 0, 0], f: [s.epsilon, 0.0, 0.0, 0.0] },
                        (n, 1, 1),
                    );
                    &self.work.normalized
                } else {
                    &self.work.x
                };

                let gate = l.gate.as_ref().unwrap();
                run(
                    "matmul_batch",
                    [&gate.buf, ffn_in, dummy, &self.work.gate, dummy, dummy],
                    PushParams { p: [gate.kind, gate.width, gate.rows, n, 0, 0, 0, 0], f: [0.0; 4] },
                    (gate.rows.div_ceil(8), t_tiles, 1),
                );
                run(
                    "matmul_batch",
                    [&l.up.buf, ffn_in, dummy, &self.work.up, dummy, dummy],
                    PushParams { p: [l.up.kind, l.up.width, l.up.rows, n, 0, 0, 0, 0], f: [0.0; 4] },
                    (l.up.rows.div_ceil(8), t_tiles, 1),
                );
                let act_count = n * s.hidden as u32;
                run(
                    "swiglu",
                    [dummy, &self.work.up, dummy, &self.work.gate, dummy, dummy],
                    PushParams { p: [act_count, u32::from(gemma.is_some()), 0, 0, 0, 0, 0, 0], f: [0.0; 4] },
                    (act_count.div_ceil(256), 1, 1),
                );
                run(
                    "matmul_batch",
                    [&l.down.buf, &self.work.gate, dummy, &self.work.down, dummy, dummy],
                    PushParams { p: [l.down.kind, l.down.width, l.down.rows, n, 0, 0, 0, 0], f: [0.0; 4] },
                    (l.down.rows.div_ceil(8), t_tiles, 1),
                );

                if let Some([_, post_ffn]) = &l.post_norms {
                    run(
                        "rmsnorm_batch",
                        [&post_ffn.buf, &self.work.down, dummy, &self.work.normalized, dummy, dummy],
                        PushParams { p: [w, 0, 0, 0, 0, 0, 0, 0], f: [s.epsilon, 0.0, 0.0, 0.0] },
                        (n, 1, 1),
                    );
                    run(
                        "residual",
                        [dummy, &self.work.normalized, dummy, &self.work.x, dummy, dummy],
                        PushParams { p: [nw, 0, 0, 0, 0, 0, 0, 0], f: [0.0; 4] },
                        (nw.div_ceil(256), 1, 1),
                    );
                } else if olmo2 {
                    if let Some(ffn_norm) = &l.ffn_norm {
                        run(
                            "rmsnorm_batch",
                            [&ffn_norm.buf, &self.work.down, dummy, &self.work.normalized, dummy, dummy],
                            PushParams { p: [w, 0, 0, 0, 0, 0, 0, 0], f: [s.epsilon, 0.0, 0.0, 0.0] },
                            (n, 1, 1),
                        );
                        run(
                            "residual",
                            [dummy, &self.work.normalized, dummy, &self.work.x, dummy, dummy],
                            PushParams { p: [nw, 0, 0, 0, 0, 0, 0, 0], f: [0.0; 4] },
                            (nw.div_ceil(256), 1, 1),
                        );
                    }
                } else {
                    run(
                        "residual",
                        [dummy, &self.work.down, dummy, &self.work.x, dummy, dummy],
                        PushParams { p: [nw, 0, 0, 0, 0, 0, 0, 0], f: [0.0; 4] },
                        (nw.div_ceil(256), 1, 1),
                    );
                }
            }

            if compute_logits {
                run(
                    "copy_last_token",
                    [dummy, &self.work.x, dummy, &self.work.projected, dummy, dummy],
                    PushParams { p: [w, n - 1, 0, 0, 0, 0, 0, 0], f: [0.0; 4] },
                    (w.div_ceil(256), 1, 1),
                );
                let final_in = if let Some(norm) = &self.weights.output_norm {
                    run(
                        "rmsnorm",
                        [&norm.buf, &self.work.projected, dummy, &self.work.normalized, dummy, dummy],
                        PushParams { p: [w, 0, 0, 0, 0, 0, 0, 0], f: [s.epsilon, 0.0, 0.0, 0.0] },
                        (1, 1, 1),
                    );
                    &self.work.normalized
                } else {
                    &self.work.projected
                };
                let head = self.weights.output.as_ref().unwrap_or(&self.weights.embedding);
                run(
                    "matvec",
                    [&head.buf, final_in, dummy, &self.work.logits, dummy, &self.work.ids],
                    PushParams { p: [head.kind, head.width, head.rows, 0, 0, 0, 0, 0], f: [0.0; 4] },
                    (head.rows.div_ceil(8), 1, 1),
                );
                if let Some(gemma) = gemma
                    && gemma.final_softcap > 0.0
                {
                    let vocab = head.rows;
                    run(
                        "softcap",
                        [dummy, dummy, dummy, &self.work.logits, dummy, dummy],
                        PushParams { p: [vocab, 0, 0, 0, 0, 0, 0, 0], f: [gemma.final_softcap, 0.0, 0.0, 0.0] },
                        (vocab.div_ceil(256), 1, 1),
                    );
                }
                unsafe {
                    let barrier = [ash::vk::MemoryBarrier::default()
                        .src_access_mask(ash::vk::AccessFlags::SHADER_WRITE)
                        .dst_access_mask(ash::vk::AccessFlags::TRANSFER_READ)];
                    device.cmd_pipeline_barrier(
                        cmd,
                        ash::vk::PipelineStageFlags::COMPUTE_SHADER,
                        ash::vk::PipelineStageFlags::TRANSFER,
                        ash::vk::DependencyFlags::empty(),
                        &barrier,
                        &[],
                        &[],
                    );
                    let region = [ash::vk::BufferCopy::default().size(logits_bytes)];
                    device.cmd_copy_buffer(
                        cmd,
                        self.work.logits.handle,
                        self.work.logits_staging.handle,
                        &region,
                    );
                }
            }
        })?;
        if let Some(err) = dispatch_err {
            return Err(err);
        }
        if compute_logits {
            self.ctx
                .read_mapped_f32(&self.work.logits_staging, &mut self.logits)?;
            if self.logits.iter().any(|x| !x.is_finite()) {
                return Err(failure("nonfinite logits; request poisoned"));
            }
        }
        Ok(())
    }
}

impl Backend for VulkanBackend {
    fn info(&self) -> &ModelInfo {
        &self.weights.spec.info
    }

    fn logits(&self) -> Result<&[f32]> {
        self.state.logits(&self.logits)
    }

    fn tokenize(&self, text: &str) -> Result<Vec<i32>> {
        self.weights
            .spec
            .tokenizer
            .as_ref()
            .map_err(failure)?
            .tokenize(text)
    }

    fn detokenize(&self, tokens: &[i32]) -> Result<String> {
        self.weights
            .spec
            .tokenizer
            .as_ref()
            .map_err(failure)?
            .detokenize(tokens)
    }

    fn reset(&mut self) -> Result<()> {
        self.state = State::default();
        Ok(())
    }

    fn evaluate(&mut self, tokens: &[i32], position: u32) -> Result<()> {
        self.state.begin(tokens, position, &self.weights.spec.info)?;
        let cap = self.work.batch_cap;
        if tokens.len() > 1 && cap > 1 {
            let mut offset = 0usize;
            while offset < tokens.len() {
                let end = (offset + cap).min(tokens.len());
                let last = end == tokens.len();
                self.evaluate_batch(&tokens[offset..end], position + offset as u32, last)?;
                offset = end;
            }
        } else {
            for (i, &token) in tokens.iter().enumerate() {
                let last = i + 1 == tokens.len();
                self.evaluate_token(token, position + i as u32, last)?;
            }
        }
        self.state.finish(tokens.len());
        Ok(())
    }

    fn sample_greedy(&mut self) -> Result<i32> {
        greedy(self.logits()?)
    }

    fn is_eog(&self, token: i32) -> bool {
        self.weights.spec.eog.contains(&token)
    }

    fn memory_usage(&self) -> Option<MemoryUsage> {
        let mut usage = self.memory.clone();
        usage.kv_live_bytes = self.state.position as u64
            * self.weights.spec.kv_heads as u64
            * self.weights.spec.dim as u64
            * self.weights.layers.len() as u64
            * 8;
        Some(usage)
    }
}
