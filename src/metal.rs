//! Native Apple Metal compute backend (`--features metal`, `--backend metal`)
//! for macOS Apple Silicon.
//!
//! Uses `StorageModeShared` unified-memory buffers (zero-copy CPU/GPU access)
//! and runtime-compiled MSL compute shaders (`src/metal/shaders.metal`) with
//! hardware `simd_sum()` reductions.

use crate::{
    Config, Error, Result,
    backend::{Backend, MemoryUsage, ModelInfo},
    gguf::{Storage, Tensor},
    transformer::{Weights, invalid},
};
use metal::{
    Buffer, CommandQueue, CompileOptions, ComputePipelineState, Device as MtlDevice,
    MTLResourceOptions, MTLSize,
};
use serde::Serialize;
use std::{collections::HashMap, path::Path};

const SHADER_SOURCE: &str = include_str!("metal/shaders.metal");

const ENTRY_POINTS: &[&str] = &[
    "embedding",
    "embedding_batch",
    "rmsnorm",
    "rmsnorm_batch",
    "matvec",
    "matmul_batch",
    "add_row_bias",
    "add_row_bias_batch",
    "head_norm",
    "head_norm_batch",
    "rotary",
    "rotary_batch",
    "cache_kv",
    "cache_kv_batch",
    "attention",
    "attention_batch",
    "swiglu",
    "residual",
    "scale_vec",
    "softcap",
    "router",
    "combine_experts",
    "copy_last_token",
];

fn failure(message: impl std::fmt::Display) -> Error {
    Error::Backend(format!("Metal: {message}"))
}

#[derive(Clone, Debug, Serialize)]
pub struct MetalOptions {
    pub device: usize,
    pub memory_limit_bytes: u64,
}

impl Default for MetalOptions {
    fn default() -> Self {
        Self {
            device: 0,
            memory_limit_bytes: u64::MAX,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct PushParams {
    p: [u32; 8],
    f: [f32; 4],
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
    probabilities: Buffer,
    logits: Buffer,
    dummy: Buffer,
    batch_cap: usize,
}

pub struct MetalBackend {
    _device: MtlDevice,
    queue: CommandQueue,
    pipelines: HashMap<&'static str, ComputePipelineState>,
    weights: Weights<DeviceTensor>,
    keys: Vec<Buffer>,
    values: Vec<Buffer>,
    work: Workspace,
    logits: Vec<f32>,
    state: State,
    memory: MemoryUsage,
}

fn alloc_shared(device: &MtlDevice, bytes: u64) -> Buffer {
    let size = bytes.max(4).div_ceil(4) * 4;
    let buf = device.new_buffer(size, MTLResourceOptions::StorageModeShared);
    unsafe {
        std::ptr::write_bytes(buf.contents().cast::<u8>(), 0, size as usize);
    }
    buf
}

fn upload_bytes(device: &MtlDevice, data: &[u8]) -> Buffer {
    let size = (data.len() as u64).max(4).div_ceil(4) * 4;
    let buf = device.new_buffer(size, MTLResourceOptions::StorageModeShared);
    unsafe {
        let dst = buf.contents().cast::<u8>();
        std::ptr::copy_nonoverlapping(data.as_ptr(), dst, data.len());
        if (data.len() as u64) < size {
            std::ptr::write_bytes(dst.add(data.len()), 0, (size as usize) - data.len());
        }
    }
    buf
}

fn upload_tensor(device: &MtlDevice, tensor: Tensor) -> Result<DeviceTensor> {
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
                    "Metal backend does not yet support tensor kind {kind}"
                )));
            }
            (*kind, bytes.as_slice())
        }
    };
    let buf = upload_bytes(device, bytes);
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
    options: MetalOptions,
) -> Result<Box<dyn Backend>> {
    Ok(Box::new(MetalBackend::open(path, config, options)?))
}

impl MetalBackend {
    pub fn open(path: &Path, config: &Config, options: MetalOptions) -> Result<Self> {
        config.validate()?;
        if options.memory_limit_bytes == 0 || config.context_tokens > 65535 {
            return Err(invalid(
                "Metal requires a positive device budget and context <= 65535",
            ));
        }
        let all = MtlDevice::all();
        let device = if options.device == 0 {
            MtlDevice::system_default()
                .or_else(|| all.into_iter().next())
                .ok_or_else(|| failure("no Metal GPU device found"))?
        } else {
            all.into_iter()
                .nth(options.device)
                .ok_or_else(|| failure(format!("Metal device index {} out of range", options.device)))?
        };
        let device_name = device.name().to_owned();

        let mut raw = Weights::open(path, config)?;
        if raw.position_embedding.is_some()
            || raw.embedding_norm.is_some()
            || raw.output_norm_bias.is_some()
            || raw.layers.iter().any(|l| l.affine.is_some() || l.gate.is_none())
        {
            return Err(invalid(format!(
                "Metal backend does not yet support {} architecture",
                raw.spec.info.architecture
            )));
        }
        if raw.spec.dim > 256 {
            return Err(invalid(format!(
                "Metal backend requires head_dim <= 256, got {}",
                raw.spec.dim
            )));
        }
        raw.spec.info.backend = format!("metal ({device_name})");

        let library = device
            .new_library_with_source(SHADER_SOURCE, &CompileOptions::new())
            .map_err(|e| failure(format!("MSL compile error: {e}")))?;
        let mut pipelines = HashMap::with_capacity(ENTRY_POINTS.len());
        for &name in ENTRY_POINTS {
            let func = library
                .get_function(name, None)
                .map_err(|e| failure(format!("MSL function {name}: {e}")))?;
            let state = device
                .new_compute_pipeline_state_with_function(&func)
                .map_err(|e| failure(format!("Metal pipeline {name}: {e}")))?;
            pipelines.insert(name, state);
        }
        let queue = device.new_command_queue();

        let s = &raw.spec;
        let kv_elements = s.info.context_tokens as u64 * s.kv_heads as u64 * s.dim as u64;
        let kv_bytes_per_layer = kv_elements * 4;
        let kv_total_bytes = kv_bytes_per_layer * 2 * raw.layers.len() as u64;

        let mut weights_bytes = 0u64;
        let weights = raw.map(|t| {
            let dt = upload_tensor(&device, t)?;
            weights_bytes += dt.buf.length();
            Ok(dt)
        })?;

        let s = &weights.spec;
        let alloc_f32 = |n: usize| alloc_shared(&device, (n as u64) * 4);
        let qw = s.heads * s.dim;
        let kw = s.kv_heads * s.dim;
        let batch_cap = if s.experts == 0 {
            (s.info.batch_tokens as usize).clamp(1, 64)
        } else {
            1
        };
        let work = Workspace {
            x: alloc_f32(s.width * batch_cap),
            normalized: alloc_f32(s.width * batch_cap),
            q: alloc_f32(qw * batch_cap),
            k: alloc_f32(kw * batch_cap),
            v: alloc_f32(kw * batch_cap),
            attended: alloc_f32(qw * batch_cap),
            projected: alloc_f32(s.width * batch_cap),
            gate: alloc_f32(s.used * s.hidden * batch_cap),
            up: alloc_f32(s.used * s.hidden * batch_cap),
            down: alloc_f32(s.used * s.width * batch_cap),
            router: alloc_f32(s.experts.max(1)),
            ids: alloc_shared(&device, (s.used.max(1) as u64) * 4),
            token_ids: alloc_shared(&device, (batch_cap as u64) * 4),
            probabilities: alloc_f32(s.used.max(1)),
            logits: alloc_f32(s.info.vocab_tokens as usize),
            dummy: alloc_shared(&device, 16),
            batch_cap,
        };
        let workspace_bytes = work.x.length()
            + work.normalized.length()
            + work.q.length()
            + work.k.length()
            + work.v.length()
            + work.attended.length()
            + work.projected.length()
            + work.gate.length()
            + work.up.length()
            + work.down.length()
            + work.router.length()
            + work.ids.length()
            + work.token_ids.length()
            + work.probabilities.length();
        let logits_bytes = work.logits.length();

        let mut keys = Vec::with_capacity(weights.layers.len());
        let mut values = Vec::with_capacity(weights.layers.len());
        for _ in 0..weights.layers.len() {
            keys.push(alloc_shared(&device, kv_bytes_per_layer));
            values.push(alloc_shared(&device, kv_bytes_per_layer));
        }
        let reservation = weights_bytes + kv_total_bytes + workspace_bytes + logits_bytes;
        if reservation > options.memory_limit_bytes {
            return Err(Error::Invalid(format!(
                "Metal reservation ({reservation} B) exceeds limit ({} B)",
                options.memory_limit_bytes
            )));
        }
        let memory = MemoryUsage {
            allocation_domain: "metal",
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
            _device: device,
            queue,
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

    fn dispatch(
        &self,
        enc: &metal::ComputeCommandEncoderRef,
        name: &str,
        bufs: [&Buffer; 6],
        params: PushParams,
        grid: (u32, u32, u32),
    ) {
        let pipe = &self.pipelines[name];
        enc.set_compute_pipeline_state(pipe);
        for (i, buf) in bufs.iter().enumerate() {
            enc.set_buffer(i as u64, Some(buf), 0);
        }
        enc.set_bytes(
            6,
            std::mem::size_of::<PushParams>() as u64,
            (&params as *const PushParams).cast(),
        );
        enc.dispatch_thread_groups(
            MTLSize::new(u64::from(grid.0), u64::from(grid.1), u64::from(grid.2)),
            MTLSize::new(256, 1, 1),
        );
    }

    fn read_logits(&mut self) -> Result<()> {
        unsafe {
            let src = self.work.logits.contents().cast::<f32>();
            std::ptr::copy_nonoverlapping(src, self.logits.as_mut_ptr(), self.logits.len());
        }
        if self.logits.iter().any(|x| !x.is_finite()) {
            return Err(failure("nonfinite logits; request poisoned"));
        }
        Ok(())
    }

    fn evaluate_token(&mut self, token: i32, position: u32, compute_logits: bool) -> Result<()> {
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
        let dummy = &self.work.dummy;

        let cmd = self.queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();

        self.dispatch(
            enc,
            "embedding",
            [&self.weights.embedding.buf, dummy, dummy, &self.work.x, dummy, dummy],
            PushParams { p: [token as u32, w, self.weights.embedding.kind, 0, 0, 0, 0, 0], f: [0.0; 4] },
            (w.div_ceil(256), 1, 1),
        );
        if gemma.is_some() {
            self.dispatch(
                enc,
                "scale_vec",
                [dummy, dummy, dummy, &self.work.x, dummy, dummy],
                PushParams { p: [w, 0, 0, 0, 0, 0, 0, 0], f: [(s.width as f32).sqrt(), 0.0, 0.0, 0.0] },
                (w.div_ceil(256), 1, 1),
            );
        }

        for (index, l) in self.weights.layers.iter().enumerate() {
            let gemma_attn = gemma.map(|g| g.attention(s, index));
            let rope_base = gemma_attn.map_or(s.rope_base, |(_, base, _)| base);

            let attn_in = if olmo2 {
                &self.work.x
            } else if let Some(norm) = &l.norm {
                self.dispatch(
                    enc,
                    "rmsnorm",
                    [&norm.buf, &self.work.x, dummy, &self.work.normalized, dummy, dummy],
                    PushParams { p: [w, 0, 0, 0, 0, 0, 0, 0], f: [s.epsilon, 0.0, 0.0, 0.0] },
                    (1, 1, 1),
                );
                &self.work.normalized
            } else {
                &self.work.x
            };

            for (tensor, out) in [(&l.q, &self.work.q), (&l.k, &self.work.k), (&l.v, &self.work.v)] {
                self.dispatch(
                    enc,
                    "matvec",
                    [&tensor.buf, attn_in, dummy, out, dummy, &self.work.ids],
                    PushParams { p: [tensor.kind, tensor.width, tensor.rows, 0, 0, 0, 0, 0], f: [0.0; 4] },
                    (tensor.rows.div_ceil(8), 1, 1),
                );
            }
            if let Some(biases) = &l.qkv_bias {
                for (out, bias) in [(&self.work.q, &biases[0]), (&self.work.k, &biases[1]), (&self.work.v, &biases[2])] {
                    self.dispatch(
                        enc,
                        "add_row_bias",
                        [&bias.buf, dummy, dummy, out, dummy, dummy],
                        PushParams { p: [bias.width, 0, 0, 0, 0, 0, 0, 0], f: [0.0; 4] },
                        (bias.width.div_ceil(256), 1, 1),
                    );
                }
            }

            let factors_buf = self.weights.rope_factors.as_ref().map_or(dummy, |f| &f.buf);
            let has_factors = u32::from(self.weights.rope_factors.is_some());
            for (norm, vec_buf, size) in [(&l.q_norm, &self.work.q, qw), (&l.k_norm, &self.work.k, kw)] {
                if let Some(norm) = norm {
                    let norm_w = if olmo2 || olmoe { size } else { dim };
                    self.dispatch(
                        enc,
                        "head_norm",
                        [&norm.buf, dummy, dummy, vec_buf, dummy, dummy],
                        PushParams { p: [norm_w, 0, 0, 0, 0, 0, 0, 0], f: [s.epsilon, 0.0, 0.0, 0.0] },
                        (size / norm_w, 1, 1),
                    );
                }
                let pairs = (size / dim) * (rotary_dim / 2);
                self.dispatch(
                    enc,
                    "rotary",
                    [dummy, dummy, factors_buf, vec_buf, dummy, dummy],
                    PushParams {
                        p: [size, dim, position, neox, rotary_dim, has_factors, 0, 0],
                        f: [rope_base, s.rope_attn_factor, 0.0, 0.0],
                    },
                    (pairs.div_ceil(256), 1, 1),
                );
            }

            self.dispatch(
                enc,
                "cache_kv",
                [dummy, &self.work.k, &self.work.v, &self.keys[index], &self.values[index], dummy],
                PushParams { p: [kw, position, 0, 0, 0, 0, 0, 0], f: [0.0; 4] },
                (kw.div_ceil(256), 1, 1),
            );

            let (window, scale, softcap) = match gemma_attn {
                Some((win, _, sc)) => (win as u32, sc, gemma.unwrap().attention_softcap),
                None => (0u32, (s.dim as f32).sqrt().recip(), 0.0f32),
            };
            self.dispatch(
                enc,
                "attention",
                [&self.values[index], &self.work.q, &self.keys[index], &self.work.attended, dummy, dummy],
                PushParams {
                    p: [dim, heads, kv_heads, position + 1, window, 0, 0, 0],
                    f: [scale, softcap, 0.0, 0.0],
                },
                (heads, 1, 1),
            );

            self.dispatch(
                enc,
                "matvec",
                [&l.output.buf, &self.work.attended, dummy, &self.work.projected, dummy, &self.work.ids],
                PushParams { p: [l.output.kind, l.output.width, l.output.rows, 0, 0, 0, 0, 0], f: [0.0; 4] },
                (l.output.rows.div_ceil(8), 1, 1),
            );

            if let Some([post_attn, _]) = &l.post_norms {
                self.dispatch(
                    enc,
                    "rmsnorm",
                    [&post_attn.buf, &self.work.projected, dummy, &self.work.normalized, dummy, dummy],
                    PushParams { p: [w, 0, 0, 0, 0, 0, 0, 0], f: [s.epsilon, 0.0, 0.0, 0.0] },
                    (1, 1, 1),
                );
                self.dispatch(
                    enc,
                    "residual",
                    [dummy, &self.work.normalized, dummy, &self.work.x, dummy, dummy],
                    PushParams { p: [w, 0, 0, 0, 0, 0, 0, 0], f: [0.0; 4] },
                    (w.div_ceil(256), 1, 1),
                );
            } else if olmo2 {
                if let Some(norm) = &l.norm {
                    self.dispatch(
                        enc,
                        "rmsnorm",
                        [&norm.buf, &self.work.projected, dummy, &self.work.normalized, dummy, dummy],
                        PushParams { p: [w, 0, 0, 0, 0, 0, 0, 0], f: [s.epsilon, 0.0, 0.0, 0.0] },
                        (1, 1, 1),
                    );
                    self.dispatch(
                        enc,
                        "residual",
                        [dummy, &self.work.normalized, dummy, &self.work.x, dummy, dummy],
                        PushParams { p: [w, 0, 0, 0, 0, 0, 0, 0], f: [0.0; 4] },
                        (w.div_ceil(256), 1, 1),
                    );
                }
            } else {
                self.dispatch(
                    enc,
                    "residual",
                    [dummy, &self.work.projected, dummy, &self.work.x, dummy, dummy],
                    PushParams { p: [w, 0, 0, 0, 0, 0, 0, 0], f: [0.0; 4] },
                    (w.div_ceil(256), 1, 1),
                );
            }

            let ffn_in = if olmo2 {
                &self.work.x
            } else if let Some(ffn_norm) = &l.ffn_norm {
                self.dispatch(
                    enc,
                    "rmsnorm",
                    [&ffn_norm.buf, &self.work.x, dummy, &self.work.normalized, dummy, dummy],
                    PushParams { p: [w, 0, 0, 0, 0, 0, 0, 0], f: [s.epsilon, 0.0, 0.0, 0.0] },
                    (1, 1, 1),
                );
                &self.work.normalized
            } else {
                &self.work.x
            };

            let moe = u32::from(l.router.is_some());
            if let Some(router_t) = &l.router {
                self.dispatch(
                    enc,
                    "matvec",
                    [&router_t.buf, ffn_in, dummy, &self.work.router, dummy, &self.work.ids],
                    PushParams { p: [router_t.kind, router_t.width, router_t.rows, 0, 0, 0, 0, 0], f: [0.0; 4] },
                    (router_t.rows.div_ceil(8), 1, 1),
                );
                self.dispatch(
                    enc,
                    "router",
                    [dummy, &self.work.router, dummy, &self.work.probabilities, dummy, &self.work.ids],
                    PushParams { p: [experts, used, u32::from(olmoe), 0, 0, 0, 0, 0], f: [0.0; 4] },
                    (1, 1, 1),
                );
            }

            let gate = l.gate.as_ref().unwrap();
            self.dispatch(
                enc,
                "matvec",
                [&gate.buf, ffn_in, dummy, &self.work.gate, dummy, &self.work.ids],
                PushParams { p: [gate.kind, gate.width, gate.rows, moe, 0, 0, 0, 0], f: [0.0; 4] },
                (gate.rows.div_ceil(8), used, 1),
            );
            self.dispatch(
                enc,
                "matvec",
                [&l.up.buf, ffn_in, dummy, &self.work.up, dummy, &self.work.ids],
                PushParams { p: [l.up.kind, l.up.width, l.up.rows, moe, 0, 0, 0, 0], f: [0.0; 4] },
                (l.up.rows.div_ceil(8), used, 1),
            );
            let act_count = (s.hidden * s.used) as u32;
            self.dispatch(
                enc,
                "swiglu",
                [dummy, &self.work.up, dummy, &self.work.gate, dummy, dummy],
                PushParams { p: [act_count, u32::from(gemma.is_some()), 0, 0, 0, 0, 0, 0], f: [0.0; 4] },
                (act_count.div_ceil(256), 1, 1),
            );
            self.dispatch(
                enc,
                "matvec",
                [&l.down.buf, &self.work.gate, dummy, &self.work.down, dummy, &self.work.ids],
                PushParams { p: [l.down.kind, l.down.width, l.down.rows, moe, 1, 0, 0, 0], f: [0.0; 4] },
                (l.down.rows.div_ceil(8), used, 1),
            );

            if moe != 0 {
                self.dispatch(
                    enc,
                    "combine_experts",
                    [dummy, &self.work.down, &self.work.probabilities, &self.work.x, dummy, dummy],
                    PushParams { p: [w, used, 0, 0, 0, 0, 0, 0], f: [0.0; 4] },
                    (w.div_ceil(256), 1, 1),
                );
            } else if let Some([_, post_ffn]) = &l.post_norms {
                self.dispatch(
                    enc,
                    "rmsnorm",
                    [&post_ffn.buf, &self.work.down, dummy, &self.work.normalized, dummy, dummy],
                    PushParams { p: [w, 0, 0, 0, 0, 0, 0, 0], f: [s.epsilon, 0.0, 0.0, 0.0] },
                    (1, 1, 1),
                );
                self.dispatch(
                    enc,
                    "residual",
                    [dummy, &self.work.normalized, dummy, &self.work.x, dummy, dummy],
                    PushParams { p: [w, 0, 0, 0, 0, 0, 0, 0], f: [0.0; 4] },
                    (w.div_ceil(256), 1, 1),
                );
            } else if olmo2 {
                if let Some(ffn_norm) = &l.ffn_norm {
                    self.dispatch(
                        enc,
                        "rmsnorm",
                        [&ffn_norm.buf, &self.work.down, dummy, &self.work.normalized, dummy, dummy],
                        PushParams { p: [w, 0, 0, 0, 0, 0, 0, 0], f: [s.epsilon, 0.0, 0.0, 0.0] },
                        (1, 1, 1),
                    );
                    self.dispatch(
                        enc,
                        "residual",
                        [dummy, &self.work.normalized, dummy, &self.work.x, dummy, dummy],
                        PushParams { p: [w, 0, 0, 0, 0, 0, 0, 0], f: [0.0; 4] },
                        (w.div_ceil(256), 1, 1),
                    );
                }
            } else {
                self.dispatch(
                    enc,
                    "residual",
                    [dummy, &self.work.down, dummy, &self.work.x, dummy, dummy],
                    PushParams { p: [w, 0, 0, 0, 0, 0, 0, 0], f: [0.0; 4] },
                    (w.div_ceil(256), 1, 1),
                );
            }
        }

        if compute_logits {
            let final_in = if let Some(norm) = &self.weights.output_norm {
                self.dispatch(
                    enc,
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
            self.dispatch(
                enc,
                "matvec",
                [&head.buf, final_in, dummy, &self.work.logits, dummy, &self.work.ids],
                PushParams { p: [head.kind, head.width, head.rows, 0, 0, 0, 0, 0], f: [0.0; 4] },
                (head.rows.div_ceil(8), 1, 1),
            );
            if let Some(gemma) = gemma
                && gemma.final_softcap > 0.0
            {
                let vocab = head.rows;
                self.dispatch(
                    enc,
                    "softcap",
                    [dummy, dummy, dummy, &self.work.logits, dummy, dummy],
                    PushParams { p: [vocab, 0, 0, 0, 0, 0, 0, 0], f: [gemma.final_softcap, 0.0, 0.0, 0.0] },
                    (vocab.div_ceil(256), 1, 1),
                );
            }
        }

        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
        if compute_logits {
            self.read_logits()?;
        }
        Ok(())
    }

    fn evaluate_batch(&mut self, tokens: &[i32], position: u32, compute_logits: bool) -> Result<()> {
        let n = tokens.len() as u32;
        unsafe {
            let dst = self.work.token_ids.contents().cast::<i32>();
            std::ptr::copy_nonoverlapping(tokens.as_ptr(), dst, tokens.len());
        }
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
        let dummy = &self.work.dummy;
        let t_tiles = n.div_ceil(4);

        let cmd = self.queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();

        self.dispatch(
            enc,
            "embedding_batch",
            [&self.weights.embedding.buf, dummy, dummy, &self.work.x, dummy, &self.work.token_ids],
            PushParams { p: [w, self.weights.embedding.kind, 0, 0, 0, 0, 0, 0], f: [0.0; 4] },
            (w.div_ceil(256), n, 1),
        );
        if gemma.is_some() {
            let nw = n * w;
            self.dispatch(
                enc,
                "scale_vec",
                [dummy, dummy, dummy, &self.work.x, dummy, dummy],
                PushParams { p: [nw, 0, 0, 0, 0, 0, 0, 0], f: [(s.width as f32).sqrt(), 0.0, 0.0, 0.0] },
                (nw.div_ceil(256), 1, 1),
            );
        }

        for (index, l) in self.weights.layers.iter().enumerate() {
            let gemma_attn = gemma.map(|g| g.attention(s, index));
            let rope_base = gemma_attn.map_or(s.rope_base, |(_, base, _)| base);

            let attn_in = if olmo2 {
                &self.work.x
            } else if let Some(norm) = &l.norm {
                self.dispatch(
                    enc,
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
                self.dispatch(
                    enc,
                    "matmul_batch",
                    [&tensor.buf, attn_in, dummy, out, dummy, dummy],
                    PushParams { p: [tensor.kind, tensor.width, tensor.rows, n, 0, 0, 0, 0], f: [0.0; 4] },
                    (tensor.rows.div_ceil(8), t_tiles, 1),
                );
            }
            if let Some(biases) = &l.qkv_bias {
                for (out, bias) in [(&self.work.q, &biases[0]), (&self.work.k, &biases[1]), (&self.work.v, &biases[2])] {
                    self.dispatch(
                        enc,
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
                    self.dispatch(
                        enc,
                        "head_norm_batch",
                        [&norm.buf, dummy, dummy, vec_buf, dummy, dummy],
                        PushParams { p: [norm_w, size, 0, 0, 0, 0, 0, 0], f: [s.epsilon, 0.0, 0.0, 0.0] },
                        (size / norm_w, n, 1),
                    );
                }
                let pairs = (size / dim) * (rotary_dim / 2);
                self.dispatch(
                    enc,
                    "rotary_batch",
                    [dummy, dummy, factors_buf, vec_buf, dummy, dummy],
                    PushParams {
                        p: [size, dim, position, neox, rotary_dim, has_factors, 0, 0],
                        f: [rope_base, s.rope_attn_factor, 0.0, 0.0],
                    },
                    (pairs.div_ceil(256), n, 1),
                );
            }

            self.dispatch(
                enc,
                "cache_kv_batch",
                [dummy, &self.work.k, &self.work.v, &self.keys[index], &self.values[index], dummy],
                PushParams { p: [kw, position, 0, 0, 0, 0, 0, 0], f: [0.0; 4] },
                (kw.div_ceil(256), n, 1),
            );

            let (window, scale, softcap) = match gemma_attn {
                Some((win, _, sc)) => (win as u32, sc, gemma.unwrap().attention_softcap),
                None => (0u32, (s.dim as f32).sqrt().recip(), 0.0f32),
            };
            self.dispatch(
                enc,
                "attention_batch",
                [&self.values[index], &self.work.q, &self.keys[index], &self.work.attended, dummy, dummy],
                PushParams {
                    p: [dim, heads, kv_heads, position + 1, window, 0, 0, 0],
                    f: [scale, softcap, 0.0, 0.0],
                },
                (heads, n, 1),
            );

            self.dispatch(
                enc,
                "matmul_batch",
                [&l.output.buf, &self.work.attended, dummy, &self.work.projected, dummy, dummy],
                PushParams { p: [l.output.kind, l.output.width, l.output.rows, n, 0, 0, 0, 0], f: [0.0; 4] },
                (l.output.rows.div_ceil(8), t_tiles, 1),
            );

            let nw = n * w;
            if let Some([post_attn, _]) = &l.post_norms {
                self.dispatch(
                    enc,
                    "rmsnorm_batch",
                    [&post_attn.buf, &self.work.projected, dummy, &self.work.normalized, dummy, dummy],
                    PushParams { p: [w, 0, 0, 0, 0, 0, 0, 0], f: [s.epsilon, 0.0, 0.0, 0.0] },
                    (n, 1, 1),
                );
                self.dispatch(
                    enc,
                    "residual",
                    [dummy, &self.work.normalized, dummy, &self.work.x, dummy, dummy],
                    PushParams { p: [nw, 0, 0, 0, 0, 0, 0, 0], f: [0.0; 4] },
                    (nw.div_ceil(256), 1, 1),
                );
            } else if olmo2 {
                if let Some(norm) = &l.norm {
                    self.dispatch(
                        enc,
                        "rmsnorm_batch",
                        [&norm.buf, &self.work.projected, dummy, &self.work.normalized, dummy, dummy],
                        PushParams { p: [w, 0, 0, 0, 0, 0, 0, 0], f: [s.epsilon, 0.0, 0.0, 0.0] },
                        (n, 1, 1),
                    );
                    self.dispatch(
                        enc,
                        "residual",
                        [dummy, &self.work.normalized, dummy, &self.work.x, dummy, dummy],
                        PushParams { p: [nw, 0, 0, 0, 0, 0, 0, 0], f: [0.0; 4] },
                        (nw.div_ceil(256), 1, 1),
                    );
                }
            } else {
                self.dispatch(
                    enc,
                    "residual",
                    [dummy, &self.work.projected, dummy, &self.work.x, dummy, dummy],
                    PushParams { p: [nw, 0, 0, 0, 0, 0, 0, 0], f: [0.0; 4] },
                    (nw.div_ceil(256), 1, 1),
                );
            }

            let ffn_in = if olmo2 {
                &self.work.x
            } else if let Some(ffn_norm) = &l.ffn_norm {
                self.dispatch(
                    enc,
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
            self.dispatch(
                enc,
                "matmul_batch",
                [&gate.buf, ffn_in, dummy, &self.work.gate, dummy, dummy],
                PushParams { p: [gate.kind, gate.width, gate.rows, n, 0, 0, 0, 0], f: [0.0; 4] },
                (gate.rows.div_ceil(8), t_tiles, 1),
            );
            self.dispatch(
                enc,
                "matmul_batch",
                [&l.up.buf, ffn_in, dummy, &self.work.up, dummy, dummy],
                PushParams { p: [l.up.kind, l.up.width, l.up.rows, n, 0, 0, 0, 0], f: [0.0; 4] },
                (l.up.rows.div_ceil(8), t_tiles, 1),
            );
            let act_count = n * s.hidden as u32;
            self.dispatch(
                enc,
                "swiglu",
                [dummy, &self.work.up, dummy, &self.work.gate, dummy, dummy],
                PushParams { p: [act_count, u32::from(gemma.is_some()), 0, 0, 0, 0, 0, 0], f: [0.0; 4] },
                (act_count.div_ceil(256), 1, 1),
            );
            self.dispatch(
                enc,
                "matmul_batch",
                [&l.down.buf, &self.work.gate, dummy, &self.work.down, dummy, dummy],
                PushParams { p: [l.down.kind, l.down.width, l.down.rows, n, 0, 0, 0, 0], f: [0.0; 4] },
                (l.down.rows.div_ceil(8), t_tiles, 1),
            );

            if let Some([_, post_ffn]) = &l.post_norms {
                self.dispatch(
                    enc,
                    "rmsnorm_batch",
                    [&post_ffn.buf, &self.work.down, dummy, &self.work.normalized, dummy, dummy],
                    PushParams { p: [w, 0, 0, 0, 0, 0, 0, 0], f: [s.epsilon, 0.0, 0.0, 0.0] },
                    (n, 1, 1),
                );
                self.dispatch(
                    enc,
                    "residual",
                    [dummy, &self.work.normalized, dummy, &self.work.x, dummy, dummy],
                    PushParams { p: [nw, 0, 0, 0, 0, 0, 0, 0], f: [0.0; 4] },
                    (nw.div_ceil(256), 1, 1),
                );
            } else if olmo2 {
                if let Some(ffn_norm) = &l.ffn_norm {
                    self.dispatch(
                        enc,
                        "rmsnorm_batch",
                        [&ffn_norm.buf, &self.work.down, dummy, &self.work.normalized, dummy, dummy],
                        PushParams { p: [w, 0, 0, 0, 0, 0, 0, 0], f: [s.epsilon, 0.0, 0.0, 0.0] },
                        (n, 1, 1),
                    );
                    self.dispatch(
                        enc,
                        "residual",
                        [dummy, &self.work.normalized, dummy, &self.work.x, dummy, dummy],
                        PushParams { p: [nw, 0, 0, 0, 0, 0, 0, 0], f: [0.0; 4] },
                        (nw.div_ceil(256), 1, 1),
                    );
                }
            } else {
                self.dispatch(
                    enc,
                    "residual",
                    [dummy, &self.work.down, dummy, &self.work.x, dummy, dummy],
                    PushParams { p: [nw, 0, 0, 0, 0, 0, 0, 0], f: [0.0; 4] },
                    (nw.div_ceil(256), 1, 1),
                );
            }
        }

        if compute_logits {
            self.dispatch(
                enc,
                "copy_last_token",
                [dummy, &self.work.x, dummy, &self.work.projected, dummy, dummy],
                PushParams { p: [w, n - 1, 0, 0, 0, 0, 0, 0], f: [0.0; 4] },
                (w.div_ceil(256), 1, 1),
            );
            let final_in = if let Some(norm) = &self.weights.output_norm {
                self.dispatch(
                    enc,
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
            self.dispatch(
                enc,
                "matvec",
                [&head.buf, final_in, dummy, &self.work.logits, dummy, &self.work.ids],
                PushParams { p: [head.kind, head.width, head.rows, 0, 0, 0, 0, 0], f: [0.0; 4] },
                (head.rows.div_ceil(8), 1, 1),
            );
            if let Some(gemma) = gemma
                && gemma.final_softcap > 0.0
            {
                let vocab = head.rows;
                self.dispatch(
                    enc,
                    "softcap",
                    [dummy, dummy, dummy, &self.work.logits, dummy, dummy],
                    PushParams { p: [vocab, 0, 0, 0, 0, 0, 0, 0], f: [gemma.final_softcap, 0.0, 0.0, 0.0] },
                    (vocab.div_ceil(256), 1, 1),
                );
            }
        }

        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
        if compute_logits {
            self.read_logits()?;
        }
        Ok(())
    }
}

impl Backend for MetalBackend {
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
