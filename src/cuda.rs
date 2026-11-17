//! Independent CUDA kernels over packed GGUF weights. No GGML or native inference engine.
pub mod diagnostics;
pub mod reference;

use crate::{
    Config, Error, Result,
    backend::{Backend, MemoryUsage, ModelInfo},
    gguf::{Storage, Tensor},
    transformer::{Spec, Weights, invalid},
};
use cudarc::{
    driver::{
        CudaContext, CudaFunction, CudaSlice, CudaStream, LaunchConfig, PushKernelArg, result, sys,
    },
    nvrtc::{CompileOptions, compile_ptx_with_opts},
};
use serde::Serialize;
use std::{
    collections::HashMap,
    path::{Path, PathBuf},
    sync::Arc,
};

fn failure(error: impl std::fmt::Display) -> Error {
    Error::Backend(format!("CUDA: {error}"))
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, clap::ValueEnum)]
#[serde(rename_all = "lowercase")]
pub enum CudaMatmul {
    #[default]
    F32,
    /// Experimental reduced-precision projections; failed long-MoE quality gates.
    Tf32,
    /// Experimental three-product TF32; tested for long MoE, slower on small/dense prompts.
    Tf32x3,
    /// Experimental staged F32 expert tiles; dense projections stay F32.
    #[serde(rename = "f32-tiled")]
    F32Tiled,
    /// Experimental staged three-product TF32 experts; failed strict MoE quality gates.
    #[serde(rename = "tf32x3-tiled")]
    Tf32x3Tiled,
}

impl CudaMatmul {
    fn tensor_cores(self) -> bool {
        matches!(self, Self::Tf32 | Self::Tf32x3 | Self::Tf32x3Tiled)
    }

    fn group_tile(self) -> usize {
        if matches!(self, Self::F32Tiled | Self::Tf32x3Tiled) {
            8
        } else {
            4
        }
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct CudaOptions {
    pub device: usize,
    /// Owned device buffers only, excluding driver/JIT overhead and host model loading.
    pub memory_limit_bytes: u64,
    /// Opt-in private directory for persistent, checksummed NVRTC output.
    pub kernel_cache_dir: Option<PathBuf>,
    pub matmul: CudaMatmul,
    /// Opt-in bounded decode graph and token-only greedy generation.
    pub decode_graph: bool,
}

impl Default for CudaOptions {
    fn default() -> Self {
        Self {
            device: 0,
            memory_limit_bytes: 24 * 1024 * 1024 * 1024,
            kernel_cache_dir: None,
            matmul: CudaMatmul::F32,
            decode_graph: false,
        }
    }
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
        // A driver/numerical error at any point leaves the request poisoned.
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

fn greedy(logits: &[f32]) -> Result<i32> {
    if logits.is_empty() || logits.iter().any(|x| !x.is_finite()) {
        return Err(failure("invalid logits"));
    }
    let mut best = 0;
    for i in 1..logits.len() {
        if logits[i] > logits[best] {
            best = i;
        }
    }
    Ok(best as i32)
}

struct DeviceTensor {
    bytes: CudaSlice<u8>,
    kind: i32,
    width: i32,
    rows: i32,
}

impl DeviceTensor {
    fn upload(tensor: Tensor, stream: &Arc<CudaStream>) -> Result<Self> {
        let (kind, bytes) = match tensor.storage {
            Storage::Packed { kind, bytes } => (kind as i32, bytes),
            Storage::F32(values) => (0, values.into_iter().flat_map(f32::to_le_bytes).collect()),
        };
        Ok(Self {
            bytes: stream.clone_htod(&bytes).map_err(failure)?,
            kind,
            width: tensor.dims[0] as i32,
            rows: tensor.dims.get(1).copied().unwrap_or(1) as i32,
        })
    }
}

struct Workspace {
    x: CudaSlice<f32>,
    normalized: CudaSlice<f32>,
    q: CudaSlice<f32>,
    k: CudaSlice<f32>,
    v: CudaSlice<f32>,
    attended: CudaSlice<f32>,
    projected: CudaSlice<f32>,
    gate: CudaSlice<f32>,
    up: CudaSlice<f32>,
    down: CudaSlice<f32>,
    scores: CudaSlice<f32>,
    router: CudaSlice<f32>,
    ids: CudaSlice<i32>,
    probabilities: CudaSlice<f32>,
    status: CudaSlice<i32>,
    logits: CudaSlice<f32>,
    tokens: Option<CudaSlice<i32>>,
    groups: Option<CudaSlice<i32>>,
}

fn prefill_capacity(spec: &Spec) -> usize {
    (spec.info.batch_tokens as usize).min(512)
}

fn fused_attention_supported(spec: &Spec) -> bool {
    spec.dim <= 256 && spec.info.context_tokens <= 1024
}

fn routing_elements(spec: &Spec) -> Result<usize> {
    spec.used
        .checked_mul(prefill_capacity(spec))
        .ok_or_else(|| invalid("routing workspace overflow"))
}

fn group_elements(spec: &Spec) -> Result<usize> {
    group_elements_with(spec, 4)
}

fn group_elements_with(spec: &Spec, tile: usize) -> Result<usize> {
    if spec.experts == 0 || prefill_capacity(spec) == 1 {
        return Ok(0);
    }
    let assignments = routing_elements(spec)?;
    // At most three padding slots per nonempty expert; no expert is selected
    // twice by a token. Five integers describe an expert and four assignments.
    assignments
        .div_ceil(tile)
        .checked_add(spec.experts - 1)
        .map(|n| n.min(assignments))
        .and_then(|n| n.checked_mul(tile + 1))
        .ok_or_else(|| invalid("group workspace overflow"))
}

fn workspace_sizes(spec: &Spec) -> Result<[usize; 12]> {
    let product = |a: usize, b: usize| {
        a.checked_mul(b)
            .ok_or_else(|| invalid("workspace overflow"))
    };
    let mut sizes = [
        spec.width,
        spec.width,
        product(spec.heads, spec.dim)?,
        product(spec.kv_heads, spec.dim)?,
        product(spec.kv_heads, spec.dim)?,
        product(spec.heads, spec.dim)?,
        spec.width,
        product(spec.used, spec.hidden)?,
        product(spec.used, spec.hidden)?,
        product(spec.used, spec.width)?,
        product(spec.heads, spec.info.context_tokens as usize)?,
        spec.experts.max(1),
    ];
    for size in &mut sizes[..10] {
        *size = product(*size, prefill_capacity(spec))?;
    }
    if !fused_attention_supported(spec) {
        sizes[10] = product(sizes[10], prefill_capacity(spec))?;
    }
    if spec.experts > 0 {
        sizes[11] = product(sizes[11], prefill_capacity(spec))?;
    }
    if sizes[..10].iter().any(|&n| n > i32::MAX as usize) {
        return Err(invalid("batched vectors exceed CUDA indexing"));
    }
    Ok(sizes)
}

impl Workspace {
    fn new(spec: &Spec, stream: &Arc<CudaStream>, matmul: CudaMatmul) -> Result<Self> {
        let [w, _, qw, kw, _, _, _, gates, _, down, scores, router] = workspace_sizes(spec)?;
        let alloc = |n| stream.alloc_zeros::<f32>(n).map_err(failure);
        Ok(Self {
            x: alloc(w)?,
            normalized: alloc(w)?,
            q: alloc(qw)?,
            k: alloc(kw)?,
            v: alloc(kw)?,
            attended: alloc(qw)?,
            projected: alloc(w)?,
            gate: alloc(gates)?,
            up: alloc(gates)?,
            down: alloc(down)?,
            scores: alloc(scores)?,
            router: alloc(router)?,
            ids: stream
                .alloc_zeros(routing_elements(spec)?)
                .map_err(failure)?,
            probabilities: stream
                .clone_htod(&vec![1.0; routing_elements(spec)?])
                .map_err(failure)?,
            status: stream.alloc_zeros(1).map_err(failure)?,
            logits: alloc(spec.info.vocab_tokens as usize)?,
            tokens: if prefill_capacity(spec) > 1 {
                Some(
                    stream
                        .alloc_zeros(prefill_capacity(spec))
                        .map_err(failure)?,
                )
            } else {
                None
            },
            groups: if group_elements_with(spec, matmul.group_tile())? > 0 {
                Some(
                    stream
                        .alloc_zeros(group_elements_with(spec, matmul.group_tile())?)
                        .map_err(failure)?,
                )
            } else {
                None
            },
        })
    }
}

macro_rules! launch {
    ($backend:ident, $name:expr, $grid:expr, $($arg:expr),+ $(,)?) => {{
        let cfg = LaunchConfig { grid_dim: $grid, block_dim: (256, 1, 1), shared_mem_bytes: 0 };
        let mut args = $backend.stream.launch_builder(&$backend.functions[$name]);
        $(args.arg($arg);)+
        // Kernel signatures, validated tensor shapes, and fixed 256-thread reductions agree.
        unsafe { args.launch(cfg) }.map_err(failure)?;
    }};
}

macro_rules! launch_position {
    ($backend:ident, $device:expr, $name:literal, $dynamic:literal, $grid:expr,
     [$($before:expr),*], $value:expr, [$($after:expr),*]) => {{
        if $device {
            let control = &$backend.decode.as_ref()
                .ok_or_else(|| failure("decode control missing"))?.control;
            launch!($backend, $dynamic, $grid, $($before,)* control $(, $after)*);
        } else {
            launch!($backend, $name, $grid, $($before,)* $value $(, $after)*);
        }
    }};
}

#[cfg(test)]
mod arithmetic;
mod batch;
mod cache;
mod decode;
#[cfg(test)]
mod grouped_bench;
#[cfg(test)]
mod packed_decode;
#[cfg(test)]
mod prefill;

pub struct CudaBackend {
    // Destroy captured references before their buffers and modules.
    decode: Option<decode::DecodeState>,
    weights: Weights<DeviceTensor>,
    stream: Arc<CudaStream>,
    functions: HashMap<&'static str, CudaFunction>,
    work: Workspace,
    keys: Vec<CudaSlice<f32>>,
    values: Vec<CudaSlice<f32>>,
    logits: Vec<f32>,
    state: State,
    memory: MemoryUsage,
    matmul: CudaMatmul,
    diagnostic: Option<diagnostics::Capture>,
}

impl CudaBackend {
    pub fn open(path: &Path, config: &Config, options: CudaOptions) -> Result<Self> {
        config.validate()?;
        if !cfg!(target_os = "linux") {
            return Err(invalid("CUDA backend currently supports Linux only"));
        }
        if options.memory_limit_bytes == 0 || config.context_tokens > 65535 {
            return Err(invalid(
                "CUDA requires a positive device budget and context <= 65535",
            ));
        }
        // cudarc's lazy loader panics for missing libraries; make that a normal configuration error.
        for name in ["libcuda.so.1", "libnvrtc.so.13"] {
            // These are fixed CUDA libraries, not user-supplied executable paths.
            unsafe { libloading::Library::new(name) }.map_err(|e| {
                failure(format!(
                    "{name}: {e}; set LD_LIBRARY_PATH to CUDA 13 libraries"
                ))
            })?;
        }
        let context = CudaContext::new(options.device).map_err(failure)?;
        let major = context
            .attribute(sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)
            .map_err(failure)?;
        let minor = context
            .attribute(sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)
            .map_err(failure)?;
        if options.matmul.tensor_cores() && major < 8 {
            return Err(invalid(
                "TF32 tensor-core mode requires compute capability >=8.0",
            ));
        }
        let base_source = concat!(
            include_str!("cuda/kernels.cu"),
            "\n",
            include_str!("cuda/prefill.cu"),
            "\n",
            include_str!("cuda/attention.cu")
        );
        let base_source = if options.decode_graph {
            format!("{base_source}\n{}", include_str!("cuda/decode.cu"))
        } else {
            base_source.to_owned()
        };
        let source = if options.matmul.tensor_cores() {
            format!("{base_source}\n{}", include_str!("cuda/tensorcore.cu"))
        } else {
            base_source
        };
        let ptx = cache::compile(
            options.kernel_cache_dir.as_deref(),
            &source,
            CompileOptions {
                fmad: Some(false),
                options: vec![format!("--gpu-architecture=compute_{major}{minor}")],
                ..Default::default()
            },
        )?;
        let module = context.load_module(ptx).map_err(failure)?;
        let mut functions = HashMap::new();
        for name in [
            "embedding",
            "check_finite",
            "matvec",
            "normalize",
            "head_norm",
            "rotary",
            "cache_kv",
            "attention_scores",
            "attention_values",
            "router",
            "swiglu",
            "residual",
            "combine",
            "prefill_matmul",
            "prefill_embedding",
            "group_experts",
            "grouped_matmul",
            "combine_batch",
            "attention_fused",
            "prefill_matmul_tiled",
            "group_experts_tiled",
            "grouped_matmul_tiled",
        ] {
            functions.insert(name, module.load_function(name).map_err(failure)?);
        }
        if options.matmul.tensor_cores() {
            for name in [
                "prefill_matmul_tf32",
                "grouped_matmul_tf32",
                "prefill_matmul_tf32x3",
                "grouped_matmul_tf32x3",
                "prefill_matmul_tf32x3_tiled",
                "grouped_matmul_tf32x3_tiled",
            ] {
                functions.insert(name, module.load_function(name).map_err(failure)?);
            }
        }
        if options.decode_graph {
            for name in [
                "embedding_decode",
                "rotary_decode",
                "cache_kv_decode",
                "attention_fused_decode",
                "greedy_partials",
                "greedy_finish",
            ] {
                functions.insert(name, module.load_function(name).map_err(failure)?);
            }
        }
        let weights = Weights::open(path, config)?;
        let s = &weights.spec;
        if options.decode_graph && !fused_attention_supported(s) {
            return Err(invalid(
                "decode graph requires head dimension <=256 and context <=1024",
            ));
        }
        if s.used > 65535
            || s.hidden
                .checked_mul(s.used)
                .is_none_or(|n| n > i32::MAX as usize)
        {
            return Err(invalid("expert batch exceeds CUDA indexing"));
        }
        let weight_bytes = weights.bytes();
        let kv_elements = (s.kv_heads as u64)
            .checked_mul(s.dim as u64)
            .and_then(|n| n.checked_mul(u64::from(config.context_tokens)))
            .ok_or_else(|| invalid("KV overflow"))?;
        let kv_bytes = kv_elements
            .checked_mul(weights.layers.len() as u64)
            .and_then(|n| n.checked_mul(8))
            .ok_or_else(|| invalid("KV byte overflow"))?;
        let routing = routing_elements(s)? as u64;
        let groups = if options.matmul.group_tile() == 4 {
            group_elements(s)?
        } else {
            group_elements_with(s, options.matmul.group_tile())?
        } as u64;
        let workspace_bytes = workspace_sizes(s)?
            .into_iter()
            .try_fold(0u64, |sum, n| sum.checked_add(n as u64))
            .and_then(|n| n.checked_add(routing * 2 + groups + 1))
            .and_then(|n| {
                n.checked_add(if prefill_capacity(s) > 1 {
                    prefill_capacity(s) as u64
                } else {
                    0
                })
            })
            .and_then(|n| n.checked_mul(4))
            .ok_or_else(|| invalid("workspace byte overflow"))?;
        let logits_bytes = u64::from(s.info.vocab_tokens) * 4;
        let reservation = weight_bytes
            .checked_add(kv_bytes)
            .and_then(|n| n.checked_add(workspace_bytes))
            .and_then(|n| n.checked_add(logits_bytes))
            .and_then(|n| {
                n.checked_add(if options.decode_graph {
                    decode::DecodeState::bytes(s.info.vocab_tokens as usize)
                } else {
                    0
                })
            })
            .ok_or_else(|| invalid("device reservation overflow"))?;
        let (free, _) = result::mem_get_info().map_err(failure)?;
        admit(reservation, options.memory_limit_bytes, free as u64)?;
        let memory = MemoryUsage {
            allocation_domain: "cuda",
            weights_allocated_bytes: weight_bytes,
            kv_allocated_bytes: kv_bytes,
            kv_live_bytes: 0,
            workspace_allocated_bytes: workspace_bytes,
            logits_allocated_bytes: logits_bytes,
            token_state_allocated_bytes: if options.decode_graph {
                decode::DecodeState::bytes(s.info.vocab_tokens as usize)
            } else {
                0
            },
            prefix_capacity_bytes: 0,
            prefix_allocated_bytes: 0,
            worker_scratch_bound_bytes: 0,
            numerical_reservation_bytes: reservation,
        };
        let stream = if options.decode_graph {
            // This backend owns one stream and never shares device buffers.
            unsafe {
                context.disable_event_tracking();
            }
            context.new_stream().map_err(failure)?
        } else {
            context.default_stream()
        };
        let decode = if options.decode_graph {
            Some(decode::DecodeState::new(
                &stream,
                s.info.vocab_tokens as usize,
            )?)
        } else {
            None
        };
        let work = Workspace::new(s, &stream, options.matmul)?;
        let mut keys = Vec::new();
        let mut values = Vec::new();
        for _ in &weights.layers {
            keys.push(stream.alloc_zeros(kv_elements as usize).map_err(failure)?);
            values.push(stream.alloc_zeros(kv_elements as usize).map_err(failure)?);
        }
        let logits = vec![0.0; s.info.vocab_tokens as usize];
        let mut weights = weights.map(|t| DeviceTensor::upload(t, &stream))?;
        weights.spec.info.backend = format!(
            "rust-cuda-packed:device{}:{}",
            options.device,
            context.name().map_err(failure)?
        );
        stream.synchronize().map_err(failure)?;
        let mut backend = Self {
            decode,
            weights,
            stream,
            functions,
            work,
            keys,
            values,
            logits,
            state: State::default(),
            memory,
            matmul: options.matmul,
            diagnostic: None,
        };
        if options.decode_graph {
            backend.capture_decode_graph()?;
        }
        Ok(backend)
    }

    fn step(
        &mut self,
        token: i32,
        position: usize,
        output_logits: bool,
        device_state: bool,
    ) -> Result<()> {
        let s = &self.weights.spec;
        let w = s.width as i32;
        let qw = (s.heads * s.dim) as i32;
        let kw = (s.kv_heads * s.dim) as i32;
        let dim = s.dim as i32;
        let pos = position as i32;
        let neox = i32::from(s.neox);
        let heads = s.heads as i32;
        let kv_heads = s.kv_heads as i32;
        let context = s.info.context_tokens as i32;
        let used = s.used as i32;
        let experts = s.experts as i32;
        let count = pos + 1;
        let vector = |n: i32| ((n as u32).div_ceil(256), 1, 1);
        let t = &self.weights.embedding;
        launch_position!(
            self,
            device_state,
            "embedding",
            "embedding_decode",
            vector(w),
            [&t.bytes, &t.kind, &mut self.work.x, &w],
            &token,
            []
        );
        for (index, l) in self.weights.layers.iter().enumerate() {
            launch!(
                self,
                "normalize",
                (1, 1, 1),
                &self.work.x,
                &l.norm.bytes,
                &mut self.work.normalized,
                &w,
                &s.epsilon
            );
            for (tensor, output) in [
                (&l.q, &mut self.work.q),
                (&l.k, &mut self.work.k),
                (&l.v, &mut self.work.v),
            ] {
                launch!(
                    self,
                    "matvec",
                    (tensor.rows as u32, 1, 1),
                    &tensor.bytes,
                    &tensor.kind,
                    &self.work.normalized,
                    output,
                    &tensor.width,
                    &tensor.rows,
                    &self.work.ids,
                    &0i32,
                    &0i32
                );
            }
            for (norm, vector, size) in [
                (&l.q_norm, &mut self.work.q, qw),
                (&l.k_norm, &mut self.work.k, kw),
            ] {
                if let Some(norm) = norm {
                    launch!(
                        self,
                        "head_norm",
                        ((size / dim) as u32, 1, 1),
                        &mut *vector,
                        &norm.bytes,
                        &dim,
                        &s.epsilon
                    );
                }
                launch_position!(
                    self,
                    device_state,
                    "rotary",
                    "rotary_decode",
                    (((size / 2) as u32).div_ceil(256), 1, 1),
                    [vector, &size, &dim],
                    &pos,
                    [&s.rope_base, &neox]
                );
            }
            launch_position!(
                self,
                device_state,
                "cache_kv",
                "cache_kv_decode",
                vector(kw),
                [
                    &self.work.k,
                    &self.work.v,
                    &mut self.keys[index],
                    &mut self.values[index],
                    &kw
                ],
                &pos,
                []
            );
            if device_state {
                let control = &self
                    .decode
                    .as_ref()
                    .ok_or_else(|| failure("decode control missing"))?
                    .control;
                launch!(
                    self,
                    "attention_fused_decode",
                    (heads as u32, 1, 1),
                    &self.work.q,
                    &self.keys[index],
                    &self.values[index],
                    &mut self.work.attended,
                    &dim,
                    &heads,
                    &kv_heads,
                    control
                );
            } else {
                launch!(
                    self,
                    "attention_scores",
                    (heads as u32, count as u32, 1),
                    &self.work.q,
                    &self.keys[index],
                    &mut self.work.scores,
                    &dim,
                    &heads,
                    &kv_heads,
                    &context,
                    &count
                );
                launch!(
                    self,
                    "attention_values",
                    (heads as u32, 1, 1),
                    &mut self.work.scores,
                    &self.values[index],
                    &mut self.work.attended,
                    &dim,
                    &heads,
                    &kv_heads,
                    &context,
                    &count
                );
            }
            let t = &l.output;
            launch!(
                self,
                "matvec",
                (t.rows as u32, 1, 1),
                &t.bytes,
                &t.kind,
                &self.work.attended,
                &mut self.work.projected,
                &t.width,
                &t.rows,
                &self.work.ids,
                &0i32,
                &0i32
            );
            launch!(
                self,
                "residual",
                vector(w),
                &mut self.work.x,
                &self.work.projected,
                &w
            );
            launch!(
                self,
                "normalize",
                (1, 1, 1),
                &self.work.x,
                &l.ffn_norm.bytes,
                &mut self.work.normalized,
                &w,
                &s.epsilon
            );
            if let Some(t) = &l.router {
                launch!(
                    self,
                    "matvec",
                    (t.rows as u32, 1, 1),
                    &t.bytes,
                    &t.kind,
                    &self.work.normalized,
                    &mut self.work.router,
                    &t.width,
                    &t.rows,
                    &self.work.ids,
                    &0i32,
                    &0i32
                );
                launch!(
                    self,
                    "router",
                    (1, 1, 1),
                    &self.work.router,
                    &experts,
                    &used,
                    &mut self.work.ids,
                    &mut self.work.probabilities,
                    &mut self.work.status
                );
            }
            let moe = i32::from(l.router.is_some());
            for (t, out) in [(&l.gate, &mut self.work.gate), (&l.up, &mut self.work.up)] {
                launch!(
                    self,
                    "matvec",
                    (t.rows as u32, used as u32, 1),
                    &t.bytes,
                    &t.kind,
                    &self.work.normalized,
                    out,
                    &t.width,
                    &t.rows,
                    &self.work.ids,
                    &moe,
                    &0i32
                );
            }
            let activated = (s.hidden * s.used) as i32;
            launch!(
                self,
                "swiglu",
                vector(activated),
                &mut self.work.gate,
                &self.work.up,
                &activated
            );
            let t = &l.down;
            launch!(
                self,
                "matvec",
                (t.rows as u32, used as u32, 1),
                &t.bytes,
                &t.kind,
                &self.work.gate,
                &mut self.work.down,
                &t.width,
                &t.rows,
                &self.work.ids,
                &moe,
                &1i32
            );
            launch!(
                self,
                "combine",
                vector(w),
                &mut self.work.x,
                &self.work.down,
                &self.work.probabilities,
                &w,
                &used
            );
        }
        if !output_logits {
            // Skipped logits must not hide a failed intermediate token's numerical state.
            launch!(
                self,
                "check_finite",
                vector(w),
                &self.work.x,
                &w,
                &mut self.work.status
            );
            return Ok(());
        }
        self.project_logits(0)
    }

    fn project_logits(&mut self, offset: usize) -> Result<()> {
        let s = &self.weights.spec;
        let w = s.width as i32;
        let input = self.work.x.slice(offset..offset + s.width);
        launch!(
            self,
            "normalize",
            (1, 1, 1),
            &input,
            &self.weights.output_norm.bytes,
            &mut self.work.normalized,
            &w,
            &s.epsilon
        );
        let t = self
            .weights
            .output
            .as_ref()
            .unwrap_or(&self.weights.embedding);
        launch!(
            self,
            "matvec",
            (t.rows as u32, 1, 1),
            &t.bytes,
            &t.kind,
            &self.work.normalized,
            &mut self.work.logits,
            &t.width,
            &t.rows,
            &self.work.ids,
            &0i32,
            &0i32
        );
        Ok(())
    }

    fn finish_evaluation(&mut self) -> Result<()> {
        self.stream
            .memcpy_dtoh(&self.work.logits, &mut self.logits)
            .map_err(failure)?;
        let mut status = [0i32];
        self.stream
            .memcpy_dtoh(&self.work.status, &mut status[..])
            .map_err(failure)?;
        self.stream.synchronize().map_err(failure)?;
        if status[0] != 0 || self.logits.iter().any(|x| !x.is_finite()) {
            return Err(failure(
                "nonfinite router scores or logits; request poisoned",
            ));
        }
        Ok(())
    }
}

fn admit(reservation: u64, budget: u64, free: u64) -> Result<()> {
    let headroom = 2 * 1024 * 1024 * 1024;
    if reservation > budget || reservation > free.saturating_sub(headroom) {
        return Err(invalid(format!(
            "device buffers need {reservation} bytes; budget {budget}, free {free}; preserving 2 GiB headroom"
        )));
    }
    Ok(())
}

impl Backend for CudaBackend {
    fn info(&self) -> &ModelInfo {
        &self.weights.spec.info
    }
    fn logits(&self) -> Result<&[f32]> {
        if self.decode.as_ref().is_some_and(|d| d.sampled.is_some()) {
            return Err(failure(
                "token-only evaluation; use evaluate to materialize host logits",
            ));
        }
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
        self.stream.synchronize().map_err(failure)?;
        self.stream
            .memset_zeros(&mut self.work.status)
            .map_err(failure)?;
        self.stream.synchronize().map_err(failure)?;
        self.state = State::default();
        self.diagnostic = None;
        if let Some(decode) = &mut self.decode {
            decode.sampled = None;
        }
        Ok(())
    }
    fn evaluate(&mut self, tokens: &[i32], position: u32) -> Result<()> {
        self.evaluate_inner(tokens, position, true)
    }
    fn evaluate_for_sampling(&mut self, tokens: &[i32], position: u32) -> Result<()> {
        self.evaluate_inner(tokens, position, self.decode.is_none())
    }
    fn sample_tokens(&mut self, maximum: usize, mode: crate::GenerationMode) -> Result<Vec<i32>> {
        self.sample_decode_tokens(maximum, mode)
    }
    fn sample_greedy(&mut self) -> Result<i32> {
        self.state.logits(&[])?;
        if let Some(token) = self.decode.as_ref().and_then(|d| d.sampled) {
            Ok(token)
        } else {
            greedy(self.logits()?)
        }
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

impl CudaBackend {
    fn evaluate_inner(&mut self, tokens: &[i32], position: u32, host_logits: bool) -> Result<()> {
        self.state
            .begin(tokens, position, &self.weights.spec.info)?;
        if let Some(capture) = &self.diagnostic {
            capture.validate_batch(tokens.len(), prefill_capacity(&self.weights.spec))?;
        }
        if let Some(decode) = &mut self.decode {
            decode.sampled = None;
        }
        let capacity = prefill_capacity(&self.weights.spec);
        for (index, chunk) in tokens.chunks(capacity).enumerate() {
            let offset = index * capacity;
            let output_logits = offset + chunk.len() == tokens.len();
            if chunk.len() > 1 {
                self.batched_prefill(chunk, position as usize + offset, output_logits)?;
            } else if self.decode.is_some() && output_logits {
                self.replay_decode(chunk[0], position as usize + offset)?;
            } else {
                self.step(chunk[0], position as usize + offset, output_logits, false)?;
            }
        }
        if host_logits {
            self.finish_evaluation()?;
        } else {
            // Batched prefill is not captured, so run its final device reduction here.
            if tokens.len() != 1 {
                self.prepare_decode(0, position as usize + tokens.len() - 1, 0)?;
                self.device_sample()?;
            }
            let sampled = self.read_decode_tokens(1)?[0];
            self.decode.as_mut().unwrap().sampled = Some(sampled);
        }
        self.state.finish(tokens.len());
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    pub(super) fn packed_fixture(kind: u32, width: usize, rows: usize, experts: usize) -> Tensor {
        use half::f16;
        let (block, size, _) = crate::gguf::layout(kind).unwrap();
        let mut bytes = vec![0u8; width * rows * experts / block * size];
        for (b, data) in bytes.chunks_exact_mut(size).enumerate() {
            for (i, byte) in data.iter_mut().enumerate() {
                *byte = ((b * 17 + i * 13) % 251) as u8;
            }
            if kind == 0 {
                data.copy_from_slice(&((b % 31) as f32 / 64.0).to_le_bytes());
            } else {
                let offset = if kind == 14 { 208 } else { 0 };
                data[offset..offset + 2]
                    .copy_from_slice(&f16::from_f32(0.015625).to_bits().to_le_bytes());
                if kind == 12 {
                    data[2..4].copy_from_slice(&f16::from_f32(0.0078125).to_bits().to_le_bytes());
                }
            }
        }
        let storage = if kind == 0 {
            Storage::F32(
                bytes
                    .chunks_exact(4)
                    .map(|p| f32::from_le_bytes(p.try_into().unwrap()))
                    .collect(),
            )
        } else {
            Storage::Packed { kind, bytes }
        };
        Tensor {
            dims: vec![width, rows, experts],
            storage,
            accelerated: false,
            block_elements: block,
        }
    }

    #[test]
    fn admission_errors_preserve_logits_and_failed_execution_requires_reset() {
        let info = ModelInfo {
            backend: "test".into(),
            revision: "test".into(),
            architecture: "test".into(),
            tensor_types: vec![],
            weight_bytes: 0,
            parameters: 0,
            context_tokens: 4,
            batch_tokens: 2,
            vocab_tokens: 3,
            capabilities: crate::backend::Capabilities {
                token_step: true,
                chunked_prefill: true,
            },
        };
        let mut state = State::default();
        assert!(state.logits(&[0.0]).is_err());
        state.begin(&[0, 1], 0, &info).unwrap();
        state.finish(2);
        for (tokens, position) in [(vec![3], 2), (vec![0], 1), (vec![0, 1, 2], 2)] {
            assert!(state.begin(&tokens, position, &info).is_err());
            assert!(state.logits(&[0.0]).is_ok());
        }
        state.begin(&[0, 2], 2, &info).unwrap();
        assert!(state.logits(&[0.0]).is_err());
        assert!(state.begin(&[0], 2, &info).is_err());
        state.finish(2);
        assert!(state.begin(&[0], 4, &info).is_err());
    }

    #[test]
    #[ignore = "requires a CUDA 13 NVRTC/driver and a free GPU; run serially in release mode"]
    fn cuda_quantizers_expert_offsets_and_router_match_independent_cpu_kernels() -> Result<()> {
        use crate::gguf::layout;
        let context = CudaContext::new(0).unwrap();
        let module = context
            .load_module(
                compile_ptx_with_opts(
                    include_str!("cuda/kernels.cu"),
                    CompileOptions {
                        fmad: Some(false),
                        arch: Some("compute_75"),
                        ..Default::default()
                    },
                )
                .unwrap(),
            )
            .unwrap();
        struct Harness {
            stream: Arc<CudaStream>,
            functions: HashMap<&'static str, CudaFunction>,
        }
        let harness = Harness {
            stream: context.default_stream(),
            functions: ["matvec", "router", "check_finite"]
                .into_iter()
                .map(|name| (name, module.load_function(name).unwrap()))
                .collect(),
        };
        for (kind, width) in [0, 1, 6, 8, 12, 14].into_iter().flat_map(|kind| {
            let block = layout(kind).unwrap().0;
            [block, 512, 768, 2048]
                .into_iter()
                .map(move |width| (kind, width))
        }) {
            let rows = 3;
            let tensor = packed_fixture(kind, width, rows, 3);
            let input: Vec<f32> = (0..width * 2)
                .map(|i| (i % 23) as f32 / 23.0 - 0.5)
                .collect();
            let expected: Vec<_> = (0..=1)
                .map(|batched| {
                    let mut out = vec![0.0; rows * 2];
                    for (slot, expert) in [2, 0].into_iter().enumerate() {
                        let start = if batched == 1 { slot * width } else { 0 };
                        tensor.multiply_expert(
                            expert,
                            &input[start..start + width],
                            &mut out[slot * rows..(slot + 1) * rows],
                        );
                    }
                    out
                })
                .collect();
            let t = DeviceTensor::upload(tensor, &harness.stream).unwrap();
            let x = harness.stream.clone_htod(&input).unwrap();
            let ids = harness.stream.clone_htod(&[2i32, 0]).unwrap();
            let mut out = harness.stream.alloc_zeros::<f32>(rows * 2).unwrap();
            for (batched, expected) in expected.iter().enumerate() {
                let batched = batched as i32;
                launch!(
                    harness,
                    "matvec",
                    (rows as u32, 2, 1),
                    &t.bytes,
                    &t.kind,
                    &x,
                    &mut out,
                    &t.width,
                    &t.rows,
                    &ids,
                    &1i32,
                    &batched
                );
                let actual = harness.stream.clone_dtoh(&out).unwrap();
                for (a, e) in actual.iter().zip(expected) {
                    assert!(
                        (a - e).abs() < 0.0001 + e.abs() * 1e-5,
                        "kind={kind}, width={width}, batched={batched}: {a} != {e}"
                    );
                }
            }
        }
        for (scores, used) in [
            (vec![1000.0, 1000.0, 999.0, -1000.0], 3),
            (vec![-0.0, 0.0, 0.0, -1.0], 3),
            (vec![f32::NAN, 1.0, 0.0, -1.0], 3),
            (vec![0.0; 128], 8),
            (
                (0..128)
                    .map(|i| ((i * 13 + 7) % 31) as f32 * 0.125)
                    .collect(),
                8,
            ),
            (
                (0..513)
                    .map(|i| ((i * 17 + 3) % 47) as f32 * 0.25)
                    .collect(),
                17,
            ),
            (
                {
                    let mut scores = vec![1.0; 513];
                    scores[330] = f32::INFINITY;
                    scores
                },
                8,
            ),
        ] {
            let count = scores.len() as i32;
            let selected = used as i32;
            let x = harness.stream.clone_htod(&scores).unwrap();
            let mut ids = harness.stream.alloc_zeros::<i32>(used).unwrap();
            let mut probabilities = harness.stream.alloc_zeros::<f32>(used).unwrap();
            let mut status = harness.stream.alloc_zeros::<i32>(1).unwrap();
            launch!(
                harness,
                "router",
                (1, 1, 1),
                &x,
                &count,
                &selected,
                &mut ids,
                &mut probabilities,
                &mut status
            );
            let flag = harness.stream.clone_dtoh(&status).unwrap()[0];
            if scores.iter().any(|v| !v.is_finite()) {
                assert_eq!(flag, 1);
                continue;
            }
            assert_eq!(flag, 0);
            let expected = crate::transformer::route(&scores, used).unwrap();
            assert_eq!(
                harness.stream.clone_dtoh(&ids).unwrap(),
                expected.iter().map(|p| p.0 as i32).collect::<Vec<_>>()
            );
            for (actual, expected) in harness
                .stream
                .clone_dtoh(&probabilities)
                .unwrap()
                .iter()
                .zip(expected)
            {
                assert!((actual - expected.1).abs() < 1e-6);
            }
        }
        let mut status = harness.stream.alloc_zeros::<i32>(1).unwrap();
        for values in [[f32::NAN, 1.0], [1.0, 2.0]] {
            let x = harness.stream.clone_htod(&values).unwrap();
            launch!(harness, "check_finite", (1, 1, 1), &x, &2i32, &mut status);
        }
        assert_eq!(harness.stream.clone_dtoh(&status).unwrap(), [1]);
        harness.stream.memset_zeros(&mut status).unwrap();
        let x = harness.stream.clone_htod(&[1.0f32, 2.0]).unwrap();
        launch!(harness, "check_finite", (1, 1, 1), &x, &2i32, &mut status);
        assert_eq!(harness.stream.clone_dtoh(&status).unwrap(), [0]);
        Ok(())
    }

    #[test]
    fn device_budget_checks_exact_boundary_and_shared_gpu_headroom() {
        let free = 3 * 1024 * 1024 * 1024;
        assert!(admit(1234, 1234, free).is_ok());
        assert!(admit(1234, 1233, free).is_err());
        assert!(admit(1025 * 1024 * 1024, free, free).is_err());
        assert!(admit(1, 1, 0).is_err());
    }
}
