//! Independent CUDA kernels over packed GGUF weights. No GGML or native inference engine.
pub mod diagnostics;

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
    /// Experimental staged/register F32 experts and packed decode; no activation quantization.
    #[serde(rename = "f32-tiled")]
    F32Tiled,
    /// Experimental staged three-product TF32 experts; failed strict MoE quality gates.
    #[serde(rename = "tf32x3-tiled")]
    Tf32x3Tiled,
    /// Research-only residual INT8 prefill; full-model quality is not admitted.
    #[serde(rename = "residual-int8-research")]
    ResidualInt8Research,
    /// Research-only three-component integer MMA prefill; no production promotion.
    #[serde(rename = "residual-mma3-research")]
    ResidualMma3Research,
}

impl CudaMatmul {
    fn f32_tiled(self) -> bool {
        matches!(
            self,
            Self::F32Tiled | Self::ResidualInt8Research | Self::ResidualMma3Research
        )
    }

    fn residual_components(self) -> usize {
        match self {
            Self::ResidualInt8Research => 2,
            Self::ResidualMma3Research => 3,
            _ => 0,
        }
    }

    fn residual_min_tokens(self) -> usize {
        if self == Self::ResidualMma3Research {
            32
        } else {
            256
        }
    }

    fn tensor_cores(self) -> bool {
        matches!(self, Self::Tf32 | Self::Tf32x3 | Self::Tf32x3Tiled)
    }

    fn group_tile(self) -> usize {
        if self.f32_tiled() || self == Self::Tf32x3Tiled {
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
    #[cfg(feature = "cuda-qwen35-matvec-cache-research")]
    k_coefficients: Option<CudaSlice<f32>>,
    #[cfg(feature = "cuda-fp16-corrected-prefill-research")]
    fp16: Option<[CudaSlice<u16>; 2]>,
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
            #[cfg(feature = "cuda-qwen35-matvec-cache-research")]
            k_coefficients: None,
            #[cfg(feature = "cuda-fp16-corrected-prefill-research")]
            fp16: None,
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
    group_counts: Option<CudaSlice<i32>>,
    residual_int8: Option<ResidualWorkspace>,
    residual_mma3: Option<PreciseWorkspace>,
}

struct ResidualWorkspace {
    q0: CudaSlice<i8>,
    q1: CudaSlice<i8>,
    scales: CudaSlice<f32>,
    sums: CudaSlice<i32>,
}

struct PreciseWorkspace {
    values: CudaSlice<i8>,
    scales: CudaSlice<f32>,
    sums: CudaSlice<i32>,
}

fn residual_workspace_elements(spec: &Spec, matmul: CudaMatmul) -> Result<usize> {
    if matmul.residual_components() == 0 || prefill_capacity(spec) < matmul.residual_min_tokens() {
        return Ok(0);
    }
    let sizes = workspace_sizes(spec)?;
    let largest = sizes[0].max(sizes[2]).max(sizes[7]);
    largest
        .checked_add(31)
        .map(|n| n / 32 * 32)
        .filter(|&n| {
            n <= i32::MAX as usize
                / if matmul == CudaMatmul::ResidualMma3Research {
                    3
                } else {
                    1
                }
        })
        .ok_or_else(|| invalid("residual INT8 workspace exceeds CUDA indexing"))
}

fn residual_workspace_bytes(spec: &Spec, matmul: CudaMatmul) -> Result<u64> {
    let count = residual_workspace_elements(spec, matmul)? as u64;
    let components = matmul.residual_components() as u64;
    count
        .checked_mul(components)
        .and_then(|n| n.checked_add(count / 32 * components * 8))
        .ok_or_else(|| invalid("residual INT8 workspace byte overflow"))
}

fn prefill_capacity(spec: &Spec) -> usize {
    (spec.info.batch_tokens as usize).min(512)
}

fn decode_score_reuse_options(arch: String, f32_profile: bool) -> Vec<String> {
    let mut options = vec![arch];
    if cfg!(feature = "cuda-decode-score-reuse-research") && f32_profile {
        options.push("-DCUDA_DECODE_SCORE_REUSE_RESEARCH=1".into());
    }
    options
}

fn fused_attention_supported(spec: &Spec) -> bool {
    spec.dim <= 256
        && spec.info.context_tokens <= 1024
        && !matches!(spec.info.architecture.as_str(), "gemma2" | "gemma3")
}

#[cfg(any(
    feature = "cuda-qwen3-batched-attention-research",
    feature = "cuda-llama-batched-attention-research"
))]
fn batched_attention_supported(spec: &Spec) -> bool {
    let shape = (spec.heads, spec.kv_heads, spec.dim);
    let model = match spec.info.architecture.as_str() {
        "qwen3" => cfg!(feature = "cuda-qwen3-batched-attention-research") && shape == (16, 8, 128),
        "llama" => cfg!(feature = "cuda-llama-batched-attention-research") && shape == (32, 8, 64),
        _ => false,
    };
    model && spec.experts == 0 && fused_attention_supported(spec) && prefill_capacity(spec) >= 256
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
    // At most tile-1 padding slots per nonempty expert; a descriptor holds
    // one expert ID followed by tile original assignment IDs.
    assignments
        .div_ceil(tile)
        .checked_add(spec.experts - 1)
        .map(|n| n.min(assignments))
        .and_then(|n| n.checked_mul(tile + 1))
        .ok_or_else(|| invalid("group workspace overflow"))
}

fn register_group_shape(spec: &Spec, matmul: CudaMatmul, tokens: usize) -> bool {
    matmul.f32_tiled()
        && tokens >= 256
        && spec.width == 2048
        && spec.hidden == 768
        && spec.experts == 128
        && spec.used == 8
}

fn group_workspace_elements(spec: &Spec, matmul: CudaMatmul) -> Result<usize> {
    let base = if matmul.group_tile() == 4 {
        group_elements(spec)?
    } else {
        group_elements_with(spec, matmul.group_tile())?
    };
    if register_group_shape(spec, matmul, prefill_capacity(spec))
        || (matmul.residual_components() > 0
            && prefill_capacity(spec) >= matmul.residual_min_tokens())
    {
        Ok(base.max(group_elements_with(spec, 32)?))
    } else {
        Ok(base)
    }
}

const QWEN3_DECODE_TILE: u32 = if cfg!(feature = "cuda-qwen3-decode-tile-128-research") {
    128
} else {
    160
};

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
    if cfg!(feature = "cuda-qwen3-decode-tile-128-research")
        && spec.info.architecture == "qwen3"
        && spec.dim == 128
        && fused_attention_supported(spec)
    {
        let tiles = spec.info.context_tokens.div_ceil(QWEN3_DECODE_TILE) as usize;
        sizes[10] = sizes[10].max(product(spec.heads, product(tiles, 130)?)?);
    } else if !fused_attention_supported(spec) {
        sizes[10] = product(sizes[10], prefill_capacity(spec))?;
    }
    #[cfg(any(
        feature = "cuda-qwen3-batched-attention-research",
        feature = "cuda-llama-batched-attention-research"
    ))]
    if batched_attention_supported(spec) {
        let capacity = prefill_capacity(spec).min(512);
        sizes[10] = sizes[10].max(product(spec.heads, product(capacity, capacity)?)?);
    }
    if spec.experts > 0 {
        sizes[11] = product(sizes[11], prefill_capacity(spec))?;
    }
    if sizes[..10].iter().any(|&n| n > i32::MAX as usize) {
        return Err(invalid("batched vectors exceed CUDA indexing"));
    }
    Ok(sizes)
}

fn parallel_group_count_elements(spec: &Spec, matmul: CudaMatmul) -> usize {
    if cfg!(feature = "cuda-parallel-grouping-research")
        && matmul == CudaMatmul::ResidualMma3Research
        && prefill_capacity(spec) >= matmul.residual_min_tokens()
    {
        spec.experts
    } else {
        0
    }
}

impl Workspace {
    fn new(
        spec: &Spec,
        stream: &Arc<CudaStream>,
        matmul: CudaMatmul,
        precise: CudaMatmul,
    ) -> Result<Self> {
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
            groups: if group_workspace_elements(spec, matmul)? > 0 {
                Some(
                    stream
                        .alloc_zeros(group_workspace_elements(spec, matmul)?)
                        .map_err(failure)?,
                )
            } else {
                None
            },
            group_counts: match parallel_group_count_elements(spec, matmul) {
                0 => None,
                n => Some(stream.alloc_zeros(n).map_err(failure)?),
            },
            residual_int8: {
                let count = residual_workspace_elements(spec, matmul)?;
                if count == 0 || matmul != CudaMatmul::ResidualInt8Research {
                    None
                } else {
                    Some(ResidualWorkspace {
                        q0: stream.alloc_zeros(count).map_err(failure)?,
                        q1: stream.alloc_zeros(count).map_err(failure)?,
                        scales: alloc(count / 32 * 2)?,
                        sums: stream.alloc_zeros(count / 32 * 2).map_err(failure)?,
                    })
                }
            },
            residual_mma3: {
                let count = residual_workspace_elements(spec, precise)?;
                if count == 0 || precise != CudaMatmul::ResidualMma3Research {
                    None
                } else {
                    Some(PreciseWorkspace {
                        values: stream.alloc_zeros(count * 3).map_err(failure)?,
                        scales: alloc(count / 32 * 3)?,
                        sums: stream.alloc_zeros(count / 32 * 3).map_err(failure)?,
                    })
                }
            },
        })
    }
}

macro_rules! launch {
    ($backend:ident, $name:expr, $grid:expr; $threads:expr, $($arg:expr),+ $(,)?) => {{
        let cfg = LaunchConfig { grid_dim: $grid, block_dim: ($threads, 1, 1), shared_mem_bytes: 0 };
        let mut args = $backend.stream.launch_builder(&$backend.functions[$name]);
        $(args.arg($arg);)+
        // Kernel signatures, validated shapes, and per-kernel thread counts agree.
        unsafe { args.launch(cfg) }.map_err(failure)?;
    }};
    ($backend:ident, $name:expr, $grid:expr, $($arg:expr),+ $(,)?) => {{
        launch!($backend, $name, $grid; 256, $($arg),+)
    }};
}

macro_rules! q8_dp4a {
    ($b:ident, $t:expr, $x:ident, $y:expr) => {{
        q8_dp4a!($b, $t, $x, $y, None::<i32>)
    }};
    ($b:ident, $t:expr, $x:ident, $y:expr, $prepared:expr) => {{
        let t = $t;
        if cfg!(feature = "cuda-llama-q8-dp4a-research")
            && t.kind == 8
            && t.width > 0
            && t.width % 32 == 0
            && $b.functions.contains_key("llama_q8_dp4a")
            && let Some(work) = $b.work.residual_mma3.as_mut()
        {
            let width = t.width as usize;
            if width > $b.work.$x.len()
                || 3 * width > work.values.len()
                || 3 * (width / 32) > work.scales.len().min(work.sums.len())
            {
                return Err(invalid("Q8 DP4A input exceeds workspace"));
            }

            if !cfg!(feature = "cuda-llama-q8-prep-reuse-research") || $prepared != Some(t.width) {
                launch!($b, "residual_quantize32_precise3", ((t.width as u32).div_ceil(256), 1, 1),
                    &$b.work.$x, &mut work.values, &mut work.scales, &mut work.sums,
                    &t.width, &mut $b.work.status);
            }
            launch!($b, "llama_q8_dp4a", ((t.rows as u32).div_ceil(4), 1, 1); 128,
                &t.bytes, &work.values, &work.scales, $y, &t.width, &t.rows);
            true
        } else {
            false
        }
    }};
}

macro_rules! qwen3_matvec_128 {
    ($b:ident, $t:expr, $x:ident, $y:expr) => {{
        let t = $t;
        if cfg!(feature = "cuda-qwen3-layer-matvec-128-research")
            && matches!(t.kind, 12 | 14)
            && t.width % 256 == 0
            && $b.functions.contains_key("qwen3_layer_matvec_128")
        {
            launch!($b, "qwen3_layer_matvec_128", (t.rows as u32, 1, 1); 128,
                &t.bytes, &t.kind, &$b.work.$x, $y, &t.width);
            true
        } else {
            false
        }
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

macro_rules! alibi_attention {
    ($backend:ident, $layer:expr, $heads:expr, $kv_heads:expr, $dim:expr, $context:expr,
     $count:expr, $tokens:expr) => {{
        let tokens = $tokens;
        launch!(
            $backend,
            "alibi_attention_scores",
            ($heads as u32, ($count + tokens - 1) as u32, tokens as u32),
            &$backend.work.q,
            &$backend.keys[$layer],
            &mut $backend.work.scores,
            &$dim,
            &$heads,
            &$kv_heads,
            &$context,
            &$count,
            &crate::transformer::BLOOM_ALIBI_MAX
        );
        launch!(
            $backend,
            "attention_values",
            ($heads as u32, tokens as u32, 1),
            &mut $backend.work.scores,
            &$backend.values[$layer],
            &mut $backend.work.attended,
            &$dim,
            &$heads,
            &$kv_heads,
            &$context,
            &$count
        );
    }};
}

macro_rules! gemma_attention {
    ($backend:ident, $layer:expr, $heads:expr, $kv_heads:expr, $dim:expr, $context:expr,
     $count:expr, $tokens:expr, $window:expr, $scale:expr, $softcap:expr) => {{
        let tokens = $tokens;
        launch!(
            $backend,
            "gemma_attention_scores",
            ($heads as u32, ($count + tokens - 1) as u32, tokens as u32),
            &$backend.work.q,
            &$backend.keys[$layer],
            &mut $backend.work.scores,
            &$dim,
            &$heads,
            &$kv_heads,
            &$context,
            &$count,
            &$window,
            &$scale,
            &$softcap
        );
        launch!(
            $backend,
            "gemma_attention_values",
            ($heads as u32, tokens as u32, 1),
            &mut $backend.work.scores,
            &$backend.values[$layer],
            &mut $backend.work.attended,
            &$dim,
            &$heads,
            &$kv_heads,
            &$context,
            &$count,
            &$window
        );
    }};
}

macro_rules! gemma_post_norm {
    ($backend:ident, $input:ident, $norm:expr, $tokens:expr, $width:expr, $epsilon:expr) => {{
        let elements = $tokens * $width;
        launch!(
            $backend,
            "normalize",
            ($tokens as u32, 1, 1),
            &$backend.work.$input,
            &$norm.bytes,
            &mut $backend.work.normalized,
            &$width,
            &$epsilon
        );
        launch!(
            $backend,
            "residual",
            ((elements as u32).div_ceil(256), 1, 1),
            &mut $backend.work.x,
            &$backend.work.normalized,
            &elements
        );
    }};
}

macro_rules! transformer_norm {
    ($backend:ident, $input:expr, $weight:expr, $bias:expr, $n:expr, $width:expr, $eps:expr) => {{
        let grid = ($n as u32, 1, 1);
        if let (Some(weight), Some(bias)) = ($weight, $bias) {
            launch!(
                $backend,
                "layer_norm_affine",
                grid,
                $input,
                &weight.bytes,
                &bias.bytes,
                &mut $backend.work.normalized,
                &$width,
                &$eps
            );
        } else if let Some(weight) = $weight {
            launch!(
                $backend,
                "normalize",
                grid,
                $input,
                &weight.bytes,
                &mut $backend.work.normalized,
                &$width,
                &$eps
            );
        } else {
            launch!(
                $backend,
                "layer_norm_plain",
                grid,
                $input,
                &mut $backend.work.normalized,
                &$width,
                &$eps
            );
        }
    }};
}

macro_rules! affine_ffn {
    ($backend:ident, $layer:expr, $bias:expr, $n:expr, $width:expr, $parallel:expr) => {{
        let (layer, bias, n) = ($layer, $bias, $n);
        let up = &layer.up;
        let down = &layer.down;
        let tiles = (n as u32).div_ceil(4);
        launch!(
            $backend,
            "prefill_matmul",
            (up.rows as u32, tiles, 1),
            &up.bytes,
            &up.kind,
            &$backend.work.normalized,
            &mut $backend.work.up,
            &up.width,
            &up.rows,
            &n
        );
        launch!(
            $backend,
            "gelu_row_bias",
            ((up.rows as u32).div_ceil(256), n as u32, 1),
            &mut $backend.work.up,
            &bias.up.bytes,
            &up.rows
        );
        launch!(
            $backend,
            "prefill_matmul",
            (down.rows as u32, tiles, 1),
            &down.bytes,
            &down.kind,
            &$backend.work.up,
            &mut $backend.work.down,
            &down.width,
            &down.rows,
            &n
        );
        launch!(
            $backend,
            "add_row_bias",
            (($width as u32).div_ceil(256), n as u32, 1),
            &mut $backend.work.down,
            &bias.down.bytes,
            &$width
        );
        let count = n * $width;
        launch!(
            $backend,
            "residual",
            ((count as u32).div_ceil(256), 1, 1),
            &mut $backend.work.x,
            &$backend.work.down,
            &count
        );
        if $parallel {
            launch!(
                $backend,
                "residual",
                ((count as u32).div_ceil(256), 1, 1),
                &mut $backend.work.x,
                &$backend.work.projected,
                &count
            );
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
mod model_quality;
#[cfg(test)]
mod model_replay;
#[cfg(test)]
mod packed_decode;
#[cfg(test)]
mod prefill;
pub mod reference;
#[cfg(test)]
mod residual;
#[cfg(test)]
mod residual_precise;

pub struct CudaBackend {
    // Destroy captured references before their buffers and modules.
    decode: Option<decode::DecodeState>,
    device_greedy: Option<decode::DeviceGreedy>,
    #[cfg(any(
        feature = "cuda-qwen3-batched-attention-research",
        feature = "cuda-llama-batched-attention-research"
    ))]
    attention_blas: Option<cudarc::cublas::CudaBlas>,
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
    #[cfg(feature = "cuda-fp16-corrected-prefill-research")]
    fp16: Option<fp16_gemm::Prefill>,
    #[cfg(feature = "cuda-llama-ffn-mmq-research")]
    ffn_mmq: Option<ffn_mmq::ReplayKernel>,
}

/// Open the native CUDA backend for the model's architecture.
pub fn open_backend(
    path: &Path,
    config: &Config,
    options: CudaOptions,
) -> Result<Box<dyn Backend>> {
    let metadata = crate::gguf::Gguf::read_metadata(path)?;
    if matches!(metadata.get("general.architecture"), Some(crate::gguf::Value::Text(arch)) if arch == "qwen35")
    {
        return reference::qwen35::device::open(path, config, options);
    }
    Ok(Box::new(CudaBackend::open(path, config, options)?))
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
        if options.matmul == CudaMatmul::ResidualInt8Research
            && (major < 6 || (major == 6 && minor < 1))
        {
            return Err(invalid(
                "residual INT8 research requires compute capability >=6.1",
            ));
        }
        if options.matmul == CudaMatmul::ResidualMma3Research && major < 8 {
            return Err(invalid(
                "residual MMA3 research requires compute capability >=8.0",
            ));
        }
        let base_source = concat!(
            include_str!("cuda/kernels.cu"),
            "\n",
            include_str!("cuda/prefill.cu"),
            "\n",
            include_str!("cuda/attention.cu")
        );
        #[cfg(feature = "cuda-packed-reduction-research")]
        let packed_reduction_source;
        #[cfg(feature = "cuda-packed-reduction-research")]
        let base_source = if options.matmul == CudaMatmul::ResidualMma3Research {
            let prefix = if cfg!(feature = "cuda-packed-down-research") {
                "#define CUDA_PACKED_DOWN_RESEARCH 1\n"
            } else {
                ""
            };
            let gate_up = if cfg!(feature = "cuda-packed-gate-up-research") {
                "#define CUDA_PACKED_GATE_UP_RESEARCH 1\n"
            } else {
                ""
            };
            packed_reduction_source =
                format!("{prefix}{gate_up}#define CUDA_PACKED_REDUCTION_RESEARCH 1\n{base_source}");
            &packed_reduction_source
        } else {
            base_source
        };
        #[cfg(feature = "cuda-block-reduction-research")]
        let block_reduction_source;
        #[cfg(feature = "cuda-block-reduction-research")]
        let base_source = if options.matmul == CudaMatmul::ResidualMma3Research {
            block_reduction_source =
                format!("#define CUDA_BLOCK_REDUCTION_RESEARCH 1\n{base_source}");
            &block_reduction_source
        } else {
            base_source
        };
        let base_source = if options.decode_graph || cfg!(feature = "cuda-device-greedy-research") {
            format!("{base_source}\n{}", include_str!("cuda/decode.cu"))
        } else {
            base_source.to_owned()
        };
        let source = if options.matmul.tensor_cores() {
            format!("{base_source}\n{}", include_str!("cuda/tensorcore.cu"))
        } else {
            base_source
        };
        let source = match options.matmul {
            CudaMatmul::ResidualInt8Research => format!(
                "{source}\n{}\n{}",
                include_str!("cuda/residual.cu"),
                include_str!("cuda/residual_tile.cu")
            ),
            CudaMatmul::ResidualMma3Research => format!(
                "{source}\n{}\n{}\n{}",
                include_str!("cuda/residual.cu"),
                include_str!("cuda/residual_precise.cu"),
                include_str!("cuda/residual_mma.cu")
            ),
            _ => source,
        };
        let source = if cfg!(feature = "cuda-parallel-grouping-research")
            && options.matmul == CudaMatmul::ResidualMma3Research
        {
            format!("{source}\n{}", include_str!("cuda/grouping_parallel.cu"))
        } else {
            source
        };
        let source = if options.matmul.f32_tiled() {
            format!(
                "#define CUDA_PACKED_DECODE 1\n#define CUDA_REGISTER_PREFILL 1\n#define CUDA_NATIVE_HALF 1\n{source}"
            )
        } else {
            source
        };
        let source = if options.matmul == CudaMatmul::ResidualMma3Research {
            format!("#define CUDA_NORMALIZED_ATTENTION 1\n{source}")
        } else {
            source
        };
        let source = if cfg!(feature = "cuda-mma3-wide-rows-research")
            && options.matmul == CudaMatmul::ResidualMma3Research
        {
            format!("#define CUDA_MMA3_WIDE_ROWS_RESEARCH 1\n{source}")
        } else {
            source
        };
        let source = if cfg!(feature = "cuda-mma3-splitk-research")
            && options.matmul == CudaMatmul::ResidualMma3Research
        {
            format!("#define CUDA_MMA3_SPLITK_RESEARCH 1\n{source}")
        } else {
            source
        };
        let source = if cfg!(feature = "cuda-decode-fusion-research")
            && options.matmul == CudaMatmul::ResidualMma3Research
        {
            format!("#define CUDA_DECODE_FUSION_RESEARCH 1\n{source}")
        } else {
            source
        };
        #[cfg(feature = "cuda-attention-reuse-research")]
        let source = if options.matmul == CudaMatmul::ResidualMma3Research {
            let pair = if cfg!(feature = "cuda-attention-query-pair-research") {
                "#define CUDA_ATTENTION_QUERY_PAIR_RESEARCH 1\n"
            } else {
                ""
            };
            let batched = if cfg!(feature = "cuda-qwen3-batched-attention-research") {
                "#define CUDA_QWEN3_BATCHED_ATTENTION 1\n"
            } else {
                ""
            };
            format!("{pair}{batched}#define CUDA_ATTENTION_REUSE_RESEARCH 1\n{source}")
        } else {
            source
        };
        let qwen3_two_rows = cfg!(feature = "cuda-qwen3-prefill-2x16-research")
            && options.matmul == CudaMatmul::F32
            && matches!(
                crate::gguf::Gguf::read_metadata(path)?.get("general.architecture"),
                Some(crate::gguf::Value::Text(arch)) if arch == "qwen3"
            );
        let mut compiler_options = decode_score_reuse_options(
            format!("--gpu-architecture=compute_{major}{minor}"),
            options.matmul == CudaMatmul::F32,
        );
        if qwen3_two_rows {
            compiler_options.push("-DCUDA_REGISTER_PREFILL=1".into());
        }
        let packed_llama = cfg!(feature = "cuda-packed-prefill-research")
            && options.matmul == CudaMatmul::F32
            && !options.decode_graph
            && major >= 8
            && matches!(
                crate::gguf::Gguf::read_metadata(path)?.get("general.architecture"),
                Some(crate::gguf::Value::Text(arch)) if arch == "llama"
            );
        let source = if packed_llama {
            compiler_options.push("-DCUDA_DENSE_PACKED_REPLAY=1".into());
            format!(
                "{source}\n{}\n{}\n{}",
                include_str!("cuda/residual.cu"),
                include_str!("cuda/residual_precise.cu"),
                include_str!("cuda/residual_mma.cu"),
            )
        } else {
            source
        };
        let q8_dp4a = packed_llama && cfg!(feature = "cuda-llama-q8-dp4a-research");
        let warp_scores = packed_llama && cfg!(feature = "cuda-llama-warp-scores-research");
        let split_llama_attention = packed_llama
            && cfg!(feature = "cuda-llama-split-decode-attention-research");
        if warp_scores {
            compiler_options.push("-DCUDA_LLAMA_WARP_SCORES=1".into());
        }
        let q8_qkv = q8_dp4a && cfg!(feature = "cuda-llama-q8-fused-qkv-research");
        if q8_qkv {
            compiler_options.push("-DCUDA_LLAMA_Q8_FUSED_QKV=1".into());
        }
        let q8_norm_pack = q8_qkv && cfg!(feature = "cuda-llama-q8-norm-pack-research");
        if q8_norm_pack {
            compiler_options.push("-DCUDA_LLAMA_Q8_NORM_PACK=1".into());
        }
        let q8_swiglu = q8_dp4a && cfg!(feature = "cuda-llama-q8-fused-swiglu-research");
        let q8_swiglu_pairs = q8_swiglu && cfg!(feature = "cuda-llama-q8-swiglu-pairs-research");
        if q8_swiglu_pairs {
            compiler_options.push("-DCUDA_LLAMA_Q8_SWIGLU_PAIRS=1".into());
        }
        if q8_swiglu {
            compiler_options.push("-DCUDA_LLAMA_Q8_FUSED_SWIGLU=1".into());
        }
        let qwen3_mma3 = (cfg!(feature = "cuda-qwen3-dp4a-head-research")
            || cfg!(feature = "cuda-qwen3-split-decode-attention-research")
            || cfg!(feature = "cuda-qwen3-warp-reduction-research"))
            && options.matmul == CudaMatmul::ResidualMma3Research
            && matches!(
                crate::gguf::Gguf::read_metadata(path)?.get("general.architecture"),
                Some(crate::gguf::Value::Text(arch)) if arch == "qwen3"
            );
        if qwen3_mma3
            && cfg!(feature = "cuda-qwen3-warp-reduction-research")
            && !cfg!(feature = "cuda-packed-reduction-research")
        {
            compiler_options.push("-DCUDA_PACKED_REDUCTION_RESEARCH=1".into());
        }
        let qwen3_qkv = qwen3_mma3 && cfg!(feature = "cuda-qwen3-fused-qkv-research");
        if qwen3_qkv {
            compiler_options.push("-DCUDA_QWEN3_FUSED_QKV=1".into());
            if cfg!(feature = "cuda-qwen3-qkv-128-research") {
                compiler_options.push("-DCUDA_QWEN3_QKV_128=1".into());
            }
        }
        let qwen3_head_rope = qwen3_qkv && cfg!(feature = "cuda-qwen3-fused-head-rope-research");
        let qwen3_prefill_head_rope = cfg!(feature = "cuda-qwen3-prefill-fused-head-rope-research")
            && matches!(
                crate::gguf::Gguf::read_metadata(path)?.get("general.architecture"),
                Some(crate::gguf::Value::Text(arch)) if arch == "qwen3"
            );
        if qwen3_prefill_head_rope {
            compiler_options.push("-DCUDA_QWEN3_PREFILL_FUSED_HEAD_ROPE=1".into());
        }
        let qwen3_layer_matvec =
            qwen3_qkv && cfg!(feature = "cuda-qwen3-layer-matvec-128-research");
        if qwen3_layer_matvec {
            compiler_options.push("-DCUDA_QWEN3_LAYER_MATVEC_128=1".into());
        }
        if qwen3_head_rope {
            compiler_options.push("-DCUDA_QWEN3_FUSED_HEAD_ROPE=1".into());
        }
        let k_dp4a_head = qwen3_mma3 && cfg!(feature = "cuda-qwen3-dp4a-head-research");
        let split_decode_attention = qwen3_mma3
            && options.decode_graph
            && cfg!(feature = "cuda-qwen3-split-decode-attention-research");
        if split_decode_attention {
            compiler_options.push("-DCUDA_QWEN3_SPLIT_DECODE_ATTENTION=1".into());
        }
        if split_llama_attention {
            compiler_options.push("-DCUDA_LLAMA_SPLIT_DECODE_ATTENTION=1".into());
        }
        let source = if k_dp4a_head {
            compiler_options.push("-DCUDA_DENSE_PACKED_REPLAY=1".into());
            format!("{source}\n{}", include_str!("cuda/k_dp4a.cu"))
        } else {
            source
        };
        let value_split = cfg!(feature = "cuda-llama-decode-value-split-research")
            && options.matmul == CudaMatmul::F32
            && matches!(
                crate::gguf::Gguf::read_metadata(path)?.get("general.architecture"),
                Some(crate::gguf::Value::Text(arch)) if arch == "llama"
            );
        if value_split {
            compiler_options.push("-DCUDA_LLAMA_DECODE_VALUE_SPLIT=1".into());
            if cfg!(feature = "cuda-llama-value-unroll-research") {
                compiler_options.push("-DCUDA_LLAMA_VALUE_UNROLL=1".into());
            }
        }
        let query_pair_64 = cfg!(feature = "cuda-llama-prefill-query-pair-research")
            && options.matmul == CudaMatmul::F32
            && matches!(
                crate::gguf::Gguf::read_metadata(path)?.get("general.architecture"),
                Some(crate::gguf::Value::Text(arch)) if arch == "llama"
            );
        if query_pair_64 {
            compiler_options.push("-DCUDA_LLAMA_PREFILL_QUERY_PAIR=1".into());
        }
        if cfg!(feature = "cuda-llama-batched-attention-research") {
            compiler_options.push("-DCUDA_LLAMA_BATCHED_ATTENTION=1".into());
            if cfg!(feature = "cuda-llama-grouped-prefill-attention-research") {
                compiler_options.push("-DCUDA_LLAMA_GROUPED_PREFILL_ATTENTION=1".into());
            }
        }
        if cfg!(feature = "cuda-batched-attention-warp-softmax-research") {
            compiler_options.push("-DCUDA_BATCHED_ATTENTION_WARP_SOFTMAX=1".into());
        }
        if cfg!(feature = "cuda-qwen3-grouped-prefill-attention-research") {
            compiler_options.push("-DCUDA_QWEN3_GROUPED_PREFILL_ATTENTION=1".into());
            compiler_options.push("-DCUDA_LLAMA_GROUPED_PREFILL_ATTENTION=1".into());
        }
        let source = if q8_dp4a {
            format!("{source}\n{}", include_str!("cuda/q8_decode.cu"))
        } else {
            source
        };
        let ptx = cache::compile(
            options.kernel_cache_dir.as_deref(),
            &source,
            CompileOptions {
                fmad: Some(false),
                options: compiler_options,
                ..Default::default()
            },
        )?;
        let module = context.load_module(ptx).map_err(failure)?;
        let mut functions = HashMap::new();
        if qwen3_layer_matvec {
            let name = "qwen3_layer_matvec_128";
            functions.insert(name, module.load_function(name).map_err(failure)?);
        }
        if query_pair_64 {
            let name = "attention_fused_query_pair_64";
            functions.insert(name, module.load_function(name).map_err(failure)?);
        }
        if split_decode_attention {
            for name in ["attention_decode_partials", "attention_decode_merge"] {
                functions.insert(name, module.load_function(name).map_err(failure)?);
            }
        }
        if split_llama_attention {
            for name in ["llama_split_decode_partials", "llama_split_decode_merge"] {
                functions.insert(name, module.load_function(name).map_err(failure)?);
            }
        }
        if k_dp4a_head {
            let name = "k_dp4a_head";
            functions.insert(name, module.load_function(name).map_err(failure)?);
        }
        if value_split {
            let name = "attention_values_split";
            functions.insert(name, module.load_function(name).map_err(failure)?);
        }
        if q8_dp4a {
            let name = "llama_q8_dp4a";
            functions.insert(name, module.load_function(name).map_err(failure)?);
        }
        if warp_scores {
            let name = "llama_warp_scores_64";
            functions.insert(name, module.load_function(name).map_err(failure)?);
        }
        if packed_llama && cfg!(feature = "cuda-llama-factored-rope-cache-research") {
            let name = "llama_factored_rope_cache";
            functions.insert(name, module.load_function(name).map_err(failure)?);
        }
        if q8_qkv {
            let name = "llama_q8_dp4a_qkv";
            functions.insert(name, module.load_function(name).map_err(failure)?);
        }
        if q8_swiglu {
            let name = "llama_q8_dp4a_swiglu";
            functions.insert(name, module.load_function(name).map_err(failure)?);
        }
        if q8_swiglu_pairs {
            let name = "llama_q8_dp4a_swiglu_pairs";
            functions.insert(name, module.load_function(name).map_err(failure)?);
        }
        for name in [
            "embedding",
            "check_finite",
            "matvec",
            "normalize",
            "layer_norm_affine",
            "layer_norm_plain",
            "add_position_embedding",
            "gelu_row_bias",
            "head_norm",
            "rotary",
            "rotary_partial_neox",
            "rotary_factored",
            "add_row_bias",
            "cache_kv",
            "attention_scores",
            "alibi_attention_scores",
            "attention_values",
            "gemma_attention_scores",
            "gemma_attention_values",
            "gemma_embedding_scale",
            "gemma_geglu",
            "gemma_softcap",
            "router",
            "router_full_softmax",
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
            let entry = if name == "matvec" && options.matmul.f32_tiled() {
                "matvec_packed_f32"
            } else {
                name
            };
            functions.insert(name, module.load_function(entry).map_err(failure)?);
        }
        if qwen3_prefill_head_rope {
            let name = "prefill_qkv_head_norm_rope_cache";
            functions.insert(name, module.load_function(name).map_err(failure)?);
        }
        if qwen3_two_rows {
            let name = "prefill_matmul_ref2x16";
            functions.insert(name, module.load_function(name).map_err(failure)?);
        }
        if packed_llama {
            for name in ["residual_quantize32_precise3", "dense_packed_replay_mma3"] {
                functions.insert(name, module.load_function(name).map_err(failure)?);
            }
        }
        if q8_norm_pack {
            let name = "llama_q8_norm_pack";
            functions.insert(name, module.load_function(name).map_err(failure)?);
        }
        if options.matmul.f32_tiled() {
            for name in [
                "group_experts_reg32",
                "grouped_matmul_reg32x32",
                "grouped_matmul_reg32_counted",
                "grouped_matmul_coalesced_k32",
                "grouped_matmul_unroll8",
                "prefill_matmul_ref2x16",
            ] {
                functions.insert(name, module.load_function(name).map_err(failure)?);
            }
            if options.matmul == CudaMatmul::ResidualInt8Research {
                for name in ["residual_quantize32", "residual_matmul32"] {
                    functions.insert(name, module.load_function(name).map_err(failure)?);
                }
            }
            if options.matmul == CudaMatmul::ResidualMma3Research {
                if cfg!(feature = "cuda-attention-query-pair-research") {
                    let name = "attention_fused_query_pair";
                    functions.insert(name, module.load_function(name).map_err(failure)?);
                }
                if cfg!(feature = "cuda-mma3-wide-rows-research") {
                    let name = "residual_matmul_mma3_rows64";
                    functions.insert(name, module.load_function(name).map_err(failure)?);
                }
                if cfg!(feature = "cuda-mma3-splitk-research") {
                    for name in ["residual_matmul_mma3_splitk", "residual_splitk_reduce"] {
                        functions.insert(name, module.load_function(name).map_err(failure)?);
                    }
                }
                if cfg!(feature = "cuda-packed-down-research") {
                    let name = "matvec_packed_down_warp";
                    functions.insert(name, module.load_function(name).map_err(failure)?);
                }
                if cfg!(feature = "cuda-packed-gate-up-research") {
                    let name = "matvec_packed_gate_up";
                    functions.insert(name, module.load_function(name).map_err(failure)?);
                }
                if cfg!(feature = "cuda-decode-fusion-research") {
                    for name in [
                        "matvec_packed_qkv",
                        "router_warp128",
                        "qkv_head_norm_rope_cache",
                        "qkv_head_norm_rope_cache_decode",
                    ] {
                        functions.insert(name, module.load_function(name).map_err(failure)?);
                    }
                } else if qwen3_qkv {
                    let name = "matvec_packed_qkv";
                    functions.insert(name, module.load_function(name).map_err(failure)?);
                    if qwen3_head_rope {
                        for name in [
                            "qkv_head_norm_rope_cache",
                            "qkv_head_norm_rope_cache_decode",
                        ] {
                            functions.insert(name, module.load_function(name).map_err(failure)?);
                        }
                    }
                }
                for name in ["residual_quantize32_precise3", "residual_matmul_mma3"] {
                    functions.insert(name, module.load_function(name).map_err(failure)?);
                }
                if cfg!(feature = "cuda-parallel-grouping-research") {
                    for name in ["group_parallel_count32", "group_parallel_emit32"] {
                        functions.insert(name, module.load_function(name).map_err(failure)?);
                    }
                }
            }
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
        let device_greedy = cfg!(feature = "cuda-device-greedy-research")
            && !options.decode_graph
            && s.experts == 0
            && s.info.architecture == "llama";
        if device_greedy {
            for name in ["greedy_partials", "greedy_finish_id"] {
                functions.insert(name, module.load_function(name).map_err(failure)?);
            }
        }
        if matches!(
            s.info.architecture.as_str(),
            "gpt2" | "starcoder2" | "bloom" | "gptneox" | "olmo" | "olmo2" | "olmoe"
        ) && (options.decode_graph || options.matmul != CudaMatmul::F32)
        {
            return Err(invalid(format!(
                "{} CUDA requires f32 matmul without a decode graph",
                s.info.architecture
            )));
        }
        if s.info.architecture == "qwen2"
            && (options.decode_graph || options.matmul != CudaMatmul::F32)
        {
            return Err(invalid(
                "qwen2 CUDA requires f32 matmul without a decode graph",
            ));
        }
        if s.info.tensor_types.iter().any(|kind| kind == "Q5_K")
            && (s.experts != 0 || options.decode_graph || options.matmul != CudaMatmul::F32)
        {
            return Err(invalid(
                "Q5_K CUDA requires dense f32 matmul without a decode graph",
            ));
        }
        if s.info.tensor_types.iter().any(|kind| {
            matches!(
                kind.as_str(),
                "Q4_0" | "Q4_1" | "Q5_1" | "Q2_K" | "Q3_K" | "IQ4_NL"
            )
        }) && (s.experts != 0 || options.decode_graph || options.matmul != CudaMatmul::F32)
        {
            return Err(invalid(
                "Q4_0/Q4_1/Q5_1/Q2_K/Q3_K/IQ4_NL CUDA requires dense f32 matmul without a decode graph",
            ));
        }
        if s.info.tensor_types.iter().any(|kind| kind == "BF16")
            && (s.experts != 0 || options.decode_graph || options.matmul != CudaMatmul::F32)
        {
            return Err(invalid(
                "BF16 CUDA requires dense f32 matmul without a decode graph",
            ));
        }
        if s.info.architecture == "phi3"
            && (options.decode_graph || options.matmul != CudaMatmul::F32)
        {
            return Err(invalid(
                "phi3 CUDA requires f32 matmul without a decode graph",
            ));
        }
        if weights.rope_factors.is_some()
            && (options.decode_graph || options.matmul != CudaMatmul::F32)
        {
            return Err(invalid(
                "factored RoPE CUDA requires f32 matmul without a decode graph",
            ));
        }
        if weights.gemma.is_some() && (options.decode_graph || options.matmul != CudaMatmul::F32) {
            return Err(invalid(
                "Gemma CUDA requires f32 matmul without a decode graph",
            ));
        }
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
        #[cfg(any(
            feature = "cuda-qwen3-batched-attention-research",
            feature = "cuda-llama-batched-attention-research"
        ))]
        let batched_attention = major >= 8
            && batched_attention_supported(s)
            && matches!(
                (s.info.architecture.as_str(), options.matmul),
                ("qwen3", CudaMatmul::ResidualMma3Research) | ("llama", CudaMatmul::F32)
            );
        #[cfg(any(
            feature = "cuda-qwen3-batched-attention-research",
            feature = "cuda-llama-batched-attention-research"
        ))]
        if batched_attention {
            let name = "qwen3_batched_causal_softmax";
            functions.insert(name, module.load_function(name).map_err(failure)?);
            let llama_grouped = cfg!(feature = "cuda-llama-grouped-prefill-attention-research")
                && s.info.architecture == "llama";
            let qwen3_grouped = cfg!(feature = "cuda-qwen3-grouped-prefill-attention-research")
                && s.info.architecture == "qwen3";
            if llama_grouped || qwen3_grouped
            {
                for name in [
                    if llama_grouped { "llama_grouped_causal_softmax" } else { "qwen3_grouped_causal_softmax" },
                    "llama_grouped_q_pack",
                    "llama_grouped_out_unpack",
                ] {
                    functions.insert(name, module.load_function(name).map_err(failure)?);
                }
            }
        }
        #[cfg(feature = "cuda-fp16-corrected-prefill-research")]
        let is_llama = s.info.architecture == "llama";
        #[cfg(feature = "cuda-fp16-corrected-prefill-research")]
        let (fp16_bytes, fp16_inputs) = if major >= 8
            && s.experts == 0
            && ((is_llama && options.matmul == CudaMatmul::F32)
                || (s.info.architecture == "qwen3"
                    && options.matmul == CudaMatmul::ResidualMma3Research))
            && prefill_capacity(s) >= 32
        {
            let sizes = workspace_sizes(s)?;
            (
                fp16_gemm::dense_bytes(&weights, is_llama)?,
                sizes[0].max(sizes[2]).max(sizes[7]),
            )
        } else {
            (0, 0)
        };
        #[cfg(feature = "cuda-fp16-corrected-prefill-research")]
        let weight_bytes = weight_bytes
            .checked_add(fp16_bytes)
            .ok_or_else(|| invalid("FP16 weight byte overflow"))?;
        let kv_elements = (s.kv_heads as u64)
            .checked_mul(s.dim as u64)
            .and_then(|n| n.checked_mul(u64::from(config.context_tokens)))
            .ok_or_else(|| invalid("KV overflow"))?;
        let kv_bytes = kv_elements
            .checked_mul(weights.layers.len() as u64)
            .and_then(|n| n.checked_mul(8))
            .ok_or_else(|| invalid("KV byte overflow"))?;
        let routing = routing_elements(s)? as u64;
        let groups = group_workspace_elements(s, options.matmul)? as u64;
        let workspace_bytes = workspace_sizes(s)?
            .into_iter()
            .try_fold(0u64, |sum, n| sum.checked_add(n as u64))
            .and_then(|n| n.checked_add(routing * 2 + groups + 1))
            .and_then(|n| n.checked_add(parallel_group_count_elements(s, options.matmul) as u64))
            .and_then(|n| {
                n.checked_add(if prefill_capacity(s) > 1 {
                    prefill_capacity(s) as u64
                } else {
                    0
                })
            })
            .and_then(|n| n.checked_mul(4))
            .ok_or_else(|| invalid("workspace byte overflow"))?;
        let precise = if packed_llama && s.experts == 0 {
            CudaMatmul::ResidualMma3Research
        } else {
            options.matmul
        };
        let workspace_bytes = workspace_bytes
            .checked_add(residual_workspace_bytes(s, precise)?)
            .ok_or_else(|| invalid("workspace byte overflow"))?;
        #[cfg(feature = "cuda-fp16-corrected-prefill-research")]
        let workspace_bytes = workspace_bytes
            .checked_add(if fp16_bytes > 0 {
                4 * fp16_inputs as u64
            } else {
                0
            })
            .ok_or_else(|| invalid("FP16 workspace byte overflow"))?;
        let logits_bytes = u64::from(s.info.vocab_tokens) * 4;
        #[cfg(feature = "cuda-llama-ffn-mmq-research")]
        let ffn_mmq = (major == 12
            && minor == 0
            && is_llama
            && options.matmul == CudaMatmul::F32
            && s.experts == 0
            && s.width == 2048
            && s.hidden == 8192
            && prefill_capacity(s) >= 512)
            .then(|| ffn_mmq::Bridge::open(&context))
            .transpose()?;
        #[cfg(feature = "cuda-llama-ffn-mmq-research")]
        let workspace_bytes = workspace_bytes
            .checked_add(ffn_mmq.as_ref().map_or(0, ffn_mmq::Bridge::bytes))
            .ok_or_else(|| invalid("FFN MMQ workspace byte overflow"))?;
        let token_bytes = if options.decode_graph {
            decode::DecodeState::bytes(s.info.vocab_tokens as usize)
        } else if device_greedy {
            decode::DeviceGreedy::bytes(s.info.vocab_tokens as usize)
        } else {
            0
        };
        let reservation = weight_bytes
            .checked_add(kv_bytes)
            .and_then(|n| n.checked_add(workspace_bytes))
            .and_then(|n| n.checked_add(logits_bytes))
            .and_then(|n| n.checked_add(token_bytes))
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
            token_state_allocated_bytes: token_bytes,
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
        let work = Workspace::new(s, &stream, options.matmul, precise)?;
        let device_greedy = device_greedy
            .then(|| decode::DeviceGreedy::new(&stream, s.info.vocab_tokens as usize))
            .transpose()?;
        let mut keys = Vec::new();
        let mut values = Vec::new();
        for _ in &weights.layers {
            keys.push(stream.alloc_zeros(kv_elements as usize).map_err(failure)?);
            values.push(stream.alloc_zeros(kv_elements as usize).map_err(failure)?);
        }
        let logits = vec![0.0; s.info.vocab_tokens as usize];
        #[cfg(any(
            feature = "cuda-qwen3-batched-attention-research",
            feature = "cuda-llama-batched-attention-research"
        ))]
        let attention_blas = batched_attention
            .then(|| cudarc::cublas::CudaBlas::new(stream.clone()).map_err(failure))
            .transpose()?;
        let mut weights = weights.map(|t| DeviceTensor::upload(t, &stream))?;
        #[cfg(feature = "cuda-fp16-corrected-prefill-research")]
        let fp16 = (fp16_bytes > 0)
            .then(|| fp16_gemm::Prefill::dense(&stream, &mut weights, fp16_inputs, is_llama))
            .transpose()?;
        #[cfg(feature = "cuda-llama-ffn-mmq-research")]
        let ffn_mmq = ffn_mmq.map(|bridge| bridge.allocate(&stream)).transpose()?;
        weights.spec.info.backend = format!(
            "rust-cuda-packed:device{}:{}",
            options.device,
            context.name().map_err(failure)?
        );
        stream.synchronize().map_err(failure)?;
        let mut backend = Self {
            decode,
            device_greedy,
            #[cfg(any(
                feature = "cuda-qwen3-batched-attention-research",
                feature = "cuda-llama-batched-attention-research"
            ))]
            attention_blas,
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
            #[cfg(feature = "cuda-fp16-corrected-prefill-research")]
            fp16,
            #[cfg(feature = "cuda-llama-ffn-mmq-research")]
            ffn_mmq,
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
        let embedding_norm = self.weights.embedding_norm.as_ref();
        if let Some([weight, bias]) = embedding_norm {
            transformer_norm!(
                self,
                &self.work.x,
                Some(weight),
                Some(bias),
                1,
                w,
                s.epsilon
            );
            std::mem::swap(&mut self.work.x, &mut self.work.normalized);
        }
        let position_embedding = self.weights.position_embedding.as_ref();
        if let Some(t) = position_embedding {
            launch!(
                self,
                "add_position_embedding",
                vector(w),
                &mut self.work.x,
                &t.bytes,
                &t.kind,
                &w,
                &pos
            );
        }
        let gemma = self.weights.gemma.as_ref();
        if gemma.is_some() {
            let scale = (w as f32).sqrt();
            launch!(
                self,
                "gemma_embedding_scale",
                vector(w),
                &mut self.work.x,
                &w,
                &scale
            );
        }
        for (index, l) in self.weights.layers.iter().enumerate() {
            let gemma_attention = gemma.map(|g| g.attention(s, index));
            let rope_base = gemma_attention.map_or(s.rope_base, |(_, base, _)| base);
            let affine = l.affine.as_ref();
            let olmo2 = s.info.architecture == "olmo2";
            let olmoe = s.info.architecture == "olmoe";
            let packed_qkv = self.functions.contains_key("llama_q8_dp4a_qkv")
                && l.q.kind == 8
                && l.k.kind == 8
                && l.v.kind == 8
                && l.q.width == w
                && l.k.width == w
                && l.v.width == w
                && w % 32 == 0
                && l.k.rows == l.v.rows
                && self.work.residual_mma3.is_some();
            let norm_pack = l.norm.as_ref().filter(|_| {
                packed_qkv
                    && w == 2048
                    && affine.is_none()
                    && self.functions.contains_key("llama_q8_norm_pack")
            });
            if olmo2 {
                self.stream
                    .memcpy_dtod(
                        &self.work.x.slice(..s.width),
                        &mut self.work.normalized.slice_mut(..s.width),
                    )
                    .map_err(failure)?;
            } else if norm_pack.is_none() {
                transformer_norm!(
                    self,
                    &self.work.x,
                    l.norm.as_ref(),
                    affine.map(|b| &b.norm),
                    1,
                    w,
                    s.epsilon
                );
            }
            if packed_qkv && let Some(scratch) = self.work.residual_mma3.as_mut() {
                if 3 * w as usize > scratch.values.len()
                    || 3 * (w as usize / 32) > scratch.scales.len().min(scratch.sums.len())
                {
                    return Err(invalid("Q8 DP4A QKV input exceeds workspace"));
                }
                if let Some(norm) = norm_pack {
                    launch!(
                        self,
                        "llama_q8_norm_pack",
                        (8, 1, 1),
                        &self.work.x,
                        &norm.bytes,
                        &mut scratch.values,
                        &mut scratch.scales,
                        &mut scratch.sums,
                        &mut self.work.status,
                        &s.epsilon
                    );
                } else {
                    launch!(
                        self,
                        "residual_quantize32_precise3",
                        ((w as u32).div_ceil(256), 1, 1),
                        &self.work.normalized,
                        &mut scratch.values,
                        &mut scratch.scales,
                        &mut scratch.sums,
                        &w,
                        &mut self.work.status
                    );
                }
                launch!(self, "llama_q8_dp4a_qkv", ((l.q.rows.max(l.k.rows) as u32).div_ceil(4), 3, 1); 128,
                    &l.q.bytes, &l.k.bytes, &l.v.bytes, &scratch.values, &scratch.scales,
                    &mut self.work.q, &mut self.work.k, &mut self.work.v, &w, &l.q.rows, &l.k.rows);
            } else if (cfg!(feature = "cuda-decode-fusion-research")
                || (cfg!(feature = "cuda-qwen3-fused-qkv-research")
                    && self.functions.contains_key("matvec_packed_qkv")))
                && self.matmul == CudaMatmul::ResidualMma3Research
                && l.q.width == l.k.width
                && l.k.width == l.v.width
                && l.q.width % 256 == 0
                && matches!(l.q.kind, 12 | 14)
                && matches!(l.k.kind, 12 | 14)
                && matches!(l.v.kind, 12 | 14)
            {
                let total_rows = (l.q.rows + l.k.rows + l.v.rows) as u32;
                let threads = if cfg!(feature = "cuda-qwen3-qkv-128-research")
                    && s.info.architecture == "qwen3"
                {
                    128
                } else {
                    256
                };
                launch!(
                    self,
                    "matvec_packed_qkv",
                    (total_rows, 1, 1); threads,
                    &l.q.bytes,
                    &l.q.kind,
                    &l.q.rows,
                    &mut self.work.q,
                    &l.k.bytes,
                    &l.k.kind,
                    &l.k.rows,
                    &mut self.work.k,
                    &l.v.bytes,
                    &l.v.kind,
                    &l.v.rows,
                    &mut self.work.v,
                    &self.work.normalized,
                    &l.q.width
                );
            } else {
                let mut prepared = None;
                for (tensor, output) in [
                    (&l.q, &mut self.work.q),
                    (&l.k, &mut self.work.k),
                    (&l.v, &mut self.work.v),
                ] {
                    if q8_dp4a!(self, tensor, normalized, &mut *output, prepared) {
                        prepared = Some(tensor.width);
                        continue;
                    }
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
            }
            if let Some(biases) = &l.qkv_bias {
                for (projection, bias) in [&mut self.work.q, &mut self.work.k, &mut self.work.v]
                    .into_iter()
                    .zip(biases)
                {
                    launch!(
                        self,
                        "add_row_bias",
                        vector(bias.width),
                        projection,
                        &bias.bytes,
                        &bias.width
                    );
                }
            }
            if (cfg!(feature = "cuda-decode-fusion-research")
                || (cfg!(feature = "cuda-qwen3-fused-head-rope-research")
                    && self.functions.contains_key("qkv_head_norm_rope_cache")))
                && self.matmul == CudaMatmul::ResidualMma3Research
                && dim == 128
                && qw == heads * 128
                && kw == kv_heads * 128
                && l.q_norm.is_some()
                && l.k_norm.is_some()
            {
                let qn = l.q_norm.as_ref().unwrap();
                let kn = l.k_norm.as_ref().unwrap();
                launch_position!(
                    self,
                    device_state,
                    "qkv_head_norm_rope_cache",
                    "qkv_head_norm_rope_cache_decode",
                    ((heads + kv_heads) as u32, 1, 1),
                    [
                        &mut self.work.q,
                        &qn.bytes,
                        &heads,
                        &mut self.work.k,
                        &kn.bytes,
                        &self.work.v,
                        &mut self.keys[index],
                        &mut self.values[index],
                        &kv_heads,
                        &s.epsilon
                    ],
                    &pos,
                    [&s.rope_base, &neox]
                );
            } else if !device_state
                && (dim, heads, kv_heads) == (64, 32, 8)
                && s.rotary_dim == s.dim
                && position_embedding.is_none()
                && embedding_norm.is_none()
                && l.q_norm.is_none()
                && l.k_norm.is_none()
                && self.functions.contains_key("llama_factored_rope_cache")
                && let Some(factors) = &self.weights.rope_factors
            {
                launch!(self, "llama_factored_rope_cache", vector((qw + kw) / 2),
                    &mut self.work.q, &self.work.k, &self.work.v,
                    &mut self.keys[index], &mut self.values[index], &qw, &kw,
                    &pos, &rope_base, &neox, &factors.bytes, &s.rope_attn_factor);
            } else {
                for (norm, vector, size) in [
                    (&l.q_norm, &mut self.work.q, qw),
                    (&l.k_norm, &mut self.work.k, kw),
                ] {
                    if let Some(norm) = norm {
                        let norm_width = if olmo2 || olmoe { size } else { dim };
                        launch!(
                            self,
                            "head_norm",
                            ((size / norm_width) as u32, 1, 1),
                            &mut *vector,
                            &norm.bytes,
                            &norm_width,
                            &s.epsilon
                        );
                    }
                    if position_embedding.is_some() || embedding_norm.is_some() {
                        continue;
                    }
                    let grid = (((size / 2) as u32).div_ceil(256), 1, 1);
                    if s.rotary_dim != s.dim {
                        let rotary_dim = s.rotary_dim as i32;
                        launch!(self, "rotary_partial_neox", grid;
                            256, vector, &size, &dim, &pos, &rope_base, &rotary_dim);
                    } else if let Some(factors) = &self.weights.rope_factors {
                        launch!(
                            self,
                            "rotary_factored",
                            grid,
                            vector,
                            &size,
                            &dim,
                            &pos,
                            &rope_base,
                            &neox,
                            &factors.bytes,
                            &s.rope_attn_factor
                        );
                    } else {
                        launch_position!(
                            self,
                            device_state,
                            "rotary",
                            "rotary_decode",
                            grid,
                            [vector, &size, &dim],
                            &pos,
                            [&rope_base, &neox]
                        );
                    }
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
            }
            if embedding_norm.is_some() {
                alibi_attention!(self, index, heads, kv_heads, dim, context, count, 1);
            } else if let Some((window, _, scale)) = gemma_attention {
                gemma_attention!(
                    self,
                    index,
                    heads,
                    kv_heads,
                    dim,
                    context,
                    count,
                    1,
                    window,
                    scale,
                    gemma.unwrap().attention_softcap
                );
            } else if device_state {
                let control = &self
                    .decode
                    .as_ref()
                    .ok_or_else(|| failure("decode control missing"))?
                    .control;
                const TILE: u32 = QWEN3_DECODE_TILE;
                let tiles = (context as u32).div_ceil(TILE);
                if dim == 128
                    && (tiles as usize) * 130 * heads as usize <= self.work.scores.len()
                    && self.functions.contains_key("attention_decode_partials")
                {
                    launch!(
                        self,
                        "attention_decode_partials",
                        (heads as u32, tiles, 1),
                        &self.work.q,
                        &self.keys[index],
                        &self.values[index],
                        &mut self.work.attended,
                        &mut self.work.scores,
                        &heads,
                        &kv_heads,
                        &TILE,
                        control
                    );
                    launch!(
                        self,
                        "attention_decode_merge",
                        (heads as u32, 1, 1); 128,
                        &self.work.scores,
                        &mut self.work.attended,
                        &tiles,
                        &TILE,
                        control
                    );
                } else {
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
                }
            } else if (dim, heads, kv_heads) == (64, 32, 8)
                && count >= 128
                && self.functions.contains_key("llama_split_decode_partials")
            {
                const TILE: u32 = 128;
                let tiles = (count as u32).div_ceil(TILE);
                // At least 128 live tokens leave room for 66 floats per tile.
                debug_assert!(heads as usize * tiles as usize * 66 <= self.work.scores.len());
                launch!(self, "llama_split_decode_partials", (heads as u32, tiles, 1),
                    &self.work.q, &self.keys[index], &self.values[index],
                    &mut self.work.attended, &mut self.work.scores, &heads,
                    &kv_heads, &TILE, &count);
                if tiles > 1 {
                    launch!(self, "llama_split_decode_merge", (heads as u32, 1, 1); 64,
                        &self.work.scores, &mut self.work.attended, &tiles, &TILE, &count);
                }
            } else {
                if dim == 64 && self.functions.contains_key("llama_warp_scores_64") {
                    launch!(self, "llama_warp_scores_64", (heads as u32, (count as u32).div_ceil(4), 1); 128,
                        &self.work.q, &self.keys[index], &mut self.work.scores, &heads,
                        &kv_heads, &context, &count);
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
                }
                launch!(
                    self,
                    if count >= 128
                        && matches!(dim, 64 | 128)
                        && self.functions.contains_key("attention_values_split")
                    {
                        "attention_values_split"
                    } else {
                        "attention_values"
                    },
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
            if !q8_dp4a!(self, t, attended, &mut self.work.projected)
                && !qwen3_matvec_128!(self, t, attended, &mut self.work.projected)
            {
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
            }
            if let Some(b) = affine {
                launch!(
                    self,
                    "add_row_bias",
                    vector(w),
                    &mut self.work.projected,
                    &b.attention.bytes,
                    &w
                );
            }
            if let Some([post_attn, _]) = &l.post_norms {
                gemma_post_norm!(self, projected, post_attn, 1, w, s.epsilon);
            } else if !s.parallel_residual {
                launch!(
                    self,
                    "residual",
                    vector(w),
                    &mut self.work.x,
                    &self.work.projected,
                    &w
                );
            }
            if olmo2 {
                self.stream
                    .memcpy_dtod(
                        &self.work.x.slice(..s.width),
                        &mut self.work.normalized.slice_mut(..s.width),
                    )
                    .map_err(failure)?;
            } else {
                transformer_norm!(
                    self,
                    &self.work.x,
                    l.ffn_norm.as_ref(),
                    affine.map(|b| &b.ffn_norm),
                    1,
                    w,
                    s.epsilon
                );
            }
            if let Some(b) = affine {
                affine_ffn!(self, l, b, 1i32, w, s.parallel_residual);
                continue;
            }
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
                if cfg!(feature = "cuda-decode-fusion-research")
                    && self.matmul == CudaMatmul::ResidualMma3Research
                    && experts == 128
                    && used <= 32
                {
                    launch!(
                        self,
                        "router_warp128",
                        (1, 1, 1); 32,
                        &self.work.router,
                        &used,
                        &mut self.work.ids,
                        &mut self.work.probabilities,
                        &mut self.work.status
                    );
                } else {
                    launch!(
                        self,
                        if olmoe {
                            "router_full_softmax"
                        } else {
                            "router"
                        },
                        (1, 1, 1),
                        &self.work.router,
                        &experts,
                        &used,
                        &mut self.work.ids,
                        &mut self.work.probabilities,
                        &mut self.work.status
                    );
                }
            }
            let moe = i32::from(l.router.is_some());
            let gate = l.gate.as_ref().expect("gated CUDA layer");
            if self.functions.contains_key("llama_q8_dp4a_swiglu")
                && moe == 0
                && gemma.is_none()
                && s.used == 1
                && gate.kind == 8
                && l.up.kind == 8
                && gate.width == w
                && l.up.width == w
                && w % 32 == 0
                && gate.rows == l.up.rows
                && gate.rows as usize == s.hidden
                && let Some(scratch) = self.work.residual_mma3.as_mut()
            {
                if 3 * w as usize > scratch.values.len()
                    || 3 * (w as usize / 32) > scratch.scales.len().min(scratch.sums.len())
                {
                    return Err(invalid("Q8 DP4A SwiGLU input exceeds workspace"));
                }
                launch!(
                    self,
                    "residual_quantize32_precise3",
                    ((w as u32).div_ceil(256), 1, 1),
                    &self.work.normalized,
                    &mut scratch.values,
                    &mut scratch.scales,
                    &mut scratch.sums,
                    &w,
                    &mut self.work.status
                );
                let (kernel, rows_per_block) = if gate.rows % 2 == 0
                    && self.functions.contains_key("llama_q8_dp4a_swiglu_pairs")
                {
                    ("llama_q8_dp4a_swiglu_pairs", 8)
                } else {
                    ("llama_q8_dp4a_swiglu", 4)
                };
                launch!(self, kernel, ((gate.rows as u32).div_ceil(rows_per_block), 1, 1); 128,
                    &gate.bytes, &l.up.bytes, &scratch.values, &scratch.scales,
                    &mut self.work.gate, &w, &gate.rows);
            } else if cfg!(feature = "cuda-packed-gate-up-research")
                && self.matmul == CudaMatmul::ResidualMma3Research
                && moe != 0
                && gate.width == 2048
                && l.up.width == 2048
                && gate.rows == l.up.rows
                && matches!(gate.kind, 12 | 14)
                && matches!(l.up.kind, 12 | 14)
            {
                launch!(
                    self,
                    "matvec_packed_gate_up",
                    (gate.rows as u32, used as u32, 1),
                    &gate.bytes,
                    &gate.kind,
                    &l.up.bytes,
                    &l.up.kind,
                    &self.work.normalized,
                    &mut self.work.gate,
                    &gate.rows,
                    &self.work.ids
                );
            } else {
                let mut prepared = None;
                for (t, out) in [(gate, &mut self.work.gate), (&l.up, &mut self.work.up)] {
                    if moe == 0 && q8_dp4a!(self, t, normalized, &mut *out, prepared) {
                        prepared = Some(t.width);
                        continue;
                    }
                    if moe == 0 && qwen3_matvec_128!(self, t, normalized, &mut *out) {
                        continue;
                    }
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
                    if gemma.is_some() {
                        "gemma_geglu"
                    } else {
                        "swiglu"
                    },
                    vector(activated),
                    &mut self.work.gate,
                    &self.work.up,
                    &activated
                );
            }
            let t = &l.down;
            if cfg!(feature = "cuda-packed-down-research")
                && self.matmul == CudaMatmul::ResidualMma3Research
                && moe != 0
                && t.width == 768
                && matches!(t.kind, 12 | 14)
            {
                launch!(
                    self,
                    "matvec_packed_down_warp",
                    ((t.rows as u32).div_ceil(8), used as u32, 1),
                    &t.bytes,
                    &t.kind,
                    &self.work.gate,
                    &mut self.work.down,
                    &t.rows,
                    &self.work.ids
                );
            } else if moe != 0
                || (!q8_dp4a!(self, t, gate, &mut self.work.down)
                    && !qwen3_matvec_128!(self, t, gate, &mut self.work.down))
            {
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
            }
            if let Some([_, post_ffn]) = &l.post_norms {
                gemma_post_norm!(self, down, post_ffn, 1, w, s.epsilon);
            } else {
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
        transformer_norm!(
            self,
            &input,
            self.weights.output_norm.as_ref(),
            self.weights.output_norm_bias.as_ref(),
            1,
            w,
            s.epsilon
        );
        let t = self
            .weights
            .output
            .as_ref()
            .unwrap_or(&self.weights.embedding);
        if cfg!(feature = "cuda-qwen3-dp4a-head-research")
            && matches!(t.kind, 12 | 14)
            && t.width % 256 == 0
            && self.functions.contains_key("k_dp4a_head")
            && let Some(scratch) = self.work.residual_mma3.as_mut()
        {
            launch!(
                self,
                "residual_quantize32_precise3",
                ((w as u32).div_ceil(256), 1, 1),
                &self.work.normalized,
                &mut scratch.values,
                &mut scratch.scales,
                &mut scratch.sums,
                &w,
                &mut self.work.status
            );
            launch!(self, "k_dp4a_head", ((t.rows as u32).div_ceil(4), 1, 1); 128,
                &t.bytes, &t.kind, &scratch.values, &scratch.scales,
                &mut self.work.logits, &w, &t.rows);
        } else if !q8_dp4a!(self, t, normalized, &mut self.work.logits) {
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
        }
        if let Some(gemma) = &self.weights.gemma
            && gemma.final_softcap > 0.0
        {
            launch!(
                self,
                "gemma_softcap",
                ((t.rows as u32).div_ceil(256), 1, 1),
                &mut self.work.logits,
                &t.rows,
                &gemma.final_softcap,
                &mut self.work.status
            );
        }
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
                "nonfinite router scores, logits or residual INT8 arithmetic; request poisoned",
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
        if self.decode.as_ref().is_some_and(|d| d.sampled.is_some())
            || self.device_greedy.as_ref().is_some_and(|d| d.sampled.is_some())
        {
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
        if let Some(device) = &mut self.device_greedy {
            device.sampled = None;
        }
        Ok(())
    }
    fn evaluate(&mut self, tokens: &[i32], position: u32) -> Result<()> {
        self.evaluate_inner(tokens, position, true)
    }
    fn evaluate_for_sampling(&mut self, tokens: &[i32], position: u32) -> Result<()> {
        self.evaluate_inner(tokens, position, self.decode.is_none() && self.device_greedy.is_none())
    }
    fn sample_tokens(&mut self, maximum: usize, mode: crate::GenerationMode) -> Result<Vec<i32>> {
        self.sample_decode_tokens(maximum, mode)
    }
    fn sample_greedy(&mut self) -> Result<i32> {
        self.state.logits(&[])?;
        if let Some(token) = self.decode.as_ref().and_then(|d| d.sampled)
            .or_else(|| self.device_greedy.as_ref().and_then(|d| d.sampled))
        {
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
        if let Some(device) = &mut self.device_greedy {
            device.sampled = None;
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
        } else if let Some(device) = &mut self.device_greedy {
            device.enqueue(&self.stream, &self.functions, &self.work.logits, &mut self.work.status)?;
            device.read(&self.stream, self.weights.spec.info.vocab_tokens)?;
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
    fn register_group_workspace_covers_large_and_fallback_chunks() {
        let mut spec = Spec {
            info: ModelInfo {
                backend: "test".into(),
                revision: "test".into(),
                architecture: "qwen3moe".into(),
                tensor_types: vec![],
                weight_bytes: 0,
                parameters: 0,
                context_tokens: 1024,
                batch_tokens: 512,
                vocab_tokens: 3,
                capabilities: crate::backend::Capabilities {
                    token_step: true,
                    chunked_prefill: true,
                },
            },
            width: 2048,
            heads: 32,
            kv_heads: 4,
            dim: 128,
            rotary_dim: 128,
            parallel_residual: false,
            hidden: 768,
            experts: 128,
            used: 8,
            epsilon: 1e-6,
            rope_base: 1_000_000.0,
            rope_attn_factor: 1.0,
            neox: true,
            eog: vec![],
            tokenizer: Err(invalid("tokenizer unused by workspace sizing")),
        };
        for capacity in [1, 2, 31, 32, 33, 46, 127, 128, 255, 256, 257, 511, 512] {
            spec.info.batch_tokens = capacity;
            let count =
                residual_workspace_elements(&spec, CudaMatmul::ResidualInt8Research).unwrap();
            assert_eq!(
                count,
                if capacity < 256 {
                    0
                } else {
                    capacity as usize * 768 * 8
                }
            );
            assert_eq!(
                residual_workspace_bytes(&spec, CudaMatmul::ResidualInt8Research).unwrap(),
                count as u64 * 2 + count as u64 / 32 * 16
            );
            let precise_count = if capacity < 32 {
                0
            } else {
                capacity as usize * 768 * 8
            };
            assert_eq!(
                residual_workspace_elements(&spec, CudaMatmul::ResidualMma3Research).unwrap(),
                precise_count
            );
            assert_eq!(
                residual_workspace_bytes(&spec, CudaMatmul::ResidualMma3Research).unwrap(),
                precise_count as u64 * 3 + precise_count as u64 / 32 * 24
            );
            assert_eq!(
                residual_workspace_elements(&spec, CudaMatmul::F32Tiled).unwrap(),
                0
            );
            assert_eq!(
                group_workspace_elements(&spec, CudaMatmul::ResidualInt8Research).unwrap(),
                group_workspace_elements(&spec, CudaMatmul::F32Tiled).unwrap()
            );
            let reserved = group_workspace_elements(&spec, CudaMatmul::F32Tiled).unwrap();
            let precise_reserved =
                group_workspace_elements(&spec, CudaMatmul::ResidualMma3Research).unwrap();
            assert_eq!(
                precise_reserved,
                if capacity >= 32 {
                    reserved.max(group_elements_with(&spec, 32).unwrap())
                } else {
                    reserved
                }
            );
            let legacy = group_elements_with(&spec, 8).unwrap();
            if capacity < 256 {
                assert_eq!(reserved, legacy);
            }
            if capacity == 512 {
                assert_eq!((reserved - legacy) * size_of::<i32>(), 10_656);
            }
            for chunk in 1..=capacity {
                spec.info.batch_tokens = chunk;
                let tile = if register_group_shape(&spec, CudaMatmul::F32Tiled, chunk as usize) {
                    32
                } else {
                    8
                };
                assert!(group_elements_with(&spec, tile).unwrap() <= reserved);
                assert!(group_elements_with(&spec, 8).unwrap() <= reserved);
                assert!(group_elements_with(&spec, 8).unwrap() <= precise_reserved);
                if chunk >= 32 {
                    assert!(group_elements_with(&spec, 32).unwrap() <= precise_reserved);
                }
            }
        }
        spec.experts = 64;
        assert_eq!(
            group_workspace_elements(&spec, CudaMatmul::F32Tiled).unwrap(),
            group_elements_with(&spec, 8).unwrap()
        );
        assert_eq!(
            group_workspace_elements(&spec, CudaMatmul::F32).unwrap(),
            group_elements(&spec).unwrap()
        );
        assert!(
            group_workspace_elements(&spec, CudaMatmul::ResidualInt8Research).unwrap()
                >= group_elements_with(&spec, 32).unwrap()
        );
        assert!(
            group_workspace_elements(&spec, CudaMatmul::ResidualMma3Research).unwrap()
                >= group_elements_with(&spec, 32).unwrap()
        );
        spec.used = 1;
        assert_eq!(
            residual_workspace_elements(&spec, CudaMatmul::ResidualInt8Research).unwrap(),
            4096 * 512
        );
        spec.hidden = 8192;
        assert_eq!(
            residual_workspace_elements(&spec, CudaMatmul::ResidualInt8Research).unwrap(),
            8192 * 512
        );
        assert_eq!(
            residual_workspace_elements(&spec, CudaMatmul::ResidualMma3Research).unwrap(),
            8192 * 512
        );
        spec.hidden = 1_400_000;
        assert!(residual_workspace_elements(&spec, CudaMatmul::ResidualInt8Research).is_ok());
        assert!(residual_workspace_elements(&spec, CudaMatmul::ResidualMma3Research).is_err());
        spec.hidden = usize::MAX;
        assert!(residual_workspace_elements(&spec, CudaMatmul::ResidualInt8Research).is_err());
        assert!(residual_workspace_elements(&spec, CudaMatmul::ResidualMma3Research).is_err());
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
    fn qwen35_cuda_delta_tracks_decayed_state_across_tokens() {
        const DIM: usize = 128;
        const HEADS: usize = 16;
        const INNER: usize = HEADS * DIM;
        let context = CudaContext::new(0).unwrap();
        let module = context
            .load_module(
                compile_ptx_with_opts(
                    include_str!("cuda/qwen35_delta.cu"),
                    CompileOptions {
                        fmad: Some(false),
                        arch: Some("compute_75"),
                        ..Default::default()
                    },
                )
                .unwrap(),
            )
            .unwrap();
        let stream = context.default_stream();
        let kernel = module.load_function("qwen35_delta_step").unwrap();
        let mut state = stream.alloc_zeros::<f32>(INNER * DIM).unwrap();
        let mut output = stream.alloc_zeros::<f32>(INNER).unwrap();
        let initial_value: Vec<_> = (0..INNER).map(|i| (i % 31) as f32 * 0.125).collect();
        let mut projection = vec![0.0; 3 * INNER];
        projection[2 * INNER..].copy_from_slice(&initial_value);
        for head in 0..HEADS {
            projection[head * DIM + head] = 1.0;
            projection[INNER + head * DIM + head] = 1.0;
        }
        let projection = stream.clone_htod(&projection).unwrap();
        let decay = stream.clone_htod(&[0.5f32; HEADS]).unwrap();
        let beta = stream.clone_htod(&[0.25f32; HEADS]).unwrap();
        for factor in [0.25, 0.34375, 0.37890625] {
            let mut args = stream.launch_builder(&kernel);
            args.arg(&projection)
                .arg(&decay)
                .arg(&beta)
                .arg(&mut state)
                .arg(&mut output);
            // A block handles one head; each of its 128 threads owns one state row.
            unsafe {
                args.launch(LaunchConfig {
                    grid_dim: (HEADS as u32, 1, 1),
                    block_dim: (DIM as u32, 1, 1),
                    shared_mem_bytes: 0,
                })
            }
            .unwrap();
            let values = stream.clone_dtoh(&output).unwrap();
            for (actual, input) in values.iter().zip(&initial_value) {
                assert!((actual - input * factor / (DIM as f32).sqrt()).abs() < 1e-7);
            }
        }
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
#[cfg(feature = "cuda-llama-ffn-mmq-research")]
mod ffn_mmq;
#[cfg(any(
    all(test, feature = "cuda-fp16-prefill-research"),
    feature = "cuda-fp16-corrected-prefill-research"
))]
mod fp16_gemm;
