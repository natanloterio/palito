use super::*;
use cudarc::cublas::{CudaBlas, result, sys};
use cudarc::driver::{DevicePtr, DevicePtrMut};

#[cfg(feature = "cuda-fp16-corrected-prefill-research")]
pub(super) fn dense_bytes(weights: &Weights<Tensor>, is_llama: bool) -> Result<u64> {
    cache_bytes(weights
        .layers
        .iter()
        .flat_map(|l| {
            [&l.q, &l.k, &l.v, &l.output, &l.up, &l.down]
                .into_iter()
                .chain(l.gate.as_ref())
        })
        .filter(|t| matches!(t.storage, Storage::Packed { kind, .. } if dense_kind(kind as i32, is_llama))))
}

#[cfg(feature = "cuda-fp16-corrected-prefill-research")]
pub(super) fn cache_bytes<'a>(mut tensors: impl Iterator<Item = &'a Tensor>) -> Result<u64> {
    tensors.try_fold(0u64, |sum, t| {
        let count = t
            .dims
            .first()
            .zip(t.dims.get(1))
            .and_then(|(w, r)| w.checked_mul(*r))
            .filter(|&n| n <= i32::MAX as usize)
            .ok_or_else(|| invalid("FP16 weight expansion too large"))?;
        sum.checked_add(4 * count as u64)
            .ok_or_else(|| invalid("FP16 weight byte overflow"))
    })
}

#[cfg(feature = "cuda-fp16-corrected-prefill-research")]
fn dense_kind(kind: i32, is_llama: bool) -> bool {
    if is_llama {
        kind == 8
    } else {
        matches!(kind, 12 | 14)
    }
}

#[cfg(feature = "cuda-fp16-corrected-prefill-research")]
pub(super) struct Prefill {
    gemm: Gemm,
    expand: CudaFunction,
    x: [CudaSlice<u16>; 2],
    #[cfg(feature = "cuda-qwen35-ffn-weight-only-research")]
    hybrid_ffn_weight_only: bool,
    #[cfg(feature = "cuda-qwen3-fp16-down-weight-only-research")]
    qwen3_down_weight_only: bool,
    #[cfg(any(
        feature = "cuda-qwen3-async-prefill-research",
        feature = "cuda-llama-async-prefill-research"
    ))]
    async_tile: Option<CudaFunction>,
}

#[cfg(feature = "cuda-fp16-corrected-prefill-research")]
impl Prefill {
    pub(super) fn dense(
        s: &Arc<CudaStream>,
        weights: &mut Weights<DeviceTensor>,
        inputs: usize,
        is_llama: bool,
    ) -> Result<Self> {
        let this = Self::new(s, inputs)?;
        #[cfg(feature = "cuda-qwen3-fp16-down-weight-only-research")]
        let this = Self {
            qwen3_down_weight_only: !is_llama,
            ..this
        };
        #[cfg(any(
            feature = "cuda-qwen3-async-prefill-research",
            feature = "cuda-llama-async-prefill-research"
        ))]
        let this = if (is_llama && cfg!(feature = "cuda-llama-async-prefill-research"))
            || (!is_llama && cfg!(feature = "cuda-qwen3-async-prefill-research"))
        {
            Self {
                async_tile: Some(compile_tile(s, "corrected_fp16_tile_async_n64")?),
                ..this
            }
        } else {
            this
        };
        for l in &mut weights.layers {
            for t in [
                &mut l.q,
                &mut l.k,
                &mut l.v,
                &mut l.output,
                &mut l.up,
                &mut l.down,
            ]
            .into_iter()
            .chain(l.gate.as_mut())
            .filter(|t| dense_kind(t.kind, is_llama))
            {
                this.cache(t)?;
            }
        }
        Ok(this)
    }

    pub(super) fn new(s: &Arc<CudaStream>, inputs: usize) -> Result<Self> {
        Ok(Self {
            gemm: Gemm::new(s)?,
            expand: compile(s)?,
            #[cfg(feature = "cuda-qwen35-ffn-weight-only-research")]
            hybrid_ffn_weight_only: false,
            #[cfg(feature = "cuda-qwen3-fp16-down-weight-only-research")]
            qwen3_down_weight_only: false,
            #[cfg(any(
                feature = "cuda-qwen3-async-prefill-research",
                feature = "cuda-llama-async-prefill-research"
            ))]
            async_tile: None,
            x: [
                s.alloc_zeros(inputs).map_err(failure)?,
                s.alloc_zeros(inputs).map_err(failure)?,
            ],
        })
    }

    pub(super) fn cache(&self, t: &mut DeviceTensor) -> Result<()> {
        let count = t
            .width
            .checked_mul(t.rows)
            .filter(|&n| n > 0)
            .ok_or_else(|| invalid("FP16 weight expansion overflow"))?;
        let s = &self.gemm.stream;
        let mut high = s.alloc_zeros(count as usize).map_err(failure)?;
        let mut low = s.alloc_zeros(count as usize).map_err(failure)?;
        expand(s, &self.expand, &t.bytes, t.kind, &mut high, count, false)?;
        expand(s, &self.expand, &t.bytes, t.kind, &mut low, count, true)?;
        t.fp16 = Some([high, low]);
        Ok(())
    }

    #[cfg(feature = "cuda-llama-prefill-fp16-gate-up-research")]
    pub(super) fn project_q8_gate_up(
        &mut self,
        x: &CudaSlice<f32>,
        gate: &DeviceTensor,
        up: &DeviceTensor,
        gate_out: &mut CudaSlice<f32>,
        up_out: &mut CudaSlice<f32>,
        n: i32,
    ) -> Result<()> {
        let count = gate.width * n;
        if x.len() < count as usize {
            return Err(invalid("invalid FP16 Q8 gate/up dimensions"));
        }
        let s = &self.gemm.stream;
        expand(s, &self.expand, x, 0, &mut self.x[0], count, false)?;
        for (t, out) in [(gate, gate_out), (up, up_out)] {
            let [high, _] = t
                .fp16
                .as_ref()
                .ok_or_else(|| invalid("missing FP16 Q8 gate/up cache"))?;
            self.gemm
                .project(high, &self.x[0], out, t.width, t.rows, n, 1.0, 0.0)?;
        }
        Ok(())
    }

    #[cfg(feature = "cuda-llama-short-kv-fp16-research")]
    pub(super) fn project_q8_kv(
        &mut self,
        x: &CudaSlice<f32>,
        k: &DeviceTensor,
        v: &DeviceTensor,
        k_out: &mut CudaSlice<f32>,
        v_out: &mut CudaSlice<f32>,
        n: i32,
    ) -> Result<bool> {
        let (Some([kh, kl]), Some([vh, vl])) = (k.fp16.as_ref(), v.fp16.as_ref()) else {
            return Ok(false);
        };
        if [(k.kind, k.width, k.rows), (v.kind, v.width, v.rows)] != [(8, 2048, 512); 2] {
            return Ok(false);
        }
        let count = k.width * n;
        if x.len() < count as usize {
            return Err(invalid("invalid FP16 Q8 K/V activation size"));
        }
        expand(&self.gemm.stream, &self.expand, x, 0, &mut self.x[0], count, false)?;
        for (high, low, out) in [(kh, kl, k_out), (vh, vl, v_out)] {
            self.gemm.project(high, &self.x[0], out, k.width, k.rows, n, 1.0, 0.0)?;
            self.gemm.project(low, &self.x[0], out, k.width, k.rows, n, 1.0 / 1024.0, 1.0)?;
        }
        Ok(true)
    }

    #[cfg(feature = "cuda-qwen35-ffn-weight-only-research")]
    pub(super) fn with_hybrid_ffn_weight_only(mut self) -> Self {
        self.hybrid_ffn_weight_only = true;
        self
    }

    #[cfg(any(
        feature = "cuda-qwen3-prefill-kv-weight-only-research",
        feature = "cuda-qwen3-prefill-output-weight-only-research"
    ))]
    pub(super) fn project_qwen3_attention_weights(
        &mut self,
        t: &DeviceTensor,
        x: &CudaSlice<f32>,
        y: &mut CudaSlice<f32>,
        width: i32,
        n: i32,
    ) -> Result<bool> {
        let Some([high, low]) = t.fp16.as_ref().filter(|_| {
            self.qwen3_down_weight_only
                && matches!(t.kind, 12 | 14)
                && t.width == width
                && (t.rows == 1024
                    || (cfg!(feature = "cuda-qwen3-prefill-q-weight-only-research")
                        && width == 1024
                        && t.rows == 2048))
        }) else {
            return Ok(false);
        };
        let count = width * n;
        if x.len() < count as usize {
            return Err(invalid("FP16 attention projection input exceeds workspace"));
        }
        expand(&self.gemm.stream, &self.expand, x, 0, &mut self.x[0], count, false)?;
        self.gemm.project(high, &self.x[0], y, width, t.rows, n, 1.0, 0.0)?;
        self.gemm.project(low, &self.x[0], y, width, t.rows, n, 1.0 / 1024.0, 1.0)?;
        Ok(true)
    }

    pub(super) fn project(
        &mut self,
        t: &DeviceTensor,
        x: &CudaSlice<f32>,
        y: &mut CudaSlice<f32>,
        n: i32,
    ) -> Result<bool> {
        let Some([high, low]) = t.fp16.as_ref().filter(|_| n >= 32) else {
            return Ok(false);
        };
        let count = t
            .width
            .checked_mul(n)
            .filter(|&n| n > 0 && n as usize <= x.len())
            .ok_or_else(|| invalid("FP16 activation size overflow"))?;
        let s = &self.gemm.stream;
        expand(s, &self.expand, x, 0, &mut self.x[0], count, false)?;
        #[cfg(feature = "cuda-qwen3-fp16-down-weight-only-research")]
        let qwen3_weight_only = self.qwen3_down_weight_only
            && (n >= 384
                || (cfg!(feature = "cuda-qwen3-fp16-ffn-weight-only-256-research") && n >= 128)
                || cfg!(feature = "cuda-qwen3-fp16-ffn-weight-only-46-research"))
            && (matches!((t.kind, t.width, t.rows), (12 | 14, 3072, 1024))
                || (cfg!(feature = "cuda-qwen3-fp16-gate-up-weight-only-research")
                    && (t.kind, t.width, t.rows) == (12, 1024, 3072)));
        #[cfg(not(feature = "cuda-qwen3-fp16-down-weight-only-research"))]
        let qwen3_weight_only = false;
        let hybrid_gate_up_short =
            (32..128).contains(&n) && cfg!(feature = "cuda-qwen35-ffn-gate-up-single-term-46-research");
        #[cfg(feature = "cuda-qwen35-ffn-weight-only-research")]
        let hybrid_ffn = self.hybrid_ffn_weight_only
            && (n >= 384
                || (cfg!(feature = "cuda-qwen35-ffn-weight-only-256-research") && n >= 128)
                || hybrid_gate_up_short)
            && (t.kind, t.width, t.rows) == (12, 1024, 3584);
        #[cfg(not(feature = "cuda-qwen35-ffn-weight-only-research"))]
        let hybrid_ffn = false;
        #[cfg(feature = "cuda-qwen35-qkv-weight-only-research")]
        let hybrid_large_chunk = n >= 384
            || (cfg!(feature = "cuda-qwen35-qkv-down-weight-only-256-research") && n >= 128);
        #[cfg(feature = "cuda-qwen35-qkv-weight-only-research")]
        let hybrid_qkv = self.hybrid_ffn_weight_only
            && hybrid_large_chunk
            && (t.kind, t.width, t.rows) == (13, 1024, 6144);
        #[cfg(not(feature = "cuda-qwen35-qkv-weight-only-research"))]
        let hybrid_qkv = false;
        #[cfg(feature = "cuda-qwen35-down-weight-only-research")]
        let hybrid_down = self.hybrid_ffn_weight_only
            && hybrid_large_chunk
            && matches!((t.kind, t.width, t.rows), (12 | 14, 3584, 1024));
        #[cfg(not(feature = "cuda-qwen35-down-weight-only-research"))]
        let hybrid_down = false;
        let hybrid_output = cfg!(feature = "cuda-qwen35-output-weight-only-research")
            && self.hybrid_ffn_weight_only
            && (n >= 384 || (n >= 128 && cfg!(feature = "cuda-qwen35-output-single-term-256-research")))
            && matches!((t.kind, t.width, t.rows), (12..=14, 2048, 1024));
        if qwen3_weight_only
            || hybrid_ffn
            || hybrid_qkv
            || hybrid_down
            || hybrid_output
            || (cfg!(feature = "cuda-llama-short-ffn-weight-only-research")
                && (32..128).contains(&n)
                && matches!((t.kind, t.width, t.rows), (8, 2048, 8192) | (8, 8192, 2048)))
            || (cfg!(feature = "cuda-llama-attention-projection-weight-only-research")
                && (n >= 384
                    || (cfg!(
                        feature = "cuda-llama-attention-projection-weight-only-256-research"
                    ) && n >= 128))
                && (t.kind, t.width, t.rows) == (8, 2048, 2048))
            || (cfg!(feature = "cuda-llama-fp16-down-weight-only-research")
                && ((matches!(n, 256 | 512) && (t.kind, t.width, t.rows) == (8, 8192, 2048))
                    || (cfg!(feature = "cuda-llama-fp16-wide-weight-only-research")
                        && matches!(n, 256 | 512)
                        && (t.kind, t.width, t.rows) == (8, 2048, 8192))))
        {
            self.gemm
                .project(high, &self.x[0], y, t.width, t.rows, n, 1.0, 0.0)?;
            let hybrid_gate_up_single_term =
                cfg!(feature = "cuda-qwen35-ffn-gate-up-single-term-research")
                    && hybrid_ffn
                    && (n >= 384
                        || (n >= 128
                            && cfg!(feature = "cuda-qwen35-ffn-gate-up-single-term-256-research"))
                        || hybrid_gate_up_short);
            if hybrid_gate_up_single_term
                || (cfg!(feature = "cuda-llama-prefill-qo-single-term-research")
                    && n >= 384
                    && (t.kind, t.width, t.rows) == (8, 2048, 2048))
                || (cfg!(feature = "cuda-qwen35-ffn-down-single-term-research")
                    && hybrid_down
                    && (n >= 384
                        || (n >= 128
                            && cfg!(feature = "cuda-qwen35-ffn-down-single-term-256-research"))))
                || (cfg!(feature = "cuda-qwen35-output-single-term-research") && hybrid_output)
                || (cfg!(feature = "cuda-qwen35-qkv-single-term-research")
                    && hybrid_qkv
                    && (n >= 384 || (n >= 128 && cfg!(feature = "cuda-qwen35-qkv-single-term-256-research"))))
                || (n >= 384 && n % 128 == 0
                    && cfg!(feature = "cuda-qwen3-prefill-gate-up-single-term-research")
                    && qwen3_weight_only
                    && (t.kind, t.width, t.rows) == (12, 1024, 3072))
                || (n >= 384 && n % 128 == 0
                    && cfg!(feature = "cuda-qwen3-fp16-down-single-term-research")
                    && qwen3_weight_only
                    && matches!((t.kind, t.width, t.rows), (12 | 14, 3072, 1024)))
            {
                return Ok(true);
            }
            self.gemm
                .project(low, &self.x[0], y, t.width, t.rows, n, 1.0 / 1024.0, 1.0)?;
            return Ok(true);
        }
        expand(s, &self.expand, x, 0, &mut self.x[1], count, true)?;
        #[cfg(any(
            feature = "cuda-qwen3-async-prefill-research",
            feature = "cuda-llama-async-prefill-research"
        ))]
        if n >= 256
            && match (t.kind, t.width, t.rows) {
                (8, 2048, 512 | 8192) => n == 512,
                (12 | 14, 1024 | 2048, 1024) => true,
                _ => false,
            }
            && let Some(tile) = &self.async_tile
        {
            return launch_tile(
                s, tile, high, low, &self.x[0], &self.x[1], y, t.width, t.rows, n, 32, 64,
            )
            .map(|()| true);
        }
        self.gemm
            .project(high, &self.x[0], y, t.width, t.rows, n, 1.0, 0.0)?;
        self.gemm
            .project(high, &self.x[1], y, t.width, t.rows, n, 1.0 / 1024.0, 1.0)?;
        self.gemm
            .project(low, &self.x[0], y, t.width, t.rows, n, 1.0 / 1024.0, 1.0)?;
        Ok(true)
    }
}

pub(super) fn compile(stream: &Arc<CudaStream>) -> Result<CudaFunction> {
    compile_expand(stream, "expand_prefill_fp16", None)
}

#[cfg(test)]
pub(super) fn compile_replay_merge(stream: &Arc<CudaStream>) -> Result<CudaFunction> {
    compile_expand(stream, "merge_prefill_fp16", Some("CUDA_FP16_MERGE_REPLAY"))
}

#[cfg(any(
    test,
    feature = "cuda-qwen3-async-prefill-research",
    feature = "cuda-llama-async-prefill-research"
))]
pub(super) fn compile_tile(stream: &Arc<CudaStream>, name: &str) -> Result<CudaFunction> {
    compile_expand(stream, name, Some("CUDA_FP16_TILE_REPLAY"))
}

#[cfg(any(
    test,
    feature = "cuda-qwen3-async-prefill-research",
    feature = "cuda-llama-async-prefill-research"
))]
pub(super) fn launch_tile(
    s: &Arc<CudaStream>,
    kernel: &CudaFunction,
    wh: &CudaSlice<u16>,
    wl: &CudaSlice<u16>,
    xh: &CudaSlice<u16>,
    xl: &CudaSlice<u16>,
    out: &mut impl DevicePtrMut<f32>,
    width: i32,
    rows: i32,
    tokens: i32,
    tile_rows: u32,
    tile_tokens: u32,
) -> Result<()> {
    launch_tile_parts(
        s,
        kernel,
        wh,
        wl,
        xh,
        xl,
        out,
        width,
        rows,
        tokens,
        tile_rows,
        tile_tokens,
        1,
    )
}

#[cfg(any(
    test,
    feature = "cuda-qwen3-async-prefill-research",
    feature = "cuda-llama-async-prefill-research"
))]
pub(super) fn launch_tile_parts(
    s: &Arc<CudaStream>,
    kernel: &CudaFunction,
    wh: &CudaSlice<u16>,
    wl: &CudaSlice<u16>,
    xh: &CudaSlice<u16>,
    xl: &CudaSlice<u16>,
    out: &mut impl DevicePtrMut<f32>,
    width: i32,
    rows: i32,
    tokens: i32,
    tile_rows: u32,
    tile_tokens: u32,
    parts: u32,
) -> Result<()> {
    let size = |a: i32, b: i32| a.checked_mul(b).filter(|&n| n > 0).map(|n| n as usize);
    if size(width, rows) != Some(wh.len())
        || wh.len() != wl.len()
        || size(width, tokens).is_none_or(|n| n > xh.len() || n > xl.len())
        || !matches!(parts, 1 | 2)
        || (parts == 2 && width % 128 != 0)
        || size(rows, tokens)
            .and_then(|n| n.checked_mul(parts as usize))
            .is_none_or(|n| n > out.len())
    {
        return Err(invalid("invalid corrected FP16 tile dimensions"));
    }
    let mut args = s.launch_builder(kernel);
    let (ptr, _write) = out.device_ptr_mut(s);
    args.arg(wh)
        .arg(wl)
        .arg(xh)
        .arg(xl)
        .arg(&ptr)
        .arg(&width)
        .arg(&rows)
        .arg(&tokens);
    unsafe {
        args.launch(LaunchConfig {
            grid_dim: (
                (rows as u32).div_ceil(tile_rows),
                (tokens as u32).div_ceil(tile_tokens),
                parts,
            ),
            block_dim: (tile_rows * 4, 1, 1),
            shared_mem_bytes: 0,
        })
    }
    .map_err(failure)?;
    Ok(())
}

fn compile_expand(
    stream: &Arc<CudaStream>,
    name: &str,
    define: Option<&str>,
) -> Result<CudaFunction> {
    let (major, minor) = stream.context().compute_capability().map_err(failure)?;
    let mut options = vec![format!("--gpu-architecture=compute_{major}{minor}")];
    if let Some(define) = define {
        options.push(format!("-D{define}=1"));
    }
    let ptx = compile_ptx_with_opts(
        concat!(
            include_str!("kernels.cu"),
            "\n",
            include_str!("fp16_gemm.cu")
        ),
        CompileOptions {
            fmad: Some(false),
            options,
            ..Default::default()
        },
    )
    .map_err(failure)?;
    stream
        .context()
        .load_module(ptx)
        .and_then(|module| module.load_function(name))
        .map_err(failure)
}

pub(super) fn expand<'a, T>(
    stream: &Arc<CudaStream>,
    function: &CudaFunction,
    src: &'a CudaSlice<T>,
    kind: i32,
    dst: &mut CudaSlice<u16>,
    count: i32,
    residual: bool,
) -> Result<()> {
    if count <= 0 || count as usize > dst.len() {
        return Err(invalid("invalid FP16 expansion length"));
    }
    let mut args = stream.launch_builder(function);
    let residual = i32::from(residual);
    args.arg(src).arg(&kind).arg(dst).arg(&count).arg(&residual);
    unsafe { args.launch(LaunchConfig::for_num_elems(count as u32)) }.map_err(failure)?;
    Ok(())
}

pub(super) struct Gemm {
    stream: Arc<CudaStream>,
    blas: CudaBlas,
}

impl Gemm {
    pub(super) fn new(stream: &Arc<CudaStream>) -> Result<Self> {
        let blas = CudaBlas::new(stream.clone()).map_err(failure)?;
        Ok(Self {
            stream: stream.clone(),
            blas,
        })
    }

    pub(super) fn project(
        &self,
        weight: &CudaSlice<u16>,
        input: &CudaSlice<u16>,
        output: &mut impl DevicePtrMut<f32>,
        width: i32,
        rows: i32,
        tokens: i32,
        alpha: f32,
        beta: f32,
    ) -> Result<()> {
        self.project_impl(
            weight,
            input,
            output,
            width,
            rows,
            tokens,
            alpha,
            beta,
            sys::cudaDataType::CUDA_R_16F,
            sys::cublasComputeType_t::CUBLAS_COMPUTE_32F,
            false,
        )
    }

    #[cfg(test)]
    pub(super) fn project_nn(
        &self,
        weight: &CudaSlice<u16>,
        input: &CudaSlice<u16>,
        output: &mut impl DevicePtrMut<f32>,
        width: i32,
        rows: i32,
        tokens: i32,
        alpha: f32,
        beta: f32,
    ) -> Result<()> {
        self.project_impl(
            weight,
            input,
            output,
            width,
            rows,
            tokens,
            alpha,
            beta,
            sys::cudaDataType::CUDA_R_16F,
            sys::cublasComputeType_t::CUBLAS_COMPUTE_32F,
            true,
        )
    }

    #[cfg(test)]
    pub(super) fn transpose_weights(
        &self,
        input: &CudaSlice<u16>,
        width: i32,
        rows: i32,
    ) -> Result<CudaSlice<u16>> {
        let s = &self.stream;
        let count = width
            .checked_mul(rows)
            .filter(|&n| n > 0 && n as usize == input.len())
            .ok_or_else(|| invalid("invalid FP16 transpose dimensions"))?;
        let function = compile_expand(s, "transpose_prefill_fp16", Some("CUDA_FP16_NN_REPLAY"))?;
        let mut output = s.alloc_zeros(count as usize).map_err(failure)?;
        let mut args = s.launch_builder(&function);
        args.arg(input).arg(&mut output).arg(&width).arg(&rows);
        unsafe { args.launch(LaunchConfig::for_num_elems(count as u32)) }.map_err(failure)?;
        s.synchronize().map_err(failure)?;
        Ok(output)
    }

    fn project_impl<T>(
        &self,
        weight: &CudaSlice<T>,
        input: &CudaSlice<T>,
        output: &mut impl DevicePtrMut<f32>,
        width: i32,
        rows: i32,
        tokens: i32,
        alpha: f32,
        beta: f32,
        ty: sys::cudaDataType,
        compute: sys::cublasComputeType_t,
        nn: bool,
    ) -> Result<()> {
        let elements = |a: i32, b: i32| (a as usize).saturating_mul(b as usize);
        if width <= 0
            || rows <= 0
            || tokens <= 0
            || weight.len() != elements(width, rows)
            || input.len() < elements(width, tokens)
            || output.len() < elements(rows, tokens)
        {
            return Err(invalid("invalid FP16 prefill matrix dimensions"));
        }
        let s = &self.stream;
        s.context().bind_to_thread().map_err(failure)?;
        let (w, _read_w) = weight.device_ptr(s);
        let (x, _read_x) = input.device_ptr(s);
        let (y, _write_y) = output.device_ptr_mut(s);
        // Token-major Rust output is column-major [rows, tokens] for cuBLAS.
        unsafe {
            result::gemm_ex(
                *self.blas.handle(),
                if nn {
                    sys::cublasOperation_t::CUBLAS_OP_N
                } else {
                    sys::cublasOperation_t::CUBLAS_OP_T
                },
                sys::cublasOperation_t::CUBLAS_OP_N,
                rows,
                tokens,
                width,
                (&alpha as *const f32).cast(),
                w as *const _,
                ty,
                if nn { rows } else { width },
                x as *const _,
                ty,
                width,
                (&beta as *const f32).cast(),
                y as *mut _,
                sys::cudaDataType::CUDA_R_32F,
                rows,
                compute,
                sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT,
            )
        }
        .map_err(failure)
    }

    #[cfg(test)]
    pub(super) fn replay_f32(
        &self,
        t: &DeviceTensor,
        input: &CudaSlice<f32>,
        tokens: i32,
        reference: &[f32],
        name: &str,
        tf32: bool,
    ) -> Result<()> {
        let (label, compute) = if tf32 {
            (
                "TF32",
                sys::cublasComputeType_t::CUBLAS_COMPUTE_32F_FAST_TF32,
            )
        } else {
            (
                "BF16X9",
                sys::cublasComputeType_t::CUBLAS_COMPUTE_32F_EMULATED_16BFX9,
            )
        };
        let s = &self.stream;
        let count = t
            .width
            .checked_mul(t.rows)
            .filter(|&n| n > 0)
            .ok_or_else(|| invalid("invalid F32 GEMM replay dimensions"))?;
        let expand = compile_expand(s, "expand_prefill_f32", Some("CUDA_FP32_GEMM_REPLAY"))?;
        let mut weights = s.alloc_zeros(count as usize).map_err(failure)?;
        let mut args = s.launch_builder(&expand);
        args.arg(&t.bytes)
            .arg(&t.kind)
            .arg(&mut weights)
            .arg(&count);
        unsafe { args.launch(LaunchConfig::for_num_elems(count as u32)) }.map_err(failure)?;
        let end = 8 + reference.len();
        let mut guard = vec![1337.0; end + 8];
        guard[8..end].fill(f32::NAN);
        let mut y = s.clone_htod(&guard).map_err(failure)?;
        let mut launch = || {
            self.project_impl(
                &weights,
                input,
                &mut y.slice_mut(8..end),
                t.width,
                t.rows,
                tokens,
                1.0,
                0.0,
                sys::cudaDataType::CUDA_R_32F,
                compute,
                false,
            )
        };
        for _ in 0..3 {
            launch()?;
        }
        let mut samples = Vec::new();
        for _ in 0..7 {
            let flag = Some(cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT);
            let start = s.record_event(flag).map_err(failure)?;
            launch()?;
            let finish = s.record_event(flag).map_err(failure)?;
            samples.push(start.elapsed_ms(&finish).map_err(failure)?);
        }
        let output = s.clone_dtoh(&y).map_err(failure)?;
        if output[..8] != guard[..8]
            || output[end..] != guard[end..]
            || output[8..end].iter().any(|a| !a.is_finite())
        {
            return Err(failure(format!("{label} replay {name} invalid output")));
        }
        let max_abs = output[8..end]
            .iter()
            .zip(reference)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f32::max);
        println!(
            "DENSE_{label}_REPLAY {}",
            serde_json::json!({
                "tensor": name, "tokens": tokens, "kind": t.kind, "prep_plus_matmul_ms": samples,
                "max_abs_vs_f32_capture": max_abs, "expanded_weight_bytes": 4 * weights.len()
            })
        );
        Ok(())
    }
}
