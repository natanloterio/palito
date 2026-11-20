use super::*;
use crate::cuda::{
    Backend, CompileOptions, CudaContext, CudaFunction, CudaMatmul, CudaOptions, CudaSlice,
    CudaStream, DeviceTensor, HashMap, LaunchConfig, PushKernelArg, State, cache, failure, greedy,
    sys,
};
use std::{path::Path, sync::Arc};

enum DeviceAttention {
    Dense {
        qg: DeviceTensor,
        k: DeviceTensor,
        v: DeviceTensor,
        q_norm: DeviceTensor,
        k_norm: DeviceTensor,
        output: DeviceTensor,
        keys: CudaSlice<f32>,
        values: CudaSlice<f32>,
    },
    Delta {
        qkv: DeviceTensor,
        z: DeviceTensor,
        alpha: DeviceTensor,
        beta: DeviceTensor,
        dt: DeviceTensor,
        decay: DeviceTensor,
        conv: DeviceTensor,
        norm: DeviceTensor,
        output: DeviceTensor,
        history: CudaSlice<f32>,
        state: CudaSlice<f32>,
    },
}

struct DeviceLayer {
    attention: DeviceAttention,
    input_norm: DeviceTensor,
    ffn_norm: DeviceTensor,
    gate: DeviceTensor,
    up: DeviceTensor,
    down: DeviceTensor,
}

struct Hybrid {
    graph: Option<DecodeGraph>,
    device_greedy: Option<(crate::cuda::decode::DeviceGreedy, CudaSlice<i32>)>,
    #[cfg(feature = "cuda-qwen35-recurrent-gdn-512-research")]
    gdn: Option<Gdn512>,
    #[cfg(feature = "cuda-fp16-corrected-prefill-research")]
    fp16: Option<crate::cuda::fp16_gemm::Prefill>,
    stream: Arc<CudaStream>,
    functions: HashMap<&'static str, CudaFunction>,
    info: ModelInfo,
    tokenizer: Result<Tokenizer>,
    eog: Vec<i32>,
    epsilon: f32,
    base: f32,
    embedding: DeviceTensor,
    output_norm: DeviceTensor,
    layers: Vec<DeviceLayer>,
    work: Work,
    logits: Vec<f32>,
    state: State,
    capacity: usize,
    #[cfg(test)]
    projection_probe: bool,
    #[cfg(all(test, feature = "cuda-fp16-prefill-research"))]
    fp16_probe: Option<(CudaFunction, crate::cuda::fp16_gemm::Gemm)>,
}

struct DecodeGraph {
    graphs: std::collections::BTreeMap<(u32, bool), cudarc::driver::CudaGraph>,
    control: CudaSlice<i32>,
}

impl Drop for Hybrid {
    fn drop(&mut self) {
        if self.graph.is_some() {
            self.stream.context().record_err(self.stream.synchronize());
        }
    }
}

#[cfg(feature = "cuda-qwen35-recurrent-gdn-512-research")]
type GdnLaunch = unsafe extern "C" fn(sys::CUstream, u64, u64, u64, u64, u64, u64, u64, u64) -> i32;

#[cfg(feature = "cuda-qwen35-recurrent-gdn-512-research")]
struct Gdn512 {
    _library: libloading::Library,
    stream: Arc<CudaStream>,
    launch: GdnLaunch,
    logs: CudaSlice<f32>,
    state_in: CudaSlice<f32>,
    state_out: CudaSlice<f32>,
}

#[cfg(feature = "cuda-qwen35-recurrent-gdn-512-research")]
impl Drop for Gdn512 {
    fn drop(&mut self) {
        if let Err(error) = self.stream.synchronize() {
            eprintln!("Qwen3.5 GDN synchronization on drop: {error}");
        }
    }
}

#[cfg(feature = "cuda-qwen35-recurrent-gdn-512-research")]
impl Gdn512 {
    const STATE: usize = GROUPS * STATE_DIM * STATE_DIM;
    const LOGS: usize = 512 * GROUPS;
    const BYTES: u64 = ((2 * Self::STATE + Self::LOGS) * size_of::<f32>()) as u64;

    fn open(stream: &Arc<CudaStream>) -> Result<Self> {
        let library = unsafe { libloading::Library::new("libli_qwen35_gdn_research.so") }
            .map_err(failure)?;
        let launch = unsafe { *library.get::<GdnLaunch>(b"li_qwen35_gdn_512\0").map_err(failure)? };
        Ok(Self {
            _library: library,
            stream: stream.clone(),
            launch,
            logs: stream.alloc_zeros(Self::LOGS).map_err(failure)?,
            state_in: stream.alloc_zeros(Self::STATE).map_err(failure)?,
            state_out: stream.alloc_zeros(Self::STATE).map_err(failure)?,
        })
    }

    fn run(&mut self, work: &mut Work, state: &mut CudaSlice<f32>) -> Result<()> {
        use cudarc::driver::{DevicePtr, DevicePtrMut};
        let stream = &self.stream;
        stream.context().bind_to_thread().map_err(failure)?;
        let (qkv, _qkv) = work.qkv.device_ptr(stream);
        let (decay, _decay) = work.alpha.device_ptr(stream);
        let (beta, _beta) = work.beta.device_ptr(stream);
        let (state, _state) = state.device_ptr_mut(stream);
        let (out, _out) = work.attended.device_ptr_mut(stream);
        let (logs, _logs) = self.logs.device_ptr_mut(stream);
        let (state_in, _in) = self.state_in.device_ptr_mut(stream);
        let (state_out, _state_out) = self.state_out.device_ptr_mut(stream);
        let status = unsafe {
            (self.launch)(stream.cu_stream(), qkv, decay, beta, state, out, logs, state_in, state_out)
        };
        if status != 0 {
            return Err(failure(format!("Qwen3.5 GDN launch: CUDA status {status}")));
        }
        Ok(())
    }
}

struct Work {
    splitk: Option<CudaSlice<f32>>,
    packed: Option<PackedWork>,
    x: CudaSlice<f32>,
    normalized: CudaSlice<f32>,
    projected: CudaSlice<f32>,
    gate: CudaSlice<f32>,
    up: CudaSlice<f32>,
    qkv: CudaSlice<f32>,
    q: CudaSlice<f32>,
    k: CudaSlice<f32>,
    v: CudaSlice<f32>,
    attention_gate: CudaSlice<f32>,
    attended: CudaSlice<f32>,
    alpha: CudaSlice<f32>,
    beta: CudaSlice<f32>,
    scores: CudaSlice<f32>,
    logits: CudaSlice<f32>,
    ids: CudaSlice<i32>,
}

struct PackedWork {
    values: CudaSlice<i8>,
    scales: CudaSlice<f32>,
    sums: CudaSlice<i32>,
    status: CudaSlice<i32>,
}

pub(in crate::cuda) fn open(
    path: &Path,
    config: &Config,
    options: CudaOptions,
) -> Result<Box<dyn Backend>> {
    Ok(Box::new(open_hybrid(path, config, options, false)?))
}

fn open_hybrid(
    path: &Path,
    config: &Config,
    options: CudaOptions,
    projection_probe: bool,
) -> Result<Hybrid> {
    config.validate()?;
    let decode_graph = options.decode_graph && cfg!(feature = "cuda-qwen35-decode-graph-research");
    if options.matmul != CudaMatmul::F32 || (options.decode_graph && !decode_graph) {
        return Err(invalid(
            "Qwen3.5 CUDA requires f32 matmul without a decode graph",
        ));
    }
    if !cfg!(target_os = "linux") || config.context_tokens > 65535 {
        return Err(invalid("Qwen3.5 CUDA requires Linux and context <= 65535"));
    }
    let gguf = Gguf::open(path, false)?;
    let weight_bytes: u64 = gguf.tensors.values().map(Tensor::allocated_bytes).sum();
    let model = Qwen35::load(gguf, config, false)?;
    let context = config.context_tokens as usize;
    let kv = KV_HEADS * HEAD_DIM * context;
    let history = CHANNELS * (KERNEL - 1);
    let state = GROUPS * STATE_DIM * STATE_DIM;
    let vocab = model.info.vocab_tokens as usize;
    let capacity = config.prefill_chunk.min(512) as usize;
    let packed = cfg!(feature = "cuda-packed-prefill-research") && capacity >= 32;
    let packed_count = capacity * model.width.max(model.hidden);
    #[cfg(feature = "cuda-fp16-corrected-prefill-research")]
    let fp16_bytes = if capacity >= 32 && !projection_probe {
        crate::cuda::fp16_gemm::cache_bytes(
            model
                .layers
                .iter()
                .flat_map(|l| {
                    let common = [&l.gate, &l.up, &l.down];
                    let attn = match &l.attention {
                        Attention::Dense(t) => {
                            [Some(&t.qg), Some(&t.k), Some(&t.v), Some(&t.output), None]
                        }
                        Attention::Delta(t) => [
                            Some(&t.qkv),
                            Some(&t.z),
                            Some(&t.alpha),
                            Some(&t.beta),
                            Some(&t.output),
                        ],
                    };
                    common.into_iter().chain(attn.into_iter().flatten())
                })
                .filter(|t| {
                    matches!(
                        t.storage,
                        crate::gguf::Storage::Packed { kind: 12..=14, .. }
                    )
                }),
        )?
    } else {
        0
    };
    #[cfg(feature = "cuda-fp16-corrected-prefill-research")]
    let packed = packed && fp16_bytes == 0;
    let head_dp4a = cfg!(feature = "cuda-qwen35-dp4a-head-research")
        && model.width % 256 == 0
        && matches!(
            model.embedding.storage,
            crate::gguf::Storage::Packed { kind: 12..=14, .. }
        );
    let packed = packed || head_dp4a;
    #[cfg(feature = "cuda-qwen35-matvec-cache-research")]
    let coefficient_bytes = model
        .layers
        .iter()
        .flat_map(|l| {
            let attention = match &l.attention {
                Attention::Dense(t) => [&t.qg, &t.k, &t.v, &t.output, &t.output],
                Attention::Delta(t) => [&t.qkv, &t.z, &t.alpha, &t.beta, &t.output],
            };
            let count = if matches!(l.attention, Attention::Dense(_)) {
                4
            } else {
                5
            };
            [&l.gate, &l.up, &l.down]
                .into_iter()
                .chain(attention.into_iter().take(count))
        })
        .try_fold(0u64, |sum, t| {
            if matches!(
                t.storage,
                crate::gguf::Storage::Packed { kind: 12..=14, .. }
            ) && t.dims[0] % 256 == 0
            {
                sum.checked_add(64 * (t.dims[0] as u64 / 256) * t.dims[1] as u64)
                    .ok_or_else(|| invalid("hybrid K coefficient byte count overflow"))
            } else {
                Ok(sum)
            }
        })?;
    let splitk_elements = if cfg!(feature = "cuda-qwen35-splitk-research") && capacity >= 256 {
        capacity * CHANNELS * 4
    } else {
        0
    };
    let scores = HEADS * if cfg!(feature = "cuda-qwen35-split-decode-attention-research") && context > 512 {
        context.max(context.div_ceil(128) * (HEAD_DIM + 2))
    } else {
        context
    };
    let floats = 6 * 2 * kv
        + 18 * (history + state)
        + capacity
            * (3 * model.width
                + 2 * model.hidden
                + CHANNELS
                + 3 * INNER
                + 2 * KV_HEADS * HEAD_DIM
                + 2 * GROUPS)
        + scores
        + vocab;
    let packed_bytes = if packed {
        3 * packed_count + 24 * packed_count.div_ceil(32) + 4
    } else {
        0
    };
    let required = weight_bytes
        + 4 * (floats as u64 + capacity as u64 + splitk_elements as u64)
        + packed_bytes as u64
        + if decode_graph { 8 } else { 0 }
        + if cfg!(feature = "cuda-device-greedy-research") {
            crate::cuda::decode::DeviceGreedy::bytes(vocab) + 4
        } else {
            0
        };
    #[cfg(feature = "cuda-qwen35-matvec-cache-research")]
    let required = required
        .checked_add(coefficient_bytes)
        .ok_or_else(|| invalid("hybrid K coefficient device budget overflow"))?;
    #[cfg(feature = "cuda-fp16-corrected-prefill-research")]
    let required = required
        .checked_add(if fp16_bytes == 0 {
            0
        } else {
            fp16_bytes + 4 * packed_count as u64
        })
        .ok_or_else(|| invalid("Qwen3.5 FP16 device budget overflow"))?;
    if required > options.memory_limit_bytes {
        return Err(invalid(format!(
            "Qwen3.5 CUDA requires {required} device bytes"
        )));
    }
    for name in ["libcuda.so.1", "libnvrtc.so.13"] {
        // cudarc uses lazy loading, so report unavailable libraries before calling it.
        unsafe { libloading::Library::new(name) }.map_err(failure)?;
    }
    let cuda = CudaContext::new(options.device).map_err(failure)?;
    let major = cuda
        .attribute(sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)
        .map_err(failure)?;
    let minor = cuda
        .attribute(sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)
        .map_err(failure)?;
    #[cfg(feature = "cuda-qwen35-recurrent-gdn-512-research")]
    let gdn_enabled = capacity == 512 && (major, minor) == (12, 0) && !projection_probe;
    #[cfg(feature = "cuda-qwen35-recurrent-gdn-512-research")]
    let required = if gdn_enabled {
        required.checked_add(Gdn512::BYTES)
            .ok_or_else(|| invalid("Qwen3.5 GDN device budget overflow"))?
    } else {
        required
    };
    let source = concat!(
        include_str!("../../kernels.cu"),
        "\n#define CUDA_REGISTER_PREFILL 1\n",
        include_str!("../../prefill.cu"),
        "\n#define CUDA_NORMALIZED_ATTENTION 1\n",
        include_str!("../../attention.cu"),
        "\n#define QWEN35_HYBRID\n",
        include_str!("../../qwen35_delta.cu")
    );
    let splitk = splitk_elements > 0;
    if (splitk || projection_probe || packed) && major < 8 {
        return Err(invalid(
            "Qwen3.5 tensor-core research requires compute capability >= 8.0",
        ));
    }
    let splitk_source;
    let source = if projection_probe || splitk || packed {
        let tensorcore = include_str!("../../tensorcore.cu");
        splitk_source = if projection_probe || packed {
            format!(
                "{source}\n{tensorcore}\n{}\n{}\n{}",
                include_str!("../../residual.cu"),
                include_str!("../../residual_precise.cu"),
                include_str!("../../residual_mma.cu"),
            )
        } else {
            format!("{source}\n{tensorcore}")
        };
        &splitk_source
    } else {
        source
    };
    let head_source;
    let source = if head_dp4a {
        head_source = format!("{source}\n{}", include_str!("../../k_dp4a.cu"));
        &head_source
    } else {
        source
    };
    let greedy_source;
    let source = if cfg!(feature = "cuda-device-greedy-research") {
        greedy_source = format!("{source}\n{}", include_str!("../../decode.cu"));
        &greedy_source
    } else {
        source
    };
    let ptx = cache::compile(
        options.kernel_cache_dir.as_deref(),
        source,
        CompileOptions {
            fmad: Some(false),
            options: {
                let mut options = crate::cuda::decode_score_reuse_options(
                    format!("--gpu-architecture=compute_{major}{minor}"),
                    true,
                );
                if decode_graph {
                    options.push("-DCUDA_QWEN35_DECODE_GRAPH=1".into());
                }
                if cfg!(feature = "cuda-qwen35-split-decode-attention-research") {
                    options.push("-DCUDA_QWEN35_SPLIT_DECODE_ATTENTION=1".into());
                }
                if projection_probe {
                    options.push("-DCUDA_QWEN35_PROJECTION_PROBE=1".into());
                    options.push("-DCUDA_QWEN35_Q5_WARP_RESEARCH=1".into());
                }
                if projection_probe || packed {
                    options.push("-DCUDA_DENSE_PACKED_REPLAY=1".into());
                }
                if cfg!(feature = "cuda-qwen35-matvec-reduction-research") {
                    options.push("-DCUDA_QWEN35_MATVEC_REDUCTION=1".into());
                }
                if cfg!(feature = "cuda-qwen35-matvec-cache-research") {
                    options.push("-DCUDA_QWEN35_MATVEC_CACHE=1".into());
                }
                if cfg!(feature = "cuda-qwen35-matvec-fixed-width-research") {
                    options.push("-DCUDA_QWEN35_MATVEC_FIXED_WIDTH=1".into());
                }
                if cfg!(feature = "cuda-qwen35-matvec-ffn-width-research") {
                    options.push("-DCUDA_QWEN35_MATVEC_FFN_WIDTH=1".into());
                }
                if cfg!(feature = "cuda-qwen35-fused-ffn-fixed-width-research") {
                    options.push("-DCUDA_QWEN35_SWIGLU_FIXED_WIDTH=1".into());
                }
                if cfg!(feature = "cuda-qwen35-fused-ffn-decode-research") {
                    options.push("-DCUDA_QWEN35_FUSED_FFN=1".into());
                }
                if cfg!(feature = "cuda-qwen35-value-columns-research") {
                    options.push("-DCUDA_QWEN35_VALUE_COLUMNS=1".into());
                }
                if cfg!(feature = "cuda-qwen35-recurrent-fma-research") {
                    options.push("-DCUDA_QWEN35_RECURRENT_FMA=1".into());
                }
                if cfg!(feature = "cuda-qwen35-recurrent-stage-qk-research") {
                    options.push("-DCUDA_QWEN35_RECURRENT_STAGE_QK=1".into());
                }
                if cfg!(feature = "cuda-qwen35-recurrent-panel64-research") {
                    options.push("-DCUDA_QWEN35_RECURRENT_PANEL64=1".into());
                }
                if cfg!(feature = "cuda-qwen35-recurrent-warp-research") {
                    options.push("-DCUDA_QWEN35_RECURRENT_WARP=1".into());
                    if cfg!(feature = "cuda-qwen35-recurrent-direct-qk-research") {
                        options.push("-DCUDA_QWEN35_RECURRENT_DIRECT_QK=1".into());
                    }
                    if cfg!(feature = "cuda-qwen35-recurrent-eight-segments-research") {
                        options.push("-DCUDA_QWEN35_RECURRENT_EIGHT_SEGMENTS=1".into());
                    }
                    if cfg!(feature = "cuda-qwen35-recurrent-four-chains-research") {
                        options.push("-DCUDA_QWEN35_RECURRENT_FOUR_CHAINS=1".into());
                    }
                    if cfg!(feature = "cuda-qwen35-recurrent-norm-gate-research") {
                        options.push("-DCUDA_QWEN35_RECURRENT_NORM_GATE=1".into());
                    }
                    if cfg!(feature = "cuda-qwen35-residual-norm-research") {
                        options.push("-DCUDA_QWEN35_RESIDUAL_NORM=1".into());
                    }
                }
                if cfg!(feature = "cuda-qwen35-conv-tile-research") {
                    options.push("-DCUDA_QWEN35_CONV_TILE=1".into());
                }
                if cfg!(feature = "cuda-qwen35-prefill-query-pair-research") {
                    options.push("-DCUDA_QWEN35_PREFILL_QUERY_PAIR=1".into());
                }
                if projection_probe || splitk {
                    options.push("-DCUDA_QWEN35_SPLITK_PROBE=1".into());
                }
                if cfg!(feature = "cuda-qwen35-q5-prefill-research") {
                    options.push("-DCUDA_QWEN35_Q5_PREFILL_RESEARCH=1".into());
                }
                options
            },
            ..Default::default()
        },
    )?;
    let module = cuda.load_module(ptx).map_err(failure)?;
    let mut functions = HashMap::new();
    if cfg!(feature = "cuda-qwen35-split-decode-attention-research") {
        for name in ["qwen35_split_decode_partials", "qwen35_split_decode_merge"] {
            functions.insert(name, module.load_function(name).map_err(failure)?);
        }
    }
    if cfg!(feature = "cuda-device-greedy-research") {
        for name in ["greedy_partials", "greedy_finish_id"] {
            functions.insert(name, module.load_function(name).map_err(failure)?);
        }
    }
    if cfg!(feature = "cuda-qwen35-residual-norm-research") {
        let name = "residual_normalize";
        functions.insert(name, module.load_function(name).map_err(failure)?);
    }
    if cfg!(feature = "cuda-qwen35-recurrent-norm-gate-research") {
        let name = "qwen35_delta_norm_gate";
        functions.insert(name, module.load_function(name).map_err(failure)?);
    }
    if cfg!(feature = "cuda-qwen35-value-columns-research") {
        for name in [
            "qwen35_attention_normalize",
            "qwen35_attention_value_columns",
        ] {
            functions.insert(name, module.load_function(name).map_err(failure)?);
        }
    }
    if cfg!(feature = "cuda-qwen35-conv-tile-research") {
        for name in ["qwen35_conv_save", "qwen35_conv_tile"] {
            functions.insert(name, module.load_function(name).map_err(failure)?);
        }
    }
    if cfg!(feature = "cuda-qwen35-recurrent-panel64-research") {
        let name = "qwen35_delta_panel64";
        functions.insert(name, module.load_function(name).map_err(failure)?);
    }
    if cfg!(feature = "cuda-qwen35-recurrent-warp-research") {
        let name = "qwen35_delta_warp";
        functions.insert(name, module.load_function(name).map_err(failure)?);
    }
    if cfg!(feature = "cuda-qwen35-prefill-query-pair-research") {
        let name = "attention_fused_query_pair_256";
        functions.insert(name, module.load_function(name).map_err(failure)?);
    }
    #[cfg(feature = "cuda-qwen35-matvec-cache-research")]
    for name in ["qwen35_expand_k_coefficients", "qwen35_cached_matvec"] {
        functions.insert(name, module.load_function(name).map_err(failure)?);
    }
    if cfg!(feature = "cuda-qwen35-fused-ffn-decode-research") {
        let name = "qwen35_cached_swiglu";
        functions.insert(name, module.load_function(name).map_err(failure)?);
    }
    if head_dp4a {
        let name = "k_dp4a_head";
        functions.insert(name, module.load_function(name).map_err(failure)?);
    }
    if decode_graph {
        for name in [
            "embedding_decode", "cache_kv_decode", "qwen35_rotary_graph",
            "qwen35_attention_scores_graph", "qwen35_attention_values_graph",
        ] {
            functions.insert(name, module.load_function(name).map_err(failure)?);
        }
        if functions.contains_key("qwen35_attention_normalize") {
            for name in ["qwen35_attention_normalize_graph", "qwen35_attention_value_columns_graph"] {
                functions.insert(name, module.load_function(name).map_err(failure)?);
            }
        }
    }
    for name in [
        "embedding",
        "prefill_embedding",
        "prefill_matmul",
        "prefill_matmul_ref2x16",
        "matvec",
        "normalize",
        "residual",
        "swiglu",
        "head_norm",
        "rotary_partial_neox",
        "cache_kv",
        "attention_scores",
        "attention_values",
        "attention_fused",
        "qwen35_conv",
        "qwen35_l2",
        "qwen35_delta_gates",
        "qwen35_delta_batch",
        "qwen35_delta_panel",
        "qwen35_delta_norm",
        "qwen35_gate",
        "qwen35_split_qg",
    ] {
        functions.insert(name, module.load_function(name).map_err(failure)?);
    }
    if projection_probe || splitk {
        for name in ["qwen35_q5_tf32x3_split", "qwen35_q5_split_reduce"] {
            functions.insert(name, module.load_function(name).map_err(failure)?);
        }
    }
    if projection_probe {
        for name in [
            "qwen35_pack_tf32x3",
            "qwen35_q5_tf32x3_packed_split",
            "qwen35_q4_tf32x3_packed",
            "prefill_matmul_tf32x3_tiled",
            "qwen35_q5_tf32x3_warp_split",
            "residual_quantize32_precise1",
            "dense_packed_replay_mma1",
        ] {
            functions.insert(name, module.load_function(name).map_err(failure)?);
        }
    }
    if projection_probe || packed {
        for name in ["residual_quantize32_precise3", "dense_packed_replay_mma3"] {
            functions.insert(name, module.load_function(name).map_err(failure)?);
        }
    }
    let stream = if decode_graph {
        // All hybrid buffers and launches are confined to this stream.
        unsafe { cuda.disable_event_tracking() };
        cuda.new_stream().map_err(failure)?
    } else {
        cuda.default_stream()
    };
    #[cfg(feature = "cuda-qwen35-recurrent-gdn-512-research")]
    let gdn = if gdn_enabled {
        let (free, _) = cudarc::driver::result::mem_get_info().map_err(failure)?;
        crate::cuda::admit(required, options.memory_limit_bytes, free as u64)?;
        Some(Gdn512::open(&stream)?)
    } else {
        None
    };
    #[cfg(feature = "cuda-qwen35-matvec-cache-research")]
    if coefficient_bytes != 0 {
        let (free, _) = cudarc::driver::result::mem_get_info().map_err(failure)?;
        crate::cuda::admit(required, options.memory_limit_bytes, free as u64)?;
    }
    #[cfg(feature = "cuda-fp16-corrected-prefill-research")]
    let fp16 = if fp16_bytes > 0 {
        if major < 8 {
            return Err(invalid("Qwen3.5 FP16 correction requires SM80+"));
        }
        let (free, _) = cudarc::driver::result::mem_get_info().map_err(failure)?;
        crate::cuda::admit(required, options.memory_limit_bytes, free as u64)?;
        let prefill = crate::cuda::fp16_gemm::Prefill::new(&stream, packed_count)?;
        #[cfg(feature = "cuda-qwen35-ffn-weight-only-research")]
        let prefill = prefill.with_hybrid_ffn_weight_only();
        Some(prefill)
    } else {
        None
    };
    let alloc = |n| stream.alloc_zeros::<f32>(n).map_err(failure);
    let upload = |tensor| DeviceTensor::upload(tensor, &stream);
    let project_weight = |tensor| -> Result<_> {
        let t = upload(tensor)?;
        #[cfg(feature = "cuda-qwen35-matvec-cache-research")]
        let t = {
            let mut t = t;
            if matches!(t.kind, 12..=14) && t.width % 256 == 0 {
                let blocks = (t.width / 256)
                    .checked_mul(t.rows)
                    .ok_or_else(|| invalid("hybrid K coefficient count overflow"))?;
                let count = blocks
                    .checked_mul(16)
                    .ok_or_else(|| invalid("hybrid K coefficient count overflow"))?;
                let mut coefficients = alloc(count as usize)?;
                let mut args = stream.launch_builder(&functions["qwen35_expand_k_coefficients"]);
                args.arg(&t.bytes)
                    .arg(&t.kind)
                    .arg(&mut coefficients)
                    .arg(&count);
                unsafe { args.launch(LaunchConfig::for_num_elems(count as u32)) }
                    .map_err(failure)?;
                t.k_coefficients = Some(coefficients);
            }
            t
        };
        #[cfg(feature = "cuda-fp16-corrected-prefill-research")]
        let t = {
            let mut t = t;
            if let Some(fp16) = fp16.as_ref().filter(|_| matches!(t.kind, 12..=14)) {
                fp16.cache(&mut t)?;
            }
            t
        };
        Ok(t)
    };
    let plain = |tensor: Tensor| {
        if !matches!(tensor.storage, crate::gguf::Storage::F32(_)) {
            return Err(invalid("Qwen3.5 CUDA convolution requires F32 weights"));
        }
        upload(tensor)
    };
    let mut layers = Vec::with_capacity(model.layers.len());
    for layer in model.layers {
        let attention = match layer.attention {
            Attention::Dense(t) => DeviceAttention::Dense {
                qg: project_weight(t.qg)?,
                k: project_weight(t.k)?,
                v: project_weight(t.v)?,
                q_norm: upload(t.q_norm)?,
                k_norm: upload(t.k_norm)?,
                output: project_weight(t.output)?,
                keys: alloc(kv)?,
                values: alloc(kv)?,
            },
            Attention::Delta(t) => DeviceAttention::Delta {
                qkv: project_weight(t.qkv)?,
                z: project_weight(t.z)?,
                alpha: project_weight(t.alpha)?,
                beta: project_weight(t.beta)?,
                dt: upload(t.dt)?,
                decay: upload(t.decay)?,
                conv: plain(t.conv)?,
                norm: upload(t.norm)?,
                output: project_weight(t.output)?,
                history: alloc(history)?,
                state: alloc(state)?,
            },
        };
        layers.push(DeviceLayer {
            attention,
            input_norm: upload(layer.input_norm)?,
            ffn_norm: upload(layer.ffn_norm)?,
            gate: project_weight(layer.gate)?,
            up: project_weight(layer.up)?,
            down: project_weight(layer.down)?,
        });
    }
    let work = Work {
        packed: if packed {
            Some(PackedWork {
                values: stream.alloc_zeros(3 * packed_count).map_err(failure)?,
                scales: stream
                    .alloc_zeros(3 * packed_count.div_ceil(32))
                    .map_err(failure)?,
                sums: stream
                    .alloc_zeros(3 * packed_count.div_ceil(32))
                    .map_err(failure)?,
                status: stream.alloc_zeros(1).map_err(failure)?,
            })
        } else {
            None
        },
        splitk: if splitk {
            Some(alloc(splitk_elements)?)
        } else {
            None
        },
        x: alloc(capacity * model.width)?,
        normalized: alloc(capacity * model.width)?,
        projected: alloc(capacity * model.width)?,
        gate: alloc(capacity * model.hidden)?,
        up: alloc(capacity * model.hidden)?,
        qkv: alloc(capacity * CHANNELS)?,
        q: alloc(capacity * INNER)?,
        k: alloc(capacity * KV_HEADS * HEAD_DIM)?,
        v: alloc(capacity * KV_HEADS * HEAD_DIM)?,
        attention_gate: alloc(capacity * INNER)?,
        attended: alloc(capacity * INNER)?,
        alpha: alloc(capacity * GROUPS)?,
        beta: alloc(capacity * GROUPS)?,
        scores: alloc(scores)?,
        logits: alloc(vocab)?,
        ids: stream.alloc_zeros(capacity).map_err(failure)?,
    };
    #[cfg(all(test, feature = "cuda-fp16-prefill-research"))]
    let fp16_probe = projection_probe
        .then(|| -> Result<_> {
            use crate::cuda::fp16_gemm;
            Ok((fp16_gemm::compile(&stream)?, fp16_gemm::Gemm::new(&stream)?))
        })
        .transpose()?;
    let graph = if decode_graph {
        Some(DecodeGraph {
            control: stream.alloc_zeros(2).map_err(failure)?,
            graphs: Default::default(),
        })
    } else {
        None
    };
    let device_greedy = if cfg!(feature = "cuda-device-greedy-research") {
        Some((
            crate::cuda::decode::DeviceGreedy::new(&stream, vocab)?,
            stream.alloc_zeros(1).map_err(failure)?,
        ))
    } else {
        None
    };
    Ok(Hybrid {
        graph,
        device_greedy,
        #[cfg(feature = "cuda-qwen35-recurrent-gdn-512-research")]
        gdn,
        #[cfg(feature = "cuda-fp16-corrected-prefill-research")]
        fp16,
        embedding: upload(model.embedding)?,
        output_norm: upload(model.output_norm)?,
        stream,
        functions,
        layers,
        work,
        info: ModelInfo {
            backend: "rust-cuda".into(),
            ..model.info
        },
        tokenizer: model.tokenizer,
        eog: model.eog,
        epsilon: model.epsilon,
        base: model.base,
        logits: vec![0.0; vocab],
        state: State::default(),
        capacity,
        #[cfg(test)]
        projection_probe,
        #[cfg(all(test, feature = "cuda-fp16-prefill-research"))]
        fp16_probe,
    })
}

#[cfg(test)]
fn probe_projection(
    stream: &Arc<CudaStream>,
    functions: &HashMap<&'static str, CudaFunction>,
    tensor: &DeviceTensor,
    input: &CudaSlice<f32>,
    tokens: i32,
    stages: (&str, &str),
    current: &str,
    sibling: Option<&DeviceTensor>,
    #[cfg(feature = "cuda-fp16-prefill-research")] fp16: Option<&(
        CudaFunction,
        crate::cuda::fp16_gemm::Gemm,
    )>,
) -> Result<()> {
    let export = std::env::var_os("LI_REPLAY_EXPORT_HYBRID_GATE_UP");
    let export_stem = if let Some(dir) = export {
        use std::sync::atomic::{AtomicUsize, Ordering};
        static GATE: AtomicUsize = AtomicUsize::new(0);
        static UP: AtomicUsize = AtomicUsize::new(0);
        let counter = match (tensor.kind, tensor.width, tensor.rows, tokens, stages.1) {
            (12, 1024, 3584, 512, "gate") => &GATE,
            (12, 1024, 3584, 512, "up") => &UP,
            _ => return Ok(()),
        };
        let layer = counter.fetch_add(1, Ordering::Relaxed);
        if !matches!(layer, 0 | 12 | 23) {
            return Ok(());
        }
        Some(std::path::PathBuf::from(dir).join(format!("hybrid-layer{layer}-{}", stages.1)))
    } else {
        None
    };
    let modes = [("prefill_matmul", 1, 4), ("prefill_matmul_ref2x16", 2, 16)];
    let elements = tensor.rows as usize * tokens as usize;
    let mut buffers = [
        stream.alloc_zeros::<f32>(elements).map_err(failure)?,
        stream.alloc_zeros::<f32>(elements).map_err(failure)?,
    ];
    let mut samples: [Vec<f32>; 2] = Default::default();
    for (round, order) in [[0, 1], [1, 0]].into_iter().enumerate() {
        for i in order {
            let (kernel, rows, tile) = modes[i];
            let launch = |output: &mut CudaSlice<f32>| {
                let mut args = stream.launch_builder(&functions[kernel]);
                args.arg(&tensor.bytes)
                    .arg(&tensor.kind)
                    .arg(input)
                    .arg(output)
                    .arg(&tensor.width)
                    .arg(&tensor.rows)
                    .arg(&tokens);
                unsafe {
                    args.launch(LaunchConfig {
                        grid_dim: (
                            (tensor.rows as u32).div_ceil(rows),
                            (tokens as u32).div_ceil(tile),
                            1,
                        ),
                        block_dim: (256, 1, 1),
                        shared_mem_bytes: 0,
                    })
                }
                .map_err(failure)?;
                Ok::<_, crate::Error>(())
            };
            launch(&mut buffers[i])?;
            if let Some(stem) = export_stem.as_ref() {
                let weights = stream.clone_dtoh(&tensor.bytes).map_err(failure)?;
                let input = stream.clone_dtoh(input).map_err(failure)?;
                let output = stream.clone_dtoh(&buffers[i]).map_err(failure)?;
                let count = tensor.width as usize * tokens as usize;
                let packed = tensor.width as usize / 256 * tensor.rows as usize * 144;
                if weights.len() < packed || input.len() < count || output.len() != elements {
                    return Err(invalid("unexpected hybrid gate/up replay operand size"));
                }
                let write = |suffix: &str, data: &[u8]| {
                    let mut path = stem.as_os_str().to_os_string();
                    path.push(suffix);
                    std::fs::write(path, data)
                };
                write(".packed", &weights[..packed])?;
                for (suffix, values) in [("-input.f32", &input[..count]), ("-output.f32", &output)] {
                    let bytes: Vec<_> = values.iter().flat_map(|x| x.to_le_bytes()).collect();
                    write(suffix, &bytes)?;
                }
                return Ok(());
            }
            for _ in 0..3 {
                let flag = Some(sys::CUevent_flags::CU_EVENT_DEFAULT);
                let start = stream.record_event(flag).map_err(failure)?;
                launch(&mut buffers[i])?;
                let end = stream.record_event(flag).map_err(failure)?;
                samples[i].push(start.elapsed_ms(&end).map_err(failure)?);
            }
        }
        if round == 0 {
            let control = stream.clone_dtoh(&buffers[0]).map_err(failure)?;
            let candidate = stream.clone_dtoh(&buffers[1]).map_err(failure)?;
            if control
                .iter()
                .zip(&candidate)
                .any(|(a, b)| !a.is_finite() || a.to_bits() != b.to_bits())
            {
                return Err(failure(format!(
                    "Qwen3.5 projection bits differ: {stages:?}, kind {}, {}x{}",
                    tensor.kind, tensor.width, tensor.rows
                )));
            }
        }
    }
    println!(
        "HYBRID_REPLAY {}",
        serde_json::json!({
            "input":stages.0, "output":stages.1, "tokens":tokens, "kind":tensor.kind,
            "width":tensor.width, "rows":tensor.rows, "current":current,
            "legacy_ms":samples[0], "tile_ms":samples[1], "outputs":elements
        })
    );
    if matches!(tensor.kind, 12..=14) && tensor.width % 256 == 0 {
        let expected = stream.clone_dtoh(&buffers[1]).map_err(failure)?;
        crate::cuda::model_replay::benchmark_dense_packed_panel(
            stream, functions, tensor, input, tokens, &expected, stages.1,
        )?;
        #[cfg(feature = "cuda-fp16-prefill-research")]
        if let Some((expand, gemm)) = fp16 {
            crate::cuda::model_replay::benchmark_dense_fp16_panel(
                stream, expand, gemm, tensor, input, tokens, &expected, stages.1,
            )?;
        }
    }
    if matches!(
        (tensor.kind, tensor.width, tensor.rows),
        (13, 1024, 6144) | (13, 2048, 1024)
    ) {
        let expected = stream.clone_dtoh(&buffers[1]).map_err(failure)?;
        let mut partial = stream.alloc_zeros::<f32>(elements * 8).map_err(failure)?;
        let mut output = stream.alloc_zeros::<f32>(elements).map_err(failure)?;
        let mut times: [Vec<f32>; 5] = Default::default();
        let mut errors = [0.0f32; 5];
        let mut expected_split = Vec::new();
        let count = elements as i32;
        for order in [[0, 3, 1, 4, 2], [2, 4, 1, 3, 0]] {
            for i in order {
                let splits = [1i32, 4, 4, 2, 8][i];
                let launch = |target: &mut CudaSlice<f32>| {
                    let name = if i == 2 {
                        "qwen35_q5_tf32x3_warp_split"
                    } else {
                        "qwen35_q5_tf32x3_split"
                    };
                    let mut args = stream.launch_builder(&functions[name]);
                    args.arg(&tensor.bytes)
                        .arg(input)
                        .arg(target)
                        .arg(&tensor.width)
                        .arg(&tensor.rows)
                        .arg(&tokens);
                    unsafe {
                        args.launch(LaunchConfig {
                            grid_dim: (
                                (tensor.rows as u32).div_ceil(128),
                                (tokens as u32).div_ceil(8),
                                splits as u32,
                            ),
                            block_dim: (256, 1, 1),
                            shared_mem_bytes: 0,
                        })
                    }
                    .map_err(failure)?;
                    Ok::<_, crate::Error>(())
                };
                for sample in 0..10 {
                    let flag = Some(sys::CUevent_flags::CU_EVENT_DEFAULT);
                    let start = stream.record_event(flag).map_err(failure)?;
                    if splits == 1 {
                        launch(&mut output)?;
                    } else {
                        launch(&mut partial)?;
                        let mut reduce =
                            stream.launch_builder(&functions["qwen35_q5_split_reduce"]);
                        reduce
                            .arg(&partial)
                            .arg(&mut output)
                            .arg(&count)
                            .arg(&splits);
                        unsafe { reduce.launch(LaunchConfig::for_num_elems(count as u32)) }
                            .map_err(failure)?;
                    }
                    let end = stream.record_event(flag).map_err(failure)?;
                    let elapsed = start.elapsed_ms(&end).map_err(failure)?;
                    if sample >= 3 {
                        times[i].push(elapsed);
                    }
                }
                let actual = stream.clone_dtoh(&output).map_err(failure)?;
                for (&a, &b) in actual.iter().zip(&expected) {
                    if !a.is_finite() {
                        return Err(failure("nonfinite Qwen3.5 tensor-core projection"));
                    }
                    errors[i] = errors[i].max((a - b).abs());
                }
                if i == 1 && expected_split.is_empty() {
                    expected_split = actual;
                } else if i == 2
                    && actual
                        .iter()
                        .zip(&expected_split)
                        .any(|(a, b)| a.to_bits() != b.to_bits())
                {
                    return Err(failure(
                        "Qwen3.5 cooperative Q5_K decode changed Split-K bits",
                    ));
                }
            }
        }
        println!(
            "HYBRID_SPLITK {}",
            serde_json::json!({
                "tokens":tokens, "width":tensor.width, "rows":tensor.rows,
                "f32_tile_ms":samples[1], "tf32x3_ms":times[0], "split4_ms":times[1],
                "warp_split4_ms":times[2], "split2_ms":times[3], "split8_ms":times[4],
                "max_abs_errors":errors, "max_split_scratch_bytes":elements * 8 * size_of::<f32>()
            })
        );
        if let Some(z) = sibling.filter(|z| {
            (tensor.width, tensor.rows, z.kind, z.width, z.rows) == (1024, 6144, 12, 1024, 2048)
        }) {
            let splits = 4i32;
            let mut packed = stream
                .alloc_zeros::<f32>(2 * tokens as usize * tensor.width as usize)
                .map_err(failure)?;
            let mut z_output = stream
                .alloc_zeros::<f32>(z.rows as usize * tokens as usize)
                .map_err(failure)?;
            let mut times: [Vec<f32>; 3] = Default::default();
            let mut expected_qkv = Vec::new();
            let mut baseline_z = Vec::new();
            let mut expected_z = Vec::new();
            let mut max_z_error = 0.0f32;
            for order in [[0, 1, 2], [2, 1, 0]] {
                for i in order {
                    for sample in 0..10 {
                        let flag = Some(sys::CUevent_flags::CU_EVENT_DEFAULT);
                        let start = stream.record_event(flag).map_err(failure)?;
                        let kernel_input = if i == 2 {
                            let mut pack = stream.launch_builder(&functions["qwen35_pack_tf32x3"]);
                            pack.arg(input)
                                .arg(&mut packed)
                                .arg(&tensor.width)
                                .arg(&tokens);
                            unsafe {
                                pack.launch(LaunchConfig {
                                    grid_dim: (tensor.width as u32 / 32, tokens as u32 / 8, 1),
                                    block_dim: (256, 1, 1),
                                    shared_mem_bytes: 0,
                                })
                            }
                            .map_err(failure)?;
                            &packed
                        } else {
                            input
                        };
                        let kernel = if i == 2 {
                            "qwen35_q5_tf32x3_packed_split"
                        } else {
                            "qwen35_q5_tf32x3_split"
                        };
                        let mut args = stream.launch_builder(&functions[kernel]);
                        args.arg(&tensor.bytes)
                            .arg(kernel_input)
                            .arg(&mut partial)
                            .arg(&tensor.width)
                            .arg(&tensor.rows)
                            .arg(&tokens);
                        unsafe {
                            args.launch(LaunchConfig {
                                grid_dim: (
                                    tensor.rows as u32 / 128,
                                    tokens as u32 / 8,
                                    splits as u32,
                                ),
                                block_dim: (256, 1, 1),
                                shared_mem_bytes: 0,
                            })
                        }
                        .map_err(failure)?;
                        let mut reduce =
                            stream.launch_builder(&functions["qwen35_q5_split_reduce"]);
                        reduce
                            .arg(&partial)
                            .arg(&mut output)
                            .arg(&count)
                            .arg(&splits);
                        unsafe { reduce.launch(LaunchConfig::for_num_elems(count as u32)) }
                            .map_err(failure)?;
                        let (kernel, rows, tile) = [
                            ("prefill_matmul_ref2x16", 2, 16),
                            ("prefill_matmul_tf32x3_tiled", 128, 8),
                            ("qwen35_q4_tf32x3_packed", 128, 8),
                        ][i];
                        let mut args = stream.launch_builder(&functions[kernel]);
                        args.arg(&z.bytes);
                        if i != 2 {
                            args.arg(&z.kind);
                        }
                        args.arg(kernel_input)
                            .arg(&mut z_output)
                            .arg(&z.width)
                            .arg(&z.rows)
                            .arg(&tokens);
                        unsafe {
                            args.launch(LaunchConfig {
                                grid_dim: (z.rows as u32 / rows, tokens as u32 / tile, 1),
                                block_dim: (256, 1, 1),
                                shared_mem_bytes: 0,
                            })
                        }
                        .map_err(failure)?;
                        let end = stream.record_event(flag).map_err(failure)?;
                        let elapsed = start.elapsed_ms(&end).map_err(failure)?;
                        if sample >= 3 {
                            times[i].push(elapsed);
                        }
                    }
                    let actual = stream.clone_dtoh(&output).map_err(failure)?;
                    if expected_qkv.is_empty() {
                        expected_qkv = actual;
                    } else if actual
                        .iter()
                        .zip(&expected_qkv)
                        .any(|(a, b)| !a.is_finite() || a.to_bits() != b.to_bits())
                    {
                        return Err(failure("Qwen3.5 packed TF32x3 changed Split-K bits"));
                    }
                    let actual_z = stream.clone_dtoh(&z_output).map_err(failure)?;
                    if i == 0 && baseline_z.is_empty() {
                        baseline_z = actual_z;
                    } else if i == 1 && expected_z.is_empty() {
                        for (&a, &b) in actual_z.iter().zip(&baseline_z) {
                            if !a.is_finite() || !b.is_finite() {
                                return Err(failure("Nonfinite Qwen3.5 Z projection"));
                            }
                            max_z_error = max_z_error.max((a - b).abs());
                        }
                        expected_z = actual_z;
                    } else if i == 2
                        && actual_z
                            .iter()
                            .zip(&expected_z)
                            .any(|(a, b)| a.to_bits() != b.to_bits())
                    {
                        return Err(failure("Qwen3.5 shared TF32x3 pack changed Z bits"));
                    }
                }
            }
            println!(
                "HYBRID_PREPACK_SHARED {}",
                serde_json::json!({
                    "tokens":tokens, "outputs":elements,
                    "split_plus_z_tile_ms":times[0], "unpacked_tf32x3_ms":times[1],
                    "shared_pack_tf32x3_ms":times[2], "max_z_error_vs_tile":max_z_error,
                    "packed_bytes":packed.len() * size_of::<f32>()
                })
            );
        }
    }
    Ok(())
}

#[test]
#[ignore = "requires QWEN35_GGUF and QWEN35_PROBE_IDS; serial CUDA"]
fn benchmark_qwen35_projection_probe() -> Result<()> {
    let model = std::env::var("QWEN35_GGUF").map_err(failure)?;
    let input = std::env::var("QWEN35_PROBE_IDS").map_err(failure)?;
    let ids: Vec<i32> = serde_json::from_slice(&std::fs::read(input)?)?;
    if !matches!(ids.len(), 256 | 512) {
        return Err(invalid("Qwen3.5 probe expects a 256- or 512-token chunk"));
    }
    let config = Config {
        context_tokens: 1024,
        batch_tokens: 512,
        prefill_chunk: 512,
        threads: 4,
    };
    let mut backend = open_hybrid(Path::new(&model), &config, CudaOptions::default(), true)?;
    backend.evaluate(&ids, 0)
}

macro_rules! project {
    (@sibling) => { None };
    (@sibling $sibling:expr) => { Some($sibling) };
    ($model:ident, $t:expr, $x:ident, $y:ident, $n:ident $(, $sibling:expr)?) => {{
        let t = $t;
        if $n == 1 {
            project!($model, t, $x, $y);
        } else {
            // Only dispatch the wider tile for measured packed shapes and batch sizes.
            let (kernel, rows, tokens) = match (t.kind, t.width, t.rows) {
                (12, 1024, 2048 | 3584 | 4096) | (12, 2048 | 3584, 1024) | (14, 3584, 1024)
                    if $n >= 128 =>
                {
                    ("prefill_matmul_ref2x16", 2, 16)
                }
                (13, 1024, 6144) | (13, 2048, 1024)
                    if cfg!(feature = "cuda-qwen35-q5-prefill-research") && $n >= 256 =>
                {
                    ("prefill_matmul_ref2x16", 2, 16)
                }
                _ => ("prefill_matmul", 1, 4),
            };
            #[cfg(test)]
            if $model.projection_probe && matches!(t.kind, 8 | 12 | 13 | 14) {
                probe_projection(
                    &$model.stream,
                    &$model.functions,
                    t,
                    &$model.work.$x,
                    $n,
                    (stringify!($x), stringify!($y)),
                    kernel,
                    project!(@sibling $($sibling)?),
                    #[cfg(feature = "cuda-fp16-prefill-research")]
                    $model.fp16_probe.as_ref(),
                )?;
            }
            let splits = match (t.kind, t.width, t.rows) {
                (13, 1024, 6144) => 4i32,
                (13, 2048, 1024) if cfg!(feature = "cuda-qwen35-down-splitk-research") => 8,
                _ => 0,
            };
            #[cfg(feature = "cuda-fp16-corrected-prefill-research")]
            let fp16 = if let Some(fp16) = $model.fp16.as_mut() {
                fp16.project(t, &$model.work.$x, &mut $model.work.$y, $n)?
            } else {
                false
            };
            #[cfg(not(feature = "cuda-fp16-corrected-prefill-research"))]
            let fp16 = false;
            if fp16 {
            } else if let Some(packed) = $model.work.packed.as_mut()
                .filter(|_| $n >= 32 && matches!(t.kind,12..=14) && t.width % 256 == 0)
            {
                let count = t.width * $n;
                launch!($model, "residual_quantize32_precise3", ((count as u32).div_ceil(256), 1, 1),
                    &$model.work.$x, &mut packed.values, &mut packed.scales, &mut packed.sums,
                    &count, &mut packed.status);
                launch!($model, "dense_packed_replay_mma3", ((t.rows as u32).div_ceil(32), ($n as u32).div_ceil(32), 1); 128,
                    &t.bytes, &t.kind, &packed.values, &packed.scales, &packed.sums,
                    &mut $model.work.$y, &t.width, &t.rows, &$n, &mut packed.status);
            } else if let Some(partials) = $model
                .work
                .splitk
                .as_mut()
                .filter(|_| $n >= 256 && splits != 0)
            {
                let count = t.rows * $n;
                launch!(
                    $model,
                    "qwen35_q5_tf32x3_split",
                    ((t.rows as u32).div_ceil(128), ($n as u32).div_ceil(8), splits as u32),
                    &t.bytes,
                    &$model.work.$x,
                    &mut *partials,
                    &t.width,
                    &t.rows,
                    &$n
                );
                launch!(
                    $model,
                    "qwen35_q5_split_reduce",
                    ((count as u32).div_ceil(256), 1, 1),
                    &*partials,
                    &mut $model.work.$y,
                    &count,
                    &splits
                );
            } else {
                launch!(
                    $model,
                    kernel,
                    (
                        (t.rows as u32).div_ceil(rows),
                        ($n as u32).div_ceil(tokens),
                        1
                    ),
                    &t.bytes,
                    &t.kind,
                    &$model.work.$x,
                    &mut $model.work.$y,
                    &t.width,
                    &t.rows,
                    &$n
                );
            }
        }
    }};
    ($model:ident, $t:expr, $x:ident, $y:ident) => {{
        let t = $t;
        #[cfg(feature = "cuda-qwen35-matvec-cache-research")]
        let cached = if let Some(coefficients) = &t.k_coefficients {
            launch!($model, "qwen35_cached_matvec", (t.rows as u32, 1, 1),
                &t.bytes, coefficients, &t.kind, &$model.work.$x, &mut $model.work.$y, &t.width);
            true
        } else { false };
        #[cfg(not(feature = "cuda-qwen35-matvec-cache-research"))]
        let cached = false;
        if !cached { launch!(
            $model,
            "matvec",
            (t.rows as u32, 1, 1),
            &t.bytes,
            &t.kind,
            &$model.work.$x,
            &mut $model.work.$y,
            &t.width,
            &t.rows,
            &$model.work.ids,
            &0i32,
            &0i32
        ); }
    }};
}

impl Hybrid {
    fn replay_decode(&mut self, token: i32, pos: i32, head_dp4a: bool) -> Result<()> {
        let count = pos as u32 + 1;
        // Bound inactive score CTAs without capturing one graph per position.
        let grid = count.next_power_of_two().min(self.info.context_tokens);
        let split = count >= 128 && self.functions.contains_key("qwen35_attention_normalize");
        let key = (grid, split);
        if !self.graph.as_ref().unwrap().graphs.contains_key(&key) {
            self.stream.synchronize().map_err(failure)?;
            self.stream
                .begin_capture(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)
                .map_err(failure)?;
            let submitted = self.chunk(&[token], pos, Some(key))
                .and_then(|()| self.project_logits(0, head_dp4a));
            let captured = self.stream.end_capture(
                sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH,
            );
            submitted?;
            let graph = captured.map_err(failure)?
                .ok_or_else(|| failure("Qwen3.5 decode graph capture was empty"))?;
            self.graph.as_mut().unwrap().graphs.insert(key, graph);
        }
        let graph = self.graph.as_mut().unwrap();
        self.stream.memcpy_htod(&[token, pos], &mut graph.control).map_err(failure)?;
        graph.graphs[&key].launch().map_err(failure)
    }

    fn chunk(&mut self, tokens: &[i32], pos: i32, capture: Option<(u32, bool)>) -> Result<()> {
        let vector = |n: i32| ((n as u32).div_ceil(256), 1, 1);
        let w = self.embedding.width;
        let inner = INNER as i32;
        let kw = (KV_HEADS * HEAD_DIM) as i32;
        let heads = HEADS as i32;
        let kv_heads = KV_HEADS as i32;
        let dim = HEAD_DIM as i32;
        let rotary = ROTARY as i32;
        let context = self.info.context_tokens as i32;
        let n = tokens.len() as i32;
        let control = capture.map(|_| &self.graph.as_ref().unwrap().control);
        if let Some(control) = control {
            launch!(self, "embedding_decode", vector(w),
                &self.embedding.bytes, &self.embedding.kind, &mut self.work.x, &w, control);
        } else if let [token] = tokens {
            launch!(
                self,
                "embedding",
                vector(w),
                &self.embedding.bytes,
                &self.embedding.kind,
                &mut self.work.x,
                &w,
                token
            );
        } else {
            self.stream
                .memcpy_htod(tokens, &mut self.work.ids.slice_mut(..tokens.len()))
                .map_err(failure)?;
            launch!(
                self,
                "prefill_embedding",
                ((w as u32).div_ceil(256), n as u32, 1),
                &self.embedding.bytes,
                &self.embedding.kind,
                &self.work.ids,
                &mut self.work.x,
                &w
            );
        }
        for layer in &mut self.layers {
            launch!(
                self,
                "normalize",
                (n as u32, 1, 1),
                &self.work.x,
                &layer.input_norm.bytes,
                &mut self.work.normalized,
                &w,
                &self.epsilon
            );
            match &mut layer.attention {
                DeviceAttention::Dense {
                    qg,
                    k,
                    v,
                    q_norm,
                    k_norm,
                    output,
                    keys,
                    values,
                } => {
                    project!(self, qg, normalized, qkv, n);
                    project!(self, k, normalized, k, n);
                    project!(self, v, normalized, v, n);
                    launch!(
                        self,
                        "qwen35_split_qg",
                        (vector(inner).0, n as u32, 1),
                        &self.work.qkv,
                        &mut self.work.q,
                        &mut self.work.attention_gate
                    );
                    for (data, weight, length) in [
                        (&mut self.work.q, &*q_norm, inner),
                        (&mut self.work.k, &*k_norm, kw),
                    ] {
                        launch!(
                            self,
                            "head_norm",
                            ((n * length / dim) as u32, 1, 1),
                            &mut *data,
                            &weight.bytes,
                            &dim,
                            &self.epsilon
                        );
                        if let Some(control) = control {
                            launch!(self, "qwen35_rotary_graph", (vector(length / 2).0, 1, 1),
                                data, &length, &dim, control, &self.base, &rotary);
                        } else {
                            launch!(
                            self,
                            "rotary_partial_neox",
                            (vector(length / 2).0, n as u32, 1),
                            data,
                            &length,
                            &dim,
                            &pos,
                            &self.base,
                            &rotary
                            );
                        }
                    }
                    if let Some(control) = control {
                        launch!(self, "cache_kv_decode", vector(kw),
                            &self.work.k, &self.work.v, &mut *keys, &mut *values, &kw, control);
                    } else {
                        launch!(
                        self,
                        "cache_kv",
                        (vector(kw).0, n as u32, 1),
                        &self.work.k,
                        &self.work.v,
                        &mut *keys,
                        &mut *values,
                        &kw,
                        &pos
                        );
                    }
                    // The fused kernel keeps at most 1024 scores per query in shared
                    // memory and masks each query to its own causal prefix.
                    let live_count = pos + 1;
                    let bucket = capture.map_or(live_count as u32, |(grid, _)| grid);
                    if n == 1 && bucket > 512
                        && self.functions.contains_key("qwen35_split_decode_partials")
                    {
                        let tile = 128u32;
                        let tiles = bucket.div_ceil(tile);
                        macro_rules! split_attention {
                            ($position:expr) => {
                                launch!(self, "qwen35_split_decode_partials", (HEADS as u32, tiles, 1),
                                    &self.work.q, &*keys, &*values, &mut self.work.attended,
                                    &mut self.work.scores, &heads, &kv_heads, &tile, &live_count, $position);
                                launch!(self, "qwen35_split_decode_merge", (HEADS as u32, 1, 1),
                                    &self.work.scores, &mut self.work.attended, &tiles, &tile, &live_count, $position);
                            };
                        }
                        if let Some(control) = control {
                            split_attention!(control);
                        } else {
                            split_attention!(&0u64);
                        }
                    } else if let Some((score_grid, split)) = capture {
                        let control = control.unwrap();
                        launch!(self, "qwen35_attention_scores_graph", (HEADS as u32, score_grid, 1),
                            &self.work.q, &*keys, &mut self.work.scores, &dim, &heads, &kv_heads, &context, control);
                        if split {
                            launch!(self, "qwen35_attention_normalize_graph", (HEADS as u32, 1, 1),
                                &mut self.work.scores, &heads, &kv_heads, &context, control);
                            launch!(self, "qwen35_attention_value_columns_graph", (HEADS as u32, 8, 1); 32,
                                &self.work.scores, &*values, &mut self.work.attended, &heads, &kv_heads, &context, control);
                        } else {
                            launch!(self, "qwen35_attention_values_graph", (HEADS as u32, 1, 1),
                                &mut self.work.scores, &*values, &mut self.work.attended, &dim, &heads, &kv_heads, &context, control);
                        }
                    } else if cfg!(feature = "cuda-qwen35-prefill-query-pair-research")
                        && dim == 256
                        && n >= 32
                        && n % 2 == 0
                        && pos + n <= 1024
                    {
                        let first_count = pos + 1;
                        launch!(
                            self,
                            "attention_fused_query_pair_256",
                            (HEADS as u32, n as u32 / 2, 1),
                            &self.work.q,
                            &*keys,
                            &*values,
                            &mut self.work.attended,
                            &heads,
                            &kv_heads,
                            &first_count
                        );
                    } else if n > 1 && pos + n <= 1024 {
                        let first_count = pos + 1;
                        launch!(
                            self,
                            "attention_fused",
                            (HEADS as u32, n as u32, 1),
                            &self.work.q,
                            &*keys,
                            &*values,
                            &mut self.work.attended,
                            &dim,
                            &heads,
                            &kv_heads,
                            &first_count
                        );
                    } else {
                        for t in 0..tokens.len() {
                            let count = pos + t as i32 + 1;
                            let row = t * INNER..(t + 1) * INNER;
                            let q_row = self.work.q.slice(row.clone());
                            let mut out_row = self.work.attended.slice_mut(row);
                            launch!(
                                self,
                                "attention_scores",
                                (HEADS as u32, count as u32, 1),
                                &q_row,
                                &*keys,
                                &mut self.work.scores,
                                &dim,
                                &heads,
                                &kv_heads,
                                &context,
                                &count
                            );
                            if count >= 128
                                && self.functions.contains_key("qwen35_attention_normalize")
                            {
                                launch!(
                                    self,
                                    "qwen35_attention_normalize",
                                    (HEADS as u32, 1, 1),
                                    &mut self.work.scores,
                                    &heads,
                                    &kv_heads,
                                    &context,
                                    &count
                                );
                                launch!(self, "qwen35_attention_value_columns", (HEADS as u32, 8, 1); 32,
                                    &self.work.scores, &*values, &mut out_row, &heads, &kv_heads, &context, &count);
                                continue;
                            }
                            launch!(
                                self,
                                "attention_values",
                                (HEADS as u32, 1, 1),
                                &mut self.work.scores,
                                &*values,
                                &mut out_row,
                                &dim,
                                &heads,
                                &kv_heads,
                                &context,
                                &count
                            );
                        }
                    }
                    launch!(
                        self,
                        "qwen35_gate",
                        (vector(inner).0, n as u32, 1),
                        &mut self.work.attended,
                        &self.work.attention_gate,
                        &0i32
                    );
                    project!(self, output, attended, projected, n);
                }
                DeviceAttention::Delta {
                    qkv,
                    z,
                    alpha,
                    beta,
                    dt,
                    decay,
                    conv,
                    norm,
                    output,
                    history,
                    state,
                } => {
                    project!(self, qkv, normalized, qkv, n, z);
                    project!(self, z, normalized, attention_gate, n);
                    project!(self, alpha, normalized, alpha, n);
                    project!(self, beta, normalized, beta, n);
                    if n >= 32 && cfg!(feature = "cuda-qwen35-conv-tile-research") {
                        let tiles = (n as u32).div_ceil(16);
                        let halo_elements = tiles as usize * 3 * CHANNELS;
                        if halo_elements > self.work.q.len() {
                            return Err(failure("Qwen3.5 convolution halo exceeds query scratch"));
                        }
                        // Dense attention does not use this scratch in a recurrent layer.
                        let grid = (vector(CHANNELS as i32).0, tiles, 1);
                        let mut halo = self.work.q.slice_mut(..halo_elements);
                        launch!(
                            self,
                            "qwen35_conv_save",
                            grid,
                            &self.work.qkv,
                            history,
                            &mut halo,
                            &n
                        );
                        let halo = self.work.q.slice(..halo_elements);
                        launch!(
                            self,
                            "qwen35_conv_tile",
                            grid,
                            &mut self.work.qkv,
                            &halo,
                            &conv.bytes,
                            &n
                        );
                    } else {
                        launch!(
                            self,
                            "qwen35_conv",
                            vector(CHANNELS as i32),
                            &mut self.work.qkv,
                            history,
                            &conv.bytes,
                            &n
                        );
                    }
                    launch!(self, "qwen35_l2", (2 * GROUPS as u32, n as u32, 1); STATE_DIM as u32,
                        &mut self.work.qkv, &self.epsilon);
                    launch!(self, "qwen35_delta_gates", (n as u32, 1, 1); GROUPS as u32,
                        &mut self.work.alpha, &mut self.work.beta, &dt.bytes, &decay.bytes);
                    #[cfg(feature = "cuda-qwen35-recurrent-gdn-512-research")]
                    let gdn_used = if let Some(gdn) = self.gdn.as_mut().filter(|_| n == 512) {
                        gdn.run(&mut self.work, state)?;
                        true
                    } else {
                        false
                    };
                    #[cfg(not(feature = "cuda-qwen35-recurrent-gdn-512-research"))]
                    let gdn_used = false;
                    if gdn_used {
                        // The bridge wrote attended output and restored the persistent F32 layout.
                    } else if n >= 256 && cfg!(feature = "cuda-qwen35-recurrent-warp-research") {
                        let columns =
                            if cfg!(feature = "cuda-qwen35-recurrent-eight-segments-research") {
                                4
                            } else {
                                8
                            };
                        launch!(self, "qwen35_delta_warp", (GROUPS as u32, (STATE_DIM / columns) as u32, 1); 32,
                            &self.work.qkv, &self.work.alpha, &self.work.beta, state, &mut self.work.attended, &n);
                    } else if n >= 32 && cfg!(feature = "cuda-qwen35-recurrent-panel64-research") {
                        launch!(self, "qwen35_delta_panel64", (GROUPS as u32, (STATE_DIM / 64) as u32, 1); 64,
                            &self.work.qkv, &self.work.alpha, &self.work.beta, state, &mut self.work.attended, &n);
                    } else if n >= 16
                        || (n == 1 && cfg!(feature = "cuda-qwen35-decode-panel-research"))
                    {
                        launch!(self, "qwen35_delta_panel", (GROUPS as u32, (STATE_DIM / 32) as u32, 1); 32,
                            &self.work.qkv, &self.work.alpha, &self.work.beta, state, &mut self.work.attended, &n);
                    } else {
                        launch!(self, "qwen35_delta_batch", (GROUPS as u32, 1, 1); STATE_DIM as u32,
                            &self.work.qkv, &self.work.alpha, &self.work.beta, state, &mut self.work.attended, &n);
                    }
                    if cfg!(feature = "cuda-qwen35-recurrent-norm-gate-research") {
                        launch!(self, "qwen35_delta_norm_gate", (GROUPS as u32 * n as u32, 1, 1); STATE_DIM as u32,
                            &mut self.work.attended, &self.work.attention_gate, &norm.bytes, &self.epsilon);
                    } else {
                        launch!(self, "qwen35_delta_norm", (GROUPS as u32 * n as u32, 1, 1); STATE_DIM as u32,
                            &mut self.work.attended, &norm.bytes, &self.epsilon);
                        launch!(
                            self,
                            "qwen35_gate",
                            ((inner as u32).div_ceil(256), n as u32, 1),
                            &mut self.work.attended,
                            &self.work.attention_gate,
                            &1i32
                        );
                    }
                    project!(self, output, attended, projected, n);
                }
            }
            let elements = n * w;
            if cfg!(feature = "cuda-qwen35-residual-norm-research") {
                launch!(self, "residual_normalize", (n as u32, 1, 1); 256,
                    &mut self.work.x, &self.work.projected, &layer.ffn_norm.bytes,
                    &mut self.work.normalized, &w, &self.epsilon);
            } else {
                launch!(
                    self,
                    "residual",
                    vector(elements),
                    &mut self.work.x,
                    &self.work.projected,
                    &elements
                );
                launch!(
                    self,
                    "normalize",
                    (n as u32, 1, 1),
                    &self.work.x,
                    &layer.ffn_norm.bytes,
                    &mut self.work.normalized,
                    &w,
                    &self.epsilon
                );
            }
            #[cfg(feature = "cuda-qwen35-fused-ffn-decode-research")]
            let fused = if n == 1
                && layer.gate.kind == 12
                && layer.up.kind == 12
                && layer.gate.width == layer.up.width
                && layer.gate.rows == layer.up.rows
                && let (Some(gc), Some(uc)) = (&layer.gate.k_coefficients, &layer.up.k_coefficients)
            {
                launch!(
                    self,
                    "qwen35_cached_swiglu",
                    (layer.gate.rows as u32, 1, 1),
                    &layer.gate.bytes,
                    gc,
                    &layer.up.bytes,
                    uc,
                    &self.work.normalized,
                    &mut self.work.gate,
                    &layer.gate.width
                );
                true
            } else {
                false
            };
            #[cfg(not(feature = "cuda-qwen35-fused-ffn-decode-research"))]
            let fused = false;
            if !fused {
                project!(self, &layer.gate, normalized, gate, n);
                project!(self, &layer.up, normalized, up, n);
                let hidden_elements = n * layer.gate.rows;
                launch!(
                    self,
                    "swiglu",
                    vector(hidden_elements),
                    &mut self.work.gate,
                    &self.work.up,
                    &hidden_elements
                );
            }
            project!(self, &layer.down, gate, projected, n);
            launch!(
                self,
                "residual",
                vector(elements),
                &mut self.work.x,
                &self.work.projected,
                &elements
            );
        }
        Ok(())
    }
}

impl Backend for Hybrid {
    fn info(&self) -> &ModelInfo {
        &self.info
    }
    fn tokenize(&self, text: &str) -> Result<Vec<i32>> {
        self.tokenizer.as_ref().map_err(failure)?.tokenize(text)
    }
    fn detokenize(&self, tokens: &[i32]) -> Result<String> {
        self.tokenizer.as_ref().map_err(failure)?.detokenize(tokens)
    }
    fn reset(&mut self) -> Result<()> {
        self.state.failed = true;
        self.stream.synchronize().map_err(failure)?;
        for layer in &mut self.layers {
            if let DeviceAttention::Delta { history, state, .. } = &mut layer.attention {
                self.stream.memset_zeros(history).map_err(failure)?;
                self.stream.memset_zeros(state).map_err(failure)?;
            }
        }
        self.stream.synchronize().map_err(failure)?;
        self.state = State::default();
        if let Some((device, _)) = &mut self.device_greedy {
            device.sampled = None;
        }
        Ok(())
    }
    fn evaluate(&mut self, tokens: &[i32], position: u32) -> Result<()> {
        self.evaluate_inner(tokens, position, true)
    }
    fn evaluate_for_sampling(&mut self, tokens: &[i32], position: u32) -> Result<()> {
        self.evaluate_inner(tokens, position, self.device_greedy.is_none())
    }
    fn logits(&self) -> Result<&[f32]> {
        if self.device_greedy.as_ref().is_some_and(|(device, _)| device.sampled.is_some()) {
            return Err(failure("token-only evaluation; use evaluate to materialize host logits"));
        }
        self.state.logits(&self.logits)
    }
    fn sample_greedy(&mut self) -> Result<i32> {
        self.state.logits(&[])?;
        if let Some(token) = self.device_greedy.as_ref().and_then(|(device, _)| device.sampled) {
            Ok(token)
        } else {
            greedy(self.logits()?)
        }
    }
    fn is_eog(&self, token: i32) -> bool {
        self.eog.contains(&token)
    }
}

impl Hybrid {
    fn evaluate_inner(&mut self, tokens: &[i32], position: u32, host_logits: bool) -> Result<()> {
        self.state.begin(tokens, position, &self.info)?;
        if let Some((device, _)) = &mut self.device_greedy {
            device.sampled = None;
        }
        let head_dp4a = cfg!(feature = "cuda-qwen35-dp4a-head-research")
            && self.functions.contains_key("k_dp4a_head");
        let packed = (tokens.len() >= 32 || head_dp4a) && self.work.packed.is_some();
        if packed {
            self.stream
                .memset_zeros(&mut self.work.packed.as_mut().unwrap().status)
                .map_err(failure)?;
        }
        if let [token] = tokens && position != 0 && self.graph.is_some() {
            self.replay_decode(*token, position as i32, head_dp4a)?;
        } else {
            let mut last = 0;
            for (i, chunk) in tokens.chunks(self.capacity).enumerate() {
                self.chunk(chunk, position as i32 + (i * self.capacity) as i32, None)?;
                last = chunk.len() - 1;
            }
            self.project_logits(last, head_dp4a)?;
        }
        if !host_logits {
            let (device, scratch_status) = self.device_greedy.as_mut().unwrap();
            let status = if packed {
                &mut self.work.packed.as_mut().unwrap().status
            } else {
                self.stream.memset_zeros(scratch_status).map_err(failure)?;
                scratch_status
            };
            device.enqueue(&self.stream, &self.functions, &self.work.logits, status)?;
            device.read(&self.stream, self.info.vocab_tokens)?;
            self.state.finish(tokens.len());
            return Ok(());
        }
        self.stream
            .memcpy_dtoh(&self.work.logits, &mut self.logits)
            .map_err(failure)?;
        self.stream.synchronize().map_err(failure)?;
        if packed {
            let status = self
                .stream
                .clone_dtoh(&self.work.packed.as_ref().unwrap().status)
                .map_err(failure)?[0];
            if status != 0 {
                return Err(failure(format!(
                    "Qwen3.5 packed projection failed ({status}); request poisoned"
                )));
            }
        }
        if self.logits.iter().any(|x| !x.is_finite()) {
            return Err(failure("nonfinite Qwen3.5 logits; request poisoned"));
        }
        self.state.finish(tokens.len());
        Ok(())
    }
    fn project_logits(&mut self, last: usize, head_dp4a: bool) -> Result<()> {
        let w = self.embedding.width;
        let row = self
            .work
            .x
            .slice(last * w as usize..(last + 1) * w as usize);
        launch!(
            self,
            "normalize",
            (1, 1, 1),
            &row,
            &self.output_norm.bytes,
            &mut self.work.normalized,
            &w,
            &self.epsilon
        );
        if head_dp4a {
            let t = &self.embedding;
            let scratch = self
                .work
                .packed
                .as_mut()
                .ok_or_else(|| failure("Qwen3.5 DP4A head workspace missing"))?;
            launch!(
                self,
                "residual_quantize32_precise3",
                ((w as u32).div_ceil(256), 1, 1),
                &self.work.normalized,
                &mut scratch.values,
                &mut scratch.scales,
                &mut scratch.sums,
                &w,
                &mut scratch.status
            );
            launch!(self, "k_dp4a_head", ((t.rows as u32).div_ceil(4), 1, 1); 128,
                &t.bytes, &t.kind, &scratch.values, &scratch.scales,
                &mut self.work.logits, &w, &t.rows);
        } else {
            project!(self, &self.embedding, normalized, logits);
        }
        Ok(())
    }
}
