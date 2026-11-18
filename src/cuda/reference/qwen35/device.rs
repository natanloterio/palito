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
}

struct Work {
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

pub(in crate::cuda) fn open(
    path: &Path,
    config: &Config,
    options: CudaOptions,
) -> Result<Box<dyn Backend>> {
    config.validate()?;
    if options.matmul != CudaMatmul::F32 || options.decode_graph {
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
    let floats = 6 * 2 * kv
        + 18 * (history + state)
        + capacity
            * (3 * model.width
                + 2 * model.hidden
                + CHANNELS
                + 3 * INNER
                + 2 * KV_HEADS * HEAD_DIM
                + 2 * GROUPS)
        + HEADS * context
        + vocab;
    let required = weight_bytes + 4 * (floats as u64 + capacity as u64);
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
    let ptx = cache::compile(
        options.kernel_cache_dir.as_deref(),
        concat!(
            "#define CUDA_PACKED_DECODE 1\n",
            include_str!("../../kernels.cu"),
            "\n#define CUDA_REGISTER_PREFILL 1\n",
            include_str!("../../prefill.cu"),
            "\n#define CUDA_NORMALIZED_ATTENTION 1\n",
            include_str!("../../attention.cu"),
            "\n#define QWEN35_HYBRID\n",
            include_str!("../../qwen35_delta.cu")
        ),
        CompileOptions {
            fmad: Some(false),
            options: vec![format!("--gpu-architecture=compute_{major}{minor}")],
            ..Default::default()
        },
    )?;
    let module = cuda.load_module(ptx).map_err(failure)?;
    let mut functions = HashMap::new();
    for name in [
        "embedding",
        "prefill_embedding",
        "prefill_matmul",
        "prefill_matmul_ref2x16",
        "matvec",
        "matvec_packed_f32",
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
    let stream = cuda.default_stream();
    let alloc = |n| stream.alloc_zeros::<f32>(n).map_err(failure);
    let upload = |tensor| DeviceTensor::upload(tensor, &stream);
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
                qg: upload(t.qg)?,
                k: upload(t.k)?,
                v: upload(t.v)?,
                q_norm: upload(t.q_norm)?,
                k_norm: upload(t.k_norm)?,
                output: upload(t.output)?,
                keys: alloc(kv)?,
                values: alloc(kv)?,
            },
            Attention::Delta(t) => DeviceAttention::Delta {
                qkv: upload(t.qkv)?,
                z: upload(t.z)?,
                alpha: upload(t.alpha)?,
                beta: upload(t.beta)?,
                dt: upload(t.dt)?,
                decay: upload(t.decay)?,
                conv: plain(t.conv)?,
                norm: upload(t.norm)?,
                output: upload(t.output)?,
                history: alloc(history)?,
                state: alloc(state)?,
            },
        };
        layers.push(DeviceLayer {
            attention,
            input_norm: upload(layer.input_norm)?,
            ffn_norm: upload(layer.ffn_norm)?,
            gate: upload(layer.gate)?,
            up: upload(layer.up)?,
            down: upload(layer.down)?,
        });
    }
    let work = Work {
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
        scores: alloc(HEADS * context)?,
        logits: alloc(vocab)?,
        ids: stream.alloc_zeros(capacity).map_err(failure)?,
    };
    Ok(Box::new(Hybrid {
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
    }))
}

macro_rules! project {
    ($model:ident, $t:expr, $x:ident, $y:ident, $n:ident) => {{
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
                _ => ("prefill_matmul", 1, 4),
            };
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
    }};
    ($model:ident, $t:expr, $x:ident, $y:ident) => {{
        let t = $t;
        let kernel = if matches!(t.kind, 12 | 14) && t.width % 256 == 0 {
            "matvec_packed_f32"
        } else {
            "matvec"
        };
        launch!(
            $model,
            kernel,
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
        );
    }};
}

impl Hybrid {
    fn chunk(&mut self, tokens: &[i32], pos: i32) -> Result<()> {
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
        if let [token] = tokens {
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
                    // The fused kernel keeps at most 1024 scores per query in shared
                    // memory and masks each query to its own causal prefix.
                    if n > 1 && pos + n <= 1024 {
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
                    project!(self, qkv, normalized, qkv, n);
                    project!(self, z, normalized, attention_gate, n);
                    project!(self, alpha, normalized, alpha, n);
                    project!(self, beta, normalized, beta, n);
                    launch!(
                        self,
                        "qwen35_conv",
                        vector(CHANNELS as i32),
                        &mut self.work.qkv,
                        history,
                        &conv.bytes,
                        &n
                    );
                    launch!(self, "qwen35_l2", (2 * GROUPS as u32, n as u32, 1); STATE_DIM as u32,
                        &mut self.work.qkv, &self.epsilon);
                    launch!(self, "qwen35_delta_gates", (n as u32, 1, 1); GROUPS as u32,
                        &mut self.work.alpha, &mut self.work.beta, &dt.bytes, &decay.bytes);
                    if n >= 16 {
                        launch!(self, "qwen35_delta_panel", (GROUPS as u32, (STATE_DIM / 32) as u32, 1); 32,
                            &self.work.qkv, &self.work.alpha, &self.work.beta, state, &mut self.work.attended, &n);
                    } else {
                        launch!(self, "qwen35_delta_batch", (GROUPS as u32, 1, 1); STATE_DIM as u32,
                            &self.work.qkv, &self.work.alpha, &self.work.beta, state, &mut self.work.attended, &n);
                    }
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
                    project!(self, output, attended, projected, n);
                }
            }
            let elements = n * w;
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
        Ok(())
    }
    fn evaluate(&mut self, tokens: &[i32], position: u32) -> Result<()> {
        self.state.begin(tokens, position, &self.info)?;
        let mut last = 0;
        for (i, chunk) in tokens.chunks(self.capacity).enumerate() {
            self.chunk(chunk, position as i32 + (i * self.capacity) as i32)?;
            last = chunk.len() - 1;
        }
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
        project!(self, &self.embedding, normalized, logits);
        self.stream
            .memcpy_dtoh(&self.work.logits, &mut self.logits)
            .map_err(failure)?;
        self.stream.synchronize().map_err(failure)?;
        if self.logits.iter().any(|x| !x.is_finite()) {
            return Err(failure("nonfinite Qwen3.5 logits; request poisoned"));
        }
        self.state.finish(tokens.len());
        Ok(())
    }
    fn logits(&self) -> Result<&[f32]> {
        self.state.logits(&self.logits)
    }
    fn sample_greedy(&mut self) -> Result<i32> {
        greedy(self.logits()?)
    }
    fn is_eog(&self, token: i32) -> bool {
        self.eog.contains(&token)
    }
}
