use crate::{
    Config, Result,
    backend::{Capabilities, ModelInfo},
    cpu::{attend_with_divisor, norm, rotary},
    gguf::{Gguf, Tensor, Value},
    tokenizer::Tokenizer,
    transformer::invalid,
};
pub(in crate::cuda) mod device;

struct Dense {
    qg: Tensor,
    k: Tensor,
    v: Tensor,
    q_norm: Tensor,
    k_norm: Tensor,
    output: Tensor,
    keys: Vec<f32>,
    values: Vec<f32>,
}

struct Delta {
    qkv: Tensor,
    z: Tensor,
    beta: Tensor,
    alpha: Tensor,
    dt: Tensor,
    decay: Tensor,
    conv: Tensor,
    norm: Tensor,
    output: Tensor,
    history: Vec<f32>,
    state: Vec<f32>,
}

enum Attention {
    Dense(Dense),
    Delta(Delta),
}

struct Layer {
    attention: Attention,
    input_norm: Tensor,
    ffn_norm: Tensor,
    gate: Tensor,
    up: Tensor,
    down: Tensor,
}

pub(super) struct Qwen35 {
    pub info: ModelInfo,
    pub tokenizer: Result<Tokenizer>,
    pub eog: Vec<i32>,
    width: usize,
    hidden: usize,
    epsilon: f32,
    base: f32,
    embedding: Tensor,
    output_norm: Tensor,
    layers: Vec<Layer>,
}

// Only the measured 0.8B text layout is admitted. Other schedules and
// multimodal position streams require separate validation.
const HEADS: usize = 8;
const KV_HEADS: usize = 2;
const HEAD_DIM: usize = 256;
const ROTARY: usize = 64;
const GROUPS: usize = 16;
const STATE_DIM: usize = 128;
const INNER: usize = GROUPS * STATE_DIM;
const CHANNELS: usize = 3 * INNER;
const KERNEL: usize = 4;

impl Qwen35 {
    pub fn load(mut gguf: Gguf, config: &Config, host_state: bool) -> Result<Self> {
        let integer = |key| gguf.integer(&format!("qwen35.{key}"));
        for (key, expected) in [
            ("block_count", 24),
            ("embedding_length", 1024),
            ("feed_forward_length", 3584),
            ("full_attention_interval", 4),
            ("attention.head_count", HEADS),
            ("attention.head_count_kv", KV_HEADS),
            ("attention.key_length", HEAD_DIM),
            ("attention.value_length", HEAD_DIM),
            ("rope.dimension_count", ROTARY),
            ("ssm.conv_kernel", KERNEL),
            ("ssm.state_size", STATE_DIM),
            ("ssm.group_count", GROUPS),
            ("ssm.time_step_rank", GROUPS),
            ("ssm.inner_size", INNER),
        ] {
            if integer(key)? != expected {
                return Err(invalid(format!("unsupported qwen35 {key}")));
            }
        }
        if !matches!(
            gguf.metadata.get("qwen35.rope.dimension_sections"),
            Some(Value::Integers(values)) if values == &[11, 11, 10, 0]
        ) || !matches!(
            gguf.metadata.get("qwen35.attention.causal"),
            None | Some(Value::Bool(true))
        ) || [
            "attention.recurrent_layers",
            "attention.sliding_window",
            "attention.scale",
            "rope.scaling.type",
            "rope.scaling.factor",
            "rope.scaling.attn_factor",
            "rope.scale_linear",
        ]
        .iter()
        .any(|key| gguf.metadata.contains_key(&format!("qwen35.{key}")))
        {
            return Err(invalid(
                "unsupported qwen35 schedule or attention/rotary settings",
            ));
        }
        let context = config.context_tokens as usize;
        if context == 0 || context > integer("context_length")? {
            return Err(invalid("invalid qwen35 context"));
        }
        let width = integer("embedding_length")?;
        let hidden = integer("feed_forward_length")?;
        let count = integer("block_count")?;
        let epsilon = gguf.float("qwen35.attention.layer_norm_rms_epsilon")?;
        let base = gguf.float("qwen35.rope.freq_base")?;
        if !epsilon.is_finite() || epsilon <= 0.0 || !base.is_finite() || base <= 0.0 {
            return Err(invalid("invalid qwen35 epsilon or rotary base"));
        }
        let vocab = match gguf.metadata.get("tokenizer.ggml.tokens") {
            Some(Value::Strings(tokens))
                if !tokens.is_empty() && tokens.len() <= i32::MAX as usize =>
            {
                tokens.len()
            }
            _ => return Err(invalid("invalid qwen35 vocabulary")),
        };
        let eos = gguf.integer("tokenizer.ggml.eos_token_id")?;
        if eos >= vocab {
            return Err(invalid("qwen35 EOS exceeds vocabulary"));
        }
        let mut eog = vec![eos as i32];
        if let (Some(Value::Strings(tokens)), Some(Value::Integers(types))) = (
            gguf.metadata.get("tokenizer.ggml.tokens"),
            gguf.metadata.get("tokenizer.ggml.token_type"),
        ) {
            for (id, (text, kind)) in tokens.iter().zip(types).enumerate() {
                if id != eos
                    && *kind == 3
                    && matches!(
                        text.as_str(),
                        "<|endoftext|>"
                            | "<|im_end|>"
                            | "<|fim_pad|>"
                            | "<|repo_name|>"
                            | "<|file_sep|>"
                    )
                {
                    eog.push(id as i32);
                }
            }
        }
        let tokenizer = Tokenizer::from_metadata(&gguf.metadata);
        let embedding = gguf.tensor("token_embd.weight", &[width, vocab])?;
        let output_norm = gguf.tensor("output_norm.weight", &[width])?;
        let cache_len = context
            .checked_mul(KV_HEADS * HEAD_DIM)
            .ok_or_else(|| invalid("qwen35 KV capacity overflow"))?;
        let mut layers = Vec::with_capacity(count);
        for i in 0..count {
            let mut tensor = |name, shape: &[usize]| gguf.tensor(&format!("blk.{i}.{name}"), shape);
            let attention = if (i + 1) % 4 == 0 {
                Attention::Dense(Dense {
                    qg: tensor("attn_q.weight", &[width, 2 * HEADS * HEAD_DIM])?,
                    k: tensor("attn_k.weight", &[width, KV_HEADS * HEAD_DIM])?,
                    v: tensor("attn_v.weight", &[width, KV_HEADS * HEAD_DIM])?,
                    q_norm: tensor("attn_q_norm.weight", &[HEAD_DIM])?,
                    k_norm: tensor("attn_k_norm.weight", &[HEAD_DIM])?,
                    output: tensor("attn_output.weight", &[HEADS * HEAD_DIM, width])?,
                    keys: vec![0.0; if host_state { cache_len } else { 0 }],
                    values: vec![0.0; if host_state { cache_len } else { 0 }],
                })
            } else {
                Attention::Delta(Delta {
                    qkv: tensor("attn_qkv.weight", &[width, CHANNELS])?,
                    z: tensor("attn_gate.weight", &[width, INNER])?,
                    beta: tensor("ssm_beta.weight", &[width, GROUPS])?,
                    alpha: tensor("ssm_alpha.weight", &[width, GROUPS])?,
                    dt: tensor("ssm_dt.bias", &[GROUPS])?,
                    decay: tensor("ssm_a", &[GROUPS])?,
                    conv: tensor("ssm_conv1d.weight", &[KERNEL, CHANNELS])?,
                    norm: tensor("ssm_norm.weight", &[STATE_DIM])?,
                    output: tensor("ssm_out.weight", &[INNER, width])?,
                    history: vec![
                        0.0;
                        if host_state {
                            CHANNELS * (KERNEL - 1)
                        } else {
                            0
                        }
                    ],
                    state: vec![
                        0.0;
                        if host_state {
                            GROUPS * STATE_DIM * STATE_DIM
                        } else {
                            0
                        }
                    ],
                })
            };
            layers.push(Layer {
                attention,
                input_norm: tensor("attn_norm.weight", &[width])?,
                ffn_norm: tensor("post_attention_norm.weight", &[width])?,
                gate: tensor("ffn_gate.weight", &[width, hidden])?,
                up: tensor("ffn_up.weight", &[width, hidden])?,
                down: tensor("ffn_down.weight", &[hidden, width])?,
            });
        }
        if !gguf.tensors.is_empty() {
            return Err(invalid("unsupported tensors remain in qwen35 text model"));
        }
        Ok(Self {
            info: ModelInfo {
                backend: "rust-cpu-transformer-reference".into(),
                revision: env!("CARGO_PKG_VERSION").into(),
                architecture: "qwen35".into(),
                tensor_types: gguf.tensor_types,
                weight_bytes: gguf.file_bytes,
                parameters: gguf.parameters,
                context_tokens: config.context_tokens,
                batch_tokens: config.batch_tokens,
                vocab_tokens: vocab as u32,
                capabilities: Capabilities {
                    token_step: true,
                    chunked_prefill: true,
                },
            },
            tokenizer,
            eog,
            width,
            hidden,
            epsilon,
            base,
            embedding,
            output_norm,
            layers,
        })
    }

    pub fn reset(&mut self) {
        for layer in &mut self.layers {
            if let Attention::Delta(delta) = &mut layer.attention {
                delta.history.fill(0.0);
                delta.state.fill(0.0);
            }
        }
    }

    pub fn step(&mut self, token: i32, pos: usize, logits: &mut [f32]) {
        let mut x = vec![0.0; self.width];
        let mut normalized = vec![0.0; self.width];
        let mut projected = vec![0.0; self.width];
        let mut gate = vec![0.0; self.hidden];
        let mut up = vec![0.0; self.hidden];
        self.embedding.row(token as usize, &mut x);
        for layer in &mut self.layers {
            norm(&x, &layer.input_norm, self.epsilon, &mut normalized);
            match &mut layer.attention {
                Attention::Dense(attn) => {
                    attn.step(&normalized, pos, self.epsilon, self.base, &mut projected)
                }
                Attention::Delta(attn) => attn.step(&normalized, self.epsilon, &mut projected),
            }
            for (x, y) in x.iter_mut().zip(&projected) {
                *x += y;
            }
            norm(&x, &layer.ffn_norm, self.epsilon, &mut normalized);
            layer.gate.multiply_columns(&normalized, 1, &mut gate);
            layer.up.multiply_columns(&normalized, 1, &mut up);
            for (g, u) in gate.iter_mut().zip(&up) {
                *g = silu(*g) * u;
            }
            layer.down.multiply_columns(&gate, 1, &mut projected);
            for (x, y) in x.iter_mut().zip(&projected) {
                *x += y;
            }
        }
        norm(&x, &self.output_norm, self.epsilon, &mut normalized);
        self.embedding.multiply_columns(&normalized, 1, logits);
    }
}

fn sigmoid(x: f32) -> f32 {
    (1.0 + (-x).exp()).recip()
}

fn silu(x: f32) -> f32 {
    x * sigmoid(x)
}

impl Dense {
    fn step(&mut self, input: &[f32], pos: usize, epsilon: f32, base: f32, output: &mut [f32]) {
        let mut qg = vec![0.0; 2 * HEADS * HEAD_DIM];
        let mut q = vec![0.0; HEADS * HEAD_DIM];
        let mut k = vec![0.0; KV_HEADS * HEAD_DIM];
        let mut scratch = vec![0.0; k.len()];
        self.qg.multiply_columns(input, 1, &mut qg);
        self.k.multiply_columns(input, 1, &mut scratch);
        for (dst, src) in q
            .chunks_exact_mut(HEAD_DIM)
            .zip(qg.chunks_exact(2 * HEAD_DIM))
        {
            norm(&src[..HEAD_DIM], &self.q_norm, epsilon, dst);
            // Text supplies the same position on all used MRoPE axes.
            rotary(&mut dst[..ROTARY], ROTARY, pos, base, true);
        }
        for (dst, src) in k
            .chunks_exact_mut(HEAD_DIM)
            .zip(scratch.chunks_exact(HEAD_DIM))
        {
            norm(src, &self.k_norm, epsilon, dst);
            rotary(&mut dst[..ROTARY], ROTARY, pos, base, true);
        }
        let width = KV_HEADS * HEAD_DIM;
        self.keys[pos * width..(pos + 1) * width].copy_from_slice(&k);
        self.v
            .multiply_columns(input, 1, &mut self.values[pos * width..(pos + 1) * width]);
        let mut attended = vec![0.0; q.len()];
        attend_with_divisor(
            &q,
            &self.keys,
            &self.values,
            HEADS,
            KV_HEADS,
            HEAD_DIM,
            pos + 1,
            (HEAD_DIM as f32).sqrt(),
            0.0,
            0.0,
            &mut attended,
            &mut vec![0.0; pos + 1],
        );
        for (dst, src) in attended
            .chunks_exact_mut(HEAD_DIM)
            .zip(qg.chunks_exact(2 * HEAD_DIM))
        {
            for (x, g) in dst.iter_mut().zip(&src[HEAD_DIM..]) {
                *x *= sigmoid(*g);
            }
        }
        self.output.multiply_columns(&attended, 1, output);
    }
}

impl Delta {
    fn step(&mut self, input: &[f32], epsilon: f32, output: &mut [f32]) {
        let mut qkv = vec![0.0; CHANNELS];
        let mut z = vec![0.0; INNER];
        let mut alpha = [0.0; GROUPS];
        let mut beta = [0.0; GROUPS];
        self.qkv.multiply_columns(input, 1, &mut qkv);
        self.z.multiply_columns(input, 1, &mut z);
        self.alpha.multiply_columns(input, 1, &mut alpha);
        self.beta.multiply_columns(input, 1, &mut beta);
        for ((x, history), filter) in qkv
            .iter_mut()
            .zip(self.history.chunks_exact_mut(KERNEL - 1))
            .zip(self.conv.floats().chunks_exact(KERNEL))
        {
            let sum = dot(history, &filter[..KERNEL - 1]) + *x * filter[KERNEL - 1];
            history.rotate_left(1);
            history[KERNEL - 2] = *x;
            *x = silu(sum);
        }
        for head in qkv[..2 * INNER].chunks_exact_mut(STATE_DIM) {
            let scale = (dot(head, head) + epsilon).sqrt().recip();
            head.iter_mut().for_each(|x| *x *= scale);
        }
        let mut out = vec![0.0; INNER];
        let mut mixed = [0.0; STATE_DIM];
        for (h, state) in self
            .state
            .chunks_exact_mut(STATE_DIM * STATE_DIM)
            .enumerate()
        {
            let range = h * STATE_DIM..(h + 1) * STATE_DIM;
            let q = &qkv[range.clone()];
            let k = &qkv[INNER..][range.clone()];
            let v = &qkv[2 * INNER..][range.clone()];
            let a = alpha[h] + self.dt.floats()[h];
            let softplus = if a > 20.0 { a } else { a.exp().ln_1p() };
            let decay = (softplus * self.decay.floats()[h]).exp();
            let beta = sigmoid(beta[h]);
            for (j, row) in state.chunks_exact_mut(STATE_DIM).enumerate() {
                row.iter_mut().for_each(|x| *x *= decay);
                let delta = (v[j] - dot(row, k)) * beta;
                for (x, k) in row.iter_mut().zip(k) {
                    *x += delta * k;
                }
                mixed[j] = dot(row, q) / (STATE_DIM as f32).sqrt();
            }
            norm(&mixed, &self.norm, epsilon, &mut out[range.clone()]);
            for (x, z) in out[range.clone()].iter_mut().zip(&z[range]) {
                *x *= silu(*z);
            }
        }
        self.output.multiply_columns(&out, 1, output);
    }
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}
