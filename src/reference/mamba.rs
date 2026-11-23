use crate::{
    Config, Result,
    backend::{Capabilities, ModelInfo},
    cpu::norm,
    gguf::{Gguf, Tensor, Value},
    tokenizer::Tokenizer,
    transformer::invalid,
};

struct Layer {
    norm: Tensor,
    input: Tensor,
    conv: Vec<f32>,
    conv_bias: Tensor,
    parameters: Tensor,
    dt: Tensor,
    dt_bias: Tensor,
    decay: Vec<f32>,
    skip: Tensor,
    output: Tensor,
    history: Vec<f32>,
    state: Vec<f32>,
}

pub(super) struct Mamba {
    pub info: ModelInfo,
    pub tokenizer: Result<Tokenizer>,
    pub eos: i32,
    epsilon: f32,
    width: usize,
    inner: usize,
    state_dim: usize,
    kernel: usize,
    rank: usize,
    embedding: Tensor,
    output_norm: Tensor,
    output: Option<Tensor>,
    layers: Vec<Layer>,
}

fn rows(tensor: Tensor, width: usize, count: usize) -> Vec<f32> {
    let mut result = vec![0.0; width * count];
    for (i, dst) in result.chunks_exact_mut(width).enumerate() {
        tensor.row(i, dst);
    }
    result
}

impl Mamba {
    pub fn load(mut gguf: Gguf, config: &Config) -> Result<Self> {
        config.validate()?;
        let integer = |key| gguf.integer(&format!("mamba.{key}"));
        let width = integer("embedding_length")?;
        let inner = integer("ssm.inner_size")?;
        let state_dim = integer("ssm.state_size")?;
        let kernel = integer("ssm.conv_kernel")?;
        let rank = integer("ssm.time_step_rank")?;
        let count = integer("block_count")?;
        let epsilon = gguf.float("mamba.attention.layer_norm_rms_epsilon")?;
        if (width, inner, state_dim, kernel, rank, count) != (768, 1536, 16, 4, 48, 24)
            || !matches!(
                gguf.metadata.get("mamba.ssm.dt_b_c_rms"),
                None | Some(Value::Bool(false))
            )
            || !epsilon.is_finite()
            || epsilon <= 0.0
            || config.context_tokens as usize > integer("context_length")?
        {
            return Err(invalid(
                "unsupported Mamba-1 dimensions, RMS mode or context",
            ));
        }
        let vocab = match gguf.metadata.get("tokenizer.ggml.tokens") {
            Some(Value::Strings(v)) if !v.is_empty() && v.len() <= i32::MAX as usize => v.len(),
            _ => return Err(invalid("invalid Mamba vocabulary")),
        };
        let eos = gguf.integer("tokenizer.ggml.eos_token_id")?;
        if eos >= vocab {
            return Err(invalid("Mamba EOS exceeds vocabulary"));
        }
        let tokenizer = Tokenizer::from_metadata(&gguf.metadata);
        let embedding = gguf.tensor("token_embd.weight", &[width, vocab])?;
        let output_norm = gguf.tensor("output_norm.weight", &[width])?;
        let output = gguf
            .tensors
            .contains_key("output.weight")
            .then(|| gguf.tensor("output.weight", &[width, vocab]))
            .transpose()?;
        let mut layers = Vec::with_capacity(count);
        for i in 0..count {
            let mut tensor = |name, shape: &[usize]| gguf.tensor(&format!("blk.{i}.{name}"), shape);
            layers.push(Layer {
                norm: tensor("attn_norm.weight", &[width])?,
                input: tensor("ssm_in.weight", &[width, 2 * inner])?,
                conv: rows(
                    tensor("ssm_conv1d.weight", &[kernel, inner])?,
                    kernel,
                    inner,
                ),
                conv_bias: tensor("ssm_conv1d.bias", &[inner])?,
                parameters: tensor("ssm_x.weight", &[inner, rank + 2 * state_dim])?,
                dt: tensor("ssm_dt.weight", &[rank, inner])?,
                dt_bias: tensor("ssm_dt.bias", &[inner])?,
                decay: rows(tensor("ssm_a", &[state_dim, inner])?, state_dim, inner),
                skip: tensor("ssm_d", &[inner])?,
                output: tensor("ssm_out.weight", &[inner, width])?,
                history: vec![0.0; (kernel - 1) * inner],
                state: vec![0.0; state_dim * inner],
            });
        }
        if !gguf.tensors.is_empty() {
            return Err(invalid("unsupported tensors remain in Mamba text model"));
        }
        Ok(Self {
            info: ModelInfo {
                backend: "rust-cpu-transformer-reference".into(),
                revision: env!("CARGO_PKG_VERSION").into(),
                architecture: "mamba".into(),
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
            eos: eos as i32,
            epsilon,
            width,
            inner,
            state_dim,
            kernel,
            rank,
            embedding,
            output_norm,
            output,
            layers,
        })
    }

    pub fn reset(&mut self) {
        for layer in &mut self.layers {
            layer.history.fill(0.0);
            layer.state.fill(0.0);
        }
    }

    pub fn step(&mut self, token: i32, logits: &mut [f32]) {
        let mut residual = vec![0.0; self.width];
        let mut normalized = vec![0.0; self.width];
        let mut projected = vec![0.0; self.width];
        let mut xz = vec![0.0; 2 * self.inner];
        let mut params = vec![0.0; self.rank + 2 * self.state_dim];
        let mut dt = vec![0.0; self.inner];
        self.embedding.row(token as usize, &mut residual);
        for layer in &mut self.layers {
            norm(&residual, &layer.norm, self.epsilon, &mut normalized);
            layer.input.multiply_columns(&normalized, 1, &mut xz);
            let (x, z) = xz.split_at_mut(self.inner);
            for (i, (x, history)) in x
                .iter_mut()
                .zip(layer.history.chunks_exact_mut(self.kernel - 1))
                .enumerate()
            {
                let filter = &layer.conv[i * self.kernel..(i + 1) * self.kernel];
                let y = history.iter().zip(filter).map(|(h, w)| h * w).sum::<f32>()
                    + *x * filter[self.kernel - 1];
                history.rotate_left(1);
                history[self.kernel - 2] = *x;
                *x = silu(y + layer.conv_bias.floats()[i]);
            }
            layer.parameters.multiply_columns(x, 1, &mut params);
            layer.dt.multiply_columns(&params[..self.rank], 1, &mut dt);
            let (b, c) = params[self.rank..].split_at(self.state_dim);
            for (i, (x, state)) in x
                .iter_mut()
                .zip(layer.state.chunks_exact_mut(self.state_dim))
                .enumerate()
            {
                let t = dt[i] + layer.dt_bias.floats()[i];
                let t = if t > 20.0 { t } else { t.exp().ln_1p() };
                let scaled_x = *x * t;
                let a = &layer.decay[i * self.state_dim..(i + 1) * self.state_dim];
                let y = state
                    .iter_mut()
                    .zip(a)
                    .zip(b.iter().zip(c))
                    .map(|((s, a), (b, c))| {
                        *s = *s * (t * a).exp() + b * scaled_x;
                        *s * c
                    })
                    .sum::<f32>();
                *x = (y + *x * layer.skip.floats()[i]) * silu(z[i]);
            }
            layer.output.multiply_columns(x, 1, &mut projected);
            for (r, p) in residual.iter_mut().zip(&projected) {
                *r += p;
            }
        }
        norm(&residual, &self.output_norm, self.epsilon, &mut normalized);
        self.output
            .as_ref()
            .unwrap_or(&self.embedding)
            .multiply_columns(&normalized, 1, logits);
    }
}

fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}
