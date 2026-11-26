//! Independent dense transformer execution on CPU with packed weights and causal batched prefill.
//! Consumes the shared `transformer::Weights` normalized representation used by CUDA, Metal, and Vulkan.
pub mod plt1_simd;

use crate::{
    Config, Error, GenerationMode, Result,
    backend::{Backend, MemoryUsage, ModelInfo, SpeculationStats},
    gguf::{Gguf, Tensor},
    tokenizer::Tokenizer,
    transformer::{Gemma, Weights},
};
use rayon::prelude::*;
use serde::Serialize;
use std::{path::Path, sync::Arc};

#[derive(Clone, Debug, Default, Serialize)]
pub struct CpuOptions {
    pub expand_weights: bool,
    pub sequential_prefill: bool,
    pub memory_limit_bytes: Option<u64>,
    pub prefix_cache_bytes: u64,
    pub speculative_window: u32,
    pub scalar_kernels: bool,
}

struct Workspace {
    state: Vec<f32>,
    normalized: Vec<f32>,
    query: Vec<f32>,
    key: Vec<f32>,
    value: Vec<f32>,
    attention: Vec<f32>,
    projected: Vec<f32>,
    gate: Vec<f32>,
    up: Vec<f32>,
    columns: Vec<f32>,
    scores: Vec<f32>,
    saved_logits: Vec<f32>,
    verification: Vec<f32>,
}

impl Workspace {
    fn new(sizes: [usize; 13]) -> Self {
        let [
            state,
            normalized,
            query,
            key,
            value,
            attention,
            projected,
            gate,
            up,
            columns,
            scores,
            saved_logits,
            verification,
        ] = sizes.map(|size| vec![0.0; size]);
        Self {
            state,
            normalized,
            query,
            key,
            value,
            attention,
            projected,
            gate,
            up,
            columns,
            scores,
            saved_logits,
            verification,
        }
    }

    fn allocated_bytes(&self) -> u64 {
        [
            &self.state,
            &self.normalized,
            &self.query,
            &self.key,
            &self.value,
            &self.attention,
            &self.projected,
            &self.gate,
            &self.up,
            &self.columns,
            &self.scores,
            &self.saved_logits,
            &self.verification,
        ]
        .iter()
        .map(|buffer| buffer.capacity() as u64 * 4)
        .sum()
    }
}

struct Layer {
    attention_norm: Option<Tensor>,
    query: Tensor,
    key: Tensor,
    value: Tensor,
    qkv_bias: Option<[Tensor; 3]>,
    q_norm: Option<Tensor>,
    k_norm: Option<Tensor>,
    attention_output: Tensor,
    ffn_norm: Option<Tensor>,
    gate: Tensor,
    up: Tensor,
    down: Tensor,
    post_norms: Option<[Tensor; 2]>,
    keys: Vec<f32>,
    values: Vec<f32>,
}

struct Prefix {
    tokens: Vec<i32>,
    logits: Vec<f32>,
    keys: Vec<Vec<f32>>,
    values: Vec<Vec<f32>>,
}

impl Prefix {
    fn allocated_bytes(&self) -> u64 {
        (self.tokens.capacity()
            + self.logits.capacity()
            + self
                .keys
                .iter()
                .chain(&self.values)
                .map(Vec::capacity)
                .sum::<usize>()) as u64
            * 4
    }
}

/// A single-use, backend-owned append transaction. Reset invalidates the handle.
#[derive(Debug)]
pub struct Transaction {
    owner: Arc<()>,
    serial: u64,
}

pub struct CpuBackend {
    info: ModelInfo,
    pool: Arc<rayon::ThreadPool>,
    embedding: Tensor,
    layers: Vec<Layer>,
    output_norm: Option<Tensor>,
    output: Option<Tensor>,
    rope_factors: Option<Tensor>,
    gemma: Option<Gemma>,
    width: usize,
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
    epsilon: f32,
    rope_base: f32,
    rope_attn_factor: f32,
    neox: bool,
    eog: Vec<i32>,
    position: usize,
    logits: Vec<f32>,
    logits_valid: bool,
    tokenizer: Result<Tokenizer>,
    options: CpuOptions,
    workspace: Workspace,
    memory: MemoryUsage,
    history: Vec<i32>,
    prefix: Option<Prefix>,
    owner: Arc<()>,
    transaction: Option<(u64, usize)>,
    transaction_serial: u64,
    speculation: SpeculationStats,
    failed: bool,
}

fn invalid(message: impl Into<String>) -> Error {
    Error::Invalid(format!("Rust CPU: {}", message.into()))
}

pub(crate) fn norm(input: &[f32], weights: &Tensor, epsilon: f32, output: &mut [f32]) {
    let inverse = (input.iter().map(|x| x * x).sum::<f32>() / input.len() as f32 + epsilon)
        .sqrt()
        .recip();
    for ((value, x), w) in output.iter_mut().zip(input).zip(weights.floats()) {
        *value = x * inverse * w;
    }
}

pub(crate) fn rotary(vector: &mut [f32], head_dim: usize, position: usize, base: f32, neox: bool) {
    rotary_with_factors(vector, head_dim, position, base, neox, None, 1.0);
}

pub(crate) fn rotary_with_factors(
    vector: &mut [f32],
    head_dim: usize,
    position: usize,
    base: f32,
    neox: bool,
    factors: Option<&[f32]>,
    magnitude: f32,
) {
    for head in vector.chunks_exact_mut(head_dim) {
        for pair in 0..head_dim / 2 {
            let angle = position as f32 * base.powf(-((2 * pair) as f32) / head_dim as f32);
            let angle = factors.map_or(angle, |factors| angle / factors[pair]);
            let (sin, cos) = angle.sin_cos();
            let (sin, cos) = (sin * magnitude, cos * magnitude);
            let (a, b) = if neox {
                (pair, pair + head_dim / 2)
            } else {
                (2 * pair, 2 * pair + 1)
            };
            let (x, y) = (head[a], head[b]);
            head[a] = x * cos - y * sin;
            head[b] = x * sin + y * cos;
        }
    }
}

#[cfg(test)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn attend(
    query: &[f32],
    keys: &[f32],
    values: &[f32],
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
    count: usize,
    output: &mut [f32],
    scores: &mut [f32],
) {
    attend_with_divisor(
        query,
        keys,
        values,
        heads,
        kv_heads,
        head_dim,
        count,
        (head_dim as f32).sqrt(),
        0.0,
        0.0,
        output,
        scores,
    );
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn attend_with_divisor(
    query: &[f32],
    keys: &[f32],
    values: &[f32],
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
    count: usize,
    divisor: f32,
    softcap: f32,
    alibi_max: f32,
    output: &mut [f32],
    scores: &mut [f32],
) {
    output.fill(0.0);
    let kv_width = kv_heads * head_dim;
    for head in 0..heads {
        let slope = if alibi_max > 0.0 {
            let base_heads = 1 << heads.ilog2();
            if head < base_heads {
                2.0_f32
                    .powf(-alibi_max / base_heads as f32)
                    .powi(head as i32 + 1)
            } else {
                2.0_f32
                    .powf(-alibi_max / (2 * base_heads) as f32)
                    .powi((2 * (head - base_heads) + 1) as i32)
            }
        } else {
            0.0
        };
        let kv_head = head / (heads / kv_heads);
        let offset = kv_head * head_dim;
        let q = &query[head * head_dim..(head + 1) * head_dim];
        let scores = &mut scores[..count];
        for (time, score) in scores.iter_mut().enumerate() {
            *score = q
                .iter()
                .zip(&keys[time * kv_width + offset..time * kv_width + offset + head_dim])
                .map(|(q, k)| q * k)
                .sum::<f32>()
                / divisor;
            if softcap > 0.0 {
                *score = if score.is_finite() {
                    softcap * (*score * softcap.recip()).tanh()
                } else {
                    f32::NAN
                };
            }
            if alibi_max > 0.0 {
                *score -= slope * (count - 1 - time) as f32;
            }
        }
        let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        for value in &mut *scores {
            *value = (*value - max).exp();
        }
        let total: f32 = scores.iter().sum();
        for (time, score) in scores.iter().enumerate() {
            for dim in 0..head_dim {
                output[head * head_dim + dim] +=
                    score / total * values[time * kv_width + offset + dim];
            }
        }
    }
}

/// Whether the fast packed CPU path can run this model. Other layouts (MoE,
/// affine FFN, recurrent/hybrid) use the reference backend.
pub fn has_fast_path(path: &Path) -> Result<bool> {
    let (metadata, _tensors) = Gguf::read_layout(path)?;
    let arch = match metadata.get("general.architecture") {
        Some(crate::gguf::Value::Text(arch)) => arch.as_str(),
        _ => return Ok(false),
    };
    let dense_gated = matches!(
        arch,
        "llama" | "qwen2" | "qwen3" | "phi3" | "olmo2" | "gemma2" | "gemma3"
    );
    let has_experts = metadata.contains_key(&format!("{arch}.expert_count"));
    Ok(dense_gated && !has_experts)
}

impl CpuBackend {
    pub fn open(path: &Path, config: &Config) -> Result<Self> {
        Self::open_with_options(path, config, CpuOptions::default())
    }

    pub fn open_with_options(path: &Path, config: &Config, options: CpuOptions) -> Result<Self> {
        config.validate()?;
        if options.speculative_window > config.batch_tokens {
            return Err(invalid("speculative window exceeds batch capacity"));
        }
        let mut model = Gguf::open(path, options.expand_weights)?;
        if options.scalar_kernels {
            for tensor in model.tensors.values_mut() {
                tensor.accelerated = false;
            }
        }
        let weights_allocated_bytes: u64 =
            model.tensors.values().map(Tensor::allocated_bytes).sum();
        let weights = Weights::from_gguf(model, config)?;
        if weights.spec.experts > 0
            || weights.position_embedding.is_some()
            || weights.embedding_norm.is_some()
            || weights.output_norm_bias.is_some()
            || weights.spec.rotary_dim != weights.spec.dim
            || weights
                .layers
                .iter()
                .any(|l| l.affine.is_some() || l.gate.is_none())
        {
            return Err(invalid(format!(
                "fast CPU path does not support {} layout; use ReferenceBackend",
                weights.spec.info.architecture
            )));
        }
        let s = &weights.spec;
        let width = s.width;
        let heads = s.heads;
        let kv_heads = s.kv_heads;
        let head_dim = s.dim;
        let q_width = heads * head_dim;
        let kv_width = kv_heads * head_dim;
        let layer_count = weights.layers.len();
        let hidden = s.hidden;
        let vocab = s.info.vocab_tokens as usize;
        let product = |a: usize, b: usize| {
            a.checked_mul(b)
                .ok_or_else(|| invalid("allocation size overflow"))
        };
        let batch = config.batch_tokens as usize;
        let context = config.context_tokens as usize;
        let bw = product(batch, width)?;
        let bq = product(batch, q_width)?;
        let bk = product(batch, kv_width)?;
        let bh = product(batch, hidden)?;
        let sizes = [
            bw,
            bw,
            bq,
            bk,
            bk,
            bq,
            bw,
            bh,
            bh,
            product(batch, width.max(q_width).max(hidden).max(kv_width))?,
            context,
            vocab,
            product(options.speculative_window as usize, vocab)?,
        ];
        let workspace_bytes = sizes
            .iter()
            .try_fold(0u64, |sum, &size| {
                sum.checked_add(size as u64)
                    .ok_or_else(|| invalid("workspace overflow"))
            })?
            .checked_mul(4)
            .ok_or_else(|| invalid("workspace byte overflow"))?;
        let kv_allocated_bytes = (product(product(context, kv_width)?, layer_count)? as u64)
            .checked_mul(8)
            .ok_or_else(|| invalid("KV byte overflow"))?;
        let logits_bytes = vocab as u64 * 4;
        let history_bytes = context as u64 * 4;
        let maximum_prefix = kv_allocated_bytes
            .checked_add(logits_bytes)
            .and_then(|bytes| bytes.checked_add(history_bytes))
            .ok_or_else(|| invalid("prefix reservation overflow"))?;
        let prefix_capacity_bytes = options.prefix_cache_bytes.min(maximum_prefix);
        let worker_scratch_bound_bytes = u64::from(config.threads) * 1056;
        let numerical_reservation_bytes = [
            weights_allocated_bytes,
            workspace_bytes,
            kv_allocated_bytes,
            logits_bytes,
            history_bytes,
            prefix_capacity_bytes,
            worker_scratch_bound_bytes,
        ]
        .into_iter()
        .try_fold(0u64, |sum, size| {
            sum.checked_add(size)
                .ok_or_else(|| invalid("numerical reservation overflow"))
        })?;
        if options
            .memory_limit_bytes
            .is_some_and(|limit| numerical_reservation_bytes > limit)
        {
            return Err(invalid(format!(
                "numerical reservation {numerical_reservation_bytes} bytes exceeds memory limit"
            )));
        }
        let cache_elements = (config.context_tokens as usize)
            .checked_mul(kv_width)
            .ok_or_else(|| invalid("KV cache size overflow"))?;
        let mut info = weights.spec.info;
        info.backend = match (
            options.expand_weights,
            !options.scalar_kernels && crate::kernels::avx2_available(),
        ) {
            (true, _) => "rust-cpu-f32-reference",
            (false, true) => "rust-cpu-packed-avx2",
            (false, false) => "rust-cpu-packed-portable",
        }
        .into();
        let epsilon = weights.spec.epsilon;
        let rope_base = weights.spec.rope_base;
        let rope_attn_factor = weights.spec.rope_attn_factor;
        let neox = weights.spec.neox;
        let eog = weights.spec.eog;
        let tokenizer = weights.spec.tokenizer;
        let layers = weights
            .layers
            .into_iter()
            .map(|l| Layer {
                attention_norm: l.norm,
                query: l.q,
                key: l.k,
                value: l.v,
                qkv_bias: l.qkv_bias,
                q_norm: l.q_norm,
                k_norm: l.k_norm,
                attention_output: l.output,
                ffn_norm: l.ffn_norm,
                gate: l.gate.expect("dense gated layer"),
                up: l.up,
                down: l.down,
                post_norms: l.post_norms,
                keys: vec![0.0; cache_elements],
                values: vec![0.0; cache_elements],
            })
            .collect();
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(config.threads as usize)
            .build()
            .map_err(|error| invalid(format!("thread pool: {error}")))?;
        let workspace = Workspace::new(sizes);
        let workspace_allocated_bytes = workspace.allocated_bytes();
        Ok(Self {
            info,
            pool: Arc::new(pool),
            embedding: weights.embedding,
            layers,
            output_norm: weights.output_norm,
            output: weights.output,
            rope_factors: weights.rope_factors,
            gemma: weights.gemma,
            width,
            heads,
            kv_heads,
            head_dim,
            epsilon,
            rope_base,
            rope_attn_factor,
            neox,
            eog,
            position: 0,
            logits: vec![0.0; vocab],
            logits_valid: false,
            tokenizer,
            workspace,
            history: Vec::with_capacity(context),
            memory: MemoryUsage {
                allocation_domain: "host",
                weights_allocated_bytes,
                kv_allocated_bytes,
                kv_live_bytes: 0,
                workspace_allocated_bytes,
                logits_allocated_bytes: logits_bytes,
                token_state_allocated_bytes: history_bytes,
                prefix_capacity_bytes,
                prefix_allocated_bytes: 0,
                worker_scratch_bound_bytes,
                numerical_reservation_bytes,
            },
            options,
            prefix: None,
            owner: Arc::new(()),
            transaction: None,
            transaction_serial: 0,
            speculation: SpeculationStats::default(),
            failed: false,
        })
    }

    fn forward(&mut self, tokens: &[i32], verify: bool) -> Result<()> {
        let width = self.width;
        let q_width = self.heads * self.head_dim;
        let kv_width = self.kv_heads * self.head_dim;
        let count = tokens.len();
        let hidden = self.layers[0].gate.dims[1];
        let olmo2 = self.info.architecture == "olmo2";
        let gemma_arch = self.info.architecture == "gemma3";
        let factors = self.rope_factors.as_ref().map(Tensor::floats);
        let w = &mut self.workspace;
        for (&token, output) in tokens.iter().zip(w.state.chunks_mut(width)) {
            self.embedding.row(token as usize, output);
            if self.gemma.is_some() {
                let scale = (width as f32).sqrt();
                for v in output {
                    *v *= scale;
                }
            }
        }
        for (layer_idx, layer) in self.layers.iter_mut().enumerate() {
            let gemma_attn = self.gemma.as_ref().map(|g| {
                let dim = if gemma_arch && g.local_layers.len() == 62 {
                    width / self.heads
                } else {
                    self.head_dim
                };
                let (window, base) = if g.local_layers[layer_idx] {
                    (g.window, g.local_rope_base)
                } else {
                    (0, self.rope_base)
                };
                (window, base, (dim as f32).sqrt().recip(), g.attention_softcap)
            });
            let rope_base = gemma_attn.map_or(self.rope_base, |(_, base, _, _)| base);

            let attn_input = if olmo2 || layer.attention_norm.is_none() {
                &w.state[..count * width]
            } else {
                let attn_norm = layer.attention_norm.as_ref().unwrap();
                w.normalized[..count * width]
                    .par_chunks_mut(width)
                    .zip(w.state[..count * width].par_chunks(width))
                    .for_each(|(output, input)| norm(input, attn_norm, self.epsilon, output));
                &w.normalized[..count * width]
            };
            layer.query.multiply(
                attn_input,
                count,
                &mut w.query[..count * q_width],
                &mut w.columns,
            );
            layer.key.multiply(
                attn_input,
                count,
                &mut w.key[..count * kv_width],
                &mut w.columns,
            );
            layer.value.multiply(
                attn_input,
                count,
                &mut w.value[..count * kv_width],
                &mut w.columns,
            );
            if let Some([qb, kb, vb]) = &layer.qkv_bias {
                for time in 0..count {
                    for (dst, b) in w.query[time * q_width..(time + 1) * q_width]
                        .iter_mut()
                        .zip(qb.floats())
                    {
                        *dst += b;
                    }
                    for (dst, b) in w.key[time * kv_width..(time + 1) * kv_width]
                        .iter_mut()
                        .zip(kb.floats())
                    {
                        *dst += b;
                    }
                    for (dst, b) in w.value[time * kv_width..(time + 1) * kv_width]
                        .iter_mut()
                        .zip(vb.floats())
                    {
                        *dst += b;
                    }
                }
            }
            for time in 0..count {
                let position = self.position + time;
                let query = &mut w.query[time * q_width..(time + 1) * q_width];
                let key = &mut w.key[time * kv_width..(time + 1) * kv_width];
                if let Some(qn) = &layer.q_norm {
                    let nw = if olmo2 { q_width } else { self.head_dim };
                    let mut tmp = vec![0.0; nw];
                    for chunk in query.chunks_exact_mut(nw) {
                        norm(chunk, qn, self.epsilon, &mut tmp);
                        chunk.copy_from_slice(&tmp);
                    }
                }
                if let Some(kn) = &layer.k_norm {
                    let nw = if olmo2 { kv_width } else { self.head_dim };
                    let mut tmp = vec![0.0; nw];
                    for chunk in key.chunks_exact_mut(nw) {
                        norm(chunk, kn, self.epsilon, &mut tmp);
                        chunk.copy_from_slice(&tmp);
                    }
                }
                rotary_with_factors(
                    key,
                    self.head_dim,
                    position,
                    rope_base,
                    self.neox,
                    factors,
                    self.rope_attn_factor,
                );
                rotary_with_factors(
                    query,
                    self.head_dim,
                    position,
                    rope_base,
                    self.neox,
                    factors,
                    self.rope_attn_factor,
                );
                layer.keys[position * kv_width..(position + 1) * kv_width].copy_from_slice(key);
                layer.values[position * kv_width..(position + 1) * kv_width]
                    .copy_from_slice(&w.value[time * kv_width..(time + 1) * kv_width]);
            }
            for time in 0..count {
                let total = self.position + time + 1;
                let (start, divisor, softcap) = if let Some((window, _, scale, softcap)) = gemma_attn
                {
                    let start = if window > 0 && total > window {
                        total - window
                    } else {
                        0
                    };
                    for v in &mut w.query[time * q_width..(time + 1) * q_width] {
                        *v *= scale;
                    }
                    (start, 1.0, softcap)
                } else {
                    (0, (self.head_dim as f32).sqrt(), 0.0)
                };
                attend_with_divisor(
                    &w.query[time * q_width..(time + 1) * q_width],
                    &layer.keys[start * kv_width..],
                    &layer.values[start * kv_width..],
                    self.heads,
                    self.kv_heads,
                    self.head_dim,
                    total - start,
                    divisor,
                    softcap,
                    0.0,
                    &mut w.attention[time * q_width..(time + 1) * q_width],
                    &mut w.scores,
                );
            }
            layer.attention_output.multiply(
                &w.attention[..count * q_width],
                count,
                &mut w.projected[..count * width],
                &mut w.columns,
            );
            if let Some([post_attn, _]) = &layer.post_norms {
                w.normalized[..count * width]
                    .par_chunks_mut(width)
                    .zip(w.projected[..count * width].par_chunks(width))
                    .for_each(|(output, input)| norm(input, post_attn, self.epsilon, output));
                for (x, update) in w.state.iter_mut().zip(&w.normalized).take(count * width) {
                    *x += update;
                }
            } else if olmo2 {
                let post_attn = layer.attention_norm.as_ref().unwrap();
                w.normalized[..count * width]
                    .par_chunks_mut(width)
                    .zip(w.projected[..count * width].par_chunks(width))
                    .for_each(|(output, input)| norm(input, post_attn, self.epsilon, output));
                for (x, update) in w.state.iter_mut().zip(&w.normalized).take(count * width) {
                    *x += update;
                }
            } else {
                for (x, update) in w.state.iter_mut().zip(&w.projected).take(count * width) {
                    *x += update;
                }
            }

            let ffn_input = if olmo2 || layer.ffn_norm.is_none() {
                &w.state[..count * width]
            } else {
                let ffn_norm = layer.ffn_norm.as_ref().unwrap();
                w.normalized[..count * width]
                    .par_chunks_mut(width)
                    .zip(w.state[..count * width].par_chunks(width))
                    .for_each(|(output, input)| norm(input, ffn_norm, self.epsilon, output));
                &w.normalized[..count * width]
            };
            layer.gate.multiply(
                ffn_input,
                count,
                &mut w.gate[..count * hidden],
                &mut w.columns,
            );
            layer.up.multiply(
                ffn_input,
                count,
                &mut w.up[..count * hidden],
                &mut w.columns,
            );
            if self.gemma.is_some() {
                for (g, u) in w.gate.iter_mut().zip(&w.up).take(count * hidden) {
                    let gelu =
                        0.5 * *g * (1.0 + (0.797_884_6 * *g * (1.0 + 0.044715 * *g * *g)).tanh());
                    *g = gelu * u;
                }
            } else {
                for (g, u) in w.gate.iter_mut().zip(&w.up).take(count * hidden) {
                    *g = *g / (1.0 + (-*g).exp()) * u;
                }
            }
            layer.down.multiply(
                &w.gate[..count * hidden],
                count,
                &mut w.projected[..count * width],
                &mut w.columns,
            );
            if let Some([_, post_ffn]) = &layer.post_norms {
                w.normalized[..count * width]
                    .par_chunks_mut(width)
                    .zip(w.projected[..count * width].par_chunks(width))
                    .for_each(|(output, input)| norm(input, post_ffn, self.epsilon, output));
                for (x, update) in w.state.iter_mut().zip(&w.normalized).take(count * width) {
                    *x += update;
                }
            } else if olmo2 {
                let post_ffn = layer.ffn_norm.as_ref().unwrap();
                w.normalized[..count * width]
                    .par_chunks_mut(width)
                    .zip(w.projected[..count * width].par_chunks(width))
                    .for_each(|(output, input)| norm(input, post_ffn, self.epsilon, output));
                for (x, update) in w.state.iter_mut().zip(&w.normalized).take(count * width) {
                    *x += update;
                }
            } else {
                for (x, update) in w.state.iter_mut().zip(&w.projected).take(count * width) {
                    *x += update;
                }
            }
        }
        let start = if verify { 0 } else { (count - 1) * width };
        let final_input = if let Some(output_norm) = &self.output_norm {
            for (input, output) in w.state[start..count * width]
                .chunks(width)
                .zip(w.normalized[start..count * width].chunks_mut(width))
            {
                norm(input, output_norm, self.epsilon, output);
            }
            &w.normalized[start..count * width]
        } else {
            &w.state[start..count * width]
        };
        let output = self.output.as_ref().unwrap_or(&self.embedding);
        if verify {
            output.multiply_columns(
                final_input,
                count,
                &mut w.verification[..count * self.logits.len()],
            );
            if let Some(cap) = self.gemma.as_ref().map(|g| g.final_softcap).filter(|&c| c > 0.0) {
                for v in &mut w.verification[..count * self.logits.len()] {
                    *v = cap * (*v / cap).tanh();
                }
            }
            for (row, value) in self.logits.iter_mut().enumerate() {
                *value = w.verification[row * count + count - 1];
            }
        } else {
            output.multiply(final_input, 1, &mut self.logits, &mut w.columns);
            if let Some(cap) = self.gemma.as_ref().map(|g| g.final_softcap).filter(|&c| c > 0.0) {
                for v in &mut self.logits {
                    *v = cap * (*v / cap).tanh();
                }
            }
        }
        if self.logits.iter().any(|value| !value.is_finite())
            || (verify
                && w.verification[..count * self.logits.len()]
                    .iter()
                    .any(|value| !value.is_finite()))
        {
            return Err(Error::Backend(
                "Rust CPU: non-finite logits; reset required".into(),
            ));
        }
        self.logits_valid = true;
        self.position += count;
        self.history.extend_from_slice(tokens);
        Ok(())
    }

    pub fn logits(&self) -> Result<&[f32]> {
        if !self.logits_valid {
            return Err(Error::Backend(
                "Rust CPU: no valid logits; evaluate first".into(),
            ));
        }
        Ok(&self.logits)
    }

    pub fn begin_transaction(&mut self) -> Result<Transaction> {
        self.logits()?;
        if self.transaction.is_some() {
            return Err(invalid("nested transactions are not supported"));
        }
        self.transaction_serial = self
            .transaction_serial
            .checked_add(1)
            .ok_or_else(|| invalid("transaction serial exhausted"))?;
        self.workspace.saved_logits.copy_from_slice(&self.logits);
        self.transaction = Some((self.transaction_serial, self.position));
        Ok(Transaction {
            owner: Arc::clone(&self.owner),
            serial: self.transaction_serial,
        })
    }

    fn transaction_position(&self, transaction: &Transaction) -> Result<usize> {
        match self.transaction {
            Some((serial, position))
                if Arc::ptr_eq(&transaction.owner, &self.owner)
                    && serial == transaction.serial
                    && position <= self.position =>
            {
                Ok(position)
            }
            _ => Err(invalid("stale or foreign transaction")),
        }
    }

    pub fn commit_transaction(&mut self, transaction: Transaction) -> Result<()> {
        self.transaction_position(&transaction)?;
        self.logits()?;
        self.transaction = None;
        Ok(())
    }

    pub fn rollback_transaction(&mut self, transaction: Transaction) -> Result<()> {
        let position = self.transaction_position(&transaction)?;
        self.position = position;
        self.history.truncate(position);
        self.logits.copy_from_slice(&self.workspace.saved_logits);
        self.logits_valid = true;
        self.transaction = None;
        self.failed = false;
        Ok(())
    }

    fn validate_evaluation(&self, tokens: &[i32], position: u32) -> Result<()> {
        if self.failed {
            return Err(Error::Backend(
                "Rust CPU: failed evaluation; reset or rollback required".into(),
            ));
        }
        if tokens.is_empty()
            || tokens.len() > self.info.batch_tokens as usize
            || position as usize != self.position
            || self
                .position
                .checked_add(tokens.len())
                .is_none_or(|end| end > self.info.context_tokens as usize)
            || tokens
                .iter()
                .any(|&token| token < 0 || token as u32 >= self.info.vocab_tokens)
        {
            return Err(invalid(
                "invalid tokens, batch size, position, or context capacity",
            ));
        }
        Ok(())
    }

    /// Commits only the greedy-matching draft prefix. Rejected KV rows remain
    /// allocated but are made unreachable by truncating the active position.
    pub fn verify_draft(&mut self, draft: &[i32]) -> Result<usize> {
        self.validate_evaluation(draft, self.position as u32)?;
        if draft.len() > self.options.speculative_window as usize {
            return Err(invalid("draft exceeds configured speculative window"));
        }
        let first = self.sample_greedy()?;
        let transaction = self.begin_transaction()?;
        let start = self.position;
        let pool = Arc::clone(&self.pool);
        self.logits_valid = false;
        if let Err(error) = pool.install(|| self.forward(draft, true)) {
            self.rollback_transaction(transaction)?;
            return Err(error);
        }
        let mut accepted = 0;
        let mut predicted = first;
        for (index, &token) in draft.iter().enumerate() {
            if token != predicted {
                break;
            }
            accepted += 1;
            let mut best = 0;
            for row in 1..self.logits.len() {
                if self.workspace.verification[row * draft.len() + index]
                    > self.workspace.verification[best * draft.len() + index]
                {
                    best = row;
                }
            }
            predicted = best as i32;
        }
        if accepted == 0 {
            self.rollback_transaction(transaction)?;
        } else {
            self.position = start + accepted;
            self.history.truncate(self.position);
            for (row, value) in self.logits.iter_mut().enumerate() {
                *value = self.workspace.verification[row * draft.len() + accepted - 1];
            }
            self.commit_transaction(transaction)?;
        }
        self.speculation.proposed_tokens += draft.len() as u64;
        self.speculation.accepted_tokens += accepted as u64;
        self.speculation.verification_batches += 1;
        Ok(accepted)
    }

    fn propose(&self, maximum: usize) -> Vec<i32> {
        for length in (1..=4.min(self.history.len())).rev() {
            let suffix = &self.history[self.history.len() - length..];
            for start in (0..self.history.len() - length).rev() {
                if &self.history[start..start + length] == suffix {
                    return self.history[start + length..]
                        .iter()
                        .take(maximum)
                        .copied()
                        .collect();
                }
            }
        }
        Vec::new()
    }
}

impl Backend for CpuBackend {
    fn logits(&self) -> Result<&[f32]> {
        CpuBackend::logits(self)
    }
    fn info(&self) -> &ModelInfo {
        &self.info
    }
    fn tokenize(&self, text: &str) -> Result<Vec<i32>> {
        self.tokenizer
            .as_ref()
            .map_err(|error| invalid(format!("text unavailable: {error}")))?
            .tokenize(text)
    }
    fn detokenize(&self, tokens: &[i32]) -> Result<String> {
        self.tokenizer
            .as_ref()
            .map_err(|error| invalid(format!("text unavailable: {error}")))?
            .detokenize(tokens)
    }
    fn reset(&mut self) -> Result<()> {
        self.position = 0;
        self.logits_valid = false;
        self.history.clear();
        self.transaction = None;
        self.speculation = SpeculationStats::default();
        self.failed = false;
        // Old cache entries are unreachable: attention only reads up to position.
        Ok(())
    }
    fn evaluate(&mut self, tokens: &[i32], position: u32) -> Result<()> {
        self.validate_evaluation(tokens, position)?;
        self.logits_valid = false;
        let pool = Arc::clone(&self.pool);
        let result = pool.install(|| {
            if self.options.sequential_prefill {
                for token in tokens.chunks(1) {
                    self.forward(token, false)?;
                }
            } else {
                self.forward(tokens, false)?;
            }
            Ok(())
        });
        if result.is_err() {
            self.logits_valid = false;
            self.failed = true;
        }
        result
    }
    fn sample_greedy(&mut self) -> Result<i32> {
        let logits = self.logits()?;
        let mut best = 0;
        for i in 1..logits.len() {
            if logits[i] > logits[best] {
                best = i;
            }
        }
        Ok(best as i32)
    }
    fn is_eog(&self, token: i32) -> bool {
        self.eog.contains(&token)
    }
    fn memory_usage(&self) -> Option<MemoryUsage> {
        let mut memory = self.memory.clone();
        memory.kv_live_bytes = self.position as u64
            * self.kv_heads as u64
            * self.head_dim as u64
            * self.layers.len() as u64
            * 8;
        memory.prefix_allocated_bytes = self.prefix.as_ref().map_or(0, Prefix::allocated_bytes);
        Some(memory)
    }
    fn restore_prefix(&mut self, prompt: &[i32]) -> Result<usize> {
        if self.position != 0 || self.logits_valid || self.transaction.is_some() {
            return Err(invalid("prefix restoration requires a reset backend"));
        }
        let Some(prefix) = self
            .prefix
            .as_ref()
            .filter(|prefix| prompt.starts_with(&prefix.tokens))
        else {
            return Ok(0);
        };
        let width = self.kv_heads * self.head_dim * prefix.tokens.len();
        for ((layer, keys), values) in self.layers.iter_mut().zip(&prefix.keys).zip(&prefix.values)
        {
            layer.keys[..width].copy_from_slice(keys);
            layer.values[..width].copy_from_slice(values);
        }
        self.position = prefix.tokens.len();
        self.history.extend_from_slice(&prefix.tokens);
        self.logits.copy_from_slice(&prefix.logits);
        self.logits_valid = true;
        Ok(self.position)
    }
    fn store_prefix(&mut self, prompt: &[i32]) -> Result<()> {
        if self.memory.prefix_capacity_bytes == 0 {
            return Ok(());
        }
        if self.transaction.is_some() || self.history != prompt || !self.logits_valid {
            return Err(invalid(
                "prefix snapshot requires exactly the evaluated prompt",
            ));
        }
        let width = self.kv_heads * self.head_dim * prompt.len();
        let required = (width as u64 * self.layers.len() as u64 * 2
            + self.logits.len() as u64
            + prompt.len() as u64)
            * 4;
        if required > self.memory.prefix_capacity_bytes
            || self
                .prefix
                .as_ref()
                .is_some_and(|prefix| prefix.tokens == prompt)
        {
            return Ok(());
        }
        // Evict before allocating the replacement so two snapshots cannot exceed the budget.
        self.prefix = None;
        self.prefix = Some(Prefix {
            tokens: prompt.to_vec(),
            logits: self.logits.clone(),
            keys: self
                .layers
                .iter()
                .map(|layer| layer.keys[..width].to_vec())
                .collect(),
            values: self
                .layers
                .iter()
                .map(|layer| layer.values[..width].to_vec())
                .collect(),
        });
        Ok(())
    }
    fn sample_tokens(&mut self, maximum: usize, mode: GenerationMode) -> Result<Vec<i32>> {
        if maximum == 0 {
            return Err(invalid("sample maximum must be positive"));
        }
        let maximum_draft = maximum
            .saturating_sub(1)
            .min(self.options.speculative_window as usize);
        if maximum_draft == 0 {
            return Ok(vec![self.sample_greedy()?]);
        }
        let mut draft = self.propose(maximum_draft);
        if mode == GenerationMode::UntilEog {
            if let Some(end) = draft.iter().position(|&token| self.is_eog(token)) {
                draft.truncate(end);
            }
        }
        if draft.is_empty() {
            return Ok(vec![self.sample_greedy()?]);
        }
        let accepted = self.verify_draft(&draft)?;
        draft.truncate(accepted);
        draft.push(self.sample_greedy()?);
        Ok(draft)
    }
    fn speculation_stats(&self) -> Option<SpeculationStats> {
        (self.options.speculative_window > 0).then(|| self.speculation.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn rotary_pairs_preserve_norm_and_use_absolute_positions() {
        let mut x = [1.0, 0.0, 0.0, 1.0];
        rotary(&mut x, 4, 2, 100.0, false);
        assert!((x[0] - 2.0f32.cos()).abs() < 1e-6);
        assert!((x[1] - 2.0f32.sin()).abs() < 1e-6);
        assert!((x[2] + 0.2f32.sin()).abs() < 1e-6);
        assert!((x.iter().map(|v| v * v).sum::<f32>() - 2.0).abs() < 1e-6);
    }
    #[test]
    fn grouped_attention_reads_only_causal_positions_and_maps_kv_heads() {
        let query = [0.0; 8];
        let keys = [0.0; 12];
        let values = [
            1.0,
            2.0,
            10.0,
            20.0,
            3.0,
            4.0,
            30.0,
            40.0,
            f32::NAN,
            f32::NAN,
            f32::NAN,
            f32::NAN,
        ];
        let mut output = [0.0; 8];
        attend(
            &query,
            &keys,
            &values,
            4,
            2,
            2,
            2,
            &mut output,
            &mut [0.0; 3],
        );
        assert_eq!(output, [2.0, 3.0, 2.0, 3.0, 20.0, 30.0, 20.0, 30.0]);
    }
}
