//! Independent dense llama execution with packed weights and causal batched prefill.
use crate::{
    Config, Error, GenerationMode, Result,
    backend::{Backend, Capabilities, MemoryUsage, ModelInfo, SpeculationStats},
    gguf::{Gguf, Tensor},
    tokenizer::Tokenizer,
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
    attention_norm: Tensor,
    query: Tensor,
    key: Tensor,
    value: Tensor,
    attention_output: Tensor,
    ffn_norm: Tensor,
    gate: Tensor,
    up: Tensor,
    down: Tensor,
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
    output_norm: Tensor,
    output: Option<Tensor>,
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
    epsilon: f32,
    rope_base: f32,
    eos: i32,
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

fn rope(vector: &mut [f32], head_dim: usize, position: usize, base: f32) {
    rotary(vector, head_dim, position, base, false);
}

pub(crate) fn rotary(vector: &mut [f32], head_dim: usize, position: usize, base: f32, neox: bool) {
    for head in vector.chunks_exact_mut(head_dim) {
        for pair in 0..head_dim / 2 {
            let angle = position as f32 * base.powf(-((2 * pair) as f32) / head_dim as f32);
            let (sin, cos) = angle.sin_cos();
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
    output.fill(0.0);
    let kv_width = kv_heads * head_dim;
    for head in 0..heads {
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
                / (head_dim as f32).sqrt();
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
        if model.text("general.architecture")? != "llama" {
            return Err(invalid("only dense llama architecture is implemented"));
        }
        for key in [
            "llama.rope.scaling.type",
            "llama.attention.sliding_window",
            "llama.expert_count",
            "llama.attention.logit_softcap",
        ] {
            if model.metadata.contains_key(key) {
                return Err(invalid(format!("unsupported model feature {key}")));
            }
        }
        let width = model.integer("llama.embedding_length")?;
        let heads = model.integer("llama.attention.head_count")?;
        let kv_heads = model.integer("llama.attention.head_count_kv")?;
        let layer_count = model.integer("llama.block_count")?;
        let hidden = model.integer("llama.feed_forward_length")?;
        let trained_context = model.integer("llama.context_length")?;
        if width == 0
            || heads == 0
            || kv_heads == 0
            || layer_count == 0
            || hidden == 0
            || width % heads != 0
            || heads % kv_heads != 0
            || (width / heads) % 2 != 0
            || config.context_tokens as usize > trained_context
        {
            return Err(invalid(
                "invalid dimensions or context exceeds training capacity",
            ));
        }
        let head_dim = width / heads;
        if model.integer("llama.rope.dimension_count")? != head_dim {
            return Err(invalid("partial rotary embeddings are not supported"));
        }
        let kv_width = kv_heads * head_dim;
        let epsilon = model.float("llama.attention.layer_norm_rms_epsilon")?;
        let rope_base = model.float("llama.rope.freq_base")?;
        let vocab = model
            .tensors
            .get("token_embd.weight")
            .filter(|tensor| tensor.dims.len() == 2)
            .map(|tensor| tensor.dims[1])
            .ok_or_else(|| invalid("missing embedding matrix"))?;
        let eos = i32::try_from(model.integer("tokenizer.ggml.eos_token_id")?)
            .map_err(|_| invalid("EOG token exceeds i32"))?;
        if vocab == 0 || vocab > i32::MAX as usize || eos < 0 || eos as usize >= vocab {
            return Err(invalid("invalid vocabulary or EOG token"));
        }
        let tokenizer = Tokenizer::from_metadata(&model.metadata).and_then(|tokenizer| {
            if tokenizer.vocab_tokens() != vocab {
                return Err(invalid(
                    "tokenizer vocabulary does not match embedding rows",
                ));
            }
            Ok(tokenizer)
        });
        let weights_allocated_bytes: u64 =
            model.tensors.values().map(Tensor::allocated_bytes).sum();
        let product = |a: usize, b: usize| {
            a.checked_mul(b)
                .ok_or_else(|| invalid("allocation size overflow"))
        };
        let batch = config.batch_tokens as usize;
        let context = config.context_tokens as usize;
        let bw = product(batch, width)?;
        let bk = product(batch, kv_width)?;
        let bh = product(batch, hidden)?;
        let sizes = [
            bw,
            bw,
            bw,
            bk,
            bk,
            bw,
            bw,
            bh,
            bh,
            product(batch, width.max(hidden).max(kv_width))?,
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
        let embedding = model.tensor("token_embd.weight", &[width, vocab])?;
        let output_norm = model.tensor("output_norm.weight", &[width])?;
        let output = if model.tensors.contains_key("output.weight") {
            Some(model.tensor("output.weight", &[width, vocab])?)
        } else {
            None
        };
        let cache_elements = (config.context_tokens as usize)
            .checked_mul(kv_width)
            .ok_or_else(|| invalid("KV cache size overflow"))?;
        let mut layers = Vec::new();
        for index in 0..layer_count {
            let prefix = format!("blk.{index}");
            layers.push(Layer {
                attention_norm: model.tensor(&format!("{prefix}.attn_norm.weight"), &[width])?,
                query: model.tensor(&format!("{prefix}.attn_q.weight"), &[width, width])?,
                key: model.tensor(&format!("{prefix}.attn_k.weight"), &[width, kv_width])?,
                value: model.tensor(&format!("{prefix}.attn_v.weight"), &[width, kv_width])?,
                attention_output: model
                    .tensor(&format!("{prefix}.attn_output.weight"), &[width, width])?,
                ffn_norm: model.tensor(&format!("{prefix}.ffn_norm.weight"), &[width])?,
                gate: model.tensor(&format!("{prefix}.ffn_gate.weight"), &[width, hidden])?,
                up: model.tensor(&format!("{prefix}.ffn_up.weight"), &[width, hidden])?,
                down: model.tensor(&format!("{prefix}.ffn_down.weight"), &[hidden, width])?,
                keys: vec![0.0; cache_elements],
                values: vec![0.0; cache_elements],
            });
        }
        if !model.tensors.is_empty() {
            let mut names: Vec<_> = model.tensors.keys().cloned().collect();
            names.sort();
            return Err(invalid(format!("unsupported tensors: {names:?}")));
        }
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(config.threads as usize)
            .build()
            .map_err(|error| invalid(format!("thread pool: {error}")))?;
        let workspace = Workspace::new(sizes);
        let workspace_allocated_bytes = workspace.allocated_bytes();
        Ok(Self {
            info: ModelInfo {
                backend: match (
                    options.expand_weights,
                    !options.scalar_kernels && crate::kernels::avx2_available(),
                ) {
                    (true, _) => "rust-cpu-f32-reference",
                    (false, true) => "rust-cpu-packed-avx2",
                    (false, false) => "rust-cpu-packed-portable",
                }
                .into(),
                revision: env!("CARGO_PKG_VERSION").into(),
                architecture: "llama".into(),
                tensor_types: model.tensor_types,
                weight_bytes: model.file_bytes,
                parameters: model.parameters,
                context_tokens: config.context_tokens,
                batch_tokens: config.batch_tokens,
                vocab_tokens: vocab as u32,
                capabilities: Capabilities {
                    token_step: true,
                    chunked_prefill: true,
                },
            },
            pool: Arc::new(pool),
            embedding,
            layers,
            output_norm,
            output,
            heads,
            kv_heads,
            head_dim,
            epsilon,
            rope_base,
            eos,
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
        let width = self.embedding.dims[0];
        let kv_width = self.kv_heads * self.head_dim;
        let count = tokens.len();
        let hidden = self.layers[0].gate.dims[1];
        let w = &mut self.workspace;
        for (&token, output) in tokens.iter().zip(w.state.chunks_mut(width)) {
            self.embedding.row(token as usize, output);
        }
        for layer in &mut self.layers {
            w.normalized[..count * width]
                .par_chunks_mut(width)
                .zip(w.state[..count * width].par_chunks(width))
                .for_each(|(output, input)| {
                    norm(input, &layer.attention_norm, self.epsilon, output)
                });
            layer.query.multiply(
                &w.normalized[..count * width],
                count,
                &mut w.query[..count * width],
                &mut w.columns,
            );
            layer.key.multiply(
                &w.normalized[..count * width],
                count,
                &mut w.key[..count * kv_width],
                &mut w.columns,
            );
            layer.value.multiply(
                &w.normalized[..count * width],
                count,
                &mut w.value[..count * kv_width],
                &mut w.columns,
            );
            for time in 0..count {
                let position = self.position + time;
                let key = &mut w.key[time * kv_width..(time + 1) * kv_width];
                rope(key, self.head_dim, position, self.rope_base);
                rope(
                    &mut w.query[time * width..(time + 1) * width],
                    self.head_dim,
                    position,
                    self.rope_base,
                );
                layer.keys[position * kv_width..(position + 1) * kv_width].copy_from_slice(key);
                layer.values[position * kv_width..(position + 1) * kv_width]
                    .copy_from_slice(&w.value[time * kv_width..(time + 1) * kv_width]);
            }
            for time in 0..count {
                attend(
                    &w.query[time * width..(time + 1) * width],
                    &layer.keys,
                    &layer.values,
                    self.heads,
                    self.kv_heads,
                    self.head_dim,
                    self.position + time + 1,
                    &mut w.attention[time * width..(time + 1) * width],
                    &mut w.scores,
                );
            }
            layer.attention_output.multiply(
                &w.attention[..count * width],
                count,
                &mut w.projected[..count * width],
                &mut w.columns,
            );
            for (x, update) in w.state.iter_mut().zip(&w.projected).take(count * width) {
                *x += update;
            }
            w.normalized[..count * width]
                .par_chunks_mut(width)
                .zip(w.state[..count * width].par_chunks(width))
                .for_each(|(output, input)| norm(input, &layer.ffn_norm, self.epsilon, output));
            layer.gate.multiply(
                &w.normalized[..count * width],
                count,
                &mut w.gate[..count * hidden],
                &mut w.columns,
            );
            layer.up.multiply(
                &w.normalized[..count * width],
                count,
                &mut w.up[..count * hidden],
                &mut w.columns,
            );
            for (g, u) in w.gate.iter_mut().zip(&w.up).take(count * hidden) {
                *g = *g / (1.0 + (-*g).exp()) * u;
            }
            layer.down.multiply(
                &w.gate[..count * hidden],
                count,
                &mut w.projected[..count * width],
                &mut w.columns,
            );
            for (x, update) in w.state.iter_mut().zip(&w.projected).take(count * width) {
                *x += update;
            }
        }
        let start = if verify { 0 } else { (count - 1) * width };
        for (input, output) in w.state[start..count * width]
            .chunks(width)
            .zip(w.normalized[start..count * width].chunks_mut(width))
        {
            norm(input, &self.output_norm, self.epsilon, output);
        }
        let output = self.output.as_ref().unwrap_or(&self.embedding);
        if verify {
            output.multiply_columns(
                &w.normalized[..count * width],
                count,
                &mut w.verification[..count * self.logits.len()],
            );
            for (row, value) in self.logits.iter_mut().enumerate() {
                *value = w.verification[row * count + count - 1];
            }
        } else {
            output.multiply(
                &w.normalized[start..count * width],
                1,
                &mut self.logits,
                &mut w.columns,
            );
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
        token == self.eos
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
        rope(&mut x, 4, 2, 100.0);
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
