//! Slow independent packed CPU backend covering every dense, hybrid, and Mamba layout.
//! It serves architectures the fast CPU path lacks and is the oracle for CUDA.
use crate::{
    Config, Error, Result,
    backend::{Backend, ModelInfo},
    cpu::{attend_with_divisor, norm, rotary_with_factors},
    gguf::{Gguf, Storage, Tensor},
    transformer::{BLOOM_ALIBI_MAX, Weights, route, route_softmax},
};
mod mamba;
pub(crate) mod qwen35;
pub(crate) mod state;
use mamba::Mamba;
use qwen35::Qwen35;
use state::{State, greedy};

enum Model {
    Dense(Weights<Tensor>),
    Hybrid(Qwen35),
    Mamba(Mamba),
}

impl Model {
    fn info(&self) -> &ModelInfo {
        match self {
            Self::Dense(model) => &model.spec.info,
            Self::Hybrid(model) => &model.info,
            Self::Mamba(model) => &model.info,
        }
    }

    fn tokenizer(&self) -> &Result<crate::tokenizer::Tokenizer> {
        match self {
            Self::Dense(model) => &model.spec.tokenizer,
            Self::Hybrid(model) => &model.tokenizer,
            Self::Mamba(model) => &model.tokenizer,
        }
    }
}
use rayon::ThreadPool;
use std::{
    fs::{File, OpenOptions},
    io::{BufWriter, Write},
    path::Path,
    sync::Arc,
};

struct Trace {
    writer: BufWriter<File>,
    position: usize,
}

fn gelu(x: f32) -> f32 {
    0.5 * x * (1.0 + (0.797_884_6 * x * (1.0 + 0.044715 * x * x)).tanh())
}

fn normalize(
    input: &[f32],
    weight: Option<&Tensor>,
    bias: Option<&Tensor>,
    epsilon: f32,
    output: &mut [f32],
) {
    if let (Some(weight), None) = (weight, bias) {
        return norm(input, weight, epsilon, output);
    }
    let mean = input.iter().map(|&x| f64::from(x)).sum::<f64>() as f32 / input.len() as f32;
    for (out, x) in output.iter_mut().zip(input) {
        *out = x - mean;
    }
    let variance =
        (output.iter().map(|x| f64::from(x * x)).sum::<f64>() / input.len() as f64) as f32;
    let scale = (variance + epsilon).sqrt().recip();
    if let (Some(weight), Some(bias)) = (weight, bias) {
        for ((out, weight), bias) in output.iter_mut().zip(weight.floats()).zip(bias.floats()) {
            *out = *out * scale * weight + bias;
        }
    } else {
        for out in output {
            *out *= scale;
        }
    }
}

fn record<T: serde::Serialize>(
    trace: &mut Option<Trace>,
    position: usize,
    layer: usize,
    stage: &str,
    values: &[T],
) -> Result<()> {
    if let Some(trace) = trace.as_mut().filter(|t| t.position == position) {
        serde_json::to_writer(
            &mut trace.writer,
            &serde_json::json!({
                "position": position, "layer": layer, "stage": stage, "values": values,
            }),
        )?;
        trace.writer.write_all(b"\n")?;
    }
    Ok(())
}

pub struct ReferenceBackend {
    model: Model,
    keys: Vec<Vec<f32>>,
    values: Vec<Vec<f32>>,
    logits: Vec<f32>,
    state: State,
    pool: Arc<ThreadPool>,
    trace: Option<Trace>,
}

impl ReferenceBackend {
    pub fn open(path: &Path, config: &Config) -> Result<Self> {
        let gguf = Gguf::open(path, false)?;
        let model = match gguf.text("general.architecture")? {
            "qwen35" => Model::Hybrid(Qwen35::load(gguf, config, true)?),
            "mamba" => Model::Mamba(Mamba::load(gguf, config)?),
            _ => Model::Dense(Weights::from_gguf(gguf, config)?),
        };
        let (width, count) = match &model {
            Model::Dense(weights) => (
                weights.spec.kv_heads * weights.spec.dim,
                weights.layers.len(),
            ),
            Model::Hybrid(_) | Model::Mamba(_) => (0, 0),
        };
        let elements = width
            .checked_mul(config.context_tokens as usize)
            .ok_or_else(|| Error::Invalid("reference KV capacity overflow".into()))?;
        let logits = vec![0.0; model.info().vocab_tokens as usize];
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(config.threads as usize)
            .build()
            .map_err(|e| Error::Backend(e.to_string()))?;
        Ok(Self {
            model,
            logits,
            state: State::default(),
            pool: Arc::new(pool),
            trace: None,
            keys: (0..count).map(|_| vec![0.0; elements]).collect(),
            values: (0..count).map(|_| vec![0.0; elements]).collect(),
        })
    }

    /// Record the specified token's intermediate values to a new diagnostic JSONL file.
    pub fn trace_at(&mut self, path: &Path, position: usize) -> Result<()> {
        if self.trace.is_some()
            || position < self.state.position
            || position >= self.model.info().context_tokens as usize
            || !matches!(self.model, Model::Dense(_))
        {
            return Err(Error::Invalid(
                "invalid or already configured reference trace".into(),
            ));
        }
        self.trace = Some(Trace {
            writer: BufWriter::new(OpenOptions::new().write(true).create_new(true).open(path)?),
            position,
        });
        Ok(())
    }

    fn step(&mut self, token: i32, position: usize) -> Result<()> {
        let model = match &mut self.model {
            Model::Dense(model) => model,
            Model::Hybrid(model) => {
                model.step(token, position, &mut self.logits);
                return Ok(());
            }
            Model::Mamba(model) => {
                model.step(token, &mut self.logits);
                return Ok(());
            }
        };
        let s = &model.spec;
        let w = s.width;
        let qw = s.heads * s.dim;
        let kw = s.kv_heads * s.dim;
        let factors = model.rope_factors.as_ref().map(|t| match &t.storage {
            Storage::F32(values) => values.as_slice(),
            _ => unreachable!("loader validates frequency divisors"),
        });
        let mut x = vec![0.0; w];
        let mut normalized = vec![0.0; w];
        let mut q = vec![0.0; qw];
        let mut k = vec![0.0; kw];
        let mut v = vec![0.0; kw];
        let mut attended = vec![0.0; qw];
        let mut projected = vec![0.0; w];
        let mut scores = vec![0.0; position + 1];
        let mut gate = vec![0.0; s.hidden];
        let mut up = vec![0.0; s.hidden];
        let mut routed = vec![0.0; s.experts];
        model.embedding.row(token as usize, &mut x);
        let alibi_max = if let Some([weight, bias]) = &model.embedding_norm {
            normalize(&x, Some(weight), Some(bias), s.epsilon, &mut normalized);
            std::mem::swap(&mut x, &mut normalized);
            BLOOM_ALIBI_MAX
        } else {
            0.0
        };
        if let Some(embedding) = &model.position_embedding {
            embedding.row(position, &mut projected);
            for (value, positional) in x.iter_mut().zip(&projected) {
                *value += positional;
            }
        }
        if model.gemma.is_some() {
            let scale = (w as f32).sqrt();
            for value in &mut x {
                *value *= scale;
            }
        }
        for (index, l) in model.layers.iter().enumerate() {
            let gemma_attention = model.gemma.as_ref().map(|g| g.attention(s, index));
            let (window, rope_base, _) = gemma_attention.unwrap_or((0, s.rope_base, 1.0));
            let start = if window == 0 {
                0
            } else {
                (position + 1).saturating_sub(window as usize)
            };
            let affine = l.affine.as_ref();
            let olmo2 = s.info.architecture == "olmo2";
            let olmoe = s.info.architecture == "olmoe";
            if olmo2 {
                normalized.copy_from_slice(&x);
            } else {
                normalize(
                    &x,
                    l.norm.as_ref(),
                    affine.map(|b| &b.norm),
                    s.epsilon,
                    &mut normalized,
                );
            }
            record(&mut self.trace, position, index, "attn_norm", &normalized)?;
            l.q.multiply_columns(&normalized, 1, &mut q);
            l.k.multiply_columns(&normalized, 1, &mut k);
            l.v.multiply_columns(&normalized, 1, &mut v);
            if let Some(biases) = &l.qkv_bias {
                for (projection, bias) in [&mut q, &mut k, &mut v].into_iter().zip(biases) {
                    for (value, bias) in projection.iter_mut().zip(bias.floats()) {
                        *value += bias;
                    }
                }
            }
            record(&mut self.trace, position, index, "Qproj", &q)?;
            record(&mut self.trace, position, index, "Kproj", &k)?;
            record(&mut self.trace, position, index, "Vproj", &v)?;
            for (vector, weights) in [(&mut q, &l.q_norm), (&mut k, &l.k_norm)] {
                if let Some(weights) = weights {
                    let norm_width = if olmo2 || olmoe { vector.len() } else { s.dim };
                    let mut scratch = vec![0.0; norm_width];
                    for head in vector.chunks_exact_mut(norm_width) {
                        norm(head, weights, s.epsilon, &mut scratch);
                        head.copy_from_slice(&scratch);
                    }
                }
                if model.position_embedding.is_none() && alibi_max == 0.0 {
                    for head in vector.chunks_exact_mut(s.dim) {
                        rotary_with_factors(
                            &mut head[..s.rotary_dim],
                            s.rotary_dim,
                            position,
                            rope_base,
                            s.neox,
                            factors,
                            s.rope_attn_factor,
                        );
                    }
                }
            }
            record(&mut self.trace, position, index, "Qcur", &q)?;
            record(&mut self.trace, position, index, "Kcur", &k)?;
            record(&mut self.trace, position, index, "Vcur", &v)?;
            self.keys[index][position * kw..(position + 1) * kw].copy_from_slice(&k);
            self.values[index][position * kw..(position + 1) * kw].copy_from_slice(&v);
            let mut divisor = (s.dim as f32).sqrt();
            if let Some((_, _, scale)) = gemma_attention {
                for value in &mut q {
                    *value *= scale;
                }
                divisor = 1.0;
            }
            attend_with_divisor(
                &q,
                &self.keys[index][start * kw..],
                &self.values[index][start * kw..],
                s.heads,
                s.kv_heads,
                s.dim,
                position + 1 - start,
                divisor,
                model.gemma.as_ref().map_or(0.0, |g| g.attention_softcap),
                alibi_max,
                &mut attended,
                &mut scores,
            );
            l.output.multiply_columns(&attended, 1, &mut projected);
            if let Some(b) = affine {
                for (value, bias) in projected.iter_mut().zip(b.attention.floats()) {
                    *value += bias;
                }
            }
            let attention_output = if let Some([post_attn, _]) = &l.post_norms {
                norm(&projected, post_attn, s.epsilon, &mut normalized);
                &normalized
            } else {
                &projected
            };
            if !s.parallel_residual {
                for (x, y) in x.iter_mut().zip(attention_output) {
                    *x += y;
                }
            }
            record(&mut self.trace, position, index, "ffn_inp", &x)?;
            if olmo2 {
                normalized.copy_from_slice(&x);
            } else {
                normalize(
                    &x,
                    l.ffn_norm.as_ref(),
                    affine.map(|b| &b.ffn_norm),
                    s.epsilon,
                    &mut normalized,
                );
            }
            record(&mut self.trace, position, index, "ffn_norm", &normalized)?;
            if let Some(b) = affine {
                l.up.multiply_columns(&normalized, 1, &mut up);
                for (value, bias) in up.iter_mut().zip(b.up.floats()) {
                    *value = gelu(*value + bias);
                }
                if s.parallel_residual {
                    normalized.copy_from_slice(&projected);
                }
                l.down.multiply_columns(&up, 1, &mut projected);
                for ((x, value), bias) in x.iter_mut().zip(&projected).zip(b.down.floats()) {
                    *x += value + bias;
                }
                if s.parallel_residual {
                    for (value, attention) in x.iter_mut().zip(&normalized) {
                        *value += attention;
                    }
                }
                record(&mut self.trace, position, index, "l_out", &x)?;
                continue;
            }
            let gate_weights = l.gate.as_ref().expect("gated layer");
            let selected = if let Some(router) = &l.router {
                router.multiply_columns(&normalized, 1, &mut routed);
                record(&mut self.trace, position, index, "ffn_moe_logits", &routed)?;
                if olmoe {
                    route_softmax(&routed, s.used, false)?
                } else {
                    route(&routed, s.used)?
                }
            } else {
                vec![(0, 1.0)]
            };
            record(
                &mut self.trace,
                position,
                index,
                "ffn_moe_topk",
                &selected.iter().map(|p| p.0).collect::<Vec<_>>(),
            )?;
            record(
                &mut self.trace,
                position,
                index,
                "ffn_moe_weights_norm",
                &selected.iter().map(|p| p.1).collect::<Vec<_>>(),
            )?;
            let mut combined = vec![0.0; w];
            for (expert, probability) in selected {
                if l.router.is_some() {
                    gate_weights.multiply_expert(expert, &normalized, &mut gate);
                    l.up.multiply_expert(expert, &normalized, &mut up);
                } else {
                    gate_weights.multiply_columns(&normalized, 1, &mut gate);
                    l.up.multiply_columns(&normalized, 1, &mut up);
                }
                for (g, u) in gate.iter_mut().zip(&up) {
                    if model.gemma.is_some() {
                        *g = gelu(*g) * u;
                    } else {
                        *g = *g / (1.0 + (-*g).exp()) * u;
                    }
                }
                if l.router.is_some() {
                    l.down.multiply_expert(expert, &gate, &mut projected);
                } else {
                    l.down.multiply_columns(&gate, 1, &mut projected);
                }
                for (sum, &value) in combined.iter_mut().zip(&projected) {
                    *sum += probability * value;
                }
            }
            let output = if let Some([_, post_ffn]) = &l.post_norms {
                norm(&combined, post_ffn, s.epsilon, &mut normalized);
                &normalized
            } else {
                &combined
            };
            for (x, y) in x.iter_mut().zip(output) {
                *x += y;
            }
            record(&mut self.trace, position, index, "ffn_moe_out", &combined)?;
            record(&mut self.trace, position, index, "l_out", &x)?;
        }
        normalize(
            &x,
            model.output_norm.as_ref(),
            model.output_norm_bias.as_ref(),
            s.epsilon,
            &mut normalized,
        );
        model
            .output
            .as_ref()
            .unwrap_or(&model.embedding)
            .multiply_columns(&normalized, 1, &mut self.logits);
        if self.logits.iter().any(|x| !x.is_finite()) {
            return Err(Error::Backend("nonfinite reference logits".into()));
        }
        if let Some(gemma) = &model.gemma
            && gemma.final_softcap > 0.0
        {
            let cap = gemma.final_softcap;
            for logit in &mut self.logits {
                *logit = cap * (*logit / cap).tanh();
            }
        }
        Ok(())
    }
}

impl Backend for ReferenceBackend {
    fn info(&self) -> &ModelInfo {
        self.model.info()
    }
    fn logits(&self) -> Result<&[f32]> {
        self.state.logits(&self.logits)
    }
    fn tokenize(&self, text: &str) -> Result<Vec<i32>> {
        self.model
            .tokenizer()
            .as_ref()
            .map_err(|e| Error::Backend(e.to_string()))?
            .tokenize(text)
    }
    fn detokenize(&self, tokens: &[i32]) -> Result<String> {
        self.model
            .tokenizer()
            .as_ref()
            .map_err(|e| Error::Backend(e.to_string()))?
            .detokenize(tokens)
    }
    fn reset(&mut self) -> Result<()> {
        self.state = State::default();
        match &mut self.model {
            Model::Hybrid(model) => model.reset(),
            Model::Mamba(model) => model.reset(),
            Model::Dense(_) => {}
        }
        Ok(())
    }
    fn evaluate(&mut self, tokens: &[i32], position: u32) -> Result<()> {
        self.state.begin(tokens, position, self.model.info())?;
        let pool = Arc::clone(&self.pool);
        pool.install(|| -> Result<()> {
            for (i, &token) in tokens.iter().enumerate() {
                self.step(token, position as usize + i)?;
            }
            Ok(())
        })?;
        if let Some(trace) = &mut self.trace {
            trace.writer.flush()?;
        }
        self.state.finish(tokens.len());
        Ok(())
    }
    fn sample_greedy(&mut self) -> Result<i32> {
        greedy(self.logits()?)
    }
    fn is_eog(&self, token: i32) -> bool {
        match &self.model {
            Model::Dense(model) => model.spec.eog.contains(&token),
            Model::Hybrid(model) => model.eog.contains(&token),
            Model::Mamba(model) => model.eos == token,
        }
    }
}
