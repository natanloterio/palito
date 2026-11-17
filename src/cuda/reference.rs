//! Slow independent packed CPU oracle for the CUDA transformer's model layout.
use super::{State, greedy};
use crate::{
    Config, Error, Result,
    backend::{Backend, ModelInfo},
    cpu::{attend, norm, rotary},
    gguf::Tensor,
    transformer::{Weights, route},
};
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
    weights: Weights<Tensor>,
    keys: Vec<Vec<f32>>,
    values: Vec<Vec<f32>>,
    logits: Vec<f32>,
    state: State,
    pool: Arc<ThreadPool>,
    trace: Option<Trace>,
}

impl ReferenceBackend {
    pub fn open(path: &Path, config: &Config) -> Result<Self> {
        let weights = Weights::open(path, config)?;
        let width = weights.spec.kv_heads * weights.spec.dim;
        let elements = width
            .checked_mul(config.context_tokens as usize)
            .ok_or_else(|| Error::Invalid("reference KV capacity overflow".into()))?;
        let count = weights.layers.len();
        let logits = vec![0.0; weights.spec.info.vocab_tokens as usize];
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(config.threads as usize)
            .build()
            .map_err(|e| Error::Backend(e.to_string()))?;
        Ok(Self {
            weights,
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
            || position >= self.weights.spec.info.context_tokens as usize
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
        let model = &self.weights;
        let s = &model.spec;
        let w = s.width;
        let qw = s.heads * s.dim;
        let kw = s.kv_heads * s.dim;
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
        for (index, l) in model.layers.iter().enumerate() {
            norm(&x, &l.norm, s.epsilon, &mut normalized);
            record(&mut self.trace, position, index, "attn_norm", &normalized)?;
            l.q.multiply_columns(&normalized, 1, &mut q);
            l.k.multiply_columns(&normalized, 1, &mut k);
            l.v.multiply_columns(&normalized, 1, &mut v);
            record(&mut self.trace, position, index, "Qproj", &q)?;
            record(&mut self.trace, position, index, "Kproj", &k)?;
            record(&mut self.trace, position, index, "Vproj", &v)?;
            for (vector, weights) in [(&mut q, &l.q_norm), (&mut k, &l.k_norm)] {
                if let Some(weights) = weights {
                    let mut scratch = vec![0.0; s.dim];
                    for head in vector.chunks_exact_mut(s.dim) {
                        norm(head, weights, s.epsilon, &mut scratch);
                        head.copy_from_slice(&scratch);
                    }
                }
                rotary(vector, s.dim, position, s.rope_base, s.neox);
            }
            record(&mut self.trace, position, index, "Qcur", &q)?;
            record(&mut self.trace, position, index, "Kcur", &k)?;
            record(&mut self.trace, position, index, "Vcur", &v)?;
            self.keys[index][position * kw..(position + 1) * kw].copy_from_slice(&k);
            self.values[index][position * kw..(position + 1) * kw].copy_from_slice(&v);
            attend(
                &q,
                &self.keys[index],
                &self.values[index],
                s.heads,
                s.kv_heads,
                s.dim,
                position + 1,
                &mut attended,
                &mut scores,
            );
            l.output.multiply_columns(&attended, 1, &mut projected);
            for (x, y) in x.iter_mut().zip(&projected) {
                *x += y;
            }
            record(&mut self.trace, position, index, "ffn_inp", &x)?;
            norm(&x, &l.ffn_norm, s.epsilon, &mut normalized);
            record(&mut self.trace, position, index, "ffn_norm", &normalized)?;
            let selected = if let Some(router) = &l.router {
                router.multiply_columns(&normalized, 1, &mut routed);
                record(&mut self.trace, position, index, "ffn_moe_logits", &routed)?;
                route(&routed, s.used)?
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
                    l.gate.multiply_expert(expert, &normalized, &mut gate);
                    l.up.multiply_expert(expert, &normalized, &mut up);
                } else {
                    l.gate.multiply_columns(&normalized, 1, &mut gate);
                    l.up.multiply_columns(&normalized, 1, &mut up);
                }
                for (g, u) in gate.iter_mut().zip(&up) {
                    *g = *g / (1.0 + (-*g).exp()) * u;
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
            for (x, y) in x.iter_mut().zip(&combined) {
                *x += y;
            }
            record(&mut self.trace, position, index, "ffn_moe_out", &combined)?;
            record(&mut self.trace, position, index, "l_out", &x)?;
        }
        norm(&x, &model.output_norm, s.epsilon, &mut normalized);
        model
            .output
            .as_ref()
            .unwrap_or(&model.embedding)
            .multiply_columns(&normalized, 1, &mut self.logits);
        if self.logits.iter().any(|x| !x.is_finite()) {
            return Err(Error::Backend("nonfinite reference logits".into()));
        }
        Ok(())
    }
}

impl Backend for ReferenceBackend {
    fn info(&self) -> &ModelInfo {
        &self.weights.spec.info
    }
    fn logits(&self) -> Result<&[f32]> {
        self.state.logits(&self.logits)
    }
    fn tokenize(&self, text: &str) -> Result<Vec<i32>> {
        self.weights
            .spec
            .tokenizer
            .as_ref()
            .map_err(|e| Error::Backend(e.to_string()))?
            .tokenize(text)
    }
    fn detokenize(&self, tokens: &[i32]) -> Result<String> {
        self.weights
            .spec
            .tokenizer
            .as_ref()
            .map_err(|e| Error::Backend(e.to_string()))?
            .detokenize(tokens)
    }
    fn reset(&mut self) -> Result<()> {
        self.state = State::default();
        Ok(())
    }
    fn evaluate(&mut self, tokens: &[i32], position: u32) -> Result<()> {
        self.state
            .begin(tokens, position, &self.weights.spec.info)?;
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
        self.weights.spec.eog.contains(&token)
    }
}
