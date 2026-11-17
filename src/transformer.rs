use crate::{
    Config, Error, Result,
    backend::{Capabilities, ModelInfo},
    gguf::{Gguf, Tensor, Value},
    tokenizer::Tokenizer,
};
use std::path::Path;

pub(crate) fn invalid(message: impl std::fmt::Display) -> Error {
    Error::Invalid(format!("Rust transformer: {message}"))
}

pub(crate) struct Spec {
    pub info: ModelInfo,
    pub width: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub dim: usize,
    pub hidden: usize,
    pub experts: usize,
    pub used: usize,
    pub epsilon: f32,
    pub rope_base: f32,
    pub neox: bool,
    pub eog: Vec<i32>,
    pub tokenizer: Result<Tokenizer>,
}

pub(crate) struct Layer<T> {
    pub norm: T,
    pub q: T,
    pub k: T,
    pub v: T,
    pub output: T,
    pub q_norm: Option<T>,
    pub k_norm: Option<T>,
    pub ffn_norm: T,
    pub router: Option<T>,
    pub gate: T,
    pub up: T,
    pub down: T,
}

pub(crate) struct Weights<T> {
    pub spec: Spec,
    pub embedding: T,
    pub output_norm: T,
    pub output: Option<T>,
    pub layers: Vec<Layer<T>>,
}

impl<T> Weights<T> {
    pub fn map<U>(self, mut convert: impl FnMut(T) -> Result<U>) -> Result<Weights<U>> {
        let mut layers = Vec::with_capacity(self.layers.len());
        for layer in self.layers {
            layers.push(Layer {
                norm: convert(layer.norm)?,
                q: convert(layer.q)?,
                k: convert(layer.k)?,
                v: convert(layer.v)?,
                output: convert(layer.output)?,
                q_norm: layer.q_norm.map(&mut convert).transpose()?,
                k_norm: layer.k_norm.map(&mut convert).transpose()?,
                ffn_norm: convert(layer.ffn_norm)?,
                router: layer.router.map(&mut convert).transpose()?,
                gate: convert(layer.gate)?,
                up: convert(layer.up)?,
                down: convert(layer.down)?,
            });
        }
        Ok(Weights {
            spec: self.spec,
            embedding: convert(self.embedding)?,
            output_norm: convert(self.output_norm)?,
            output: self.output.map(convert).transpose()?,
            layers,
        })
    }
}

impl Weights<Tensor> {
    pub fn open(path: &Path, config: &Config) -> Result<Self> {
        config.validate()?;
        let mut model = Gguf::open(path, false)?;
        let arch = model.text("general.architecture")?.to_owned();
        let moe = match arch.as_str() {
            "llama" => false,
            "qwen3moe" => true,
            _ => return Err(invalid(format!("unsupported architecture {arch}"))),
        };
        for suffix in [
            "rope.scaling.type",
            "attention.sliding_window",
            "attention.logit_softcap",
            "expert_shared_count",
        ] {
            if model.metadata.contains_key(&format!("{arch}.{suffix}")) {
                return Err(invalid(format!("unsupported model feature {suffix}")));
            }
        }
        if !moe && model.metadata.contains_key("llama.expert_count") {
            return Err(invalid("llama MoE is not supported"));
        }
        let integer = |suffix: &str| model.integer(&format!("{arch}.{suffix}"));
        let width = integer("embedding_length")?;
        let heads = integer("attention.head_count")?;
        let kv_heads = integer("attention.head_count_kv")?;
        let count = integer("block_count")?;
        let context = integer("context_length")?;
        let dim = if moe {
            integer("attention.key_length")?
        } else {
            width
                .checked_div(heads)
                .ok_or_else(|| invalid("zero head count"))?
        };
        let hidden = integer(if moe {
            "expert_feed_forward_length"
        } else {
            "feed_forward_length"
        })?;
        let (experts, used) = if moe {
            (integer("expert_count")?, integer("expert_used_count")?)
        } else {
            (0, 1)
        };
        let epsilon = model.float(&format!("{arch}.attention.layer_norm_rms_epsilon"))?;
        let rope_base = model.float(&format!("{arch}.rope.freq_base"))?;
        if width == 0
            || count == 0
            || heads == 0
            || kv_heads == 0
            || hidden == 0
            || heads % kv_heads != 0
            || dim == 0
            || dim % 2 != 0
            || !epsilon.is_finite()
            || !rope_base.is_finite()
            || epsilon <= 0.0
            || rope_base <= 0.0
            || config.context_tokens as usize > context
            || (moe && (experts == 0 || used == 0 || used > experts))
            || (!moe && (width % heads != 0 || integer("rope.dimension_count")? != dim))
            || (moe && integer("attention.value_length")? != dim)
        {
            return Err(invalid(
                "invalid dimensions, rotary layout, experts, or context",
            ));
        }
        if let Some(value) = model.metadata.get(&format!("{arch}.rope.dimension_count")) {
            if !matches!(value, Value::Integer(n) if *n == dim as u64) {
                return Err(invalid("partial rotary embeddings are not supported"));
            }
        }
        let q_width = heads
            .checked_mul(dim)
            .ok_or_else(|| invalid("query width overflow"))?;
        let kv_width = kv_heads
            .checked_mul(dim)
            .ok_or_else(|| invalid("KV width overflow"))?;
        for size in [
            width, heads, kv_heads, dim, hidden, experts, used, q_width, kv_width, count,
        ] {
            if size > i32::MAX as usize {
                return Err(invalid("dimensions exceed CUDA indexing"));
            }
        }
        let vocab = model
            .tensors
            .get("token_embd.weight")
            .filter(|t| t.dims.len() == 2)
            .map(|t| t.dims[1])
            .ok_or_else(|| invalid("missing embedding matrix"))?;
        if vocab == 0 || vocab > i32::MAX as usize {
            return Err(invalid("invalid vocabulary size"));
        }
        let eos = i32::try_from(model.integer("tokenizer.ggml.eos_token_id")?)
            .map_err(|_| invalid("invalid EOG token"))?;
        if eos < 0 || eos as usize >= vocab {
            return Err(invalid("EOG exceeds vocabulary"));
        }
        let mut eog = vec![eos];
        if moe {
            if let Some(value) = model.metadata.get("tokenizer.ggml.tokens") {
                let Value::Strings(tokens) = value else {
                    return Err(invalid("invalid Qwen vocabulary"));
                };
                let Some(Value::Integers(types)) = model.metadata.get("tokenizer.ggml.token_type")
                else {
                    return Err(invalid("missing Qwen token types"));
                };
                if tokens.len() != vocab || types.len() != vocab {
                    return Err(invalid("Qwen vocabulary/types do not match embeddings"));
                }
                for (id, (text, kind)) in tokens.iter().zip(types).enumerate() {
                    if *kind == 3
                        && matches!(text.as_str(), "<|endoftext|>" | "<|im_end|>")
                        && !eog.contains(&(id as i32))
                    {
                        eog.push(id as i32);
                    }
                }
            }
        }
        let tokenizer = Tokenizer::from_metadata(&model.metadata).and_then(|tokenizer| {
            if tokenizer.vocab_tokens() != vocab {
                return Err(invalid("tokenizer vocabulary mismatch"));
            }
            Ok(tokenizer)
        });
        let info = ModelInfo {
            backend: "rust-cpu-transformer-reference".into(),
            revision: env!("CARGO_PKG_VERSION").into(),
            architecture: arch,
            tensor_types: model.tensor_types.clone(),
            weight_bytes: model.file_bytes,
            parameters: model.parameters,
            context_tokens: config.context_tokens,
            batch_tokens: config.batch_tokens,
            vocab_tokens: vocab as u32,
            capabilities: Capabilities {
                token_step: true,
                chunked_prefill: true,
            },
        };
        let embedding = model.tensor("token_embd.weight", &[width, vocab])?;
        let output_norm = model.tensor("output_norm.weight", &[width])?;
        let output = if model.tensors.contains_key("output.weight") {
            Some(model.tensor("output.weight", &[width, vocab])?)
        } else {
            None
        };
        let mut layers = Vec::with_capacity(count);
        for i in 0..count {
            let mut tensor =
                |name: &str, dims: &[usize]| model.tensor(&format!("blk.{i}.{name}.weight"), dims);
            layers.push(Layer {
                norm: tensor("attn_norm", &[width])?,
                q: tensor("attn_q", &[width, q_width])?,
                k: tensor("attn_k", &[width, kv_width])?,
                v: tensor("attn_v", &[width, kv_width])?,
                output: tensor("attn_output", &[q_width, width])?,
                q_norm: if moe {
                    Some(tensor("attn_q_norm", &[dim])?)
                } else {
                    None
                },
                k_norm: if moe {
                    Some(tensor("attn_k_norm", &[dim])?)
                } else {
                    None
                },
                ffn_norm: tensor("ffn_norm", &[width])?,
                router: if moe {
                    Some(tensor("ffn_gate_inp", &[width, experts])?)
                } else {
                    None
                },
                gate: if moe {
                    tensor("ffn_gate_exps", &[width, hidden, experts])?
                } else {
                    tensor("ffn_gate", &[width, hidden])?
                },
                up: if moe {
                    tensor("ffn_up_exps", &[width, hidden, experts])?
                } else {
                    tensor("ffn_up", &[width, hidden])?
                },
                down: if moe {
                    tensor("ffn_down_exps", &[hidden, width, experts])?
                } else {
                    tensor("ffn_down", &[hidden, width])?
                },
            });
        }
        if !model.tensors.is_empty() {
            return Err(invalid("unsupported tensors remain in model"));
        }
        Ok(Self {
            spec: Spec {
                info,
                width,
                heads,
                kv_heads,
                dim,
                hidden,
                experts,
                used,
                epsilon,
                rope_base,
                neox: moe,
                eog,
                tokenizer,
            },
            embedding,
            output_norm,
            output,
            layers,
        })
    }

    pub fn bytes(&self) -> u64 {
        let mut bytes = self.embedding.allocated_bytes()
            + self.output_norm.allocated_bytes()
            + self.output.as_ref().map_or(0, Tensor::allocated_bytes);
        for l in &self.layers {
            for t in [
                &l.norm,
                &l.q,
                &l.k,
                &l.v,
                &l.output,
                &l.ffn_norm,
                &l.gate,
                &l.up,
                &l.down,
            ] {
                bytes += t.allocated_bytes();
            }
            for t in [&l.q_norm, &l.k_norm, &l.router].into_iter().flatten() {
                bytes += t.allocated_bytes();
            }
        }
        bytes
    }
}

pub(crate) fn route(scores: &[f32], used: usize) -> Result<Vec<(usize, f32)>> {
    if used == 0 || used > scores.len() || scores.iter().any(|v| !v.is_finite()) {
        return Err(invalid("invalid router scores or expert count"));
    }
    let mut ids: Vec<_> = (0..scores.len()).collect();
    ids.sort_unstable_by(|&a, &b| {
        scores[b]
            .partial_cmp(&scores[a])
            .expect("finite scores")
            .then(a.cmp(&b))
    });
    ids.truncate(used);
    let max = scores[ids[0]];
    let sum: f32 = ids.iter().map(|&id| (scores[id] - max).exp()).sum();
    Ok(ids
        .into_iter()
        .map(|id| (id, (scores[id] - max).exp() / sum))
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn routing_is_stable_normalized_and_rejects_nonfinite_scores() {
        let selected = route(&[-1000.0, 1000.0, 1000.0, 999.0], 3).unwrap();
        assert_eq!(selected.iter().map(|x| x.0).collect::<Vec<_>>(), [1, 2, 3]);
        assert!((selected.iter().map(|x| x.1).sum::<f32>() - 1.0).abs() < 1e-6);
        assert!(route(&[f32::NAN], 1).is_err());
        assert!(route(&[0.0], 2).is_err());
        assert!(route(&[0.0], 0).is_err());
    }
}
