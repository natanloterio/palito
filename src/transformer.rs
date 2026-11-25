use crate::{
    Config, Error, Result,
    backend::{Capabilities, ModelInfo},
    gguf::{Gguf, Storage, Tensor, Value},
    tokenizer::Tokenizer,
};
use std::path::Path;

pub(crate) fn invalid(message: impl std::fmt::Display) -> Error {
    Error::Invalid(format!("Rust transformer: {message}"))
}

pub(crate) const BLOOM_ALIBI_MAX: f32 = 8.0;

pub(crate) struct Spec {
    pub info: ModelInfo,
    pub width: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub dim: usize,
    pub rotary_dim: usize,
    pub parallel_residual: bool,
    pub hidden: usize,
    pub experts: usize,
    pub used: usize,
    pub epsilon: f32,
    pub rope_base: f32,
    pub rope_attn_factor: f32,
    pub neox: bool,
    pub eog: Vec<i32>,
    pub tokenizer: Result<Tokenizer>,
}

pub(crate) struct Layer<T> {
    pub norm: Option<T>,
    pub q: T,
    pub k: T,
    pub v: T,
    pub qkv_bias: Option<[T; 3]>,
    pub output: T,
    pub q_norm: Option<T>,
    pub k_norm: Option<T>,
    pub ffn_norm: Option<T>,
    pub router: Option<T>,
    pub gate: Option<T>,
    pub up: T,
    pub down: T,
    pub post_norms: Option<[T; 2]>,
    pub affine: Option<AffineLayer<T>>,
}

pub(crate) struct AffineLayer<T> {
    pub norm: T,
    pub attention: T,
    pub ffn_norm: T,
    pub up: T,
    pub down: T,
}

pub(crate) struct Gemma {
    pub local_layers: Vec<bool>,
    pub window: usize,
    pub local_rope_base: f32,
    pub final_softcap: f32,
    pub attention_softcap: f32,
}

impl Gemma {
    pub fn attention(&self, spec: &Spec, layer: usize) -> (i32, f32, f32) {
        let dim = if spec.info.architecture == "gemma3" && self.local_layers.len() == 62 {
            spec.width / spec.heads
        } else {
            spec.dim
        };
        let (window, base) = if self.local_layers[layer] {
            (self.window as i32, self.local_rope_base)
        } else {
            (0, spec.rope_base)
        };
        (window, base, (dim as f32).sqrt().recip())
    }
}

pub(crate) struct Weights<T> {
    pub spec: Spec,
    pub embedding: T,
    pub position_embedding: Option<T>,
    pub embedding_norm: Option<[T; 2]>,
    pub output_norm: Option<T>,
    pub output_norm_bias: Option<T>,
    pub output: Option<T>,
    pub rope_factors: Option<T>,
    pub layers: Vec<Layer<T>>,
    pub gemma: Option<Gemma>,
}

impl<T> Weights<T> {
    #[cfg_attr(not(any(feature = "cuda", feature = "vulkan", all(feature = "metal", target_os = "macos"))), allow(dead_code))]
    pub fn map<U>(self, mut convert: impl FnMut(T) -> Result<U>) -> Result<Weights<U>> {
        let mut layers = Vec::with_capacity(self.layers.len());
        for layer in self.layers {
            layers.push(Layer {
                norm: layer.norm.map(&mut convert).transpose()?,
                q: convert(layer.q)?,
                k: convert(layer.k)?,
                v: convert(layer.v)?,
                qkv_bias: layer
                    .qkv_bias
                    .map(|[q, k, v]| Ok::<_, Error>([convert(q)?, convert(k)?, convert(v)?]))
                    .transpose()?,
                output: convert(layer.output)?,
                q_norm: layer.q_norm.map(&mut convert).transpose()?,
                k_norm: layer.k_norm.map(&mut convert).transpose()?,
                ffn_norm: layer.ffn_norm.map(&mut convert).transpose()?,
                router: layer.router.map(&mut convert).transpose()?,
                gate: layer.gate.map(&mut convert).transpose()?,
                up: convert(layer.up)?,
                down: convert(layer.down)?,
                post_norms: layer
                    .post_norms
                    .map(|[a, b]| Ok::<_, Error>([convert(a)?, convert(b)?]))
                    .transpose()?,
                affine: layer
                    .affine
                    .map(|b| {
                        Ok::<_, Error>(AffineLayer {
                            norm: convert(b.norm)?,
                            attention: convert(b.attention)?,
                            ffn_norm: convert(b.ffn_norm)?,
                            up: convert(b.up)?,
                            down: convert(b.down)?,
                        })
                    })
                    .transpose()?,
            });
        }
        Ok(Weights {
            spec: self.spec,
            embedding: convert(self.embedding)?,
            position_embedding: self.position_embedding.map(&mut convert).transpose()?,
            embedding_norm: self
                .embedding_norm
                .map(|[weight, bias]| Ok::<_, Error>([convert(weight)?, convert(bias)?]))
                .transpose()?,
            output_norm: self.output_norm.map(&mut convert).transpose()?,
            output_norm_bias: self.output_norm_bias.map(&mut convert).transpose()?,
            output: self.output.map(&mut convert).transpose()?,
            rope_factors: self.rope_factors.map(convert).transpose()?,
            layers,
            gemma: self.gemma,
        })
    }
}

impl Weights<Tensor> {
    #[cfg_attr(
        any(
            feature = "qtip-research",
            not(any(feature = "cuda", feature = "vulkan", all(feature = "metal", target_os = "macos")))
        ),
        allow(dead_code)
    )]
    pub fn open(path: &Path, config: &Config) -> Result<Self> {
        config.validate()?;
        Self::from_gguf(Gguf::open(path, false)?, config)
    }

    /// Rejects QTIP3 tensors: only the CUDA backend executes them (open_qtip3).
    pub(crate) fn from_gguf(model: Gguf, config: &Config) -> Result<Self> {
        if model.has_qtip3() {
            return Err(invalid(
                "QTIP3 tensors need the CUDA backend built with the qtip-research feature",
            ));
        }
        Self::from_gguf_any(model, config)
    }

    /// Like `open`, but accepts QTIP3 projections (research, Llama only).
    #[cfg(feature = "qtip-research")]
    pub(crate) fn open_qtip3(path: &Path, config: &Config) -> Result<Self> {
        config.validate()?;
        let model = Gguf::open(path, false)?;
        if model.has_qtip3() && model.text("general.architecture")? != "llama" {
            return Err(invalid(
                "QTIP3 tensors are supported for the llama architecture only",
            ));
        }
        Self::from_gguf_any(model, config)
    }

    fn from_gguf_any(mut model: Gguf, config: &Config) -> Result<Self> {
        let arch = model.text("general.architecture")?.to_owned();
        let (qwen, moe) = match arch.as_str() {
            "llama" => (false, false),
            "qwen3" => (true, false),
            "qwen3moe" => (true, true),
            "olmoe" => (false, true),
            "gemma2" | "gemma3" | "phi3" | "qwen2" | "starcoder2" | "gpt2" | "bloom"
            | "gptneox" | "olmo" | "olmo2" => (false, false),
            _ => return Err(invalid(format!("unsupported architecture {arch}"))),
        };
        let gemma2 = arch == "gemma2";
        let gemma3 = arch == "gemma3";
        let gemma = gemma2 || gemma3;
        let phi = arch == "phi3";
        let longrope = phi
            && ["rope_factors_short.weight", "rope_factors_long.weight"]
                .iter()
                .any(|name| model.tensors.contains_key(*name));
        let qwen2 = arch == "qwen2";
        let starcoder2 = arch == "starcoder2";
        let gpt2 = arch == "gpt2";
        let bloom = arch == "bloom";
        let gptneox = arch == "gptneox";
        let olmo = arch == "olmo";
        let olmo2 = arch == "olmo2";
        let olmoe = arch == "olmoe";
        if olmoe && model.metadata.contains_key("olmoe.expert_weights_scale") {
            return Err(invalid("olmoe expert weight scaling is not supported"));
        }
        if olmo && model.metadata.contains_key("olmo.attention.clamp_kqv") {
            return Err(invalid("olmo attention clamp is not supported"));
        }
        let affine_ffn = starcoder2 || gpt2 || bloom || gptneox;
        let parallel_residual = if gptneox {
            match model.metadata.get("gptneox.use_parallel_residual") {
                Some(Value::Bool(value)) => *value,
                _ => return Err(invalid("gptneox requires boolean use_parallel_residual")),
            }
        } else {
            false
        };
        let head_norm = qwen || gemma3;
        for suffix in [
            "rope.scaling.type",
            "rope.scaling.factor",
            "rope.scaling.attn_factor",
            "rope.scale_linear",
            "attention.sliding_window",
            "attention.logit_softcap",
            "attn_logit_softcapping",
            "expert_shared_count",
        ] {
            if !(gemma && suffix == "attention.sliding_window")
                && !(gemma2 && suffix == "attn_logit_softcapping")
                && !(longrope && suffix == "rope.scaling.attn_factor")
                && model.metadata.contains_key(&format!("{arch}.{suffix}"))
            {
                return Err(invalid(format!("unsupported model feature {suffix}")));
            }
        }
        if !moe && model.metadata.contains_key(&format!("{arch}.expert_count")) {
            return Err(invalid(format!("{arch} does not support expert_count")));
        }
        let integer = |suffix: &str| model.integer(&format!("{arch}.{suffix}"));
        let width = integer("embedding_length")?;
        let heads = integer("attention.head_count")?;
        let kv_heads = if (gpt2 || gptneox)
            && !model
                .metadata
                .contains_key(&format!("{arch}.attention.head_count_kv"))
        {
            heads
        } else {
            integer("attention.head_count_kv")?
        };
        let count = integer("block_count")?;
        let context = integer("context_length")?;
        let dim = if head_norm || gemma2 {
            integer("attention.key_length")?
        } else {
            width
                .checked_div(heads)
                .ok_or_else(|| invalid("zero head count"))?
        };
        let hidden = integer(if moe && !olmoe {
            "expert_feed_forward_length"
        } else {
            "feed_forward_length"
        })?;
        let (experts, used) = if moe {
            (integer("expert_count")?, integer("expert_used_count")?)
        } else {
            (0, 1)
        };
        let epsilon = model.float(&format!(
            "{arch}.attention.{}",
            if affine_ffn || olmo {
                "layer_norm_epsilon"
            } else {
                "layer_norm_rms_epsilon"
            }
        ))?;
        let rope_key = format!("{arch}.rope.freq_base");
        let rope_base = if gpt2
            || bloom
            || ((phi || gemma2 || gptneox) && !model.metadata.contains_key(&rope_key))
        {
            10_000.0
        } else {
            model.float(&rope_key)?
        };
        if phi || qwen2 || affine_ffn || olmo || olmo2 || olmoe {
            for suffix in ["attention.key_length", "attention.value_length"] {
                if model.metadata.contains_key(&format!("{arch}.{suffix}"))
                    && integer(suffix)? != dim
                {
                    return Err(invalid(format!(
                        "unsupported {arch} attention head dimension"
                    )));
                }
            }
            if !matches!(
                model.metadata.get(&format!("{arch}.attention.causal")),
                None | Some(Value::Bool(true))
            ) {
                return Err(invalid(format!("{arch} requires causal attention")));
            }
        }
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
            || (!head_norm && !gemma2 && width % heads != 0)
            || (!head_norm
                && !gemma2
                && !qwen2
                && !affine_ffn
                && !olmo
                && !olmo2
                && !olmoe
                && integer("rope.dimension_count")? != dim)
            || ((head_norm || gemma2) && integer("attention.value_length")? != dim)
        {
            return Err(invalid(
                "invalid dimensions, rotary layout, experts, or context",
            ));
        }
        let rotary_dim = if gptneox {
            let rotary_dim = integer("rope.dimension_count")?;
            if rotary_dim == 0 || rotary_dim > dim || rotary_dim % 2 != 0 || kv_heads != heads {
                return Err(invalid("invalid gptneox rotary dimension or KV head count"));
            }
            rotary_dim
        } else if let Some(value) = model.metadata.get(&format!("{arch}.rope.dimension_count")) {
            if !matches!(value, Value::Integer(n) if *n == dim as u64) {
                return Err(invalid("partial rotary embeddings are not supported"));
            }
            dim
        } else {
            dim
        };
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
        let gemma_config = if gemma {
            if model
                .metadata
                .contains_key(&format!("{arch}.rope.dimension_count_swa"))
                || model
                    .metadata
                    .contains_key(&format!("{arch}.attention.scale"))
            {
                return Err(invalid(format!(
                    "unsupported {arch} attention or rotary override"
                )));
            }
            if !matches!(
                model.metadata.get(&format!("{arch}.attention.causal")),
                None | Some(Value::Bool(true))
            ) {
                return Err(invalid(format!("{arch} requires causal attention")));
            }
            let window = if model
                .metadata
                .contains_key(&format!("{arch}.attention.sliding_window"))
            {
                integer("attention.sliding_window")?
            } else if gemma2 {
                4096
            } else {
                0
            };
            if window > i32::MAX as usize {
                return Err(invalid(format!(
                    "{arch} sliding window exceeds indexing range"
                )));
            }
            let pattern = model
                .metadata
                .get(&format!("{arch}.attention.sliding_window_pattern"));
            let local_layers = match (window, pattern) {
                (0, _) => vec![false; count],
                (_, Some(Value::Bools(layers))) if layers.len() == count => layers.clone(),
                (_, None | Some(Value::Integer(_))) => {
                    let period = if pattern.is_some() {
                        integer("attention.sliding_window_pattern")?
                    } else if gemma2 {
                        2
                    } else {
                        6
                    };
                    (0..count)
                        .map(|layer| period == 0 || layer % period != period - 1)
                        .collect()
                }
                _ => return Err(invalid(format!("invalid {arch} sliding window pattern"))),
            };
            let local_rope_key = format!("{arch}.rope.freq_base_swa");
            let local_rope_base = if model.metadata.contains_key(&local_rope_key) {
                model.float(&local_rope_key)?
            } else if gemma2 {
                rope_base
            } else {
                10_000.0
            };
            let final_softcap = match model
                .metadata
                .get(&format!("{arch}.final_logit_softcapping"))
            {
                None => 0.0,
                Some(Value::Float(value)) if value.is_finite() && *value >= 0.0 => *value,
                _ => return Err(invalid(format!("invalid {arch} final logit softcap"))),
            };
            let attention_softcap = if gemma2 {
                if count != 26 || final_softcap <= 0.0 {
                    return Err(invalid(
                        "gemma2 requires the 26-layer layout and a positive final softcap",
                    ));
                }
                model.float("gemma2.attn_logit_softcapping")?
            } else {
                0.0
            };
            Some(Gemma {
                local_layers,
                window,
                local_rope_base,
                final_softcap,
                attention_softcap,
            })
        } else {
            None
        };
        let eos = i32::try_from(model.integer("tokenizer.ggml.eos_token_id")?)
            .map_err(|_| invalid("invalid EOG token"))?;
        if eos < 0 || eos as usize >= vocab {
            return Err(invalid("EOG exceeds vocabulary"));
        }
        let mut eog = vec![eos];
        if starcoder2
            && let Some(Value::Strings(tokens)) = model.metadata.get("tokenizer.ggml.tokens")
        {
            for (id, text) in tokens.iter().take(vocab).enumerate() {
                if matches!(text.as_str(), "<|endoftext|>" | "<fim_pad>")
                    && !eog.contains(&(id as i32))
                {
                    eog.push(id as i32);
                }
            }
        }
        if phi && let Some(Value::Strings(tokens)) = model.metadata.get("tokenizer.ggml.tokens") {
            for (id, text) in tokens.iter().take(vocab).enumerate() {
                if crate::tokenizer::phi3_stop_piece(text) && !eog.contains(&(id as i32)) {
                    eog.push(id as i32);
                }
            }
        }
        if arch == "llama"
            && matches!(model.metadata.get("tokenizer.ggml.pre"),
                Some(Value::Text(pre)) if matches!(pre.as_str(), "llama-bpe" | "llama3" | "llama-v3"))
            && let Some(Value::Strings(tokens)) = model.metadata.get("tokenizer.ggml.tokens")
            && let Some(Value::Integers(types)) = model.metadata.get("tokenizer.ggml.token_type")
        {
            for (id, (text, kind)) in tokens.iter().zip(types).take(vocab).enumerate() {
                if *kind == 3
                    && matches!(
                        text.as_str(),
                        "<|end_of_text|>" | "<|eot_id|>" | "<|eom_id|>"
                    )
                    && !eog.contains(&(id as i32))
                {
                    eog.push(id as i32);
                }
            }
        }
        if qwen || qwen2 || olmo2 {
            if let Some(value) = model.metadata.get("tokenizer.ggml.tokens") {
                let Value::Strings(tokens) = value else {
                    return Err(invalid("invalid Qwen vocabulary"));
                };
                let Some(Value::Integers(types)) = model.metadata.get("tokenizer.ggml.token_type")
                else {
                    return Err(invalid("missing Qwen token types"));
                };
                if tokens.len() != vocab || types.len() != vocab {
                    return Err(invalid("vocabulary/types do not match embeddings"));
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
        if gemma && let Some(Value::Strings(tokens)) = model.metadata.get("tokenizer.ggml.tokens") {
            let eot = match model.metadata.get("tokenizer.ggml.eot_token_id") {
                Some(Value::Integer(id)) if *id < vocab as u64 => Some(*id as usize),
                None => tokens.iter().position(|token| token == "<end_of_turn>"),
                _ => return Err(invalid("invalid gemma3 end-of-turn token ID")),
            };
            if let Some(id) = eot.filter(|&id| id < vocab && id != eos as usize) {
                eog.push(id as i32);
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
            architecture: arch.clone(),
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
        let embedding_norm = if bloom {
            Some([
                model.tensor("token_embd_norm.weight", &[width])?,
                model.tensor("token_embd_norm.bias", &[width])?,
            ])
        } else {
            None
        };
        let position_embedding = gpt2
            .then(|| model.tensor("position_embd.weight", &[width, context]))
            .transpose()?;
        let output_norm = (!olmo)
            .then(|| model.tensor("output_norm.weight", &[width]))
            .transpose()?;
        let output_norm_bias = affine_ffn
            .then(|| model.tensor("output_norm.bias", &[width]))
            .transpose()?;
        if gemma2 && model.tensors.contains_key("output.weight") {
            return Err(invalid("gemma2 requires tied output weights"));
        }
        let output = if model.tensors.contains_key("output.weight") {
            Some(model.tensor("output.weight", &[width, vocab])?)
        } else {
            None
        };
        let read_divisors = |model: &mut Gguf, name| {
            let factors = model.tensor(name, &[dim / 2])?;
            if !matches!(&factors.storage, Storage::F32(values)
                if values.iter().all(|v| v.is_finite() && *v > 0.0))
            {
                return Err(invalid(
                    "RoPE frequency divisors must be positive finite F32",
                ));
            }
            Ok(factors)
        };
        let (rope_factors, rope_attn_factor) = if longrope {
            let short = read_divisors(&mut model, "rope_factors_short.weight")?;
            let long = read_divisors(&mut model, "rope_factors_long.weight")?;
            let original = model.integer("phi3.rope.scaling.original_context_length")?;
            let magnitude = model.float("phi3.rope.scaling.attn_factor")?;
            if original == 0 || original >= context || !magnitude.is_finite() || magnitude <= 0.0 {
                return Err(invalid(
                    "invalid Phi-3 LongRoPE context or attention factor",
                ));
            }
            // The pinned oracle selects once using its padded single-sequence context,
            // never at a token position (cached keys must use the same factors).
            let padded_context = u64::from(config.context_tokens).div_ceil(256) * 256;
            let factors = if padded_context > original as u64 {
                long
            } else {
                short
            };
            (Some(factors), magnitude)
        } else if arch == "llama" && model.tensors.contains_key("rope_freqs.weight") {
            (Some(read_divisors(&mut model, "rope_freqs.weight")?), 1.0)
        } else {
            (None, 1.0)
        };
        let mut layers = Vec::with_capacity(count);
        for i in 0..count {
            let qkv_bias = if gpt2 || bloom || gptneox {
                let fused =
                    model.tensor(&format!("blk.{i}.attn_qkv.bias"), &[q_width + 2 * kv_width])?;
                let (q, kv) = fused.split_vector(q_width)?;
                let (k, v) = kv.split_vector(kv_width)?;
                Some([q, k, v])
            } else if qwen2 || starcoder2 {
                Some([
                    model.tensor(&format!("blk.{i}.attn_q.bias"), &[q_width])?,
                    model.tensor(&format!("blk.{i}.attn_k.bias"), &[kv_width])?,
                    model.tensor(&format!("blk.{i}.attn_v.bias"), &[kv_width])?,
                ])
            } else {
                None
            };
            let affine = if affine_ffn {
                let mut bias = |name, size| model.tensor(&format!("blk.{i}.{name}.bias"), &[size]);
                Some(AffineLayer {
                    norm: bias("attn_norm", width)?,
                    attention: bias("attn_output", width)?,
                    ffn_norm: bias("ffn_norm", width)?,
                    up: bias("ffn_up", hidden)?,
                    down: bias("ffn_down", width)?,
                })
            } else {
                None
            };
            let mut tensor =
                |name: &str, dims: &[usize]| model.tensor(&format!("blk.{i}.{name}.weight"), dims);
            let (q, k, v) = if phi || gpt2 || bloom || gptneox {
                let (q, kv) =
                    tensor("attn_qkv", &[width, q_width + 2 * kv_width])?.split_rows(q_width)?;
                let (k, v) = kv.split_rows(kv_width)?;
                (q, k, v)
            } else {
                (
                    tensor("attn_q", &[width, q_width])?,
                    tensor("attn_k", &[width, kv_width])?,
                    tensor("attn_v", &[width, kv_width])?,
                )
            };
            let (gate, up) = if phi {
                let (gate, up) = tensor("ffn_up", &[width, 2 * hidden])?.split_rows(hidden)?;
                (Some(gate), up)
            } else if moe {
                (
                    Some(tensor("ffn_gate_exps", &[width, hidden, experts])?),
                    tensor("ffn_up_exps", &[width, hidden, experts])?,
                )
            } else {
                (
                    if affine_ffn {
                        None
                    } else {
                        Some(tensor("ffn_gate", &[width, hidden])?)
                    },
                    tensor("ffn_up", &[width, hidden])?,
                )
            };
            layers.push(Layer {
                norm: (!olmo && !olmo2)
                    .then(|| tensor("attn_norm", &[width]))
                    .transpose()?,
                q,
                k,
                v,
                qkv_bias,
                output: tensor("attn_output", &[q_width, width])?,
                q_norm: if olmo2 || olmoe {
                    Some(tensor("attn_q_norm", &[q_width])?)
                } else if head_norm {
                    Some(tensor("attn_q_norm", &[dim])?)
                } else {
                    None
                },
                k_norm: if olmo2 || olmoe {
                    Some(tensor("attn_k_norm", &[kv_width])?)
                } else if head_norm {
                    Some(tensor("attn_k_norm", &[dim])?)
                } else {
                    None
                },
                ffn_norm: (!olmo && !olmo2)
                    .then(|| tensor("ffn_norm", &[width]))
                    .transpose()?,
                router: if moe {
                    Some(tensor("ffn_gate_inp", &[width, experts])?)
                } else {
                    None
                },
                gate,
                up,
                down: if moe {
                    tensor("ffn_down_exps", &[hidden, width, experts])?
                } else {
                    tensor("ffn_down", &[hidden, width])?
                },
                post_norms: if gemma || olmo2 {
                    Some([
                        tensor("post_attention_norm", &[width])?,
                        tensor("post_ffw_norm", &[width])?,
                    ])
                } else {
                    None
                },
                affine,
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
                rotary_dim,
                parallel_residual,
                hidden,
                experts,
                used,
                epsilon,
                rope_base,
                rope_attn_factor,
                neox: head_norm
                    || phi
                    || qwen2
                    || gemma2
                    || starcoder2
                    || gptneox
                    || olmo2
                    || olmoe,
                eog,
                tokenizer,
            },
            embedding,
            position_embedding,
            embedding_norm,
            output_norm,
            output_norm_bias,
            output,
            rope_factors,
            layers,
            gemma: gemma_config,
        })
    }

    #[cfg_attr(not(any(feature = "cuda", feature = "vulkan", all(feature = "metal", target_os = "macos"))), allow(dead_code))]
    pub fn bytes(&self) -> u64 {
        let mut bytes = self.embedding.allocated_bytes()
            + self
                .position_embedding
                .as_ref()
                .map_or(0, Tensor::allocated_bytes)
            + self.output_norm.as_ref().map_or(0, Tensor::allocated_bytes)
            + self
                .output_norm_bias
                .as_ref()
                .map_or(0, Tensor::allocated_bytes)
            + self.output.as_ref().map_or(0, Tensor::allocated_bytes)
            + self
                .rope_factors
                .as_ref()
                .map_or(0, Tensor::allocated_bytes);
        if let Some([weight, bias]) = &self.embedding_norm {
            bytes += weight.allocated_bytes() + bias.allocated_bytes();
        }
        for l in &self.layers {
            for t in [&l.q, &l.k, &l.v, &l.output, &l.up, &l.down] {
                bytes += t.allocated_bytes();
            }
            for t in [
                &l.norm,
                &l.ffn_norm,
                &l.q_norm,
                &l.k_norm,
                &l.router,
                &l.gate,
            ]
            .into_iter()
            .flatten()
            {
                bytes += t.allocated_bytes();
            }
            if let Some(b) = &l.affine {
                bytes += [&b.norm, &b.attention, &b.ffn_norm, &b.up, &b.down]
                    .into_iter()
                    .map(Tensor::allocated_bytes)
                    .sum::<u64>();
            }
            if let Some([a, b]) = &l.post_norms {
                bytes += a.allocated_bytes() + b.allocated_bytes();
            }
            if let Some(biases) = &l.qkv_bias {
                bytes += biases.iter().map(Tensor::allocated_bytes).sum::<u64>();
            }
        }
        bytes
    }
}

pub(crate) fn route(scores: &[f32], used: usize) -> Result<Vec<(usize, f32)>> {
    route_softmax(scores, used, true)
}

pub(crate) fn route_softmax(
    scores: &[f32],
    used: usize,
    normalize_selected: bool,
) -> Result<Vec<(usize, f32)>> {
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
    let sum: f32 = if normalize_selected {
        ids.iter().map(|&id| (scores[id] - max).exp()).sum()
    } else {
        scores.iter().map(|&score| (score - max).exp()).sum()
    };
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
