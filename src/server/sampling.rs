//! Sampling fields of OpenAI requests: `temperature`, `top_p`, `seed`,
//! `presence_penalty`, `frequency_penalty` and `logit_bias`, plus the
//! llama.cpp extensions `top_k`, `min_p`, `repeat_penalty` and `repeat_last_n`.
//!
//! `logit_bias` takes OpenAI's object (`{"15339": -100}`) or llama.cpp's
//! array of pairs (`[[15339, -100]]`); a bias of `false` bans the token.
//! Unlike llama.cpp, token ids outside the vocabulary and string keys that are
//! not token ids are rejected rather than ignored.

use palito::sampling::Sampling;
use serde::Deserialize;
use serde_json::Value;

#[derive(Deserialize, Default)]
pub struct SamplingFields {
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub top_k: Option<u32>,
    pub min_p: Option<f32>,
    pub repeat_penalty: Option<f32>,
    pub repeat_last_n: Option<u32>,
    pub presence_penalty: Option<f32>,
    pub frequency_penalty: Option<f32>,
    pub logit_bias: Option<Value>,
    pub seed: Option<u64>,
}

impl SamplingFields {
    /// Request settings over the server defaults, validated for a model with
    /// `vocab_tokens` tokens.
    pub fn resolve(&self, defaults: &Sampling, vocab_tokens: u32) -> Result<Sampling, String> {
        let logit_bias = match &self.logit_bias {
            None | Some(Value::Null) => defaults.logit_bias.clone(),
            Some(value) => logit_bias(value)?,
        };
        let sampling = Sampling {
            temperature: self.temperature.unwrap_or(defaults.temperature),
            top_p: self.top_p.unwrap_or(defaults.top_p),
            top_k: self.top_k.unwrap_or(defaults.top_k),
            min_p: self.min_p.unwrap_or(defaults.min_p),
            repeat_penalty: self.repeat_penalty.unwrap_or(defaults.repeat_penalty),
            repeat_last_n: self.repeat_last_n.unwrap_or(defaults.repeat_last_n),
            presence_penalty: self.presence_penalty.unwrap_or(defaults.presence_penalty),
            frequency_penalty: self.frequency_penalty.unwrap_or(defaults.frequency_penalty),
            logit_bias,
            seed: self.seed.or(defaults.seed),
        };
        sampling
            .validate()
            .and_then(|()| sampling.check_vocabulary(vocab_tokens))
            .map_err(|error| error.to_string())?;
        Ok(sampling)
    }
}

fn logit_bias(value: &Value) -> Result<Vec<(u32, f32)>, String> {
    let entry = |token: Option<u32>, bias: &Value| Some((token?, bias_value(bias)?));
    let entries: Option<Vec<_>> = match value {
        Value::Object(map) => map
            .iter()
            .map(|(token, bias)| entry(token.parse().ok(), bias))
            .collect(),
        Value::Array(items) => items
            .iter()
            .map(|item| match item.as_array().map(Vec::as_slice) {
                Some([token, bias]) => {
                    entry(token.as_u64().and_then(|t| u32::try_from(t).ok()), bias)
                }
                _ => None,
            })
            .collect(),
        _ => None,
    };
    entries.ok_or_else(|| "logit_bias must map token ids to numbers (or false to ban them)".into())
}

fn bias_value(bias: &Value) -> Option<f32> {
    match bias {
        Value::Bool(false) => Some(f32::NEG_INFINITY),
        Value::Number(number) => number.as_f64().map(|b| b as f32),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn fields(body: Value) -> SamplingFields {
        serde_json::from_value(body).unwrap()
    }

    #[test]
    fn request_fields_override_the_server_defaults() {
        let defaults = Sampling { repeat_penalty: 1.1, min_p: 0.05, ..Sampling::GREEDY };
        let resolved = fields(json!({
            "min_p": 0.1, "presence_penalty": 0.5, "frequency_penalty": -0.25,
            "repeat_last_n": 32, "logit_bias": {"7": -100, "9": false}
        }))
        .resolve(&defaults, 10)
        .unwrap();
        assert_eq!(resolved.min_p, 0.1);
        assert_eq!((resolved.repeat_penalty, resolved.repeat_last_n), (1.1, 32));
        assert_eq!((resolved.presence_penalty, resolved.frequency_penalty), (0.5, -0.25));
        let mut bias = resolved.logit_bias.clone();
        bias.sort_by_key(|pair| pair.0);
        assert_eq!(bias, vec![(7, -100.0), (9, f32::NEG_INFINITY)]);
        let kept = fields(json!({})).resolve(&defaults, 10).unwrap();
        assert_eq!(kept, defaults);
        let array = fields(json!({"logit_bias": [[3, 2.5]]})).resolve(&defaults, 10).unwrap();
        assert_eq!(array.logit_bias, vec![(3, 2.5)]);
    }

    #[test]
    fn rejects_bad_sampling_fields() {
        for body in [
            json!({"min_p": 2}),
            json!({"repeat_penalty": 0}),
            json!({"logit_bias": {"hello": 1}}),
            json!({"logit_bias": {"3": "x"}}),
            json!({"logit_bias": {"10": 1}}),
            json!({"logit_bias": [[1]]}),
            json!({"logit_bias": [[-1, 1]]}),
            json!({"logit_bias": 5}),
        ] {
            assert!(fields(body.clone()).resolve(&Sampling::GREEDY, 10).is_err(), "{body}");
        }
        assert!(serde_json::from_value::<SamplingFields>(json!({"repeat_last_n": -1})).is_err());
    }
}
