pub mod backend;
pub mod chat;
pub mod engine;
pub mod ffi;
pub mod stream;
pub mod cpu;
#[cfg(feature = "cuda")]
pub mod cuda;
mod gguf;
mod kernels;
#[cfg(feature = "ktransformers")]
pub mod ktransformers;
pub mod tokenizer;
#[cfg(feature = "cuda")]
mod transformer;

use backend::{Backend, ModelInfo};
use serde::Serialize;
use std::time::Instant;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum Error {
    #[error("invalid request or configuration: {0}")]
    Invalid(String),
    #[error("backend failure: {0}")]
    Backend(String),
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error(transparent)]
    Json(#[from] serde_json::Error),
    /// A streaming callback asked to stop generation.
    #[error("generation cancelled")]
    Cancelled,
}

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Clone, Debug, Serialize)]
pub struct Config {
    pub context_tokens: u32,
    pub batch_tokens: u32,
    pub prefill_chunk: u32,
    pub threads: u32,
}

impl Config {
    pub fn validate(&self) -> Result<()> {
        if self.context_tokens == 0
            || self.context_tokens > i32::MAX as u32
            || self.batch_tokens == 0
            || self.batch_tokens > self.context_tokens
            || self.prefill_chunk == 0
            || self.prefill_chunk > self.batch_tokens
            || self.threads == 0
            || self.threads > i32::MAX as u32
        {
            return Err(Error::Invalid(
                "require 0 < chunk <= batch <= context <= i32::MAX and 0 < threads <= i32::MAX"
                    .into(),
            ));
        }
        Ok(())
    }
}

/// Trained context length from GGUF metadata (`<architecture>.context_length`),
/// read from the file header only.
pub fn model_context_tokens(path: &std::path::Path) -> Result<u32> {
    let metadata = gguf::Gguf::read_metadata(path)?;
    let arch = match metadata.get("general.architecture") {
        Some(gguf::Value::Text(arch)) => arch.clone(),
        _ => return Err(Error::Invalid("GGUF: missing general.architecture".into())),
    };
    match metadata.get(&format!("{arch}.context_length")) {
        Some(gguf::Value::Integer(n)) if *n > 0 => Ok(u32::try_from(*n).unwrap_or(u32::MAX)),
        _ => Err(Error::Invalid(format!("GGUF: missing {arch}.context_length"))),
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum GenerationMode {
    UntilEog,
    FixedLength,
}

#[derive(Debug, Serialize)]
pub struct Sample {
    pub prefill_ns: u64,
    pub ttft_ns: u64,
    pub decode_ns: u64,
    pub decode_steps: usize,
    pub generated_tokens: Vec<i32>,
    pub cached_prompt_tokens: usize,
    pub speculation: Option<backend::SpeculationStats>,
}

pub struct Executor<B: Backend> {
    backend: B,
    config: Config,
}

impl<B: Backend> Executor<B> {
    pub fn new(backend: B, config: Config) -> Result<Self> {
        config.validate()?;
        let info = backend.info();
        if !info.capabilities.token_step
            || (!info.capabilities.chunked_prefill && config.prefill_chunk < config.context_tokens)
        {
            return Err(Error::Invalid(
                "backend lacks token-step or requested chunked-prefill capability".into(),
            ));
        }
        if config.context_tokens > info.context_tokens || config.batch_tokens > info.batch_tokens {
            return Err(Error::Invalid(
                "requested plan exceeds backend context or batch capacity".into(),
            ));
        }
        Ok(Self { backend, config })
    }

    pub fn info(&self) -> &ModelInfo {
        self.backend.info()
    }

    pub fn memory_usage(&self) -> Option<backend::MemoryUsage> {
        self.backend.memory_usage()
    }

    pub fn tokenize(&self, text: &str) -> Result<Vec<i32>> {
        self.backend.tokenize(text)
    }

    pub fn detokenize(&self, tokens: &[i32]) -> Result<String> {
        self.backend.detokenize(tokens)
    }

    pub fn is_eog(&self, token: i32) -> bool {
        self.backend.is_eog(token)
    }

    pub fn generate(
        &mut self,
        prompt: &[i32],
        max_tokens: u32,
        mode: GenerationMode,
    ) -> Result<Sample> {
        self.generate_streaming(prompt, max_tokens, mode, |_| Ok(()))
    }

    /// `generate`, passing each generated token to `on_token` as soon as it is
    /// sampled (same tokens, same order). An `on_token` error aborts the request.
    pub fn generate_streaming(
        &mut self,
        prompt: &[i32],
        max_tokens: u32,
        mode: GenerationMode,
        mut on_token: impl FnMut(i32) -> Result<()>,
    ) -> Result<Sample> {
        let evaluated = prompt
            .len()
            .checked_add(max_tokens.saturating_sub(1) as usize);
        if prompt.is_empty()
            || max_tokens == 0
            || evaluated.is_none_or(|n| n > self.config.context_tokens as usize)
            || prompt
                .iter()
                .any(|&t| t < 0 || t as u32 >= self.info().vocab_tokens)
        {
            return Err(Error::Invalid(
                "empty prompt, invalid token/count, or prompt + generated - 1 exceeds context"
                    .into(),
            ));
        }
        // Every request, including a retry after partial native work, starts clean.
        self.backend.reset()?;
        let start = Instant::now();
        let cached_prompt_tokens = self.backend.restore_prefix(prompt)?;
        if cached_prompt_tokens > prompt.len() {
            return Err(Error::Backend(
                "prefix restoration exceeds prompt length".into(),
            ));
        }
        for (index, chunk) in prompt[cached_prompt_tokens..]
            .chunks(self.config.prefill_chunk as usize)
            .enumerate()
        {
            let position =
                (cached_prompt_tokens + index * self.config.prefill_chunk as usize) as u32;
            self.backend.evaluate_for_sampling(chunk, position)?;
        }
        self.backend.store_prefix(prompt)?;
        let prefill_ns = nanos(start);
        let first = self.backend.sample_greedy()?;
        let ttft_ns = nanos(start);
        on_token(first)?;
        let mut generated_tokens = vec![first];
        let decode_start = Instant::now();
        while generated_tokens.len() < max_tokens as usize {
            let last = *generated_tokens.last().expect("first token exists");
            if mode == GenerationMode::UntilEog && self.backend.is_eog(last) {
                break;
            }
            let position = (prompt.len() + generated_tokens.len() - 1) as u32;
            self.backend.evaluate_for_sampling(&[last], position)?;
            let maximum = max_tokens as usize - generated_tokens.len();
            let sampled = self
                .backend
                .sample_tokens_streaming(maximum, mode, &mut on_token)?;
            if sampled.is_empty()
                || sampled.len() > maximum
                || sampled
                    .iter()
                    .any(|&token| token < 0 || token as u32 >= self.info().vocab_tokens)
            {
                return Err(Error::Backend("invalid sampled token batch".into()));
            }
            generated_tokens.extend(sampled);
        }
        let decode_steps = generated_tokens.len() - 1;
        Ok(Sample {
            prefill_ns,
            ttft_ns,
            decode_ns: if decode_steps == 0 {
                0
            } else {
                nanos(decode_start)
            },
            decode_steps,
            generated_tokens,
            cached_prompt_tokens,
            speculation: self.backend.speculation_stats(),
        })
    }
}

fn nanos(start: Instant) -> u64 {
    u64::try_from(start.elapsed().as_nanos()).expect("elapsed time exceeds 584 years")
}

#[cfg(test)]
mod tests {
    use super::*;
    use backend::Capabilities;

    struct Fake {
        info: ModelInfo,
        calls: Vec<(Vec<i32>, u32)>,
        position: u32,
        resets: usize,
        fail_once: bool,
    }

    impl Fake {
        fn new() -> Self {
            Self {
                info: ModelInfo {
                    backend: "test".into(),
                    revision: "test".into(),
                    architecture: "test".into(),
                    tensor_types: vec![],
                    weight_bytes: 0,
                    parameters: 0,
                    context_tokens: 8,
                    batch_tokens: 3,
                    vocab_tokens: 32,
                    capabilities: Capabilities {
                        token_step: true,
                        chunked_prefill: true,
                    },
                },
                calls: vec![],
                position: 0,
                resets: 0,
                fail_once: false,
            }
        }
    }

    impl Backend for Fake {
        fn info(&self) -> &ModelInfo {
            &self.info
        }
        fn tokenize(&self, _: &str) -> Result<Vec<i32>> {
            Ok(vec![1])
        }
        fn detokenize(&self, _: &[i32]) -> Result<String> {
            Ok(String::new())
        }
        fn reset(&mut self) -> Result<()> {
            self.position = 0;
            self.resets += 1;
            Ok(())
        }
        fn evaluate(&mut self, tokens: &[i32], position: u32) -> Result<()> {
            assert_eq!(self.position, position);
            self.position += tokens.len() as u32;
            self.calls.push((tokens.to_vec(), position));
            if std::mem::take(&mut self.fail_once) {
                return Err(Error::Backend("injected partial evaluation".into()));
            }
            Ok(())
        }
        fn sample_greedy(&mut self) -> Result<i32> {
            Ok(self.position as i32)
        }
        fn is_eog(&self, token: i32) -> bool {
            token == 5
        }
    }

    fn config() -> Config {
        Config {
            context_tokens: 8,
            batch_tokens: 3,
            prefill_chunk: 3,
            threads: 1,
        }
    }

    #[test]
    fn chunks_prefill_and_advances_decode_positions() {
        let mut executor = Executor::new(Fake::new(), config()).unwrap();
        let sample = executor
            .generate(&[1, 2, 3, 4], 3, GenerationMode::FixedLength)
            .unwrap();
        assert_eq!(
            executor.backend.calls,
            vec![(vec![1, 2, 3], 0), (vec![4], 3), (vec![4], 4), (vec![5], 5)]
        );
        assert_eq!(sample.generated_tokens, vec![4, 5, 6]);
        assert_eq!(sample.decode_steps, 2);
    }

    #[test]
    fn streaming_reports_every_generated_token_and_sink_errors_abort() {
        let mut executor = Executor::new(Fake::new(), config()).unwrap();
        let mut streamed = Vec::new();
        let sample = executor
            .generate_streaming(&[1, 2, 3, 4], 3, GenerationMode::FixedLength, |token| {
                streamed.push(token);
                Ok(())
            })
            .unwrap();
        assert_eq!(streamed, sample.generated_tokens);
        let mut seen = 0;
        let aborted =
            executor.generate_streaming(&[1, 2], 4, GenerationMode::FixedLength, |_| {
                seen += 1;
                if seen == 2 {
                    Err(Error::Invalid("closed output".into()))
                } else {
                    Ok(())
                }
            });
        assert!(aborted.is_err());
        assert_eq!(seen, 2);
        // The next request starts clean after the aborted one.
        let again = executor
            .generate(&[1, 2, 3, 4], 3, GenerationMode::FixedLength)
            .unwrap();
        assert_eq!(again.generated_tokens, sample.generated_tokens);
    }
    #[test]
    fn admits_exact_capacity_and_rejects_overflow_before_reset() {
        let mut executor = Executor::new(Fake::new(), config()).unwrap();
        executor
            .generate(&[1; 6], 3, GenerationMode::FixedLength)
            .unwrap();
        assert!(
            executor
                .generate(&[1; 6], 4, GenerationMode::FixedLength)
                .is_err()
        );
        assert_eq!(executor.backend.resets, 1);
        for (prompt, count) in [(&[][..], 1), (&[1][..], 0), (&[-1][..], 1), (&[32][..], 1)] {
            assert!(
                executor
                    .generate(prompt, count, GenerationMode::FixedLength)
                    .is_err()
            );
        }
    }

    #[test]
    fn next_request_resets_after_partial_failure() {
        let mut fake = Fake::new();
        fake.fail_once = true;
        let mut executor = Executor::new(fake, config()).unwrap();
        assert!(
            executor
                .generate(&[1, 2], 2, GenerationMode::FixedLength)
                .is_err()
        );
        let sample = executor
            .generate(&[1, 2], 1, GenerationMode::FixedLength)
            .unwrap();
        assert_eq!(sample.generated_tokens, vec![2]);
        assert_eq!(sample.decode_steps, 0);
        assert_eq!(sample.decode_ns, 0);
        assert_eq!(executor.backend.resets, 2);
    }

    #[test]
    fn fixed_length_and_eog_modes_are_distinct() {
        let mut executor = Executor::new(Fake::new(), config()).unwrap();
        let sample = executor
            .generate(&[1; 5], 3, GenerationMode::UntilEog)
            .unwrap();
        assert_eq!(sample.generated_tokens, vec![5]);
        let fixed = executor
            .generate(&[1; 5], 3, GenerationMode::FixedLength)
            .unwrap();
        assert_eq!(fixed.generated_tokens, vec![5, 6, 7]);
    }

    #[test]
    fn rejects_unsupported_backend_and_invalid_plan() {
        let mut fake = Fake::new();
        fake.info.capabilities.token_step = false;
        assert!(Executor::new(fake, config()).is_err());
        let mut bad = config();
        bad.prefill_chunk = 4;
        assert!(Executor::new(Fake::new(), bad).is_err());
        let mut smaller = Fake::new();
        smaller.info.context_tokens = 4;
        assert!(Executor::new(smaller, config()).is_err());
    }
}
