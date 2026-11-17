use crate::{GenerationMode, Result};
use serde::Serialize;

#[derive(Clone, Debug, Serialize)]
pub struct Capabilities {
    pub token_step: bool,
    pub chunked_prefill: bool,
}

#[derive(Clone, Debug, Serialize)]
pub struct ModelInfo {
    pub backend: String,
    pub revision: String,
    pub architecture: String,
    pub tensor_types: Vec<String>,
    pub weight_bytes: u64,
    pub parameters: u64,
    pub context_tokens: u32,
    pub batch_tokens: u32,
    pub vocab_tokens: u32,
    pub capabilities: Capabilities,
}

#[derive(Clone, Debug, Serialize)]
pub struct MemoryUsage {
    pub allocation_domain: &'static str,
    pub weights_allocated_bytes: u64,
    pub kv_allocated_bytes: u64,
    pub kv_live_bytes: u64,
    pub workspace_allocated_bytes: u64,
    pub logits_allocated_bytes: u64,
    pub token_state_allocated_bytes: u64,
    pub prefix_capacity_bytes: u64,
    pub prefix_allocated_bytes: u64,
    pub worker_scratch_bound_bytes: u64,
    pub numerical_reservation_bytes: u64,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct SpeculationStats {
    pub proposed_tokens: u64,
    pub accepted_tokens: u64,
    pub verification_batches: u64,
}

/// A single-owner token-step backend. No native types escape this boundary.
///
/// Adapters must preserve token IDs and explicit positions, return errors instead
/// of silently converting weights, and complete device work before returning
/// from `evaluate`. Reset must discard active request state, including failed work.
/// An explicitly enabled immutable prefix cache may survive reset.
/// A backend without token-step access must report that capability as false.
pub trait Backend {
    fn info(&self) -> &ModelInfo;
    fn logits(&self) -> Result<&[f32]> {
        Err(crate::Error::Backend(
            "backend does not expose logits".into(),
        ))
    }
    fn tokenize(&self, text: &str) -> Result<Vec<i32>>;
    fn detokenize(&self, tokens: &[i32]) -> Result<String>;
    fn reset(&mut self) -> Result<()>;
    fn evaluate(&mut self, tokens: &[i32], position: u32) -> Result<()>;
    /// Complete evaluation for greedy generation. Full host logits need not be
    /// materialized; callers requiring them must use `evaluate` instead.
    fn evaluate_for_sampling(&mut self, tokens: &[i32], position: u32) -> Result<()> {
        self.evaluate(tokens, position)
    }
    fn sample_greedy(&mut self) -> Result<i32>;
    fn is_eog(&self, token: i32) -> bool;
    fn memory_usage(&self) -> Option<MemoryUsage> {
        None
    }
    fn restore_prefix(&mut self, _prompt: &[i32]) -> Result<usize> {
        Ok(0)
    }
    fn store_prefix(&mut self, _prompt: &[i32]) -> Result<()> {
        Ok(())
    }
    /// Returns sampled tokens with only the final token left unevaluated.
    fn sample_tokens(&mut self, maximum: usize, _mode: GenerationMode) -> Result<Vec<i32>> {
        if maximum == 0 {
            return Err(crate::Error::Invalid(
                "sample maximum must be positive".into(),
            ));
        }
        Ok(vec![self.sample_greedy()?])
    }
    fn speculation_stats(&self) -> Option<SpeculationStats> {
        None
    }
}

impl<B: Backend + ?Sized> Backend for Box<B> {
    fn info(&self) -> &ModelInfo {
        (**self).info()
    }
    fn logits(&self) -> Result<&[f32]> {
        (**self).logits()
    }
    fn tokenize(&self, text: &str) -> Result<Vec<i32>> {
        (**self).tokenize(text)
    }
    fn detokenize(&self, tokens: &[i32]) -> Result<String> {
        (**self).detokenize(tokens)
    }
    fn reset(&mut self) -> Result<()> {
        (**self).reset()
    }
    fn evaluate(&mut self, tokens: &[i32], position: u32) -> Result<()> {
        (**self).evaluate(tokens, position)
    }
    fn evaluate_for_sampling(&mut self, tokens: &[i32], position: u32) -> Result<()> {
        (**self).evaluate_for_sampling(tokens, position)
    }
    fn sample_greedy(&mut self) -> Result<i32> {
        (**self).sample_greedy()
    }
    fn is_eog(&self, token: i32) -> bool {
        (**self).is_eog(token)
    }
    fn memory_usage(&self) -> Option<MemoryUsage> {
        (**self).memory_usage()
    }
    fn restore_prefix(&mut self, prompt: &[i32]) -> Result<usize> {
        (**self).restore_prefix(prompt)
    }
    fn store_prefix(&mut self, prompt: &[i32]) -> Result<()> {
        (**self).store_prefix(prompt)
    }
    fn sample_tokens(&mut self, maximum: usize, mode: GenerationMode) -> Result<Vec<i32>> {
        (**self).sample_tokens(maximum, mode)
    }
    fn speculation_stats(&self) -> Option<SpeculationStats> {
        (**self).speculation_stats()
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct RequestBackendInfo {
    pub backend: String,
    pub endpoint: String,
    pub model: String,
    pub model_path: String,
    pub tokenizer_path: Option<String>,
    pub model_type: Option<String>,
    pub weight_version: String,
    pub server_version: String,
    pub kernel_method: String,
    pub cpu_weight_path: String,
    pub cpu_threads: u32,
    pub num_gpu_experts: Option<u32>,
    pub deferred_experts_per_token: u32,
    pub context_tokens: Option<u32>,
    pub prefix_cache_enabled: bool,
    pub capabilities: Capabilities,
}

#[derive(Serialize)]
#[serde(tag = "kind", content = "value", rename_all = "snake_case")]
pub enum GenerationInput {
    Text(String),
    TokenIds(Vec<i32>),
}

pub struct CompletionRequest {
    pub input: GenerationInput,
    pub max_tokens: u32,
    pub mode: GenerationMode,
}

#[derive(Debug, Serialize)]
pub struct Completion {
    pub text: String,
    pub generated_tokens: Vec<i32>,
    pub prompt_tokens: u32,
    pub cached_tokens: u32,
    pub finish_reason: String,
    pub elapsed_ns: u64,
}

/// A whole-request service owns token stepping, batching, and KV lifetimes.
/// It must not be presented as a token-step `Backend` or used by `Executor`.
/// Timings include transport and service scheduling, not isolated kernel phases.
pub trait RequestBackend {
    fn info(&self) -> &RequestBackendInfo;
    fn generate(&mut self, request: &CompletionRequest) -> Result<Completion>;
}
