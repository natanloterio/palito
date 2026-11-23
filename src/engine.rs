//! High-level embedding API: open a GGUF model once, then run chat or plain
//! completion with streamed text.
//!
//! `Engine::open` applies Palito's defaults: the model's trained context (CUDA:
//! at most 65535, shrunk to fit free device memory), the fastest CUDA profile
//! that opens for the model, and prefill chunks of up to 4096 tokens. `palito
//! chat`, `palito serve` and the C API (`crate::ffi`) all build on this type.

mod open;
mod output;
mod slots;
mod text;
mod tools;

pub use output::SlotReply;
pub use slots::SlotEngine;
pub use tools::ToolCompletion;

pub(crate) use output::{ReplyOutput, RunStats};

use crate::{
    Config, Error, Executor, GenerationMode, Result, Sample,
    backend::Backend,
    chat::{ChatTemplate, Message},
    grammar::{Constraint, JsonDecoder, JsonSchema, LogitConstraint},
    sampling::{Sampler, Sampling},
    stop,
    tokenizer::Tokenizer,
};
use output::{ReplyPlan, TextOutput};
use std::{
    path::{Path, PathBuf},
    sync::Arc,
    time::Instant,
};
use text::{TextSink, TrimmedText};

const DEFAULT_THREADS: u32 = 4;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Device {
    Cpu,
    Cuda,
    Vulkan,
}

#[derive(Clone, Debug)]
pub struct EngineOptions {
    /// Default: CUDA when built with the `cuda` feature, else CPU.
    pub device: Device,
    /// None: the model's trained context (CUDA: fitted to device memory).
    pub context_tokens: Option<u32>,
    pub threads: u32,
    /// None: 4096 with the tiled CUDA kernels, else 128.
    pub batch_tokens: Option<u32>,
    /// None: 4096 with the tiled CUDA kernels, else 64 (at most the batch).
    pub prefill_chunk: Option<u32>,
    pub gpu_device: usize,
    /// None: free device memory minus the 2 GiB headroom.
    pub gpu_memory_limit_bytes: Option<u64>,
    /// None: $PALITO_CUDA_KERNEL_CACHE, else $XDG_CACHE_HOME (or ~/.cache)/palito/kernels.
    pub kernel_cache_dir: Option<PathBuf>,
    /// None: try int8-fast, then f32-tiled, then f32. Some: only this profile.
    #[cfg(feature = "cuda")]
    pub cuda_matmul: Option<crate::cuda::CudaMatmul>,
    /// None: on for the tiled profiles, dropped if it cannot be captured.
    #[cfg(feature = "cuda")]
    pub decode_graph: Option<bool>,
    /// Jinja source replacing the GGUF `tokenizer.chat_template`.
    pub chat_template: Option<String>,
    /// CUDA: reuse the previous prompt's K/V when the next prompt starts with it
    /// (a chat resent with one more turn). Greedy output is unchanged.
    pub prefix_cache: bool,
}

impl Default for EngineOptions {
    fn default() -> Self {
        Self {
            device: if cfg!(feature = "cuda") {
                Device::Cuda
            } else {
                Device::Cpu
            },
            context_tokens: None,
            threads: DEFAULT_THREADS,
            batch_tokens: None,
            prefill_chunk: None,
            gpu_device: 0,
            gpu_memory_limit_bytes: None,
            kernel_cache_dir: None,
            #[cfg(feature = "cuda")]
            cuda_matmul: None,
            #[cfg(feature = "cuda")]
            decode_graph: None,
            chat_template: None,
            prefix_cache: true,
        }
    }
}

#[derive(Clone, Debug)]
pub struct GenerationOptions {
    /// None: until end of turn or a full context.
    pub max_tokens: Option<u32>,
    /// Reasoning for templates that support it (Qwen3 `enable_thinking`).
    pub enable_thinking: bool,
    /// Default: greedy.
    pub sampling: Sampling,
    /// Generation ends at the first of these; it is not part of the text.
    pub stop: Vec<String>,
    /// Constrains the output to JSON matching this schema
    /// (`JsonSchema::any_object()` for any object). Turns reasoning off: the
    /// constrained JSON is the whole reply.
    pub json_schema: Option<JsonSchema>,
}

impl GenerationOptions {
    /// Whether the chat template is asked for reasoning (`<think>`).
    pub fn thinking(&self) -> bool {
        self.enable_thinking && self.json_schema.is_none()
    }
}

impl Default for GenerationOptions {
    fn default() -> Self {
        Self {
            max_tokens: None,
            enable_thinking: true,
            sampling: Sampling::GREEDY,
            stop: Vec::new(),
            json_schema: None,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FinishReason {
    /// The model ended its turn, or a stop sequence appeared.
    Stop,
    /// `max_tokens` was reached.
    Length,
    /// The context is full.
    ContextFull,
    /// The text callback asked to stop.
    Cancelled,
}

#[derive(Clone, Debug)]
pub struct Completion {
    /// Generated text (for chat, trimmed like `palito chat` replies), without
    /// any stop sequence.
    pub text: String,
    pub finish_reason: FinishReason,
    pub prompt_tokens: usize,
    /// Prompt tokens whose evaluation was reused from the previous request.
    pub cached_tokens: usize,
    pub generated_tokens: usize,
    /// Prompt evaluation plus the first token, in nanoseconds.
    pub ttft_ns: u64,
    /// Decode steps per second after the first token (0 with fewer than two tokens).
    pub tokens_per_second: f64,
}

pub struct Engine<B: Backend + ?Sized = dyn Backend> {
    executor: Executor<Box<B>>,
    tokenizer: Arc<Tokenizer>,
    /// End-of-generation token IDs.
    eog: Arc<[i32]>,
    /// None when the model has no chat template (only `complete` works).
    template: Option<ChatTemplate>,
    profile: &'static str,
    notices: Vec<String>,
    /// Built on the first JSON-constrained request.
    json_decoder: Option<JsonDecoder>,
    /// Built on the first request with tools.
    tool_tokens: Option<crate::tools::ToolTokens>,
}

impl Engine {
    pub fn open(path: &Path, options: &EngineOptions) -> Result<Self> {
        let opened = open::open(path, options)?;
        let mut engine = Self::from_backend(
            path,
            opened.backend,
            opened.config,
            opened.profile,
            options.chat_template.as_deref(),
        )?;
        engine.notices = opened.notices;
        Ok(engine)
    }

    /// Wraps a backend opened by the caller (e.g. with validation-only CPU
    /// options). `config` must fit the backend.
    pub fn from_backend(
        path: &Path,
        backend: Box<dyn Backend>,
        config: Config,
        profile: &'static str,
        chat_template: Option<&str>,
    ) -> Result<Self> {
        Self::wrap(path, backend, config, profile, chat_template)
    }
}

impl<B: Backend + ?Sized> Engine<B> {
    fn wrap(
        path: &Path,
        backend: Box<B>,
        config: Config,
        profile: &'static str,
        chat_template: Option<&str>,
    ) -> Result<Self> {
        let tokenizer_adds_bos = !backend.tokenize("")?.is_empty();
        let template = match chat_template {
            Some(source) => Some(ChatTemplate::from_model(path, Some(source), tokenizer_adds_bos)?),
            None => ChatTemplate::from_model(path, None, tokenizer_adds_bos).ok(),
        };
        let executor = Executor::new(backend, config)?;
        executor.detokenize(&[])?;
        let vocabulary = executor.info().vocab_tokens as i32;
        let eog = (0..vocabulary).filter(|&id| executor.is_eog(id)).collect();
        Ok(Self {
            executor,
            tokenizer: Arc::new(Tokenizer::open(path)?),
            eog,
            template,
            profile,
            notices: Vec::new(),
            json_decoder: None,
            tool_tokens: None,
        })
    }

    /// Context tokens actually opened (may be below the requested context).
    pub fn context_tokens(&self) -> u32 {
        self.executor.info().context_tokens
    }

    /// Numerical profile in use: "cpu", or the CUDA matmul profile.
    pub fn profile(&self) -> &'static str {
        self.profile
    }

    /// What `open` changed from the request: profile fallbacks, a context
    /// shrunk to fit device memory.
    pub fn notices(&self) -> &[String] {
        &self.notices
    }

    pub fn has_chat_template(&self) -> bool {
        self.template.is_some()
    }

    pub fn tokenize(&self, text: &str) -> Result<Vec<i32>> {
        self.executor.tokenize(text)
    }

    /// The prompt `chat` would evaluate for `messages`.
    pub fn render_chat(&self, messages: &[Message], enable_thinking: bool) -> Result<String> {
        let template = self.template.as_ref().ok_or_else(|| {
            Error::Invalid("model has no chat template; use plain completion".into())
        })?;
        template.render_thinking(messages, true, enable_thinking)
    }

    /// Renders `messages` with the model's chat template and generates the
    /// assistant reply. `on_text` receives the reply as it is generated and
    /// returns false to stop.
    pub fn chat(
        &mut self,
        messages: &[Message],
        options: &GenerationOptions,
        on_text: impl FnMut(&str) -> bool,
    ) -> Result<Completion> {
        let prompt = self.render_chat(messages, options.thinking())?;
        let mut completion = self.generate(&prompt, options, TrimmedText::new(on_text))?;
        completion.text = completion.text.trim().to_owned();
        Ok(completion)
    }

    /// Continues `prompt` without a chat template.
    pub fn complete(
        &mut self,
        prompt: &str,
        options: &GenerationOptions,
        mut on_text: impl FnMut(&str) -> bool,
    ) -> Result<Completion> {
        self.generate(prompt, options, move |text: &str| on_text(text))
    }

    fn json_decoder(&mut self) -> Result<&JsonDecoder> {
        if self.json_decoder.is_none() {
            let vocab = self.executor.info().vocab_tokens as usize;
            let executor = &self.executor;
            let tokenizer = &self.tokenizer;
            let decoder = JsonDecoder::new(
                vocab,
                |id| tokenizer.token_bytes(id),
                |id| executor.is_eog(id),
            )?;
            self.json_decoder = Some(decoder);
        }
        self.json_decoder
            .as_ref()
            .ok_or_else(|| Error::Backend("JSON decoder missing".into()))
    }

    /// Validates a text request and works out its tokens and budget.
    fn text_request(&mut self, prompt: &str, options: &GenerationOptions) -> Result<TextRequest> {
        stop::validate(&options.stop)?;
        let sampler = Sampler::new(options.sampling)?;
        let constraint = match &options.json_schema {
            Some(schema) => Some(self.json_decoder()?.constraint(schema)?),
            None => None,
        };
        let tokens = self.executor.tokenize(prompt)?;
        let context = self.context_tokens() as usize;
        if tokens.is_empty() || tokens.len() >= context {
            return Err(Error::Invalid(format!(
                "prompt of {} tokens does not fit in the {context}-token context",
                tokens.len()
            )));
        }
        let room = (context + 1 - tokens.len()) as u32;
        let budget = options.max_tokens.map_or(room, |max| max.min(room));
        if budget == 0 {
            return Err(Error::Invalid("max_tokens must be positive".into()));
        }
        Ok(TextRequest { tokens, budget, room, sampler, constraint })
    }

    fn generate(
        &mut self,
        prompt: &str,
        options: &GenerationOptions,
        sink: impl TextSink,
    ) -> Result<Completion> {
        let TextRequest { tokens, budget, room, mut sampler, mut constraint } =
            self.text_request(prompt, options)?;
        let plan = self.reply_plan(tokens.len(), budget, room);
        let mut output = TextOutput::new(plan, &options.stop, sink);
        let mode = GenerationMode::UntilEog;
        let result = if sampler.is_greedy() && constraint.is_none() {
            self.executor
                .generate_streaming(&tokens, budget, mode, |token| output.token(token))
        } else {
            self.executor.generate_constrained(
                &tokens,
                budget,
                mode,
                &mut sampler,
                constraint.as_mut().map(|c| c as &mut dyn LogitConstraint),
                |token| output.token(token),
            )
        };
        let sample = match result {
            Ok(sample) => Some(sample),
            Err(Error::Cancelled) => None,
            Err(error) => return Err(error),
        };
        if sample.is_some() {
            output.drain()?;
        }
        output.complete(sample.as_ref().map(run_stats))
    }

    fn reply_plan(&self, prompt_tokens: usize, budget: u32, room: u32) -> ReplyPlan {
        ReplyPlan {
            tokenizer: self.tokenizer.clone(),
            eog: self.eog.clone(),
            prompt_tokens,
            budget,
            room,
        }
    }
}

struct TextRequest {
    tokens: Vec<i32>,
    budget: u32,
    room: u32,
    sampler: Sampler,
    constraint: Option<Constraint>,
}

fn run_stats(sample: &Sample) -> RunStats {
    RunStats {
        cached_tokens: sample.cached_prompt_tokens,
        ttft_ns: sample.ttft_ns,
        tokens_per_second: decode_rate(sample),
    }
}

fn decode_rate(sample: &Sample) -> f64 {
    if sample.decode_ns == 0 {
        0.0
    } else {
        sample.decode_steps as f64 * 1e9 / sample.decode_ns as f64
    }
}

/// Timing for generations that end early (a stop sequence or a cancelled
/// callback), where the executor returns no `Sample`.
struct TokenClock {
    start: Instant,
    first: Option<Instant>,
    last: Option<Instant>,
    tokens: usize,
}

impl TokenClock {
    fn new() -> Self {
        Self {
            start: Instant::now(),
            first: None,
            last: None,
            tokens: 0,
        }
    }

    fn tick(&mut self) {
        let now = Instant::now();
        self.first.get_or_insert(now);
        self.last = Some(now);
        self.tokens += 1;
    }

    fn stats(&self) -> (u64, f64) {
        let (Some(first), Some(last)) = (self.first, self.last) else {
            return (0, 0.0);
        };
        let ttft = (first - self.start).as_nanos() as u64;
        let decode = (last - first).as_secs_f64();
        let rate = if self.tokens > 1 && decode > 0.0 {
            (self.tokens - 1) as f64 / decode
        } else {
            0.0
        };
        (ttft, rate)
    }
}
