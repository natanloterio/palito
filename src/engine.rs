//! High-level embedding API: open a GGUF model once, then run chat or plain
//! completion with streamed text.
//!
//! `Engine::open` applies the same defaults as `palito chat`: the model's trained
//! context (CUDA: at most 65535, shrunk to fit free device memory), the fastest
//! CUDA profile that opens for the model, and prefill chunks of up to 4096 tokens.
//! The C API (`crate::ffi`) and other embedders build on this type.

use crate::{
    Config, Error, Executor, GenerationMode, Result, Sample,
    backend::Backend,
    chat::{ChatTemplate, Message},
    cpu::{CpuBackend, CpuOptions},
    stream::TextStream,
    tokenizer::Tokenizer,
};
use std::path::{Path, PathBuf};

const CPU_BATCH_TOKENS: u32 = 128;
const CPU_PREFILL_CHUNK: u32 = 64;
const DEFAULT_THREADS: u32 = 4;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Device {
    Cpu,
    Cuda,
}

#[derive(Clone, Debug)]
pub struct EngineOptions {
    /// Default: CUDA when built with the `cuda` feature, else CPU.
    pub device: Device,
    /// None: the model's trained context (CUDA: fitted to device memory).
    pub context_tokens: Option<u32>,
    pub threads: u32,
    pub gpu_device: usize,
    /// None: free device memory minus the 2 GiB headroom.
    pub gpu_memory_limit_bytes: Option<u64>,
    /// None: $PALITO_CUDA_KERNEL_CACHE, else $XDG_CACHE_HOME (or ~/.cache)/palito/kernels.
    pub kernel_cache_dir: Option<PathBuf>,
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
            gpu_device: 0,
            gpu_memory_limit_bytes: None,
            kernel_cache_dir: None,
        }
    }
}

#[derive(Clone, Debug)]
pub struct GenerationOptions {
    /// None: until end of turn or a full context.
    pub max_tokens: Option<u32>,
    /// Reasoning for templates that support it (Qwen3 `enable_thinking`).
    pub enable_thinking: bool,
}

impl Default for GenerationOptions {
    fn default() -> Self {
        Self {
            max_tokens: None,
            enable_thinking: true,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FinishReason {
    /// The model ended its turn.
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
    /// Generated text (for chat, trimmed like `palito chat` replies).
    pub text: String,
    pub finish_reason: FinishReason,
    pub prompt_tokens: usize,
    pub generated_tokens: usize,
    /// Prompt evaluation plus the first token, in nanoseconds (0 when cancelled).
    pub ttft_ns: u64,
    /// Decode steps per second after the first token (0 when cancelled or fewer
    /// than two tokens).
    pub tokens_per_second: f64,
}

pub struct Engine {
    executor: Executor<Box<dyn Backend>>,
    tokenizer: Tokenizer,
    /// None when the model has no chat template (only `complete` works).
    template: Option<ChatTemplate>,
    profile: &'static str,
}

impl Engine {
    pub fn open(path: &Path, options: &EngineOptions) -> Result<Self> {
        if options.threads == 0 {
            return Err(Error::Invalid("threads must be positive".into()));
        }
        let trained = crate::model_context_tokens(path)?;
        let (backend, config, profile) = match options.device {
            Device::Cpu => open_cpu(path, options, trained)?,
            Device::Cuda => open_cuda(path, options, trained)?,
        };
        let tokenizer_adds_bos = !backend.tokenize("")?.is_empty();
        let template = ChatTemplate::from_model(path, None, tokenizer_adds_bos).ok();
        let executor = Executor::new(backend, config)?;
        executor.detokenize(&[])?;
        Ok(Self {
            executor,
            tokenizer: Tokenizer::open(path)?,
            template,
            profile,
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

    pub fn has_chat_template(&self) -> bool {
        self.template.is_some()
    }

    pub fn tokenize(&self, text: &str) -> Result<Vec<i32>> {
        self.executor.tokenize(text)
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
        let template = self.template.as_ref().ok_or_else(|| {
            Error::Invalid("model has no chat template; use plain completion".into())
        })?;
        let prompt = template.render_thinking(messages, true, options.enable_thinking)?;
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

    fn generate(
        &mut self,
        prompt: &str,
        options: &GenerationOptions,
        mut sink: impl TextSink,
    ) -> Result<Completion> {
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
        let mut stream = TextStream::new();
        let mut generated = Vec::new();
        let tokenizer = &self.tokenizer;
        let detokenize = |ids: &[i32]| tokenizer.detokenize(ids);
        let result =
            self.executor
                .generate_streaming(&tokens, budget, GenerationMode::UntilEog, |token| {
                    generated.push(token);
                    let piece = stream.push(token, detokenize)?;
                    if piece.is_empty() || sink.write(&piece) {
                        Ok(())
                    } else {
                        Err(Error::Cancelled)
                    }
                });
        let rest = stream.finish(detokenize)?;
        let sample = match result {
            Ok(sample) => Some(sample),
            Err(Error::Cancelled) => None,
            Err(error) => return Err(error),
        };
        if sample.is_some() && !rest.is_empty() {
            sink.write(&rest);
        }
        sink.end();
        let ended = generated.last().is_some_and(|&t| self.executor.is_eog(t));
        let finish_reason = match &sample {
            None => FinishReason::Cancelled,
            Some(_) if ended => FinishReason::Stop,
            Some(_) if budget < room => FinishReason::Length,
            Some(_) => FinishReason::ContextFull,
        };
        Ok(Completion {
            text: tokenizer.detokenize(&generated)?,
            finish_reason,
            prompt_tokens: tokens.len(),
            generated_tokens: generated.len(),
            ttft_ns: sample.as_ref().map_or(0, |s| s.ttft_ns),
            tokens_per_second: sample.as_ref().map_or(0.0, decode_rate),
        })
    }
}

fn decode_rate(sample: &Sample) -> f64 {
    if sample.decode_ns == 0 {
        0.0
    } else {
        sample.decode_steps as f64 * 1e9 / sample.decode_ns as f64
    }
}

/// Receives streamed text; `write` returns false to stop generation.
trait TextSink {
    fn write(&mut self, text: &str) -> bool;
    fn end(&mut self) {}
}

impl<F: FnMut(&str) -> bool> TextSink for F {
    fn write(&mut self, text: &str) -> bool {
        self(text)
    }
}

/// Streams chat text without leading whitespace and holds trailing whitespace
/// until more text follows, so the streamed reply equals the trimmed one.
struct TrimmedText<F> {
    inner: F,
    started: bool,
    pending: String,
}

impl<F: FnMut(&str) -> bool> TrimmedText<F> {
    fn new(inner: F) -> Self {
        Self {
            inner,
            started: false,
            pending: String::new(),
        }
    }
}

impl<F: FnMut(&str) -> bool> TextSink for TrimmedText<F> {
    fn write(&mut self, text: &str) -> bool {
        let text = if self.started {
            text
        } else {
            text.trim_start()
        };
        if text.is_empty() {
            return true;
        }
        self.started = true;
        let body = text.trim_end();
        let keep_going = if body.is_empty() {
            true
        } else {
            let piece = std::mem::take(&mut self.pending) + body;
            (self.inner)(&piece)
        };
        self.pending.push_str(&text[body.len()..]);
        keep_going
    }
}

fn open_cpu(
    path: &Path,
    options: &EngineOptions,
    trained: u32,
) -> Result<(Box<dyn Backend>, Config, &'static str)> {
    let context = options.context_tokens.unwrap_or(trained);
    let config = Config {
        context_tokens: context,
        batch_tokens: CPU_BATCH_TOKENS.min(context),
        prefill_chunk: CPU_PREFILL_CHUNK.min(context),
        threads: options.threads,
    };
    let backend = CpuBackend::open_with_options(path, &config, CpuOptions::default())?;
    Ok((Box::new(backend), config, "cpu"))
}

#[cfg(not(feature = "cuda"))]
fn open_cuda(
    _: &Path,
    _: &EngineOptions,
    _: u32,
) -> Result<(Box<dyn Backend>, Config, &'static str)> {
    Err(Error::Invalid(
        "this build has no CUDA support; use the CPU device".into(),
    ))
}

#[cfg(feature = "cuda")]
fn open_cuda(
    path: &Path,
    options: &EngineOptions,
    trained: u32,
) -> Result<(Box<dyn Backend>, Config, &'static str)> {
    use crate::cuda::{CudaMatmul, CudaOptions, open_backend};
    const CONTEXT_MAX: u32 = 65535;
    const FAST_PREFILL_TOKENS: u32 = 4096;
    let (context, fit) = match options.context_tokens {
        Some(context) => (context, false),
        None => (trained.min(CONTEXT_MAX), true),
    };
    // The chat CLI's order: fastest profile first, then the ones every
    // quantization supports (Q5_K needs f32 without the decode graph).
    let attempts = [
        (CudaMatmul::Int8Fast, true, "int8-fast"),
        (CudaMatmul::F32Tiled, true, "f32-tiled"),
        (CudaMatmul::F32Tiled, false, "f32-tiled"),
        (CudaMatmul::F32, false, "f32"),
    ];
    let mut first_error = None;
    for (matmul, graph, name) in attempts {
        let (batch, chunk) = if matmul == CudaMatmul::F32 {
            (CPU_BATCH_TOKENS.min(context), CPU_PREFILL_CHUNK.min(context))
        } else {
            let batch = FAST_PREFILL_TOKENS.min(context);
            (batch, batch)
        };
        let config = Config {
            context_tokens: context,
            batch_tokens: batch,
            prefill_chunk: chunk,
            threads: options.threads,
        };
        let cuda = CudaOptions {
            device: options.gpu_device,
            memory_limit_bytes: options.gpu_memory_limit_bytes.unwrap_or(u64::MAX),
            kernel_cache_dir: options.kernel_cache_dir.clone().or_else(default_kernel_cache),
            matmul,
            decode_graph: graph,
            fit_context: fit,
        };
        match open_backend(path, &config, cuda) {
            Ok(backend) => {
                let opened = backend.info().context_tokens;
                let config = Config {
                    context_tokens: opened,
                    batch_tokens: config.batch_tokens.min(opened),
                    prefill_chunk: config.prefill_chunk.min(opened),
                    ..config
                };
                return Ok((backend, config, name));
            }
            // Files and memory budgets fail the same way for every profile.
            Err(error @ Error::Io(_)) => return Err(error),
            Err(error) => {
                first_error.get_or_insert(error);
            }
        }
    }
    Err(first_error.expect("at least one CUDA profile was tried"))
}

#[cfg(feature = "cuda")]
fn default_kernel_cache() -> Option<PathBuf> {
    let from_env = |name: &str| std::env::var_os(name).filter(|value| !value.is_empty());
    if let Some(dir) = from_env("PALITO_CUDA_KERNEL_CACHE") {
        return Some(PathBuf::from(dir));
    }
    let base = from_env("XDG_CACHE_HOME")
        .map(PathBuf::from)
        .or_else(|| from_env("HOME").map(|home| PathBuf::from(home).join(".cache")))?;
    Some(base.join("palito").join("kernels"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trimmed_text_streams_the_trimmed_reply() {
        let mut out = String::new();
        {
            let mut sink = TrimmedText::new(|piece: &str| {
                out.push_str(piece);
                true
            });
            for piece in ["\n\n", "  Hello", ", world", "  ", "\n", "again", " \n"] {
                assert!(sink.write(piece));
            }
        }
        assert_eq!(out, "Hello, world  \nagain");
    }

    #[test]
    fn trimmed_text_reports_stop() {
        let mut sink = TrimmedText::new(|_: &str| false);
        assert!(sink.write(" "));
        assert!(!sink.write("text"));
    }
}
