//! Backend, model and execution-plan options shared by every local model
//! command, and opening the backend they select.

use crate::cli_ui;
use clap::{Args, ValueEnum, ValueHint};
use palito::{
    Config, Result,
    backend::Backend,
    cpu::{CpuBackend, CpuOptions},
};
use serde_json::json;
use std::path::PathBuf;

#[derive(Clone, Copy, PartialEq, Eq, ValueEnum)]
pub(crate) enum LocalBackend {
    Cpu,
    #[cfg(feature = "cuda")]
    Cuda,
    #[cfg(feature = "vulkan")]
    Vulkan,
    #[cfg(all(feature = "metal", target_os = "macos"))]
    Metal,
    /// Slow independent packed CPU oracle for the CUDA model layouts.
    #[cfg(feature = "cuda")]
    Reference,
}

/// Read only by the CUDA backend, so a global setting never breaks CPU runs.
#[cfg(feature = "cuda")]
const KERNEL_CACHE_ENV: &str = "PALITO_CUDA_KERNEL_CACHE";
const DEFAULT_GENERATED_TOKENS: u32 = 32;
const DEFAULT_CONTEXT_TOKENS: u32 = 512;
/// Largest CUDA context (CudaBackend::open).
#[cfg(feature = "cuda")]
const CUDA_CONTEXT_MAX: u32 = 65535;
const DEFAULT_BATCH_TOKENS: u32 = 128;
const DEFAULT_PREFILL_CHUNK: u32 = 64;

/// Backend, model and execution-plan options shared by every local model command.
#[derive(Args)]
pub(crate) struct RuntimeArgs {
    /// Execution backend [default: cuda for .palito files; GGUF: metal on macOS, else cpu].
    #[arg(long, value_enum, env = "PALITO_BACKEND", help_heading = "Backend")]
    backend: Option<LocalBackend>,
    /// Model: a .palito/.gguf file or a downloaded model name (see 'palito list').
    #[arg(short, long, env = "PALITO_MODEL", value_name = "MODEL", value_hint = ValueHint::FilePath)]
    pub(crate) model: PathBuf,
    /// Download the model first if it is a name that is not downloaded yet, without asking.
    #[arg(long)]
    pub(crate) pull: bool,
    /// Context tokens [default: 512; chat: the model's trained context].
    #[arg(
        long,
        value_name = "N",
        long_help = "Context tokens [default: 512; chat: the model's trained context (GGUF \
                     context_length; CUDA: at most 65535, shrunk to the largest that fits in \
                     free device memory)]"
    )]
    context: Option<u32>,
    /// Maximum generated tokens [default: 32; chat: until end of turn or a full context].
    #[arg(long, value_name = "N")]
    tokens: Option<u32>,
    /// CPU threads for host-side work.
    #[arg(long, default_value_t = 4, value_name = "N", help_heading = "Backend")]
    threads: u32,
    /// Prefill batch capacity [default: 128; chat on CUDA: 4096, capped by --context].
    #[arg(long, value_name = "N", help_heading = "Advanced", hide_short_help = true)]
    batch: Option<u32>,
    /// Tokens per prefill call [default: 64; chat on CUDA: 4096, capped by --context and --batch].
    #[arg(long, value_name = "N", help_heading = "Advanced", hide_short_help = true)]
    prefill_chunk: Option<u32>,
    /// Expand weights for differential validation rather than retaining packed GGUF blocks.
    #[arg(long, help_heading = "Validation", hide_short_help = true)]
    pub(crate) f32_weights: bool,
    /// Evaluate prefill one token at a time for differential validation.
    #[arg(long, help_heading = "Validation", hide_short_help = true)]
    sequential_prefill: bool,
    /// Disable runtime-dispatched AVX2 specialization for validation.
    #[arg(long, help_heading = "Validation", hide_short_help = true)]
    scalar_kernels: bool,
    /// Bound numerical reservations (e.g. 8GiB), not process RSS or transient model loading.
    #[arg(long, alias = "memory-limit", value_name = "SIZE", value_parser = cli_ui::parse_size,
        help_heading = "Advanced", hide_short_help = true)]
    pub(crate) memory_limit_bytes: Option<u64>,
    /// Opt-in, one-entry immutable prefix cache with a numerical payload budget (e.g. 512MiB).
    #[arg(long, alias = "prefix-cache", default_value = "0", value_name = "SIZE",
        value_parser = cli_ui::parse_size, help_heading = "Advanced", hide_short_help = true)]
    prefix_cache_bytes: u64,
    /// Opt-in n-gram draft verification; zero disables speculation.
    #[arg(long, default_value_t = 0, value_name = "N", help_heading = "Advanced", hide_short_help = true)]
    speculative_window: u32,
    /// GPU index to run on [default: 0].
    #[cfg(feature = "cuda")]
    #[arg(long, alias = "cuda-device", requires = "backend", value_name = "INDEX", help_heading = "CUDA")]
    gpu_device: Option<usize>,
    /// Device-buffer limit (e.g. 20GiB); excludes driver/JIT overhead and host model loading.
    #[cfg(feature = "cuda")]
    #[arg(long, alias = "gpu-memory-limit", value_name = "SIZE", value_parser = cli_ui::parse_size,
        help_heading = "CUDA")]
    gpu_memory_limit_bytes: Option<u64>,
    /// Directory for the persistent CUDA kernel cache [default: $PALITO_CUDA_KERNEL_CACHE].
    #[cfg(feature = "cuda")]
    #[arg(long, value_name = "DIR", value_hint = ValueHint::DirPath, help_heading = "CUDA",
        hide_short_help = true)]
    cuda_kernel_cache: Option<PathBuf>,
    /// CUDA matrix profile [default: f32; chat: int8-fast, f32-tiled below sm_80].
    #[cfg(feature = "cuda")]
    #[arg(
        long,
        value_enum,
        help_heading = "CUDA",
        long_help = "CUDA matrix profile [default: f32; chat: int8-fast, f32-tiled below sm_80].\n\
                     \n\
                     f32: existing kernels and arithmetic.\n\
                     f32-tiled: measured F32 prefill/decode kernels, no activation quantization.\n\
                     int8-prefill: f32-tiled decode, tensor-core prefill (INT8 projections, FP16 \
                     hi/lo attention); faster long prompts, approximate logits (same top token on \
                     the quality gates).\n\
                     int8-fast: int8-prefill at llama.cpp precision (one 8-bit plane per 32 \
                     activations, single FP16 attention, TF32 router); fastest prefill, logits \
                     about 1 off F32 and generated text can differ."
    )]
    cuda_matmul: Option<palito::cuda::CudaMatmul>,
    /// Captured decode with token-only greedy output (f32: context <= 1024). On by
    /// default for chat with f32-tiled, int8-prefill and int8-fast.
    #[cfg(feature = "cuda")]
    #[arg(long, help_heading = "CUDA", hide_short_help = true)]
    cuda_decode_graph: bool,
    /// Disable the decode graph chat enables by default.
    #[cfg(feature = "cuda")]
    #[arg(long, conflicts_with = "cuda_decode_graph", help_heading = "CUDA", hide_short_help = true)]
    no_cuda_decode_graph: bool,
    /// Re-evaluate the whole conversation every turn instead of reusing the
    /// previous prompt's KV cache (chat with --backend cuda).
    #[cfg(feature = "cuda")]
    #[arg(long, help_heading = "CUDA", hide_short_help = true)]
    no_prefix_cache: bool,
    /// --context defaulted to the model's trained context (chat): CUDA may shrink it
    /// to fit device memory.
    #[arg(skip)]
    model_context: bool,
}

impl RuntimeArgs {
    /// Resolves a model name to its downloaded file (pulling it if needed),
    /// then fails with the path and a hint when --model is not a model.
    pub(crate) fn check_model(&mut self) -> Result<()> {
        crate::hub::commands::resolve_model(&mut self.model, self.pull)?;
        cli_ui::check_model(&self.model)
    }

    fn options(&self) -> CpuOptions {
        CpuOptions {
            expand_weights: self.f32_weights,
            sequential_prefill: self.sequential_prefill,
            memory_limit_bytes: self.memory_limit_bytes,
            prefix_cache_bytes: self.prefix_cache_bytes,
            speculative_window: self.speculative_window,
            scalar_kernels: self.scalar_kernels,
        }
    }

    fn resolved_backend(&self) -> LocalBackend {
        if let Some(backend) = self.backend {
            return backend;
        }
        #[cfg(feature = "cuda")]
        if palito::plt1::is_plt1_file(&self.model) {
            return LocalBackend::Cuda;
        }
        // Every Apple Silicon Mac has Metal; .palito files are CPU-only there.
        #[cfg(all(feature = "metal", target_os = "macos"))]
        if !palito::plt1::is_plt1_file(&self.model) {
            return LocalBackend::Metal;
        }
        LocalBackend::Cpu
    }

    pub(crate) fn open(&self) -> Result<Box<dyn Backend>> {
        #[allow(unused_variables)]
        let backend = self.resolved_backend();
        #[cfg(feature = "cuda")]
        {
            if backend != LocalBackend::Cuda
                && (self.gpu_device.is_some()
                    || self.gpu_memory_limit_bytes.is_some()
                    || self.cuda_kernel_cache.is_some()
                    || self
                        .cuda_matmul
                        .is_some_and(|m| m != palito::cuda::CudaMatmul::F32)
                    || self.cuda_decode_graph
                    || self.no_cuda_decode_graph
                    || self.no_prefix_cache)
            {
                return Err(palito::Error::Invalid(
                    "CUDA options require --backend cuda".into(),
                ));
            }
            self.check_cpu_policy()?;
            match backend {
                LocalBackend::Cuda => {
                    return palito::cuda::open_backend(
                        &self.model,
                        &self.config(),
                        self.cuda_options(),
                    );
                }
                LocalBackend::Reference => {
                    return Ok(Box::new(
                        palito::reference::ReferenceBackend::open(
                            &self.model,
                            &self.config(),
                        )?,
                    ));
                }
                #[cfg(feature = "vulkan")]
                LocalBackend::Vulkan => {}
                #[cfg(all(feature = "metal", target_os = "macos"))]
                LocalBackend::Metal => {}
                LocalBackend::Cpu => {}
            }
        }
        #[cfg(feature = "vulkan")]
        if backend == LocalBackend::Vulkan {
            self.check_cpu_policy()?;
            let options = palito::vulkan::VulkanOptions::default();
            return palito::vulkan::open_backend(&self.model, &self.config(), options);
        }
        #[cfg(all(feature = "metal", target_os = "macos"))]
        if backend == LocalBackend::Metal {
            self.check_cpu_policy()?;
            let options = palito::metal::MetalOptions::default();
            return palito::metal::open_backend(&self.model, &self.config(), options);
        }
        if !palito::cpu::has_fast_path(&self.model)? {
            return self.open_cpu_reference();
        }
        Ok(Box::new(CpuBackend::open_with_options(
            &self.model,
            &self.config(),
            self.options(),
        )?))
    }

    /// Layouts without the fast path run on the reference backend, which has
    /// none of the fast path's policy options.
    fn open_cpu_reference(&self) -> Result<Box<dyn Backend>> {
        if self.f32_weights
            || self.sequential_prefill
            || self.memory_limit_bytes.is_some()
            || self.prefix_cache_bytes != 0
            || self.speculative_window != 0
            || self.scalar_kernels
        {
            return Err(palito::Error::Invalid(
                "CPU policy options are not supported for this model architecture".into(),
            ));
        }
        Ok(Box::new(palito::reference::ReferenceBackend::open(
            &self.model,
            &self.config(),
        )?))
    }

    #[cfg(feature = "cuda")]
    fn cuda_options(&self) -> palito::cuda::CudaOptions {
        let mut options = palito::cuda::CudaOptions::default();
        if let Some(device) = self.gpu_device {
            options.device = device;
        }
        if let Some(limit) = self.gpu_memory_limit_bytes {
            options.memory_limit_bytes = limit;
        }
        options.kernel_cache_dir = self.cuda_kernel_cache.clone().or_else(|| {
            std::env::var_os(KERNEL_CACHE_ENV)
                .filter(|dir| !dir.is_empty())
                .map(PathBuf::from)
        });
        let is_plt1 = palito::plt1::is_plt1_file(&self.model);
        options.matmul = self.cuda_matmul.unwrap_or(if is_plt1 {
            palito::cuda::CudaMatmul::F32Tiled
        } else {
            palito::cuda::CudaMatmul::default()
        });
        options.decode_graph = self.cuda_decode_graph || (is_plt1 && !self.no_cuda_decode_graph);
        options.fit_context = self.model_context;
        options
    }

    #[cfg(any(feature = "cuda", feature = "vulkan", all(feature = "metal", target_os = "macos")))]
    fn check_cpu_policy(&self) -> Result<()> {
        if self.resolved_backend() != LocalBackend::Cpu
            && (self.f32_weights
                || self.sequential_prefill
                || self.memory_limit_bytes.is_some()
                || self.prefix_cache_bytes != 0
                || self.speculative_window != 0
                || self.scalar_kernels)
        {
            return Err(palito::Error::Invalid(
                "CPU policy options require --backend cpu".into(),
            ));
        }
        Ok(())
    }

    /// Engine options for `--backend cuda` (chat and serve defaults: model
    /// context, fastest profile that opens, decode graph, 4096-token prefill).
    #[cfg(feature = "cuda")]
    pub(crate) fn cuda_engine_options(&self) -> Result<palito::engine::EngineOptions> {
        self.check_cpu_policy()?;
        Ok(palito::engine::EngineOptions {
            device: palito::engine::Device::Cuda,
            context_tokens: self.context,
            threads: self.threads,
            batch_tokens: self.batch,
            prefill_chunk: self.prefill_chunk,
            gpu_device: self.gpu_device.unwrap_or(0),
            gpu_memory_limit_bytes: self.gpu_memory_limit_bytes,
            kernel_cache_dir: self.cuda_kernel_cache.clone(),
            cuda_matmul: self.cuda_matmul,
            decode_graph: if self.cuda_decode_graph {
                Some(true)
            } else if self.no_cuda_decode_graph {
                Some(false)
            } else {
                None
            },
            chat_template: None,
            prefix_cache: !self.no_prefix_cache,
        })
    }

    /// Name of the backend this run resolves to, for messages.
    pub(crate) fn backend_name(&self) -> &'static str {
        match self.resolved_backend() {
            LocalBackend::Cpu => "cpu",
            #[cfg(feature = "cuda")]
            LocalBackend::Cuda => "cuda",
            #[cfg(feature = "vulkan")]
            LocalBackend::Vulkan => "vulkan",
            #[cfg(all(feature = "metal", target_os = "macos"))]
            LocalBackend::Metal => "metal",
            #[cfg(feature = "cuda")]
            LocalBackend::Reference => "cpu-reference",
        }
    }

    /// True when the backend is CUDA (chat then opens through `Engine`).
    pub(crate) fn is_cuda(&self) -> bool {
        #[cfg(feature = "cuda")]
        {
            self.resolved_backend() == LocalBackend::Cuda
        }
        #[cfg(not(feature = "cuda"))]
        false
    }

    /// "cpu" or "cpu-reference".
    pub(crate) fn cpu_profile(&self) -> &'static str {
        #[cfg(feature = "cuda")]
        if self.resolved_backend() == LocalBackend::Reference {
            return "cpu-reference";
        }
        "cpu"
    }

    pub(crate) fn tokens(&self) -> u32 {
        self.tokens.unwrap_or(DEFAULT_GENERATED_TOKENS)
    }

    pub(crate) fn context(&self) -> u32 {
        self.context.unwrap_or(DEFAULT_CONTEXT_TOKENS)
    }

    /// Explicit --tokens for chat replies; None generates until end of turn or a full
    /// context.
    pub(crate) fn chat_tokens(&self) -> Option<u32> {
        self.tokens
    }

    /// Without --context, use the model's trained context (CUDA: at most
    /// CUDA_CONTEXT_MAX, fitted to device memory when opening).
    pub(crate) fn use_model_context(&mut self) -> Result<()> {
        if self.context.is_none() {
            let trained = palito::model_context_tokens(&self.model)?;
            #[cfg(feature = "cuda")]
            let cap = if self.resolved_backend() == LocalBackend::Cuda {
                CUDA_CONTEXT_MAX
            } else {
                u32::MAX
            };
            #[cfg(not(feature = "cuda"))]
            let cap = u32::MAX;
            self.context = Some(trained.min(cap));
            self.model_context = true;
        }
        Ok(())
    }

    fn batch(&self) -> u32 {
        self.batch.unwrap_or(DEFAULT_BATCH_TOKENS)
    }

    fn prefill_chunk(&self) -> u32 {
        self.prefill_chunk.unwrap_or(DEFAULT_PREFILL_CHUNK)
    }

    pub(crate) fn execution(&self, info: &palito::backend::ModelInfo) -> serde_json::Value {
        if info.backend == "rust-cpu-transformer-reference" {
            return json!({"backend": "cpu-reference", "prefill": "sequential token steps"});
        }
        #[allow(unused_variables)]
        let backend = self.resolved_backend();
        #[cfg(feature = "cuda")]
        match backend {
            LocalBackend::Cuda => {
                let options = self.cuda_options();
                let kv_precision = if options.matmul == palito::cuda::CudaMatmul::Int8Fast {
                    "f16"
                } else {
                    "f32"
                };
                return json!({
                    "backend": "cuda", "options": options,
                    "expert_residency": "all weights GPU-resident",
                    "prefill": if self.batch() > 1 {
                        "layer-wise batched (internal capacity <=4096)"
                    } else { "sequential token steps" },
                    "kv_precision": kv_precision,
                });
            }
            LocalBackend::Reference => {}
            #[cfg(feature = "vulkan")]
            LocalBackend::Vulkan => {}
            #[cfg(all(feature = "metal", target_os = "macos"))]
            LocalBackend::Metal => {}
            LocalBackend::Cpu => {}
        }
        #[cfg(feature = "vulkan")]
        if backend == LocalBackend::Vulkan {
            return json!({
                "backend": "vulkan",
                "options": palito::vulkan::VulkanOptions::default(),
                "expert_residency": "all weights GPU-resident",
                "prefill": "layer-wise batched (internal capacity <=64)",
                "kv_precision": "f32",
            });
        }
        #[cfg(all(feature = "metal", target_os = "macos"))]
        if backend == LocalBackend::Metal {
            return json!({
                "backend": "metal",
                "options": palito::metal::MetalOptions::default(),
                "expert_residency": "Apple Silicon unified memory (StorageModeShared)",
                "prefill": "layer-wise batched (internal capacity <=64)",
                "kv_precision": "f32",
            });
        }
        json!(self.options())
    }

    pub(crate) fn config(&self) -> Config {
        Config {
            context_tokens: self.context(),
            batch_tokens: self.batch(),
            prefill_chunk: self.prefill_chunk(),
            threads: self.threads,
        }
    }
}

