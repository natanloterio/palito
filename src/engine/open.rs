//! Opening a backend with `palito chat`'s defaults and CUDA profile fallbacks.

use super::{Device, EngineOptions};
use crate::{
    Config, Error, Result,
    backend::Backend,
    cpu::{CpuBackend, CpuOptions},
};
use std::path::Path;

pub(super) const CPU_BATCH_TOKENS: u32 = 128;
pub(super) const CPU_PREFILL_CHUNK: u32 = 64;

pub(super) struct Opened {
    pub backend: Box<dyn Backend>,
    pub config: Config,
    pub profile: &'static str,
    /// Fallbacks and context shrinking worth telling the user about.
    pub notices: Vec<String>,
}

pub(super) fn open(path: &Path, options: &EngineOptions) -> Result<Opened> {
    if options.threads == 0 {
        return Err(Error::Invalid("threads must be positive".into()));
    }
    let trained = crate::model_context_tokens(path)?;
    match options.device {
        Device::Cpu => open_cpu(path, options, trained),
        Device::Cuda => open_cuda(path, options, trained),
    }
}

fn open_cpu(path: &Path, options: &EngineOptions, trained: u32) -> Result<Opened> {
    let context = options.context_tokens.unwrap_or(trained);
    let batch = options.batch_tokens.unwrap_or(CPU_BATCH_TOKENS.min(context));
    let config = Config {
        context_tokens: context,
        batch_tokens: batch,
        prefill_chunk: options.prefill_chunk.unwrap_or(CPU_PREFILL_CHUNK.min(batch)),
        threads: options.threads,
    };
    let backend = CpuBackend::open_with_options(path, &config, CpuOptions::default())?;
    Ok(Opened {
        backend: Box::new(backend),
        config,
        profile: "cpu",
        notices: Vec::new(),
    })
}

#[cfg(not(feature = "cuda"))]
fn open_cuda(_: &Path, _: &EngineOptions, _: u32) -> Result<Opened> {
    Err(Error::Invalid(
        "this build has no CUDA support; use the CPU device".into(),
    ))
}

#[cfg(feature = "cuda")]
fn open_cuda(path: &Path, options: &EngineOptions, trained: u32) -> Result<Opened> {
    use crate::cuda::{CudaMatmul, CudaOptions, open_backend};
    const CONTEXT_MAX: u32 = 65535;
    let (context, fit) = match options.context_tokens {
        Some(context) => (context, false),
        None => (trained.min(CONTEXT_MAX), true),
    };
    let attempts = cuda_attempts(options);
    let mut notices = Vec::new();
    let mut first_error = None;
    for (index, &(matmul, graph)) in attempts.iter().enumerate() {
        let config = cuda_config(options, context, matmul);
        let cuda = CudaOptions {
            device: options.gpu_device,
            memory_limit_bytes: options.gpu_memory_limit_bytes.unwrap_or(u64::MAX),
            kernel_cache_dir: options
                .kernel_cache_dir
                .clone()
                .or_else(default_kernel_cache),
            matmul,
            decode_graph: graph,
            fit_context: fit,
            prefix_cache: options.prefix_cache,
            slots: 1,
        };
        let error = match open_backend(path, &config, cuda) {
            Ok(backend) => {
                let opened = backend.info().context_tokens;
                if opened < context {
                    notices.push(format!(
                        "context {opened} tokens: {context} do not fit in device memory"
                    ));
                }
                let config = Config {
                    context_tokens: opened,
                    batch_tokens: config.batch_tokens.min(opened),
                    prefill_chunk: config.prefill_chunk.min(opened),
                    ..config
                };
                return Ok(Opened {
                    backend,
                    config,
                    profile: profile_name(matmul),
                    notices,
                });
            }
            // Files and memory budgets fail the same way for every profile.
            Err(error @ Error::Io(_)) => return Err(error),
            Err(error) => error,
        };
        if let Some(&(next, _)) = attempts.get(index + 1) {
            if next != matmul {
                let using = if next == CudaMatmul::F32 {
                    "the f32 profile".to_owned()
                } else {
                    profile_name(next).to_owned()
                };
                notices.push(format!(
                    "{} unavailable: {error}; using {using}",
                    profile_name(matmul)
                ));
            } else if options.cuda_matmul.is_some() {
                notices.push(format!(
                    "decode graph unavailable: {error}; continuing without it"
                ));
            }
        }
        first_error.get_or_insert(error);
    }
    Err(first_error.expect("at least one CUDA profile was tried"))
}

/// (profile, decode graph) pairs in the order tried. Without an explicit
/// profile: int8-fast, f32-tiled, then f32 (which every quantization supports;
/// Q5_K needs it without the decode graph). An implicit decode graph is
/// dropped before the profile is.
#[cfg(feature = "cuda")]
fn cuda_attempts(options: &EngineOptions) -> Vec<(crate::cuda::CudaMatmul, bool)> {
    use crate::cuda::CudaMatmul;
    let graphs = |matmul: CudaMatmul| -> Vec<bool> {
        match options.decode_graph {
            Some(graph) => vec![graph],
            None if fast_decode(matmul) => vec![true, false],
            None => vec![false],
        }
    };
    let profiles = match options.cuda_matmul {
        Some(matmul) => vec![(matmul, graphs(matmul))],
        // int8-fast falls straight to f32-tiled: without its graph it is not
        // faster than f32-tiled with one.
        None => vec![
            (CudaMatmul::Int8Fast, graphs(CudaMatmul::Int8Fast)[..1].to_vec()),
            (CudaMatmul::F32Tiled, graphs(CudaMatmul::F32Tiled)),
            (CudaMatmul::F32, graphs(CudaMatmul::F32)),
        ],
    };
    profiles
        .into_iter()
        .flat_map(|(matmul, graphs)| graphs.into_iter().map(move |graph| (matmul, graph)))
        .collect()
}

/// Profiles with f32-tiled decode: decode graph and large prefill chunks pay off.
#[cfg(feature = "cuda")]
fn fast_decode(matmul: crate::cuda::CudaMatmul) -> bool {
    use crate::cuda::CudaMatmul;
    matches!(
        matmul,
        CudaMatmul::F32Tiled | CudaMatmul::Int8Prefill | CudaMatmul::Int8Fast
    )
}

/// Prefill batch/chunk: 4096 tokens with the tiled kernels (one chunk covers a
/// 4k prompt), else 128/64.
#[cfg(feature = "cuda")]
fn cuda_config(options: &EngineOptions, context: u32, matmul: crate::cuda::CudaMatmul) -> Config {
    const FAST_PREFILL_TOKENS: u32 = 4096;
    let fast = fast_decode(matmul);
    let batch = options.batch_tokens.unwrap_or(if fast {
        FAST_PREFILL_TOKENS.min(context)
    } else {
        CPU_BATCH_TOKENS.min(context)
    });
    let chunk = options.prefill_chunk.unwrap_or(if fast {
        FAST_PREFILL_TOKENS.min(batch)
    } else {
        CPU_PREFILL_CHUNK.min(batch)
    });
    Config {
        context_tokens: context,
        batch_tokens: batch,
        prefill_chunk: chunk,
        threads: options.threads,
    }
}

#[cfg(feature = "cuda")]
fn profile_name(matmul: crate::cuda::CudaMatmul) -> &'static str {
    use crate::cuda::CudaMatmul;
    use clap::ValueEnum;
    match matmul {
        CudaMatmul::F32 => "f32",
        CudaMatmul::F32Tiled => "f32-tiled",
        CudaMatmul::Int8Prefill => "int8-prefill",
        CudaMatmul::Int8Fast => "int8-fast",
        // Research profiles: named once per open, so leaking is bounded.
        other => other
            .to_possible_value()
            .map_or("cuda", |value| Box::leak(value.get_name().to_owned().into_boxed_str())),
    }
}

#[cfg(feature = "cuda")]
fn default_kernel_cache() -> Option<std::path::PathBuf> {
    use std::path::PathBuf;
    let from_env = |name: &str| std::env::var_os(name).filter(|value| !value.is_empty());
    if let Some(dir) = from_env("PALITO_CUDA_KERNEL_CACHE") {
        return Some(PathBuf::from(dir));
    }
    let base = from_env("XDG_CACHE_HOME")
        .map(PathBuf::from)
        .or_else(|| from_env("HOME").map(|home| PathBuf::from(home).join(".cache")))?;
    Some(base.join("palito").join("kernels"))
}
