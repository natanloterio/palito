//! Unified `palito convert` command: converts a GGUF model into a single
//! self-contained `.palito` (`PLT1`) bundle in one step (calibration Hessians ->
//! GPU-accelerated HYB Q=5 quantization -> skeleton GGUF + optional EAGLE-3 bundling).

use clap::Args;
use palito::{Error, Result};
use std::path::PathBuf;

#[derive(Args)]
pub(crate) struct ConvertArgs {
    /// Source GGUF model (e.g. Q6_K, Q8_0, or F16).
    #[arg(short, long)]
    pub(crate) input: PathBuf,
    /// Destination single-file `.palito` bundle path.
    #[arg(short, long)]
    pub(crate) output: PathBuf,
    /// Optional pre-packed EAGLE-3 draft head (`.palito`) to embed for speculative decoding.
    #[arg(long)]
    pub(crate) eagle3: Option<PathBuf>,
    /// Optional pre-computed Hessians directory (skips calibration step).
    #[arg(long)]
    pub(crate) hessians: Option<PathBuf>,
    /// Optional calibration text file (e.g. WikiText-2 `wiki.train.raw`). If omitted and
    /// `--hessians` is not set, uses a built-in English/code calibration corpus on CUDA.
    #[arg(long)]
    pub(crate) calibration_text: Option<PathBuf>,
    /// Skip Hessian calibration and use damped identity `H = I` (faster, lower quality).
    #[arg(long)]
    pub(crate) no_hessians: bool,
    /// Bits per weight preset: `3` (3.0 bpw, L=10 B=6 Q=5, recommended) or `2` (2.0 bpw, L=12 B=4 Q=5).
    #[arg(long, default_value_t = 3, value_parser = clap::value_parser!(u32).range(2..=3))]
    pub(crate) bits: u32,
    /// Calibration chunks to run when collecting Hessians (default: 64 with --calibration-text, 8 with built-in corpus).
    #[arg(long)]
    pub(crate) chunks: Option<usize>,
    /// Tokens per calibration chunk.
    #[arg(long, default_value_t = 512)]
    pub(crate) context: u32,
    /// Layers per calibration forward pass (0 = all layers at once).
    #[arg(long, default_value_t = 16)]
    pub(crate) layers_per_pass: usize,
    /// Skip per-expert `ffn_down_exps` Hessians on MoE models (saves RAM/disk).
    #[arg(long)]
    pub(crate) skip_down_exps: bool,
    /// Do not fold RMSNorm and residual R_1 Hadamard rotation into weights.
    #[arg(long)]
    pub(crate) no_fold_rht: bool,
    /// Optional limit on layers to convert (for quick smoke tests).
    #[arg(long)]
    pub(crate) max_layers: Option<usize>,
    /// Keep intermediate Hessians and raw layers `.palito` files instead of deleting them.
    #[arg(long)]
    pub(crate) keep_temp: bool,
    /// CPU threads for tokenization and host math.
    #[arg(long, default_value_t = 8)]
    pub(crate) threads: u32,
    /// Overwrite --output if it already exists.
    #[arg(short, long)]
    pub(crate) force: bool,
}

/// Files a conversion creates: temporaries (kept with --keep-temp) and the
/// output while it is being written. Whatever is still listed is deleted when
/// the guard drops (error or panic) or on Ctrl-C, so a failed conversion does
/// not leave gigabytes of `.tmp` data or a truncated `.palito` behind.
#[cfg(feature = "qtip-research")]
#[derive(Clone)]
struct Cleanup {
    state: std::sync::Arc<std::sync::Mutex<CleanupState>>,
}

#[cfg(feature = "qtip-research")]
#[derive(Default)]
struct CleanupState {
    temps: Vec<PathBuf>,
    partial: Option<PathBuf>,
    keep_temps: bool,
}

#[cfg(feature = "qtip-research")]
impl CleanupState {
    fn run(&mut self) {
        if let Some(path) = self.partial.take() {
            let _ = std::fs::remove_file(path);
        }
        for path in self.temps.drain(..) {
            if self.keep_temps {
                eprintln!("Kept {}", path.display());
                continue;
            }
            let _ = if path.is_dir() {
                std::fs::remove_dir_all(&path)
            } else {
                std::fs::remove_file(&path)
            };
        }
    }
}

#[cfg(feature = "qtip-research")]
impl Cleanup {
    fn install(keep_temps: bool) -> CleanupGuard {
        let cleanup = Self {
            state: std::sync::Arc::new(std::sync::Mutex::new(CleanupState {
                keep_temps,
                ..CleanupState::default()
            })),
        };
        let on_interrupt = cleanup.clone();
        let _ = ctrlc::set_handler(move || {
            eprintln!("\nInterrupted; removing partial files.");
            if let Ok(mut state) = on_interrupt.state.lock() {
                state.run();
            }
            std::process::exit(130);
        });
        CleanupGuard(cleanup)
    }

    fn with<T>(&self, f: impl FnOnce(&mut CleanupState) -> T) -> T {
        let mut state = self.state.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        f(&mut state)
    }

    fn temp(&self, path: PathBuf) {
        self.with(|state| state.temps.push(path));
    }

    fn writing(&self, path: Option<PathBuf>) {
        self.with(|state| state.partial = path);
    }
}

/// The handle `run` owns; dropping it (success, error or panic) cleans up.
#[cfg(feature = "qtip-research")]
struct CleanupGuard(Cleanup);

#[cfg(feature = "qtip-research")]
impl std::ops::Deref for CleanupGuard {
    type Target = Cleanup;
    fn deref(&self) -> &Cleanup {
        &self.0
    }
}

#[cfg(feature = "qtip-research")]
impl Drop for CleanupGuard {
    fn drop(&mut self) {
        self.0.with(CleanupState::run);
    }
}

#[cfg(not(feature = "qtip-research"))]
pub(crate) fn run(_args: ConvertArgs) -> Result<()> {
    Err(Error::Invalid(
        "`palito convert` requires a build with `--features cuda,qtip-research` (or `--features qtip-research`)"
            .into(),
    ))
}

#[cfg(feature = "qtip-research")]
pub(crate) fn run(args: ConvertArgs) -> Result<()> {
    use palito::qtip::{
        hessian::Hessians,
        plt1_export::{Plt1ExportOptions, export_gguf_to_plt1, pack_palito_bundle},
    };
    use std::time::Instant;

    if !args.input.is_file() {
        return Err(Error::Invalid(format!(
            "input GGUF not found: {}",
            args.input.display()
        )));
    }
    if args.output.exists() && !args.force {
        return Err(Error::Invalid(format!(
            "{} already exists\n  hint: pass --force to overwrite it",
            args.output.display()
        )));
    }
    let cleanup = Cleanup::install(args.keep_temp);
    if let Some(parent) = args.output.parent()
        && !parent.as_os_str().is_empty()
    {
        std::fs::create_dir_all(parent)?;
    }

    let t_total = Instant::now();
    let (state_bits, pair_bits) = match args.bits {
        2 => (12u32, 4u32),
        _ => (10u32, 6u32),
    };

    // Step 1/3: Calibration Hessians.
    let hessians = if args.no_hessians {
        eprintln!("[1/3] Skipping Hessian calibration (--no-hessians: using identity H = I)");
        Hessians::empty()
    } else if let Some(dir) = &args.hessians {
        eprintln!("[1/3] Loading pre-computed Hessians from {}", dir.display());
        Hessians::open(dir)?
    } else {
        #[cfg(feature = "cuda")]
        {
            use palito::cuda::CudaMatmul;
            use palito::qtip::hessian_cuda::{CudaHessianOptions, collect_hessians_cuda};

            let dir = args.output.with_extension("hessians.tmp");
            if dir.exists() {
                eprintln!("Removing stale {}", dir.display());
                std::fs::remove_dir_all(&dir)?;
            }
            cleanup.temp(dir.clone());
            let chunks = args.chunks.unwrap_or(if args.calibration_text.is_some() {
                64
            } else {
                8
            });
            eprintln!(
                "[1/3] Collecting calibration Hessians on CUDA ({chunks} chunks x {} tokens)...",
                args.context
            );
            collect_hessians_cuda(&CudaHessianOptions {
                model: args.input.clone(),
                text: args.calibration_text.clone(),
                output: dir.clone(),
                chunks,
                context: args.context,
                layers_per_pass: args.layers_per_pass,
                skip_down_exps: args.skip_down_exps,
                min_expert_tokens: 32,
                matmul: CudaMatmul::F32,
                threads: args.threads,
            })?;
            Hessians::open(&dir)?
        }
        #[cfg(not(feature = "cuda"))]
        {
            eprintln!(
                "[1/3] Built without CUDA: using identity H = I (pass --hessians <DIR> for Hessian-calibrated CPU export)"
            );
            Hessians::empty()
        }
    };

    // Step 2/3: Quantize layers into PLT1 HYB Q=5.
    let layers_tmp = args.output.with_extension("layers.palito.tmp");
    cleanup.temp(layers_tmp.clone());
    let opts = Plt1ExportOptions {
        state_bits,
        pair_bits,
        lut_bits: 5,
        fold_residual_rht: !args.no_fold_rht,
        max_layers: args.max_layers,
        ..Default::default()
    };
    eprintln!(
        "[2/3] Quantizing {} -> PLT1 HYB Q=5 ({}.0 bpw, L={state_bits}, B={pair_bits})...",
        args.input.display(),
        args.bits
    );
    let t_quant = Instant::now();
    let (reports, index) = export_gguf_to_plt1(
        &args.input,
        &layers_tmp,
        &hessians,
        &opts,
        |report, done, total| {
            let proxy = report
                .proxy_error
                .map(|p| format!(" proxy={p:.5}"))
                .unwrap_or_default();
            eprintln!(
                "  [{done:3}/{total:3}] {:32} rel={:.4}{proxy}",
                report.name, report.relative_error
            );
        },
    )?;
    let mean_rel = reports.iter().map(|r| r.relative_error).sum::<f64>()
        / reports.len().max(1) as f64;
    eprintln!(
        "  Quantized {} tensors ({:.2} MB) in {:.1}s | mean rel={mean_rel:.4}",
        index.tensors.len(),
        std::fs::metadata(&layers_tmp)?.len() as f64 / 1e6,
        t_quant.elapsed().as_secs_f64()
    );

    // Step 3/3: Bundle skeleton GGUF + PLT1 layers + optional EAGLE-3 into one `.palito` file.
    eprintln!("[3/3] Bundling self-contained {}...", args.output.display());
    cleanup.writing(Some(args.output.clone()));
    let bundle = pack_palito_bundle(
        &args.input,
        &layers_tmp,
        args.eagle3.as_deref(),
        &args.output,
    )?;

    cleanup.writing(None);
    drop(cleanup);

    eprintln!(
        "Done in {:.1}s -> {} ({:.2} MB total | skeleton {:.2} MB | EAGLE-3: {} tensors)",
        t_total.elapsed().as_secs_f64(),
        args.output.display(),
        bundle.total_bytes as f64 / 1e6,
        bundle.skeleton_bytes as f64 / 1e6,
        bundle.eagle3_tensors
    );
    eprintln!(
        "Run with: palito chat --model {} --stats",
        args.output.display()
    );
    Ok(())
}
