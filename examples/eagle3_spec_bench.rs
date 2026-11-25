//! End-to-end EAGLE-3 speculative decoding benchmark on Palito CUDA.
//!
//! Loads a quantized target model (e.g. Llama 3.1 8B QTIP3 + Q6_K head) and a
//! PLT1 EAGLE-3 draft head (`eagle3-q8_0.palito`), runs greedy generation both
//! autoregressively and with EAGLE-3 (`k <= 7` draft chain + `verify_chain_greedy`),
//! verifies 100% bit-exact token parity (`exact_parity=true`), and reports
//! wall-clock tok/s, acceptance length `tau`, and speedup.

#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("eagle3-spec-bench requires the cuda feature");
    std::process::exit(2);
}

#[cfg(feature = "cuda")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use clap::Parser;
    use palito::{
        Config,
        backend::Backend,
        cuda::{CudaBackend, CudaMatmul, CudaOptions},
        speculative::SpecTree,
    };
    use std::{path::PathBuf, time::Instant};

    #[derive(Parser)]
    struct Args {
        #[arg(long)]
        model: PathBuf,
        #[arg(long)]
        eagle3: PathBuf,
        #[arg(long, default_value_t = 128)]
        tokens: usize,
        #[arg(long, default_value_t = 1024)]
        context: u32,
        /// Draft chain length per cycle (1..=15; total verified = 1 root + draft_len <= 16).
        #[arg(long, default_value_t = 5)]
        draft_len: usize,
        /// Use 16-node branching draft tree (depth 6, top-3/top-2) with bitmask tree attention.
        #[arg(long)]
        tree: bool,
        /// Use compact 8-node branching draft tree (depth 5, top-2, 5 EAGLE-3 steps).
        #[arg(long)]
        tree8: bool,
        /// Hybrid SuffixDecoding (fast-path) + EAGLE-3 (fallback) for maximum tok/s.
        #[arg(long)]
        hybrid: bool,
        /// Optional PLT1 (.palito) HYB Q=5 projection overlay.
        #[arg(long)]
        plt1: Option<PathBuf>,
        /// CUDA matmul profile: f32-tiled or int8-fast (FP16 KV + Tensor-Core attention).
        #[arg(long, default_value = "f32-tiled")]
        matmul: String,
    }

    let args = Args::parse();
    let draft_len = args.draft_len.clamp(1, 15);
    let matmul = match args.matmul.as_str() {
        "f32-tiled" => CudaMatmul::F32Tiled,
        "int8-fast" => CudaMatmul::Int8Fast,
        other => return Err(format!("unsupported matmul {other}").into()),
    };

    let config = Config {
        context_tokens: args.context,
        batch_tokens: 1024,
        prefill_chunk: 1024,
        threads: 4,
    };
    let mut backend = CudaBackend::open(
        &args.model,
        &config,
        CudaOptions {
            matmul,
            decode_graph: true,
            ..Default::default()
        },
    )?;
    #[cfg(feature = "qtip-research")]
    if let Some(plt1_path) = &args.plt1 {
        backend.load_plt1_projections(plt1_path)?;
        println!("Loaded PLT1 HYB Q=5 overlay: {}", plt1_path.display());
    }

    let prompts = [
        (
            "general",
            "<|start_header_id|>user<|end_header_id|>\n\nExplain in clear, structured steps how GPU tensor cores accelerate matrix multiplication in modern large language models, covering warp-level MMA instructions, shared memory tiling, and memory bandwidth bottlenecks.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
        ),
        (
            "coding",
            "<|start_header_id|>user<|end_header_id|>\n\nWrite a complete, idiomatic Rust implementation of an LRU cache with O(1) get and put operations, including unit tests.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
        ),
        (
            "refactor",
            "<|start_header_id|>user<|end_header_id|>\n\nRewrite this Rust struct and impl block to add a `scale: f32` field to `TensorHeader`, keeping all existing fields and methods intact and returning the full updated code:\n```rust\npub struct TensorHeader {\n    pub name: String,\n    pub rows: u32,\n    pub cols: u32,\n    pub offset: u64,\n    pub size: u64,\n}\nimpl TensorHeader {\n    pub fn byte_end(&self) -> u64 {\n        self.offset + self.size\n    }\n    pub fn num_elements(&self) -> u64 {\n        u64::from(self.rows) * u64::from(self.cols)\n    }\n}\n```<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
        ),
    ];

    // 1. Measure baseline BEFORE loading EAGLE-3 (so baseline has zero tap overhead).
    let mut baselines = Vec::new();
    for (label, text) in prompts {
        let prompt = backend.tokenize(text)?;
        let run_base = |b: &mut CudaBackend| -> Result<(Vec<i32>, f64), palito::Error> {
            b.reset()?;
            b.evaluate_for_sampling(&prompt, 0)?;
            let mut pos = prompt.len() as u32;
            let mut out = Vec::with_capacity(args.tokens);
            let t0 = Instant::now();
            let mut next = b.sample_greedy()?;
            out.push(next);
            while out.len() < args.tokens && !b.is_eog(next) {
                b.evaluate_for_sampling(&[next], pos)?;
                pos += 1;
                next = b.sample_greedy()?;
                out.push(next);
            }
            Ok((out, t0.elapsed().as_secs_f64()))
        };
        let _ = run_base(&mut backend)?; // warmup
        let (toks, secs) = run_base(&mut backend)?;
        baselines.push((label, prompt, toks, secs));
    }

    // 2. Load PLT1 EAGLE-3 draft head and run speculative decoding on the same prompts.
    backend.load_eagle3(&args.eagle3)?;

    for (label, prompt, base_tokens, base_secs) in baselines {
        let base_tps = base_tokens.len() as f64 / base_secs;
        let run_eagle = |b: &mut CudaBackend| -> Result<(Vec<i32>, f64, usize), palito::Error> {
            b.reset()?;
            let mut suffix = palito::speculative::SuffixDrafter::new(8);
            suffix.push_tokens(&prompt.iter().map(|&t| t as u32).collect::<Vec<_>>());
            b.evaluate_for_sampling(&prompt, 0)?;
            let prompt_len = prompt.len();
            let mut pos = prompt_len as u32;
            let mut out = Vec::with_capacity(args.tokens);

            let first = b.sample_greedy()?;
            out.push(first);
            suffix.push_tokens(&[first as u32]);
            let mut root = first;

            // Prompt-phase EAGLE-3 KV cache initialization (part of prefill, before decode timer).
            let mut init_tokens = Vec::with_capacity(prompt_len);
            init_tokens.extend_from_slice(&prompt[1..]);
            init_tokens.push(first);
            b.eagle3_ingest_tapped_to_dev(&init_tokens, 0)?;
            let mut eagle_pos = prompt_len;
            let mut steps = 0usize;
            let mut t_draft = 0.0f64;
            let mut t_verify = 0.0f64;
            let mut t_ingest = 0.0f64;
            let t0 = Instant::now();

            while out.len() < args.tokens && !b.is_eog(root) {
                let remaining = args.tokens - out.len();
                steps += 1;

                let td0 = Instant::now();
                let suffix_draft = if args.hybrid {
                    suffix.draft(draft_len.min(remaining))
                } else {
                    Vec::new()
                };
                let use_tree = (args.tree || args.tree8) && suffix_draft.len() < 3;
                let (tree, used_suffix) = if suffix_draft.len() >= 3 {
                    (SpecTree::from_chain(root as u32, &suffix_draft), true)
                } else if args.tree8 {
                    (b.eagle3_draft_tree_8(root, eagle_pos)?, false)
                } else if args.tree {
                    (b.eagle3_draft_tree(root, eagle_pos)?, false)
                } else {
                    let k = draft_len.min(remaining);
                    let draft = b.eagle3_draft_chain_from_slot0(eagle_pos, k)?;
                    (
                        SpecTree::from_chain(
                            root as u32,
                            &draft.iter().map(|&tok| tok as u32).collect::<Vec<_>>(),
                        ),
                        false,
                    )
                };
                t_draft += td0.elapsed().as_secs_f64();

                let tv0 = Instant::now();
                let target_i32 = if !used_suffix && use_tree {
                    b.verify_tree_greedy(&tree, pos)?
                } else {
                    let chain_i32: Vec<i32> = tree.tokens.iter().map(|&tok| tok as i32).collect();
                    b.verify_chain_greedy(&chain_i32, pos)?
                };
                t_verify += tv0.elapsed().as_secs_f64();
                let target_u32: Vec<u32> = target_i32.iter().map(|&t| t as u32).collect();
                let (accepted_u32, path_nodes) = tree.verify_greedy(&target_u32);

                let mut accepted_i32 = Vec::with_capacity(accepted_u32.len());
                let mut hit_eog = false;
                for &tok_u32 in &accepted_u32 {
                    if out.len() >= args.tokens {
                        break;
                    }
                    let tok = tok_u32 as i32;
                    out.push(tok);
                    suffix.push_tokens(&[tok_u32]);
                    accepted_i32.push(tok);
                    root = tok;
                    if b.is_eog(tok) {
                        hit_eog = true;
                        break;
                    }
                }
                let pushed = accepted_i32.len();
                if use_tree {
                    b.compact_tree_kv(pos, &path_nodes[..pushed])?;
                } else {
                    b.rewind_to(pos + pushed as u32)?;
                }
                pos += pushed as u32;
                if hit_eog || out.len() >= args.tokens {
                    break;
                }
                // Feed the accepted tokens paired with target's freshly tapped features.
                // Zero-sync: writes first draft token directly to GPU slot 0, so the
                // next step's draft chain runs back-to-back with ZERO sync between ingest & draft!
                let ti0 = Instant::now();
                let need_eagle_next = !args.hybrid
                    || suffix.draft(draft_len.min(args.tokens - out.len())).len() < 3;
                if need_eagle_next && !args.tree && !args.tree8 {
                    b.eagle3_ingest_tapped_to_dev(&accepted_i32, eagle_pos)?;
                } else {
                    b.eagle3_ingest_tapped_opt(&accepted_i32, eagle_pos, false)?;
                }
                t_ingest += ti0.elapsed().as_secs_f64();
                eagle_pos += pushed;
            }
            let total = t0.elapsed().as_secs_f64();
            eprintln!(
                "  [{label:7} breakdown] draft={:.1} ms ({:.2}/step) | verify={:.1} ms ({:.2}/step) | ingest={:.1} ms ({:.2}/step)",
                t_draft * 1e3,
                t_draft * 1e3 / steps.max(1) as f64,
                t_verify * 1e3,
                t_verify * 1e3 / steps.max(1) as f64,
                t_ingest * 1e3,
                t_ingest * 1e3 / steps.max(1) as f64,
            );
            Ok((out, total, steps))
        };

        let _ = run_eagle(&mut backend)?; // warmup
        let (spec_tokens, spec_secs, steps) = run_eagle(&mut backend)?;
        let spec_tps = spec_tokens.len() as f64 / spec_secs;
        let tau = spec_tokens.len() as f64 / steps.max(1) as f64;
        let parity = base_tokens == spec_tokens;
        println!(
            "[{label:7}] Baseline: {:6.1} tok/s ({:.2} ms) | EAGLE-3 (k={draft_len}): {:6.1} tok/s ({:.2} ms) | steps={steps:2} | tau={tau:.2} | speedup={:.2}x | exact_parity={parity}",
            base_tps,
            base_secs * 1e3,
            spec_tps,
            spec_secs * 1e3,
            base_secs / spec_secs
        );
    }
    Ok(())
}
