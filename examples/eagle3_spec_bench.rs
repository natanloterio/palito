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
        /// Draft chain length per cycle (1..=7; total verified = 1 root + draft_len <= 8).
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
    }

    let args = Args::parse();
    let draft_len = args.draft_len.clamp(1, 7);

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
            matmul: CudaMatmul::F32Tiled,
            decode_graph: true,
            ..Default::default()
        },
    )?;

    let prompts = [
        (
            "general",
            "<|start_header_id|>user<|end_header_id|>\n\nExplain in clear, structured steps how GPU tensor cores accelerate matrix multiplication in modern large language models, covering warp-level MMA instructions, shared memory tiling, and memory bandwidth bottlenecks.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
        ),
        (
            "coding",
            "<|start_header_id|>user<|end_header_id|>\n\nWrite a complete, idiomatic Rust implementation of an LRU cache with O(1) get and put operations, including unit tests.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
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
            let mut next_draft = b.eagle3_ingest_tapped(&init_tokens, 0)?;
            let mut eagle_pos = prompt_len;
            let mut steps = 0usize;
            let t0 = Instant::now();

            while out.len() < args.tokens && !b.is_eog(root) {
                let remaining = args.tokens - out.len();
                steps += 1;

                let suffix_draft = if args.hybrid {
                    suffix.draft(7.min(remaining))
                } else {
                    Vec::new()
                };
                let use_tree = (args.tree || args.tree8) && suffix_draft.len() < 3;
                let (tree, target_i32) = if suffix_draft.len() >= 3 {
                    let t = SpecTree::from_chain(root as u32, &suffix_draft);
                    let chain_i32: Vec<i32> = t.tokens.iter().map(|&tok| tok as i32).collect();
                    let tgt = b.verify_chain_greedy(&chain_i32, pos)?;
                    (t, tgt)
                } else if args.tree8 {
                    let t = b.eagle3_draft_tree_8(root, eagle_pos)?;
                    let tgt = b.verify_tree_greedy(&t, pos)?;
                    (t, tgt)
                } else if args.tree {
                    let t = b.eagle3_draft_tree(root, eagle_pos)?;
                    let tgt = b.verify_tree_greedy(&t, pos)?;
                    (t, tgt)
                } else {
                    let k = draft_len.min(remaining);
                    let draft = b.eagle3_draft_chain(next_draft, eagle_pos, k)?;
                    let t = SpecTree::from_chain(
                        root as u32,
                        &draft.iter().map(|&tok| tok as u32).collect::<Vec<_>>(),
                    );
                    let chain_i32: Vec<i32> = t.tokens.iter().map(|&tok| tok as i32).collect();
                    let tgt = b.verify_chain_greedy(&chain_i32, pos)?;
                    (t, tgt)
                };
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
                next_draft = b.eagle3_ingest_tapped(&accepted_i32, eagle_pos)?;
                eagle_pos += pushed;
            }
            Ok((out, t0.elapsed().as_secs_f64(), steps))
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
