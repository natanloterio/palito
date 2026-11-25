//! End-to-end SuffixDecoding speculative verification benchmark on Palito CUDA.
//!
//! Runs greedy generation twice on the same prompt:
//!   1. Standard autoregressive decode (baseline)
//!   2. SuffixDecoding (`SuffixDrafter` + `SpecTree` + `CudaBackend::verify_chain_greedy`)
//! Verifies 100% exact token parity (lossless) and reports tok/s, acceptance
//! length (tau), and net wall-clock speedup.

#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("suffix-spec-bench requires the cuda feature");
    std::process::exit(2);
}

#[cfg(feature = "cuda")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use clap::Parser;
    use palito::{
        Config,
        backend::Backend,
        cuda::{CudaBackend, CudaMatmul, CudaOptions},
        speculative::{SpecTree, SuffixDrafter},
    };
    use std::{path::PathBuf, time::Instant};

    #[derive(Parser)]
    struct Args {
        #[arg(long)]
        model: PathBuf,
        #[arg(long, default_value_t = 128)]
        tokens: usize,
        #[arg(long, default_value_t = 1024)]
        context: u32,
        /// Maximum draft length per step (up to 7; chain = 1 root + max_draft <= 8).
        #[arg(long, default_value_t = 7)]
        max_draft: usize,
        /// Minimum suffix match length before triggering batched verification.
        #[arg(long, default_value_t = 2)]
        min_draft: usize,
        /// CUDA matmul profile: f32-tiled or int8-fast.
        #[arg(long, default_value = "int8-fast")]
        matmul: String,
    }

    let args = Args::parse();
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

    // Realistic coding/refactoring prompt where SuffixDecoding shines (repeating struct/API names).
    let prompt_text = "\
Rewrite the following Rust struct and impl block to add a `scale: f32` field to `TensorHeader`, \
keeping all existing fields and methods intact and returning the full updated code:\n\
```rust\n\
pub struct TensorHeader {\n\
    pub name: String,\n\
    pub rows: u32,\n\
    pub cols: u32,\n\
    pub offset: u64,\n\
    pub size: u64,\n\
}\n\
impl TensorHeader {\n\
    pub fn byte_end(&self) -> u64 {\n\
        self.offset + self.size\n\
    }\n\
    pub fn num_elements(&self) -> u64 {\n\
        u64::from(self.rows) * u64::from(self.cols)\n\
    }\n\
}\n\
```\n\
Updated Rust code:\n```rust\n";

    let prompt = backend.tokenize(prompt_text)?;
    println!(
        "Model: {} | prompt={} tokens | generate={} tokens | matmul={}",
        args.model.display(),
        prompt.len(),
        args.tokens,
        args.matmul
    );

    // 1. Baseline autoregressive greedy decode (warmup + measured).
    let run_baseline = |backend: &mut CudaBackend| -> Result<(Vec<i32>, f64), palito::Error> {
        backend.reset()?;
        backend.evaluate_for_sampling(&prompt, 0)?;
        let mut pos = prompt.len() as u32;
        let mut out = Vec::with_capacity(args.tokens);
        let t0 = Instant::now();
        let mut next = backend.sample_greedy()?;
        out.push(next);
        while out.len() < args.tokens && !backend.is_eog(next) {
            backend.evaluate_for_sampling(&[next], pos)?;
            pos += 1;
            next = backend.sample_greedy()?;
            out.push(next);
        }
        let secs = t0.elapsed().as_secs_f64();
        Ok((out, secs))
    };

    let _ = run_baseline(&mut backend)?; // warmup
    let (base_tokens, base_secs) = run_baseline(&mut backend)?;
    let base_tps = base_tokens.len() as f64 / base_secs;
    println!(
        "Baseline:       {} tok in {:.2} ms ({:.1} tok/s, {:.3} ms/tok)",
        base_tokens.len(),
        base_secs * 1e3,
        base_tps,
        base_secs * 1e3 / base_tokens.len() as f64
    );

    // 2. SuffixDecoding speculative decode (warmup + measured).
    let max_draft = args.max_draft.min(7);
    let run_spec = |backend: &mut CudaBackend| -> Result<(Vec<i32>, f64, usize, usize, usize), palito::Error> {
        backend.reset()?;
        let mut drafter = SuffixDrafter::new(8);
        let prompt_u32: Vec<u32> = prompt.iter().map(|&t| t as u32).collect();
        drafter.push_tokens(&prompt_u32);

        backend.evaluate_for_sampling(&prompt, 0)?;
        let mut pos = prompt.len() as u32;
        let mut out = Vec::with_capacity(args.tokens);
        let t0 = Instant::now();

        let first = backend.sample_greedy()?;
        out.push(first);
        drafter.push_tokens(&[first as u32]);
        let mut root = first;

        let mut steps = 0usize;
        let mut spec_steps = 0usize;
        let mut total_proposed = 0usize;

        while out.len() < args.tokens && !backend.is_eog(root) {
            let remaining = args.tokens - out.len();
            let draft_u32 = drafter.draft(max_draft.min(remaining));
            steps += 1;

            if draft_u32.len() >= args.min_draft {
                spec_steps += 1;
                total_proposed += draft_u32.len();
                let tree = SpecTree::from_chain(root as u32, &draft_u32);
                let chain_i32: Vec<i32> = tree.tokens.iter().map(|&t| t as i32).collect();
                let target_i32 = backend.verify_chain_greedy(&chain_i32, pos)?;
                let target_u32: Vec<u32> = target_i32.iter().map(|&t| t as u32).collect();
                let (accepted, _) = tree.verify_greedy(&target_u32);

                let mut hit_eog = false;
                let mut pushed = 0usize;
                for &tok_u32 in &accepted {
                    if out.len() >= args.tokens {
                        break;
                    }
                    let tok = tok_u32 as i32;
                    out.push(tok);
                    drafter.push_tokens(&[tok_u32]);
                    root = tok;
                    pushed += 1;
                    if backend.is_eog(tok) {
                        hit_eog = true;
                        break;
                    }
                }
                // KV cache holds valid entries for `pos .. pos + pushed` (root + accepted[:-1]).
                pos += pushed as u32;
                backend.rewind_to(pos)?;
                if hit_eog {
                    break;
                }
            } else {
                backend.evaluate_for_sampling(&[root], pos)?;
                pos += 1;
                let next = backend.sample_greedy()?;
                out.push(next);
                drafter.push_tokens(&[next as u32]);
                root = next;
            }
        }
        let secs = t0.elapsed().as_secs_f64();
        Ok((out, secs, steps, spec_steps, total_proposed))
    };

    let _ = run_spec(&mut backend)?; // warmup
    let (spec_tokens, spec_secs, steps, spec_steps, proposed) = run_spec(&mut backend)?;
    let spec_tps = spec_tokens.len() as f64 / spec_secs;
    let tau = spec_tokens.len() as f64 / steps.max(1) as f64;

    let parity = base_tokens == spec_tokens;
    println!(
        "SuffixDecoding: {} tok in {:.2} ms ({:.1} tok/s, {:.3} ms/tok) | steps={steps} (spec={spec_steps}, proposed={proposed}) | tau={tau:.2} | speedup={:.2}x | exact_parity={parity}",
        spec_tokens.len(),
        spec_secs * 1e3,
        spec_tps,
        spec_secs * 1e3 / spec_tokens.len() as f64,
        base_secs / spec_secs
    );
    if !parity {
        let diff_idx = base_tokens
            .iter()
            .zip(&spec_tokens)
            .position(|(a, b)| a != b);
        println!("WARNING: token mismatch at index {diff_idx:?} (note: int8-fast prefill differs slightly from f32-tiled decode; re-run with --matmul f32-tiled for bit-exact parity)");
    }
    Ok(())
}
