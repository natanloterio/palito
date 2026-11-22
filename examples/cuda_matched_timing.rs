//! Fixed-input timing counterpart to native/matched_timing.cpp.
#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("cuda_matched_timing requires the cuda feature");
    std::process::exit(2);
}

#[cfg(feature = "cuda")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use palito::{
        Config,
        backend::Backend,
        cuda::{CudaBackend, CudaMatmul, CudaOptions},
    };
    use serde_json::json;
    use std::{fs, path::Path, time::Instant};

    let args: Vec<_> = std::env::args().skip(1).collect();
    let [model, fixture, warmups, runs] = args.as_slice() else {
        return Err("usage: cuda_matched_timing MODEL FIXTURE WARMUPS RUNS".into());
    };
    let (warmups, runs): (usize, usize) = (warmups.parse()?, runs.parse()?);
    if runs == 0 || runs > 100 || warmups > 20 {
        return Err("invalid timing counts".into());
    }
    let text = fs::read_to_string(fixture)?;
    let lines: Vec<_> = text.lines().collect();
    let [prompt, continuation] = lines.as_slice() else {
        return Err("fixture needs a prompt line and a continuation line".into());
    };
    let parse = |s: &str| {
        s.split_whitespace()
            .map(str::parse)
            .collect::<Result<Vec<i32>, _>>()
    };
    let prompt = parse(prompt)?;
    let continuation = parse(continuation)?;
    if prompt.is_empty() || prompt.len() > 512 || continuation.len() > 31 {
        return Err("fixture exceeds timing shape limits".into());
    }
    let mut backend = CudaBackend::open(
        Path::new(model),
        &Config {
            context_tokens: 1024,
            batch_tokens: 512,
            prefill_chunk: 512,
            threads: 4,
        },
        CudaOptions {
            matmul: CudaMatmul::ResidualMma3Research,
            decode_graph: true,
            ..Default::default()
        },
    )?;
    for greedy in [true, false] {
        for i in 0..if greedy { 1 } else { warmups + runs } {
            let mut ids = Vec::with_capacity(1 + continuation.len());
            let start = Instant::now();
            backend.reset()?;
            let prefill_start = Instant::now();
            backend.evaluate_for_sampling(&prompt, 0)?;
            ids.push(backend.sample_greedy()?);
            let prefill_end = Instant::now();
            for (step, &forced) in continuation.iter().enumerate() {
                let id = if greedy { ids[step] } else { forced };
                backend.evaluate_for_sampling(&[id], (prompt.len() + step) as u32)?;
                ids.push(backend.sample_greedy()?);
            }
            let end = Instant::now();
            if greedy || i >= warmups {
                println!(
                    "{}",
                    json!({
                        "greedy":greedy,"sample":i.saturating_sub(warmups),"output_ids":ids,
                        "prefill_ns":(prefill_end-prefill_start).as_nanos(),
                        "decode_ns":(end-prefill_end).as_nanos(),"request_ns":(end-start).as_nanos()
                    })
                );
            }
        }
    }
    Ok(())
}
