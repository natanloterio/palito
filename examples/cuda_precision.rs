use clap::Parser;
use palito::{
    Config, Executor, GenerationMode, Result,
    backend::Backend,
    cuda::{CudaBackend, CudaMatmul, CudaOptions},
};
use serde::Deserialize;
use std::{
    collections::BTreeMap,
    fs::{self, OpenOptions},
    path::PathBuf,
    time::Instant,
};

#[derive(Parser)]
struct Args {
    #[arg(long)]
    model: PathBuf,
    #[arg(long)]
    cases: PathBuf,
    #[arg(long)]
    output: PathBuf,
    #[arg(long, value_enum, default_value_t = CudaMatmul::F32)]
    matmul: CudaMatmul,
    #[arg(long)]
    decode_graph: bool,
    #[arg(long, default_value_t = 128)]
    batch: u32,
    #[arg(long, default_value_t = 128)]
    chunk: usize,
    #[arg(long, default_value_t = 3)]
    warmups: usize,
    #[arg(long, default_value_t = 7)]
    repetitions: usize,
}

#[derive(Deserialize)]
struct Case {
    ids: Vec<i32>,
    generated: u32,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let cases: BTreeMap<String, Case> = serde_json::from_slice(&fs::read(args.cases)?)?;
    if args.chunk == 0
        || args.chunk > args.batch as usize
        || args.batch > 512
        || args.repetitions == 0
        || cases.is_empty()
        || cases
            .values()
            .any(|c| c.ids.is_empty() || c.ids.len() > 1024)
    {
        return Err(palito::Error::Invalid(
            "invalid precision cases".into(),
        ));
    }
    let output = OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(args.output)?;
    let config = Config {
        context_tokens: 1024,
        batch_tokens: args.batch,
        prefill_chunk: args.chunk as u32,
        threads: 4,
    };
    let start = Instant::now();
    let mut backend = CudaBackend::open(
        &args.model,
        &config,
        CudaOptions {
            matmul: args.matmul,
            decode_graph: args.decode_graph,
            ..CudaOptions::default()
        },
    )?;
    let load_ns = start.elapsed().as_nanos();
    let mut logits = BTreeMap::new();
    for (name, case) in &cases {
        backend.reset()?;
        for (index, chunk) in case.ids.chunks(args.chunk).enumerate() {
            backend.evaluate(chunk, (index * args.chunk) as u32)?;
        }
        logits.insert(name, backend.logits()?.to_vec());
    }
    let memory = backend.memory_usage();
    let mut executor = Executor::new(backend, config)?;
    let mut measurements = BTreeMap::new();
    for (name, case) in &cases {
        for _ in 0..args.warmups {
            executor.generate(&case.ids, case.generated, GenerationMode::FixedLength)?;
        }
        let mut samples = Vec::new();
        for _ in 0..args.repetitions {
            samples.push(executor.generate(
                &case.ids,
                case.generated,
                GenerationMode::FixedLength,
            )?);
        }
        measurements.insert(name, samples);
    }
    serde_json::to_writer(
        output,
        &serde_json::json!({
            "mode": args.matmul, "decode_graph":args.decode_graph,
            "batch":args.batch, "chunk":args.chunk, "load_ns": load_ns, "memory": memory,
            "logits": logits, "measurements": measurements,
            "warmups":args.warmups, "repetitions":args.repetitions,
        }),
    )?;
    Ok(())
}
