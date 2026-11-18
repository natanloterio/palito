//! Batched-prefill only. --ids reads a JSON token array and --output names a new
//! JSONL file. --cases reads cuda-precision's case map and --output names a new
//! directory, with one JSONL per case. Graph replay is deliberately disabled.
use clap::Parser;
use local_inference::{
    Config, Error, Result,
    backend::Backend,
    cuda::{CudaBackend, CudaMatmul, CudaOptions, diagnostics::CaptureOptions},
};
use serde::Deserialize;
use std::{collections::BTreeMap, fs, path::PathBuf};

#[derive(Parser)]
struct Args {
    #[arg(long)]
    model: PathBuf,
    #[arg(long, required_unless_present = "cases", conflicts_with = "cases")]
    ids: Option<PathBuf>,
    #[arg(long, required_unless_present = "ids", conflicts_with = "ids")]
    cases: Option<PathBuf>,
    /// Comma-separated zero-based layer indices.
    #[arg(long, value_delimiter = ',', required = true)]
    layers: Vec<usize>,
    /// Projection stages to retain; omit to retain all. Routing is always saved.
    #[arg(long, value_delimiter = ',')]
    stages: Vec<String>,
    /// Comma-separated global zero-based prompt positions.
    #[arg(long, value_delimiter = ',', required = true)]
    positions: Vec<usize>,
    #[arg(long)]
    output: PathBuf,
    #[arg(long, value_enum, default_value_t = CudaMatmul::F32)]
    matmul: CudaMatmul,
    /// Retained numeric payload and record storage, excluding JSON encoding.
    #[arg(long, default_value_t = 64 * 1024 * 1024)]
    max_bytes: usize,
    #[arg(long, default_value_t = 1024)]
    context: u32,
    #[arg(long, default_value_t = 128)]
    batch: u32,
    #[arg(long, default_value_t = 128)]
    chunk: usize,
}

#[derive(Deserialize)]
struct Case {
    ids: Vec<i32>,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let many = args.cases.is_some();
    let cases: BTreeMap<String, Case> = if let Some(path) = &args.cases {
        serde_json::from_slice(&fs::read(path)?)?
    } else {
        BTreeMap::from([(
            "ids".into(),
            Case {
                ids: serde_json::from_slice(&fs::read(args.ids.as_ref().unwrap())?)?,
            },
        )])
    };
    if args.chunk < 2
        || args.chunk > args.batch as usize
        || args.chunk > 512
        || args.batch > args.context
        || cases.is_empty()
        || args.max_bytes == 0
        || cases.iter().any(|(name, case)| {
            name.is_empty()
                || name == "."
                || name == ".."
                || !name
                    .bytes()
                    .all(|c| c.is_ascii_alphanumeric() || b"_-.".contains(&c))
                || case.ids.len() > args.context as usize
                || case.ids.len() < 2
                || case.ids.chunks(args.chunk).any(|chunk| chunk.len() < 2)
                || args.positions.iter().any(|&p| p >= case.ids.len())
        })
    {
        return Err(Error::Invalid(
            "capture needs valid cases, positions and 2..=512-token chunks; no singleton tails"
                .into(),
        ));
    }
    let config = Config {
        context_tokens: args.context,
        batch_tokens: args.batch,
        prefill_chunk: args.chunk as u32,
        threads: 4,
    };
    config.validate()?;
    if many {
        fs::create_dir(&args.output)?;
    } else if args.output.try_exists()? {
        return Err(Error::Invalid("capture output already exists".into()));
    }
    let mut backend = CudaBackend::open(
        &args.model,
        &config,
        CudaOptions {
            matmul: args.matmul,
            decode_graph: false,
            ..CudaOptions::default()
        },
    )?;
    for (name, case) in cases {
        backend.reset()?;
        backend.enable_capture(CaptureOptions {
            layers: args.layers.clone(),
            stages: args.stages.clone(),
            positions: args.positions.clone(),
            max_bytes: args.max_bytes,
        })?;
        for (index, chunk) in case.ids.chunks(args.chunk).enumerate() {
            backend.evaluate(chunk, (index * args.chunk) as u32)?;
        }
        let output = if many {
            args.output.join(format!("{name}.jsonl"))
        } else {
            args.output.clone()
        };
        backend.save_capture(&output)?;
        println!("{}", output.display());
    }
    Ok(())
}
