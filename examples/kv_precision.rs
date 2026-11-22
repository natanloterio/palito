//! Full-vocabulary logit dumps for comparing CUDA profiles (e.g. int8-fast with
//! F16 KV against the F32 oracle).
//!
//! `dump` prefills the first `--tokens` tokens of a text file and records the
//! logits at the prompt boundary and after each of `--steps - 1` decode steps.
//! Decode inputs are the run's own greedy tokens, or `--teacher`'s (a previous
//! dump's JSON) so every run sees the same positions. `compare` reports, per
//! step, max_abs, cosine and whether the argmax agrees.
use clap::{Parser, Subcommand};
use palito::{
    Config, Error, Result,
    backend::Backend,
    cuda::{CudaBackend, CudaMatmul, CudaOptions},
};
use serde::{Deserialize, Serialize};
use std::{fs, path::PathBuf};

#[derive(Parser)]
struct Args {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    Dump {
        #[arg(long)]
        model: PathBuf,
        #[arg(long)]
        prompt: PathBuf,
        #[arg(long)]
        tokens: usize,
        #[arg(long, value_enum)]
        matmul: CudaMatmul,
        #[arg(long)]
        decode_graph: bool,
        #[arg(long, default_value_t = 4096)]
        context: u32,
        #[arg(long, default_value_t = 512)]
        chunk: u32,
        #[arg(long, default_value_t = 16)]
        steps: usize,
        #[arg(long)]
        teacher: Option<PathBuf>,
        /// Writes <output>.json (ids) and <output>.bin (steps x vocab f32 LE).
        #[arg(long)]
        output: PathBuf,
    },
    Compare {
        #[arg(long)]
        reference: PathBuf,
        #[arg(long)]
        candidate: PathBuf,
    },
}

#[derive(Serialize, Deserialize)]
struct Dump {
    matmul: String,
    decode_graph: bool,
    prompt: Vec<i32>,
    /// Token fed after each recorded step (the last one is never evaluated).
    inputs: Vec<i32>,
    argmax: Vec<i32>,
    vocab: usize,
}

fn argmax(logits: &[f32]) -> i32 {
    let mut best = 0;
    for (index, value) in logits.iter().enumerate() {
        if *value > logits[best] {
            best = index;
        }
    }
    best as i32
}

fn with_extension(path: &std::path::Path, extension: &str) -> PathBuf {
    let mut name = path.as_os_str().to_owned();
    name.push(extension);
    PathBuf::from(name)
}

#[allow(clippy::too_many_arguments)]
fn dump(
    model: PathBuf,
    prompt: PathBuf,
    tokens: usize,
    matmul: CudaMatmul,
    decode_graph: bool,
    context: u32,
    chunk: u32,
    steps: usize,
    teacher: Option<PathBuf>,
    output: PathBuf,
) -> Result<()> {
    if tokens == 0 || steps == 0 || chunk == 0 || tokens + steps > context as usize {
        return Err(Error::Invalid(
            "need 0 < tokens, tokens + steps <= context".into(),
        ));
    }
    let teacher: Option<Dump> = teacher
        .map(|path| -> Result<_> { Ok(serde_json::from_slice(&fs::read(path)?)?) })
        .transpose()?;
    let config = Config {
        context_tokens: context,
        batch_tokens: chunk,
        prefill_chunk: chunk,
        threads: 4,
    };
    let mut backend = CudaBackend::open(
        &model,
        &config,
        CudaOptions {
            matmul,
            decode_graph,
            ..CudaOptions::default()
        },
    )?;
    let ids = match &teacher {
        Some(teacher) => teacher.prompt.clone(),
        None => {
            let text = fs::read_to_string(&prompt)?;
            let ids = backend.tokenize(&text)?;
            if ids.len() < tokens {
                return Err(Error::Invalid(format!(
                    "prompt has only {} tokens",
                    ids.len()
                )));
            }
            ids[..tokens].to_vec()
        }
    };
    backend.reset()?;
    for (index, piece) in ids.chunks(chunk as usize).enumerate() {
        backend.evaluate(piece, (index * chunk as usize) as u32)?;
    }
    let mut record = Dump {
        matmul: format!("{matmul:?}"),
        decode_graph,
        prompt: ids.clone(),
        inputs: Vec::new(),
        argmax: Vec::new(),
        vocab: 0,
    };
    let mut bytes = Vec::new();
    for step in 0..steps {
        let logits = backend.logits()?;
        record.vocab = logits.len();
        record.argmax.push(argmax(logits));
        bytes.extend(logits.iter().flat_map(|value| value.to_le_bytes()));
        let next = match &teacher {
            Some(teacher) => *teacher
                .inputs
                .get(step)
                .ok_or_else(|| Error::Invalid("teacher has fewer steps".into()))?,
            None => record.argmax[step],
        };
        record.inputs.push(next);
        if step + 1 < steps {
            backend.evaluate(&[next], (ids.len() + step) as u32)?;
        }
    }
    fs::write(with_extension(&output, ".bin"), bytes)?;
    fs::write(
        with_extension(&output, ".json"),
        serde_json::to_vec(&record)?,
    )?;
    Ok(())
}

fn load(path: &std::path::Path) -> Result<(Dump, Vec<f32>)> {
    let record: Dump = serde_json::from_slice(&fs::read(with_extension(path, ".json"))?)?;
    let logits = fs::read(with_extension(path, ".bin"))?
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();
    Ok((record, logits))
}

fn compare(reference: PathBuf, candidate: PathBuf) -> Result<()> {
    let (expected, expected_logits) = load(&reference)?;
    let (actual, actual_logits) = load(&candidate)?;
    if expected.vocab != actual.vocab || expected.prompt != actual.prompt {
        return Err(Error::Invalid(
            "dumps differ in vocabulary or prompt".into(),
        ));
    }
    let steps = expected.argmax.len().min(actual.argmax.len());
    let (mut worst_abs, mut worst_cosine, mut same) = (0.0f64, 1.0f64, 0);
    let mut prompt_boundary = (0.0f64, 1.0f64);
    for step in 0..steps {
        let range = step * expected.vocab..(step + 1) * expected.vocab;
        let (e, a) = (&expected_logits[range.clone()], &actual_logits[range]);
        let (mut max_abs, mut dot, mut ee, mut aa) = (0.0f64, 0.0, 0.0, 0.0);
        for (&x, &y) in e.iter().zip(a) {
            let (x, y) = (f64::from(x), f64::from(y));
            max_abs = max_abs.max((x - y).abs());
            dot += x * y;
            ee += x * x;
            aa += y * y;
        }
        let cosine = dot / (ee.sqrt() * aa.sqrt());
        let agree = expected.argmax[step] == actual.argmax[step];
        same += usize::from(agree);
        if step == 0 {
            prompt_boundary = (max_abs, cosine);
        }
        worst_abs = worst_abs.max(max_abs);
        worst_cosine = worst_cosine.min(cosine);
        println!(
            "{}",
            serde_json::json!({"step": step, "max_abs": max_abs, "cosine": cosine,
                "same_argmax": agree})
        );
    }
    println!(
        "{}",
        serde_json::json!({"summary": true, "prompt_tokens": expected.prompt.len(),
            "steps": steps, "max_abs": worst_abs, "min_cosine": worst_cosine,
            "same_argmax": same, "prompt_boundary_max_abs": prompt_boundary.0,
            "prompt_boundary_cosine": prompt_boundary.1})
    );
    Ok(())
}

fn main() -> Result<()> {
    match Args::parse().command {
        Command::Dump {
            model,
            prompt,
            tokens,
            matmul,
            decode_graph,
            context,
            chunk,
            steps,
            teacher,
            output,
        } => dump(
            model,
            prompt,
            tokens,
            matmul,
            decode_graph,
            context,
            chunk,
            steps,
            teacher,
            output,
        ),
        Command::Compare {
            reference,
            candidate,
        } => compare(reference, candidate),
    }
}
