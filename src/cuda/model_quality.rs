use super::*;
use crate::{Executor, GenerationMode};
use serde::Deserialize;
use serde_json::json;
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    fs::{self, OpenOptions},
    io::Write,
};

#[derive(Deserialize)]
struct Case {
    ids: Vec<i32>,
    chunk: usize,
    steps: usize,
}

fn prefill(backend: &mut CudaBackend, case: &Case) -> Result<()> {
    backend.reset()?;
    for (index, chunk) in case.ids.chunks(case.chunk).enumerate() {
        backend.evaluate(chunk, (index * case.chunk) as u32)?;
    }
    Ok(())
}

#[test]
#[ignore = "M8_QUALITY_MODEL/CASES/OUTPUT/MODE and optional TEACHER/BATCH; full-vocabulary arithmetic research, serial GPU"]
fn residual_full_model_quality() -> Result<()> {
    let required = |key| std::env::var(key).map_err(|e| invalid(format!("{key}: {e}")));
    let model = PathBuf::from(required("M8_QUALITY_MODEL")?);
    let output = PathBuf::from(required("M8_QUALITY_OUTPUT")?);
    let batch = std::env::var_os("M8_QUALITY_BATCH")
        .map(|value| {
            value
                .to_str()
                .and_then(|value| value.parse::<u32>().ok())
                .filter(|&value| (1..=512).contains(&value))
                .ok_or_else(|| invalid("M8_QUALITY_BATCH must be in 1..=512"))
        })
        .transpose()?
        .unwrap_or(512);
    let mode_name = required("M8_QUALITY_MODE")?;
    let fp16 = mode_name == "fp16-projection-control";
    let mode = match mode_name.as_str() {
        "f32-tiled" => CudaMatmul::F32Tiled,
        "residual-int8-research" => CudaMatmul::ResidualInt8Research,
        "residual-mma3-research" | "fp16-projection-control" => CudaMatmul::ResidualMma3Research,
        _ => {
            return Err(invalid(
                "quality mode must be f32-tiled, residual-int8-research, residual-mma3-research or fp16-projection-control",
            ));
        }
    };
    let cases: BTreeMap<String, Case> =
        serde_json::from_slice(&fs::read(required("M8_QUALITY_CASES")?)?)?;
    let teacher: Option<BTreeMap<String, Vec<i32>>> = std::env::var_os("M8_QUALITY_TEACHER")
        .map(|path| -> Result<_> { Ok(serde_json::from_slice(&fs::read(path)?)?) })
        .transpose()?;
    if cases.is_empty()
        || cases.iter().any(|(name, c)| {
            name.is_empty()
                || !name
                    .bytes()
                    .all(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b'_')
                || c.ids.is_empty()
                || c.ids.len() > 512
                || c.chunk == 0
                || c.chunk > batch as usize
                || c.steps == 0
                || c.steps > 32
        })
        || teacher.as_ref().is_some_and(|t| {
            t.len() != cases.len()
                || cases
                    .iter()
                    .any(|(name, c)| t.get(name).is_none_or(|v| v.len() != c.steps))
        })
    {
        return Err(invalid(
            "invalid bounded full-model quality cases or teacher schedule",
        ));
    }
    fs::create_dir(&output)?;
    let output = output.canonicalize()?;
    let config = Config {
        context_tokens: 1024,
        batch_tokens: batch,
        prefill_chunk: batch,
        threads: 4,
    };
    let mut backend = CudaBackend::open(
        &model,
        &config,
        CudaOptions {
            matmul: mode,
            decode_graph: true,
            ..CudaOptions::default()
        },
    )?;
    let initial_memory = backend.memory_usage().unwrap();
    if fp16 {
        let context = backend.stream.context();
        let major = context
            .attribute(sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)
            .map_err(failure)?;
        let minor = context
            .attribute(sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)
            .map_err(failure)?;
        let ptx = compile_ptx_with_opts(
            concat!(
                "#define CUDA_NATIVE_HALF 1\n",
                include_str!("kernels.cu"),
                "\n",
                include_str!("residual.cu"),
                "\n",
                include_str!("fp16_projection.cu")
            ),
            CompileOptions {
                fmad: Some(false),
                options: vec![format!("--gpu-architecture=compute_{major}{minor}")],
                ..Default::default()
            },
        )
        .map_err(failure)?;
        let module = context.load_module(ptx).map_err(failure)?;
        backend.functions.insert(
            "projection_fp16_packed",
            module
                .load_function("projection_fp16_packed")
                .map_err(failure)?,
        );
    }
    let extra = residual_workspace_bytes(&backend.weights.spec, mode)?;
    let actual_bytes = if let Some(work) = &backend.work.residual_int8 {
        assert!(backend.work.residual_mma3.is_none());
        let actual_bytes = (work.q0.len() + work.q1.len()) as u64
            + (work.scales.len() + work.sums.len()) as u64 * 4;
        assert_eq!(
            work.q0.len(),
            residual_workspace_elements(&backend.weights.spec, mode)?
        );
        actual_bytes
    } else if let Some(work) = &backend.work.residual_mma3 {
        assert_eq!(
            work.values.len(),
            residual_workspace_elements(&backend.weights.spec, mode)? * 3
        );
        work.values.len() as u64 + (work.scales.len() + work.sums.len()) as u64 * 4
    } else {
        0
    };
    assert_eq!(extra, actual_bytes);
    if extra > 0 {
        let id = cases.values().next().unwrap().ids[0];
        backend.evaluate(&[id], 0)?;
        let saved = backend.logits()?.to_vec();
        assert!(backend.evaluate(&[], 1).is_err());
        assert_eq!(backend.logits()?, saved);
        let bad = backend
            .stream
            .clone_htod(&[f32::NAN; 32])
            .map_err(failure)?;
        if mode == CudaMatmul::ResidualMma3Research {
            let work = backend.work.residual_mma3.as_mut().unwrap();
            launch!(
                backend,
                "residual_quantize32_precise3",
                (1, 1, 1),
                &bad,
                &mut work.values,
                &mut work.scales,
                &mut work.sums,
                &32i32,
                &mut backend.work.status
            );
        } else {
            let work = backend.work.residual_int8.as_mut().unwrap();
            launch!(
                backend,
                "residual_quantize32",
                (1, 1, 1),
                &bad,
                &mut work.q0,
                &mut work.q1,
                &mut work.scales,
                &mut work.sums,
                &32i32,
                &mut backend.work.status
            );
        }
        assert!(backend.evaluate(&[id], 1).is_err());
        assert!(backend.logits().is_err());
        assert!(backend.evaluate(&[id], 1).is_err());
        backend.reset()?;
        backend.evaluate(&[id], 0)?;
        assert_eq!(backend.logits()?, saved);
    } else {
        assert_eq!(extra, 0);
    }
    assert!(
        admit(
            initial_memory.numerical_reservation_bytes,
            initial_memory.numerical_reservation_bytes - 1,
            u64::MAX
        )
        .is_err()
    );
    admit(
        initial_memory.numerical_reservation_bytes,
        initial_memory.numerical_reservation_bytes,
        u64::MAX,
    )?;
    println!(
        "INT8_MODEL_PROVENANCE {}",
        json!({"mode":mode,"fp16_projection_override":fp16,"memory":initial_memory,
        "quantization_payload_bytes":extra,"quantization_workspace_reused_across_all_stages_and_layers":true,
        "exact_budget_boundary_checked":true,"quantization_failure_poison_reset_checked":extra>0,
        "decode_graph":true,"config":config})
    );
    let mut schedule = BTreeMap::new();
    for (name, case) in &cases {
        prefill(&mut backend, case)?;
        let mut chosen = Vec::new();
        for step in 0..=case.steps {
            let values = backend.logits()?;
            let greedy = greedy(values)?;
            let bytes: Vec<_> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
            let path = output.join(format!("{name}-{step}.f32"));
            OpenOptions::new()
                .create_new(true)
                .write(true)
                .open(&path)?
                .write_all(&bytes)?;
            println!(
                "INT8_MODEL_LOGITS {}",
                json!({"case":name,"step":step,"mode":mode,
                "values":values.len(),"greedy":greedy,"path":path,
                "sha256":format!("{:x}",Sha256::digest(&bytes))})
            );
            if step < case.steps {
                let token = teacher.as_ref().map_or(greedy, |t| t[name][step]);
                chosen.push(token);
                backend.evaluate(&[token], (case.ids.len() + step) as u32)?;
            }
        }
        schedule.insert(name, chosen);
        let memory = backend.memory_usage().unwrap();
        assert_eq!(
            memory.numerical_reservation_bytes,
            initial_memory.numerical_reservation_bytes
        );
        assert_eq!(
            memory.workspace_allocated_bytes,
            initial_memory.workspace_allocated_bytes
        );
    }
    serde_json::to_writer(
        OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(output.join("teacher.json"))?,
        &schedule,
    )?;
    let longest = cases.values().max_by_key(|case| case.ids.len()).unwrap();
    let capacity_ids: Vec<_> = longest.ids.iter().copied().cycle().take(1024).collect();
    backend.reset()?;
    for (index, chunk) in capacity_ids.chunks(batch as usize).enumerate() {
        backend.evaluate(chunk, (index * batch as usize) as u32)?;
    }
    let saved = backend.logits()?.to_vec();
    assert!(backend.evaluate(&[capacity_ids[0]], 1024).is_err());
    assert_eq!(backend.logits()?, saved);
    let memory = backend.memory_usage().unwrap();
    assert_eq!(
        memory.numerical_reservation_bytes,
        initial_memory.numerical_reservation_bytes
    );
    assert_eq!(
        memory.workspace_allocated_bytes,
        initial_memory.workspace_allocated_bytes
    );
    println!(
        "INT8_MODEL_CAPACITY {}",
        json!({"mode":mode,"exact_context":1024,
        "one_over_rejected_without_poisoning":true,"memory":memory})
    );
    let mut executor = Executor::new(backend, config)?;
    for (name, case) in &cases {
        let sample =
            executor.generate(&case.ids, case.steps as u32, GenerationMode::FixedLength)?;
        println!(
            "INT8_MODEL_GENERATION {}",
            json!({"case":name,"mode":mode,"sample":sample,
            "prefill_chunk":batch,"scope":"exact-ID check only; one un-warmed sample is not speed acceptance"})
        );
    }
    Ok(())
}
