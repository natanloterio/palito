//! Bounded M8.3 experiment: unchanged matvec ABI, F32 operands, changed K tree.
use super::prefill::Harness;
use super::tests::packed_fixture;
use super::*;
use serde_json::json;

const GUARD: f32 = 12345.0;
const KERNELS: [&str; 2] = ["matvec", "matvec_packed_f32"];

#[allow(clippy::too_many_arguments)]
fn project(
    harness: &Harness,
    name: &str,
    tensor: &DeviceTensor,
    input: &CudaSlice<f32>,
    output: &mut CudaSlice<f32>,
    ids: &CudaSlice<i32>,
    moe: i32,
    batched: i32,
    extra_rows: u32,
) -> Result<()> {
    let selections = ids.len();
    let x = input.slice(8..input.len() - 8);
    let mut y = output.slice_mut(8..8 + selections * tensor.rows as usize);
    let cfg = LaunchConfig {
        grid_dim: (tensor.rows as u32 + extra_rows, selections as u32, 1),
        block_dim: (
            if name == "matvec_packed_k_pair" {
                128
            } else {
                256
            },
            1,
            1,
        ),
        shared_mem_bytes: 0,
    };
    let mut args = harness.stream.launch_builder(&harness.functions[name]);
    args.arg(&tensor.bytes)
        .arg(&tensor.kind)
        .arg(&x)
        .arg(&mut y)
        .arg(&tensor.width)
        .arg(&tensor.rows)
        .arg(ids)
        .arg(&moe)
        .arg(&batched);
    // Candidate overlaunches exercise the uniform row guard.
    unsafe { args.launch(cfg) }.map_err(failure)?;
    Ok(())
}

fn guarded_input(harness: &Harness, input: &[f32]) -> Result<CudaSlice<f32>> {
    let mut values = vec![f32::NAN; input.len() + 16];
    values[8..8 + input.len()].copy_from_slice(input);
    harness.stream.clone_htod(&values).map_err(failure)
}

fn guarded_output(harness: &Harness, count: usize) -> Result<CudaSlice<f32>> {
    harness
        .stream
        .clone_htod(&vec![GUARD; count + 16])
        .map_err(failure)
}

fn read_output(harness: &Harness, output: &CudaSlice<f32>) -> Result<Vec<f32>> {
    let values = harness.stream.clone_dtoh(output).map_err(failure)?;
    assert!(values[..8].iter().all(|&x| x == GUARD), "prefix guard");
    assert!(
        values[values.len() - 8..].iter().all(|&x| x == GUARD),
        "suffix guard"
    );
    Ok(values[8..values.len() - 8].to_vec())
}

pub(super) fn benchmark_down(
    h: &Harness,
    tensor: &DeviceTensor,
    input: &[f32],
    routing: &[i32],
    position: usize,
    capture_hash: &str,
) -> Result<()> {
    let input = guarded_input(h, input)?;
    let ids = h.stream.clone_htod(routing).map_err(failure)?;
    let count = tensor.rows as usize * routing.len();
    let mut outputs = [
        super::residual::matrix_output(h, count)?,
        super::residual::matrix_output(h, count)?,
        super::residual::matrix_output(h, count)?,
    ];
    let run = |arm: usize, output: &mut CudaSlice<f32>| -> Result<()> {
        if arm == 0 {
            return project(
                h,
                "matvec_packed_f32",
                tensor,
                &input,
                output,
                &ids,
                1,
                1,
                0,
            );
        }
        let (kernel, rows_per_block) = if arm == 1 {
            ("matvec_packed_down_two_row", 2)
        } else {
            ("matvec_packed_down_warp", 8)
        };
        let x = input.slice(8..input.len() - 8);
        let mut y = output.slice_mut(8..8 + count);
        launch!(
            h,
            kernel,
            (
                (tensor.rows as u32).div_ceil(rows_per_block),
                routing.len() as u32,
                1
            ),
            &tensor.bytes,
            &tensor.kind,
            &x,
            &mut y,
            &tensor.rows,
            &ids
        );
        Ok(())
    };
    for (arm, output) in outputs.iter_mut().enumerate() {
        run(arm, output)?;
    }
    let flag = h.stream.alloc_zeros::<i32>(1).map_err(failure)?;
    let base = super::residual::read_matrix_output(h, &flag, &outputs[0])?;
    for output in &outputs[1..] {
        let candidate = super::residual::read_matrix_output(h, &flag, output)?;
        assert!(
            base.iter()
                .zip(&candidate)
                .all(|(&a, &b)| a.to_bits() == b.to_bits())
        );
    }
    let mut samples = [Vec::new(), Vec::new(), Vec::new()];
    for repetition in 0..10 {
        for slot in 0..3 {
            let i = (repetition + slot) % 3;
            let start = h
                .stream
                .record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))
                .map_err(failure)?;
            run(i, &mut outputs[i])?;
            let end = h
                .stream
                .record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))
                .map_err(failure)?;
            end.synchronize().map_err(failure)?;
            if repetition >= 3 {
                samples[i].push(start.elapsed_ms(&end).map_err(failure)?);
            }
        }
    }
    println!(
        "PACKED_DOWN_SCREEN {}",
        json!({
            "capture_sha256":capture_hash, "position":position, "kind":tensor.kind, "ids":routing,
            "output_sha256":super::model_replay::float_hash(&base), "exact":true,
            "baseline_ms":samples[0], "staged_ms":samples[1], "warp_ms":samples[2],
            "repetitions":7, "warmups":3,
        })
    );
    Ok(())
}

pub(super) fn benchmark_gate_up(
    h: &Harness,
    gate: &DeviceTensor,
    up: &DeviceTensor,
    input: &[f32],
    routing: &[i32],
    position: usize,
    capture_hash: &str,
) -> Result<()> {
    assert_eq!((gate.width, up.width), (2048, 2048));
    assert_eq!(gate.rows, up.rows);
    assert!(matches!(gate.kind, 12 | 14) && matches!(up.kind, 12 | 14));
    let input = guarded_input(h, input)?;
    let ids = h.stream.clone_htod(routing).map_err(failure)?;
    let count = gate.rows as usize * routing.len();
    let mut scratch = guarded_output(h, count)?;
    let mut outputs = [guarded_output(h, count)?, guarded_output(h, count)?];
    let mut run = |arm: usize, output: &mut CudaSlice<f32>| -> Result<()> {
        if arm == 0 {
            project(h, "matvec_packed_f32", gate, &input, output, &ids, 1, 0, 0)?;
            project(
                h,
                "matvec_packed_f32",
                up,
                &input,
                &mut scratch,
                &ids,
                1,
                0,
                0,
            )?;
            let mut y = output.slice_mut(8..8 + count);
            let u = scratch.slice(8..8 + count);
            let count_i32 = count as i32;
            launch!(
                h,
                "swiglu",
                ((count as u32).div_ceil(256), 1, 1),
                &mut y,
                &u,
                &count_i32
            );
        } else {
            let x = input.slice(8..input.len() - 8);
            let mut y = output.slice_mut(8..8 + count);
            launch!(
                h,
                "matvec_packed_gate_up",
                (gate.rows as u32 + 1, routing.len() as u32, 1),
                &gate.bytes,
                &gate.kind,
                &up.bytes,
                &up.kind,
                &x,
                &mut y,
                &gate.rows,
                &ids
            );
        }
        Ok(())
    };
    let mut samples = [Vec::new(), Vec::new()];
    for repetition in 0..10 {
        for slot in 0..2 {
            let arm = (repetition + slot) % 2;
            let start = h
                .stream
                .record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))
                .map_err(failure)?;
            run(arm, &mut outputs[arm])?;
            let end = h
                .stream
                .record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))
                .map_err(failure)?;
            end.synchronize().map_err(failure)?;
            if repetition >= 3 {
                samples[arm].push(start.elapsed_ms(&end).map_err(failure)?);
            }
        }
    }
    read_output(h, &scratch)?;
    let baseline = read_output(h, &outputs[0])?;
    let candidate = read_output(h, &outputs[1])?;
    assert!(
        baseline
            .iter()
            .zip(&candidate)
            .all(|(&a, &b)| a.is_finite() && a.to_bits() == b.to_bits())
    );
    println!(
        "PACKED_GATE_UP_SCREEN {}",
        json!({"capture_sha256":capture_hash, "position":position, "kinds":[gate.kind, up.kind],
            "activation_sha256":super::model_replay::float_hash(&baseline), "exact":true,
            "baseline_ms":samples[0], "fused_ms":samples[1], "repetitions":7, "warmups":3})
    );
    Ok(())
}

pub(super) fn benchmark_k_pair(
    h: &Harness,
    tensor: &DeviceTensor,
    input: &[f32],
    tensor_name: &str,
    position: usize,
    capture_hash: &str,
) -> Result<()> {
    let input = guarded_input(h, input)?;
    let ids = h.stream.clone_htod(&[0i32]).map_err(failure)?;
    let mut outputs = [
        guarded_output(h, tensor.rows as usize)?,
        guarded_output(h, tensor.rows as usize)?,
    ];
    let kernels = ["matvec_packed_f32", "matvec_packed_k_pair"];
    let mut samples = [Vec::new(), Vec::new()];
    for repetition in 0..10 {
        for slot in 0..2 {
            let arm = (repetition + slot) % 2;
            let start = h
                .stream
                .record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))
                .map_err(failure)?;
            project(
                h,
                kernels[arm],
                tensor,
                &input,
                &mut outputs[arm],
                &ids,
                0,
                0,
                arm as u32,
            )?;
            let end = h
                .stream
                .record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))
                .map_err(failure)?;
            end.synchronize().map_err(failure)?;
            if repetition >= 3 {
                samples[arm].push(start.elapsed_ms(&end).map_err(failure)?);
            }
        }
    }
    let baseline = read_output(h, &outputs[0])?;
    let candidate = read_output(h, &outputs[1])?;
    assert!(
        baseline
            .iter()
            .zip(&candidate)
            .all(|(&a, &b)| a.is_finite() && a.to_bits() == b.to_bits())
    );
    println!(
        "PACKED_K_PAIR_SCREEN {}",
        json!({"capture_sha256":capture_hash, "position":position, "tensor":tensor_name,
            "kind":tensor.kind,"output_sha256":super::model_replay::float_hash(&baseline), "exact":true,
            "baseline_ms":samples[0], "pair_ms":samples[1], "repetitions":7, "warmups":3})
    );
    Ok(())
}

fn decoded_weights(tensor: &Tensor) -> Vec<f32> {
    match &tensor.storage {
        Storage::F32(values) => values.clone(),
        Storage::Packed { kind, bytes } => {
            let (block, size, _) = crate::gguf::layout(*kind).unwrap();
            let mut values = vec![0.0; tensor.dims.iter().product()];
            for (packed, decoded) in bytes.chunks_exact(size).zip(values.chunks_exact_mut(block)) {
                crate::gguf::decode(*kind, packed, decoded);
            }
            values
        }
    }
}

fn assert_close(actual: &[f32], expected: &[f32], context: &str) {
    assert_eq!(actual.len(), expected.len());
    for (i, (&a, &e)) in actual.iter().zip(expected).enumerate() {
        assert!(
            a.is_finite() && e.is_finite() && (a - e).abs() <= 0.0001 + e.abs() * 1e-5,
            "{context} output={i}: {a} != {e}"
        );
    }
}

#[test]
#[ignore = "requires CUDA 13; run serially in release mode"]
fn packed_decode_formats_experts_batched_tails_and_guards() -> Result<()> {
    let harness = Harness::new()?;
    let selected = [2i32, 0, 1, 2];
    let ids = harness.stream.clone_htod(&selected).map_err(failure)?;
    for kind in [0, 1, 6, 8, 12, 14] {
        let block = crate::gguf::layout(kind)?.0;
        // The last case has rows crossing storage blocks, but complete storage.
        // 9 blocks also exercises a second iteration of the eight-warp K loop.
        for (width, rows) in [
            (block, 1),
            (3 * block, 9),
            (9 * block, 3),
            (block + 1, block),
        ] {
            let tensor = packed_fixture(kind, width, rows, 3);
            let weights = decoded_weights(&tensor);
            let t = DeviceTensor::upload(tensor, &harness.stream)?;
            let input = super::prefill::inputs(width, selected.len());
            let x = guarded_input(&harness, &input)?;
            for moe in [0i32, 1] {
                for batched in [0i32, 1] {
                    let mut expected = Vec::with_capacity(rows * selected.len());
                    for (selection, &id) in selected.iter().enumerate() {
                        let expert = if moe != 0 { id as usize } else { 0 };
                        let token = if batched != 0 { selection } else { 0 };
                        let input = &input[token * width..(token + 1) * width];
                        for row in 0..rows {
                            let start = (expert * rows + row) * width;
                            expected.push(
                                weights[start..start + width]
                                    .iter()
                                    .zip(input)
                                    .map(|(&w, &v)| f64::from(w) * f64::from(v))
                                    .sum::<f64>() as f32,
                            );
                        }
                    }
                    let context = format!(
                        "kind={kind} width={width} rows={rows} moe={moe} batched={batched}"
                    );
                    let mut reference = Vec::new();
                    for name in KERNELS {
                        let mut output = guarded_output(&harness, expected.len())?;
                        project(
                            &harness,
                            name,
                            &t,
                            &x,
                            &mut output,
                            &ids,
                            moe,
                            batched,
                            if name == "matvec_packed_f32" { 3 } else { 0 },
                        )?;
                        let actual = read_output(&harness, &output)?;
                        assert_close(&actual, &expected, &format!("{name} {context}"));
                        if name == "matvec" {
                            reference = actual;
                        } else if !matches!(kind, 12 | 14) || width % 256 != 0 {
                            assert_eq!(actual, reference, "fallback changed {context}");
                        } else {
                            assert_close(&actual, &reference, &context);
                        }
                    }
                }
            }
        }
    }
    Ok(())
}

#[test]
#[ignore = "requires CUDA 13; exhaustive one-hot packed reconstruction, run serially"]
fn packed_decode_one_hot_reconstruction_matches_cpu() -> Result<()> {
    let harness = Harness::new()?;
    let selected = [2i32, 0, 1];
    let ids = harness.stream.clone_htod(&selected).map_err(failure)?;
    let width = 512;
    let rows = 3;
    for kind in [12, 14] {
        let mut tensor = packed_fixture(kind, width, rows, 3);
        if let Storage::Packed { bytes, .. } = &mut tensor.storage {
            let size = crate::gguf::layout(kind)?.1;
            // Non-dyadic, negative and subnormal scales expose reconstruction
            // reassociation that the default fixture's power-of-two scales hide.
            let halves = [0x0401u16, 0x2937, 0xb155, 0x3555, 0x0001, 0x8001];
            for (i, block) in bytes.chunks_exact_mut(size).enumerate() {
                let offset = if kind == 12 { 0 } else { 208 };
                block[offset..offset + 2].copy_from_slice(&halves[i % halves.len()].to_le_bytes());
                if kind == 12 {
                    block[2..4].copy_from_slice(&halves[(i + 2) % halves.len()].to_le_bytes());
                }
            }
        }
        let expected = decoded_weights(&tensor);
        let t = DeviceTensor::upload(tensor, &harness.stream)?;
        let mut input = vec![0.0; width];
        let mut output = guarded_output(&harness, rows * selected.len())?;
        for col in 0..width {
            input[col] = 1.0;
            let x = guarded_input(&harness, &input)?;
            project(
                &harness,
                "matvec_packed_f32",
                &t,
                &x,
                &mut output,
                &ids,
                1,
                0,
                0,
            )?;
            let actual = read_output(&harness, &output)?;
            for (selection, &expert) in selected.iter().enumerate() {
                for row in 0..rows {
                    // Addition of zero may erase a reconstructed negative zero.
                    assert_eq!(
                        actual[selection * rows + row],
                        expected[(expert as usize * rows + row) * width + col],
                        "kind={kind} expert={expert} row={row} col={col}"
                    );
                }
            }
            input[col] = 0.0;
        }
    }
    Ok(())
}

#[test]
#[ignore = "requires CUDA 13; synthetic event microbenchmark, run serially with --nocapture"]
fn benchmark_packed_decode_f32_per_shape() -> Result<()> {
    let harness = Harness::new()?;
    for kind in [12, 14] {
        for (class, width, rows, selected) in [
            ("dense_q", 2048, 4096, vec![0i32]),
            ("dense_output", 2048, 2048, vec![0]),
            ("dense_kv", 2048, 512, vec![0]),
            ("expert_gate_up", 2048, 768, vec![7, 0, 3, 1, 7, 5, 2, 6]),
            ("expert_down", 768, 2048, vec![7, 0, 3, 1, 7, 5, 2, 6]),
        ] {
            let moe = i32::from(selected.len() > 1);
            let experts = if moe != 0 { 8 } else { 1 };
            let t =
                DeviceTensor::upload(packed_fixture(kind, width, rows, experts), &harness.stream)?;
            let ids = harness.stream.clone_htod(&selected).map_err(failure)?;
            let x = guarded_input(&harness, &super::prefill::inputs(width, selected.len()))?;
            let mut outputs = [
                guarded_output(&harness, rows * selected.len())?,
                guarded_output(&harness, rows * selected.len())?,
            ];
            for batched in 0..=moe {
                for _ in 0..4 {
                    for (slot, name) in KERNELS.iter().enumerate() {
                        project(
                            &harness,
                            name,
                            &t,
                            &x,
                            &mut outputs[slot],
                            &ids,
                            moe,
                            batched,
                            0,
                        )?;
                    }
                }
                let reference = read_output(&harness, &outputs[0])?;
                let actual = read_output(&harness, &outputs[1])?;
                assert_close(&actual, &reference, class);
                for repetition in 0..7 {
                    // Alternate order; compile, allocations, copies, validation
                    // and warmup stay outside each event interval.
                    for turn in 0..2 {
                        let slot = (turn + repetition) % 2;
                        let name = KERNELS[slot];
                        harness.stream.synchronize().map_err(failure)?;
                        let start = harness
                            .stream
                            .record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))
                            .map_err(failure)?;
                        for _ in 0..20 {
                            project(
                                &harness,
                                name,
                                &t,
                                &x,
                                &mut outputs[slot],
                                &ids,
                                moe,
                                batched,
                                0,
                            )?;
                        }
                        let end = harness
                            .stream
                            .record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))
                            .map_err(failure)?;
                        end.synchronize().map_err(failure)?;
                        println!(
                            "PACKED_DECODE_BENCH {}",
                            json!({
                                "class": class, "kind": kind, "width": width, "rows": rows,
                                "selections": selected.len(), "ids": selected,
                                "moe": moe, "batched": batched, "kernel": name,
                                "repetition": repetition, "launches": 20,
                                "cuda_event_ms": start.elapsed_ms(&end).map_err(failure)? / 20.0,
                                "fmad": false, "operands": "f32",
                                "weight_bytes": width * rows * experts / 256
                                    * crate::gguf::layout(kind)?.1,
                                "grid": [rows, selected.len(), 1], "threads": 256,
                                "synthetic": true, "captured_graph": false,
                            })
                        );
                    }
                }
                for output in &outputs {
                    let _ = read_output(&harness, output)?;
                }
            }
        }
    }
    Ok(())
}

#[derive(serde::Deserialize)]
struct DecodeTrialCase {
    ids: Vec<i32>,
    generated: u32,
}

struct DecodeTrialReference {
    logits: Vec<Vec<f32>>,
    generated: Vec<i32>,
}

fn recapture_decode_graph(backend: &mut CudaBackend) -> Result<()> {
    // open() already captured the production function. Replacing the function
    // map alone cannot change that graph's recorded kernel handles.
    backend.capture_decode_graph()?;
    backend.reset()
}

fn compare_trial_logits(
    actual: &[f32],
    expected: &[f32],
    phase: &str,
    case: &str,
    step: usize,
    position: usize,
) -> Result<()> {
    assert_eq!(actual.len(), expected.len(), "{phase}/{case}/{step}");
    let actual_argmax = greedy(actual)?;
    let expected_argmax = greedy(expected)?;
    let (mut max_abs, mut dot, mut actual_norm, mut expected_norm) = (0.0f64, 0.0, 0.0, 0.0);
    for (&a, &e) in actual.iter().zip(expected) {
        let (a, e) = (f64::from(a), f64::from(e));
        max_abs = max_abs.max((a - e).abs());
        dot += a * e;
        actual_norm += a * a;
        expected_norm += e * e;
    }
    let cosine = if actual_norm == 0.0 || expected_norm == 0.0 {
        if actual == expected { 1.0 } else { 0.0 }
    } else {
        dot / (actual_norm.sqrt() * expected_norm.sqrt())
    };
    let passed = max_abs <= 0.01 && cosine >= 0.99999 && actual_argmax == expected_argmax;
    println!(
        "PACKED_DECODE_TRIAL {}",
        json!({
            "kind": "logit_comparison", "phase": phase, "case": case,
            "step": step, "position": position, "reference": "A1",
            "elements": actual.len(), "max_abs": max_abs, "cosine": cosine,
            "actual_argmax": actual_argmax, "reference_argmax": expected_argmax,
            "passed": passed,
        })
    );
    assert!(passed, "{phase}/{case}/step={step}: full-logit gate failed");
    Ok(())
}

fn trial_full_logits(
    backend: &mut CudaBackend,
    case: &DecodeTrialCase,
    reference: Option<&DecodeTrialReference>,
    phase: &str,
    name: &str,
) -> Result<DecodeTrialReference> {
    backend.reset()?;
    for (index, chunk) in case.ids.chunks(512).enumerate() {
        backend.evaluate(chunk, (index * 512) as u32)?;
    }
    let mut result = DecodeTrialReference {
        logits: Vec::new(),
        generated: Vec::new(),
    };
    // Materialize every vocabulary logit at the prompt boundary and at each
    // generated decode position. This untimed path also uses the captured graph.
    for step in 0..case.generated as usize {
        let logits = backend.logits()?;
        let token = greedy(logits)?;
        if let Some(reference) = reference {
            compare_trial_logits(
                logits,
                &reference.logits[step],
                phase,
                name,
                step,
                case.ids.len() + step - 1,
            )?;
        } else {
            result.logits.push(logits.to_vec());
        }
        result.generated.push(token);
        if step + 1 < case.generated as usize {
            backend.evaluate(&[token], (case.ids.len() + step) as u32)?;
        }
    }
    if let Some(reference) = reference {
        assert_eq!(
            result.generated, reference.generated,
            "{phase}/{name}: full-logit generation IDs changed"
        );
    }
    println!(
        "PACKED_DECODE_TRIAL {}",
        json!({
            "kind": "quality", "phase": phase, "case": name, "reference": "A1",
            "full_logit_positions": case.generated, "generated_ids": result.generated,
            "passed": true, "reference_capture": reference.is_none(),
        })
    );
    Ok(result)
}

#[test]
#[ignore = "requires CUDA, M8_DECODE_MODEL and M8_DECODE_CASES local paths; ABBA, run serially with --nocapture"]
fn packed_decode_real_model_abba_trial() -> Result<()> {
    use crate::{Executor, GenerationMode};
    use sha2::{Digest, Sha256};
    use std::{collections::BTreeMap, time::Instant};

    let model = std::fs::canonicalize(std::env::var("M8_DECODE_MODEL").map_err(failure)?)?;
    let cases_path = std::fs::canonicalize(std::env::var("M8_DECODE_CASES").map_err(failure)?)?;
    if !model.is_file() || !cases_path.is_file() {
        return Err(invalid(
            "M8 decode model and cases must be explicit local files",
        ));
    }
    let cases_bytes = std::fs::read(&cases_path)?;
    let cases: BTreeMap<String, DecodeTrialCase> = serde_json::from_slice(&cases_bytes)?;
    let config = Config {
        context_tokens: 1024,
        batch_tokens: 512,
        prefill_chunk: 512,
        threads: 4,
    };
    if cases.is_empty()
        || !cases.values().any(|case| case.generated > 1)
        || cases.values().any(|case| {
            case.ids.is_empty()
                || case.generated == 0
                || case.ids.iter().any(|&id| id < 0)
                || case
                    .ids
                    .len()
                    .checked_add(case.generated.saturating_sub(1) as usize)
                    .is_none_or(|count| count > config.context_tokens as usize)
        })
    {
        return Err(invalid(
            "M8 decode cases need nonempty IDs, generated > 0, prompt + generated - 1 <= 1024, and at least one decode case",
        ));
    }
    let metadata = std::fs::metadata(&model)?;
    let modified = metadata.modified()?;
    let prototype_source = concat!("#define CUDA_M8_PROTOTYPES 1\n", include_str!("kernels.cu"));
    let phases = [("A1", false), ("B1", true), ("B2", true), ("A2", false)];
    println!(
        "PACKED_DECODE_TRIAL {}",
        json!({
            "kind": "provenance", "version": 1, "order": ["A1", "B1", "B2", "A2"],
            "model": model, "model_bytes": metadata.len(),
            "model_modified_unix_ns": modified.duration_since(std::time::UNIX_EPOCH).map_err(failure)?.as_nanos(),
            "model_identity": "canonical_path_size_mtime_not_content_hash",
            "cases": cases_path, "cases_sha256": format!("{:x}", Sha256::digest(&cases_bytes)),
            "prototype_source_sha256": format!("{:x}", Sha256::digest(prototype_source)),
            "baseline_matvec_override": "matvec",
            "candidate_matvec_override": "matvec_packed_f32",
            "both_variants_explicitly_installed_and_recaptured": true,
            "baseline_source_sha256": format!("{:x}", Sha256::digest(concat!(
                include_str!("kernels.cu"), "\n", include_str!("prefill.cu"), "\n",
                include_str!("attention.cu"), "\n", include_str!("decode.cu")
            ))),
            "cargo_lock_sha256": format!("{:x}", Sha256::digest(include_str!("../../Cargo.lock"))),
            "config": config, "matmul": "f32-tiled", "decode_graph": true,
            "warmups_per_case_per_phase": 3, "samples_per_case_per_phase": 7,
            "generation_mode": "fixed_length", "fmad": false, "operands": "f32",
            "max_abs_limit": 0.01, "cosine_minimum": 0.99999,
            "exact_argmax_and_ids": true, "prefix_cache": false, "device": 0,
            "timing_boundary": "Executor::generate; reset included in elapsed_ns, model load/JIT/graph capture/full-logit checks excluded",
            "decode_boundary": "Sample.decode_ns includes token-only graph replay, sampling and host control; excludes prompt and first token",
            "exclusive_gpu_assumed": false,
        })
    );
    let mut references = BTreeMap::<String, DecodeTrialReference>::new();
    let mut timings: BTreeMap<String, [Vec<u64>; 4]> = cases
        .keys()
        .map(|name| (name.clone(), std::array::from_fn(|_| Vec::new())))
        .collect();
    for (phase_index, (phase, candidate)) in phases.into_iter().enumerate() {
        let current = std::fs::metadata(&model)?;
        assert_eq!(current.len(), metadata.len(), "model changed during ABBA");
        assert_eq!(current.modified()?, modified, "model changed during ABBA");
        let load_start = Instant::now();
        let mut backend = CudaBackend::open(
            &model,
            &config,
            CudaOptions {
                matmul: CudaMatmul::F32Tiled,
                decode_graph: true,
                ..CudaOptions::default()
            },
        )?;
        let load_ns = load_start.elapsed().as_nanos();
        let context = backend.stream.context().clone();
        let major = context
            .attribute(sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)
            .map_err(failure)?;
        let minor = context
            .attribute(sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)
            .map_err(failure)?;
        let install_start = Instant::now();
        let ptx = compile_ptx_with_opts(
            prototype_source,
            CompileOptions {
                fmad: Some(false),
                options: vec![format!("--gpu-architecture=compute_{major}{minor}")],
                ..Default::default()
            },
        )
        .map_err(failure)?;
        let trial_module = context.load_module(ptx).map_err(failure)?;
        let kernel = if candidate {
            "matvec_packed_f32"
        } else {
            "matvec"
        };
        // Override both arms: F32Tiled may itself select packed decode in the
        // future, but the A phases must still execute the legacy implementation.
        backend.functions.insert(
            "matvec",
            trial_module.load_function(kernel).map_err(failure)?,
        );
        recapture_decode_graph(&mut backend)?;
        let install_ns = install_start.elapsed().as_nanos();
        println!(
            "PACKED_DECODE_TRIAL {}",
            json!({
                "kind": "phase", "phase": phase, "phase_index": phase_index,
                "candidate": candidate, "kernel": kernel,
                "load_ns": load_ns, "kernel_compile_install_recapture_ns": install_ns,
                "graph_recaptured_after_replacement": true,
                "compute_capability": [major, minor], "device_name": context.name().map_err(failure)?,
                "backend": backend.info(), "memory": backend.memory_usage(),
            })
        );
        for (name, case) in &cases {
            if case
                .ids
                .iter()
                .any(|&id| id as u32 >= backend.info().vocab_tokens)
            {
                return Err(invalid(format!(
                    "{name}: prompt token outside model vocabulary"
                )));
            }
            let reference =
                trial_full_logits(&mut backend, case, references.get(name), phase, name)?;
            if phase_index == 0 {
                references.insert(name.clone(), reference);
            }
        }
        let mut executor = Executor::new(backend, config.clone())?;
        for (name, case) in &cases {
            for iteration in 0..10 {
                let wall = Instant::now();
                let sample =
                    executor.generate(&case.ids, case.generated, GenerationMode::FixedLength)?;
                let elapsed_ns = wall.elapsed().as_nanos();
                let warmup = iteration < 3;
                let passed = sample.generated_tokens == references[name].generated;
                println!(
                    "PACKED_DECODE_TRIAL {}",
                    json!({
                        "kind": if warmup { "warmup" } else { "sample" },
                        "phase": phase, "phase_index": phase_index, "candidate": candidate,
                        "case": name, "prompt_tokens": case.ids.len(), "generated": case.generated,
                        "iteration": if warmup { iteration } else { iteration - 3 },
                        "elapsed_ns": elapsed_ns, "sample": sample,
                        "decode_tokens_per_second": (sample.decode_ns != 0).then(||
                            sample.decode_steps as f64 * 1e9 / sample.decode_ns as f64),
                        "exact_ids_vs_A1": passed,
                    })
                );
                assert!(
                    passed,
                    "{phase}/{name}/iteration={iteration}: generated IDs changed"
                );
                assert_eq!(sample.cached_prompt_tokens, 0, "unexpected prefix reuse");
                if !warmup {
                    timings.get_mut(name).unwrap()[phase_index].push(sample.decode_ns);
                }
            }
        }
        // Keep the separately loaded module alive until every graph and backend
        // function has been destroyed, then release the stream before the next load.
        let stream = executor.backend.stream.clone();
        stream.synchronize().map_err(failure)?;
        drop(executor);
        drop(trial_module);
        stream.synchronize().map_err(failure)?;
        drop(stream);
        drop(context);
        println!(
            "PACKED_DECODE_TRIAL {}",
            json!({"kind": "phase_dropped", "phase": phase, "synchronized": true})
        );
    }
    for (name, mut samples) in timings {
        let medians = samples.each_mut().map(|values| {
            assert_eq!(values.len(), 7);
            values.sort_unstable();
            values[3]
        });
        let a = medians[0] as f64 + medians[3] as f64;
        let b = medians[1] as f64 + medians[2] as f64;
        println!(
            "PACKED_DECODE_TRIAL {}",
            json!({
                "kind": "summary", "case": name, "order": ["A1", "B1", "B2", "A2"],
                "median_decode_ns": medians,
                "abba_decode_speedup": (b > 0.0).then(|| a / b),
                "baseline_A2_over_A1": (medians[0] > 0).then(|| medians[3] as f64 / medians[0] as f64),
                "all_full_logits_argmax_and_generated_ids_passed": true,
            })
        );
    }
    Ok(())
}
