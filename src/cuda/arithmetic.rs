//! Bounded SM80+ arithmetic controls, not production admission or dispatch.
use super::{prefill::Harness, tests::packed_fixture, *};

const CONTROL: &str = "prefill_matmul_tf32x4_control";
const THREE_PRODUCT: &str = "prefill_matmul_tf32x3_tiled";
const GUARD: f32 = 1337.0;

fn decoded_weights(tensor: &Tensor) -> Vec<f32> {
    match &tensor.storage {
        Storage::F32(values) => values.clone(),
        Storage::Packed { kind, bytes } => {
            let (block, size, _) = crate::gguf::layout(*kind).unwrap();
            let mut values = vec![0.0; tensor.dims[0] * tensor.dims[1]];
            for (packed, decoded) in bytes.chunks_exact(size).zip(values.chunks_exact_mut(block)) {
                crate::gguf::decode(*kind, packed, decoded);
            }
            values
        }
    }
}

fn cpu_reference(tensor: &Tensor, input: &[f32]) -> Vec<f32> {
    let width = tensor.dims[0];
    let weights = decoded_weights(tensor);
    input
        .chunks_exact(width)
        .flat_map(|x| {
            weights.chunks_exact(width).map(move |row| {
                row.iter()
                    .zip(x)
                    .map(|(&w, &v)| f64::from(w) * f64::from(v))
                    .sum::<f64>() as f32
            })
        })
        .collect()
}

fn project(
    harness: &Harness,
    kernel: &str,
    tensor: &DeviceTensor,
    input: &CudaSlice<f32>,
    tokens: usize,
) -> Result<Vec<f32>> {
    let count = tokens * tensor.rows as usize;
    let mut output = harness
        .stream
        .clone_htod(&vec![GUARD; count + 16])
        .map_err(failure)?;
    {
        let x = input.slice(8..8 + tokens * tensor.width as usize);
        let mut y = output.slice_mut(8..8 + count);
        let n = tokens as i32;
        let f32_reference = kernel == "prefill_matmul";
        let cfg = LaunchConfig {
            grid_dim: (
                (tensor.rows as u32).div_ceil(if f32_reference { 1 } else { 128 }),
                (tokens as u32).div_ceil(if f32_reference { 4 } else { 8 }),
                1,
            ),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        // All launches use the existing packed ABI and bounded token-major views.
        unsafe {
            harness
                .stream
                .launch_builder(&harness.functions[kernel])
                .arg(&tensor.bytes)
                .arg(&tensor.kind)
                .arg(&x)
                .arg(&mut y)
                .arg(&tensor.width)
                .arg(&tensor.rows)
                .arg(&n)
                .launch(cfg)
        }
        .map_err(failure)?;
    }
    let actual = harness.stream.clone_dtoh(&output).map_err(failure)?;
    assert!(
        actual[..8]
            .iter()
            .chain(&actual[8 + count..])
            .all(|&v| v == GUARD),
        "{kernel}: output guard modified"
    );
    assert!(actual[8..8 + count].iter().all(|v| v.is_finite()));
    Ok(actual[8..8 + count].to_vec())
}

fn report(kernel: &str, reference: &str, shape: [usize; 3], kind: i32, a: &[f32], b: &[f32]) {
    assert_eq!(a.len(), b.len());
    let mut max_abs = 0.0f64;
    let mut violations = 0usize;
    let mut first_violation = None;
    let (mut dot, mut aa, mut bb) = (0.0f64, 0.0f64, 0.0f64);
    for (index, (&a, &b)) in a.iter().zip(b).enumerate() {
        assert!(a.is_finite() && b.is_finite());
        let error = (a - b).abs();
        let limit = 0.005 + 0.003 * b.abs();
        if error > limit {
            violations += 1;
            first_violation.get_or_insert_with(|| {
                serde_json::json!({
                    "output": index, "actual": a, "expected": b,
                    "abs_error": error, "limit": limit,
                })
            });
        }
        let (a, b) = (f64::from(a), f64::from(b));
        max_abs = max_abs.max((a - b).abs());
        dot += a * b;
        aa += a * a;
        bb += b * b;
    }
    let cosine = if aa == 0.0 || bb == 0.0 {
        if aa == bb { 1.0 } else { 0.0 }
    } else {
        dot / (aa.sqrt() * bb.sqrt())
    };
    println!(
        "ARITHMETIC_CONTROL {}",
        serde_json::json!({
            "kernel": kernel, "reference": reference, "kind": kind,
            "width": shape[0], "rows": shape[1], "tokens": shape[2],
            "max_abs": max_abs, "cosine": cosine,
            "tolerance_abs": 0.005, "tolerance_rel": 0.003,
            "tolerance_violations": violations, "first_violation": first_violation,
            "scope": "primitive_diagnostic_not_model_admission",
        })
    );
}

fn check_case(
    harness: &Harness,
    tensor: Tensor,
    input: Vec<f32>,
    exact_dyadic: bool,
    diagnostic_baseline: bool,
) -> Result<()> {
    let (width, rows) = (tensor.dims[0], tensor.dims[1]);
    assert_eq!(input.len() % width, 0);
    let tokens = input.len() / width;
    let expected = cpu_reference(&tensor, &input);
    let tensor = DeviceTensor::upload(tensor, &harness.stream)?;
    let mut guarded = vec![GUARD; input.len() + 16];
    guarded[8..8 + input.len()].copy_from_slice(&input);
    let x = harness.stream.clone_htod(&guarded).map_err(failure)?;
    let f32_reference = project(harness, "prefill_matmul", &tensor, &x, tokens)?;
    let three = project(harness, THREE_PRODUCT, &tensor, &x, tokens)?;
    let four = project(harness, CONTROL, &tensor, &x, tokens)?;
    report(
        CONTROL,
        THREE_PRODUCT,
        [width, rows, tokens],
        tensor.kind,
        &four,
        &three,
    );
    for (name, values) in [
        ("prefill_matmul", &f32_reference),
        (THREE_PRODUCT, &three),
        (CONTROL, &four),
    ] {
        report(
            name,
            "f64_dot_f32_operands",
            [width, rows, tokens],
            tensor.kind,
            values,
            &expected,
        );
    }
    for (name, values) in [(THREE_PRODUCT, &three), (CONTROL, &four)] {
        report(
            name,
            "gpu_f32",
            [width, rows, tokens],
            tensor.kind,
            values,
            &f32_reference,
        );
    }
    // Report every comparison before enforcing candidate precision: the rejected
    // three-product baseline must not hide a four-product failure on large cases.
    for (name, values) in [
        ("prefill_matmul", &f32_reference),
        (THREE_PRODUCT, &three),
        (CONTROL, &four),
    ] {
        if diagnostic_baseline && name == THREE_PRODUCT {
            continue;
        }
        for (reference, expected) in [
            ("f64_dot_f32_operands", &expected),
            ("gpu_f32", &f32_reference),
        ] {
            for (i, (&a, &e)) in values.iter().zip(expected).enumerate() {
                assert!(
                    (a - e).abs() <= 0.005 + 0.003 * e.abs(),
                    "{name} vs {reference} kind={} width={width} rows={rows} tokens={tokens} output={i}: {a} != {e}; primitive control remains unpromoted",
                    tensor.kind,
                );
            }
        }
    }
    // This crafted dyadic dot is exactly representable, unlike arbitrary inputs.
    // Its high*high terms cancel and both low*low terms in every pair survive.
    if exact_dyadic {
        assert_eq!(four, expected, "dyadic residual*residual contribution lost");
    }
    assert_eq!(
        harness.stream.clone_dtoh(&x).map_err(failure)?,
        guarded,
        "input or input guards modified"
    );
    Ok(())
}

fn fixture(kind: u32, width: usize, rows: usize) -> Tensor {
    let mut tensor = packed_fixture(kind, width, rows, 1);
    match &mut tensor.storage {
        Storage::F32(values) => {
            for (i, value) in values.iter_mut().enumerate() {
                *value = ((i * 17 % 251) as f32 - 125.0) / 137.0;
            }
        }
        Storage::Packed { bytes, .. } if kind == 1 => {
            for (i, value) in bytes.chunks_exact_mut(2).enumerate() {
                value.copy_from_slice(
                    &half::f16::from_f32(((i * 17 % 251) as f32 - 125.0) / 137.0)
                        .to_bits()
                        .to_le_bytes(),
                );
            }
        }
        Storage::Packed { bytes, .. } => {
            let size = crate::gguf::layout(kind).unwrap().1;
            for block in bytes.chunks_exact_mut(size) {
                let offset = if kind == 14 { 208 } else { 0 };
                block[offset..offset + 2].copy_from_slice(&0x2345u16.to_le_bytes());
                if kind == 12 {
                    block[2..4].copy_from_slice(&0x1f13u16.to_le_bytes());
                }
            }
        }
    }
    tensor
}

fn inputs(width: usize, tokens: usize) -> Vec<f32> {
    (0..width * tokens)
        .map(|i| ((i * 29 % 257) as f32 - 128.0) / 131.0)
        .collect()
}

#[test]
#[ignore = "SM80+ bounded arithmetic diagnostic; run serially with --nocapture"]
fn tf32x4_control_cancellation_residual_product_and_tails() -> Result<()> {
    let harness = Harness::new()?;
    let (width, rows, tokens) = (257, 129, 9);
    let d = 1.0f32 / 4096.0;
    let mut tensor = packed_fixture(0, width, rows, 1);
    if let Storage::F32(weights) = &mut tensor.storage {
        for (i, value) in weights.iter_mut().enumerate() {
            let k = i % width;
            let sign = if (k + i / width) % 2 == 0 { 1.0 } else { -1.0 };
            *value = if k == width - 1 {
                0.0
            } else {
                sign * (1.0 + d)
            };
        }
    }
    let input = (0..width * tokens)
        .map(|i| {
            let sign = if i / width % 2 == 0 { 1.0 } else { -1.0 };
            sign * if i % width % 2 == 0 { 1.0 + d } else { 1.0 - d }
        })
        .collect();
    check_case(&harness, tensor, input, true, false)
}

#[test]
#[ignore = "SM80+ packed arithmetic diagnostic; run serially with --nocapture"]
fn tf32x4_control_packed_kinds_tails_and_decoder_sanity() -> Result<()> {
    let harness = Harness::new()?;
    for kind in [0, 1, 6, 8, 12, 14] {
        let block = crate::gguf::layout(kind)?.0;
        for width in [block, if kind <= 1 { 257 } else { block * 3 }] {
            let tensor = fixture(kind, width, 129);
            // Same K=32 staging/decoder used by both arithmetic controls.
            let device = DeviceTensor::upload(fixture(kind, width, 129), &harness.stream)?;
            let mut status = harness.stream.clone_htod(&[0i32]).map_err(failure)?;
            let k_tile = 32i32;
            unsafe {
                harness
                    .stream
                    .launch_builder(&harness.functions["projection_decode_check"])
                    .arg(&device.bytes)
                    .arg(&device.kind)
                    .arg(&mut status)
                    .arg(&device.width)
                    .arg(&device.rows)
                    .arg(&k_tile)
                    .launch(LaunchConfig {
                        grid_dim: (129u32.div_ceil(8), (width as u32).div_ceil(32), 1),
                        block_dim: (256, 1, 1),
                        shared_mem_bytes: 0,
                    })
            }
            .map_err(failure)?;
            assert_eq!(harness.stream.clone_dtoh(&status).map_err(failure)?, [0]);
            check_case(&harness, tensor, inputs(width, 9), false, false)?;
        }
    }
    Ok(())
}

#[test]
#[ignore = "SM80+ realistic Q4_K arithmetic diagnostic; run serially with --nocapture"]
fn tf32x4_control_realistic_q4k_projection() -> Result<()> {
    check_realistic_projection(12)
}

#[test]
#[ignore = "SM80+ realistic Q6_K arithmetic diagnostic; run serially with --nocapture"]
fn tf32x4_control_realistic_q6k_projection() -> Result<()> {
    check_realistic_projection(14)
}

fn check_realistic_projection(kind: u32) -> Result<()> {
    let harness = Harness::new()?;
    let (width, rows, tokens) = (2048, 768, 64);
    check_case(
        &harness,
        fixture(kind, width, rows),
        inputs(width, tokens),
        false,
        true,
    )
}
