//! Test-only precision iteration after the two-component full-model rejection.
use super::residual::MatrixCandidate;
use super::{prefill::Harness, *};
use serde_json::{Value, json};
use sha2::Digest;
use std::collections::BTreeMap;
use std::io::{BufRead, BufReader};

#[path = "residual_block.rs"]
mod block;

pub(super) struct Quantized {
    pub(super) values: Vec<i8>,
    pub(super) scales: Vec<f32>,
    pub(super) sums: Vec<i32>,
}

fn quantize(input: &[f32]) -> Result<Quantized> {
    quantize_groups(input, 32)
}

pub(super) fn quantize_groups(input: &[f32], group_size: usize) -> Result<Quantized> {
    if !matches!(group_size, 32 | 256) {
        return Err(invalid("precise INT8 supports only 32/256-value groups"));
    }
    if input.is_empty()
        || input.len() % group_size != 0
        || input.len() > i32::MAX as usize / 3
        || input.iter().any(|x| !x.is_finite())
    {
        return Err(invalid(
            "precise INT8 requires finite bounded complete groups",
        ));
    }
    let mut result = Quantized {
        values: vec![0; input.len() * 3],
        scales: Vec::new(),
        sums: vec![0; input.len() / 32 * 3],
    };
    for (block, source) in input.chunks_exact(group_size).enumerate() {
        let mut remaining = source.to_vec();
        let mut reconstructed = vec![0.0f32; group_size];
        for component in 0..3 {
            let maximum = remaining.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
            let d = if maximum == 0.0 {
                0.0
            } else {
                (maximum / 127.0).max(f32::from_bits(1))
            };
            for (lane, value) in remaining.iter_mut().enumerate() {
                let q = if d == 0.0 {
                    0
                } else {
                    (*value / d).round_ties_even().clamp(-127.0, 127.0) as i8
                };
                reconstructed[lane] += d * f32::from(q);
                *value = (-d).mul_add(f32::from(q), *value);
                if !value.is_finite() || !reconstructed[lane].is_finite() {
                    return Err(invalid("precise INT8 reconstruction overflow"));
                }
                let i = block * group_size + lane;
                result.values[component * input.len() + i] = q;
                result.sums[i / 32 * 3 + component] += i32::from(q);
            }
            result.scales.push(d);
        }
    }
    Ok(result)
}

fn reconstruction(input: &[f32], quantized: &Quantized, group_size: usize) -> Value {
    let mut maxima = [0.0f64; 3];
    for (i, &expected) in input.iter().enumerate() {
        let mut actual = 0.0f64;
        for (component, maximum) in maxima.iter_mut().enumerate() {
            actual += f64::from(quantized.scales[i / group_size * 3 + component])
                * f64::from(quantized.values[component * input.len() + i]);
            *maximum = maximum.max((actual - f64::from(expected)).abs());
        }
    }
    json!({"values":input.len(), "maximum_absolute_reconstruction_error_by_components":maxima,
        "scope":"Real-valued component reconstruction only; not matrix or full-model acceptance"})
}

fn fixture() -> Vec<f32> {
    (0..256)
        .map(|i| match i / 32 {
            0 => {
                if i % 2 == 0 {
                    0.0
                } else {
                    -0.0
                }
            }
            1 => f32::from_bits((i % 32 + 1) as u32),
            2 => {
                if i % 32 == 0 {
                    1000.0
                } else {
                    0.0013 * i as f32
                }
            }
            _ => (i as f32 * 0.371).sin() * 17.0 + (i % 7) as f32 * 0.013,
        })
        .collect()
}

#[test]
fn precise_three_component_reconstruction_and_errors() -> Result<()> {
    for input in [
        vec![],
        vec![0.0; 31],
        vec![f32::NAN; 32],
        vec![f32::MAX; 32],
    ] {
        assert!(quantize(&input).is_err());
    }
    let input = fixture();
    let quantized = quantize(&input)?;
    for (block, source) in input.chunks_exact(32).enumerate() {
        let maximum = source
            .iter()
            .map(|&x| f64::from(x).abs())
            .fold(0.0, f64::max);
        for (lane, &expected) in source.iter().enumerate() {
            let i = block * 32 + lane;
            let actual: f64 = (0..3)
                .map(|c| {
                    f64::from(quantized.scales[block * 3 + c])
                        * f64::from(quantized.values[c * input.len() + i])
                })
                .sum();
            assert!((actual - f64::from(expected)).abs() <= maximum * 8e-8 + 3e-45);
        }
    }
    let errors = reconstruction(&input, &quantized, 32);
    let maxima = errors["maximum_absolute_reconstruction_error_by_components"]
        .as_array()
        .unwrap();
    assert!(maxima[2].as_f64().unwrap() * 100.0 < maxima[1].as_f64().unwrap());
    Ok(())
}

fn check_gpu(harness: &Harness, input: &[f32]) -> Result<Value> {
    check_gpu_groups(harness, input, 32)
}

pub(super) fn check_gpu_groups(
    harness: &Harness,
    input: &[f32],
    group_size: usize,
) -> Result<Value> {
    if input.is_empty() || input.len() % group_size != 0 || input.len() > i32::MAX as usize / 3 {
        return Err(invalid(
            "precise INT8 GPU input shape exceeds kernel indexing",
        ));
    }
    let expected = quantize_groups(input, group_size);
    let x = harness.stream.clone_htod(input).map_err(failure)?;
    let mut q = harness
        .stream
        .clone_htod(&vec![85i8; input.len() * 3 + 16])
        .map_err(failure)?;
    let metadata = input.len() / group_size * 3;
    let sum_count = input.len() / 32 * 3;
    let mut d = harness
        .stream
        .clone_htod(&vec![1337.0f32; metadata + 16])
        .map_err(failure)?;
    let mut sums = harness
        .stream
        .clone_htod(&vec![1337i32; sum_count + 16])
        .map_err(failure)?;
    let mut flag = harness.stream.alloc_zeros::<i32>(1).map_err(failure)?;
    let mut q_view = q.slice_mut(8..8 + input.len() * 3);
    let mut d_view = d.slice_mut(8..8 + metadata);
    let mut sum_view = sums.slice_mut(8..8 + sum_count);
    let count = i32::try_from(input.len()).map_err(failure)?;
    launch!(
        harness,
        if group_size == 32 {
            "residual_quantize32_precise3"
        } else {
            "residual_quantize256_precise3"
        },
        ((count as u32).div_ceil(256), 1, 1),
        &x,
        &mut q_view,
        &mut d_view,
        &mut sum_view,
        &count,
        &mut flag
    );
    let observed_flag = harness.stream.clone_dtoh(&flag).map_err(failure)?[0];
    let q = harness.stream.clone_dtoh(&q).map_err(failure)?;
    let d = harness.stream.clone_dtoh(&d).map_err(failure)?;
    let sums = harness.stream.clone_dtoh(&sums).map_err(failure)?;
    assert!(q[..8].iter().chain(&q[q.len() - 8..]).all(|&x| x == 85));
    assert!(d[..8].iter().chain(&d[d.len() - 8..]).all(|&x| x == 1337.0));
    assert!(
        sums[..8]
            .iter()
            .chain(&sums[sums.len() - 8..])
            .all(|&x| x == 1337)
    );
    match expected {
        Ok(expected) => {
            assert_eq!(observed_flag, 0);
            assert_eq!(&q[8..q.len() - 8], expected.values);
            assert_eq!(&sums[8..sums.len() - 8], expected.sums);
            assert!(
                d[8..d.len() - 8]
                    .iter()
                    .zip(&expected.scales)
                    .all(|(&a, &b)| a.to_bits() == b.to_bits())
            );
            Ok(reconstruction(input, &expected, group_size))
        }
        Err(_) => {
            assert_eq!(
                observed_flag,
                if input.iter().any(|x| !x.is_finite()) {
                    1
                } else {
                    2
                }
            );
            Ok(json!({"expected_error_status":observed_flag, "guards_intact":true}))
        }
    }
}

#[test]
#[ignore = "serial CUDA precise3 quantization CPU-bit oracle, guards and optional real captures"]
fn precise_three_component_gpu_oracle() -> Result<()> {
    let harness = Harness::new()?;
    for input in [
        vec![f32::NAN; 32],
        vec![f32::INFINITY; 32],
        vec![f32::NEG_INFINITY; 32],
        vec![f32::MAX; 32],
        fixture(),
    ] {
        println!("PRECISE3_QUANT {}", check_gpu(&harness, &input)?);
    }
    if let Some(paths) = std::env::var_os("M8_PRECISE_CAPTURES") {
        for path in std::env::split_paths(&paths) {
            let reader = BufReader::new(std::fs::File::open(&path).map_err(failure)?);
            let mut input = Vec::new();
            for line in reader.lines() {
                let mut record: Value =
                    serde_json::from_str(&line.map_err(failure)?).map_err(failure)?;
                if record["kind"] == "projection" {
                    input.extend(
                        serde_json::from_value::<Vec<f32>>(record["values"].take())
                            .map_err(failure)?,
                    );
                }
            }
            let report = check_gpu(&harness, &input)?;
            println!(
                "PRECISE3_CAPTURE {}",
                json!({"capture":path, "result":report})
            );
        }
    }
    Ok(())
}

#[test]
#[ignore = "SM80+ integer MMA matrices, sparse routing, layouts, tails and CPU affine oracle; serial CUDA"]
fn precise_three_component_mma_matrices() -> Result<()> {
    check_mma_matrices(32)
}

pub(super) fn check_mma_matrices(group_size: usize) -> Result<()> {
    let harness = Harness::new()?;
    let mut fixtures = 0usize;
    let mut outputs = 0usize;
    let mut worst = 0.0f64;
    for kind in [12, 14] {
        for (width, rows, tokens) in [(256, 35, 35), (768, 33, 5), (2048, 5, 33), (4096, 3, 3)] {
            let ids: Vec<_> = (0..tokens)
                .flat_map(|t| (0..8).map(move |s| ((t % 2 * 4 + s) % 10) as i32))
                .collect();
            let routing: super::grouped_bench::RoutingFixture = serde_json::from_value(json!({
                "layer":0,"position":0,"experts":11,"used":8,"tokens":tokens,"ids":ids,
            }))
            .map_err(failure)?;
            routing.validate()?;
            let groups = harness
                .stream
                .clone_htod(&routing.descriptors(32))
                .map_err(failure)?;
            for (grouped, major) in [(false, 0), (true, 0), (true, 1)] {
                let input_count = tokens * if grouped && major == 1 { 8 } else { 1 };
                let input = super::prefill::inputs(width, input_count);
                let prepared = quantize_groups(&input, group_size)?;
                let reconstructed: Vec<f64> = (0..input.len())
                    .map(|i| {
                        (0..3)
                            .map(|c| {
                                f64::from(prepared.scales[i / group_size * 3 + c])
                                    * f64::from(prepared.values[c * input.len() + i])
                            })
                            .sum()
                    })
                    .collect();
                let tensor = super::tests::packed_fixture(kind, width, rows, 11);
                let count = tokens * rows * if grouped { 8 } else { 1 };
                let mut expected = Vec::with_capacity(count);
                let mut decoded = vec![0.0f32; width];
                for assignment in 0..count / rows {
                    let expert = if grouped {
                        routing.ids[assignment] as usize
                    } else {
                        0
                    };
                    let source = if grouped && major == 0 {
                        assignment / 8
                    } else {
                        assignment
                    };
                    for row in 0..rows {
                        tensor.row(expert * rows + row, &mut decoded);
                        let (mut value, mut magnitude) = (0.0f64, 0.0f64);
                        for k in 0..width {
                            let term = f64::from(decoded[k]) * reconstructed[source * width + k];
                            value += term;
                            magnitude += term.abs();
                        }
                        expected.push((value, magnitude));
                    }
                }
                let tensor = DeviceTensor::upload(tensor, &harness.stream)?;
                let q = harness
                    .stream
                    .clone_htod(&prepared.values)
                    .map_err(failure)?;
                let scales = harness
                    .stream
                    .clone_htod(&prepared.scales)
                    .map_err(failure)?;
                let sums = harness.stream.clone_htod(&prepared.sums).map_err(failure)?;
                let mut output = harness
                    .stream
                    .clone_htod(&vec![f32::NAN; count + 16])
                    .map_err(failure)?;
                let mut view = output.slice_mut(8..8 + count);
                let mut flag = harness.stream.alloc_zeros::<i32>(1).map_err(failure)?;
                let packs = i32::try_from(width * rows * 11 / 4).map_err(failure)?;
                launch!(
                    harness,
                    "residual_packed_quant_check",
                    ((packs as u32).div_ceil(256), 1, 1),
                    &tensor.bytes,
                    &tensor.kind,
                    &packs,
                    &mut flag
                );
                let n = tokens as i32;
                let used = 8i32;
                let is_grouped = i32::from(grouped);
                let grid = (
                    (rows as u32).div_ceil(32),
                    if grouped {
                        (groups.len() / 33) as u32
                    } else {
                        (tokens as u32).div_ceil(32)
                    },
                    1,
                );
                let kernel = if group_size == 32 {
                    "residual_matmul_mma3"
                } else {
                    "residual_matmul_mma3_block"
                };
                launch!(harness, kernel, grid;128,
                    &tensor.bytes, &tensor.kind, &q, &scales, &sums, &mut view,
                    &tensor.width, &tensor.rows, &n, &groups, &used, &is_grouped, &major, &mut flag);
                let actual = harness.stream.clone_dtoh(&output).map_err(failure)?;
                if group_size == 32 {
                    let mut pipeline = harness
                        .stream
                        .clone_htod(&vec![f32::NAN; count + 16])
                        .map_err(failure)?;
                    let mut pipeline_view = pipeline.slice_mut(8..8 + count);
                    launch!(harness, "residual_matmul_mma3_pipeline", grid;128,
                    &tensor.bytes, &tensor.kind, &q, &scales, &sums, &mut pipeline_view,
                    &tensor.width, &tensor.rows, &n, &groups, &used, &is_grouped, &major, &mut flag);
                    let pipeline = harness.stream.clone_dtoh(&pipeline).map_err(failure)?;
                    assert!(
                        pipeline[..8]
                            .iter()
                            .chain(&pipeline[count + 8..])
                            .all(|x| x.is_nan())
                    );
                    assert!(
                        actual[8..8 + count]
                            .iter()
                            .zip(&pipeline[8..8 + count])
                            .all(|(a, b)| a.to_bits() == b.to_bits()),
                        "pipeline changed kind{kind} width{width} grouped{grouped} major{major}"
                    );
                }
                assert_eq!(harness.stream.clone_dtoh(&flag).map_err(failure)?, [0]);
                assert!(
                    actual[..8]
                        .iter()
                        .chain(&actual[count + 8..])
                        .all(|x| x.is_nan())
                );
                for (&actual, &(expected, magnitude)) in actual[8..8 + count].iter().zip(&expected)
                {
                    assert!(actual.is_finite());
                    let error = (f64::from(actual) - expected).abs();
                    assert!(
                        error <= 1e-5 + 128.0 * f64::from(f32::EPSILON) * magnitude,
                        "kind{kind} width{width} grouped{grouped} major{major}: {actual} != {expected}"
                    );
                    worst = worst.max(error);
                }
                fixtures += 1;
                outputs += count;
            }
        }
    }
    println!(
        "PRECISE3_MMA {}",
        json!({"fixtures":fixtures,"outputs":outputs,
        "activation_scale_group":group_size,
        "worst_absolute_affine_oracle_error":worst,"all_finite_written_and_guarded":true,
        "scope":"Full synthetic matrix correctness with CPU-prepared components; not actual-model quality or throughput"})
    );
    Ok(())
}

struct Prepared {
    values: CudaSlice<i8>,
    scales: CudaSlice<f32>,
    sums: CudaSlice<i32>,
    input_hash: String,
}

impl Prepared {
    fn new(harness: &Harness, input: &[f32], x: &CudaSlice<f32>, stage: &str) -> Result<Self> {
        Self::new_groups(harness, input, x, stage, 32)
    }

    fn new_groups(
        harness: &Harness,
        input: &[f32],
        x: &CudaSlice<f32>,
        stage: &str,
        group_size: usize,
    ) -> Result<Self> {
        Self::new_components(harness, input, x, stage, group_size, 3)
    }

    fn new_components(
        harness: &Harness,
        input: &[f32],
        x: &CudaSlice<f32>,
        stage: &str,
        group_size: usize,
        components: usize,
    ) -> Result<Self> {
        let host = quantize_groups(input, group_size)?;
        let count = input.len() as i32;
        let value_count = input.len() * components;
        let metadata = host.scales.len() / 3 * components;
        let sum_count = host.sums.len() / 3 * components;
        let kernel = match (group_size, components) {
            (32, 1) => "residual_quantize32_precise1",
            (32, 3) => "residual_quantize32_precise3",
            (256, 3) => "residual_quantize256_precise3",
            _ => return Err(invalid("unsupported replay quantizer configuration")),
        };
        let mut result = Self {
            values: harness
                .stream
                .clone_htod(&vec![85i8; value_count + 16])
                .map_err(failure)?,
            scales: harness
                .stream
                .clone_htod(&vec![1337.0f32; metadata + 16])
                .map_err(failure)?,
            sums: harness
                .stream
                .clone_htod(&vec![1337i32; sum_count + 16])
                .map_err(failure)?,
            input_hash: super::model_replay::float_hash(input),
        };
        let mut flag = harness.stream.alloc_zeros::<i32>(1).map_err(failure)?;
        let mut timings = Vec::new();
        for repetition in 0..10 {
            let mut q = result.values.slice_mut(8..8 + value_count);
            let mut d = result.scales.slice_mut(8..8 + metadata);
            let mut s = result.sums.slice_mut(8..8 + sum_count);
            let start = harness
                .stream
                .record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))
                .map_err(failure)?;
            launch!(
                harness,
                kernel,
                ((count as u32).div_ceil(256), 1, 1),
                x,
                &mut q,
                &mut d,
                &mut s,
                &count,
                &mut flag
            );
            let end = harness
                .stream
                .record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))
                .map_err(failure)?;
            end.synchronize().map_err(failure)?;
            super::residual::status(harness, &flag)?;
            if repetition >= 3 {
                timings.push(start.elapsed_ms(&end).map_err(failure)?);
            }
        }
        let q = harness.stream.clone_dtoh(&result.values).map_err(failure)?;
        let d = harness.stream.clone_dtoh(&result.scales).map_err(failure)?;
        let s = harness.stream.clone_dtoh(&result.sums).map_err(failure)?;
        assert!(q[..8].iter().chain(&q[q.len() - 8..]).all(|&x| x == 85));
        assert!(d[..8].iter().chain(&d[d.len() - 8..]).all(|&x| x == 1337.0));
        assert!(s[..8].iter().chain(&s[s.len() - 8..]).all(|&x| x == 1337));
        assert_eq!(&q[8..q.len() - 8], &host.values[..value_count]);
        assert!(
            s[8..s.len() - 8].iter().copied().eq(host
                .sums
                .chunks_exact(3)
                .flat_map(|group| group[..components].iter().copied()))
        );
        assert!(
            d[8..d.len() - 8]
                .iter()
                .zip(
                    host.scales
                        .chunks_exact(3)
                        .flat_map(|group| &group[..components])
                )
                .all(|(&a, &b)| a.to_bits() == b.to_bits())
        );
        println!(
            "RESIDUAL_PREPARE {}",
            json!({
                "stage":stage,"values":input.len(),"input_sha256":result.input_hash,
                "activation_scale_group":group_size,
                "cuda_event_ms":timings,"reconstruction":reconstruction(input,&host,group_size),
                "workspace_bytes":value_count + (metadata+sum_count)*4,
                "cpu_quantization_bits_exact":true,
                "components":components,
                "includes":"activation components, scales and sums",
            })
        );
        Ok(result)
    }
}

struct Matrix<'a> {
    projection: &'a super::model_replay::Projection<'a>,
    prepared: &'a Prepared,
    planar_weights: Option<&'a CudaSlice<u8>>,
    flag: CudaSlice<i32>,
    kernel: &'static str,
}

pub(super) fn benchmark_gate_up(
    h: &Harness,
    tensors: Vec<Tensor>,
    routing: &super::grouped_bench::RoutingFixture,
    input: &[f32],
    frozen_down_input: &[f32],
    capture_hash: &str,
    model_hash: &str,
    screen_compact: bool,
    screen_k64: bool,
    screen_mixed: bool,
    screen_wide: bool,
    screen_same_expert: bool,
) -> Result<()> {
    use super::{
        model_replay::float_hash,
        residual::{matrix_output, read_matrix_output},
    };
    let tokens = routing.tokens as i32;
    let separate_candidate = screen_compact || screen_k64 || screen_mixed || screen_wide || screen_same_expert;
    let (down_kind, expected) = match capture_hash {
        "b3fe94936ce6b3ebdcc00469f6cda14f17f48c8dc60deff37ce511336b76fd14" if tokens == 256 => (
            12,
            [
                "f4c735bd771d5e10aae8c8a713ce1cfed82f1852d24d6a188765ac51fadd401a",
                "f4abbed123cf5c826cf11c1b5bf4b1726619c72223dc322c5c186c6b31aa3e07",
                "87284d39714165c3e07b16e4f4d3d01de8fb00847d2e19f1a4261f73cc84a895",
            ],
        ),
        "baef860c8adea671aadad32133680eb76f2980e5e90aaa69d86710f432803b81" if tokens == 256 => (
            14,
            [
                "e5e8cc9dc13848606753e9c8474a8680394d2aa85ee2f902e0ddd6d86a2ef44d",
                "9cda0e40e552a0fda171f3d5c4d2eecfe4b802a193867fbab219b5b98a484b90",
                "7acfe589592654145c2ff14febd6105484f71be8de95c127431feccb5664a172",
            ],
        ),
        "1c41262c1e3acd17503c89480f69fa83bf12e09e7ca8af71abf48a2ed1ad6d4b" if tokens == 256 => (
            14,
            [
                "3f9e48007e15bf3f9eb156b1a816f5ed85c11492c1a2636a28c32104e7a56a14",
                "e894ff398c06f6ea570a2f3bf37e8f92b5fa91c36400d08e1e9a0144a9b22870",
                "b7add5cb6dd4909960c1da8a35988f68d8c87ed597af0580f04431462ec0ad66",
            ],
        ),
        "fa590dfeea71a36cda673eb6f3e5044d43abea1e64b750fd92ac08d2fdbf5dbe" if tokens == 46 => (
            14,
            [
                "5b8d0b23e812fa015ad2bdc2092eefac9092cd4324c071ead3dfa3f615c2f324",
                "33532d6381b0e0ed2e9a0c885d470dc207b5ee033653a4290e72dd82e9fb713a",
                "8c97cfecb9200d8fb6258cfda3a47723f9de039ef432fbb335b501b3160afc11",
            ],
        ),
        "5e34c0a3dd2429ce061f8617bf7f7c9ed27e2f7aaad32d4f317260d60e13bd17" if tokens == 512 => (
            14,
            [
                "3abc650704fda20ace16e8a5a89f4bfa6885ff58d2b5e25daa064d23309868d4",
                "f5443f9c25192dfb72f9bf3d5d74f9fc76aaed9cf5f3cd4df002086a9123d9d4",
                "35caa31cd45b8e02a1fbb5d97d83b04f82587148c6060c26fd27502ad10c15fa",
            ],
        ),
        "80e9b95e7ab172bd9348676bcc92000fd5b55138af6d7299324f93301f5b67d1" if tokens == 32 => (
            12,
            [
                "e2823df5e5482e7a2a087fb4f5cfa6333863baf34cebdd8006d68cf07df6402c",
                "9a608e18eca1a8e11c113e33b236e9acbad10c5901645beb53e85550d4d6e966",
                "2fb3617acaceda429df25abd45d97a00e55eef8bdbf3aa198444251112782aae",
            ],
        ),
        "dd961ab91ccfcc44f1f40fd75d6d798c1bf0605d05e82234d22edd4c5abaaadb" if tokens == 46 => (
            14,
            [
                "bf7f8ae78b3b8e67eb8038bca72ada298d37fb87c393a841cd60ad674da38c51",
                "0c01c3544a973b318a3499d475512db6b47e2fe22e49a2ab7fbf18c14fb8aea7",
                "76bb4d0fec5abc345ee26012f1d1dd58853c909ee9bc72292a28655f51f11279",
            ],
        ),
        _ => return Err(invalid("unexpected frozen gate/up replay capture")),
    };
    if model_hash != "398bf479282bf374f996f55b0f742c4b45b798e51bf45ca9b7dda0e400d6aa86" {
        return Err(invalid("gate/up replay pin mismatch"));
    }
    let Ok([gate, up, down]) = <[Tensor; 3]>::try_from(tensors) else {
        return Err(invalid("gate/up replay requires three expert tensors"));
    };
    let [gate, up, down] = [gate, up, down].map(|t| DeviceTensor::upload(t, &h.stream));
    let (gate, up, down) = (gate?, up?, down?);
    assert_eq!([gate.kind, up.kind, down.kind], [12, 12, down_kind]);
    let ids = h.stream.clone_htod(&routing.ids).map_err(failure)?;
    let x = h.stream.clone_htod(input).map_err(failure)?;
    let frozen = h.stream.clone_htod(frozen_down_input).map_err(failure)?;
    let mut qu = Prepared::new(h, input, &x, "ffn_norm")?;
    let mut qd = Prepared::new(h, frozen_down_input, &frozen, "ffn_down_inp")?;
    let tiles = [32, 8];
    let words = tiles.map(|tile| routing.slots(tile) * (tile + 1));
    let mut descriptors = [
        h.stream
            .clone_htod(&vec![-777i32; words[0] + 16])
            .map_err(failure)?,
        h.stream
            .clone_htod(&vec![-777i32; words[1] + 16])
            .map_err(failure)?,
    ];
    let mut counts = h.stream.alloc_zeros::<i32>(128).map_err(failure)?;
    let schedule_words = 2 * routing.slots(32);
    let mut schedule_buffer = h.stream.clone_htod(&vec![-777i32; schedule_words + 16]).map_err(failure)?;
    let mut status = h.stream.alloc_zeros::<i32>(1).map_err(failure)?;
    let count = frozen_down_input.len();
    let activation_count = count as i32;
    let assignments = tokens * 8;
    let mut g = matrix_output(h, count)?;
    let mut u = matrix_output(h, count)?;
    let mut compact_up = matrix_output(h, count)?;
    let mut fused = matrix_output(h, count)?;
    let mut y = matrix_output(h, tokens as usize * 8 * 2048)?;
    let mut candidate_y = matrix_output(h, tokens as usize * 8 * 2048)?;
    let mut run = |candidate: bool,
                   activation: &mut CudaSlice<f32>,
                   up_output: &mut CudaSlice<f32>,
                   output: &mut CudaSlice<f32>,
                   panel: bool,
                   binding: bool|
     -> Result<()> {
        let compact = candidate && screen_compact;
        let mixed = candidate && screen_mixed;
        let same_expert = candidate && screen_same_expert;
        let merge = candidate && !separate_candidate;
        let layout = usize::from(compact);
        let slots = routing.slots(tiles[layout]) as i32;
        let mut groups = descriptors[layout].slice_mut(8..8 + words[layout]);
        let emit = if compact {
            "group_parallel_emit8"
        } else {
            "group_parallel_emit32"
        };
        launch!(h, "group_parallel_count32", (128,1,1);32,
            &ids, &mut counts, &assignments);
        launch!(h, emit, (129,1,1);32,
            &ids, &counts, &mut groups, &assignments, &128i32, &slots);
        let groups = groups.as_view();
        let mut schedule = schedule_buffer.slice_mut(8..8 + schedule_words);
        if mixed || same_expert {
            let emitter = if mixed { "group_parallel_mixed_schedule" } else { "group_parallel_same_expert_schedule" };
            launch!(h, emitter, (1,1,1);32,
                &groups, &slots, &mut schedule);
        }
        macro_rules! quantize {
            ($src:expr, $size:expr, $prep:ident) => {{
                let n = $size as i32;
                let mut q = $prep.values.slice_mut(8..$prep.values.len() - 8);
                let mut d = $prep.scales.slice_mut(8..$prep.scales.len() - 8);
                let mut s = $prep.sums.slice_mut(8..$prep.sums.len() - 8);
                launch!(
                    h,
                    "residual_quantize32_precise3",
                    ((n as u32).div_ceil(256), 1, 1),
                    $src,
                    &mut q,
                    &mut d,
                    &mut s,
                    &n,
                    &mut status
                );
            }};
        }
        quantize!(&x, input.len(), qu);
        macro_rules! matrix {
            ($tensor:ident, $prep:ident, $dst:expr, $assignment_major:expr) => {{
                let q = $prep.values.slice(8..$prep.values.len()-8);
                let d = $prep.scales.slice(8..$prep.scales.len()-8);
                let s = $prep.sums.slice(8..$prep.sums.len()-8);
                let grid = (($tensor.rows as u32).div_ceil(32), slots as u32, 1);
                if candidate && screen_wide {
                    let grid = (($tensor.rows as u32).div_ceil(64), slots as u32, 1);
                    launch!(h, "residual_matmul_mma3_rows64", grid;256,
                        &$tensor.bytes, &$tensor.kind, &q, &d, &s, $dst,
                        &$tensor.width, &$tensor.rows, &tokens, &groups, &8i32, &1i32,
                        &$assignment_major, &mut status);
                } else if mixed || same_expert {
                    let schedule = schedule.as_view();
                    let kernel = if mixed { "residual_matmul_mma3_mixed" } else { "residual_matmul_mma3_same_expert" };
                    launch!(h, kernel, grid;128,
                        &$tensor.bytes, &$tensor.kind, &q, &d, &s, $dst,
                        &$tensor.width, &$tensor.rows, &tokens, &groups, &8i32,
                        &$assignment_major, &schedule, &mut status);
                } else if compact {
                    launch!(h, "residual_matmul_mma3_compact8", grid;64,
                        &$tensor.bytes, &$tensor.kind, &q, &d, &s, $dst,
                        &$tensor.width, &$tensor.rows, &tokens, &groups, &8i32,
                        &$assignment_major, &mut status);
                } else {
                    let kernel = if candidate && screen_k64 && $tensor.kind == 12 && $assignment_major == 0 {
                        "residual_matmul_mma3_sync_k64"
                    } else {
                        "residual_matmul_mma3"
                    };
                    launch!(h, kernel, grid;128,
                        &$tensor.bytes, &$tensor.kind, &q, &d, &s, $dst,
                        &$tensor.width, &$tensor.rows, &tokens, &groups, &8i32, &1i32,
                        &$assignment_major, &mut status);
                }
            }};
        }
        let mut a = activation.slice_mut(8..8 + count);
        if merge {
            let q = qu.values.slice(8..qu.values.len() - 8);
            let d = qu.scales.slice(8..qu.scales.len() - 8);
            let s = qu.sums.slice(8..qu.sums.len() - 8);
            launch!(h, "residual_matmul_mma3_gate_up_q4", (24,slots as u32,1);128,
                &gate.bytes, &up.bytes, &q, &d, &s, &mut a,
                &gate.width, &gate.rows, &tokens, &groups, &8i32, &mut status);
        } else {
            matrix!(gate, qu, &mut a, 0i32);
            let mut b = up_output.slice_mut(8..8 + count);
            matrix!(up, qu, &mut b, 0i32);
            if !binding {
                let b = b.as_view();
                launch!(
                    h,
                    "swiglu",
                    ((count as u32).div_ceil(256), 1, 1),
                    &mut a,
                    &b,
                    &activation_count
                );
            }
        }
        if panel {
            if binding {
                quantize!(&frozen, count, qd);
            } else {
                let a = a.as_view();
                quantize!(&a, count, qd);
            }
            let mut z = output.slice_mut(8..output.len() - 8);
            matrix!(down, qd, &mut z, 1i32);
        }
        Ok(())
    };
    run(false, &mut g, &mut u, &mut y, true, true)?;
    let bound_gate = h.stream.clone_dtoh(&g).map_err(failure)?;
    let bound_up = h.stream.clone_dtoh(&u).map_err(failure)?;
    let bound_down = h.stream.clone_dtoh(&y).map_err(failure)?;
    if separate_candidate {
        run(
            true,
            &mut fused,
            &mut compact_up,
            &mut candidate_y,
            true,
            true,
        )?;
        assert_eq!(
            h.stream
                .clone_dtoh(&fused)
                .map_err(failure)?
                .iter()
                .map(|x| x.to_bits())
                .collect::<Vec<_>>(),
            bound_gate.iter().map(|x| x.to_bits()).collect::<Vec<_>>()
        );
        assert_eq!(
            h.stream
                .clone_dtoh(&candidate_y)
                .map_err(failure)?
                .iter()
                .map(|x| x.to_bits())
                .collect::<Vec<_>>(),
            bound_down.iter().map(|x| x.to_bits()).collect::<Vec<_>>()
        );
        assert_eq!(
            h.stream
                .clone_dtoh(&compact_up)
                .map_err(failure)?
                .iter()
                .map(|x| x.to_bits())
                .collect::<Vec<_>>(),
            bound_up.iter().map(|x| x.to_bits()).collect::<Vec<_>>()
        );
    }
    let mut samples = [[Vec::new(), Vec::new()], [Vec::new(), Vec::new()]];
    run(false, &mut g, &mut u, &mut y, true, false)?;
    run(
        true,
        &mut fused,
        &mut compact_up,
        &mut candidate_y,
        true,
        false,
    )?;
    let base_panel = h.stream.clone_dtoh(&y).map_err(failure)?;
    let candidate_panel = h.stream.clone_dtoh(&candidate_y).map_err(failure)?;
    let base_activation = h.stream.clone_dtoh(&g).map_err(failure)?;
    let candidate_activation = h.stream.clone_dtoh(&fused).map_err(failure)?;
    for (boundary, arm) in samples.iter_mut().enumerate() {
        for repetition in 0..10 {
            for slot in 0..2 {
                let i = (repetition + slot) % 2;
                let start = h
                    .stream
                    .record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))
                    .map_err(failure)?;
                run(i == 1, &mut g, &mut u, &mut y, boundary == 1, false)?;
                let end = h
                    .stream
                    .record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))
                    .map_err(failure)?;
                end.synchronize().map_err(failure)?;
                if repetition >= 3 {
                    arm[i].push(start.elapsed_ms(&end).map_err(failure)?);
                }
            }
        }
    }
    read_matrix_output(h, &status, &u)?;
    if screen_mixed || screen_same_expert {
        let schedule = h.stream.clone_dtoh(&schedule_buffer).map_err(failure)?;
        assert!(schedule[..8].iter().chain(&schedule[8 + schedule_words..]).all(|&x| x == -777));
    }
    let check_guarded = |v: &[f32]| {
        assert!(v[..8].iter().chain(&v[v.len() - 8..]).all(|x| *x == 1337.0));
        assert!(v[8..v.len() - 8].iter().all(|x| x.is_finite()));
        float_hash(&v[8..v.len() - 8])
    };
    assert_eq!(
        [
            check_guarded(&bound_gate),
            check_guarded(&bound_up),
            check_guarded(&bound_down)
        ],
        expected
    );
    assert_eq!(check_guarded(&base_panel), check_guarded(&candidate_panel));
    assert_eq!(
        check_guarded(&base_activation),
        check_guarded(&candidate_activation)
    );
    for layout in 0..=usize::from(screen_compact) {
        let actual = h.stream.clone_dtoh(&descriptors[layout]).map_err(failure)?;
        assert_eq!(
            actual[8..8 + words[layout]],
            routing.descriptors(tiles[layout])
        );
        assert!(
            actual[..8]
                .iter()
                .chain(&actual[8 + words[layout]..])
                .all(|&x| x == -777)
        );
    }
    for prepare in [&qu, &qd] {
        let q = h.stream.clone_dtoh(&prepare.values).map_err(failure)?;
        let s = h.stream.clone_dtoh(&prepare.sums).map_err(failure)?;
        let d = h.stream.clone_dtoh(&prepare.scales).map_err(failure)?;
        assert!(q[..8].iter().chain(&q[q.len() - 8..]).all(|&x| x == 85));
        assert!(s[..8].iter().chain(&s[s.len() - 8..]).all(|&x| x == 1337));
        assert!(d[..8].iter().chain(&d[d.len() - 8..]).all(|&x| x == 1337.0));
    }
    println!(
        "{} {}",
        if screen_compact {
            "COMPACT8_SCREEN"
        } else if screen_k64 {
            "SYNC_K64_SCREEN"
        } else if screen_mixed {
            "MIXED_EXPERT_SCREEN"
        } else if screen_wide {
            "WIDE_EXPERT_PANEL_SCREEN"
        } else if screen_same_expert {
            "SAME_EXPERT_PANEL_SCREEN"
        } else {
            "GATE_UP_SCREEN"
        },
        json!({
            "tokens":tokens, "capture_sha256":capture_hash, "model_sha256":model_hash,
            "mma_source_sha256":format!("{:x}",sha2::Sha256::digest(include_bytes!("residual_mma.cu"))),
            "baseline_projection_hashes":expected, "activation_sha256":check_guarded(&base_activation),
            "panel_sha256":check_guarded(&base_panel), "exact":true, "warmups":3, "repetitions":7,
            "replaced_path_cuda_event_ms":if separate_candidate {
                json!({"baseline":samples[0][0],"candidate":samples[0][1]})
            } else { json!({"separate":samples[0][0],"fused":samples[0][1]}) },
            "panel_cuda_event_ms":if separate_candidate {
                json!({"baseline":samples[1][0],"candidate":samples[1][1]})
            } else { json!({"separate":samples[1][0],"fused":samples[1][1]}) },
        })
    );
    Ok(())
}

impl super::residual::MatrixCandidate for Matrix<'_> {
    fn label(&self) -> &'static str {
        self.kernel
    }

    fn launch(&mut self, output: &mut CudaSlice<f32>) -> Result<()> {
        let p = self.projection;
        let harness = p.harness;
        let q = self
            .prepared
            .values
            .slice(8..self.prepared.values.len() - 8);
        let scales = self
            .prepared
            .scales
            .slice(8..self.prepared.scales.len() - 8);
        let sums = self.prepared.sums.slice(8..self.prepared.sums.len() - 8);
        let mut y = output.slice_mut(8..output.len() - 8);
        let tokens = p.tokens as i32;
        let grouped = i32::from(p.grouped);
        let used = 8i32;
        let rows64 = self.kernel == "residual_matmul_mma3_rows64";
        let grid = (
            (p.tensor.rows as u32).div_ceil(if rows64 { 64 } else { 32 }),
            if p.grouped {
                (p.groups.len() / 33) as u32
            } else {
                (p.tokens as u32).div_ceil(32)
            },
            1,
        );
        if matches!(
            self.kernel,
            "projection_fp16_control" | "projection_fp16_packed"
        ) {
            launch!(harness, self.kernel, grid;128,
                &p.tensor.bytes, &p.tensor.kind, &p.input, &mut y,
                &p.tensor.width, &p.tensor.rows, &tokens, p.groups, &used,
                &grouped, &p.assignment_major, &mut self.flag);
            return Ok(());
        }
        let wide = match self.kernel {
            "residual_matmul_mma3_cols8" => Some("residual_matmul_mma3_wide8"),
            "residual_matmul_mma3_cols16" => Some("residual_matmul_mma3_wide16"),
            _ => None,
        };
        let threads = if rows64 {
            256
        } else if wide.is_some() {
            64
        } else {
            128
        };
        let weights = if self.kernel == "residual_matmul_mma3_planar_q4" {
            self.planar_weights
                .ok_or_else(|| invalid("missing planar weight buffer"))?
        } else {
            &p.tensor.bytes
        };
        launch!(harness, self.kernel, grid;threads,
            weights, &p.tensor.kind, &q, &scales, &sums, &mut y,
            &p.tensor.width, &p.tensor.rows, &tokens, p.groups, &used,
            &grouped, &p.assignment_major, &mut self.flag);
        if let Some(kernel) = wide {
            launch!(harness, kernel, grid;128,
                &p.tensor.bytes, &p.tensor.kind, &q, &scales, &sums, &mut y,
                &p.tensor.width, &p.tensor.rows, &tokens, p.groups, &used,
                &grouped, &p.assignment_major, &mut self.flag);
        }
        Ok(())
    }

    fn read(&self, output: &CudaSlice<f32>) -> Result<Vec<f32>> {
        super::residual::read_matrix_output(self.projection.harness, &self.flag, output)
    }
}

pub(super) struct Experiment<'a> {
    harness: &'a Harness,
    prepared: BTreeMap<&'static str, Prepared>,
    pipeline: bool,
    block: bool,
    block_prepared: BTreeMap<&'static str, Prepared>,
    control_prepared: Option<BTreeMap<&'static str, Prepared>>,
    narrow: bool,
    fp16: bool,
    cache_scales: bool,
    wide_rows: bool,
    registers: bool,
    planar: bool,
}

impl<'a> Experiment<'a> {
    pub(super) fn new(harness: &'a Harness, pipeline: bool) -> Self {
        Self {
            harness,
            prepared: BTreeMap::new(),
            pipeline,
            block: false,
            block_prepared: BTreeMap::new(),
            control_prepared: None,
            narrow: false,
            fp16: false,
            cache_scales: false,
            wide_rows: false,
            registers: false,
            planar: false,
        }
    }

    pub(super) fn new_block(harness: &'a Harness) -> Self {
        Self {
            block: true,
            ..Self::new(harness, false)
        }
    }

    pub(super) fn new_control(harness: &'a Harness) -> Self {
        Self {
            control_prepared: Some(BTreeMap::new()),
            ..Self::new(harness, false)
        }
    }

    pub(super) fn new_narrow(harness: &'a Harness) -> Self {
        Self {
            narrow: true,
            ..Self::new(harness, false)
        }
    }

    pub(super) fn new_fp16(harness: &'a Harness) -> Self {
        Self {
            fp16: true,
            ..Self::new(harness, false)
        }
    }

    pub(super) fn new_cached_scales(harness: &'a Harness) -> Self {
        Self {
            cache_scales: true,
            ..Self::new(harness, false)
        }
    }

    pub(super) fn new_wide_rows(harness: &'a Harness) -> Self {
        Self {
            wide_rows: true,
            ..Self::new(harness, false)
        }
    }

    pub(super) fn new_registers(harness: &'a Harness) -> Self {
        Self {
            registers: true,
            ..Self::new(harness, false)
        }
    }

    pub(super) fn new_planar(harness: &'a Harness) -> Self {
        Self {
            planar: true,
            ..Self::new(harness, false)
        }
    }

    pub(super) fn probe(&mut self, probe: super::residual::Probe<'_>) -> Result<()> {
        let super::residual::Probe {
            tensor,
            input,
            expected,
            stage,
            name,
            tensor_hash,
            routing,
            grouped,
            assignment_major,
            groups,
        } = probe;
        if tensor.dims[0] % 256 != 0
            || !matches!(&tensor.storage, Storage::Packed { kind: 12 | 14, .. })
        {
            return Err(invalid("precise MMA requires block-aligned Q4_K/Q6_K"));
        }
        let harness = self.harness;
        let x = harness.stream.clone_htod(input).map_err(failure)?;
        if !self.prepared.contains_key(stage) {
            self.prepared
                .insert(stage, Prepared::new(harness, input, &x, stage)?);
        }
        let prepared = &self.prepared[stage];
        assert_eq!(prepared.input_hash, super::model_replay::float_hash(input));
        let planar_weights = if self.planar {
            let Storage::Packed { kind: 12, bytes } = &tensor.storage else {
                return Err(invalid("planar replay requires Q4_K"));
            };
            let blocks = tensor.dims[0] / 256;
            let rows = tensor.dims[1];
            if grouped || rows % 32 != 0 || bytes.len() != rows * blocks * 144 {
                return Err(invalid("planar replay requires complete dense 32-row tiles"));
            }
            let start = std::time::Instant::now();
            let mut planar = vec![0; bytes.len()];
            for tile in 0..rows / 32 {
                for block in 0..blocks {
                    for word in 0..36 {
                        for row in 0..32 {
                            let src = (((tile * 32 + row) * blocks + block) * 36 + word) * 4;
                            let dst = (((tile * blocks + block) * 36 + word) * 32 + row) * 4;
                            planar[dst..dst + 4].copy_from_slice(&bytes[src..src + 4]);
                        }
                    }
                }
            }
            println!(
                "PLANAR_REPACK {}",
                json!({"tensor":name, "bytes":planar.len(), "host_ms":start.elapsed().as_secs_f64()*1000.0})
            );
            Some(harness.stream.clone_htod(&planar).map_err(failure)?)
        } else {
            None
        };
        let small_groups = super::model_replay::small_groups(harness, routing, grouped)?;
        let projection = super::model_replay::Projection {
            harness,
            tensor: DeviceTensor::upload(tensor, &harness.stream)?,
            input: x,
            groups,
            small_groups: small_groups.as_ref(),
            tokens: routing.tokens,
            grouped,
            assignment_major,
        };
        let mut candidate = Matrix {
            projection: &projection,
            prepared,
            planar_weights: planar_weights.as_ref(),
            flag: harness.stream.alloc_zeros::<i32>(1).map_err(failure)?,
            kernel: "residual_matmul_mma3",
        };
        if let Some(directory) = std::env::var_os("M8_MMA_EXPORT_DIR") {
            let selected = std::env::var("M8_MMA_EXPORT_TENSOR").map_err(failure)?;
            if selected == name {
                let directory = PathBuf::from(directory);
                std::fs::create_dir(&directory)?;
                let arrays = [
                    (
                        "weights.bin",
                        harness
                            .stream
                            .clone_dtoh(&projection.tensor.bytes)
                            .map_err(failure)?,
                    ),
                    (
                        "values.bin",
                        harness
                            .stream
                            .clone_dtoh(&prepared.values)
                            .map_err(failure)?[8..prepared.values.len() - 8]
                            .iter()
                            .map(|&v| v as u8)
                            .collect(),
                    ),
                    (
                        "scales.bin",
                        harness
                            .stream
                            .clone_dtoh(&prepared.scales)
                            .map_err(failure)?[8..prepared.scales.len() - 8]
                            .iter()
                            .flat_map(|v| v.to_le_bytes())
                            .collect(),
                    ),
                    (
                        "sums.bin",
                        harness.stream.clone_dtoh(&prepared.sums).map_err(failure)?
                            [8..prepared.sums.len() - 8]
                            .iter()
                            .flat_map(|v| v.to_le_bytes())
                            .collect(),
                    ),
                    (
                        "groups.bin",
                        harness
                            .stream
                            .clone_dtoh(groups)
                            .map_err(failure)?
                            .iter()
                            .flat_map(|v| v.to_le_bytes())
                            .collect(),
                    ),
                ];
                for (file, bytes) in arrays {
                    std::fs::write(directory.join(file), bytes)?;
                }
                let mut output = super::residual::matrix_output(harness, expected.len())?;
                super::residual::MatrixCandidate::launch(&mut candidate, &mut output)?;
                let actual = super::residual::MatrixCandidate::read(&candidate, &output)?;
                std::fs::write(
                    directory.join("expected.bin"),
                    actual
                        .iter()
                        .flat_map(|v| v.to_le_bytes())
                        .collect::<Vec<_>>(),
                )?;
                let metadata = json!({
                    "tensor":name, "tensor_sha256":tensor_hash,
                    "input_sha256":prepared.input_hash, "kind":projection.tensor.kind,
                    "width":projection.tensor.width, "rows":projection.tensor.rows,
                    "tokens":routing.tokens, "used":8, "grouped":grouped,
                    "assignment_major":assignment_major,
                    "group_slots":groups.len()/33, "outputs":expected.len(),
                    "expected_sha256":super::model_replay::float_hash(&actual),
                    "source_sha256":format!("{:x}",sha2::Sha256::digest(include_bytes!("residual_mma.cu"))),
                    "scope":"Actual operands and current MMA3 output for an ABI-compatible standalone hardware profiler; not timing admission."
                });
                std::fs::write(directory.join("fixture.json"), metadata.to_string())?;
                println!("MMA_PROFILE_FIXTURE {metadata}");
            }
        }
        if self.planar || self.registers || self.wide_rows {
            return benchmark_pair(
                &mut candidate,
                prepared,
                if self.planar {
                    "residual_matmul_mma3_planar_q4"
                } else if self.registers {
                    "residual_matmul_mma3_registers"
                } else {
                    "residual_matmul_mma3_rows64"
                },
                expected,
                name,
                tensor_hash,
            );
        }
        if self.cache_scales {
            return benchmark_pair(
                &mut candidate,
                prepared,
                "residual_matmul_mma3_cached_scales",
                expected,
                name,
                tensor_hash,
            );
        }
        if self.fp16 {
            benchmark_pair(
                &mut candidate,
                prepared,
                "projection_fp16_control",
                expected,
                name,
                tensor_hash,
            )?;
            return benchmark_pair(
                &mut candidate,
                prepared,
                "projection_fp16_packed",
                expected,
                name,
                tensor_hash,
            );
        }
        if self.narrow {
            for kernel in ["residual_matmul_mma3_cols8", "residual_matmul_mma3_cols16"] {
                benchmark_pair(
                    &mut candidate,
                    prepared,
                    kernel,
                    expected,
                    name,
                    tensor_hash,
                )?;
            }
            return Ok(());
        }
        if let Some(control) = &mut self.control_prepared {
            if let std::collections::btree_map::Entry::Vacant(slot) = control.entry(stage) {
                slot.insert(Prepared::new_components(
                    harness,
                    input,
                    &projection.input,
                    stage,
                    32,
                    1,
                )?);
            }
            assert_eq!(control[stage].input_hash, prepared.input_hash);
            benchmark_pair(
                &mut candidate,
                &control[stage],
                "residual_matmul_mma1",
                expected,
                name,
                tensor_hash,
            )?;
            return benchmark_pair(
                &mut candidate,
                prepared,
                "residual_matmul_mma3_skip_empty",
                expected,
                name,
                tensor_hash,
            );
        }
        if self.block {
            if !self.block_prepared.contains_key(stage) {
                self.block_prepared.insert(
                    stage,
                    Prepared::new_groups(harness, input, &projection.input, stage, 256)?,
                );
            }
            let block = &self.block_prepared[stage];
            assert_eq!(block.input_hash, prepared.input_hash);
            return benchmark_pair(
                &mut candidate,
                block,
                "residual_matmul_mma3_block",
                expected,
                name,
                tensor_hash,
            );
        }
        if self.pipeline {
            return benchmark_pair(
                &mut candidate,
                prepared,
                "residual_matmul_mma3_pipeline",
                expected,
                name,
                tensor_hash,
            );
        }
        super::residual::benchmark_matrix_candidate(
            &projection,
            &mut candidate,
            expected,
            super::residual::MatrixMetadata {
                name,
                tensor_hash,
                stage,
                input_hash: &prepared.input_hash,
                shared_bytes: 5888,
                preparation: "all three components cached once per input stage; see RESIDUAL_PREPARE",
            },
        )
    }
}

fn benchmark_pair(
    baseline: &mut Matrix<'_>,
    prepared: &Prepared,
    kernel: &'static str,
    expected: &[f32],
    name: &str,
    tensor_hash: &str,
) -> Result<()> {
    let p = baseline.projection;
    let harness = p.harness;
    let mut binding = super::residual::matrix_output(harness, expected.len())?;
    p.launch(p.production_mode(), &mut binding)?;
    super::model_replay::assert_output(
        &harness.stream.clone_dtoh(&binding).map_err(failure)?,
        expected,
        "paired projection actual production binding",
    )?;
    let mut candidate = Matrix {
        projection: p,
        prepared,
        planar_weights: baseline.planar_weights,
        flag: harness.stream.alloc_zeros::<i32>(1).map_err(failure)?,
        kernel,
    };
    let mut output = super::residual::matrix_output(harness, expected.len())?;
    baseline.launch(&mut output)?;
    let reference = baseline.read(&output)?;
    output = super::residual::matrix_output(harness, expected.len())?;
    candidate.launch(&mut output)?;
    let actual = candidate.read(&output)?;
    let bitwise_equal = reference
        .iter()
        .zip(&actual)
        .all(|(a, b)| a.to_bits() == b.to_bits());
    let (label, record, shared_bytes, exact) = match kernel {
        "residual_matmul_mma3_block" => ("block", "BLOCK_MATRIX", 7040, false),
        "residual_matmul_mma3_pipeline" => ("pipeline", "PIPELINE_MATRIX", 11776, true),
        "residual_matmul_mma1" => ("one", "PRECISION_CONTROL_MATRIX", 3072, false),
        "residual_matmul_mma3_skip_empty" => ("skip_empty", "SKIP_EMPTY_MATRIX", 5888, true),
        "residual_matmul_mma3_cached_scales" => {
            ("cached_scales", "CACHED_SCALES_MATRIX", 5888, true)
        }
        "residual_matmul_mma3_rows64" => ("rows64", "WIDE_ROWS_MATRIX", 7424, true),
        "residual_matmul_mma3_registers" => ("registers", "REGISTERS_MATRIX", 0, true),
        "residual_matmul_mma3_planar_q4" => ("planar", "PLANAR_MATRIX", 5888, true),
        "residual_matmul_mma3_cols8" => ("cols8", "NARROW_MATRIX", 2624, true),
        "residual_matmul_mma3_cols16" => ("cols16", "NARROW_MATRIX", 3712, true),
        "projection_fp16_control" => ("fp16", "FP16_MATRIX", 4480, false),
        "projection_fp16_packed" => ("fp16_packed", "FP16_PACKED_MATRIX", 4480, false),
        _ => return Err(invalid("unknown paired matrix kernel")),
    };
    if exact {
        assert!(bitwise_equal, "{kernel} changed actual {name} output bits");
    }
    let mut samples: BTreeMap<&str, Vec<f32>> = BTreeMap::new();
    for repetition in 0..10 {
        for slot in 0..2 {
            let candidate_turn = (repetition + slot) % 2 == 1;
            let start = harness
                .stream
                .record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))
                .map_err(failure)?;
            if candidate_turn {
                candidate.launch(&mut output)?;
            } else {
                baseline.launch(&mut output)?;
            }
            let end = harness
                .stream
                .record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))
                .map_err(failure)?;
            end.synchronize().map_err(failure)?;
            if repetition >= 3 {
                samples
                    .entry(if candidate_turn { label } else { "baseline" })
                    .or_default()
                    .push(start.elapsed_ms(&end).map_err(failure)?);
            }
        }
    }
    baseline.read(&output)?;
    candidate.read(&output)?;
    println!(
        "{} {}",
        record,
        json!({
            "tensor":name, "tensor_sha256":tensor_hash, "kind":p.tensor.kind,
            "input_sha256":baseline.prepared.input_hash,
            "width":p.tensor.width, "rows":p.tensor.rows, "tokens":p.tokens,
            "grouped":p.grouped, "assignment_major":p.assignment_major,
            "output_values":expected.len(), "production_binding_bitwise_equal":true,
            "baseline_output_sha256":super::model_replay::float_hash(&reference),
            "candidate_output_sha256":super::model_replay::float_hash(&actual),
            "bitwise_equal":bitwise_equal, "guards_and_status_checked":true,
            "errors":super::residual::metrics(&actual,expected),
            "baseline_errors":super::residual::metrics(&reference,expected),
            "cuda_event_ms":samples, "baseline_shared_bytes":5888, "candidate_shared_bytes":shared_bytes,
            "scope":"Actual intermediate matrices, alternating3-warmup/7-event samples; common preparation and grouping reported separately."
        })
    );
    Ok(())
}
