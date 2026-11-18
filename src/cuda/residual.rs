//! Bounded arithmetic diagnostics, never production activation conversion.
use super::{grouped_bench::RoutingFixture, prefill::Harness, *};
use serde_json::{Value, json};
use std::collections::BTreeMap;

const MODES: [&str; 3] = [
    "residual_f32_dot",
    "residual_int8_dot",
    "residual_tf32x4_dot",
];

#[derive(Debug)]
struct Quantized {
    q0: Vec<i8>,
    q1: Vec<i8>,
    scales: Vec<f32>,
    sums: Vec<i32>,
}

fn quantize_cpu(input: &[f32]) -> Result<Quantized> {
    if input.is_empty() || input.len() % 32 != 0 || input.iter().any(|v| !v.is_finite()) {
        return Err(invalid(
            "residual quantization requires finite complete 32-value blocks",
        ));
    }
    let mut result = Quantized {
        q0: vec![],
        q1: vec![],
        scales: vec![],
        sums: vec![],
    };
    for block in input.chunks_exact(32) {
        let maximum = block.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
        let scale = |maximum: f32| {
            if maximum == 0.0 {
                0.0
            } else {
                (maximum / 127.0).max(f32::from_bits(1))
            }
        };
        let quant = |x: f32, d: f32| {
            if d == 0.0 {
                0
            } else {
                (x / d).round_ties_even().clamp(-127.0, 127.0) as i8
            }
        };
        let d0 = scale(maximum);
        let q0: Vec<_> = block.iter().map(|&x| quant(x, d0)).collect();
        let residual: Vec<_> = block
            .iter()
            .zip(&q0)
            .map(|(&x, &q)| x - d0 * f32::from(q))
            .collect();
        if residual.iter().any(|x| !x.is_finite()) {
            return Err(invalid("residual reconstruction overflow"));
        }
        let d1 = scale(residual.iter().map(|x| x.abs()).fold(0.0, f32::max));
        let q1: Vec<_> = residual.iter().map(|&x| quant(x, d1)).collect();
        result.scales.extend([d0, d1]);
        result.sums.extend([
            q0.iter().map(|&x| i32::from(x)).sum::<i32>(),
            q1.iter().map(|&x| i32::from(x)).sum::<i32>(),
        ]);
        result.q0.extend(q0);
        result.q1.extend(q1);
    }
    Ok(result)
}

pub(super) fn metrics(actual: &[f32], expected: &[f32]) -> Value {
    assert_eq!(actual.len(), expected.len());
    assert!(!actual.is_empty());
    let (mut max_abs, mut squared, mut dot, mut aa, mut bb) = (0.0f64, 0.0, 0.0, 0.0, 0.0);
    for (&a, &b) in actual.iter().zip(expected) {
        assert!(a.is_finite() && b.is_finite());
        let (a, b) = (f64::from(a), f64::from(b));
        max_abs = max_abs.max((a - b).abs());
        squared += (a - b) * (a - b);
        dot += a * b;
        aa += a * a;
        bb += b * b;
    }
    let cosine = if aa == 0.0 || bb == 0.0 {
        if aa == bb { 1.0 } else { 0.0 }
    } else {
        dot / (aa.sqrt() * bb.sqrt())
    };
    json!({"values":actual.len(), "max_abs":max_abs, "rmse":(squared / actual.len() as f64).sqrt(),
        "cosine":cosine, "within_logit_thresholds_diagnostic_only":max_abs <= 0.01 && cosine >= 0.99999})
}

pub(super) fn status(harness: &Harness, flag: &CudaSlice<i32>) -> Result<()> {
    let value = harness.stream.clone_dtoh(flag).map_err(failure)?[0];
    if value != 0 {
        return Err(failure(format!(
            "residual primitive status {value}: nonfinite input(1), reconstruction overflow(2), dot overflow(4), unsupported kind(8), or unsupported MMA capability(16)"
        )));
    }
    Ok(())
}

struct Prepared {
    q0: CudaSlice<i8>,
    q1: CudaSlice<i8>,
    scales: CudaSlice<f32>,
    sums: CudaSlice<i32>,
    host: Quantized,
    timings: Vec<f32>,
    input_hash: String,
}

impl Prepared {
    fn new(harness: &Harness, input: &[f32], x: &CudaSlice<f32>) -> Result<Self> {
        if input.is_empty() || input.len() % 32 != 0 || input.len() > i32::MAX as usize {
            return Err(invalid(
                "residual GPU quantization requires bounded complete 32-value blocks",
            ));
        }
        let count = input.len() as i32;
        let blocks = input.len() / 32;
        let mut q0 = harness
            .stream
            .clone_htod(&vec![85i8; input.len() + 16])
            .map_err(failure)?;
        let mut q1 = harness
            .stream
            .clone_htod(&vec![85i8; input.len() + 16])
            .map_err(failure)?;
        let mut scales = harness
            .stream
            .clone_htod(&vec![1337.0f32; blocks * 2 + 16])
            .map_err(failure)?;
        let mut sums = harness
            .stream
            .clone_htod(&vec![1337i32; blocks * 2 + 16])
            .map_err(failure)?;
        let mut flag = harness.stream.alloc_zeros::<i32>(1).map_err(failure)?;
        let mut timings = Vec::new();
        for repetition in 0..10 {
            let mut a = q0.slice_mut(8..8 + input.len());
            let mut b = q1.slice_mut(8..8 + input.len());
            let mut d = scales.slice_mut(8..8 + blocks * 2);
            let mut s = sums.slice_mut(8..8 + blocks * 2);
            let start = harness
                .stream
                .record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))
                .map_err(failure)?;
            launch!(
                harness,
                "residual_quantize32",
                ((input.len() as u32).div_ceil(256), 1, 1),
                x,
                &mut a,
                &mut b,
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
            status(harness, &flag)?;
            if repetition >= 3 {
                timings.push(start.elapsed_ms(&end).map_err(failure)?);
            }
        }
        let a = harness.stream.clone_dtoh(&q0).map_err(failure)?;
        let b = harness.stream.clone_dtoh(&q1).map_err(failure)?;
        let d = harness.stream.clone_dtoh(&scales).map_err(failure)?;
        let s = harness.stream.clone_dtoh(&sums).map_err(failure)?;
        assert!(a[..8].iter().chain(&a[a.len() - 8..]).all(|&x| x == 85));
        assert!(b[..8].iter().chain(&b[b.len() - 8..]).all(|&x| x == 85));
        assert!(d[..8].iter().chain(&d[d.len() - 8..]).all(|&x| x == 1337.0));
        assert!(s[..8].iter().chain(&s[s.len() - 8..]).all(|&x| x == 1337));
        let host = quantize_cpu(input)?;
        assert_eq!(&a[8..a.len() - 8], host.q0);
        assert_eq!(&b[8..b.len() - 8], host.q1);
        assert_eq!(&s[8..s.len() - 8], host.sums);
        assert!(
            d[8..d.len() - 8]
                .iter()
                .zip(&host.scales)
                .all(|(&x, &y)| x.to_bits() == y.to_bits())
        );
        Ok(Self {
            q0,
            q1,
            scales,
            sums,
            host,
            timings,
            input_hash: super::model_replay::float_hash(input),
        })
    }

    fn reconstruction(&self, input: &[f32]) -> Value {
        let first: Vec<_> = self
            .host
            .q0
            .iter()
            .enumerate()
            .map(|(i, &q)| self.host.scales[i / 32 * 2] * f32::from(q))
            .collect();
        let both: Vec<_> = first
            .iter()
            .enumerate()
            .map(|(i, &x)| x + self.host.scales[i / 32 * 2 + 1] * f32::from(self.host.q1[i]))
            .collect();
        json!({"one_component":metrics(&first,input), "two_components":metrics(&both,input)})
    }
}

pub(super) struct Experiment<'a> {
    harness: &'a Harness,
    prepared: BTreeMap<&'static str, Prepared>,
    tiled: bool,
}

pub(super) struct Probe<'a> {
    pub tensor: Tensor,
    pub input: &'a [f32],
    pub expected: &'a [f32],
    pub stage: &'static str,
    pub name: &'a str,
    pub tensor_hash: &'a str,
    pub routing: &'a RoutingFixture,
    pub grouped: bool,
    pub assignment_major: i32,
    pub groups: &'a CudaSlice<i32>,
}

// Each metadata group is reduced in i32 before scaling: Q4 bound32*15*127=60960,
// Q6 bound16*32*127=65024; activation-sum bound32*127=4064.
fn affine_oracle(tensor: &Tensor, quant: &Quantized, row: usize, input: usize) -> (f64, f64) {
    let Storage::Packed { kind, bytes } = &tensor.storage else {
        unreachable!()
    };
    let width = tensor.dims[0];
    let size = crate::gguf::layout(*kind).unwrap().1;
    let (mut result, mut magnitude) = (0.0, 0.0);
    for group in 0..width / 32 {
        let block = &bytes[(row * width + group * 32) / 256 * size..][..size];
        let i = group * 32 % 256;
        for half in 0..if *kind == 12 { 1 } else { 2 } {
            let start = input * width + group * 32 + half * 16;
            let mut dots = [0i32; 2];
            for k in 0..if *kind == 12 { 32 } else { 16 } {
                let index = i + half * 16 + k;
                let q = if *kind == 12 {
                    i32::from(
                        (block[16 + index / 64 * 32 + index % 32] >> (4 * (index / 32 % 2))) & 15,
                    )
                } else {
                    let h = index / 128;
                    let quarter = index % 128 / 32;
                    let low =
                        (block[h * 64 + index % 32 + quarter % 2 * 32] >> (4 * (quarter / 2))) & 15;
                    let high = (block[128 + h * 32 + index % 32] >> (2 * quarter)) & 3;
                    i32::from(low | high << 4) - 32
                };
                dots[0] += q * i32::from(quant.q0[start + k]);
                dots[1] += q * i32::from(quant.q1[start + k]);
            }
            let (a, b) = if *kind == 12 {
                let g = i / 32;
                let s = &block[4..16];
                let scale = if g < 4 {
                    s[g] & 63
                } else {
                    (s[g + 4] & 15) | ((s[g - 4] >> 6) << 4)
                };
                let minimum = if g < 4 {
                    s[g + 4] & 63
                } else {
                    (s[g + 4] >> 4) | ((s[g] >> 6) << 4)
                };
                (
                    crate::gguf::half(block, 0) * f32::from(scale),
                    crate::gguf::half(block, 2) * f32::from(minimum),
                )
            } else {
                (
                    crate::gguf::half(block, 208) * f32::from(block[192 + i / 16 + half] as i8),
                    0.0,
                )
            };
            for (component, dot) in dots.into_iter().enumerate() {
                let index = (start / 32) * 2 + component;
                let positive = f64::from(a) * f64::from(dot);
                let negative = if *kind == 12 {
                    f64::from(b) * f64::from(quant.sums[index])
                } else {
                    0.0
                };
                let scale = f64::from(quant.scales[index]);
                result += (positive - negative) * scale;
                magnitude += (positive.abs() + negative.abs()) * scale.abs();
            }
        }
    }
    (result, magnitude)
}

struct Dots<'a> {
    harness: &'a Harness,
    tensor: &'a DeviceTensor,
    input: &'a CudaSlice<f32>,
    prepared: &'a Prepared,
    pairs: CudaSlice<i32>,
    count: usize,
    flag: CudaSlice<i32>,
}

impl Dots<'_> {
    fn launch(&mut self, mode: &str, output: &mut CudaSlice<f32>) -> Result<()> {
        let mut y = output.slice_mut(8..8 + self.count);
        let count = self.count as i32;
        let mut args = self
            .harness
            .stream
            .launch_builder(&self.harness.functions[mode]);
        args.arg(&self.tensor.bytes).arg(&self.tensor.kind);
        let q0 = self.prepared.q0.slice(8..self.prepared.q0.len() - 8);
        let q1 = self.prepared.q1.slice(8..self.prepared.q1.len() - 8);
        let scales = self
            .prepared
            .scales
            .slice(8..self.prepared.scales.len() - 8);
        let sums = self.prepared.sums.slice(8..self.prepared.sums.len() - 8);
        if mode == "residual_int8_dot" {
            args.arg(&q0).arg(&q1).arg(&scales).arg(&sums);
        } else {
            args.arg(self.input);
        }
        args.arg(&self.pairs)
            .arg(&mut y)
            .arg(&self.tensor.width)
            .arg(&count)
            .arg(&mut self.flag);
        let f32_mode = mode == "residual_f32_dot";
        let cfg = LaunchConfig {
            grid_dim: (
                if f32_mode {
                    self.count as u32
                } else {
                    (self.count as u32).div_ceil(4)
                },
                1,
                1,
            ),
            block_dim: (if f32_mode { 256 } else { 128 }, 1, 1),
            shared_mem_bytes: 0,
        };
        // Pairs are bounded to validated packed rows and captured input rows.
        unsafe { args.launch(cfg) }.map_err(failure)?;
        Ok(())
    }

    fn read(&self, output: &CudaSlice<f32>) -> Result<Vec<f32>> {
        status(self.harness, &self.flag)?;
        let values = self.harness.stream.clone_dtoh(output).map_err(failure)?;
        assert!(
            values[..8]
                .iter()
                .chain(&values[values.len() - 8..])
                .all(|&x| x == 1337.0)
        );
        assert!(values[8..values.len() - 8].iter().all(|v| v.is_finite()));
        Ok(values[8..values.len() - 8].to_vec())
    }
}

impl<'a> Experiment<'a> {
    pub fn new(harness: &'a Harness) -> Self {
        Self {
            harness,
            prepared: BTreeMap::new(),
            tiled: false,
        }
    }

    pub fn new_tiled(harness: &'a Harness) -> Self {
        Self {
            tiled: true,
            ..Self::new(harness)
        }
    }

    pub fn probe(&mut self, probe: Probe<'_>) -> Result<()> {
        let Probe {
            tensor,
            input,
            expected,
            stage,
            name,
            routing,
            grouped,
            assignment_major,
            groups,
            tensor_hash,
        } = probe;
        let (width, rows) = (tensor.dims[0], tensor.dims[1]);
        if width % 256 != 0 || !matches!(&tensor.storage, Storage::Packed { kind: 12 | 14, .. }) {
            return Err(invalid(
                "residual dot research supports block-aligned Q4_K/Q6_K only",
            ));
        }
        let harness = self.harness;
        let x = harness.stream.clone_htod(input).map_err(failure)?;
        if !self.prepared.contains_key(stage) {
            let prepared = Prepared::new(harness, input, &x)?;
            println!(
                "RESIDUAL_PREPARE {}",
                json!({"stage":stage,"values":input.len(),
                "input_sha256":prepared.input_hash, "cuda_event_ms":prepared.timings,
                "reconstruction":prepared.reconstruction(input),
                "workspace_bytes":input.len()*2 + input.len()/32*16,
                "cpu_quantization_bits_exact":true, "includes":"both quantizations, scales and activation sums; no scatter needed"})
            );
            self.prepared.insert(stage, prepared);
        }
        let prepared = &self.prepared[stage];
        assert_eq!(prepared.input_hash, super::model_replay::float_hash(input));
        let small_groups = super::model_replay::small_groups(harness, routing, grouped)?;
        if self.tiled {
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
            return benchmark_matrix(&projection, prepared, expected, name, tensor_hash, stage);
        }
        let selected = [
            0,
            1,
            rows / 8,
            rows / 4,
            rows / 2,
            rows * 3 / 4,
            rows - 2,
            rows - 1,
        ];
        let assignments = routing.tokens * if grouped { routing.used } else { 1 };
        let mut pairs = Vec::new();
        let mut captured = Vec::new();
        for assignment in 0..assignments {
            let expert = if grouped {
                routing.ids[assignment] as usize
            } else {
                0
            };
            let source = if grouped && assignment_major == 0 {
                assignment / routing.used
            } else {
                assignment
            };
            for row in selected {
                pairs.extend([(expert * rows + row) as i32, source as i32]);
                captured.push(expected[assignment * rows + row]);
            }
        }
        let mut oracle = Vec::new();
        let count = captured.len();
        let indices: Vec<_> = (0..16).map(|n| n * (count - 1) / 15).collect();
        let mut decoded = vec![0.0f32; width];
        for &index in &indices {
            let row = pairs[index * 2] as usize;
            let source = pairs[index * 2 + 1] as usize;
            tensor.row(row, &mut decoded);
            let reference = decoded
                .iter()
                .zip(&input[source * width..(source + 1) * width])
                .map(|(&w, &x)| f64::from(w) * f64::from(x))
                .sum::<f64>();
            let (affine, magnitude) = affine_oracle(&tensor, &prepared.host, row, source);
            let reconstructed = decoded
                .iter()
                .enumerate()
                .map(|(k, &w)| {
                    let i = source * width + k;
                    let reconstructed = f64::from(prepared.host.scales[i / 32 * 2])
                        * f64::from(prepared.host.q0[i])
                        + f64::from(prepared.host.scales[i / 32 * 2 + 1])
                            * f64::from(prepared.host.q1[i]);
                    f64::from(w) * reconstructed
                })
                .sum::<f64>();
            oracle.push((index, reference, reconstructed, affine, magnitude));
        }
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
        let production = projection.production_mode();
        let mut full = harness
            .stream
            .clone_htod(&vec![1337.0; expected.len() + 16])
            .map_err(failure)?;
        projection.launch(production, &mut full)?;
        super::model_replay::assert_output(
            &harness.stream.clone_dtoh(&full).map_err(failure)?,
            expected,
            "production binding",
        )?;
        drop(full);
        let mut dots = Dots {
            harness,
            tensor: &projection.tensor,
            input: &projection.input,
            prepared,
            pairs: harness.stream.clone_htod(&pairs).map_err(failure)?,
            count,
            flag: harness.stream.alloc_zeros::<i32>(1).map_err(failure)?,
        };
        let mut output = harness
            .stream
            .clone_htod(&vec![1337.0; count + 16])
            .map_err(failure)?;
        let mut outputs = BTreeMap::new();
        for mode in MODES {
            harness
                .stream
                .memcpy_htod(&vec![1337.0; count + 16], &mut output)
                .map_err(failure)?;
            dots.launch(mode, &mut output)?;
            outputs.insert(mode, dots.read(&output)?);
        }
        let mut oracle_records = Vec::new();
        for (index, reference, reconstructed, affine, magnitude) in oracle {
            let actual = f64::from(outputs["residual_int8_dot"][index]);
            let error = (actual - affine).abs();
            assert!(
                error <= 1e-5 + 64.0 * f64::from(f32::EPSILON) * magnitude,
                "DP4A affine oracle output={index}: {actual} != {affine}"
            );
            let baseline = f64::from(outputs["residual_f32_dot"][index]);
            assert!(
                (baseline - reference).abs() <= 0.001 + reference.abs() * 1e-4,
                "F32 independent sampled oracle: {baseline} != {reference}"
            );
            oracle_records.push(json!({"index":index,"f64_decoded_dot":reference,
                "f64_affine_two_component_dot":affine,"gpu_integer_dot":actual,
                "integer_merge_rounding_error":error,
                "activation_reconstruction_dot_error":reconstructed-reference,
                "decoder_affine_rounding_error":affine-reconstructed,
                "f32_reduction_error":baseline-reference,
                "affine_vs_original_error":(affine-reference).abs()}));
        }
        for repetition in 0..3 {
            for slot in 0..MODES.len() {
                dots.launch(MODES[(slot + repetition) % MODES.len()], &mut output)?;
            }
        }
        let mut timings: BTreeMap<&str, Vec<f32>> = BTreeMap::new();
        for repetition in 0..7 {
            for slot in 0..MODES.len() {
                let mode = MODES[(slot + repetition) % MODES.len()];
                let start = harness
                    .stream
                    .record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))
                    .map_err(failure)?;
                dots.launch(mode, &mut output)?;
                let end = harness
                    .stream
                    .record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))
                    .map_err(failure)?;
                end.synchronize().map_err(failure)?;
                timings
                    .entry(mode)
                    .or_default()
                    .push(start.elapsed_ms(&end).map_err(failure)?);
            }
        }
        status(harness, &dots.flag)?;
        let comparisons: BTreeMap<_, _> = MODES
            .into_iter()
            .map(|mode| {
                (
                    mode,
                    json!({
                        "output_sha256":super::model_replay::float_hash(&outputs[mode]),
                        "vs_captured":metrics(&outputs[mode],&captured),
                        "vs_gpu_f32":metrics(&outputs[mode],&outputs["residual_f32_dot"]),
                    }),
                )
            })
            .collect();
        println!(
            "RESIDUAL_DOTS {}",
            json!({"tensor":name,"tensor_sha256":tensor_hash,"input_stage":stage,
                "expected_full_output_sha256":super::model_replay::float_hash(expected),
                "kind":projection.tensor.kind,"width":width,"rows":rows,"tokens":routing.tokens,
                "grouped":grouped,"assignment_major":assignment_major,"sampled_rows":selected,
                "sampled_dots":count,"production_binding_bitwise_equal":true,
                "comparisons":comparisons,"cuda_event_ms":timings,"cpu_oracles":oracle_records,
                "scope":"sampled dot primitives, not GEMM throughput, full logits or model admission",
                "float_control":"four separately accumulated scalar TF32 products, not tensor-core throughput",
                "prepared_once_per_stage":true,"preparation_amortization_branches":if stage=="attn_norm"{3}else if stage=="ffn_norm"{2}else{1},
            })
        );
        Ok(())
    }
}

struct Matrix<'a> {
    projection: &'a super::model_replay::Projection<'a>,
    prepared: &'a Prepared,
    flag: CudaSlice<i32>,
}

impl Matrix<'_> {
    fn launch(&mut self, output: &mut CudaSlice<f32>) -> Result<()> {
        let p = self.projection;
        let harness = p.harness;
        let q0 = self.prepared.q0.slice(8..self.prepared.q0.len() - 8);
        let q1 = self.prepared.q1.slice(8..self.prepared.q1.len() - 8);
        let scales = self
            .prepared
            .scales
            .slice(8..self.prepared.scales.len() - 8);
        let sums = self.prepared.sums.slice(8..self.prepared.sums.len() - 8);
        let mut y = output.slice_mut(8..output.len() - 8);
        let tokens = p.tokens as i32;
        let grouped = i32::from(p.grouped);
        let used = 8i32;
        let grid = (
            (p.tensor.rows as u32).div_ceil(32),
            if p.grouped {
                (p.groups.len() / 33) as u32
            } else {
                (p.tokens as u32).div_ceil(32)
            },
            1,
        );
        launch!(harness,"residual_matmul32",grid;128,
            &p.tensor.bytes,&p.tensor.kind,&q0,&q1,&scales,&sums,&mut y,
            &p.tensor.width,&p.tensor.rows,&tokens,p.groups,&used,&grouped,
            &p.assignment_major,&mut self.flag);
        Ok(())
    }
}

#[test]
#[ignore = "full INT8 matrix tails, sparse routing, input layouts and independent affine oracles; serial CUDA release"]
fn residual_matrix_tails_routing_and_affine_oracles() -> Result<()> {
    let harness = Harness::new()?;
    for kind in [12, 14] {
        for (width, rows, tokens) in [(256, 35, 35), (768, 33, 5), (2048, 5, 33), (4096, 3, 3)] {
            let ids: Vec<_> = (0..tokens)
                .flat_map(|t| (0..8).map(move |s| ((t % 2 * 4 + s) % 10) as i32))
                .collect();
            let routing: RoutingFixture = serde_json::from_value(json!({
                "layer":0,"position":0,"experts":11,"used":8,"tokens":tokens,"ids":ids,
            }))
            .map_err(failure)?;
            routing.validate()?;
            let groups = harness
                .stream
                .clone_htod(&routing.descriptors(32))
                .map_err(failure)?;
            for (grouped, assignment_major) in [(false, 0), (true, 0), (true, 1)] {
                let tensor = super::tests::packed_fixture(kind, width, rows, 11);
                let input_count = tokens
                    * if grouped && assignment_major == 1 {
                        8
                    } else {
                        1
                    };
                let input = super::prefill::inputs(width, input_count);
                let x = harness.stream.clone_htod(&input).map_err(failure)?;
                let prepared = Prepared::new(&harness, &input, &x)?;
                let count = tokens * rows * if grouped { 8 } else { 1 };
                let mut oracles = Vec::new();
                for index in 0..count {
                    let assignment = index / rows;
                    let expert = if grouped {
                        routing.ids[assignment] as usize
                    } else {
                        0
                    };
                    let row = expert * rows + index % rows;
                    let source = if grouped && assignment_major == 0 {
                        assignment / 8
                    } else {
                        assignment
                    };
                    let (affine, magnitude) = affine_oracle(&tensor, &prepared.host, row, source);
                    oracles.push((index, affine, magnitude));
                }
                let projection = super::model_replay::Projection {
                    harness: &harness,
                    tensor: DeviceTensor::upload(tensor, &harness.stream)?,
                    input: x,
                    groups: &groups,
                    small_groups: None,
                    tokens,
                    grouped,
                    assignment_major,
                };
                let production = if grouped {
                    super::model_replay::EXPERT[0]
                } else {
                    super::model_replay::DENSE[0]
                };
                let mut output = matrix_output(&harness, count)?;
                let mut matrix = Matrix {
                    projection: &projection,
                    prepared: &prepared,
                    flag: harness.stream.alloc_zeros::<i32>(1).map_err(failure)?,
                };
                assert!(
                    matrix.read(&output).is_err(),
                    "unwritten matrix outputs must fail"
                );
                matrix.launch(&mut output)?;
                let actual = matrix.read(&output)?;
                for (index, affine, magnitude) in oracles {
                    assert!(
                        (f64::from(actual[index]) - affine).abs()
                            <= 1e-5 + 128.0 * f64::from(f32::EPSILON) * magnitude,
                        "kind={kind} width={width} grouped={grouped} layout={assignment_major} index={index}"
                    );
                }
                output = matrix_output(&harness, count)?;
                projection.launch(production, &mut output)?;
                let reference = matrix.read(&output)?;
                let comparison = metrics(&actual, &reference);
                println!(
                    "RESIDUAL_MATRIX_FIXTURE {}",
                    json!({
                        "kind":kind,"width":width,"rows":rows,"tokens":tokens,"grouped":grouped,
                        "assignment_major":assignment_major,"all_affine_oracles_checked":count,
                        "vs_f32":comparison,"scope":"synthetic intermediate matrix, not full-model admission",
                    })
                );
            }
        }
    }
    let mut tensor = super::tests::packed_fixture(12, 256, 1, 1);
    if let Storage::Packed { bytes, .. } = &mut tensor.storage {
        bytes[..2].copy_from_slice(&half::f16::from_f32(1.0).to_bits().to_le_bytes());
        bytes[2..4].fill(0);
        bytes[4..].fill(255);
    }
    let input = vec![1e36; 256];
    let x = harness.stream.clone_htod(&input).map_err(failure)?;
    let prepared = Prepared::new(&harness, &input, &x)?;
    let groups = harness.stream.clone_htod(&[-1i32; 33]).map_err(failure)?;
    let projection = super::model_replay::Projection {
        harness: &harness,
        tensor: DeviceTensor::upload(tensor, &harness.stream)?,
        input: x,
        groups: &groups,
        small_groups: None,
        tokens: 1,
        grouped: false,
        assignment_major: 0,
    };
    let mut matrix = Matrix {
        projection: &projection,
        prepared: &prepared,
        flag: harness.stream.alloc_zeros::<i32>(1).map_err(failure)?,
    };
    let mut output = matrix_output(&harness, 1)?;
    matrix.launch(&mut output)?;
    assert!(
        matrix
            .read(&output)
            .unwrap_err()
            .to_string()
            .contains("status 4")
    );
    Ok(())
}

impl Matrix<'_> {
    fn read(&self, output: &CudaSlice<f32>) -> Result<Vec<f32>> {
        read_matrix_output(self.projection.harness, &self.flag, output)
    }
}

pub(super) fn read_matrix_output(
    harness: &Harness,
    flag: &CudaSlice<i32>,
    output: &CudaSlice<f32>,
) -> Result<Vec<f32>> {
    status(harness, flag)?;
    let values = harness.stream.clone_dtoh(output).map_err(failure)?;
    if values[..8]
        .iter()
        .chain(&values[values.len() - 8..])
        .any(|&v| v != 1337.0)
        || values[8..values.len() - 8].iter().any(|v| !v.is_finite())
    {
        return Err(failure(
            "residual matrix changed guards or left nonfinite/unwritten outputs",
        ));
    }
    Ok(values[8..values.len() - 8].to_vec())
}

pub(super) fn matrix_output(harness: &Harness, count: usize) -> Result<CudaSlice<f32>> {
    let mut values = vec![f32::NAN; count + 16];
    values[..8].fill(1337.0);
    values[count + 8..].fill(1337.0);
    harness.stream.clone_htod(&values).map_err(failure)
}

fn benchmark_matrix(
    projection: &super::model_replay::Projection<'_>,
    prepared: &Prepared,
    expected: &[f32],
    name: &str,
    tensor_hash: &str,
    stage: &str,
) -> Result<()> {
    let mut candidate = Matrix {
        projection,
        prepared,
        flag: projection
            .harness
            .stream
            .alloc_zeros::<i32>(1)
            .map_err(failure)?,
    };
    benchmark_matrix_candidate(
        projection,
        &mut candidate,
        expected,
        MatrixMetadata {
            name,
            tensor_hash,
            stage,
            input_hash: &prepared.input_hash,
            shared_bytes: 4480,
            preparation: "both components cached once per input stage; see RESIDUAL_PREPARE",
        },
    )
}

pub(super) trait MatrixCandidate {
    fn label(&self) -> &'static str;
    fn launch(&mut self, output: &mut CudaSlice<f32>) -> Result<()>;
    fn read(&self, output: &CudaSlice<f32>) -> Result<Vec<f32>>;
}

impl MatrixCandidate for Matrix<'_> {
    fn label(&self) -> &'static str {
        "residual_matmul32"
    }
    fn launch(&mut self, output: &mut CudaSlice<f32>) -> Result<()> {
        Matrix::launch(self, output)
    }
    fn read(&self, output: &CudaSlice<f32>) -> Result<Vec<f32>> {
        Matrix::read(self, output)
    }
}

pub(super) struct MatrixMetadata<'a> {
    pub name: &'a str,
    pub tensor_hash: &'a str,
    pub stage: &'a str,
    pub input_hash: &'a str,
    pub shared_bytes: usize,
    pub preparation: &'a str,
}

pub(super) fn benchmark_matrix_candidate(
    projection: &super::model_replay::Projection<'_>,
    candidate: &mut impl MatrixCandidate,
    expected: &[f32],
    metadata: MatrixMetadata<'_>,
) -> Result<()> {
    let MatrixMetadata {
        name,
        tensor_hash,
        stage,
        input_hash,
        shared_bytes,
        preparation,
    } = metadata;
    let harness = projection.harness;
    let production = projection.production_mode();
    let mut output = matrix_output(harness, expected.len())?;
    projection.launch(production, &mut output)?;
    super::model_replay::assert_output(
        &harness.stream.clone_dtoh(&output).map_err(failure)?,
        expected,
        "full-matrix production binding",
    )?;
    output = matrix_output(harness, expected.len())?;
    candidate.launch(&mut output)?;
    let actual = candidate.read(&output)?;
    let errors = metrics(&actual, expected);
    let worst = (0..actual.len())
        .max_by(|&a, &b| {
            (f64::from(actual[a]) - f64::from(expected[a]))
                .abs()
                .total_cmp(&(f64::from(actual[b]) - f64::from(expected[b])).abs())
        })
        .unwrap();
    let output_hash = super::model_replay::float_hash(&actual);
    let worst_values = json!({"index":worst,"actual":actual[worst],"expected":expected[worst]});
    drop(actual);
    let mut samples: BTreeMap<&str, Vec<f32>> = BTreeMap::new();
    for repetition in 0..10 {
        for slot in 0..2 {
            let integer = (repetition + slot) % 2 == 1;
            let start = harness
                .stream
                .record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))
                .map_err(failure)?;
            if integer {
                candidate.launch(&mut output)?;
            } else {
                projection.launch(production, &mut output)?;
            }
            let end = harness
                .stream
                .record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))
                .map_err(failure)?;
            end.synchronize().map_err(failure)?;
            if repetition >= 3 {
                samples
                    .entry(if integer {
                        candidate.label()
                    } else {
                        "production_f32"
                    })
                    .or_default()
                    .push(start.elapsed_ms(&end).map_err(failure)?);
            }
        }
    }
    candidate.read(&output)?;
    println!(
        "RESIDUAL_MATRIX {}",
        json!({
            "tensor":name,"tensor_sha256":tensor_hash,"input_stage":stage,
            "input_sha256":input_hash,"expected_sha256":super::model_replay::float_hash(expected),
            "output_sha256":output_hash,"width":projection.tensor.width,"rows":projection.tensor.rows,
            "kind":projection.tensor.kind,"tokens":projection.tokens,"grouped":projection.grouped,
            "production_kernel":production.kernel,
            "production_tile":[production.rows,production.tokens],
            "assignment_major":projection.assignment_major,"output_values":expected.len(),
            "production_binding_bitwise_equal":true,"all_outputs_finite_written_and_guarded":true,
            "vs_captured":errors,"worst_output":worst_values,"cuda_event_ms":samples,
            "preparation_amortization_branches":if stage=="attn_norm"{3}else if stage=="ffn_norm"{2}else{1},
            "tile":[32,32,32],"threads":128,"static_shared_bytes":shared_bytes,
            "preparation":preparation,
            "layout":"quantized input gathering and packed-weight decoding occur inside the measured tile kernel; no separate scatter",
            "scope":"full intermediate projections, not full-model logits or arithmetic admission",
        })
    );
    Ok(())
}

#[test]
fn residual_quantization_edges_and_integer_bounds() {
    assert!(quantize_cpu(&[]).is_err());
    assert!(quantize_cpu(&[0.0; 31]).is_err());
    assert!(quantize_cpu(&[f32::NAN; 32]).is_err());
    assert!(quantize_cpu(&[f32::INFINITY; 32]).is_err());
    let x: Vec<_> = (0..64)
        .map(|i| {
            if i < 32 {
                if i % 2 == 0 { 0.0 } else { -0.0 }
            } else {
                f32::from_bits((i - 31) as u32)
            }
        })
        .collect();
    let q = quantize_cpu(&x).unwrap();
    assert!(q.q0[..32].iter().all(|&x| x == 0));
    assert_eq!(q.scales[0], 0.0);
    for (i, &x) in x.iter().enumerate() {
        let reconstructed = q.scales[i / 32 * 2] * f32::from(q.q0[i])
            + q.scales[i / 32 * 2 + 1] * f32::from(q.q1[i]);
        assert_eq!(reconstructed, x);
    }
    let fixture: Vec<_> = (0..32)
        .map(|i| {
            if i % 2 == 0 {
                127.0
            } else {
                (i as f32 - 16.0) / 2.0
            }
        })
        .collect();
    let q = quantize_cpu(&fixture).unwrap();
    assert_eq!(q.scales[0], 1.0);
    assert_eq!(q.q0[1], -8);
    assert_eq!(q.q0[3], -6);
    assert!(q.q0.iter().chain(&q.q1).all(|&x| (-127..=127).contains(&x)));
}

#[test]
#[ignore = "bounded residual quantization and affine DP4A CUDA correctness; serial release"]
fn residual_gpu_edges_affine_signed_and_cancellation() -> Result<()> {
    let harness = Harness::new()?;
    for invalid in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let input = vec![invalid; 32];
        let x = harness.stream.clone_htod(&input).map_err(failure)?;
        assert!(Prepared::new(&harness, &input, &x).is_err());
    }
    let extreme = vec![f32::MAX; 32];
    let x = harness.stream.clone_htod(&extreme).map_err(failure)?;
    assert!(quantize_cpu(&extreme).is_err());
    assert!(Prepared::new(&harness, &extreme, &x).is_err());
    for kind in [12, 14] {
        let mut tensor = super::tests::packed_fixture(kind, 256, 8, 1);
        if let Storage::Packed { bytes, .. } = &mut tensor.storage {
            let size = crate::gguf::layout(kind)?.1;
            for (row, block) in bytes.chunks_exact_mut(size).enumerate() {
                let offset = if kind == 14 { 208 } else { 0 };
                block[offset..offset + 2]
                    .copy_from_slice(&half::f16::from_f32(0.01371).to_bits().to_le_bytes());
                if row < 2 {
                    if kind == 12 {
                        block[4..16].fill(255);
                        block[16..].fill(if row == 0 { 255 } else { 0 });
                    } else {
                        block[..192].fill(if row == 0 { 0 } else { 255 });
                        block[192..208].fill(if row == 0 { 128 } else { 127 });
                    }
                }
            }
        }
        let mut input: Vec<_> = (0..5 * 256)
            .map(|i| match i / 256 {
                0 => {
                    if i % 2 == 0 {
                        0.0
                    } else {
                        -0.0
                    }
                }
                1 => f32::from_bits((i % 31 + 1) as u32),
                2 => {
                    if i % 2 == 0 {
                        1.0
                    } else {
                        -1.0 + 1e-5
                    }
                }
                3 => (i as f32 * 0.17).sin() * 3.0,
                _ => {
                    if i % 2 == 0 {
                        127.0
                    } else {
                        (i % 32) as f32 / 2.0 - 8.0
                    }
                }
            })
            .collect();
        input[3 * 256] = 127.0;
        let x = harness.stream.clone_htod(&input).map_err(failure)?;
        let prepared = Prepared::new(&harness, &input, &x)?;
        let mut pairs = Vec::new();
        let mut oracle = Vec::new();
        let mut decoded = vec![0.0; 256];
        for source in 0..5 {
            for row in 0..8 {
                pairs.extend([row as i32, source as i32]);
                tensor.row(row, &mut decoded);
                let reference = decoded
                    .iter()
                    .zip(&input[source * 256..(source + 1) * 256])
                    .map(|(&w, &v)| f64::from(w) * f64::from(v))
                    .sum::<f64>();
                let (affine, magnitude) = affine_oracle(&tensor, &prepared.host, row, source);
                oracle.push((reference, affine, magnitude));
            }
        }
        let tensor = DeviceTensor::upload(tensor, &harness.stream)?;
        let mut dots = Dots {
            harness: &harness,
            tensor: &tensor,
            input: &x,
            prepared: &prepared,
            pairs: harness.stream.clone_htod(&pairs).map_err(failure)?,
            count: 40,
            flag: harness.stream.alloc_zeros::<i32>(1).map_err(failure)?,
        };
        let mut output = harness
            .stream
            .clone_htod(&vec![1337.0; 56])
            .map_err(failure)?;
        for mode in MODES {
            dots.launch(mode, &mut output)?;
            let actual = dots.read(&output)?;
            for (&actual, &(reference, affine, magnitude)) in actual.iter().zip(&oracle) {
                let target = if mode == "residual_int8_dot" {
                    affine
                } else {
                    reference
                };
                assert!(
                    (f64::from(actual) - target).abs()
                        <= 0.001 + 64.0 * f64::from(f32::EPSILON) * magnitude,
                    "{mode} kind={kind}: {actual} != {target}"
                );
            }
            let huge = vec![1e36; 256];
            let x = harness.stream.clone_htod(&huge).map_err(failure)?;
            let prepared = Prepared::new(&harness, &huge, &x)?;
            for mode in MODES {
                let mut dots = Dots {
                    harness: &harness,
                    tensor: &tensor,
                    input: &x,
                    prepared: &prepared,
                    pairs: harness.stream.clone_htod(&[0i32, 0]).map_err(failure)?,
                    count: 1,
                    flag: harness.stream.alloc_zeros::<i32>(1).map_err(failure)?,
                };
                let mut output = harness
                    .stream
                    .clone_htod(&vec![1337.0; 17])
                    .map_err(failure)?;
                dots.launch(mode, &mut output)?;
                assert!(
                    dots.read(&output).is_err(),
                    "{mode} must surface dot overflow"
                );
            }
        }
    }
    Ok(())
}
