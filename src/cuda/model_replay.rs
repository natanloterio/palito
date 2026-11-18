//! Research-only replay of real packed model weights and complete capture operands.
use super::{grouped_bench::RoutingFixture, prefill::Harness, *};
use serde::Deserialize;
use serde_json::json;
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    fs::File,
    io::{BufRead, BufReader, Read},
};

const MAX_CAPTURE_BYTES: u64 = 512 * 1024 * 1024;
const STAGES: [&str; 15] = [
    "attn_norm",
    "Qproj",
    "Kproj",
    "Vproj",
    "Qcur",
    "Kcur",
    "Vcur",
    "attn_attended",
    "attn_output",
    "ffn_norm",
    "ffn_inp",
    "ffn_gate",
    "ffn_up",
    "ffn_down_inp",
    "ffn_down",
];

#[derive(Deserialize)]
#[serde(tag = "kind", rename_all = "lowercase")]
enum Record {
    Metadata {
        version: u32,
        matmul: String,
    },
    Projection {
        layer: usize,
        position: usize,
        stage: String,
        values: Vec<f32>,
    },
    Routing {
        #[serde(flatten)]
        fixture: RoutingFixture,
    },
}

#[derive(Default)]
struct Operands {
    metadata: bool,
    routing: Option<RoutingFixture>,
    projections: BTreeMap<String, BTreeMap<usize, Vec<f32>>>,
}

impl Operands {
    fn push(&mut self, record: Record, layer: usize) -> Result<()> {
        match record {
            Record::Metadata { version, matmul } => {
                if self.metadata || version != 1 || matmul != "f32-tiled" {
                    return Err(invalid(
                        "replay needs one version1 F32Tiled capture metadata record",
                    ));
                }
                self.metadata = true;
            }
            Record::Projection {
                layer: selected,
                position,
                stage,
                values,
            } if selected == layer && STAGES.contains(&stage.as_str()) => {
                if position >= 512 || values.is_empty() || values.iter().any(|v| !v.is_finite()) {
                    return Err(invalid("invalid/nonfinite model replay projection"));
                }
                if self
                    .projections
                    .entry(stage)
                    .or_default()
                    .insert(position, values)
                    .is_some()
                {
                    return Err(invalid("duplicate model replay projection position"));
                }
            }
            Record::Routing { fixture } if fixture.layer == Some(layer) => {
                fixture.validate()?;
                if self.routing.is_some()
                    || fixture.position != Some(0)
                    || !matches!(fixture.tokens, 32 | 46 | 128 | 256 | 512)
                    || fixture.experts != 128
                    || fixture.used != 8
                {
                    return Err(invalid(
                        "replay requires one complete 32/46/128/256/512-token 128-expert/top8 chunk",
                    ));
                }
                self.routing = Some(fixture);
            }
            _ => {}
        }
        Ok(())
    }

    fn load(path: &Path, layer: usize) -> Result<Self> {
        let file = File::open(path)?;
        if file.metadata()?.len() > MAX_CAPTURE_BYTES {
            return Err(invalid("model replay capture exceeds 512MiB file bound"));
        }
        let mut result = Self::default();
        for line in BufReader::new(file).lines() {
            result.push(serde_json::from_str(&line?)?, layer)?;
        }
        if !result.metadata || result.routing.is_none() {
            return Err(invalid(
                "model replay capture lacks metadata or selected routing",
            ));
        }
        Ok(result)
    }

    fn matrix(&self, stage: &str, width: usize, tokens: usize) -> Result<Vec<f32>> {
        let positions = self
            .projections
            .get(stage)
            .ok_or_else(|| invalid(format!("missing model replay stage {stage}")))?;
        if positions.len() != tokens
            || positions.keys().copied().ne(0..tokens)
            || positions.values().any(|row| row.len() != width)
        {
            return Err(invalid(format!(
                "incomplete/wrong-width model replay stage {stage}"
            )));
        }
        Ok(positions.values().flatten().copied().collect())
    }
}

fn file_hash(path: &Path) -> Result<String> {
    let mut reader = BufReader::new(File::open(path)?);
    let mut hash = Sha256::new();
    let mut buffer = vec![0; 1024 * 1024];
    loop {
        let count = reader.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        hash.update(&buffer[..count]);
    }
    Ok(format!("{:x}", hash.finalize()))
}

pub(super) fn float_hash(values: &[f32]) -> String {
    let mut hash = Sha256::new();
    for value in values {
        hash.update(value.to_le_bytes());
    }
    format!("{:x}", hash.finalize())
}

#[derive(Clone, Copy)]
pub(super) struct Mode {
    pub(super) kernel: &'static str,
    pub(super) rows: u32,
    pub(super) tokens: u32,
    threads: u32,
}

pub(super) const DENSE: [Mode; 3] = [
    Mode {
        kernel: "prefill_matmul_ref2x16",
        rows: 2,
        tokens: 16,
        threads: 256,
    },
    Mode {
        kernel: "prefill_matmul_refcompact2x16",
        rows: 2,
        tokens: 16,
        threads: 256,
    },
    Mode {
        kernel: "prefill_matmul_refinterleave2x16",
        rows: 2,
        tokens: 16,
        threads: 256,
    },
];
pub(super) const EXPERT: [Mode; 3] = [
    Mode {
        kernel: "grouped_matmul_unroll8",
        rows: 32,
        tokens: 32,
        threads: 128,
    },
    Mode {
        kernel: "grouped_matmul_coalesced_k32",
        rows: 32,
        tokens: 32,
        threads: 128,
    },
    Mode {
        kernel: "grouped_matmul_unroll4",
        rows: 32,
        tokens: 32,
        threads: 128,
    },
];

pub(super) struct Projection<'a> {
    pub(super) harness: &'a Harness,
    pub(super) tensor: DeviceTensor,
    pub(super) input: CudaSlice<f32>,
    pub(super) groups: &'a CudaSlice<i32>,
    pub(super) small_groups: Option<&'a CudaSlice<i32>>,
    pub(super) tokens: usize,
    pub(super) grouped: bool,
    pub(super) assignment_major: i32,
}

impl Projection<'_> {
    pub(super) fn production_mode(&self) -> Mode {
        if self.tokens >= 256 {
            if self.grouped { EXPERT[0] } else { DENSE[0] }
        } else if self.grouped {
            Mode {
                kernel: "grouped_matmul_tiled",
                rows: 32,
                tokens: 8,
                threads: 256,
            }
        } else {
            Mode {
                kernel: "prefill_matmul",
                rows: 1,
                tokens: 4,
                threads: 256,
            }
        }
    }

    pub(super) fn launch(&self, mode: Mode, output: &mut CudaSlice<f32>) -> Result<()> {
        let groups = if self.grouped && mode.tokens == 8 {
            self.small_groups
                .ok_or_else(|| failure("small-prefix production groups missing"))?
        } else {
            self.groups
        };
        let mut y = output.slice_mut(8..output.len() - 8);
        let mut args = self
            .harness
            .stream
            .launch_builder(&self.harness.functions[mode.kernel]);
        let used = 8i32;
        let tokens = self.tokens as i32;
        args.arg(&self.tensor.bytes)
            .arg(&self.tensor.kind)
            .arg(&self.input)
            .arg(&mut y)
            .arg(&self.tensor.width)
            .arg(&self.tensor.rows);
        if self.grouped {
            args.arg(groups).arg(&used).arg(&self.assignment_major);
        } else {
            args.arg(&tokens);
        }
        let cfg = LaunchConfig {
            grid_dim: (
                (self.tensor.rows as u32).div_ceil(mode.rows),
                if self.grouped {
                    (groups.len() / (mode.tokens as usize + 1)) as u32
                } else {
                    (self.tokens as u32).div_ceil(mode.tokens)
                },
                1,
            ),
            block_dim: (mode.threads, 1, 1),
            shared_mem_bytes: 0,
        };
        // All modes use the existing projection ABI with the recorded block geometry.
        unsafe { args.launch(cfg) }.map_err(failure)?;
        Ok(())
    }
}

pub(super) fn small_groups(
    harness: &Harness,
    routing: &RoutingFixture,
    grouped: bool,
) -> Result<Option<CudaSlice<i32>>> {
    if grouped && routing.tokens < 256 {
        Ok(Some(
            harness
                .stream
                .clone_htod(&routing.descriptors(8))
                .map_err(failure)?,
        ))
    } else {
        Ok(None)
    }
}

pub(super) fn assert_output(actual: &[f32], expected: &[f32], kernel: &str) -> Result<()> {
    if actual.len() != expected.len() + 16
        || actual[..8]
            .iter()
            .chain(&actual[actual.len() - 8..])
            .any(|&v| v != 1337.0)
        || actual[8..actual.len() - 8]
            .iter()
            .zip(expected)
            .any(|(&a, &b)| !a.is_finite() || a.to_bits() != b.to_bits())
    {
        return Err(failure(format!(
            "{kernel}: real-model replay changed captured output bits or guards"
        )));
    }
    Ok(())
}

#[test]
#[ignore = "requires M8_REPLAY_MODEL, M8_REPLAY_MODEL_SHA256, M8_REPLAY_CAPTURE, M8_REPLAY_LAYER; serial release --nocapture"]
fn benchmark_model_projection_replay() -> Result<()> {
    run_model_projection_replay(Replay::Structural)
}

#[test]
#[ignore = "same M8_REPLAY_* corpus as structural replay; arithmetic diagnostics only, serial release"]
fn benchmark_residual_model_primitives() -> Result<()> {
    run_model_projection_replay(Replay::Primitives)
}

#[test]
#[ignore = "same M8_REPLAY_* corpus; full residual matrix versus production, serial release"]
fn benchmark_residual_model_matrices() -> Result<()> {
    run_model_projection_replay(Replay::Matrix)
}

#[test]
#[ignore = "same M8_REPLAY_* corpus; precise3 integer tensor-core matrices, serial SM80+"]
fn benchmark_precise_mma_model_matrices() -> Result<()> {
    run_model_projection_replay(Replay::PreciseMma)
}

#[test]
#[ignore = "same M8_REPLAY_* corpus; bit-exact asynchronous MMA pipeline versus admitted kernel, serial SM80+"]
fn benchmark_precise_pipeline_model_matrices() -> Result<()> {
    run_model_projection_replay(Replay::PrecisePipeline)
}

#[test]
#[ignore = "same development M8_REPLAY_* corpus; block-affine arithmetic screen versus admitted MMA3, serial SM80+"]
fn benchmark_block_affine_model_matrices() -> Result<()> {
    run_model_projection_replay(Replay::BlockAffine)
}

#[test]
#[ignore = "same development M8_REPLAY_* corpus; matched one-component and sparse MMA controls, serial SM80+"]
fn benchmark_matched_projection_controls() -> Result<()> {
    run_model_projection_replay(Replay::MatchedControls)
}

#[test]
#[ignore = "same development M8_REPLAY_* corpus; exact narrow/32-column expert MMA, serial SM80+"]
fn benchmark_narrow_expert_mma3() -> Result<()> {
    run_model_projection_replay(Replay::NarrowExpert)
}

#[test]
#[ignore = "same development M8_REPLAY_* corpus; FP16 operand precision control, serial SM80+"]
fn benchmark_fp16_projection_control() -> Result<()> {
    run_model_projection_replay(Replay::Fp16)
}

#[test]
#[ignore = "same development M8_REPLAY_* corpus; exact MMA3 weight metadata cache, serial SM80+"]
fn benchmark_mma3_cached_scales() -> Result<()> {
    run_model_projection_replay(Replay::CachedScales)
}

#[derive(Clone, Copy)]
enum Replay {
    Structural,
    Primitives,
    Matrix,
    PreciseMma,
    PrecisePipeline,
    BlockAffine,
    MatchedControls,
    NarrowExpert,
    Fp16,
    CachedScales,
    GateUp,
    Compact8,
    SyncK64,
    MixedExpert,
    WideExpertPanel,
    SameExpertPanel,
    PackedDown,
    PackedGateUp,
    PackedKPair,
    WideRows,
    DenseWideRows,
    DenseRegisters,
    DensePlanar,
}

#[test]
#[ignore = "frozen model-backed expert panel; serial CUDA"]
fn benchmark_exact_mma3_gate_up() -> Result<()> {
    run_model_projection_replay(Replay::GateUp)
}

#[test]
#[ignore = "frozen compact expert panel; serial CUDA"]
fn benchmark_mma3_compact8() -> Result<()> {
    run_model_projection_replay(Replay::Compact8)
}

#[test]
#[ignore = "frozen mixed expert schedule validation; serial CUDA"]
fn mixed_expert_schedule_matches_cpu() -> Result<()> {
    let path = PathBuf::from(std::env::var("M8_REPLAY_CAPTURE").map_err(failure)?);
    let layer = std::env::var("M8_REPLAY_LAYER").map_err(failure)?.parse().map_err(failure)?;
    let pin = match layer {
        0 => "fa590dfeea71a36cda673eb6f3e5044d43abea1e64b750fd92ac08d2fdbf5dbe",
        47 => "dd961ab91ccfcc44f1f40fd75d6d798c1bf0605d05e82234d22edd4c5abaaadb",
        _ => return Err(invalid("unexpected mixed schedule replay layer")),
    };
    if file_hash(&path)? != pin {
        return Err(invalid("mixed schedule replay pin mismatch"));
    }
    let routing = Operands::load(&path, layer)?
        .routing
        .ok_or_else(|| invalid("missing mixed schedule routing"))?;
    let descriptors = routing.descriptors(32);
    let slots = routing.slots(32);
    let mut expected = Vec::new();
    let mut short = Vec::new();
    for (i, descriptor) in descriptors.chunks_exact(33).enumerate() {
        if descriptor[0] < 0 {
            continue;
        }
        if descriptor[1..].iter().filter(|&&id| id >= 0).count() > 8 {
            expected.extend([i as i32, -2]);
        } else {
            short.push(i as i32);
        }
    }
    expected.extend(short);
    let live = expected.len().div_ceil(2);
    expected.resize(slots * 2, -1);
    let h = Harness::new()?;
    let groups = h.stream.clone_htod(&descriptors).map_err(failure)?;
    let mut guarded = h.stream.clone_htod(&vec![-777i32; slots * 2 + 16]).map_err(failure)?;
    let mut schedule = guarded.slice_mut(8..8 + slots * 2);
    let slots_arg = slots as i32;
    launch!(h, "group_parallel_mixed_schedule", (1, 1, 1);32,
        &groups, &slots_arg, &mut schedule);
    let actual = h.stream.clone_dtoh(&guarded).map_err(failure)?;
    assert_eq!(&actual[8..8 + slots * 2], &expected);
    assert!(actual[..8].iter().chain(&actual[8 + slots * 2..]).all(|&x| x == -777));
    println!("MIXED_EXPERT_SCHEDULE layer={layer} live_ctas={live} exact=true");
    Ok(())
}

#[test]
#[ignore = "frozen 256-token same-expert schedule validation; serial CUDA"]
fn same_expert_schedule_matches_cpu() -> Result<()> {
    let path = PathBuf::from(std::env::var("M8_REPLAY_CAPTURE").map_err(failure)?);
    let pin = "1c41262c1e3acd17503c89480f69fa83bf12e09e7ca8af71abf48a2ed1ad6d4b";
    if file_hash(&path)? != pin {
        return Err(invalid("same-expert schedule replay pin mismatch"));
    }
    let routing = Operands::load(&path, 0)?
        .routing
        .ok_or_else(|| invalid("missing same-expert routing"))?;
    let slots = routing.slots(32);
    let descriptors = routing.descriptors(32);
    let mut expected = Vec::new();
    let mut pairs = 0;
    let mut tile = 0;
    while tile < slots && descriptors[tile * 33] >= 0 {
        let pair = tile + 1 < slots && descriptors[tile * 33] == descriptors[(tile + 1) * 33];
        expected.extend([tile as i32, if pair { tile as i32 + 1 } else { -2 }]);
        pairs += usize::from(pair);
        tile += 1 + usize::from(pair);
    }
    let live = expected.len() / 2;
    expected.resize(slots * 2, -1);
    let h = Harness::new()?;
    let groups = h.stream.clone_htod(&descriptors).map_err(failure)?;
    let mut guarded = h.stream.clone_htod(&vec![-777i32; slots * 2 + 16]).map_err(failure)?;
    let mut schedule = guarded.slice_mut(8..8 + slots * 2);
    let slots_arg = slots as i32;
    launch!(h, "group_parallel_same_expert_schedule", (1, 1, 1);32,
        &groups, &slots_arg, &mut schedule);
    let actual = h.stream.clone_dtoh(&guarded).map_err(failure)?;
    assert_eq!(&actual[8..8 + slots * 2], &expected);
    assert!(actual[..8].iter().chain(&actual[8 + slots * 2..]).all(|&x| x == -777));
    println!("SAME_EXPERT_SCHEDULE live_ctas={live} pairs={pairs} exact=true");
    Ok(())
}

#[test]
#[ignore = "frozen synchronous K64 expert panel; serial CUDA"]
fn benchmark_mma3_sync_k64() -> Result<()> {
    run_model_projection_replay(Replay::SyncK64)
}

#[test]
#[ignore = "frozen mixed expert MMA3 panel; serial CUDA"]
fn benchmark_mma3_mixed_expert() -> Result<()> {
    run_model_projection_replay(Replay::MixedExpert)
}

#[test]
#[ignore = "frozen 256-token wide expert panel; serial CUDA"]
fn benchmark_mma3_wide_expert_panel() -> Result<()> {
    run_model_projection_replay(Replay::WideExpertPanel)
}

#[test]
#[ignore = "frozen 256-token same-expert panel; serial CUDA"]
fn benchmark_mma3_same_expert_panel() -> Result<()> {
    run_model_projection_replay(Replay::SameExpertPanel)
}

#[test]
#[ignore = "query driver resource attributes from one CUDA module"]
fn mixed_expert_kernel_resources() -> Result<()> {
    let h = Harness::new()?;
    for (name, threads) in [
        ("residual_matmul_mma3", 128),
        ("residual_matmul_mma3_compact8", 64),
        ("residual_matmul_mma3_mixed", 128),
        ("residual_matmul_mma3_same_expert", 128),
    ] {
        let f = &h.functions[name];
        println!("EXPERT_KERNEL_RESOURCES {}", json!({
            "kernel":name,"threads":threads,"registers":f.num_regs().map_err(failure)?,
            "shared_bytes":f.shared_size_bytes().map_err(failure)?,
            "local_bytes":f.local_size_bytes().map_err(failure)?,
            "predicted_blocks_per_sm":f.occupancy_max_active_blocks_per_multiprocessor(threads, 0, None).map_err(failure)?
        }));
    }
    Ok(())
}

#[test]
#[ignore = "frozen decode-shaped expert down replay; serial CUDA"]
fn benchmark_exact_packed_down() -> Result<()> {
    run_model_projection_replay(Replay::PackedDown)
}

#[test]
#[ignore = "frozen decode-shaped expert gate/up replay; serial CUDA"]
fn benchmark_exact_packed_gate_up() -> Result<()> {
    run_model_projection_replay(Replay::PackedGateUp)
}

#[test]
#[ignore = "frozen decode-shaped dense matvec replay; serial CUDA"]
fn benchmark_exact_packed_k_pair() -> Result<()> {
    run_model_projection_replay(Replay::PackedKPair)
}

#[test]
#[ignore = "model-backed 64-row MMA3 projection replay; serial CUDA"]
fn benchmark_mma3_wide_rows() -> Result<()> {
    run_model_projection_replay(Replay::WideRows)
}

#[test]
#[ignore = "model-backed dense 64-row MMA3 projection replay; serial CUDA"]
fn benchmark_mma3_dense_wide_rows() -> Result<()> {
    run_model_projection_replay(Replay::DenseWideRows)
}

#[test]
#[ignore = "frozen short-prefill dense MMA3 projection replay; serial CUDA"]
fn benchmark_mma3_dense_registers() -> Result<()> {
    run_model_projection_replay(Replay::DenseRegisters)
}

#[test]
#[ignore = "frozen short-prefill Q4_K planar projection replay; serial CUDA"]
fn benchmark_mma3_dense_planar() -> Result<()> {
    run_model_projection_replay(Replay::DensePlanar)
}

#[test]
#[ignore = "frozen decode-shaped FFN normalization replay; serial CUDA"]
fn benchmark_retained_normalize() -> Result<()> {
    let path = PathBuf::from(std::env::var("M8_REPLAY_CAPTURE").map_err(failure)?);
    let layer: usize = std::env::var("M8_REPLAY_LAYER").map_err(failure)?.parse().map_err(failure)?;
    let pin = match layer {
        0 => "fa590dfeea71a36cda673eb6f3e5044d43abea1e64b750fd92ac08d2fdbf5dbe",
        47 => "dd961ab91ccfcc44f1f40fd75d6d798c1bf0605d05e82234d22edd4c5abaaadb",
        _ => return Err(invalid("unexpected normalization replay layer")),
    };
    let model_path = PathBuf::from(std::env::var("M8_REPLAY_MODEL").map_err(failure)?);
    if file_hash(&path)? != pin
        || file_hash(&model_path)? != "398bf479282bf374f996f55b0f742c4b45b798e51bf45ca9b7dda0e400d6aa86"
    {
        return Err(invalid("normalization replay pin mismatch"));
    }
    let operands = Operands::load(&path, layer)?;
    let inputs = operands.matrix("ffn_inp", 2048, 46)?;
    let captured = operands.matrix("ffn_norm", 2048, 46)?;
    let mut model = crate::gguf::Gguf::open(&model_path, false)?;
    let weights = model.tensor(&format!("blk.{layer}.ffn_norm.weight"), &[2048])?;
    let epsilon = model.float("qwen3moe.attention.layer_norm_rms_epsilon")?;
    drop(model);
    let h = Harness::with_admitted_attention()?;
    let weights = DeviceTensor::upload(weights, &h.stream)?;
    let width = 2048i32;
    for position in [0, 45] {
        let range = position * 2048..(position + 1) * 2048;
        let x = h.stream.clone_htod(&inputs[range.clone()]).map_err(failure)?;
        let mut outputs = [
            super::residual::matrix_output(&h, 2048)?,
            super::residual::matrix_output(&h, 2048)?,
        ];
        let mut samples = [Vec::new(), Vec::new()];
        for repetition in 0..10 {
            for slot in 0..2 {
                let arm = (repetition + slot) % 2;
                let mut y = outputs[arm].slice_mut(8..2056);
                let start = h.stream.record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT)).map_err(failure)?;
                if arm == 0 {
                    launch!(h, "normalize", (1, 1, 1), &x, &weights.bytes, &mut y, &width, &epsilon);
                } else {
                    launch!(h, "normalize_retained2048", (1, 1, 1), &x, &weights.bytes, &mut y, &epsilon);
                }
                let end = h.stream.record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT)).map_err(failure)?;
                end.synchronize().map_err(failure)?;
                if repetition >= 3 {
                    samples[arm].push(start.elapsed_ms(&end).map_err(failure)?);
                }
            }
        }
        let flag = h.stream.alloc_zeros::<i32>(1).map_err(failure)?;
        let baseline = super::residual::read_matrix_output(&h, &flag, &outputs[0])?;
        let candidate = super::residual::read_matrix_output(&h, &flag, &outputs[1])?;
        assert!(baseline.iter().zip(&candidate).all(|(&a, &b)| a.is_finite() && a.to_bits() == b.to_bits()));
        println!("NORMALIZE_SCREEN {}", json!({"capture_sha256":pin, "layer":layer, "position":position,
            "output_sha256":float_hash(&baseline), "captured_sha256":float_hash(&captured[range]), "exact":true,
            "baseline_ms":samples[0], "retained_ms":samples[1], "repetitions":7, "warmups":3}));
    }
    Ok(())
}

#[test]
#[ignore = "frozen 512-token GPU attention replay; serial CUDA"]
fn benchmark_temporal_attention_pair() -> Result<()> {
    use super::residual::matrix_output;
    let path = PathBuf::from(std::env::var("M8_REPLAY_CAPTURE").map_err(failure)?);
    let layer: usize = std::env::var("M8_REPLAY_LAYER")
        .map_err(failure)?
        .parse()
        .map_err(failure)?;
    let pin = match layer {
        0 => "75acbe42d4d774ec1296127c3d58c56db4d72662adb387af9dceb73e84915d64",
        37 => "7686003f8d04631bb4eb79b87d65c5baed8a1c8422348d1ba7891d42a35a0840",
        _ => return Err(invalid("unexpected temporal attention layer")),
    };
    if file_hash(&path)? != pin {
        return Err(invalid("temporal attention capture pin mismatch"));
    }
    let operands = Operands::load(&path, layer)?;
    let tokens = 512;
    let harness = Harness::with_admitted_attention()?;
    let q = harness
        .stream
        .clone_htod(&operands.matrix("Qcur", 4096, tokens)?)
        .map_err(failure)?;
    let k = harness
        .stream
        .clone_htod(&operands.matrix("Kcur", 512, tokens)?)
        .map_err(failure)?;
    let v = harness
        .stream
        .clone_htod(&operands.matrix("Vcur", 512, tokens)?)
        .map_err(failure)?;
    let expected = operands.matrix("attn_attended", 4096, tokens)?;
    let mut outputs = [
        matrix_output(&harness, expected.len())?,
        matrix_output(&harness, expected.len())?,
    ];
    let run = |arm: usize, output: &mut CudaSlice<f32>| -> Result<()> {
        let mut y = output.slice_mut(8..8 + expected.len());
        if arm == 0 {
            launch!(
                harness,
                "attention_fused",
                (32, 512, 1),
                &q,
                &k,
                &v,
                &mut y,
                &128i32,
                &32i32,
                &4i32,
                &1i32
            );
        } else {
            launch!(
                harness,
                "attention_fused_query_pair",
                (32, 256, 1),
                &q,
                &k,
                &v,
                &mut y,
                &32i32,
                &4i32,
                &1i32
            );
        }
        Ok(())
    };
    for (arm, output) in outputs.iter_mut().enumerate() {
        run(arm, output)?;
        assert_output(
            &harness.stream.clone_dtoh(output).map_err(failure)?,
            &expected,
            "temporal attention",
        )?;
    }
    let mut samples = [Vec::new(), Vec::new()];
    for repetition in 0..10 {
        for slot in 0..2 {
            let arm = (repetition + slot) % 2;
            let start = harness
                .stream
                .record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))
                .map_err(failure)?;
            run(arm, &mut outputs[arm])?;
            let end = harness
                .stream
                .record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))
                .map_err(failure)?;
            end.synchronize().map_err(failure)?;
            if repetition >= 3 {
                samples[arm].push(start.elapsed_ms(&end).map_err(failure)?);
            }
        }
    }
    println!(
        "TEMPORAL_ATTENTION {}",
        json!({"layer":layer,"capture_sha256":pin,"output_sha256":float_hash(&expected),
        "baseline_ms":samples[0],"pair_ms":samples[1],"all_outputs_exact":true,"warmups":3,"runs":7})
    );
    Ok(())
}

fn run_model_projection_replay(mode: Replay) -> Result<()> {
    let arithmetic = !matches!(mode, Replay::Structural);
    let required = |name| std::env::var(name).map_err(|e| invalid(format!("{name}: {e}")));
    let model_path = PathBuf::from(required("M8_REPLAY_MODEL")?);
    let capture_path = PathBuf::from(required("M8_REPLAY_CAPTURE")?);
    let layer: usize = required("M8_REPLAY_LAYER")?.parse().map_err(failure)?;
    let expected_hash = required("M8_REPLAY_MODEL_SHA256")?;
    if File::open(&model_path)?.metadata()?.len() > 24 * 1024 * 1024 * 1024 {
        return Err(invalid("research replay model exceeds 24GiB file bound"));
    }
    let model_hash = file_hash(&model_path)?;
    if model_hash != expected_hash {
        return Err(invalid("model replay GGUF SHA256 mismatch"));
    }
    let operands = Operands::load(&capture_path, layer)?;
    let capture_hash = file_hash(&capture_path)?;
    let routing = operands.routing.as_ref().unwrap();
    let tokens = routing.tokens;
    let plans = [
        ("attn_q", "attn_norm", "Qproj", 2048, 4096, false, 0),
        ("attn_k", "attn_norm", "Kproj", 2048, 512, false, 0),
        ("attn_v", "attn_norm", "Vproj", 2048, 512, false, 0),
        (
            "attn_output",
            "attn_attended",
            "attn_output",
            4096,
            2048,
            false,
            0,
        ),
        ("ffn_gate_exps", "ffn_norm", "ffn_gate", 2048, 768, true, 0),
        ("ffn_up_exps", "ffn_norm", "ffn_up", 2048, 768, true, 0),
        (
            "ffn_down_exps",
            "ffn_down_inp",
            "ffn_down",
            768,
            2048,
            true,
            1,
        ),
    ];
    // Reuse the validated GGUF parser; release unselected host tensors before GPU work.
    let mut model = crate::gguf::Gguf::open(&model_path, false)?;
    let mut tensors = Vec::new();
    for &(name, input, output, width, rows, grouped, assignment_major) in &plans {
        let name = format!("blk.{layer}.{name}.weight");
        let dims = if grouped {
            vec![width, rows, 128]
        } else {
            vec![width, rows]
        };
        let tensor = model.tensor(&name, &dims)?;
        if !matches!(&tensor.storage, Storage::Packed { kind: 12 | 14, .. }) {
            return Err(invalid(format!(
                "model replay requires packed Q4_K/Q6_K: {name}"
            )));
        }
        operands.matrix(
            input,
            width * if assignment_major == 1 { 8 } else { 1 },
            tokens,
        )?;
        operands.matrix(output, rows * if grouped { 8 } else { 1 }, tokens)?;
        tensors.push(tensor);
    }
    drop(model);
    let harness = if matches!(mode, Replay::PackedDown | Replay::PackedGateUp | Replay::PackedKPair) {
        Harness::with_packed_reduction()?
    } else {
        Harness::new()?
    };
    if matches!(mode, Replay::PackedKPair) {
        let wanted = match (layer, tokens) {
            (0, 46) => "fa590dfeea71a36cda673eb6f3e5044d43abea1e64b750fd92ac08d2fdbf5dbe",
            (47, 46) => "dd961ab91ccfcc44f1f40fd75d6d798c1bf0605d05e82234d22edd4c5abaaadb",
            _ => return Err(invalid("unexpected packed K-pair replay fixture")),
        };
        if capture_hash != wanted
            || model_hash != "398bf479282bf374f996f55b0f742c4b45b798e51bf45ca9b7dda0e400d6aa86"
        {
            return Err(invalid("packed K-pair replay pin mismatch"));
        }
        for (index, tensor) in tensors.into_iter().enumerate() {
            if !matches!(index, 0 | 3) {
                continue;
            }
            let (name, stage, _, width, _, _, _) = plans[index];
            let tensor = DeviceTensor::upload(tensor, &harness.stream)?;
            let input = operands.matrix(stage, width, tokens)?;
            for position in [0, tokens - 1] {
                super::packed_decode::benchmark_k_pair(&harness, &tensor,
                    &input[position * width..(position + 1) * width], name, position, &capture_hash)?;
            }
        }
        return Ok(());
    }
    if matches!(mode, Replay::PackedGateUp) {
        let wanted = match (layer, tokens) {
            (0, 46) => "fa590dfeea71a36cda673eb6f3e5044d43abea1e64b750fd92ac08d2fdbf5dbe",
            (47, 46) => "dd961ab91ccfcc44f1f40fd75d6d798c1bf0605d05e82234d22edd4c5abaaadb",
            _ => return Err(invalid("unexpected packed gate/up replay fixture")),
        };
        if capture_hash != wanted
            || model_hash != "398bf479282bf374f996f55b0f742c4b45b798e51bf45ca9b7dda0e400d6aa86"
        {
            return Err(invalid("packed gate/up replay pin mismatch"));
        }
        let mut experts = tensors.into_iter().skip(4);
        let gate = DeviceTensor::upload(experts.next().unwrap(), &harness.stream)?;
        let up = DeviceTensor::upload(experts.next().unwrap(), &harness.stream)?;
        let input = operands.matrix("ffn_norm", 2048, tokens)?;
        for position in [0, tokens - 1] {
            super::packed_decode::benchmark_gate_up(&harness, &gate, &up,
                &input[position * 2048..(position + 1) * 2048],
                &routing.ids[position * 8..(position + 1) * 8], position, &capture_hash)?;
        }
        return Ok(());
    }
    if matches!(mode, Replay::PackedDown) {
        let wanted = match (layer, tokens) {
            (0, 46) => "fa590dfeea71a36cda673eb6f3e5044d43abea1e64b750fd92ac08d2fdbf5dbe",
            (37, 32) => "80e9b95e7ab172bd9348676bcc92000fd5b55138af6d7299324f93301f5b67d1",
            _ => return Err(invalid("unexpected packed down replay fixture")),
        };
        if capture_hash != wanted
            || model_hash != "398bf479282bf374f996f55b0f742c4b45b798e51bf45ca9b7dda0e400d6aa86"
        {
            return Err(invalid("packed down replay pin mismatch"));
        }
        let down = DeviceTensor::upload(tensors.pop().unwrap(), &harness.stream)?;
        let input = operands.matrix("ffn_down_inp", 8 * 768, tokens)?;
        for token in [0, tokens - 1] {
            super::packed_decode::benchmark_down(
                &harness,
                &down,
                &input[token * 8 * 768..(token + 1) * 8 * 768],
                &routing.ids[token * 8..(token + 1) * 8],
                token,
                &capture_hash,
            )?;
        }
        return Ok(());
    }
    if matches!(mode, Replay::GateUp | Replay::Compact8 | Replay::SyncK64 | Replay::MixedExpert | Replay::WideExpertPanel | Replay::SameExpertPanel) {
        return super::residual_precise::benchmark_gate_up(
            &harness,
            tensors.split_off(4),
            routing,
            &operands.matrix("ffn_norm", 2048, tokens)?,
            &operands.matrix("ffn_down_inp", 8 * 768, tokens)?,
            &capture_hash,
            &model_hash,
            matches!(mode, Replay::Compact8),
            matches!(mode, Replay::SyncK64),
            matches!(mode, Replay::MixedExpert),
            matches!(mode, Replay::WideExpertPanel),
            matches!(mode, Replay::SameExpertPanel),
        );
    }
    let ids = harness.stream.clone_htod(&routing.ids).map_err(failure)?;
    let descriptor_count = routing.slots(32);
    let token_count = tokens as i32;
    let slots = descriptor_count as i32;
    let mut groups = harness
        .stream
        .alloc_zeros::<i32>(descriptor_count * 33)
        .map_err(failure)?;
    let mut grouping_ms = Vec::new();
    for repetition in 0..10 {
        let start = harness
            .stream
            .record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))
            .map_err(failure)?;
        launch!(
            harness,
            "group_experts_reg32",
            (1, 1, 1),
            &ids,
            &mut groups,
            &128i32,
            &8i32,
            &token_count,
            &slots
        );
        let end = harness
            .stream
            .record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))
            .map_err(failure)?;
        end.synchronize().map_err(failure)?;
        if repetition >= 3 {
            grouping_ms.push(start.elapsed_ms(&end).map_err(failure)?);
        }
    }
    if harness.stream.clone_dtoh(&groups).map_err(failure)? != routing.descriptors(32) {
        return Err(failure(
            "model replay GPU routing descriptors disagree with captured IDs",
        ));
    }
    let production_grouping_ms = if tokens < 256 {
        let slots = routing.slots(8) as i32;
        let mut legacy = harness
            .stream
            .alloc_zeros::<i32>(slots as usize * 9)
            .map_err(failure)?;
        let mut samples = Vec::new();
        for repetition in 0..10 {
            let start = harness
                .stream
                .record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))
                .map_err(failure)?;
            launch!(
                harness,
                "group_experts_tiled",
                (1, 1, 1),
                &ids,
                &mut legacy,
                &128i32,
                &8i32,
                &token_count,
                &slots
            );
            let end = harness
                .stream
                .record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))
                .map_err(failure)?;
            end.synchronize().map_err(failure)?;
            if repetition >= 3 {
                samples.push(start.elapsed_ms(&end).map_err(failure)?);
            }
        }
        assert_eq!(
            harness.stream.clone_dtoh(&legacy).map_err(failure)?,
            routing.descriptors(8)
        );
        samples
    } else {
        grouping_ms.clone()
    };
    let source_hash = format!(
        "{:x}",
        Sha256::digest(concat!(
            include_str!("kernels.cu"),
            include_str!("prefill.cu"),
            include_str!("attention.cu"),
            include_str!("prefill.rs"),
            include_str!("model_replay.rs"),
            include_str!("residual.rs"),
            include_str!("residual.cu"),
            include_str!("residual_tile.cu"),
            include_str!("residual_precise.rs"),
            include_str!("residual_precise.cu"),
            include_str!("residual_mma.cu"),
            include_str!("residual_pipeline.cu"),
            include_str!("residual_block.rs"),
            include_str!("residual_block.cu"),
            include_str!("fp16_projection.cu")
        ))
    );
    println!(
        "{} {}",
        match mode {
            Replay::Structural => "MODEL_REPLAY_PROVENANCE",
            Replay::Primitives => "RESIDUAL_PROVENANCE",
            Replay::Matrix => "RESIDUAL_MATRIX_PROVENANCE",
            Replay::PreciseMma => "PRECISE_MMA_PROVENANCE",
            Replay::PrecisePipeline => "PRECISE_PIPELINE_PROVENANCE",
            Replay::BlockAffine => "BLOCK_PROVENANCE",
            Replay::MatchedControls => "MATCHED_PROJECTION_PROVENANCE",
            Replay::NarrowExpert => "NARROW_EXPERT_PROVENANCE",
            Replay::Fp16 => "FP16_PROVENANCE",
            Replay::CachedScales => "CACHED_SCALES_PROVENANCE",
            Replay::WideRows | Replay::DenseWideRows => "WIDE_ROWS_PROVENANCE",
            Replay::DenseRegisters => "REGISTERS_PROVENANCE",
            Replay::DensePlanar => "PLANAR_PROVENANCE",
            Replay::GateUp | Replay::Compact8 | Replay::SyncK64 | Replay::MixedExpert | Replay::WideExpertPanel | Replay::SameExpertPanel | Replay::PackedDown | Replay::PackedGateUp | Replay::PackedKPair => unreachable!(),
        },
        json!({
            "version":1, "model":model_path, "model_sha256":model_hash,
            "capture":capture_path, "capture_sha256":capture_hash, "layer":layer, "tokens":tokens,
            "source_sha256":source_hash, "cargo_lock_sha256":format!("{:x}", Sha256::digest(include_str!("../../Cargo.lock"))),
            "grouping_cuda_event_ms":grouping_ms, "routing_counts":routing.validate()?,
            "production_grouping_cuda_event_ms":production_grouping_ms,
            "production_group_tile":if tokens<256{8}else{32},
            "candidate_group_tile":32,
            "operands":"actual packed GGUF weights, captured F32 activations, outputs and routing",
            "timing_excludes":"model loading, host hashing, JSON parsing, copies, correctness checks; grouping reported separately",
            "quality_claim":"structural replay self-consistency only, not arithmetic or full-model acceptance",
            "activation_precision":"f32", "warmups":3, "repetitions":7,
            "arithmetic_research":arithmetic,
        })
    );
    let mut experiment = if matches!(mode, Replay::Matrix) {
        super::residual::Experiment::new_tiled(&harness)
    } else {
        super::residual::Experiment::new(&harness)
    };
    let mut precise = match mode {
        Replay::BlockAffine => super::residual_precise::Experiment::new_block(&harness),
        Replay::MatchedControls => super::residual_precise::Experiment::new_control(&harness),
        Replay::NarrowExpert => super::residual_precise::Experiment::new_narrow(&harness),
        Replay::Fp16 => super::residual_precise::Experiment::new_fp16(&harness),
        Replay::CachedScales => super::residual_precise::Experiment::new_cached_scales(&harness),
        Replay::DenseRegisters => super::residual_precise::Experiment::new_registers(&harness),
        Replay::DensePlanar => super::residual_precise::Experiment::new_planar(&harness),
        Replay::WideRows | Replay::DenseWideRows => {
            super::residual_precise::Experiment::new_wide_rows(&harness)
        }
        _ => super::residual_precise::Experiment::new(
            &harness,
            matches!(mode, Replay::PrecisePipeline),
        ),
    };
    for ((name, input_stage, output_stage, width, rows, grouped, assignment_major), tensor) in
        plans.into_iter().zip(tensors)
    {
        if (matches!(mode, Replay::NarrowExpert | Replay::WideRows) && !grouped)
            || (matches!(mode, Replay::DenseWideRows | Replay::DenseRegisters) && grouped)
            || (matches!(mode, Replay::DensePlanar)
                && (grouped || !matches!(&tensor.storage, Storage::Packed { kind: 12, .. })))
        {
            continue;
        }
        let input = operands.matrix(
            input_stage,
            width * if assignment_major == 1 { 8 } else { 1 },
            tokens,
        )?;
        let expected = operands.matrix(output_stage, rows * if grouped { 8 } else { 1 }, tokens)?;
        let Storage::Packed { bytes, .. } = &tensor.storage else {
            unreachable!()
        };
        let tensor_hash = format!("{:x}", Sha256::digest(bytes));
        if arithmetic {
            let tensor_name = format!("blk.{layer}.{name}.weight");
            let probe = super::residual::Probe {
                tensor,
                input: &input,
                expected: &expected,
                stage: input_stage,
                name: &tensor_name,
                routing,
                grouped,
                assignment_major,
                groups: &groups,
                tensor_hash: &tensor_hash,
            };
            if matches!(
                mode,
                Replay::PreciseMma
                    | Replay::PrecisePipeline
                    | Replay::BlockAffine
                    | Replay::MatchedControls
                    | Replay::NarrowExpert
                    | Replay::Fp16
                    | Replay::CachedScales
                    | Replay::WideRows
                    | Replay::DenseWideRows
                    | Replay::DenseRegisters
                    | Replay::DensePlanar
            ) {
                precise.probe(probe)?;
            } else {
                experiment.probe(probe)?;
            }
            continue;
        }
        let projection = Projection {
            harness: &harness,
            tensor: DeviceTensor::upload(tensor, &harness.stream)?,
            input: harness.stream.clone_htod(&input).map_err(failure)?,
            groups: &groups,
            small_groups: None,
            tokens,
            grouped,
            assignment_major,
        };
        let modes = if grouped { &EXPERT } else { &DENSE };
        let mut output = harness
            .stream
            .clone_htod(&vec![1337.0; expected.len() + 16])
            .map_err(failure)?;
        for &mode in modes {
            harness
                .stream
                .memcpy_htod(&vec![1337.0; expected.len() + 16], &mut output)
                .map_err(failure)?;
            projection.launch(mode, &mut output)?;
            let actual = harness.stream.clone_dtoh(&output).map_err(failure)?;
            assert_output(&actual, &expected, mode.kernel)?;
        }
        for warmup in 0..3 {
            for slot in 0..modes.len() {
                projection.launch(modes[(slot + warmup) % modes.len()], &mut output)?;
            }
        }
        let mut samples: BTreeMap<&str, Vec<f32>> = BTreeMap::new();
        for repetition in 0..7 {
            for slot in 0..modes.len() {
                let mode = modes[(slot + repetition) % modes.len()];
                let start = harness
                    .stream
                    .record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))
                    .map_err(failure)?;
                projection.launch(mode, &mut output)?;
                let end = harness
                    .stream
                    .record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))
                    .map_err(failure)?;
                end.synchronize().map_err(failure)?;
                samples
                    .entry(mode.kernel)
                    .or_default()
                    .push(start.elapsed_ms(&end).map_err(failure)?);
            }
        }
        println!(
            "MODEL_REPLAY {}",
            json!({
                "tensor":format!("blk.{layer}.{name}.weight"), "tensor_sha256":tensor_hash,
                "input_sha256":float_hash(&input), "captured_output_sha256":float_hash(&expected),
                "kind":projection.tensor.kind, "width":width, "rows":rows, "tokens":tokens,
                "grouped":grouped, "assignment_major":assignment_major,
                "all_outputs_bitwise_equal":true, "output_values":expected.len(),
                "baseline_kernel":modes[0].kernel, "cuda_event_ms":samples,
            })
        );
    }
    Ok(())
}

#[test]
fn replay_rejects_bad_routing_and_multiple_chunks() {
    let routing = |position, tokens| {
        serde_json::from_value::<Record>(json!({
            "kind":"routing", "version":1, "layer":0, "position":position,
            "tokens":tokens, "experts":128, "used":8,
            "ids":(0..tokens * 8).map(|i| i % 8).collect::<Vec<_>>(),
        }))
        .unwrap()
    };
    let mut operands = Operands::default();
    assert!(operands.push(routing(1, 256), 0).is_err());
    assert!(operands.push(routing(0, 257), 0).is_err());
    operands.push(routing(0, 256), 0).unwrap();
    assert!(operands.push(routing(0, 256), 0).is_err());
    for tokens in [32, 46, 128] {
        let mut operands = Operands::default();
        operands.push(routing(0, tokens), 0).unwrap();
        assert_eq!(operands.routing.unwrap().tokens, tokens);
    }
}

#[test]
fn replay_rejects_incomplete_duplicate_and_nonfinite_operands() {
    let mut operands = Operands::default();
    let projection = |position, values| Record::Projection {
        layer: 0,
        position,
        stage: "attn_norm".into(),
        values,
    };
    operands.push(projection(1, vec![1.0, 2.0]), 0).unwrap();
    assert!(operands.matrix("attn_norm", 2, 2).is_err());
    operands.push(projection(0, vec![3.0, 4.0]), 0).unwrap();
    assert_eq!(
        operands.matrix("attn_norm", 2, 2).unwrap(),
        [3.0, 4.0, 1.0, 2.0]
    );
    assert!(operands.matrix("attn_norm", 1, 2).is_err());
    assert!(operands.push(projection(0, vec![1.0]), 0).is_err());
    assert!(operands.push(projection(2, vec![f32::NAN]), 0).is_err());
    assert!(operands.push(projection(512, vec![1.0]), 0).is_err());
    assert!(
        operands
            .push(
                Record::Metadata {
                    version: 1,
                    matmul: "f32".into()
                },
                0
            )
            .is_err()
    );
    let record = || Record::Metadata {
        version: 1,
        matmul: "f32-tiled".into(),
    };
    operands.push(record(), 0).unwrap();
    assert!(operands.push(record(), 0).is_err());
    let mut actual = vec![1337.0; 18];
    actual[8..10].copy_from_slice(&[0.0, 1.0]);
    assert_output(&actual, &[0.0, 1.0], "test").unwrap();
    assert!(assert_output(&actual, &[-0.0, 1.0], "test").is_err());
    actual[0] = 0.0;
    assert!(assert_output(&actual, &[0.0, 1.0], "test").is_err());
}
