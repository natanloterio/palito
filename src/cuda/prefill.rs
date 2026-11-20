//! Correctness and isolated performance harness for packed prompt projections.
use super::*;
use crate::cuda::tests::packed_fixture;
use serde_json::json;
use std::time::Instant;

const DECODE_CHECK_SOURCE: &str = r#"
extern "C" __global__ void half_decode_check(const u8 *input, unsigned int *bits) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < 65536) bits[i] = __float_as_uint(half_value(input + i * 2));
}
template<int Kind, int KTile>
__device__ void check_projection_decode(const u8 *w, int *status, int width, int rows,
                                        float *workspace) {
    float (*decoded)[KTile + 1] = reinterpret_cast<float (*)[KTile + 1]>(workspace);
    int first_row = blockIdx.x * 8, first_k = blockIdx.y * KTile;
    projection_weights<Kind, 8, KTile>(w, decoded, width, rows, 0, first_row, first_k);
    __syncthreads();
    for (int i = threadIdx.x; i < 8 * KTile; i += 256) {
        int r = i / KTile, k = i % KTile;
        if (first_row + r < rows && first_k + k < width) {
            float expected = weight(w, Kind, (u64)(first_row + r) * width + first_k + k);
            if (__float_as_uint(expected) != __float_as_uint(decoded[r][k]))
                atomicExch(status, 1);
        }
    }
}
template<int Kind>
__device__ void check_projection_decode_tile(const u8 *w, int *status, int width, int rows,
                                             int k_tile, float *workspace) {
    if (k_tile == 32) check_projection_decode<Kind, 32>(w, status, width, rows, workspace);
    else if (k_tile == 64) check_projection_decode<Kind, 64>(w, status, width, rows, workspace);
    else check_projection_decode<Kind, 256>(w, status, width, rows, workspace);
}
extern "C" __global__ void projection_decode_check(const u8 *w, int kind, int *status,
                                                   int width, int rows, int k_tile) {
    __shared__ float workspace[8 * 257];
    switch (kind) {
        case 0: check_projection_decode_tile<0>(w, status, width, rows, k_tile, workspace); break;
        case 1: check_projection_decode_tile<1>(w, status, width, rows, k_tile, workspace); break;
        case 6: check_projection_decode_tile<6>(w, status, width, rows, k_tile, workspace); break;
        case 8: check_projection_decode_tile<8>(w, status, width, rows, k_tile, workspace); break;
        case 12: check_projection_decode_tile<12>(w, status, width, rows, k_tile, workspace); break;
        case 14: check_projection_decode_tile<14>(w, status, width, rows, k_tile, workspace); break;
    }
}
"#;

pub(super) struct Harness {
    pub(super) stream: Arc<CudaStream>,
    pub(super) functions: HashMap<&'static str, CudaFunction>,
    ids: CudaSlice<i32>,
}

#[derive(Clone, Copy, Debug)]
enum Mode {
    Sequential,
    Independent,
    Tiled,
    WideTiled,
    Rows2,
    Staged,
    Tf32x3,
    StagedTf32x3,
    Register64x16,
    Register32x32,
    Register64x32,
    Reference2x8,
    Reference2x16,
    Reference4x16,
    Reuse2x16,
    Reuse4x8,
    Reuse4x16,
    Reuse4x8Rows4,
    ReferencePacked2x16,
    ReferenceUnroll2,
    ReferenceUnroll4,
    ReferenceUnroll8,
    ReferenceUnroll16,
    ReferenceCompact2x16,
    ReferenceCompact2x8,
    Reference64Lanes2x8,
    Reference64Lanes2x16,
    Reference32Lanes4x4,
    Reference32Lanes4x8,
    ReferenceStatic,
    ReferenceInterleaved8,
    ReferenceInterleaved16,
}

const REGISTER_MODES: &[Mode] = &[
    Mode::Staged,
    Mode::Register64x16,
    Mode::Register32x32,
    Mode::Register64x32,
];

impl Harness {
    pub(super) fn new() -> Result<Self> {
        Self::with_normalized_attention(false)
    }

    fn with_normalized_attention(normalize: bool) -> Result<Self> {
        Self::with_attention_options(normalize, false)
    }

    fn with_attention_options(normalize: bool, reuse: bool) -> Result<Self> {
        Self::compile(normalize, reuse, false, false)
    }

    pub(super) fn with_packed_reduction() -> Result<Self> {
        Self::compile(false, false, true, false)
    }

    pub(super) fn with_admitted_attention() -> Result<Self> {
        Self::compile(true, true, false, true)
    }

    pub(super) fn with_admitted_decode() -> Result<Self> {
        Self::compile(true, true, true, true)
    }

    fn compile(
        normalize: bool,
        reuse: bool,
        packed_reduction: bool,
        block_reduction: bool,
    ) -> Result<Self> {
        let context = CudaContext::new(0).map_err(failure)?;
        let major = context
            .attribute(sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)
            .map_err(failure)?;
        let minor = context
            .attribute(sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)
            .map_err(failure)?;
        let source = format!(
            "{}\n{}\n{}\n{}\n#define CUDA_M8_PROTOTYPES 1\n#define CUDA_NATIVE_HALF 1\n#define CUDA_DENSE_PACKED_REPLAY 1\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}",
            if block_reduction {
                "#define CUDA_BLOCK_REDUCTION_RESEARCH 1"
            } else {
                ""
            },
            if normalize {
                "#define CUDA_NORMALIZED_ATTENTION 1"
            } else {
                ""
            },
            if reuse {
                "#define CUDA_ATTENTION_REUSE_RESEARCH 1"
            } else {
                ""
            },
            if packed_reduction {
                "#define CUDA_PACKED_REDUCTION_RESEARCH 1"
            } else {
                ""
            },
            include_str!("kernels.cu"),
            include_str!("prefill.cu"),
            include_str!("attention.cu"),
            include_str!("tiles8.cu"),
            include_str!("rows2.cu"),
            include_str!("tensorcore.cu"),
            include_str!("residual.cu"),
            include_str!("residual_tile.cu"),
            include_str!("residual_precise.cu"),
            include_str!("residual_mma.cu"),
            include_str!("residual_pipeline.cu"),
            include_str!("residual_block.cu"),
            include_str!("grouping_parallel.cu"),
            include_str!("fp16_projection.cu"),
            DECODE_CHECK_SOURCE,
        );
        let source = if std::env::var_os("LI_REPLAY_Q8_ALIGNED").is_some() {
            format!(
                "{source}\n#define CUDA_LLAMA_Q8_ALIGNED_REPLAY 1\n{}",
                include_str!("q8_decode.cu")
            )
        } else {
            source
        };
        let ptx = compile_ptx_with_opts(
            source,
            CompileOptions {
                fmad: Some(false),
                options: vec![format!("--gpu-architecture=compute_{major}{minor}")],
                ..Default::default()
            },
        )
        .map_err(failure)?;
        let module = context.load_module(ptx).map_err(failure)?;
        let functions = [
            "matvec",
            "matvec_packed_f32",
            "matvec_packed_down_two_row",
            "matvec_packed_down_warp",
            "prefill_matmul",
            "dense_packed_replay_mma1",
            "dense_packed_replay_mma3",
            "dense_q8_replay_mma3_rows64",
            "dense_q8_replay_mma3_wide_warp",
            "group_experts",
            "grouped_matmul",
            "prefill_matmul8",
            "attention_fused",
            "attention_fused_query_pair",
            "attention_scores",
            "attention_values",
            "prefill_matmul_rows2",
            "grouped_matmul_rows2",
            "prefill_matmul_tf32",
            "grouped_matmul_tf32",
            "prefill_matmul_tf32x3",
            "prefill_matmul_tf32x4_control",
            "grouped_matmul_tf32x3",
            "group_experts_tiled",
            "prefill_matmul_tiled",
            "grouped_matmul_tiled",
            "prefill_matmul_tf32x3_tiled",
            "grouped_matmul_tf32x3_tiled",
            "projection_decode_check",
            "half_decode_check",
            "residual_quantize32",
            "residual_quantize32_precise3",
            "residual_quantize32_precise1",
            "residual_packed_quant_check",
            "residual_matmul_mma3",
            "residual_matmul_mma3_gate_up_q4",
            "residual_matmul_mma3_rows64",
            "residual_matmul_mma3_compact8",
            "matvec_packed_gate_up",
            "matvec_packed_k_pair",
            "residual_matmul_mma3_registers",
            "residual_matmul_mma3_planar_q4",
            "residual_matmul_mma3_sync_k64",
            "residual_matmul_mma3_mixed",
            "residual_matmul_mma3_same_expert",
            "residual_matmul_mma3_same_expert_warp8",
            "residual_matmul_mma3_splitk",
            "residual_splitk_reduce",
            "residual_matmul_mma1",
            "residual_matmul_mma3_skip_empty",
            "residual_matmul_mma3_cached_scales",
            "projection_fp16_control",
            "projection_fp16_packed",
            "residual_matmul_mma3_cols8",
            "residual_matmul_mma3_wide8",
            "residual_matmul_mma3_cols16",
            "residual_matmul_mma3_wide16",
            "residual_matmul_mma3_pipeline",
            "residual_matmul_mma3_true_pipeline",
            "residual_quantize256_precise3",
            "residual_matmul_mma3_block",
            "residual_int8_dot",
            "residual_f32_dot",
            "residual_tf32x4_dot",
            "residual_matmul32",
            "prefill_matmul_reg64x16",
            "prefill_matmul_reg32x32",
            "prefill_matmul_reg64x32",
            "group_experts_reg32",
            "group_parallel_count32",
            "group_parallel_emit32",
            "group_parallel_emit8",
            "group_parallel_mixed_schedule",
            "group_parallel_same_expert_schedule",
            "grouped_matmul_reg32x32",
            "prefill_matmul_ref2x8",
            "prefill_matmul_ref2x16",
            "prefill_matmul_ref4x16",
            "prefill_matmul_reuse2x16",
            "prefill_matmul_reuse4x8",
            "prefill_matmul_reuse4x16",
            "prefill_matmul_reuse4x8_r4",
            "prefill_matmul_refpacked2x16",
            "prefill_matmul_ref2x16_u2",
            "prefill_matmul_ref2x16_u4",
            "prefill_matmul_ref2x16_u8",
            "prefill_matmul_ref2x16_u16",
            "prefill_matmul_refcompact2x16",
            "prefill_matmul_refcompact2x8",
            "prefill_matmul_ref64_2x8",
            "prefill_matmul_ref64_2x16",
            "prefill_matmul_ref32_4x4",
            "prefill_matmul_ref32_4x8",
            "prefill_matmul_ref2048",
            "prefill_matmul_ref4096",
            "prefill_matmul_refinterleave2x16",
            "prefill_matmul_refinterleave2x8",
            "grouped_matmul_reg32_counted",
            "grouped_gate_up_reg32x32",
            "grouped_matmul_coalesced_k32",
            "grouped_matmul_coalesced_k64",
            "grouped_matmul_coalesced_k128",
            "grouped_matmul_balanced32",
            "grouped_matmul_wide64",
            "grouped_matmul_balanced64",
            "grouped_matmul_balanced128",
            "grouped_matmul_unroll2",
            "grouped_matmul_unroll4",
            "grouped_matmul_unroll8",
            "grouped_matmul_unroll16",
            "grouped_matmul_unroll32",
            "grouped_gate_up_reg16_counted",
            "grouped_gate_up_reg16_k32",
            "swiglu",
            "normalize",
            "head_norm",
            "rotary",
            "cache_kv",
            "router",
            "residual",
            "matvec_packed_qkv",
            "router_warp128",
            "qkv_head_norm_rope_cache",
            "qkv_head_norm_rope_cache_decode",
            "residual_normalize",
        ]
        .into_iter()
        .chain(block_reduction.then_some("normalize_retained2048"))
        .chain(
            std::env::var_os("LI_REPLAY_Q8_ALIGNED")
                .is_some()
                .then_some(["llama_q8_dp4a", "llama_q8_dp4a_aligned", "llama_q8_repack"])
                .into_iter()
                .flatten(),
        )
        .map(|name| {
            let function = module
                .load_function(name)
                .map_err(|error| failure(format!("{name}: {error}")))?;
            Ok((name, function))
        })
        .collect::<Result<_>>()?;
        let stream = context.default_stream();
        let ids = stream.clone_htod(&[0i32]).map_err(failure)?;
        Ok(Self {
            stream,
            functions,
            ids,
        })
    }

    fn project(
        &self,
        mode: Mode,
        tensor: &DeviceTensor,
        input: &CudaSlice<f32>,
        output: &mut CudaSlice<f32>,
        tokens: usize,
    ) -> Result<()> {
        let iterations = if matches!(mode, Mode::Sequential) {
            tokens
        } else {
            1
        };
        for token in 0..iterations {
            let count = if iterations == 1 { tokens } else { 1 };
            let x =
                input.slice(token * tensor.width as usize..(token + count) * tensor.width as usize);
            let mut y = output.slice_mut(
                8 + token * tensor.rows as usize..8 + (token + count) * tensor.rows as usize,
            );
            let tile = match mode {
                Mode::Tiled | Mode::Rows2 | Mode::Reference32Lanes4x4 => 4,
                Mode::WideTiled | Mode::Staged | Mode::Tf32x3 | Mode::StagedTf32x3 => 8,
                Mode::Register64x16 => 16,
                Mode::Register32x32 | Mode::Register64x32 => 32,
                Mode::Reference2x8
                | Mode::Reuse4x8
                | Mode::Reuse4x8Rows4
                | Mode::ReferenceCompact2x8
                | Mode::ReferenceInterleaved8
                | Mode::Reference64Lanes2x8
                | Mode::Reference32Lanes4x8 => 8,
                Mode::Reference2x16
                | Mode::Reference4x16
                | Mode::Reuse2x16
                | Mode::Reuse4x16
                | Mode::ReferencePacked2x16
                | Mode::ReferenceUnroll2
                | Mode::ReferenceUnroll4
                | Mode::ReferenceUnroll8
                | Mode::ReferenceUnroll16
                | Mode::ReferenceCompact2x16
                | Mode::ReferenceStatic
                | Mode::ReferenceInterleaved16
                | Mode::Reference64Lanes2x16 => 16,
                _ => 1,
            };
            let tiled = tile > 1;
            let name = match mode {
                Mode::WideTiled => "prefill_matmul8",
                Mode::Tiled => "prefill_matmul",
                Mode::Rows2 => "prefill_matmul_rows2",
                Mode::Staged => "prefill_matmul_tiled",
                Mode::Tf32x3 => "prefill_matmul_tf32x3",
                Mode::StagedTf32x3 => "prefill_matmul_tf32x3_tiled",
                Mode::Register64x16 => "prefill_matmul_reg64x16",
                Mode::Register32x32 => "prefill_matmul_reg32x32",
                Mode::Register64x32 => "prefill_matmul_reg64x32",
                Mode::Reference2x8 => "prefill_matmul_ref2x8",
                Mode::Reference2x16 => "prefill_matmul_ref2x16",
                Mode::Reference4x16 => "prefill_matmul_ref4x16",
                Mode::Reuse2x16 => "prefill_matmul_reuse2x16",
                Mode::Reuse4x8 => "prefill_matmul_reuse4x8",
                Mode::Reuse4x16 => "prefill_matmul_reuse4x16",
                Mode::Reuse4x8Rows4 => "prefill_matmul_reuse4x8_r4",
                Mode::ReferencePacked2x16 => "prefill_matmul_refpacked2x16",
                Mode::ReferenceUnroll2 => "prefill_matmul_ref2x16_u2",
                Mode::ReferenceUnroll4 => "prefill_matmul_ref2x16_u4",
                Mode::ReferenceUnroll8 => "prefill_matmul_ref2x16_u8",
                Mode::ReferenceUnroll16 => "prefill_matmul_ref2x16_u16",
                Mode::ReferenceCompact2x16 => "prefill_matmul_refcompact2x16",
                Mode::ReferenceCompact2x8 => "prefill_matmul_refcompact2x8",
                Mode::Reference64Lanes2x8 => "prefill_matmul_ref64_2x8",
                Mode::Reference64Lanes2x16 => "prefill_matmul_ref64_2x16",
                Mode::Reference32Lanes4x4 => "prefill_matmul_ref32_4x4",
                Mode::Reference32Lanes4x8 => "prefill_matmul_ref32_4x8",
                Mode::ReferenceInterleaved8 => "prefill_matmul_refinterleave2x8",
                Mode::ReferenceInterleaved16 => "prefill_matmul_refinterleave2x16",
                Mode::ReferenceStatic => match tensor.width {
                    2048 => "prefill_matmul_ref2048",
                    4096 => "prefill_matmul_ref4096",
                    _ => {
                        return Err(failure(
                            "fixed-width dense trial requires width 2048 or 4096",
                        ));
                    }
                },
                _ => "matvec",
            };
            let cfg = LaunchConfig {
                grid_dim: (
                    (tensor.rows as u32).div_ceil(match mode {
                        Mode::Rows2 => 2,
                        Mode::Staged => 32,
                        Mode::Register64x16 | Mode::Register64x32 => 64,
                        Mode::Register32x32 => 32,
                        Mode::Reference2x8
                        | Mode::Reference2x16
                        | Mode::Reuse2x16
                        | Mode::ReferencePacked2x16
                        | Mode::ReferenceUnroll2
                        | Mode::ReferenceUnroll4
                        | Mode::ReferenceUnroll8
                        | Mode::ReferenceUnroll16
                        | Mode::ReferenceCompact2x16
                        | Mode::ReferenceCompact2x8
                        | Mode::ReferenceStatic
                        | Mode::ReferenceInterleaved8
                        | Mode::ReferenceInterleaved16
                        | Mode::Reference64Lanes2x8
                        | Mode::Reference64Lanes2x16 => 2,
                        Mode::Reference4x16
                        | Mode::Reuse4x8
                        | Mode::Reuse4x16
                        | Mode::Reuse4x8Rows4
                        | Mode::Reference32Lanes4x4
                        | Mode::Reference32Lanes4x8 => 4,
                        Mode::Tf32x3 | Mode::StagedTf32x3 => 128,
                        _ => 1,
                    }),
                    count.div_ceil(tile) as u32,
                    1,
                ),
                block_dim: (
                    match mode {
                        Mode::Register64x16
                        | Mode::Register32x32
                        | Mode::Reuse2x16
                        | Mode::Reuse4x8Rows4
                        | Mode::Reference64Lanes2x8
                        | Mode::Reference64Lanes2x16
                        | Mode::Reference32Lanes4x4
                        | Mode::Reference32Lanes4x8 => 128,
                        Mode::Reference4x16 => 512,
                        _ => 256,
                    },
                    1,
                    1,
                ),
                shared_mem_bytes: 0,
            };
            let count = count as i32;
            let zero = 0i32;
            let one = 1i32;
            let mut args = self.stream.launch_builder(&self.functions[name]);
            args.arg(&tensor.bytes)
                .arg(&tensor.kind)
                .arg(&x)
                .arg(&mut y)
                .arg(&tensor.width)
                .arg(&tensor.rows);
            if tiled {
                args.arg(&count);
            } else {
                args.arg(&self.ids).arg(&zero).arg(&one);
            }
            // Fixture shapes match packed storage; views bound every token-major
            // read/write; the selected mode determines its fixed CTA size.
            unsafe { args.launch(cfg) }.map_err(failure)?;
        }
        Ok(())
    }

    fn check(
        &self,
        tensor: &DeviceTensor,
        input: &CudaSlice<f32>,
        expected: &[f32],
        tokens: usize,
    ) -> Result<()> {
        self.check_modes(
            tensor,
            input,
            expected,
            tokens,
            &[
                Mode::Sequential,
                Mode::Independent,
                Mode::Tiled,
                Mode::WideTiled,
                Mode::Rows2,
            ],
            true,
        )
    }

    fn check_modes(
        &self,
        tensor: &DeviceTensor,
        input: &CudaSlice<f32>,
        expected: &[f32],
        tokens: usize,
        modes: &[Mode],
        exact: bool,
    ) -> Result<()> {
        let mut reference = None;
        for &mode in modes {
            let mut output = self
                .stream
                .clone_htod(&vec![1337.0f32; expected.len() + 16])
                .map_err(failure)?;
            self.project(mode, tensor, input, &mut output, tokens)?;
            let actual = self.stream.clone_dtoh(&output).map_err(failure)?;
            assert!(actual[..8].iter().all(|&x| x == 1337.0));
            assert!(actual[8 + expected.len()..].iter().all(|&x| x == 1337.0));
            let values = &actual[8..8 + expected.len()];
            for (index, (&a, &e)) in values.iter().zip(expected).enumerate() {
                assert!(
                    a.is_finite()
                        && (a - e).abs()
                            <= if exact {
                                0.0001 + e.abs() * 1e-5
                            } else {
                                0.005 + e.abs() * 0.003
                            },
                    "{mode:?} kind={} width={} tokens={tokens} output={index}: {a} != {e}",
                    tensor.kind,
                    tensor.width,
                );
            }
            let bits: Vec<_> = values.iter().map(|x| x.to_bits()).collect();
            if exact {
                if let Some(reference) = &reference {
                    assert_eq!(&bits, reference, "{mode:?}: changed GPU accumulation");
                } else {
                    reference = Some(bits);
                }
            }
        }
        Ok(())
    }
}

pub(super) fn inputs(width: usize, tokens: usize) -> Vec<f32> {
    (0..width * tokens)
        .map(|i| ((i * 13 + i / width * 7) % 97) as f32 / 97.0 - 0.5)
        .collect()
}

fn cpu_product(tensor: &Tensor, input: &[f32], rows: usize) -> Vec<f32> {
    cpu_product_precision(tensor, input, rows, false)
}

fn cpu_product_precision(tensor: &Tensor, input: &[f32], rows: usize, tf32: bool) -> Vec<f32> {
    let weights = match &tensor.storage {
        Storage::F32(values) => values.clone(),
        Storage::Packed { kind, bytes } => {
            let (block, size, _) = crate::gguf::layout(*kind).unwrap();
            let mut weights = vec![0.0; tensor.dims[0] * rows];
            for (packed, decoded) in bytes
                .chunks_exact(size)
                .zip(weights.chunks_exact_mut(block))
            {
                crate::gguf::decode(*kind, packed, decoded);
            }
            weights
        }
    };
    let mut output = vec![0.0; input.len() / tensor.dims[0] * rows];
    for (x, y) in input
        .chunks_exact(tensor.dims[0])
        .zip(output.chunks_exact_mut(rows))
    {
        for (row, value) in weights.chunks_exact(tensor.dims[0]).zip(y) {
            *value = row
                .iter()
                .zip(x)
                .map(|(&w, &v)| {
                    if tf32 {
                        f64::from(round_tf32(w)) * f64::from(round_tf32(v))
                    } else {
                        f64::from(w) * f64::from(v)
                    }
                })
                .sum::<f64>() as f32;
        }
    }
    output
}

fn round_tf32(value: f32) -> f32 {
    let bits = value.to_bits();
    if bits & 0x7f800000 == 0x7f800000 {
        return value;
    }
    f32::from_bits(bits.wrapping_add(0xfff + ((bits >> 13) & 1)) & !0x1fff)
}

#[test]
#[ignore = "requires compute capability >=8.0; TF32 prototype, run serially"]
fn tensorcore_tiles_match_rounded_cpu_with_rows_tokens_and_storage_tails() -> Result<()> {
    check_tensorcore_tiles("prefill_matmul_tf32", true)?;
    check_tensorcore_tiles("prefill_matmul_tf32x3", false)
}

#[test]
#[ignore = "experimental tiled GPU projections; run serially"]
fn staged_prefill_tiles_match_cpu_with_rows_tokens_and_storage_tails() -> Result<()> {
    check_tensorcore_tiles("prefill_matmul_tiled", false)?;
    check_tensorcore_tiles("prefill_matmul_tf32x3_tiled", false)
}

#[test]
#[ignore = "direct staged GPU weight decoding; run serially"]
fn staged_weight_decode_is_bitwise_equal_to_scalar_decoder() -> Result<()> {
    let harness = Harness::new()?;
    for kind in [0, 1, 6, 8, 12, 14] {
        for width in [256usize, 768] {
            let mut tensor = packed_fixture(kind, width, 9, 2);
            if let Storage::Packed { bytes, .. } = &mut tensor.storage
                && matches!(kind, 12 | 14)
            {
                let half_bits = [0x2345u16, 0x3555, 0x0001, 0x7bff, 0xb123, 0x8000, 0x0401, 0];
                let (_, size, _) = crate::gguf::layout(kind)?;
                for (b, block) in bytes.chunks_exact_mut(size).enumerate() {
                    let offset = if kind == 14 { 208 } else { 0 };
                    block[offset..offset + 2]
                        .copy_from_slice(&half_bits[b % half_bits.len()].to_le_bytes());
                    if kind == 12 {
                        block[2..4]
                            .copy_from_slice(&half_bits[(b + 3) % half_bits.len()].to_le_bytes());
                    }
                }
            }
            let tensor = DeviceTensor::upload(tensor, &harness.stream)?;
            let rows = 18i32;
            for k_tile in [32i32, 64, 256] {
                let mut initial = vec![1337i32; 17];
                initial[8] = 0;
                let mut status = harness.stream.clone_htod(&initial).map_err(failure)?;
                {
                    let mut check = status.slice_mut(8..9);
                    launch!(
                        harness,
                        "projection_decode_check",
                        (
                            (rows as u32).div_ceil(8),
                            (width as u32).div_ceil(k_tile as u32),
                            1
                        ),
                        &tensor.bytes,
                        &tensor.kind,
                        &mut check,
                        &tensor.width,
                        &rows,
                        &k_tile
                    );
                }
                let actual = harness.stream.clone_dtoh(&status).map_err(failure)?;
                assert_eq!(
                    actual, initial,
                    "direct decode kind={kind} width={width} k_tile={k_tile}"
                );
            }
        }
    }
    Ok(())
}

#[test]
#[ignore = "M8 register-tiled F32 correctness; run serially in release mode"]
fn register_prefill_tiles_match_cpu_ordered_f32_and_guards() -> Result<()> {
    let harness = Harness::new()?;
    for kind in [12, 14] {
        // The narrow non-aligned rows cross packed-block boundaries. Their total
        // element counts remain multiples of 256, as required by the fixture.
        for (width, rows, tokens) in [
            (256usize, 1usize, 1usize),
            (768, 33, 17),
            (2048, 65, 33),
            (4096, 3, 1),
            (65, 256, 3),
            (257, 256, 2),
            (256, 3, 512),
        ] {
            let tensor = packed_fixture(kind, width, rows, 1);
            let input = inputs(width, tokens);
            let cpu = cpu_product(&tensor, &input, rows);
            let tensor = DeviceTensor::upload(tensor, &harness.stream)?;
            let x = harness.stream.clone_htod(&input).map_err(failure)?;
            let mut reference = harness
                .stream
                .alloc_zeros::<f32>(rows * tokens + 16)
                .map_err(failure)?;
            harness.project(Mode::Staged, &tensor, &x, &mut reference, tokens)?;
            let reference = harness.stream.clone_dtoh(&reference).map_err(failure)?;
            let ordered = &reference[8..8 + cpu.len()];
            for (i, (&a, &e)) in ordered.iter().zip(&cpu).enumerate() {
                assert!(
                    a.is_finite() && (a - e).abs() <= 0.002 + 2e-5 * e.abs(),
                    "ordered F32 kind={kind} width={width} rows={rows} tokens={tokens} output={i}: {a} != {e}"
                );
            }
            // These variants change reuse/ILP only, not the serial K reduction.
            // Require every output bit to match the existing staged F32 kernel.
            harness.check_modes(&tensor, &x, ordered, tokens, REGISTER_MODES, true)?;
        }
    }
    Ok(())
}

#[test]
#[ignore = "reference-order dense tiles; serial CUDA release correctness"]
fn reference_order_prefill_matches_legacy_bits_cpu_and_guards() -> Result<()> {
    check_reference_prefill_modes(&[
        Mode::Tiled,
        Mode::Reference2x8,
        Mode::Reference2x16,
        Mode::Reference4x16,
        Mode::Reuse2x16,
        Mode::Reuse4x8,
        Mode::Reuse4x16,
        Mode::Reuse4x8Rows4,
        Mode::ReferencePacked2x16,
        Mode::ReferenceUnroll2,
        Mode::ReferenceUnroll4,
        Mode::ReferenceUnroll8,
        Mode::ReferenceUnroll16,
        Mode::ReferenceCompact2x16,
        Mode::ReferenceCompact2x8,
        Mode::Reference64Lanes2x8,
        Mode::Reference64Lanes2x16,
        Mode::Reference32Lanes4x4,
        Mode::Reference32Lanes4x8,
    ])
}

#[test]
#[ignore = "all half bit patterns; serial CUDA release correctness"]
fn native_half_decode_preserves_all_legacy_bits() -> Result<()> {
    let harness = Harness::new()?;
    let input: Vec<u8> = (0..=u16::MAX).flat_map(u16::to_le_bytes).collect();
    let input = harness.stream.clone_htod(&input).map_err(failure)?;
    let mut bits = harness.stream.alloc_zeros::<u32>(65536).map_err(failure)?;
    launch!(harness, "half_decode_check", (256, 1, 1), &input, &mut bits);
    let bits = harness.stream.clone_dtoh(&bits).map_err(failure)?;
    for (h, actual) in bits.into_iter().enumerate() {
        let h = h as u32;
        let sign = (h & 0x8000) << 16;
        let exponent = (h >> 10) & 31;
        let fraction = h & 1023;
        let expected = if exponent == 0 {
            let value = fraction as f32 * (1.0 / 16_777_216.0);
            if sign == 0 { value } else { -value }
        } else {
            f32::from_bits(sign | ((exponent + 112) << 23) | (fraction << 13))
        };
        assert_eq!(actual, expected.to_bits(), "half bits {h:04x}");
    }
    Ok(())
}

#[test]
#[ignore = "selected reference-order dense kernels; serial CUDA release correctness"]
fn interleaved_reference_prefill_matches_legacy_bits_cpu_and_guards() -> Result<()> {
    check_reference_prefill_modes(&[
        Mode::Tiled,
        Mode::Reference2x16,
        Mode::ReferenceInterleaved8,
        Mode::ReferenceInterleaved16,
    ])
}

fn check_reference_prefill_modes(modes: &[Mode]) -> Result<()> {
    check_reference_prefill_shapes(
        modes,
        &[
            (256, 1, 1),
            (768, 7, 17),
            (2048, 33, 33),
            (4096, 3, 7),
            (65, 256, 3),
            (257, 256, 2),
            (256, 3, 257),
        ],
    )
}

#[test]
#[ignore = "fixed-width reference-order dense kernels; serial CUDA release correctness"]
fn fixed_width_reference_prefill_matches_legacy_bits_cpu_and_guards() -> Result<()> {
    check_reference_prefill_shapes(
        &[Mode::Tiled, Mode::Reference2x16, Mode::ReferenceStatic],
        &[
            (2048, 1, 1),
            (2048, 3, 17),
            (2048, 33, 257),
            (4096, 1, 7),
            (4096, 3, 33),
            (4096, 33, 17),
        ],
    )
}

fn check_reference_prefill_shapes(modes: &[Mode], shapes: &[(usize, usize, usize)]) -> Result<()> {
    let harness = Harness::new()?;
    for kind in [12, 14] {
        for &(width, rows, tokens) in shapes {
            let tensor = packed_fixture(kind, width, rows, 1);
            let input = inputs(width, tokens);
            let expected = cpu_product(&tensor, &input, rows);
            let tensor = DeviceTensor::upload(tensor, &harness.stream)?;
            let x = harness.stream.clone_htod(&input).map_err(failure)?;
            harness.check_modes(&tensor, &x, &expected, tokens, modes, true)?;
        }
    }
    Ok(())
}

fn check_tensorcore_tiles(kernel: &'static str, rounded: bool) -> Result<()> {
    let harness = Harness::new()?;
    for kind in [0, 1, 6, 8, 12, 14] {
        let block = crate::gguf::layout(kind)?.0;
        for width in [block, 768, 2048] {
            for (tokens, rows) in [(1usize, 19usize), (3, 19), (8, 19), (17, 129)] {
                let tensor = packed_fixture(kind, width, rows, 1);
                let input = inputs(width, tokens);
                let expected = cpu_product_precision(&tensor, &input, rows, rounded);
                let tensor = DeviceTensor::upload(tensor, &harness.stream)?;
                let x = harness.stream.clone_htod(&input).map_err(failure)?;
                let mut out = harness
                    .stream
                    .clone_htod(&vec![1337.0f32; tokens * rows + 16])
                    .map_err(failure)?;
                let n = tokens as i32;
                {
                    let mut y = out.slice_mut(8..8 + tokens * rows);
                    launch!(
                        harness,
                        kernel,
                        (
                            (rows as u32).div_ceil(if kernel == "prefill_matmul_tiled" {
                                32
                            } else {
                                128
                            }),
                            (tokens as u32).div_ceil(8),
                            1
                        ),
                        &tensor.bytes,
                        &tensor.kind,
                        &x,
                        &mut y,
                        &tensor.width,
                        &tensor.rows,
                        &n
                    );
                }
                let actual = harness.stream.clone_dtoh(&out).map_err(failure)?;
                assert!(
                    actual[..8]
                        .iter()
                        .chain(&actual[8 + expected.len()..])
                        .all(|&x| x == 1337.0)
                );
                for (i, (&a, &b)) in actual[8..8 + expected.len()]
                    .iter()
                    .zip(&expected)
                    .enumerate()
                {
                    assert!(
                        a.is_finite() && (a - b).abs() <= 0.005 + 0.003 * b.abs(),
                        "{kernel} kind={kind} width={width} tokens={tokens} output={i}: {a} != {b}"
                    );
                }
            }
        }
    }
    Ok(())
}

#[test]
#[ignore = "requires compute capability >=8.0; grouped TF32 prototype, run serially"]
fn tensorcore_grouped_projection_matches_rounded_cpu() -> Result<()> {
    check_tensorcore_grouped("grouped_matmul_tf32", true)?;
    check_tensorcore_grouped("grouped_matmul_tf32x3", false)
}

fn check_tensorcore_grouped(kernel: &'static str, rounded: bool) -> Result<()> {
    let harness = Harness::new()?;
    let (width, rows, experts, tokens, used) = (256usize, 19usize, 3usize, 7usize, 2usize);
    let slots = (tokens * used).div_ceil(4) + experts - 1;
    let selected: Vec<i32> = (0..tokens)
        .flat_map(|t| (0..used).map(move |s| ((t * 5 + s) % experts) as i32))
        .collect();
    let ids = harness.stream.clone_htod(&selected).map_err(failure)?;
    let mut groups = harness
        .stream
        .alloc_zeros::<i32>(slots * 5)
        .map_err(failure)?;
    let (ne, nu, nt, ns) = (experts as i32, used as i32, tokens as i32, slots as i32);
    launch!(
        harness,
        "group_experts",
        (1, 1, 1),
        &ids,
        &mut groups,
        &ne,
        &nu,
        &nt,
        &ns
    );
    for kind in [0, 1, 6, 8, 12, 14] {
        for batched in [0i32, 1] {
            let input_count = tokens * if batched == 1 { used } else { 1 };
            let input = inputs(width, input_count);
            let tensor = packed_fixture(kind, width, rows, experts);
            let expected = cpu_product_precision(&tensor, &input, rows * experts, rounded);
            let tensor = DeviceTensor::upload(tensor, &harness.stream)?;
            let input = harness.stream.clone_htod(&input).map_err(failure)?;
            let mut out = harness
                .stream
                .clone_htod(&vec![1337.0f32; tokens * used * rows + 16])
                .map_err(failure)?;
            {
                let mut output = out.slice_mut(8..8 + tokens * used * rows);
                launch!(
                    harness,
                    kernel,
                    ((rows as u32).div_ceil(128), slots as u32, 1),
                    &tensor.bytes,
                    &tensor.kind,
                    &input,
                    &mut output,
                    &tensor.width,
                    &tensor.rows,
                    &groups,
                    &nu,
                    &batched
                );
            }
            let actual = harness.stream.clone_dtoh(&out).map_err(failure)?;
            assert!(
                actual[..8]
                    .iter()
                    .chain(&actual[8 + tokens * used * rows..])
                    .all(|&x| x == 1337.0)
            );
            for assignment in 0..tokens * used {
                let x = if batched == 1 {
                    assignment
                } else {
                    assignment / used
                };
                let expert = selected[assignment] as usize;
                for row in 0..rows {
                    let a = actual[8 + assignment * rows + row];
                    let b = expected[(x * experts + expert) * rows + row];
                    assert!(
                        a.is_finite() && (a - b).abs() <= 0.005 + 0.003 * b.abs(),
                        "{kernel} kind={kind} assignment={assignment} row={row}: {a} != {b}"
                    );
                }
            }
        }
    }
    Ok(())
}

#[test]
#[ignore = "requires CUDA 13; isolated prototype, run serially in release mode"]
fn packed_prefill_tiles_match_cpu_and_gpu_with_token_tails() -> Result<()> {
    let harness = Harness::new()?;
    for kind in [0, 1, 6, 8, 12, 14] {
        let block = crate::gguf::layout(kind)?.0;
        for width in [block, 256, 768, 2048] {
            let tensor = packed_fixture(kind, width, 5, 1);
            for tokens in [1, 3, 4, 5, 17] {
                let input = inputs(width, tokens);
                let expected = cpu_product(&tensor, &input, 5);
                let x = harness.stream.clone_htod(&input).map_err(failure)?;
                // Upload a fresh fixture because DeviceTensor owns its storage.
                let t = DeviceTensor::upload(packed_fixture(kind, width, 5, 1), &harness.stream)?;
                harness.check(&t, &x, &expected, tokens)?;
            }
        }
    }
    Ok(())
}

#[test]
#[ignore = "requires CUDA 13; grouped packed kernels, run serially in release mode"]
fn grouped_expert_tiles_preserve_assignments_tails_and_accumulation() -> Result<()> {
    check_grouped_tiles("grouped_matmul", 1)?;
    check_grouped_tiles("grouped_matmul_rows2", 2)
}

fn check_grouped_tiles(kernel: &'static str, row_tile: u32) -> Result<()> {
    check_grouped_projection(kernel, row_tile, 4, false)
}

#[test]
#[ignore = "experimental tiled grouped GPU projections; run serially"]
fn staged_grouped_tiles_preserve_assignments_tails_and_guards() -> Result<()> {
    check_grouped_projection("grouped_matmul_tiled", 32, 8, true)?;
    check_grouped_projection("grouped_matmul_tf32x3_tiled", 128, 8, true)
}

#[test]
#[ignore = "M8 grouped register32 F32 correctness; run serially in release mode"]
fn register_grouped_tiles_preserve_assignments_tails_guards_and_cpu() -> Result<()> {
    check_grouped_projection("grouped_matmul_reg32x32", 32, 32, false)
}

#[test]
#[ignore = "count-aware register tiles; serial CUDA release correctness"]
fn counted_grouped_tiles_preserve_assignments_tails_guards_and_cpu() -> Result<()> {
    check_grouped_projection("grouped_matmul_reg32_counted", 32, 32, false)
}

#[test]
#[ignore = "coalesced grouped staging and K tiles; serial CUDA release correctness"]
fn coalesced_grouped_tiles_preserve_assignments_tails_guards_and_cpu() -> Result<()> {
    for kernel in [
        "grouped_matmul_coalesced_k32",
        "grouped_matmul_coalesced_k64",
        "grouped_matmul_coalesced_k128",
    ] {
        check_grouped_projection(kernel, 32, 32, false)?;
    }
    Ok(())
}

#[test]
#[ignore = "balanced and wider expert mappings; serial CUDA release correctness"]
fn balanced_grouped_tiles_preserve_assignments_tails_guards_and_cpu() -> Result<()> {
    for (kernel, rows) in [
        ("grouped_matmul_balanced32", 32),
        ("grouped_matmul_wide64", 64),
        ("grouped_matmul_balanced64", 64),
        ("grouped_matmul_balanced128", 128),
    ] {
        check_grouped_projection(kernel, rows, 32, false)?;
    }
    Ok(())
}

#[test]
#[ignore = "ordered partially unrolled expert loops; serial CUDA release correctness"]
fn unrolled_grouped_tiles_preserve_assignments_tails_guards_and_cpu() -> Result<()> {
    for kernel in [
        "grouped_matmul_unroll2",
        "grouped_matmul_unroll4",
        "grouped_matmul_unroll8",
        "grouped_matmul_unroll16",
        "grouped_matmul_unroll32",
    ] {
        check_grouped_projection(kernel, 32, 32, false)?;
    }
    Ok(())
}

#[test]
#[ignore = "selected unroll8 expert kernel; serial CUDA release and sanitizers"]
fn unroll8_grouped_tiles_preserve_assignments_tails_guards_and_cpu() -> Result<()> {
    check_grouped_projection("grouped_matmul_unroll8", 32, 32, false)
}

fn check_grouped_projection(
    kernel: &'static str,
    row_tile: u32,
    group_tile: usize,
    approximate: bool,
) -> Result<()> {
    let harness = Harness::new()?;
    for kind in [0, 1, 6, 8, 12, 14] {
        if group_tile == 32 && !matches!(kind, 12 | 14) {
            continue;
        }
        let cases = [
            (3usize, 2usize, 3usize, 256usize, 5usize, false),
            (17, 3, 3, 256, 5, false),
            (128, 1, 3, 256, 5, false),
            (7, 2, 257, 256, 5, false),
            (1, 1, 1, 256, 33, false),
            (9, 3, 5, 768, 35, true),
            (17, 2, 3, 512, 129, true),
        ];
        let register_cases = if group_tile == 32 {
            vec![(65, 2, 7, 256, 33, true), (17, 2, 3, 257, 256, true)]
        } else {
            vec![]
        };
        for (tokens, used, experts, width, rows, skewed) in cases.into_iter().chain(register_cases)
        {
            let slots = (tokens * used).min((tokens * used).div_ceil(group_tile) + experts - 1);
            let selected: Vec<i32> = (0..tokens)
                .flat_map(|t| {
                    (0..used).map(move |s| {
                        (if skewed {
                            if s == 0 {
                                0
                            } else {
                                1 + (t + s - 1) % (experts - 1)
                            }
                        } else {
                            (t * 5 + s + if experts > 256 { experts - 1 } else { 0 }) % experts
                        }) as i32
                    })
                })
                .collect();
            let ids = harness.stream.clone_htod(&selected).map_err(failure)?;
            let mut groups = harness
                .stream
                .clone_htod(&vec![1337i32; slots * (group_tile + 1) + 16])
                .map_err(failure)?;
            let (ne, nu, nt, ns) = (experts as i32, used as i32, tokens as i32, slots as i32);
            {
                let mut descriptors = groups.slice_mut(8..8 + slots * (group_tile + 1));
                launch!(
                    harness,
                    match group_tile {
                        32 => "group_experts_reg32",
                        8 => "group_experts_tiled",
                        _ => "group_experts",
                    },
                    (1, 1, 1),
                    &ids,
                    &mut descriptors,
                    &ne,
                    &nu,
                    &nt,
                    &ns
                );
            }
            let grouped = harness.stream.clone_dtoh(&groups).map_err(failure)?;
            assert!(
                grouped[..8]
                    .iter()
                    .chain(&grouped[grouped.len() - 8..])
                    .all(|&x| x == 1337)
            );
            let mut seen = vec![false; selected.len()];
            for tile in grouped[8..grouped.len() - 8].chunks_exact(group_tile + 1) {
                if tile[0] < 0 {
                    assert!(tile.iter().all(|&x| x == -1));
                    continue;
                }
                for &assignment in &tile[1..] {
                    if assignment >= 0 {
                        let a = assignment as usize;
                        assert_eq!(selected[a], tile[0]);
                        assert!(!seen[a], "duplicate grouped assignment");
                        seen[a] = true;
                    }
                }
            }
            assert!(seen.into_iter().all(|x| x));
            let t =
                DeviceTensor::upload(packed_fixture(kind, width, rows, experts), &harness.stream)?;
            for batched in [0i32, 1] {
                let input_count = if batched == 1 { tokens * used } else { tokens };
                let host_input = inputs(width, input_count);
                let cpu = (group_tile == 32).then(|| {
                    cpu_product(
                        &packed_fixture(kind, width, rows, experts),
                        &host_input,
                        rows * experts,
                    )
                });
                let x = harness.stream.clone_htod(&host_input).map_err(failure)?;
                let length = tokens * used * rows;
                let mut out = harness
                    .stream
                    .clone_htod(&vec![1337.0f32; length + 16])
                    .map_err(failure)?;
                let mut reference = harness.stream.alloc_zeros::<f32>(length).map_err(failure)?;
                {
                    let mut y = out.slice_mut(8..8 + length);
                    let descriptors = groups.slice(8..8 + slots * (group_tile + 1));
                    let mut args = harness.stream.launch_builder(&harness.functions[kernel]);
                    args.arg(&t.bytes)
                        .arg(&t.kind)
                        .arg(&x)
                        .arg(&mut y)
                        .arg(&t.width)
                        .arg(&t.rows)
                        .arg(&descriptors)
                        .arg(&nu)
                        .arg(&batched);
                    unsafe {
                        args.launch(LaunchConfig {
                            grid_dim: ((rows as u32).div_ceil(row_tile), slots as u32, 1),
                            block_dim: (
                                if group_tile == 32
                                    && !matches!(
                                        kernel,
                                        "grouped_matmul_wide64" | "grouped_matmul_balanced128"
                                    )
                                {
                                    128
                                } else {
                                    256
                                },
                                1,
                                1,
                            ),
                            shared_mem_bytes: 0,
                        })
                    }
                    .map_err(failure)?;
                }
                for token in 0..tokens {
                    let stride = if batched == 1 { used } else { 1 };
                    let input = x.slice(token * stride * width..(token + 1) * stride * width);
                    let selected = ids.slice(token * used..(token + 1) * used);
                    let mut y = reference.slice_mut(token * used * rows..(token + 1) * used * rows);
                    launch!(
                        harness,
                        "matvec",
                        (rows as u32, used as u32, 1),
                        &t.bytes,
                        &t.kind,
                        &input,
                        &mut y,
                        &t.width,
                        &t.rows,
                        &selected,
                        &1i32,
                        &batched
                    );
                }
                let actual = harness.stream.clone_dtoh(&out).map_err(failure)?;
                let expected = harness.stream.clone_dtoh(&reference).map_err(failure)?;
                assert!(
                    actual[..8]
                        .iter()
                        .chain(&actual[8 + length..])
                        .all(|&x| x == 1337.0)
                );
                for (i, (&a, &b)) in actual[8..8 + length].iter().zip(&expected).enumerate() {
                    assert!(
                        a.is_finite()
                            && if approximate {
                                (a - b).abs() <= 0.005 + 0.003 * b.abs()
                            } else if group_tile == 32 {
                                (a - b).abs() <= 0.002 + 2e-5 * b.abs()
                            } else {
                                a.to_bits() == b.to_bits()
                            },
                        "{kernel} kind={kind} tokens={tokens} used={used} experts={experts} batched={batched} output={i}: {a} != {b}"
                    );
                    if let Some(cpu) = &cpu {
                        let assignment = i / rows;
                        let input = if batched == 1 {
                            assignment
                        } else {
                            assignment / used
                        };
                        let expert = selected[assignment] as usize;
                        let e = cpu[(input * experts + expert) * rows + i % rows];
                        assert!(
                            (a - e).abs() <= 0.002 + 2e-5 * e.abs(),
                            "{kernel} CPU kind={kind} tokens={tokens} used={used} experts={experts} batched={batched} output={i}: {a} != {e}"
                        );
                    }
                }
            }
        }
    }
    Ok(())
}

#[test]
#[ignore = "requires CUDA 13; fused attention, run serially in release mode"]
fn fused_attention_matches_unfused_at_causal_and_context_boundaries() -> Result<()> {
    let harness = Harness::new()?;
    check_attention(&harness, 1)?;
    check_attention(&harness, 128)
}

#[test]
#[ignore = "CUDA attention structural experiment; complete outputs and isolated timing, serial GPU"]
fn normalized_attention_preserves_bits_and_measures_complete_heads() -> Result<()> {
    compare_attention_options(false)
}

#[test]
#[ignore = "CUDA attention reuse experiment; complete outputs and isolated timing, serial GPU"]
fn reused_attention_preserves_bits_and_measures_complete_heads() -> Result<()> {
    compare_attention_options(true)
}

fn compare_attention_options(reuse: bool) -> Result<()> {
    let normalized = Harness::with_attention_options(true, reuse)?;
    for tokens in [1, 7, 128, 512] {
        check_attention(&normalized, tokens)?;
    }
    let harnesses = [Harness::with_normalized_attention(reuse)?, normalized];
    let (heads, kv_heads, dim, tokens, context) = (32usize, 4usize, 128usize, 512usize, 1024usize);
    let queries = inputs(heads * dim, tokens);
    let keys = inputs(kv_heads * dim, context);
    let values: Vec<_> = keys
        .iter()
        .enumerate()
        .map(|(i, &v)| v * ((i % 7) as f32 - 3.0))
        .collect();
    let mut buffers = Vec::new();
    for harness in &harnesses {
        buffers.push((
            harness.stream.clone_htod(&queries).map_err(failure)?,
            harness.stream.clone_htod(&keys).map_err(failure)?,
            harness.stream.clone_htod(&values).map_err(failure)?,
            harness
                .stream
                .alloc_zeros::<f32>(queries.len())
                .map_err(failure)?,
        ));
    }
    let mut samples = [Vec::new(), Vec::new()];
    let (d, h, k, count) = (dim as i32, heads as i32, kv_heads as i32, 1i32);
    for repetition in 0..10 {
        for slot in 0..2 {
            let index = (repetition + slot) % 2;
            let harness = &harnesses[index];
            let (q, keys, values, output) = &mut buffers[index];
            let start = harness
                .stream
                .record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))
                .map_err(failure)?;
            launch!(
                harness,
                "attention_fused",
                (heads as u32, tokens as u32, 1),
                &*q,
                &*keys,
                &*values,
                output,
                &d,
                &h,
                &k,
                &count
            );
            let end = harness
                .stream
                .record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))
                .map_err(failure)?;
            end.synchronize().map_err(failure)?;
            if repetition >= 3 {
                samples[index].push(start.elapsed_ms(&end).map_err(failure)?);
            }
        }
    }
    let expected = harnesses[0]
        .stream
        .clone_dtoh(&buffers[0].3)
        .map_err(failure)?;
    let actual = harnesses[1]
        .stream
        .clone_dtoh(&buffers[1].3)
        .map_err(failure)?;
    assert!(
        actual
            .iter()
            .zip(&expected)
            .all(|(&a, &b)| a.is_finite() && a.to_bits() == b.to_bits())
    );
    println!(
        "NORMALIZED_ATTENTION {}",
        json!({
            "heads":heads,"kv_heads":kv_heads,"dim":dim,"tokens":tokens,
            "output_values":actual.len(),"bitwise_equal":true,
            "reference_cuda_event_ms":samples[0],"normalized_cuda_event_ms":samples[1],
            "boundary_fixtures":48,
            "reuse_query_and_reduce_max":reuse,
            "scope":"Synthetic complete attention; not full-model or inference-speed admission"
        })
    );
    Ok(())
}

fn check_attention(harness: &Harness, tokens: usize) -> Result<()> {
    let (heads, kv_heads, context) = (6usize, 2usize, 1024usize);
    for dim in [2usize, 64, 128, 256] {
        for position in [0usize, 7, context - tokens] {
            let q = harness
                .stream
                .clone_htod(&inputs(heads * dim, tokens))
                .map_err(failure)?;
            let mut keys = inputs(kv_heads * dim, context);
            let mut values: Vec<_> = keys
                .iter()
                .enumerate()
                .map(|(i, &v)| v * ((i % 7) as f32 - 3.0))
                .collect();
            keys[(position + tokens) * kv_heads * dim..].fill(f32::NAN);
            values[(position + tokens) * kv_heads * dim..].fill(f32::NAN);
            let keys = harness.stream.clone_htod(&keys).map_err(failure)?;
            let values = harness.stream.clone_htod(&values).map_err(failure)?;
            let mut scores = harness
                .stream
                .alloc_zeros::<f32>(tokens * heads * context)
                .map_err(failure)?;
            let mut expected = harness
                .stream
                .alloc_zeros::<f32>(tokens * heads * dim)
                .map_err(failure)?;
            let mut out = harness
                .stream
                .clone_htod(&vec![1337.0f32; tokens * heads * dim + 16])
                .map_err(failure)?;
            let (d, h, k, c, count) = (
                dim as i32,
                heads as i32,
                kv_heads as i32,
                context as i32,
                (position + 1) as i32,
            );
            launch!(
                harness,
                "attention_scores",
                (heads as u32, (position + tokens) as u32, tokens as u32),
                &q,
                &keys,
                &mut scores,
                &d,
                &h,
                &k,
                &c,
                &count
            );
            launch!(
                harness,
                "attention_values",
                (heads as u32, tokens as u32, 1),
                &mut scores,
                &values,
                &mut expected,
                &d,
                &h,
                &k,
                &c,
                &count
            );
            {
                let mut output = out.slice_mut(8..8 + tokens * heads * dim);
                launch!(
                    harness,
                    "attention_fused",
                    (heads as u32, tokens as u32, 1),
                    &q,
                    &keys,
                    &values,
                    &mut output,
                    &d,
                    &h,
                    &k,
                    &count
                );
            }
            let actual = harness.stream.clone_dtoh(&out).map_err(failure)?;
            let expected = harness.stream.clone_dtoh(&expected).map_err(failure)?;
            assert!(
                actual[..8]
                    .iter()
                    .chain(&actual[8 + expected.len()..])
                    .all(|&x| x == 1337.0)
            );
            for (i, (&a, &b)) in actual[8..8 + expected.len()]
                .iter()
                .zip(&expected)
                .enumerate()
            {
                assert!(
                    a.is_finite() && a.to_bits() == b.to_bits(),
                    "fused dim={dim} position={position} output={i}: {a} != {b}"
                );
            }
        }
    }
    Ok(())
}

#[test]
#[ignore = "GPU microbenchmark, not model inference; run serially with --nocapture"]
fn benchmark_packed_prefill_weight_reuse() -> Result<()> {
    benchmark_prefill_modes(
        &[
            Mode::Sequential,
            Mode::Independent,
            Mode::Tiled,
            Mode::WideTiled,
            Mode::Rows2,
        ],
        &[1, 4, 16, 64],
        true,
    )
}

#[test]
#[ignore = "GPU tiled F32 microbenchmark; run serially with --nocapture"]
fn benchmark_prefill_tiled_f32() -> Result<()> {
    benchmark_prefill_modes(&[Mode::Tiled, Mode::Staged], &[64, 128, 256], false)
}

#[test]
#[ignore = "GPU tiled TF32x3 microbenchmark; run serially with --nocapture"]
fn benchmark_prefill_tiled_tf32x3() -> Result<()> {
    benchmark_prefill_modes(&[Mode::Tf32x3, Mode::StagedTf32x3], &[64, 128, 256], false)
}

fn benchmark_prefill_modes(modes: &[Mode], token_counts: &[usize], exact: bool) -> Result<()> {
    benchmark_prefill_shapes(
        modes,
        token_counts,
        &[(2048, 768), (768, 2048), (2048, 4096)],
        exact,
        false,
        exact,
    )
}

// Separate selectors keep the representative corpus out of existing/default
// benchmark runs. These are isolated dense operations, not grouped routing.
#[test]
#[ignore = "M8 real dense shapes, 1/16/64 tokens; serial release --nocapture"]
fn benchmark_register_prefill_dense_small() -> Result<()> {
    benchmark_register_shapes(&[(2048, 4096), (4096, 2048), (2048, 512)], &[1, 16, 64])
}

#[test]
#[ignore = "M8 real dense shapes, 128/256/512 tokens; serial release --nocapture"]
fn benchmark_register_prefill_dense_large() -> Result<()> {
    benchmark_register_shapes(&[(2048, 4096), (4096, 2048), (2048, 512)], &[128, 256, 512])
}

#[test]
#[ignore = "M8 isolated expert shapes, 1/4/16/32/64 assignments; serial release --nocapture"]
fn benchmark_register_prefill_expert_small() -> Result<()> {
    benchmark_register_shapes(&[(2048, 768), (768, 2048)], &[1, 4, 16, 32, 64])
}

#[test]
#[ignore = "M8 isolated expert shapes, 128/256/512 assignments; serial release --nocapture"]
fn benchmark_register_prefill_expert_large() -> Result<()> {
    benchmark_register_shapes(&[(2048, 768), (768, 2048)], &[128, 256, 512])
}

fn benchmark_register_shapes(shapes: &[(usize, usize)], token_counts: &[usize]) -> Result<()> {
    benchmark_prefill_shapes(
        &[
            Mode::Tiled,
            Mode::Staged,
            Mode::Register64x16,
            Mode::Register32x32,
            Mode::Register64x32,
        ],
        token_counts,
        shapes,
        false,
        true,
        false,
    )
}

#[test]
#[ignore = "queries driver resource limits; serial release --nocapture"]
fn reference_order_dense_kernel_resources() -> Result<()> {
    let harness = Harness::new()?;
    for (name, threads) in [
        ("prefill_matmul_ref2x8", 256),
        ("prefill_matmul_ref2x16", 256),
        ("prefill_matmul_ref4x16", 512),
        ("prefill_matmul_reuse2x16", 128),
        ("prefill_matmul_reuse4x8", 256),
        ("prefill_matmul_reuse4x16", 256),
        ("prefill_matmul_reuse4x8_r4", 128),
        ("prefill_matmul_refpacked2x16", 256),
        ("prefill_matmul_ref2x16_u2", 256),
        ("prefill_matmul_ref2x16_u4", 256),
        ("prefill_matmul_ref2x16_u8", 256),
        ("prefill_matmul_ref2x16_u16", 256),
        ("prefill_matmul_refcompact2x16", 256),
        ("prefill_matmul_refcompact2x8", 256),
        ("prefill_matmul_ref64_2x8", 128),
        ("prefill_matmul_ref64_2x16", 128),
        ("prefill_matmul_ref32_4x4", 128),
        ("prefill_matmul_ref32_4x8", 128),
        ("prefill_matmul_ref2048", 256),
        ("prefill_matmul_ref4096", 256),
        ("prefill_matmul_refinterleave2x16", 256),
        ("prefill_matmul_refinterleave2x8", 256),
    ] {
        let function = &harness.functions[name];
        println!(
            "DENSE_RESOURCES {}",
            json!({
                "kernel":name,
                "registers_per_thread":function.num_regs().map_err(failure)?,
                "static_shared_bytes":function.shared_size_bytes().map_err(failure)?,
                "local_bytes_per_thread":function.local_size_bytes().map_err(failure)?,
                "driver_predicted_max_blocks_per_sm":function
                    .occupancy_max_active_blocks_per_multiprocessor(threads, 0, None)
                    .map_err(failure)?,
                "threads":threads,
                "measured_occupancy":false,
            })
        );
    }
    Ok(())
}

#[test]
#[ignore = "reference-order dense trials; serial release --nocapture"]
fn benchmark_reference_order_prefill() -> Result<()> {
    benchmark_prefill_shapes(
        &[
            Mode::Reference2x16,
            Mode::ReferenceInterleaved8,
            Mode::ReferenceInterleaved16,
        ],
        &[256, 512],
        &[(2048, 4096), (4096, 2048), (2048, 512)],
        true,
        false,
        false,
    )
}

#[test]
#[ignore = "requires QWEN35_GGUF; serial CUDA release --nocapture"]
fn benchmark_qwen35_reference_prefill() -> Result<()> {
    let path = std::env::var_os("QWEN35_GGUF").expect("set the GGUF fixture path");
    let gguf = crate::gguf::Gguf::open(std::path::Path::new(&path), false)?;
    let harness = Harness::new()?;
    let modes = [
        Mode::Tiled,
        Mode::Reference2x16,
        Mode::ReferenceInterleaved8,
        Mode::ReferenceInterleaved16,
        Mode::Reference4x16,
        Mode::ReferencePacked2x16,
    ];
    let mut shapes = std::collections::BTreeMap::new();
    for (name, tensor) in gguf.tensors {
        if let (
            true,
            &[width, rows],
            Storage::Packed {
                kind: kind @ (8 | 12 | 13 | 14),
                ..
            },
        ) = (
            name.starts_with("blk."),
            tensor.dims.as_slice(),
            &tensor.storage,
        ) {
            let entry = shapes
                .entry((*kind, width, rows))
                .or_insert_with(|| (name.clone(), None));
            if entry.1.is_none() || name < entry.0 {
                *entry = (name, Some(tensor));
            }
        }
    }
    assert!(!shapes.is_empty(), "no packed Qwen3.5 projection matrices");
    for ((kind, width, rows), (name, tensor)) in shapes {
        let modes = if matches!(kind, 12 | 14) {
            &modes[..]
        } else {
            &modes[..2]
        };
        let t = DeviceTensor::upload(tensor.expect("selected weight tensor"), &harness.stream)?;
        for tokens in [16, 128, 512] {
            let x = harness
                .stream
                .clone_htod(&inputs(width, tokens))
                .map_err(failure)?;
            let mut output = harness
                .stream
                .alloc_zeros::<f32>(rows * tokens + 16)
                .map_err(failure)?;
            harness.project(Mode::Tiled, &t, &x, &mut output, tokens)?;
            let expected = harness.stream.clone_dtoh(&output).map_err(failure)?;
            harness.check_modes(&t, &x, &expected[8..8 + rows * tokens], tokens, modes, true)?;
            let mut samples = [
                Vec::new(),
                Vec::new(),
                Vec::new(),
                Vec::new(),
                Vec::new(),
                Vec::new(),
            ];
            let timing = Some(sys::CUevent_flags::CU_EVENT_DEFAULT);
            for repetition in 0..8 {
                for slot in 0..modes.len() {
                    let i = (slot + repetition) % modes.len();
                    let start = harness.stream.record_event(timing).map_err(failure)?;
                    for _ in 0..8 {
                        harness.project(modes[i], &t, &x, &mut output, tokens)?;
                    }
                    let end = harness.stream.record_event(timing).map_err(failure)?;
                    end.synchronize().map_err(failure)?;
                    if repetition > 0 {
                        samples[i].push(start.elapsed_ms(&end).map_err(failure)? / 8.0);
                    }
                }
            }
            for (mode, values) in modes.iter().zip(&mut samples) {
                values.sort_by(f32::total_cmp);
                println!(
                    "QWEN35_PROJECTION {}",
                    json!({
                        "tensor":name, "kind":kind, "width":width, "rows":rows, "tokens":tokens,
                        "mode":format!("{mode:?}"), "median_ms":values[values.len()/2],
                        "samples_ms":values,
                    })
                );
            }
        }
    }
    Ok(())
}

fn benchmark_prefill_shapes(
    modes: &[Mode],
    token_counts: &[usize],
    shapes: &[(usize, usize)],
    exact: bool,
    register_contract: bool,
    cpu_reference: bool,
) -> Result<()> {
    let harness = Harness::new()?;
    for kind in [12, 14] {
        for &(width, rows) in shapes {
            for &tokens in token_counts {
                let tensor = packed_fixture(kind, width, rows, 1);
                let input = inputs(width, tokens);
                let expected = cpu_reference.then(|| cpu_product(&tensor, &input, rows));
                let t = DeviceTensor::upload(tensor, &harness.stream)?;
                let x = harness.stream.clone_htod(&input).map_err(failure)?;
                let expected = if let Some(expected) = expected {
                    expected
                } else {
                    let mut reference = harness
                        .stream
                        .alloc_zeros::<f32>(rows * tokens + 16)
                        .map_err(failure)?;
                    harness.project(Mode::Tiled, &t, &x, &mut reference, tokens)?;
                    harness.stream.clone_dtoh(&reference).map_err(failure)?[8..8 + rows * tokens]
                        .to_vec()
                };
                harness.check_modes(&t, &x, &expected, tokens, modes, exact)?;
                if register_contract {
                    let mut ordered = harness
                        .stream
                        .alloc_zeros::<f32>(rows * tokens + 16)
                        .map_err(failure)?;
                    harness.project(Mode::Staged, &t, &x, &mut ordered, tokens)?;
                    let ordered = harness.stream.clone_dtoh(&ordered).map_err(failure)?;
                    harness.check_modes(
                        &t,
                        &x,
                        &ordered[8..8 + rows * tokens],
                        tokens,
                        REGISTER_MODES,
                        true,
                    )?;
                }
                let mut output = harness
                    .stream
                    .alloc_zeros::<f32>(rows * tokens + 16)
                    .map_err(failure)?;
                for warmup in 0..3 {
                    for slot in 0..modes.len() {
                        let mode = modes[(slot + warmup) % modes.len()];
                        harness.project(mode, &t, &x, &mut output, tokens)?;
                    }
                }
                harness.stream.synchronize().map_err(failure)?;
                let launches_per_sample = if register_contract { 1 } else { 8 };
                for repetition in 0..7 {
                    for slot in 0..modes.len() {
                        let mode = modes[(slot + repetition) % modes.len()];
                        harness.stream.synchronize().map_err(failure)?;
                        let start = harness
                            .stream
                            .record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))
                            .map_err(failure)?;
                        let wall = Instant::now();
                        for _ in 0..launches_per_sample {
                            harness.project(mode, &t, &x, &mut output, tokens)?;
                        }
                        let end = harness
                            .stream
                            .record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))
                            .map_err(failure)?;
                        end.synchronize().map_err(failure)?;
                        let wall_ms =
                            wall.elapsed().as_secs_f64() * 1000.0 / launches_per_sample as f64;
                        let device_ms =
                            start.elapsed_ms(&end).map_err(failure)? / launches_per_sample as f32;
                        let register_geometry = match mode {
                            Mode::Register64x16 => Some((64, 16, 64, 128, 20_992)),
                            Mode::Register32x32 => Some((32, 32, 64, 128, 16_768)),
                            Mode::Register64x32 => Some((64, 32, 32, 256, 12_672)),
                            _ => None,
                        };
                        println!(
                            "PREFILL_BENCH {}",
                            json!({
                                "kind": kind, "width": width, "rows": rows, "tokens": tokens,
                                "mode": format!("{mode:?}"), "repetition": repetition,
                                "launches_per_sample": launches_per_sample,
                                "register_geometry_rows_tokens_k_threads_shared_bytes": register_geometry,
                                "wall_ms": wall_ms, "cuda_event_ms": device_ms,
                                "weight_bytes": width * rows / crate::gguf::layout(kind)?.0
                                    * crate::gguf::layout(kind)?.1,
                                "activation_bytes": (width * tokens + rows * tokens + 16) * 4,
                            })
                        );
                    }
                }
            }
        }
    }
    Ok(())
}
