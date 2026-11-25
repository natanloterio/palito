//! QTIP3 projections (feature `qtip-research`): the fused trellis matvec of
//! `src/qtip/matvec.cu`, which applies both 256-block Hadamards itself. A
//! single projection is one memset and one launch (`matvec`); a Llama layer
//! made only of QTIP3 projections runs in four launches without memsets
//! (`multi`: q/k/v, output, gate/up, and down with SwiGLU fused in).

use super::{DeviceTensor, cache, failure};
use crate::{
    Result,
    gguf::{qtip_symbol_bits, qtip_tile_bytes},
    transformer::invalid,
};
use cudarc::{
    driver::{
        CudaContext, CudaFunction, CudaSlice, CudaStream, DevicePtr, LaunchConfig, PushKernelArg,
        sys,
    },
    nvrtc::CompileOptions,
};
use std::{path::Path, sync::Arc};

const CHUNK: usize = 256;
const TILE: usize = 16;
/// Blocks per launch the split aims for (best on an RTX 5090, 170 SMs).
const TARGET_BLOCKS: usize = 340;
/// 1MAD code normalization (`qtip::codebook`).
const CODE_NORM: f32 = 147.800_54;
const THREADS: u32 = 512;
const STATIC_SHARED_LIMIT: u32 = 48 * 1024;

/// Device side of one QTIP3 matrix, next to its tiles in `DeviceTensor::bytes`.
pub(super) struct QtipTensor {
    /// Row signs, then column signs.
    signs: CudaSlice<f32>,
    /// `sigma / 147.80054 / 256`: the code scale and both `1 / 16` norms.
    scale: f32,
    tiles_per_split: i32,
    splits: u32,
    shared_bytes: u32,
}

/// Column tiles per split: about `TARGET_BLOCKS` blocks in all, and a split's
/// columns divide 256 or are a multiple of it (the kernel rotates whole chunks).
fn tiles_per_split(col_tiles: usize, row_blocks: usize) -> usize {
    let wanted = col_tiles.div_ceil(TARGET_BLOCKS.div_ceil(row_blocks).clamp(1, col_tiles));
    if wanted >= CHUNK / TILE {
        wanted.div_ceil(CHUNK / TILE) * (CHUNK / TILE)
    } else {
        wanted.next_power_of_two()
    }
}

impl QtipTensor {
    /// Splits a QTIP3 tensor's host bytes (tiles, then the companion
    /// `[sigma, row signs, column signs]` folded in by `gguf`) and uploads the
    /// signs; `bytes` keeps only the tiles.
    pub(super) fn upload(
        bytes: &mut Vec<u8>,
        kind: u32,
        cols: usize,
        rows: usize,
        stream: &Arc<CudaStream>,
    ) -> Result<Self> {
        let per_tile = qtip_tile_bytes(kind).ok_or_else(|| invalid("not a QTIP tensor type"))?;
        let tile_bytes = rows * cols / (TILE * TILE) * per_tile;
        if bytes.len() != tile_bytes + 4 * (1 + rows + cols) {
            return Err(invalid("QTIP3 tensor bytes do not match its shape"));
        }
        let companion: Vec<f32> = bytes[tile_bytes..]
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().expect("4-byte chunks")))
            .collect();
        bytes.truncate(tile_bytes);
        let col_tiles = cols / TILE;
        let per_split = tiles_per_split(col_tiles, rows / CHUNK);
        let chunks = (per_split * TILE).div_ceil(CHUNK);
        Ok(Self {
            signs: stream.clone_htod(&companion[1..]).map_err(failure)?,
            scale: companion[0] / CODE_NORM / (CHUNK as f32),
            tiles_per_split: per_split as i32,
            splits: col_tiles.div_ceil(per_split) as u32,
            shared_bytes: (chunks * CHUNK * 4) as u32,
        })
    }
}

/// One matrix of a `qtip_matvec_multi` launch (device pointers).
#[derive(Clone, Copy, Default)]
struct Matrix {
    tiles: u64,
    col_signs: u64,
    row_signs: u64,
    y: u64,
    row_blocks: i32,
    scale: f32,
}

/// Tokens per `qtip_matmul_mma` block (QTIP_MMA_TOKENS).
const MATMUL_TOKENS: usize = 128;
/// Column tiles per `qtip_matmul_mma` step (QTIP_MMA_STEP_TILES).
const MATMUL_STEP_TILES: usize = 2;
/// Warps (16 rows each) per `qtip_matmul_mma` block (QTIP_MMA_WARPS).
const MATMUL_WARPS: usize = 4;
/// `qtip_matmul_i8`: tokens and warps per block (QTIP_I8_TOKENS,
/// QTIP_I8_WARPS), and the widest input `qtip_rotate_input_i8` holds in
/// registers (QTIP_I8_MAX_CHUNKS 256-chunks).
const MATMUL_I8_TOKENS: usize = 128;
const MATMUL_I8_WARPS: usize = 4;
const MATMUL_I8_MAX_CHUNKS: usize = 64;
/// Prefill matmul blocks the K split aims for, and the scratch it may use.
const MATMUL_TARGET_BLOCKS: usize = 1024;
const MATMUL_SCRATCH_FLOATS: usize = 16 << 20;
/// Fewest column tiles per split (shorter splits only add partial planes).
const MATMUL_MIN_SPLIT_TILES: usize = 8;

/// Column tiles per K split of `qtip_matmul_mma`: enough splits to reach
/// about MATMUL_TARGET_BLOCKS blocks, within the scratch budget.
fn matmul_split_tiles(col_tiles: usize, blocks: usize, plane: usize) -> usize {
    let wanted = MATMUL_TARGET_BLOCKS.div_ceil(blocks.max(1));
    let affordable = (MATMUL_SCRATCH_FLOATS / plane.max(1)).max(1);
    let splits = wanted
        .min(affordable)
        .min(col_tiles / MATMUL_MIN_SPLIT_TILES)
        .max(1);
    // A multiple of the kernel's tiles per step (QTIP_MMA_STEP_TILES).
    col_tiles
        .div_ceil(splits)
        .next_multiple_of(MATMUL_STEP_TILES)
}

fn pointer<T>(slice: &CudaSlice<T>, stream: &CudaStream) -> u64 {
    slice.device_ptr(stream).0
}

/// The kernels of one trellis width K (QTIP_K), compiled from the same source.
struct Variant {
    function: CudaFunction,
    multi: CudaFunction,
    multi_swiglu: CudaFunction,
    small_k: CudaFunction,
    small_k_multi: CudaFunction,
    small_k_multi_swiglu: CudaFunction,
    rotate_input: CudaFunction,
    rotate_output: CudaFunction,
    product: CudaFunction,
    rotate_input_i8: CudaFunction,
    product_i8: CudaFunction,
}

pub(super) struct QtipKernel {
    /// Indexed by K - 2: QTIP2, then QTIP3; only the widths a model uses.
    variants: [Option<Variant>; 2],
    /// Prefill scratch, grown on demand: rotated FP16 inputs and the rotated
    /// F32 product.
    rotated_input: Option<CudaSlice<u16>>,
    partial_output: Option<CudaSlice<f32>>,
    quantized_input: Option<CudaSlice<i8>>,
    input_scales: Option<CudaSlice<f32>>,
}

/// `slot`, reallocated (zeroed) when shorter than `len`.
fn grown<'a, T: cudarc::driver::DeviceRepr + cudarc::driver::ValidAsZeroBits>(
    slot: &'a mut Option<CudaSlice<T>>,
    len: usize,
    stream: &Arc<CudaStream>,
) -> Result<&'a mut CudaSlice<T>> {
    if slot.as_ref().is_none_or(|b| b.len() < len) {
        *slot = Some(stream.alloc_zeros(len).map_err(failure)?);
    }
    Ok(slot.as_mut().expect("allocated above"))
}

/// One warp per 256-chunk of each token row, eight warps per block.
fn rotate_config(width: usize, tokens: usize) -> LaunchConfig {
    LaunchConfig {
        grid_dim: ((width / CHUNK).div_ceil(8) as u32, tokens as u32, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    }
}

impl Variant {
    /// Compiles the kernels for K = `symbol_bits` (L = 12) and opts them into
    /// the dynamic shared memory of the widest split among `tensors`.
    fn load(
        context: &Arc<CudaContext>,
        architecture: (i32, i32),
        cache_dir: Option<&Path>,
        symbol_bits: u32,
        tensors: &[&DeviceTensor],
    ) -> Result<Self> {
        let source = format!(
            "#define QTIP_L 12\n#define QTIP_K {symbol_bits}\n#define QTIP_UNROLL 4\n\
             #define QTIP_MMA_WARPS {MATMUL_WARPS}\n#define QTIP_MMA_STEP_TILES {MATMUL_STEP_TILES}\n\
             {}\n{}",
            include_str!("../qtip/matvec.cu"),
            include_str!("../qtip/matmul.cu")
        );
        let ptx = cache::compile(
            cache_dir,
            &source,
            CompileOptions {
                options: vec![format!(
                    "--gpu-architecture=compute_{}{}",
                    architecture.0, architecture.1
                )],
                ..Default::default()
            },
        )?;
        let module = context.load_module(ptx).map_err(failure)?;
        let load = |name: &str| module.load_function(name).map_err(failure);
        let variant = Self {
            function: load("qtip_matvec_fused")?,
            multi: load("qtip_matvec_multi")?,
            multi_swiglu: load("qtip_matvec_multi_swiglu")?,
            small_k: load("qtip_matvec_small_k")?,
            small_k_multi: load("qtip_matvec_small_k_multi")?,
            small_k_multi_swiglu: load("qtip_matvec_small_k_multi_swiglu")?,
            rotate_input: load("qtip_rotate_input")?,
            rotate_output: load("qtip_rotate_output")?,
            product: load("qtip_matmul_mma")?,
            rotate_input_i8: load("qtip_rotate_input_i8")?,
            product_i8: load("qtip_matmul_i8")?,
        };
        let shared = tensors
            .iter()
            .filter_map(|t| t.qtip.as_ref())
            .map(|q| q.shared_bytes)
            .max()
            .unwrap_or(0);
        // A multi launch splits by its combined row blocks; a whole row bounds it.
        let multi_shared = tensors
            .iter()
            .map(|t| (t.width as usize).div_ceil(CHUNK) * CHUNK * 4)
            .max()
            .unwrap_or(0) as u32;
        for (f, bytes) in [
            (&variant.function, shared),
            (&variant.multi, multi_shared),
            (&variant.multi_swiglu, multi_shared),
        ] {
            if bytes > STATIC_SHARED_LIMIT {
                f.set_attribute(
                    sys::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                    bytes as i32,
                )
                .map_err(failure)?;
            }
        }
        Ok(variant)
    }
}

impl QtipKernel {
    /// Compiles the fused kernels (L = 12) for each trellis width K that
    /// `tensors` use (QTIP3: K = 3, QTIP2: K = 2).
    pub(super) fn load<'a>(
        context: &Arc<CudaContext>,
        architecture: (i32, i32),
        cache_dir: Option<&Path>,
        tensors: impl Iterator<Item = &'a DeviceTensor>,
    ) -> Result<Self> {
        let tensors: Vec<&DeviceTensor> = tensors.filter(|t| t.qtip.is_some()).collect();
        let mut variants = [None, None];
        for symbol_bits in [2u32, 3] {
            let used: Vec<&DeviceTensor> = tensors
                .iter()
                .copied()
                .filter(|t| qtip_symbol_bits(t.kind as u32) == Some(symbol_bits))
                .collect();
            if !used.is_empty() {
                variants[symbol_bits as usize - 2] = Some(Variant::load(
                    context,
                    architecture,
                    cache_dir,
                    symbol_bits,
                    &used,
                )?);
            }
        }
        Ok(Self {
            variants,
            rotated_input: None,
            partial_output: None,
            quantized_input: None,
            input_scales: None,
        })
    }

    /// The kernels for a QTIP tensor type.
    fn variant(variants: &[Option<Variant>; 2], kind: i32) -> Result<&Variant> {
        qtip_symbol_bits(kind as u32)
            .and_then(|k| variants[k as usize - 2].as_ref())
            .ok_or_else(|| invalid("no QTIP kernels for this tensor type"))
    }

    /// `outputs[i][..rows_i] = W_i x` for up to three QTIP3 tensors of the same
    /// width in one launch. With `x_up`, the input is SwiGLU(x, x_up). Each
    /// `zero` buffer (a later projection's output, untouched here) is zeroed
    /// on the way. Outputs must already be zero unless `memset_outputs`.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn multi(
        &self,
        stream: &Arc<CudaStream>,
        tensors: &[&DeviceTensor],
        outputs: &mut [&mut CudaSlice<f32>],
        x: &CudaSlice<f32>,
        x_up: Option<&CudaSlice<f32>>,
        zero: &mut [(&mut CudaSlice<f32>, usize)],
        memset_outputs: bool,
    ) -> Result<()> {
        let count = tensors.len();
        let cols = tensors.first().map_or(0, |t| t.width as usize);
        if !(1..=3).contains(&count) || outputs.len() != count || zero.len() > 3 {
            return Err(invalid(
                "QTIP3 multi launch takes 1-3 tensors and 0-3 zero buffers",
            ));
        }
        if x.len() < cols || x_up.is_some_and(|u| u.len() < cols) {
            return Err(invalid("QTIP3 multi input is too small"));
        }
        let mut m = [Matrix::default(); 3];
        let mut row_blocks = 0;
        for (i, (t, y)) in tensors.iter().zip(outputs.iter_mut()).enumerate() {
            let q = t
                .qtip
                .as_ref()
                .filter(|_| t.kind == tensors[0].kind && t.width as usize == cols)
                .ok_or_else(|| invalid("multi launch needs QTIP tensors of one type and width"))?;
            let rows = t.rows as usize;
            if y.len() < rows {
                return Err(invalid("QTIP3 multi output is too small"));
            }
            if memset_outputs {
                stream
                    .memset_zeros(&mut y.slice_mut(..rows))
                    .map_err(failure)?;
            }
            m[i] = Matrix {
                tiles: pointer(&t.bytes, stream),
                row_signs: pointer(&q.signs, stream),
                col_signs: pointer(&q.signs, stream) + 4 * rows as u64,
                y: pointer(&**y, stream),
                row_blocks: (rows / CHUNK) as i32,
                scale: q.scale,
            };
            row_blocks += rows / CHUNK;
        }
        // Unused slots repeat matrix 0; rb0 and rb1 keep the kernel off them.
        let first = m[0];
        m.iter_mut().skip(count).for_each(|slot| *slot = first);
        let (rb0, rb1) = match count {
            1 => (row_blocks as i32, 0),
            2 => (m[0].row_blocks, row_blocks as i32),
            _ => (m[0].row_blocks, m[1].row_blocks),
        };
        let mut zeros = [(0u64, 0i32); 3];
        for (slot, (buffer, n)) in zeros.iter_mut().zip(zero.iter()) {
            if buffer.len() < *n {
                return Err(invalid("QTIP3 zero buffer is too small"));
            }
            *slot = (pointer(&**buffer, stream), *n as i32);
        }
        let col_tiles = cols / TILE;
        let per_split = tiles_per_split(col_tiles, row_blocks) as i32;
        let config = LaunchConfig {
            grid_dim: (
                row_blocks as u32,
                col_tiles.div_ceil(per_split as usize) as u32,
                1,
            ),
            block_dim: (THREADS, 1, 1),
            shared_mem_bytes: ((per_split as usize * TILE).div_ceil(CHUNK) * CHUNK * 4) as u32,
        };
        let (x, x_up) = (pointer(x, stream), x_up.map_or(0, |u| pointer(u, stream)));
        let cols = cols as i32;
        let variant = Self::variant(&self.variants, tensors[0].kind)?;
        let function = if x_up != 0 {
            &variant.multi_swiglu
        } else {
            &variant.multi
        };
        let tiles = m.map(|m| m.tiles);
        let col_signs = m.map(|m| m.col_signs);
        let row_signs = m.map(|m| m.row_signs);
        let ys = m.map(|m| m.y);
        let scales = m.map(|m| m.scale);
        let zero_ptrs = zeros.map(|z| z.0);
        let zero_counts = zeros.map(|z| z.1);
        let mut args = stream.launch_builder(function);
        for group in [&tiles, &col_signs, &row_signs, &ys] {
            args.arg(&group[0]).arg(&group[1]).arg(&group[2]);
        }
        args.arg(&scales[0]).arg(&scales[1]).arg(&scales[2]);
        args.arg(&rb0).arg(&rb1).arg(&x).arg(&x_up);
        args.arg(&zero_ptrs[0])
            .arg(&zero_ptrs[1])
            .arg(&zero_ptrs[2]);
        args.arg(&zero_counts[0])
            .arg(&zero_counts[1])
            .arg(&zero_counts[2]);
        args.arg(&cols).arg(&per_split);
        // QTIP_MULTI_PARAMS order. Pointers come from live slices checked
        // above; the zero buffers are disjoint from this launch's inputs and
        // outputs, and rb0/rb1 keep the grid on the `count` real matrices.
        unsafe { args.launch(config) }.map_err(failure)?;
        Ok(())
    }

    /// Fused small-batch (`2 <= tokens <= 16`) multi-matrix launch for Q/K/V,
    /// Gate/Up, or Down+SwiGLU (1 single launch instead of 2–3 + swiglu).
    #[allow(clippy::too_many_arguments)]
    pub(super) fn matmul_small_k_multi(
        &self,
        stream: &Arc<CudaStream>,
        matrices: &[(&DeviceTensor, &mut CudaSlice<f32>)],
        x: &CudaSlice<f32>,
        x_up: Option<&CudaSlice<f32>>,
        tokens: usize,
    ) -> Result<()> {
        if matrices.is_empty() || matrices.len() > 3 || !(2..=16).contains(&tokens) {
            return Err(invalid("matmul_small_k_multi needs 1..=3 matrices and 2..=16 tokens"));
        }
        let kind = matrices[0].0.kind;
        let cols = matrices[0].0.width as usize;
        let variant = Self::variant(&self.variants, kind)?;
        let mut tiles = [0u64; 3];
        let mut col_signs = [0u64; 3];
        let mut row_signs = [0u64; 3];
        let mut ys = [0u64; 3];
        let mut scales = [0.0f32; 3];
        let mut rows_arr = [0i32; 3];
        let mut row_blocks = [0usize; 3];

        // Zero output slices and gather raw pointers.
        let mut total_rb = 0usize;
        for (i, (t, y)) in matrices.iter().enumerate() {
            let q = t
                .qtip
                .as_ref()
                .ok_or_else(|| invalid("matmul_small_k_multi needs QTIP tensors"))?;
            let r = t.rows as usize;
            let mut y_mut = y.slice(0..tokens * r);
            // SAFETY: each y in `matrices` is a distinct output buffer; zeroing via raw ptr below.
            let y_ptr = pointer(y, stream);
            unsafe {
                cudarc::driver::result::memset_d8_async(
                    y_ptr,
                    0,
                    tokens * r * std::mem::size_of::<f32>(),
                    stream.cu_stream(),
                )
                .map_err(failure)?;
            }
            let _ = &mut y_mut;
            tiles[i] = pointer(&t.bytes, stream);
            row_signs[i] = pointer(&q.signs, stream);
            col_signs[i] = row_signs[i] + (r * std::mem::size_of::<f32>()) as u64;
            ys[i] = y_ptr;
            scales[i] = q.scale;
            rows_arr[i] = r as i32;
            row_blocks[i] = r / CHUNK;
            total_rb += row_blocks[i];
        }
        let rb0 = row_blocks[0] as i32;
        let rb1 = if matrices.len() >= 2 {
            row_blocks[1] as i32
        } else {
            total_rb as i32
        };
        let total_chunks = cols / CHUNK;
        let splits = 340usize.div_ceil(total_rb).clamp(1, total_chunks);
        let chunks_per_split = total_chunks.div_ceil(splits);
        let splits = total_chunks.div_ceil(chunks_per_split);
        let x_ptr = pointer(x, stream);
        let x_up_ptr = x_up.map_or(0u64, |u| pointer(u, stream));
        let cols_i = cols as i32;
        let tokens_i = tokens as i32;
        let cps_i = chunks_per_split as i32;

        let func = if x_up.is_some() {
            &variant.small_k_multi_swiglu
        } else {
            &variant.small_k_multi
        };
        let config = LaunchConfig {
            grid_dim: (total_rb as u32, splits as u32, 1),
            block_dim: (512, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut args = stream.launch_builder(func);
        for g in [&tiles, &col_signs, &row_signs, &ys] {
            args.arg(&g[0]).arg(&g[1]).arg(&g[2]);
        }
        args.arg(&scales[0]).arg(&scales[1]).arg(&scales[2]);
        args.arg(&rows_arr[0]).arg(&rows_arr[1]).arg(&rows_arr[2]);
        args.arg(&rb0).arg(&rb1).arg(&x_ptr).arg(&x_up_ptr);
        args.arg(&cols_i).arg(&tokens_i).arg(&cps_i);
        unsafe { args.launch(config) }.map_err(failure)?;
        Ok(())
    }

    /// `y[t * rows + r] = (W x_t)[r]` for `tokens` token-major inputs
    /// (prefill) on tensor cores (`src/qtip/matmul.cu`). Exact weights and
    /// FP16 inputs after rotation (about 2e-4 from the decode matvec), or
    /// with `fast` (int8-fast) int8 weights round(code / 4) and int8 inputs
    /// (about 1.2%, twice as fast). Needs sm_80+, rows % 64 == 0 and, with
    /// `fast`, at most 8192 columns.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn matmul(
        &mut self,
        stream: &Arc<CudaStream>,
        tensor: &DeviceTensor,
        x: &CudaSlice<f32>,
        y: &mut CudaSlice<f32>,
        tokens: usize,
        fast: bool,
    ) -> Result<()> {
        let q = tensor
            .qtip
            .as_ref()
            .ok_or_else(|| invalid("QTIP matmul needs a QTIP tensor"))?;
        let variant = Self::variant(&self.variants, tensor.kind)?;
        let (rows, cols) = (tensor.rows as usize, tensor.width as usize);
        let (warps, block_tokens) = if fast {
            (MATMUL_I8_WARPS, MATMUL_I8_TOKENS)
        } else {
            (MATMUL_WARPS, MATMUL_TOKENS)
        };
        let row_block = 16 * warps;
        if !rows.is_multiple_of(row_block) || !cols.is_multiple_of(CHUNK) {
            return Err(invalid("QTIP3 matmul needs rows % 64 and cols % 256 == 0"));
        }
        if fast && cols / CHUNK > MATMUL_I8_MAX_CHUNKS {
            return Err(invalid("QTIP3 int8 matmul supports up to 16384 columns"));
        }
        if x.len() < tokens * cols || y.len() < tokens * rows {
            return Err(invalid("QTIP3 matmul buffers are too small"));
        }
        // Small-batch (k <= 16, e.g. speculative tree verification): use the 1-launch
        // fused multi-vector Tensor-Core kernel instead of the 3-launch 128-token MMA path.
        if (2..=16).contains(&tokens) && rows.is_multiple_of(CHUNK) {
            let mut y_slice = y.slice_mut(0..tokens * rows);
            stream.memset_zeros(&mut y_slice).map_err(failure)?;
            let row_blocks = rows / CHUNK;
            let total_chunks = cols / CHUNK;
            let splits = 340usize
                .div_ceil(row_blocks)
                .clamp(1, total_chunks);
            let chunks_per_split = total_chunks.div_ceil(splits);
            let splits = total_chunks.div_ceil(chunks_per_split);
            let col_signs = q.signs.slice(rows..);
            let row_signs = q.signs.slice(..rows);
            let (rows_i, cols_i, tokens_i, cps_i) = (
                rows as i32,
                cols as i32,
                tokens as i32,
                chunks_per_split as i32,
            );
            let config = LaunchConfig {
                grid_dim: (row_blocks as u32, splits as u32, 1),
                block_dim: (512, 1, 1),
                shared_mem_bytes: 0,
            };
            let mut args = stream.launch_builder(&variant.small_k);
            args.arg(&tensor.bytes)
                .arg(x)
                .arg(&col_signs)
                .arg(&row_signs)
                .arg(&mut y_slice)
                .arg(&rows_i)
                .arg(&cols_i)
                .arg(&tokens_i)
                .arg(&cps_i)
                .arg(&q.scale);
            unsafe { args.launch(config) }.map_err(failure)?;
            return Ok(());
        }
        let col_tiles = cols / TILE;
        let blocks = (rows / row_block) * tokens.div_ceil(block_tokens);
        let split_tiles = matmul_split_tiles(col_tiles, blocks, tokens * rows);
        let splits = col_tiles.div_ceil(split_tiles);
        let partial = grown(&mut self.partial_output, splits * tokens * rows, stream)?;
        let (cols_i, rows_i, tokens_i) = (cols as i32, rows as i32, tokens as i32);
        let (split_tiles_i, splits_i) = (split_tiles as i32, splits as i32);
        let col_signs = q.signs.slice(rows..);
        let product = LaunchConfig {
            grid_dim: (
                (rows / row_block) as u32,
                tokens.div_ceil(block_tokens) as u32,
                splits as u32,
            ),
            block_dim: (32 * warps as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        // Launch bounds: x holds tokens * cols values and y tokens * rows
        // (checked); tensor.bytes holds rows / 16 row groups of cols / 16
        // tiles; `partial` holds `splits` planes of tokens * rows values and
        // split_tiles is a multiple of both kernels' tiles per step.
        let output_scale = if fast {
            let quantized = grown(&mut self.quantized_input, tokens * cols, stream)?;
            let scales = grown(&mut self.input_scales, tokens, stream)?;
            let mut args = stream.launch_builder(&variant.rotate_input_i8);
            args.arg(x)
                .arg(&col_signs)
                .arg(&mut *quantized)
                .arg(&mut *scales);
            args.arg(&cols_i);
            let config = LaunchConfig {
                grid_dim: (tokens as u32, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            };
            unsafe { args.launch(config) }.map_err(failure)?;
            let mut args = stream.launch_builder(&variant.product_i8);
            args.arg(&tensor.bytes).arg(&*quantized).arg(&*scales);
            args.arg(&mut *partial).arg(&cols_i).arg(&rows_i);
            args.arg(&tokens_i).arg(&split_tiles_i);
            unsafe { args.launch(product) }.map_err(failure)?;
            q.scale
        } else {
            let rotated = grown(&mut self.rotated_input, tokens * cols, stream)?;
            let mut args = stream.launch_builder(&variant.rotate_input);
            args.arg(x).arg(&col_signs).arg(&mut *rotated).arg(&cols_i);
            unsafe { args.launch(rotate_config(cols, tokens)) }.map_err(failure)?;
            let mut args = stream.launch_builder(&variant.product);
            args.arg(&tensor.bytes).arg(&*rotated).arg(&mut *partial);
            args.arg(&cols_i)
                .arg(&rows_i)
                .arg(&tokens_i)
                .arg(&split_tiles_i);
            unsafe { args.launch(product) }.map_err(failure)?;
            // X' was divided by 16 to stay well inside the FP16 range.
            q.scale * 16.0
        };
        let mut args = stream.launch_builder(&variant.rotate_output);
        args.arg(&*partial).arg(&q.signs).arg(y).arg(&rows_i);
        args.arg(&output_scale).arg(&splits_i).arg(&tokens_i);
        // Row signs are the first `rows` values of q.signs.
        unsafe { args.launch(rotate_config(rows, tokens)) }.map_err(failure)?;
        Ok(())
    }

    /// `y[..rows] = W x[..cols]` for a QTIP3 tensor.
    pub(super) fn matvec(
        &self,
        stream: &Arc<CudaStream>,
        t: &DeviceTensor,
        x: &CudaSlice<f32>,
        y: &mut CudaSlice<f32>,
    ) -> Result<()> {
        let q = t
            .qtip
            .as_ref()
            .ok_or_else(|| invalid("not a QTIP tensor"))?;
        let variant = Self::variant(&self.variants, t.kind)?;
        let (rows, cols) = (t.rows as usize, t.width as usize);
        if x.len() < cols || y.len() < rows {
            return Err(invalid("QTIP3 matvec buffers are too small"));
        }
        stream
            .memset_zeros(&mut y.slice_mut(..rows))
            .map_err(failure)?;
        let row_signs = q.signs.slice(..rows);
        let col_signs = q.signs.slice(rows..);
        let config = LaunchConfig {
            grid_dim: ((rows / CHUNK) as u32, q.splits, 1),
            block_dim: (THREADS, 1, 1),
            shared_mem_bytes: q.shared_bytes,
        };
        let mut args = stream.launch_builder(&variant.function);
        args.arg(&t.bytes)
            .arg(x)
            .arg(&col_signs)
            .arg(&row_signs)
            .arg(y)
            .arg(&t.width)
            .arg(&q.tiles_per_split)
            .arg(&q.scale);
        // Signature of qtip_matvec_fused; shapes are 256-aligned (gguf checks)
        // and the split satisfies its chunk constraint (tiles_per_split).
        unsafe { args.launch(config) }.map_err(failure)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A QTIP tensor of `kind` with random trellis words (every bitstream
    /// decodes) and signs, with its companion, as `gguf` folds it.
    fn random_tensor(
        kind: u32,
        cols: usize,
        rows: usize,
        seed: u64,
        stream: &Arc<CudaStream>,
    ) -> DeviceTensor {
        let mut state = seed;
        let mut next = move || {
            state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = state;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        };
        let tiles = rows * cols / (TILE * TILE) * qtip_tile_bytes(kind).expect("a QTIP kind");
        let mut bytes: Vec<u8> = (0..tiles).map(|_| next() as u8).collect();
        let sign = |v: u64| if v & 1 == 0 { 1.0f32 } else { -1.0 };
        let companion = std::iter::once(0.02f32).chain((0..rows + cols).map(|_| sign(next())));
        bytes.extend(companion.flat_map(f32::to_le_bytes));
        let tensor = crate::gguf::Tensor {
            dims: vec![cols, rows],
            storage: crate::gguf::Storage::Packed { kind, bytes },
            accelerated: true,
            block_elements: 256,
        };
        DeviceTensor::upload(tensor, stream).unwrap()
    }

    #[test]
    #[ignore = "requires CUDA (sm_80+); run serially in release mode"]
    fn batched_matmul_matches_the_decode_matvec() -> Result<()> {
        let context = CudaContext::new(0).map_err(failure)?;
        let stream = context.default_stream();
        let major = context
            .attribute(sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)
            .map_err(failure)?;
        let minor = context
            .attribute(sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)
            .map_err(failure)?;
        // q/k/v, gate and down shapes, then Llama 3.1 8B ffn_down (56 chunks,
        // not a power of two); 130 tokens cross a 128-token block.
        // Every shape as QTIP3, then as QTIP2 (one kernel variant each).
        let shapes = [(2048, 512), (2048, 8192), (8192, 2048), (14336, 4096)];
        let kinds = [crate::gguf::QTIP3_KIND, crate::gguf::QTIP2_KIND];
        let shapes: Vec<(usize, usize)> = kinds.iter().flat_map(|_| shapes).collect();
        let tensors: Vec<DeviceTensor> = shapes
            .iter()
            .enumerate()
            .map(|(i, &(c, r))| {
                let kind = kinds[i / (shapes.len() / kinds.len())];
                random_tensor(kind, c, r, i as u64 + 1, &stream)
            })
            .collect();
        let mut kernel = QtipKernel::load(&context, (major, minor), None, tensors.iter())?;
        // Exact path, then int8-fast (looser bound).
        let cases = [(false, 1e-3), (true, 0.02)];
        for ((tensor, &(cols, rows)), &(fast, bound)) in tensors
            .iter()
            .zip(&shapes)
            .flat_map(|t| cases.iter().map(move |c| (t, c)))
        {
            for tokens in [3, 130] {
                let host: Vec<f32> = (0..tokens * cols)
                    .map(|i| ((i * 7919 % 2003) as f32 - 1001.0) / 500.0)
                    .collect();
                let x = stream.clone_htod(&host).map_err(failure)?;
                let mut y = stream.alloc_zeros::<f32>(tokens * rows).map_err(failure)?;
                kernel.matmul(&stream, tensor, &x, &mut y, tokens, fast)?;
                let batched = stream.clone_dtoh(&y).map_err(failure)?;
                let (mut error, mut norm) = (0.0f64, 0.0f64);
                for t in 0..tokens {
                    let xt = stream
                        .clone_htod(&host[t * cols..(t + 1) * cols])
                        .map_err(failure)?;
                    let mut yt = stream.alloc_zeros::<f32>(rows).map_err(failure)?;
                    kernel.matvec(&stream, tensor, &xt, &mut yt)?;
                    let reference = stream.clone_dtoh(&yt).map_err(failure)?;
                    for (a, b) in batched[t * rows..(t + 1) * rows].iter().zip(&reference) {
                        error += f64::from(a - b).powi(2);
                        norm += f64::from(*b).powi(2);
                    }
                }
                // About 2e-4 exact, 1.2% int8-fast.
                let relative = (error / norm).sqrt();
                assert!(
                    relative < bound,
                    "{cols}x{rows}, {tokens} tokens, fast {fast}: {relative}"
                );
            }
        }
        Ok(())
    }

    #[test]
    fn splits_cover_whole_chunks() {
        for (cols, rows) in [
            (2048, 512),
            (2048, 2048),
            (2048, 8192),
            (8192, 2048),
            (16384, 4096),
            (14336, 4096),
            (4096, 14336),
        ] {
            let col_tiles = cols / TILE;
            let per = tiles_per_split(col_tiles, rows / CHUNK);
            let span = per * TILE;
            assert!(
                CHUNK.is_multiple_of(span) || span.is_multiple_of(CHUNK),
                "{cols}x{rows}: {span}"
            );
            assert!(per <= col_tiles.next_power_of_two().max(CHUNK / TILE));
        }
    }
}
