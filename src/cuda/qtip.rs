//! QTIP3 projections (feature `qtip-research`): the fused trellis matvec of
//! `src/qtip/matvec.cu`, which applies both 256-block Hadamards itself. A
//! single projection is one memset and one launch (`matvec`); a Llama layer
//! made only of QTIP3 projections runs in four launches without memsets
//! (`multi`: q/k/v, output, gate/up, and down with SwiGLU fused in).

use super::{DeviceTensor, cache, failure};
use crate::{
    Result,
    gguf::{QTIP3_KIND, QTIP3_TILE_BYTES},
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
        cols: usize,
        rows: usize,
        stream: &Arc<CudaStream>,
    ) -> Result<Self> {
        let tile_bytes = rows * cols / (TILE * TILE) * QTIP3_TILE_BYTES;
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

pub(super) struct QtipKernel {
    function: CudaFunction,
    multi: CudaFunction,
    multi_swiglu: CudaFunction,
    rotate_input: CudaFunction,
    rotate_output: CudaFunction,
    mma: CudaFunction,
    /// Prefill scratch, grown on demand: rotated FP16 inputs and the rotated
    /// F32 product.
    rotated_input: Option<CudaSlice<u16>>,
    rotated_output: Option<CudaSlice<f32>>,
}

impl QtipKernel {
    /// Compiles the fused kernel (L = 12, K = 3) and opts it into the dynamic
    /// shared memory of the widest split among `tensors`.
    pub(super) fn load<'a>(
        context: &Arc<CudaContext>,
        architecture: (i32, i32),
        cache_dir: Option<&Path>,
        tensors: impl Iterator<Item = &'a DeviceTensor>,
    ) -> Result<Self> {
        let source = format!(
            "#define QTIP_L 12\n#define QTIP_K 3\n#define QTIP_UNROLL 4\n\
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
        let function = module.load_function("qtip_matvec_fused").map_err(failure)?;
        let multi = module.load_function("qtip_matvec_multi").map_err(failure)?;
        let multi_swiglu = module
            .load_function("qtip_matvec_multi_swiglu")
            .map_err(failure)?;
        let rotate_input = module.load_function("qtip_rotate_input").map_err(failure)?;
        let rotate_output = module
            .load_function("qtip_rotate_output")
            .map_err(failure)?;
        let mma = module.load_function("qtip_matmul_mma").map_err(failure)?;
        let tensors: Vec<&DeviceTensor> = tensors.filter(|t| t.qtip.is_some()).collect();
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
            (&function, shared),
            (&multi, multi_shared),
            (&multi_swiglu, multi_shared),
        ] {
            if bytes > STATIC_SHARED_LIMIT {
                f.set_attribute(
                    sys::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                    bytes as i32,
                )
                .map_err(failure)?;
            }
        }
        Ok(Self {
            function,
            multi,
            multi_swiglu,
            rotate_input,
            rotate_output,
            mma,
            rotated_input: None,
            rotated_output: None,
        })
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
                .filter(|_| t.kind == QTIP3_KIND as i32 && t.width as usize == cols)
                .ok_or_else(|| invalid("multi launch needs QTIP3 tensors of one width"))?;
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
        let function = if x_up != 0 {
            &self.multi_swiglu
        } else {
            &self.multi
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

    /// `y[t * rows + r] = (W x_t)[r]` for `tokens` token-major inputs (prefill),
    /// on FP16 tensor cores: the inputs are rounded to FP16 after rotation,
    /// the weights are exact, sums are F32. Needs sm_80+ and rows % 64 == 0.
    pub(super) fn matmul(
        &mut self,
        stream: &Arc<CudaStream>,
        tensor: &DeviceTensor,
        x: &CudaSlice<f32>,
        y: &mut CudaSlice<f32>,
        tokens: usize,
    ) -> Result<()> {
        let q = tensor
            .qtip
            .as_ref()
            .ok_or_else(|| invalid("QTIP3 matmul needs a QTIP3 tensor"))?;
        let (rows, cols) = (tensor.rows as usize, tensor.width as usize);
        if rows % 64 != 0 || cols % CHUNK != 0 {
            return Err(invalid("QTIP3 matmul needs rows % 64 and cols % 256 == 0"));
        }
        if x.len() < tokens * cols || y.len() < tokens * rows {
            return Err(invalid("QTIP3 matmul buffers are too small"));
        }
        let col_tiles = cols / TILE;
        let row_block = 16 * MATMUL_WARPS;
        if rows % row_block != 0 {
            return Err(invalid(
                "QTIP3 matmul rows must be a multiple of the row block",
            ));
        }
        let blocks = (rows / row_block) * tokens.div_ceil(MATMUL_TOKENS);
        let split_tiles = matmul_split_tiles(col_tiles, blocks, tokens * rows);
        let splits = col_tiles.div_ceil(split_tiles);
        let grow = |len: usize, need: usize| len < need;
        if self
            .rotated_input
            .as_ref()
            .is_none_or(|b| grow(b.len(), tokens * cols))
        {
            self.rotated_input = Some(stream.alloc_zeros(tokens * cols).map_err(failure)?);
        }
        let planes = splits * tokens * rows;
        if self
            .rotated_output
            .as_ref()
            .is_none_or(|b| grow(b.len(), planes))
        {
            self.rotated_output = Some(stream.alloc_zeros(planes).map_err(failure)?);
        }
        let (xr, yr) = (
            self.rotated_input.as_mut().unwrap(),
            self.rotated_output.as_mut().unwrap(),
        );
        let col_signs = q.signs.slice(rows..);
        let (cols_i, rows_i, tokens_i) = (cols as i32, rows as i32, tokens as i32);
        let chunks = |n: usize| (n / CHUNK).div_ceil(8) as u32;
        let rotate = |n: usize| LaunchConfig {
            grid_dim: (chunks(n), tokens as u32, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut args = stream.launch_builder(&self.rotate_input);
        args.arg(x).arg(&col_signs).arg(&mut *xr).arg(&cols_i);
        // Bounds checked above; x and xr hold tokens * cols values.
        unsafe { args.launch(rotate(cols)) }.map_err(failure)?;
        let mut args = stream.launch_builder(&self.mma);
        args.arg(&tensor.bytes).arg(&*xr).arg(&mut *yr);
        let (split_tiles_i, splits_i) = (split_tiles as i32, splits as i32);
        args.arg(&cols_i)
            .arg(&rows_i)
            .arg(&tokens_i)
            .arg(&split_tiles_i);
        let config = LaunchConfig {
            grid_dim: (
                (rows / row_block) as u32,
                tokens.div_ceil(MATMUL_TOKENS) as u32,
                splits as u32,
            ),
            block_dim: (32 * MATMUL_WARPS as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        // tensor.bytes holds rows / 16 row groups of cols / 16 tiles; yr
        // holds `splits` planes of tokens * rows values.
        unsafe { args.launch(config) }.map_err(failure)?;
        let scale16 = q.scale * 16.0;
        let mut args = stream.launch_builder(&self.rotate_output);
        args.arg(&*yr)
            .arg(&q.signs)
            .arg(y)
            .arg(&rows_i)
            .arg(&scale16);
        args.arg(&splits_i).arg(&tokens_i);
        // Bounds checked above; row signs are the first rows values.
        unsafe { args.launch(rotate(rows)) }.map_err(failure)?;
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
            .filter(|_| t.kind == QTIP3_KIND as i32)
            .ok_or_else(|| invalid("not a QTIP3 tensor"))?;
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
        let mut args = stream.launch_builder(&self.function);
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

    /// A QTIP3 tensor of random trellis words (every bitstream decodes) and
    /// signs, with its companion, as `gguf` folds it.
    fn random_tensor(
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
        let tiles = rows * cols / (TILE * TILE) * QTIP3_TILE_BYTES;
        let mut bytes: Vec<u8> = (0..tiles).map(|_| next() as u8).collect();
        let sign = |v: u64| if v & 1 == 0 { 1.0f32 } else { -1.0 };
        let companion = std::iter::once(0.02f32).chain((0..rows + cols).map(|_| sign(next())));
        bytes.extend(companion.flat_map(f32::to_le_bytes));
        let tensor = crate::gguf::Tensor {
            dims: vec![cols, rows],
            storage: crate::gguf::Storage::Packed {
                kind: QTIP3_KIND,
                bytes,
            },
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
        // q/k/v, gate and down shapes; 130 tokens cross a 128-token block.
        let shapes = [(2048, 512), (2048, 8192), (8192, 2048)];
        let tensors: Vec<DeviceTensor> = shapes
            .iter()
            .enumerate()
            .map(|(i, &(c, r))| random_tensor(c, r, i as u64 + 1, &stream))
            .collect();
        let mut kernel = QtipKernel::load(&context, (major, minor), None, tensors.iter())?;
        for (tensor, &(cols, rows)) in tensors.iter().zip(&shapes) {
            for tokens in [3, 130] {
                let host: Vec<f32> = (0..tokens * cols)
                    .map(|i| ((i * 7919 % 2003) as f32 - 1001.0) / 500.0)
                    .collect();
                let x = stream.clone_htod(&host).map_err(failure)?;
                let mut y = stream.alloc_zeros::<f32>(tokens * rows).map_err(failure)?;
                kernel.matmul(&stream, tensor, &x, &mut y, tokens)?;
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
                // FP16 inputs after rotation, exact weights: about 2e-4.
                let relative = (error / norm).sqrt();
                assert!(
                    relative < 1e-3,
                    "{cols}x{rows}, {tokens} tokens: {relative}"
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
