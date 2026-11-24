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

fn pointer<T>(slice: &CudaSlice<T>, stream: &CudaStream) -> u64 {
    slice.device_ptr(stream).0
}

pub(super) struct QtipKernel {
    function: CudaFunction,
    multi: CudaFunction,
    multi_swiglu: CudaFunction,
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
            "#define QTIP_L 12\n#define QTIP_K 3\n#define QTIP_UNROLL 4\n{}",
            include_str!("../qtip/matvec.cu")
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
