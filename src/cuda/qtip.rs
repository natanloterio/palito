//! QTIP3 projections (feature `qtip-research`): the fused trellis matvec of
//! `src/qtip/matvec.cu`, which applies both 256-block Hadamards itself, so a
//! QTIP3 projection is one memset and one launch.

use super::{DeviceTensor, cache, failure};
use crate::{
    Result,
    gguf::{QTIP3_KIND, QTIP3_TILE_BYTES},
    transformer::invalid,
};
use cudarc::{
    driver::{CudaContext, CudaFunction, CudaSlice, CudaStream, LaunchConfig, PushKernelArg, sys},
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

pub(super) struct QtipKernel {
    function: CudaFunction,
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
        let shared = tensors
            .filter_map(|t| t.qtip.as_ref())
            .map(|q| q.shared_bytes)
            .max()
            .unwrap_or(0);
        if shared > STATIC_SHARED_LIMIT {
            function
                .set_attribute(
                    sys::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                    shared as i32,
                )
                .map_err(failure)?;
        }
        Ok(Self { function })
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
