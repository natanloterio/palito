//! GPU-native `PLT1` (`.palito`) HYB `Q=5` tensor loader & dispatcher for Blackwell SM_120.
//!
//! Uploads 128B-aligned MMA-swizzled HYB tiles and their 32-entry `half2` warp-register
//! LUTs directly from a `.palito` file to CUDA VRAM, dispatching:
//! - `hyb_matvec_fused` for `B=1` decode (zero-SMEM `Q=5` warp-register LUT + folded RHT)
//! - `hyb_matvec_mma` for `k = 8, 16` speculative verification (direct `mma.sync.aligned.m16n8k16`)

use crate::{
    Error, Result,
    plt1::{Codec, Plt1Index, TensorDesc, tensor_flags},
};
use cudarc::{
    driver::{CudaContext, CudaFunction, CudaSlice, CudaStream, LaunchConfig, PushKernelArg},
    nvrtc::{CompileOptions, compile_ptx_with_opts},
};
use std::{
    collections::HashMap,
    fs::File,
    io::{Read, Seek, SeekFrom},
    path::Path,
    sync::Arc,
};

fn cuda_err(e: impl std::fmt::Display) -> Error {
    Error::Backend(format!("CUDA PLT1: {e}"))
}

pub struct Plt1GpuTensor {
    pub desc: TensorDesc,
    pub tiles: CudaSlice<u32>,
    pub lut: CudaSlice<u32>,
}

pub struct Plt1HybRuntime {
    pub index: Plt1Index,
    pub tensors: HashMap<String, Plt1GpuTensor>,
    stream: Arc<CudaStream>,
    /// Compiled `(pair_bits, skip_in, skip_out)` -> `(fused_fn, mma8_fn, mma16_fn)`.
    kernels: HashMap<(u8, bool, bool), (CudaFunction, CudaFunction, CudaFunction)>,
}

impl Plt1HybRuntime {
    /// Opens a `.palito` (`PLT1`) file, uploads all `QtipHybQ5` tensors + LUTs to
    /// GPU VRAM, and compiles the specialized SM_120 HYB `Q=5` kernels.
    pub fn open(path: &Path, ctx: &Arc<CudaContext>) -> Result<Self> {
        let mut file = File::open(path)?;
        let index = Plt1Index::read_from(&mut file)?;
        let stream = ctx.default_stream();
        let mut tensors = HashMap::new();
        let mut kernel_keys = Vec::new();

        for desc in &index.tensors {
            if desc.codec != Codec::QtipHybQ5 {
                continue;
            }
            file.seek(SeekFrom::Start(desc.data_offset))?;
            let mut raw = vec![0u8; desc.data_size as usize];
            file.read_exact(&mut raw)?;
            let words: Vec<u32> = raw
                .chunks_exact(4)
                .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
                .collect();
            let tiles = stream.clone_htod(&words).map_err(cuda_err)?;

            let lut_entries = 1usize << desc.lut_bits;
            file.seek(SeekFrom::Start(desc.lut_offset))?;
            let mut lut_raw = vec![0u8; lut_entries * 4];
            file.read_exact(&mut lut_raw)?;
            let lut_words: Vec<u32> = lut_raw
                .chunks_exact(4)
                .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
                .collect();
            let lut = stream.clone_htod(&lut_words).map_err(cuda_err)?;

            let skip_in = (desc.flags & tensor_flags::SKIP_IN_RHT) != 0;
            let skip_out = (desc.flags & tensor_flags::SKIP_OUT_RHT) != 0;
            let key = (desc.pair_bits, skip_in, skip_out);
            if !kernel_keys.contains(&key) {
                kernel_keys.push(key);
            }
            tensors.insert(
                desc.name.clone(),
                Plt1GpuTensor {
                    desc: desc.clone(),
                    tiles,
                    lut,
                },
            );
        }

        let mut kernels = HashMap::new();
        for (pair_bits, skip_in, skip_out) in kernel_keys {
            let compile_for = |vec_k: u32| -> Result<(CudaFunction, CudaFunction)> {
                let src = format!(
                    "#define HYB_B {pair_bits}\n#define HYB_Q 5\n#define HYB_UNROLL 4\n\
                     #define HYB_SKIP_IN_RHT {}\n#define HYB_SKIP_OUT_RHT {}\n\
                     #define HYB_VEC_K {vec_k}\n{}",
                    usize::from(skip_in),
                    usize::from(skip_out),
                    include_str!("../qtip/hyb_matvec.cu")
                );
                let ptx = compile_ptx_with_opts(
                    src,
                    CompileOptions {
                        arch: Some("compute_120"),
                        ..Default::default()
                    },
                )
                .map_err(cuda_err)?;
                let module = ctx.load_module(ptx).map_err(cuda_err)?;
                Ok((
                    module.load_function("hyb_matvec_fused").map_err(cuda_err)?,
                    module.load_function("hyb_matvec_mma").map_err(cuda_err)?,
                ))
            };
            let (fused, mma8) = compile_for(8)?;
            let (_, mma16) = compile_for(16)?;
            kernels.insert((pair_bits, skip_in, skip_out), (fused, mma8, mma16));
        }

        Ok(Self {
            index,
            tensors,
            stream,
            kernels,
        })
    }

    /// Runs `hyb_matvec_mma` (`k = 8` or `16` tokens) on a loaded PLT1 HYB `Q=5` tensor.
    /// `x_h2`: `[k, cols/2]` packed FP16x2, `y`: `[k, rows]` FP32 (zeroed internally).
    pub fn matvec_mma(
        &self,
        name: &str,
        x_h2: &CudaSlice<u32>,
        y: &mut CudaSlice<f32>,
        k: usize,
    ) -> Result<()> {
        if k != 8 && k != 16 {
            return Err(Error::Invalid("PLT1 matvec_mma supports k=8 or k=16".into()));
        }
        let t = self
            .tensors
            .get(name)
            .ok_or_else(|| Error::Invalid(format!("PLT1 tensor {name} not found")))?;
        let skip_in = (t.desc.flags & tensor_flags::SKIP_IN_RHT) != 0;
        let skip_out = (t.desc.flags & tensor_flags::SKIP_OUT_RHT) != 0;
        let (_, mma8, mma16) = &self.kernels[&(t.desc.pair_bits, skip_in, skip_out)];
        let func = if k == 8 { mma8 } else { mma16 };

        let rows = t.desc.rows as usize;
        let cols = t.desc.cols as usize;
        let row_blocks = rows / 256;
        let col_tiles = cols / 16;
        let splits = (340usize / row_blocks).clamp(1, col_tiles);
        let tps = col_tiles.div_ceil(splits);
        let splits = col_tiles.div_ceil(tps);

        let mut y_slice = y.slice_mut(0..k * rows);
        self.stream.memset_zeros(&mut y_slice).map_err(cuda_err)?;

        let (rows_i, cols_i, tps_i) = (rows as i32, cols as i32, tps as i32);
        let mut b = self.stream.launch_builder(func);
        b.arg(&t.tiles)
            .arg(&t.lut)
            .arg(x_h2)
            .arg(&mut y_slice)
            .arg(&rows_i)
            .arg(&cols_i)
            .arg(&tps_i)
            .arg(&t.desc.scale);
        unsafe {
            b.launch(LaunchConfig {
                grid_dim: (row_blocks as u32, splits as u32, 1),
                block_dim: (512, 1, 1),
                shared_mem_bytes: 0,
            })
        }
        .map_err(cuda_err)?;
        Ok(())
    }
}
