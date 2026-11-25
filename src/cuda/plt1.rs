//! GPU-native `PLT1` (`.palito`) HYB `Q=5` tensor loader & full-layer dispatcher
//! for Blackwell SM_120 (RTX 5090).
//!
//! Uploads 128B-aligned MMA-swizzled HYB tiles, their 32-entry `half2` warp-register
//! LUTs, per-tensor sign vectors, and optional shared `R_1` residual-stream signs
//! directly to CUDA VRAM. Executes all 7 projections per layer in 4 fused launches
//! (`QKV`, `O`, `Gate+Up`, `Down+SwiGLU`) for `1 <= tokens <= 16`.

use crate::{
    Error, Result,
    plt1::{Codec, Plt1Index, TensorDesc, flags, tensor_flags},
};
use cudarc::{
    driver::{
        CudaContext, CudaFunction, CudaSlice, CudaStream, DevicePtr, LaunchConfig, PushKernelArg,
    },
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
    /// `[row_signs (rows), col_signs (cols)]` as F32.
    pub signs: CudaSlice<f32>,
}

pub struct Plt1Layer {
    pub q: Plt1GpuTensor,
    pub k: Plt1GpuTensor,
    pub v: Plt1GpuTensor,
    pub o: Plt1GpuTensor,
    pub gate: Plt1GpuTensor,
    pub up: Plt1GpuTensor,
    pub down: Plt1GpuTensor,
}

pub struct Plt1HybRuntime {
    pub index: Plt1Index,
    pub layers: Vec<Plt1Layer>,
    pub r1_signs: Option<CudaSlice<f32>>,
    pub folded_rmsnorm: bool,
    stream: Arc<CudaStream>,
    r1_enter_fn: CudaFunction,
    r1_exit_fn: CudaFunction,
    /// `(pair_bits, skip_in, skip_out)` -> `(small_k_multi, small_k_multi_swiglu)`.
    kernels: HashMap<(u8, bool, bool), (CudaFunction, CudaFunction)>,
}

impl Plt1HybRuntime {
    pub fn open(path: &Path, ctx: &Arc<CudaContext>, stream: Arc<CudaStream>) -> Result<Self> {
        let mut file = File::open(path)?;
        let index = Plt1Index::read_from(&mut file)?;

        let read_f32_tensor = |file: &mut File, name: &str| -> Result<Option<CudaSlice<f32>>> {
            let Some(desc) = index.tensors.iter().find(|t| t.name == name) else {
                return Ok(None);
            };
            file.seek(SeekFrom::Start(desc.data_offset))?;
            let mut raw = vec![0u8; desc.data_size as usize];
            file.read_exact(&mut raw)?;
            let vals: Vec<f32> = raw
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                .collect();
            Ok(Some(stream.clone_htod(&vals).map_err(cuda_err)?))
        };

        let r1_signs = read_f32_tensor(&mut file, "plt1.r1_signs")?;
        let folded_rmsnorm = (index.header.flags & flags::FOLDED_RMSNORM) != 0;

        let mut kernel_keys = Vec::new();
        let mut load_hyb = |file: &mut File, name: &str| -> Result<Plt1GpuTensor> {
            let desc = index
                .tensors
                .iter()
                .find(|t| t.name == name && t.codec == Codec::QtipHybQ5)
                .ok_or_else(|| Error::Invalid(format!("missing PLT1 HYB Q=5 tensor {name}")))?
                .clone();
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

            let signs_name = format!("{name}.signs");
            let signs = match read_f32_tensor(file, &signs_name)? {
                Some(s) => s,
                None => stream
                    .clone_htod(&vec![1.0f32; (desc.rows + desc.cols) as usize])
                    .map_err(cuda_err)?,
            };

            let skip_in = (desc.flags & tensor_flags::SKIP_IN_RHT) != 0;
            let skip_out = (desc.flags & tensor_flags::SKIP_OUT_RHT) != 0;
            let key = (desc.pair_bits, skip_in, skip_out);
            if !kernel_keys.contains(&key) {
                kernel_keys.push(key);
            }
            Ok(Plt1GpuTensor {
                desc,
                tiles,
                lut,
                signs,
            })
        };

        let n_layers = index.header.n_layers as usize;
        let mut layers = Vec::with_capacity(n_layers);
        for l in 0..n_layers {
            layers.push(Plt1Layer {
                q: load_hyb(&mut file, &format!("blk.{l}.attn_q.weight"))?,
                k: load_hyb(&mut file, &format!("blk.{l}.attn_k.weight"))?,
                v: load_hyb(&mut file, &format!("blk.{l}.attn_v.weight"))?,
                o: load_hyb(&mut file, &format!("blk.{l}.attn_output.weight"))?,
                gate: load_hyb(&mut file, &format!("blk.{l}.ffn_gate.weight"))?,
                up: load_hyb(&mut file, &format!("blk.{l}.ffn_up.weight"))?,
                down: load_hyb(&mut file, &format!("blk.{l}.ffn_down.weight"))?,
            });
        }

        let mut kernels = HashMap::new();
        let mut r1_fns = None;
        for (pair_bits, skip_in, skip_out) in kernel_keys {
            let src = format!(
                "#define HYB_L 12\n#define HYB_B {pair_bits}\n#define HYB_Q 5\n#define HYB_UNROLL 4\n\
                 #define HYB_SKIP_IN_RHT {}\n#define HYB_SKIP_OUT_RHT {}\n\
                 #define HYB_VEC_K 8\n{}",
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
            if r1_fns.is_none() {
                r1_fns = Some((
                    module.load_function("hyb_r1_enter").map_err(cuda_err)?,
                    module.load_function("hyb_r1_exit").map_err(cuda_err)?,
                ));
            }
            kernels.insert(
                (pair_bits, skip_in, skip_out),
                (
                    module.load_function("hyb_small_k_multi").map_err(cuda_err)?,
                    module
                        .load_function("hyb_small_k_multi_swiglu")
                        .map_err(cuda_err)?,
                ),
            );
        }
        let (r1_enter_fn, r1_exit_fn) =
            r1_fns.ok_or_else(|| Error::Invalid("PLT1 file contained no HYB layers".into()))?;

        Ok(Self {
            index,
            layers,
            r1_signs,
            folded_rmsnorm,
            stream,
            r1_enter_fn,
            r1_exit_fn,
            kernels,
        })
    }

    pub fn r1_enter(&self, x: &mut CudaSlice<f32>, cols: usize, tokens: usize) -> Result<()> {
        let Some(r1) = &self.r1_signs else {
            return Ok(());
        };
        let cols_i = cols as i32;
        let mut b = self.stream.launch_builder(&self.r1_enter_fn);
        b.arg(x).arg(r1).arg(&cols_i);
        unsafe {
            b.launch(LaunchConfig {
                grid_dim: ((cols / 256) as u32, tokens as u32, 1),
                block_dim: (32, 1, 1),
                shared_mem_bytes: 0,
            })
        }
        .map_err(cuda_err)?;
        Ok(())
    }

    pub fn r1_exit(&self, x: &mut CudaSlice<f32>, cols: usize, tokens: usize) -> Result<()> {
        let Some(r1) = &self.r1_signs else {
            return Ok(());
        };
        let cols_i = cols as i32;
        let mut b = self.stream.launch_builder(&self.r1_exit_fn);
        b.arg(x).arg(r1).arg(&cols_i);
        unsafe {
            b.launch(LaunchConfig {
                grid_dim: ((cols / 256) as u32, tokens as u32, 1),
                block_dim: (32, 1, 1),
                shared_mem_bytes: 0,
            })
        }
        .map_err(cuda_err)?;
        Ok(())
    }

    /// Runs 1..=3 HYB Q=5 projections sharing input `x` (and optional `x_up` for SwiGLU)
    /// for `1 <= tokens <= 16`.
    pub fn run_multi(
        &self,
        matrices: &[(&Plt1GpuTensor, &mut CudaSlice<f32>)],
        x: &CudaSlice<f32>,
        x_up: Option<&CudaSlice<f32>>,
        tokens: usize,
    ) -> Result<()> {
        let first = matrices[0].0;
        let skip_in = (first.desc.flags & tensor_flags::SKIP_IN_RHT) != 0;
        let skip_out = (first.desc.flags & tensor_flags::SKIP_OUT_RHT) != 0;
        let (fn_plain, fn_swiglu) = &self.kernels[&(first.desc.pair_bits, skip_in, skip_out)];
        let func = if x_up.is_some() { fn_swiglu } else { fn_plain };

        let cols = first.desc.cols as usize;
        let mut tiles = [0u64; 3];
        let mut luts = [0u64; 3];
        let mut col_signs = [0u64; 3];
        let mut row_signs = [0u64; 3];
        let mut ys = [0u64; 3];
        let mut scales = [0.0f32; 3];
        let mut rows_arr = [0i32; 3];
        let mut rbs = [0usize; 3];
        let mut total_rb = 0usize;

        for (i, (t, y)) in matrices.iter().enumerate() {
            let r = t.desc.rows as usize;
            let y_ptr = y.device_ptr(&self.stream).0;
            unsafe {
                cudarc::driver::result::memset_d8_async(
                    y_ptr,
                    0,
                    tokens * r * std::mem::size_of::<f32>(),
                    self.stream.cu_stream(),
                )
                .map_err(cuda_err)?;
            }
            tiles[i] = t.tiles.device_ptr(&self.stream).0;
            luts[i] = t.lut.device_ptr(&self.stream).0;
            let s_ptr = t.signs.device_ptr(&self.stream).0;
            row_signs[i] = s_ptr;
            col_signs[i] = s_ptr + (r * std::mem::size_of::<f32>()) as u64;
            ys[i] = y_ptr;
            scales[i] = t.desc.scale;
            rows_arr[i] = r as i32;
            rbs[i] = r / 256;
            total_rb += rbs[i];
        }
        let rb0 = rbs[0] as i32;
        let rb1 = if matrices.len() >= 2 {
            rbs[1] as i32
        } else {
            total_rb as i32
        };
        let total_chunks = cols / 256;
        let splits = 340usize.div_ceil(total_rb).clamp(1, total_chunks);
        let cps = total_chunks.div_ceil(splits);
        let splits = total_chunks.div_ceil(cps);

        let x_ptr = x.device_ptr(&self.stream).0;
        let x_up_ptr = x_up.map_or(0u64, |u| u.device_ptr(&self.stream).0);
        let (cols_i, tokens_i, cps_i) = (cols as i32, tokens as i32, cps as i32);

        let mut b = self.stream.launch_builder(func);
        for g in [&tiles, &luts, &col_signs, &row_signs, &ys] {
            b.arg(&g[0]).arg(&g[1]).arg(&g[2]);
        }
        b.arg(&scales[0]).arg(&scales[1]).arg(&scales[2]);
        b.arg(&rows_arr[0]).arg(&rows_arr[1]).arg(&rows_arr[2]);
        b.arg(&rb0).arg(&rb1).arg(&x_ptr).arg(&x_up_ptr);
        b.arg(&cols_i).arg(&tokens_i).arg(&cps_i);
        unsafe {
            b.launch(LaunchConfig {
                grid_dim: (total_rb as u32, splits as u32, 1),
                block_dim: (512, 1, 1),
                shared_mem_bytes: 0,
            })
        }
        .map_err(cuda_err)?;
        Ok(())
    }
}
