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
    /// `(pair_bits, skip_in, skip_out)` -> `(k8_plain, k8_swiglu, k16_plain, k16_swiglu)`.
    kernels: HashMap<(u8, bool, bool), (CudaFunction, CudaFunction, CudaFunction, CudaFunction)>,
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
        let state_bits: u32 = index
            .header
            .flags
            .ne(&0)
            .then(|| {
                let sec = index
                    .sections
                    .iter()
                    .find(|s| s.kind == crate::plt1::SectionKind::MetadataJson)?;
                file.seek(SeekFrom::Start(sec.offset)).ok()?;
                let mut buf = vec![0u8; sec.size as usize];
                file.read_exact(&mut buf).ok()?;
                let v: serde_json::Value = serde_json::from_slice(&buf).ok()?;
                v.get("state_bits")?.as_u64().map(|n| n as u32)
            })
            .flatten()
            .unwrap_or(12);

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
            let compile_max_k = |max_k: u32, want_r1: bool| -> Result<(CudaFunction, CudaFunction, Option<(CudaFunction, CudaFunction)>)> {
                let src = format!(
                    "#define HYB_L {state_bits}\n#define HYB_B {pair_bits}\n#define HYB_Q 5\n#define HYB_UNROLL 4\n\
                     #define HYB_SKIP_IN_RHT {}\n#define HYB_SKIP_OUT_RHT {}\n\
                     #define HYB_MAX_K {max_k}\n#define HYB_VEC_K 8\n{}",
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
                let r1 = if want_r1 {
                    Some((
                        module.load_function("hyb_r1_enter").map_err(cuda_err)?,
                        module.load_function("hyb_r1_exit").map_err(cuda_err)?,
                    ))
                } else {
                    None
                };
                Ok((
                    module.load_function("hyb_small_k_multi").map_err(cuda_err)?,
                    module
                        .load_function("hyb_small_k_multi_swiglu")
                        .map_err(cuda_err)?,
                    r1,
                ))
            };
            let (k8_p, k8_s, r1) = compile_max_k(8, r1_fns.is_none())?;
            if r1_fns.is_none() {
                r1_fns = r1;
            }
            let (k16_p, k16_s, _) = compile_max_k(16, false)?;
            kernels.insert((pair_bits, skip_in, skip_out), (k8_p, k8_s, k16_p, k16_s));
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
    /// When `zero_self` is false, caller guarantees outputs were already zeroed by the
    /// previous projection's `zero_next` pointers (eliminating 224 memset_d8_async calls/step).
    #[allow(clippy::too_many_arguments)]
    pub fn run_multi(
        &self,
        matrices: &[(&Plt1GpuTensor, &mut CudaSlice<f32>)],
        x: &CudaSlice<f32>,
        x_up: Option<&CudaSlice<f32>>,
        tokens: usize,
        zero_next: &[(&mut CudaSlice<f32>, usize)],
        zero_self: bool,
    ) -> Result<()> {
        let first = matrices[0].0;
        let skip_in = (first.desc.flags & tensor_flags::SKIP_IN_RHT) != 0;
        let skip_out = (first.desc.flags & tensor_flags::SKIP_OUT_RHT) != 0;
        let (k8_p, k8_s, k16_p, k16_s) = &self.kernels[&(first.desc.pair_bits, skip_in, skip_out)];
        let func = match (tokens <= 8, x_up.is_some()) {
            (true, false) => k8_p,
            (true, true) => k8_s,
            (false, false) => k16_p,
            (false, true) => k16_s,
        };

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
            if zero_self {
                unsafe {
                    cudarc::driver::result::memset_d8_async(
                        y_ptr,
                        0,
                        tokens * r * std::mem::size_of::<f32>(),
                        self.stream.cu_stream(),
                    )
                    .map_err(cuda_err)?;
                }
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
        let mut zs = [0u64; 3];
        let mut ns = [0i32; 3];
        for (i, (buf, count)) in zero_next.iter().take(3).enumerate() {
            zs[i] = buf.device_ptr(&self.stream).0;
            ns[i] = *count as i32;
        }

        let rb0 = rbs[0] as i32;
        let rb1 = if matrices.len() >= 2 {
            rbs[1] as i32
        } else {
            total_rb as i32
        };
        let total_chunks = cols / 256;
        // Fit exact 2-wave multiples on RTX 5090's 170 SMs (avoid 3rd tail wave on Gate+Up!).
        let splits = (340usize / total_rb).clamp(1, total_chunks);
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
        b.arg(&zs[0]).arg(&zs[1]).arg(&zs[2]);
        b.arg(&ns[0]).arg(&ns[1]).arg(&ns[2]);
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::qtip::{
        hyb::{HybCodebook, HybQuantizer, decode_hyb_tile_mma_l, swizzle_tile_to_mma},
        incoherence::Incoherence,
    };

    #[test]
    fn hyb_small_k_multi_matches_cpu_with_folded_rht() {
        let ctx = CudaContext::new(0).unwrap();
        let stream = ctx.default_stream();
        let (rows, cols) = (256usize, 256usize);
        let cb = HybCodebook::gaussian_for_q(5);
        let q = HybQuantizer::new(12, 4, cb.clone()).unwrap();
        let w: Vec<f32> = (0..rows * cols)
            .map(|i| ((i * 7919 % 1000) as f32 - 500.0) / 500.0)
            .collect();
        let rht = Incoherence::with_block(rows, cols, 42, Some(256)).unwrap();
        let w_rot = rht.forward(&w);
        let sigma = (w_rot.iter().map(|v| v * v).sum::<f32>() / w_rot.len() as f32).sqrt();

        let mut tiles_u32 = Vec::new();
        let mut recon_rot = vec![0.0f32; rows * cols];
        for g in 0..(rows / 16) {
            for k in 0..(cols / 16) {
                let mut tile = [0.0f32; 256];
                for r in 0..16 {
                    for c in 0..16 {
                        tile[r * 16 + c] = w_rot[(g * 16 + r) * cols + k * 16 + c];
                    }
                }
                let swz = swizzle_tile_to_mma(&tile);
                let norm: Vec<f32> = swz.iter().map(|&v| v / sigma).collect();
                let words = q.quantize_normalized(&norm).unwrap();
                let dec = decode_hyb_tile_mma_l(&words, 12, 4, &cb, sigma);
                for r in 0..16 {
                    for c in 0..16 {
                        recon_rot[(g * 16 + r) * cols + k * 16 + c] = dec[r * 16 + c];
                    }
                }
                tiles_u32.extend_from_slice(&words);
            }
        }
        let w_rec = rht.inverse(&recon_rot);

        // Compile hyb_small_k_multi with SKIP_IN=1, SKIP_OUT=0 (like QKV/GateUp).
        let src = format!(
            "#define HYB_L 12\n#define HYB_B 4\n#define HYB_Q 5\n#define HYB_UNROLL 4\n\
             #define HYB_SKIP_IN_RHT 1\n#define HYB_SKIP_OUT_RHT 0\n\
             #define HYB_MAX_K 8\n#define HYB_VEC_K 8\n{}",
            include_str!("../qtip/hyb_matvec.cu")
        );
        let ptx = compile_ptx_with_opts(
            src,
            CompileOptions {
                arch: Some("compute_120"),
                ..Default::default()
            },
        )
        .unwrap();
        let module = ctx.load_module(ptx).unwrap();
        let r1_enter = module.load_function("hyb_r1_enter").unwrap();
        let func = module.load_function("hyb_small_k_multi").unwrap();

        let x: Vec<f32> = (0..cols).map(|i| ((i * 31 % 200) as f32 - 100.0) / 100.0).collect();
        let mut d_x = stream.clone_htod(&x).unwrap();
        let d_r1 = stream.clone_htod(rht.col_signs()).unwrap();
        let cols_i = cols as i32;
        unsafe {
            let mut b = stream.launch_builder(&r1_enter);
            b.arg(&mut d_x).arg(&d_r1).arg(&cols_i);
            b.launch(LaunchConfig {
                grid_dim: (1, 1, 1),
                block_dim: (32, 1, 1),
                shared_mem_bytes: 0,
            })
            .unwrap();
        }

        let d_tiles = stream.clone_htod(&tiles_u32).unwrap();
        let d_lut = stream.clone_htod(&cb.lut_f16x2).unwrap();
        let d_rs = stream.clone_htod(rht.row_signs()).unwrap();
        let d_y = stream.alloc_zeros::<f32>(rows).unwrap();
        let t_ptr = d_tiles.device_ptr(&stream).0;
        let l_ptr = d_lut.device_ptr(&stream).0;
        let cs_ptr = d_r1.device_ptr(&stream).0;
        let rs_ptr = d_rs.device_ptr(&stream).0;
        let y_ptr = d_y.device_ptr(&stream).0;
        let x_ptr = d_x.device_ptr(&stream).0;
        let scale = sigma / 16.0;
        let (rows_i, rb0, tokens_i, cps_i, zero_u64, zero_i) = (rows as i32, 1i32, 1i32, 1i32, 0u64, 0i32);
        unsafe {
            let mut b = stream.launch_builder(&func);
            b.arg(&t_ptr).arg(&t_ptr).arg(&t_ptr);
            b.arg(&l_ptr).arg(&l_ptr).arg(&l_ptr);
            b.arg(&cs_ptr).arg(&cs_ptr).arg(&cs_ptr);
            b.arg(&rs_ptr).arg(&rs_ptr).arg(&rs_ptr);
            b.arg(&y_ptr).arg(&y_ptr).arg(&y_ptr);
            b.arg(&scale).arg(&scale).arg(&scale);
            b.arg(&rows_i).arg(&rows_i).arg(&rows_i);
            b.arg(&rb0).arg(&rb0).arg(&x_ptr).arg(&zero_u64);
            b.arg(&zero_u64).arg(&zero_u64).arg(&zero_u64);
            b.arg(&zero_i).arg(&zero_i).arg(&zero_i);
            b.arg(&cols_i).arg(&tokens_i).arg(&cps_i);
            b.launch(LaunchConfig {
                grid_dim: (1, 1, 1),
                block_dim: (512, 1, 1),
                shared_mem_bytes: 0,
            })
            .unwrap();
        }
        let gpu_y = stream.clone_dtoh(&d_y).unwrap();
        let mut cpu_y = vec![0.0f32; rows];
        for r in 0..rows {
            cpu_y[r] = (0..cols).map(|c| w_rec[r * cols + c] * x[c]).sum();
        }
        let max_err = gpu_y
            .iter()
            .zip(&cpu_y)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        let max_ref = cpu_y.iter().map(|v| v.abs()).fold(1e-6f32, f32::max);
        let rel = max_err / max_ref;
        assert!(rel < 2e-3, "hyb_small_k_multi mismatch: rel={rel:.4e} (gpu[0]={:.4}, cpu[0]={:.4})", gpu_y[0], cpu_y[0]);
    }

    #[test]
    fn hyb_small_k_multi_swiglu_and_skip_out_match_cpu() {
        let ctx = CudaContext::new(0).unwrap();
        let stream = ctx.default_stream();
        let (rows, cols, tokens) = (256usize, 512usize, 6usize);
        let cb = HybCodebook::gaussian_for_q(5);
        let q = HybQuantizer::new(12, 4, cb.clone()).unwrap();
        let w: Vec<f32> = (0..rows * cols)
            .map(|i| ((i * 7919 % 1000) as f32 - 500.0) / 500.0)
            .collect();
        let rht = Incoherence::with_block(rows, cols, 99, Some(256)).unwrap();
        let w_rot = rht.forward(&w);
        let sigma = (w_rot.iter().map(|v| v * v).sum::<f32>() / w_rot.len() as f32).sqrt();

        let mut tiles_u32 = Vec::new();
        let mut recon_rot = vec![0.0f32; rows * cols];
        for g in 0..(rows / 16) {
            for k in 0..(cols / 16) {
                let mut tile = [0.0f32; 256];
                for r in 0..16 {
                    for c in 0..16 {
                        tile[r * 16 + c] = w_rot[(g * 16 + r) * cols + k * 16 + c];
                    }
                }
                let swz = swizzle_tile_to_mma(&tile);
                let norm: Vec<f32> = swz.iter().map(|&v| v / sigma).collect();
                let words = q.quantize_normalized(&norm).unwrap();
                let dec = decode_hyb_tile_mma_l(&words, 12, 4, &cb, sigma);
                for r in 0..16 {
                    for c in 0..16 {
                        recon_rot[(g * 16 + r) * cols + k * 16 + c] = dec[r * 16 + c];
                    }
                }
                tiles_u32.extend_from_slice(&words);
            }
        }
        let w_rec = rht.inverse(&recon_rot);

        // Compile hyb_small_k_multi_swiglu with SKIP_IN=0, SKIP_OUT=1 (like Down_proj).
        let src = format!(
            "#define HYB_L 12\n#define HYB_B 4\n#define HYB_Q 5\n#define HYB_UNROLL 4\n\
             #define HYB_SKIP_IN_RHT 0\n#define HYB_SKIP_OUT_RHT 1\n\
             #define HYB_MAX_K 8\n#define HYB_VEC_K 8\n{}",
            include_str!("../qtip/hyb_matvec.cu")
        );
        let ptx = compile_ptx_with_opts(
            src,
            CompileOptions {
                arch: Some("compute_120"),
                ..Default::default()
            },
        )
        .unwrap();
        let module = ctx.load_module(ptx).unwrap();
        let r1_exit = module.load_function("hyb_r1_exit").unwrap();
        let func = module.load_function("hyb_small_k_multi_swiglu").unwrap();

        let x_gate: Vec<f32> = (0..tokens * cols)
            .map(|i| ((i * 31 % 200) as f32 - 100.0) / 100.0)
            .collect();
        let x_up: Vec<f32> = (0..tokens * cols)
            .map(|i| ((i * 53 % 200) as f32 - 100.0) / 100.0)
            .collect();
        let d_xg = stream.clone_htod(&x_gate).unwrap();
        let d_xu = stream.clone_htod(&x_up).unwrap();
        let d_tiles = stream.clone_htod(&tiles_u32).unwrap();
        let d_lut = stream.clone_htod(&cb.lut_f16x2).unwrap();
        let d_cs = stream.clone_htod(rht.col_signs()).unwrap();
        let d_rs = stream.clone_htod(rht.row_signs()).unwrap();
        let mut d_y = stream.alloc_zeros::<f32>(tokens * rows).unwrap();

        let t_ptr = d_tiles.device_ptr(&stream).0;
        let l_ptr = d_lut.device_ptr(&stream).0;
        let cs_ptr = d_cs.device_ptr(&stream).0;
        let rs_ptr = d_rs.device_ptr(&stream).0;
        let y_ptr = d_y.device_ptr(&stream).0;
        let xg_ptr = d_xg.device_ptr(&stream).0;
        let xu_ptr = d_xu.device_ptr(&stream).0;
        let scale = sigma / 16.0;
        let (rows_i, cols_i, rb0, tokens_i, cps_i, zero_u64, zero_i) =
            (rows as i32, cols as i32, 1i32, tokens as i32, 1i32, 0u64, 0i32);
        unsafe {
            let mut b = stream.launch_builder(&func);
            b.arg(&t_ptr).arg(&t_ptr).arg(&t_ptr);
            b.arg(&l_ptr).arg(&l_ptr).arg(&l_ptr);
            b.arg(&cs_ptr).arg(&cs_ptr).arg(&cs_ptr);
            b.arg(&rs_ptr).arg(&rs_ptr).arg(&rs_ptr);
            b.arg(&y_ptr).arg(&y_ptr).arg(&y_ptr);
            b.arg(&scale).arg(&scale).arg(&scale);
            b.arg(&rows_i).arg(&rows_i).arg(&rows_i);
            b.arg(&rb0).arg(&rb0).arg(&xg_ptr).arg(&xu_ptr);
            b.arg(&zero_u64).arg(&zero_u64).arg(&zero_u64);
            b.arg(&zero_i).arg(&zero_i).arg(&zero_i);
            b.arg(&cols_i).arg(&tokens_i).arg(&cps_i);
            b.launch(LaunchConfig {
                grid_dim: (1, 2, 1), // 2 splits!
                block_dim: (512, 1, 1),
                shared_mem_bytes: 0,
            })
            .unwrap();
            let mut b2 = stream.launch_builder(&r1_exit);
            b2.arg(&mut d_y).arg(&d_rs).arg(&rows_i);
            b2.launch(LaunchConfig {
                grid_dim: (1, tokens as u32, 1),
                block_dim: (32, 1, 1),
                shared_mem_bytes: 0,
            })
            .unwrap();
        }
        let gpu_y = stream.clone_dtoh(&d_y).unwrap();
        let mut cpu_y = vec![0.0f32; tokens * rows];
        for t in 0..tokens {
            for r in 0..rows {
                cpu_y[t * rows + r] = (0..cols)
                    .map(|c| {
                        let g = x_gate[t * cols + c];
                        let u = x_up[t * cols + c];
                        let act = g / (1.0 + (-g).exp()) * u;
                        w_rec[r * cols + c] * act
                    })
                    .sum();
            }
        }
        let max_err = gpu_y
            .iter()
            .zip(&cpu_y)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        let max_ref = cpu_y.iter().map(|v| v.abs()).fold(1e-6f32, f32::max);
        let rel = max_err / max_ref;
        assert!(
            rel < 2e-3,
            "swiglu/skip_out mismatch: rel={rel:.4e} (gpu[0]={:.4}, cpu[0]={:.4})",
            gpu_y[0],
            cpu_y[0]
        );
    }

    #[test]
    fn hyb_small_k_multi_b6_l10_matches_cpu() {
        let ctx = CudaContext::new(0).unwrap();
        let stream = ctx.default_stream();
        let (rows, cols, tokens) = (256usize, 512usize, 6usize);
        let cb = HybCodebook::gaussian_for_q(5);
        let q = HybQuantizer::new(10, 6, cb.clone()).unwrap();
        let w: Vec<f32> = (0..rows * cols)
            .map(|i| ((i * 7919 % 1000) as f32 - 500.0) / 500.0)
            .collect();
        let rht = Incoherence::with_block(rows, cols, 77, Some(256)).unwrap();
        let w_rot = rht.forward(&w);
        let sigma = (w_rot.iter().map(|v| v * v).sum::<f32>() / w_rot.len() as f32).sqrt();

        let mut tiles_u32 = Vec::new();
        let mut recon_rot = vec![0.0f32; rows * cols];
        for g in 0..(rows / 16) {
            for k in 0..(cols / 16) {
                let mut tile = [0.0f32; 256];
                for r in 0..16 {
                    for c in 0..16 {
                        tile[r * 16 + c] = w_rot[(g * 16 + r) * cols + k * 16 + c];
                    }
                }
                let swz = swizzle_tile_to_mma(&tile);
                let norm: Vec<f32> = swz.iter().map(|&v| v / sigma).collect();
                let words = q.quantize_normalized(&norm).unwrap();
                let dec = decode_hyb_tile_mma_l(&words, 10, 6, &cb, sigma);
                for r in 0..16 {
                    for c in 0..16 {
                        recon_rot[(g * 16 + r) * cols + k * 16 + c] = dec[r * 16 + c];
                    }
                }
                tiles_u32.extend_from_slice(&words);
            }
        }
        let w_rec = rht.inverse(&recon_rot);

        let src = format!(
            "#define HYB_L 10\n#define HYB_B 6\n#define HYB_Q 5\n#define HYB_UNROLL 4\n\
             #define HYB_SKIP_IN_RHT 1\n#define HYB_SKIP_OUT_RHT 0\n\
             #define HYB_MAX_K 8\n#define HYB_VEC_K 8\n{}",
            include_str!("../qtip/hyb_matvec.cu")
        );
        let ptx = compile_ptx_with_opts(
            src,
            CompileOptions {
                arch: Some("compute_120"),
                ..Default::default()
            },
        )
        .unwrap();
        let module = ctx.load_module(ptx).unwrap();
        let r1_enter = module.load_function("hyb_r1_enter").unwrap();
        let func = module.load_function("hyb_small_k_multi").unwrap();

        let x: Vec<f32> = (0..tokens * cols)
            .map(|i| ((i * 37 % 200) as f32 - 100.0) / 100.0)
            .collect();
        let mut d_x = stream.clone_htod(&x).unwrap();
        let d_r1 = stream.clone_htod(rht.col_signs()).unwrap();
        let cols_i = cols as i32;
        unsafe {
            let mut b = stream.launch_builder(&r1_enter);
            b.arg(&mut d_x).arg(&d_r1).arg(&cols_i);
            b.launch(LaunchConfig {
                grid_dim: ((cols / 256) as u32, tokens as u32, 1),
                block_dim: (32, 1, 1),
                shared_mem_bytes: 0,
            })
            .unwrap();
        }
        let d_tiles = stream.clone_htod(&tiles_u32).unwrap();
        let d_lut = stream.clone_htod(&cb.lut_f16x2).unwrap();
        let d_rs = stream.clone_htod(rht.row_signs()).unwrap();
        let d_y = stream.alloc_zeros::<f32>(tokens * rows).unwrap();
        let t_ptr = d_tiles.device_ptr(&stream).0;
        let l_ptr = d_lut.device_ptr(&stream).0;
        let cs_ptr = d_r1.device_ptr(&stream).0;
        let rs_ptr = d_rs.device_ptr(&stream).0;
        let y_ptr = d_y.device_ptr(&stream).0;
        let x_ptr = d_x.device_ptr(&stream).0;
        let scale = sigma / 16.0;
        let (rows_i, rb0, tokens_i, cps_i, zero_u64, zero_i) =
            (rows as i32, 1i32, tokens as i32, 1i32, 0u64, 0i32);
        unsafe {
            let mut b = stream.launch_builder(&func);
            b.arg(&t_ptr).arg(&t_ptr).arg(&t_ptr);
            b.arg(&l_ptr).arg(&l_ptr).arg(&l_ptr);
            b.arg(&cs_ptr).arg(&cs_ptr).arg(&cs_ptr);
            b.arg(&rs_ptr).arg(&rs_ptr).arg(&rs_ptr);
            b.arg(&y_ptr).arg(&y_ptr).arg(&y_ptr);
            b.arg(&scale).arg(&scale).arg(&scale);
            b.arg(&rows_i).arg(&rows_i).arg(&rows_i);
            b.arg(&rb0).arg(&rb0).arg(&x_ptr).arg(&zero_u64);
            b.arg(&zero_u64).arg(&zero_u64).arg(&zero_u64);
            b.arg(&zero_i).arg(&zero_i).arg(&zero_i);
            b.arg(&cols_i).arg(&tokens_i).arg(&cps_i);
            b.launch(LaunchConfig {
                grid_dim: (1, 2, 1),
                block_dim: (512, 1, 1),
                shared_mem_bytes: 0,
            })
            .unwrap();
        }
        let gpu_y = stream.clone_dtoh(&d_y).unwrap();
        let mut cpu_y = vec![0.0f32; tokens * rows];
        for t in 0..tokens {
            for r in 0..rows {
                cpu_y[t * rows + r] = (0..cols)
                    .map(|c| w_rec[r * cols + c] * x[t * cols + c])
                    .sum();
            }
        }
        let max_err = gpu_y
            .iter()
            .zip(&cpu_y)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        let max_ref = cpu_y.iter().map(|v| v.abs()).fold(1e-6f32, f32::max);
        let rel = max_err / max_ref;
        assert!(rel < 2e-3, "B=6 L=10 mismatch: rel={rel:.4e}");
    }
}
