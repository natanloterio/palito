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
    /// Per-expert scales `[n_experts]` for 3D MoE tensors; `None` for dense 2D tensors.
    pub scales: Option<CudaSlice<f32>>,
}

pub struct Plt1Layer {
    pub q: Plt1GpuTensor,
    pub k: Plt1GpuTensor,
    pub v: Plt1GpuTensor,
    pub o: Plt1GpuTensor,
    pub gate: Plt1GpuTensor,
    pub up: Plt1GpuTensor,
    pub down: Plt1GpuTensor,
    pub moe_ffn: bool,
}

pub struct Plt1HybRuntime {
    pub index: Plt1Index,
    pub layers: Vec<Plt1Layer>,
    pub r1_signs: Option<CudaSlice<f32>>,
    pub folded_rmsnorm: bool,
    stream: Arc<CudaStream>,
    r1_enter_fn: CudaFunction,
    r1_exit_fn: CudaFunction,
    /// `(pair_bits, skip_in, skip_out)` -> `(k8_plain, k8_swiglu, k16_plain, k16_swiglu, moe_plain, moe_swiglu)`.
    kernels: HashMap<
        (u8, bool, bool),
        (
            CudaFunction,
            CudaFunction,
            CudaFunction,
            CudaFunction,
            CudaFunction,
            CudaFunction,
        ),
    >,
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
            let scales = read_f32_tensor(file, &format!("{name}.scales"))?;

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
                scales,
            })
        };

        let n_layers = index.header.n_layers as usize;
        let mut layers = Vec::with_capacity(n_layers);
        for l in 0..n_layers {
            let moe_ffn = index
                .tensors
                .iter()
                .any(|t| t.name == format!("blk.{l}.ffn_gate_exps.weight"));
            let (gate, up, down) = if moe_ffn {
                (
                    load_hyb(&mut file, &format!("blk.{l}.ffn_gate_exps.weight"))?,
                    load_hyb(&mut file, &format!("blk.{l}.ffn_up_exps.weight"))?,
                    load_hyb(&mut file, &format!("blk.{l}.ffn_down_exps.weight"))?,
                )
            } else {
                (
                    load_hyb(&mut file, &format!("blk.{l}.ffn_gate.weight"))?,
                    load_hyb(&mut file, &format!("blk.{l}.ffn_up.weight"))?,
                    load_hyb(&mut file, &format!("blk.{l}.ffn_down.weight"))?,
                )
            };
            layers.push(Plt1Layer {
                q: load_hyb(&mut file, &format!("blk.{l}.attn_q.weight"))?,
                k: load_hyb(&mut file, &format!("blk.{l}.attn_k.weight"))?,
                v: load_hyb(&mut file, &format!("blk.{l}.attn_v.weight"))?,
                o: load_hyb(&mut file, &format!("blk.{l}.attn_output.weight"))?,
                gate,
                up,
                down,
                moe_ffn,
            });
        }

        let mut kernels = HashMap::new();
        let mut r1_fns = None;
        for (pair_bits, skip_in, skip_out) in kernel_keys {
            let compile_max_k = |max_k: u32,
                                 want_r1: bool,
                                 want_moe: bool|
             -> Result<(
                CudaFunction,
                CudaFunction,
                Option<(CudaFunction, CudaFunction)>,
                Option<(CudaFunction, CudaFunction)>,
            )> {
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
                let moe = if want_moe {
                    Some((
                        module.load_function("hyb_moe_small_k").map_err(cuda_err)?,
                        module
                            .load_function("hyb_moe_small_k_swiglu")
                            .map_err(cuda_err)?,
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
                    moe,
                ))
            };
            let (k8_p, k8_s, r1, moe) = compile_max_k(8, r1_fns.is_none(), true)?;
            if r1_fns.is_none() {
                r1_fns = r1;
            }
            let (moe_p, moe_s) = moe.unwrap();
            let (k16_p, k16_s, _, _) = compile_max_k(16, false, false)?;
            kernels.insert(
                (pair_bits, skip_in, skip_out),
                (k8_p, k8_s, k16_p, k16_s, moe_p, moe_s),
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
        let (k8_p, k8_s, k16_p, k16_s, _, _) =
            &self.kernels[&(first.desc.pair_bits, skip_in, skip_out)];
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

    /// Runs fused MoE Gate + Up HYB Q=5 projections for `tokens * used` active expert assignments.
    #[allow(clippy::too_many_arguments)]
    pub fn run_moe_gate_up(
        &self,
        gate: &Plt1GpuTensor,
        up: &Plt1GpuTensor,
        x: &CudaSlice<f32>,
        ids: &CudaSlice<i32>,
        y_gate: &mut CudaSlice<f32>,
        y_up: &mut CudaSlice<f32>,
        tokens: usize,
        used: usize,
    ) -> Result<()> {
        let skip_in = (gate.desc.flags & tensor_flags::SKIP_IN_RHT) != 0;
        let skip_out = (gate.desc.flags & tensor_flags::SKIP_OUT_RHT) != 0;
        let (_, _, _, _, moe_p, _) = &self.kernels[&(gate.desc.pair_bits, skip_in, skip_out)];
        let r0 = gate.desc.rows as usize;
        let r1 = up.desc.rows as usize;
        let cols = gate.desc.cols as i32;
        let rb0 = (r0 / 256) as i32;
        let total_rb = (r0 / 256 + r1 / 256) as u32;
        let assignments = (tokens * used) as u32;
        let used_i = used as i32;
        let (rows0_i, rows1_i) = (r0 as i32, r1 as i32);

        let t0 = gate.tiles.device_ptr(&self.stream).0;
        let t1 = up.tiles.device_ptr(&self.stream).0;
        let l0 = gate.lut.device_ptr(&self.stream).0;
        let l1 = up.lut.device_ptr(&self.stream).0;
        let sc0 = gate
            .scales
            .as_ref()
            .ok_or_else(|| Error::Invalid("MoE gate missing .scales".into()))?
            .device_ptr(&self.stream)
            .0;
        let sc1 = up
            .scales
            .as_ref()
            .ok_or_else(|| Error::Invalid("MoE up missing .scales".into()))?
            .device_ptr(&self.stream)
            .0;
        let s0 = gate.signs.device_ptr(&self.stream).0;
        let s1 = up.signs.device_ptr(&self.stream).0;
        let c0 = s0 + (r0 * std::mem::size_of::<f32>()) as u64;
        let c1 = s1 + (r1 * std::mem::size_of::<f32>()) as u64;
        let y0 = y_gate.device_ptr(&self.stream).0;
        let y1 = y_up.device_ptr(&self.stream).0;
        let x_ptr = x.device_ptr(&self.stream).0;
        let ids_ptr = ids.device_ptr(&self.stream).0;

        let mut b = self.stream.launch_builder(moe_p);
        b.arg(&t0).arg(&t1).arg(&l0).arg(&l1);
        b.arg(&sc0).arg(&sc1).arg(&c0).arg(&c1).arg(&s0).arg(&s1);
        b.arg(&y0).arg(&y1).arg(&rows0_i).arg(&rows1_i).arg(&rb0);
        b.arg(&x_ptr).arg(&ids_ptr).arg(&cols).arg(&used_i);
        unsafe {
            b.launch(LaunchConfig {
                grid_dim: (total_rb, assignments, 1),
                block_dim: (512, 1, 1),
                shared_mem_bytes: 0,
            })
        }
        .map_err(cuda_err)?;
        Ok(())
    }

    /// Runs fused SwiGLU + MoE Down HYB Q=5 projection for `tokens * used` active assignments.
    #[allow(clippy::too_many_arguments)]
    pub fn run_moe_down(
        &self,
        down: &Plt1GpuTensor,
        x_gate: &CudaSlice<f32>,
        x_up: &CudaSlice<f32>,
        ids: &CudaSlice<i32>,
        y_down: &mut CudaSlice<f32>,
        tokens: usize,
        used: usize,
    ) -> Result<()> {
        let skip_in = (down.desc.flags & tensor_flags::SKIP_IN_RHT) != 0;
        let skip_out = (down.desc.flags & tensor_flags::SKIP_OUT_RHT) != 0;
        let (_, _, _, _, _, moe_s) = &self.kernels[&(down.desc.pair_bits, skip_in, skip_out)];
        let r0 = down.desc.rows as usize;
        let cols = down.desc.cols as i32;
        let total_rb = (r0 / 256) as u32;
        let assignments = (tokens * used) as u32;
        let used_i = used as i32;
        let rows0_i = r0 as i32;

        let t0 = down.tiles.device_ptr(&self.stream).0;
        let l0 = down.lut.device_ptr(&self.stream).0;
        let sc0 = down
            .scales
            .as_ref()
            .ok_or_else(|| Error::Invalid("MoE down missing .scales".into()))?
            .device_ptr(&self.stream)
            .0;
        let s0 = down.signs.device_ptr(&self.stream).0;
        let c0 = s0 + (r0 * std::mem::size_of::<f32>()) as u64;
        let y0 = y_down.device_ptr(&self.stream).0;
        let xg = x_gate.device_ptr(&self.stream).0;
        let xu = x_up.device_ptr(&self.stream).0;
        let ids_ptr = ids.device_ptr(&self.stream).0;

        let mut b = self.stream.launch_builder(moe_s);
        b.arg(&t0).arg(&l0).arg(&sc0).arg(&c0).arg(&s0).arg(&y0);
        b.arg(&rows0_i).arg(&xg).arg(&xu).arg(&ids_ptr).arg(&cols).arg(&used_i);
        unsafe {
            b.launch(LaunchConfig {
                grid_dim: (total_rb, assignments, 1),
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

    #[test]
    fn hyb_moe_small_k_and_swiglu_match_cpu() {
        let ctx = CudaContext::new(0).unwrap();
        let stream = ctx.default_stream();
        let (n_experts, used, tokens, d_model, d_ff) = (4usize, 2usize, 3usize, 512usize, 256usize);
        let cb = HybCodebook::gaussian_for_q(5);
        let q = HybQuantizer::new(10, 6, cb.clone()).unwrap();

        let quant_moe = |rows: usize,
                         cols: usize,
                         seed: u64,
                         r1: Option<Vec<f32>>,
                         r1_is_col: bool| {
            let base = Incoherence::with_block(rows, cols, seed, Some(256)).unwrap();
            let rht = match r1 {
                Some(s) if r1_is_col => {
                    Incoherence::with_signs(rows, cols, base.row_signs().to_vec(), s, Some(256))
                        .unwrap()
                }
                Some(s) => {
                    Incoherence::with_signs(rows, cols, s, base.col_signs().to_vec(), Some(256))
                        .unwrap()
                }
                None => base,
            };
            let mut all_tiles = Vec::new();
            let mut scales = Vec::new();
            let mut recs = Vec::new();
            for e in 0..n_experts {
                let w: Vec<f32> = (0..rows * cols)
                    .map(|i| (((i * 7919 + e * 104729) % 1000) as f32 - 500.0) / 500.0)
                    .collect();
                let w_rot = rht.forward(&w);
                let sigma = (w_rot.iter().map(|v| v * v).sum::<f32>() / w_rot.len() as f32).sqrt();
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
                        all_tiles.extend_from_slice(&words);
                    }
                }
                recs.push(rht.inverse(&recon_rot));
                scales.push(sigma / 16.0);
            }
            (rht, all_tiles, scales, recs)
        };

        let r1_signs = Incoherence::with_block(d_model, d_model, 111, Some(256))
            .unwrap()
            .col_signs()
            .to_vec();
        let (rht_g, tiles_g, sc_g, rec_g) =
            quant_moe(d_ff, d_model, 222, Some(r1_signs.clone()), true);
        let (rht_u, tiles_u, sc_u, rec_u) =
            quant_moe(d_ff, d_model, 333, Some(r1_signs.clone()), true);
        let (rht_d, tiles_d, sc_d, rec_d) =
            quant_moe(d_model, d_ff, 444, Some(r1_signs.clone()), false);

        let compile = |skip_in: usize, skip_out: usize| {
            let src = format!(
                "#define HYB_L 10\n#define HYB_B 6\n#define HYB_Q 5\n#define HYB_UNROLL 4\n\
                 #define HYB_SKIP_IN_RHT {skip_in}\n#define HYB_SKIP_OUT_RHT {skip_out}\n\
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
            ctx.load_module(ptx).unwrap()
        };
        let mod_in = compile(1, 0);
        let mod_out = compile(0, 1);
        let r1_enter = mod_in.load_function("hyb_r1_enter").unwrap();
        let r1_exit = mod_out.load_function("hyb_r1_exit").unwrap();
        let f_gu = mod_in.load_function("hyb_moe_small_k").unwrap();
        let f_down = mod_out.load_function("hyb_moe_small_k_swiglu").unwrap();

        let x: Vec<f32> = (0..tokens * d_model)
            .map(|i| ((i * 29 % 200) as f32 - 100.0) / 100.0)
            .collect();
        let ids: Vec<i32> = vec![0, 2, 1, 3, 2, 0];
        let mut d_x = stream.clone_htod(&x).unwrap();
        let d_r1 = stream.clone_htod(&r1_signs).unwrap();
        let d_ids = stream.clone_htod(&ids).unwrap();
        let d_lut = stream.clone_htod(&cb.lut_f16x2).unwrap();

        let d_tg = stream.clone_htod(&tiles_g).unwrap();
        let d_tu = stream.clone_htod(&tiles_u).unwrap();
        let d_td = stream.clone_htod(&tiles_d).unwrap();
        let d_sg = stream.clone_htod(&sc_g).unwrap();
        let d_su = stream.clone_htod(&sc_u).unwrap();
        let d_sd = stream.clone_htod(&sc_d).unwrap();
        let d_rg = stream.clone_htod(rht_g.row_signs()).unwrap();
        let d_ru = stream.clone_htod(rht_u.row_signs()).unwrap();
        let d_cd = stream.clone_htod(rht_d.col_signs()).unwrap();

        let d_yg = stream.alloc_zeros::<f32>(tokens * used * d_ff).unwrap();
        let d_yu = stream.alloc_zeros::<f32>(tokens * used * d_ff).unwrap();
        let mut d_yd = stream.alloc_zeros::<f32>(tokens * used * d_model).unwrap();

        let (d_model_i, d_ff_i, used_i, rb0_i) =
            (d_model as i32, d_ff as i32, used as i32, (d_ff / 256) as i32);
        unsafe {
            let mut b = stream.launch_builder(&r1_enter);
            b.arg(&mut d_x).arg(&d_r1).arg(&d_model_i);
            b.launch(LaunchConfig {
                grid_dim: ((d_model / 256) as u32, tokens as u32, 1),
                block_dim: (32, 1, 1),
                shared_mem_bytes: 0,
            })
            .unwrap();

            let (tg, tu, td) = (
                d_tg.device_ptr(&stream).0,
                d_tu.device_ptr(&stream).0,
                d_td.device_ptr(&stream).0,
            );
            let lut = d_lut.device_ptr(&stream).0;
            let (sg, su, sd) = (
                d_sg.device_ptr(&stream).0,
                d_su.device_ptr(&stream).0,
                d_sd.device_ptr(&stream).0,
            );
            let (rg, ru, cd, r1p) = (
                d_rg.device_ptr(&stream).0,
                d_ru.device_ptr(&stream).0,
                d_cd.device_ptr(&stream).0,
                d_r1.device_ptr(&stream).0,
            );
            let (yg, yu, yd) = (
                d_yg.device_ptr(&stream).0,
                d_yu.device_ptr(&stream).0,
                d_yd.device_ptr(&stream).0,
            );
            let (xp, idsp) = (d_x.device_ptr(&stream).0, d_ids.device_ptr(&stream).0);

            let mut b1 = stream.launch_builder(&f_gu);
            b1.arg(&tg).arg(&tu).arg(&lut).arg(&lut);
            b1.arg(&sg).arg(&su).arg(&r1p).arg(&r1p).arg(&rg).arg(&ru);
            b1.arg(&yg).arg(&yu).arg(&d_ff_i).arg(&d_ff_i).arg(&rb0_i);
            b1.arg(&xp).arg(&idsp).arg(&d_model_i).arg(&used_i);
            b1.launch(LaunchConfig {
                grid_dim: ((2 * d_ff / 256) as u32, (tokens * used) as u32, 1),
                block_dim: (512, 1, 1),
                shared_mem_bytes: 0,
            })
            .unwrap();

            let mut b2 = stream.launch_builder(&f_down);
            b2.arg(&td).arg(&lut).arg(&sd).arg(&cd).arg(&r1p).arg(&yd);
            b2.arg(&d_model_i).arg(&yg).arg(&yu).arg(&idsp).arg(&d_ff_i).arg(&used_i);
            b2.launch(LaunchConfig {
                grid_dim: ((d_model / 256) as u32, (tokens * used) as u32, 1),
                block_dim: (512, 1, 1),
                shared_mem_bytes: 0,
            })
            .unwrap();

            let mut b3 = stream.launch_builder(&r1_exit);
            b3.arg(&mut d_yd).arg(&d_r1).arg(&d_model_i);
            b3.launch(LaunchConfig {
                grid_dim: ((d_model / 256) as u32, (tokens * used) as u32, 1),
                block_dim: (32, 1, 1),
                shared_mem_bytes: 0,
            })
            .unwrap();
        }
        let gpu_yd = stream.clone_dtoh(&d_yd).unwrap();
        let mut cpu_yd = vec![0.0f32; tokens * used * d_model];
        for assign in 0..(tokens * used) {
            let tok = assign / used;
            let e = ids[assign] as usize;
            let mut g_out = vec![0.0f32; d_ff];
            let mut u_out = vec![0.0f32; d_ff];
            for r in 0..d_ff {
                g_out[r] = (0..d_model)
                    .map(|c| rec_g[e][r * d_model + c] * x[tok * d_model + c])
                    .sum();
                u_out[r] = (0..d_model)
                    .map(|c| rec_u[e][r * d_model + c] * x[tok * d_model + c])
                    .sum();
            }
            for r in 0..d_model {
                cpu_yd[assign * d_model + r] = (0..d_ff)
                    .map(|c| {
                        let act = g_out[c] / (1.0 + (-g_out[c]).exp()) * u_out[c];
                        rec_d[e][r * d_ff + c] * act
                    })
                    .sum();
            }
        }
        let max_err = gpu_yd
            .iter()
            .zip(&cpu_yd)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        let max_ref = cpu_yd.iter().map(|v| v.abs()).fold(1e-6f32, f32::max);
        let rel = max_err / max_ref;
        assert!(rel < 2e-3, "MoE HYB B=6 L=10 mismatch: rel={rel:.4e}");
    }
}
