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
    sync::{
        Arc,
        atomic::{AtomicU64, AtomicUsize, Ordering},
    },
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

const MOE_MAX_ASSIGNS: usize = 256;

pub struct Plt1HybRuntime {
    pub index: Plt1Index,
    pub layers: Vec<Plt1Layer>,
    pub r1_signs: Option<CudaSlice<f32>>,
    pub folded_rmsnorm: bool,
    stream: Arc<CudaStream>,
    r1_enter_fn: CudaFunction,
    r1_exit_fn: CudaFunction,
    moe_group_fn: CudaFunction,
    splitk_reduce_fn: CudaFunction,
    /// Scratch buffer for deterministic split-K reduction (`[splits, tokens_or_assigns, total_rows]`).
    splitk_scratch: CudaSlice<f32>,
    moe_group_experts: CudaSlice<i32>,
    moe_group_offsets: CudaSlice<i32>,
    moe_group_assigns: CudaSlice<i32>,
    moe_num_groups: CudaSlice<i32>,
    /// Tracks `(ids_ptr, total_assigns)` grouped by `run_moe_gate_up` so the paired
    /// `run_moe_down` skips redundant re-grouping on the same layer.
    last_moe_group: (AtomicU64, AtomicUsize),
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
                Option<(CudaFunction, CudaFunction, CudaFunction, CudaFunction)>,
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
                        module
                            .load_function("hyb_moe_group_assignments")
                            .map_err(cuda_err)?,
                        module
                            .load_function("hyb_splitk_reduce_multi")
                            .map_err(cuda_err)?,
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
        let (r1_enter_fn, r1_exit_fn, moe_group_fn, splitk_reduce_fn) =
            r1_fns.ok_or_else(|| Error::Invalid("PLT1 file contained no HYB layers".into()))?;
        let compute_splits = |denom: usize, total_chunks: usize| -> usize {
            let s = (340usize / denom.max(1)).clamp(1, total_chunks);
            let cps = total_chunks.div_ceil(s);
            total_chunks.div_ceil(cps)
        };
        let mut max_scratch_elems = 1usize;
        for l in &layers {
            let dense_cases = [
                (
                    (l.q.desc.rows + l.k.desc.rows + l.v.desc.rows) as usize,
                    l.q.desc.cols as usize,
                ),
                (l.o.desc.rows as usize, l.o.desc.cols as usize),
            ];
            for (total_rows, cols) in dense_cases {
                let splits = compute_splits(total_rows / 256, cols / 256);
                if splits > 1 {
                    max_scratch_elems = max_scratch_elems.max(splits * 16 * total_rows);
                }
            }
            if l.moe_ffn {
                let gu_rows = (l.gate.desc.rows + l.up.desc.rows) as usize;
                let gu_cols = l.gate.desc.cols as usize;
                let d_rows = l.down.desc.rows as usize;
                let d_cols = l.down.desc.cols as usize;
                for used in 1..=16usize {
                    let max_assigns = (16 * used).min(MOE_MAX_ASSIGNS);
                    let s_gu = compute_splits((gu_rows / 256) * used, gu_cols / 256);
                    if s_gu > 1 {
                        max_scratch_elems = max_scratch_elems.max(s_gu * max_assigns * gu_rows);
                    }
                    let s_d = compute_splits((d_rows / 256) * used, d_cols / 256);
                    if s_d > 1 {
                        max_scratch_elems = max_scratch_elems.max(s_d * max_assigns * d_rows);
                    }
                }
            } else {
                let gu_rows = (l.gate.desc.rows + l.up.desc.rows) as usize;
                let s_gu = compute_splits(gu_rows / 256, l.gate.desc.cols as usize / 256);
                if s_gu > 1 {
                    max_scratch_elems = max_scratch_elems.max(s_gu * 16 * gu_rows);
                }
                let d_rows = l.down.desc.rows as usize;
                let s_d = compute_splits(d_rows / 256, l.down.desc.cols as usize / 256);
                if s_d > 1 {
                    max_scratch_elems = max_scratch_elems.max(s_d * 16 * d_rows);
                }
            }
        }
        let splitk_scratch = stream
            .alloc_zeros::<f32>(max_scratch_elems)
            .map_err(cuda_err)?;
        let moe_group_experts = stream.alloc_zeros::<i32>(MOE_MAX_ASSIGNS).map_err(cuda_err)?;
        let moe_group_offsets = stream
            .alloc_zeros::<i32>(MOE_MAX_ASSIGNS + 1)
            .map_err(cuda_err)?;
        let moe_group_assigns = stream.alloc_zeros::<i32>(MOE_MAX_ASSIGNS).map_err(cuda_err)?;
        let moe_num_groups = stream.alloc_zeros::<i32>(1).map_err(cuda_err)?;

        Ok(Self {
            index,
            layers,
            r1_signs,
            folded_rmsnorm,
            stream,
            r1_enter_fn,
            r1_exit_fn,
            moe_group_fn,
            splitk_reduce_fn,
            splitk_scratch,
            moe_group_experts,
            moe_group_offsets,
            moe_group_assigns,
            moe_num_groups,
            last_moe_group: (AtomicU64::new(0), AtomicUsize::new(0)),
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

    #[allow(clippy::too_many_arguments)]
    fn launch_splitk_reduce(
        &self,
        partials_ptr: u64,
        y0: u64,
        y1: u64,
        y2: u64,
        rows0: i32,
        rows1: i32,
        rows2: i32,
        tokens: usize,
        splits: usize,
    ) -> Result<()> {
        let total_rows = (rows0 + rows1 + rows2) as u32;
        let tokens_i = tokens as i32;
        let splits_i = splits as i32;
        let mut b = self.stream.launch_builder(&self.splitk_reduce_fn);
        b.arg(&partials_ptr)
            .arg(&y0)
            .arg(&y1)
            .arg(&y2)
            .arg(&rows0)
            .arg(&rows1)
            .arg(&rows2)
            .arg(&tokens_i)
            .arg(&splits_i);
        unsafe {
            b.launch(LaunchConfig {
                grid_dim: (total_rows.div_ceil(256), tokens as u32, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            })
        }
        .map_err(cuda_err)?;
        Ok(())
    }

    /// Runs 1..=3 HYB Q=5 projections sharing input `x` (and optional `x_up` for SwiGLU)
    /// for `1 <= tokens <= 16`.
    /// Uses deterministic split-K reduction (no `atomicAdd`, no pre-zeroing required).
    #[allow(clippy::too_many_arguments)]
    pub fn run_multi(
        &self,
        matrices: &[(&Plt1GpuTensor, &mut CudaSlice<f32>)],
        x: &CudaSlice<f32>,
        x_up: Option<&CudaSlice<f32>>,
        tokens: usize,
        _zero_next: &[(&mut CudaSlice<f32>, usize)],
        _zero_self: bool,
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
        let zs = [0u64; 3];
        let ns = [0i32; 3];

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
        let partials_ptr = if splits > 1 {
            self.splitk_scratch.device_ptr(&self.stream).0
        } else {
            0u64
        };
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
        b.arg(&cols_i).arg(&tokens_i).arg(&cps_i).arg(&partials_ptr);
        unsafe {
            b.launch(LaunchConfig {
                grid_dim: (total_rb as u32, splits as u32, 1),
                block_dim: (512, 1, 1),
                shared_mem_bytes: 0,
            })
        }
        .map_err(cuda_err)?;
        if splits > 1 {
            self.launch_splitk_reduce(
                partials_ptr,
                ys[0],
                ys[1],
                ys[2],
                rows_arr[0],
                rows_arr[1],
                rows_arr[2],
                tokens,
                splits,
            )?;
        }
        Ok(())
    }

    fn group_moe_assignments(&self, ids: &CudaSlice<i32>, total_assigns: usize) -> Result<()> {
        if total_assigns > MOE_MAX_ASSIGNS {
            return Err(Error::Invalid(format!(
                "MoE assignments {total_assigns} exceeds max {MOE_MAX_ASSIGNS}"
            )));
        }
        let ids_ptr = ids.device_ptr(&self.stream).0;
        let ge_ptr = self.moe_group_experts.device_ptr(&self.stream).0;
        let go_ptr = self.moe_group_offsets.device_ptr(&self.stream).0;
        let ga_ptr = self.moe_group_assigns.device_ptr(&self.stream).0;
        let ng_ptr = self.moe_num_groups.device_ptr(&self.stream).0;
        let total_i = total_assigns as i32;
        let mut b = self.stream.launch_builder(&self.moe_group_fn);
        b.arg(&ids_ptr)
            .arg(&total_i)
            .arg(&ge_ptr)
            .arg(&go_ptr)
            .arg(&ga_ptr)
            .arg(&ng_ptr);
        unsafe {
            b.launch(LaunchConfig {
                grid_dim: (1, 1, 1),
                block_dim: (32, 1, 1),
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
        let total_assigns = tokens * used;
        let ids_ptr = ids.device_ptr(&self.stream).0;
        self.group_moe_assignments(ids, total_assigns)?;
        self.last_moe_group.0.store(ids_ptr, Ordering::Relaxed);
        self.last_moe_group
            .1
            .store(total_assigns, Ordering::Relaxed);

        let skip_in = (gate.desc.flags & tensor_flags::SKIP_IN_RHT) != 0;
        let skip_out = (gate.desc.flags & tensor_flags::SKIP_OUT_RHT) != 0;
        let (_, _, _, _, moe_p, _) = &self.kernels[&(gate.desc.pair_bits, skip_in, skip_out)];
        let r0 = gate.desc.rows as usize;
        let r1 = up.desc.rows as usize;
        let cols = gate.desc.cols as usize;
        let rb0 = (r0 / 256) as i32;
        let total_rb = r0 / 256 + r1 / 256;
        let total_chunks = cols / 256;
        // Keep split count independent of `tokens` so single-token decode (B=1) and
        // batched speculative verification (B=2..16) have identical FP32 accumulation order.
        let splits = (340usize / (total_rb * used).max(1)).clamp(1, total_chunks);
        let cps = total_chunks.div_ceil(splits);
        let splits = total_chunks.div_ceil(cps);

        let y0 = y_gate.device_ptr(&self.stream).0;
        let y1 = y_up.device_ptr(&self.stream).0;
        let partials_ptr = if splits > 1 {
            self.splitk_scratch.device_ptr(&self.stream).0
        } else {
            0u64
        };
        let total_assigns_i = total_assigns as i32;

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
        let x_ptr = x.device_ptr(&self.stream).0;
        let ge_ptr = self.moe_group_experts.device_ptr(&self.stream).0;
        let go_ptr = self.moe_group_offsets.device_ptr(&self.stream).0;
        let ga_ptr = self.moe_group_assigns.device_ptr(&self.stream).0;
        let ng_ptr = self.moe_num_groups.device_ptr(&self.stream).0;
        let (rows0_i, rows1_i, cols_i, used_i, cps_i, splits_i) = (
            r0 as i32,
            r1 as i32,
            cols as i32,
            used as i32,
            cps as i32,
            splits as i32,
        );

        let mut b = self.stream.launch_builder(moe_p);
        b.arg(&t0).arg(&t1).arg(&l0).arg(&l1);
        b.arg(&sc0).arg(&sc1).arg(&c0).arg(&c1).arg(&s0).arg(&s1);
        b.arg(&y0).arg(&y1).arg(&rows0_i).arg(&rows1_i).arg(&rb0);
        b.arg(&x_ptr)
            .arg(&ge_ptr)
            .arg(&go_ptr)
            .arg(&ga_ptr)
            .arg(&ng_ptr)
            .arg(&cols_i)
            .arg(&used_i)
            .arg(&cps_i)
            .arg(&splits_i)
            .arg(&partials_ptr)
            .arg(&total_assigns_i);
        unsafe {
            b.launch(LaunchConfig {
                grid_dim: (total_rb as u32, total_assigns as u32, splits as u32),
                block_dim: (512, 1, 1),
                shared_mem_bytes: 0,
            })
        }
        .map_err(cuda_err)?;
        if splits > 1 {
            self.launch_splitk_reduce(
                partials_ptr,
                y0,
                y1,
                0u64,
                rows0_i,
                rows1_i,
                0i32,
                total_assigns,
                splits,
            )?;
        }
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
        let total_assigns = tokens * used;
        let ids_ptr = ids.device_ptr(&self.stream).0;
        let prev_ptr = self.last_moe_group.0.swap(0, Ordering::Relaxed);
        let prev_n = self.last_moe_group.1.swap(0, Ordering::Relaxed);
        if prev_ptr != ids_ptr || prev_n != total_assigns {
            self.group_moe_assignments(ids, total_assigns)?;
        }

        let skip_in = (down.desc.flags & tensor_flags::SKIP_IN_RHT) != 0;
        let skip_out = (down.desc.flags & tensor_flags::SKIP_OUT_RHT) != 0;
        let (_, _, _, _, _, moe_s) = &self.kernels[&(down.desc.pair_bits, skip_in, skip_out)];
        let r0 = down.desc.rows as usize;
        let cols = down.desc.cols as usize;
        let total_rb = r0 / 256;
        let total_chunks = cols / 256;
        // Keep split count independent of `tokens` for bit-exact B=1 vs B=2..16 parity.
        let splits = (340usize / (total_rb * used).max(1)).clamp(1, total_chunks);
        let cps = total_chunks.div_ceil(splits);
        let splits = total_chunks.div_ceil(cps);

        let y0 = y_down.device_ptr(&self.stream).0;
        let partials_ptr = if splits > 1 {
            self.splitk_scratch.device_ptr(&self.stream).0
        } else {
            0u64
        };
        let total_assigns_i = total_assigns as i32;

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
        let xg = x_gate.device_ptr(&self.stream).0;
        let xu = x_up.device_ptr(&self.stream).0;
        let ge_ptr = self.moe_group_experts.device_ptr(&self.stream).0;
        let go_ptr = self.moe_group_offsets.device_ptr(&self.stream).0;
        let ga_ptr = self.moe_group_assigns.device_ptr(&self.stream).0;
        let ng_ptr = self.moe_num_groups.device_ptr(&self.stream).0;
        let (rows0_i, cols_i, used_i, cps_i, splits_i) = (
            r0 as i32,
            cols as i32,
            used as i32,
            cps as i32,
            splits as i32,
        );

        let mut b = self.stream.launch_builder(moe_s);
        b.arg(&t0).arg(&l0).arg(&sc0).arg(&c0).arg(&s0).arg(&y0);
        b.arg(&rows0_i)
            .arg(&xg)
            .arg(&xu)
            .arg(&ge_ptr)
            .arg(&go_ptr)
            .arg(&ga_ptr)
            .arg(&ng_ptr)
            .arg(&cols_i)
            .arg(&used_i)
            .arg(&cps_i)
            .arg(&splits_i)
            .arg(&partials_ptr)
            .arg(&total_assigns_i);
        unsafe {
            b.launch(LaunchConfig {
                grid_dim: (total_rb as u32, total_assigns as u32, splits as u32),
                block_dim: (512, 1, 1),
                shared_mem_bytes: 0,
            })
        }
        .map_err(cuda_err)?;
        if splits > 1 {
            self.launch_splitk_reduce(
                partials_ptr,
                y0,
                0u64,
                0u64,
                rows0_i,
                0i32,
                0i32,
                total_assigns,
                splits,
            )?;
        }
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
            b.arg(&cols_i).arg(&tokens_i).arg(&cps_i).arg(&zero_u64);
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
            b.arg(&cols_i).arg(&tokens_i).arg(&cps_i).arg(&zero_u64);
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
            b.arg(&cols_i).arg(&tokens_i).arg(&cps_i).arg(&zero_u64);
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
        let f_group = mod_in.load_function("hyb_moe_group_assignments").unwrap();
        let f_gu = mod_in.load_function("hyb_moe_small_k").unwrap();
        let f_down = mod_out.load_function("hyb_moe_small_k_swiglu").unwrap();

        let x: Vec<f32> = (0..tokens * d_model)
            .map(|i| ((i * 29 % 200) as f32 - 100.0) / 100.0)
            .collect();
        let ids: Vec<i32> = vec![0, 2, 1, 3, 2, 0];
        let total_assigns = tokens * used;
        let mut d_x = stream.clone_htod(&x).unwrap();
        let d_r1 = stream.clone_htod(&r1_signs).unwrap();
        let d_ids = stream.clone_htod(&ids).unwrap();
        let d_lut = stream.clone_htod(&cb.lut_f16x2).unwrap();

        let d_ge = stream.alloc_zeros::<i32>(MOE_MAX_ASSIGNS).unwrap();
        let d_go = stream.alloc_zeros::<i32>(MOE_MAX_ASSIGNS + 1).unwrap();
        let d_ga = stream.alloc_zeros::<i32>(MOE_MAX_ASSIGNS).unwrap();
        let d_ng = stream.alloc_zeros::<i32>(1).unwrap();

        let d_tg = stream.clone_htod(&tiles_g).unwrap();
        let d_tu = stream.clone_htod(&tiles_u).unwrap();
        let d_td = stream.clone_htod(&tiles_d).unwrap();
        let d_sg = stream.clone_htod(&sc_g).unwrap();
        let d_su = stream.clone_htod(&sc_u).unwrap();
        let d_sd = stream.clone_htod(&sc_d).unwrap();
        let d_rg = stream.clone_htod(rht_g.row_signs()).unwrap();
        let d_ru = stream.clone_htod(rht_u.row_signs()).unwrap();
        let d_cd = stream.clone_htod(rht_d.col_signs()).unwrap();

        let d_yg = stream.alloc_zeros::<f32>(total_assigns * d_ff).unwrap();
        let d_yu = stream.alloc_zeros::<f32>(total_assigns * d_ff).unwrap();
        let mut d_yd = stream.alloc_zeros::<f32>(total_assigns * d_model).unwrap();

        let (d_model_i, d_ff_i, used_i, rb0_i, total_i) = (
            d_model as i32,
            d_ff as i32,
            used as i32,
            (d_ff / 256) as i32,
            total_assigns as i32,
        );
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
            let (ge, go, ga, ng) = (
                d_ge.device_ptr(&stream).0,
                d_go.device_ptr(&stream).0,
                d_ga.device_ptr(&stream).0,
                d_ng.device_ptr(&stream).0,
            );

            let mut bg = stream.launch_builder(&f_group);
            bg.arg(&idsp)
                .arg(&total_i)
                .arg(&ge)
                .arg(&go)
                .arg(&ga)
                .arg(&ng);
            bg.launch(LaunchConfig {
                grid_dim: (1, 1, 1),
                block_dim: (32, 1, 1),
                shared_mem_bytes: 0,
            })
            .unwrap();

            // Gate+Up with splits=2 (d_model=512 -> 2 chunks, cps=1)
            let (cps_gu, splits_gu, null_u64, ta_i) = (1i32, 2i32, 0u64, total_assigns as i32);
            let mut b1 = stream.launch_builder(&f_gu);
            b1.arg(&tg).arg(&tu).arg(&lut).arg(&lut);
            b1.arg(&sg).arg(&su).arg(&r1p).arg(&r1p).arg(&rg).arg(&ru);
            b1.arg(&yg).arg(&yu).arg(&d_ff_i).arg(&d_ff_i).arg(&rb0_i);
            b1.arg(&xp)
                .arg(&ge)
                .arg(&go)
                .arg(&ga)
                .arg(&ng)
                .arg(&d_model_i)
                .arg(&used_i)
                .arg(&cps_gu)
                .arg(&splits_gu)
                .arg(&null_u64)
                .arg(&ta_i);
            b1.launch(LaunchConfig {
                grid_dim: ((2 * d_ff / 256) as u32, total_assigns as u32, splits_gu as u32),
                block_dim: (512, 1, 1),
                shared_mem_bytes: 0,
            })
            .unwrap();

            // Down with splits=1 (d_ff=256 -> 1 chunk, cps=1)
            let (cps_d, splits_d) = (1i32, 1i32);
            let mut b2 = stream.launch_builder(&f_down);
            b2.arg(&td).arg(&lut).arg(&sd).arg(&cd).arg(&r1p).arg(&yd);
            b2.arg(&d_model_i)
                .arg(&yg)
                .arg(&yu)
                .arg(&ge)
                .arg(&go)
                .arg(&ga)
                .arg(&ng)
                .arg(&d_ff_i)
                .arg(&used_i)
                .arg(&cps_d)
                .arg(&splits_d)
                .arg(&null_u64)
                .arg(&ta_i);
            b2.launch(LaunchConfig {
                grid_dim: ((d_model / 256) as u32, total_assigns as u32, splits_d as u32),
                block_dim: (512, 1, 1),
                shared_mem_bytes: 0,
            })
            .unwrap();

            let mut b3 = stream.launch_builder(&r1_exit);
            b3.arg(&mut d_yd).arg(&d_r1).arg(&d_model_i);
            b3.launch(LaunchConfig {
                grid_dim: ((d_model / 256) as u32, total_assigns as u32, 1),
                block_dim: (32, 1, 1),
                shared_mem_bytes: 0,
            })
            .unwrap();
        }
        let gpu_yd = stream.clone_dtoh(&d_yd).unwrap();
        let mut cpu_yd = vec![0.0f32; total_assigns * d_model];
        for assign in 0..total_assigns {
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

    #[test]
    fn hyb_moe_grouped_duplicates_and_split_k_match_cpu() {
        let ctx = CudaContext::new(0).unwrap();
        let stream = ctx.default_stream();
        // 12 experts, top-8, tokens=8 (64 assigns with heavy duplicates -> g_len up to 8)
        // plus a second check with tokens=12, used=4 (48 assigns where expert 0 appears 12 times -> sub-group split 8+4!).
        let (n_experts, d_model, d_ff) = (12usize, 1024usize, 512usize);
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
                    .map(|i| (((i * 6151 + e * 99991) % 1000) as f32 - 500.0) / 500.0)
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

        let r1_signs = Incoherence::with_block(d_model, d_model, 555, Some(256))
            .unwrap()
            .col_signs()
            .to_vec();
        let (rht_g, tiles_g, sc_g, rec_g) =
            quant_moe(d_ff, d_model, 666, Some(r1_signs.clone()), true);
        let (rht_u, tiles_u, sc_u, rec_u) =
            quant_moe(d_ff, d_model, 777, Some(r1_signs.clone()), true);
        let (rht_d, tiles_d, sc_d, rec_d) =
            quant_moe(d_model, d_ff, 888, Some(r1_signs.clone()), false);

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
        let f_group = mod_in.load_function("hyb_moe_group_assignments").unwrap();
        let f_gu = mod_in.load_function("hyb_moe_small_k").unwrap();
        let f_down = mod_out.load_function("hyb_moe_small_k_swiglu").unwrap();

        let d_r1 = stream.clone_htod(&r1_signs).unwrap();
        let d_lut = stream.clone_htod(&cb.lut_f16x2).unwrap();
        let d_ge = stream.alloc_zeros::<i32>(MOE_MAX_ASSIGNS).unwrap();
        let d_go = stream.alloc_zeros::<i32>(MOE_MAX_ASSIGNS + 1).unwrap();
        let d_ga = stream.alloc_zeros::<i32>(MOE_MAX_ASSIGNS).unwrap();
        let d_ng = stream.alloc_zeros::<i32>(1).unwrap();
        let d_tg = stream.clone_htod(&tiles_g).unwrap();
        let d_tu = stream.clone_htod(&tiles_u).unwrap();
        let d_td = stream.clone_htod(&tiles_d).unwrap();
        let d_sg = stream.clone_htod(&sc_g).unwrap();
        let d_su = stream.clone_htod(&sc_u).unwrap();
        let d_sd = stream.clone_htod(&sc_d).unwrap();
        let d_rg = stream.clone_htod(rht_g.row_signs()).unwrap();
        let d_ru = stream.clone_htod(rht_u.row_signs()).unwrap();
        let d_cd = stream.clone_htod(rht_d.col_signs()).unwrap();

        let run_case = |tokens: usize, used: usize, ids: &[i32], splits_gu: i32, splits_d: i32| {
            let total_assigns = tokens * used;
            let x: Vec<f32> = (0..tokens * d_model)
                .map(|i| ((i * 41 % 200) as f32 - 100.0) / 100.0)
                .collect();
            let mut d_x = stream.clone_htod(&x).unwrap();
            let d_ids = stream.clone_htod(ids).unwrap();
            // Pre-poison output buffers when splits == 1 to verify direct store overwrites!
            let init_gu = if splits_gu == 1 { 1234.5f32 } else { 0.0f32 };
            let init_d = if splits_d == 1 { -987.25f32 } else { 0.0f32 };
            let d_yg = stream.clone_htod(&vec![init_gu; total_assigns * d_ff]).unwrap();
            let d_yu = stream.clone_htod(&vec![init_gu; total_assigns * d_ff]).unwrap();
            let mut d_yd = stream.clone_htod(&vec![init_d; total_assigns * d_model]).unwrap();

            let (d_model_i, d_ff_i, used_i, rb0_i, total_i) = (
                d_model as i32,
                d_ff as i32,
                used as i32,
                (d_ff / 256) as i32,
                total_assigns as i32,
            );
            let cps_gu = ((d_model / 256) as i32 + splits_gu - 1) / splits_gu;
            let cps_d = ((d_ff / 256) as i32 + splits_d - 1) / splits_d;

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
                let (ge, go, ga, ng) = (
                    d_ge.device_ptr(&stream).0,
                    d_go.device_ptr(&stream).0,
                    d_ga.device_ptr(&stream).0,
                    d_ng.device_ptr(&stream).0,
                );

                let mut bg = stream.launch_builder(&f_group);
                bg.arg(&idsp)
                    .arg(&total_i)
                    .arg(&ge)
                    .arg(&go)
                    .arg(&ga)
                    .arg(&ng);
                bg.launch(LaunchConfig {
                    grid_dim: (1, 1, 1),
                    block_dim: (32, 1, 1),
                    shared_mem_bytes: 0,
                })
                .unwrap();

                let (null_u64, ta_i) = (0u64, total_assigns as i32);
                let mut b1 = stream.launch_builder(&f_gu);
                b1.arg(&tg).arg(&tu).arg(&lut).arg(&lut);
                b1.arg(&sg).arg(&su).arg(&r1p).arg(&r1p).arg(&rg).arg(&ru);
                b1.arg(&yg).arg(&yu).arg(&d_ff_i).arg(&d_ff_i).arg(&rb0_i);
                b1.arg(&xp)
                    .arg(&ge)
                    .arg(&go)
                    .arg(&ga)
                    .arg(&ng)
                    .arg(&d_model_i)
                    .arg(&used_i)
                    .arg(&cps_gu)
                    .arg(&splits_gu)
                    .arg(&null_u64)
                    .arg(&ta_i);
                b1.launch(LaunchConfig {
                    grid_dim: (
                        (2 * d_ff / 256) as u32,
                        total_assigns as u32,
                        splits_gu as u32,
                    ),
                    block_dim: (512, 1, 1),
                    shared_mem_bytes: 0,
                })
                .unwrap();

                let mut b2 = stream.launch_builder(&f_down);
                b2.arg(&td).arg(&lut).arg(&sd).arg(&cd).arg(&r1p).arg(&yd);
                b2.arg(&d_model_i)
                    .arg(&yg)
                    .arg(&yu)
                    .arg(&ge)
                    .arg(&go)
                    .arg(&ga)
                    .arg(&ng)
                    .arg(&d_ff_i)
                    .arg(&used_i)
                    .arg(&cps_d)
                    .arg(&splits_d)
                    .arg(&null_u64)
                    .arg(&ta_i);
                b2.launch(LaunchConfig {
                    grid_dim: (
                        (d_model / 256) as u32,
                        total_assigns as u32,
                        splits_d as u32,
                    ),
                    block_dim: (512, 1, 1),
                    shared_mem_bytes: 0,
                })
                .unwrap();

                let mut b3 = stream.launch_builder(&r1_exit);
                b3.arg(&mut d_yd).arg(&d_r1).arg(&d_model_i);
                b3.launch(LaunchConfig {
                    grid_dim: ((d_model / 256) as u32, total_assigns as u32, 1),
                    block_dim: (32, 1, 1),
                    shared_mem_bytes: 0,
                })
                .unwrap();
            }

            let gpu_yd = stream.clone_dtoh(&d_yd).unwrap();
            let mut cpu_yd = vec![0.0f32; total_assigns * d_model];
            for assign in 0..total_assigns {
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
            assert!(
                rel < 2e-3,
                "MoE grouped (tokens={tokens}, used={used}, splits_gu={splits_gu}, splits_d={splits_d}) mismatch: rel={rel:.4e}"
            );
        };

        // Case 1: tokens=8, used=8 with heavy duplicates across tokens (splits=1 and splits>1).
        let ids_8x8: Vec<i32> = (0..64)
            .map(|i| {
                let tok = i / 8;
                let slot = i % 8;
                // Experts 0, 1, 2 hit by all 8 tokens (g_len=8); others hit by 2..6 tokens.
                if slot < 3 {
                    slot as i32
                } else {
                    (3 + (tok + slot * 3) % (n_experts - 3)) as i32
                }
            })
            .collect();
        run_case(8, 8, &ids_8x8, 1, 1);
        run_case(8, 8, &ids_8x8, 4, 2);

        // Case 2: tokens=12, used=4 where expert 0 appears 12 times (> 8 -> splits into 8 + 4 sub-groups!).
        let ids_12x4: Vec<i32> = (0..48)
            .map(|i| {
                let tok = i / 4;
                let slot = i % 4;
                if slot == 0 {
                    0
                } else {
                    (1 + (tok + slot * 2) % (n_experts - 1)) as i32
                }
            })
            .collect();
        run_case(12, 4, &ids_12x4, 1, 1);
        run_case(12, 4, &ids_12x4, 2, 2);
    }

    #[test]
    #[ignore]
    fn bench_qwen3_30b_a3b_moe_hyb() {
        let ctx = CudaContext::new(0).unwrap();
        let stream = ctx.default_stream();
        // Qwen3-30B-A3B MoE FFN dimensions: 128 experts, top-8, d_model=2048, d_ff=768, B=6 L=10 Q=5.
        let (n_experts, used, d_model, d_ff) = (128usize, 8usize, 2048usize, 768usize);
        let words_per_tile = 24usize; // B=6 -> 128 * 6 / 32 = 24 u32/tile
        let gu_tiles_words = n_experts * (d_ff / 16) * (d_model / 16) * words_per_tile;
        let d_tiles_words = n_experts * (d_model / 16) * (d_ff / 16) * words_per_tile;

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
        let f_group = mod_in.load_function("hyb_moe_group_assignments").unwrap();
        let f_gu = mod_in.load_function("hyb_moe_small_k").unwrap();
        let f_down = mod_out.load_function("hyb_moe_small_k_swiglu").unwrap();

        let cb = HybCodebook::gaussian_for_q(5);
        let d_lut = stream.clone_htod(&cb.lut_f16x2).unwrap();
        let d_tg = stream.alloc_zeros::<u32>(gu_tiles_words).unwrap();
        let d_tu = stream.alloc_zeros::<u32>(gu_tiles_words).unwrap();
        let d_td = stream.alloc_zeros::<u32>(d_tiles_words).unwrap();
        let d_sg = stream.clone_htod(&vec![0.01f32; n_experts]).unwrap();
        let d_su = stream.clone_htod(&vec![0.01f32; n_experts]).unwrap();
        let d_sd = stream.clone_htod(&vec![0.01f32; n_experts]).unwrap();
        let d_r1 = stream.clone_htod(&vec![1.0f32; d_model]).unwrap();
        let d_rg = stream.clone_htod(&vec![1.0f32; d_ff]).unwrap();
        let d_ru = stream.clone_htod(&vec![1.0f32; d_ff]).unwrap();
        let d_cd = stream.clone_htod(&vec![1.0f32; d_ff]).unwrap();
        let d_ge = stream.alloc_zeros::<i32>(MOE_MAX_ASSIGNS).unwrap();
        let d_go = stream.alloc_zeros::<i32>(MOE_MAX_ASSIGNS + 1).unwrap();
        let d_ga = stream.alloc_zeros::<i32>(MOE_MAX_ASSIGNS).unwrap();
        let d_ng = stream.alloc_zeros::<i32>(1).unwrap();

        for &(label, tokens, unique_pool) in &[
            ("B=1 (8 unique experts)", 1usize, 128usize),
            ("k=8 (uniform 128-pool, ~50 unique)", 8usize, 128usize),
            ("k=8 (realistic skewed, 32 unique)", 8usize, 32usize),
        ] {
            let total_assigns = tokens * used;
            // Deterministic pseudo-random top-8 expert IDs per token (distinct within each token).
            let mut ids = Vec::with_capacity(total_assigns);
            let mut rng = 0x1234_5678_9abc_def0u64 ^ (tokens as u64 * 17) ^ (unique_pool as u64);
            for _ in 0..tokens {
                let mut picked = Vec::with_capacity(used);
                while picked.len() < used {
                    rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                    let e = ((rng >> 33) as usize % unique_pool) as i32;
                    if !picked.contains(&e) {
                        picked.push(e);
                    }
                }
                ids.extend_from_slice(&picked);
            }
            let d_ids = stream.clone_htod(&ids).unwrap();
            let d_x = stream.clone_htod(&vec![0.1f32; tokens * d_model]).unwrap();
            let d_yg = stream.alloc_zeros::<f32>(total_assigns * d_ff).unwrap();
            let d_yu = stream.alloc_zeros::<f32>(total_assigns * d_ff).unwrap();
            let d_yd = stream.alloc_zeros::<f32>(total_assigns * d_model).unwrap();

            let rb_gu = 2 * (d_ff / 256);
            let ch_gu = d_model / 256;
            let splits_gu = (340usize / (rb_gu * total_assigns).max(1)).clamp(1, ch_gu);
            let cps_gu = ch_gu.div_ceil(splits_gu);
            let splits_gu = ch_gu.div_ceil(cps_gu);

            let rb_d = d_model / 256;
            let ch_d = d_ff / 256;
            let splits_d = (340usize / (rb_d * total_assigns).max(1)).clamp(1, ch_d);
            let cps_d = ch_d.div_ceil(splits_d);
            let splits_d = ch_d.div_ceil(cps_d);

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
            let (ge, go, ga, ng) = (
                d_ge.device_ptr(&stream).0,
                d_go.device_ptr(&stream).0,
                d_ga.device_ptr(&stream).0,
                d_ng.device_ptr(&stream).0,
            );
            let (d_model_i, d_ff_i, used_i, rb0_i, total_i) = (
                d_model as i32,
                d_ff as i32,
                used as i32,
                (d_ff / 256) as i32,
                total_assigns as i32,
            );
            let (cps_gu_i, splits_gu_i, cps_d_i, splits_d_i) =
                (cps_gu as i32, splits_gu as i32, cps_d as i32, splits_d as i32);

            let launch_once = || unsafe {
                let mut bg = stream.launch_builder(&f_group);
                bg.arg(&idsp)
                    .arg(&total_i)
                    .arg(&ge)
                    .arg(&go)
                    .arg(&ga)
                    .arg(&ng);
                bg.launch(LaunchConfig {
                    grid_dim: (1, 1, 1),
                    block_dim: (32, 1, 1),
                    shared_mem_bytes: 0,
                })
                .unwrap();

                if splits_gu > 1 {
                    cudarc::driver::result::memset_d8_async(
                        yg,
                        0,
                        total_assigns * d_ff * std::mem::size_of::<f32>(),
                        stream.cu_stream(),
                    )
                    .unwrap();
                    cudarc::driver::result::memset_d8_async(
                        yu,
                        0,
                        total_assigns * d_ff * std::mem::size_of::<f32>(),
                        stream.cu_stream(),
                    )
                    .unwrap();
                }
                let (null_u64, ta_i) = (0u64, total_assigns as i32);
                let mut b1 = stream.launch_builder(&f_gu);
                b1.arg(&tg).arg(&tu).arg(&lut).arg(&lut);
                b1.arg(&sg).arg(&su).arg(&r1p).arg(&r1p).arg(&rg).arg(&ru);
                b1.arg(&yg).arg(&yu).arg(&d_ff_i).arg(&d_ff_i).arg(&rb0_i);
                b1.arg(&xp)
                    .arg(&ge)
                    .arg(&go)
                    .arg(&ga)
                    .arg(&ng)
                    .arg(&d_model_i)
                    .arg(&used_i)
                    .arg(&cps_gu_i)
                    .arg(&splits_gu_i)
                    .arg(&null_u64)
                    .arg(&ta_i);
                b1.launch(LaunchConfig {
                    grid_dim: (rb_gu as u32, total_assigns as u32, splits_gu as u32),
                    block_dim: (512, 1, 1),
                    shared_mem_bytes: 0,
                })
                .unwrap();

                if splits_d > 1 {
                    cudarc::driver::result::memset_d8_async(
                        yd,
                        0,
                        total_assigns * d_model * std::mem::size_of::<f32>(),
                        stream.cu_stream(),
                    )
                    .unwrap();
                }
                let mut b2 = stream.launch_builder(&f_down);
                b2.arg(&td).arg(&lut).arg(&sd).arg(&cd).arg(&r1p).arg(&yd);
                b2.arg(&d_model_i)
                    .arg(&yg)
                    .arg(&yu)
                    .arg(&ge)
                    .arg(&go)
                    .arg(&ga)
                    .arg(&ng)
                    .arg(&d_ff_i)
                    .arg(&used_i)
                    .arg(&cps_d_i)
                    .arg(&splits_d_i)
                    .arg(&null_u64)
                    .arg(&ta_i);
                b2.launch(LaunchConfig {
                    grid_dim: (rb_d as u32, total_assigns as u32, splits_d as u32),
                    block_dim: (512, 1, 1),
                    shared_mem_bytes: 0,
                })
                .unwrap();
            };

            // Warmup
            for _ in 0..20 {
                launch_once();
            }
            stream.synchronize().unwrap();
            let ng_host = stream.clone_dtoh(&d_ng).unwrap()[0];

            let iters = 200usize;
            let t0 = std::time::Instant::now();
            for _ in 0..iters {
                launch_once();
            }
            stream.synchronize().unwrap();
            let us_per_layer = t0.elapsed().as_secs_f64() * 1e6 / iters as f64;
            let ms_48_layers = us_per_layer * 48.0 / 1000.0;
            println!(
                "[MoE HYB Q=5] {label}: groups={ng_host}, splits=(gu:{splits_gu}, d:{splits_d}) -> {us_per_layer:.2} us/layer, 48-layer={ms_48_layers:.3} ms"
            );
        }
    }
}
