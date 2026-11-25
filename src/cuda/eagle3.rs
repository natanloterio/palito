//! Native CUDA EAGLE-3 single-layer speculative draft head loaded from PLT1 (`.palito`).
//!
//! Reuses Palito's existing vectorized `Q8_0` `matvec_packed_rows`, `normalize_warp`,
//! `rotary`, `cache_kv`, `attention_fused`, `swiglu`, `residual`, `embedding`, and
//! `greedy_partials`/`greedy_finish` kernels — zero extra CUDA compilation needed.

use super::{CudaBackend, DeviceTensor, failure, invalid, matvec_grid};

fn vector(n: i32) -> (u32, u32, u32) {
    ((n as u32).div_ceil(256), 1, 1)
}
use crate::{
    Result,
    plt1::{Plt1Index, SectionKind},
};
use cudarc::driver::{CudaSlice, PushKernelArg};
use std::{
    fs::File,
    io::{Read, Seek, SeekFrom},
    path::Path,
};

pub(super) struct Eagle3State {
    pub(super) tap_layers: [usize; 3],
    pub(super) tap_hidden: [CudaSlice<f32>; 3],
    draft_vocab: usize,
    d2t: Vec<i32>,
    d_d2t: CudaSlice<i32>,
    draft_chain_dev: CudaSlice<i32>,
    rope_theta: f32,
    eps: f32,
    // Q8_0 (kind=8) projections
    fc: DeviceTensor,
    q_proj: DeviceTensor,
    k_proj: DeviceTensor,
    v_proj: DeviceTensor,
    o_proj: DeviceTensor,
    gate_proj: DeviceTensor,
    up_proj: DeviceTensor,
    down_proj: DeviceTensor,
    lm_head: DeviceTensor,
    // F32 RMSNorm weights (raw bytes)
    input_layernorm: CudaSlice<u8>,
    hidden_norm: CudaSlice<u8>,
    post_attention_layernorm: CudaSlice<u8>,
    norm: CudaSlice<u8>,
    // 1-layer EAGLE-3 KV cache
    keys: CudaSlice<f32>,
    values: CudaSlice<f32>,
    // Scratch buffers
    fc_in: CudaSlice<f32>,
    g: CudaSlice<f32>,
    g_save0: CudaSlice<f32>,
    g_save1: CudaSlice<f32>,
    emb: CudaSlice<f32>,
    inp_cat: CudaSlice<f32>,
    q: CudaSlice<f32>,
    k: CudaSlice<f32>,
    v: CudaSlice<f32>,
    attn: CudaSlice<f32>,
    proj: CudaSlice<f32>,
    norm_buf: CudaSlice<f32>,
    gate: CudaSlice<f32>,
    up: CudaSlice<f32>,
    draft_logits: CudaSlice<f32>,
    batch_toks: CudaSlice<i32>,
}

fn read_bytes(file: &mut File, offset: u64, size: u64) -> Result<Vec<u8>> {
    file.seek(SeekFrom::Start(offset))?;
    let mut buf = vec![0u8; size as usize];
    file.read_exact(&mut buf)?;
    Ok(buf)
}

impl CudaBackend {
    /// Loads an EAGLE-3 draft head from a PLT1 (`.palito`) file containing an
    /// `Eagle3SpecHead` section and Q8_0 projections, enabling 3-layer hidden-state
    /// tapping on the target model.
    pub fn load_eagle3(&mut self, path: &Path) -> Result<()> {
        let mut file = File::open(path)?;
        let index = Plt1Index::read_from(&mut file)?;
        let sec = index
            .sections
            .iter()
            .find(|s| s.kind == SectionKind::Eagle3SpecHead)
            .ok_or_else(|| invalid("PLT1 file has no Eagle3SpecHead section"))?;
        let sec_bytes = read_bytes(&mut file, sec.offset, sec.size)?;
        if sec_bytes.len() < 24 {
            return Err(invalid("Eagle3SpecHead section too short"));
        }
        let u32_at = |o: usize| u32::from_le_bytes(sec_bytes[o..o + 4].try_into().unwrap());
        let f32_at = |o: usize| f32::from_le_bytes(sec_bytes[o..o + 4].try_into().unwrap());
        let tap_layers = [u32_at(0) as usize, u32_at(4) as usize, u32_at(8) as usize];
        let draft_vocab = u32_at(12) as usize;
        let rope_theta = f32_at(16);
        let eps = f32_at(20);
        if sec_bytes.len() < 24 + draft_vocab * 4 {
            return Err(invalid("Eagle3SpecHead d2t table truncated"));
        }
        let d2t: Vec<i32> = sec_bytes[24..24 + draft_vocab * 4]
            .chunks_exact(4)
            .map(|c| i32::from_le_bytes(c.try_into().unwrap()))
            .collect();

        let load_proj = |file: &mut File, name: &str| -> Result<DeviceTensor> {
            let desc = index
                .tensors
                .iter()
                .find(|t| t.name == name)
                .ok_or_else(|| invalid(format!("missing EAGLE-3 tensor {name}")))?;
            let raw = read_bytes(file, desc.data_offset, desc.data_size)?;
            let bytes = self.stream.clone_htod(&raw).map_err(failure)?;
            let kind = match desc.codec {
                crate::plt1::Codec::Q8_0 => 8,
                crate::plt1::Codec::Q6K => 14,
                crate::plt1::Codec::Q4K => 12,
                _ => 8,
            };
            Ok(DeviceTensor {
                bytes,
                kind,
                width: desc.cols as i32,
                rows: desc.rows as i32,
                #[cfg(feature = "qtip-research")]
                qtip: None,
            })
        };
        let load_norm = |file: &mut File, name: &str| -> Result<CudaSlice<u8>> {
            let desc = index
                .tensors
                .iter()
                .find(|t| t.name == name)
                .ok_or_else(|| invalid(format!("missing EAGLE-3 norm {name}")))?;
            let raw = read_bytes(file, desc.data_offset, desc.data_size)?;
            self.stream.clone_htod(&raw).map_err(failure)
        };

        let fc = load_proj(&mut file, "fc.weight")?;
        let q_proj = load_proj(&mut file, "midlayer.self_attn.q_proj.weight")?;
        let k_proj = load_proj(&mut file, "midlayer.self_attn.k_proj.weight")?;
        let v_proj = load_proj(&mut file, "midlayer.self_attn.v_proj.weight")?;
        let o_proj = load_proj(&mut file, "midlayer.self_attn.o_proj.weight")?;
        let gate_proj = load_proj(&mut file, "midlayer.mlp.gate_proj.weight")?;
        let up_proj = load_proj(&mut file, "midlayer.mlp.up_proj.weight")?;
        let down_proj = load_proj(&mut file, "midlayer.mlp.down_proj.weight")?;
        let lm_head = load_proj(&mut file, "lm_head.weight")?;

        let input_layernorm = load_norm(&mut file, "midlayer.input_layernorm.weight")?;
        let hidden_norm = load_norm(&mut file, "midlayer.hidden_norm.weight")?;
        let post_attention_layernorm =
            load_norm(&mut file, "midlayer.post_attention_layernorm.weight")?;
        let norm = load_norm(&mut file, "norm.weight")?;

        let s = &self.weights.spec;
        let width = s.width;
        let q_width = q_proj.rows as usize;
        let kv_width = k_proj.rows as usize;
        let inter = gate_proj.rows as usize;
        let context = s.info.context_tokens as usize;
        let batch_cap = self.work.x.len() / width;

        let alloc = |n: usize| self.stream.alloc_zeros::<f32>(n).map_err(failure);
        let tap_hidden = [
            alloc(batch_cap * width)?,
            alloc(batch_cap * width)?,
            alloc(batch_cap * width)?,
        ];

        let d_d2t = self.stream.clone_htod(&d2t).map_err(failure)?;
        let draft_chain_dev = self.stream.alloc_zeros::<i32>(16).map_err(failure)?;

        self.eagle3 = Some(Eagle3State {
            tap_layers,
            tap_hidden,
            draft_vocab,
            d2t,
            d_d2t,
            draft_chain_dev,
            rope_theta,
            eps,
            fc,
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            gate_proj,
            up_proj,
            down_proj,
            lm_head,
            input_layernorm,
            hidden_norm,
            post_attention_layernorm,
            norm,
            keys: alloc(context * kv_width)?,
            values: alloc(context * kv_width)?,
            fc_in: alloc(8 * 3 * width)?,
            g: alloc(8 * width)?,
            g_save0: alloc(width)?,
            g_save1: alloc(width)?,
            emb: alloc(8 * width)?,
            inp_cat: alloc(8 * 2 * width)?,
            q: alloc(8 * q_width)?,
            k: alloc(8 * kv_width)?,
            v: alloc(8 * kv_width)?,
            attn: alloc(8 * q_width)?,
            proj: alloc(8 * width)?,
            norm_buf: alloc(8 * width)?,
            gate: alloc(8 * inter)?,
            up: alloc(8 * inter)?,
            draft_logits: alloc(draft_vocab)?,
            batch_toks: self.stream.alloc_zeros::<i32>(8).map_err(failure)?,
        });
        Ok(())
    }

    /// Runs one single-token EAGLE-3 midlayer step using `eagle.g` and `token`,
    /// writing KV at `pos` and updating `eagle.g` in-place.
    /// If `dev_token_slot` is `Some(slot)`, reads the token ID directly from
    /// `eagle.draft_chain_dev[slot]` on GPU (zero CPU-GPU sync!).
    fn eagle3_midlayer_step_impl(
        &mut self,
        token: i32,
        dev_token_slot: Option<usize>,
        pos: usize,
    ) -> Result<()> {
        let s = &self.weights.spec;
        let width = s.width;
        let width_i = width as i32;
        let heads = s.heads as i32;
        let kv_heads = s.kv_heads as i32;
        let dim = s.dim as i32;
        let qw = (s.heads * s.dim) as i32;
        let kw = (s.kv_heads * s.dim) as i32;
        let pos_i = pos as i32;
        let count_i = (pos + 1) as i32;
        let neox = 1i32;

        // 1. Embedding lookup into eagle.emb (reusing target embed_tokens).
        {
            let e = self.eagle3.as_mut().unwrap();
            let emb_w = &self.weights.embedding;
            if let Some(slot) = dev_token_slot {
                let func = self.functions["prefill_embedding"].clone();
                let tok_slice = e.draft_chain_dev.slice(slot..slot + 1);
                let mut b = self.stream.launch_builder(&func);
                b.arg(&emb_w.bytes)
                    .arg(&emb_w.kind)
                    .arg(&tok_slice)
                    .arg(&mut e.emb)
                    .arg(&width_i);
                unsafe {
                    b.launch(cudarc::driver::LaunchConfig {
                        grid_dim: (vector(width_i).0, 1, 1),
                        block_dim: (256, 1, 1),
                        shared_mem_bytes: 0,
                    })
                }
                .map_err(failure)?;
            } else {
                let func = self.functions["embedding"].clone();
                let mut b = self.stream.launch_builder(&func);
                b.arg(&emb_w.bytes)
                    .arg(&emb_w.kind)
                    .arg(&mut e.emb)
                    .arg(&width_i)
                    .arg(&token);
                unsafe {
                    b.launch(cudarc::driver::LaunchConfig {
                        grid_dim: vector(width_i),
                        block_dim: (256, 1, 1),
                        shared_mem_bytes: 0,
                    })
                }
                .map_err(failure)?;
            }
        }

        // 2. Fused input_layernorm(emb) + hidden_norm(g) -> inp_cat in 1 single launch!
        {
            let e = self.eagle3.as_mut().unwrap();
            let norm_cat_fn = self.functions["eagle3_norm_cat"].clone();
            let eps = e.eps;
            let mut b = self.stream.launch_builder(&norm_cat_fn);
            b.arg(&e.emb)
                .arg(&e.g)
                .arg(&e.input_layernorm)
                .arg(&e.hidden_norm)
                .arg(&mut e.inp_cat)
                .arg(&width_i)
                .arg(&eps);
            unsafe {
                b.launch(cudarc::driver::LaunchConfig {
                    grid_dim: (1, 2, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                })
            }
            .map_err(failure)?;
        }

        // 3. Q, K, V projections (fused into 1 launch via matvec_packed_qkv_rows for Q4_K!).
        let matvec_fn = self.functions["matvec"].clone();
        let matmul = self.matmul;
        {
            let e = self.eagle3.as_mut().unwrap();
            if matmul.f32_tiled()
                && e.q_proj.kind == 12
                && e.k_proj.kind == 12
                && e.v_proj.kind == 12
                && e.q_proj.width % 256 == 0
            {
                let qkv_fn = self.functions["matvec_packed_qkv_rows"].clone();
                let total_rows = e.q_proj.rows + e.k_proj.rows + e.v_proj.rows;
                let mut b = self.stream.launch_builder(&qkv_fn);
                b.arg(&e.q_proj.bytes)
                    .arg(&e.q_proj.kind)
                    .arg(&e.q_proj.rows)
                    .arg(&mut e.q)
                    .arg(&e.k_proj.bytes)
                    .arg(&e.k_proj.kind)
                    .arg(&e.k_proj.rows)
                    .arg(&mut e.k)
                    .arg(&e.v_proj.bytes)
                    .arg(&e.v_proj.kind)
                    .arg(&e.v_proj.rows)
                    .arg(&mut e.v)
                    .arg(&e.inp_cat)
                    .arg(&e.q_proj.width);
                unsafe {
                    b.launch(cudarc::driver::LaunchConfig {
                        grid_dim: matvec_grid(matmul, 12, total_rows, 1, e.q_proj.width),
                        block_dim: (256, 1, 1),
                        shared_mem_bytes: 0,
                    })
                }
                .map_err(failure)?;
            } else {
                let zero = 0i32;
                for (proj, out) in [
                    (&e.q_proj, &mut e.q),
                    (&e.k_proj, &mut e.k),
                    (&e.v_proj, &mut e.v),
                ] {
                    let mut b = self.stream.launch_builder(&matvec_fn);
                    b.arg(&proj.bytes)
                        .arg(&proj.kind)
                        .arg(&e.inp_cat)
                        .arg(out)
                        .arg(&proj.width)
                        .arg(&proj.rows)
                        .arg(&self.work.ids)
                        .arg(&zero)
                        .arg(&zero);
                    unsafe {
                        b.launch(cudarc::driver::LaunchConfig {
                            grid_dim: matvec_grid(matmul, proj.kind, proj.rows, 1, proj.width),
                            block_dim: (256, 1, 1),
                            shared_mem_bytes: 0,
                        })
                    }
                    .map_err(failure)?;
                }
            }
        }

        // 4. Fused RoPE (Q+K) + cache_kv in 1 launch (instead of 3!), then attention_fused.
        {
            let e = self.eagle3.as_mut().unwrap();
            let theta = e.rope_theta;
            let null_factors = 0u64;
            let mag = 1.0f32;
            let total_pairs = ((qw / 2 + kw / 2) as u32).div_ceil(256);
            let fused_rk = self.functions["rotary_factored_cache_kv_batch"].clone();
            {
                let mut b = self.stream.launch_builder(&fused_rk);
                b.arg(&mut e.q)
                    .arg(&qw)
                    .arg(&mut e.k)
                    .arg(&e.v)
                    .arg(&mut e.keys)
                    .arg(&mut e.values)
                    .arg(&kw)
                    .arg(&dim)
                    .arg(&pos_i)
                    .arg(&theta)
                    .arg(&neox)
                    .arg(&null_factors)
                    .arg(&mag);
                unsafe {
                    b.launch(cudarc::driver::LaunchConfig {
                        grid_dim: (total_pairs, 1, 1),
                        block_dim: (256, 1, 1),
                        shared_mem_bytes: 0,
                    })
                }
                .map_err(failure)?;
            }
            let attn_fn = self.functions["attention_warp128"].clone();
            {
                let mut b = self.stream.launch_builder(&attn_fn);
                b.arg(&e.q)
                    .arg(&e.keys)
                    .arg(&e.values)
                    .arg(&mut e.attn)
                    .arg(&heads)
                    .arg(&kv_heads)
                    .arg(&count_i);
                unsafe {
                    b.launch(cudarc::driver::LaunchConfig {
                        grid_dim: (heads as u32, 1, 1),
                        block_dim: (32, 1, 1),
                        shared_mem_bytes: 0,
                    })
                }
                .map_err(failure)?;
            }
        }

        // 5. O_proj + residual into g.
        {
            let e = self.eagle3.as_mut().unwrap();
            let zero = 0i32;
            let mut b = self.stream.launch_builder(&matvec_fn);
            b.arg(&e.o_proj.bytes)
                .arg(&e.o_proj.kind)
                .arg(&e.attn)
                .arg(&mut e.proj)
                .arg(&e.o_proj.width)
                .arg(&e.o_proj.rows)
                .arg(&self.work.ids)
                .arg(&zero)
                .arg(&zero);
            unsafe {
                b.launch(cudarc::driver::LaunchConfig {
                    grid_dim: matvec_grid(matmul, e.o_proj.kind, e.o_proj.rows, 1, e.o_proj.width),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                })
            }
            .map_err(failure)?;

            let res_fn = self.functions["residual"].clone();
            let mut rb = self.stream.launch_builder(&res_fn);
            rb.arg(&mut e.g).arg(&e.proj).arg(&width_i);
            unsafe {
                rb.launch(cudarc::driver::LaunchConfig {
                    grid_dim: vector(width_i),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                })
            }
            .map_err(failure)?;
        }

        // 6. post_attention_layernorm(g) -> gate/up -> swiglu -> down -> residual into g.
        {
            let e = self.eagle3.as_mut().unwrap();
            let norm_fn = self.functions["normalize"].clone();
            let eps = e.eps;
            let mut nb = self.stream.launch_builder(&norm_fn);
            nb.arg(&e.g)
                .arg(&e.post_attention_layernorm)
                .arg(&mut e.norm_buf)
                .arg(&width_i)
                .arg(&eps);
            unsafe {
                nb.launch(cudarc::driver::LaunchConfig {
                    grid_dim: (1, 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                })
            }
            .map_err(failure)?;

            let zero = 0i32;
            if matmul.f32_tiled()
                && e.gate_proj.kind == 12
                && e.up_proj.kind == 12
                && e.gate_proj.width % 256 == 0
            {
                let qkv_fn = self.functions["matvec_packed_qkv_rows"].clone();
                let total_rows = e.gate_proj.rows + e.up_proj.rows;
                let mut b = self.stream.launch_builder(&qkv_fn);
                b.arg(&e.gate_proj.bytes)
                    .arg(&e.gate_proj.kind)
                    .arg(&e.gate_proj.rows)
                    .arg(&mut e.gate)
                    .arg(&e.up_proj.bytes)
                    .arg(&e.up_proj.kind)
                    .arg(&e.up_proj.rows)
                    .arg(&mut e.up)
                    .arg(&e.up_proj.bytes)
                    .arg(&e.up_proj.kind)
                    .arg(&zero)
                    .arg(&mut e.proj)
                    .arg(&e.norm_buf)
                    .arg(&e.gate_proj.width);
                unsafe {
                    b.launch(cudarc::driver::LaunchConfig {
                        grid_dim: matvec_grid(matmul, 12, total_rows, 1, e.gate_proj.width),
                        block_dim: (256, 1, 1),
                        shared_mem_bytes: 0,
                    })
                }
                .map_err(failure)?;
            } else {
                for (proj, out) in [(&e.gate_proj, &mut e.gate), (&e.up_proj, &mut e.up)] {
                    let mut b = self.stream.launch_builder(&matvec_fn);
                    b.arg(&proj.bytes)
                        .arg(&proj.kind)
                        .arg(&e.norm_buf)
                        .arg(out)
                        .arg(&proj.width)
                        .arg(&proj.rows)
                        .arg(&self.work.ids)
                        .arg(&zero)
                        .arg(&zero);
                    unsafe {
                        b.launch(cudarc::driver::LaunchConfig {
                            grid_dim: matvec_grid(matmul, proj.kind, proj.rows, 1, proj.width),
                            block_dim: (256, 1, 1),
                            shared_mem_bytes: 0,
                        })
                    }
                    .map_err(failure)?;
                }
            }

            let inter_i = e.gate_proj.rows;
            let swiglu_fn = self.functions["swiglu"].clone();
            let mut sb = self.stream.launch_builder(&swiglu_fn);
            sb.arg(&mut e.gate).arg(&e.up).arg(&inter_i);
            unsafe {
                sb.launch(cudarc::driver::LaunchConfig {
                    grid_dim: vector(inter_i),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                })
            }
            .map_err(failure)?;

            let mut db = self.stream.launch_builder(&matvec_fn);
            db.arg(&e.down_proj.bytes)
                .arg(&e.down_proj.kind)
                .arg(&e.gate)
                .arg(&mut e.proj)
                .arg(&e.down_proj.width)
                .arg(&e.down_proj.rows)
                .arg(&self.work.ids)
                .arg(&zero)
                .arg(&zero);
            unsafe {
                db.launch(cudarc::driver::LaunchConfig {
                    grid_dim: matvec_grid(
                        matmul,
                        e.down_proj.kind,
                        e.down_proj.rows,
                        1,
                        e.down_proj.width,
                    ),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                })
            }
            .map_err(failure)?;

            let res_fn = self.functions["residual"].clone();
            let mut rb = self.stream.launch_builder(&res_fn);
            rb.arg(&mut e.g).arg(&e.proj).arg(&width_i);
            unsafe {
                rb.launch(cudarc::driver::LaunchConfig {
                    grid_dim: vector(width_i),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                })
            }
            .map_err(failure)?;
        }

        Ok(())
    }

    fn eagle3_midlayer_step(&mut self, token: i32, pos: usize) -> Result<()> {
        self.eagle3_midlayer_step_impl(token, None, pos)
    }

    /// Projects `eagle.g` -> `norm` -> `lm_head (32k)` -> `eagle3_argmax_d2t`
    /// directly into `eagle.draft_chain_dev[slot]` on GPU with ZERO CPU sync!
    fn eagle3_sample_greedy_to_slot(&mut self, slot: usize) -> Result<()> {
        let width_i = self.weights.spec.width as i32;
        let matvec_fn = self.functions["matvec"].clone();
        let norm_fn = self.functions["normalize"].clone();
        let argmax_fn = self.functions["eagle3_argmax_d2t"].clone();
        let matmul = self.matmul;
        let draft_vocab = self
            .eagle3
            .as_ref()
            .ok_or_else(|| invalid("EAGLE-3 not loaded"))?
            .draft_vocab;

        let e = self.eagle3.as_mut().unwrap();
        let eps = e.eps;
        let mut nb = self.stream.launch_builder(&norm_fn);
        nb.arg(&e.g)
            .arg(&e.norm)
            .arg(&mut e.norm_buf)
            .arg(&width_i)
            .arg(&eps);
        unsafe {
            nb.launch(cudarc::driver::LaunchConfig {
                grid_dim: (1, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            })
        }
        .map_err(failure)?;

        let zero = 0i32;
        let mut lb = self.stream.launch_builder(&matvec_fn);
        lb.arg(&e.lm_head.bytes)
            .arg(&e.lm_head.kind)
            .arg(&e.norm_buf)
            .arg(&mut e.draft_logits)
            .arg(&e.lm_head.width)
            .arg(&e.lm_head.rows)
            .arg(&self.work.ids)
            .arg(&zero)
            .arg(&zero);
        unsafe {
            lb.launch(cudarc::driver::LaunchConfig {
                grid_dim: matvec_grid(
                    matmul,
                    e.lm_head.kind,
                    e.lm_head.rows,
                    1,
                    e.lm_head.width,
                ),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            })
        }
        .map_err(failure)?;

        let count = draft_vocab as i32;
        let slot_i = slot as i32;
        let mut ab = self.stream.launch_builder(&argmax_fn);
        ab.arg(&e.draft_logits)
            .arg(&count)
            .arg(&e.d_d2t)
            .arg(&mut e.draft_chain_dev)
            .arg(&slot_i)
            .arg(&mut self.work.status);
        unsafe {
            ab.launch(cudarc::driver::LaunchConfig {
                grid_dim: (1, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            })
        }
        .map_err(failure)?;
        Ok(())
    }

    /// Projects `eagle.g` through `norm` -> `lm_head (32k)` -> device argmax,
    /// mapping `draft_id -> draft_id + d2t[draft_id]` into target vocab ID.
    fn eagle3_sample_greedy(&mut self) -> Result<i32> {
        let width_i = self.weights.spec.width as i32;
        let matvec_fn = self.functions["matvec"].clone();
        let norm_fn = self.functions["normalize"].clone();
        let matmul = self.matmul;
        let draft_vocab = self
            .eagle3
            .as_ref()
            .ok_or_else(|| invalid("EAGLE-3 not loaded"))?
            .draft_vocab;

        {
            let e = self.eagle3.as_mut().unwrap();
            let eps = e.eps;
            let mut nb = self.stream.launch_builder(&norm_fn);
            nb.arg(&e.g)
                .arg(&e.norm)
                .arg(&mut e.norm_buf)
                .arg(&width_i)
                .arg(&eps);
            unsafe {
                nb.launch(cudarc::driver::LaunchConfig {
                    grid_dim: (1, 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                })
            }
            .map_err(failure)?;

            let zero = 0i32;
            let mut lb = self.stream.launch_builder(&matvec_fn);
            lb.arg(&e.lm_head.bytes)
                .arg(&e.lm_head.kind)
                .arg(&e.norm_buf)
                .arg(&mut e.draft_logits)
                .arg(&e.lm_head.width)
                .arg(&e.lm_head.rows)
                .arg(&self.work.ids)
                .arg(&zero)
                .arg(&zero);
            unsafe {
                lb.launch(cudarc::driver::LaunchConfig {
                    grid_dim: matvec_grid(
                        matmul,
                        e.lm_head.kind,
                        e.lm_head.rows,
                        1,
                        e.lm_head.width,
                    ),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                })
            }
            .map_err(failure)?;
        }

        // Single-launch device argmax over 32k draft_logits (0 H2D copies!).
        let draft_id = {
            if self.work.batch_tokens.is_none() {
                self.work.batch_tokens = Some(self.stream.alloc_zeros(16).map_err(failure)?);
            }
            let e = self.eagle3.as_ref().unwrap();
            let count = draft_vocab as i32;
            let f = self.functions["greedy_batch_argmax"].clone();
            let out_tok = self.work.batch_tokens.as_mut().unwrap();
            let mut b1 = self.stream.launch_builder(&f);
            b1.arg(&e.draft_logits)
                .arg(&count)
                .arg(out_tok)
                .arg(&mut self.work.status);
            unsafe {
                b1.launch(cudarc::driver::LaunchConfig {
                    grid_dim: (1, 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                })
            }
            .map_err(failure)?;
            let mut res = [0i32; 1];
            self.stream
                .memcpy_dtoh(
                    &self.work.batch_tokens.as_ref().unwrap().slice(0..1),
                    &mut res,
                )
                .map_err(failure)?;
            self.stream.synchronize().map_err(failure)?;
            res[0] as usize
        };
        let offset = self.eagle3.as_ref().unwrap().d2t[draft_id];
        Ok(draft_id as i32 + offset)
    }

    /// Projects `eagle.g` -> `norm` -> `lm_head (32k)` and returns the top-`k`
    /// target vocabulary token IDs (`k <= 4`).
    fn eagle3_sample_topk(&mut self, k: usize) -> Result<Vec<i32>> {
        let width_i = self.weights.spec.width as i32;
        let matvec_fn = self.functions["matvec"].clone();
        let norm_fn = self.functions["normalize"].clone();
        let matmul = self.matmul;
        let draft_vocab = self
            .eagle3
            .as_ref()
            .ok_or_else(|| invalid("EAGLE-3 not loaded"))?
            .draft_vocab;

        {
            let e = self.eagle3.as_mut().unwrap();
            let eps = e.eps;
            let mut nb = self.stream.launch_builder(&norm_fn);
            nb.arg(&e.g)
                .arg(&e.norm)
                .arg(&mut e.norm_buf)
                .arg(&width_i)
                .arg(&eps);
            unsafe {
                nb.launch(cudarc::driver::LaunchConfig {
                    grid_dim: (1, 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                })
            }
            .map_err(failure)?;

            let zero = 0i32;
            let mut lb = self.stream.launch_builder(&matvec_fn);
            lb.arg(&e.lm_head.bytes)
                .arg(&e.lm_head.kind)
                .arg(&e.norm_buf)
                .arg(&mut e.draft_logits)
                .arg(&e.lm_head.width)
                .arg(&e.lm_head.rows)
                .arg(&self.work.ids)
                .arg(&zero)
                .arg(&zero);
            unsafe {
                lb.launch(cudarc::driver::LaunchConfig {
                    grid_dim: matvec_grid(
                        matmul,
                        e.lm_head.kind,
                        e.lm_head.rows,
                        1,
                        e.lm_head.width,
                    ),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                })
            }
            .map_err(failure)?;
        }

        let e = self.eagle3.as_ref().unwrap();
        let mut host_logits = vec![0.0f32; draft_vocab];
        self.stream
            .memcpy_dtoh(&e.draft_logits, &mut host_logits)
            .map_err(failure)?;
        self.stream.synchronize().map_err(failure)?;

        let mut best: Vec<(f32, usize)> = Vec::with_capacity(k + 1);
        for (idx, &val) in host_logits.iter().enumerate() {
            if best.len() < k || val > best.last().unwrap().0 {
                best.push((val, idx));
                best.sort_by(|a, b| b.0.total_cmp(&a.0));
                if best.len() > k {
                    best.pop();
                }
            }
        }
        Ok(best
            .into_iter()
            .map(|(_, idx)| idx as i32 + e.d2t[idx])
            .collect())
    }

    /// Generates a 16-node depth-6 EAGLE-3 draft tree rooted at `root` (with
    /// `eagle.g` already populated by `eagle3_ingest_tapped`), branching top-3
    /// at depth 1 and top-2 at depth 2 for high acceptance length `tau`.
    pub fn eagle3_draft_tree(
        &mut self,
        root: i32,
        start_pos: usize,
    ) -> Result<crate::speculative::SpecTree> {
        let mut nodes: Vec<(u32, i8)> = Vec::with_capacity(16);
        nodes.push((root as u32, -1)); // node 0

        // Save root g_0 into g_save0.
        {
            let width = self.weights.spec.width;
            let e = self.eagle3.as_mut().unwrap();
            let src = e.g.slice(0..width);
            self.stream
                .memcpy_dtod(&src, &mut e.g_save0)
                .map_err(failure)?;
        }

        // Depth 1: top-3 children of root (0 extra midlayer steps!).
        let top3 = self.eagle3_sample_topk(3)?;
        let (c0, c1, c2) = (top3[0], top3[1], top3[2]);
        nodes.push((c0 as u32, 0)); // node 1
        nodes.push((c1 as u32, 0)); // node 2
        nodes.push((c2 as u32, 0)); // node 3

        // Branch 1: c0 -> top-2 (nodes 4, 5)
        self.eagle3_midlayer_step(c0, start_pos)?;
        {
            let width = self.weights.spec.width;
            let e = self.eagle3.as_mut().unwrap();
            let src = e.g.slice(0..width);
            self.stream
                .memcpy_dtod(&src, &mut e.g_save1)
                .map_err(failure)?;
        }
        let b1_top2 = self.eagle3_sample_topk(2)?;
        let (c00, c01) = (b1_top2[0], b1_top2[1]);
        nodes.push((c00 as u32, 1)); // node 4
        nodes.push((c01 as u32, 1)); // node 5

        // Extend c00 greedily 4 more steps (depths 3, 4, 5, 6 -> nodes 6, 7, 8, 9).
        let mut cur_tok = c00;
        let mut cur_parent = 4i8;
        for step in 0..4usize {
            self.eagle3_midlayer_step(cur_tok, start_pos + 1 + step)?;
            let next = self.eagle3_sample_greedy()?;
            let idx = nodes.len() as i8;
            nodes.push((next as u32, cur_parent));
            cur_tok = next;
            cur_parent = idx;
        }

        // Restore g_save1 (after c0 at start_pos) -> extend c01 greedily 2 steps (nodes 10, 11).
        {
            let width = self.weights.spec.width;
            let e = self.eagle3.as_mut().unwrap();
            let mut dst = e.g.slice_mut(0..width);
            self.stream
                .memcpy_dtod(&e.g_save1, &mut dst)
                .map_err(failure)?;
        }
        cur_tok = c01;
        cur_parent = 5i8;
        for step in 0..2usize {
            self.eagle3_midlayer_step(cur_tok, start_pos + 1 + step)?;
            let next = self.eagle3_sample_greedy()?;
            let idx = nodes.len() as i8;
            nodes.push((next as u32, cur_parent));
            cur_tok = next;
            cur_parent = idx;
        }

        // Branch 2: restore g_save0 -> c1 -> top-2 (nodes 12, 13) + 1 greedy step on c10 (node 14).
        {
            let width = self.weights.spec.width;
            let e = self.eagle3.as_mut().unwrap();
            let mut dst = e.g.slice_mut(0..width);
            self.stream
                .memcpy_dtod(&e.g_save0, &mut dst)
                .map_err(failure)?;
        }
        self.eagle3_midlayer_step(c1, start_pos)?;
        let b2_top2 = self.eagle3_sample_topk(2)?;
        let (c10, c11) = (b2_top2[0], b2_top2[1]);
        nodes.push((c10 as u32, 2)); // node 12
        nodes.push((c11 as u32, 2)); // node 13
        self.eagle3_midlayer_step(c10, start_pos + 1)?;
        let c100 = self.eagle3_sample_greedy()?;
        nodes.push((c100 as u32, 12)); // node 14

        // Branch 3: restore g_save0 -> c2 -> top-1 (node 15).
        {
            let width = self.weights.spec.width;
            let e = self.eagle3.as_mut().unwrap();
            let mut dst = e.g.slice_mut(0..width);
            self.stream
                .memcpy_dtod(&e.g_save0, &mut dst)
                .map_err(failure)?;
        }
        self.eagle3_midlayer_step(c2, start_pos)?;
        let c20 = self.eagle3_sample_greedy()?;
        nodes.push((c20 as u32, 3)); // node 15

        Ok(crate::speculative::SpecTree::from_nodes(&nodes))
    }

    /// Compact 8-node depth-5 EAGLE-3 draft tree (only 5 midlayer steps +
    /// 1 single `mma.m16n8k16` verify pass on the target model!).
    /// Topology: root (0) -> top-2 (1, 2); extend 1 greedily to depth 5 (3, 4, 5, 6);
    /// extend 2 greedily 1 step (7).
    pub fn eagle3_draft_tree_8(
        &mut self,
        root: i32,
        start_pos: usize,
    ) -> Result<crate::speculative::SpecTree> {
        let mut nodes: Vec<(u32, i8)> = Vec::with_capacity(8);
        nodes.push((root as u32, -1)); // node 0

        {
            let width = self.weights.spec.width;
            let e = self.eagle3.as_mut().unwrap();
            let src = e.g.slice(0..width);
            self.stream
                .memcpy_dtod(&src, &mut e.g_save0)
                .map_err(failure)?;
        }

        let top2 = self.eagle3_sample_topk(2)?;
        let (c0, c1) = (top2[0], top2[1]);
        nodes.push((c0 as u32, 0)); // node 1
        nodes.push((c1 as u32, 0)); // node 2

        // Extend c0 greedily 4 steps (depths 2, 3, 4, 5 -> nodes 3, 4, 5, 6).
        let mut cur_tok = c0;
        let mut cur_parent = 1i8;
        for step in 0..4usize {
            self.eagle3_midlayer_step(cur_tok, start_pos + step)?;
            let next = self.eagle3_sample_greedy()?;
            let idx = nodes.len() as i8;
            nodes.push((next as u32, cur_parent));
            cur_tok = next;
            cur_parent = idx;
        }

        // Restore g_save0 -> extend c1 1 step (node 7).
        {
            let width = self.weights.spec.width;
            let e = self.eagle3.as_mut().unwrap();
            let mut dst = e.g.slice_mut(0..width);
            self.stream
                .memcpy_dtod(&e.g_save0, &mut dst)
                .map_err(failure)?;
        }
        self.eagle3_midlayer_step(c1, start_pos)?;
        let c10 = self.eagle3_sample_greedy()?;
        nodes.push((c10 as u32, 2)); // node 7

        Ok(crate::speculative::SpecTree::from_nodes(&nodes))
    }

    /// Batched EAGLE-3 midlayer pass for `2 <= n <= 8` Q4_K tokens already loaded
    /// into `eagle.g[0..n*H]`, writing KV at `start_pos..start_pos+n` and copying
    /// the last token's output `g[n-1]` into `eagle.g[0..H]`.
    fn eagle3_midlayer_batch(&mut self, tokens: &[i32], start_pos: usize) -> Result<()> {
        let n = tokens.len();
        let n_i = n as i32;
        let s = &self.weights.spec;
        let width = s.width;
        let width_i = width as i32;
        let heads = s.heads as i32;
        let kv_heads = s.kv_heads as i32;
        let dim = s.dim as i32;
        let qw = (s.heads * s.dim) as i32;
        let kw = (s.kv_heads * s.dim) as i32;
        let pos_i = start_pos as i32;
        let count_i = (start_pos + 1) as i32;
        let neox = 1i32;

        // 1. Embeddings for all `n` tokens via prefill_embedding (reusing pre-allocated batch_toks).
        {
            let e = self.eagle3.as_mut().unwrap();
            self.stream
                .memcpy_htod(tokens, &mut e.batch_toks.slice_mut(0..n))
                .map_err(failure)?;
            let emb_w = &self.weights.embedding;
            let func = self.functions["prefill_embedding"].clone();
            let mut b = self.stream.launch_builder(&func);
            b.arg(&emb_w.bytes)
                .arg(&emb_w.kind)
                .arg(&e.batch_toks)
                .arg(&mut e.emb)
                .arg(&width_i);
            unsafe {
                b.launch(cudarc::driver::LaunchConfig {
                    grid_dim: ((width as u32).div_ceil(256), n as u32, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                })
            }
            .map_err(failure)?;
        }

        // 2. Fused normalize(emb) + normalize(g) -> inp_cat[t * 2H ..] in 1 single launch!
        {
            let e = self.eagle3.as_mut().unwrap();
            let norm_cat_fn = self.functions["eagle3_norm_cat"].clone();
            let eps = e.eps;
            let mut b = self.stream.launch_builder(&norm_cat_fn);
            b.arg(&e.emb)
                .arg(&e.g)
                .arg(&e.input_layernorm)
                .arg(&e.hidden_norm)
                .arg(&mut e.inp_cat)
                .arg(&width_i)
                .arg(&eps);
            unsafe {
                b.launch(cudarc::driver::LaunchConfig {
                    grid_dim: (n as u32, 2, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                })
            }
            .map_err(failure)?;
        }

        // 3. Q, K, V projections (single weight read across all `n` tokens via matvec_q4k_multi8!).
        let q4k_fn = self.functions["matvec_q4k_multi8"].clone();
        {
            let e = self.eagle3.as_mut().unwrap();
            for (proj, out) in [
                (&e.q_proj, &mut e.q),
                (&e.k_proj, &mut e.k),
                (&e.v_proj, &mut e.v),
            ] {
                let mut b = self.stream.launch_builder(&q4k_fn);
                b.arg(&proj.bytes)
                    .arg(&e.inp_cat)
                    .arg(out)
                    .arg(&proj.width)
                    .arg(&proj.rows)
                    .arg(&n_i);
                unsafe {
                    b.launch(cudarc::driver::LaunchConfig {
                        grid_dim: ((proj.rows as u32).div_ceil(2), 1, 1),
                        block_dim: (256, 1, 1),
                        shared_mem_bytes: 0,
                    })
                }
                .map_err(failure)?;
            }
        }

        // 4. Fused batched RoPE (Q+K) + cache_kv in 1 launch, then causal attention_fused.
        {
            let e = self.eagle3.as_mut().unwrap();
            let theta = e.rope_theta;
            let null_factors = 0u64;
            let mag = 1.0f32;
            let total_pairs = ((qw / 2 + kw / 2) as u32).div_ceil(256);
            let fused_rk = self.functions["rotary_factored_cache_kv_batch"].clone();
            {
                let mut b = self.stream.launch_builder(&fused_rk);
                b.arg(&mut e.q)
                    .arg(&qw)
                    .arg(&mut e.k)
                    .arg(&e.v)
                    .arg(&mut e.keys)
                    .arg(&mut e.values)
                    .arg(&kw)
                    .arg(&dim)
                    .arg(&pos_i)
                    .arg(&theta)
                    .arg(&neox)
                    .arg(&null_factors)
                    .arg(&mag);
                unsafe {
                    b.launch(cudarc::driver::LaunchConfig {
                        grid_dim: (total_pairs, n as u32, 1),
                        block_dim: (256, 1, 1),
                        shared_mem_bytes: 0,
                    })
                }
                .map_err(failure)?;
            }
            let attn_fn = self.functions["attention_warp128"].clone();
            {
                let mut b = self.stream.launch_builder(&attn_fn);
                b.arg(&e.q)
                    .arg(&e.keys)
                    .arg(&e.values)
                    .arg(&mut e.attn)
                    .arg(&heads)
                    .arg(&kv_heads)
                    .arg(&count_i);
                unsafe {
                    b.launch(cudarc::driver::LaunchConfig {
                        grid_dim: (heads as u32, n as u32, 1),
                        block_dim: (32, 1, 1),
                        shared_mem_bytes: 0,
                    })
                }
                .map_err(failure)?;
            }
        }

        // 5. O_proj + residual + post_attention_layernorm + MLP (gate/up/swiglu/down) + residual.
        let total_w = (n * width) as i32;
        {
            let e = self.eagle3.as_mut().unwrap();
            let mut ob = self.stream.launch_builder(&q4k_fn);
            ob.arg(&e.o_proj.bytes)
                .arg(&e.attn)
                .arg(&mut e.proj)
                .arg(&e.o_proj.width)
                .arg(&e.o_proj.rows)
                .arg(&n_i);
            unsafe {
                ob.launch(cudarc::driver::LaunchConfig {
                    grid_dim: ((e.o_proj.rows as u32).div_ceil(2), 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                })
            }
            .map_err(failure)?;

            let res_fn = self.functions["residual"].clone();
            {
                let mut rb = self.stream.launch_builder(&res_fn);
                rb.arg(&mut e.g).arg(&e.proj).arg(&total_w);
                unsafe {
                    rb.launch(cudarc::driver::LaunchConfig {
                        grid_dim: vector(total_w),
                        block_dim: (256, 1, 1),
                        shared_mem_bytes: 0,
                    })
                }
                .map_err(failure)?;
            }

            let norm_fn = self.functions["normalize"].clone();
            let eps = e.eps;
            {
                let mut nb = self.stream.launch_builder(&norm_fn);
                nb.arg(&e.g)
                    .arg(&e.post_attention_layernorm)
                    .arg(&mut e.norm_buf)
                    .arg(&width_i)
                    .arg(&eps);
                unsafe {
                    nb.launch(cudarc::driver::LaunchConfig {
                        grid_dim: (n as u32, 1, 1),
                        block_dim: (256, 1, 1),
                        shared_mem_bytes: 0,
                    })
                }
                .map_err(failure)?;
            }

            for (proj, out) in [(&e.gate_proj, &mut e.gate), (&e.up_proj, &mut e.up)] {
                let mut b = self.stream.launch_builder(&q4k_fn);
                b.arg(&proj.bytes)
                    .arg(&e.norm_buf)
                    .arg(out)
                    .arg(&proj.width)
                    .arg(&proj.rows)
                    .arg(&n_i);
                unsafe {
                    b.launch(cudarc::driver::LaunchConfig {
                        grid_dim: ((proj.rows as u32).div_ceil(2), 1, 1),
                        block_dim: (256, 1, 1),
                        shared_mem_bytes: 0,
                    })
                }
                .map_err(failure)?;
            }

            let total_inter = (n * e.gate_proj.rows as usize) as i32;
            let swiglu_fn = self.functions["swiglu"].clone();
            {
                let mut sb = self.stream.launch_builder(&swiglu_fn);
                sb.arg(&mut e.gate).arg(&e.up).arg(&total_inter);
                unsafe {
                    sb.launch(cudarc::driver::LaunchConfig {
                        grid_dim: vector(total_inter),
                        block_dim: (256, 1, 1),
                        shared_mem_bytes: 0,
                    })
                }
                .map_err(failure)?;
            }

            {
                let mut db = self.stream.launch_builder(&q4k_fn);
                db.arg(&e.down_proj.bytes)
                    .arg(&e.gate)
                    .arg(&mut e.proj)
                    .arg(&e.down_proj.width)
                    .arg(&e.down_proj.rows)
                    .arg(&n_i);
                unsafe {
                    db.launch(cudarc::driver::LaunchConfig {
                        grid_dim: ((e.down_proj.rows as u32).div_ceil(2), 1, 1),
                        block_dim: (256, 1, 1),
                        shared_mem_bytes: 0,
                    })
                }
                .map_err(failure)?;
            }

            {
                let mut rb = self.stream.launch_builder(&res_fn);
                rb.arg(&mut e.g).arg(&e.proj).arg(&total_w);
                unsafe {
                    rb.launch(cudarc::driver::LaunchConfig {
                        grid_dim: vector(total_w),
                        block_dim: (256, 1, 1),
                        shared_mem_bytes: 0,
                    })
                }
                .map_err(failure)?;
            }

            // Copy last token's hidden state g[n-1] into g[0] for subsequent drafting.
            let last_off = (n - 1) * width;
            let (mut left, right) = e.g.split_at_mut(last_off);
            let src = right.slice(0..width);
            let mut dst = left.slice_mut(0..width);
            self.stream.memcpy_dtod(&src, &mut dst).map_err(failure)?;
        }

        Ok(())
    }

    /// Ingests `tokens.len()` accepted tokens paired with the target's tapped
    /// 3-layer hidden states (`tap_hidden[0..3][0..tokens.len()]`) starting at
    /// EAGLE-3 KV position `start_pos`, returning the first drafted token.
    pub fn eagle3_ingest_tapped(&mut self, tokens: &[i32], start_pos: usize) -> Result<i32> {
        Ok(self
            .eagle3_ingest_tapped_opt(tokens, start_pos, true)?
            .expect("sample_next=true always returns a token"))
    }

    /// Same as `eagle3_ingest_tapped`, but skips `lm_head` + GPU sync when
    /// `sample_next == false` (used by hybrid mode when SuffixDecoding handles
    /// the next step).
    pub fn eagle3_pending_start(&self) -> usize {
        self.eagle3_pending_start
    }

    pub fn eagle3_ingest_tapped_opt(
        &mut self,
        tokens: &[i32],
        start_pos: usize,
        sample_next: bool,
    ) -> Result<Option<i32>> {
        let width = self.weights.spec.width;
        // When target runs in the PLT1 R_1-rotated residual basis, unrotate the
        // 3 tapped hidden states back to the standard basis before EAGLE-3's `fc`.
        #[cfg(feature = "qtip-research")]
        if let Some(plt1) = &self.plt1_hyb
            && plt1.r1_signs.is_some()
        {
            let e = self.eagle3.as_mut().unwrap();
            for slot in 0..3 {
                plt1.r1_exit(&mut e.tap_hidden[slot], width, tokens.len())?;
            }
        }

        let is_q4k = self
            .eagle3
            .as_ref()
            .is_some_and(|e| e.fc.kind == 12);

        if is_q4k {
            let q4k_fn = self.functions["matvec_q4k_multi8"].clone();
            let mut offset = 0usize;
            while offset < tokens.len() {
                let chunk_len = (tokens.len() - offset).min(8);
                if chunk_len == 1 {
                    let j = offset;
                    let matvec_fn = self.functions["matvec"].clone();
                    let matmul = self.matmul;
                    {
                        let e = self.eagle3.as_mut().unwrap();
                        for slot in 0..3 {
                            let src = e.tap_hidden[slot].slice(j * width..(j + 1) * width);
                            let mut dst = e.fc_in.slice_mut(slot * width..(slot + 1) * width);
                            self.stream.memcpy_dtod(&src, &mut dst).map_err(failure)?;
                        }
                        let zero = 0i32;
                        let mut fb = self.stream.launch_builder(&matvec_fn);
                        fb.arg(&e.fc.bytes)
                            .arg(&e.fc.kind)
                            .arg(&e.fc_in)
                            .arg(&mut e.g)
                            .arg(&e.fc.width)
                            .arg(&e.fc.rows)
                            .arg(&self.work.ids)
                            .arg(&zero)
                            .arg(&zero);
                        unsafe {
                            fb.launch(cudarc::driver::LaunchConfig {
                                grid_dim: matvec_grid(matmul, e.fc.kind, e.fc.rows, 1, e.fc.width),
                                block_dim: (256, 1, 1),
                                shared_mem_bytes: 0,
                            })
                        }
                        .map_err(failure)?;
                    }
                    self.eagle3_midlayer_step(tokens[j], start_pos + j)?;
                } else {
                    {
                        let e = self.eagle3.as_mut().unwrap();
                        let concat_fn = self.functions["eagle3_concat3"].clone();
                        let t0 = e.tap_hidden[0].slice(offset * width..(offset + chunk_len) * width);
                        let t1 = e.tap_hidden[1].slice(offset * width..(offset + chunk_len) * width);
                        let t2 = e.tap_hidden[2].slice(offset * width..(offset + chunk_len) * width);
                        let width_i = width as i32;
                        let mut cb = self.stream.launch_builder(&concat_fn);
                        cb.arg(&t0)
                            .arg(&t1)
                            .arg(&t2)
                            .arg(&mut e.fc_in)
                            .arg(&width_i);
                        unsafe {
                            cb.launch(cudarc::driver::LaunchConfig {
                                grid_dim: ((width as u32).div_ceil(256), chunk_len as u32, 1),
                                block_dim: (256, 1, 1),
                                shared_mem_bytes: 0,
                            })
                        }
                        .map_err(failure)?;
                        let n_i = chunk_len as i32;
                        let mut fb = self.stream.launch_builder(&q4k_fn);
                        fb.arg(&e.fc.bytes)
                            .arg(&e.fc_in)
                            .arg(&mut e.g)
                            .arg(&e.fc.width)
                            .arg(&e.fc.rows)
                            .arg(&n_i);
                        unsafe {
                            fb.launch(cudarc::driver::LaunchConfig {
                                grid_dim: ((e.fc.rows as u32).div_ceil(2), 1, 1),
                                block_dim: (256, 1, 1),
                                shared_mem_bytes: 0,
                            })
                        }
                        .map_err(failure)?;
                    }
                    self.eagle3_midlayer_batch(
                        &tokens[offset..offset + chunk_len],
                        start_pos + offset,
                    )?;
                }
                offset += chunk_len;
            }
            return if sample_next {
                Ok(Some(self.eagle3_sample_greedy()?))
            } else {
                Ok(None)
            };
        }

        let matvec_fn = self.functions["matvec"].clone();
        let matmul = self.matmul;
        for (j, &tok) in tokens.iter().enumerate() {
            // Concatenate [tap0[j], tap1[j], tap2[j]] into eagle.fc_in -> fc -> eagle.g.
            {
                let e = self.eagle3.as_mut().unwrap();
                for slot in 0..3 {
                    let src = e.tap_hidden[slot].slice(j * width..(j + 1) * width);
                    let mut dst = e.fc_in.slice_mut(slot * width..(slot + 1) * width);
                    self.stream.memcpy_dtod(&src, &mut dst).map_err(failure)?;
                }
                let zero = 0i32;
                let mut fb = self.stream.launch_builder(&matvec_fn);
                fb.arg(&e.fc.bytes)
                    .arg(&e.fc.kind)
                    .arg(&e.fc_in)
                    .arg(&mut e.g)
                    .arg(&e.fc.width)
                    .arg(&e.fc.rows)
                    .arg(&self.work.ids)
                    .arg(&zero)
                    .arg(&zero);
                unsafe {
                    fb.launch(cudarc::driver::LaunchConfig {
                        grid_dim: matvec_grid(matmul, e.fc.kind, e.fc.rows, 1, e.fc.width),
                        block_dim: (256, 1, 1),
                        shared_mem_bytes: 0,
                    })
                }
                .map_err(failure)?;
            }
            self.eagle3_midlayer_step(tok, start_pos + j)?;
        }
        if sample_next {
            Ok(Some(self.eagle3_sample_greedy()?))
        } else {
            Ok(None)
        }
    }

    /// Zero-sync device-chained ingest: ingests `tokens` paired with tapped hidden
    /// states and writes the first drafted token into `draft_chain_dev[0]` on GPU
    /// (zero CPU-GPU sync!). Pair with `eagle3_draft_chain_from_slot0`.
    pub fn eagle3_ingest_tapped_to_dev(&mut self, tokens: &[i32], start_pos: usize) -> Result<()> {
        self.eagle3_ingest_tapped_opt(tokens, start_pos, false)?;
        self.eagle3_sample_greedy_to_slot(0)
    }

    /// Generates `count` greedy draft tokens starting from `draft_chain_dev[0]`
    /// (written by `eagle3_ingest_tapped_to_dev`) with ZERO intermediate syncs
    /// across `ingest` + all `count - 1` draft steps (1 single D2H read at the end!).
    pub fn eagle3_draft_chain_from_slot0(
        &mut self,
        start_pos: usize,
        count: usize,
    ) -> Result<Vec<i32>> {
        if count == 0 {
            return Ok(Vec::new());
        }
        for step in 0..(count - 1) {
            self.eagle3_midlayer_step_impl(0, Some(step), start_pos + step)?;
            self.eagle3_sample_greedy_to_slot(step + 1)?;
        }
        let mut drafted = vec![0i32; count];
        {
            let e = self.eagle3.as_ref().unwrap();
            self.stream
                .memcpy_dtoh(&e.draft_chain_dev.slice(0..count), &mut drafted)
                .map_err(failure)?;
        }
        self.stream.synchronize().map_err(failure)?;
        Ok(drafted)
    }

    /// Generates `count` greedy draft tokens starting from `first_draft` (which
    /// was returned by `eagle3_ingest_tapped`) at EAGLE-3 KV positions
    /// `start_pos .. start_pos + count - 1`.
    /// Runs all `count - 1` steps back-to-back on the GPU with ZERO intermediate
    /// CPU-GPU syncs (1 single D2H read at the end!).
    pub fn eagle3_draft_chain(
        &mut self,
        first_draft: i32,
        start_pos: usize,
        count: usize,
    ) -> Result<Vec<i32>> {
        if count == 0 {
            return Ok(Vec::new());
        }
        if count == 1 {
            return Ok(vec![first_draft]);
        }
        // Step 0 uses `first_draft` (host scalar); steps 1..count-1 read their
        // input token directly from `eagle.draft_chain_dev[step - 1]` on GPU!
        self.eagle3_midlayer_step_impl(first_draft, None, start_pos)?;
        self.eagle3_sample_greedy_to_slot(0)?;
        for step in 1..(count - 1) {
            self.eagle3_midlayer_step_impl(0, Some(step - 1), start_pos + step)?;
            self.eagle3_sample_greedy_to_slot(step)?;
        }
        let mut tail = vec![0i32; count - 1];
        {
            let e = self.eagle3.as_ref().unwrap();
            self.stream
                .memcpy_dtoh(&e.draft_chain_dev.slice(0..count - 1), &mut tail)
                .map_err(failure)?;
        }
        self.stream.synchronize().map_err(failure)?;
        let mut drafted = Vec::with_capacity(count);
        drafted.push(first_draft);
        drafted.extend_from_slice(&tail);
        Ok(drafted)
    }
}
