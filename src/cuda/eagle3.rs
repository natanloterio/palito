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

        let load_q8 = |file: &mut File, name: &str| -> Result<DeviceTensor> {
            let desc = index
                .tensors
                .iter()
                .find(|t| t.name == name)
                .ok_or_else(|| invalid(format!("missing EAGLE-3 tensor {name}")))?;
            let raw = read_bytes(file, desc.data_offset, desc.data_size)?;
            let bytes = self.stream.clone_htod(&raw).map_err(failure)?;
            Ok(DeviceTensor {
                bytes,
                kind: 8, // Q8_0
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

        let fc = load_q8(&mut file, "fc.weight")?;
        let q_proj = load_q8(&mut file, "midlayer.self_attn.q_proj.weight")?;
        let k_proj = load_q8(&mut file, "midlayer.self_attn.k_proj.weight")?;
        let v_proj = load_q8(&mut file, "midlayer.self_attn.v_proj.weight")?;
        let o_proj = load_q8(&mut file, "midlayer.self_attn.o_proj.weight")?;
        let gate_proj = load_q8(&mut file, "midlayer.mlp.gate_proj.weight")?;
        let up_proj = load_q8(&mut file, "midlayer.mlp.up_proj.weight")?;
        let down_proj = load_q8(&mut file, "midlayer.mlp.down_proj.weight")?;
        let lm_head = load_q8(&mut file, "lm_head.weight")?;

        let input_layernorm = load_norm(&mut file, "midlayer.input_layernorm.weight")?;
        let hidden_norm = load_norm(&mut file, "midlayer.hidden_norm.weight")?;
        let post_attention_layernorm =
            load_norm(&mut file, "midlayer.post_attention_layernorm.weight")?;
        let norm = load_norm(&mut file, "norm.weight")?;

        let s = &self.weights.spec;
        let width = s.width;
        let kv_width = s.kv_heads * s.dim;
        let inter = gate_proj.rows as usize;
        let context = s.info.context_tokens as usize;
        let batch_cap = self.work.x.len() / width;

        let alloc = |n: usize| self.stream.alloc_zeros::<f32>(n).map_err(failure);
        let tap_hidden = [
            alloc(batch_cap * width)?,
            alloc(batch_cap * width)?,
            alloc(batch_cap * width)?,
        ];

        self.eagle3 = Some(Eagle3State {
            tap_layers,
            tap_hidden,
            draft_vocab,
            d2t,
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
            fc_in: alloc(3 * width)?,
            g: alloc(width)?,
            emb: alloc(width)?,
            inp_cat: alloc(2 * width)?,
            q: alloc(width)?,
            k: alloc(kv_width)?,
            v: alloc(kv_width)?,
            attn: alloc(width)?,
            proj: alloc(width)?,
            norm_buf: alloc(width)?,
            gate: alloc(inter)?,
            up: alloc(inter)?,
            draft_logits: alloc(draft_vocab)?,
        });
        Ok(())
    }

    /// Runs one single-token EAGLE-3 midlayer step using `eagle.g` and `token`,
    /// writing KV at `pos` and updating `eagle.g` in-place.
    fn eagle3_midlayer_step(&mut self, token: i32, pos: usize) -> Result<()> {
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

        // 2. inp_cat[0..H] = input_layernorm(emb), inp_cat[H..2H] = hidden_norm(g).
        {
            let e = self.eagle3.as_mut().unwrap();
            let norm_fn = self.functions["normalize"].clone();
            let eps = e.eps;
            {
                let mut dst = e.inp_cat.slice_mut(0..width);
                let mut b = self.stream.launch_builder(&norm_fn);
                b.arg(&e.emb)
                    .arg(&e.input_layernorm)
                    .arg(&mut dst)
                    .arg(&width_i)
                    .arg(&eps);
                unsafe {
                    b.launch(cudarc::driver::LaunchConfig {
                        grid_dim: (1, 1, 1),
                        block_dim: (256, 1, 1),
                        shared_mem_bytes: 0,
                    })
                }
                .map_err(failure)?;
            }
            {
                let mut dst = e.inp_cat.slice_mut(width..2 * width);
                let mut b = self.stream.launch_builder(&norm_fn);
                b.arg(&e.g)
                    .arg(&e.hidden_norm)
                    .arg(&mut dst)
                    .arg(&width_i)
                    .arg(&eps);
                unsafe {
                    b.launch(cudarc::driver::LaunchConfig {
                        grid_dim: (1, 1, 1),
                        block_dim: (256, 1, 1),
                        shared_mem_bytes: 0,
                    })
                }
                .map_err(failure)?;
            }
        }

        // 3. Q, K, V projections (Q8_0, width=2*H).
        let matvec_fn = self.functions["matvec"].clone();
        let matmul = self.matmul;
        {
            let e = self.eagle3.as_mut().unwrap();
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

        // 4. RoPE (base=rope_theta=10000, neox=1) on Q and K + cache_kv + attention_fused.
        {
            let e = self.eagle3.as_mut().unwrap();
            let rotary_fn = self.functions["rotary"].clone();
            let theta = e.rope_theta;
            for (buf, n_el) in [(&mut e.q, qw), (&mut e.k, kw)] {
                let mut b = self.stream.launch_builder(&rotary_fn);
                b.arg(buf)
                    .arg(&n_el)
                    .arg(&dim)
                    .arg(&pos_i)
                    .arg(&theta)
                    .arg(&neox);
                unsafe {
                    b.launch(cudarc::driver::LaunchConfig {
                        grid_dim: vector(n_el / 2),
                        block_dim: (256, 1, 1),
                        shared_mem_bytes: 0,
                    })
                }
                .map_err(failure)?;
            }
            let cache_fn = self.functions["cache_kv"].clone();
            {
                let mut b = self.stream.launch_builder(&cache_fn);
                b.arg(&e.k)
                    .arg(&e.v)
                    .arg(&mut e.keys)
                    .arg(&mut e.values)
                    .arg(&kw)
                    .arg(&pos_i);
                unsafe {
                    b.launch(cudarc::driver::LaunchConfig {
                        grid_dim: vector(kw),
                        block_dim: (256, 1, 1),
                        shared_mem_bytes: 0,
                    })
                }
                .map_err(failure)?;
            }
            let attn_fn = self.functions["attention_fused"].clone();
            {
                let mut b = self.stream.launch_builder(&attn_fn);
                b.arg(&e.q)
                    .arg(&e.keys)
                    .arg(&e.values)
                    .arg(&mut e.attn)
                    .arg(&dim)
                    .arg(&heads)
                    .arg(&kv_heads)
                    .arg(&count_i);
                unsafe {
                    b.launch(cudarc::driver::LaunchConfig {
                        grid_dim: (heads as u32, 1, 1),
                        block_dim: (256, 1, 1),
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

        // Device argmax over the 32k draft_logits into decode slot 0.
        self.prepare_decode(0, 0, 0)?;
        let draft_id = {
            let e = self.eagle3.as_ref().unwrap();
            let count = draft_vocab as i32;
            let blocks = (count as u32).div_ceil(256) as i32;
            let gp = self.functions["greedy_partials"].clone();
            let gf = self.functions["greedy_finish"].clone();
            let decode = self
                .decode
                .as_mut()
                .ok_or_else(|| failure("EAGLE-3 requires decode state"))?;
            let mut b1 = self.stream.launch_builder(&gp);
            b1.arg(&e.draft_logits)
                .arg(&count)
                .arg(&mut decode.scores)
                .arg(&mut decode.ids)
                .arg(&mut self.work.status);
            unsafe {
                b1.launch(cudarc::driver::LaunchConfig {
                    grid_dim: (blocks as u32, 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                })
            }
            .map_err(failure)?;
            let mut b2 = self.stream.launch_builder(&gf);
            b2.arg(&decode.scores)
                .arg(&decode.ids)
                .arg(&blocks)
                .arg(&mut decode.control)
                .arg(&mut decode.tokens)
                .arg(&mut self.work.status);
            unsafe {
                b2.launch(cudarc::driver::LaunchConfig {
                    grid_dim: (1, 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                })
            }
            .map_err(failure)?;
            self.read_decode_tokens(1)?[0] as usize
        };
        let offset = self.eagle3.as_ref().unwrap().d2t[draft_id];
        Ok(draft_id as i32 + offset)
    }

    /// Ingests `tokens.len()` accepted tokens paired with the target's tapped
    /// 3-layer hidden states (`tap_hidden[0..3][0..tokens.len()]`) starting at
    /// EAGLE-3 KV position `start_pos`, returning the first drafted token.
    pub fn eagle3_ingest_tapped(&mut self, tokens: &[i32], start_pos: usize) -> Result<i32> {
        let width = self.weights.spec.width;
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
        self.eagle3_sample_greedy()
    }

    /// Generates `count` greedy draft tokens starting from `first_draft` (which
    /// was returned by `eagle3_ingest_tapped`) at EAGLE-3 KV positions
    /// `start_pos .. start_pos + count - 1`.
    pub fn eagle3_draft_chain(
        &mut self,
        first_draft: i32,
        start_pos: usize,
        count: usize,
    ) -> Result<Vec<i32>> {
        let mut drafted = Vec::with_capacity(count);
        if count == 0 {
            return Ok(drafted);
        }
        drafted.push(first_draft);
        let mut prev = first_draft;
        for step in 0..(count - 1) {
            self.eagle3_midlayer_step(prev, start_pos + step)?;
            let next = self.eagle3_sample_greedy()?;
            drafted.push(next);
            prev = next;
        }
        Ok(drafted)
    }
}
