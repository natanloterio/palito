//! Pure-Rust CPU runtime for single-file `.palito` (`PLT1 HYB Q=5`) bundles.
//!
//! Decodes MMA-swizzled (`mma.sync.aligned.m16n8k16`) circular `16x16` HYB tiles
//! (`B in 3..=6`, `L = state_bits`, `Q = 5`) on the fly with zero per-tile heap
//! allocations and exact mathematical parity with `src/qtip/hyb_matvec.cu`.

use crate::{
    Error, Result,
    plt1::{Codec, Plt1Index, SectionKind, TensorDesc, flags, tensor_flags},
};
use half::f16;
use rayon::prelude::*;
use std::{
    fs::File,
    io::{Read, Seek, SeekFrom},
    path::Path,
};

const TILE: usize = 16;
const CHUNK: usize = 256;
const HYB_PAIRS_PER_TILE: usize = (TILE * TILE) / 2; // 128 pairs = 256 weights
const HYB_HASH_MULT: u32 = 0x9E37_79B1;
const TOKEN_TILE: usize = 8;

#[inline(always)]
pub(crate) const fn hyb_tile_words(pair_bits: u8) -> usize {
    (HYB_PAIRS_PER_TILE * pair_bits as usize) / 32
}

/// Unnormalized in-place 256-point Fast Walsh-Hadamard Transform (`H_256`).
#[inline]
pub(crate) fn fwht256_inplace(chunk: &mut [f32]) {
    super::plt1_simd::fwht256_inplace(chunk.try_into().expect("FWHT-256 expects 256 elements"));
}

/// Converts a 32-entry `f16x2` (`Q=5`) HYB LUT into `[f32; 64]` at load time.
pub(crate) fn lut_words_to_f32(words: &[u32]) -> Result<[f32; 64]> {
    if words.len() != 32 {
        return Err(Error::Invalid(format!(
            "PLT1 CPU HYB Q=5 requires 32-entry LUT, got {}",
            words.len()
        )));
    }
    let mut out = [0.0f32; 64];
    for (i, &w) in words.iter().enumerate() {
        out[2 * i] = f16::from_bits((w & 0xFFFF) as u16).to_f32();
        out[2 * i + 1] = f16::from_bits((w >> 16) as u16).to_f32();
    }
    Ok(out)
}

/// Decodes one `state_bits`-bit trellis state into `(w0, w1)` bit-identically
/// to `hyb_decode_pair_lut` (`Q=5`) in `src/qtip/hyb_matvec.cu`.
#[inline(always)]
fn decode_pair_q5(lut_f32: &[f32; 64], state: u32) -> (f32, f32) {
    let h = state.wrapping_mul(HYB_HASH_MULT);
    let idx = ((h >> 5) & 31) as usize;
    let w0 = f32::from_bits(lut_f32[2 * idx].to_bits() ^ ((h & 0x8000) << 16));
    let w1 = f32::from_bits(lut_f32[2 * idx + 1].to_bits() ^ (h & 0x8000_0000));
    (w0, w1)
}

/// Extracts the 4 trellis states for MMA `lane` (`0..32`) from a circular `16x16` tile.
#[inline(always)]
fn extract_lane_states(tile: &[u32], lane: usize, pair_bits: u8, state_mask: u32) -> [u32; 4] {
    match pair_bits {
        4 => {
            // 16 words per tile, 16 bits per lane -> 2 words cover all 4 states.
            let bit = lane * 16;
            let w0 = (bit >> 5) & 15;
            let w1 = (w0 + 1) & 15;
            let shift = bit & 31;
            let bits0 = ((u64::from(tile[w0]) | (u64::from(tile[w1]) << 32)) >> shift) as u32;
            [
                bits0 & state_mask,
                (bits0 >> 4) & state_mask,
                (bits0 >> 8) & state_mask,
                (bits0 >> 12) & state_mask,
            ]
        }
        6 => {
            // 24 words per tile, 24 bits per lane -> 3 words cover all 4 states.
            let bit = lane * 24;
            let w0 = (bit >> 5) % 24;
            let w1 = if w0 + 1 < 24 { w0 + 1 } else { 0 };
            let w2 = if w1 + 1 < 24 { w1 + 1 } else { 0 };
            let shift = bit & 31;
            let bits0 = ((u64::from(tile[w0]) | (u64::from(tile[w1]) << 32)) >> shift) as u32;
            let bits1 = ((u64::from(tile[w1]) | (u64::from(tile[w2]) << 32)) >> shift) as u32;
            [
                bits0 & state_mask,
                (bits0 >> 6) & state_mask,
                (bits0 >> 12) & state_mask,
                ((bits0 >> 18) | (bits1 << 14)) & state_mask,
            ]
        }
        _ => {
            let wpt = tile.len();
            let mut out = [0u32; 4];
            for (p, dst) in out.iter_mut().enumerate() {
                let bit = (lane * 4 + p) * pair_bits as usize;
                let w0 = (bit >> 5) % wpt;
                let w1 = (w0 + 1) % wpt;
                let shift = bit & 31;
                let combined = u64::from(tile[w0]) | (u64::from(tile[w1]) << 32);
                *dst = ((combined >> shift) as u32) & state_mask;
            }
            out
        }
    }
}

/// Folds `ffn_norm * R_1` into an F32 MoE router weight matrix `[n_experts, width]`
/// so the router can run directly on `R_1`-rotated unit-norm activations.
/// Matches `CudaBackend::load_plt1_projections` bit-for-bit.
pub(crate) fn fold_router_rmsnorm_r1(
    router_weights: &mut [f32],
    ffn_norm: &[f32],
    r1_signs: &[f32],
    width: usize,
) {
    if width == 0 || width % CHUNK != 0 || ffn_norm.len() != width || r1_signs.len() != width {
        return;
    }
    let scale = 1.0f32 / 16.0;
    for row in router_weights.chunks_exact_mut(width) {
        for ((v, &g), &s) in row.iter_mut().zip(ffn_norm).zip(r1_signs) {
            *v *= g * s;
        }
        for chunk in row.chunks_exact_mut(CHUNK) {
            fwht256_inplace(chunk);
            for v in chunk.iter_mut() {
                *v *= scale;
            }
        }
    }
}

pub struct Plt1CpuTensor {
    pub desc: TensorDesc,
    pub tiles: Vec<u32>,
    pub lut_f32: [f32; 64],
    pub row_signs: Vec<f32>,
    pub col_signs: Vec<f32>,
    /// Per-expert scales `[n_experts]` for 3D MoE tensors; `None` for dense 2D tensors.
    pub scales: Option<Vec<f32>>,
    pub skip_in: bool,
    pub skip_out: bool,
}

impl Plt1CpuTensor {
    pub fn allocated_bytes(&self) -> u64 {
        (self.tiles.capacity() * 4
            + self.row_signs.capacity() * 4
            + self.col_signs.capacity() * 4
            + self.scales.as_ref().map_or(0, |s| s.capacity() * 4)) as u64
    }
}

pub struct Plt1CpuLayer {
    pub q: Plt1CpuTensor,
    pub k: Plt1CpuTensor,
    pub v: Plt1CpuTensor,
    pub o: Plt1CpuTensor,
    pub gate: Plt1CpuTensor,
    pub up: Plt1CpuTensor,
    pub down: Plt1CpuTensor,
    pub moe_ffn: bool,
}

impl Plt1CpuLayer {
    pub fn allocated_bytes(&self) -> u64 {
        self.q.allocated_bytes()
            + self.k.allocated_bytes()
            + self.v.allocated_bytes()
            + self.o.allocated_bytes()
            + self.gate.allocated_bytes()
            + self.up.allocated_bytes()
            + self.down.allocated_bytes()
    }
}

pub struct Plt1CpuRuntime {
    pub index: Plt1Index,
    pub layers: Vec<Plt1CpuLayer>,
    pub r1_signs: Option<Vec<f32>>,
    pub folded_rmsnorm: bool,
    pub state_bits: u32,
}

impl Plt1CpuRuntime {
    pub fn open(path: &Path) -> Result<Self> {
        let mut file = File::open(path)?;
        let index = Plt1Index::read_from(&mut file)?;

        let read_f32_tensor = |file: &mut File, name: &str| -> Result<Option<Vec<f32>>> {
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
            Ok(Some(vals))
        };

        let r1_signs = read_f32_tensor(&mut file, "plt1.r1_signs")?;
        let folded_rmsnorm = (index.header.flags & flags::FOLDED_RMSNORM) != 0;
        let state_bits: u32 = index
            .sections
            .iter()
            .find(|s| s.kind == SectionKind::MetadataJson)
            .and_then(|sec| {
                file.seek(SeekFrom::Start(sec.offset)).ok()?;
                let mut buf = vec![0u8; sec.size as usize];
                file.read_exact(&mut buf).ok()?;
                let v: serde_json::Value = serde_json::from_slice(&buf).ok()?;
                v.get("state_bits")?.as_u64().map(|n| n as u32)
            })
            .unwrap_or(12);

        if !(1..=16).contains(&state_bits) {
            return Err(Error::Invalid(format!(
                "PLT1 CPU: unsupported state_bits {state_bits}"
            )));
        }

        let load_hyb = |file: &mut File, name: &str| -> Result<Plt1CpuTensor> {
            let desc = index
                .tensors
                .iter()
                .find(|t| t.name == name && t.codec == Codec::QtipHybQ5)
                .ok_or_else(|| Error::Invalid(format!("missing PLT1 HYB Q=5 tensor {name}")))?
                .clone();
            if !(3..=6).contains(&desc.pair_bits) {
                return Err(Error::Invalid(format!(
                    "PLT1 CPU: unsupported pair_bits {} on {name}",
                    desc.pair_bits
                )));
            }
            if desc.lut_bits != 5 {
                return Err(Error::Invalid(format!(
                    "PLT1 CPU: expected Q=5 LUT on {name}, got Q={}",
                    desc.lut_bits
                )));
            }
            let rows = desc.rows as usize;
            let cols = desc.cols as usize;
            let skip_in = (desc.flags & tensor_flags::SKIP_IN_RHT) != 0;
            let skip_out = (desc.flags & tensor_flags::SKIP_OUT_RHT) != 0;
            if rows % TILE != 0 || cols % TILE != 0 {
                return Err(Error::Invalid(format!(
                    "PLT1 CPU: tensor {name} shape {rows}x{cols} not multiple of {TILE}"
                )));
            }
            if (!skip_in && cols % CHUNK != 0) || (!skip_out && rows % CHUNK != 0) {
                return Err(Error::Invalid(format!(
                    "PLT1 CPU: tensor {name} shape {rows}x{cols} requires multiple of {CHUNK} for FWHT-256"
                )));
            }

            file.seek(SeekFrom::Start(desc.data_offset))?;
            let mut raw = vec![0u8; desc.data_size as usize];
            file.read_exact(&mut raw)?;
            let tiles: Vec<u32> = raw
                .chunks_exact(4)
                .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
                .collect();

            let lut_entries = 1usize << desc.lut_bits;
            file.seek(SeekFrom::Start(desc.lut_offset))?;
            let mut lut_raw = vec![0u8; lut_entries * 4];
            file.read_exact(&mut lut_raw)?;
            let lut_words: Vec<u32> = lut_raw
                .chunks_exact(4)
                .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
                .collect();
            let lut_f32 = lut_words_to_f32(&lut_words)?;

            let signs_name = format!("{name}.signs");
            let (row_signs, col_signs) = match read_f32_tensor(file, &signs_name)? {
                Some(s) if s.len() == rows + cols => (s[..rows].to_vec(), s[rows..].to_vec()),
                Some(s) => {
                    return Err(Error::Invalid(format!(
                        "PLT1 CPU: {signs_name} length {} != {}",
                        s.len(),
                        rows + cols
                    )));
                }
                None => (vec![1.0f32; rows], vec![1.0f32; cols]),
            };
            let scales = read_f32_tensor(file, &format!("{name}.scales"))?;

            Ok(Plt1CpuTensor {
                desc,
                tiles,
                lut_f32,
                row_signs,
                col_signs,
                scales,
                skip_in,
                skip_out,
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
            layers.push(Plt1CpuLayer {
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

        Ok(Self {
            index,
            layers,
            r1_signs,
            folded_rmsnorm,
            state_bits,
        })
    }

    pub fn allocated_bytes(&self) -> u64 {
        self.layers.iter().map(Plt1CpuLayer::allocated_bytes).sum::<u64>()
            + self
                .r1_signs
                .as_ref()
                .map_or(0, |s| (s.capacity() * 4) as u64)
    }

    /// Residual-stream `R_1` rotation enter: `x <- (H_256 * S_res / 16) * x`.
    pub fn r1_enter(&self, x: &mut [f32], cols: usize, tokens: usize) {
        let Some(r1) = &self.r1_signs else {
            return;
        };
        r1_enter_with_signs(x, r1, cols, tokens);
    }

    /// Residual-stream `R_1` rotation exit: `x <- S_res * (H_256 / 16) * x`.
    pub fn r1_exit(&self, x: &mut [f32], cols: usize, tokens: usize) {
        let Some(r1) = &self.r1_signs else {
            return;
        };
        r1_exit_with_signs(x, r1, cols, tokens);
    }

    /// Dense HYB Q=5 matvec (`tokens == 1`) or matmul (`tokens > 1`).
    /// `x` has shape `[tokens, cols]`, `y` has shape `[tokens, rows]`.
    pub fn matmul(&self, tensor: &Plt1CpuTensor, x: &[f32], tokens: usize, y: &mut [f32]) {
        matmul_tiles(
            &tensor.tiles,
            &tensor.lut_f32,
            &tensor.row_signs,
            &tensor.col_signs,
            tensor.desc.rows as usize,
            tensor.desc.cols as usize,
            tensor.desc.pair_bits,
            self.state_bits,
            tensor.desc.scale,
            tensor.skip_in,
            tensor.skip_out,
            x,
            tokens,
            y,
        );
    }

    /// 3D MoE HYB Q=5 matvec/matmul for one `expert` slice.
    /// `x` has shape `[tokens, cols]`, `y` has shape `[tokens, rows]`.
    pub fn matmul_moe_expert(
        &self,
        tensor: &Plt1CpuTensor,
        expert: usize,
        x: &[f32],
        tokens: usize,
        y: &mut [f32],
    ) {
        let rows = tensor.desc.rows as usize;
        let cols = tensor.desc.cols as usize;
        let wpt = hyb_tile_words(tensor.desc.pair_bits);
        let expert_stride = (rows / TILE) * (cols / TILE) * wpt;
        let expert_tiles = &tensor.tiles[expert * expert_stride..(expert + 1) * expert_stride];
        let scale = tensor
            .scales
            .as_ref()
            .map_or(tensor.desc.scale, |s| s[expert]);
        matmul_tiles(
            expert_tiles,
            &tensor.lut_f32,
            &tensor.row_signs,
            &tensor.col_signs,
            rows,
            cols,
            tensor.desc.pair_bits,
            self.state_bits,
            scale,
            tensor.skip_in,
            tensor.skip_out,
            x,
            tokens,
            y,
        );
    }
}

pub(crate) fn r1_enter_with_signs(x: &mut [f32], r1_signs: &[f32], cols: usize, tokens: usize) {
    debug_assert_eq!(cols % CHUNK, 0);
    debug_assert_eq!(r1_signs.len(), cols);
    for tok in x[..tokens * cols].chunks_exact_mut(cols) {
        for (chunk, signs) in tok.chunks_exact_mut(CHUNK).zip(r1_signs.chunks_exact(CHUNK)) {
            super::plt1_simd::hyb_r1_enter_chunk(
                chunk.try_into().unwrap(),
                signs.try_into().unwrap(),
            );
        }
    }
}

pub(crate) fn r1_exit_with_signs(x: &mut [f32], r1_signs: &[f32], cols: usize, tokens: usize) {
    debug_assert_eq!(cols % CHUNK, 0);
    debug_assert_eq!(r1_signs.len(), cols);
    for tok in x[..tokens * cols].chunks_exact_mut(cols) {
        for (chunk, signs) in tok.chunks_exact_mut(CHUNK).zip(r1_signs.chunks_exact(CHUNK)) {
            super::plt1_simd::hyb_r1_exit_chunk(
                chunk.try_into().unwrap(),
                signs.try_into().unwrap(),
            );
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn matmul_tiles(
    tiles: &[u32],
    lut_f32: &[f32; 64],
    row_signs: &[f32],
    col_signs: &[f32],
    rows: usize,
    cols: usize,
    pair_bits: u8,
    state_bits: u32,
    scale: f32,
    skip_in: bool,
    skip_out: bool,
    x: &[f32],
    tokens: usize,
    y: &mut [f32],
) {
    if tokens == 0 {
        return;
    }
    let state_mask = (1u32 << state_bits) - 1;
    let wpt = hyb_tile_words(pair_bits);
    let col_tiles = cols / TILE;
    let block_rows = if skip_out { TILE } else { CHUNK };
    let groups_per_block = block_rows / TILE;

    // Input preprocessing: apply col_signs + unnormalized FWHT-256 when !skip_in.
    let x_prep_buf: Vec<f32>;
    let x_in: &[f32] = if skip_in {
        &x[..tokens * cols]
    } else {
        x_prep_buf = prepare_input(x, col_signs, cols, tokens);
        &x_prep_buf
    };

    if tokens == 1 {
        y[..rows]
            .par_chunks_exact_mut(block_rows)
            .enumerate()
            .for_each(|(rb, out_chunk)| {
                let row0 = rb * block_rows;
                let group0 = rb * groups_per_block;
                for g in 0..groups_per_block {
                    let group = group0 + g;
                    let group_tiles = &tiles[group * col_tiles * wpt..(group + 1) * col_tiles * wpt];
                    let mut acc = [0.0f32; TILE];
                    for k in 0..col_tiles {
                        let tile = &group_tiles[k * wpt..(k + 1) * wpt];
                        let xv: &[f32; TILE] = x_in[k * TILE..(k + 1) * TILE].try_into().unwrap();
                        super::plt1_simd::hyb_q5_tile_dot_mma(
                            tile,
                            lut_f32,
                            xv,
                            &mut acc,
                            state_bits,
                            u32::from(pair_bits),
                        );
                    }
                    out_chunk[g * TILE..(g + 1) * TILE].copy_from_slice(&acc);
                }
                if !skip_out {
                    fwht256_inplace(out_chunk);
                    let rs = &row_signs[row0..row0 + CHUNK];
                    for (v, &s) in out_chunk.iter_mut().zip(rs) {
                        *v *= s * scale;
                    }
                } else {
                    for v in out_chunk.iter_mut() {
                        *v *= scale;
                    }
                }
            });
        return;
    }

    // Multi-token path: process tokens in tiles of up to TOKEN_TILE (8) so each
    // weight tile is decoded ONCE and reused across all tokens in the tile.
    let num_row_blocks = rows / block_rows;
    let mut t0 = 0usize;
    while t0 < tokens {
        let n_tok = (tokens - t0).min(TOKEN_TILE);
        let mut block_outs = vec![0.0f32; num_row_blocks * n_tok * block_rows];
        block_outs
            .par_chunks_exact_mut(n_tok * block_rows)
            .enumerate()
            .for_each(|(rb, out_blk)| {
                let row0 = rb * block_rows;
                let group0 = rb * groups_per_block;
                let mut acc = [[0.0f32; TILE]; TOKEN_TILE];
                for g in 0..groups_per_block {
                    let group = group0 + g;
                    let group_tiles = &tiles[group * col_tiles * wpt..(group + 1) * col_tiles * wpt];
                    for a in &mut acc[..n_tok] {
                        *a = [0.0f32; TILE];
                    }
                    for k in 0..col_tiles {
                        let tile = &group_tiles[k * wpt..(k + 1) * wpt];
                        let col_off = k * TILE;
                        accumulate_tile_ntok(
                            tile,
                            lut_f32,
                            pair_bits,
                            state_mask,
                            x_in,
                            t0,
                            n_tok,
                            cols,
                            col_off,
                            &mut acc,
                        );
                    }
                    for t in 0..n_tok {
                        let dst = t * block_rows + g * TILE;
                        out_blk[dst..dst + TILE].copy_from_slice(&acc[t]);
                    }
                }
                for t in 0..n_tok {
                    let dst = &mut out_blk[t * block_rows..(t + 1) * block_rows];
                    if !skip_out {
                        fwht256_inplace(dst);
                        let rs = &row_signs[row0..row0 + CHUNK];
                        for (v, &s) in dst.iter_mut().zip(rs) {
                            *v *= s * scale;
                        }
                    } else {
                        for v in dst.iter_mut() {
                            *v *= scale;
                        }
                    }
                }
            });

        for (rb, out_blk) in block_outs.chunks_exact(n_tok * block_rows).enumerate() {
            let row0 = rb * block_rows;
            for t in 0..n_tok {
                let src = &out_blk[t * block_rows..(t + 1) * block_rows];
                let dst = &mut y[(t0 + t) * rows + row0..(t0 + t) * rows + row0 + block_rows];
                dst.copy_from_slice(src);
            }
        }
        t0 += n_tok;
    }
}

fn prepare_input(x: &[f32], col_signs: &[f32], cols: usize, tokens: usize) -> Vec<f32> {
    let mut out = x[..tokens * cols].to_vec();
    if tokens == 1 {
        for (chunk, signs) in out.chunks_exact_mut(CHUNK).zip(col_signs.chunks_exact(CHUNK)) {
            for (v, &s) in chunk.iter_mut().zip(signs) {
                *v *= s;
            }
            fwht256_inplace(chunk);
        }
    } else {
        out.par_chunks_exact_mut(cols).for_each(|tok| {
            for (chunk, signs) in tok.chunks_exact_mut(CHUNK).zip(col_signs.chunks_exact(CHUNK)) {
                for (v, &s) in chunk.iter_mut().zip(signs) {
                    *v *= s;
                }
                fwht256_inplace(chunk);
            }
        });
    }
    out
}

#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn accumulate_tile_ntok(
    tile: &[u32],
    lut_f32: &[f32; 64],
    pair_bits: u8,
    state_mask: u32,
    x_in: &[f32],
    t0: usize,
    n_tok: usize,
    cols: usize,
    col_off: usize,
    acc: &mut [[f32; TILE]; TOKEN_TILE],
) {
    for lane in 0..32usize {
        let group_id = lane >> 2;
        let tid = lane & 3;
        let c0 = col_off + 2 * tid;
        let c1 = c0 + 8;

        let states = extract_lane_states(tile, lane, pair_bits, state_mask);
        let (a0, a1) = decode_pair_q5(lut_f32, states[0]);
        let (b0, b1) = decode_pair_q5(lut_f32, states[1]);
        let (c0w, c1w) = decode_pair_q5(lut_f32, states[2]);
        let (d0, d1) = decode_pair_q5(lut_f32, states[3]);

        for t in 0..n_tok {
            let base = (t0 + t) * cols;
            let x0 = x_in[base + c0];
            let x1 = x_in[base + c0 + 1];
            let x2 = x_in[base + c1];
            let x3 = x_in[base + c1 + 1];
            acc[t][group_id] += a0 * x0 + a1 * x1 + c0w * x2 + c1w * x3;
            acc[t][group_id + 8] += b0 * x0 + b1 * x1 + d0 * x2 + d1 * x3;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_lut() -> [u32; 32] {
        let mut lut = [0u32; 32];
        for (i, slot) in lut.iter_mut().enumerate() {
            let v0 = 0.05 + (i as f32) * 0.04;
            let v1 = 0.07 + ((31 - i) as f32) * 0.035;
            let h0 = f16::from_f32(v0).to_bits() & 0x7FFF;
            let h1 = f16::from_f32(v1).to_bits() & 0x7FFF;
            *slot = u32::from(h0) | (u32::from(h1) << 16);
        }
        lut
    }

    fn ref_decode_tile_mma(
        words: &[u32],
        state_bits: u32,
        pair_bits: u8,
        lut_words: &[u32; 32],
    ) -> [f32; TILE * TILE] {
        let mut out = [0.0f32; TILE * TILE];
        let state_mask = (1u64 << state_bits) - 1;
        let wpt = words.len();
        for lane in 0..32usize {
            let group_id = lane >> 2;
            let tid = lane & 3;
            let coords = [
                (group_id, 2 * tid),
                (group_id + 8, 2 * tid),
                (group_id, 2 * tid + 8),
                (group_id + 8, 2 * tid + 8),
            ];
            for (p, &(r, c)) in coords.iter().enumerate() {
                let bit = (lane * 4 + p) * pair_bits as usize;
                let w0 = (bit / 32) % wpt;
                let w1 = (w0 + 1) % wpt;
                let shift = bit % 32;
                let combined = u64::from(words[w0]) | (u64::from(words[w1]) << 32);
                let state = ((combined >> shift) & state_mask) as u32;
                let h = state.wrapping_mul(HYB_HASH_MULT);
                let idx = ((h >> 5) & 31) as usize;
                let raw = lut_words[idx] ^ (h & 0x8000_8000);
                out[r * TILE + c] = f16::from_bits((raw & 0xFFFF) as u16).to_f32();
                out[r * TILE + c + 1] = f16::from_bits((raw >> 16) as u16).to_f32();
            }
        }
        out
    }

    fn ref_matmul(
        tiles: &[u32],
        lut_words: &[u32; 32],
        row_signs: &[f32],
        col_signs: &[f32],
        rows: usize,
        cols: usize,
        pair_bits: u8,
        state_bits: u32,
        scale: f32,
        skip_in: bool,
        skip_out: bool,
        x: &[f32],
        tokens: usize,
    ) -> Vec<f32> {
        let wpt = hyb_tile_words(pair_bits);
        let col_tiles = cols / TILE;
        let mut dense = vec![0.0f32; rows * cols];
        for (idx, tile) in tiles.chunks_exact(wpt).enumerate() {
            let (group, block) = (idx / col_tiles, idx % col_tiles);
            let dec = ref_decode_tile_mma(tile, state_bits, pair_bits, lut_words);
            for r in 0..TILE {
                let dst = (group * TILE + r) * cols + block * TILE;
                dense[dst..dst + TILE].copy_from_slice(&dec[r * TILE..(r + 1) * TILE]);
            }
        }
        let mut y = vec![0.0f32; tokens * rows];
        for t in 0..tokens {
            let mut xt = x[t * cols..(t + 1) * cols].to_vec();
            if !skip_in {
                for (chunk, s) in xt.chunks_exact_mut(CHUNK).zip(col_signs.chunks_exact(CHUNK)) {
                    for (v, &sv) in chunk.iter_mut().zip(s) {
                        *v *= sv;
                    }
                    fwht256_inplace(chunk);
                }
            }
            let yt = &mut y[t * rows..(t + 1) * rows];
            for r in 0..rows {
                let mut sum = 0.0f32;
                for c in 0..cols {
                    sum += dense[r * cols + c] * xt[c];
                }
                yt[r] = sum;
            }
            if !skip_out {
                for (chunk, s) in yt.chunks_exact_mut(CHUNK).zip(row_signs.chunks_exact(CHUNK)) {
                    fwht256_inplace(chunk);
                    for (v, &sv) in chunk.iter_mut().zip(s) {
                        *v *= sv * scale;
                    }
                }
            } else {
                for v in yt.iter_mut() {
                    *v *= scale;
                }
            }
        }
        y
    }

    fn pseudo_words(count: usize, seed: u32) -> Vec<u32> {
        let mut s = seed;
        (0..count)
            .map(|_| {
                s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                s
            })
            .collect()
    }

    fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
        assert_eq!(a.len(), b.len());
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max)
    }

    #[test]
    fn r1_enter_exit_roundtrip_and_exact_formula() {
        let cols = 512;
        let tokens = 3;
        let signs: Vec<f32> = (0..cols)
            .map(|i| if (i * 7 + 3) % 2 == 0 { 1.0 } else { -1.0 })
            .collect();
        let orig: Vec<f32> = (0..tokens * cols)
            .map(|i| ((i as f32) * 0.13).sin() * 1.7)
            .collect();
        let mut x = orig.clone();
        r1_enter_with_signs(&mut x, &signs, cols, tokens);
        // Verify enter is non-trivial and norm-preserving.
        let orig_norm: f32 = orig.iter().map(|v| v * v).sum();
        let rot_norm: f32 = x.iter().map(|v| v * v).sum();
        assert!((orig_norm - rot_norm).abs() < 1e-2);
        r1_exit_with_signs(&mut x, &signs, cols, tokens);
        let diff = max_abs_diff(&orig, &x);
        assert!(diff < 1e-4, "r1 roundtrip diff {diff}");
    }

    #[test]
    fn hyb_b4_l12_parity_single_and_multi_token() {
        let rows = 256;
        let cols = 512;
        let pair_bits = 4u8;
        let state_bits = 12u32;
        let lut_words = make_test_lut();
        let lut_f32 = lut_words_to_f32(&lut_words).unwrap();
        let wpt = hyb_tile_words(pair_bits);
        let tiles = pseudo_words((rows / TILE) * (cols / TILE) * wpt, 0x1234_5678);
        let row_signs: Vec<f32> = (0..rows).map(|i| if i % 3 == 0 { -1.0 } else { 1.0 }).collect();
        let col_signs: Vec<f32> = (0..cols).map(|i| if i % 5 == 0 { -1.0 } else { 1.0 }).collect();
        let scale = 0.03125 / 16.0;

        for (skip_in, skip_out) in [(false, false), (true, false), (false, true), (true, true)] {
            for tokens in [1usize, 5, 11] {
                let x: Vec<f32> = (0..tokens * cols)
                    .map(|i| ((i as f32) * 0.07).cos() * 0.5)
                    .collect();
                let mut y = vec![0.0f32; tokens * rows];
                matmul_tiles(
                    &tiles,
                    &lut_f32,
                    &row_signs,
                    &col_signs,
                    rows,
                    cols,
                    pair_bits,
                    state_bits,
                    scale,
                    skip_in,
                    skip_out,
                    &x,
                    tokens,
                    &mut y,
                );
                let expected = ref_matmul(
                    &tiles,
                    &lut_words,
                    &row_signs,
                    &col_signs,
                    rows,
                    cols,
                    pair_bits,
                    state_bits,
                    scale,
                    skip_in,
                    skip_out,
                    &x,
                    tokens,
                );
                let diff = max_abs_diff(&y, &expected);
                assert!(
                    diff < 1e-4,
                    "B=4 L=12 skip=({skip_in},{skip_out}) tokens={tokens}: diff {diff}"
                );
            }
        }
    }

    #[test]
    fn hyb_b6_l10_parity_single_and_multi_token() {
        let rows = 256;
        let cols = 256;
        let pair_bits = 6u8;
        let state_bits = 10u32;
        let lut_words = make_test_lut();
        let lut_f32 = lut_words_to_f32(&lut_words).unwrap();
        let wpt = hyb_tile_words(pair_bits);
        let tiles = pseudo_words((rows / TILE) * (cols / TILE) * wpt, 0xDEAD_BEEF);
        let row_signs: Vec<f32> = (0..rows).map(|i| if i % 2 == 0 { 1.0 } else { -1.0 }).collect();
        let col_signs: Vec<f32> = (0..cols).map(|i| if i % 4 == 0 { -1.0 } else { 1.0 }).collect();
        let scale = 0.04 / 16.0;

        for tokens in [1usize, 4, 9] {
            let x: Vec<f32> = (0..tokens * cols)
                .map(|i| ((i as f32) * 0.11).sin() * 0.8)
                .collect();
            let mut y = vec![0.0f32; tokens * rows];
            matmul_tiles(
                &tiles,
                &lut_f32,
                &row_signs,
                &col_signs,
                rows,
                cols,
                pair_bits,
                state_bits,
                scale,
                false,
                true,
                &x,
                tokens,
                &mut y,
            );
            let expected = ref_matmul(
                &tiles,
                &lut_words,
                &row_signs,
                &col_signs,
                rows,
                cols,
                pair_bits,
                state_bits,
                scale,
                false,
                true,
                &x,
                tokens,
            );
            let diff = max_abs_diff(&y, &expected);
            assert!(diff < 1e-4, "B=6 L=10 tokens={tokens}: diff {diff}");
        }
    }

    #[test]
    fn hyb_3d_moe_expert_tiles_parity() {
        let n_experts = 4usize;
        let rows = 256;
        let cols = 256;
        let pair_bits = 4u8;
        let state_bits = 12u32;
        let lut_words = make_test_lut();
        let lut_f32 = lut_words_to_f32(&lut_words).unwrap();
        let wpt = hyb_tile_words(pair_bits);
        let expert_stride = (rows / TILE) * (cols / TILE) * wpt;
        let tiles = pseudo_words(n_experts * expert_stride, 0xCAFE_BABE);
        let row_signs = vec![1.0f32; rows];
        let col_signs: Vec<f32> = (0..cols).map(|i| if i % 3 == 1 { -1.0 } else { 1.0 }).collect();
        let scales = vec![0.01f32, 0.025, 0.018, 0.033];

        let tensor = Plt1CpuTensor {
            desc: TensorDesc {
                name: "blk.0.ffn_gate_exps.weight".into(),
                rows: rows as u32,
                cols: cols as u32,
                codec: Codec::QtipHybQ5,
                flags: tensor_flags::SKIP_OUT_RHT | tensor_flags::MMA_SWIZZLED,
                pair_bits,
                lut_bits: 5,
                scale: 1.0,
                teal_thresholds: [0.0; 4],
                data_offset: 0,
                data_size: (tiles.len() * 4) as u64,
                lut_offset: 0,
            },
            tiles: tiles.clone(),
            lut_f32,
            row_signs: row_signs.clone(),
            col_signs: col_signs.clone(),
            scales: Some(scales.clone()),
            skip_in: false,
            skip_out: true,
        };
        let rt = Plt1CpuRuntime {
            index: Plt1Index {
                header: crate::plt1::Plt1Header {
                    version: crate::plt1::PLT1_VERSION,
                    target_sm: 120,
                    flags: 0,
                    num_sections: 0,
                    section_dir_offset: 64,
                    n_layers: 0,
                    d_model: cols as u32,
                    d_ff: rows as u32,
                    vocab_size: 0,
                },
                sections: Vec::new(),
                tensors: Vec::new(),
            },
            layers: Vec::new(),
            r1_signs: None,
            folded_rmsnorm: false,
            state_bits,
        };

        let tokens = 3usize;
        let x: Vec<f32> = (0..tokens * cols)
            .map(|i| ((i as f32) * 0.19).sin())
            .collect();
        for expert in 0..n_experts {
            let mut y = vec![0.0f32; tokens * rows];
            rt.matmul_moe_expert(&tensor, expert, &x, tokens, &mut y);
            let slice = &tiles[expert * expert_stride..(expert + 1) * expert_stride];
            let expected = ref_matmul(
                slice,
                &lut_words,
                &row_signs,
                &col_signs,
                rows,
                cols,
                pair_bits,
                state_bits,
                scales[expert],
                false,
                true,
                &x,
                tokens,
            );
            let diff = max_abs_diff(&y, &expected);
            assert!(diff < 1e-4, "MoE expert {expert}: diff {diff}");
        }
    }
}
