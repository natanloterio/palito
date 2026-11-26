//! CUDA-accelerated batch circular 2-pass tail-biting Viterbi quantizer for QTIP HYB.
//!
//! Compiles `hyb_viterbi_cuda.cu` via NVRTC for Blackwell (`compute_120`) and
//! quantizes an entire batch of `N` MMA-swizzled, sigma-normalized `16x16` tiles
//! in a single GPU launch (one threadblock per tile).

use super::{
    hyb::{HybCodebook, hyb_tile_words},
    ldlq::TILE,
};
use crate::{Error, Result};
use cudarc::{
    driver::{CudaContext, CudaFunction, CudaSlice, CudaStream, LaunchConfig, PushKernelArg},
    nvrtc::{CompileOptions, compile_ptx_with_opts},
};
use std::sync::{Arc, Mutex};

const WEIGHTS_PER_TILE: usize = TILE * TILE; // 256

fn cuda_err(e: impl std::fmt::Display) -> Error {
    Error::Backend(format!("CUDA HYB Viterbi: {e}"))
}

struct ScratchBuffers {
    cap_tiles: usize,
    d_in: CudaSlice<f32>,
    d_words: CudaSlice<u32>,
    d_decoded: CudaSlice<f32>,
}

/// GPU-accelerated HYB Viterbi quantizer for a fixed `(state_bits, pair_bits, codebook)`.
pub struct CudaHybQuantizer {
    pub state_bits: u32,
    pub pair_bits: u32,
    pub codebook: HybCodebook,
    block_threads: u32,
    words_per_tile: usize,
    _ctx: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    kernel: CudaFunction,
    d_lut: CudaSlice<u32>,
    scratch: Mutex<Option<ScratchBuffers>>,
}

impl CudaHybQuantizer {
    /// Compiles the specialized HYB Viterbi kernel for `(state_bits, pair_bits, codebook.q_bits)`
    /// targeting `compute_120` and uploads the codebook LUT to the device.
    pub fn new(
        ctx: Arc<CudaContext>,
        state_bits: u32,
        pair_bits: u32,
        codebook: HybCodebook,
    ) -> Result<Self> {
        if !(8..=13).contains(&state_bits) {
            return Err(Error::Invalid(format!(
                "CUDA HYB state_bits must be in 8..=13, got {state_bits}"
            )));
        }
        if !(3..=6).contains(&pair_bits) || state_bits <= pair_bits {
            return Err(Error::Invalid(format!(
                "CUDA HYB pair_bits must be in 3..=6 and < state_bits, got {pair_bits}"
            )));
        }
        let head_bits = state_bits - pair_bits;
        let heads = 1u32 << head_bits;
        let block_threads = if heads <= 256 { 256u32 } else { 512u32 };
        if heads > block_threads || block_threads / heads > 32 {
            return Err(Error::Invalid(format!(
                "Unsupported CUDA HYB (L={state_bits}, B={pair_bits}) geometry (heads={heads})"
            )));
        }

        let src = format!(
            "#define HYB_L {state_bits}\n#define HYB_B {pair_bits}\n#define HYB_Q {}\n{}",
            codebook.q_bits,
            include_str!("hyb_viterbi_cuda.cu")
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
        let kernel = module
            .load_function("hyb_viterbi_quantize")
            .map_err(cuda_err)?;
        let stream = ctx.default_stream();
        let d_lut = stream.clone_htod(&codebook.lut_f16x2).map_err(cuda_err)?;

        Ok(Self {
            state_bits,
            pair_bits,
            codebook,
            block_threads,
            words_per_tile: hyb_tile_words(pair_bits),
            _ctx: ctx,
            stream,
            kernel,
            d_lut,
            scratch: Mutex::new(None),
        })
    }

    /// Quantizes `N` MMA-swizzled, sigma-normalized `16x16` tiles (`N * 256` floats)
    /// in a single GPU launch.
    ///
    /// Returns `(packed_words, decoded_norm_mma)` where:
    /// - `packed_words` has `N * hyb_tile_words(pair_bits)` `u32`s with the exact same
    ///   circular LSB-first packing as `HybQuantizer::quantize_normalized`.
    /// - `decoded_norm_mma` has `N * 256` `f32`s in UNSWIZZLED `16x16` row-major order
    ///   per tile (matching `decode_hyb_tile_mma_l` with `scale = 1.0`).
    pub fn quantize_tiles_normalized(
        &self,
        tiles_swizzled_norm: &[f32],
    ) -> Result<(Vec<u32>, Vec<f32>)> {
        if !tiles_swizzled_norm.len().is_multiple_of(WEIGHTS_PER_TILE) {
            return Err(Error::Invalid(format!(
                "tiles_swizzled_norm length {} must be a multiple of {}",
                tiles_swizzled_norm.len(),
                WEIGHTS_PER_TILE
            )));
        }
        let n_tiles = tiles_swizzled_norm.len() / WEIGHTS_PER_TILE;
        if n_tiles == 0 {
            return Ok((Vec::new(), Vec::new()));
        }

        let mut guard = self
            .scratch
            .lock()
            .map_err(|_| Error::Backend("CUDA HYB scratch mutex poisoned".into()))?;
        let need_alloc = match guard.as_ref() {
            Some(s) => s.cap_tiles < n_tiles,
            None => true,
        };
        if need_alloc {
            let cap = n_tiles;
            let d_in = self
                .stream
                .alloc_zeros::<f32>(cap * WEIGHTS_PER_TILE)
                .map_err(cuda_err)?;
            let d_words = self
                .stream
                .alloc_zeros::<u32>(cap * self.words_per_tile)
                .map_err(cuda_err)?;
            let d_decoded = self
                .stream
                .alloc_zeros::<f32>(cap * WEIGHTS_PER_TILE)
                .map_err(cuda_err)?;
            *guard = Some(ScratchBuffers {
                cap_tiles: cap,
                d_in,
                d_words,
                d_decoded,
            });
        }
        let scratch = guard.as_mut().unwrap();

        let in_elems = n_tiles * WEIGHTS_PER_TILE;
        let out_word_elems = n_tiles * self.words_per_tile;
        let mut d_in_view = scratch.d_in.slice_mut(0..in_elems);
        self.stream
            .memcpy_htod(tiles_swizzled_norm, &mut d_in_view)
            .map_err(cuda_err)?;

        let n_tiles_u32 = n_tiles as u32;
        let mut b = self.stream.launch_builder(&self.kernel);
        b.arg(&scratch.d_in)
            .arg(&self.d_lut)
            .arg(&mut scratch.d_words)
            .arg(&mut scratch.d_decoded)
            .arg(&n_tiles_u32);
        unsafe {
            b.launch(LaunchConfig {
                grid_dim: (n_tiles_u32, 1, 1),
                block_dim: (self.block_threads, 1, 1),
                shared_mem_bytes: 0,
            })
        }
        .map_err(cuda_err)?;

        let mut words = vec![0u32; out_word_elems];
        let mut decoded = vec![0.0f32; in_elems];
        self.stream
            .memcpy_dtoh(&scratch.d_words.slice(0..out_word_elems), &mut words)
            .map_err(cuda_err)?;
        self.stream
            .memcpy_dtoh(&scratch.d_decoded.slice(0..in_elems), &mut decoded)
            .map_err(cuda_err)?;
        self.stream.synchronize().map_err(cuda_err)?;

        Ok((words, decoded))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::qtip::hyb::{HybQuantizer, decode_hyb_tile_mma_l, swizzle_tile_to_mma};
    use std::time::Instant;

    fn generate_gaussian_tiles(n_tiles: usize, seed: u64) -> (Vec<f32>, Vec<f32>) {
        let mut state = seed;
        let mut next_u = || -> f64 {
            state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = state;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            ((z ^ (z >> 31)) >> 11) as f64 / ((1u64 << 53) as f64)
        };

        let mut rm_all = Vec::with_capacity(n_tiles * WEIGHTS_PER_TILE);
        let mut mma_all = Vec::with_capacity(n_tiles * WEIGHTS_PER_TILE);
        for _ in 0..n_tiles {
            let mut tile_rm = [0.0f32; WEIGHTS_PER_TILE];
            for chunk in tile_rm.chunks_exact_mut(2) {
                let u1 = next_u().max(1e-12);
                let u2 = next_u();
                let r = (-2.0 * u1.ln()).sqrt();
                let theta = std::f64::consts::TAU * u2;
                chunk[0] = (r * theta.cos()) as f32;
                chunk[1] = (r * theta.sin()) as f32;
            }
            let tile_mma = swizzle_tile_to_mma(&tile_rm);
            rm_all.extend_from_slice(&tile_rm);
            mma_all.extend_from_slice(&tile_mma);
        }
        (rm_all, mma_all)
    }

    fn verify_and_bench_config(
        ctx: &Arc<CudaContext>,
        state_bits: u32,
        pair_bits: u32,
        n_tiles: usize,
    ) {
        let cb = HybCodebook::gaussian_for_q(5);
        let gpu_q = CudaHybQuantizer::new(ctx.clone(), state_bits, pair_bits, cb.clone())
            .expect("compile CUDA HYB Viterbi");
        let cpu_q = HybQuantizer::new(state_bits, pair_bits, cb.clone()).expect("CPU HYB");
        let wpt = hyb_tile_words(pair_bits);

        let (rm_tiles, mma_tiles) =
            generate_gaussian_tiles(n_tiles, 0xCAFE_BABE_0000_0001 ^ u64::from(state_bits));

        // Warm-up GPU
        let (gpu_words, gpu_decoded) = gpu_q
            .quantize_tiles_normalized(&mma_tiles)
            .expect("GPU quantize");

        assert_eq!(gpu_words.len(), n_tiles * wpt);
        assert_eq!(gpu_decoded.len(), n_tiles * WEIGHTS_PER_TILE);

        // Verify GPU decoded_norm_mma matches CPU decode_hyb_tile_mma_l(&gpu_words, L, B, &cb, 1.0)
        let mut max_decode_diff = 0.0f32;
        let mut gpu_se = 0.0f64;
        let mut total_e = 0.0f64;
        for t in 0..n_tiles {
            let w_slice = &gpu_words[t * wpt..(t + 1) * wpt];
            let ref_dec = decode_hyb_tile_mma_l(w_slice, state_bits, pair_bits, &cb, 1.0);
            let gpu_dec = &gpu_decoded[t * WEIGHTS_PER_TILE..(t + 1) * WEIGHTS_PER_TILE];
            let orig_rm = &rm_tiles[t * WEIGHTS_PER_TILE..(t + 1) * WEIGHTS_PER_TILE];
            for i in 0..WEIGHTS_PER_TILE {
                let diff = (ref_dec[i] - gpu_dec[i]).abs();
                if diff > max_decode_diff {
                    max_decode_diff = diff;
                }
                let e = f64::from(orig_rm[i] - gpu_dec[i]);
                gpu_se += e * e;
                total_e += f64::from(orig_rm[i]) * f64::from(orig_rm[i]);
            }
        }
        assert!(
            max_decode_diff < 1e-5,
            "L={state_bits}, B={pair_bits}: GPU decoded vs CPU decode_hyb_tile_mma_l max diff = {max_decode_diff}"
        );

        // Compare against CPU HybQuantizer::quantize_normalized
        let t_cpu = Instant::now();
        let mut cpu_se = 0.0f64;
        let mut exact_word_matches = 0usize;
        for t in 0..n_tiles {
            let mma_slice = &mma_tiles[t * WEIGHTS_PER_TILE..(t + 1) * WEIGHTS_PER_TILE];
            let orig_rm = &rm_tiles[t * WEIGHTS_PER_TILE..(t + 1) * WEIGHTS_PER_TILE];
            let cpu_words = cpu_q.quantize_normalized(mma_slice).expect("CPU quantize");
            if cpu_words == gpu_words[t * wpt..(t + 1) * wpt] {
                exact_word_matches += 1;
            }
            let cpu_dec = decode_hyb_tile_mma_l(&cpu_words, state_bits, pair_bits, &cb, 1.0);
            for i in 0..WEIGHTS_PER_TILE {
                let e = f64::from(orig_rm[i] - cpu_dec[i]);
                cpu_se += e * e;
            }
        }
        let cpu_elapsed = t_cpu.elapsed().as_secs_f64();

        let gpu_mse = gpu_se / total_e;
        let cpu_mse = cpu_se / total_e;
        let rel_mse_diff = (gpu_mse - cpu_mse).abs() / cpu_mse;

        assert!(
            rel_mse_diff < 0.001,
            "L={state_bits}, B={pair_bits}: GPU MSE {gpu_mse:.6} vs CPU MSE {cpu_mse:.6} (rel diff {:.4}%)",
            rel_mse_diff * 100.0
        );

        // Measure GPU throughput over multiple batches (steady-state with reused scratch)
        let iters = 20usize;
        let t_gpu = Instant::now();
        for _ in 0..iters {
            let _ = gpu_q
                .quantize_tiles_normalized(&mma_tiles)
                .expect("GPU bench");
        }
        let gpu_elapsed = t_gpu.elapsed().as_secs_f64() / iters as f64;

        let gpu_tps = n_tiles as f64 / gpu_elapsed;
        let cpu_tps = n_tiles as f64 / cpu_elapsed;
        let speedup = gpu_tps / cpu_tps;

        // 8B model ~ 7.0B linear weights (~27.3M 16x16 tiles); 30B MoE ~ 29.5B linear weights (~115.2M tiles)
        let est_8b_sec = (7.0e9 / 256.0) / gpu_tps;
        let est_30b_sec = (29.5e9 / 256.0) / gpu_tps;

        println!(
            "[HYB L={state_bits} B={pair_bits} Q=5 ({:.1} bpw)] N={n_tiles} tiles:\n  \
             max_decode_diff={max_decode_diff:.2e}, exact_tile_match={exact_word_matches}/{n_tiles}\n  \
             GPU MSE={gpu_mse:.6}, CPU MSE={cpu_mse:.6}, rel_diff={:.5}%\n  \
             GPU={gpu_tps:.0} tiles/s ({:.3} ms/batch) vs CPU(1T)={cpu_tps:.0} tiles/s ({:.1} ms/batch) -> {speedup:.1}x\n  \
             Est Viterbi-only time: 8B={est_8b_sec:.1}s, 30B={est_30b_sec:.1}s",
            f64::from(pair_bits) / 2.0,
            rel_mse_diff * 100.0,
            gpu_elapsed * 1e3,
            cpu_elapsed * 1e3,
        );
    }

    #[test]
    fn cuda_hyb_viterbi_parity_and_throughput() {
        let ctx = CudaContext::new(0).expect("CUDA device 0");
        verify_and_bench_config(&ctx, 10, 6, 256);
        verify_and_bench_config(&ctx, 12, 4, 256);
    }
}
