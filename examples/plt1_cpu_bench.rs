//! Standalone CPU microbenchmark comparing scalar vs SIMD for PLT1 HYB Q=5:
//! - `fwht256_inplace`
//! - `hyb_q5_chunk_matvec_256` on `4096x4096` and `14336x4096` for both
//!   `(B=4, L=12)` (2.0 bpw) and `(B=6, L=10)` (3.0 bpw).
//!
//! Run with:
//!     cargo run --release --example plt1-cpu-bench

use half::f16;
use palito::cpu::plt1_simd::{
    HYB_CHUNK, HYB_TILE, f16x2_lut_to_f32, fwht256_inplace, fwht256_inplace_scalar,
    hyb_q5_chunk_matvec_256, hyb_q5_chunk_matvec_256_scalar, hyb_tile_words,
};
use rayon::prelude::*;
use std::hint::black_box;
use std::time::Instant;

fn lcg_next(state: &mut u64) -> u32 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    (*state >> 32) as u32
}

fn rand_f32(state: &mut u64) -> f32 {
    let u = (lcg_next(state) as f64) / (u32::MAX as f64);
    (u * 2.0 - 1.0) as f32
}

fn make_lut(seed: u64) -> [f32; 64] {
    let mut s = seed;
    let mut packed = [0u32; 32];
    for slot in &mut packed {
        let a = (rand_f32(&mut s).abs() * 2.0 + 0.05).min(3.5);
        let b = (rand_f32(&mut s).abs() * 2.0 + 0.05).min(3.5);
        let h0 = f16::from_f32(a).to_bits() & 0x7FFF;
        let h1 = f16::from_f32(b).to_bits() & 0x7FFF;
        *slot = u32::from(h0) | (u32::from(h1) << 16);
    }
    f16x2_lut_to_f32(&packed)
}

fn bench_fwht256() {
    let mut s = 0x1234_5678_9ABC_DEF0u64;
    let mut orig = [0.0f32; 256];
    for v in &mut orig {
        *v = rand_f32(&mut s);
    }

    let mut ref_out = orig;
    let mut simd_out = orig;
    fwht256_inplace_scalar(&mut ref_out);
    fwht256_inplace(&mut simd_out);

    let norm: f32 = ref_out.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-6);
    let max_err = ref_out
        .iter()
        .zip(&simd_out)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let rel_err = max_err / norm;

    let iters = 200_000usize;

    let mut buf = orig;
    let t0 = Instant::now();
    for _ in 0..iters {
        fwht256_inplace_scalar(black_box(&mut buf));
    }
    let scalar_ms = t0.elapsed().as_secs_f64() * 1e3;

    let mut buf = orig;
    let t1 = Instant::now();
    for _ in 0..iters {
        fwht256_inplace(black_box(&mut buf));
    }
    let simd_ms = t1.elapsed().as_secs_f64() * 1e3;

    let scalar_ns = (scalar_ms * 1e6) / iters as f64;
    let simd_ns = (simd_ms * 1e6) / iters as f64;
    let speedup = scalar_ms / simd_ms;
    let bytes_per_call = (256 * std::mem::size_of::<f32>() * 2) as f64; // read + write
    let simd_gbs = (bytes_per_call * iters as f64) / (simd_ms * 1e6);

    println!("=== FWHT-256 ({iters} iters) ===");
    println!(
        "  scalar: {scalar_ns:6.1} ns/call | SIMD: {simd_ns:6.1} ns/call | speedup: {speedup:5.2}x | eff: {simd_gbs:6.1} GB/s | max_rel_err: {rel_err:.2e}"
    );
}

fn bench_matvec(rows: usize, cols: usize, state_bits: u32, pair_bits: u32, lut_f32: &[f32; 64]) {
    assert_eq!(rows % HYB_CHUNK, 0);
    assert_eq!(cols % HYB_CHUNK, 0);

    let mut s = 0xDEAD_BEEF_0000_0001u64 ^ ((rows as u64) << 16) ^ (pair_bits as u64);
    let wpt = hyb_tile_words(pair_bits);
    let col_tiles = cols / HYB_TILE;
    let row_blocks = rows / HYB_CHUNK;
    let words_per_block = 16 * col_tiles * wpt;
    let total_words = row_blocks * words_per_block;

    let tiles: Vec<u32> = (0..total_words).map(|_| lcg_next(&mut s)).collect();
    let x: Vec<f32> = (0..cols).map(|_| rand_f32(&mut s)).collect();
    let row_signs: Vec<f32> = (0..rows)
        .map(|_| if (lcg_next(&mut s) & 1) == 0 { 1.0 } else { -1.0 })
        .collect();
    let scale = 1.0 / 16.0;

    let mut y_scalar = vec![0.0f32; rows];
    let mut y_simd = vec![0.0f32; rows];

    // Correctness check (single-pass full matrix)
    for b in 0..row_blocks {
        let bt = &tiles[b * words_per_block..(b + 1) * words_per_block];
        let rs: &[f32; 256] = (&row_signs[b * 256..(b + 1) * 256]).try_into().unwrap();
        let ys: &mut [f32; 256] = (&mut y_scalar[b * 256..(b + 1) * 256]).try_into().unwrap();
        let yv: &mut [f32; 256] = (&mut y_simd[b * 256..(b + 1) * 256]).try_into().unwrap();
        hyb_q5_chunk_matvec_256_scalar(bt, lut_f32, &x, None, Some(rs), ys, cols, state_bits, pair_bits, scale);
        hyb_q5_chunk_matvec_256(bt, lut_f32, &x, None, Some(rs), yv, cols, state_bits, pair_bits, scale);
    }

    let norm: f32 = y_scalar.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-6);
    let max_diff = y_scalar
        .iter()
        .zip(&y_simd)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let max_rel_err = max_diff / norm;

    let weight_bytes = (total_words * std::mem::size_of::<u32>()) as f64;
    let iters_1t = if rows <= 4096 { 10 } else { 4 };
    let iters_mt = if rows <= 4096 { 40 } else { 15 };

    // 1-thread scalar
    let t0 = Instant::now();
    for _ in 0..iters_1t {
        for b in 0..row_blocks {
            let bt = &tiles[b * words_per_block..(b + 1) * words_per_block];
            let rs: &[f32; 256] = (&row_signs[b * 256..(b + 1) * 256]).try_into().unwrap();
            let ys: &mut [f32; 256] = (&mut y_scalar[b * 256..(b + 1) * 256]).try_into().unwrap();
            hyb_q5_chunk_matvec_256_scalar(
                black_box(bt),
                lut_f32,
                black_box(&x),
                None,
                Some(rs),
                ys,
                cols,
                state_bits,
                pair_bits,
                scale,
            );
        }
    }
    let scalar_1t_ms = (t0.elapsed().as_secs_f64() * 1e3) / iters_1t as f64;

    // 1-thread SIMD
    let t1 = Instant::now();
    for _ in 0..iters_1t {
        for b in 0..row_blocks {
            let bt = &tiles[b * words_per_block..(b + 1) * words_per_block];
            let rs: &[f32; 256] = (&row_signs[b * 256..(b + 1) * 256]).try_into().unwrap();
            let yv: &mut [f32; 256] = (&mut y_simd[b * 256..(b + 1) * 256]).try_into().unwrap();
            hyb_q5_chunk_matvec_256(
                black_box(bt),
                lut_f32,
                black_box(&x),
                None,
                Some(rs),
                yv,
                cols,
                state_bits,
                pair_bits,
                scale,
            );
        }
    }
    let simd_1t_ms = (t1.elapsed().as_secs_f64() * 1e3) / iters_1t as f64;

    // Multi-threaded (rayon) scalar
    let t2 = Instant::now();
    for _ in 0..iters_mt {
        y_scalar
            .par_chunks_exact_mut(256)
            .enumerate()
            .for_each(|(b, ys_slice)| {
                let bt = &tiles[b * words_per_block..(b + 1) * words_per_block];
                let rs: &[f32; 256] = (&row_signs[b * 256..(b + 1) * 256]).try_into().unwrap();
                let ys: &mut [f32; 256] = ys_slice.try_into().unwrap();
                hyb_q5_chunk_matvec_256_scalar(
                    black_box(bt),
                    lut_f32,
                    black_box(&x),
                    None,
                    Some(rs),
                    ys,
                    cols,
                    state_bits,
                    pair_bits,
                    scale,
                );
            });
    }
    let scalar_mt_ms = (t2.elapsed().as_secs_f64() * 1e3) / iters_mt as f64;

    // Multi-threaded (rayon) SIMD
    let t3 = Instant::now();
    for _ in 0..iters_mt {
        y_simd
            .par_chunks_exact_mut(256)
            .enumerate()
            .for_each(|(b, yv_slice)| {
                let bt = &tiles[b * words_per_block..(b + 1) * words_per_block];
                let rs: &[f32; 256] = (&row_signs[b * 256..(b + 1) * 256]).try_into().unwrap();
                let yv: &mut [f32; 256] = yv_slice.try_into().unwrap();
                hyb_q5_chunk_matvec_256(
                    black_box(bt),
                    lut_f32,
                    black_box(&x),
                    None,
                    Some(rs),
                    yv,
                    cols,
                    state_bits,
                    pair_bits,
                    scale,
                );
            });
    }
    let simd_mt_ms = (t3.elapsed().as_secs_f64() * 1e3) / iters_mt as f64;

    let simd_1t_gbs = weight_bytes / (simd_1t_ms * 1e6);
    let simd_mt_gbs = weight_bytes / (simd_mt_ms * 1e6);
    let speedup_1t = scalar_1t_ms / simd_1t_ms;
    let speedup_mt = scalar_mt_ms / simd_mt_ms;
    let bpw = pair_bits as f64 / 2.0;

    println!(
        "  {rows:5}x{cols:<4} (B={pair_bits}, L={state_bits:2}, {bpw:.1} bpw, {:.2} MB):",
        weight_bytes / (1024.0 * 1024.0)
    );
    println!(
        "    1-thread : scalar {scalar_1t_ms:7.3} ms | SIMD {simd_1t_ms:7.3} ms | {speedup_1t:5.2}x | eff {simd_1t_gbs:6.2} GB/s"
    );
    println!(
        "    rayon MT : scalar {scalar_mt_ms:7.3} ms | SIMD {simd_mt_ms:7.3} ms | {speedup_mt:5.2}x | eff {simd_mt_gbs:6.2} GB/s | max_rel_err {max_rel_err:.2e}"
    );
}

fn main() {
    println!("PLT1 HYB Q=5 CPU SIMD Microbenchmark (rayon threads: {})", rayon::current_num_threads());
    bench_fwht256();
    println!("\n=== hyb_q5_chunk_matvec_256 (SKIP_IN_RHT, with OUT_RHT) ===");
    let lut_f32 = make_lut(0xA1B2_C3D4);
    for &(rows, cols) in &[(4096usize, 4096usize), (14336usize, 4096usize)] {
        bench_matvec(rows, cols, 12, 4, &lut_f32);
        bench_matvec(rows, cols, 10, 6, &lut_f32);
    }
}
