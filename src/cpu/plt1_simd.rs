//! Standalone CPU SIMD microkernels for `PLT1 HYB Q=5` inference.
//!
//! Provides runtime-dispatched x86_64 AVX2+FMA, aarch64 NEON, and scalar fallback
//! implementations for:
//! - Exact IEEE-754 `f16x2` -> `f32` LUT conversion for the 32-entry `Q=5` codebook
//! - Unnormalized length-256 Walsh-Hadamard transform (`fwht256_inplace`)
//! - Residual-stream R1 rotation enter/exit (`hyb_r1_enter_chunk`, `hyb_r1_exit_chunk`)
//! - MMA-order `16x16` HYB Q=5 tile dot-product (`hyb_q5_tile_dot_mma`)
//! - Full 256-row block matvec (`hyb_q5_chunk_matvec_256`) with optional input/output RHT

use half::f16;

/// Knuth 32-bit multiplicative hash constant (bit-identical to `src/qtip/hyb.rs` and CUDA).
pub const HYB_HASH_MULT: u32 = 0x9E37_79B1;
pub const HYB_TILE: usize = 16;
pub const HYB_CHUNK: usize = 256;
pub const HYB_PAIRS_PER_TILE: usize = (HYB_TILE * HYB_TILE) / 2;

/// Number of `u32` words per circular `16x16` HYB tile for `pair_bits` in `3..=6`.
#[inline]
pub const fn hyb_tile_words(pair_bits: u32) -> usize {
    (HYB_PAIRS_PER_TILE * pair_bits as usize) / 32
}

/// Exact IEEE-754 `f16x2` -> `f32` conversion for the 32-entry `HYB Q=5` codebook.
///
/// Output layout: `[w0_0, w1_0, w0_1, w1_1, ..., w0_31, w1_31]`.
pub fn f16x2_lut_to_f32(lut_f16x2: &[u32; 32]) -> [f32; 64] {
    let mut out = [0.0f32; 64];
    for (i, &packed) in lut_f16x2.iter().enumerate() {
        out[2 * i] = f16::from_bits((packed & 0xFFFF) as u16).to_f32();
        out[2 * i + 1] = f16::from_bits((packed >> 16) as u16).to_f32();
    }
    out
}

// ---------------------------------------------------------------------------
// Length-256 Walsh-Hadamard Transform (unnormalized)
// ---------------------------------------------------------------------------

/// Scalar reference implementation of the unnormalized length-256 FWHT.
pub fn fwht256_inplace_scalar(v: &mut [f32; 256]) {
    let mut span = 1usize;
    while span < 256 {
        let step = span * 2;
        let mut base = 0usize;
        while base < 256 {
            for j in 0..span {
                let a = v[base + j];
                let b = v[base + j + span];
                v[base + j] = a + b;
                v[base + j + span] = a - b;
            }
            base += step;
        }
        span = step;
    }
}

/// Unnormalized length-256 Walsh-Hadamard transform with runtime SIMD dispatch.
#[inline]
pub fn fwht256_inplace(v: &mut [f32; 256]) {
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2") {
            // SAFETY: AVX2 availability was verified at runtime above.
            unsafe { fwht256_inplace_avx2(v) };
            return;
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: NEON is always available on aarch64.
        unsafe { fwht256_inplace_neon(v) };
        return;
    }
    #[allow(unreachable_code)]
    fwht256_inplace_scalar(v);
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn fwht256_inplace_avx2(v: &mut [f32; 256]) {
    use std::arch::x86_64::*;

    unsafe {
        let sign1 = _mm256_setr_ps(0.0, -0.0, 0.0, -0.0, 0.0, -0.0, 0.0, -0.0);
        let sign2 = _mm256_setr_ps(0.0, 0.0, -0.0, -0.0, 0.0, 0.0, -0.0, -0.0);
        let sign4 = _mm256_setr_ps(0.0, 0.0, 0.0, 0.0, -0.0, -0.0, -0.0, -0.0);

        let ptr = v.as_mut_ptr();

        // Pass 1: 4 blocks of 64 floats (8 YMM vectors each).
        // Performs stages 1, 2, 4 (in-lane), 8, 16, 32 (cross-vector) in registers.
        for blk in 0..4usize {
            let base = ptr.add(blk * 64);
            let mut r = [_mm256_setzero_ps(); 8];
            for i in 0..8usize {
                let mut x = _mm256_loadu_ps(base.add(i * 8));
                // Stage 1 (span = 1)
                let even = _mm256_moveldup_ps(x);
                let odd = _mm256_movehdup_ps(x);
                x = _mm256_add_ps(even, _mm256_xor_ps(odd, sign1));
                // Stage 2 (span = 2)
                let a2 = _mm256_permute_ps(x, 0b01_00_01_00);
                let b2 = _mm256_permute_ps(x, 0b11_10_11_10);
                x = _mm256_add_ps(a2, _mm256_xor_ps(b2, sign2));
                // Stage 4 (span = 4)
                let lo4 = _mm256_permute2f128_ps(x, x, 0x00);
                let hi4 = _mm256_permute2f128_ps(x, x, 0x11);
                r[i] = _mm256_add_ps(lo4, _mm256_xor_ps(hi4, sign4));
            }
            // Stage 8 (span = 8: pairs (0,1), (2,3), (4,5), (6,7))
            for i in (0..8).step_by(2) {
                let a = r[i];
                let b = r[i + 1];
                r[i] = _mm256_add_ps(a, b);
                r[i + 1] = _mm256_sub_ps(a, b);
            }
            // Stage 16 (span = 16: pairs (0,2), (1,3), (4,6), (5,7))
            for &i in &[0usize, 1, 4, 5] {
                let a = r[i];
                let b = r[i + 2];
                r[i] = _mm256_add_ps(a, b);
                r[i + 2] = _mm256_sub_ps(a, b);
            }
            // Stage 32 (span = 32: pairs (0,4), (1,5), (2,6), (3,7))
            for i in 0..4usize {
                let a = r[i];
                let b = r[i + 4];
                r[i] = _mm256_add_ps(a, b);
                r[i + 4] = _mm256_sub_ps(a, b);
            }
            for i in 0..8usize {
                _mm256_storeu_ps(base.add(i * 8), r[i]);
            }
        }

        // Pass 2: Stages 64 and 128 across the 4 blocks of 64 floats.
        for j in 0..8usize {
            let off = j * 8;
            let b0 = _mm256_loadu_ps(ptr.add(off));
            let b1 = _mm256_loadu_ps(ptr.add(64 + off));
            let b2 = _mm256_loadu_ps(ptr.add(128 + off));
            let b3 = _mm256_loadu_ps(ptr.add(192 + off));

            let t0 = _mm256_add_ps(b0, b1);
            let t1 = _mm256_sub_ps(b0, b1);
            let t2 = _mm256_add_ps(b2, b3);
            let t3 = _mm256_sub_ps(b2, b3);

            _mm256_storeu_ps(ptr.add(off), _mm256_add_ps(t0, t2));
            _mm256_storeu_ps(ptr.add(64 + off), _mm256_add_ps(t1, t3));
            _mm256_storeu_ps(ptr.add(128 + off), _mm256_sub_ps(t0, t2));
            _mm256_storeu_ps(ptr.add(192 + off), _mm256_sub_ps(t1, t3));
        }
    }
}

#[cfg(target_arch = "aarch64")]
unsafe fn fwht256_inplace_neon(v: &mut [f32; 256]) {
    use std::arch::aarch64::*;

    unsafe {
        let ptr = v.as_mut_ptr();

        // Pass 1: 4 blocks of 64 floats (16 NEON registers each).
        for blk in 0..4usize {
            let base = ptr.add(blk * 64);
            let mut r = [vdupq_n_f32(0.0); 16];
            for i in 0..16usize {
                let x = vld1q_f32(base.add(i * 4));
                // Stage 1 (span = 1)
                let even = vuzp1q_f32(x, x);
                let odd = vuzp2q_f32(x, x);
                let sum1 = vaddq_f32(even, odd);
                let diff1 = vsubq_f32(even, odd);
                let y = vzip1q_f32(sum1, diff1);
                // Stage 2 (span = 2)
                let lo = vcombine_f32(vget_low_f32(y), vget_low_f32(y));
                let hi = vcombine_f32(vget_high_f32(y), vget_high_f32(y));
                let sum2 = vaddq_f32(lo, hi);
                let diff2 = vsubq_f32(lo, hi);
                r[i] = vcombine_f32(vget_low_f32(sum2), vget_low_f32(diff2));
            }
            // Stage 4 (span = 4)
            for i in (0..16).step_by(2) {
                let a = r[i];
                let b = r[i + 1];
                r[i] = vaddq_f32(a, b);
                r[i + 1] = vsubq_f32(a, b);
            }
            // Stage 8 (span = 8)
            for base_i in (0..16).step_by(4) {
                for j in 0..2 {
                    let a = r[base_i + j];
                    let b = r[base_i + j + 2];
                    r[base_i + j] = vaddq_f32(a, b);
                    r[base_i + j + 2] = vsubq_f32(a, b);
                }
            }
            // Stage 16 (span = 16)
            for base_i in (0..16).step_by(8) {
                for j in 0..4 {
                    let a = r[base_i + j];
                    let b = r[base_i + j + 4];
                    r[base_i + j] = vaddq_f32(a, b);
                    r[base_i + j + 4] = vsubq_f32(a, b);
                }
            }
            // Stage 32 (span = 32)
            for j in 0..8 {
                let a = r[j];
                let b = r[j + 8];
                r[j] = vaddq_f32(a, b);
                r[j + 8] = vsubq_f32(a, b);
            }
            for i in 0..16usize {
                vst1q_f32(base.add(i * 4), r[i]);
            }
        }

        // Pass 2: Stages 64 and 128 across the 4 blocks of 64 floats.
        for j in 0..16usize {
            let off = j * 4;
            let b0 = vld1q_f32(ptr.add(off));
            let b1 = vld1q_f32(ptr.add(64 + off));
            let b2 = vld1q_f32(ptr.add(128 + off));
            let b3 = vld1q_f32(ptr.add(192 + off));

            let t0 = vaddq_f32(b0, b1);
            let t1 = vsubq_f32(b0, b1);
            let t2 = vaddq_f32(b2, b3);
            let t3 = vsubq_f32(b2, b3);

            vst1q_f32(ptr.add(off), vaddq_f32(t0, t2));
            vst1q_f32(ptr.add(64 + off), vaddq_f32(t1, t3));
            vst1q_f32(ptr.add(128 + off), vsubq_f32(t0, t2));
            vst1q_f32(ptr.add(192 + off), vsubq_f32(t1, t3));
        }
    }
}

// ---------------------------------------------------------------------------
// Residual-stream R1 enter / exit (matching hyb_r1_enter / hyb_r1_exit)
// ---------------------------------------------------------------------------

/// Residual-stream R1 rotation enter: `x <- (H_256 * S_res / 16) * x`.
#[inline]
pub fn hyb_r1_enter_chunk(x: &mut [f32; 256], r1_signs: &[f32; 256]) {
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2") {
            // SAFETY: AVX2 availability verified at runtime.
            unsafe { hyb_r1_enter_chunk_avx2(x, r1_signs) };
            return;
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: NEON is always available on aarch64.
        unsafe { hyb_r1_enter_chunk_neon(x, r1_signs) };
        return;
    }
    #[allow(unreachable_code)]
    {
        for i in 0..256 {
            x[i] *= r1_signs[i];
        }
        fwht256_inplace_scalar(x);
        for v in x.iter_mut() {
            *v *= 1.0 / 16.0;
        }
    }
}

/// Residual-stream R1 rotation exit: `x <- S_res * (H_256 / 16) * x`.
#[inline]
pub fn hyb_r1_exit_chunk(x: &mut [f32; 256], r1_signs: &[f32; 256]) {
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2") {
            // SAFETY: AVX2 availability verified at runtime.
            unsafe { hyb_r1_exit_chunk_avx2(x, r1_signs) };
            return;
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: NEON is always available on aarch64.
        unsafe { hyb_r1_exit_chunk_neon(x, r1_signs) };
        return;
    }
    #[allow(unreachable_code)]
    {
        fwht256_inplace_scalar(x);
        for i in 0..256 {
            x[i] *= r1_signs[i] * (1.0 / 16.0);
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn hyb_r1_enter_chunk_avx2(x: &mut [f32; 256], r1_signs: &[f32; 256]) {
    use std::arch::x86_64::*;
    unsafe {
        let xp = x.as_mut_ptr();
        let sp = r1_signs.as_ptr();
        for i in (0..256).step_by(8) {
            let vx = _mm256_loadu_ps(xp.add(i));
            let vs = _mm256_loadu_ps(sp.add(i));
            _mm256_storeu_ps(xp.add(i), _mm256_mul_ps(vx, vs));
        }
        fwht256_inplace_avx2(x);
        let inv16 = _mm256_set1_ps(1.0 / 16.0);
        for i in (0..256).step_by(8) {
            let vx = _mm256_loadu_ps(xp.add(i));
            _mm256_storeu_ps(xp.add(i), _mm256_mul_ps(vx, inv16));
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn hyb_r1_exit_chunk_avx2(x: &mut [f32; 256], r1_signs: &[f32; 256]) {
    use std::arch::x86_64::*;
    unsafe {
        fwht256_inplace_avx2(x);
        let xp = x.as_mut_ptr();
        let sp = r1_signs.as_ptr();
        let inv16 = _mm256_set1_ps(1.0 / 16.0);
        for i in (0..256).step_by(8) {
            let vx = _mm256_loadu_ps(xp.add(i));
            let vs = _mm256_loadu_ps(sp.add(i));
            let scaled = _mm256_mul_ps(_mm256_mul_ps(vx, vs), inv16);
            _mm256_storeu_ps(xp.add(i), scaled);
        }
    }
}

#[cfg(target_arch = "aarch64")]
unsafe fn hyb_r1_enter_chunk_neon(x: &mut [f32; 256], r1_signs: &[f32; 256]) {
    use std::arch::aarch64::*;
    unsafe {
        let xp = x.as_mut_ptr();
        let sp = r1_signs.as_ptr();
        for i in (0..256).step_by(4) {
            vst1q_f32(xp.add(i), vmulq_f32(vld1q_f32(xp.add(i)), vld1q_f32(sp.add(i))));
        }
        fwht256_inplace_neon(x);
        let inv16 = vdupq_n_f32(1.0 / 16.0);
        for i in (0..256).step_by(4) {
            vst1q_f32(xp.add(i), vmulq_f32(vld1q_f32(xp.add(i)), inv16));
        }
    }
}

#[cfg(target_arch = "aarch64")]
unsafe fn hyb_r1_exit_chunk_neon(x: &mut [f32; 256], r1_signs: &[f32; 256]) {
    use std::arch::aarch64::*;
    unsafe {
        fwht256_inplace_neon(x);
        let xp = x.as_mut_ptr();
        let sp = r1_signs.as_ptr();
        let inv16 = vdupq_n_f32(1.0 / 16.0);
        for i in (0..256).step_by(4) {
            let scaled = vmulq_f32(vmulq_f32(vld1q_f32(xp.add(i)), vld1q_f32(sp.add(i))), inv16);
            vst1q_f32(xp.add(i), scaled);
        }
    }
}

// ---------------------------------------------------------------------------
// HYB Q=5 MMA-order 16x16 tile dot-product
// ---------------------------------------------------------------------------

/// Scalar reference implementation of `hyb_q5_tile_dot_mma`.
///
/// Decodes a `16x16` HYB Q=5 tile stored in `mma.sync.aligned.m16n8k16` A-fragment
/// order and accumulates `acc[r] += sum_c W[r, c] * x_tile[c]` for `r in 0..16`.
pub fn hyb_q5_tile_dot_mma_scalar(
    words: &[u32],
    lut_f32: &[f32; 64],
    x_tile: &[f32; 16],
    acc: &mut [f32; 16],
    state_bits: u32,
    pair_bits: u32,
) {
    debug_assert!((1..=16).contains(&state_bits));
    debug_assert_eq!(words.len(), hyb_tile_words(pair_bits));
    let total_words = words.len();
    let mask = (1u64 << state_bits) - 1;

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
            let w0 = (bit / 32) % total_words;
            let w1 = (w0 + 1) % total_words;
            let shift = bit & 31;
            let combined = u64::from(words[w0]) | (u64::from(words[w1]) << 32);
            let state = ((combined >> shift) & mask) as u32;
            let h = state.wrapping_mul(HYB_HASH_MULT);
            let idx = ((h >> 5) & 31) as usize;
            let s0 = (h << 16) & 0x8000_0000;
            let s1 = h & 0x8000_0000;
            let w_0 = f32::from_bits(lut_f32[2 * idx].to_bits() ^ s0);
            let w_1 = f32::from_bits(lut_f32[2 * idx + 1].to_bits() ^ s1);
            acc[r] = w_0.mul_add(x_tile[c], acc[r]);
            acc[r] = w_1.mul_add(x_tile[c + 1], acc[r]);
        }
    }
}

/// Decodes a `16x16` HYB Q=5 MMA-swizzled tile and accumulates `W * x_tile` into
/// `acc[0..16]` directly in MMA register order without materializing a `16x16` matrix.
///
/// Supports `(L=12, B=4)` (16 words), `(L=10, B=6)` (24 words), and generic `(L, B)`.
#[inline]
pub fn hyb_q5_tile_dot_mma(
    words: &[u32],
    lut_f32: &[f32; 64],
    x_tile: &[f32; 16],
    acc: &mut [f32; 16],
    state_bits: u32,
    pair_bits: u32,
) {
    debug_assert!((1..=16).contains(&state_bits));
    debug_assert_eq!(words.len(), hyb_tile_words(pair_bits));

    #[cfg(target_arch = "x86_64")]
    {
        if (pair_bits == 4 || pair_bits == 6)
            && std::is_x86_feature_detected!("avx2")
            && std::is_x86_feature_detected!("fma")
        {
            // SAFETY: AVX2+FMA verified at runtime; words.len() matches pair_bits.
            unsafe {
                if pair_bits == 4 {
                    hyb_q5_tile_dot_mma_b4_avx2(words, lut_f32, x_tile, acc, state_bits);
                } else {
                    hyb_q5_tile_dot_mma_b6_avx2(words, lut_f32, x_tile, acc, state_bits);
                }
            }
            return;
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if pair_bits == 4 || pair_bits == 6 {
            // SAFETY: NEON is always available on aarch64.
            unsafe {
                if pair_bits == 4 {
                    hyb_q5_tile_dot_mma_b4_neon(words, lut_f32, x_tile, acc, state_bits);
                } else {
                    hyb_q5_tile_dot_mma_b6_neon(words, lut_f32, x_tile, acc, state_bits);
                }
            }
            return;
        }
    }

    hyb_q5_tile_dot_mma_scalar(words, lut_f32, x_tile, acc, state_bits, pair_bits);
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn avx2_decode_and_fma_8(
    states: std::arch::x86_64::__m256i,
    lut_ptr: *const f32,
    x0: f32,
    x1: f32,
    acc: std::arch::x86_64::__m256,
    hash_mult: std::arch::x86_64::__m256i,
    idx_mask: std::arch::x86_64::__m256i,
    sign_bit: std::arch::x86_64::__m256i,
) -> std::arch::x86_64::__m256 {
    use std::arch::x86_64::*;
    unsafe {
        let h = _mm256_mullo_epi32(states, hash_mult);
        let idx = _mm256_and_si256(_mm256_srli_epi32(h, 5), idx_mask);
        let w0_mag = _mm256_i32gather_ps::<8>(lut_ptr, idx);
        let w1_mag = _mm256_i32gather_ps::<8>(lut_ptr.add(1), idx);
        let s0 = _mm256_and_si256(_mm256_slli_epi32(h, 16), sign_bit);
        let s1 = _mm256_and_si256(h, sign_bit);
        let w0 = _mm256_xor_ps(w0_mag, _mm256_castsi256_ps(s0));
        let w1 = _mm256_xor_ps(w1_mag, _mm256_castsi256_ps(s1));
        let acc = _mm256_fmadd_ps(w0, _mm256_set1_ps(x0), acc);
        _mm256_fmadd_ps(w1, _mm256_set1_ps(x1), acc)
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn avx2_tile_b4_step(
    words_ptr: *const u32,
    lut_ptr: *const f32,
    x_tile: &[f32; 16],
    mut acc_lo: std::arch::x86_64::__m256,
    mut acc_hi: std::arch::x86_64::__m256,
    state_mask: std::arch::x86_64::__m256i,
    hash_mult: std::arch::x86_64::__m256i,
    idx_mask: std::arch::x86_64::__m256i,
    sign_bit: std::arch::x86_64::__m256i,
    rot1: std::arch::x86_64::__m256i,
) -> (std::arch::x86_64::__m256, std::arch::x86_64::__m256) {
    use std::arch::x86_64::*;

    unsafe {
        // Load 16 u32 words and deinterleave into w_even[g] = words[2g], w_odd[g] = words[2g+1].
        let v0 = _mm256_loadu_ps(words_ptr as *const f32);
        let v1 = _mm256_loadu_ps(words_ptr.add(8) as *const f32);
        let w_even = _mm256_permute4x64_epi64(
            _mm256_castps_si256(_mm256_shuffle_ps(v0, v1, 0x88)),
            0b11_01_10_00,
        );
        let w_odd = _mm256_permute4x64_epi64(
            _mm256_castps_si256(_mm256_shuffle_ps(v0, v1, 0xDD)),
            0b11_01_10_00,
        );
        let w_next = _mm256_permutevar8x32_epi32(w_even, rot1);

        let s16 = _mm256_or_si256(_mm256_srli_epi32(w_even, 16), _mm256_slli_epi32(w_odd, 16));
        let s48 = _mm256_or_si256(_mm256_srli_epi32(w_odd, 16), _mm256_slli_epi32(w_next, 16));

        macro_rules! step_win {
            ($win:expr, $c0:expr, $c1:expr) => {{
                let st0 = _mm256_and_si256($win, state_mask);
                let st1 = _mm256_and_si256(_mm256_srli_epi32($win, 4), state_mask);
                let st2 = _mm256_and_si256(_mm256_srli_epi32($win, 8), state_mask);
                let st3 = _mm256_and_si256(_mm256_srli_epi32($win, 12), state_mask);
                acc_lo = avx2_decode_and_fma_8(st0, lut_ptr, x_tile[$c0], x_tile[$c0 + 1], acc_lo, hash_mult, idx_mask, sign_bit);
                acc_hi = avx2_decode_and_fma_8(st1, lut_ptr, x_tile[$c0], x_tile[$c0 + 1], acc_hi, hash_mult, idx_mask, sign_bit);
                acc_lo = avx2_decode_and_fma_8(st2, lut_ptr, x_tile[$c1], x_tile[$c1 + 1], acc_lo, hash_mult, idx_mask, sign_bit);
                acc_hi = avx2_decode_and_fma_8(st3, lut_ptr, x_tile[$c1], x_tile[$c1 + 1], acc_hi, hash_mult, idx_mask, sign_bit);
            }};
        }

        step_win!(w_even, 0, 8);
        step_win!(s16, 2, 10);
        step_win!(w_odd, 4, 12);
        step_win!(s48, 6, 14);

        (acc_lo, acc_hi)
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn avx2_tile_b6_step(
    words_ptr: *const u32,
    lut_ptr: *const f32,
    x_tile: &[f32; 16],
    mut acc_lo: std::arch::x86_64::__m256,
    mut acc_hi: std::arch::x86_64::__m256,
    state_mask: std::arch::x86_64::__m256i,
    hash_mult: std::arch::x86_64::__m256i,
    idx_mask: std::arch::x86_64::__m256i,
    sign_bit: std::arch::x86_64::__m256i,
    rot1: std::arch::x86_64::__m256i,
) -> (std::arch::x86_64::__m256, std::arch::x86_64::__m256) {
    use std::arch::x86_64::*;

    unsafe {
        // Each group_id occupies 96 bits = 3 u32 words: w0[g]=words[3g], w1[g]=words[3g+1], w2[g]=words[3g+2].
        let w = std::slice::from_raw_parts(words_ptr, 24);
        let w0 = _mm256_setr_epi32(
            w[0] as i32, w[3] as i32, w[6] as i32, w[9] as i32,
            w[12] as i32, w[15] as i32, w[18] as i32, w[21] as i32,
        );
        let w1 = _mm256_setr_epi32(
            w[1] as i32, w[4] as i32, w[7] as i32, w[10] as i32,
            w[13] as i32, w[16] as i32, w[19] as i32, w[22] as i32,
        );
        let w2 = _mm256_setr_epi32(
            w[2] as i32, w[5] as i32, w[8] as i32, w[11] as i32,
            w[14] as i32, w[17] as i32, w[20] as i32, w[23] as i32,
        );
        let w3 = _mm256_permutevar8x32_epi32(w0, rot1);

        // tid = 0: bit offsets 0, 6, 12, 18 -> cols (0,1) and (8,9)
        {
            let st0 = _mm256_and_si256(w0, state_mask);
            let st1 = _mm256_and_si256(_mm256_srli_epi32(w0, 6), state_mask);
            let st2 = _mm256_and_si256(_mm256_srli_epi32(w0, 12), state_mask);
            let st3 = _mm256_and_si256(
                _mm256_or_si256(_mm256_srli_epi32(w0, 18), _mm256_slli_epi32(w1, 14)),
                state_mask,
            );
            acc_lo = avx2_decode_and_fma_8(st0, lut_ptr, x_tile[0], x_tile[1], acc_lo, hash_mult, idx_mask, sign_bit);
            acc_hi = avx2_decode_and_fma_8(st1, lut_ptr, x_tile[0], x_tile[1], acc_hi, hash_mult, idx_mask, sign_bit);
            acc_lo = avx2_decode_and_fma_8(st2, lut_ptr, x_tile[8], x_tile[9], acc_lo, hash_mult, idx_mask, sign_bit);
            acc_hi = avx2_decode_and_fma_8(st3, lut_ptr, x_tile[8], x_tile[9], acc_hi, hash_mult, idx_mask, sign_bit);
        }

        // tid = 1: bit offsets 24, 30, 36, 42 -> cols (2,3) and (10,11)
        {
            let b24 = _mm256_or_si256(_mm256_srli_epi32(w0, 24), _mm256_slli_epi32(w1, 8));
            let st0 = _mm256_and_si256(b24, state_mask);
            let st1 = _mm256_and_si256(_mm256_srli_epi32(b24, 6), state_mask);
            let st2 = _mm256_and_si256(_mm256_srli_epi32(w1, 4), state_mask);
            let st3 = _mm256_and_si256(_mm256_srli_epi32(w1, 10), state_mask);
            acc_lo = avx2_decode_and_fma_8(st0, lut_ptr, x_tile[2], x_tile[3], acc_lo, hash_mult, idx_mask, sign_bit);
            acc_hi = avx2_decode_and_fma_8(st1, lut_ptr, x_tile[2], x_tile[3], acc_hi, hash_mult, idx_mask, sign_bit);
            acc_lo = avx2_decode_and_fma_8(st2, lut_ptr, x_tile[10], x_tile[11], acc_lo, hash_mult, idx_mask, sign_bit);
            acc_hi = avx2_decode_and_fma_8(st3, lut_ptr, x_tile[10], x_tile[11], acc_hi, hash_mult, idx_mask, sign_bit);
        }

        // tid = 2: bit offsets 48, 54, 60, 66 -> cols (4,5) and (12,13)
        {
            let b48 = _mm256_or_si256(_mm256_srli_epi32(w1, 16), _mm256_slli_epi32(w2, 16));
            let st0 = _mm256_and_si256(b48, state_mask);
            let st1 = _mm256_and_si256(_mm256_srli_epi32(b48, 6), state_mask);
            let st2 = _mm256_and_si256(_mm256_srli_epi32(b48, 12), state_mask);
            let st3 = _mm256_and_si256(_mm256_srli_epi32(w2, 2), state_mask);
            acc_lo = avx2_decode_and_fma_8(st0, lut_ptr, x_tile[4], x_tile[5], acc_lo, hash_mult, idx_mask, sign_bit);
            acc_hi = avx2_decode_and_fma_8(st1, lut_ptr, x_tile[4], x_tile[5], acc_hi, hash_mult, idx_mask, sign_bit);
            acc_lo = avx2_decode_and_fma_8(st2, lut_ptr, x_tile[12], x_tile[13], acc_lo, hash_mult, idx_mask, sign_bit);
            acc_hi = avx2_decode_and_fma_8(st3, lut_ptr, x_tile[12], x_tile[13], acc_hi, hash_mult, idx_mask, sign_bit);
        }

        // tid = 3: bit offsets 72, 78, 84, 90 -> cols (6,7) and (14,15)
        {
            let b72 = _mm256_or_si256(_mm256_srli_epi32(w2, 8), _mm256_slli_epi32(w3, 24));
            let st0 = _mm256_and_si256(b72, state_mask);
            let st1 = _mm256_and_si256(_mm256_srli_epi32(b72, 6), state_mask);
            let st2 = _mm256_and_si256(_mm256_srli_epi32(b72, 12), state_mask);
            let st3 = _mm256_and_si256(
                _mm256_or_si256(_mm256_srli_epi32(w2, 26), _mm256_slli_epi32(w3, 6)),
                state_mask,
            );
            acc_lo = avx2_decode_and_fma_8(st0, lut_ptr, x_tile[6], x_tile[7], acc_lo, hash_mult, idx_mask, sign_bit);
            acc_hi = avx2_decode_and_fma_8(st1, lut_ptr, x_tile[6], x_tile[7], acc_hi, hash_mult, idx_mask, sign_bit);
            acc_lo = avx2_decode_and_fma_8(st2, lut_ptr, x_tile[14], x_tile[15], acc_lo, hash_mult, idx_mask, sign_bit);
            acc_hi = avx2_decode_and_fma_8(st3, lut_ptr, x_tile[14], x_tile[15], acc_hi, hash_mult, idx_mask, sign_bit);
        }

        (acc_lo, acc_hi)
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn hyb_q5_tile_dot_mma_b4_avx2(
    words: &[u32],
    lut_f32: &[f32; 64],
    x_tile: &[f32; 16],
    acc: &mut [f32; 16],
    state_bits: u32,
) {
    use std::arch::x86_64::*;
    unsafe {
        let state_mask = _mm256_set1_epi32(((1u32 << state_bits) - 1) as i32);
        let hash_mult = _mm256_set1_epi32(HYB_HASH_MULT as i32);
        let idx_mask = _mm256_set1_epi32(31);
        let sign_bit = _mm256_set1_epi32(0x8000_0000u32 as i32);
        let rot1 = _mm256_setr_epi32(1, 2, 3, 4, 5, 6, 7, 0);

        let acc_lo = _mm256_loadu_ps(acc.as_ptr());
        let acc_hi = _mm256_loadu_ps(acc.as_ptr().add(8));
        let (acc_lo, acc_hi) = avx2_tile_b4_step(
            words.as_ptr(),
            lut_f32.as_ptr(),
            x_tile,
            acc_lo,
            acc_hi,
            state_mask,
            hash_mult,
            idx_mask,
            sign_bit,
            rot1,
        );
        _mm256_storeu_ps(acc.as_mut_ptr(), acc_lo);
        _mm256_storeu_ps(acc.as_mut_ptr().add(8), acc_hi);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn hyb_q5_tile_dot_mma_b6_avx2(
    words: &[u32],
    lut_f32: &[f32; 64],
    x_tile: &[f32; 16],
    acc: &mut [f32; 16],
    state_bits: u32,
) {
    use std::arch::x86_64::*;
    unsafe {
        let state_mask = _mm256_set1_epi32(((1u32 << state_bits) - 1) as i32);
        let hash_mult = _mm256_set1_epi32(HYB_HASH_MULT as i32);
        let idx_mask = _mm256_set1_epi32(31);
        let sign_bit = _mm256_set1_epi32(0x8000_0000u32 as i32);
        let rot1 = _mm256_setr_epi32(1, 2, 3, 4, 5, 6, 7, 0);

        let acc_lo = _mm256_loadu_ps(acc.as_ptr());
        let acc_hi = _mm256_loadu_ps(acc.as_ptr().add(8));
        let (acc_lo, acc_hi) = avx2_tile_b6_step(
            words.as_ptr(),
            lut_f32.as_ptr(),
            x_tile,
            acc_lo,
            acc_hi,
            state_mask,
            hash_mult,
            idx_mask,
            sign_bit,
            rot1,
        );
        _mm256_storeu_ps(acc.as_mut_ptr(), acc_lo);
        _mm256_storeu_ps(acc.as_mut_ptr().add(8), acc_hi);
    }
}

#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn neon_decode_and_fma_4(
    states: std::arch::aarch64::uint32x4_t,
    lut_f32: &[f32; 64],
    x0: f32,
    x1: f32,
    acc: std::arch::aarch64::float32x4_t,
) -> std::arch::aarch64::float32x4_t {
    use std::arch::aarch64::*;
    unsafe {
        let h = vmulq_u32(states, vdupq_n_u32(HYB_HASH_MULT));
        let mut h_arr = [0u32; 4];
        vst1q_u32(h_arr.as_mut_ptr(), h);

        let mut w0_arr = [0.0f32; 4];
        let mut w1_arr = [0.0f32; 4];
        for i in 0..4 {
            let hi = h_arr[i];
            let idx = ((hi >> 5) & 31) as usize;
            let s0 = (hi << 16) & 0x8000_0000;
            let s1 = hi & 0x8000_0000;
            w0_arr[i] = f32::from_bits(lut_f32[2 * idx].to_bits() ^ s0);
            w1_arr[i] = f32::from_bits(lut_f32[2 * idx + 1].to_bits() ^ s1);
        }
        let w0 = vld1q_f32(w0_arr.as_ptr());
        let w1 = vld1q_f32(w1_arr.as_ptr());
        let acc = vfmaq_n_f32(acc, w0, x0);
        vfmaq_n_f32(acc, w1, x1)
    }
}

#[cfg(target_arch = "aarch64")]
unsafe fn hyb_q5_tile_dot_mma_b4_neon(
    words: &[u32],
    lut_f32: &[f32; 64],
    x_tile: &[f32; 16],
    acc: &mut [f32; 16],
    state_bits: u32,
) {
    use std::arch::aarch64::*;
    unsafe {
        let state_mask = vdupq_n_u32((1u32 << state_bits) - 1);

        for half in 0..2usize {
            let g0 = half * 4;
            let w_even_arr = [
                words[2 * g0],
                words[2 * (g0 + 1)],
                words[2 * (g0 + 2)],
                words[2 * (g0 + 3)],
            ];
            let w_odd_arr = [
                words[2 * g0 + 1],
                words[2 * (g0 + 1) + 1],
                words[2 * (g0 + 2) + 1],
                words[2 * (g0 + 3) + 1],
            ];
            let w_next_arr = [
                words[(2 * (g0 + 1)) % 16],
                words[(2 * (g0 + 2)) % 16],
                words[(2 * (g0 + 3)) % 16],
                words[(2 * (g0 + 4)) % 16],
            ];
            let w_even = vld1q_u32(w_even_arr.as_ptr());
            let w_odd = vld1q_u32(w_odd_arr.as_ptr());
            let w_next = vld1q_u32(w_next_arr.as_ptr());
            let s16 = vorrq_u32(vshrq_n_u32::<16>(w_even), vshlq_n_u32::<16>(w_odd));
            let s48 = vorrq_u32(vshrq_n_u32::<16>(w_odd), vshlq_n_u32::<16>(w_next));

            let mut acc_lo = vld1q_f32(acc.as_ptr().add(g0));
            let mut acc_hi = vld1q_f32(acc.as_ptr().add(g0 + 8));

            let windows = [w_even, s16, w_odd, s48];
            for (tid, &win) in windows.iter().enumerate() {
                let c0 = 2 * tid;
                let c1 = c0 + 8;
                let st0 = vandq_u32(win, state_mask);
                let st1 = vandq_u32(vshrq_n_u32::<4>(win), state_mask);
                let st2 = vandq_u32(vshrq_n_u32::<8>(win), state_mask);
                let st3 = vandq_u32(vshrq_n_u32::<12>(win), state_mask);
                acc_lo = neon_decode_and_fma_4(st0, lut_f32, x_tile[c0], x_tile[c0 + 1], acc_lo);
                acc_hi = neon_decode_and_fma_4(st1, lut_f32, x_tile[c0], x_tile[c0 + 1], acc_hi);
                acc_lo = neon_decode_and_fma_4(st2, lut_f32, x_tile[c1], x_tile[c1 + 1], acc_lo);
                acc_hi = neon_decode_and_fma_4(st3, lut_f32, x_tile[c1], x_tile[c1 + 1], acc_hi);
            }

            vst1q_f32(acc.as_mut_ptr().add(g0), acc_lo);
            vst1q_f32(acc.as_mut_ptr().add(g0 + 8), acc_hi);
        }
    }
}

#[cfg(target_arch = "aarch64")]
unsafe fn hyb_q5_tile_dot_mma_b6_neon(
    words: &[u32],
    lut_f32: &[f32; 64],
    x_tile: &[f32; 16],
    acc: &mut [f32; 16],
    state_bits: u32,
) {
    use std::arch::aarch64::*;
    unsafe {
        let state_mask = vdupq_n_u32((1u32 << state_bits) - 1);

        for half in 0..2usize {
            let g0 = half * 4;
            let w0_arr = [words[3 * g0], words[3 * (g0 + 1)], words[3 * (g0 + 2)], words[3 * (g0 + 3)]];
            let w1_arr = [words[3 * g0 + 1], words[3 * (g0 + 1) + 1], words[3 * (g0 + 2) + 1], words[3 * (g0 + 3) + 1]];
            let w2_arr = [words[3 * g0 + 2], words[3 * (g0 + 1) + 2], words[3 * (g0 + 2) + 2], words[3 * (g0 + 3) + 2]];
            let w3_arr = [
                words[(3 * (g0 + 1)) % 24],
                words[(3 * (g0 + 2)) % 24],
                words[(3 * (g0 + 3)) % 24],
                words[(3 * (g0 + 4)) % 24],
            ];
            let w0 = vld1q_u32(w0_arr.as_ptr());
            let w1 = vld1q_u32(w1_arr.as_ptr());
            let w2 = vld1q_u32(w2_arr.as_ptr());
            let w3 = vld1q_u32(w3_arr.as_ptr());

            let mut acc_lo = vld1q_f32(acc.as_ptr().add(g0));
            let mut acc_hi = vld1q_f32(acc.as_ptr().add(g0 + 8));

            {
                let st0 = vandq_u32(w0, state_mask);
                let st1 = vandq_u32(vshrq_n_u32::<6>(w0), state_mask);
                let st2 = vandq_u32(vshrq_n_u32::<12>(w0), state_mask);
                let st3 = vandq_u32(vorrq_u32(vshrq_n_u32::<18>(w0), vshlq_n_u32::<14>(w1)), state_mask);
                acc_lo = neon_decode_and_fma_4(st0, lut_f32, x_tile[0], x_tile[1], acc_lo);
                acc_hi = neon_decode_and_fma_4(st1, lut_f32, x_tile[0], x_tile[1], acc_hi);
                acc_lo = neon_decode_and_fma_4(st2, lut_f32, x_tile[8], x_tile[9], acc_lo);
                acc_hi = neon_decode_and_fma_4(st3, lut_f32, x_tile[8], x_tile[9], acc_hi);
            }
            {
                let b24 = vorrq_u32(vshrq_n_u32::<24>(w0), vshlq_n_u32::<8>(w1));
                let st0 = vandq_u32(b24, state_mask);
                let st1 = vandq_u32(vshrq_n_u32::<6>(b24), state_mask);
                let st2 = vandq_u32(vshrq_n_u32::<4>(w1), state_mask);
                let st3 = vandq_u32(vshrq_n_u32::<10>(w1), state_mask);
                acc_lo = neon_decode_and_fma_4(st0, lut_f32, x_tile[2], x_tile[3], acc_lo);
                acc_hi = neon_decode_and_fma_4(st1, lut_f32, x_tile[2], x_tile[3], acc_hi);
                acc_lo = neon_decode_and_fma_4(st2, lut_f32, x_tile[10], x_tile[11], acc_lo);
                acc_hi = neon_decode_and_fma_4(st3, lut_f32, x_tile[10], x_tile[11], acc_hi);
            }
            {
                let b48 = vorrq_u32(vshrq_n_u32::<16>(w1), vshlq_n_u32::<16>(w2));
                let st0 = vandq_u32(b48, state_mask);
                let st1 = vandq_u32(vshrq_n_u32::<6>(b48), state_mask);
                let st2 = vandq_u32(vshrq_n_u32::<12>(b48), state_mask);
                let st3 = vandq_u32(vshrq_n_u32::<2>(w2), state_mask);
                acc_lo = neon_decode_and_fma_4(st0, lut_f32, x_tile[4], x_tile[5], acc_lo);
                acc_hi = neon_decode_and_fma_4(st1, lut_f32, x_tile[4], x_tile[5], acc_hi);
                acc_lo = neon_decode_and_fma_4(st2, lut_f32, x_tile[12], x_tile[13], acc_lo);
                acc_hi = neon_decode_and_fma_4(st3, lut_f32, x_tile[12], x_tile[13], acc_hi);
            }
            {
                let b72 = vorrq_u32(vshrq_n_u32::<8>(w2), vshlq_n_u32::<24>(w3));
                let st0 = vandq_u32(b72, state_mask);
                let st1 = vandq_u32(vshrq_n_u32::<6>(b72), state_mask);
                let st2 = vandq_u32(vshrq_n_u32::<12>(b72), state_mask);
                let st3 = vandq_u32(vorrq_u32(vshrq_n_u32::<26>(w2), vshlq_n_u32::<6>(w3)), state_mask);
                acc_lo = neon_decode_and_fma_4(st0, lut_f32, x_tile[6], x_tile[7], acc_lo);
                acc_hi = neon_decode_and_fma_4(st1, lut_f32, x_tile[6], x_tile[7], acc_hi);
                acc_lo = neon_decode_and_fma_4(st2, lut_f32, x_tile[14], x_tile[15], acc_lo);
                acc_hi = neon_decode_and_fma_4(st3, lut_f32, x_tile[14], x_tile[15], acc_hi);
            }

            vst1q_f32(acc.as_mut_ptr().add(g0), acc_lo);
            vst1q_f32(acc.as_mut_ptr().add(g0 + 8), acc_hi);
        }
    }
}

// ---------------------------------------------------------------------------
// Full 256-row block matvec
// ---------------------------------------------------------------------------

/// Scalar reference implementation of `hyb_q5_chunk_matvec_256`.
#[allow(clippy::too_many_arguments)]
pub fn hyb_q5_chunk_matvec_256_scalar(
    block_tiles: &[u32],
    lut_f32: &[f32; 64],
    x: &[f32],
    col_signs: Option<&[f32]>,
    row_signs: Option<&[f32; 256]>,
    y_out: &mut [f32; 256],
    cols: usize,
    state_bits: u32,
    pair_bits: u32,
    scale: f32,
) {
    debug_assert_eq!(cols % HYB_TILE, 0);
    debug_assert!(col_signs.is_none() || cols % HYB_CHUNK == 0);
    let wpt = hyb_tile_words(pair_bits);
    let col_tiles = cols / HYB_TILE;
    debug_assert!(block_tiles.len() >= 16 * col_tiles * wpt);
    debug_assert!(x.len() >= cols);

    let mut y_acc = [0.0f32; 256];

    if let Some(cs) = col_signs {
        let chunks = cols / HYB_CHUNK;
        let mut x_chunk = [0.0f32; HYB_CHUNK];
        for ch in 0..chunks {
            let c0 = ch * HYB_CHUNK;
            for i in 0..HYB_CHUNK {
                x_chunk[i] = x[c0 + i] * cs[c0 + i];
            }
            fwht256_inplace_scalar(&mut x_chunk);
            for group in 0..16usize {
                let acc: &mut [f32; 16] = (&mut y_acc[group * 16..(group + 1) * 16])
                    .try_into()
                    .unwrap();
                let group_base = group * col_tiles * wpt;
                for tk in 0..16usize {
                    let k = ch * 16 + tk;
                    let tile = &block_tiles[group_base + k * wpt..group_base + (k + 1) * wpt];
                    let x_tile: &[f32; 16] = (&x_chunk[tk * 16..(tk + 1) * 16]).try_into().unwrap();
                    hyb_q5_tile_dot_mma_scalar(tile, lut_f32, x_tile, acc, state_bits, pair_bits);
                }
            }
        }
    } else {
        for group in 0..16usize {
            let acc: &mut [f32; 16] = (&mut y_acc[group * 16..(group + 1) * 16])
                .try_into()
                .unwrap();
            let group_base = group * col_tiles * wpt;
            for k in 0..col_tiles {
                let tile = &block_tiles[group_base + k * wpt..group_base + (k + 1) * wpt];
                let x_tile: &[f32; 16] = (&x[k * 16..(k + 1) * 16]).try_into().unwrap();
                hyb_q5_tile_dot_mma_scalar(tile, lut_f32, x_tile, acc, state_bits, pair_bits);
            }
        }
    }

    if let Some(rs) = row_signs {
        fwht256_inplace_scalar(&mut y_acc);
        for i in 0..256 {
            y_out[i] = y_acc[i] * rs[i] * scale;
        }
    } else {
        for i in 0..256 {
            y_out[i] = y_acc[i] * scale;
        }
    }
}

/// Processes a full 256-row HYB Q=5 block across `cols` for a single vector `x`,
/// with optional input RHT (`col_signs`), optional output RHT (`row_signs`), and `scale`.
///
/// Zero heap allocations; cleanly reusable by `Plt1CpuRuntime` inside `rayon` row-block loops.
#[allow(clippy::too_many_arguments)]
pub fn hyb_q5_chunk_matvec_256(
    block_tiles: &[u32],
    lut_f32: &[f32; 64],
    x: &[f32],
    col_signs: Option<&[f32]>,
    row_signs: Option<&[f32; 256]>,
    y_out: &mut [f32; 256],
    cols: usize,
    state_bits: u32,
    pair_bits: u32,
    scale: f32,
) {
    debug_assert_eq!(cols % HYB_TILE, 0);
    debug_assert!(col_signs.is_none() || cols % HYB_CHUNK == 0);
    let wpt = hyb_tile_words(pair_bits);
    let col_tiles = cols / HYB_TILE;
    debug_assert!(block_tiles.len() >= 16 * col_tiles * wpt);
    debug_assert!(x.len() >= cols);

    #[cfg(target_arch = "x86_64")]
    {
        if (pair_bits == 4 || pair_bits == 6)
            && std::is_x86_feature_detected!("avx2")
            && std::is_x86_feature_detected!("fma")
        {
            // SAFETY: AVX2+FMA verified at runtime; slice lengths checked above.
            unsafe {
                hyb_q5_chunk_matvec_256_avx2(
                    block_tiles,
                    lut_f32,
                    x,
                    col_signs,
                    row_signs,
                    y_out,
                    cols,
                    state_bits,
                    pair_bits,
                    scale,
                );
            }
            return;
        }
    }

    // Fallback path (aarch64 NEON via fwht256_inplace + hyb_q5_tile_dot_mma, or scalar).
    let mut y_acc = [0.0f32; 256];

    if let Some(cs) = col_signs {
        let chunks = cols / HYB_CHUNK;
        let mut x_chunk = [0.0f32; HYB_CHUNK];
        for ch in 0..chunks {
            let c0 = ch * HYB_CHUNK;
            for i in 0..HYB_CHUNK {
                x_chunk[i] = x[c0 + i] * cs[c0 + i];
            }
            fwht256_inplace(&mut x_chunk);
            for group in 0..16usize {
                let acc: &mut [f32; 16] = (&mut y_acc[group * 16..(group + 1) * 16])
                    .try_into()
                    .unwrap();
                let group_base = group * col_tiles * wpt;
                for tk in 0..16usize {
                    let k = ch * 16 + tk;
                    let tile = &block_tiles[group_base + k * wpt..group_base + (k + 1) * wpt];
                    let x_tile: &[f32; 16] = (&x_chunk[tk * 16..(tk + 1) * 16]).try_into().unwrap();
                    hyb_q5_tile_dot_mma(tile, lut_f32, x_tile, acc, state_bits, pair_bits);
                }
            }
        }
    } else {
        for group in 0..16usize {
            let acc: &mut [f32; 16] = (&mut y_acc[group * 16..(group + 1) * 16])
                .try_into()
                .unwrap();
            let group_base = group * col_tiles * wpt;
            for k in 0..col_tiles {
                let tile = &block_tiles[group_base + k * wpt..group_base + (k + 1) * wpt];
                let x_tile: &[f32; 16] = (&x[k * 16..(k + 1) * 16]).try_into().unwrap();
                hyb_q5_tile_dot_mma(tile, lut_f32, x_tile, acc, state_bits, pair_bits);
            }
        }
    }

    if let Some(rs) = row_signs {
        fwht256_inplace(&mut y_acc);
        for i in 0..256 {
            y_out[i] = y_acc[i] * rs[i] * scale;
        }
    } else {
        for i in 0..256 {
            y_out[i] = y_acc[i] * scale;
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[allow(clippy::too_many_arguments)]
unsafe fn hyb_q5_chunk_matvec_256_avx2(
    block_tiles: &[u32],
    lut_f32: &[f32; 64],
    x: &[f32],
    col_signs: Option<&[f32]>,
    row_signs: Option<&[f32; 256]>,
    y_out: &mut [f32; 256],
    cols: usize,
    state_bits: u32,
    pair_bits: u32,
    scale: f32,
) {
    use std::arch::x86_64::*;

    unsafe {
        let wpt = hyb_tile_words(pair_bits);
        let col_tiles = cols / HYB_TILE;
        let state_mask = _mm256_set1_epi32(((1u32 << state_bits) - 1) as i32);
        let hash_mult = _mm256_set1_epi32(HYB_HASH_MULT as i32);
        let idx_mask = _mm256_set1_epi32(31);
        let sign_bit = _mm256_set1_epi32(0x8000_0000u32 as i32);
        let rot1 = _mm256_setr_epi32(1, 2, 3, 4, 5, 6, 7, 0);
        let lut_ptr = lut_f32.as_ptr();
        let tiles_ptr = block_tiles.as_ptr();

        let mut y_acc = [0.0f32; 256];

        if let Some(cs) = col_signs {
            let chunks = cols / HYB_CHUNK;
            let mut x_chunk = [0.0f32; HYB_CHUNK];
            for ch in 0..chunks {
                let c0 = ch * HYB_CHUNK;
                let xp = x.as_ptr().add(c0);
                let cp = cs.as_ptr().add(c0);
                let dp = x_chunk.as_mut_ptr();
                for i in (0..HYB_CHUNK).step_by(8) {
                    let vx = _mm256_loadu_ps(xp.add(i));
                    let vc = _mm256_loadu_ps(cp.add(i));
                    _mm256_storeu_ps(dp.add(i), _mm256_mul_ps(vx, vc));
                }
                fwht256_inplace_avx2(&mut x_chunk);

                for group in 0..16usize {
                    let acc_ptr = y_acc.as_mut_ptr().add(group * 16);
                    let mut acc_lo = _mm256_loadu_ps(acc_ptr);
                    let mut acc_hi = _mm256_loadu_ps(acc_ptr.add(8));
                    let group_base = group * col_tiles * wpt + ch * 16 * wpt;
                    if pair_bits == 4 {
                        for tk in 0..16usize {
                            let x_tile = &*(x_chunk.as_ptr().add(tk * 16) as *const [f32; 16]);
                            let (lo, hi) = avx2_tile_b4_step(
                                tiles_ptr.add(group_base + tk * 16),
                                lut_ptr,
                                x_tile,
                                acc_lo,
                                acc_hi,
                                state_mask,
                                hash_mult,
                                idx_mask,
                                sign_bit,
                                rot1,
                            );
                            acc_lo = lo;
                            acc_hi = hi;
                        }
                    } else {
                        for tk in 0..16usize {
                            let x_tile = &*(x_chunk.as_ptr().add(tk * 16) as *const [f32; 16]);
                            let (lo, hi) = avx2_tile_b6_step(
                                tiles_ptr.add(group_base + tk * 24),
                                lut_ptr,
                                x_tile,
                                acc_lo,
                                acc_hi,
                                state_mask,
                                hash_mult,
                                idx_mask,
                                sign_bit,
                                rot1,
                            );
                            acc_lo = lo;
                            acc_hi = hi;
                        }
                    }
                    _mm256_storeu_ps(acc_ptr, acc_lo);
                    _mm256_storeu_ps(acc_ptr.add(8), acc_hi);
                }
            }
        } else {
            // SKIP_IN_RHT fast path: keep acc_lo / acc_hi in YMM registers across all col_tiles.
            for group in 0..16usize {
                let mut acc_lo = _mm256_setzero_ps();
                let mut acc_hi = _mm256_setzero_ps();
                let group_base = group * col_tiles * wpt;
                if pair_bits == 4 {
                    for k in 0..col_tiles {
                        let x_tile = &*(x.as_ptr().add(k * 16) as *const [f32; 16]);
                        let (lo, hi) = avx2_tile_b4_step(
                            tiles_ptr.add(group_base + k * 16),
                            lut_ptr,
                            x_tile,
                            acc_lo,
                            acc_hi,
                            state_mask,
                            hash_mult,
                            idx_mask,
                            sign_bit,
                            rot1,
                        );
                        acc_lo = lo;
                        acc_hi = hi;
                    }
                } else {
                    for k in 0..col_tiles {
                        let x_tile = &*(x.as_ptr().add(k * 16) as *const [f32; 16]);
                        let (lo, hi) = avx2_tile_b6_step(
                            tiles_ptr.add(group_base + k * 24),
                            lut_ptr,
                            x_tile,
                            acc_lo,
                            acc_hi,
                            state_mask,
                            hash_mult,
                            idx_mask,
                            sign_bit,
                            rot1,
                        );
                        acc_lo = lo;
                        acc_hi = hi;
                    }
                }
                let acc_ptr = y_acc.as_mut_ptr().add(group * 16);
                _mm256_storeu_ps(acc_ptr, acc_lo);
                _mm256_storeu_ps(acc_ptr.add(8), acc_hi);
            }
        }

        let vscale = _mm256_set1_ps(scale);
        let yp = y_out.as_mut_ptr();
        let ap = y_acc.as_ptr();
        if let Some(rs) = row_signs {
            fwht256_inplace_avx2(&mut y_acc);
            let rp = rs.as_ptr();
            for i in (0..256).step_by(8) {
                let va = _mm256_loadu_ps(ap.add(i));
                let vr = _mm256_loadu_ps(rp.add(i));
                _mm256_storeu_ps(yp.add(i), _mm256_mul_ps(_mm256_mul_ps(va, vr), vscale));
            }
        } else {
            for i in (0..256).step_by(8) {
                let va = _mm256_loadu_ps(ap.add(i));
                _mm256_storeu_ps(yp.add(i), _mm256_mul_ps(va, vscale));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lcg_next(state: &mut u64) -> u32 {
        *state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (*state >> 32) as u32
    }

    fn rand_f32(state: &mut u64) -> f32 {
        let u = (lcg_next(state) as f64) / (u32::MAX as f64);
        (u * 2.0 - 1.0) as f32
    }

    fn make_test_lut(seed: u64) -> [u32; 32] {
        let mut s = seed;
        let mut lut = [0u32; 32];
        for slot in &mut lut {
            let a = (rand_f32(&mut s).abs() * 2.0 + 0.05).min(3.5);
            let b = (rand_f32(&mut s).abs() * 2.0 + 0.05).min(3.5);
            let h0 = f16::from_f32(a).to_bits() & 0x7FFF;
            let h1 = f16::from_f32(b).to_bits() & 0x7FFF;
            *slot = u32::from(h0) | (u32::from(h1) << 16);
        }
        lut
    }

    #[test]
    fn fwht256_simd_matches_scalar_and_is_involution() {
        let mut s = 0x1234_5678_9ABC_DEF0u64;
        let mut orig = [0.0f32; 256];
        for v in &mut orig {
            *v = rand_f32(&mut s);
        }

        let mut a = orig;
        let mut b = orig;
        fwht256_inplace_scalar(&mut a);
        fwht256_inplace(&mut b);

        let max_diff = a
            .iter()
            .zip(&b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max);
        assert!(max_diff < 1e-6, "fwht256 SIMD vs scalar max_diff={max_diff}");

        // Involution: FWHT(FWHT(x)) == 256 * x
        fwht256_inplace(&mut b);
        let max_inv_err = orig
            .iter()
            .zip(&b)
            .map(|(x, y)| (y - 256.0 * x).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_inv_err < 1e-4,
            "fwht256 involution error max_inv_err={max_inv_err}"
        );
    }

    #[test]
    fn hyb_r1_enter_exit_roundtrip() {
        let mut s = 0xCAFE_BABE_DEAD_BEEFu64;
        let mut x = [0.0f32; 256];
        let mut signs = [1.0f32; 256];
        for i in 0..256 {
            x[i] = rand_f32(&mut s);
            signs[i] = if (lcg_next(&mut s) & 1) == 0 { 1.0 } else { -1.0 };
        }
        let orig = x;

        hyb_r1_enter_chunk(&mut x, &signs);
        hyb_r1_exit_chunk(&mut x, &signs);

        let max_err = orig
            .iter()
            .zip(&x)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(max_err < 1e-6, "R1 enter/exit roundtrip max_err={max_err}");
    }

    #[test]
    fn hyb_q5_tile_dot_mma_simd_matches_scalar() {
        let lut_f16x2 = make_test_lut(0x1111_2222);
        let lut_f32 = f16x2_lut_to_f32(&lut_f16x2);
        let mut s = 0x9999_8888_7777_6666u64;

        for &(state_bits, pair_bits) in &[(12u32, 4u32), (10u32, 6u32), (16u32, 4u32), (16u32, 6u32), (12u32, 3u32), (12u32, 5u32)] {
            let wpt = hyb_tile_words(pair_bits);
            let words: Vec<u32> = (0..wpt).map(|_| lcg_next(&mut s)).collect();
            let mut x_tile = [0.0f32; 16];
            for v in &mut x_tile {
                *v = rand_f32(&mut s);
            }

            let mut acc_scalar = [0.0f32; 16];
            let mut acc_simd = [0.0f32; 16];
            hyb_q5_tile_dot_mma_scalar(&words, &lut_f32, &x_tile, &mut acc_scalar, state_bits, pair_bits);
            hyb_q5_tile_dot_mma(&words, &lut_f32, &x_tile, &mut acc_simd, state_bits, pair_bits);

            let norm: f32 = acc_scalar.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-6);
            let max_diff = acc_scalar
                .iter()
                .zip(&acc_simd)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            let rel_err = max_diff / norm;
            assert!(
                rel_err < 1e-5,
                "tile dot (L={state_bits}, B={pair_bits}) rel_err={rel_err} (max_diff={max_diff})"
            );
        }
    }

    #[test]
    fn hyb_q5_chunk_matvec_256_simd_matches_scalar() {
        let lut_f16x2 = make_test_lut(0xABCD_EF01);
        let lut_f32 = f16x2_lut_to_f32(&lut_f16x2);
        let mut s = 0x5555_4444_3333_2222u64;
        let cols = 512usize;
        let col_tiles = cols / HYB_TILE;

        let x: Vec<f32> = (0..cols).map(|_| rand_f32(&mut s)).collect();
        let col_signs: Vec<f32> = (0..cols)
            .map(|_| if (lcg_next(&mut s) & 1) == 0 { 1.0 } else { -1.0 })
            .collect();
        let mut row_signs = [1.0f32; 256];
        for r in &mut row_signs {
            *r = if (lcg_next(&mut s) & 1) == 0 { 1.0 } else { -1.0 };
        }

        for &(state_bits, pair_bits) in &[(12u32, 4u32), (10u32, 6u32)] {
            let wpt = hyb_tile_words(pair_bits);
            let tiles: Vec<u32> = (0..16 * col_tiles * wpt).map(|_| lcg_next(&mut s)).collect();
            for &(use_in, use_out) in &[(false, false), (true, false), (false, true), (true, true)] {
                let cs = use_in.then_some(col_signs.as_slice());
                let rs = use_out.then_some(&row_signs);
                let mut y_scalar = [0.0f32; 256];
                let mut y_simd = [0.0f32; 256];
                let scale = 0.03125f32;

                hyb_q5_chunk_matvec_256_scalar(
                    &tiles, &lut_f32, &x, cs, rs, &mut y_scalar, cols, state_bits, pair_bits, scale,
                );
                hyb_q5_chunk_matvec_256(
                    &tiles, &lut_f32, &x, cs, rs, &mut y_simd, cols, state_bits, pair_bits, scale,
                );

                let norm: f32 = y_scalar.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-6);
                let max_diff = y_scalar
                    .iter()
                    .zip(&y_simd)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0f32, f32::max);
                let rel_err = max_diff / norm;
                assert!(
                    rel_err < 1e-5,
                    "chunk_matvec (L={state_bits}, B={pair_bits}, in={use_in}, out={use_out}) rel_err={rel_err}"
                );
            }
        }
    }

    #[cfg(feature = "qtip-research")]
    #[test]
    fn hyb_q5_tile_dot_mma_matches_qtip_reference_decoder() {
        use crate::qtip::hyb::{HybCodebook, decode_hyb_tile_mma_l};

        let codebook = HybCodebook::gaussian_for_q(5);
        let lut_f16x2: [u32; 32] = codebook.lut_f16x2.as_slice().try_into().unwrap();
        let lut_f32 = f16x2_lut_to_f32(&lut_f16x2);

        let mut s = 0xDEAD_BEEF_1337_0001u64;
        for &(state_bits, pair_bits) in &[(12u32, 4u32), (10u32, 6u32)] {
            let wpt = hyb_tile_words(pair_bits);
            let words: Vec<u32> = (0..wpt).map(|_| lcg_next(&mut s)).collect();
            let mut x_tile = [0.0f32; 16];
            for v in &mut x_tile {
                *v = rand_f32(&mut s);
            }

            let decoded = decode_hyb_tile_mma_l(&words, state_bits, pair_bits, &codebook, 1.0);
            let mut expected = [0.0f32; 16];
            for r in 0..16 {
                let mut sum = 0.0f32;
                for c in 0..16 {
                    sum += decoded[r * 16 + c] * x_tile[c];
                }
                expected[r] = sum;
            }

            let mut actual = [0.0f32; 16];
            hyb_q5_tile_dot_mma(&words, &lut_f32, &x_tile, &mut actual, state_bits, pair_bits);

            let norm: f32 = expected.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-6);
            let max_diff = expected
                .iter()
                .zip(&actual)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            let rel_err = max_diff / norm;
            assert!(
                rel_err < 1e-5,
                "qtip reference mismatch (L={state_bits}, B={pair_bits}): rel_err={rel_err} (max_diff={max_diff})"
            );
        }
    }
}
