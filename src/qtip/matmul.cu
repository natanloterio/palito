// Batched QTIP3 projection for prefill on FP16 tensor cores (sm_80+),
// appended to matvec.cu. With W = scale * D_r H Ŵ H D_c (unnormalized
// 256-block Hadamards H, sign diagonals D), Y = W X for T tokens is:
//   1. qtip_rotate_input:  X' = H D_c X / 16 in FP16 (one warp per 256-chunk);
//   2. qtip_matmul_mma:    Y' = Ŵ X', each warp decoding a 16x16 trellis tile
//                          once and feeding it to mma.m16n8k16 for 64 tokens;
//   3. qtip_rotate_output: Y = 16 scale D_r H Y'.
// Step 2 splits K over blockIdx.z (each split writes its own partial plane
// of Y', and step 3 sums the planes in order): deterministic, no atomics.
// The decoded codes are integers in [-510, 510], exact in FP16; X' is the
// only rounded operand (FP16, relative error 2^-11), accumulation is F32.
// Layouts are token-major: x[t * cols + c], y[t * rows + r].

// FP16 bits; NVRTC builds without cuda_fp16.h.
typedef unsigned short qtip_f16;

__device__ __forceinline__ qtip_f16 qtip_to_f16(float v) {
    qtip_f16 h;
    asm("cvt.rn.f16.f32 %0, %1;" : "=h"(h) : "f"(v));
    return h;
}

#define QTIP_MMA_TOKENS 128
#define QTIP_MMA_WARPS 4
// Padding (in halves) keeps the fragment loads free of bank conflicts.
#define QTIP_MMA_PAD 8

// grid (ceil(cols / 256 / 8), tokens), block 256: warp w handles chunk
// blockIdx.x * 8 + w of token blockIdx.y.
extern "C" __global__ void qtip_rotate_input(const float* __restrict__ x,
                                             const float* __restrict__ col_signs,
                                             qtip_f16* __restrict__ xr, int cols) {
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int chunk = blockIdx.x * 8 + warp;
    if (chunk * QTIP_CHUNK >= cols) {
        return;
    }
    const size_t row = (size_t)blockIdx.y * cols;
    float v[8];
#pragma unroll
    for (int e = 0; e < 8; ++e) {
        const int i = chunk * QTIP_CHUNK + lane + 32 * e;
        v[e] = x[row + i] * col_signs[i];
    }
    qtip_fwht256_warp(v);
#pragma unroll
    for (int e = 0; e < 8; ++e) {
        xr[row + chunk * QTIP_CHUNK + lane + 32 * e] = qtip_to_f16(v[e] * 0.0625f);
    }
}

// grid (ceil(rows / 256 / 8), tokens), block 256: y = scale16 * D_r H y',
// with y' the sum of `splits` planes of tokens * rows values.
extern "C" __global__ void qtip_rotate_output(const float* __restrict__ yr,
                                              const float* __restrict__ row_signs,
                                              float* __restrict__ y, int rows, float scale16,
                                              int splits, int tokens) {
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int chunk = blockIdx.x * 8 + warp;
    if (chunk * QTIP_CHUNK >= rows) {
        return;
    }
    const size_t row = (size_t)blockIdx.y * rows;
    float v[8];
#pragma unroll
    for (int e = 0; e < 8; ++e) {
        v[e] = 0.0f;
    }
    const size_t plane = (size_t)tokens * rows;
    for (int s = 0; s < splits; ++s) {
#pragma unroll
        for (int e = 0; e < 8; ++e) {
            v[e] += yr[s * plane + row + chunk * QTIP_CHUNK + lane + 32 * e];
        }
    }
    qtip_fwht256_warp(v);
#pragma unroll
    for (int e = 0; e < 8; ++e) {
        const int i = chunk * QTIP_CHUNK + lane + 32 * e;
        y[row + i] = v[e] * row_signs[i] * scale16;
    }
}

__device__ __forceinline__ void qtip_mma(float (&c)[4], const unsigned (&a)[4], unsigned b0,
                                         unsigned b1) {
#if __CUDA_ARCH__ >= 800
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
#endif
}

// Four 8x8 FP16 matrices from shared memory; lane i gives row i % 8 of
// matrix i / 8.
__device__ __forceinline__ void qtip_ldmatrix4(unsigned (&r)[4], const void* row) {
#if __CUDA_ARCH__ >= 800
    const unsigned address = (unsigned)__cvta_generic_to_shared(row);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(address));
#endif
}

// grid (rows / 64, ceil(tokens / 128), splits), block 128. Warp w owns row
// group blockIdx.x * 4 + w (16 rotated rows) and the block's 128 tokens over
// column tiles [z * split_tiles, (z + 1) * split_tiles): each tile is decoded
// once and feeds 16 mma.m16n8k16. Plane z of yr gets that part of the
// rotated, unscaled product Ŵ X' (F32). Inputs and decoded tiles are double
// buffered, so one barrier per column tile suffices.
extern "C" __global__ void __launch_bounds__(QTIP_MMA_WARPS * 32)
qtip_matmul_mma(const unsigned* __restrict__ tiles, const qtip_f16* __restrict__ xr,
                float* __restrict__ yr, int cols, int rows, int tokens, int split_tiles) {
    constexpr int stride = QTIP_TILE + QTIP_MMA_PAD;
    __shared__ __align__(16) qtip_f16 xs[2][QTIP_MMA_TOKENS][stride];
    __shared__ __align__(16) qtip_f16 ws[2][QTIP_MMA_WARPS][QTIP_TILE][stride];
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int col_tiles = cols / QTIP_TILE;
    const int group = blockIdx.x * QTIP_MMA_WARPS + warp;
    const int token0 = blockIdx.y * QTIP_MMA_TOKENS;
    const int k0 = blockIdx.z * split_tiles;
    const int steps = min(k0 + split_tiles, col_tiles) - k0;
    const unsigned* base = tiles + ((size_t)group * col_tiles + k0) * QTIP_WORDS;
    yr += (size_t)blockIdx.z * tokens * rows;
    // Decode layout: lane holds tile elements lane * 8 .. + 8 (row lane / 2).
    const int first = lane * QTIP_PER_LANE;
    const int bit = first * QTIP_K;
    const int hi_word = bit / 32, shift = bit % 32;
    const int lo_word = min(hi_word + 1, 31);
    const int w_row = first / QTIP_TILE, w_col = first % QTIP_TILE;
    // Input staging: thread i loads the 16 columns (32 bytes) of token i.
    const int x_token = threadIdx.x;
    const bool x_valid = token0 + x_token < tokens;
    const uint4* x_src =
        reinterpret_cast<const uint4*>(xr + (size_t)(token0 + x_token) * cols + k0 * QTIP_TILE);
    // ldmatrix row addresses: matrix m = lane / 8, row lane % 8.
    const int m = lane / 8, r = lane % 8;
    const int a_row = r + (m % 2) * 8, a_col = (m / 2) * 8;
    const int b_token = (m / 2) * 8 + r, b_col = (m % 2) * 8;

    float c[QTIP_MMA_TOKENS / 8][4] = {};
    const uint4 zero = make_uint4(0, 0, 0, 0);
    unsigned word = lane < QTIP_WORDS ? __ldg(base + lane) : 0u;
    uint4 x_lo = x_valid ? x_src[0] : zero, x_hi = x_valid ? x_src[1] : zero;
    for (int k = 0; k < steps; ++k) {
        const int b = k & 1;
        *reinterpret_cast<uint4*>(&xs[b][x_token][0]) = x_lo;
        *reinterpret_cast<uint4*>(&xs[b][x_token][8]) = x_hi;
        const unsigned hi = __shfl_sync(0xffffffffu, word, hi_word);
        const unsigned lo = __shfl_sync(0xffffffffu, word, lo_word);
        // Prefetch the next tile and input columns while this one computes.
        if (k + 1 < steps) {
            word = lane < QTIP_WORDS ? __ldg(base + (size_t)(k + 1) * QTIP_WORDS + lane) : 0u;
            x_lo = x_valid ? x_src[2 * (k + 1)] : zero;
            x_hi = x_valid ? x_src[2 * (k + 1) + 1] : zero;
        }
        const unsigned top = __funnelshift_l(lo, hi, shift);
        const unsigned low = lo << shift;
        qtip_f16 decoded[QTIP_PER_LANE];
#pragma unroll
        for (int j = 0; j < QTIP_PER_LANE; ++j) {
            const int end = QTIP_K * j + QTIP_L;
            const unsigned field =
                end <= 32 ? (top >> (32 - end)) : __funnelshift_l(low, top, end - 32);
            decoded[j] = qtip_to_f16(qtip_code(field & ((1u << QTIP_L) - 1u)));
        }
        *reinterpret_cast<uint4*>(&ws[b][warp][w_row][w_col]) =
            *reinterpret_cast<const uint4*>(decoded);
        __syncthreads();
        unsigned a[4];
        qtip_ldmatrix4(a, &ws[b][warp][a_row][a_col]);
#pragma unroll
        for (int n = 0; n < QTIP_MMA_TOKENS / 8; n += 2) {
            unsigned bf[4];
            qtip_ldmatrix4(bf, &xs[b][n * 8 + b_token][b_col]);
            qtip_mma(c[n], a, bf[0], bf[1]);
            qtip_mma(c[n + 1], a, bf[2], bf[3]);
        }
    }
    const int g = lane / 4, t = lane % 4;
    const int row = group * QTIP_TILE + g;
#pragma unroll
    for (int n = 0; n < QTIP_MMA_TOKENS / 8; ++n) {
        const int token = token0 + n * 8 + 2 * t;
        if (token < tokens) {
            yr[(size_t)token * rows + row] = c[n][0];
            yr[(size_t)token * rows + row + 8] = c[n][2];
        }
        if (token + 1 < tokens) {
            yr[(size_t)(token + 1) * rows + row] = c[n][1];
            yr[(size_t)(token + 1) * rows + row + 8] = c[n][3];
        }
    }
}
