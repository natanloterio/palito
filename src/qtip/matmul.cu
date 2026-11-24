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
#ifndef QTIP_MMA_WARPS
#define QTIP_MMA_WARPS 4
#endif
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

#ifndef QTIP_MMA_STEP_TILES
#define QTIP_MMA_STEP_TILES 2
#endif

// grid (rows / 64, ceil(tokens / 128), splits), block 128, split_tiles a
// multiple of QTIP_MMA_STEP_TILES. Warp w owns row group blockIdx.x * 4 + w
// (16 rotated rows) and the block's 128 tokens over column tiles
// [z * split_tiles, (z + 1) * split_tiles), QTIP_MMA_STEP_TILES per step:
// each tile is decoded once and feeds 16 mma.m16n8k16. Plane z of yr gets
// that part of the rotated, unscaled product Ŵ X' (F32). Inputs and decoded
// tiles are double buffered, so one barrier per step suffices.
extern "C" __global__ void __launch_bounds__(QTIP_MMA_WARPS * 32)
qtip_matmul_mma(const unsigned* __restrict__ tiles, const qtip_f16* __restrict__ xr,
                float* __restrict__ yr, int cols, int rows, int tokens, int split_tiles) {
    constexpr int S = QTIP_MMA_STEP_TILES;
    constexpr int stride = S * QTIP_TILE + QTIP_MMA_PAD;
    __shared__ __align__(16) qtip_f16 xs[2][QTIP_MMA_TOKENS][stride];
    __shared__ __align__(16) qtip_f16 ws[2][QTIP_MMA_WARPS][QTIP_TILE][stride];
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int col_tiles = cols / QTIP_TILE;
    const int group = blockIdx.x * QTIP_MMA_WARPS + warp;
    const int token0 = blockIdx.y * QTIP_MMA_TOKENS;
    const int k0 = blockIdx.z * split_tiles;
    const int steps = (min(k0 + split_tiles, col_tiles) - k0) / S;
    const unsigned* base = tiles + ((size_t)group * col_tiles + k0) * QTIP_WORDS;
    yr += (size_t)blockIdx.z * tokens * rows;
    // Decode layout: lane holds tile elements lane * 8 .. + 8 (row lane / 2).
    const int first = lane * QTIP_PER_LANE;
    const int bit = first * QTIP_K;
    const int hi_word = bit / 32, shift = bit % 32;
    const int lo_word = min(hi_word + 1, 31);
    const int w_row = first / QTIP_TILE, w_col = first % QTIP_TILE;
    // Input staging: each step moves 128 tokens x 2S uint4; thread i takes
    // uint4 j = i + T v (v < V) of token j / (2S), part j % (2S).
    constexpr int T = QTIP_MMA_WARPS * 32;
    constexpr int V = (QTIP_MMA_TOKENS * 2 * S + T - 1) / T;
    int x_token[V], x_part[V];
    bool x_valid[V];
    const uint4* x_src[V];
#pragma unroll
    for (int v = 0; v < V; ++v) {
        const int j = threadIdx.x + T * v;
        x_token[v] = j / (2 * S);
        x_part[v] = j % (2 * S);
        x_valid[v] = j < QTIP_MMA_TOKENS * 2 * S && token0 + x_token[v] < tokens;
        x_src[v] = reinterpret_cast<const uint4*>(xr + (size_t)(token0 + x_token[v]) * cols +
                                                  k0 * QTIP_TILE) + x_part[v];
    }
    // ldmatrix row addresses: matrix m = lane / 8, row lane % 8.
    const int m = lane / 8, r = lane % 8;
    const int a_row = r + (m % 2) * 8, a_col = (m / 2) * 8;
    const int b_token = (m / 2) * 8 + r, b_col = (m % 2) * 8;

    float c[QTIP_MMA_TOKENS / 8][4] = {};
    const uint4 zero = make_uint4(0, 0, 0, 0);
    const bool loads = lane < QTIP_WORDS;
    unsigned words[S];
    uint4 xv[V];
#pragma unroll
    for (int u = 0; u < S; ++u) {
        words[u] = loads ? __ldg(base + (size_t)u * QTIP_WORDS + lane) : 0u;
    }
#pragma unroll
    for (int v = 0; v < V; ++v) {
        xv[v] = x_valid[v] ? x_src[v][0] : zero;
    }
    for (int k = 0; k < steps; ++k) {
        const int b = k & 1;
        unsigned current[S];
#pragma unroll
        for (int v = 0; v < V; ++v) {
            if (threadIdx.x + T * v < QTIP_MMA_TOKENS * 2 * S) {
                *reinterpret_cast<uint4*>(&xs[b][x_token[v]][8 * x_part[v]]) = xv[v];
            }
        }
#pragma unroll
        for (int u = 0; u < S; ++u) {
            current[u] = words[u];
        }
        // Prefetch the next step's tiles and inputs while this one computes.
        if (k + 1 < steps) {
#pragma unroll
            for (int u = 0; u < S; ++u) {
                const int tile = (k + 1) * S + u;
                words[u] = loads ? __ldg(base + (size_t)tile * QTIP_WORDS + lane) : 0u;
            }
#pragma unroll
            for (int v = 0; v < V; ++v) {
                xv[v] = x_valid[v] ? x_src[v][2 * S * (k + 1)] : zero;
            }
        }
#pragma unroll
        for (int u = 0; u < S; ++u) {
            const unsigned hi = __shfl_sync(0xffffffffu, current[u], hi_word);
            const unsigned lo = __shfl_sync(0xffffffffu, current[u], lo_word);
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
            *reinterpret_cast<uint4*>(&ws[b][warp][w_row][16 * u + w_col]) =
                *reinterpret_cast<const uint4*>(decoded);
        }
        __syncthreads();
#pragma unroll
        for (int u = 0; u < S; ++u) {
            unsigned a[4];
            qtip_ldmatrix4(a, &ws[b][warp][a_row][16 * u + a_col]);
#pragma unroll
            for (int n = 0; n < QTIP_MMA_TOKENS / 8; n += 2) {
                unsigned bf[4];
                qtip_ldmatrix4(bf, &xs[b][n * 8 + b_token][16 * u + b_col]);
                qtip_mma(c[n], a, bf[0], bf[1]);
                qtip_mma(c[n + 1], a, bf[2], bf[3]);
            }
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

// ---- int8-fast profile -----------------------------------------------------
// The llama.cpp-precision variant (--cuda-matmul int8-fast): weights
// round(code / 4) in int8, inputs X' = H D_c X quantized per token,
// q = round(X' / s_t) with s_t = max|X'_t| / 127; Y' = 4 s_t (W8 q) exact in
// int32. About 1.2% from the exact product (the code / 4 rounding and the
// int8 inputs about equally); measured +0.6% wiki perplexity with half the
// FP16 kernel's time. Same three steps, with qtip_rotate_output taking
// scale (not 16 scale): X' is not divided by 16 here.

#define QTIP_I8_MAX_CHUNKS 32
#define QTIP_I8_TOKENS 128
#define QTIP_I8_WARPS 4

// grid tokens, block 256 (8 warps; warp w rotates chunks w, w + 8, ...):
// xq[t * cols + c] int8, scales[t] = s_t.
extern "C" __global__ void __launch_bounds__(256)
qtip_rotate_input_i8(const float* __restrict__ x, const float* __restrict__ col_signs,
                     signed char* __restrict__ xq, float* __restrict__ scales, int cols) {
    __shared__ float maxima[8];
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int chunks = cols / QTIP_CHUNK;
    const size_t row = (size_t)blockIdx.x * cols;
    float v[QTIP_I8_MAX_CHUNKS / 8][8];
    float peak = 0.0f;
#pragma unroll
    for (int i = 0; i < QTIP_I8_MAX_CHUNKS / 8; ++i) {
        const int chunk = warp + 8 * i;
        if (chunk < chunks) {
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                const int c = chunk * QTIP_CHUNK + lane + 32 * e;
                v[i][e] = x[row + c] * col_signs[c];
            }
            qtip_fwht256_warp(v[i]);
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                peak = fmaxf(peak, fabsf(v[i][e]));
            }
        }
    }
#pragma unroll
    for (int span = 16; span > 0; span >>= 1) {
        peak = fmaxf(peak, __shfl_xor_sync(0xffffffffu, peak, span));
    }
    if (lane == 0) {
        maxima[warp] = peak;
    }
    __syncthreads();
    peak = maxima[0];
#pragma unroll
    for (int w = 1; w < 8; ++w) {
        peak = fmaxf(peak, maxima[w]);
    }
    const float scale = peak / 127.0f;
    const float inverse = peak > 0.0f ? 127.0f / peak : 0.0f;
    if (threadIdx.x == 0) {
        scales[blockIdx.x] = scale;
    }
#pragma unroll
    for (int i = 0; i < QTIP_I8_MAX_CHUNKS / 8; ++i) {
        const int chunk = warp + 8 * i;
        if (chunk < chunks) {
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                xq[row + chunk * QTIP_CHUNK + lane + 32 * e] =
                    (signed char)__float2int_rn(v[i][e] * inverse);
            }
        }
    }
}

__device__ __forceinline__ void qtip_mma_i8(int (&c)[4], const unsigned (&a)[4], unsigned b0,
                                            unsigned b1) {
#if __CUDA_ARCH__ >= 800
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
#endif
}

// Eight int8 weights round(code / 4) of one lane's half tile row, packed.
__device__ __forceinline__ uint2 qtip_decode_i8(unsigned word, int hi_word, int lo_word,
                                                int shift) {
    const unsigned hi = __shfl_sync(0xffffffffu, word, hi_word);
    const unsigned lo = __shfl_sync(0xffffffffu, word, lo_word);
    const unsigned top = __funnelshift_l(lo, hi, shift);
    const unsigned low = lo << shift;
    unsigned packed[2] = {0u, 0u};
#pragma unroll
    for (int j = 0; j < QTIP_PER_LANE; ++j) {
        const int end = QTIP_K * j + QTIP_L;
        const unsigned field =
            end <= 32 ? (top >> (32 - end)) : __funnelshift_l(low, top, end - 32);
        const int q = min(__float2int_rn(qtip_code(field & ((1u << QTIP_L) - 1u)) * 0.25f), 127);
        packed[j / 4] |= (unsigned)(q & 0xff) << (8 * (j % 4));
    }
    return make_uint2(packed[0], packed[1]);
}

// grid (rows / 64, ceil(tokens / 128), splits), block 128, split_tiles even:
// as qtip_matmul_mma on int8. Each step covers two column tiles (32 columns,
// one mma.m16n8k32 per 8 tokens); plane z of yr gets 4 s_t (W8 q).
extern "C" __global__ void __launch_bounds__(QTIP_I8_WARPS * 32)
qtip_matmul_i8(const unsigned* __restrict__ tiles, const signed char* __restrict__ xq,
               const float* __restrict__ scales, float* __restrict__ yr, int cols, int rows,
               int tokens, int split_tiles) {
    // 32 bytes per row plus 16 of padding: conflict-free ldmatrix.
    constexpr int stride = 48;
    __shared__ __align__(16) signed char xs[2][QTIP_I8_TOKENS][stride];
    __shared__ __align__(16) signed char ws[2][QTIP_I8_WARPS][QTIP_TILE][stride];
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int col_tiles = cols / QTIP_TILE;
    const int group = blockIdx.x * QTIP_I8_WARPS + warp;
    const int token0 = blockIdx.y * QTIP_I8_TOKENS;
    const int k0 = blockIdx.z * split_tiles;
    const int steps = (min(k0 + split_tiles, col_tiles) - k0) / 2;
    const unsigned* base = tiles + ((size_t)group * col_tiles + k0) * QTIP_WORDS;
    yr += (size_t)blockIdx.z * tokens * rows;
    const int first = lane * QTIP_PER_LANE;
    const int bit = first * QTIP_K;
    const int hi_word = bit / 32, shift = bit % 32;
    const int lo_word = min(hi_word + 1, 31);
    const int w_row = first / QTIP_TILE, w_col = first % QTIP_TILE;
    const int x_token = threadIdx.x;
    const bool x_valid = token0 + x_token < tokens;
    const uint4* x_src =
        reinterpret_cast<const uint4*>(xq + (size_t)(token0 + x_token) * cols + k0 * QTIP_TILE);
    const int m = lane / 8, r = lane % 8;
    const int a_row = r + (m % 2) * 8, a_col = (m / 2) * 16;
    const int b_token = (m / 2) * 8 + r, b_col = (m % 2) * 16;

    int c[QTIP_I8_TOKENS / 8][4] = {};
    const uint4 zero = make_uint4(0, 0, 0, 0);
    const bool loads = lane < QTIP_WORDS;
    unsigned w0 = loads ? __ldg(base + lane) : 0u;
    unsigned w1 = loads ? __ldg(base + QTIP_WORDS + lane) : 0u;
    uint4 x_lo = x_valid ? x_src[0] : zero, x_hi = x_valid ? x_src[1] : zero;
    for (int k = 0; k < steps; ++k) {
        const int b = k & 1;
        *reinterpret_cast<uint4*>(&xs[b][x_token][0]) = x_lo;
        *reinterpret_cast<uint4*>(&xs[b][x_token][16]) = x_hi;
        const unsigned u0 = w0, u1 = w1;
        if (k + 1 < steps) {
            const unsigned* next = base + (size_t)(2 * k + 2) * QTIP_WORDS;
            w0 = loads ? __ldg(next + lane) : 0u;
            w1 = loads ? __ldg(next + QTIP_WORDS + lane) : 0u;
            x_lo = x_valid ? x_src[2 * (k + 1)] : zero;
            x_hi = x_valid ? x_src[2 * (k + 1) + 1] : zero;
        }
        *reinterpret_cast<uint2*>(&ws[b][warp][w_row][w_col]) =
            qtip_decode_i8(u0, hi_word, lo_word, shift);
        *reinterpret_cast<uint2*>(&ws[b][warp][w_row][16 + w_col]) =
            qtip_decode_i8(u1, hi_word, lo_word, shift);
        __syncthreads();
        unsigned a[4];
        qtip_ldmatrix4(a, &ws[b][warp][a_row][a_col]);
#pragma unroll
        for (int n = 0; n < QTIP_I8_TOKENS / 8; n += 2) {
            unsigned bf[4];
            qtip_ldmatrix4(bf, &xs[b][n * 8 + b_token][b_col]);
            qtip_mma_i8(c[n], a, bf[0], bf[1]);
            qtip_mma_i8(c[n + 1], a, bf[2], bf[3]);
        }
    }
    const int g = lane / 4, t = lane % 4;
    const int row = group * QTIP_TILE + g;
#pragma unroll
    for (int n = 0; n < QTIP_I8_TOKENS / 8; ++n) {
        const int token = token0 + n * 8 + 2 * t;
        if (token < tokens) {
            const float s = 4.0f * scales[token];
            yr[(size_t)token * rows + row] = (float)c[n][0] * s;
            yr[(size_t)token * rows + row + 8] = (float)c[n][2] * s;
        }
        if (token + 1 < tokens) {
            const float s = 4.0f * scales[token + 1];
            yr[(size_t)(token + 1) * rows + row] = (float)c[n][1] * s;
            yr[(size_t)(token + 1) * rows + row + 8] = (float)c[n][3] * s;
        }
    }
}
