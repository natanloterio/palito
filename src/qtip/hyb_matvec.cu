// QTIP HYB (L=16, Q=9, V=2) GPU kernels for Blackwell SM_120 (RTX 5090):
// 1. hyb_matvec_fused: B=1 decode with 32x bank-replicated 64KB SMEM LUT,
//    optional offline-folded RHT (HYB_SKIP_IN_RHT / HYB_SKIP_OUT_RHT), and
//    exact circular tile sizes (48B / 64B / 80B / 96B for 1.5 / 2.0 / 2.5 / 3.0 bpw).
// 2. hyb_matvec_multivec: Multi-vector verification (k = 4, 8, 16 tokens) that
//    reads and decodes each weight ONCE and applies it to k activation vectors
//    in registers — solving the short-prefill / speculative verification bottleneck.

#ifndef HYB_L
#define HYB_L 16
#endif
#ifndef HYB_B
#define HYB_B 4
#endif
#ifndef HYB_Q
#define HYB_Q 5
#endif
#ifndef HYB_MAX_K
#define HYB_MAX_K 16
#endif
#define HYB_STATE_MASK ((1u << HYB_L) - 1u)
#ifndef HYB_UNROLL
#define HYB_UNROLL 4
#endif
#ifndef HYB_SKIP_IN_RHT
#define HYB_SKIP_IN_RHT 0
#endif
#ifndef HYB_SKIP_OUT_RHT
#define HYB_SKIP_OUT_RHT 0
#endif
#ifndef HYB_VEC_K
#define HYB_VEC_K 8
#endif

#define HYB_TILE 16
#define HYB_CHUNK 256
#define HYB_FUSED_WARPS 16
#define HYB_PAIRS_PER_TILE 128
#define HYB_WORDS ((HYB_PAIRS_PER_TILE * HYB_B) / 32)
#define HYB_HASH_MULT 0x9E3779B1u

static_assert(HYB_B >= 3 && HYB_B <= 6, "HYB_B must be in 3..=6 (1.5..=3.0 bpw)");
static_assert(HYB_WORDS <= 24, "HYB_WORDS must fit within one warp");

__device__ __forceinline__ void hyb_fwht256_warp(float (&v)[8]) {
    const int lane = threadIdx.x & 31;
#pragma unroll
    for (int span = 1; span < 8; span <<= 1) {
#pragma unroll
        for (int e = 0; e < 8; ++e) {
            if ((e & span) == 0) {
                const float a = v[e], b = v[e + span];
                v[e] = a + b;
                v[e + span] = a - b;
            }
        }
    }
#pragma unroll
    for (int span = 1; span < 32; span <<= 1) {
#pragma unroll
        for (int e = 0; e < 8; ++e) {
            const float other = __shfl_xor_sync(0xffffffffu, v[e], span);
            v[e] = (lane & span) ? other - v[e] : v[e] + other;
        }
    }
}

#define HYB_LUT_SIZE (1 << HYB_Q)
#define HYB_LUT_MASK ((HYB_LUT_SIZE - 1) << 5)

__device__ __forceinline__ void hyb_decode_pair_lut(
    const unsigned* __restrict__ smem_lut, unsigned reg_lut, unsigned state, int lane, float& wa, float& wb) {
    const unsigned h = state * HYB_HASH_MULT;
#if HYB_Q == 5
    // Zero-SMEM warp-register LUT: 32 entries held across the 32 lanes of each warp!
    // One __shfl_sync instruction, 0 SMEM, 0 bank conflicts, 100% SM occupancy.
    const unsigned raw = __shfl_sync(0xffffffffu, reg_lut, (h >> 5) & 31u) ^ (h & 0x80008000u);
#else
    // Low 5 bits = lane => strictly 0 SMEM bank conflicts across all 32 lanes!
    const unsigned raw = smem_lut[(h & HYB_LUT_MASK) | (unsigned)lane] ^ (h & 0x80008000u);
#endif
    asm("cvt.f32.f16 %0, %1;" : "=f"(wa) : "h"((unsigned short)(raw & 0xFFFFu)));
    asm("cvt.f32.f16 %0, %1;" : "=f"(wb) : "h"((unsigned short)(raw >> 16)));
}

// B=1 decode kernel with zero-SMEM (Q=5) or bank-replicated SMEM LUT (Q=6..9).
// Grid: (rows / 256, splits), Block: 512 threads (16 warps).
extern "C" __global__ void __launch_bounds__(512)
hyb_matvec_fused(const unsigned* __restrict__ tiles,
                 const unsigned* __restrict__ lut_global,
                 const float* __restrict__ x,
                 const float* __restrict__ col_signs,
                 const float* __restrict__ row_signs,
                 float* __restrict__ y,
                 int cols,
                 int tiles_per_split,
                 float scale) {
    extern __shared__ unsigned smem_raw[];
#if HYB_Q == 5
    const unsigned* smem_lut = nullptr;
    float* xs = reinterpret_cast<float*>(smem_raw);
    const unsigned reg_lut = __ldg(lut_global + (threadIdx.x & 31));
#else
    unsigned* smem_lut = smem_raw; // HYB_LUT_SIZE * 32 u32
    float* xs = reinterpret_cast<float*>(smem_raw + HYB_LUT_SIZE * 32);
    const unsigned reg_lut = 0u;
    for (int i = threadIdx.x; i < HYB_LUT_SIZE * 32; i += 512) {
        smem_lut[i] = __ldg(lut_global + (i >> 5));
    }
#endif
    __shared__ float partial[HYB_CHUNK];

    const int col_tiles = cols / HYB_TILE;
    const int k0 = blockIdx.y * tiles_per_split;
    const int k1 = min(k0 + tiles_per_split, col_tiles);
    const int c_first = (k0 * HYB_TILE) / HYB_CHUNK * HYB_CHUNK;
    const int chunks = (k1 * HYB_TILE - c_first + HYB_CHUNK - 1) / HYB_CHUNK;
    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x & 31;

    for (int chunk = warp; chunk < chunks; chunk += HYB_FUSED_WARPS) {
        const int c = c_first + chunk * HYB_CHUNK;
        float v[8];
#pragma unroll
        for (int e = 0; e < 8; ++e) {
            v[e] = __ldg(x + c + lane + 32 * e);
        }
#if !HYB_SKIP_IN_RHT
        float sign[8];
#pragma unroll
        for (int e = 0; e < 8; ++e) {
            sign[e] = __ldg(col_signs + c + lane + 32 * e);
        }
#pragma unroll
        for (int e = 0; e < 8; ++e) {
            v[e] *= sign[e];
        }
        hyb_fwht256_warp(v);
#endif
#pragma unroll
        for (int e = 0; e < 8; ++e) {
            xs[chunk * HYB_CHUNK + lane + 32 * e] = v[e];
        }
    }
    __syncthreads();

    const int group = blockIdx.x * HYB_FUSED_WARPS + warp;
    const int row_in_tile = lane >> 1;
    const int col_in_tile = (lane & 1) * 8;
    const int bit = lane * 4 * HYB_B;
    const int w0_idx = (bit / 32) % HYB_WORDS;
    const int w1_idx = (w0_idx + 1) % HYB_WORDS;
#if HYB_B == 6
    const int w2_idx = (w0_idx + 2) % HYB_WORDS;
#endif
    const int shift = bit & 31;

    const unsigned* base = tiles + (size_t)group * col_tiles * HYB_WORDS;
    const float* xbase = xs + (k0 * HYB_TILE - c_first);
    float acc = 0.0f;

    for (int k = k0; k < k1; k += HYB_UNROLL) {
        unsigned words[HYB_UNROLL];
#pragma unroll
        for (int u = 0; u < HYB_UNROLL; ++u) {
            const int kk = k + u;
            words[u] = (lane < HYB_WORDS && kk < k1)
                           ? __ldg(base + (size_t)kk * HYB_WORDS + lane)
                           : 0u;
        }
#pragma unroll
        for (int u = 0; u < HYB_UNROLL; ++u) {
            const int kk = k + u;
            const unsigned r0 = __shfl_sync(0xffffffffu, words[u], w0_idx);
            const unsigned r1 = __shfl_sync(0xffffffffu, words[u], w1_idx);
#if HYB_B == 6
            const unsigned r2 = __shfl_sync(0xffffffffu, words[u], w2_idx);
#endif
            if (kk >= k1) {
                break;
            }
            const unsigned bits0 = __funnelshift_r(r0, r1, shift);
#if HYB_B == 6
            const unsigned bits1 = __funnelshift_r(r1, r2, shift);
#endif
            const float* xv = xbase + (kk - k0) * HYB_TILE + col_in_tile;
            const float4 xa = *reinterpret_cast<const float4*>(xv);
            const float4 xb = *reinterpret_cast<const float4*>(xv + 4);
            const float xx[8] = {xa.x, xa.y, xa.z, xa.w, xb.x, xb.y, xb.z, xb.w};

#pragma unroll
            for (int p = 0; p < 4; ++p) {
#if HYB_B == 6
                const unsigned state = (p < 3)
                    ? ((bits0 >> (p * 6)) & 0xFFFFu)
                    : (((bits0 >> 18) | (bits1 << 14)) & 0xFFFFu);
#else
                const unsigned state = (bits0 >> (p * HYB_B)) & 0xFFFFu;
#endif
                float wa, wb;
                hyb_decode_pair_lut(smem_lut, reg_lut, state, lane, wa, wb);
                acc = fmaf(wa, xx[2 * p], acc);
                acc = fmaf(wb, xx[2 * p + 1], acc);
            }
        }
    }

    acc += __shfl_xor_sync(0xffffffffu, acc, 1);
    if ((lane & 1) == 0) {
        partial[warp * HYB_TILE + row_in_tile] = acc;
    }
    __syncthreads();

    if (warp == 0) {
        float v[8];
#pragma unroll
        for (int e = 0; e < 8; ++e) {
            v[e] = partial[lane + 32 * e];
        }
#if !HYB_SKIP_OUT_RHT
        hyb_fwht256_warp(v);
#pragma unroll
        for (int e = 0; e < 8; ++e) {
            const int row = blockIdx.x * HYB_CHUNK + lane + 32 * e;
            atomicAdd(&y[row], v[e] * __ldg(row_signs + row) * scale);
        }
#else
#pragma unroll
        for (int e = 0; e < 8; ++e) {
            const int row = blockIdx.x * HYB_CHUNK + lane + 32 * e;
            atomicAdd(&y[row], v[e] * scale);
        }
#endif
    }
}

// Multi-vector verification kernel (k = HYB_VEC_K tokens, e.g. 4, 8, 16).
// Decodes each weight pair ONCE via the 64KB SMEM LUT and applies it to all
// HYB_VEC_K token vectors in registers. Assumes input activations x[t * cols + c]
// are already rotated (or offline-folded). Output y[t * rows + r] accumulated
// via atomicAdd.
// Grid: (rows / 256, splits), Block: 512 threads (16 warps).
// Dynamic shared memory: 64 KB (LUT) only — x is read from L1/L2 cache via __ldg
// (for k<=16, cols<=16384, x is <= 1 MB and stays 100% resident in RTX 5090's 96 MB L2!).
extern "C" __global__ void __launch_bounds__(512)
hyb_matvec_multivec(const unsigned* __restrict__ tiles,
                    const unsigned* __restrict__ lut_global,
                    const float* __restrict__ x,
                    float* __restrict__ y,
                    int rows,
                    int cols,
                    int tiles_per_split,
                    float scale) {
#if HYB_Q == 5
    const unsigned* smem_lut = nullptr;
    const unsigned reg_lut = __ldg(lut_global + (threadIdx.x & 31));
#else
    extern __shared__ unsigned smem_lut[];
    const unsigned reg_lut = 0u;
    for (int i = threadIdx.x; i < HYB_LUT_SIZE * 32; i += 512) {
        smem_lut[i] = __ldg(lut_global + (i >> 5));
    }
    __syncthreads();
#endif

    const int col_tiles = cols / HYB_TILE;
    const int k0 = blockIdx.y * tiles_per_split;
    const int k1 = min(k0 + tiles_per_split, col_tiles);
    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x & 31;

    const int group = blockIdx.x * HYB_FUSED_WARPS + warp;
    const int row_in_tile = lane >> 1;
    const int col_in_tile = (lane & 1) * 8;
    const int bit = lane * 4 * HYB_B;
    const int w0_idx = (bit / 32) % HYB_WORDS;
    const int w1_idx = (w0_idx + 1) % HYB_WORDS;
#if HYB_B == 6
    const int w2_idx = (w0_idx + 2) % HYB_WORDS;
#endif
    const int shift = bit & 31;

    const unsigned* base = tiles + (size_t)group * col_tiles * HYB_WORDS;
    float acc[HYB_VEC_K];
#pragma unroll
    for (int t = 0; t < HYB_VEC_K; ++t) {
        acc[t] = 0.0f;
    }

    for (int k = k0; k < k1; k += HYB_UNROLL) {
        unsigned words[HYB_UNROLL];
#pragma unroll
        for (int u = 0; u < HYB_UNROLL; ++u) {
            const int kk = k + u;
            words[u] = (lane < HYB_WORDS && kk < k1)
                           ? __ldg(base + (size_t)kk * HYB_WORDS + lane)
                           : 0u;
        }
#pragma unroll
        for (int u = 0; u < HYB_UNROLL; ++u) {
            const int kk = k + u;
            const unsigned r0 = __shfl_sync(0xffffffffu, words[u], w0_idx);
            const unsigned r1 = __shfl_sync(0xffffffffu, words[u], w1_idx);
#if HYB_B == 6
            const unsigned r2 = __shfl_sync(0xffffffffu, words[u], w2_idx);
#endif
            if (kk >= k1) {
                break;
            }
            const unsigned bits0 = __funnelshift_r(r0, r1, shift);
#if HYB_B == 6
            const unsigned bits1 = __funnelshift_r(r1, r2, shift);
#endif
            float w_dec[8];
#pragma unroll
            for (int p = 0; p < 4; ++p) {
#if HYB_B == 6
                const unsigned state = (p < 3)
                    ? ((bits0 >> (p * 6)) & 0xFFFFu)
                    : (((bits0 >> 18) | (bits1 << 14)) & 0xFFFFu);
#else
                const unsigned state = (bits0 >> (p * HYB_B)) & 0xFFFFu;
#endif
                hyb_decode_pair_lut(smem_lut, reg_lut, state, lane, w_dec[2 * p], w_dec[2 * p + 1]);
            }

            const int col_off = kk * HYB_TILE + col_in_tile;
#pragma unroll
            for (int t = 0; t < HYB_VEC_K; ++t) {
                const float* xv = x + (size_t)t * cols + col_off;
                const float4 xa = __ldg(reinterpret_cast<const float4*>(xv));
                const float4 xb = __ldg(reinterpret_cast<const float4*>(xv + 4));
                acc[t] = fmaf(w_dec[0], xa.x, acc[t]);
                acc[t] = fmaf(w_dec[1], xa.y, acc[t]);
                acc[t] = fmaf(w_dec[2], xa.z, acc[t]);
                acc[t] = fmaf(w_dec[3], xa.w, acc[t]);
                acc[t] = fmaf(w_dec[4], xb.x, acc[t]);
                acc[t] = fmaf(w_dec[5], xb.y, acc[t]);
                acc[t] = fmaf(w_dec[6], xb.z, acc[t]);
                acc[t] = fmaf(w_dec[7], xb.w, acc[t]);
            }
        }
    }

    const int row = group * HYB_TILE + row_in_tile;
#pragma unroll
    for (int t = 0; t < HYB_VEC_K; ++t) {
        acc[t] += __shfl_xor_sync(0xffffffffu, acc[t], 1);
        if ((lane & 1) == 0) {
            atomicAdd(&y[(size_t)t * rows + row], acc[t] * scale);
        }
    }
}

// Tensor-Core multi-vector verification kernel (k = HYB_VEC_K, multiple of 8: 8 or 16).
// Assumes the 128 pairs of each 16x16 tile are pre-swizzled offline into exact
// mma.sync.aligned.m16n8k16 A-fragment register order:
//   lane l = (groupID << 2) | tid_in_group:
//     pair 0 -> row groupID,     cols 2*tid..2*tid+1   (a0)
//     pair 1 -> row groupID + 8, cols 2*tid..2*tid+1   (a1)
//     pair 2 -> row groupID,     cols 2*tid+8..2*tid+9 (a2)
//     pair 3 -> row groupID + 8, cols 2*tid+8..2*tid+9 (a3)
// Zero SMEM staging, zero cvt.f32.f16, zero transpose: decoded half2 registers
// feed mma.sync.aligned.m16n8k16 directly!
// Input x_h2: [k, cols/2] as packed half2 (unsigned).
// Output y:   [k, rows] FP32.
extern "C" __global__ void __launch_bounds__(512)
hyb_matvec_mma(const unsigned* __restrict__ tiles,
               const unsigned* __restrict__ lut_global,
               const unsigned* __restrict__ x_h2,
               float* __restrict__ y,
               int rows,
               int cols,
               int tiles_per_split,
               float scale) {
#if HYB_Q == 5
    const unsigned* smem_lut = nullptr;
    const unsigned reg_lut = __ldg(lut_global + (threadIdx.x & 31));
#else
    extern __shared__ unsigned smem_lut[];
    const unsigned reg_lut = 0u;
    for (int i = threadIdx.x; i < HYB_LUT_SIZE * 32; i += 512) {
        smem_lut[i] = __ldg(lut_global + (i >> 5));
    }
    __syncthreads();
#endif

    const int col_tiles = cols / HYB_TILE;
    const int cols_h2 = cols / 2;
    const int k0 = blockIdx.y * tiles_per_split;
    const int k1 = min(k0 + tiles_per_split, col_tiles);
    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x & 31;
    const int group = blockIdx.x * HYB_FUSED_WARPS + warp;
    const int group_id = lane >> 2;      // 0..7
    const int tid_in_group = lane & 3;   // 0..3

    const int bit = lane * 4 * HYB_B;
    const int w0_idx = (bit / 32) % HYB_WORDS;
    const int w1_idx = (w0_idx + 1) % HYB_WORDS;
#if HYB_B == 6
    const int w2_idx = (w0_idx + 2) % HYB_WORDS;
#endif
    const int shift = bit & 31;

    const unsigned* base = tiles + (size_t)group * col_tiles * HYB_WORDS;
    constexpr int N_CHUNKS = HYB_VEC_K / 8;
    float d0[N_CHUNKS], d1[N_CHUNKS], d2[N_CHUNKS], d3[N_CHUNKS];
#pragma unroll
    for (int nc = 0; nc < N_CHUNKS; ++nc) {
        d0[nc] = 0.0f;
        d1[nc] = 0.0f;
        d2[nc] = 0.0f;
        d3[nc] = 0.0f;
    }

    for (int k = k0; k < k1; k += HYB_UNROLL) {
        unsigned words[HYB_UNROLL];
#pragma unroll
        for (int u = 0; u < HYB_UNROLL; ++u) {
            const int kk = k + u;
            words[u] = (lane < HYB_WORDS && kk < k1)
                           ? __ldg(base + (size_t)kk * HYB_WORDS + lane)
                           : 0u;
        }
#pragma unroll
        for (int u = 0; u < HYB_UNROLL; ++u) {
            const int kk = k + u;
            const unsigned r0 = __shfl_sync(0xffffffffu, words[u], w0_idx);
            const unsigned r1 = __shfl_sync(0xffffffffu, words[u], w1_idx);
#if HYB_B == 6
            const unsigned r2 = __shfl_sync(0xffffffffu, words[u], w2_idx);
#endif
            if (kk >= k1) {
                break;
            }
            const unsigned bits0 = __funnelshift_r(r0, r1, shift);
#if HYB_B == 6
            const unsigned bits1 = __funnelshift_r(r1, r2, shift);
#endif
            unsigned a[4];
#pragma unroll
            for (int p = 0; p < 4; ++p) {
#if HYB_B == 6
                const unsigned state = (p < 3)
                    ? ((bits0 >> (p * 6)) & 0xFFFFu)
                    : (((bits0 >> 18) | (bits1 << 14)) & 0xFFFFu);
#else
                const unsigned state = (bits0 >> (p * HYB_B)) & 0xFFFFu;
#endif
                const unsigned h = state * HYB_HASH_MULT;
#if HYB_Q == 5
                a[p] = __shfl_sync(0xffffffffu, reg_lut, (h >> 5) & 31u) ^ (h & 0x80008000u);
#else
                a[p] = smem_lut[(h & HYB_LUT_MASK) | (unsigned)lane] ^ (h & 0x80008000u);
#endif
            }

            const int col_h2_base = kk * 8 + tid_in_group;
#pragma unroll
            for (int nc = 0; nc < N_CHUNKS; ++nc) {
                const int tok = nc * 8 + group_id;
                const unsigned* xtok = x_h2 + (size_t)tok * cols_h2 + col_h2_base;
                const unsigned b0 = __ldg(xtok);
                const unsigned b1 = __ldg(xtok + 4);
                asm volatile(
                    "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                    "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                    : "+f"(d0[nc]), "+f"(d1[nc]), "+f"(d2[nc]), "+f"(d3[nc])
                    : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
            }
        }
    }

    const int row0 = group * HYB_TILE + group_id;
    const int row1 = row0 + 8;
#pragma unroll
    for (int nc = 0; nc < N_CHUNKS; ++nc) {
        const int tok0 = nc * 8 + 2 * tid_in_group;
        const int tok1 = tok0 + 1;
        atomicAdd(&y[(size_t)tok0 * rows + row0], d0[nc] * scale);
        atomicAdd(&y[(size_t)tok1 * rows + row0], d1[nc] * scale);
        atomicAdd(&y[(size_t)tok0 * rows + row1], d2[nc] * scale);
        atomicAdd(&y[(size_t)tok1 * rows + row1], d3[nc] * scale);
    }
}

// Fused small-batch (2 <= tokens <= 16) HYB Q=5 Tensor-Core verification block:
// Zero-SMEM warp-register LUT (Q=5), MMA-swizzled circular tiles (48/64/80/96B),
// optional offline-folded RHT (HYB_SKIP_IN_RHT / HYB_SKIP_OUT_RHT), and optional
// fused SwiGLU. 1-2 mma.sync.aligned.m16n8k16 instructions compute up to 16 tokens
// directly from decoded half2 LUT registers with ZERO weight cvt!
template <bool SwiGLU>
__device__ __forceinline__ void hyb_small_k_block(
    const unsigned* __restrict__ tiles, const unsigned* __restrict__ lut_global,
    const float* __restrict__ x, const float* __restrict__ x_up,
    const float* __restrict__ col_signs, const float* __restrict__ row_signs,
    float* __restrict__ y, float* __restrict__ partials, int rows, int total_rows, int cols,
    int tokens, int chunks_per_split, float scale, int row_block, int global_row_block,
    int split) {
    __shared__ unsigned xs_h2[HYB_MAX_K][HYB_CHUNK / 2]; // 4 KB (K=8) or 8 KB (K=16)
    __shared__ float partial[HYB_MAX_K][HYB_CHUNK];      // 8 KB (K=8) or 16 KB (K=16)

    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x & 31;
    const int group_id = lane >> 2;
    const int tid = lane & 3;
    const unsigned reg_lut = __ldg(lut_global + lane);

    const int col_tiles = cols / HYB_TILE;
    const int total_chunks = cols / HYB_CHUNK;
    const int ch0 = split * chunks_per_split;
    const int ch1 = min(ch0 + chunks_per_split, total_chunks);

    const int group = row_block * HYB_FUSED_WARPS + warp;
    const int bit = lane * 4 * HYB_B;
    const int w0_idx = (bit / 32) % HYB_WORDS;
    const int w1_idx = (w0_idx + 1) % HYB_WORDS;
#if HYB_B == 6
    const int w2_idx = (w0_idx + 2) % HYB_WORDS;
#endif
    const int shift = bit & 31;

    const unsigned* base = tiles + (size_t)group * col_tiles * HYB_WORDS;
    float d0 = 0.0f, d1 = 0.0f, d2 = 0.0f, d3 = 0.0f;
#if HYB_MAX_K > 8
    float e0 = 0.0f, e1 = 0.0f, e2 = 0.0f, e3 = 0.0f;
    const bool has_hi = tokens > 8;
#endif

    for (int ch = ch0; ch < ch1; ++ch) {
        const int c = ch * HYB_CHUNK;
        if (warp < tokens) {
            float v[8];
            const float* xt = x + (size_t)warp * cols + c;
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                v[e] = __ldg(xt + lane + 32 * e);
            }
            if (SwiGLU) {
                const float* ut = x_up + (size_t)warp * cols + c;
                float up[8];
#pragma unroll
                for (int e = 0; e < 8; ++e) {
                    up[e] = __ldg(ut + lane + 32 * e);
                }
#pragma unroll
                for (int e = 0; e < 8; ++e) {
                    v[e] = v[e] / (1.0f + expf(-v[e])) * up[e];
                }
            }
#if !HYB_SKIP_IN_RHT
            float sign[8];
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                sign[e] = __ldg(col_signs + c + lane + 32 * e);
            }
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                v[e] *= sign[e];
            }
            hyb_fwht256_warp(v);
#endif
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                partial[warp][lane + 32 * e] = v[e];
            }
            __syncwarp();
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                const int j = lane + 32 * e;
                const float x0 = partial[warp][2 * j];
                const float x1 = partial[warp][2 * j + 1];
                unsigned h2;
                asm("cvt.rn.f16x2.f32 %0, %2, %1;" : "=r"(h2) : "f"(x0), "f"(x1));
                xs_h2[warp][j] = h2;
            }
        }
        __syncthreads();

        const int k0 = ch * 16;
        for (int tk = 0; tk < 16; tk += HYB_UNROLL) {
            unsigned words[HYB_UNROLL];
#pragma unroll
            for (int u = 0; u < HYB_UNROLL; ++u) {
                const int kk = k0 + tk + u;
                words[u] = (lane < HYB_WORDS) ? __ldg(base + (size_t)kk * HYB_WORDS + lane) : 0u;
            }
#pragma unroll
            for (int u = 0; u < HYB_UNROLL; ++u) {
                const unsigned r0 = __shfl_sync(0xffffffffu, words[u], w0_idx);
                const unsigned r1 = __shfl_sync(0xffffffffu, words[u], w1_idx);
#if HYB_B == 6
                const unsigned r2 = __shfl_sync(0xffffffffu, words[u], w2_idx);
#endif
                const unsigned bits0 = __funnelshift_r(r0, r1, shift);
#if HYB_B == 6
                const unsigned bits1 = __funnelshift_r(r1, r2, shift);
#endif
                unsigned a[4];
#pragma unroll
                for (int p = 0; p < 4; ++p) {
#if HYB_B == 6
                    const unsigned state = (p < 3)
                        ? ((bits0 >> (p * 6)) & HYB_STATE_MASK)
                        : (((bits0 >> 18) | (bits1 << 14)) & HYB_STATE_MASK);
#else
                    const unsigned state = (bits0 >> (p * HYB_B)) & HYB_STATE_MASK;
#endif
                    const unsigned h = state * HYB_HASH_MULT;
                    a[p] = __shfl_sync(0xffffffffu, reg_lut, (h >> 5) & 31u) ^ (h & 0x80008000u);
                }

                const int xoff = (tk + u) * 8 + tid;
                const unsigned b0 = (group_id < tokens) ? xs_h2[group_id][xoff] : 0u;
                const unsigned b1 = (group_id < tokens) ? xs_h2[group_id][xoff + 4] : 0u;
                asm volatile(
                    "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                    "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                    : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
                    : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
#if HYB_MAX_K > 8
                if (has_hi) {
                    const int g_hi = 8 + group_id;
                    const unsigned b2 = (g_hi < tokens) ? xs_h2[g_hi][xoff] : 0u;
                    const unsigned b3 = (g_hi < tokens) ? xs_h2[g_hi][xoff + 4] : 0u;
                    asm volatile(
                        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(e0), "+f"(e1), "+f"(e2), "+f"(e3)
                        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b2), "r"(b3));
                }
#endif
            }
        }
        __syncthreads();
    }

    const int tok0 = 2 * tid;
    const int tok1 = tok0 + 1;
    if (tok0 < tokens) {
        partial[tok0][warp * HYB_TILE + group_id] = d0;
        partial[tok0][warp * HYB_TILE + group_id + 8] = d2;
    }
    if (tok1 < tokens) {
        partial[tok1][warp * HYB_TILE + group_id] = d1;
        partial[tok1][warp * HYB_TILE + group_id + 8] = d3;
    }
#if HYB_MAX_K > 8
    if (has_hi) {
        if (8 + tok0 < tokens) {
            partial[8 + tok0][warp * HYB_TILE + group_id] = e0;
            partial[8 + tok0][warp * HYB_TILE + group_id + 8] = e2;
        }
        if (8 + tok1 < tokens) {
            partial[8 + tok1][warp * HYB_TILE + group_id] = e1;
            partial[8 + tok1][warp * HYB_TILE + group_id + 8] = e3;
        }
    }
#endif
    __syncthreads();

    if (warp < tokens) {
        float v[8];
#pragma unroll
        for (int e = 0; e < 8; ++e) {
            v[e] = partial[warp][lane + 32 * e];
        }
#if !HYB_SKIP_OUT_RHT
        hyb_fwht256_warp(v);
#endif
        if (gridDim.y == 1) {
            float* yt = y + (size_t)warp * rows;
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                const int row = row_block * HYB_CHUNK + lane + 32 * e;
#if !HYB_SKIP_OUT_RHT
                yt[row] = v[e] * __ldg(row_signs + row) * scale;
#else
                yt[row] = v[e] * scale;
#endif
            }
        } else if (partials != nullptr) {
            float* pt = partials + ((size_t)split * tokens + warp) * total_rows
                + global_row_block * HYB_CHUNK;
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                const int local_r = lane + 32 * e;
#if !HYB_SKIP_OUT_RHT
                const int row = row_block * HYB_CHUNK + local_r;
                pt[local_r] = v[e] * __ldg(row_signs + row) * scale;
#else
                pt[local_r] = v[e] * scale;
#endif
            }
        } else {
            float* yt = y + (size_t)warp * rows;
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                const int row = row_block * HYB_CHUNK + lane + 32 * e;
#if !HYB_SKIP_OUT_RHT
                atomicAdd(&yt[row], v[e] * __ldg(row_signs + row) * scale);
#else
                atomicAdd(&yt[row], v[e] * scale);
#endif
            }
        }
    }
}

#define HYB_SMALL_K_MULTI_PARAMS                                                            \
    const unsigned* __restrict__ t0, const unsigned* __restrict__ t1,                       \
        const unsigned* __restrict__ t2, const unsigned* __restrict__ l0,                   \
        const unsigned* __restrict__ l1, const unsigned* __restrict__ l2,                   \
        const float* __restrict__ c0, const float* __restrict__ c1,                         \
        const float* __restrict__ c2, const float* __restrict__ r0,                         \
        const float* __restrict__ r1, const float* __restrict__ r2, float* __restrict__ y0, \
        float* __restrict__ y1, float* __restrict__ y2, float s0, float s1, float s2,       \
        int rows0, int rows1, int rows2, int rb0, int rb1, const float* __restrict__ x,     \
        const float* __restrict__ x_up, float* __restrict__ z0, float* __restrict__ z1,     \
        float* __restrict__ z2, int n0, int n1, int n2, int cols, int tokens,               \
        int chunks_per_split, float* __restrict__ partials

template <bool SwiGLU>
__device__ __forceinline__ void hyb_small_k_multi_body(HYB_SMALL_K_MULTI_PARAMS) {
    // Zero the next projection's output buffers in-kernel when using legacy atomicAdd path.
    if (partials == nullptr && (n0 > 0 || n1 > 0 || n2 > 0)) {
        const int tid_global = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
        const int stride_global = gridDim.x * gridDim.y * blockDim.x;
        if (n0 > 0) {
            for (int i = tid_global; i < n0; i += stride_global) z0[i] = 0.0f;
        }
        if (n1 > 0) {
            for (int i = tid_global; i < n1; i += stride_global) z1[i] = 0.0f;
        }
        if (n2 > 0) {
            for (int i = tid_global; i < n2; i += stride_global) z2[i] = 0.0f;
        }
    }

    int row_block = blockIdx.x;
    const unsigned *tiles = t0, *lut = l0;
    const float *col_signs = c0, *row_signs = r0;
    float* y = y0;
    float scale = s0;
    int rows = rows0;
    if (row_block >= rb0) {
        row_block -= rb0;
        tiles = t1, lut = l1, col_signs = c1, row_signs = r1, y = y1, scale = s1, rows = rows1;
        if (row_block >= rb1) {
            row_block -= rb1;
            tiles = t2, lut = l2, col_signs = c2, row_signs = r2, y = y2, scale = s2, rows = rows2;
        }
    }
    const int total_rows = gridDim.x * HYB_CHUNK;
    hyb_small_k_block<SwiGLU>(tiles, lut, x, SwiGLU ? x_up : nullptr, col_signs, row_signs, y,
                              partials, rows, total_rows, cols, tokens, chunks_per_split, scale,
                              row_block, blockIdx.x, blockIdx.y);
}

#define HYB_SMALL_K_MULTI_ARGS                                                              \
    t0, t1, t2, l0, l1, l2, c0, c1, c2, r0, r1, r2, y0, y1, y2, s0, s1, s2, rows0, rows1,   \
        rows2, rb0, rb1, x, x_up, z0, z1, z2, n0, n1, n2, cols, tokens, chunks_per_split,   \
        partials

extern "C" __global__ void __launch_bounds__(512)
hyb_small_k_multi(HYB_SMALL_K_MULTI_PARAMS) {
    hyb_small_k_multi_body<false>(HYB_SMALL_K_MULTI_ARGS);
}

extern "C" __global__ void __launch_bounds__(512)
hyb_small_k_multi_swiglu(HYB_SMALL_K_MULTI_PARAMS) {
    hyb_small_k_multi_body<true>(HYB_SMALL_K_MULTI_ARGS);
}

// Deterministic split-K reduction for both dense (hyb_small_k_multi) and MoE (hyb_moe_small_k).
// Grid: (ceil(total_rows / 128), tokens, 1), Block: 32 threads (1 warp/block, float4 vectorized).
// Spreading across 8x more SMs than 256-thread blocks avoids SM underutilization on small total_rows.
extern "C" __global__ void hyb_splitk_reduce_multi(const float* __restrict__ partials,
                                                   float* __restrict__ y0,
                                                   float* __restrict__ y1,
                                                   float* __restrict__ y2,
                                                   int rows0,
                                                   int rows1,
                                                   int rows2,
                                                   int tokens,
                                                   int splits) {
    const int total_rows4 = (rows0 + rows1 + rows2) >> 2;
    const int g4 = blockIdx.x * 32 + threadIdx.x;
    const int t = blockIdx.y;
    if (g4 >= total_rows4 || t >= tokens) {
        return;
    }
    const size_t split_stride4 = (size_t)tokens * total_rows4;
    const float4* base =
        reinterpret_cast<const float4*>(partials) + (size_t)t * total_rows4 + g4;
    float4 acc = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    for (int s = 0; s < splits; ++s) {
        const float4 v = __ldg(base + (size_t)s * split_stride4);
        acc.x += v.x;
        acc.y += v.y;
        acc.z += v.z;
        acc.w += v.w;
    }
    const int r0_4 = rows0 >> 2;
    const int r1_4 = rows1 >> 2;
    if (g4 < r0_4) {
        reinterpret_cast<float4*>(y0)[(size_t)t * r0_4 + g4] = acc;
    } else if (g4 < r0_4 + r1_4) {
        reinterpret_cast<float4*>(y1)[(size_t)t * r1_4 + (g4 - r0_4)] = acc;
    } else {
        const int r2_4 = rows2 >> 2;
        reinterpret_cast<float4*>(y2)[(size_t)t * r2_4 + (g4 - r0_4 - r1_4)] = acc;
    }
}

// Once-per-step residual-stream R_1 rotation enter: x <- (H_256 * S_res / 16) * x.
// Grid: (cols / 256, tokens, 1), Block: 32 threads (1 warp per 256-chunk).
extern "C" __global__ void hyb_r1_enter(float* __restrict__ x,
                                        const float* __restrict__ r1_signs,
                                        int cols) {
    const int lane = threadIdx.x & 31;
    const int c = blockIdx.x * HYB_CHUNK;
    float* xt = x + (size_t)blockIdx.y * cols + c;
    float v[8];
#pragma unroll
    for (int e = 0; e < 8; ++e) {
        const int idx = lane + 32 * e;
        v[e] = xt[idx] * __ldg(r1_signs + c + idx);
    }
    hyb_fwht256_warp(v);
#pragma unroll
    for (int e = 0; e < 8; ++e) {
        xt[lane + 32 * e] = v[e] * (1.0f / 16.0f);
    }
}

// Once-per-step residual-stream R_1 rotation exit: x <- S_res * (H_256 / 16) * x.
// Grid: (cols / 256, tokens, 1), Block: 32 threads (1 warp per 256-chunk).
extern "C" __global__ void hyb_r1_exit(float* __restrict__ x,
                                       const float* __restrict__ r1_signs,
                                       int cols) {
    const int lane = threadIdx.x & 31;
    const int c = blockIdx.x * HYB_CHUNK;
    float* xt = x + (size_t)blockIdx.y * cols + c;
    float v[8];
#pragma unroll
    for (int e = 0; e < 8; ++e) {
        v[e] = xt[lane + 32 * e];
    }
    hyb_fwht256_warp(v);
#pragma unroll
    for (int e = 0; e < 8; ++e) {
        const int idx = lane + 32 * e;
        xt[idx] = v[e] * __ldg(r1_signs + c + idx) * (1.0f / 16.0f);
    }
}

// Group MoE assignments [0..total_assigns) by expert_id into sub-groups of size 1..=8.
// Single warp (32 threads), zero __syncthreads(), handles up to 256 assignments in < 1 us.
extern "C" __global__ void hyb_moe_group_assignments(const int* __restrict__ ids,
                                                     int total_assigns,
                                                     int* __restrict__ group_experts,
                                                     int* __restrict__ group_offsets,
                                                     int* __restrict__ group_assigns,
                                                     int* __restrict__ num_groups_out) {
    const int lane = threadIdx.x & 31;
    int my_exp[8];
    unsigned rem_mask[8];
    const int n_chunks = (total_assigns + 31) >> 5;
#pragma unroll
    for (int c = 0; c < 8; ++c) {
        const int idx = (c << 5) + lane;
        const bool valid = idx < total_assigns;
        my_exp[c] = valid ? __ldg(ids + idx) : -1;
        rem_mask[c] = __ballot_sync(0xffffffffu, valid);
    }

    int num_groups = 0;
    int out_pos = 0;
    if (lane == 0) {
        group_offsets[0] = 0;
    }

    for (int c0 = 0; c0 < n_chunks; ++c0) {
        while (rem_mask[c0] != 0u) {
            const int first_lane = __ffs(rem_mask[c0]) - 1;
            const int target_exp = __shfl_sync(0xffffffffu, my_exp[c0], first_lane);

            int g_count = 0;
            for (int c = c0; c < n_chunks; ++c) {
                const bool match = ((rem_mask[c] >> lane) & 1u) && (my_exp[c] == target_exp);
                unsigned m = __ballot_sync(0xffffffffu, match);
                rem_mask[c] &= ~m;
                while (m != 0u) {
                    const int bit = __ffs(m) - 1;
                    m &= m - 1u;
                    if (lane == 0) {
                        group_assigns[out_pos] = (c << 5) + bit;
                    }
                    ++out_pos;
                    ++g_count;
                    if (g_count == 8) {
                        if (lane == 0) {
                            group_experts[num_groups] = target_exp;
                            group_offsets[num_groups + 1] = out_pos;
                        }
                        ++num_groups;
                        g_count = 0;
                    }
                }
            }
            if (g_count > 0) {
                if (lane == 0) {
                    group_experts[num_groups] = target_exp;
                    group_offsets[num_groups + 1] = out_pos;
                }
                ++num_groups;
            }
        }
    }

    if (lane == 0) {
        num_groups_out[0] = num_groups;
    }
}

// MoE PLT1 HYB Q=5 Tensor-Core kernel for 3D expert tensors ([n_experts, rows, cols], MMA tiles).
// Grid: (total_row_blocks, max_groups, splits), Block: 512 threads (16 warps = 256 rows).
// Reads each active expert's weights ONCE per group of up to 8 assignments, decodes via
// zero-SMEM warp-register LUT (Q=5), and executes mma.sync.aligned.m16n8k16 directly!
// Direct store when splits == 1 (zero atomics, zero memset); atomicAdd when splits > 1.
template <bool SwiGLU>
__device__ __forceinline__ void hyb_moe_block(const unsigned* __restrict__ tiles,
                                              const unsigned* __restrict__ lut_global,
                                              const float* __restrict__ scales,
                                              const float* __restrict__ col_signs,
                                              const float* __restrict__ row_signs,
                                              const float* __restrict__ x,
                                              const float* __restrict__ x_up,
                                              const int* __restrict__ group_experts,
                                              const int* __restrict__ group_offsets,
                                              const int* __restrict__ group_assigns,
                                              const int* __restrict__ num_groups_ptr,
                                              float* __restrict__ y,
                                              float* __restrict__ partials,
                                              int rows,
                                              int total_rows,
                                              int total_assigns,
                                              int cols,
                                              int used,
                                              int chunks_per_split,
                                              int splits,
                                              int row_block,
                                              int global_row_block,
                                              int group_idx,
                                              int split) {
    const int num_groups = __ldg(num_groups_ptr);
    if (group_idx >= num_groups) {
        return;
    }

    const int total_chunks = cols / HYB_CHUNK;
    const int ch0 = split * chunks_per_split;
    const int ch1 = min(ch0 + chunks_per_split, total_chunks);
    if (ch0 >= ch1) {
        return;
    }

    __shared__ unsigned xs_h2[8][HYB_CHUNK / 2]; // 4 KB
    __shared__ float partial[8][HYB_CHUNK];      // 8 KB

    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x & 31;
    const int group_id = lane >> 2;  // 0..7
    const int tid = lane & 3;        // 0..3
    const unsigned reg_lut = __ldg(lut_global + lane);

    const int expert = __ldg(group_experts + group_idx);
    const int g_start = __ldg(group_offsets + group_idx);
    const int g_end = __ldg(group_offsets + group_idx + 1);
    const int g_len = g_end - g_start;
    const float scale = __ldg(scales + expert);

    const int my_assign = (warp < g_len) ? __ldg(group_assigns + g_start + warp) : 0;
    const int my_x_tok = SwiGLU ? my_assign : (my_assign / used);

    const int col_tiles = cols / HYB_TILE;
    const size_t expert_stride = (size_t)(rows / HYB_TILE) * col_tiles * HYB_WORDS;
    const int group = row_block * HYB_FUSED_WARPS + warp;
    const unsigned* base =
        tiles + (size_t)expert * expert_stride + (size_t)group * col_tiles * HYB_WORDS;

    const int bit = lane * 4 * HYB_B;
    const int w0_idx = (bit / 32) % HYB_WORDS;
    const int w1_idx = (w0_idx + 1) % HYB_WORDS;
#if HYB_B == 6
    const int w2_idx = (w0_idx + 2) % HYB_WORDS;
#endif
    const int shift = bit & 31;

    float d0 = 0.0f, d1 = 0.0f, d2 = 0.0f, d3 = 0.0f;

    for (int ch = ch0; ch < ch1; ++ch) {
        const int c = ch * HYB_CHUNK;
        if (warp < g_len) {
            float v[8];
            const float* xt = x + (size_t)my_x_tok * cols + c;
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                v[e] = __ldg(xt + lane + 32 * e);
            }
            if (SwiGLU) {
                const float* ut = x_up + (size_t)my_x_tok * cols + c;
                float up[8];
#pragma unroll
                for (int e = 0; e < 8; ++e) {
                    up[e] = __ldg(ut + lane + 32 * e);
                }
#pragma unroll
                for (int e = 0; e < 8; ++e) {
                    v[e] = v[e] / (1.0f + expf(-v[e])) * up[e];
                }
            }
#if !HYB_SKIP_IN_RHT
            float sign[8];
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                sign[e] = __ldg(col_signs + c + lane + 32 * e);
            }
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                v[e] *= sign[e];
            }
            hyb_fwht256_warp(v);
#endif
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                partial[warp][lane + 32 * e] = v[e];
            }
            __syncwarp();
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                const int j = lane + 32 * e;
                const float x0 = partial[warp][2 * j];
                const float x1 = partial[warp][2 * j + 1];
                unsigned h2;
                asm("cvt.rn.f16x2.f32 %0, %2, %1;" : "=r"(h2) : "f"(x0), "f"(x1));
                xs_h2[warp][j] = h2;
            }
        }
        __syncthreads();

        const int k0 = ch * 16;
        for (int tk = 0; tk < 16; tk += HYB_UNROLL) {
            unsigned words[HYB_UNROLL];
#pragma unroll
            for (int u = 0; u < HYB_UNROLL; ++u) {
                const int kk = k0 + tk + u;
                words[u] = (lane < HYB_WORDS) ? __ldg(base + (size_t)kk * HYB_WORDS + lane) : 0u;
            }
#pragma unroll
            for (int u = 0; u < HYB_UNROLL; ++u) {
                const unsigned r0 = __shfl_sync(0xffffffffu, words[u], w0_idx);
                const unsigned r1 = __shfl_sync(0xffffffffu, words[u], w1_idx);
#if HYB_B == 6
                const unsigned r2 = __shfl_sync(0xffffffffu, words[u], w2_idx);
#endif
                const unsigned bits0 = __funnelshift_r(r0, r1, shift);
#if HYB_B == 6
                const unsigned bits1 = __funnelshift_r(r1, r2, shift);
#endif
                unsigned a[4];
#pragma unroll
                for (int p = 0; p < 4; ++p) {
#if HYB_B == 6
                    const unsigned state = (p < 3)
                        ? ((bits0 >> (p * 6)) & HYB_STATE_MASK)
                        : (((bits0 >> 18) | (bits1 << 14)) & HYB_STATE_MASK);
#else
                    const unsigned state = (bits0 >> (p * HYB_B)) & HYB_STATE_MASK;
#endif
                    const unsigned h = state * HYB_HASH_MULT;
                    a[p] = __shfl_sync(0xffffffffu, reg_lut, (h >> 5) & 31u) ^ (h & 0x80008000u);
                }

                const int xoff = (tk + u) * 8 + tid;
                const unsigned b0 = (group_id < g_len) ? xs_h2[group_id][xoff] : 0u;
                const unsigned b1 = (group_id < g_len) ? xs_h2[group_id][xoff + 4] : 0u;
                asm volatile(
                    "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                    "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                    : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
                    : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
            }
        }
        __syncthreads();
    }

    const int tok0 = 2 * tid;
    const int tok1 = tok0 + 1;
    if (tok0 < g_len) {
        partial[tok0][warp * HYB_TILE + group_id] = d0;
        partial[tok0][warp * HYB_TILE + group_id + 8] = d2;
    }
    if (tok1 < g_len) {
        partial[tok1][warp * HYB_TILE + group_id] = d1;
        partial[tok1][warp * HYB_TILE + group_id + 8] = d3;
    }
    __syncthreads();

    if (warp < g_len) {
        float v[8];
#pragma unroll
        for (int e = 0; e < 8; ++e) {
            v[e] = partial[warp][lane + 32 * e];
        }
#if !HYB_SKIP_OUT_RHT
        hyb_fwht256_warp(v);
#endif
        if (splits == 1) {
            float* yt = y + (size_t)my_assign * rows;
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                const int row = row_block * HYB_CHUNK + lane + 32 * e;
#if !HYB_SKIP_OUT_RHT
                yt[row] = v[e] * __ldg(row_signs + row) * scale;
#else
                yt[row] = v[e] * scale;
#endif
            }
        } else if (partials != nullptr) {
            float* pt = partials + ((size_t)split * total_assigns + my_assign) * total_rows
                + global_row_block * HYB_CHUNK;
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                const int local_r = lane + 32 * e;
#if !HYB_SKIP_OUT_RHT
                const int row = row_block * HYB_CHUNK + local_r;
                pt[local_r] = v[e] * __ldg(row_signs + row) * scale;
#else
                pt[local_r] = v[e] * scale;
#endif
            }
        } else {
            float* yt = y + (size_t)my_assign * rows;
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                const int row = row_block * HYB_CHUNK + lane + 32 * e;
#if !HYB_SKIP_OUT_RHT
                atomicAdd(&yt[row], v[e] * __ldg(row_signs + row) * scale);
#else
                atomicAdd(&yt[row], v[e] * scale);
#endif
            }
        }
    }
}

extern "C" __global__ void __launch_bounds__(512)
hyb_moe_small_k(const unsigned* __restrict__ t0,
                const unsigned* __restrict__ t1,
                const unsigned* __restrict__ l0,
                const unsigned* __restrict__ l1,
                const float* __restrict__ sc0,
                const float* __restrict__ sc1,
                const float* __restrict__ c0,
                const float* __restrict__ c1,
                const float* __restrict__ r0,
                const float* __restrict__ r1,
                float* __restrict__ y0,
                float* __restrict__ y1,
                int rows0,
                int rows1,
                int rb0,
                const float* __restrict__ x,
                const int* __restrict__ group_experts,
                const int* __restrict__ group_offsets,
                const int* __restrict__ group_assigns,
                const int* __restrict__ num_groups_ptr,
                int cols,
                int used,
                int chunks_per_split,
                int splits,
                float* __restrict__ partials,
                int total_assigns) {
    int rb = blockIdx.x;
    const int total_rows = rows0 + rows1;
    if (rb < rb0) {
        hyb_moe_block<false>(t0, l0, sc0, c0, r0, x, nullptr, group_experts, group_offsets,
                             group_assigns, num_groups_ptr, y0, partials, rows0, total_rows,
                             total_assigns, cols, used, chunks_per_split, splits, rb, rb,
                             blockIdx.y, blockIdx.z);
    } else {
        hyb_moe_block<false>(t1, l1, sc1, c1, r1, x, nullptr, group_experts, group_offsets,
                             group_assigns, num_groups_ptr, y1, partials, rows1, total_rows,
                             total_assigns, cols, used, chunks_per_split, splits, rb - rb0, rb,
                             blockIdx.y, blockIdx.z);
    }
}

extern "C" __global__ void __launch_bounds__(512)
hyb_moe_small_k_swiglu(const unsigned* __restrict__ t0,
                       const unsigned* __restrict__ l0,
                       const float* __restrict__ sc0,
                       const float* __restrict__ c0,
                       const float* __restrict__ r0,
                       float* __restrict__ y0,
                       int rows0,
                       const float* __restrict__ x_gate,
                       const float* __restrict__ x_up,
                       const int* __restrict__ group_experts,
                       const int* __restrict__ group_offsets,
                       const int* __restrict__ group_assigns,
                       const int* __restrict__ num_groups_ptr,
                       int cols,
                       int used,
                       int chunks_per_split,
                       int splits,
                       float* __restrict__ partials,
                       int total_assigns) {
    hyb_moe_block<true>(t0, l0, sc0, c0, r0, x_gate, x_up, group_experts, group_offsets,
                        group_assigns, num_groups_ptr, y0, partials, rows0, rows0, total_assigns,
                        cols, used, chunks_per_split, splits, blockIdx.x, blockIdx.x, blockIdx.y,
                        blockIdx.z);
}

// ============================================================================
// Dedicated B=1 (single-token pure decode) PLT1 HYB Q=5 kernels:
// - Parallel 16-warp activation preload into xs_h2[16][128] using register
//   __shfl_xor_sync(..., 1) for f16x2 packing (zero SMEM partial round-trip).
// - Only ONE __syncthreads() per batch of up to 16 chunks (zero syncs inside
//   the weight loop, eliminating 2*N_chunks serial warp-0 stalls).
// - MoE B=1 reads expert = ids[blockIdx.y] directly, completely eliminating
//   hyb_moe_group_assignments (48 launches/token on Qwen3-30B-A3B).
// - Bit-for-bit identical mma.m16n8k16 + split-K accumulation order vs B=2..16.
// ============================================================================

#define HYB_B1_CHUNKS_PER_BATCH 16

template <bool SwiGLU>
__device__ __forceinline__ void hyb_b1_preload_chunks(const float* __restrict__ x,
                                                      const float* __restrict__ x_up,
                                                      const float* __restrict__ col_signs,
                                                      unsigned (&xs_h2)[HYB_B1_CHUNKS_PER_BATCH][HYB_CHUNK / 2],
                                                      int ch_base,
                                                      int n_ch,
                                                      int warp,
                                                      int lane) {
    if (warp < n_ch) {
        const int c = (ch_base + warp) * HYB_CHUNK;
        float v[8];
#pragma unroll
        for (int e = 0; e < 8; ++e) {
            v[e] = __ldg(x + c + lane + 32 * e);
        }
        if (SwiGLU) {
            float up[8];
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                up[e] = __ldg(x_up + c + lane + 32 * e);
            }
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                v[e] = v[e] / (1.0f + expf(-v[e])) * up[e];
            }
        }
#if !HYB_SKIP_IN_RHT
        float sign[8];
#pragma unroll
        for (int e = 0; e < 8; ++e) {
            sign[e] = __ldg(col_signs + c + lane + 32 * e);
        }
#pragma unroll
        for (int e = 0; e < 8; ++e) {
            v[e] *= sign[e];
        }
        hyb_fwht256_warp(v);
#endif
        // Pair adjacent elements (2*k, 2*k+1) via warp shuffle and pack f16x2 directly to SMEM.
#pragma unroll
        for (int e = 0; e < 8; ++e) {
            const float other = __shfl_xor_sync(0xffffffffu, v[e], 1);
            if ((lane & 1) == 0) {
                unsigned h2;
                asm("cvt.rn.f16x2.f32 %0, %2, %1;" : "=r"(h2) : "f"(v[e]), "f"(other));
                xs_h2[warp][(lane >> 1) + 16 * e] = h2;
            }
        }
    }
}

template <bool SwiGLU>
__device__ __forceinline__ void hyb_b1_core(const unsigned* __restrict__ tiles,
                                            const unsigned* __restrict__ lut_global,
                                            const float* __restrict__ x,
                                            const float* __restrict__ x_up,
                                            const float* __restrict__ col_signs,
                                            const float* __restrict__ row_signs,
                                            float* __restrict__ y,
                                            float* __restrict__ partials,
                                            int rows,
                                            int total_rows,
                                            int cols,
                                            int chunks_per_split,
                                            int splits,
                                            float scale,
                                            int row_block,
                                            int global_row_block,
                                            int split,
                                            int out_slot,
                                            int total_slots) {
    __shared__ unsigned xs_h2[HYB_B1_CHUNKS_PER_BATCH][HYB_CHUNK / 2]; // 8 KB
    __shared__ float out_buf[HYB_CHUNK];                               // 1 KB

    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x & 31;
    const int group_id = lane >> 2;
    const int tid = lane & 3;
    const unsigned reg_lut = __ldg(lut_global + lane);

    const int col_tiles = cols / HYB_TILE;
    const int total_chunks = cols / HYB_CHUNK;
    const int ch0 = split * chunks_per_split;
    const int ch1 = min(ch0 + chunks_per_split, total_chunks);
    if (ch0 >= ch1) {
        return;
    }

    const int group = row_block * HYB_FUSED_WARPS + warp;
    const int bit = lane * 4 * HYB_B;
    const int w0_idx = (bit / 32) % HYB_WORDS;
    const int w1_idx = (w0_idx + 1) % HYB_WORDS;
#if HYB_B == 6
    const int w2_idx = (w0_idx + 2) % HYB_WORDS;
#endif
    const int shift = bit & 31;

    const unsigned* base = tiles + (size_t)group * col_tiles * HYB_WORDS;
    float d0 = 0.0f, d1 = 0.0f, d2 = 0.0f, d3 = 0.0f;

    for (int ch_base = ch0; ch_base < ch1; ch_base += HYB_B1_CHUNKS_PER_BATCH) {
        const int n_ch = min(HYB_B1_CHUNKS_PER_BATCH, ch1 - ch_base);
        hyb_b1_preload_chunks<SwiGLU>(x, x_up, col_signs, xs_h2, ch_base, n_ch, warp, lane);
        __syncthreads();

        for (int lc = 0; lc < n_ch; ++lc) {
            const int k0 = (ch_base + lc) * 16;
            for (int tk = 0; tk < 16; tk += 8) {
                unsigned words[8];
#pragma unroll
                for (int u = 0; u < 8; ++u) {
                    const int kk = k0 + tk + u;
                    words[u] = (lane < HYB_WORDS) ? __ldg(base + (size_t)kk * HYB_WORDS + lane) : 0u;
                }
#pragma unroll
                for (int u = 0; u < 8; ++u) {
                    const unsigned r0 = __shfl_sync(0xffffffffu, words[u], w0_idx);
                    const unsigned r1 = __shfl_sync(0xffffffffu, words[u], w1_idx);
#if HYB_B == 6
                    const unsigned r2 = __shfl_sync(0xffffffffu, words[u], w2_idx);
#endif
                    const unsigned bits0 = __funnelshift_r(r0, r1, shift);
#if HYB_B == 6
                    const unsigned bits1 = __funnelshift_r(r1, r2, shift);
#endif
                    unsigned a[4];
#pragma unroll
                    for (int p = 0; p < 4; ++p) {
#if HYB_B == 6
                        const unsigned state = (p < 3)
                            ? ((bits0 >> (p * 6)) & HYB_STATE_MASK)
                            : (((bits0 >> 18) | (bits1 << 14)) & HYB_STATE_MASK);
#else
                        const unsigned state = (bits0 >> (p * HYB_B)) & HYB_STATE_MASK;
#endif
                        const unsigned h = state * HYB_HASH_MULT;
                        a[p] = __shfl_sync(0xffffffffu, reg_lut, (h >> 5) & 31u) ^ (h & 0x80008000u);
                    }

                    const int xoff = (tk + u) * 8 + tid;
                    const unsigned b0 = (group_id == 0) ? xs_h2[lc][xoff] : 0u;
                    const unsigned b1 = (group_id == 0) ? xs_h2[lc][xoff + 4] : 0u;
                    asm volatile(
                        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
                        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
                }
            }
        }
        if (ch_base + HYB_B1_CHUNKS_PER_BATCH < ch1) {
            __syncthreads();
        }
    }

    if (tid == 0) {
        out_buf[warp * HYB_TILE + group_id] = d0;
        out_buf[warp * HYB_TILE + group_id + 8] = d2;
    }
    __syncthreads();

    if (warp == 0) {
        float v[8];
#pragma unroll
        for (int e = 0; e < 8; ++e) {
            v[e] = out_buf[lane + 32 * e];
        }
#if !HYB_SKIP_OUT_RHT
        hyb_fwht256_warp(v);
#endif
        if (splits == 1) {
            float* yt = y + (size_t)out_slot * rows;
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                const int row = row_block * HYB_CHUNK + lane + 32 * e;
#if !HYB_SKIP_OUT_RHT
                yt[row] = v[e] * __ldg(row_signs + row) * scale;
#else
                yt[row] = v[e] * scale;
#endif
            }
        } else {
            float* pt = partials + ((size_t)split * total_slots + out_slot) * total_rows
                + global_row_block * HYB_CHUNK;
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                const int local_r = lane + 32 * e;
#if !HYB_SKIP_OUT_RHT
                const int row = row_block * HYB_CHUNK + local_r;
                pt[local_r] = v[e] * __ldg(row_signs + row) * scale;
#else
                pt[local_r] = v[e] * scale;
#endif
            }
        }
    }
}

template <bool SwiGLU>
__device__ __forceinline__ void hyb_b1_multi_body(
    const unsigned* __restrict__ t0, const unsigned* __restrict__ t1,
    const unsigned* __restrict__ t2, const unsigned* __restrict__ l0,
    const unsigned* __restrict__ l1, const unsigned* __restrict__ l2,
    const float* __restrict__ c0, const float* __restrict__ c1,
    const float* __restrict__ c2, const float* __restrict__ r0,
    const float* __restrict__ r1, const float* __restrict__ r2,
    float* __restrict__ y0, float* __restrict__ y1, float* __restrict__ y2,
    float s0, float s1, float s2,
    int rows0, int rows1, int rows2, int rb0, int rb1,
    const float* __restrict__ x, const float* __restrict__ x_up,
    int cols, int chunks_per_split, float* __restrict__ partials) {
    int row_block = blockIdx.x;
    const unsigned *tiles = t0, *lut = l0;
    const float *col_signs = c0, *row_signs = r0;
    float* y = y0;
    float scale = s0;
    int rows = rows0;
    if (row_block >= rb0) {
        row_block -= rb0;
        tiles = t1, lut = l1, col_signs = c1, row_signs = r1, y = y1, scale = s1, rows = rows1;
        if (row_block >= rb1) {
            row_block -= rb1;
            tiles = t2, lut = l2, col_signs = c2, row_signs = r2, y = y2, scale = s2, rows = rows2;
        }
    }
    const int total_rows = gridDim.x * HYB_CHUNK;
    hyb_b1_core<SwiGLU>(tiles, lut, x, SwiGLU ? x_up : nullptr, col_signs, row_signs, y, partials,
                        rows, total_rows, cols, chunks_per_split, gridDim.y, scale, row_block,
                        blockIdx.x, blockIdx.y, 0, 1);
}

extern "C" __global__ void __launch_bounds__(512, 3)
hyb_b1_multi(const unsigned* __restrict__ t0, const unsigned* __restrict__ t1,
             const unsigned* __restrict__ t2, const unsigned* __restrict__ l0,
             const unsigned* __restrict__ l1, const unsigned* __restrict__ l2,
             const float* __restrict__ c0, const float* __restrict__ c1,
             const float* __restrict__ c2, const float* __restrict__ r0,
             const float* __restrict__ r1, const float* __restrict__ r2,
             float* __restrict__ y0, float* __restrict__ y1, float* __restrict__ y2,
             float s0, float s1, float s2,
             int rows0, int rows1, int rows2, int rb0, int rb1,
             const float* __restrict__ x, const float* __restrict__ x_up,
             int cols, int chunks_per_split, float* __restrict__ partials) {
    hyb_b1_multi_body<false>(t0, t1, t2, l0, l1, l2, c0, c1, c2, r0, r1, r2, y0, y1, y2, s0, s1,
                             s2, rows0, rows1, rows2, rb0, rb1, x, x_up, cols, chunks_per_split,
                             partials);
}

extern "C" __global__ void __launch_bounds__(512, 3)
hyb_b1_multi_swiglu(const unsigned* __restrict__ t0, const unsigned* __restrict__ t1,
                    const unsigned* __restrict__ t2, const unsigned* __restrict__ l0,
                    const unsigned* __restrict__ l1, const unsigned* __restrict__ l2,
                    const float* __restrict__ c0, const float* __restrict__ c1,
                    const float* __restrict__ c2, const float* __restrict__ r0,
                    const float* __restrict__ r1, const float* __restrict__ r2,
                    float* __restrict__ y0, float* __restrict__ y1, float* __restrict__ y2,
                    float s0, float s1, float s2,
                    int rows0, int rows1, int rows2, int rb0, int rb1,
                    const float* __restrict__ x, const float* __restrict__ x_up,
                    int cols, int chunks_per_split, float* __restrict__ partials) {
    hyb_b1_multi_body<true>(t0, t1, t2, l0, l1, l2, c0, c1, c2, r0, r1, r2, y0, y1, y2, s0, s1, s2,
                            rows0, rows1, rows2, rb0, rb1, x, x_up, cols, chunks_per_split,
                            partials);
}

extern "C" __global__ void __launch_bounds__(512, 3)
hyb_moe_b1(const unsigned* __restrict__ t0,
           const unsigned* __restrict__ t1,
           const unsigned* __restrict__ l0,
           const unsigned* __restrict__ l1,
           const float* __restrict__ sc0,
           const float* __restrict__ sc1,
           const float* __restrict__ c0,
           const float* __restrict__ c1,
           const float* __restrict__ r0,
           const float* __restrict__ r1,
           float* __restrict__ y0,
           float* __restrict__ y1,
           int rows0,
           int rows1,
           int rb0,
           const float* __restrict__ x,
           const int* __restrict__ ids,
           int cols,
           int used,
           int chunks_per_split,
           int splits,
           float* __restrict__ partials) {
    const int rb = blockIdx.x;
    const int assign = blockIdx.y;
    const int expert = __ldg(ids + assign);
    const int total_rows = rows0 + rows1;
    const int col_tiles = cols / HYB_TILE;
    if (rb < rb0) {
        const size_t expert_stride = (size_t)(rows0 / HYB_TILE) * col_tiles * HYB_WORDS;
        hyb_b1_core<false>(t0 + (size_t)expert * expert_stride, l0, x, nullptr, c0, r0, y0,
                           partials, rows0, total_rows, cols, chunks_per_split, splits,
                           __ldg(sc0 + expert), rb, rb, blockIdx.z, assign, used);
    } else {
        const size_t expert_stride = (size_t)(rows1 / HYB_TILE) * col_tiles * HYB_WORDS;
        hyb_b1_core<false>(t1 + (size_t)expert * expert_stride, l1, x, nullptr, c1, r1, y1,
                           partials, rows1, total_rows, cols, chunks_per_split, splits,
                           __ldg(sc1 + expert), rb - rb0, rb, blockIdx.z, assign, used);
    }
}

extern "C" __global__ void __launch_bounds__(512)
hyb_moe_b1_swiglu(const unsigned* __restrict__ t0,
                  const unsigned* __restrict__ l0,
                  const float* __restrict__ sc0,
                  const float* __restrict__ c0,
                  const float* __restrict__ r0,
                  float* __restrict__ y0,
                  int rows0,
                  const float* __restrict__ x_gate,
                  const float* __restrict__ x_up,
                  const int* __restrict__ ids,
                  int cols,
                  int used,
                  int chunks_per_split,
                  int splits,
                  float* __restrict__ partials) {
    const int assign = blockIdx.y;
    const int expert = __ldg(ids + assign);
    const int col_tiles = cols / HYB_TILE;
    const size_t expert_stride = (size_t)(rows0 / HYB_TILE) * col_tiles * HYB_WORDS;
    const float* xg = x_gate + (size_t)assign * cols;
    const float* xu = x_up + (size_t)assign * cols;
    hyb_b1_core<true>(t0 + (size_t)expert * expert_stride, l0, xg, xu, c0, r0, y0, partials, rows0,
                      rows0, cols, chunks_per_split, splits, __ldg(sc0 + expert), blockIdx.x,
                      blockIdx.x, blockIdx.z, assign, used);
}

