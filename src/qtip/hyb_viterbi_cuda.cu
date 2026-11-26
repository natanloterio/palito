// QTIP HYB Circular 2-Pass Tail-Biting Viterbi Quantizer (CUDA / Blackwell SM_120).
//
// One threadblock quantizes one 16x16 MMA-swizzled, sigma-normalized tile (128 pairs).
// Register-resident DP state costs and precomputed codebook pairs `(va, vb)` eliminate
// shared-memory traffic for state values; double-buffered swizzled `head_cost` needs
// only ONE `__syncthreads()` per Viterbi step; backpointers `back[127][HYB_HEADS]`
// live entirely in shared memory.

#ifndef HYB_L
#define HYB_L 10
#endif
#ifndef HYB_B
#define HYB_B 6
#endif
#ifndef HYB_Q
#define HYB_Q 5
#endif

#define HYB_STATES (1u << HYB_L)
#define HYB_FAN (1u << HYB_B)
#define HYB_HEAD_BITS (HYB_L - HYB_B)
#define HYB_HEADS (1u << HYB_HEAD_BITS)
#define HYB_HEAD_MASK (HYB_HEADS - 1u)
#define HYB_SYM_MASK (HYB_FAN - 1u)
#define HYB_PAIRS 128
#define HYB_WORDS ((HYB_PAIRS * HYB_B) / 32)
#define HYB_LUT_SIZE (1u << HYB_Q)
#define HYB_LUT_MASK (HYB_LUT_SIZE - 1u)
#define HYB_HASH_MULT 0x9E3779B1u

#if HYB_HEADS <= 256
#define HYB_BLOCK_THREADS 256
#else
#define HYB_BLOCK_THREADS 512
#endif

#define HYB_STATES_PER_THREAD (HYB_STATES / HYB_BLOCK_THREADS)
#define HYB_THREADS_PER_HEAD (HYB_BLOCK_THREADS / HYB_HEADS)

static_assert(HYB_B >= 3 && HYB_B <= 6, "HYB_B must be in 3..=6");
static_assert(HYB_L > HYB_B && HYB_L <= 13, "HYB_L must satisfy HYB_B < HYB_L <= 13");
static_assert(HYB_STATES >= HYB_BLOCK_THREADS, "HYB_STATES must be >= HYB_BLOCK_THREADS");
static_assert(HYB_HEADS <= HYB_BLOCK_THREADS, "HYB_HEADS must be <= HYB_BLOCK_THREADS");
static_assert(HYB_THREADS_PER_HEAD <= 32, "HYB_THREADS_PER_HEAD must fit within a single warp");

__device__ __forceinline__ unsigned hyb_head_swizzle(unsigned h) {
    return h ^ (h >> 4);
}

__device__ __forceinline__ void hyb_decode_pair_smem(
    const unsigned* __restrict__ lut_s, unsigned state, float& wa, float& wb) {
    const unsigned h = (state & 0xFFFFu) * HYB_HASH_MULT;
    const unsigned idx = (h >> 5) & HYB_LUT_MASK;
    const unsigned raw = lut_s[idx] ^ (h & 0x80008000u);
    asm("cvt.f32.f16 %0, %1;" : "=f"(wa) : "h"((unsigned short)(raw & 0xFFFFu)));
    asm("cvt.f32.f16 %0, %1;" : "=f"(wb) : "h"((unsigned short)(raw >> 16)));
}

extern "C" __global__ void __launch_bounds__(HYB_BLOCK_THREADS)
hyb_viterbi_quantize(const float* __restrict__ tiles_in,
                     const unsigned* __restrict__ lut_global,
                     unsigned* __restrict__ out_words,
                     float* __restrict__ out_decoded,
                     unsigned n_tiles) {
    const unsigned tile_idx = blockIdx.x;
    if (tile_idx >= n_tiles) return;

    __shared__ unsigned lut_s[HYB_LUT_SIZE];
    __shared__ float weights_s[256];
    __shared__ float head_cost[2][HYB_HEADS];
    __shared__ unsigned char back[(HYB_PAIRS - 1) * HYB_HEADS];
    __shared__ unsigned st_out[HYB_PAIRS];
    __shared__ float reduce_c[HYB_BLOCK_THREADS / 32];
    __shared__ unsigned reduce_s[HYB_BLOCK_THREADS / 32];
    __shared__ unsigned wrap_bits_s;
    __shared__ unsigned best_s_s;

    const int tid = threadIdx.x;
    const float* tile_in = tiles_in + (size_t)tile_idx * 256;

    for (int i = tid; i < (int)HYB_LUT_SIZE; i += HYB_BLOCK_THREADS) {
        lut_s[i] = lut_global[i];
    }
    for (int i = tid; i < 256; i += HYB_BLOCK_THREADS) {
        weights_s[i] = tile_in[i];
    }
    __syncthreads();

    const unsigned head = (unsigned)tid / HYB_THREADS_PER_HEAD;
    const unsigned sub = (unsigned)tid % HYB_THREADS_PER_HEAD;
    const unsigned low_base = sub * HYB_STATES_PER_THREAD;
    const unsigned state_base = (head << HYB_B) | low_base;

    float va[HYB_STATES_PER_THREAD];
    float vb[HYB_STATES_PER_THREAD];
    float cost[HYB_STATES_PER_THREAD];

#pragma unroll
    for (unsigned i = 0; i < HYB_STATES_PER_THREAD; ++i) {
        hyb_decode_pair_smem(lut_s, state_base + i, va[i], vb[i]);
    }

    // -------------------------------------------------------------------------
    // Pass 1: Unconstrained Viterbi forward pass to select circular wrap_bits.
    // -------------------------------------------------------------------------
    {
        const float w0 = weights_s[0];
        const float w1 = weights_s[1];
#pragma unroll
        for (unsigned i = 0; i < HYB_STATES_PER_THREAD; ++i) {
            const float d0 = w0 - va[i];
            const float d1 = w1 - vb[i];
            cost[i] = d0 * d0 + d1 * d1;
        }
    }

    for (int p = 1; p < HYB_PAIRS; ++p) {
        float min_c = cost[0];
#pragma unroll
        for (unsigned i = 1; i < HYB_STATES_PER_THREAD; ++i) {
            min_c = fminf(min_c, cost[i]);
        }
#if HYB_THREADS_PER_HEAD > 1
#pragma unroll
        for (unsigned offset = HYB_THREADS_PER_HEAD >> 1; offset >= 1; offset >>= 1) {
            const float oc = __shfl_xor_sync(0xffffffffu, min_c, offset);
            min_c = fminf(min_c, oc);
        }
#endif
        const int buf = p & 1;
        if (sub == 0) {
            head_cost[buf][hyb_head_swizzle(head)] = min_c;
        }
        __syncthreads();

        const float w0 = weights_s[2 * p];
        const float w1 = weights_s[2 * p + 1];
#pragma unroll
        for (unsigned i = 0; i < HYB_STATES_PER_THREAD; ++i) {
            const unsigned h_idx = (state_base + i) & HYB_HEAD_MASK;
            const float d0 = w0 - va[i];
            const float d1 = w1 - vb[i];
            cost[i] = head_cost[buf][hyb_head_swizzle(h_idx)] + d0 * d0 + d1 * d1;
        }
    }

    // Block-wide argmin over all states to extract wrap_bits = best_s >> HYB_B.
    {
        float best_c = cost[0];
        unsigned best_s = state_base;
#pragma unroll
        for (unsigned i = 1; i < HYB_STATES_PER_THREAD; ++i) {
            if (cost[i] < best_c) {
                best_c = cost[i];
                best_s = state_base + i;
            }
        }
#pragma unroll
        for (int offset = 16; offset >= 1; offset >>= 1) {
            const float oc = __shfl_xor_sync(0xffffffffu, best_c, offset);
            const unsigned os = __shfl_xor_sync(0xffffffffu, best_s, offset);
            if (oc < best_c || (oc == best_c && os < best_s)) {
                best_c = oc;
                best_s = os;
            }
        }
        const int warp = tid >> 5;
        const int lane = tid & 31;
        if (lane == 0) {
            reduce_c[warp] = best_c;
            reduce_s[warp] = best_s;
        }
        __syncthreads();
        if (warp == 0) {
            constexpr int NUM_WARPS = HYB_BLOCK_THREADS / 32;
            const float inf = __int_as_float(0x7f800000);
            float wc = (lane < NUM_WARPS) ? reduce_c[lane] : inf;
            unsigned ws = (lane < NUM_WARPS) ? reduce_s[lane] : 0xffffffffu;
#pragma unroll
            for (int offset = 16; offset >= 1; offset >>= 1) {
                const float oc = __shfl_xor_sync(0xffffffffu, wc, offset);
                const unsigned os = __shfl_xor_sync(0xffffffffu, ws, offset);
                if (oc < wc || (oc == wc && os < ws)) {
                    wc = oc;
                    ws = os;
                }
            }
            if (lane == 0) {
                wrap_bits_s = ws >> HYB_B;
            }
        }
    }
    __syncthreads();
    const unsigned wrap_bits = wrap_bits_s;

    // -------------------------------------------------------------------------
    // Pass 2: Circular tail-biting Viterbi pinned to wrap_bits, with backptrs.
    // -------------------------------------------------------------------------
    {
        const float inf = __int_as_float(0x7f800000);
        const float w0 = weights_s[0];
        const float w1 = weights_s[1];
#pragma unroll
        for (unsigned i = 0; i < HYB_STATES_PER_THREAD; ++i) {
            const unsigned s = state_base + i;
            if ((s & HYB_HEAD_MASK) == wrap_bits) {
                const float d0 = w0 - va[i];
                const float d1 = w1 - vb[i];
                cost[i] = d0 * d0 + d1 * d1;
            } else {
                cost[i] = inf;
            }
        }
    }

    for (int p = 1; p < HYB_PAIRS; ++p) {
        float min_c = cost[0];
        unsigned min_low = low_base;
#pragma unroll
        for (unsigned i = 1; i < HYB_STATES_PER_THREAD; ++i) {
            if (cost[i] < min_c) {
                min_c = cost[i];
                min_low = low_base + i;
            }
        }
#if HYB_THREADS_PER_HEAD > 1
#pragma unroll
        for (unsigned offset = HYB_THREADS_PER_HEAD >> 1; offset >= 1; offset >>= 1) {
            const float oc = __shfl_xor_sync(0xffffffffu, min_c, offset);
            const unsigned ol = __shfl_xor_sync(0xffffffffu, min_low, offset);
            if (oc < min_c || (oc == min_c && ol < min_low)) {
                min_c = oc;
                min_low = ol;
            }
        }
#endif
        const int buf = p & 1;
        if (sub == 0) {
            head_cost[buf][hyb_head_swizzle(head)] = min_c;
            back[(p - 1) * HYB_HEADS + head] = (unsigned char)min_low;
        }
        __syncthreads();

        const float w0 = weights_s[2 * p];
        const float w1 = weights_s[2 * p + 1];
#pragma unroll
        for (unsigned i = 0; i < HYB_STATES_PER_THREAD; ++i) {
            const unsigned h_idx = (state_base + i) & HYB_HEAD_MASK;
            const float d0 = w0 - va[i];
            const float d1 = w1 - vb[i];
            cost[i] = head_cost[buf][hyb_head_swizzle(h_idx)] + d0 * d0 + d1 * d1;
        }
    }

    // End of Pass 2: select argmin among states with (s >> HYB_B) == wrap_bits (i.e. head == wrap_bits).
    {
        float min_c = cost[0];
        unsigned min_low = low_base;
#pragma unroll
        for (unsigned i = 1; i < HYB_STATES_PER_THREAD; ++i) {
            if (cost[i] < min_c) {
                min_c = cost[i];
                min_low = low_base + i;
            }
        }
#if HYB_THREADS_PER_HEAD > 1
#pragma unroll
        for (unsigned offset = HYB_THREADS_PER_HEAD >> 1; offset >= 1; offset >>= 1) {
            const float oc = __shfl_xor_sync(0xffffffffu, min_c, offset);
            const unsigned ol = __shfl_xor_sync(0xffffffffu, min_low, offset);
            if (oc < min_c || (oc == min_c && ol < min_low)) {
                min_c = oc;
                min_low = ol;
            }
        }
#endif
        if (head == wrap_bits && sub == 0) {
            best_s_s = (wrap_bits << HYB_B) | min_low;
        }
    }
    __syncthreads();

    // -------------------------------------------------------------------------
    // Backtrack 128 states and pack circular LSB-first words.
    // -------------------------------------------------------------------------
    if (tid == 0) {
        unsigned cur = best_s_s;
        for (int p = HYB_PAIRS - 1; p >= 1; --p) {
            st_out[p] = cur;
            const unsigned h = cur & HYB_HEAD_MASK;
            const unsigned low = (unsigned)back[(p - 1) * HYB_HEADS + h];
            cur = (h << HYB_B) | low;
        }
        st_out[0] = cur;

        unsigned words[HYB_WORDS];
#pragma unroll
        for (int w = 0; w < HYB_WORDS; ++w) {
            words[w] = 0u;
        }
        for (int p = 0; p < HYB_PAIRS; ++p) {
            const unsigned bits = st_out[p] & HYB_SYM_MASK;
            const int offset = p * HYB_B;
            const int w = offset >> 5;
            const int shift = offset & 31;
            words[w] |= bits << shift;
            if (shift + HYB_B > 32) {
                words[(w + 1) % HYB_WORDS] |= bits >> (32 - shift);
            }
        }
        unsigned* dst_w = out_words + (size_t)tile_idx * HYB_WORDS;
#pragma unroll
        for (int w = 0; w < HYB_WORDS; ++w) {
            dst_w[w] = words[w];
        }
    }
    __syncthreads();

    // -------------------------------------------------------------------------
    // Decode 128 pairs into UNSWIZZLED 16x16 row-major order (sigma = 1.0).
    // -------------------------------------------------------------------------
    if (tid < HYB_PAIRS) {
        const unsigned lane_id = (unsigned)tid >> 2;
        const unsigned p_in_lane = (unsigned)tid & 3u;
        const unsigned group_id = lane_id >> 2;
        const unsigned t_in_group = lane_id & 3u;
        const unsigned r = group_id + ((p_in_lane & 1u) ? 8u : 0u);
        const unsigned c = 2u * t_in_group + ((p_in_lane & 2u) ? 8u : 0u);
        float w0, w1;
        hyb_decode_pair_smem(lut_s, st_out[tid], w0, w1);
        float* dst_d = out_decoded + (size_t)tile_idx * 256;
        dst_d[r * 16 + c] = w0;
        dst_d[r * 16 + c + 1] = w1;
    }
}
