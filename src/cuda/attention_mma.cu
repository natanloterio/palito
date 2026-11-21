// Causal prefill attention on FP16 tensor cores (FP32 accumulation and softmax)
// for dim == 128 and GQA group == 8. Same layouts and mask as
// attention_prefill_flash4: q/output [token][heads][128], keys/values
// [position][kv_heads][128]; token t sees keys [0, count + t).
//
// grid = (kv_heads, ceil(tokens / (2 * Warps))), block = 32 * Warps.
// Warp w owns 16 query rows: tokens first + 2w and first + 2w + 1, all 8 heads of
// the block's KV head (MMA row r -> token r / 8, head r % 8). Each 64-key tile is
// staged once in shared memory as FP16 hi/lo pairs (K row-major, V transposed) and
// serves 16 * Warps query rows. S = Q K^T and O += P V use mma.m16n8k16 (f16 -> f32).
#define FA_DIM 128
#define FA_KEYS 64
#define FA_K_STRIDE (FA_DIM + 8)
#define FA_V_STRIDE (FA_KEYS + 8)

#if __CUDA_ARCH__ >= 800
#define FA_NEG_INF __int_as_float(0xff800000)

// Two FP32 values -> packed FP16 pair (lo in bits 0..15), round to nearest.
__device__ __forceinline__ unsigned int fa_pack(float lo, float hi) {
    unsigned int out;
    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(out) : "f"(hi), "f"(lo));
    return out;
}

__device__ __forceinline__ float2 fa_unpack(unsigned int packed) {
    float lo, hi;
    asm("{.reg .f16 l, h; mov.b32 {l, h}, %2; cvt.f32.f16 %0, l; cvt.f32.f16 %1, h;}"
        : "=f"(lo), "=f"(hi) : "r"(packed));
    return make_float2(lo, hi);
}

__device__ __forceinline__ void fa_mma(float c[4], const unsigned int a[4], unsigned int b0,
                                       unsigned int b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// Split: Q and K are carried as FP16 high + low parts and S takes three MMAs
// (hi*hi + lo*hi + hi*lo). SplitV: V and the probabilities P likewise,
// O += Phi*Vhi + Phi*Vlo + Plo*Vhi. Plain FP16 S or V measured max_abs ~1.0 on
// real-text logits; FP16 P alone failed the long MoE gates (cos 0.999988). With all
// splits the 600-token MoE gate stays within max_abs 0.00004 of the oracle.
template<int Warps, bool Split, bool SplitV>
__device__ __forceinline__ void attention_prefill_mma_impl(
    const float *q, const float *keys, const float *values, float *output, int heads,
    int kv_heads, int tokens, int count, const uint4 *halves = nullptr, int stride = 0) {
    constexpr int THREADS = 32 * Warps, GROUP = 8;
    __shared__ __align__(16) unsigned short ks[FA_KEYS * FA_K_STRIDE];
    __shared__ __align__(16) unsigned short kl[Split ? FA_KEYS * FA_K_STRIDE : 2];
    __shared__ __align__(16) unsigned short vl[SplitV ? FA_DIM * FA_V_STRIDE : 2];
    __shared__ __align__(16) unsigned short vt[FA_DIM * FA_V_STRIDE];
    int kv_head = blockIdx.x, tid = threadIdx.x, warp = tid / 32, lane = tid % 32;
    int group = lane / 4, thread = lane % 4;
    int first = blockIdx.y * 2 * Warps;
    int token0 = first + 2 * warp, token1 = token0 + 1;
    // Rows of tokens past the chunk compute on one valid key and are not stored.
    int limit0 = token0 < tokens ? count + token0 : 1;
    int limit1 = token1 < tokens ? count + token1 : 1;
    int warp_limit = max(limit0, limit1);
    int span = count + min(tokens - 1, first + 2 * Warps - 1);
    // Scores are kept in log2 units: Q is pre-scaled by log2(e) / sqrt(dim).
    const float qscale = 1.4426950408889634f * rsqrtf((float)FA_DIM);
    unsigned int qf[FA_DIM / 16][4], ql[Split ? FA_DIM / 16 : 1][4];
    {
        int head = kv_head * GROUP + group;
        const float *q0 = q + ((u64)min(token0, tokens - 1) * heads + head) * FA_DIM;
        const float *q1 = q + ((u64)min(token1, tokens - 1) * heads + head) * FA_DIM;
        #pragma unroll
        for (int s = 0; s < FA_DIM / 16; s++) {
            int d = 16 * s + 2 * thread;
            float2 x0 = *(const float2 *)(q0 + d), x1 = *(const float2 *)(q1 + d);
            float2 y0 = *(const float2 *)(q0 + d + 8), y1 = *(const float2 *)(q1 + d + 8);
            float v[4][2] = {{x0.x * qscale, x0.y * qscale}, {x1.x * qscale, x1.y * qscale},
                             {y0.x * qscale, y0.y * qscale}, {y1.x * qscale, y1.y * qscale}};
            #pragma unroll
            for (int e = 0; e < 4; e++) {
                qf[s][e] = fa_pack(v[e][0], v[e][1]);
                if (Split) {
                    float2 h = fa_unpack(qf[s][e]);
                    ql[Split ? s : 0][e] = fa_pack(v[e][0] - h.x, v[e][1] - h.y);
                }
            }
        }
    }
    float o[FA_DIM / 8][4];
    #pragma unroll
    for (int j = 0; j < FA_DIM / 8; j++) o[j][0] = o[j][1] = o[j][2] = o[j][3] = 0.0f;
    // Index 0: row `group` (token0); index 1: row `group + 8` (token1).
    float maximum[2] = {FA_NEG_INF, FA_NEG_INF}, total[2] = {0.0f, 0.0f};
    for (int base = 0; base < span; base += FA_KEYS) {
        int n = min(FA_KEYS, span - base);
        __syncthreads();
        if (halves) {
            // Pre-split tiles (attention_half_kv): plain 16-byte copies. Keys past
            // the span are masked; their V rows are finite (zeroed or older data).
            const uint4 *khi = halves, *klo = halves + (u64)stride * kv_heads * (FA_DIM / 8);
            const uint4 *vhi = klo + (u64)stride * kv_heads * (FA_DIM / 8);
            const uint4 *vlo = vhi + (u64)stride * kv_heads * (FA_DIM / 8);
            for (int i = tid; i < FA_KEYS * FA_DIM / 8; i += THREADS) {
                int key = i / (FA_DIM / 8), c = i % (FA_DIM / 8);
                u64 at = ((u64)(base + key) * kv_heads + kv_head) * (FA_DIM / 8) + c;
                uint4 zero = make_uint4(0u, 0u, 0u, 0u);
                *(uint4 *)&ks[key * FA_K_STRIDE + 8 * c] = key < n ? khi[at] : zero;
                *(uint4 *)&kl[key * FA_K_STRIDE + 8 * c] = key < n ? klo[at] : zero;
                int d = i / (FA_KEYS / 8), part = i % (FA_KEYS / 8);
                u64 row = ((u64)kv_head * FA_DIM + d) * (stride / 8) + base / 8 + part;
                *(uint4 *)&vt[d * FA_V_STRIDE + 8 * part] = vhi[row];
                *(uint4 *)&vl[d * FA_V_STRIDE + 8 * part] = vlo[row];
            }
        } else {
        for (int i = tid; i < FA_KEYS * FA_DIM / 4; i += THREADS) {
            int key = i / (FA_DIM / 4), c = 4 * (i % (FA_DIM / 4));
            float4 v = key < n
                ? *(const float4 *)(keys + ((u64)(base + key) * kv_heads + kv_head) * FA_DIM + c)
                : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            unsigned int *dst = (unsigned int *)&ks[key * FA_K_STRIDE + c];
            dst[0] = fa_pack(v.x, v.y);
            dst[1] = fa_pack(v.z, v.w);
            if (Split) {
                float2 h0 = fa_unpack(dst[0]), h1 = fa_unpack(dst[1]);
                unsigned int *low = (unsigned int *)&kl[key * FA_K_STRIDE + c];
                low[0] = fa_pack(v.x - h0.x, v.y - h0.y);
                low[1] = fa_pack(v.z - h1.x, v.w - h1.y);
            }
        }
        for (int i = tid; i < (FA_KEYS / 2) * (FA_DIM / 4); i += THREADS) {
            int pair = i / (FA_DIM / 4), c = 4 * (i % (FA_DIM / 4)), key = 2 * pair;
            float4 a = key < n
                ? *(const float4 *)(values + ((u64)(base + key) * kv_heads + kv_head) * FA_DIM + c)
                : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            float4 b = key + 1 < n
                ? *(const float4 *)(values + ((u64)(base + key + 1) * kv_heads + kv_head) * FA_DIM + c)
                : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            float pairs[4][2] = {{a.x, b.x}, {a.y, b.y}, {a.z, b.z}, {a.w, b.w}};
            #pragma unroll
            for (int e = 0; e < 4; e++) {
                unsigned int high = fa_pack(pairs[e][0], pairs[e][1]);
                *(unsigned int *)&vt[(c + e) * FA_V_STRIDE + key] = high;
                if (SplitV) {
                    float2 h = fa_unpack(high);
                    *(unsigned int *)&vl[(c + e) * FA_V_STRIDE + key] =
                        fa_pack(pairs[e][0] - h.x, pairs[e][1] - h.y);
                }
            }
        }
        }
        __syncthreads();
        if (base >= warp_limit) continue;
        float s[FA_KEYS / 8][4];
        #pragma unroll
        for (int j = 0; j < FA_KEYS / 8; j++) {
            s[j][0] = s[j][1] = s[j][2] = s[j][3] = 0.0f;
            const unsigned int *row = (const unsigned int *)&ks[(8 * j + group) * FA_K_STRIDE];
            #pragma unroll
            for (int k = 0; k < FA_DIM / 16; k++) {
                unsigned int b0 = row[8 * k + thread], b1 = row[8 * k + 4 + thread];
                fa_mma(s[j], qf[k], b0, b1);
                if (Split) {
                    const unsigned int *low = (const unsigned int *)&kl[(8 * j + group) * FA_K_STRIDE];
                    fa_mma(s[j], ql[Split ? k : 0], b0, b1);
                    fa_mma(s[j], qf[k], low[8 * k + thread], low[8 * k + 4 + thread]);
                }
            }
        }
        float step[2] = {FA_NEG_INF, FA_NEG_INF};
        #pragma unroll
        for (int j = 0; j < FA_KEYS / 8; j++) {
            #pragma unroll
            for (int e = 0; e < 4; e++) {
                int key = base + 8 * j + 2 * thread + e % 2;
                if (key >= (e < 2 ? limit0 : limit1)) s[j][e] = FA_NEG_INF;
                step[e / 2] = fmaxf(step[e / 2], s[j][e]);
            }
        }
        float rescale[2];
        #pragma unroll
        for (int r = 0; r < 2; r++) {
            step[r] = fmaxf(step[r], __shfl_xor_sync(0xffffffff, step[r], 1));
            step[r] = fmaxf(step[r], __shfl_xor_sync(0xffffffff, step[r], 2));
            float updated = fmaxf(maximum[r], step[r]);
            // updated is finite: every row sees key 0 in the first tile.
            rescale[r] = exp2f(maximum[r] - updated);
            maximum[r] = updated;
            total[r] *= rescale[r];
        }
        #pragma unroll
        for (int j = 0; j < FA_DIM / 8; j++) {
            o[j][0] *= rescale[0];
            o[j][1] *= rescale[0];
            o[j][2] *= rescale[1];
            o[j][3] *= rescale[1];
        }
        #pragma unroll
        for (int j = 0; j < FA_KEYS / 8; j++) {
            #pragma unroll
            for (int e = 0; e < 4; e++) {
                s[j][e] = exp2f(s[j][e] - maximum[e / 2]);
                total[e / 2] += s[j][e];
            }
        }
        #pragma unroll
        for (int k = 0; k < FA_KEYS / 16; k++) {
            float pv[4][2] = {{s[2 * k][0], s[2 * k][1]}, {s[2 * k][2], s[2 * k][3]},
                              {s[2 * k + 1][0], s[2 * k + 1][1]}, {s[2 * k + 1][2], s[2 * k + 1][3]}};
            unsigned int p[4], pl[4];
            #pragma unroll
            for (int e = 0; e < 4; e++) {
                p[e] = fa_pack(pv[e][0], pv[e][1]);
                float2 h = fa_unpack(p[e]);
                pl[e] = fa_pack(pv[e][0] - h.x, pv[e][1] - h.y);
            }
            #pragma unroll
            for (int j = 0; j < FA_DIM / 8; j++) {
                const unsigned int *row = (const unsigned int *)&vt[(8 * j + group) * FA_V_STRIDE];
                fa_mma(o[j], p, row[8 * k + thread], row[8 * k + 4 + thread]);
                if (SplitV) {
                    const unsigned int *low = (const unsigned int *)&vl[(8 * j + group) * FA_V_STRIDE];
                    fa_mma(o[j], p, low[8 * k + thread], low[8 * k + 4 + thread]);
                    fa_mma(o[j], pl, row[8 * k + thread], row[8 * k + 4 + thread]);
                }
            }
        }
    }
    #pragma unroll
    for (int r = 0; r < 2; r++) {
        total[r] += __shfl_xor_sync(0xffffffff, total[r], 1);
        total[r] += __shfl_xor_sync(0xffffffff, total[r], 2);
    }
    int head = kv_head * GROUP + group;
    #pragma unroll
    for (int r = 0; r < 2; r++) {
        int token = r ? token1 : token0;
        if (token >= tokens) continue;
        float inverse = 1.0f / total[r];
        float *out = output + ((u64)token * heads + head) * FA_DIM + 2 * thread;
        #pragma unroll
        for (int j = 0; j < FA_DIM / 8; j++)
            *(float2 *)(out + 8 * j) = make_float2(o[j][2 * r] * inverse, o[j][2 * r + 1] * inverse);
    }
}
#endif

extern "C" __global__ void __launch_bounds__(128)
attention_prefill_mma(const float *q, const float *keys, const float *values, float *output,
                      int heads, int kv_heads, int tokens, int count) {
#if __CUDA_ARCH__ >= 800
    attention_prefill_mma_impl<4, true, true>(q, keys, values, output, heads, kv_heads, tokens, count);
#endif
}

// Pre-split K/V for attention_prefill_mma_half: four planes of `stride` positions,
// K hi, K lo as [position][kv_head][128] and V hi, V lo transposed as
// [kv_head][128][stride] (FP16 bits). grid = (ceil(positions / 32), kv_heads),
// 256 threads; stride % 64 == 0 and positions <= stride.
extern "C" __global__ void __launch_bounds__(256)
attention_half_kv(const float *keys, const float *values, unsigned short *halves,
                  int kv_heads, int positions, int stride) {
#if __CUDA_ARCH__ >= 800
    __shared__ float tile[FA_DIM][33];
    int kv_head = blockIdx.y, first = blockIdx.x * 32, tid = threadIdx.x;
    u64 plane = (u64)stride * kv_heads * FA_DIM;
    unsigned short *khi = halves, *klo = halves + plane, *vhi = klo + plane, *vlo = vhi + plane;
    for (int i = tid; i < 32 * FA_DIM / 2; i += 256) {
        int p = i / (FA_DIM / 2), d = 2 * (i % (FA_DIM / 2)), pos = first + p;
        float2 k = make_float2(0.0f, 0.0f), v = make_float2(0.0f, 0.0f);
        if (pos < positions) {
            u64 at = ((u64)pos * kv_heads + kv_head) * FA_DIM + d;
            k = *(const float2 *)(keys + at);
            v = *(const float2 *)(values + at);
            unsigned int high = fa_pack(k.x, k.y);
            float2 h = fa_unpack(high);
            *(unsigned int *)(khi + at) = high;
            *(unsigned int *)(klo + at) = fa_pack(k.x - h.x, k.y - h.y);
        }
        tile[d][p] = v.x;
        tile[d + 1][p] = v.y;
    }
    __syncthreads();
    for (int i = tid; i < FA_DIM * 16; i += 256) {
        int d = i / 16, pair = 2 * (i % 16), pos = first + pair;
        if (pos >= positions) continue;
        float a = tile[d][pair], b = tile[d][pair + 1];
        unsigned int high = fa_pack(a, b);
        float2 h = fa_unpack(high);
        u64 at = ((u64)kv_head * FA_DIM + d) * stride + pos;
        *(unsigned int *)(vhi + at) = high;
        *(unsigned int *)(vlo + at) = fa_pack(a - h.x, b - h.y);
    }
#endif
}

extern "C" __global__ void __launch_bounds__(256)
attention_prefill_mma_half(const float *q, const unsigned short *halves, float *output,
                           int heads, int kv_heads, int tokens, int count, int stride) {
#if __CUDA_ARCH__ >= 800
    attention_prefill_mma_impl<8, true, true>(q, nullptr, nullptr, output, heads, kv_heads,
                                              tokens, count, (const uint4 *)halves, stride);
#endif
}

