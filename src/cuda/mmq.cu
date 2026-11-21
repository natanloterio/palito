// Integer tensor-core prefill projection for packed Q4_K / Q6_K weights and INT8
// activation planes: same ABI as residual_matmul_mma3. Activations are either C
// residual components (residual_quantize32_precise<C>, one scale/sum per component)
// or, with S16, two residual planes whose scales differ by an integer factor
// (quantize32_residual_pair: x ~ d * (K * q1 + q2), K in [1, 32768] per 32 values);
// S16 combines the two MMAs in the integer accumulator (dot = K * dot1 + dot2,
// bounded by 65024 * 32768 + 65024 < 2^31). Each block stages a 128-row x KS weight tile and a
// BN-token activation tile per component; 8 warps (4 along rows x 2 along
// tokens) issue s8 MMAs whose integer dot products are exact per 32-value (Q4_K)
// or 16-value (Q6_K) scale group. Each output accumulates in F32
//   coefficient * sum_c(dot_c * scale_c) - minimum * sum_c(sum_c * scale_c).
#define MMQ_ROWS 128

// Declared once per kernel and shared by both Kind instantiations.
template<int C, int BN, int KS, int SC>
struct MmqShared {
    static constexpr int PACKS = KS / 4, STRIDE = KS / 4 + 4, GROUPS = KS / 32;
    unsigned int wq[MMQ_ROWS][STRIDE];
    unsigned int xq[C][BN][STRIDE];
    float coefficient[MMQ_ROWS][2 * GROUPS], minimum[MMQ_ROWS][GROUPS];
    float xs[BN][GROUPS][SC];
    // Q4_K minimum term per token: sum over components of activation_sum * scale.
    float msum[BN][GROUPS];
    int factor[BN][GROUPS];
    int destination[BN], source[BN];
};

#if __CUDA_ARCH__ >= 800
template<int Kind, int C, int BN, int KS, bool S16>
__device__ void mmq_tile(const u8 *weights, const signed char *values, const float *scales,
                         const int *sums, float *output, int width, int rows, int tokens,
                         const int *groups, int used, int grouped, int assignment_major,
                         int *status, MmqShared<C, BN, KS, S16 ? 2 : C> &sh) {
    // S16 metadata per 32 values: scales {d, unused}, sums {sum of codes, factor K}.
    constexpr int SC = S16 ? 2 : C;
    static_assert(!S16 || C == 2, "split16 uses two planes");
    using Shared = MmqShared<C, BN, KS, SC>;
    constexpr int THREADS = 256, NT = BN / 16, PACKS = Shared::PACKS, GROUPS = Shared::GROUPS;
    static_assert(BN % 16 == 0 && KS % 32 == 0, "tile shape");
    int tid = threadIdx.x, warp = tid / 32, lane = tid % 32;
    int group = lane / 4, thread = lane % 4;
    int expert = grouped ? groups[blockIdx.y * (BN + 1)] : 0;
    if (grouped && expert < 0) return;
    int first_row = blockIdx.x * MMQ_ROWS;
    int input_count = width * tokens * (grouped && assignment_major ? used : 1);
    for (int t = tid; t < BN; t += THREADS) {
        int assignment;
        if (grouped) {
            // Descriptors: expert, then BN assignment IDs (BN == 32 when grouped).
            assignment = groups[blockIdx.y * (BN + 1) + 1 + t];
        } else {
            int token = blockIdx.y * BN + t;
            assignment = token < tokens ? token : -1;
        }
        sh.destination[t] = assignment;
        sh.source[t] = grouped && !assignment_major && assignment >= 0 ? assignment / used : assignment;
    }
    int warp_row = (warp / 2) * 32, warp_token = (warp % 2) * (BN / 2);
    float acc[2][NT][4];
    #pragma unroll
    for (int m = 0; m < 2; m++)
        #pragma unroll
        for (int n = 0; n < NT; n++)
            #pragma unroll
            for (int i = 0; i < 4; i++) acc[m][n][i] = 0.0f;
    __syncthreads();
    // Raw operands of the next K step are fetched into registers while the
    // current step computes; unpacking happens when they are stored.
    constexpr int W_ITEMS = (MMQ_ROWS * PACKS + THREADS - 1) / THREADS;
    constexpr int P_ITEMS = (MMQ_ROWS * GROUPS + THREADS - 1) / THREADS;
    constexpr int X_ITEMS = (C * BN * PACKS + THREADS - 1) / THREADS;
    constexpr int S_ITEMS = (BN * GROUPS + THREADS - 1) / THREADS;
    constexpr int W_WORDS = Kind == 12 ? 1 : 2;
    unsigned int wraw[W_ITEMS][W_WORDS], praw[P_ITEMS][4], xraw[X_ITEMS];
    float sraw[S_ITEMS][SC];
    int mraw[S_ITEMS][SC];
    auto block_of = [&](int row, int k) {
        return weights + (((u64)expert * rows + first_row + row) * width + k) / 256
            * (Kind == 12 ? 144 : 210);
    };
    auto fetch = [&](int k) {
        #pragma unroll
        for (int j = 0; j < W_ITEMS; j++) {
            int index = tid + j * THREADS, row = index / PACKS, pack = index % PACKS;
            wraw[j][0] = 0u;
            if (W_WORDS > 1) wraw[j][W_WORDS - 1] = 0u;
            if (index < MMQ_ROWS * PACKS && first_row + row < rows) {
                const u8 *p = block_of(row, k);
                int i = k % 256 + pack * 4;
                if (Kind == 12) {
                    wraw[j][0] = *(const unsigned int *)(p + 16 + (i / 64) * 32 + i % 32);
                } else {
                    int h = i / 128, col = i % 32, quarter = (i % 128) / 32;
                    const unsigned short *lo = (const unsigned short *)(p + h * 64 + col + (quarter % 2) * 32);
                    const unsigned short *hi = (const unsigned short *)(p + 128 + h * 32 + col);
                    wraw[j][0] = (unsigned int)lo[0] | ((unsigned int)lo[1] << 16);
                    wraw[j][W_WORDS - 1] = (unsigned int)hi[0] | ((unsigned int)hi[1] << 16);
                }
            }
        }
        #pragma unroll
        for (int j = 0; j < P_ITEMS; j++) {
            int index = tid + j * THREADS, row = index / GROUPS, g = index % GROUPS;
            praw[j][0] = praw[j][1] = praw[j][2] = praw[j][3] = 0u;
            if (index < MMQ_ROWS * GROUPS && first_row + row < rows) {
                const u8 *p = block_of(row, k);
                if (Kind == 12) {
                    const unsigned int *words = (const unsigned int *)p;
                    praw[j][0] = words[0];
                    praw[j][1] = words[1];
                    praw[j][2] = words[2];
                    praw[j][3] = words[3];
                } else {
                    int i = k % 256 + g * 32;
                    praw[j][0] = *(const unsigned short *)(p + 208);
                    praw[j][1] = *(const unsigned short *)(p + 192 + i / 16);
                }
            }
        }
        #pragma unroll
        for (int j = 0; j < X_ITEMS; j++) {
            int index = tid + j * THREADS;
            int c = index / (BN * PACKS), rest = index % (BN * PACKS);
            int t = rest / PACKS, pack = rest % PACKS;
            int s = index < C * BN * PACKS ? sh.source[t] : -1;
            xraw[j] = s >= 0
                ? *(const unsigned int *)(values + (u64)c * input_count + (u64)s * width + k + pack * 4)
                : 0u;
        }
        #pragma unroll
        for (int j = 0; j < S_ITEMS; j++) {
            int index = tid + j * THREADS, t = index / GROUPS, g = index % GROUPS;
            int s = index < BN * GROUPS ? sh.source[t] : -1;
            u64 offset = ((u64)(s < 0 ? 0 : s) * width + k) / 32 * SC + g * SC;
            #pragma unroll
            for (int c = 0; c < SC; c++) {
                sraw[j][c] = s >= 0 ? scales[offset + c] : 0.0f;
                mraw[j][c] = s >= 0 ? sums[offset + c] : 0;
            }
        }
    };
    auto store = [&](int k) {
        #pragma unroll
        for (int j = 0; j < W_ITEMS; j++) {
            int index = tid + j * THREADS, row = index / PACKS, pack = index % PACKS;
            if (index >= MMQ_ROWS * PACKS) continue;
            int i = k % 256 + pack * 4;
            unsigned int packed;
            if (Kind == 12) {
                packed = (wraw[j][0] >> (4 * ((i / 32) % 2))) & 0x0f0f0f0fu;
            } else {
                int quarter = (i % 128) / 32;
                unsigned int bits = ((wraw[j][0] >> (4 * (quarter / 2))) & 0x0f0f0f0fu)
                    | (((wraw[j][W_WORDS - 1] >> (2 * quarter)) & 0x03030303u) << 4);
                // Same signed bias as residual_packed_quant<14>.
                packed = (bits + 0x60606060u) ^ 0x80808080u;
            }
            sh.wq[row][pack] = packed;
        }
        #pragma unroll
        for (int j = 0; j < P_ITEMS; j++) {
            int index = tid + j * THREADS, row = index / GROUPS, g = index % GROUPS;
            if (index >= MMQ_ROWS * GROUPS) continue;
            float a0, a1 = 0.0f, b = 0.0f;
            if (Kind == 12) {
                int gi = (k % 256) / 32 + g;
                #define MMQ_SCALE_BYTE(n) ((int)((praw[j][1 + (n) / 4] >> (8 * ((n) % 4))) & 255))
                int sc = gi < 4 ? MMQ_SCALE_BYTE(gi) & 63
                                : (MMQ_SCALE_BYTE(gi + 4) & 15) | ((MMQ_SCALE_BYTE(gi - 4) >> 6) << 4);
                int mn = gi < 4 ? MMQ_SCALE_BYTE(gi + 4) & 63
                                : (MMQ_SCALE_BYTE(gi + 4) >> 4) | ((MMQ_SCALE_BYTE(gi) >> 6) << 4);
                #undef MMQ_SCALE_BYTE
                a0 = half_value_bits(praw[j][0] & 0xffff) * (float)sc;
                b = half_value_bits(praw[j][0] >> 16) * (float)mn;
            } else {
                float d = half_value_bits(praw[j][0]);
                a0 = d * (float)(signed char)(praw[j][1] & 255);
                a1 = d * (float)(signed char)(praw[j][1] >> 8);
            }
            sh.coefficient[row][2 * g] = a0;
            sh.coefficient[row][2 * g + 1] = Kind == 12 ? a0 : a1;
            sh.minimum[row][g] = b;
        }
        #pragma unroll
        for (int j = 0; j < X_ITEMS; j++) {
            int index = tid + j * THREADS;
            if (index >= C * BN * PACKS) continue;
            int c = index / (BN * PACKS), rest = index % (BN * PACKS);
            sh.xq[c][rest / PACKS][rest % PACKS] = xraw[j];
        }
        #pragma unroll
        for (int j = 0; j < S_ITEMS; j++) {
            int index = tid + j * THREADS, t = index / GROUPS, g = index % GROUPS;
            if (index >= BN * GROUPS) continue;
            float m = 0.0f;
            if (S16) {
                sh.xs[t][g][0] = sraw[j][0];
                sh.factor[t][g] = mraw[j][1];
                m = (float)mraw[j][0] * sraw[j][0];
            } else {
                #pragma unroll
                for (int c = 0; c < SC; c++) {
                    sh.xs[t][g][c] = sraw[j][c];
                    if (Kind == 12) m += (float)mraw[j][c] * sraw[j][c];
                }
            }
            sh.msum[t][g] = Kind == 12 ? m : 0.0f;
        }
    };
    fetch(0);
    store(0);
    __syncthreads();
    for (int k = 0; k < width; k += KS) {
        bool more = k + KS < width;
        if (more) fetch(k + KS);
        #pragma unroll
        for (int g = 0; g < GROUPS; g++) {
            #pragma unroll
            for (int half = 0; half < (Kind == 12 ? 1 : 2); half++) {
                int pack = g * 8 + half * 4 + thread;
                unsigned int a[2][4];
                float coef[2][2], mins[2][2];
                #pragma unroll
                for (int m = 0; m < 2; m++) {
                    int r = warp_row + m * 16 + group;
                    a[m][0] = sh.wq[r][pack];
                    a[m][1] = sh.wq[r + 8][pack];
                    a[m][2] = Kind == 12 ? sh.wq[r][pack + 4] : 0u;
                    a[m][3] = Kind == 12 ? sh.wq[r + 8][pack + 4] : 0u;
                    #pragma unroll
                    for (int h = 0; h < 2; h++) {
                        coef[m][h] = sh.coefficient[r + 8 * h][2 * g + half];
                        mins[m][h] = Kind == 12 ? sh.minimum[r + 8 * h][g] : 0.0f;
                    }
                }
                #pragma unroll
                for (int n = 0; n < NT; n++) {
                    int tb = warp_token + n * 8, t0 = tb + thread * 2;
                    float combined[2][4];
                    #pragma unroll
                    for (int m = 0; m < 2; m++)
                        #pragma unroll
                        for (int i = 0; i < 4; i++) combined[m][i] = 0.0f;
                    if (S16) {
                        unsigned int h0 = sh.xq[0][tb + group][pack];
                        unsigned int h1 = Kind == 12 ? sh.xq[0][tb + group][pack + 4] : 0u;
                        unsigned int l0 = sh.xq[1][tb + group][pack];
                        unsigned int l1 = Kind == 12 ? sh.xq[1][tb + group][pack + 4] : 0u;
                        float s0 = sh.xs[t0][g][0], s1 = sh.xs[t0 + 1][g][0];
                        int e0 = sh.factor[t0][g], e1 = sh.factor[t0 + 1][g];
                        #pragma unroll
                        for (int m = 0; m < 2; m++) {
                            int dot[4] = {0, 0, 0, 0};
                            residual_mma_group<Kind>(a[m][0], a[m][1], a[m][2], a[m][3], h0, h1, dot);
                            dot[0] *= e0;
                            dot[1] *= e1;
                            dot[2] *= e0;
                            dot[3] *= e1;
                            residual_mma_group<Kind>(a[m][0], a[m][1], a[m][2], a[m][3], l0, l1, dot);
                            combined[m][0] = (float)dot[0] * s0;
                            combined[m][1] = (float)dot[1] * s1;
                            combined[m][2] = (float)dot[2] * s0;
                            combined[m][3] = (float)dot[3] * s1;
                        }
                    }
                    #pragma unroll
                    for (int c = 0; c < (S16 ? 0 : C); c++) {
                        unsigned int b0 = sh.xq[c][tb + group][pack];
                        unsigned int b1 = Kind == 12 ? sh.xq[c][tb + group][pack + 4] : 0u;
                        float s0 = sh.xs[t0][g][S16 ? 0 : c], s1 = sh.xs[t0 + 1][g][S16 ? 0 : c];
                        #pragma unroll
                        for (int m = 0; m < 2; m++) {
                            int dot[4] = {0, 0, 0, 0};
                            residual_mma_group<Kind>(a[m][0], a[m][1], a[m][2], a[m][3], b0, b1, dot);
                            combined[m][0] += (float)dot[0] * s0;
                            combined[m][1] += (float)dot[1] * s1;
                            combined[m][2] += (float)dot[2] * s0;
                            combined[m][3] += (float)dot[3] * s1;
                        }
                    }
                    float m0 = Kind == 12 && half == 0 ? sh.msum[t0][g] : 0.0f;
                    float m1 = Kind == 12 && half == 0 ? sh.msum[t0 + 1][g] : 0.0f;
                    #pragma unroll
                    for (int m = 0; m < 2; m++) {
                        #pragma unroll
                        for (int i = 0; i < 4; i++) {
                            float value = coef[m][i / 2] * combined[m][i];
                            if (Kind == 12) value -= mins[m][i / 2] * (i % 2 ? m1 : m0);
                            acc[m][n][i] += value;
                        }
                    }
                }
            }
        }
        __syncthreads();
        if (more) {
            store(k + KS);
            __syncthreads();
        }
    }
    #pragma unroll
    for (int m = 0; m < 2; m++) {
        #pragma unroll
        for (int n = 0; n < NT; n++) {
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                int row = first_row + warp_row + m * 16 + group + (i / 2) * 8;
                int assignment = sh.destination[warp_token + n * 8 + thread * 2 + i % 2];
                if (row < rows && assignment >= 0) {
                    float result = acc[m][n][i];
                    if (!isfinite(result)) atomicOr(status, 4);
                    output[(u64)assignment * rows + row] = result;
                }
            }
        }
    }
}
#endif

// 256 threads; grid = (ceil(rows / 128), grouped ? slots : ceil(tokens / BN)).
// The activation layout must match the quantizer: C = 3 residual components
// (residual_quantize32_precise3) or the S16 pair (quantize32_residual_pair).
#if __CUDA_ARCH__ >= 800
#define MMQ_BODY(C, BN, KS, S16) \
    __shared__ MmqShared<C, BN, KS, S16 ? 2 : C> sh; \
    if (kind == 12) mmq_tile<12, C, BN, KS, S16>(weights, values, scales, sums, output, width, rows, tokens, groups, used, grouped, assignment_major, status, sh); \
    if (kind == 14) mmq_tile<14, C, BN, KS, S16>(weights, values, scales, sums, output, width, rows, tokens, groups, used, grouped, assignment_major, status, sh); \
    if (kind != 12 && kind != 14 && threadIdx.x == 0) atomicOr(status, 8);
#else
#define MMQ_BODY(C, BN, KS, S16) if (threadIdx.x == 0) atomicOr(status, 16);
#endif
#define MMQ_KERNEL(Name, C, BN, KS, S16) MMQ_KERNEL_BLOCKS(Name, C, BN, KS, S16, 2)
#define MMQ_KERNEL_BLOCKS(Name, C, BN, KS, S16, Blocks) \
extern "C" __global__ void __launch_bounds__(256, Blocks) \
Name(const u8 *weights, int kind, const signed char *values, const float *scales, \
     const int *sums, float *output, int width, int rows, int tokens, const int *groups, \
     int used, int grouped, int assignment_major, int *status) { \
    MMQ_BODY(C, BN, KS, S16) \
}
// int8-prefill (S16 activations, quantize32_residual_pair). Tiles measured on
// Qwen3-30B: KS 128 for 64-token tiles was slower, 32-token tiles beat 16 and 64
// for few-row projections, and expert slots preferred KS 128 over 64 or 256.
MMQ_KERNEL(mmq_s16, 2, 64, 64, true)
MMQ_KERNEL(mmq_s16_rows32, 2, 32, 128, true)
MMQ_KERNEL(mmq_s16_grouped, 2, 32, 128, true)
// Three residual components (residual_quantize32_precise3) for FFN inputs.
MMQ_KERNEL(mmq_c3, 3, 64, 64, false)
MMQ_KERNEL(mmq_c3_rows32, 3, 32, 128, false)
#ifndef MMQ_C3_GROUPED_BLOCKS
#define MMQ_C3_GROUPED_BLOCKS 2
#endif
MMQ_KERNEL_BLOCKS(mmq_c3_grouped, 3, 32, 128, false, MMQ_C3_GROUPED_BLOCKS)

// Two residual s8 planes per 32-value warp block with integer-related scales:
// d1 = max|x| / 127, q1 = round(x / d1), r = x - d1 * q1, K = floor(127 * d1 / max|r|)
// clamped to [1, 32768] (so |q2| <= 127), d = d1 / K, q2 = round(r / d).
// x ~ d * (K * q1 + q2): within K / (K + 1) of the adaptive two-component residual,
// which matters for blocks with large outliers (a uniform int16 code measured
// max_abs 1.05 on real text, a power-of-two factor 0.19).
// q[i] = q1, q[count + i] = q2; scales[2b] = d; sums[2b] = sum of K * q1 + q2,
// sums[2b + 1] = K for block b = i / 32. count % 32 == 0.
#define MMQ_PAIR_FACTOR_MAX 32768
extern "C" __global__ void quantize32_residual_pair(const float *x, signed char *q,
                                                    float *scales, int *sums, int count,
                                                    int *status) {
    int i = blockIdx.x * blockDim.x + threadIdx.x, lane = threadIdx.x % 32;
    if (i >= count) return;
    float value = x[i];
    if (__any_sync(0xffffffff, !isfinite(value))) {
        if (lane == 0) atomicOr(status, 1);
        return;
    }
    float maximum = fabsf(value);
    for (int stride = 16; stride > 0; stride /= 2)
        maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffff, maximum, stride));
    float d1 = maximum == 0.0f ? 0.0f : fmaxf(__fdiv_rn(maximum, 127.0f), __uint_as_float(1));
    int q1 = d1 == 0.0f ? 0 : max(-127, min(127, __float2int_rn(__fdiv_rn(value, d1))));
    float r = fmaf(-d1, (float)q1, value);
    float rmax = fabsf(r);
    for (int stride = 16; stride > 0; stride /= 2)
        rmax = fmaxf(rmax, __shfl_xor_sync(0xffffffff, rmax, stride));
    int factor = MMQ_PAIR_FACTOR_MAX;
    if (rmax > 0.0f) {
        float ratio = __fdiv_rn(127.0f * d1, rmax);
        factor = ratio >= (float)MMQ_PAIR_FACTOR_MAX ? MMQ_PAIR_FACTOR_MAX : max(1, (int)ratio);
    }
    float d = __fdiv_rn(d1, (float)factor);
    int q2 = d == 0.0f ? 0 : max(-127, min(127, __float2int_rn(__fdiv_rn(r, d))));
    q[i] = (signed char)q1;
    q[count + i] = (signed char)q2;
    int code = q1 * factor + q2;
    for (int stride = 16; stride > 0; stride /= 2) code += __shfl_down_sync(0xffffffff, code, stride);
    if (lane == 0) {
        scales[2 * (i / 32)] = d;
        scales[2 * (i / 32) + 1] = 0.0f;
        sums[2 * (i / 32)] = code;
        sums[2 * (i / 32) + 1] = factor;
    }
}
