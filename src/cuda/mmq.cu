// Integer tensor-core prefill projection for packed Q4_K / Q6_K weights and
// residual INT8 activations (residual_quantize32_precise<C>): same ABI as
// residual_matmul_mma3. Each block stages a 128-row x KS weight tile and a
// BN-token activation tile per component; 8 warps (4 along rows x 2 along
// tokens) issue s8 MMAs whose integer dot products are exact per 32-value (Q4_K)
// or 16-value (Q6_K) scale group. Each output accumulates in F32
//   coefficient * sum_c(dot_c * scale_c) - minimum * sum_c(sum_c * scale_c).
#define MMQ_ROWS 128

// Declared once per kernel and shared by both Kind instantiations.
template<int C, int BN, int KS>
struct MmqShared {
    static constexpr int PACKS = KS / 4, STRIDE = KS / 4 + 4, GROUPS = KS / 32;
    unsigned int wq[MMQ_ROWS][STRIDE];
    unsigned int xq[C][BN][STRIDE];
    float coefficient[MMQ_ROWS][2 * GROUPS], minimum[MMQ_ROWS][GROUPS];
    float xs[BN][GROUPS][C];
    // Q4_K minimum term per token: sum over components of activation_sum * scale.
    float msum[BN][GROUPS];
    int destination[BN], source[BN];
};

#if __CUDA_ARCH__ >= 800
template<int Kind, int C, int BN, int KS>
__device__ void mmq_tile(const u8 *weights, const signed char *values, const float *scales,
                         const int *sums, float *output, int width, int rows, int tokens,
                         const int *groups, int used, int grouped, int assignment_major,
                         int *status, MmqShared<C, BN, KS> &sh) {
    using Shared = MmqShared<C, BN, KS>;
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
    float sraw[S_ITEMS][C];
    int mraw[S_ITEMS][C];
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
            u64 offset = ((u64)(s < 0 ? 0 : s) * width + k) / 32 * C + g * C;
            #pragma unroll
            for (int c = 0; c < C; c++) {
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
            #pragma unroll
            for (int c = 0; c < C; c++) {
                sh.xs[t][g][c] = sraw[j][c];
                if (Kind == 12) m += (float)mraw[j][c] * sraw[j][c];
            }
            sh.msum[t][g] = m;
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
                    #pragma unroll
                    for (int c = 0; c < C; c++) {
                        unsigned int b0 = sh.xq[c][tb + group][pack];
                        unsigned int b1 = Kind == 12 ? sh.xq[c][tb + group][pack + 4] : 0u;
                        float s0 = sh.xs[t0][g][c], s1 = sh.xs[t0 + 1][g][c];
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
// The activation components C must match residual_quantize32_precise<C>.
#if __CUDA_ARCH__ >= 800
#define MMQ_BODY(C, BN, KS) \
    __shared__ MmqShared<C, BN, KS> sh; \
    if (kind == 12) mmq_tile<12, C, BN, KS>(weights, values, scales, sums, output, width, rows, tokens, groups, used, grouped, assignment_major, status, sh); \
    if (kind == 14) mmq_tile<14, C, BN, KS>(weights, values, scales, sums, output, width, rows, tokens, groups, used, grouped, assignment_major, status, sh); \
    if (kind != 12 && kind != 14 && threadIdx.x == 0) atomicOr(status, 8);
#else
#define MMQ_BODY(C, BN, KS) if (threadIdx.x == 0) atomicOr(status, 16);
#endif
#define MMQ_KERNEL(Name, C, BN, KS) \
extern "C" __global__ void __launch_bounds__(256, 2) \
Name(const u8 *weights, int kind, const signed char *values, const float *scales, \
     const int *sums, float *output, int width, int rows, int tokens, const int *groups, \
     int used, int grouped, int assignment_major, int *status) { \
    MMQ_BODY(C, BN, KS) \
}
// Dense projections with many token tiles.
MMQ_KERNEL(residual_matmul_mmq, 3, 64, 64)
// Dense projections with few rows (K/V): more, smaller token tiles.
MMQ_KERNEL(residual_matmul_mmq16, 3, 16, 128)
// Expert slots of 32-assignment descriptors (group_experts_fast32).
MMQ_KERNEL(residual_matmul_mmq_grouped, 3, 32, 128)
