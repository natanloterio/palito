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
// Prefetch: load the next K tile into registers during the MMAs (more registers);
// without it the block loads, stores and computes in turn and relies on occupancy.
// SplitK (dense only): blockIdx.z covers width / gridDim.z columns (a multiple of
// KS and 256) and writes partial sums to output + blockIdx.z * tokens * rows.
template<int Kind, int C, int BN, int KS, bool S16, bool Prefetch = true, bool SplitK = false>
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
    int k_begin = SplitK ? blockIdx.z * (width / gridDim.z) : 0;
    int k_end = SplitK ? k_begin + width / gridDim.z : width;
    if (SplitK) output += (u64)blockIdx.z * tokens * rows;
    fetch(k_begin);
    store(k_begin);
    __syncthreads();
    for (int k = k_begin; k < k_end; k += KS) {
        bool more = k + KS < k_end;
        if (!Prefetch && k > k_begin) {
            fetch(k);
            store(k);
            __syncthreads();
        }
        if (Prefetch && more) fetch(k + KS);
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
        if (Prefetch && more) {
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

#if __CUDA_ARCH__ >= 800
// int8-fast Q4_K tiles with a two-stage cp.async pipeline: the raw Q4_K nibbles and
// block headers, the s8 activations and their scales / code sums of K step i + 1
// stream into shared memory while step i computes (mmq_tile without prefetch loads,
// unpacks and computes in turn at one block per SM). Same arithmetic as mmq_tile<12, 1>.
__device__ __forceinline__ void mmq_copy(void *dst, const void *src, bool valid, int bytes) {
    unsigned int d = (unsigned int)__cvta_generic_to_shared(dst);
    int size = valid ? bytes : 0;
    if (bytes == 16)
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" :: "r"(d), "l"(src), "r"(size) : "memory");
    else
        asm volatile("cp.async.ca.shared.global [%0], [%1], 8, %2;" :: "r"(d), "l"(src), "r"(size) : "memory");
}

template<int BN, int KS>
struct MmqPipeShared {
    // Raw nibble words per row (KS / 2 bytes) padded to a conflict-free stride.
    static constexpr int GROUPS = KS / 32, WSTRIDE = KS / 8 + 4, XSTRIDE = KS / 4 + 4;
    unsigned int w[2][MMQ_ROWS][WSTRIDE];
    uint4 header[2][MMQ_ROWS];
    unsigned int x[2][BN][XSTRIDE];
    float xs[2][BN][GROUPS];
    int xsum[2][BN][GROUPS];
    float coefficient[MMQ_ROWS][GROUPS], minimum[MMQ_ROWS][GROUPS], msum[BN][GROUPS];
    int destination[BN], source[BN];
};

template<int BN, int KS, bool SplitK>
__device__ void mmq_pipe_q4(const u8 *weights, const signed char *values, const float *scales,
                            const int *sums, float *output, int width, int rows, int tokens,
                            const int *groups, int used, int grouped, int assignment_major,
                            int *status, MmqPipeShared<BN, KS> &sh) {
    static_assert(KS == 64 || KS == 128, "pipe K step");
    constexpr int THREADS = 256, NT = BN / 16, GROUPS = KS / 32;
    constexpr int WCHUNKS = KS / 32, XCHUNKS = KS / 16, SBYTES = GROUPS * 4;
    int tid = threadIdx.x, warp = tid / 32, lane = tid % 32;
    int group = lane / 4, thread = lane % 4;
    int expert = grouped ? groups[blockIdx.y * (BN + 1)] : 0;
    if (grouped && expert < 0) return;
    int first_row = blockIdx.x * MMQ_ROWS;
    for (int t = tid; t < BN; t += THREADS) {
        int assignment;
        if (grouped) {
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
    auto issue = [&](int stage, int k) {
        for (int i = tid; i < MMQ_ROWS * (WCHUNKS + 1); i += THREADS) {
            int row = i / (WCHUNKS + 1), c = i % (WCHUNKS + 1);
            bool valid = first_row + row < rows;
            const u8 *p = weights + (((u64)expert * rows + first_row + (valid ? row : 0)) * width + k)
                / 256 * 144;
            if (c == WCHUNKS) mmq_copy(&sh.header[stage][row], p, valid, 16);
            else mmq_copy(&sh.w[stage][row][c * 4], p + 16 + (k % 256) / 2 + c * 16, valid, 16);
        }
        for (int i = tid; i < BN * XCHUNKS; i += THREADS) {
            int t = i / XCHUNKS, c = i % XCHUNKS, s = sh.source[t];
            mmq_copy(&sh.x[stage][t][c * 4], values + (u64)(s < 0 ? 0 : s) * width + k + c * 16, s >= 0, 16);
        }
        for (int i = tid; i < 2 * BN; i += THREADS) {
            int t = i % BN, s = sh.source[t];
            u64 offset = ((u64)(s < 0 ? 0 : s) * width + k) / 32;
            if (i < BN) mmq_copy(sh.xs[stage][t], scales + offset, s >= 0, SBYTES);
            else mmq_copy(sh.xsum[stage][t], sums + offset, s >= 0, SBYTES);
        }
        asm volatile("cp.async.commit_group;" ::: "memory");
    };
    auto convert = [&](int stage, int k) {
        for (int i = tid; i < MMQ_ROWS * GROUPS; i += THREADS) {
            int row = i / GROUPS, g = i % GROUPS;
            uint4 h = sh.header[stage][row];
            unsigned int praw[4] = {h.x, h.y, h.z, h.w};
            int gi = (k % 256) / 32 + g;
            #define MMQ_SCALE_BYTE(n) ((int)((praw[1 + (n) / 4] >> (8 * ((n) % 4))) & 255))
            int sc = gi < 4 ? MMQ_SCALE_BYTE(gi) & 63
                            : (MMQ_SCALE_BYTE(gi + 4) & 15) | ((MMQ_SCALE_BYTE(gi - 4) >> 6) << 4);
            int mn = gi < 4 ? MMQ_SCALE_BYTE(gi + 4) & 63
                            : (MMQ_SCALE_BYTE(gi + 4) >> 4) | ((MMQ_SCALE_BYTE(gi) >> 6) << 4);
            #undef MMQ_SCALE_BYTE
            sh.coefficient[row][g] = half_value_bits(praw[0] & 0xffff) * (float)sc;
            sh.minimum[row][g] = half_value_bits(praw[0] >> 16) * (float)mn;
        }
        for (int i = tid; i < BN * GROUPS; i += THREADS) {
            int t = i / GROUPS, g = i % GROUPS;
            float m = 0.0f;
            m += (float)sh.xsum[stage][t][g] * sh.xs[stage][t][g];
            sh.msum[t][g] = m;
        }
    };
    int k_begin = SplitK ? blockIdx.z * (width / gridDim.z) : 0;
    int k_end = SplitK ? k_begin + width / gridDim.z : width;
    if (SplitK) output += (u64)blockIdx.z * tokens * rows;
    int steps = (k_end - k_begin) / KS;
    issue(0, k_begin);
    for (int step = 0; step < steps; step++) {
        int stage = step & 1, k = k_begin + step * KS;
        if (step + 1 < steps) {
            issue(stage ^ 1, k + KS);
            asm volatile("cp.async.wait_group 1;" ::: "memory");
        } else {
            asm volatile("cp.async.wait_group 0;" ::: "memory");
        }
        __syncthreads();
        convert(stage, k);
        __syncthreads();
        #pragma unroll
        for (int g = 0; g < GROUPS; g++) {
            int pack = g * 8 + thread, word = (g / 2) * 8 + thread, shift = 4 * (g % 2);
            unsigned int a[2][4];
            float coef[2][2], mins[2][2];
            #pragma unroll
            for (int m = 0; m < 2; m++) {
                int r = warp_row + m * 16 + group;
                a[m][0] = (sh.w[stage][r][word] >> shift) & 0x0f0f0f0fu;
                a[m][1] = (sh.w[stage][r + 8][word] >> shift) & 0x0f0f0f0fu;
                a[m][2] = (sh.w[stage][r][word + 4] >> shift) & 0x0f0f0f0fu;
                a[m][3] = (sh.w[stage][r + 8][word + 4] >> shift) & 0x0f0f0f0fu;
                #pragma unroll
                for (int h = 0; h < 2; h++) {
                    coef[m][h] = sh.coefficient[r + 8 * h][g];
                    mins[m][h] = sh.minimum[r + 8 * h][g];
                }
            }
            #pragma unroll
            for (int n = 0; n < NT; n++) {
                int tb = warp_token + n * 8, t0 = tb + thread * 2;
                unsigned int b0 = sh.x[stage][tb + group][pack];
                unsigned int b1 = sh.x[stage][tb + group][pack + 4];
                float s0 = sh.xs[stage][t0][g], s1 = sh.xs[stage][t0 + 1][g];
                float m0 = sh.msum[t0][g], m1 = sh.msum[t0 + 1][g];
                #pragma unroll
                for (int m = 0; m < 2; m++) {
                    float combined[4] = {0.0f, 0.0f, 0.0f, 0.0f};
                    int dot[4] = {0, 0, 0, 0};
                    residual_mma_group<12>(a[m][0], a[m][1], a[m][2], a[m][3], b0, b1, dot);
                    combined[0] += (float)dot[0] * s0;
                    combined[1] += (float)dot[1] * s1;
                    combined[2] += (float)dot[2] * s0;
                    combined[3] += (float)dot[3] * s1;
                    #pragma unroll
                    for (int i = 0; i < 4; i++) {
                        float value = coef[m][i / 2] * combined[i];
                        value -= mins[m][i / 2] * (i % 2 ? m1 : m0);
                        acc[m][n][i] += value;
                    }
                }
            }
        }
        __syncthreads();
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

// int8-fast pipelined tiles: Q4_K through mmq_pipe_q4, Q6_K through mmq_tile
// without prefetch (210-byte blocks are not 16-byte aligned for cp.async).
#if __CUDA_ARCH__ >= 800
// KQ: Q4_K K step (mmq_pipe_q4); KS: Q6_K K step (mmq_tile).
#define MMQ_PIPE_SIZE(BN, KQ, KS) (sizeof(MmqPipeShared<BN, KQ>) > sizeof(MmqShared<1, BN, KS, 1>) \
    ? sizeof(MmqPipeShared<BN, KQ>) : sizeof(MmqShared<1, BN, KS, 1>))
#define MMQ_PIPE_BODY(BN, KQ, KS, SK, PF) \
    __shared__ __align__(16) unsigned char raw[MMQ_PIPE_SIZE(BN, KQ, KS)]; \
    MMQ_PIPE_DISPATCH(BN, KQ, KS, SK, PF)
// Dynamic shared memory (> 48 KB): the host sizes it from mmq_dynamic_shared.
#define MMQ_PIPE_BODY_DYNAMIC(BN, KQ, KS, SK, PF) \
    extern __shared__ __align__(16) unsigned char raw[]; \
    MMQ_PIPE_DISPATCH(BN, KQ, KS, SK, PF)
#define MMQ_PIPE_DISPATCH(BN, KQ, KS, SK, PF) \
    if (kind == 12) mmq_pipe_q4<BN, KQ, SK>(weights, values, scales, sums, output, width, rows, tokens, groups, used, grouped, assignment_major, status, *reinterpret_cast<MmqPipeShared<BN, KQ> *>(raw)); \
    if (kind == 14) mmq_tile<14, 1, BN, KS, false, PF, SK>(weights, values, scales, sums, output, width, rows, tokens, groups, used, grouped, assignment_major, status, *reinterpret_cast<MmqShared<1, BN, KS, 1> *>(raw)); \
    if (kind != 12 && kind != 14 && threadIdx.x == 0) atomicOr(status, 8);
#else
#define MMQ_PIPE_BODY(BN, KQ, KS, SK, PF) if (threadIdx.x == 0) atomicOr(status, 16);
#define MMQ_PIPE_BODY_DYNAMIC(BN, KQ, KS, SK, PF) if (threadIdx.x == 0) atomicOr(status, 16);
#define MMQ_PIPE_SIZE(BN, KQ, KS) 0
#endif
#define MMQ_PIPE_KERNEL(Name, BN, KS, SK) MMQ_PIPE_KERNEL_BLOCKS(Name, BN, KS, KS, SK, 1, false)
#define MMQ_PIPE_KERNEL_DYNAMIC(Name, BN, KQ, KS, SK) \
extern "C" __global__ void __launch_bounds__(256, 1) \
Name(const u8 *weights, int kind, const signed char *values, const float *scales, \
     const int *sums, float *output, int width, int rows, int tokens, const int *groups, \
     int used, int grouped, int assignment_major, int *status) { \
    MMQ_PIPE_BODY_DYNAMIC(BN, KQ, KS, SK, false) \
}
#define MMQ_PIPE_KERNEL_BLOCKS(Name, BN, KQ, KS, SK, Blocks, PF) \
extern "C" __global__ void __launch_bounds__(256, Blocks) \
Name(const u8 *weights, int kind, const signed char *values, const float *scales, \
     const int *sums, float *output, int width, int rows, int tokens, const int *groups, \
     int used, int grouped, int assignment_major, int *status) { \
    MMQ_PIPE_BODY(BN, KQ, KS, SK, PF) \
}

// 256 threads; grid = (ceil(rows / 128), grouped ? slots : ceil(tokens / BN)).
// The activation layout must match the quantizer: C = 3 residual components
// (residual_quantize32_precise3) or the S16 pair (quantize32_residual_pair).
#if __CUDA_ARCH__ >= 800
#define MMQ_BODY(C, BN, KS, S16, PF, SK) \
    __shared__ MmqShared<C, BN, KS, S16 ? 2 : C> sh; \
    if (kind == 12) mmq_tile<12, C, BN, KS, S16, PF, SK>(weights, values, scales, sums, output, width, rows, tokens, groups, used, grouped, assignment_major, status, sh); \
    if (kind == 14) mmq_tile<14, C, BN, KS, S16, PF, SK>(weights, values, scales, sums, output, width, rows, tokens, groups, used, grouped, assignment_major, status, sh); \
    if (kind != 12 && kind != 14 && threadIdx.x == 0) atomicOr(status, 8);
#else
#define MMQ_BODY(C, BN, KS, S16, PF, SK) if (threadIdx.x == 0) atomicOr(status, 16);
#endif
#define MMQ_KERNEL(Name, C, BN, KS, S16) MMQ_KERNEL_BLOCKS(Name, C, BN, KS, S16, 2)
#define MMQ_KERNEL_BLOCKS(Name, C, BN, KS, S16, Blocks) MMQ_KERNEL_FULL(Name, C, BN, KS, S16, Blocks, true)
#define MMQ_KERNEL_FULL(Name, C, BN, KS, S16, Blocks, PF) MMQ_KERNEL_SPLIT(Name, C, BN, KS, S16, Blocks, PF, false)
#define MMQ_KERNEL_SPLIT(Name, C, BN, KS, S16, Blocks, PF, SK) \
extern "C" __global__ void __launch_bounds__(256, Blocks) \
Name(const u8 *weights, int kind, const signed char *values, const float *scales, \
     const int *sums, float *output, int width, int rows, int tokens, const int *groups, \
     int used, int grouped, int assignment_major, int *status) { \
    MMQ_BODY(C, BN, KS, S16, PF, SK) \
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
// int8-fast: one s8 plane per 32 activations, as llama.cpp's q8_1, pipelined (Q4_K)
// at one block per SM for 64/128-token tiles (register prefetch spills with one
// plane), two for 32-token tiles. Split-K variants: few blocks (short prompts),
// gridDim.z K splits, then mmq_splitk_reduce.
// Q4_K takes 128-wide K steps (half the barriers of 64; Qwen3-30B 0.5k TTFT
// 48 -> 44 ms for the 32-token tiles), Q6_K (mmq_tile) 64. The 64/128-token tiles
// exceed 48 KB of shared memory: dynamic, sized by mmq_dynamic_shared.
MMQ_PIPE_KERNEL_DYNAMIC(mmq_c1_pipe128, 128, 128, 64, false)
MMQ_PIPE_KERNEL_DYNAMIC(mmq_c1_pipe128_splitk, 128, 128, 64, true)
MMQ_PIPE_KERNEL_DYNAMIC(mmq_c1_pipe64, 64, 128, 64, false)
MMQ_PIPE_KERNEL_BLOCKS(mmq_c1_pipe32, 32, 128, 64, false, 2, true)
MMQ_PIPE_KERNEL_BLOCKS(mmq_c1_pipe32_splitk, 32, 128, 64, true, 2, true)

// Dynamic shared bytes of mmq_c1_pipe128, mmq_c1_pipe128_splitk, mmq_c1_pipe64.
extern "C" __global__ void mmq_dynamic_shared(int *bytes) {
    bytes[0] = (int)MMQ_PIPE_SIZE(128, 128, 64);
    bytes[1] = (int)MMQ_PIPE_SIZE(128, 128, 64);
    bytes[2] = (int)MMQ_PIPE_SIZE(64, 128, 64);
}

// Sums gridDim-z partials of a split-K MMQ in a fixed order (deterministic).
extern "C" __global__ void mmq_splitk_reduce(const float *partials, float *output, int count,
                                             int splits, int *status) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    float sum = partials[i];
    for (int s = 1; s < splits; s++) sum += partials[(u64)s * count + i];
    if (!isfinite(sum)) atomicOr(status, 4);
    output[i] = sum;
}

extern "C" __global__ void quantize32_q8(const float *x, signed char *q, float *scales,
                                         int *sums, int count, int *status) {
    residual_quantize32_precise<1>(x, q, scales, sums, count, status);
}

// swiglu (same expression) fused with quantize32_q8 for the down-projection input:
// the activated values never reach global memory. count % 32 == 0.
extern "C" __global__ void swiglu_quantize32_q8(const float *gate, const float *up, signed char *q,
                                                float *scales, int *sums, int count, int *status) {
    int i = blockIdx.x * blockDim.x + threadIdx.x, lane = threadIdx.x % 32;
    if (i >= count) return;
    float g = gate[i];
    float x = g / (1.0f + expf(-g)) * up[i];
    if (__any_sync(0xffffffff, !isfinite(x))) {
        atomicOr(status, 1);
        return;
    }
    float maximum = fabsf(x);
    for (int stride = 16; stride > 0; stride /= 2)
        maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffff, maximum, stride));
    float d = maximum == 0.0f ? 0.0f : fmaxf(__fdiv_rn(maximum, 127.0f), __uint_as_float(1));
    int value = d == 0.0f ? 0 : max(-127, min(127, __float2int_rn(__fdiv_rn(x, d))));
    q[i] = (signed char)value;
    for (int stride = 16; stride > 0; stride /= 2)
        value += __shfl_down_sync(0xffffffff, value, stride);
    if (lane == 0) {
        scales[i / 32] = d;
        sums[i / 32] = value;
    }
}

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

// int8-fast: per-head RMS norm (head_norm) fused with rotary (rotary): one block of
// dim / 2 threads per head row, one rotary pair per thread. grid = rows of n tokens
// x (width / dim) heads; dim / 2 <= 1024 and a multiple of 32.
extern "C" __global__ void head_norm_rotary(float *x, const u8 *w, int dim, float epsilon,
                                            int width, int position, float base, int neox) {
    __shared__ float partial[32];
    int heads = width / dim, token = blockIdx.x / heads, pair = threadIdx.x;
    x += (u64)blockIdx.x * dim;
    const float *weight = (const float *)w;
    int a = neox ? pair : 2 * pair, b = neox ? pair + dim / 2 : 2 * pair + 1;
    float u = x[a], v = x[b];
    float sum = u * u + v * v;
    for (int stride = 16; stride > 0; stride /= 2) sum += __shfl_xor_sync(0xffffffff, sum, stride);
    if (threadIdx.x % 32 == 0) partial[threadIdx.x / 32] = sum;
    __syncthreads();
    sum = 0.0f;
    for (int i = 0; i < (int)blockDim.x / 32; i++) sum += partial[i];
    float inverse = 1.0f / sqrtf(sum / (float)dim + epsilon);
    u *= inverse * weight[a];
    v *= inverse * weight[b];
    float angle = (float)(position + token) * powf(base, -(float)(2 * pair) / (float)dim);
    float sine = sinf(angle), cosine = cosf(angle);
    x[a] = u * cosine - v * sine;
    x[b] = u * sine + v * cosine;
}

