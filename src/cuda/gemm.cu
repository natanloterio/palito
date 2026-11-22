// Register-blocked F32 GEMM for packed Q4_K / Q6_K prefill projections (f32-tiled
// profile only). Each block stages a BM x GEMM_K tile of decoded weights and a
// GEMM_K x BN tile of activations in shared memory (double-buffered: the next
// tile's raw bytes are fetched into registers while the current one is computed);
// each thread accumulates TM rows x TN tokens. Weights are decoded with the same
// arithmetic as projection_weights ((d * scale) * q - dmin * min for Q4_K,
// (d * scale) * q for Q6_K) and every output traverses K in increasing order with
// one accumulator, so results match projection_register_tiled bit for bit.
// 16 measured fastest on Qwen3-30B prefill (8: 1.53 s, 16: 1.09 s, 32: 1.22 s TTFT).
#define GEMM_K 16

// Raw bytes of four consecutive weights, fetched one tile ahead of their decode.
template<int Kind>
__device__ __forceinline__ void gemm_fetch(const u8 *w, u64 index, unsigned int raw[5]) {
    if (Kind == 12) {
        const unsigned int *p = (const unsigned int *)(w + (index / 256) * 144);
        int i = index % 256;
        raw[0] = p[0];
        raw[1] = p[1];
        raw[2] = p[2];
        raw[3] = p[3];
        raw[4] = p[4 + (i / 64) * 8 + (i % 32) / 4];
    } else {
        const u8 *p = w + (index / 256) * 210;
        int i = index % 256, h = i / 128, l = i % 32, quarter = (i % 128) / 32;
        const unsigned short *low = (const unsigned short *)(p + h * 64 + l + (quarter % 2) * 32);
        const unsigned short *high = (const unsigned short *)(p + 128 + h * 32 + l);
        raw[0] = (unsigned int)low[0] | ((unsigned int)low[1] << 16);
        raw[1] = (unsigned int)high[0] | ((unsigned int)high[1] << 16);
        raw[2] = p[192 + i / 16];
        raw[3] = *(const unsigned short *)(p + 208);
        raw[4] = 0;
    }
}

// Decodes four consecutive weights (k % 4 == 0) from fetched bytes.
template<int Kind>
__device__ __forceinline__ void gemm_unpack(const unsigned int raw[5], u64 index, float out[4]) {
    int i = index % 256;
    if (Kind == 12) {
        int g = i / 32;
        #define GEMM_SCALE_BYTE(j) ((int)((raw[1 + (j) / 4] >> (8 * ((j) % 4))) & 255))
        int sc = g < 4 ? GEMM_SCALE_BYTE(g) & 63
                       : (GEMM_SCALE_BYTE(g + 4) & 15) | ((GEMM_SCALE_BYTE(g - 4) >> 6) << 4);
        int mn = g < 4 ? GEMM_SCALE_BYTE(g + 4) & 63
                       : (GEMM_SCALE_BYTE(g + 4) >> 4) | ((GEMM_SCALE_BYTE(g) >> 6) << 4);
        #undef GEMM_SCALE_BYTE
        float d = half_value_bits(raw[0] & 0xffff), dmin = half_value_bits(raw[0] >> 16);
        float scale = d * (float)sc, offset = dmin * (float)mn;
        int shift = 4 * (g % 2);
        #pragma unroll
        for (int j = 0; j < 4; j++)
            out[j] = scale * (float)((raw[4] >> (8 * j + shift)) & 15) - offset;
    } else {
        int quarter = (i % 128) / 32;
        float scale = half_value_bits(raw[3]) * (float)(signed char)raw[2];
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            int low = (raw[0] >> (8 * j + 4 * (quarter / 2))) & 15;
            int high = (raw[1] >> (8 * j + 2 * quarter)) & 3;
            out[j] = scale * (float)((low | (high << 4)) - 32);
        }
    }
}

// Grouped: blockIdx.y is an expert slot of group_expert_tiles<BN> descriptors
// (expert, then BN assignment IDs or -1); inputs are rows assignment / used
// unless batched. Dense: blockIdx.y tiles tokens.
template<int Kind, int BM, int BN, int TM, int TN, bool Grouped>
__device__ __forceinline__ void gemm_packed(const u8 *w, const float *x, float *y, int width,
                                            int rows, int tokens, const int *groups, int used,
                                            int batched) {
    constexpr int THREADS = (BM / TM) * (BN / TN);
    constexpr int W_ITEMS = BM * (GEMM_K / 4) / THREADS, X_ITEMS = BN * (GEMM_K / 4) / THREADS;
    static_assert(W_ITEMS * THREADS == BM * (GEMM_K / 4), "weight items per thread");
    static_assert(X_ITEMS * THREADS == BN * (GEMM_K / 4), "input items per thread");
    __shared__ __align__(16) float weights[2][GEMM_K][BM + 4];
    __shared__ __align__(16) float inputs[2][GEMM_K][BN + 4];
    __shared__ int sources[BN];
    const int *descriptor = Grouped ? groups + (u64)blockIdx.y * (BN + 1) : nullptr;
    int expert = Grouped ? descriptor[0] : 0;
    if (expert < 0) return;
    int first_row = blockIdx.x * BM, first_token = Grouped ? 0 : blockIdx.y * BN;
    for (int t = threadIdx.x; t < BN; t += THREADS) {
        int source;
        if (Grouped) {
            int assignment = descriptor[t + 1];
            source = assignment >= 0 ? (batched ? assignment : assignment / used) : -1;
        } else {
            source = first_token + t < tokens ? first_token + t : -1;
        }
        sources[t] = source;
    }
    int tx = threadIdx.x % (BN / TN), ty = threadIdx.x / (BN / TN);
    float sums[TM][TN];
    #pragma unroll
    for (int r = 0; r < TM; r++) {
        #pragma unroll
        for (int t = 0; t < TN; t++) sums[r][t] = 0.0f;
    }
    __syncthreads();
    unsigned int raw[W_ITEMS][5];
    float4 staged[X_ITEMS];
    auto fetch = [&](int first_k) {
        #pragma unroll
        for (int n = 0; n < W_ITEMS; n++) {
            int item = threadIdx.x + n * THREADS;
            int r = item / (GEMM_K / 4), k = (item % (GEMM_K / 4)) * 4;
            if (first_row + r < rows) {
                gemm_fetch<Kind>(w, ((u64)expert * rows + first_row + r) * width + first_k + k,
                                 raw[n]);
            } else {
                #pragma unroll
                for (int j = 0; j < 5; j++) raw[n][j] = 0;
            }
        }
        #pragma unroll
        for (int n = 0; n < X_ITEMS; n++) {
            int item = threadIdx.x + n * THREADS;
            int t = item / (GEMM_K / 4), k = (item % (GEMM_K / 4)) * 4;
            int source = sources[t];
            staged[n] = source >= 0 ? *(const float4 *)(x + (u64)source * width + first_k + k)
                                    : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        }
    };
    auto store = [&](int buffer, int first_k) {
        #pragma unroll
        for (int n = 0; n < W_ITEMS; n++) {
            int item = threadIdx.x + n * THREADS;
            int r = item / (GEMM_K / 4), k = (item % (GEMM_K / 4)) * 4;
            float decoded[4];
            gemm_unpack<Kind>(raw[n], first_k + k, decoded);
            #pragma unroll
            for (int j = 0; j < 4; j++) weights[buffer][k + j][r] = decoded[j];
        }
        #pragma unroll
        for (int n = 0; n < X_ITEMS; n++) {
            int item = threadIdx.x + n * THREADS;
            int t = item / (GEMM_K / 4), k = (item % (GEMM_K / 4)) * 4;
            inputs[buffer][k][t] = staged[n].x;
            inputs[buffer][k + 1][t] = staged[n].y;
            inputs[buffer][k + 2][t] = staged[n].z;
            inputs[buffer][k + 3][t] = staged[n].w;
        }
    };
    fetch(0);
    store(0, 0);
    __syncthreads();
    int buffer = 0;
    for (int first_k = 0; first_k < width; first_k += GEMM_K) {
        bool more = first_k + GEMM_K < width;
        if (more) fetch(first_k + GEMM_K);
        #pragma unroll
        for (int k = 0; k < GEMM_K; k++) {
            float a[TM], b[TN];
            #pragma unroll
            for (int r = 0; r < TM; r += 4) {
                float4 v = *(const float4 *)&weights[buffer][k][ty * TM + r];
                a[r] = v.x;
                a[r + 1] = v.y;
                a[r + 2] = v.z;
                a[r + 3] = v.w;
            }
            #pragma unroll
            for (int t = 0; t < TN; t += 4) {
                float4 v = *(const float4 *)&inputs[buffer][k][tx * TN + t];
                b[t] = v.x;
                b[t + 1] = v.y;
                b[t + 2] = v.z;
                b[t + 3] = v.w;
            }
            #pragma unroll
            for (int r = 0; r < TM; r++) {
                #pragma unroll
                for (int t = 0; t < TN; t++) sums[r][t] += a[r] * b[t];
            }
        }
        if (more) store(buffer ^ 1, first_k + GEMM_K);
        buffer ^= 1;
        __syncthreads();
    }
    #pragma unroll
    for (int t = 0; t < TN; t++) {
        int slot = tx * TN + t;
        int target = Grouped ? descriptor[slot + 1] : sources[slot];
        if (target < 0) continue;
        #pragma unroll
        for (int r = 0; r < TM; r++) {
            int row = first_row + ty * TM + r;
            if (row < rows) y[(u64)target * rows + row] = sums[r][t];
        }
    }
}

// Dense: grid = (ceil(rows / 64), ceil(tokens / 64)), 256 threads; width % 256 == 0.
extern "C" __global__ void __launch_bounds__(256)
gemm_packed_dense(const u8 *w, int kind, const float *x, float *y, int width, int rows,
                  int tokens) {
    if (kind == 12) gemm_packed<12, 64, 64, 4, 4, false>(w, x, y, width, rows, tokens, nullptr, 1, 0);
    if (kind == 14) gemm_packed<14, 64, 64, 4, 4, false>(w, x, y, width, rows, tokens, nullptr, 1, 0);
}

// Experts: grid = (ceil(rows / 64), slots), 128 threads, 32-token descriptors
// (group_experts_reg32); width % 256 == 0.
extern "C" __global__ void __launch_bounds__(128)
gemm_packed_grouped(const u8 *w, int kind, const float *x, float *y, int width, int rows,
                    const int *groups, int used, int batched) {
    if (kind == 12) gemm_packed<12, 64, 32, 4, 4, true>(w, x, y, width, rows, 0, groups, used, batched);
    if (kind == 14) gemm_packed<14, 64, 32, 4, 4, true>(w, x, y, width, rows, 0, groups, used, batched);
}


// Parallel group_expert_tiles<32> for experts <= 256: one 1024-thread block.
// Same descriptor layout and order (tiles by expert, assignments ascending within
// an expert) as group_experts_reg32, without the serial per-expert scans.
#define GROUP_FAST_EXPERTS 256
template<int TILE>
__device__ __forceinline__ void group_experts_fast(const int *ids, int *groups, int experts,
                                                   int used, int tokens, int slots) {
    constexpr int WARPS = 32;
    __shared__ int counts[GROUP_FAST_EXPERTS], offsets[GROUP_FAST_EXPERTS];
    __shared__ int running[GROUP_FAST_EXPERTS];
    __shared__ int warp_counts[WARPS][GROUP_FAST_EXPERTS];
    int tid = threadIdx.x, lane = tid % 32, warp = tid / 32;
    int assignments = tokens * used;
    for (int i = tid; i < slots * (TILE + 1); i += 1024) groups[i] = -1;
    for (int e = tid; e < experts; e += 1024) {
        counts[e] = 0;
        running[e] = 0;
    }
    __syncthreads();
    for (int i = tid; i < assignments; i += 1024) {
        int e = ids[i];
        if (e >= 0 && e < experts) atomicAdd(&counts[e], 1);
    }
    __syncthreads();
    if (tid == 0) {
        int base = 0;
        for (int e = 0; e < experts; e++) {
            offsets[e] = base;
            base += (counts[e] + TILE - 1) / TILE;
        }
    }
    for (int first = 0; first < assignments; first += 1024) {
        for (int i = tid; i < WARPS * experts; i += 1024) warp_counts[i / experts][i % experts] = 0;
        __syncthreads();
        int i = first + tid;
        int e = i < assignments ? ids[i] : -1;
        bool valid = e >= 0 && e < experts;
        unsigned int same = __match_any_sync(0xffffffff, valid ? e : -1);
        int rank = __popc(same & ((1u << lane) - 1));
        if (valid && rank == 0) warp_counts[warp][e] = __popc(same);
        __syncthreads();
        for (int x = tid; x < experts; x += 1024) {
            int total = running[x];
            for (int w = 0; w < WARPS; w++) {
                int c = warp_counts[w][x];
                warp_counts[w][x] = total;
                total += c;
            }
            running[x] = total;
        }
        __syncthreads();
        if (valid) {
            int selected = warp_counts[warp][e] + rank;
            int tile = offsets[e] + selected / TILE;
            groups[tile * (TILE + 1)] = e;
            groups[tile * (TILE + 1) + selected % TILE + 1] = i;
        }
        __syncthreads();
    }
}

extern "C" __global__ void __launch_bounds__(1024)
group_experts_fast32(const int *ids, int *groups, int experts, int used, int tokens, int slots) {
    group_experts_fast<32>(ids, groups, experts, used, tokens, slots);
}

extern "C" __global__ void __launch_bounds__(1024)
group_experts_fast128(const int *ids, int *groups, int experts, int used, int tokens, int slots) {
    group_experts_fast<128>(ids, groups, experts, used, tokens, slots);
}

// 64-assignment tiles for int8-fast expert MMQ (mmq_c1_grouped64).
extern "C" __global__ void __launch_bounds__(1024)
group_experts_fast64(const int *ids, int *groups, int experts, int used, int tokens, int slots) {
    group_experts_fast<64>(ids, groups, experts, used, tokens, slots);
}
