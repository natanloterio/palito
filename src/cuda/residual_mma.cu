// Test-only IMMA engine. Integer accumulators restart at every metadata group.
template<int Components, int Columns = 32, int Projections = 1, int Rows = 32>
struct ResidualMmaTileComponents {
    unsigned int weights[Projections][Rows][9], values[Components][Columns][9];
    float coefficient[Projections][Rows][2], minimum[Projections][Rows], scales[Columns][Components];
    int sums[Columns][Components], destination[Columns];
};
using ResidualMmaTile = ResidualMmaTileComponents<3>;

#if __CUDA_ARCH__ >= 800
template<int Kind>
__device__ __forceinline__ void residual_mma_group(unsigned int a0, unsigned int a1,
    unsigned int a2, unsigned int a3, unsigned int b0, unsigned int b1, int (&d)[4]) {
    if (Kind == 12) {
        asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
            : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3])
            : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
    } else {
        asm volatile("mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 "
            "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};"
            : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3])
            : "r"(a0), "r"(a1), "r"(b0));
    }
}

template<int K, int P, int Components, int Columns, int Projections, int WarpTiles, bool SkipEmpty, int Rows>
__device__ __forceinline__ void residual_mma_accumulate(
    const ResidualMmaTileComponents<Components, Columns, Projections, Rows> &s,
    float (&total)[Components][WarpTiles][4], int warp_row, int warp_token,
    int group, int thread, bool first, bool second) {
    #pragma unroll
    for (int half = 0; half < (K == 12 ? 1 : 2); half++) {
        int r = warp_row + group, pack = thread + half * 4;
        unsigned int a0 = s.weights[P][r][pack], a1 = s.weights[P][r + 8][pack];
        unsigned int a2 = K == 12 ? s.weights[P][r][thread + 4] : 0;
        unsigned int a3 = K == 12 ? s.weights[P][r + 8][thread + 4] : 0;
        #pragma unroll
        for (int component = 0; component < Components; component++) {
            #pragma unroll
            for (int tile = 0; tile < WarpTiles; tile++) {
                if (SkipEmpty && !(tile == 0 ? first : second)) continue;
                int token = warp_token + tile * 8 + group;
                unsigned int b0 = s.values[component][token][pack];
                unsigned int b1 = K == 12 ? s.values[component][token][thread + 4] : 0;
                int dot[4] = {};
                residual_mma_group<K>(a0, a1, a2, a3, b0, b1, dot);
                #pragma unroll
                for (int item = 0; item < 4; item++) {
                    int output_row = r + item / 2 * 8;
                    int output_token = warp_token + tile * 8 + thread * 2 + item % 2;
                    float value = s.coefficient[P][output_row][half] * (float)dot[item];
                    if (K == 12)
                        value -= s.minimum[P][output_row] * (float)s.sums[output_token][component];
                    total[component][tile][item] += value * s.scales[output_token][component];
                }
            }
        }
    }
}

template<int Kind, int Components, bool SkipEmpty, int Columns, bool CacheScales, int UpKind = 0, int Rows = 32, int Descriptor = 33, bool PlanarQ4 = false, bool Subgroup = false>
__device__ void residual_mma_tile(const u8 *weights, const signed char *values,
    const float *scales, const int *sums, float *output, int width, int rows,
    int tokens, const int *groups, int used, int grouped, int assignment_major,
    int *status, ResidualMmaTileComponents<Components, Columns, UpKind ? 2 : 1, Rows> &s,
    const u8 *up_weights = nullptr, int descriptor_slot = -1, int paired_slot = -1) {
    static_assert(!UpKind || (Components == 3 && !CacheScales && !SkipEmpty));
    static_assert(!PlanarQ4 || (Kind == 12 && Rows == 32 && !UpKind && !CacheScales));
    static_assert(!Subgroup || (Columns == 8 && Rows == 32 && !UpKind));
    constexpr int Projections = UpKind ? 2 : 1;
    int slot = Subgroup ? descriptor_slot : (descriptor_slot < 0 ? (int)blockIdx.y : descriptor_slot);
    int expert = grouped ? (slot >= 0 ? groups[slot * Descriptor] : -1) : 0;
    if (!Subgroup && expert < 0) return;
    int tid = Subgroup ? threadIdx.x % 64 : threadIdx.x;
    int warp = tid / 32, lane = tid % 32;
    int group = lane / 4, thread = lane % 4;
    constexpr int Threads = Rows * (Columns >= 32 ? 4 : 2);
    constexpr int WarpTiles = Columns == 8 ? 1 : Columns == 64 ? 4 : 2;
    int first_row = blockIdx.x * Rows, warp_row = (Columns >= 32 ? warp / 2 : warp) * 16;
    int warp_token = Columns >= 32 ? warp % 2 * WarpTiles * 8 : 0;
    int input_count = width * tokens * (grouped && assignment_major ? used : 1);
    if (tid < Columns) {
        int source_slot = Columns == 64 && tid >= 32 ? paired_slot : slot;
        int assignment = grouped ? (expert >= 0 ? groups[source_slot * Descriptor + tid % 32 + 1] : -1)
                                 : slot * 32 + tid;
        s.destination[tid] = grouped || assignment < tokens ? assignment : -1;
    }
    __syncthreads();
    float total[Projections][Components][WarpTiles][4] = {};
    // Both destinations are warp-uniform; each descriptor is padded at the end.
    bool first = !SkipEmpty || s.destination[warp_token] >= 0;
    bool second = !SkipEmpty || WarpTiles == 1 || s.destination[warp_token + 8] >= 0;
    float block_d = 0.0f, block_dm = 0.0f;
    for (int k = 0; k < width; k += 32) {
        const unsigned int *plane = nullptr;
        if constexpr (PlanarQ4)
            plane = (const unsigned int *)weights +
                (((u64)expert * rows + first_row) * (width / 256) + (k / 256) * 32) * 36;
        for (int index = tid; index < Rows * 8; index += Threads) {
            int row = index / 8, pack = index % 8;
            unsigned int packed = 0;
            if constexpr (PlanarQ4) {
                unsigned bits = plane[(4 + (k % 256) / 64 * 8 + pack) * 32 + row];
                packed = (bits >> (4 * ((k / 32) % 2))) & 0x0f0f0f0fu;
            } else if ((!Subgroup || expert >= 0) && first_row + row < rows) {
                const u8 *p = weights + (((u64)(expert * rows + first_row + row) * width + k) / 256)
                    * (Kind == 12 ? 144 : 210);
                packed = residual_packed_quant<Kind>(p, k % 256 + pack * 4);
            }
            s.weights[0][row][pack] = packed;
            if constexpr (UpKind != 0) {
                unsigned int up_packed = 0;
                if (first_row + row < rows) {
                    const u8 *p = up_weights + (((u64)(expert * rows + first_row + row) * width + k) / 256)
                        * (UpKind == 12 ? 144 : 210);
                    up_packed = residual_packed_quant<UpKind>(p, k % 256 + pack * 4);
                }
                s.weights[1][row][pack] = up_packed;
            }
            if (row < Columns) {
                int assignment = s.destination[row];
                int source = grouped && !assignment_major && assignment >= 0 ? assignment / used : assignment;
                u64 offset = (u64)(source < 0 ? 0 : source) * width + k + pack * 4;
                #pragma unroll
                for (int component = 0; component < Components; component++)
                    s.values[component][row][pack] = source >= 0 ?
                        *(const unsigned int *)(values + (u64)component * input_count + offset) : 0;
            }
        }
        if constexpr (Columns == 64) {
            for (int index = tid; index < 32 * 8; index += Threads) {
                int column = 32 + index / 8, pack = index % 8;
                int assignment = s.destination[column];
                int source = assignment >= 0 && !assignment_major ? assignment / used : assignment;
                u64 offset = (u64)(source < 0 ? 0 : source) * width + k + pack * 4;
                #pragma unroll
                for (int component = 0; component < Components; component++)
                    s.values[component][column][pack] = source >= 0 ?
                        *(const unsigned int *)(values + (u64)component * input_count + offset) : 0;
            }
        }
        if (tid < Rows) {
            float a0 = 0.0f, a1 = 0.0f, b = 0.0f;
            if constexpr (PlanarQ4) {
                int g = k % 256 / 32, shift = 8 * (g % 4);
                unsigned int halves = plane[tid];
                unsigned int lo = (plane[32 + tid] >> shift) & 255;
                unsigned int mid = (plane[64 + tid] >> shift) & 255;
                unsigned int hi = g < 4 ? 0 : (plane[96 + tid] >> shift) & 255;
                int sc = g < 4 ? lo & 63 : (hi & 15) | ((lo >> 6) << 4);
                int mn = g < 4 ? mid & 63 : (hi >> 4) | ((mid >> 6) << 4);
                a0 = half_value_bits(halves & 65535) * (float)sc;
                b = half_value_bits(halves >> 16) * (float)mn;
            } else if ((!Subgroup || expert >= 0) && first_row + tid < rows) {
                const u8 *p = weights + (((u64)(expert * rows + first_row + tid) * width + k) / 256)
                    * (Kind == 12 ? 144 : 210);
                if (CacheScales) {
                    if (k % 256 == 0) {
                        block_d = half_value(p + (Kind == 12 ? 0 : 208));
                        if (Kind == 12) block_dm = half_value(p + 2);
                    }
                    residual_parameters_scaled<Kind>(p, k % 256, block_d, block_dm, a0, a1, b);
                } else {
                    residual_parameters<Kind>(p, k % 256, a0, a1, b);
                }
            }
            s.coefficient[0][tid][0] = a0;
            s.coefficient[0][tid][1] = a1;
            s.minimum[0][tid] = b;
            if constexpr (UpKind != 0) {
                float u0 = 0.0f, u1 = 0.0f, ub = 0.0f;
                if (first_row + tid < rows) {
                    const u8 *p = up_weights + (((u64)(expert * rows + first_row + tid) * width + k) / 256)
                        * (UpKind == 12 ? 144 : 210);
                    residual_parameters<UpKind>(p, k % 256, u0, u1, ub);
                }
                s.coefficient[1][tid][0] = u0;
                s.coefficient[1][tid][1] = u1;
                s.minimum[1][tid] = ub;
            }
        }
        if (tid < Columns) {
            int assignment = s.destination[tid];
            int source = grouped && !assignment_major && assignment >= 0 ? assignment / used : assignment;
            u64 offset = ((u64)(source < 0 ? 0 : source) * width + k) / 32 * Components;
            #pragma unroll
            for (int component = 0; component < Components; component++) {
                s.scales[tid][component] = source >= 0 ? scales[offset + component] : 0.0f;
                s.sums[tid][component] = source >= 0 ? sums[offset + component] : 0;
            }
        }
        __syncthreads();
        residual_mma_accumulate<Kind, 0, Components, Columns, Projections, WarpTiles, SkipEmpty>(
            s, total[0], warp_row, warp_token, group, thread, first, second);
        if constexpr (UpKind != 0)
            residual_mma_accumulate<UpKind, 1, Components, Columns, Projections, WarpTiles, SkipEmpty>(
                s, total[1], warp_row, warp_token, group, thread, first, second);
        __syncthreads();
    }
    #pragma unroll
    for (int tile = 0; tile < WarpTiles; tile++) {
        #pragma unroll
        for (int item = 0; item < 4; item++) {
            int row = first_row + warp_row + group + item / 2 * 8;
            int assignment = s.destination[warp_token + tile * 8 + thread * 2 + item % 2];
            if (row < rows && assignment >= 0) {
                float result = total[0][0][tile][item];
                if (Components == 3)
                    result = (result + total[0][1][tile][item]) + total[0][2][tile][item];
                if (!isfinite(result)) atomicOr(status, 4);
                if constexpr (UpKind != 0) {
                    float up = (total[1][0][tile][item] + total[1][1][tile][item]) + total[1][2][tile][item];
                    if (!isfinite(up)) atomicOr(status, 4);
                    result = result / (1.0f + expf(-result)) * up;
                }
                output[(u64)assignment * rows + row] = result;
            }
        }
    }
}
#endif

template<int Components, bool SkipEmpty, int Columns = 32, bool CacheScales = false, int Rows = 32>
__device__ __forceinline__ void residual_matmul_mma_dispatch(const u8 *weights, int kind,
    const signed char *values, const float *scales, const int *sums, float *output,
    int width, int rows, int tokens, const int *groups, int used, int grouped,
    int assignment_major, int *status) {
#if __CUDA_ARCH__ >= 800
    __shared__ ResidualMmaTileComponents<Components, Columns, 1, Rows> tile;
    if (kind == 12) residual_mma_tile<12, Components, SkipEmpty, Columns, CacheScales, 0, Rows>(weights, values, scales, sums, output, width,
        rows, tokens, groups, used, grouped, assignment_major, status, tile);
    if (kind == 14) residual_mma_tile<14, Components, SkipEmpty, Columns, CacheScales, 0, Rows>(weights, values, scales, sums, output, width,
        rows, tokens, groups, used, grouped, assignment_major, status, tile);
    if (kind != 12 && kind != 14 && threadIdx.x == 0) atomicOr(status, 8);
#else
    if (threadIdx.x == 0) atomicOr(status, 16);
#endif
}

#define RESIDUAL_MMA_ENTRY(Name, Components, SkipEmpty, CacheScales, Rows) \
extern "C" __global__ void Name(const u8 *weights, int kind, \
    const signed char *values, const float *scales, const int *sums, float *output, \
    int width, int rows, int tokens, const int *groups, int used, int grouped, \
    int assignment_major, int *status) { \
    residual_matmul_mma_dispatch<Components, SkipEmpty, 32, CacheScales, Rows>(weights, kind, values, scales, sums, \
        output, width, rows, tokens, groups, used, grouped, assignment_major, status); \
}

RESIDUAL_MMA_ENTRY(residual_matmul_mma3, 3, false, false, 32)
#if defined(CUDA_M8_PROTOTYPES) || defined(CUDA_MMA3_WIDE_ROWS_RESEARCH)
RESIDUAL_MMA_ENTRY(residual_matmul_mma3_rows64, 3, false, false, 64)
#endif

#ifdef CUDA_M8_PROTOTYPES
#if __CUDA_ARCH__ >= 800
union ResidualMixedTiles {
    ResidualMmaTile full;
    ResidualMmaTileComponents<3, 8> pair[2];
};

union ResidualSameExpertTiles {
    ResidualMmaTile single;
    ResidualMmaTileComponents<3, 64> pair;
};

template<int Kind>
__device__ __forceinline__ void residual_mma_same_expert(const u8 *weights,
    const signed char *values, const float *scales, const int *sums, float *output,
    int width, int rows, int tokens, const int *groups, int used, int assignment_major,
    const int *schedule, int *status, ResidualSameExpertTiles &tiles) {
    int first = schedule[blockIdx.y * 2], second = schedule[blockIdx.y * 2 + 1];
    if (first < 0) return;
    if (second < 0) {
        residual_mma_tile<Kind, 3, false, 32, false>(weights, values, scales, sums,
            output, width, rows, tokens, groups, used, 1, assignment_major, status,
            tiles.single, nullptr, first);
    } else {
        residual_mma_tile<Kind, 3, false, 64, false>(weights, values, scales, sums,
            output, width, rows, tokens, groups, used, 1, assignment_major, status,
            tiles.pair, nullptr, first, second);
    }
}

template<int Kind>
__device__ void residual_mma_mixed(const u8 *weights, const signed char *values,
    const float *scales, const int *sums, float *output, int width, int rows, int tokens,
    const int *groups, int used, int assignment_major, const int *schedule, int *status,
    ResidualMixedTiles &tiles) {
    int first = schedule[blockIdx.y * 2];
    int second = schedule[blockIdx.y * 2 + 1];
    if (first < 0) return;
    if (second == -2) {
        residual_mma_tile<Kind, 3, false, 32, false>(weights, values, scales, sums,
            output, width, rows, tokens, groups, used, 1, assignment_major, status,
            tiles.full, nullptr, first);
    } else {
        int half = threadIdx.x / 64;
        int slot = half == 0 ? first : second;
        // A missing second descriptor still participates in CTA-wide barriers.
        residual_mma_tile<Kind, 3, false, 8, false, 0, 32, 33, false, true>(
            weights, values, scales, sums, output, width, rows, tokens, groups, used, 1,
            assignment_major, status, tiles.pair[half], nullptr, slot);
    }
}
#endif

extern "C" __global__ void residual_matmul_mma3_same_expert(const u8 *weights, int kind,
    const signed char *values, const float *scales, const int *sums, float *output,
    int width, int rows, int tokens, const int *groups, int used, int assignment_major,
    const int *schedule, int *status) {
#if __CUDA_ARCH__ >= 800
    __shared__ ResidualSameExpertTiles tiles;
    if (kind == 12) {
        residual_mma_same_expert<12>(weights, values, scales, sums, output, width, rows,
            tokens, groups, used, assignment_major, schedule, status, tiles);
    } else if (kind == 14) {
        residual_mma_same_expert<14>(weights, values, scales, sums, output, width, rows,
            tokens, groups, used, assignment_major, schedule, status, tiles);
    } else if (threadIdx.x == 0) atomicOr(status, 8);
#else
    if (threadIdx.x == 0) atomicOr(status, 16);
#endif
}

extern "C" __global__ void residual_matmul_mma3_mixed(const u8 *weights, int kind,
    const signed char *values, const float *scales, const int *sums, float *output,
    int width, int rows, int tokens, const int *groups, int used, int assignment_major,
    const int *schedule, int *status) {
#if __CUDA_ARCH__ >= 800
    __shared__ ResidualMixedTiles tiles;
    if (kind == 12) {
        residual_mma_mixed<12>(weights, values, scales, sums, output, width, rows, tokens,
            groups, used, assignment_major, schedule, status, tiles);
    } else if (kind == 14) {
        residual_mma_mixed<14>(weights, values, scales, sums, output, width, rows, tokens,
            groups, used, assignment_major, schedule, status, tiles);
    } else if (threadIdx.x == 0) atomicOr(status, 8);
#else
    if (threadIdx.x == 0) atomicOr(status, 16);
#endif
}

extern "C" __global__ void residual_matmul_mma3_sync_k64(const u8 *weights, int kind,
    const signed char *values, const float *scales, const int *sums, float *output,
    int width, int rows, int tokens, const int *groups, int used, int grouped,
    int assignment_major, int *status) {
#if __CUDA_ARCH__ >= 800
    if (kind != 12 || width % 256 != 0 || !grouped || assignment_major) {
        if (threadIdx.x == 0) atomicOr(status, 8);
        return;
    }
    const int *descriptor = groups + blockIdx.y * 33;
    int expert = descriptor[0];
    if (expert < 0) return;
    __shared__ ResidualMmaTile s[2];
    int tid = threadIdx.x, first_row = blockIdx.x * 32;
    int group = tid % 32 / 4, thread = tid % 4;
    int warp_row = tid / 64 * 16, warp_token = (tid / 32 % 2) * 16;
    int input_count = width * tokens;
    float total[3][2][4] = {};
    for (int k = 0; k < width; k += 64) {
        #pragma unroll
        for (int i = 0; i < 2; i++) {
            int row = tid / 8 + i * 16, pack = tid % 8;
            unsigned bits = 0;
            if (first_row + row < rows) {
                const u8 *p = weights + (((u64)expert * rows + first_row + row) *
                    (width / 256) + k / 256) * 144;
                bits = *(const unsigned int *)(p + 16 + k % 256 / 2 + pack * 4);
            }
            s[0].weights[0][row][pack] = bits & 0x0f0f0f0fu;
            s[1].weights[0][row][pack] = (bits >> 4) & 0x0f0f0f0fu;
            int assignment = descriptor[row + 1];
            int source = assignment >= 0 ? assignment / used : 0;
            #pragma unroll
            for (int stage = 0; stage < 2; stage++) {
                #pragma unroll
                for (int c = 0; c < 3; c++) {
                    u64 offset = (u64)c * input_count + (u64)source * width + k + stage * 32 + pack * 4;
                    s[stage].values[c][row][pack] = assignment >= 0 ?
                        *(const unsigned int *)(values + offset) : 0;
                }
            }
        }
        if (tid < 32) {
            int assignment = descriptor[tid + 1];
            int source = assignment >= 0 ? assignment / used : 0;
            const u8 *p = weights + (((u64)expert * rows + first_row + tid) *
                (width / 256) + k / 256) * 144;
            bool valid = first_row + tid < rows;
            float d = valid ? half_value(p) : 0.0f;
            float dm = valid ? half_value(p + 2) : 0.0f;
            #pragma unroll
            for (int stage = 0; stage < 2; stage++) {
                float a0 = 0.0f, unused = 0.0f, b = 0.0f;
                if (valid) residual_parameters_scaled<12>(p, k % 256 + stage * 32, d, dm, a0, unused, b);
                s[stage].coefficient[0][tid][0] = a0;
                s[stage].minimum[0][tid] = b;
                u64 meta = ((u64)source * width + k + stage * 32) / 32 * 3;
                #pragma unroll
                for (int c = 0; c < 3; c++) {
                    s[stage].scales[tid][c] = assignment >= 0 ? scales[meta + c] : 0.0f;
                    s[stage].sums[tid][c] = assignment >= 0 ? sums[meta + c] : 0;
                }
            }
        }
        __syncthreads();
        #pragma unroll
        for (int stage = 0; stage < 2; stage++)
            residual_mma_accumulate<12, 0, 3, 32, 1, 2, false>(
                s[stage], total, warp_row, warp_token, group, thread, true, true);
        __syncthreads();
    }
    #pragma unroll
    for (int tile = 0; tile < 2; tile++) {
        #pragma unroll
        for (int item = 0; item < 4; item++) {
            int row = first_row + warp_row + group + item / 2 * 8;
            int assignment = descriptor[1 + warp_token + tile * 8 + thread * 2 + item % 2];
            if (row < rows && assignment >= 0) {
                float result = (total[0][tile][item] + total[1][tile][item]) + total[2][tile][item];
                if (!isfinite(result)) atomicOr(status, 4);
                output[(u64)assignment * rows + row] = result;
            }
        }
    }
#else
    if (threadIdx.x == 0) atomicOr(status, 16);
#endif
}

extern "C" __global__ void residual_matmul_mma3_planar_q4(const u8 *weights, int kind,
    const signed char *values, const float *scales, const int *sums, float *output,
    int width, int rows, int tokens, const int *groups, int used, int grouped,
    int assignment_major, int *status) {
#if __CUDA_ARCH__ >= 800
    __shared__ ResidualMmaTile tile;
    if (kind == 12 && width % 256 == 0 && rows % 32 == 0) {
        residual_mma_tile<12, 3, false, 32, false, 0, 32, 33, true>(
            weights, values, scales, sums, output, width, rows, tokens, groups,
            used, grouped, assignment_major, status, tile);
    } else if (threadIdx.x == 0) atomicOr(status, 8);
#else
    if (threadIdx.x == 0) atomicOr(status, 16);
#endif
}
#endif

#ifdef CUDA_M8_PROTOTYPES
#if __CUDA_ARCH__ >= 800
template<int Kind>
__device__ void residual_mma_registers(const u8 *weights, const signed char *values,
    const float *scales, const int *sums, float *output, int width, int rows, int tokens,
    int *status) {
    int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    int group = lane / 4, thread = lane % 4;
    int row = blockIdx.x * 32 + warp / 2 * 16 + group;
    int token_base = blockIdx.y * 32 + warp % 2 * 16;
    int input_count = width * tokens;
    constexpr int BlockBytes = Kind == 12 ? 144 : 210;
    const u8 *row_weights[2];
    #pragma unroll
    for (int r = 0; r < 2; r++)
        row_weights[r] = weights + (u64)(row + r * 8 < rows ? row + r * 8 : 0)
            * (width / 256) * BlockBytes;
    float total[3][2][4] = {};
    for (int k = 0; k < width; k += 32) {
        int k_in_block = k % 256;
        const u8 *packed[2] = {row_weights[0] + k / 256 * BlockBytes,
                               row_weights[1] + k / 256 * BlockBytes};
        float coeff[2][2], minimum[2];
        #pragma unroll
        for (int r = 0; r < 2; r++)
            residual_parameters<Kind>(packed[r], k_in_block, coeff[r][0], coeff[r][1], minimum[r]);
        #pragma unroll
        for (int half = 0; half < (Kind == 12 ? 1 : 2); half++) {
            int offset = k_in_block + thread * 4 + half * 16;
            unsigned a0 = residual_packed_quant<Kind>(packed[0], offset);
            unsigned a1 = residual_packed_quant<Kind>(packed[1], offset);
            unsigned a2 = Kind == 12 ? residual_packed_quant<Kind>(packed[0], offset + 16) : 0;
            unsigned a3 = Kind == 12 ? residual_packed_quant<Kind>(packed[1], offset + 16) : 0;
            #pragma unroll
            for (int c = 0; c < 3; c++) {
                #pragma unroll
                for (int tile = 0; tile < 2; tile++) {
                    int source = token_base + tile * 8 + group;
                    unsigned b0 = 0, b1 = 0;
                    if (source < tokens) {
                        const signed char *v = values + (u64)c * input_count +
                            (u64)source * width + k + thread * 4 + half * 16;
                        b0 = *(const unsigned int *)v;
                        if constexpr (Kind == 12) b1 = *(const unsigned int *)(v + 16);
                    }
                    int dot[4] = {};
                    residual_mma_group<Kind>(a0, a1, a2, a3, b0, b1, dot);
                    #pragma unroll
                    for (int slot = 0; slot < 2; slot++) {
                        int token = token_base + tile * 8 + thread * 2 + slot;
                        if (token >= tokens) continue;
                        u64 meta = ((u64)token * width + k) / 32 * 3 + c;
                        float scale = scales[meta];
                        float sum = Kind == 12 ? (float)sums[meta] : 0.0f;
                        #pragma unroll
                        for (int r = 0; r < 2; r++) {
                            float value = coeff[r][half] * (float)dot[r * 2 + slot];
                            if constexpr (Kind == 12) value -= minimum[r] * sum;
                            total[c][tile][r * 2 + slot] += value * scale;
                        }
                    }
                }
            }
        }
    }
    #pragma unroll
    for (int tile = 0; tile < 2; tile++) {
        #pragma unroll
        for (int item = 0; item < 4; item++) {
            int dest_row = row + item / 2 * 8;
            int token = token_base + tile * 8 + thread * 2 + item % 2;
            if (dest_row < rows && token < tokens) {
                float result = (total[0][tile][item] + total[1][tile][item]) + total[2][tile][item];
                if (!isfinite(result)) atomicOr(status, 4);
                output[(u64)token * rows + dest_row] = result;
            }
        }
    }
}
#endif

extern "C" __global__ void residual_matmul_mma3_registers(const u8 *weights, int kind,
    const signed char *values, const float *scales, const int *sums, float *output,
    int width, int rows, int tokens, const int *groups, int used, int grouped,
    int assignment_major, int *status) {
#if __CUDA_ARCH__ >= 800
    if (!grouped && width % 256 == 0) {
        if (kind == 12) {
            residual_mma_registers<12>(weights, values, scales, sums, output, width, rows, tokens, status);
            return;
        }
        if (kind == 14) {
            residual_mma_registers<14>(weights, values, scales, sums, output, width, rows, tokens, status);
            return;
        }
    }
    if (threadIdx.x == 0) atomicOr(status, 8);
#else
    if (threadIdx.x == 0) atomicOr(status, 16);
#endif
}
#endif
#ifdef CUDA_M8_PROTOTYPES
extern "C" __global__ void residual_matmul_mma3_compact8(const u8 *weights, int kind,
    const signed char *values, const float *scales, const int *sums, float *output,
    int width, int rows, int tokens, const int *groups, int used, int assignment_major, int *status) {
#if __CUDA_ARCH__ >= 800
    __shared__ ResidualMmaTileComponents<3, 8> tile;
    if (kind == 12) residual_mma_tile<12, 3, false, 8, false, 0, 32, 9>(weights, values, scales, sums,
        output, width, rows, tokens, groups, used, 1, assignment_major, status, tile);
    else if (kind == 14) residual_mma_tile<14, 3, false, 8, false, 0, 32, 9>(weights, values, scales, sums,
        output, width, rows, tokens, groups, used, 1, assignment_major, status, tile);
    else if (threadIdx.x == 0) atomicOr(status, 8);
#else
    if (threadIdx.x == 0) atomicOr(status, 16);
#endif
}

extern "C" __global__ void residual_matmul_mma3_gate_up_q4(
    const u8 *gate, const u8 *up, const signed char *values, const float *scales,
    const int *sums, float *output, int width, int rows, int tokens,
    const int *groups, int used, int *status) {
#if __CUDA_ARCH__ >= 800
    __shared__ ResidualMmaTileComponents<3, 32, 2> tile;
    residual_mma_tile<12, 3, false, 32, false, 12>(gate, values, scales, sums,
        output, width, rows, tokens, groups, used, 1, 0, status, tile, up);
#else
    if (threadIdx.x == 0) atomicOr(status, 16);
#endif
}

RESIDUAL_MMA_ENTRY(residual_matmul_mma1, 1, false, false, 32)
RESIDUAL_MMA_ENTRY(residual_matmul_mma3_skip_empty, 3, true, false, 32)
RESIDUAL_MMA_ENTRY(residual_matmul_mma3_cached_scales, 3, false, true, 32)

// Keep the admitted descriptor ABI; each live descriptor belongs to exactly one pass.
#define RESIDUAL_MMA_SPLIT_ENTRY(Name, Threshold, Columns) \
extern "C" __global__ void Name(const u8 *weights, int kind, \
    const signed char *values, const float *scales, const int *sums, float *output, \
    int width, int rows, int tokens, const int *groups, int used, int grouped, \
    int assignment_major, int *status) { \
    if (!grouped || groups[blockIdx.y * 33] < 0) return; \
    bool fits = groups[blockIdx.y * 33 + 1 + Threshold] < 0; \
    if (fits != (Columns != 32)) return; \
    residual_matmul_mma_dispatch<3, false, Columns>(weights, kind, values, scales, sums, \
        output, width, rows, tokens, groups, used, grouped, assignment_major, status); \
}
RESIDUAL_MMA_SPLIT_ENTRY(residual_matmul_mma3_cols8, 8, 8)
RESIDUAL_MMA_SPLIT_ENTRY(residual_matmul_mma3_wide8, 8, 32)
RESIDUAL_MMA_SPLIT_ENTRY(residual_matmul_mma3_cols16, 16, 16)
RESIDUAL_MMA_SPLIT_ENTRY(residual_matmul_mma3_wide16, 16, 32)
#undef RESIDUAL_MMA_SPLIT_ENTRY
#endif
#undef RESIDUAL_MMA_ENTRY
