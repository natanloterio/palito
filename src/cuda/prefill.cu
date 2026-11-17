// Token-major dense projections share one weight decode across four prompt rows.
template<int Kind, bool Grouped = false, int Tile = 4>
__device__ void prompt_tile_workspace(const u8 *w, const float *x, float *y,
                           int width, int rows, int tokens, float partial[][256],
                           const int *groups = nullptr, int used = 1, int batched = 0) {
    int row = blockIdx.x, first = blockIdx.y * Tile, lane = threadIdx.x;
    int expert = Grouped ? groups[blockIdx.y * 5] : 0;
    if (expert < 0) return;
    int assignments[Tile];
    #pragma unroll
    for (int t = 0; t < Tile; t++)
        assignments[t] = Grouped ? groups[blockIdx.y * 5 + t + 1]
                                : (first + t < tokens ? first + t : -1);
    float sums[Tile] = {};
    for (int i = lane; i < width; i += 256) {
        float decoded = weight(w, Kind, ((u64)expert * rows + row) * width + i);
        #pragma unroll
        for (int t = 0; t < Tile; t++) {
            if (assignments[t] >= 0) {
                int input = Grouped && !batched ? assignments[t] / used : assignments[t];
                sums[t] += decoded * x[(u64)input * width + i];
            }
        }
    }
    #pragma unroll
    for (int t = 0; t < Tile; t++) partial[t][lane] = sums[t];
    __syncthreads();
    for (int stride = 128; stride >= 64; stride /= 2) {
        if (lane < stride) {
            #pragma unroll
            for (int t = 0; t < Tile; t++)
                partial[t][lane] += partial[t][lane + stride];
        }
        __syncthreads();
    }
    if (lane < 32) {
        #pragma unroll
        for (int t = 0; t < Tile; t++) {
            float sum = partial[t][lane] + partial[t][lane + 32];
            for (int stride = 16; stride > 0; stride /= 2)
                sum += __shfl_down_sync(0xffffffff, sum, stride);
            if (lane == 0 && assignments[t] >= 0)
                y[(u64)assignments[t] * rows + row] = sum;
        }
    }
}

template<int Kind, bool Grouped = false, int Tile = 4>
__device__ void prompt_tile(const u8 *w, const float *x, float *y,
                           int width, int rows, int tokens,
                           const int *groups = nullptr, int used = 1, int batched = 0) {
    __shared__ float partial[Tile][256];
    prompt_tile_workspace<Kind, Grouped, Tile>(
        w, x, y, width, rows, tokens, partial, groups, used, batched);
}

extern "C" __global__ void prefill_matmul(const u8 *w, int kind, const float *x,
                                        float *y, int width, int rows, int tokens) {
    __shared__ float partial[4][256];
    switch (kind) {
        case 0: prompt_tile_workspace<0>(w, x, y, width, rows, tokens, partial); break;
        case 1: prompt_tile_workspace<1>(w, x, y, width, rows, tokens, partial); break;
        case 6: prompt_tile_workspace<6>(w, x, y, width, rows, tokens, partial); break;
        case 8: prompt_tile_workspace<8>(w, x, y, width, rows, tokens, partial); break;
        case 12: prompt_tile_workspace<12>(w, x, y, width, rows, tokens, partial); break;
        case 14: prompt_tile_workspace<14>(w, x, y, width, rows, tokens, partial); break;
    }
}

extern "C" __global__ void prefill_embedding(const u8 *w, int kind, const int *tokens,
                                            float *x, int width) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int token = blockIdx.y;
    if (i < width)
        x[(u64)token * width + i] = weight(w, kind, (u64)tokens[token] * width + i);
}

template<int Tile>
__device__ void group_expert_tiles(const int *ids, int *groups, int experts,
                                  int used, int tokens, int slots, int *tile_counts) {
    int lane = threadIdx.x, assignments = tokens * used;
    for (int i = lane; i < slots * (Tile + 1); i += 256) groups[i] = -1;
    // A block handles all experts in rounds so models with >256 experts remain valid.
    int base = 0;
    for (int first = 0; first < experts; first += 256) {
        int expert = first + lane, count = 0;
        if (expert < experts) {
            for (int i = 0; i < assignments; i++) count += ids[i] == expert;
        }
        tile_counts[lane] = (count + Tile - 1) / Tile;
        __syncthreads();
        int offset = base;
        for (int i = 0; i < lane; i++) offset += tile_counts[i];
        if (expert < experts) {
            for (int i = 0, selected = 0; i < assignments; i++) {
                if (ids[i] == expert) {
                    int tile = offset + selected / Tile;
                    groups[tile * (Tile + 1)] = expert;
                    groups[tile * (Tile + 1) + selected % Tile + 1] = i;
                    selected++;
                }
            }
        }
        for (int i = 0; i < 256; i++) base += tile_counts[i];
        // Protect shared counts before the next round and initialized padding
        // before any thread in this round writes a tile descriptor.
        __syncthreads();
    }
}

extern "C" __global__ void group_experts(const int *ids, int *groups, int experts,
                                       int used, int tokens, int slots) {
    __shared__ int tile_counts[256];
    group_expert_tiles<4>(ids, groups, experts, used, tokens, slots, tile_counts);
}

// One 256-thread block; allocate 9 * min(tokens*used, ceil(tokens*used/8)+experts-1)
// ints. Negative assignment slots and unused descriptors are initialized to -1.
extern "C" __global__ void group_experts_tiled(const int *ids, int *groups, int experts,
                                             int used, int tokens, int slots) {
    __shared__ int tile_counts[256];
    group_expert_tiles<8>(ids, groups, experts, used, tokens, slots, tile_counts);
}

extern "C" __global__ void grouped_matmul(const u8 *w, int kind, const float *x, float *y,
                                         int width, int rows, const int *groups,
                                         int used, int batched) {
    __shared__ float partial[4][256];
    switch (kind) {
        case 0: prompt_tile_workspace<0, true>(w, x, y, width, rows, 0, partial, groups, used, batched); break;
        case 1: prompt_tile_workspace<1, true>(w, x, y, width, rows, 0, partial, groups, used, batched); break;
        case 6: prompt_tile_workspace<6, true>(w, x, y, width, rows, 0, partial, groups, used, batched); break;
        case 8: prompt_tile_workspace<8, true>(w, x, y, width, rows, 0, partial, groups, used, batched); break;
        case 12: prompt_tile_workspace<12, true>(w, x, y, width, rows, 0, partial, groups, used, batched); break;
        case 14: prompt_tile_workspace<14, true>(w, x, y, width, rows, 0, partial, groups, used, batched); break;
    }
}

// Each warp owns one row at a time and broadcasts quant-block metadata.
// The padded row stride avoids bank conflicts when compute lanes select rows.
template<int Kind, int RowTile, int KTile, int Warps = 8>
__device__ void projection_weights(const u8 *w, float weights[][KTile + 1],
                                   int width, int rows, int expert, int first_row, int first_k) {
    int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    for (int r = warp; r < RowTile; r += Warps) {
        int row = first_row + r;
        if (row >= rows) {
            for (int k = lane; k < KTile; k += 32) weights[r][k] = 0.0f;
            continue;
        }
        if (KTile == 32 && Kind != 0 && Kind != 1) {
            u64 index = ((u64)expert * rows + row) * width + first_k;
            if (first_k + 32 <= width && index % 32 == 0) {
                // All lanes own valid weights; only decoded operands are staged.
                weights[r][lane] = warp_weight<Kind>(w, index + lane);
                continue;
            }
        }
        int column = 0;
        while (column < KTile && first_k + column < width) {
            constexpr int Block = Kind == 12 || Kind == 14 ? 256 : 32;
            constexpr int Bytes = Kind == 6 ? 22 : Kind == 8 ? 34 : Kind == 12 ? 144 : 210;
            u64 index = ((u64)expert * rows + row) * width + first_k + column;
            int first = index % Block;
            int count = min(Block - first, min(KTile - column, width - first_k - column));
            const u8 *p = w;
            float d = 0.0f, minimum = 0.0f, scale = 0.0f, offset = 0.0f;
            unsigned int high_bits = 0;
            if (Kind != 0 && Kind != 1) {
                p = w + (index / Block) * Bytes;
                if (lane == 0) d = half_value(p + (Kind == 14 ? 208 : 0));
                d = __shfl_sync(0xffffffff, d, 0);
                if (Kind == 6) {
                    if (lane == 0)
                        high_bits = (unsigned int)p[2] | ((unsigned int)p[3] << 8)
                            | ((unsigned int)p[4] << 16) | ((unsigned int)p[5] << 24);
                    high_bits = __shfl_sync(0xffffffff, high_bits, 0);
                }
                if (Kind == 12) {
                    if (lane == 0) minimum = half_value(p + 2);
                    minimum = __shfl_sync(0xffffffff, minimum, 0);
                    if (lane < 8) {
                        const u8 *s = p + 4;
                        int sc = lane < 4 ? s[lane] & 63
                            : (s[lane + 4] & 15) | ((s[lane - 4] >> 6) << 4);
                        int mn = lane < 4 ? s[lane + 4] & 63
                            : (s[lane + 4] >> 4) | ((s[lane] >> 6) << 4);
                        scale = d * (float)sc;
                        offset = minimum * (float)mn;
                    }
                }
                if (Kind == 14 && lane < 16)
                    scale = d * (float)((const signed char *)p)[192 + lane];
            }
            for (int start = 0; start < count; start += 32) {
                int i = first + start + lane;
                float sc = 0.0f, mn = 0.0f;
                if (Kind == 12) {
                    sc = __shfl_sync(0xffffffff, scale, i / 32);
                    mn = __shfl_sync(0xffffffff, offset, i / 32);
                }
                if (Kind == 14) sc = __shfl_sync(0xffffffff, scale, i / 16);
                if (start + lane < count) {
                    float value;
                    if (Kind == 0) value = ((const float *)w)[index + start + lane];
                    if (Kind == 1) value = half_value(w + (index + start + lane) * 2);
                    if (Kind == 6) {
                        int q = ((p[6 + i % 16] >> (4 * (i / 16))) & 15)
                            + ((high_bits >> i) & 1) * 16 - 16;
                        value = d * (float)q;
                    }
                    if (Kind == 8) value = d * (float)((const signed char *)p)[2 + i];
                    if (Kind == 12) {
                        int q = (p[16 + (i / 64) * 32 + i % 32] >> (4 * ((i / 32) % 2))) & 15;
                        value = sc * (float)q - mn;
                    }
                    if (Kind == 14) {
                        int h = i / 128, l = i % 32, quarter = (i % 128) / 32;
                        int low = (p[h * 64 + l + (quarter % 2) * 32] >> (4 * (quarter / 2))) & 15;
                        int high = (p[128 + h * 32 + l] >> (2 * quarter)) & 3;
                        value = sc * (float)((low | (high << 4)) - 32);
                    }
                    weights[r][column + start + lane] = value;
                }
            }
            column += count;
        }
        for (int k = column + lane; k < KTile; k += 32) weights[r][k] = 0.0f;
    }
}

#ifdef CUDA_M8_PROTOTYPES
// Eight independent F32 accumulators/thread: two rows by four tokens.
// Shared decoded weights preserve the existing GGUF reconstruction arithmetic;
// every output traverses K in increasing order (unlike the matvec reduction tree).
template<int Kind, int RowTile, int TokenTile, int KTile, int Threads, bool Grouped = false>
__device__ void projection_register_tiled(const u8 *w, const float *x, float *y,
                                          int width, int rows, int tokens,
                                          float weights[][KTile + 1],
                                          float activations[][TokenTile + 1],
                                          const int *groups = nullptr, int used = 1, int batched = 0) {
    int expert = Grouped ? groups[blockIdx.y * (TokenTile + 1)] : 0;
    if (expert < 0) return;
    int first_row = blockIdx.x * RowTile, first_token = blockIdx.y * TokenTile;
    int row = (threadIdx.x / (TokenTile / 4)) * 2;
    int token = (threadIdx.x % (TokenTile / 4)) * 4;
    float sums[2][4] = {};
    for (int first_k = 0; first_k < width; first_k += KTile) {
        projection_weights<Kind, RowTile, KTile, Threads / 32>(
            w, weights, width, rows, expert, first_row, first_k);
        for (int i = threadIdx.x; i < KTile * TokenTile; i += Threads) {
            int k = i / TokenTile, t = i % TokenTile;
            if (Grouped) {
                int assignment = groups[blockIdx.y * (TokenTile + 1) + t + 1];
                int input = !batched && assignment >= 0 ? assignment / used : assignment;
                activations[k][t] = input >= 0 && first_k + k < width
                    ? x[(u64)input * width + first_k + k] : 0.0f;
            } else {
                activations[k][t] = first_token + t < tokens && first_k + k < width
                    ? x[(u64)(first_token + t) * width + first_k + k] : 0.0f;
            }
        }
        __syncthreads();
        int count = min(KTile, width - first_k);
        #pragma unroll 1
        for (int k = 0; k < count; k++) {
            float values[4];
            #pragma unroll
            for (int t = 0; t < 4; t++) values[t] = activations[k][token + t];
            #pragma unroll
            for (int r = 0; r < 2; r++) {
                float decoded = weights[row + r][k];
                #pragma unroll
                for (int t = 0; t < 4; t++) sums[r][t] += decoded * values[t];
            }
        }
        __syncthreads();
    }
    #pragma unroll
    for (int r = 0; r < 2; r++) {
        #pragma unroll
        for (int t = 0; t < 4; t++) {
            if (Grouped) {
                int assignment = groups[blockIdx.y * (TokenTile + 1) + token + t + 1];
                if (first_row + row + r < rows && assignment >= 0)
                    y[(u64)assignment * rows + first_row + row + r] = sums[r][t];
            } else {
                if (first_row + row + r < rows && first_token + token + t < tokens)
                    y[(u64)(first_token + token + t) * rows + first_row + row + r] = sums[r][t];
            }
        }
    }
}

// Prototype-only entry points. Kinds other than Q4_K/Q6_K intentionally do no work.
// Compile with fmad=false. No dynamic shared memory or persistent scratch.
// (rows,tokens,K)=(64,16,64), 128 threads, static shared=20,992 bytes.
extern "C" __global__ void prefill_matmul_reg64x16(const u8 *w, int kind, const float *x,
                                                 float *y, int width, int rows, int tokens) {
    __shared__ float weights[64][65], activations[64][17];
    if (kind == 12) projection_register_tiled<12, 64, 16, 64, 128>(w, x, y, width, rows, tokens, weights, activations);
    if (kind == 14) projection_register_tiled<14, 64, 16, 64, 128>(w, x, y, width, rows, tokens, weights, activations);
}

// (rows,tokens,K)=(32,32,64), 128 threads, static shared=16,768 bytes.
extern "C" __global__ void prefill_matmul_reg32x32(const u8 *w, int kind, const float *x,
                                                 float *y, int width, int rows, int tokens) {
    __shared__ float weights[32][65], activations[64][33];
    if (kind == 12) projection_register_tiled<12, 32, 32, 64, 128>(w, x, y, width, rows, tokens, weights, activations);
    if (kind == 14) projection_register_tiled<14, 32, 32, 64, 128>(w, x, y, width, rows, tokens, weights, activations);
}

// (rows,tokens,K)=(64,32,32), 256 threads, static shared=12,672 bytes.
extern "C" __global__ void prefill_matmul_reg64x32(const u8 *w, int kind, const float *x,
                                                 float *y, int width, int rows, int tokens) {
    __shared__ float weights[64][33], activations[32][33];
    if (kind == 12) projection_register_tiled<12, 64, 32, 32, 256>(w, x, y, width, rows, tokens, weights, activations);
    if (kind == 14) projection_register_tiled<14, 64, 32, 32, 256>(w, x, y, width, rows, tokens, weights, activations);
}

// 256 threads; allocate 33 * min(assignments,ceil(assignments/32)+experts-1) ints.
extern "C" __global__ void group_experts_reg32(const int *ids, int *groups, int experts,
                                             int used, int tokens, int slots) {
    __shared__ int tile_counts[256];
    group_expert_tiles<32>(ids, groups, experts, used, tokens, slots, tile_counts);
}

// 128 threads, grid=(ceil(rows/32),slots,1), static shared=16,768 bytes.
// Descriptors retain original assignment IDs, including for batched expert inputs.
extern "C" __global__ void grouped_matmul_reg32x32(const u8 *w, int kind, const float *x,
                                                 float *y, int width, int rows,
                                                 const int *groups, int used, int batched) {
    __shared__ float weights[32][65], activations[64][33];
    if (kind == 12) projection_register_tiled<12, 32, 32, 64, 128, true>(
        w, x, y, width, rows, 0, weights, activations, groups, used, batched);
    if (kind == 14) projection_register_tiled<14, 32, 32, 64, 128, true>(
        w, x, y, width, rows, 0, weights, activations, groups, used, batched);
}
#endif

template<bool Grouped>
__device__ int projection_assignment(const int *groups, int tile, int token, int tokens) {
    return Grouped ? groups[tile * 9 + token + 1]
                   : (tile * 8 + token < tokens ? tile * 8 + token : -1);
}

template<bool Grouped, int KTile>
__device__ void projection_inputs(const float *x, float activations[][8],
                                  int width, int tokens, const int *groups,
                                  int used, int batched, int first_k) {
    for (int i = threadIdx.x; i < KTile * 8; i += 256) {
        int k = i / 8, token = i % 8;
        int assignment = projection_assignment<Grouped>(groups, blockIdx.y, token, tokens);
        int input = Grouped && !batched && assignment >= 0 ? assignment / used : assignment;
        activations[k][token] = input >= 0 && first_k + k < width
            ? x[(u64)input * width + first_k + k] : 0.0f;
    }
}

// Experimental FP32 GEMM: 32 output rows x 8 assignments, staging K=256.
// Accumulation is FP32 in increasing K order, not the baseline's reduction tree.
template<int Kind, bool Grouped>
__device__ void projection_tiled(const u8 *w, const float *x, float *y,
                                 int width, int rows, int tokens, const int *groups,
                                 int used, int batched, float weights[][257],
                                 float activations[][8]) {
    int expert = Grouped ? groups[blockIdx.y * 9] : 0;
    if (expert < 0) return;
    int row = threadIdx.x / 8, token = threadIdx.x % 8;
    int first_row = blockIdx.x * 32;
    float sum = 0.0f;
    for (int first_k = 0; first_k < width; first_k += 256) {
        projection_weights<Kind, 32, 256>(w, weights, width, rows, expert, first_row, first_k);
        projection_inputs<Grouped, 256>(x, activations, width, tokens, groups, used, batched, first_k);
        __syncthreads();
        int count = min(256, width - first_k);
        for (int k = 0; k < count; k++) sum += weights[row][k] * activations[k][token];
        __syncthreads();
    }
    int assignment = projection_assignment<Grouped>(groups, blockIdx.y, token, tokens);
    if (first_row + row < rows && assignment >= 0)
        y[(u64)assignment * rows + first_row + row] = sum;
}

// Launch 256 threads, grid=(ceil(rows/32), ceil(tokens/8), 1), no dynamic scratch.
extern "C" __global__ void prefill_matmul_tiled(const u8 *w, int kind, const float *x,
                                               float *y, int width, int rows, int tokens) {
    __shared__ float weights[32][257], activations[256][8];
    switch (kind) {
        case 0: projection_tiled<0, false>(w, x, y, width, rows, tokens, nullptr, 1, 0, weights, activations); break;
        case 1: projection_tiled<1, false>(w, x, y, width, rows, tokens, nullptr, 1, 0, weights, activations); break;
        case 6: projection_tiled<6, false>(w, x, y, width, rows, tokens, nullptr, 1, 0, weights, activations); break;
        case 8: projection_tiled<8, false>(w, x, y, width, rows, tokens, nullptr, 1, 0, weights, activations); break;
        case 12: projection_tiled<12, false>(w, x, y, width, rows, tokens, nullptr, 1, 0, weights, activations); break;
        case 14: projection_tiled<14, false>(w, x, y, width, rows, tokens, nullptr, 1, 0, weights, activations); break;
    }
}

// Same block/row tile as dense; grid.y=slots and groups uses nine-int descriptors.
extern "C" __global__ void grouped_matmul_tiled(const u8 *w, int kind, const float *x,
                                               float *y, int width, int rows, const int *groups,
                                               int used, int batched) {
    __shared__ float weights[32][257], activations[256][8];
    switch (kind) {
        case 0: projection_tiled<0, true>(w, x, y, width, rows, 0, groups, used, batched, weights, activations); break;
        case 1: projection_tiled<1, true>(w, x, y, width, rows, 0, groups, used, batched, weights, activations); break;
        case 6: projection_tiled<6, true>(w, x, y, width, rows, 0, groups, used, batched, weights, activations); break;
        case 8: projection_tiled<8, true>(w, x, y, width, rows, 0, groups, used, batched, weights, activations); break;
        case 12: projection_tiled<12, true>(w, x, y, width, rows, 0, groups, used, batched, weights, activations); break;
        case 14: projection_tiled<14, true>(w, x, y, width, rows, 0, groups, used, batched, weights, activations); break;
    }
}
