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
        case 2: prompt_tile_workspace<2>(w, x, y, width, rows, tokens, partial); break;
        case 3: prompt_tile_workspace<3>(w, x, y, width, rows, tokens, partial); break;
        case 6: prompt_tile_workspace<6>(w, x, y, width, rows, tokens, partial); break;
        case 7: prompt_tile_workspace<7>(w, x, y, width, rows, tokens, partial); break;
        case 8: prompt_tile_workspace<8>(w, x, y, width, rows, tokens, partial); break;
        case 10: prompt_tile_workspace<10>(w, x, y, width, rows, tokens, partial); break;
        case 11: prompt_tile_workspace<11>(w, x, y, width, rows, tokens, partial); break;
        case 12: prompt_tile_workspace<12>(w, x, y, width, rows, tokens, partial); break;
        case 13: prompt_tile_workspace<13>(w, x, y, width, rows, tokens, partial); break;
        case 14: prompt_tile_workspace<14>(w, x, y, width, rows, tokens, partial); break;
        case 20: prompt_tile_workspace<20>(w, x, y, width, rows, tokens, partial); break;
        case 30: prompt_tile_workspace<30>(w, x, y, width, rows, tokens, partial); break;
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

#if defined(CUDA_M8_PROTOTYPES) || defined(CUDA_REGISTER_PREFILL)
// Default mapping: eight F32 accumulators/thread, two rows by four tokens.
// Shared decoded weights preserve the existing GGUF reconstruction arithmetic;
// every output traverses K in increasing order (unlike the matvec reduction tree).
template<int Kind, int RowTile, int TokenTile, int KTile, int Threads, bool Grouped = false,
         int DescriptorTile = TokenTile, bool CoalescedInputs = false,
         int RowsPerThread = 2, int TokensPerThread = 4, int KUnroll = 1>
__device__ void projection_register_tiled(const u8 *w, const float *x, float *y,
                                          int width, int rows, int tokens,
                                          float weights[][KTile + 1],
                                          float activations[][TokenTile + 1],
                                          const int *groups = nullptr, int used = 1, int batched = 0) {
    int expert = Grouped ? groups[blockIdx.y * (DescriptorTile + 1)] : 0;
    if (expert < 0) return;
    int first_row = blockIdx.x * RowTile, first_token = blockIdx.y * TokenTile;
    int row = (threadIdx.x / (TokenTile / TokensPerThread)) * RowsPerThread;
    int token = (threadIdx.x % (TokenTile / TokensPerThread)) * TokensPerThread;
    float sums[RowsPerThread][TokensPerThread] = {};
    for (int first_k = 0; first_k < width; first_k += KTile) {
        projection_weights<Kind, RowTile, KTile, Threads / 32>(
            w, weights, width, rows, expert, first_row, first_k);
        for (int i = threadIdx.x; i < KTile * TokenTile; i += Threads) {
            int k = CoalescedInputs ? i % KTile : i / TokenTile;
            int t = CoalescedInputs ? i / KTile : i % TokenTile;
            if (Grouped) {
                int assignment = groups[blockIdx.y * (DescriptorTile + 1) + t + 1];
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
        if (row < RowTile) {
            #pragma unroll (KUnroll)
            for (int k = 0; k < count; k++) {
                float values[TokensPerThread];
                #pragma unroll
                for (int t = 0; t < TokensPerThread; t++) values[t] = activations[k][token + t];
                #pragma unroll
                for (int r = 0; r < RowsPerThread; r++) {
                    float decoded = weights[row + r][k];
                    #pragma unroll
                    for (int t = 0; t < TokensPerThread; t++) sums[r][t] += decoded * values[t];
                }
            }
        }
        __syncthreads();
    }
    if (row >= RowTile) return;
    #pragma unroll
    for (int r = 0; r < RowsPerThread; r++) {
        #pragma unroll
        for (int t = 0; t < TokensPerThread; t++) {
            if (Grouped) {
                int assignment = groups[blockIdx.y * (DescriptorTile + 1) + token + t + 1];
                if (first_row + row + r < rows && assignment >= 0)
                    y[(u64)assignment * rows + first_row + row + r] = sums[r][t];
            } else {
                if (first_row + row + r < rows && first_token + token + t < tokens)
                    y[(u64)(first_token + token + t) * rows + first_row + row + r] = sums[r][t];
            }
        }
    }
}

template<int Kind, int RowTile, int TokenTile>
__device__ void projection_reference_legacy(const u8 *w, const float *x, float *y,
                                           int width, int rows, int tokens,
                                           float partial[][TokenTile][128]) {
    int row = blockIdx.x * RowTile + threadIdx.x / 128;
    int lane = threadIdx.x % 128, first_token = blockIdx.y * TokenTile;
    float sums[2][TokenTile] = {};
    for (int first_k = 0; first_k < width; first_k += 256) {
        #pragma unroll
        for (int half = 0; half < 2; half++) {
            int k = first_k + lane + half * 128;
            if (k < width && row < rows) {
                float decoded = weight(w, Kind, (u64)row * width + k);
                #pragma unroll
                for (int t = 0; t < TokenTile; t++)
                    if (first_token + t < tokens)
                        sums[half][t] += decoded * x[(u64)(first_token + t) * width + k];
            }
        }
    }
    int r = threadIdx.x / 128;
    #pragma unroll
    for (int t = 0; t < TokenTile; t++)
        partial[r][t][lane] = sums[0][t] + sums[1][t];
    __syncthreads();
    if (lane < 64) {
        #pragma unroll
        for (int t = 0; t < TokenTile; t++)
            partial[r][t][lane] += partial[r][t][lane + 64];
    }
    __syncthreads();
    if (lane < 32) {
        #pragma unroll
        for (int t = 0; t < TokenTile; t++) {
            float sum = partial[r][t][lane] + partial[r][t][lane + 32];
            for (int stride = 16; stride > 0; stride /= 2)
                sum += __shfl_down_sync(0xffffffff, sum, stride);
            if (lane == 0 && row < rows && first_token + t < tokens)
                y[(u64)(first_token + t) * rows + row] = sum;
        }
    }
}

#ifdef CUDA_M8_PROTOTYPES
// Pair the original lanes p and p+128 without changing either accumulation.
// The shared/warp reduction then reproduces the legacy 256-lane tree exactly.
template<int Kind, int RowTile, int TokenTile, int RowsPerLane = 1, bool Packed = false,
         int KUnroll = 1, bool Compact = false, int LaneGroup = 128, bool Interleaved = false>
__device__ void projection_reference_tiled(const u8 *w, const float *x, float *y,
                                          int width, int rows, int tokens,
                                          float partial[][TokenTile][Compact ? LaneGroup / 2 : LaneGroup]) {
    static_assert(RowTile % RowsPerLane == 0, "whole rows per lane group");
    static_assert(LaneGroup == 32 || LaneGroup == 64 || LaneGroup == 128, "whole warp lane groups");
    static_assert(!Interleaved || (LaneGroup == 128 && RowTile == 2 && RowsPerLane == 1),
                  "interleaved half warps own two rows");
    int group = Interleaved ? (threadIdx.x / 16) % 2 : threadIdx.x / LaneGroup;
    int first_row = blockIdx.x * RowTile + group * RowsPerLane;
    int lane = Interleaved ? threadIdx.x / 32 * 16 + threadIdx.x % 16
                          : threadIdx.x % LaneGroup;
    int first_token = blockIdx.y * TokenTile;
    float sums[RowsPerLane][256 / LaneGroup][TokenTile] = {};
    #pragma unroll (KUnroll)
    for (int first_k = 0; first_k < width; first_k += 256) {
        #pragma unroll
        for (int half = 0; half < 256 / LaneGroup; half++) {
            int k = first_k + lane + half * LaneGroup;
            if (k < width) {
                float decoded[RowsPerLane] = {};
                #pragma unroll
                for (int r = 0; r < RowsPerLane; r++) {
                    if (first_row + r < rows) {
                        u64 index = (u64)(first_row + r) * width + k;
                        decoded[r] = Packed && width % 32 == 0
                            ? warp_weight<Kind>(w, index) : weight(w, Kind, index);
                    }
                }
                #pragma unroll
                for (int t = 0; t < TokenTile; t++) {
                    if (first_token + t < tokens) {
                        float value = x[(u64)(first_token + t) * width + k];
                        #pragma unroll
                        for (int r = 0; r < RowsPerLane; r++)
                            if (first_row + r < rows) sums[r][half][t] += decoded[r] * value;
                    }
                }
            }
        }
    }
    int first_partial = group * RowsPerLane;
    // Fold whole legacy lane streams before moving the remaining partials.
    #pragma unroll
    for (int r = 0; r < RowsPerLane; r++) {
        #pragma unroll
        for (int t = 0; t < TokenTile; t++) {
            #pragma unroll
            for (int stride = 128 / LaneGroup; stride > 0; stride /= 2) {
                #pragma unroll
                for (int p = 0; p < stride; p++) sums[r][p][t] += sums[r][p + stride][t];
            }
        }
    }
    if (LaneGroup < 128) {
        if (LaneGroup == 64) {
            if (lane >= 32) {
                #pragma unroll
                for (int r = 0; r < RowsPerLane; r++) {
                    #pragma unroll
                    for (int t = 0; t < TokenTile; t++)
                        partial[first_partial + r][t][lane - 32] = sums[r][0][t];
                }
            }
            __syncthreads();
        }
        if (lane < 32) {
            #pragma unroll
            for (int r = 0; r < RowsPerLane; r++) {
                #pragma unroll
                for (int t = 0; t < TokenTile; t++) {
                    float sum = sums[r][0][t];
                    if (LaneGroup == 64) sum += partial[first_partial + r][t][lane];
                    for (int stride = 16; stride > 0; stride /= 2)
                        sum += __shfl_down_sync(0xffffffff, sum, stride);
                    if (lane == 0 && first_row + r < rows && first_token + t < tokens)
                        y[(u64)(first_token + t) * rows + first_row + r] = sum;
                }
            }
        }
        return;
    }
    #pragma unroll
    for (int r = 0; r < RowsPerLane; r++) {
        #pragma unroll
        for (int t = 0; t < TokenTile; t++) {
            if (!Compact || lane >= 64)
                partial[first_partial + r][t][Compact ? lane - 64 : lane] = sums[r][0][t];
        }
    }
    __syncthreads();
    if (lane < 64) {
        #pragma unroll
        for (int r = 0; r < RowsPerLane; r++) {
            #pragma unroll
            for (int t = 0; t < TokenTile; t++) {
                if (Compact) {
                    sums[r][0][t] += partial[first_partial + r][t][lane];
                    if (lane >= 32) partial[first_partial + r][t][lane] = sums[r][0][t];
                } else {
                    partial[first_partial + r][t][lane] += partial[first_partial + r][t][lane + 64];
                }
            }
        }
    }
    __syncthreads();
    if (Interleaved) {
        if (lane < 32) {
            #pragma unroll
            for (int t = 0; t < TokenTile; t++) {
                sums[0][0][t] = partial[first_partial][t][lane] + partial[first_partial][t][lane + 32];
                if (lane >= 16) partial[first_partial][t][lane + 64] = sums[0][0][t];
            }
        }
        __syncthreads();
        if (lane < 16) {
            #pragma unroll
            for (int t = 0; t < TokenTile; t++) {
                float sum = sums[0][0][t] + partial[first_partial][t][lane + 80];
                for (int stride = 8; stride > 0; stride /= 2)
                    sum += __shfl_down_sync(0xffffffff, sum, stride, 16);
                if (lane == 0 && first_row < rows && first_token + t < tokens)
                    y[(u64)(first_token + t) * rows + first_row] = sum;
            }
        }
        return;
    }
    if (lane < 32) {
        #pragma unroll
        for (int r = 0; r < RowsPerLane; r++) {
            #pragma unroll
            for (int t = 0; t < TokenTile; t++) {
                float sum = (Compact ? sums[r][0][t] : partial[first_partial + r][t][lane])
                    + partial[first_partial + r][t][lane + 32];
                for (int stride = 16; stride > 0; stride /= 2)
                    sum += __shfl_down_sync(0xffffffff, sum, stride);
                if (lane == 0 && first_row + r < rows && first_token + t < tokens)
                    y[(u64)(first_token + t) * rows + first_row + r] = sum;
            }
        }
    }
}

#endif
#ifdef CUDA_M8_PROTOTYPES
extern "C" __global__ void prefill_matmul_ref2x8(const u8 *w, int kind, const float *x,
                                                float *y, int width, int rows, int tokens) {
    __shared__ float partial[2][8][128];
    if (kind == 12) projection_reference_tiled<12, 2, 8>(w, x, y, width, rows, tokens, partial);
    if (kind == 14) projection_reference_tiled<14, 2, 8>(w, x, y, width, rows, tokens, partial);
}
#endif
extern "C" __global__ void prefill_matmul_ref2x16(const u8 *w, int kind, const float *x,
                                                 float *y, int width, int rows, int tokens) {
    __shared__ float partial[2][16][128];
    if (kind == 12) projection_reference_legacy<12, 2, 16>(w, x, y, width, rows, tokens, partial);
    if (kind == 14) projection_reference_legacy<14, 2, 16>(w, x, y, width, rows, tokens, partial);
}
#ifdef CUDA_M8_PROTOTYPES
extern "C" __global__ void prefill_matmul_refinterleave2x16(const u8 *w, int kind, const float *x,
                                                         float *y, int width, int rows, int tokens) {
    __shared__ float partial[2][16][128];
    if (kind == 12) projection_reference_tiled<12, 2, 16, 1, false, 1, false, 128, true>(w, x, y, width, rows, tokens, partial);
    if (kind == 14) projection_reference_tiled<14, 2, 16, 1, false, 1, false, 128, true>(w, x, y, width, rows, tokens, partial);
}
extern "C" __global__ void prefill_matmul_refinterleave2x8(const u8 *w, int kind, const float *x,
                                                        float *y, int width, int rows, int tokens) {
    __shared__ float partial[2][8][128];
    if (kind == 12) projection_reference_tiled<12, 2, 8, 1, false, 1, false, 128, true>(w, x, y, width, rows, tokens, partial);
    if (kind == 14) projection_reference_tiled<14, 2, 8, 1, false, 1, false, 128, true>(w, x, y, width, rows, tokens, partial);
}
extern "C" __global__ void prefill_matmul_ref4x16(const u8 *w, int kind, const float *x,
                                                 float *y, int width, int rows, int tokens) {
    __shared__ float partial[4][16][128];
    if (kind == 12) projection_reference_tiled<12, 4, 16>(w, x, y, width, rows, tokens, partial);
    if (kind == 14) projection_reference_tiled<14, 4, 16>(w, x, y, width, rows, tokens, partial);
}
extern "C" __global__ void prefill_matmul_reuse2x16(const u8 *w, int kind, const float *x,
                                                 float *y, int width, int rows, int tokens) {
    __shared__ float partial[2][16][128];
    if (kind == 12) projection_reference_tiled<12, 2, 16, 2>(w, x, y, width, rows, tokens, partial);
    if (kind == 14) projection_reference_tiled<14, 2, 16, 2>(w, x, y, width, rows, tokens, partial);
}
extern "C" __global__ void prefill_matmul_reuse4x8(const u8 *w, int kind, const float *x,
                                                float *y, int width, int rows, int tokens) {
    __shared__ float partial[4][8][128];
    if (kind == 12) projection_reference_tiled<12, 4, 8, 2>(w, x, y, width, rows, tokens, partial);
    if (kind == 14) projection_reference_tiled<14, 4, 8, 2>(w, x, y, width, rows, tokens, partial);
}
extern "C" __global__ void prefill_matmul_reuse4x16(const u8 *w, int kind, const float *x,
                                                 float *y, int width, int rows, int tokens) {
    __shared__ float partial[4][16][128];
    if (kind == 12) projection_reference_tiled<12, 4, 16, 2>(w, x, y, width, rows, tokens, partial);
    if (kind == 14) projection_reference_tiled<14, 4, 16, 2>(w, x, y, width, rows, tokens, partial);
}
extern "C" __global__ void prefill_matmul_reuse4x8_r4(const u8 *w, int kind, const float *x,
                                                   float *y, int width, int rows, int tokens) {
    __shared__ float partial[4][8][128];
    if (kind == 12) projection_reference_tiled<12, 4, 8, 4>(w, x, y, width, rows, tokens, partial);
    if (kind == 14) projection_reference_tiled<14, 4, 8, 4>(w, x, y, width, rows, tokens, partial);
}
extern "C" __global__ void prefill_matmul_refpacked2x16(const u8 *w, int kind, const float *x,
                                                     float *y, int width, int rows, int tokens) {
    __shared__ float partial[2][16][128];
    if (kind == 12) projection_reference_tiled<12, 2, 16, 1, true>(w, x, y, width, rows, tokens, partial);
    if (kind == 14) projection_reference_tiled<14, 2, 16, 1, true>(w, x, y, width, rows, tokens, partial);
}
extern "C" __global__ void prefill_matmul_ref2x16_u2(const u8 *w, int kind, const float *x,
                                                  float *y, int width, int rows, int tokens) {
    __shared__ float partial[2][16][128];
    if (kind == 12) projection_reference_tiled<12, 2, 16, 1, false, 2>(w, x, y, width, rows, tokens, partial);
    if (kind == 14) projection_reference_tiled<14, 2, 16, 1, false, 2>(w, x, y, width, rows, tokens, partial);
}
extern "C" __global__ void prefill_matmul_ref2x16_u4(const u8 *w, int kind, const float *x,
                                                  float *y, int width, int rows, int tokens) {
    __shared__ float partial[2][16][128];
    if (kind == 12) projection_reference_tiled<12, 2, 16, 1, false, 4>(w, x, y, width, rows, tokens, partial);
    if (kind == 14) projection_reference_tiled<14, 2, 16, 1, false, 4>(w, x, y, width, rows, tokens, partial);
}
extern "C" __global__ void prefill_matmul_ref2x16_u8(const u8 *w, int kind, const float *x,
                                                  float *y, int width, int rows, int tokens) {
    __shared__ float partial[2][16][128];
    if (kind == 12) projection_reference_tiled<12, 2, 16, 1, false, 8>(w, x, y, width, rows, tokens, partial);
    if (kind == 14) projection_reference_tiled<14, 2, 16, 1, false, 8>(w, x, y, width, rows, tokens, partial);
}
extern "C" __global__ void prefill_matmul_ref2x16_u16(const u8 *w, int kind, const float *x,
                                                   float *y, int width, int rows, int tokens) {
    __shared__ float partial[2][16][128];
    if (kind == 12) projection_reference_tiled<12, 2, 16, 1, false, 16>(w, x, y, width, rows, tokens, partial);
    if (kind == 14) projection_reference_tiled<14, 2, 16, 1, false, 16>(w, x, y, width, rows, tokens, partial);
}
extern "C" __global__ void prefill_matmul_ref2048(const u8 *w, int kind, const float *x,
                                               float *y, int width, int rows, int tokens) {
    __shared__ float partial[2][16][128];
    if (kind == 12) projection_reference_tiled<12, 2, 16>(w, x, y, 2048, rows, tokens, partial);
    if (kind == 14) projection_reference_tiled<14, 2, 16>(w, x, y, 2048, rows, tokens, partial);
}
extern "C" __global__ void prefill_matmul_ref4096(const u8 *w, int kind, const float *x,
                                               float *y, int width, int rows, int tokens) {
    __shared__ float partial[2][16][128];
    if (kind == 12) projection_reference_tiled<12, 2, 16>(w, x, y, 4096, rows, tokens, partial);
    if (kind == 14) projection_reference_tiled<14, 2, 16>(w, x, y, 4096, rows, tokens, partial);
}
extern "C" __global__ void prefill_matmul_refcompact2x16(const u8 *w, int kind, const float *x,
                                                      float *y, int width, int rows, int tokens) {
    __shared__ float partial[2][16][64];
    if (kind == 12) projection_reference_tiled<12, 2, 16, 1, false, 1, true>(w, x, y, width, rows, tokens, partial);
    if (kind == 14) projection_reference_tiled<14, 2, 16, 1, false, 1, true>(w, x, y, width, rows, tokens, partial);
}
extern "C" __global__ void prefill_matmul_refcompact2x8(const u8 *w, int kind, const float *x,
                                                     float *y, int width, int rows, int tokens) {
    __shared__ float partial[2][8][64];
    if (kind == 12) projection_reference_tiled<12, 2, 8, 1, false, 1, true>(w, x, y, width, rows, tokens, partial);
    if (kind == 14) projection_reference_tiled<14, 2, 8, 1, false, 1, true>(w, x, y, width, rows, tokens, partial);
}
extern "C" __global__ void prefill_matmul_ref64_2x8(const u8 *w, int kind, const float *x,
                                                 float *y, int width, int rows, int tokens) {
    __shared__ float partial[2][8][32];
    if (kind == 12) projection_reference_tiled<12, 2, 8, 1, false, 1, true, 64>(w, x, y, width, rows, tokens, partial);
    if (kind == 14) projection_reference_tiled<14, 2, 8, 1, false, 1, true, 64>(w, x, y, width, rows, tokens, partial);
}
extern "C" __global__ void prefill_matmul_ref64_2x16(const u8 *w, int kind, const float *x,
                                                  float *y, int width, int rows, int tokens) {
    __shared__ float partial[2][16][32];
    if (kind == 12) projection_reference_tiled<12, 2, 16, 1, false, 1, true, 64>(w, x, y, width, rows, tokens, partial);
    if (kind == 14) projection_reference_tiled<14, 2, 16, 1, false, 1, true, 64>(w, x, y, width, rows, tokens, partial);
}
extern "C" __global__ void prefill_matmul_ref32_4x4(const u8 *w, int kind, const float *x,
                                                 float *y, int width, int rows, int tokens) {
    __shared__ float partial[4][4][16];
    if (kind == 12) projection_reference_tiled<12, 4, 4, 1, false, 1, true, 32>(w, x, y, width, rows, tokens, partial);
    if (kind == 14) projection_reference_tiled<14, 4, 4, 1, false, 1, true, 32>(w, x, y, width, rows, tokens, partial);
}
extern "C" __global__ void prefill_matmul_ref32_4x8(const u8 *w, int kind, const float *x,
                                                 float *y, int width, int rows, int tokens) {
    __shared__ float partial[4][8][16];
    if (kind == 12) projection_reference_tiled<12, 4, 8, 1, false, 1, true, 32>(w, x, y, width, rows, tokens, partial);
    if (kind == 14) projection_reference_tiled<14, 4, 8, 1, false, 1, true, 32>(w, x, y, width, rows, tokens, partial);
}
#endif

// Keep the 33-int descriptors; narrow only the useful compute in partial tiles.
extern "C" __global__ void grouped_matmul_reg32_counted(const u8 *w, int kind, const float *x,
                                                      float *y, int width, int rows,
                                                      const int *groups, int used, int batched) {
    if (groups[blockIdx.y * 33] < 0) return;
    __shared__ float weights[32][65], activations[64][33];
    int tile = groups[blockIdx.y * 33 + 9] < 0 ? 8
             : groups[blockIdx.y * 33 + 17] < 0 ? 16 : 32;
    if (tile == 8) {
        auto input = reinterpret_cast<float (*)[9]>(activations);
        if (kind == 12) projection_register_tiled<12, 32, 8, 64, 128, true, 32>(
            w, x, y, width, rows, 0, weights, input, groups, used, batched);
        if (kind == 14) projection_register_tiled<14, 32, 8, 64, 128, true, 32>(
            w, x, y, width, rows, 0, weights, input, groups, used, batched);
    } else if (tile == 16) {
        auto input = reinterpret_cast<float (*)[17]>(activations);
        if (kind == 12) projection_register_tiled<12, 32, 16, 64, 128, true, 32>(
            w, x, y, width, rows, 0, weights, input, groups, used, batched);
        if (kind == 14) projection_register_tiled<14, 32, 16, 64, 128, true, 32>(
            w, x, y, width, rows, 0, weights, input, groups, used, batched);
    } else {
        if (kind == 12) projection_register_tiled<12, 32, 32, 64, 128, true>(
            w, x, y, width, rows, 0, weights, activations, groups, used, batched);
        if (kind == 14) projection_register_tiled<14, 32, 32, 64, 128, true>(
            w, x, y, width, rows, 0, weights, activations, groups, used, batched);
    }
}

template<int KTile, int KUnroll = 1>
__device__ void grouped_coalesced_register(const u8 *w, int kind, const float *x,
                                           float *y, int width, int rows,
                                           const int *groups, int used, int batched) {
    if (groups[blockIdx.y * 33] < 0) return;
    __shared__ float weights[32][KTile + 1], activations[KTile][33];
    int tile = groups[blockIdx.y * 33 + 9] < 0 ? 8
             : groups[blockIdx.y * 33 + 17] < 0 ? 16 : 32;
    if (tile == 8) {
        auto input = reinterpret_cast<float (*)[9]>(activations);
        if (kind == 12) projection_register_tiled<12, 32, 8, KTile, 128, true, 32, true, 2, 4, KUnroll>(
            w, x, y, width, rows, 0, weights, input, groups, used, batched);
        if (kind == 14) projection_register_tiled<14, 32, 8, KTile, 128, true, 32, true, 2, 4, KUnroll>(
            w, x, y, width, rows, 0, weights, input, groups, used, batched);
    } else if (tile == 16) {
        auto input = reinterpret_cast<float (*)[17]>(activations);
        if (kind == 12) projection_register_tiled<12, 32, 16, KTile, 128, true, 32, true, 2, 4, KUnroll>(
            w, x, y, width, rows, 0, weights, input, groups, used, batched);
        if (kind == 14) projection_register_tiled<14, 32, 16, KTile, 128, true, 32, true, 2, 4, KUnroll>(
            w, x, y, width, rows, 0, weights, input, groups, used, batched);
    } else {
        if (kind == 12) projection_register_tiled<12, 32, 32, KTile, 128, true, 32, true, 2, 4, KUnroll>(
            w, x, y, width, rows, 0, weights, activations, groups, used, batched);
        if (kind == 14) projection_register_tiled<14, 32, 32, KTile, 128, true, 32, true, 2, 4, KUnroll>(
            w, x, y, width, rows, 0, weights, activations, groups, used, batched);
    }
}

extern "C" __global__ void grouped_matmul_coalesced_k32(const u8 *w, int kind, const float *x,
                                                       float *y, int width, int rows,
                                                       const int *groups, int used, int batched) {
    grouped_coalesced_register<32>(w, kind, x, y, width, rows, groups, used, batched);
}
#ifdef CUDA_M8_PROTOTYPES
extern "C" __global__ void grouped_matmul_unroll2(const u8 *w, int kind, const float *x,
                                                 float *y, int width, int rows,
                                                 const int *groups, int used, int batched) {
    grouped_coalesced_register<32, 2>(w, kind, x, y, width, rows, groups, used, batched);
}
extern "C" __global__ void grouped_matmul_unroll4(const u8 *w, int kind, const float *x,
                                                 float *y, int width, int rows,
                                                 const int *groups, int used, int batched) {
    grouped_coalesced_register<32, 4>(w, kind, x, y, width, rows, groups, used, batched);
}
#endif
extern "C" __global__ void grouped_matmul_unroll8(const u8 *w, int kind, const float *x,
                                                 float *y, int width, int rows,
                                                 const int *groups, int used, int batched) {
    grouped_coalesced_register<32, 8>(w, kind, x, y, width, rows, groups, used, batched);
}
#ifdef CUDA_M8_PROTOTYPES
extern "C" __global__ void grouped_matmul_unroll16(const u8 *w, int kind, const float *x,
                                                  float *y, int width, int rows,
                                                  const int *groups, int used, int batched) {
    grouped_coalesced_register<32, 16>(w, kind, x, y, width, rows, groups, used, batched);
}
extern "C" __global__ void grouped_matmul_unroll32(const u8 *w, int kind, const float *x,
                                                  float *y, int width, int rows,
                                                  const int *groups, int used, int batched) {
    grouped_coalesced_register<32, 32>(w, kind, x, y, width, rows, groups, used, batched);
}

template<int Kind, int RowTile, int TokenTile, int Threads, int TokenValues>
__device__ void grouped_balanced_tile(const u8 *w, const float *x, float *y,
                                      int width, int rows, float weights[][33],
                                      float activations[][TokenTile + 1],
                                      const int *groups, int used, int batched) {
    constexpr int RowValues = RowTile * TokenTile / (Threads * TokenValues);
    static_assert(RowValues > 0 && RowValues * TokenValues * Threads == RowTile * TokenTile,
                  "every thread must own complete outputs");
    projection_register_tiled<Kind, RowTile, TokenTile, 32, Threads, true, 32,
                              true, RowValues, TokenValues>(
        w, x, y, width, rows, 0, weights, activations, groups, used, batched);
}

template<int RowTile, int Threads>
__device__ void grouped_balanced_register(const u8 *w, int kind, const float *x,
                                          float *y, int width, int rows,
                                          const int *groups, int used, int batched) {
    if (groups[blockIdx.y * 33] < 0) return;
    __shared__ float weights[RowTile][33], activations[32][33];
    int tile = groups[blockIdx.y * 33 + 9] < 0 ? 8
             : groups[blockIdx.y * 33 + 17] < 0 ? 16 : 32;
    if (tile == 8) {
        constexpr int TokenValues = RowTile * 8 / Threads < 4 ? 2 : 4;
        auto input = reinterpret_cast<float (*)[9]>(activations);
        if (kind == 12) grouped_balanced_tile<12, RowTile, 8, Threads, TokenValues>(
            w, x, y, width, rows, weights, input, groups, used, batched);
        if (kind == 14) grouped_balanced_tile<14, RowTile, 8, Threads, TokenValues>(
            w, x, y, width, rows, weights, input, groups, used, batched);
    } else if (tile == 16) {
        auto input = reinterpret_cast<float (*)[17]>(activations);
        if (kind == 12) grouped_balanced_tile<12, RowTile, 16, Threads, 4>(
            w, x, y, width, rows, weights, input, groups, used, batched);
        if (kind == 14) grouped_balanced_tile<14, RowTile, 16, Threads, 4>(
            w, x, y, width, rows, weights, input, groups, used, batched);
    } else {
        if (kind == 12) grouped_balanced_tile<12, RowTile, 32, Threads, 4>(
            w, x, y, width, rows, weights, activations, groups, used, batched);
        if (kind == 14) grouped_balanced_tile<14, RowTile, 32, Threads, 4>(
            w, x, y, width, rows, weights, activations, groups, used, batched);
    }
}

extern "C" __global__ void grouped_matmul_balanced32(const u8 *w, int kind, const float *x,
                                                    float *y, int width, int rows,
                                                    const int *groups, int used, int batched) {
    grouped_balanced_register<32, 128>(w, kind, x, y, width, rows, groups, used, batched);
}
extern "C" __global__ void grouped_matmul_wide64(const u8 *w, int kind, const float *x,
                                                float *y, int width, int rows,
                                                const int *groups, int used, int batched) {
    grouped_balanced_register<64, 256>(w, kind, x, y, width, rows, groups, used, batched);
}
extern "C" __global__ void grouped_matmul_balanced64(const u8 *w, int kind, const float *x,
                                                    float *y, int width, int rows,
                                                    const int *groups, int used, int batched) {
    grouped_balanced_register<64, 128>(w, kind, x, y, width, rows, groups, used, batched);
}
extern "C" __global__ void grouped_matmul_balanced128(const u8 *w, int kind, const float *x,
                                                     float *y, int width, int rows,
                                                     const int *groups, int used, int batched) {
    grouped_balanced_register<128, 256>(w, kind, x, y, width, rows, groups, used, batched);
}

extern "C" __global__ void grouped_matmul_coalesced_k64(const u8 *w, int kind, const float *x,
                                                       float *y, int width, int rows,
                                                       const int *groups, int used, int batched) {
    grouped_coalesced_register<64>(w, kind, x, y, width, rows, groups, used, batched);
}
extern "C" __global__ void grouped_matmul_coalesced_k128(const u8 *w, int kind, const float *x,
                                                        float *y, int width, int rows,
                                                        const int *groups, int used, int batched) {
    grouped_coalesced_register<128>(w, kind, x, y, width, rows, groups, used, batched);
}

template<int Kind, int RowTile = 32, int TokenTile = 32, int RowsPerThread = 2,
         bool CoalescedInputs = false, int KTile = 64>
__device__ void grouped_gate_up_register(const u8 *gate, const u8 *up, const float *x,
                                         float *y, int width, int rows, const int *groups,
                                         int used, float wg[][KTile + 1], float wu[][KTile + 1],
                                         float activations[][TokenTile + 1]) {
    int expert = groups[blockIdx.y * 33];
    if (expert < 0) return;
    int first_row = blockIdx.x * RowTile;
    int row = (threadIdx.x / (TokenTile / 4)) * RowsPerThread;
    int token = (threadIdx.x % (TokenTile / 4)) * 4;
    float g[RowsPerThread][4] = {}, u[RowsPerThread][4] = {};
    for (int first_k = 0; first_k < width; first_k += KTile) {
        projection_weights<Kind, RowTile, KTile, 4>(gate, wg, width, rows, expert, first_row, first_k);
        projection_weights<Kind, RowTile, KTile, 4>(up, wu, width, rows, expert, first_row, first_k);
        for (int i = threadIdx.x; i < KTile * TokenTile; i += 128) {
            int k = CoalescedInputs ? i % KTile : i / TokenTile;
            int t = CoalescedInputs ? i / KTile : i % TokenTile;
            int assignment = groups[blockIdx.y * 33 + t + 1];
            activations[k][t] = assignment >= 0 && first_k + k < width
                ? x[(u64)(assignment / used) * width + first_k + k] : 0.0f;
        }
        __syncthreads();
        int count = min(KTile, width - first_k);
        if (row < RowTile) {
            #pragma unroll 1
            for (int k = 0; k < count; k++) {
                float values[4];
                #pragma unroll
                for (int t = 0; t < 4; t++) values[t] = activations[k][token + t];
                #pragma unroll
                for (int r = 0; r < RowsPerThread; r++) {
                    float a = wg[row + r][k], b = wu[row + r][k];
                    #pragma unroll
                    for (int t = 0; t < 4; t++) {
                        g[r][t] += a * values[t];
                        u[r][t] += b * values[t];
                    }
                }
            }
        }
        __syncthreads();
    }
    if (row >= RowTile) return;
    #pragma unroll
    for (int r = 0; r < RowsPerThread; r++) {
        #pragma unroll
        for (int t = 0; t < 4; t++) {
            int assignment = groups[blockIdx.y * 33 + token + t + 1];
            if (first_row + row + r < rows && assignment >= 0)
                y[(u64)assignment * rows + first_row + row + r] =
                    g[r][t] / (1.0f + expf(-g[r][t])) * u[r][t];
        }
    }
}

extern "C" __global__ void grouped_gate_up_reg32x32(const u8 *gate, const u8 *up, int kind,
                                                   const float *x, float *y, int width, int rows,
                                                   const int *groups, int used) {
    __shared__ float wg[32][65], wu[32][65], activations[64][33];
    if (kind == 12) grouped_gate_up_register<12>(
        gate, up, x, y, width, rows, groups, used, wg, wu, activations);
    if (kind == 14) grouped_gate_up_register<14>(
        gate, up, x, y, width, rows, groups, used, wg, wu, activations);
}

template<int KTile>
__device__ void grouped_gate_up_counted(const u8 *gate, const u8 *up, int kind,
                                       const float *x, float *y, int width, int rows,
                                       const int *groups, int used) {
    if (groups[blockIdx.y * 33] < 0) return;
    __shared__ float wg[16][KTile + 1], wu[16][KTile + 1], activations[KTile][33];
    int tile = groups[blockIdx.y * 33 + 9] < 0 ? 8
             : groups[blockIdx.y * 33 + 17] < 0 ? 16 : 32;
    if (tile == 8) {
        auto input = reinterpret_cast<float (*)[9]>(activations);
        if (kind == 12) grouped_gate_up_register<12, 16, 8, 1, true, KTile>(
            gate, up, x, y, width, rows, groups, used, wg, wu, input);
        if (kind == 14) grouped_gate_up_register<14, 16, 8, 1, true, KTile>(
            gate, up, x, y, width, rows, groups, used, wg, wu, input);
    } else if (tile == 16) {
        auto input = reinterpret_cast<float (*)[17]>(activations);
        if (kind == 12) grouped_gate_up_register<12, 16, 16, 1, true, KTile>(
            gate, up, x, y, width, rows, groups, used, wg, wu, input);
        if (kind == 14) grouped_gate_up_register<14, 16, 16, 1, true, KTile>(
            gate, up, x, y, width, rows, groups, used, wg, wu, input);
    } else {
        if (kind == 12) grouped_gate_up_register<12, 16, 32, 1, true, KTile>(
            gate, up, x, y, width, rows, groups, used, wg, wu, activations);
        if (kind == 14) grouped_gate_up_register<14, 16, 32, 1, true, KTile>(
            gate, up, x, y, width, rows, groups, used, wg, wu, activations);
    }
}
extern "C" __global__ void grouped_gate_up_reg16_counted(const u8 *gate, const u8 *up, int kind,
                                                        const float *x, float *y, int width, int rows,
                                                        const int *groups, int used) {
    grouped_gate_up_counted<64>(gate, up, kind, x, y, width, rows, groups, used);
}
extern "C" __global__ void grouped_gate_up_reg16_k32(const u8 *gate, const u8 *up, int kind,
                                                    const float *x, float *y, int width, int rows,
                                                    const int *groups, int used) {
    grouped_gate_up_counted<32>(gate, up, kind, x, y, width, rows, groups, used);
}
#endif

// Callers dispatch only Q4_K/Q6_K. Other kinds intentionally do no work.
// Compile with fmad=false. No dynamic shared memory or persistent scratch.
// (rows,tokens,K)=(64,16,64), 128 threads, static shared=20,992 bytes.
#ifdef CUDA_M8_PROTOTYPES
extern "C" __global__ void prefill_matmul_reg64x16(const u8 *w, int kind, const float *x,
                                                 float *y, int width, int rows, int tokens) {
    __shared__ float weights[64][65], activations[64][17];
    if (kind == 12) projection_register_tiled<12, 64, 16, 64, 128>(w, x, y, width, rows, tokens, weights, activations);
    if (kind == 14) projection_register_tiled<14, 64, 16, 64, 128>(w, x, y, width, rows, tokens, weights, activations);
}
#endif

// (rows,tokens,K)=(32,32,64), 128 threads, static shared=16,768 bytes.
#ifdef CUDA_M8_PROTOTYPES
extern "C" __global__ void prefill_matmul_reg32x32(const u8 *w, int kind, const float *x,
                                                 float *y, int width, int rows, int tokens) {
    __shared__ float weights[32][65], activations[64][33];
    if (kind == 12) projection_register_tiled<12, 32, 32, 64, 128>(w, x, y, width, rows, tokens, weights, activations);
    if (kind == 14) projection_register_tiled<14, 32, 32, 64, 128>(w, x, y, width, rows, tokens, weights, activations);
}
#endif

// (rows,tokens,K)=(64,32,32), 256 threads, static shared=12,672 bytes.
#ifdef CUDA_M8_PROTOTYPES
extern "C" __global__ void prefill_matmul_reg64x32(const u8 *w, int kind, const float *x,
                                                 float *y, int width, int rows, int tokens) {
    __shared__ float weights[64][33], activations[32][33];
    if (kind == 12) projection_register_tiled<12, 64, 32, 32, 256>(w, x, y, width, rows, tokens, weights, activations);
    if (kind == 14) projection_register_tiled<14, 64, 32, 32, 256>(w, x, y, width, rows, tokens, weights, activations);
}
#endif

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
