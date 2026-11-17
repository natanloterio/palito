// Keep rejected tiling separate so default F32 retains M5's compilation input.
template<int Kind, bool Grouped = false, int Tile = 4, int RowTile = 2>
__device__ void prompt_tile_rows2(const u8 *w, const float *x, float *y,
                           int width, int rows, int tokens,
                           const int *groups = nullptr, int used = 1, int batched = 0) {
    int row = blockIdx.x * RowTile, first = blockIdx.y * Tile, lane = threadIdx.x;
    int expert = Grouped ? groups[blockIdx.y * 5] : 0;
    if (expert < 0) return;
    int assignments[Tile];
    #pragma unroll
    for (int t = 0; t < Tile; t++)
        assignments[t] = Grouped ? groups[blockIdx.y * 5 + t + 1]
                                : (first + t < tokens ? first + t : -1);
    float sums[RowTile][Tile] = {};
    for (int i = lane; i < width; i += 256) {
        float decoded[RowTile];
        #pragma unroll
        for (int r = 0; r < RowTile; r++)
            decoded[r] = row + r < rows
                ? weight(w, Kind, ((u64)expert * rows + row + r) * width + i) : 0.0f;
        #pragma unroll
        for (int t = 0; t < Tile; t++) {
            if (assignments[t] >= 0) {
                int input = Grouped && !batched ? assignments[t] / used : assignments[t];
                float activation = x[(u64)input * width + i];
                #pragma unroll
                for (int r = 0; r < RowTile; r++) sums[r][t] += decoded[r] * activation;
            }
        }
    }
    __shared__ float partial[RowTile][Tile][256];
    #pragma unroll
    for (int r = 0; r < RowTile; r++) {
        #pragma unroll
        for (int t = 0; t < Tile; t++) partial[r][t][lane] = sums[r][t];
    }
    __syncthreads();
    for (int stride = 128; stride >= 64; stride /= 2) {
        if (lane < stride) {
            #pragma unroll
            for (int r = 0; r < RowTile; r++) {
                #pragma unroll
                for (int t = 0; t < Tile; t++)
                    partial[r][t][lane] += partial[r][t][lane + stride];
            }
        }
        __syncthreads();
    }
    if (lane < 32) {
        #pragma unroll
        for (int r = 0; r < RowTile; r++) {
            #pragma unroll
            for (int t = 0; t < Tile; t++) {
                float sum = partial[r][t][lane] + partial[r][t][lane + 32];
                for (int stride = 16; stride > 0; stride /= 2)
                    sum += __shfl_down_sync(0xffffffff, sum, stride);
                if (lane == 0 && assignments[t] >= 0 && row + r < rows)
                    y[(u64)assignments[t] * rows + row + r] = sum;
            }
        }
    }
}

extern "C" __global__ void prefill_matmul_rows2(const u8 *w, int kind, const float *x,
                                              float *y, int width, int rows, int tokens) {
    switch (kind) {
        case 0: prompt_tile_rows2<0>(w, x, y, width, rows, tokens); break;
        case 1: prompt_tile_rows2<1>(w, x, y, width, rows, tokens); break;
        case 6: prompt_tile_rows2<6>(w, x, y, width, rows, tokens); break;
        case 8: prompt_tile_rows2<8>(w, x, y, width, rows, tokens); break;
        case 12: prompt_tile_rows2<12>(w, x, y, width, rows, tokens); break;
        case 14: prompt_tile_rows2<14>(w, x, y, width, rows, tokens); break;
    }
}

extern "C" __global__ void grouped_matmul_rows2(const u8 *w, int kind, const float *x,
                                              float *y, int width, int rows, const int *groups,
                                              int used, int batched) {
    switch (kind) {
        case 0: prompt_tile_rows2<0, true>(w, x, y, width, rows, 0, groups, used, batched); break;
        case 1: prompt_tile_rows2<1, true>(w, x, y, width, rows, 0, groups, used, batched); break;
        case 6: prompt_tile_rows2<6, true>(w, x, y, width, rows, 0, groups, used, batched); break;
        case 8: prompt_tile_rows2<8, true>(w, x, y, width, rows, 0, groups, used, batched); break;
        case 12: prompt_tile_rows2<12, true>(w, x, y, width, rows, 0, groups, used, batched); break;
        case 14: prompt_tile_rows2<14, true>(w, x, y, width, rows, 0, groups, used, batched); break;
    }
}
