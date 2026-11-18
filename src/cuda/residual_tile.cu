// Research-only 32-row x 32-assignment tile; packed weights and both activation
// components are staged once per metadata group and reused across the tile.
struct ResidualMatrixTile {
    unsigned int w[32][9], x0[32][9], x1[32][9];
    float coefficient[32][2], minimum[32], scales[32][2];
    int sums[32][2], destination[32];
};

template<int Kind>
__device__ void residual_matrix_tile(const u8 *w, const signed char *q0,
    const signed char *q1, const float *scales, const int *sums, float *y,
    int width, int rows, int tokens, const int *groups, int used,
    int grouped, int assignment_major, int *status, ResidualMatrixTile &s) {
    int expert = grouped ? groups[blockIdx.y * 33] : 0;
    if (expert < 0) return;
    int tid = threadIdx.x, row = tid / 4, token = tid % 4;
    int first_row = blockIdx.x * 32;
    if (tid < 32) {
        int assignment = grouped ? groups[blockIdx.y * 33 + tid + 1] : blockIdx.y * 32 + tid;
        s.destination[tid] = grouped || assignment < tokens ? assignment : -1;
    }
    __syncthreads();
    int destinations[8];
    float total0[8] = {}, total1[8] = {};
    #pragma unroll
    for (int j = 0; j < 8; j++) destinations[j] = s.destination[token + j * 4];
    for (int k = 0; k < width; k += 32) {
        for (int index = tid; index < 256; index += 128) {
            int r = index / 8, pack = index % 8;
            unsigned int packed = 0;
            if (first_row + r < rows) {
                const u8 *p = w + (((u64)(expert * rows + first_row + r) * width + k) / 256)
                    * (Kind == 12 ? 144 : 210);
                packed = residual_packed_quant<Kind>(p, k % 256 + pack * 4);
            }
            s.w[r][pack] = packed;
            int assignment = s.destination[r];
            int source = grouped && !assignment_major && assignment >= 0 ? assignment / used : assignment;
            // CUDA allocations and guarded test views are four-byte aligned;
            // widths and K are multiples of 32.
            u64 offset = (u64)(source < 0 ? 0 : source) * width + k + pack * 4;
            s.x0[r][pack] = source >= 0 ? *(const unsigned int *)(q0 + offset) : 0;
            s.x1[r][pack] = source >= 0 ? *(const unsigned int *)(q1 + offset) : 0;
        }
        if (tid < 32) {
            float a0 = 0.0f, a1 = 0.0f, b = 0.0f;
            if (first_row + tid < rows) {
                const u8 *p = w + (((u64)(expert * rows + first_row + tid) * width + k) / 256)
                    * (Kind == 12 ? 144 : 210);
                residual_parameters<Kind>(p, k % 256, a0, a1, b);
            }
            s.coefficient[tid][0] = a0;
            s.coefficient[tid][1] = a1;
            s.minimum[tid] = b;
            int assignment = s.destination[tid];
            int source = grouped && !assignment_major && assignment >= 0 ? assignment / used : assignment;
            u64 offset = ((u64)(source < 0 ? 0 : source) * width + k) / 32 * 2;
            #pragma unroll
            for (int component = 0; component < 2; component++) {
                s.scales[tid][component] = source >= 0 ? scales[offset + component] : 0.0f;
                s.sums[tid][component] = source >= 0 ? sums[offset + component] : 0;
            }
        }
        __syncthreads();
        #pragma unroll
        for (int half = 0; half < (Kind == 12 ? 1 : 2); half++) {
            int dot0[8] = {}, dot1[8] = {};
            #pragma unroll
            for (int pack = 0; pack < (Kind == 12 ? 8 : 4); pack++) {
                int p = half * 4 + pack;
                unsigned int packed = s.w[row][p];
                #pragma unroll
                for (int j = 0; j < 8; j++) {
                    if (destinations[j] >= 0) {
                        dot0[j] = residual_dp4a(packed, s.x0[token + j * 4][p], dot0[j]);
                        dot1[j] = residual_dp4a(packed, s.x1[token + j * 4][p], dot1[j]);
                    }
                }
            }
            float a = s.coefficient[row][half], b = s.minimum[row];
            #pragma unroll
            for (int j = 0; j < 8; j++) {
                if (destinations[j] >= 0) {
                    int t = token + j * 4;
                    float c0 = a * (float)dot0[j], c1 = a * (float)dot1[j];
                    if (Kind == 12) {
                        c0 -= b * (float)s.sums[t][0];
                        c1 -= b * (float)s.sums[t][1];
                    }
                    total0[j] += c0 * s.scales[t][0];
                    total1[j] += c1 * s.scales[t][1];
                }
            }
        }
        __syncthreads();
    }
    #pragma unroll
    for (int j = 0; j < 8; j++) {
        if (destinations[j] >= 0 && first_row + row < rows) {
            float result = total0[j] + total1[j];
            if (!isfinite(result)) atomicOr(status, 4);
            y[(u64)destinations[j] * rows + first_row + row] = result;
        }
    }
}

extern "C" __global__ void residual_matmul32(const u8 *w, int kind,
    const signed char *q0, const signed char *q1, const float *scales, const int *sums,
    float *y, int width, int rows, int tokens, const int *groups, int used,
    int grouped, int assignment_major, int *status) {
    __shared__ ResidualMatrixTile tile;
    if (kind == 12) residual_matrix_tile<12>(w, q0, q1, scales, sums, y, width, rows, tokens,
        groups, used, grouped, assignment_major, status, tile);
    if (kind == 14) residual_matrix_tile<14>(w, q0, q1, scales, sums, y, width, rows, tokens,
        groups, used, grouped, assignment_major, status, tile);
    if (kind != 12 && kind != 14) atomicOr(status, 8);
}
