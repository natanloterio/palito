// Test-only arithmetic: three activation scales per GGUF block and exact
// signed-byte weight planes. Integer accumulation never crosses a block.
extern "C" __global__ void residual_quantize256_precise3(const float *x,
    signed char *q, float *scales, int *sums, int count, int *status) {
    int i = blockIdx.x * 256 + threadIdx.x, lane = threadIdx.x % 32;
    if (i >= count) return;
    __shared__ float maxima[8];
    float remaining = x[i], reconstructed = 0.0f;
    if (__syncthreads_or(!isfinite(remaining))) {
        if (threadIdx.x == 0) atomicOr(status, 1);
        return;
    }
    #pragma unroll
    for (int component = 0; component < 3; component++) {
        float maximum = fabsf(remaining);
        for (int stride = 16; stride > 0; stride /= 2)
            maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffff, maximum, stride));
        if (lane == 0) maxima[threadIdx.x / 32] = maximum;
        __syncthreads();
        #pragma unroll
        for (int warp = 0; warp < 8; warp++) maximum = fmaxf(maximum, maxima[warp]);
        float d = maximum == 0.0f ? 0.0f :
            fmaxf(__fdiv_rn(maximum, 127.0f), __uint_as_float(1));
        int value = d == 0.0f ? 0 :
            max(-127, min(127, __float2int_rn(__fdiv_rn(remaining, d))));
        reconstructed += d * (float)value;
        remaining = fmaf(-d, (float)value, remaining);
        if (__syncthreads_or(!isfinite(reconstructed) || !isfinite(remaining))) {
            if (threadIdx.x == 0) atomicOr(status, 2);
            return;
        }
        q[component * count + i] = (signed char)value;
        for (int stride = 16; stride > 0; stride /= 2)
            value += __shfl_down_sync(0xffffffff, value, stride);
        if (lane == 0) sums[(i / 32) * 3 + component] = value;
        if (threadIdx.x == 0) scales[blockIdx.x * 3 + component] = d;
    }
}

struct ResidualBlockTile {
    unsigned int weights[2][32][9], values[3][32][9];
    float factors[32][2], scales[32][3];
    int minimum[32], sums[32][3], destination[32];
};

#if __CUDA_ARCH__ >= 800
template<int Kind>
__device__ __forceinline__ void residual_block_planes(const u8 *p, int i,
    unsigned int &low, unsigned int &high) {
    low = high = 0;
    #pragma unroll
    for (int byte = 0; byte < 4; byte++) {
        int k = i + byte, scale;
        if (Kind == 12) {
            int g = k / 32;
            const u8 *m = p + 4;
            scale = g < 4 ? m[g] & 63 : (m[g + 4] & 15) | ((m[g - 4] >> 6) << 4);
        } else {
            scale = ((const signed char *)p)[192 + k / 16];
        }
        int coefficient = scale * residual_quant<Kind>(p, k);
        int lo = coefficient & 127;
        int hi = (coefficient - lo) / 128;
        low |= (unsigned int)lo << (byte * 8);
        high |= (unsigned int)(u8)hi << (byte * 8);
    }
}

template<int Kind>
__device__ void residual_block_tile(const u8 *weights, const signed char *values,
    const float *scales, const int *sums, float *output, int width, int rows,
    int tokens, const int *groups, int used, int grouped, int assignment_major,
    int *status, ResidualBlockTile &s) {
    int expert = grouped ? groups[blockIdx.y * 33] : 0;
    if (expert < 0) return;
    int tid = threadIdx.x, warp = tid / 32, lane = tid % 32;
    int group = lane / 4, thread = lane % 4;
    int first_row = blockIdx.x * 32, warp_row = warp / 2 * 16;
    int warp_token = warp % 2 * 16;
    int input_count = width * tokens * (grouped && assignment_major ? used : 1);
    if (tid < 32) {
        int assignment = grouped ? groups[blockIdx.y * 33 + tid + 1] : blockIdx.y * 32 + tid;
        s.destination[tid] = grouped || assignment < tokens ? assignment : -1;
    }
    __syncthreads();
    float total[3][2][4] = {};
    for (int block = 0; block < width; block += 256) {
        if (tid < 32) {
            float d = 0.0f, minimum = 0.0f;
            if (first_row + tid < rows) {
                const u8 *p = weights + ((u64)(expert * rows + first_row + tid) * width + block)
                    / 256 * (Kind == 12 ? 144 : 210);
                d = half_value(p + (Kind == 12 ? 0 : 208));
                if (Kind == 12) minimum = half_value(p + 2);
            }
            s.factors[tid][0] = d;
            s.factors[tid][1] = minimum;
            int assignment = s.destination[tid];
            int source = grouped && !assignment_major && assignment >= 0 ? assignment / used : assignment;
            u64 offset = ((u64)(source < 0 ? 0 : source) * width + block) / 256 * 3;
            #pragma unroll
            for (int c = 0; c < 3; c++)
                s.scales[tid][c] = source >= 0 ? scales[offset + c] : 0.0f;
        }
        int integer[3][2][4] = {}, correction[3][2][4] = {};
        for (int k = 0; k < 256; k += 32) {
            for (int index = tid; index < 256; index += 128) {
                int row = index / 8, pack = index % 8;
                unsigned int low = 0, high = 0;
                if (first_row + row < rows) {
                    const u8 *p = weights + ((u64)(expert * rows + first_row + row) * width + block)
                        / 256 * (Kind == 12 ? 144 : 210);
                    residual_block_planes<Kind>(p, k + pack * 4, low, high);
                }
                s.weights[0][row][pack] = low;
                s.weights[1][row][pack] = high;
                int assignment = s.destination[row];
                int source = grouped && !assignment_major && assignment >= 0 ? assignment / used : assignment;
                u64 offset = (u64)(source < 0 ? 0 : source) * width + block + k + pack * 4;
                #pragma unroll
                for (int c = 0; c < 3; c++)
                    s.values[c][row][pack] = source >= 0 ?
                        *(const unsigned int *)(values + (u64)c * input_count + offset) : 0;
            }
            if (Kind == 12 && tid < 32) {
                int mn = 0;
                if (first_row + tid < rows) {
                    const u8 *p = weights + ((u64)(expert * rows + first_row + tid) * width + block)
                        / 256 * 144;
                    const u8 *m = p + 4;
                    int g = k / 32;
                    mn = g < 4 ? m[g + 4] & 63 : (m[g + 4] >> 4) | ((m[g] >> 6) << 4);
                }
                s.minimum[tid] = mn;
                int assignment = s.destination[tid];
                int source = grouped && !assignment_major && assignment >= 0 ? assignment / used : assignment;
                u64 offset = ((u64)(source < 0 ? 0 : source) * width + block + k) / 32 * 3;
                #pragma unroll
                for (int c = 0; c < 3; c++)
                    s.sums[tid][c] = source >= 0 ? sums[offset + c] : 0;
            }
            __syncthreads();
            int r = warp_row + group;
            #pragma unroll
            for (int c = 0; c < 3; c++) {
                #pragma unroll
                for (int tile = 0; tile < 2; tile++) {
                    int token = warp_token + tile * 8 + group;
                    unsigned int b0 = s.values[c][token][thread];
                    unsigned int b1 = s.values[c][token][thread + 4];
                    int lo[4] = {}, hi[4] = {};
                    residual_mma_group<12>(s.weights[0][r][thread], s.weights[0][r + 8][thread],
                        s.weights[0][r][thread + 4], s.weights[0][r + 8][thread + 4], b0, b1, lo);
                    residual_mma_group<12>(s.weights[1][r][thread], s.weights[1][r + 8][thread],
                        s.weights[1][r][thread + 4], s.weights[1][r + 8][thread + 4], b0, b1, hi);
                    #pragma unroll
                    for (int item = 0; item < 4; item++) {
                        integer[c][tile][item] += lo[item] + 128 * hi[item];
                        if (Kind == 12) {
                            int row = r + item / 2 * 8;
                            int t = warp_token + tile * 8 + thread * 2 + item % 2;
                            correction[c][tile][item] += s.minimum[row] * s.sums[t][c];
                        }
                    }
                }
            }
            __syncthreads();
        }
        #pragma unroll
        for (int c = 0; c < 3; c++) {
            #pragma unroll
            for (int tile = 0; tile < 2; tile++) {
                #pragma unroll
                for (int item = 0; item < 4; item++) {
                    int row = warp_row + group + item / 2 * 8;
                    int token = warp_token + tile * 8 + thread * 2 + item % 2;
                    float value = s.factors[row][0] * (float)integer[c][tile][item];
                    if (Kind == 12) value -= s.factors[row][1] * (float)correction[c][tile][item];
                    total[c][tile][item] += value * s.scales[token][c];
                }
            }
        }
        __syncthreads();
    }
    #pragma unroll
    for (int tile = 0; tile < 2; tile++) {
        #pragma unroll
        for (int item = 0; item < 4; item++) {
            int row = first_row + warp_row + group + item / 2 * 8;
            int assignment = s.destination[warp_token + tile * 8 + thread * 2 + item % 2];
            if (row < rows && assignment >= 0) {
                float result = (total[0][tile][item] + total[1][tile][item]) + total[2][tile][item];
                if (!isfinite(result)) atomicOr(status, 4);
                output[(u64)assignment * rows + row] = result;
            }
        }
    }
}
#endif

extern "C" __global__ void residual_matmul_mma3_block(const u8 *weights, int kind,
    const signed char *values, const float *scales, const int *sums, float *output,
    int width, int rows, int tokens, const int *groups, int used, int grouped,
    int assignment_major, int *status) {
#if __CUDA_ARCH__ >= 800
    __shared__ ResidualBlockTile tile;
    if (kind == 12) residual_block_tile<12>(weights, values, scales, sums, output, width,
        rows, tokens, groups, used, grouped, assignment_major, status, tile);
    if (kind == 14) residual_block_tile<14>(weights, values, scales, sums, output, width,
        rows, tokens, groups, used, grouped, assignment_major, status, tile);
    if (kind != 12 && kind != 14 && threadIdx.x == 0) atomicOr(status, 8);
#else
    if (threadIdx.x == 0) atomicOr(status, 16);
#endif
}
