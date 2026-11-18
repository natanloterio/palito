#ifdef CUDA_M8_PROTOTYPES
#if __CUDA_ARCH__ >= 800
__device__ __forceinline__ void residual_copy_async(void *destination, const void *source, bool valid) {
    unsigned int shared = (unsigned int)__cvta_generic_to_shared(destination);
    int bytes = valid ? 4 : 0;
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;"
        :: "r"(shared), "l"(source), "r"(bytes) : "memory");
}

struct ResidualPipelineIndex {
    u64 input[2], metadata;
    const u8 *weights[2], *parameters;
    bool valid[2], metadata_valid, row_valid[2], parameter_valid;
};

template<int Kind>
__device__ __forceinline__ void residual_pipeline_load(
    const signed char *values, const float *scales, const int *sums,
    int input_count, int k, const ResidualPipelineIndex &index, ResidualMmaTile &s) {
    int tid = threadIdx.x;
    u64 block_offset = (u64)(k / 256) * (Kind == 12 ? 144 : 210);
    #pragma unroll
    for (int slot = 0; slot < 2; slot++) {
        int row = tid / 8 + slot * 16, pack = tid % 8;
        unsigned int packed = 0;
        if (index.row_valid[slot]) {
            const u8 *p = index.weights[slot] + block_offset;
            packed = residual_packed_quant<Kind>(p, k % 256 + pack * 4);
        }
        s.weights[0][row][pack] = packed;
        u64 offset = index.input[slot] + k + pack * 4;
        #pragma unroll
        for (int component = 0; component < 3; component++)
            residual_copy_async(&s.values[component][row][pack],
                values + (u64)component * input_count + offset, index.valid[slot]);
    }
    if (tid < 32) {
        float a0 = 0.0f, a1 = 0.0f, b = 0.0f;
        if (index.parameter_valid) {
            const u8 *p = index.parameters + block_offset;
            residual_parameters<Kind>(p, k % 256, a0, a1, b);
        }
        s.coefficient[0][tid][0] = a0;
        s.coefficient[0][tid][1] = a1;
        s.minimum[0][tid] = b;
        u64 offset = index.metadata + k / 32 * 3;
        #pragma unroll
        for (int component = 0; component < 3; component++) {
            residual_copy_async(&s.scales[tid][component], scales + offset + component, index.metadata_valid);
            residual_copy_async(&s.sums[tid][component], sums + offset + component, index.metadata_valid);
        }
    }
    asm volatile("cp.async.commit_group;" ::: "memory");
}

template<int Kind>
__device__ void residual_pipeline_tile(const u8 *weights, const signed char *values,
    const float *scales, const int *sums, float *output, int width, int rows,
    int tokens, const int *groups, int used, int grouped, int assignment_major,
    int *status, ResidualMmaTile (&stages)[2]) {
    int expert = grouped ? groups[blockIdx.y * 33] : 0;
    if (expert < 0) return;
    int tid = threadIdx.x, warp = tid / 32, lane = tid % 32;
    int group = lane / 4, thread = lane % 4;
    int first_row = blockIdx.x * 32, warp_row = warp / 2 * 16;
    int warp_token = warp % 2 * 16;
    int input_count = width * tokens * (grouped && assignment_major ? used : 1);
    if (tid < 32) {
        int assignment = grouped ? groups[blockIdx.y * 33 + tid + 1] : blockIdx.y * 32 + tid;
        int destination = grouped || assignment < tokens ? assignment : -1;
        stages[0].destination[tid] = destination;
        stages[1].destination[tid] = destination;
    }
    __syncthreads();
    ResidualPipelineIndex index;
    #pragma unroll
    for (int slot = 0; slot < 2; slot++) {
        int row = tid / 8 + slot * 16;
        int assignment = stages[0].destination[row];
        int source = grouped && !assignment_major && assignment >= 0 ? assignment / used : assignment;
        index.valid[slot] = source >= 0;
        index.input[slot] = (u64)(source < 0 ? 0 : source) * width;
        index.row_valid[slot] = first_row + row < rows;
        index.weights[slot] = weights + (index.row_valid[slot] ?
            (u64)(expert * rows + first_row + row) * (width / 256) * (Kind == 12 ? 144 : 210) : 0);
    }
    if (tid < 32) {
        int assignment = stages[0].destination[tid];
        int source = grouped && !assignment_major && assignment >= 0 ? assignment / used : assignment;
        index.metadata_valid = source >= 0;
        index.metadata = (u64)(source < 0 ? 0 : source) * (width / 32) * 3;
        index.parameter_valid = first_row + tid < rows;
        index.parameters = weights + (index.parameter_valid ?
            (u64)(expert * rows + first_row + tid) * (width / 256) * (Kind == 12 ? 144 : 210) : 0);
    }
    residual_pipeline_load<Kind>(values, scales, sums, input_count, 0, index, stages[0]);
    asm volatile("cp.async.wait_group 0;" ::: "memory");
    __syncthreads();
    float total[3][2][4] = {};
    for (int k = 0; k < width; k += 32) {
        int current = (k / 32) % 2;
        if (k + 32 < width)
            residual_pipeline_load<Kind>(values, scales, sums, input_count,
                k + 32, index, stages[1 - current]);
        ResidualMmaTile &s = stages[current];
        // Keep the admitted kernel's floating-point operations and accumulation order.
        #pragma unroll
        for (int half = 0; half < (Kind == 12 ? 1 : 2); half++) {
            int r = warp_row + group, pack = thread + half * 4;
            unsigned int a0 = s.weights[0][r][pack], a1 = s.weights[0][r + 8][pack];
            unsigned int a2 = Kind == 12 ? s.weights[0][r][thread + 4] : 0;
            unsigned int a3 = Kind == 12 ? s.weights[0][r + 8][thread + 4] : 0;
            #pragma unroll
            for (int component = 0; component < 3; component++) {
                #pragma unroll
                for (int tile = 0; tile < 2; tile++) {
                    int token = warp_token + tile * 8 + group;
                    unsigned int b0 = s.values[component][token][pack];
                    unsigned int b1 = Kind == 12 ? s.values[component][token][thread + 4] : 0;
                    int dot[4] = {};
                    residual_mma_group<Kind>(a0, a1, a2, a3, b0, b1, dot);
                    #pragma unroll
                    for (int item = 0; item < 4; item++) {
                        int output_row = r + item / 2 * 8;
                        int output_token = warp_token + tile * 8 + thread * 2 + item % 2;
                        float value = s.coefficient[0][output_row][half] * (float)dot[item];
                        if (Kind == 12)
                            value -= s.minimum[0][output_row] * (float)s.sums[output_token][component];
                        total[component][tile][item] += value * s.scales[output_token][component];
                    }
                }
            }
        }
        asm volatile("cp.async.wait_group 0;" ::: "memory");
        __syncthreads();
    }
    #pragma unroll
    for (int tile = 0; tile < 2; tile++) {
        #pragma unroll
        for (int item = 0; item < 4; item++) {
            int row = first_row + warp_row + group + item / 2 * 8;
            int assignment = stages[0].destination[warp_token + tile * 8 + thread * 2 + item % 2];
            if (row < rows && assignment >= 0) {
                float result = (total[0][tile][item] + total[1][tile][item]) + total[2][tile][item];
                if (!isfinite(result)) atomicOr(status, 4);
                output[(u64)assignment * rows + row] = result;
            }
        }
    }
}
#endif

extern "C" __global__ __launch_bounds__(128) void residual_matmul_mma3_pipeline(const u8 *weights, int kind,
    const signed char *values, const float *scales, const int *sums, float *output,
    int width, int rows, int tokens, const int *groups, int used, int grouped,
    int assignment_major, int *status) {
#if __CUDA_ARCH__ >= 800
    __shared__ ResidualMmaTile stages[2];
    if (kind == 12) residual_pipeline_tile<12>(weights, values, scales, sums, output, width,
        rows, tokens, groups, used, grouped, assignment_major, status, stages);
    if (kind == 14) residual_pipeline_tile<14>(weights, values, scales, sums, output, width,
        rows, tokens, groups, used, grouped, assignment_major, status, stages);
    if (kind != 12 && kind != 14 && threadIdx.x == 0) atomicOr(status, 8);
#else
    if (threadIdx.x == 0) atomicOr(status, 16);
#endif
}

#if __CUDA_ARCH__ >= 800
template<int Kind>
__device__ __forceinline__ unsigned int residual_packed_quant_volatile(const u8 *p, int i) {
    if (Kind == 12) {
        unsigned int bits;
        asm volatile("ld.global.ca.u32 %0, [%1];"
            : "=r"(bits)
            : "l"(p + 16 + (i / 64) * 32 + i % 32));
        return (bits >> (4 * ((i / 32) % 2))) & 0x0f0f0f0fu;
    }
    int h = i / 128, col = i % 32, quarter = (i % 128) / 32;
    const u8 *lo_ptr = p + h * 64 + col + (quarter % 2) * 32;
    const u8 *hi_ptr = p + 128 + h * 32 + col;
    unsigned short lo0, lo1, hi0, hi1;
    asm volatile("ld.global.ca.u16 %0, [%1];" : "=h"(lo0) : "l"(lo_ptr));
    asm volatile("ld.global.ca.u16 %0, [%1];" : "=h"(lo1) : "l"(lo_ptr + 2));
    asm volatile("ld.global.ca.u16 %0, [%1];" : "=h"(hi0) : "l"(hi_ptr));
    asm volatile("ld.global.ca.u16 %0, [%1];" : "=h"(hi1) : "l"(hi_ptr + 2));
    unsigned int low = (unsigned int)lo0 | ((unsigned int)lo1 << 16);
    unsigned int high = (unsigned int)hi0 | ((unsigned int)hi1 << 16);
    unsigned int bits = ((low >> (4 * (quarter / 2))) & 0x0f0f0f0fu)
        | (((high >> (2 * quarter)) & 0x03030303u) << 4);
    return (bits + 0x60606060u) ^ 0x80808080u;
}

template<int Kind>
__device__ void residual_true_pipeline_tile(const u8 *weights, const signed char *values,
    const float *scales, const int *sums, float *output, int width, int rows,
    int tokens, const int *groups, int used, int grouped, int assignment_major,
    int *status, ResidualMmaTile (&stages)[2]) {
    int expert = grouped ? groups[blockIdx.y * 33] : 0;
    if (expert < 0) return;
    int tid = threadIdx.x, warp = tid / 32, lane = tid % 32;
    int group = lane / 4, thread = lane % 4;
    int first_row = blockIdx.x * 32, warp_row = (warp / 2) * 16;
    int warp_token = (warp % 2) * 16;
    int input_count = width * tokens * (grouped && assignment_major ? used : 1);
    if (tid < 32) {
        int assignment = grouped ? groups[blockIdx.y * 33 + tid + 1] : blockIdx.y * 32 + tid;
        int destination = grouped || assignment < tokens ? assignment : -1;
        stages[0].destination[tid] = destination;
        stages[1].destination[tid] = destination;
    }
    __syncthreads();
    ResidualPipelineIndex index;
    #pragma unroll
    for (int slot = 0; slot < 2; slot++) {
        int row = tid / 8 + slot * 16;
        int assignment = stages[0].destination[row];
        int source = grouped && !assignment_major && assignment >= 0 ? assignment / used : assignment;
        index.valid[slot] = source >= 0;
        index.input[slot] = (u64)(source < 0 ? 0 : source) * width;
        index.row_valid[slot] = first_row + row < rows;
        index.weights[slot] = weights + (index.row_valid[slot] ?
            (u64)(expert * rows + first_row + row) * (width / 256) * (Kind == 12 ? 144 : 210) : 0);
    }
    if (tid < 32) {
        int assignment = stages[0].destination[tid];
        int source = grouped && !assignment_major && assignment >= 0 ? assignment / used : assignment;
        index.metadata_valid = source >= 0;
        index.metadata = (u64)(source < 0 ? 0 : source) * (width / 32) * 3;
        index.parameter_valid = first_row + tid < rows;
        index.parameters = weights + (index.parameter_valid ?
            (u64)(expert * rows + first_row + tid) * (width / 256) * (Kind == 12 ? 144 : 210) : 0);
    }
    residual_pipeline_load<Kind>(values, scales, sums, input_count, 0, index, stages[0]);
    asm volatile("cp.async.wait_group 0;" ::: "memory");
    __syncthreads();
    float total[3][2][4] = {};
    int r0 = tid / 8, pack_col = tid % 8;
    for (int k = 0; k < width; k += 32) {
        int current = (k / 32) % 2;
        int next_k = k + 32;
        bool has_next = next_k < width;
        unsigned int next_w0 = 0, next_w1 = 0;
        float next_a0 = 0.0f, next_a1 = 0.0f, next_b = 0.0f;
        if (has_next) {
            ResidualMmaTile &dst = stages[1 - current];
            u64 block_offset = (u64)(next_k / 256) * (Kind == 12 ? 144 : 210);
            int sub = next_k % 256 + pack_col * 4;
            if (index.row_valid[0])
                next_w0 = residual_packed_quant_volatile<Kind>(index.weights[0] + block_offset, sub);
            if (index.row_valid[1])
                next_w1 = residual_packed_quant_volatile<Kind>(index.weights[1] + block_offset, sub);
            #pragma unroll
            for (int slot = 0; slot < 2; slot++) {
                int row = r0 + slot * 16;
                u64 offset = index.input[slot] + next_k + pack_col * 4;
                #pragma unroll
                for (int component = 0; component < 3; component++)
                    residual_copy_async(&dst.values[component][row][pack_col],
                        values + (u64)component * input_count + offset, index.valid[slot]);
            }
            if (tid < 32) {
                if (index.parameter_valid)
                    residual_parameters<Kind>(index.parameters + block_offset, next_k % 256,
                        next_a0, next_a1, next_b);
                u64 offset = index.metadata + next_k / 32 * 3;
                #pragma unroll
                for (int component = 0; component < 3; component++) {
                    residual_copy_async(&dst.scales[tid][component], scales + offset + component, index.metadata_valid);
                    residual_copy_async(&dst.sums[tid][component], sums + offset + component, index.metadata_valid);
                }
            }
            asm volatile("cp.async.commit_group;" ::: "memory");
        }
        ResidualMmaTile &s = stages[current];
        #pragma unroll
        for (int half = 0; half < (Kind == 12 ? 1 : 2); half++) {
            int r = warp_row + group, pack = thread + half * 4;
            unsigned int a0 = s.weights[0][r][pack], a1 = s.weights[0][r + 8][pack];
            unsigned int a2 = Kind == 12 ? s.weights[0][r][thread + 4] : 0;
            unsigned int a3 = Kind == 12 ? s.weights[0][r + 8][thread + 4] : 0;
            float coeff0 = s.coefficient[0][r][half], coeff1 = s.coefficient[0][r + 8][half];
            float min0 = Kind == 12 ? s.minimum[0][r] : 0.0f;
            float min1 = Kind == 12 ? s.minimum[0][r + 8] : 0.0f;
            #pragma unroll
            for (int component = 0; component < 3; component++) {
                #pragma unroll
                for (int tile = 0; tile < 2; tile++) {
                    int token = warp_token + tile * 8 + group;
                    unsigned int b0 = s.values[component][token][pack];
                    unsigned int b1 = Kind == 12 ? s.values[component][token][thread + 4] : 0;
                    int dot[4] = {};
                    residual_mma_group<Kind>(a0, a1, a2, a3, b0, b1, dot);
                    int t0 = warp_token + tile * 8 + thread * 2;
                    float scale0 = s.scales[t0][component], scale1 = s.scales[t0 + 1][component];
                    float sum0 = Kind == 12 ? (float)s.sums[t0][component] : 0.0f;
                    float sum1 = Kind == 12 ? (float)s.sums[t0 + 1][component] : 0.0f;
                    float v0 = coeff0 * (float)dot[0], v1 = coeff0 * (float)dot[1];
                    float v2 = coeff1 * (float)dot[2], v3 = coeff1 * (float)dot[3];
                    if (Kind == 12) {
                        v0 -= min0 * sum0;
                        v1 -= min0 * sum1;
                        v2 -= min1 * sum0;
                        v3 -= min1 * sum1;
                    }
                    total[component][tile][0] += v0 * scale0;
                    total[component][tile][1] += v1 * scale1;
                    total[component][tile][2] += v2 * scale0;
                    total[component][tile][3] += v3 * scale1;
                }
            }
        }
        if (has_next) {
            ResidualMmaTile &dst = stages[1 - current];
            dst.weights[0][r0][pack_col] = next_w0;
            dst.weights[0][r0 + 16][pack_col] = next_w1;
            if (tid < 32) {
                dst.coefficient[0][tid][0] = next_a0;
                dst.coefficient[0][tid][1] = next_a1;
                dst.minimum[0][tid] = next_b;
            }
        }
        asm volatile("cp.async.wait_group 0;" ::: "memory");
        __syncthreads();
    }
    #pragma unroll
    for (int tile = 0; tile < 2; tile++) {
        #pragma unroll
        for (int item = 0; item < 4; item++) {
            int row = first_row + warp_row + group + item / 2 * 8;
            int assignment = stages[0].destination[warp_token + tile * 8 + thread * 2 + item % 2];
            if (row < rows && assignment >= 0) {
                float result = (total[0][tile][item] + total[1][tile][item]) + total[2][tile][item];
                if (!isfinite(result)) atomicOr(status, 4);
                output[(u64)assignment * rows + row] = result;
            }
        }
    }
}
#endif

extern "C" __global__ __launch_bounds__(128) void residual_matmul_mma3_true_pipeline(
    const u8 *weights, int kind, const signed char *values, const float *scales,
    const int *sums, float *output, int width, int rows, int tokens,
    const int *groups, int used, int grouped, int assignment_major, int *status) {
#if __CUDA_ARCH__ >= 800
    __shared__ ResidualMmaTile stages[2];
    if (kind == 12) residual_true_pipeline_tile<12>(weights, values, scales, sums, output, width,
        rows, tokens, groups, used, grouped, assignment_major, status, stages);
    if (kind == 14) residual_true_pipeline_tile<14>(weights, values, scales, sums, output, width,
        rows, tokens, groups, used, grouped, assignment_major, status, stages);
    if (kind != 12 && kind != 14 && threadIdx.x == 0) atomicOr(status, 8);
#else
    if (threadIdx.x == 0) atomicOr(status, 16);
#endif
}
#endif
