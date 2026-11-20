template<bool aligned = false>
__device__ __forceinline__ void llama_q8_dp4a_impl(
    const u8 *weights, const signed char *input, const float *scales,
    float *output, int width, int rows, const u8 *weight_scales = nullptr) {
    const int row = blockIdx.x * 4 + threadIdx.x / 32;
    if (row >= rows) return;
    const int lane = threadIdx.x & 31;
    const int blocks = width / 32;
    float sum = 0.0f;
    for (int k = lane / 8; k < blocks; k += 4) {
        const u64 index = (u64)row * blocks + k;
        const u8 *block = weights + index * (aligned ? 32 : 34);
        const int col = (lane & 7) * 4;
        int w;
        if constexpr (aligned) {
            w = *(const int *)(block + col);
        } else {
            const unsigned short *q = (const unsigned short *)(block + 2 + col);
            w = (int)((unsigned int)q[0] | ((unsigned int)q[1] << 16));
        }
        float partial = 0.0f;
        #pragma unroll
        for (int plane = 0; plane < 3; ++plane) {
            const int a = *(const int *)(input + plane * width + k * 32 + col);
            int dot;
            asm("dp4a.s32.s32 %0, %1, %2, 0;" : "=r"(dot) : "r"(w), "r"(a));
            partial += (float)dot * scales[3 * k + plane];
        }
        sum += half_value(aligned ? weight_scales + index * 2 : block) * partial;
    }
    for (int step = 16; step != 0; step /= 2)
        sum += __shfl_down_sync(0xffffffff, sum, step);
    if (lane == 0) output[row] = sum;
}

extern "C" __global__ void llama_q8_dp4a(
    const u8 *weights, const signed char *input, const float *scales,
    float *output, int width, int rows) {
    llama_q8_dp4a_impl(weights, input, scales, output, width, rows);
}

#ifdef CUDA_LLAMA_Q8_ALIGNED_REPLAY
extern "C" __global__ void llama_q8_repack(const u8 *packed, u8 *values,
                                           u8 *weight_scales, int bytes) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= bytes) return;
    int block = i / 32, lane = i % 32;
    values[i] = packed[(u64)block * 34 + 2 + lane];
    if (lane < 2) weight_scales[block * 2 + lane] = packed[(u64)block * 34 + lane];
}

extern "C" __global__ void llama_q8_dp4a_aligned(
    const u8 *values, const u8 *weight_scales, const signed char *input, const float *scales,
    float *output, int width, int rows) {
    llama_q8_dp4a_impl<true>(values, input, scales, output, width, rows, weight_scales);
}
#endif

#ifdef CUDA_LLAMA_Q8_FUSED_QKV
extern "C" __global__ void llama_q8_dp4a_qkv(
    const u8 *q, const u8 *k, const u8 *v, const signed char *input, const float *scales,
    float *oq, float *ok, float *ov, int width, int q_rows, int kv_rows) {
    if (blockIdx.y == 0) llama_q8_dp4a_impl(q, input, scales, oq, width, q_rows);
    if (blockIdx.y == 1) llama_q8_dp4a_impl(k, input, scales, ok, width, kv_rows);
    if (blockIdx.y == 2) llama_q8_dp4a_impl(v, input, scales, ov, width, kv_rows);
}
#endif

#ifdef CUDA_LLAMA_Q8_FUSED_SWIGLU
template<int Rows>
__device__ __forceinline__ void llama_q8_dp4a_swiglu_rows(
    const u8 *gate, const u8 *up, const signed char *input, const float *scales,
    float *output, int width, int rows) {
    const int row = (blockIdx.x * 4 + threadIdx.x / 32) * Rows;
    if (row >= rows) return;
    const int lane = threadIdx.x & 31;
    const int blocks = width / 32;
    float sum_g[Rows] = {}, sum_u[Rows] = {};
    for (int k = lane / 8; k < blocks; k += 4) {
        const int col = (lane & 7) * 4;
        int a[3];
        float scale[3];
        #pragma unroll
        for (int plane = 0; plane < 3; ++plane) {
            a[plane] = *(const int *)(input + plane * width + k * 32 + col);
            scale[plane] = scales[3 * k + plane];
        }
        #pragma unroll
        for (int j = 0; j < Rows; ++j) {
            const u64 offset = ((u64)(row + j) * blocks + k) * 34;
            const unsigned short *g = (const unsigned short *)(gate + offset + 2 + col);
            const unsigned short *u = (const unsigned short *)(up + offset + 2 + col);
            const int wg = (int)((unsigned int)g[0] | ((unsigned int)g[1] << 16));
            const int wu = (int)((unsigned int)u[0] | ((unsigned int)u[1] << 16));
            float partial_g = 0.0f, partial_u = 0.0f;
            #pragma unroll
            for (int plane = 0; plane < 3; ++plane) {
                int dg, du;
                asm("dp4a.s32.s32 %0, %1, %2, 0;" : "=r"(dg) : "r"(wg), "r"(a[plane]));
                asm("dp4a.s32.s32 %0, %1, %2, 0;" : "=r"(du) : "r"(wu), "r"(a[plane]));
                partial_g += (float)dg * scale[plane];
                partial_u += (float)du * scale[plane];
            }
            sum_g[j] += half_value(gate + offset) * partial_g;
            sum_u[j] += half_value(up + offset) * partial_u;
        }
    }
    #pragma unroll
    for (int j = 0; j < Rows; ++j) {
        float g = sum_g[j], u = sum_u[j];
        for (int step = 16; step != 0; step /= 2) {
            g += __shfl_down_sync(0xffffffff, g, step);
            u += __shfl_down_sync(0xffffffff, u, step);
        }
        if (lane == 0) output[row + j] = g / (1.0f + expf(-g)) * u;
    }
}

extern "C" __global__ void llama_q8_dp4a_swiglu(
    const u8 *gate, const u8 *up, const signed char *input, const float *scales,
    float *output, int width, int rows) {
    llama_q8_dp4a_swiglu_rows<1>(gate, up, input, scales, output, width, rows);
}

#ifdef CUDA_LLAMA_Q8_SWIGLU_PAIRS
extern "C" __global__ void llama_q8_dp4a_swiglu_pairs(
    const u8 *gate, const u8 *up, const signed char *input, const float *scales,
    float *output, int width, int rows) {
    llama_q8_dp4a_swiglu_rows<2>(gate, up, input, scales, output, width, rows);
}
#endif
#endif
