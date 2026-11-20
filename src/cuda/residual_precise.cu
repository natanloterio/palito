// Test-only higher-accuracy decomposition. FMA accounts for product rounding
// when forming each residual; it does not change ordinary inference profiles.
template<int Components>
__device__ __forceinline__ void residual_quantize32_lane(float remaining, int i,
    signed char *q, float *scales, int *sums, int count, int *status) {
    int lane = threadIdx.x % 32;
    float reconstructed = 0.0f;
    if (__any_sync(0xffffffff, !isfinite(remaining))) {
        atomicOr(status, 1);
        return;
    }
    #pragma unroll
    for (int component = 0; component < Components; component++) {
        float maximum = fabsf(remaining);
        for (int stride = 16; stride > 0; stride /= 2)
            maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffff, maximum, stride));
        float d = maximum == 0.0f ? 0.0f :
            fmaxf(__fdiv_rn(maximum, 127.0f), __uint_as_float(1));
        int value = d == 0.0f ? 0 :
            max(-127, min(127, __float2int_rn(__fdiv_rn(remaining, d))));
        reconstructed += d * (float)value;
        remaining = fmaf(-d, (float)value, remaining);
        if (__any_sync(0xffffffff, !isfinite(reconstructed) || !isfinite(remaining))) {
            atomicOr(status, 2);
            return;
        }
        q[component * count + i] = (signed char)value;
        for (int stride = 16; stride > 0; stride /= 2)
            value += __shfl_down_sync(0xffffffff, value, stride);
        if (lane == 0) {
            scales[(i / 32) * Components + component] = d;
            sums[(i / 32) * Components + component] = value;
        }

    }
}

template<int Components>
__device__ __forceinline__ void residual_quantize32_precise(const float *x,
    signed char *q, float *scales, int *sums, int count, int *status) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count)
        residual_quantize32_lane<Components>(x[i], i, q, scales, sums, count, status);
}

#ifdef CUDA_LLAMA_Q8_NORM_PACK
extern "C" __global__ void llama_q8_norm_pack(const float *x, const u8 *w,
    signed char *q, float *scales, int *sums, int *status, float epsilon) {
    constexpr int width = 2048;
    float sum = 0.0f;
    for (int j = threadIdx.x; j < width; j += blockDim.x) sum += x[j] * x[j];
    float inv = 1.0f / sqrtf(sum_block(sum) / (float)width + epsilon);
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float normalized = x[i] * inv * reinterpret_cast<const float *>(w)[i];
    residual_quantize32_lane<3>(normalized, i, q, scales, sums, width, status);
}
#endif

extern "C" __global__ void residual_quantize32_precise3(const float *x,
    signed char *q, float *scales, int *sums, int count, int *status) {
    residual_quantize32_precise<3>(x, q, scales, sums, count, status);
}

#if defined(CUDA_M8_PROTOTYPES) || defined(CUDA_DENSE_PACKED_REPLAY)
extern "C" __global__ void residual_quantize32_precise1(const float *x,
    signed char *q, float *scales, int *sums, int count, int *status) {
    residual_quantize32_precise<1>(x, q, scales, sums, count, status);
}
#endif
