// Test-only higher-accuracy decomposition. FMA accounts for product rounding
// when forming each residual; it does not change ordinary inference profiles.
template<int Components>
__device__ __forceinline__ void residual_quantize32_precise(const float *x,
    signed char *q, float *scales, int *sums, int count, int *status) {
    int i = blockIdx.x * blockDim.x + threadIdx.x, lane = threadIdx.x % 32;
    if (i >= count) return;
    float remaining = x[i], reconstructed = 0.0f;
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

extern "C" __global__ void residual_quantize32_precise3(const float *x,
    signed char *q, float *scales, int *sums, int count, int *status) {
    residual_quantize32_precise<3>(x, q, scales, sums, count, status);
}

#ifdef CUDA_M8_PROTOTYPES
extern "C" __global__ void residual_quantize32_precise1(const float *x,
    signed char *q, float *scales, int *sums, int count, int *status) {
    residual_quantize32_precise<1>(x, q, scales, sums, count, status);
}
#endif
