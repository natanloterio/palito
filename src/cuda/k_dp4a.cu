template<int Kind>
__device__ float k_dp4a_row(const u8 *weights, const signed char *input,
    const float *scales, int width, int row, int lane) {
    constexpr int stride = Kind == 12 ? 144 : Kind == 13 ? 176 : 210;
    float sum = 0.0f;
    for (int k = lane / 8; k < width / 32; k += 4) {
        const u8 *block = weights + ((u64)row * (width / 256) + k / 8) * stride;
        const int col = (lane & 7) * 4;
        const int i = (k & 7) * 32 + col;
        const unsigned int w = residual_packed_quant<Kind>(block, i);
        float weight_scale = 0.0f, unused = 0.0f, minimum = 0.0f;
        residual_parameters<Kind>(block, i, weight_scale, unused, minimum);
        float partial = 0.0f;
        #pragma unroll
        for (int plane = 0; plane < 3; ++plane) {
            const unsigned int x = *(const unsigned int *)(input + plane * width + k * 32 + col);
            float value = weight_scale * (float)residual_dp4a(w, x, 0);
            if constexpr (Kind != 14)
                value -= minimum * (float)residual_dp4a(0x01010101u, x, 0);
            partial += value * scales[3 * k + plane];
        }
        sum += partial;
    }
    for (int step = 16; step != 0; step /= 2)
        sum += __shfl_down_sync(0xffffffff, sum, step);
    return sum;
}

extern "C" __global__ void k_dp4a_head(const u8 *weights, int kind,
    const signed char *input, const float *scales, float *output, int width, int rows) {
    const int row = blockIdx.x * 4 + threadIdx.x / 32;
    if (row >= rows) return;
    const int lane = threadIdx.x & 31;
    float sum;
    switch (kind) {
        case 12: sum = k_dp4a_row<12>(weights, input, scales, width, row, lane); break;
        case 13: sum = k_dp4a_row<13>(weights, input, scales, width, row, lane); break;
        case 14: sum = k_dp4a_row<14>(weights, input, scales, width, row, lane); break;
        default: sum = __int_as_float(0x7fffffff); break;
    }
    if (lane == 0) output[row] = sum;
}
