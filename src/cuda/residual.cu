// Research only: two symmetric INT8 components per 32 activation values.
extern "C" __global__ void residual_quantize32(const float *x, signed char *q0,
    signed char *q1, float *scales, int *sums, int count, int *status) {
    int i = blockIdx.x * blockDim.x + threadIdx.x, lane = threadIdx.x % 32;
    if (i >= count) return;
    float value = x[i], maximum = fabsf(value);
    if (__any_sync(0xffffffff, !isfinite(value))) {
        atomicOr(status, 1);
        return;
    }
    for (int stride = 16; stride > 0; stride /= 2)
        maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffff, maximum, stride));
    float d0 = maximum == 0.0f ? 0.0f : fmaxf(__fdiv_rn(maximum, 127.0f), __uint_as_float(1));
    int a = d0 == 0.0f ? 0 : max(-127, min(127, __float2int_rn(__fdiv_rn(value, d0))));
    float residual = value - d0 * (float)a, remaining = fabsf(residual);
    if (__any_sync(0xffffffff, !isfinite(residual))) {
        atomicOr(status, 2);
        return;
    }
    for (int stride = 16; stride > 0; stride /= 2)
        remaining = fmaxf(remaining, __shfl_xor_sync(0xffffffff, remaining, stride));
    float d1 = remaining == 0.0f ? 0.0f : fmaxf(__fdiv_rn(remaining, 127.0f), __uint_as_float(1));
    int b = d1 == 0.0f ? 0 : max(-127, min(127, __float2int_rn(__fdiv_rn(residual, d1))));
    if (__any_sync(0xffffffff, !isfinite(d0 * (float)a + d1 * (float)b))) {
        atomicOr(status, 2);
        return;
    }
    q0[i] = (signed char)a;
    q1[i] = (signed char)b;
    for (int stride = 16; stride > 0; stride /= 2) {
        a += __shfl_down_sync(0xffffffff, a, stride);
        b += __shfl_down_sync(0xffffffff, b, stride);
    }
    if (lane == 0) {
        scales[(i / 32) * 2] = d0;
        scales[(i / 32) * 2 + 1] = d1;
        sums[(i / 32) * 2] = a;
        sums[(i / 32) * 2 + 1] = b;
    }
}

__device__ __forceinline__ int residual_dp4a(unsigned int a, unsigned int b, int c) {
    int result;
    asm("dp4a.s32.s32 %0, %1, %2, %3;" : "=r"(result) : "r"(a), "r"(b), "r"(c));
    return result;
}

template<int Kind>
__device__ __forceinline__ int residual_quant(const u8 *p, int i) {
    if (Kind == 12) return (p[16 + (i / 64) * 32 + i % 32] >> (4 * ((i / 32) % 2))) & 15;
    int h = i / 128, lane = i % 32, quarter = (i % 128) / 32;
    int low = (p[h * 64 + lane + (quarter % 2) * 32] >> (4 * (quarter / 2))) & 15;
    int high = (p[128 + h * 32 + lane] >> (2 * quarter)) & 3;
    return (low | (high << 4)) - 32;
}

template<int Kind>
__device__ __forceinline__ unsigned int residual_packed_quant(const u8 *p, int i) {
    if (Kind == 12) {
        unsigned int bits = *(const unsigned int *)(p + 16 + (i / 64) * 32 + i % 32);
        return (bits >> (4 * ((i / 32) % 2))) & 0x0f0f0f0fu;
    }
    int h = i / 128, col = i % 32, quarter = (i % 128) / 32;
    const unsigned short *lo = (const unsigned short *)(p + h * 64 + col + (quarter % 2) * 32);
    const unsigned short *hi = (const unsigned short *)(p + 128 + h * 32 + col);
    unsigned int low = (unsigned int)lo[0] | ((unsigned int)lo[1] << 16);
    unsigned int high = (unsigned int)hi[0] | ((unsigned int)hi[1] << 16);
    unsigned int bits = ((low >> (4 * (quarter / 2))) & 0x0f0f0f0fu)
        | (((high >> (2 * quarter)) & 0x03030303u) << 4);
    // Each byte is <=63: this bias cannot carry into its neighbor.
    return (bits + 0x60606060u) ^ 0x80808080u;
}

template<int Kind>
__device__ __forceinline__ void residual_parameters_scaled(const u8 *p, int i, float d, float dm,
    float &a0, float &a1, float &b) {
    if (Kind == 12) {
        int g = i / 32;
        const u8 *m = p + 4;
        int sc = g < 4 ? m[g] & 63 : (m[g + 4] & 15) | ((m[g - 4] >> 6) << 4);
        int mn = g < 4 ? m[g + 4] & 63 : (m[g + 4] >> 4) | ((m[g] >> 6) << 4);
        a0 = d * (float)sc;
        b = dm * (float)mn;
    } else {
        a0 = d * (float)((const signed char *)p)[192 + i / 16];
        a1 = d * (float)((const signed char *)p)[193 + i / 16];
    }
}

template<int Kind>
__device__ __forceinline__ void residual_parameters(const u8 *p, int i,
    float &a0, float &a1, float &b) {
    float d = half_value(p + (Kind == 12 ? 0 : 208));
    float dm = Kind == 12 ? half_value(p + 2) : 0.0f;
    residual_parameters_scaled<Kind>(p, i, d, dm, a0, a1, b);
}

#ifdef CUDA_M8_PROTOTYPES
extern "C" __global__ void residual_packed_quant_check(const u8 *weights, int kind,
    int packs, int *status) {
    int pack = blockIdx.x * blockDim.x + threadIdx.x;
    if (pack >= packs) return;
    if (kind != 12 && kind != 14) {
        atomicOr(status, 8);
        return;
    }
    const u8 *p = weights + (u64)(pack / 64) * (kind == 12 ? 144 : 210);
    int first = pack % 64 * 4;
    unsigned int expected = 0;
    #pragma unroll
    for (int byte = 0; byte < 4; byte++) {
        int value = kind == 12 ? residual_quant<12>(p, first + byte) :
            residual_quant<14>(p, first + byte);
        expected |= (unsigned int)(u8)value << (byte * 8);
    }
    unsigned int actual = kind == 12 ? residual_packed_quant<12>(p, first) :
        residual_packed_quant<14>(p, first);
    if (actual != expected) atomicOr(status, 1);
}

template<int Kind>
__device__ void residual_integer_dot(const u8 *w, const signed char *q0,
    const signed char *q1, const float *scales, const int *sums,
    const int *pairs, float *y, int width, int count, int *status) {
    int dot = (blockIdx.x * blockDim.x + threadIdx.x) / 32, lane = threadIdx.x % 32;
    if (dot >= count) return;
    int row = pairs[dot * 2], input = pairs[dot * 2 + 1];
    float total0 = 0.0f, total1 = 0.0f;
    for (int group = lane; group < width / 32; group += 32) {
        int k = group * 32, block_i = k % 256, activation = input * width + k;
        const u8 *p = w + (((u64)row * width + k) / 256) * (Kind == 12 ? 144 : 210);
        float d0 = scales[(activation / 32) * 2], d1 = scales[(activation / 32) * 2 + 1];
        #pragma unroll
        for (int half = 0; half < (Kind == 12 ? 1 : 2); half++) {
            int dot0 = 0, dot1 = 0;
            #pragma unroll
            for (int pack = 0; pack < (Kind == 12 ? 8 : 4); pack++) {
                unsigned int a = 0, b0 = 0, b1 = 0;
                #pragma unroll
                for (int byte = 0; byte < 4; byte++) {
                    int offset = half * 16 + pack * 4 + byte;
                    a |= (unsigned int)(u8)residual_quant<Kind>(p, block_i + offset) << (byte * 8);
                    b0 |= (unsigned int)(u8)q0[activation + offset] << (byte * 8);
                    b1 |= (unsigned int)(u8)q1[activation + offset] << (byte * 8);
                }
                dot0 = residual_dp4a(a, b0, dot0);
                dot1 = residual_dp4a(a, b1, dot1);
            }
            if (Kind == 12) {
                int g = block_i / 32;
                const u8 *s = p + 4;
                int scale = g < 4 ? s[g] & 63 : (s[g + 4] & 15) | ((s[g - 4] >> 6) << 4);
                int minimum = g < 4 ? s[g + 4] & 63 : (s[g + 4] >> 4) | ((s[g] >> 6) << 4);
                float a = half_value(p) * (float)scale;
                float b = half_value(p + 2) * (float)minimum;
                total0 += (a * (float)dot0 - b * (float)sums[(activation / 32) * 2]) * d0;
                total1 += (a * (float)dot1 - b * (float)sums[(activation / 32) * 2 + 1]) * d1;
            } else {
                float a = half_value(p + 208) * (float)((const signed char *)p)[192 + block_i / 16 + half];
                total0 += (a * (float)dot0) * d0;
                total1 += (a * (float)dot1) * d1;
            }
        }
    }
    for (int stride = 16; stride > 0; stride /= 2) {
        total0 += __shfl_down_sync(0xffffffff, total0, stride);
        total1 += __shfl_down_sync(0xffffffff, total1, stride);
    }
    if (lane == 0) {
        float result = total0 + total1;
        if (!isfinite(result)) atomicOr(status, 4);
        y[dot] = result;
    }
}

extern "C" __global__ void residual_int8_dot(const u8 *w, int kind,
    const signed char *q0, const signed char *q1, const float *scales, const int *sums,
    const int *pairs, float *y, int width, int count, int *status) {
    if (kind == 12) residual_integer_dot<12>(w, q0, q1, scales, sums, pairs, y, width, count, status);
    if (kind == 14) residual_integer_dot<14>(w, q0, q1, scales, sums, pairs, y, width, count, status);
    if (kind != 12 && kind != 14) atomicOr(status, 8);
}

extern "C" __global__ void residual_f32_dot(const u8 *w, int kind,
    const float *x, const int *pairs, float *y, int width, int count, int *status) {
    int dot = blockIdx.x, lane = threadIdx.x;
    if (dot >= count) return;
    float sum = 0.0f;
    for (int k = lane; k < width; k += 256)
        sum += weight(w, kind, (u64)pairs[dot * 2] * width + k) * x[(u64)pairs[dot * 2 + 1] * width + k];
    __shared__ float partial[256];
    partial[lane] = sum;
    __syncthreads();
    for (int stride = 128; stride >= 64; stride /= 2) {
        if (lane < stride) partial[lane] += partial[lane + stride];
        __syncthreads();
    }
    if (lane < 32) {
        sum = partial[lane] + partial[lane + 32];
        for (int stride = 16; stride > 0; stride /= 2) sum += __shfl_down_sync(0xffffffff, sum, stride);
        if (lane == 0) {
            if (!isfinite(sum)) atomicOr(status, 4);
            y[dot] = sum;
        }
    }
}

// Scalar four-product control isolates decomposition error, not MMA throughput.
extern "C" __global__ void residual_tf32x4_dot(const u8 *w, int kind,
    const float *x, const int *pairs, float *y, int width, int count, int *status) {
#if __CUDA_ARCH__ >= 800
    int dot = (blockIdx.x * blockDim.x + threadIdx.x) / 32, lane = threadIdx.x % 32;
    if (dot >= count) return;
    float c0 = 0.0f, c1 = 0.0f, c2 = 0.0f, c3 = 0.0f;
    for (int k = lane; k < width; k += 32) {
        float a = weight(w, kind, (u64)pairs[dot * 2] * width + k);
        float b = x[(u64)pairs[dot * 2 + 1] * width + k];
        float ah = __uint_as_float(as_tf32(a)), bh = __uint_as_float(as_tf32(b));
        float al = __uint_as_float(as_tf32(a - ah)), bl = __uint_as_float(as_tf32(b - bh));
        c0 += ah * bh;
        c1 += ah * bl;
        c2 += al * bh;
        c3 += al * bl;
    }
    for (int stride = 16; stride > 0; stride /= 2) {
        c0 += __shfl_down_sync(0xffffffff, c0, stride);
        c1 += __shfl_down_sync(0xffffffff, c1, stride);
        c2 += __shfl_down_sync(0xffffffff, c2, stride);
        c3 += __shfl_down_sync(0xffffffff, c3, stride);
    }
    if (lane == 0) {
        float result = ((c0 + c1) + c2) + c3;
        if (!isfinite(result)) atomicOr(status, 4);
        y[dot] = result;
    }
#else
    asm("trap;");
#endif
}
#endif
