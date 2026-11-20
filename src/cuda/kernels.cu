typedef unsigned char u8;
typedef unsigned long long u64;

__device__ float half_value_bits(unsigned int h) {
#ifdef CUDA_NATIVE_HALF
    // Preserve legacy reserved-exponent handling; every finite half is exact in F32.
    if ((h & 0x7c00) != 0x7c00) {
        unsigned short packed = (unsigned short)h;
        float value;
        asm("cvt.f32.f16 %0, %1;" : "=f"(value) : "h"(packed));
        return value;
    }
#endif
    unsigned int sign = (h & 0x8000) << 16;
    unsigned int exponent = (h >> 10) & 31;
    unsigned int fraction = h & 1023;
    if (exponent == 0) {
        float value = (float)fraction * (1.0f / 16777216.0f);
        return (sign != 0) ? -value : value;
    }
    return __uint_as_float(sign | ((exponent + 112) << 23) | (fraction << 13));
}

__device__ float half_value(const u8 *p) {
    return half_value_bits((unsigned int)p[0] | ((unsigned int)p[1] << 8));
}

__device__ __forceinline__ float weight(const u8 *data, int kind, u64 index) {
    if (kind == 0) return ((const float *)data)[index];
    if (kind == 1) return half_value(data + index * 2);
    if (kind == 30) {
        const u8 *p = data + index * 2;
        return __uint_as_float(((unsigned int)p[0] << 16) | ((unsigned int)p[1] << 24));
    }
    if (kind == 2) {
        const u8 *p = data + (index / 32) * 18;
        int i = index % 32;
        int q = ((p[2 + i % 16] >> (4 * (i / 16))) & 15) - 8;
        return half_value(p) * (float)q;
    }
    if (kind == 3 || kind == 7) {
        const u8 *p = data + (index / 32) * (kind == 7 ? 24 : 20);
        int i = index % 32;
        int q = (p[(kind == 7 ? 8 : 4) + i % 16] >> (4 * (i / 16))) & 15;
        if (kind == 7) q |= ((p[4 + i / 8] >> (i % 8)) & 1) << 4;
        return half_value(p) * (float)q + half_value(p + 2);
    }
    if (kind == 10) {
        const u8 *p = data + (index / 256) * 84;
        int i = index % 256, group = p[i / 16], shift = 2 * ((i % 128) / 32);
        int q = (p[16 + (i / 128) * 32 + i % 32] >> shift) & 3;
        return half_value(p + 80) * (float)(group & 15) * (float)q
               - half_value(p + 82) * (float)(group >> 4);
    }
    if (kind == 11) {
        const u8 *p = data + (index / 256) * 110;
        int i = index % 256, group = i / 16, shift = i / 32;
        int lo = (p[96 + group % 8] >> (4 * (group / 8))) & 15;
        int hi = (p[104 + group % 4] >> (2 * (group / 4))) & 3;
        int multiplier = (lo | (hi << 4)) - 32;
        int q = (p[32 + (i / 128) * 32 + i % 32] >> (2 * (shift % 4))) & 3;
        if (!(p[i % 32] & (1 << shift))) q -= 4;
        return half_value(p + 108) * (float)multiplier * (float)q;
    }
    if (kind == 20) {
        constexpr signed char codebook[] = {-127, -104, -83, -65, -49, -35, -22, -10,
                                            1, 13, 25, 38, 53, 69, 89, 113};
        const u8 *p = data + (index / 32) * 18;
        int i = index % 32, code = (p[2 + i % 16] >> (4 * (i / 16))) & 15;
        return half_value(p) * (float)codebook[code];
    }
    int block = (kind == 6 || kind == 8) ? 32 : 256;
    int size = kind == 6 ? 22 : kind == 8 ? 34 : kind == 12 ? 144 : kind == 13 ? 176 : 210;
    const u8 *p = data + (index / block) * size;
    int i = index % block;
    if (kind == 6) {
        unsigned int high = (unsigned int)p[2] | ((unsigned int)p[3] << 8)
            | ((unsigned int)p[4] << 16) | ((unsigned int)p[5] << 24);
        int q = ((p[6 + i % 16] >> (4 * (i / 16))) & 15) + ((high >> i) & 1) * 16 - 16;
        return half_value(p) * (float)q;
    }
    if (kind == 8) return half_value(p) * (float)((const signed char *)p)[2 + i];
    if (kind == 12 || kind == 13) {
        const u8 *s = p + 4;
        int g = i / 32;
        int scale = g < 4 ? s[g] & 63 : (s[g + 4] & 15) | ((s[g - 4] >> 6) << 4);
        int minimum = g < 4 ? s[g + 4] & 63 : (s[g + 4] >> 4) | ((s[g] >> 6) << 4);
        int low_offset = kind == 13 ? 48 : 16;
        int q = (p[low_offset + (i / 64) * 32 + i % 32] >> (4 * (g % 2))) & 15;
        if (kind == 13) q |= ((p[16 + i % 32] >> g) & 1) << 4;
        return half_value(p) * (float)scale * (float)q - half_value(p + 2) * (float)minimum;
    }
    int h = i / 128, lane = i % 32, quarter = (i % 128) / 32;
    int low = (p[h * 64 + lane + (quarter % 2) * 32] >> (4 * (quarter / 2))) & 15;
    int high = (p[128 + h * 32 + lane] >> (2 * quarter)) & 3;
    int q = (low | (high << 4)) - 32;
    return half_value(p + 208) * (float)((const signed char *)p)[192 + i / 16] * (float)q;
}

// Packed rows are block-aligned; every participating warp covers a whole
// 32-value sub-block. Decode shared metadata once, not once per weight.
template<int Kind>
__device__ __forceinline__ float warp_weight(const u8 *data, u64 index) {
    if (Kind == 0 || Kind == 1) return weight(data, Kind, index);
    int block = (Kind == 6 || Kind == 8) ? 32 : 256;
    int size = Kind == 6 ? 22 : Kind == 8 ? 34 : Kind == 12 ? 144 : Kind == 13 ? 176 : 210;
    const u8 *p = data + (index / block) * size;
    int i = index % block, lane = threadIdx.x % 32;
    float d = lane == 0 ? half_value(p + (Kind == 14 ? 208 : 0)) : 0.0f;
    d = __shfl_sync(0xffffffff, d, 0);
    if (Kind == 6) {
        unsigned int high = lane == 0 ? (unsigned int)p[2] | ((unsigned int)p[3] << 8)
            | ((unsigned int)p[4] << 16) | ((unsigned int)p[5] << 24) : 0;
        high = __shfl_sync(0xffffffff, high, 0);
        int q = ((p[6 + i % 16] >> (4 * (i / 16))) & 15) + ((high >> i) & 1) * 16 - 16;
        return d * (float)q;
    }
    if (Kind == 8) return d * (float)((const signed char *)p)[2 + i];
    if (Kind == 12 || Kind == 13) {
        int packed = 0, g = i / 32;
        float minimum_d = lane == 0 ? half_value(p + 2) : 0.0f;
        minimum_d = __shfl_sync(0xffffffff, minimum_d, 0);
        if (lane == 0) {
            const u8 *s = p + 4;
            int scale = g < 4 ? s[g] & 63 : (s[g + 4] & 15) | ((s[g - 4] >> 6) << 4);
            int minimum = g < 4 ? s[g + 4] & 63 : (s[g + 4] >> 4) | ((s[g] >> 6) << 4);
            packed = scale | (minimum << 8);
        }
        packed = __shfl_sync(0xffffffff, packed, 0);
        int q = (p[(Kind == 13 ? 48 : 16) + (i / 64) * 32 + i % 32] >> (4 * (g % 2))) & 15;
        if (Kind == 13) q |= ((p[16 + i % 32] >> g) & 1) << 4;
        return d * (float)(packed & 255) * (float)q - minimum_d * (float)(packed >> 8);
    }
    int scale = lane % 16 == 0 ? (int)((const signed char *)p)[192 + i / 16] : 0;
    scale = __shfl_sync(0xffffffff, scale, lane & 16);
    int h = i / 128, quarter = (i % 128) / 32;
    int low = (p[h * 64 + lane + (quarter % 2) * 32] >> (4 * (quarter / 2))) & 15;
    int high = (p[128 + h * 32 + lane] >> (2 * quarter)) & 3;
    return d * (float)scale * (float)((low | (high << 4)) - 32);
}

#if defined(CUDA_PACKED_REDUCTION_RESEARCH) || defined(CUDA_BLOCK_REDUCTION_RESEARCH) || defined(CUDA_M8_PROTOTYPES) || defined(CUDA_QWEN35_MATVEC_REDUCTION)
__device__ __forceinline__ float reduce_first_warp_256(const float *partial, int lane, float value) {
    // Keep the 128/64/32 addition tree before reducing the last warp.
    float a = value + partial[lane + 128];
    float b = partial[lane + 64] + partial[lane + 192];
    float c = partial[lane + 32] + partial[lane + 160];
    float d = partial[lane + 96] + partial[lane + 224];
    float sum = (a + b) + (c + d);
    #pragma unroll
    for (int stride = 16; stride > 0; stride /= 2)
        sum += __shfl_down_sync(0xffffffff, sum, stride);
    return sum;
}
#endif

__device__ float sum_block(float value) {
    __shared__ float partial[256];
    int lane = threadIdx.x;
    partial[lane] = value;
    __syncthreads();
#ifdef CUDA_BLOCK_REDUCTION_RESEARCH
    if (blockDim.x == 256) {
        __shared__ float result;
        if (lane < 32) {
            float sum = reduce_first_warp_256(partial, lane, value);
            if (lane == 0) result = sum;
        }
        __syncthreads();
        return result;
    }
#endif
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (lane < stride) partial[lane] += partial[lane + stride];
        __syncthreads();
    }
    return partial[0];
}

__device__ __forceinline__ void embedding_impl(const u8 *w, int kind, float *x, int width, int token) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < width) x[i] = weight(w, kind, (u64)token * width + i);
}

extern "C" __global__ void embedding(const u8 *w, int kind, float *x, int width, int token) {
    embedding_impl(w, kind, x, width, token);
}

extern "C" __global__ void embedding_decode(const u8 *w, int kind, float *x, int width,
                                            const int *control) {
    embedding_impl(w, kind, x, width, control[0]);
}

extern "C" __global__ void check_finite(const float *x, int count, int *status) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count && !isfinite(x[i])) atomicExch(status, 1);
}

template<int Kind>
__device__ float matrix_row(const u8 *w, const float *x, u64 offset, int width) {
    float sum = 0.0f;
    for (int i = threadIdx.x; i < width; i += blockDim.x)
        sum += weight(w, Kind, offset + i) * x[i];
    __shared__ float partial[256];
    int lane = threadIdx.x;
    partial[lane] = sum;
    __syncthreads();
#ifdef CUDA_QWEN35_MATVEC_REDUCTION
    return lane < 32 ? reduce_first_warp_256(partial, lane, sum) : 0.0f;
#else
    for (int stride = 128; stride >= 64; stride /= 2) {
        if (lane < stride) partial[lane] += partial[lane + stride];
        __syncthreads();
    }
    // Keep the original 256-lane addition tree; only its final warp drops barriers.
    if (lane < 32) {
        sum = partial[lane] + partial[lane + 32];
        for (int stride = 16; stride > 0; stride /= 2)
            sum += __shfl_down_sync(0xffffffff, sum, stride);
    }
    return sum;
#endif
}

extern "C" __global__ void matvec(const u8 *w, int kind, const float *x, float *y,
                                 int width, int rows, const int *ids, int moe, int batched) {
    int row = blockIdx.x, selected = blockIdx.y;
    int expert = moe ? ids[selected] : 0;
    u64 offset = ((u64)expert * rows + row) * width;
    const float *input = x + (batched ? (u64)selected * width : 0);
    float sum;
    switch (kind) {
        case 0: sum = matrix_row<0>(w, input, offset, width); break;
        case 1: sum = matrix_row<1>(w, input, offset, width); break;
        case 2: sum = matrix_row<2>(w, input, offset, width); break;
        case 3: sum = matrix_row<3>(w, input, offset, width); break;
        case 6: sum = matrix_row<6>(w, input, offset, width); break;
        case 7: sum = matrix_row<7>(w, input, offset, width); break;
        case 8: sum = matrix_row<8>(w, input, offset, width); break;
        case 10: sum = matrix_row<10>(w, input, offset, width); break;
        case 11: sum = matrix_row<11>(w, input, offset, width); break;
        case 12: sum = matrix_row<12>(w, input, offset, width); break;
        case 13: sum = matrix_row<13>(w, input, offset, width); break;
        case 14: sum = matrix_row<14>(w, input, offset, width); break;
        case 20: sum = matrix_row<20>(w, input, offset, width); break;
        case 30: sum = matrix_row<30>(w, input, offset, width); break;
        default: sum = __int_as_float(0x7fffffff); break;
    }
    if (threadIdx.x == 0) y[(u64)selected * rows + row] = sum;
}

#ifdef CUDA_QWEN35_MATVEC_CACHE
extern "C" __global__ void qwen35_expand_k_coefficients(const u8 *w, int kind,
    float *out, int count) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    int sub = i % 16;
    int stride = kind == 12 ? 144 : kind == 13 ? 176 : 210;
    const u8 *p = w + (u64)(i / 16) * stride;
    if (kind == 14) {
        out[i] = half_value(p + 208) * (float)((const signed char *)p)[192 + sub];
    } else {
        const u8 *s = p + 4;
        int g = sub % 8;
        int scale = g < 4 ? s[g] & 63 : (s[g + 4] & 15) | ((s[g - 4] >> 6) << 4);
        int minimum = g < 4 ? s[g + 4] & 63 : (s[g + 4] >> 4) | ((s[g] >> 6) << 4);
        out[i] = sub < 8 ? half_value(p) * (float)scale
                         : half_value(p + 2) * (float)minimum;
    }
}

template<int Kind, int Blocks = 0>
__device__ float qwen35_cached_dot(const u8 *w, const float *coefficients,
    const float *x, int width) {
    constexpr int stride = Kind == 12 ? 144 : Kind == 13 ? 176 : 210;
    int i = threadIdx.x, g = i / 32, lane = i % 32;
    float sum = 0.0f;
    for (int block = 0; block < (Blocks ? Blocks : width / 256); ++block) {
        const u8 *p = w + (u64)block * stride;
        const float *c = coefficients + (u64)block * 16;
        float dequant;
        if constexpr (Kind == 14) {
            int h = i / 128, q = g % 4;
            int low = (p[h * 64 + lane + (q % 2) * 32] >> (4 * (q / 2))) & 15;
            int high = (p[128 + h * 32 + lane] >> (2 * q)) & 3;
            dequant = c[i / 16] * (float)((low | (high << 4)) - 32);
        } else {
            int q = (p[(Kind == 13 ? 48 : 16) + (i / 64) * 32 + lane] >> (4 * (g % 2))) & 15;
            if constexpr (Kind == 13) q |= ((p[16 + lane] >> g) & 1) << 4;
            dequant = c[g] * (float)q - c[g + 8];
        }
        sum += dequant * x[block * 256 + i];
    }
    return sum;
}

template<int Blocks>
__device__ float qwen35_cached_kind_dot(const u8 *w, const float *c, int kind,
                                       const float *x, int width, u64 offset) {
    switch (kind) {
        case 12: return qwen35_cached_dot<12, Blocks>(w + offset * 144, c, x, width);
        case 13: return qwen35_cached_dot<13, Blocks>(w + offset * 176, c, x, width);
        default: return qwen35_cached_dot<14, Blocks>(w + offset * 210, c, x, width);
    }
}

extern "C" __global__ void qwen35_cached_matvec(const u8 *w, const float *coefficients,
    int kind, const float *x, float *y, int width) {
    __shared__ float partial[256];
    int row = blockIdx.x, tid = threadIdx.x;
    u64 offset = (u64)row * (width / 256);
    coefficients += offset * 16;
    float sum;
#ifdef CUDA_QWEN35_MATVEC_FIXED_WIDTH
    if (width == 1024) sum = qwen35_cached_kind_dot<4>(w, coefficients, kind, x, width, offset);
    else if (width == 2048) sum = qwen35_cached_kind_dot<8>(w, coefficients, kind, x, width, offset);
#ifdef CUDA_QWEN35_MATVEC_FFN_WIDTH
    else if (width == 3584) sum = qwen35_cached_kind_dot<14>(w, coefficients, kind, x, width, offset);
#endif
    else
#endif
    sum = qwen35_cached_kind_dot<0>(w, coefficients, kind, x, width, offset);
    partial[tid] = sum;
    __syncthreads();
    if (tid < 32) {
        sum = reduce_first_warp_256(partial, tid, sum);
        if (tid == 0) y[row] = sum;
    }
}
#ifdef CUDA_QWEN35_FUSED_FFN
extern "C" __global__ void qwen35_cached_swiglu(const u8 *gate, const float *gate_coeff,
    const u8 *up, const float *up_coeff, const float *x, float *y, int width) {
    __shared__ float pg[256], pu[256];
    int tid = threadIdx.x, row = blockIdx.x;
    u64 offset = (u64)row * (width / 256);
    float g, u;
#ifdef CUDA_QWEN35_SWIGLU_FIXED_WIDTH
    if (width == 1024) {
        g = qwen35_cached_dot<12, 4>(gate + offset * 144, gate_coeff + offset * 16, x, width);
        u = qwen35_cached_dot<12, 4>(up + offset * 144, up_coeff + offset * 16, x, width);
    } else
#endif
    {
        g = qwen35_cached_dot<12>(gate + offset * 144, gate_coeff + offset * 16, x, width);
        u = qwen35_cached_dot<12>(up + offset * 144, up_coeff + offset * 16, x, width);
    }
    pg[tid] = g;
    pu[tid] = u;
    __syncthreads();
    if (tid < 32) {
        g = reduce_first_warp_256(pg, tid, g);
        u = reduce_first_warp_256(pu, tid, u);
        if (tid == 0) y[row] = g / (1.0f + expf(-g)) * u;
    }
}
#endif
#endif

#if defined(CUDA_M8_PROTOTYPES) || defined(CUDA_PACKED_DECODE)
#ifdef CUDA_PACKED_REDUCTION_RESEARCH
__device__ __forceinline__ float packed_sum_block(float value) {
    __shared__ float partial[256];
    int lane = threadIdx.x;
    partial[lane] = value;
    __syncthreads();
    if (lane >= 32) return 0.0f;
    return reduce_first_warp_256(partial, lane, value);
}
#endif

// One warp owns a complete 256-value block, with eight independent K chains.
// Q4_K blocks are 16-byte aligned; Q6_K's 210-byte stride permits only u16
// loads. Metadata stays local to the lane, without warp-wide broadcasts.
template<int Kind>
__device__ __forceinline__ float packed_matrix_lane(const u8 *w, const float *x,
                                                    u64 offset, int width, int tid) {
    float parts[8] = {};
    int lane = tid & 31;
    for (int first = (tid / 32) * 256; first < width; first += 2048) {
        const u8 *p = w + ((offset + first) / 256) * (Kind == 12 ? 144 : 210);
        float d = half_value(p + (Kind == 12 ? 0 : 208));
        if (Kind == 12) {
            int g = (lane / 8) * 2, col = (lane & 7) * 4;
            unsigned int q = *(const unsigned int *)(p + 16 + (g / 2) * 32 + col);
            float dm = half_value(p + 2);
            const u8 *s = p + 4;
            #pragma unroll
            for (int half = 0; half < 2; half++) {
                int group = g + half;
                int scale = group < 4 ? s[group] & 63
                    : (s[group + 4] & 15) | ((s[group - 4] >> 6) << 4);
                int minimum = group < 4 ? s[group + 4] & 63
                    : (s[group + 4] >> 4) | ((s[group] >> 6) << 4);
                float ds = d * (float)scale, ms = dm * (float)minimum;
                #pragma unroll
                for (int j = 0; j < 4; j++) {
                    float decoded = ds * (float)((q >> (j * 8 + half * 4)) & 15) - ms;
                    parts[half * 4 + j] += decoded * x[first + group * 32 + col + j];
                }
            }
        } else {
            int h = lane / 16, col = (lane & 15) * 2;
            unsigned int lo0 = *(const unsigned short *)(p + h * 64 + col);
            unsigned int lo1 = *(const unsigned short *)(p + h * 64 + 32 + col);
            unsigned int hi = *(const unsigned short *)(p + 128 + h * 32 + col);
            #pragma unroll
            for (int quarter = 0; quarter < 4; quarter++) {
                unsigned int low = (quarter & 1) ? lo1 : lo0;
                int group = h * 8 + quarter * 2 + col / 16;
                float ds = d * (float)((const signed char *)p)[192 + group];
                #pragma unroll
                for (int j = 0; j < 2; j++) {
                    int ql = (low >> (j * 8 + (quarter / 2) * 4)) & 15;
                    int qh = (hi >> (j * 8 + quarter * 2)) & 3;
                    float decoded = ds * (float)((ql | (qh << 4)) - 32);
                    parts[quarter * 2 + j] += decoded
                        * x[first + h * 128 + quarter * 32 + col + j];
                }
            }
        }
    }
    parts[0] += parts[4];
    parts[1] += parts[5];
    parts[2] += parts[6];
    parts[3] += parts[7];
    parts[0] += parts[2];
    parts[1] += parts[3];
    return parts[0] + parts[1];
}

template<int Kind>
__device__ __forceinline__ float packed_matrix_row(const u8 *w, const float *x,
                                                   u64 offset, int width) {
    float partial = packed_matrix_lane<Kind>(w, x, offset, width, threadIdx.x);
#ifdef CUDA_PACKED_REDUCTION_RESEARCH
    return packed_sum_block(partial);
#else
    return sum_block(partial);
#endif
}

// Experimental drop-in launch contract: grid=(rows, selections, 1),
// block=(256, 1, 1), no dynamic shared memory. Not production-dispatched.
extern "C" __global__ void matvec_packed_f32(const u8 *w, int kind, const float *x, float *y,
                                            int width, int rows, const int *ids,
                                            int moe, int batched) {
    int row = blockIdx.x, selected = blockIdx.y;
    if (row >= rows) return;
    int expert = moe ? ids[selected] : 0;
    u64 offset = ((u64)expert * rows + row) * width;
    const float *input = x + (batched ? (u64)selected * width : 0);
    float sum;
    // Non-block-aligned rows may share packed storage blocks. Keep their exact
    // scalar indexing, and never issue a vector load across a storage tail.
    if ((kind == 12 || kind == 14) && width % 256 == 0) {
        sum = kind == 12 ? packed_matrix_row<12>(w, input, offset, width)
                         : packed_matrix_row<14>(w, input, offset, width);
    } else {
        switch (kind) {
            case 0: sum = matrix_row<0>(w, input, offset, width); break;
            case 1: sum = matrix_row<1>(w, input, offset, width); break;
            case 6: sum = matrix_row<6>(w, input, offset, width); break;
            case 8: sum = matrix_row<8>(w, input, offset, width); break;
            case 12: sum = matrix_row<12>(w, input, offset, width); break;
            case 13: sum = matrix_row<13>(w, input, offset, width); break;
            case 14: sum = matrix_row<14>(w, input, offset, width); break;
            default: sum = __int_as_float(0x7fffffff); break;
        }
    }
    if (threadIdx.x == 0) y[(u64)selected * rows + row] = sum;
}

#if defined(CUDA_M8_PROTOTYPES) || defined(CUDA_PACKED_GATE_UP_RESEARCH)
extern "C" __global__ void matvec_packed_gate_up(const u8 *gate, int gate_kind,
    const u8 *up, int up_kind, const float *x, float *y, int rows, const int *ids) {
    __shared__ float gate_partial[256], up_partial[256];
    int tid = threadIdx.x, row = blockIdx.x, selected = blockIdx.y;
    if (row >= rows) return;
    u64 offset = ((u64)ids[selected] * rows + row) * 2048;
    float g = gate_kind == 12 ? packed_matrix_lane<12>(gate, x, offset, 2048, tid)
                              : packed_matrix_lane<14>(gate, x, offset, 2048, tid);
    float u = up_kind == 12 ? packed_matrix_lane<12>(up, x, offset, 2048, tid)
                            : packed_matrix_lane<14>(up, x, offset, 2048, tid);
    gate_partial[tid] = g;
    up_partial[tid] = u;
    __syncthreads();
    if (tid < 32) {
        g = reduce_first_warp_256(gate_partial, tid, g);
        u = reduce_first_warp_256(up_partial, tid, u);
        if (tid == 0) y[(u64)selected * rows + row] = g / (1.0f + expf(-g)) * u;
    }
}
#endif

#if defined(CUDA_QWEN3_QKV_128)
template<int Kind>
__device__ __forceinline__ float qwen3_qkv_reduce_128(const u8 *w, const float *x,
    u64 offset, int width) {
    __shared__ float partial[128];
    int tid = threadIdx.x;
    float pair = packed_matrix_lane<Kind>(w, x, offset, width, tid)
               + packed_matrix_lane<Kind>(w, x, offset, width, tid + 128);
    partial[tid] = pair;
    __syncthreads();
    if (tid >= 32) return 0.0f;
    float sum = (pair + partial[tid + 64]) +
                (partial[tid + 32] + partial[tid + 96]);
    #pragma unroll
    for (int stride = 16; stride > 0; stride /= 2)
        sum += __shfl_down_sync(0xffffffff, sum, stride);
    return sum;
}
#ifdef CUDA_QWEN3_LAYER_MATVEC_128
extern "C" __global__ void qwen3_layer_matvec_128(const u8 *w, int kind,
    const float *x, float *y, int width) {
    const int row = blockIdx.x;
    const u64 offset = (u64)row * width;
    float sum = kind == 12 ? qwen3_qkv_reduce_128<12>(w, x, offset, width)
                           : qwen3_qkv_reduce_128<14>(w, x, offset, width);
    if (threadIdx.x == 0) y[row] = sum;
}
#endif
#endif

#if defined(CUDA_M8_PROTOTYPES) || defined(CUDA_DECODE_FUSION_RESEARCH) || defined(CUDA_QWEN3_FUSED_QKV)
extern "C" __global__ void matvec_packed_qkv(
    const u8 *wq, int q_kind, int q_rows, float *yq,
    const u8 *wk, int k_kind, int k_rows, float *yk,
    const u8 *wv, int v_kind, int v_rows, float *yv,
    const float *x, int width) {
    int row = blockIdx.x;
    const u8 *w;
    int kind;
    float *y;
    if (row < q_rows) {
        w = wq;
        kind = q_kind;
        y = yq;
    } else if (row < q_rows + k_rows) {
        row -= q_rows;
        w = wk;
        kind = k_kind;
        y = yk;
    } else if (row < q_rows + k_rows + v_rows) {
        row -= q_rows + k_rows;
        w = wv;
        kind = v_kind;
        y = yv;
    } else {
        return;
    }
    u64 offset = (u64)row * width;
#ifdef CUDA_QWEN3_QKV_128
    float sum = kind == 12 ? qwen3_qkv_reduce_128<12>(w, x, offset, width)
                           : qwen3_qkv_reduce_128<14>(w, x, offset, width);
#else
    float sum = kind == 12 ? packed_matrix_row<12>(w, x, offset, width)
                           : packed_matrix_row<14>(w, x, offset, width);
#endif
    if (threadIdx.x == 0) y[row] = sum;
}
#endif

#ifdef CUDA_M8_PROTOTYPES
template<int Kind>
__device__ __forceinline__ void packed_k_pair(const u8 *w, const float *x, float *y, int width) {
    __shared__ float partial[128];
    int tid = threadIdx.x, row = blockIdx.x;
    u64 offset = (u64)row * width;
    float pair = packed_matrix_lane<Kind>(w, x, offset, width, tid) +
                 packed_matrix_lane<Kind>(w, x, offset, width, tid + 128);
    partial[tid] = pair;
    __syncthreads();
    if (tid < 32) {
        float sum = (pair + partial[tid + 64]) +
                    (partial[tid + 32] + partial[tid + 96]);
        #pragma unroll
        for (int stride = 16; stride > 0; stride /= 2)
            sum += __shfl_down_sync(0xffffffff, sum, stride);
        if (tid == 0) y[row] = sum;
    }
}

extern "C" __global__ void matvec_packed_k_pair(
    const u8 *w, int kind, const float *x, float *y, int width, int rows,
    const int *ids, int moe, int batched) {
    if (blockIdx.x >= rows || moe || batched || width % 256 != 0) return;
    if (kind == 12) packed_k_pair<12>(w, x, y, width);
    if (kind == 14) packed_k_pair<14>(w, x, y, width);
}

extern "C" __global__ void matvec_packed_down_two_row(
    const u8 *w, int kind, const float *x, float *y, int rows, const int *ids) {
    __shared__ float input[768], partial[2][96];
    int tid = threadIdx.x, selected = blockIdx.y;
    #pragma unroll
    for (int k = tid; k < 768; k += 256) input[k] = x[selected * 768 + k];
    __syncthreads();
    int local_row = tid / 128, task = tid % 128;
    int row = blockIdx.x * 2 + local_row;
    if (row < rows && task < 96) {
        u64 offset = ((u64)ids[selected] * rows + row) * 768;
        partial[local_row][task] = kind == 12
            ? packed_matrix_lane<12>(w, input, offset, 768, task)
            : packed_matrix_lane<14>(w, input, offset, 768, task);
    }
    __syncthreads();
    if (row < rows && task < 32) {
        // Logical warps 3..7 have no input blocks, but keep their zero additions.
        float a = __fadd_rn(partial[local_row][task], 0.0f);
        float b = __fadd_rn(partial[local_row][task + 64], 0.0f);
        float c = __fadd_rn(partial[local_row][task + 32], 0.0f);
        float sum = (a + b) + __fadd_rn(c, 0.0f);
        #pragma unroll
        for (int stride = 16; stride > 0; stride /= 2)
            sum += __shfl_down_sync(0xffffffff, sum, stride);
        if (task == 0) y[(u64)selected * rows + row] = sum;
    }
}
#endif
#if defined(CUDA_M8_PROTOTYPES) || defined(CUDA_PACKED_DOWN_RESEARCH)
template<int Kind>
__device__ __forceinline__ float packed_down_warp_row(const u8 *w, const float *x,
                                                      u64 offset, int lane) {
    float a = __fadd_rn(packed_matrix_lane<Kind>(w, x, offset, 768, lane), 0.0f);
    float b = __fadd_rn(packed_matrix_lane<Kind>(w, x, offset, 768, lane + 64), 0.0f);
    float c = __fadd_rn(packed_matrix_lane<Kind>(w, x, offset, 768, lane + 32), 0.0f);
    return (a + b) + __fadd_rn(c, 0.0f);
}

extern "C" __global__ void matvec_packed_down_warp(
    const u8 *w, int kind, const float *x, float *y, int rows, const int *ids) {
    int row = blockIdx.x * 8 + threadIdx.x / 32, lane = threadIdx.x % 32;
    if (row >= rows) return;
    int selected = blockIdx.y;
    u64 offset = ((u64)ids[selected] * rows + row) * 768;
    const float *input = x + selected * 768;
    float sum = kind == 12 ? packed_down_warp_row<12>(w, input, offset, lane)
                           : packed_down_warp_row<14>(w, input, offset, lane);
    #pragma unroll
    for (int stride = 16; stride > 0; stride /= 2)
        sum += __shfl_down_sync(0xffffffff, sum, stride);
    if (lane == 0) y[(u64)selected * rows + row] = sum;
}
#endif
#endif

extern "C" __global__ void normalize(const float *x, const u8 *w, float *y, int n, float epsilon) {
    x += (u64)blockIdx.x * n;
    y += (u64)blockIdx.x * n;
    float sum = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) sum += x[i] * x[i];
    float inverse = 1.0f / sqrtf(sum_block(sum) / (float)n + epsilon);
    for (int i = threadIdx.x; i < n; i += blockDim.x) y[i] = x[i] * inverse * ((const float *)w)[i];
}

template<bool Affine>
__device__ __forceinline__ void layer_norm_impl(const float *x, const float *w,
                                                 const float *bias, float *y, int n, float epsilon) {
    x += (u64)blockIdx.x * n;
    y += (u64)blockIdx.x * n;
    float sum = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) sum += x[i];
    float mean = sum_block(sum) / (float)n;
    __syncthreads();
    sum = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        float centered = x[i] - mean;
        sum += centered * centered;
    }
    float inverse = 1.0f / sqrtf(sum_block(sum) / (float)n + epsilon);
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        float centered = (x[i] - mean) * inverse;
        y[i] = Affine ? centered * w[i] + bias[i] : centered;
    }
}

extern "C" __global__ void layer_norm_affine(const float *x, const float *w,
                                              const float *bias, float *y, int n, float epsilon) {
    layer_norm_impl<true>(x, w, bias, y, n, epsilon);
}

extern "C" __global__ void layer_norm_plain(const float *x, float *y, int n, float epsilon) {
    layer_norm_impl<false>(x, nullptr, nullptr, y, n, epsilon);
}

#if defined(CUDA_M8_PROTOTYPES) && defined(CUDA_BLOCK_REDUCTION_RESEARCH)
extern "C" __global__ void normalize_retained2048(const float *x, const u8 *w, float *y, float epsilon) {
    __shared__ float partial[256], inverse;
    int lane = threadIdx.x;
    float retained[8], sum = 0.0f;
    #pragma unroll
    for (int k = 0; k < 8; k++) {
        float value = x[lane + k * 256];
        retained[k] = value;
        sum += value * value;
    }
    partial[lane] = sum;
    __syncthreads();
    if (lane < 32) {
        float total = reduce_first_warp_256(partial, lane, sum);
        if (lane == 0) inverse = 1.0f / sqrtf(total / 2048.0f + epsilon);
    }
    __syncthreads();
    #pragma unroll
    for (int k = 0; k < 8; k++) {
        int i = lane + k * 256;
        y[i] = retained[k] * inverse * ((const float *)w)[i];
    }
}
#endif

extern "C" __global__ void head_norm(float *x, const u8 *w, int dim, float epsilon) {
    x += (u64)blockIdx.x * dim;
    float sum = 0.0f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) sum += x[i] * x[i];
    float inverse = 1.0f / sqrtf(sum_block(sum) / (float)dim + epsilon);
    for (int i = threadIdx.x; i < dim; i += blockDim.x) x[i] *= inverse * ((const float *)w)[i];
}

__device__ __forceinline__ void rotary_pair(float *x, int p, int dim, int rotated,
    int position, float base, int neox, const float *factors, float magnitude,
    const float *v_row = nullptr, float *k_dst = nullptr, float *v_dst = nullptr) {
    int pair = p % (rotated / 2), head = p / (rotated / 2);
    int a = head * dim + (neox ? pair : 2 * pair);
    int b = neox ? a + rotated / 2 : a + 1;
    float angle = (float)position * powf(base, -(float)(2 * pair) / (float)rotated);
    if (factors) angle /= factors[pair];
    float sine = sinf(angle) * magnitude, cosine = cosf(angle) * magnitude;
    float u = x[a], v = x[b];
    float xa = u * cosine - v * sine, xb = u * sine + v * cosine;
    if (k_dst) {
        k_dst[a] = xa;
        k_dst[b] = xb;
        v_dst[a] = v_row[a];
        v_dst[b] = v_row[b];
    } else {
        x[a] = xa;
        x[b] = xb;
    }
}

template<bool Partial = false>
__device__ __forceinline__ void rotary_impl(float *x, int n, int dim, int position, float base, int neox,
                                           const float *factors, float magnitude, int rotary_dim = 0) {
    x += (u64)blockIdx.y * n;
    position += blockIdx.y;
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    int rotated = Partial ? rotary_dim : dim;
    if (p >= (Partial ? n / dim * (rotated / 2) : n / 2)) return;
    rotary_pair(x, p, dim, rotated, position, base, neox, factors, magnitude);
}

extern "C" __global__ void llama_factored_rope_cache(float *q, float *k, const float *v,
    float *keys, float *values, int qw, int kw, int position, float base,
    int neox, const float *factors, float magnitude) {
    const int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p < qw / 2) {
        rotary_pair(q, p, 64, 64, position, base, neox, factors, magnitude);
    } else if (p < (qw + kw) / 2) {
        const u64 dst = (u64)position * kw;
        rotary_pair(k, p - qw / 2, 64, 64, position, base, neox, factors, magnitude,
                    v, keys + dst, values + dst);
    }
}

extern "C" __global__ void rotary(float *x, int n, int dim, int position, float base, int neox) {
    rotary_impl(x, n, dim, position, base, neox, nullptr, 1.0f);
}

extern "C" __global__ void rotary_partial_neox(float *x, int n, int dim, int position,
                                               float base, int rotary_dim) {
    rotary_impl<true>(x, n, dim, position, base, 1, nullptr, 1.0f, rotary_dim);
}

#ifdef CUDA_QWEN35_DECODE_GRAPH
extern "C" __global__ void qwen35_rotary_graph(float *x, int n, int dim, const int *control,
                                                float base, int rotary_dim) {
    rotary_impl<true>(x, n, dim, control[1], base, 1, nullptr, 1.0f, rotary_dim);
}
#endif

extern "C" __global__ void rotary_factored(float *x, int n, int dim, int position, float base,
                                           int neox, const float *factors, float magnitude) {
    rotary_impl(x, n, dim, position, base, neox, factors, magnitude);
}

extern "C" __global__ void rotary_decode(float *x, int n, int dim, const int *control,
                                         float base, int neox) {
    rotary_impl(x, n, dim, control[1], base, neox, nullptr, 1.0f);
}

__device__ __forceinline__ void cache_kv_impl(const float *k, const float *v,
                                              float *keys, float *values, int width, int position) {
    k += (u64)blockIdx.y * width;
    v += (u64)blockIdx.y * width;
    position += blockIdx.y;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < width) {
        keys[(u64)position * width + i] = k[i];
        values[(u64)position * width + i] = v[i];
    }
}

extern "C" __global__ void cache_kv(const float *k, const float *v, float *keys, float *values,
                                   int width, int position) {
    cache_kv_impl(k, v, keys, values, width, position);
}

extern "C" __global__ void cache_kv_decode(const float *k, const float *v,
                                           float *keys, float *values, int width,
                                           const int *control) {
    cache_kv_impl(k, v, keys, values, width, control[1]);
}

template<bool Gemma, bool Alibi = false>
__device__ __forceinline__ void attention_scores_impl(const float *q, const float *keys,
    float *scores, int dim, int heads, int kv_heads, int context, int count, int window, float scale,
    float softcap) {
    int head = blockIdx.x, time = blockIdx.y;
    count += blockIdx.z;
    if (time >= count || (Gemma && window > 0 && time < count - window)) return;
    q += (u64)blockIdx.z * heads * dim;
    scores += (u64)blockIdx.z * heads * context;
    int kv_head = head / (heads / kv_heads);
    float sum = 0.0f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x)
        sum += (Gemma ? q[(u64)head * dim + i] * scale : q[(u64)head * dim + i])
             * keys[((u64)time * kv_heads + kv_head) * dim + i];
    sum = sum_block(sum);
    if (threadIdx.x == 0) {
        float score = Gemma ? sum : sum / sqrtf((float)dim);
        if (Gemma && softcap > 0.0f)
            score = isfinite(score) ? softcap * tanhf(score * (1.0f / softcap))
                                    : __int_as_float(0x7fffffff);
        if (Alibi) {
            int base_heads = 1 << (31 - __clz((unsigned int)heads));
            float slope = head < base_heads
                ? powf(exp2f(-scale / base_heads), head + 1)
                : powf(exp2f(-scale / (2 * base_heads)), 2 * (head - base_heads) + 1);
            score -= slope * (count - 1 - time);
        }
        scores[(u64)head * context + time] = score;
    }
}

extern "C" __global__ void attention_scores(const float *q, const float *keys, float *scores,
                                            int dim, int heads, int kv_heads, int context, int count) {
    attention_scores_impl<false>(q, keys, scores, dim, heads, kv_heads, context, count, 0, 1.0f, 0.0f);
}

#ifdef CUDA_LLAMA_WARP_SCORES
extern "C" __global__ void llama_warp_scores_64(const float *q, const float *keys,
    float *scores, int heads, int kv_heads, int context, int count) {
    const int head = blockIdx.x;
    const int time = blockIdx.y * 4 + threadIdx.x / 32;
    if (time >= count) return;
    const int lane = threadIdx.x & 31;
    const int kv_head = head / (heads / kv_heads);
    const float *query = q + head * 64;
    const float *key = keys + ((u64)time * kv_heads + kv_head) * 64;
    float sum = (0.0f + query[lane] * key[lane])
              + (0.0f + query[lane + 32] * key[lane + 32]);
    #pragma unroll
    for (int stride = 16; stride != 0; stride /= 2)
        sum += __shfl_down_sync(0xffffffff, sum, stride);
    if (lane == 0) scores[(u64)head * context + time] = sum / 8.0f;
}
#endif

extern "C" __global__ void alibi_attention_scores(const float *q, const float *keys, float *scores,
    int dim, int heads, int kv_heads, int context, int count, float max_bias) {
    attention_scores_impl<false, true>(q, keys, scores, dim, heads, kv_heads,
                                       context, count, 0, max_bias, 0.0f);
}

extern "C" __global__ void gemma_attention_scores(const float *q, const float *keys, float *scores,
    int dim, int heads, int kv_heads, int context, int count, int window, float scale, float softcap) {
    attention_scores_impl<true>(q, keys, scores, dim, heads, kv_heads, context, count, window, scale, softcap);
}

template<bool Gemma, bool Split = false, bool NormalizeOnly = false>
__device__ __forceinline__ void attention_values_impl(float *scores, const float *values,
    float *output, int dim, int heads, int kv_heads, int context, int count, int window) {
    count += blockIdx.y;
    int start = Gemma && window > 0 && count > window ? count - window : 0;
    scores += (u64)blockIdx.y * heads * context;
    if constexpr (!NormalizeOnly) output += (u64)blockIdx.y * heads * dim;
    __shared__ float maxima[256];
    int head = blockIdx.x, lane = threadIdx.x;
    float *s = scores + (u64)head * context;
    float maximum = -__int_as_float(0x7f800000);
    for (int t = start + lane; t < count; t += blockDim.x) maximum = fmaxf(maximum, s[t]);
    maxima[lane] = maximum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (lane < stride) maxima[lane] = fmaxf(maxima[lane], maxima[lane + stride]);
        __syncthreads();
    }
    maximum = maxima[0];
    float total = 0.0f;
    for (int t = start + lane; t < count; t += blockDim.x) {
        s[t] = expf(s[t] - maximum);
        total += s[t];
    }
    total = sum_block(total);
#ifdef CUDA_DECODE_SCORE_REUSE_RESEARCH
    if (!Gemma) {
        for (int t = start + lane; t < count; t += blockDim.x)
            s[t] /= total;
        __syncthreads();
    }
#endif
    if constexpr (NormalizeOnly) return;
    int kv_head = head / (heads / kv_heads);
#ifdef CUDA_LLAMA_DECODE_VALUE_SPLIT
    if constexpr (Split) {
        __shared__ float partial_values[256];
        const int part = lane / dim;
        const int i = lane % dim;
        const int parts = 256 / dim;
        const int begin = (count * part) / parts;
        const int end = (count * (part + 1)) / parts;
        float value = 0.0f;
#ifdef CUDA_LLAMA_VALUE_UNROLL
        #pragma unroll 4
#endif
        for (int t = begin; t < end; ++t)
            value += s[t] * values[((u64)t * kv_heads + kv_head) * dim + i];
        partial_values[lane] = value;
        __syncthreads();
        if (part == 0) {
            value = partial_values[i] + partial_values[i + dim];
            if (parts == 4) value += partial_values[i + 2 * dim] + partial_values[i + 3 * dim];
            output[(u64)head * dim + i] = value;
        }
        return;
    }
#endif
    for (int i = lane; i < dim; i += blockDim.x) {
        float value = 0.0f;
        for (int t = start; t < count; t++)
#ifdef CUDA_DECODE_SCORE_REUSE_RESEARCH
            value += (Gemma ? s[t] / total : s[t]) * values[((u64)t * kv_heads + kv_head) * dim + i];
#else
            value += s[t] / total * values[((u64)t * kv_heads + kv_head) * dim + i];
#endif
        output[(u64)head * dim + i] = value;
    }
}

extern "C" __global__ void attention_values(float *scores, const float *values, float *output,
                                            int dim, int heads, int kv_heads, int context, int count) {
    attention_values_impl<false>(scores, values, output, dim, heads, kv_heads, context, count, 0);
}

#ifdef CUDA_QWEN35_DECODE_GRAPH
extern "C" __global__ void qwen35_attention_scores_graph(const float *q, const float *keys, float *scores,
    int dim, int heads, int kv_heads, int context, const int *control) {
    attention_scores_impl<false>(q, keys, scores, dim, heads, kv_heads, context, control[1] + 1, 0, 1.0f, 0.0f);
}

extern "C" __global__ void qwen35_attention_values_graph(float *scores, const float *values, float *output,
    int dim, int heads, int kv_heads, int context, const int *control) {
    attention_values_impl<false>(scores, values, output, dim, heads, kv_heads, context, control[1] + 1, 0);
}
#endif

#if defined(CUDA_QWEN35_VALUE_COLUMNS) && defined(CUDA_DECODE_SCORE_REUSE_RESEARCH)
extern "C" __global__ void qwen35_attention_normalize(float *scores,
    int heads, int kv_heads, int context, int count) {
    attention_values_impl<false, false, true>(
        scores, nullptr, nullptr, 256, heads, kv_heads, context, count, 0);
}

__device__ __forceinline__ void qwen35_attention_value_columns_impl(const float *scores,
    const float *values, float *output, int heads, int kv_heads, int context, int count) {
    constexpr int dim = 256;
    const int head = blockIdx.x;
    const int col = blockIdx.y * 32 + threadIdx.x;
    const int kv_head = head / (heads / kv_heads);
    const float *score = scores + (u64)head * context;
    const float *value = values + kv_head * dim + col;
    float sum = 0.0f;
    for (int t = 0; t < count; ++t)
        sum += score[t] * value[(u64)t * kv_heads * dim];
    output[head * dim + col] = sum;
}

extern "C" __global__ void qwen35_attention_value_columns(const float *scores,
    const float *values, float *output, int heads, int kv_heads, int context, int count) {
    qwen35_attention_value_columns_impl(scores, values, output, heads, kv_heads, context, count);
}

#ifdef CUDA_QWEN35_DECODE_GRAPH
extern "C" __global__ void qwen35_attention_normalize_graph(float *scores,
    int heads, int kv_heads, int context, const int *control) {
    attention_values_impl<false, false, true>(
        scores, nullptr, nullptr, 256, heads, kv_heads, context, control[1] + 1, 0);
}

extern "C" __global__ void qwen35_attention_value_columns_graph(const float *scores,
    const float *values, float *output, int heads, int kv_heads, int context, const int *control) {
    qwen35_attention_value_columns_impl(scores, values, output, heads, kv_heads, context, control[1] + 1);
}
#endif
#endif

#ifdef CUDA_LLAMA_DECODE_VALUE_SPLIT
extern "C" __global__ void attention_values_split(float *scores, const float *values, float *output,
    int dim, int heads, int kv_heads, int context, int count) {
    attention_values_impl<false, true>(scores, values, output, dim, heads, kv_heads, context, count, 0);
}
#endif

extern "C" __global__ void gemma_attention_values(float *scores, const float *values, float *output,
    int dim, int heads, int kv_heads, int context, int count, int window) {
    attention_values_impl<true>(scores, values, output, dim, heads, kv_heads, context, count, window);
}

extern "C" __global__ void gemma_embedding_scale(float *x, int count, float scale) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) x[i] *= scale;
}

__device__ __forceinline__ float gelu(float x) {
    return 0.5f * x * (1.0f + tanhf(0.7978846f * x * (1.0f + 0.044715f * x * x)));
}

extern "C" __global__ void gemma_geglu(float *gate, const float *up, int count) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) gate[i] = gelu(gate[i]) * up[i];
}

extern "C" __global__ void gelu_row_bias(float *up, const float *bias, int width) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < width) {
        u64 offset = (u64)blockIdx.y * width + i;
        up[offset] = gelu(up[offset] + bias[i]);
    }
}

extern "C" __global__ void gemma_softcap(float *logits, int count, float cap, int *status) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) {
        float x = logits[i];
        if (!isfinite(x)) atomicExch(status, 1);
        else logits[i] = cap * tanhf(x / cap);
    }
}

template<bool FullSoftmax>
__device__ __forceinline__ void router_impl(const float *scores, int count, int used,
                                            int *ids, float *weights, int *status) {
    __shared__ float best_scores[256];
    __shared__ int best_ids[256];
    __shared__ float maximum, total;
    __shared__ int invalid;
    scores += (u64)blockIdx.x * count;
    ids += (u64)blockIdx.x * used;
    weights += (u64)blockIdx.x * used;
    int lane = threadIdx.x;
    if (lane == 0) invalid = 0;
    __syncthreads();
    for (int i = lane; i < used; i += blockDim.x) { ids[i] = 0; weights[i] = 0.0f; }
    for (int i = lane; i < count; i += blockDim.x) {
        if (!isfinite(scores[i])) {
            atomicExch(status, 1);
            atomicExch(&invalid, 1);
        }
    }
    if (lane == 0) total = 0.0f;
    __syncthreads();
    if (invalid) return;
    for (int s = 0; s < used; s++) {
        int best = -1;
        float score = -__int_as_float(0x7f800000);
        for (int i = lane; i < count; i += blockDim.x) {
            bool selected = false;
            for (int j = 0; j < s; j++) if (ids[j] == i) selected = true;
            if (!selected && (best < 0 || scores[i] > score)) { best = i; score = scores[i]; }
        }
        best_ids[lane] = best;
        best_scores[lane] = score;
        __syncthreads();
        for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
            if (lane < stride) {
                int other = best_ids[lane + stride];
                int current = best_ids[lane];
                float other_score = best_scores[lane + stride], current_score = best_scores[lane];
                if (other >= 0 && (current < 0 || other_score > current_score
                    || (other_score == current_score && other < current))) {
                    best_ids[lane] = other;
                    best_scores[lane] = other_score;
                }
            }
            __syncthreads();
        }
        if (lane == 0) {
            ids[s] = best_ids[0];
            if (s == 0) maximum = best_scores[0];
            weights[s] = expf(best_scores[0] - maximum);
            total += weights[s];
        }
        __syncthreads();
    }
    float denominator = total;
    if (FullSoftmax) {
        float local = 0.0f;
        for (int i = lane; i < count; i += blockDim.x) local += expf(scores[i] - maximum);
        denominator = sum_block(local);
    }
    for (int s = lane; s < used; s += blockDim.x) weights[s] /= denominator;
}

extern "C" __global__ void router(const float *scores, int count, int used,
                                  int *ids, float *weights, int *status) {
    router_impl<false>(scores, count, used, ids, weights, status);
}

extern "C" __global__ void router_full_softmax(const float *scores, int count, int used,
                                               int *ids, float *weights, int *status) {
    router_impl<true>(scores, count, used, ids, weights, status);
}
#if defined(CUDA_M8_PROTOTYPES) || defined(CUDA_DECODE_FUSION_RESEARCH)
extern "C" __global__ void router_warp128(const float *scores, int used,
                                          int *ids, float *weights, int *status) {
    scores += (u64)blockIdx.x * 128;
    ids += (u64)blockIdx.x * used;
    weights += (u64)blockIdx.x * used;
    int lane = threadIdx.x;
    float val[4];
    bool bad = false;
    #pragma unroll
    for (int k = 0; k < 4; k++) {
        float v = scores[lane + k * 32];
        val[k] = v;
        if (!isfinite(v)) bad = true;
    }
    if (__any_sync(0xffffffffu, bad)) {
        for (int i = lane; i < used; i += 32) {
            ids[i] = 0;
            weights[i] = 0.0f;
        }
        if (lane == 0) atomicExch(status, 1);
        return;
    }
    bool chosen[4] = {false, false, false, false};
    float maximum = 0.0f, total = 0.0f, my_weight = 0.0f;
    for (int s = 0; s < used; s++) {
        int best_id = -1;
        float best_score = -__int_as_float(0x7f800000);
        #pragma unroll
        for (int k = 0; k < 4; k++) {
            int idx = lane + k * 32;
            float sc = val[k];
            if (!chosen[k] && (best_id < 0 || sc > best_score
                || (sc == best_score && idx < best_id))) {
                best_id = idx;
                best_score = sc;
            }
        }
        #pragma unroll
        for (int stride = 16; stride > 0; stride /= 2) {
            int other_id = __shfl_down_sync(0xffffffffu, best_id, stride);
            float other_score = __shfl_down_sync(0xffffffffu, best_score, stride);
            if (other_id >= 0 && (best_id < 0 || other_score > best_score
                || (other_score == best_score && other_id < best_id))) {
                best_id = other_id;
                best_score = other_score;
            }
        }
        int win_id = __shfl_sync(0xffffffffu, best_id, 0);
        float win_score = __shfl_sync(0xffffffffu, best_score, 0);
        #pragma unroll
        for (int k = 0; k < 4; k++) {
            if (lane + k * 32 == win_id) chosen[k] = true;
        }
        if (s == 0) maximum = win_score;
        float w = expf(win_score - maximum);
        total += w;
        if (lane == s) {
            ids[s] = win_id;
            my_weight = w;
        }
    }
    if (lane < used) weights[lane] = my_weight / total;
}

#endif

#if defined(CUDA_M8_PROTOTYPES) || defined(CUDA_DECODE_FUSION_RESEARCH) || defined(CUDA_QWEN3_FUSED_HEAD_ROPE) || defined(CUDA_QWEN3_PREFILL_FUSED_HEAD_ROPE)
__device__ __forceinline__ void qkv_head_norm_rope_cache_impl(
    float *q, const u8 *q_norm, int heads,
    float *k, const u8 *k_norm, const float *v,
    float *keys, float *values, int kv_heads,
    float epsilon, int position, float base, int neox) {
    __shared__ float normed[128];
    int b = blockIdx.x, tid = threadIdx.x;
    bool is_q = b < heads;
    int head = is_q ? b : b - heads;
    if (!is_q && head >= kv_heads) return;
    float *vec = (is_q ? q : k) + (u64)head * 128;
    const float *nw = (const float *)(is_q ? q_norm : k_norm);
    float val = tid < 128 ? vec[tid] : 0.0f;
    float sq = tid < 128 ? val * val : 0.0f;
    float inverse = 1.0f / sqrtf(sum_block(sq) / 128.0f + epsilon);
    if (tid < 128) normed[tid] = val * (inverse * nw[tid]);
    __syncthreads();
    if (tid < 64) {
        int pair = tid;
        int a = neox ? pair : 2 * pair;
        int b_idx = neox ? a + 64 : a + 1;
        float angle = (float)position * powf(base, -(float)(2 * pair) / 128.0f);
        float sine = sinf(angle), cosine = cosf(angle);
        float u = normed[a], w_val = normed[b_idx];
        float ra = u * cosine - w_val * sine;
        float rb = u * sine + w_val * cosine;
        vec[a] = ra;
        vec[b_idx] = rb;
        if (!is_q) {
            u64 kv_base = ((u64)position * kv_heads + head) * 128;
            keys[kv_base + a] = ra;
            keys[kv_base + b_idx] = rb;
        }
    }
    if (!is_q && tid >= 128) {
        int i = tid - 128;
        u64 kv_base = ((u64)position * kv_heads + head) * 128;
        values[kv_base + i] = v[(u64)head * 128 + i];
    }
}

#ifdef CUDA_QWEN3_PREFILL_FUSED_HEAD_ROPE
extern "C" __global__ void prefill_qkv_head_norm_rope_cache(
    float *q, const u8 *q_norm, int heads,
    float *k, const u8 *k_norm, const float *v,
    float *keys, float *values, int kv_heads,
    float epsilon, int position, float base) {
    u64 row = blockIdx.y;
    qkv_head_norm_rope_cache_impl(
        q + row * heads * 128, q_norm, heads,
        k + row * kv_heads * 128, k_norm, v + row * kv_heads * 128,
        keys, values, kv_heads, epsilon, position + (int)row, base, 1);
}
#endif

extern "C" __global__ void qkv_head_norm_rope_cache(
    float *q, const u8 *q_norm, int heads,
    float *k, const u8 *k_norm, const float *v,
    float *keys, float *values, int kv_heads,
    float epsilon, int position, float base, int neox) {
    qkv_head_norm_rope_cache_impl(
        q, q_norm, heads, k, k_norm, v, keys, values, kv_heads, epsilon, position, base, neox);
}

extern "C" __global__ void qkv_head_norm_rope_cache_decode(
    float *q, const u8 *q_norm, int heads,
    float *k, const u8 *k_norm, const float *v,
    float *keys, float *values, int kv_heads,
    float epsilon, const int *control, float base, int neox) {
    qkv_head_norm_rope_cache_impl(
        q, q_norm, heads, k, k_norm, v, keys, values, kv_heads, epsilon, control[1], base, neox);
}
#endif

#if defined(CUDA_M8_PROTOTYPES) || defined(CUDA_QWEN35_RESIDUAL_NORM)
extern "C" __global__ void residual_normalize(float *x, const float *projected,
    const u8 *w, float *y, int n, float epsilon) {
#ifdef CUDA_QWEN35_RESIDUAL_NORM
    const u64 offset = (u64)blockIdx.x * n;
    x += offset;
    projected += offset;
    y += offset;
#endif
    float sum = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        float xi = x[i] + projected[i];
        x[i] = xi;
        sum += xi * xi;
    }
    float inverse = 1.0f / sqrtf(sum_block(sum) / (float)n + epsilon);
    for (int i = threadIdx.x; i < n; i += blockDim.x)
        y[i] = x[i] * inverse * ((const float *)w)[i];
}
#endif

extern "C" __global__ void swiglu(float *gate, const float *up, int count) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) gate[i] = gate[i] / (1.0f + expf(-gate[i])) * up[i];
}

extern "C" __global__ void residual(float *x, const float *projected, int width) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < width) x[i] += projected[i];
}

extern "C" __global__ void add_row_bias(float *x, const float *bias, int width) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < width) x[(u64)blockIdx.y * width + i] += bias[i];
}

extern "C" __global__ void add_position_embedding(float *x, const u8 *table, int kind,
                                                   int width, int position) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < width) {
        u64 row = blockIdx.y;
        x[row * width + i] += weight(table, kind, ((u64)position + row) * width + i);
    }
}

extern "C" __global__ void combine(float *x, const float *experts, const float *weights, int width, int used) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < width) {
        float sum = 0.0f;
        for (int s = 0; s < used; s++) sum += weights[s] * experts[(u64)s * width + i];
        x[i] += sum;
    }
}

extern "C" __global__ void combine_batch(float *x, const float *experts,
                                        const float *weights, int width, int used) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int token = blockIdx.y;
    if (i < width) {
        float sum = 0.0f;
        for (int s = 0; s < used; s++)
            sum += weights[(u64)token * used + s]
                 * experts[((u64)token * used + s) * width + i];
        x[(u64)token * width + i] += sum;
    }
}
