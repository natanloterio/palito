typedef unsigned char u8;
typedef unsigned long long u64;

// KV cache element access. F32 caches (every profile but int8-fast) load and store
// floats unchanged; int8-fast stores FP16 bits (round to nearest, the conversion its
// FP16 prefill attention applies) and every kernel accumulates in F32.
typedef unsigned short kv_f16;

__device__ __forceinline__ float kv_load(const float *p, u64 i) { return p[i]; }

__device__ __forceinline__ float kv_load(const kv_f16 *p, u64 i) {
    float value;
    asm("cvt.f32.f16 %0, %1;" : "=f"(value) : "h"(p[i]));
    return value;
}

__device__ __forceinline__ float2 kv_unpack2(unsigned int packed) {
    float lo, hi;
    asm("{.reg .f16 l, h; mov.b32 {l, h}, %2; cvt.f32.f16 %0, l; cvt.f32.f16 %1, h;}"
        : "=f"(lo), "=f"(hi) : "r"(packed));
    return make_float2(lo, hi);
}

// Four consecutive elements; i % 4 == 0 (16-byte F32 / 8-byte F16 aligned).
__device__ __forceinline__ float4 kv_load4(const float *p, u64 i) {
    return *(const float4 *)(p + i);
}

__device__ __forceinline__ float4 kv_load4(const kv_f16 *p, u64 i) {
    uint2 bits = *(const uint2 *)(p + i);
    float2 a = kv_unpack2(bits.x), b = kv_unpack2(bits.y);
    return make_float4(a.x, a.y, b.x, b.y);
}

__device__ __forceinline__ void kv_store(float *p, u64 i, float value) { p[i] = value; }

__device__ __forceinline__ void kv_store(kv_f16 *p, u64 i, float value) {
    unsigned short bits;
    asm("cvt.rn.f16.f32 %0, %1;" : "=h"(bits) : "f"(value));
    p[i] = bits;
}

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
    int size = Kind == 6 ? 22 : Kind == 8 ? 34 : Kind == 12 ? 144 : 210;
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
    if (Kind == 12) {
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
        int q = (p[16 + (i / 64) * 32 + i % 32] >> (4 * (g % 2))) & 15;
        return d * (float)(packed & 255) * (float)q - minimum_d * (float)(packed >> 8);
    }
    int scale = lane % 16 == 0 ? (int)((const signed char *)p)[192 + i / 16] : 0;
    scale = __shfl_sync(0xffffffff, scale, lane & 16);
    int h = i / 128, quarter = (i % 128) / 32;
    int low = (p[h * 64 + lane + (quarter % 2) * 32] >> (4 * (quarter / 2))) & 15;
    int high = (p[128 + h * 32 + lane] >> (2 * quarter)) & 3;
    return d * (float)scale * (float)((low | (high << 4)) - 32);
}

#if defined(CUDA_PACKED_REDUCTION_RESEARCH) || defined(CUDA_BLOCK_REDUCTION_RESEARCH) || defined(CUDA_M8_PROTOTYPES)
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
// Exact float of an integer in [0, 2^23): avoids the low-throughput I2F conversion.
__device__ __forceinline__ float small_uint_float(unsigned int value) {
    return __int_as_float(0x4b000000 | value) - 8388608.0f;
}

// Accumulates one packed 256-value block (Q4_K or Q6_K) of a row into this lane's
// eight partial sums; `x` points at the block's first input value.
template<int Kind>
__device__ __forceinline__ void packed_block_accumulate(const u8 *p, const float *x, int lane,
                                                        float *parts) {
    if (Kind == 12) {
        // 144-byte blocks are 16-byte aligned: wide loads for d/dmin, the 12 scale
        // bytes and the input; arithmetic and its order match the scalar decode.
        int g = (lane / 8) * 2, col = (lane & 7) * 4;
        unsigned int q = *(const unsigned int *)(p + 16 + (g / 2) * 32 + col);
        unsigned int dd = *(const unsigned int *)p;
        float d = half_value_bits(dd & 0xffff), dm = half_value_bits(dd >> 16);
        const unsigned int *sw = (const unsigned int *)(p + 4);
        unsigned int s0 = sw[0], s1 = sw[1], s2 = sw[2];
        #pragma unroll
        for (int half = 0; half < 2; half++) {
            int group = g + half, shift = 8 * (group & 3);
            // Bytes s[group & 3], s[(group & 3) + 4] and s[(group & 3) + 8] of the scales.
            unsigned int lo = (s0 >> shift) & 255, mid = (s1 >> shift) & 255;
            unsigned int hi = (s2 >> shift) & 255;
            int scale = group < 4 ? lo & 63 : (hi & 15) | ((lo >> 6) << 4);
            int minimum = group < 4 ? mid & 63 : (hi >> 4) | ((mid >> 6) << 4);
            float ds = d * (float)scale, ms = dm * (float)minimum;
            float4 xv = *(const float4 *)(x + group * 32 + col);
            float xs[4] = {xv.x, xv.y, xv.z, xv.w};
            #pragma unroll
            for (int j = 0; j < 4; j++) {
                float decoded = ds * small_uint_float((q >> (j * 8 + half * 4)) & 15) - ms;
                parts[half * 4 + j] += decoded * xs[j];
            }
        }
    } else {
        float d = half_value(p + 208);
        int h = lane / 16, col = (lane & 15) * 2;
        unsigned int lo0 = *(const unsigned short *)(p + h * 64 + col);
        unsigned int lo1 = *(const unsigned short *)(p + h * 64 + 32 + col);
        unsigned int hi = *(const unsigned short *)(p + 128 + h * 32 + col);
        #pragma unroll
        for (int quarter = 0; quarter < 4; quarter++) {
            unsigned int low = (quarter & 1) ? lo1 : lo0;
            int group = h * 8 + quarter * 2 + col / 16;
            float ds = d * (float)((const signed char *)p)[192 + group];
            float2 xv = *(const float2 *)(x + h * 128 + quarter * 32 + col);
            int ql0 = (low >> ((quarter / 2) * 4)) & 15;
            int qh0 = (hi >> (quarter * 2)) & 3;
            int ql1 = (low >> (8 + (quarter / 2) * 4)) & 15;
            int qh1 = (hi >> (8 + quarter * 2)) & 3;
            parts[quarter * 2] += ds * (small_uint_float(ql0 | (qh0 << 4)) - 32.0f) * xv.x;
            parts[quarter * 2 + 1] += ds * (small_uint_float(ql1 | (qh1 << 4)) - 32.0f) * xv.y;
        }
    }
}

__device__ __forceinline__ float fold_parts(float *parts) {
    parts[0] += parts[4];
    parts[1] += parts[5];
    parts[2] += parts[6];
    parts[3] += parts[7];
    parts[0] += parts[2];
    parts[1] += parts[3];
    return parts[0] + parts[1];
}

template<int Kind>
__device__ __forceinline__ float packed_lane_span(const u8 *w, const float *x, u64 offset,
                                                  int width, int lane, int start, int step) {
    float parts[8] = {};
    for (int first = start; first < width; first += step) {
        const u8 *p = w + ((offset + first) / 256) * (Kind == 12 ? 144 : 210);
        packed_block_accumulate<Kind>(p, x + first, lane, parts);
    }
    return fold_parts(parts);
}

__device__ __forceinline__ float warp_sum(float value) {
    #pragma unroll
    for (int stride = 16; stride > 0; stride /= 2)
        value += __shfl_xor_sync(0xffffffff, value, stride);
    return value;
}

// Q4_K with 8 lanes per 256-value block: each lane issues one 16-byte load of
// quantized values (32 values), so a warp covers 4 blocks per step. More bytes in
// flight per lane than packed_lane_span; F32 arithmetic, different summation order.
__device__ __forceinline__ void q4k_wide_block(const u8 *p, const float *x, int sub,
                                               float *parts) {
    uint4 qv = *(const uint4 *)(p + 16 + sub * 16);
    unsigned int dd = *(const unsigned int *)p;
    float d = half_value_bits(dd & 0xffff), dm = half_value_bits(dd >> 16);
    const unsigned int *sw = (const unsigned int *)(p + 4);
    unsigned int s0 = sw[0], s1 = sw[1], s2 = sw[2];
    unsigned int qw[4] = {qv.x, qv.y, qv.z, qv.w};
    int pair = sub >> 1;
    #pragma unroll
    for (int half = 0; half < 2; half++) {
        // Bytes 16*(sub&1)..+15 of 32-byte chunk `pair`: low nibbles belong to group
        // 2*pair, high nibbles to group 2*pair+1.
        int group = 2 * pair + half, shift = 8 * (group & 3);
        unsigned int lo = (s0 >> shift) & 255, mid = (s1 >> shift) & 255;
        unsigned int hi = (s2 >> shift) & 255;
        int scale = group < 4 ? lo & 63 : (hi & 15) | ((lo >> 6) << 4);
        int minimum = group < 4 ? mid & 63 : (hi >> 4) | ((mid >> 6) << 4);
        float ds = d * (float)scale, ms = dm * (float)minimum;
        const float *xb = x + 32 * group + (sub & 1) * 16;
        #pragma unroll
        for (int word = 0; word < 4; word++) {
            float4 xv = *(const float4 *)(xb + 4 * word);
            float xs[4] = {xv.x, xv.y, xv.z, xv.w};
            #pragma unroll
            for (int b = 0; b < 4; b++) {
                float decoded = ds * small_uint_float((qw[word] >> (8 * b + 4 * half)) & 15) - ms;
                parts[half * 4 + b] += decoded * xs[b];
            }
        }
    }
}

// One warp's share (`part` of `warps`) of a Q4_K row, already warp-reduced.
__device__ __forceinline__ float q4k_wide_row_part(const u8 *w, const float *x, u64 offset,
                                                   int width, int lane, int part, int warps) {
    float parts[8] = {};
    for (int first = (part * 4 + (lane >> 3)) * 256; first < width; first += warps * 1024)
        q4k_wide_block(w + ((offset + first) / 256) * 144, x + first, lane & 7, parts);
    return warp_sum(fold_parts(parts));
}

// Vectorized Q4_K multi-vector product (1 <= tokens <= 8):
// 4 warps per output row (2 rows/block, grid = ceil(rows / 2)), reads each 144-byte
// Q4_K superblock ONCE across all `tokens <= 8` input vectors with 4x K-parallelism
// and zero atomics/memsets.
extern "C" __global__ void matvec_q4k_multi8(const u8 *w, const float *x, float *y,
                                             int width, int rows, int tokens) {
    __shared__ float smem[2][4][8];
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int local_row = warp >> 2;
    int part = warp & 3;
    int row = blockIdx.x * 2 + local_row;
    if (row >= rows) return;

    u64 offset = (u64)row * width;
    int sub = lane & 7;
    int pair = sub >> 1;

    float acc[8];
    #pragma unroll
    for (int t = 0; t < 8; t++) acc[t] = 0.0f;

    for (int first = (part * 4 + (lane >> 3)) * 256; first < width; first += 4096) {
        const u8 *p = w + ((offset + first) / 256) * 144;
        uint4 qv = *(const uint4 *)(p + 16 + sub * 16);
        unsigned int dd = *(const unsigned int *)p;
        float d = half_value_bits(dd & 0xffff), dm = half_value_bits(dd >> 16);
        const unsigned int *sw = (const unsigned int *)(p + 4);
        unsigned int s0 = sw[0], s1 = sw[1], s2 = sw[2];
        unsigned int qw[4] = {qv.x, qv.y, qv.z, qv.w};

        #pragma unroll
        for (int half = 0; half < 2; half++) {
            int group = 2 * pair + half, shift = 8 * (group & 3);
            unsigned int lo = (s0 >> shift) & 255, mid = (s1 >> shift) & 255;
            unsigned int hi = (s2 >> shift) & 255;
            int scale = group < 4 ? lo & 63 : (hi & 15) | ((lo >> 6) << 4);
            int minimum = group < 4 ? mid & 63 : (hi >> 4) | ((mid >> 6) << 4);
            float ds = d * (float)scale, ms = dm * (float)minimum;
            int xbase = first + 32 * group + (sub & 1) * 16;

            #pragma unroll
            for (int word = 0; word < 4; word++) {
                float wd[4];
                #pragma unroll
                for (int b = 0; b < 4; b++) {
                    wd[b] = ds * small_uint_float((qw[word] >> (8 * b + 4 * half)) & 15) - ms;
                }
                int xoff = xbase + 4 * word;
                #pragma unroll
                for (int t = 0; t < 8; t++) {
                    if (t < tokens) {
                        float4 xv = __ldg((const float4 *)(x + (u64)t * width + xoff));
                        acc[t] = fmaf(wd[0], xv.x, acc[t]);
                        acc[t] = fmaf(wd[1], xv.y, acc[t]);
                        acc[t] = fmaf(wd[2], xv.z, acc[t]);
                        acc[t] = fmaf(wd[3], xv.w, acc[t]);
                    }
                }
            }
        }
    }

    #pragma unroll
    for (int t = 0; t < 8; t++) {
        if (t < tokens) {
            float sum = warp_sum(acc[t]);
            if (lane == 0) smem[local_row][part][t] = sum;
        }
    }
    __syncthreads();
    if (part == 0 && lane < tokens) {
        float total = smem[local_row][0][lane] + smem[local_row][1][lane]
                    + smem[local_row][2][lane] + smem[local_row][3][lane];
        y[(u64)lane * rows + row] = total;
    }
}

// Q5_K with 8 lanes per 256-value block: same layout as q4k_wide_block plus the
// 32-byte high-bit array `qh` at p + 16.
__device__ __forceinline__ void q5k_wide_block(const u8 *p, const float *x, int sub,
                                               float *parts) {
    uint4 qv = *(const uint4 *)(p + 48 + sub * 16);
    uint4 qhv = *(const uint4 *)(p + 16 + (sub & 1) * 16);
    unsigned int dd = *(const unsigned int *)p;
    float d = half_value_bits(dd & 0xffff), dm = half_value_bits(dd >> 16);
    const unsigned int *sw = (const unsigned int *)(p + 4);
    unsigned int s0 = sw[0], s1 = sw[1], s2 = sw[2];
    unsigned int qw[4] = {qv.x, qv.y, qv.z, qv.w};
    unsigned int qhw[4] = {qhv.x, qhv.y, qhv.z, qhv.w};
    int pair = sub >> 1;
    #pragma unroll
    for (int half = 0; half < 2; half++) {
        int group = 2 * pair + half, shift = 8 * (group & 3);
        unsigned int lo = (s0 >> shift) & 255, mid = (s1 >> shift) & 255;
        unsigned int hi = (s2 >> shift) & 255;
        int scale = group < 4 ? lo & 63 : (hi & 15) | ((lo >> 6) << 4);
        int minimum = group < 4 ? mid & 63 : (hi >> 4) | ((mid >> 6) << 4);
        float ds = d * (float)scale, ms = dm * (float)minimum;
        const float *xb = x + 32 * group + (sub & 1) * 16;
        #pragma unroll
        for (int word = 0; word < 4; word++) {
            float4 xv = *(const float4 *)(xb + 4 * word);
            float xs[4] = {xv.x, xv.y, xv.z, xv.w};
            #pragma unroll
            for (int b = 0; b < 4; b++) {
                unsigned int q = ((qw[word] >> (8 * b + 4 * half)) & 15)
                    | ((((qhw[word] >> (8 * b)) >> group) & 1) << 4);
                float decoded = ds * small_uint_float(q) - ms;
                parts[half * 4 + b] += decoded * xs[b];
            }
        }
    }
}

__device__ __forceinline__ float q5k_wide_row_part(const u8 *w, const float *x, u64 offset,
                                                   int width, int lane, int part, int warps) {
    float parts[8] = {};
    for (int first = (part * 4 + (lane >> 3)) * 256; first < width; first += warps * 1024)
        q5k_wide_block(w + ((offset + first) / 256) * 176, x + first, lane & 7, parts);
    return warp_sum(fold_parts(parts));
}

// Q6_K with 16 lanes per 256-value block (4 values/lane/quarter, float4 loads on x),
// so a warp covers 2 blocks (512 elements) per step.
__device__ __forceinline__ void q6k_wide_block(const u8 *p, const float *x, int sub,
                                               float *parts) {
    float d = half_value(p + 208);
    int h = sub >> 3, col = (sub & 7) * 4;
    const unsigned short *p_lo0 = (const unsigned short *)(p + h * 64 + col);
    const unsigned short *p_lo1 = (const unsigned short *)(p + h * 64 + 32 + col);
    const unsigned short *p_hi = (const unsigned short *)(p + 128 + h * 32 + col);
    unsigned int lo0 = (unsigned int)p_lo0[0] | ((unsigned int)p_lo0[1] << 16);
    unsigned int lo1 = (unsigned int)p_lo1[0] | ((unsigned int)p_lo1[1] << 16);
    unsigned int hi = (unsigned int)p_hi[0] | ((unsigned int)p_hi[1] << 16);
    #pragma unroll
    for (int quarter = 0; quarter < 4; quarter++) {
        unsigned int low = (quarter & 1) ? lo1 : lo0;
        int group = h * 8 + quarter * 2 + (col >> 4);
        float ds = d * (float)((const signed char *)p)[192 + group];
        float4 xv = *(const float4 *)(x + h * 128 + quarter * 32 + col);
        float xs[4] = {xv.x, xv.y, xv.z, xv.w};
        int lshift = (quarter >> 1) * 4;
        int hshift = quarter * 2;
        #pragma unroll
        for (int b = 0; b < 4; b++) {
            int ql = (low >> (8 * b + lshift)) & 15;
            int qh = (hi >> (8 * b + hshift)) & 3;
            float decoded = ds * (small_uint_float(ql | (qh << 4)) - 32.0f);
            parts[(quarter & 1) * 4 + b] += decoded * xs[b];
        }
    }
}

__device__ __forceinline__ float q6k_wide_row_part(const u8 *w, const float *x, u64 offset,
                                                   int width, int lane, int part, int warps) {
    float parts[8] = {};
    for (int first = (part * 2 + (lane >> 4)) * 256; first < width; first += warps * 512)
        q6k_wide_block(w + ((offset + first) / 256) * 210, x + first, lane & 15, parts);
    return warp_sum(fold_parts(parts));
}

// Vectorized Q6_K multi-vector product for speculative LM head (1 <= tokens <= 8):
// 1 warp per output row (8 rows/block), reads each 210-byte Q6_K superblock ONCE
// and multiplies by all `tokens` input vectors in registers via L1-cached __ldg.
extern "C" __global__ void matvec_q6k_multi8(const u8 *w, const float *x, float *y,
                                             int width, int rows, int tokens) {
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int row = blockIdx.x * 8 + warp;
    if (row >= rows) return;

    u64 offset = (u64)row * width;
    int sub = lane & 15;
    int h = sub >> 3;
    int col = (sub & 7) * 4;

    float acc[8];
    #pragma unroll
    for (int t = 0; t < 8; t++) acc[t] = 0.0f;

    for (int first = (lane >> 4) * 256; first < width; first += 512) {
        const u8 *p = w + ((offset + first) / 256) * 210;
        float d = half_value(p + 208);
        const unsigned short *p_lo0 = (const unsigned short *)(p + h * 64 + col);
        const unsigned short *p_lo1 = (const unsigned short *)(p + h * 64 + 32 + col);
        const unsigned short *p_hi = (const unsigned short *)(p + 128 + h * 32 + col);
        unsigned int lo0 = (unsigned int)p_lo0[0] | ((unsigned int)p_lo0[1] << 16);
        unsigned int lo1 = (unsigned int)p_lo1[0] | ((unsigned int)p_lo1[1] << 16);
        unsigned int hi = (unsigned int)p_hi[0] | ((unsigned int)p_hi[1] << 16);

        #pragma unroll
        for (int quarter = 0; quarter < 4; quarter++) {
            unsigned int low = (quarter & 1) ? lo1 : lo0;
            int group = h * 8 + quarter * 2 + (col >> 4);
            float ds = d * (float)((const signed char *)p)[192 + group];
            int lshift = (quarter >> 1) * 4;
            int hshift = quarter * 2;
            float wd[4];
            #pragma unroll
            for (int b = 0; b < 4; b++) {
                int ql = (low >> (8 * b + lshift)) & 15;
                int qh = (hi >> (8 * b + hshift)) & 3;
                wd[b] = ds * (small_uint_float(ql | (qh << 4)) - 32.0f);
            }
            int xoff = first + h * 128 + quarter * 32 + col;
            #pragma unroll
            for (int t = 0; t < 8; t++) {
                if (t < tokens) {
                    float4 xv = __ldg((const float4 *)(x + (u64)t * width + xoff));
                    acc[t] = fmaf(wd[0], xv.x, acc[t]);
                    acc[t] = fmaf(wd[1], xv.y, acc[t]);
                    acc[t] = fmaf(wd[2], xv.z, acc[t]);
                    acc[t] = fmaf(wd[3], xv.w, acc[t]);
                }
            }
        }
    }

    #pragma unroll
    for (int t = 0; t < 8; t++) {
        if (t < tokens) {
            float sum = warp_sum(acc[t]);
            if (lane == 0) y[(u64)t * rows + row] = sum;
        }
    }
}

// Vectorized Q6_K multi-vector product for tree verification (1 <= tokens <= 16):
// 1 warp per output row (8 rows/block), reads each 210-byte Q6_K superblock ONCE
// and multiplies by all `tokens <= 16` input vectors in registers.
extern "C" __global__ void matvec_q6k_multi16(const u8 *w, const float *x, float *y,
                                              int width, int rows, int tokens) {
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int row = blockIdx.x * 8 + warp;
    if (row >= rows) return;

    u64 offset = (u64)row * width;
    int sub = lane & 15;
    int h = sub >> 3;
    int col = (sub & 7) * 4;

    float acc[16];
    #pragma unroll
    for (int t = 0; t < 16; t++) acc[t] = 0.0f;

    for (int first = (lane >> 4) * 256; first < width; first += 512) {
        const u8 *p = w + ((offset + first) / 256) * 210;
        float d = half_value(p + 208);
        const unsigned short *p_lo0 = (const unsigned short *)(p + h * 64 + col);
        const unsigned short *p_lo1 = (const unsigned short *)(p + h * 64 + 32 + col);
        const unsigned short *p_hi = (const unsigned short *)(p + 128 + h * 32 + col);
        unsigned int lo0 = (unsigned int)p_lo0[0] | ((unsigned int)p_lo0[1] << 16);
        unsigned int lo1 = (unsigned int)p_lo1[0] | ((unsigned int)p_lo1[1] << 16);
        unsigned int hi = (unsigned int)p_hi[0] | ((unsigned int)p_hi[1] << 16);

        #pragma unroll
        for (int quarter = 0; quarter < 4; quarter++) {
            unsigned int low = (quarter & 1) ? lo1 : lo0;
            int group = h * 8 + quarter * 2 + (col >> 4);
            float ds = d * (float)((const signed char *)p)[192 + group];
            int lshift = (quarter >> 1) * 4;
            int hshift = quarter * 2;
            float wd[4];
            #pragma unroll
            for (int b = 0; b < 4; b++) {
                int ql = (low >> (8 * b + lshift)) & 15;
                int qh = (hi >> (8 * b + hshift)) & 3;
                wd[b] = ds * (small_uint_float(ql | (qh << 4)) - 32.0f);
            }
            int xoff = first + h * 128 + quarter * 32 + col;
            #pragma unroll
            for (int t = 0; t < 16; t++) {
                if (t < tokens) {
                    float4 xv = __ldg((const float4 *)(x + (u64)t * width + xoff));
                    acc[t] = fmaf(wd[0], xv.x, acc[t]);
                    acc[t] = fmaf(wd[1], xv.y, acc[t]);
                    acc[t] = fmaf(wd[2], xv.z, acc[t]);
                    acc[t] = fmaf(wd[3], xv.w, acc[t]);
                }
            }
        }
    }

    #pragma unroll
    for (int t = 0; t < 16; t++) {
        if (t < tokens) {
            float sum = warp_sum(acc[t]);
            if (lane == 0) y[(u64)t * rows + row] = sum;
        }
    }
}

template<int Kind>
__device__ __forceinline__ float packed_matrix_lane(const u8 *w, const float *x,
                                                    u64 offset, int width, int tid) {
    return packed_lane_span<Kind>(w, x, offset, width, tid & 31, (tid / 32) * 256, 2048);
}

template<int Kind>
__device__ __forceinline__ float warp_scalar_row(const u8 *w, const float *x, u64 offset,
                                                 int width, int lane, int part = 0,
                                                 int warps = 1) {
    float sum = 0.0f;
    for (int i = part * 32 + lane; i < width; i += warps * 32)
        sum += weight(w, Kind, offset + i) * x[i];
    return warp_sum(sum);
}

// Warps cooperating on one output row. Narrow-row shapes keep one warp per row
// (no barrier); few-row shapes split a row's 256-value blocks across up to 8
// warps so the grid still fills the GPU. Mirrors `matvec_warps_per_row` in
// cuda.rs, which sizes the grid: grid=(ceil(rows * warps / 8), selections, 1).
// Block-aligned Q4_K rows use the wide layout, where one warp step covers 4 blocks.
__device__ __forceinline__ bool q4k_wide(int kind, int width) {
    return (kind == 12 || kind == 13) && width % 256 == 0;
}

__device__ __forceinline__ int matvec_warps_per_row(int kind, int rows, int selections,
                                                    int width) {
    int blocks = width / 256 > 0 ? width / 256 : 1, warps = 1;
    if (q4k_wide(kind, width)) {
        int target = blocks / 8 > 8 ? 8 : (blocks / 8 > 1 ? blocks / 8 : 1);
        warps = 1;
        while (warps * 2 <= target) warps *= 2;
        while (warps < 8 && warps * 4 < blocks && rows * selections * warps < 2048) warps *= 2;
        return warps;
    }
    while (warps < 8 && warps < blocks && rows * selections * warps < 16384) warps *= 2;
    return warps;
}

// Q8_0 row part: 8 lanes per 32-element block (4 values/lane), so one warp
// processes 4 blocks (128 elements) per step with float4 loads on x.
__device__ __forceinline__ float q8_0_row_part(const u8 *w, const float *x, u64 offset,
                                               int width, int lane, int part, int warps) {
    int num_blocks = width / 32;
    const u8 *row_ptr = w + (offset / 32) * 34;
    int sub = lane >> 3;
    int elem = (lane & 7) * 4;
    float acc = 0.0f;
    for (int b = part * 4 + sub; b < num_blocks; b += warps * 4) {
        const u8 *p = row_ptr + (u64)b * 34;
        float d = half_value(p);
        unsigned short lo = *(const unsigned short *)(p + 2 + elem);
        unsigned short hi = *(const unsigned short *)(p + 4 + elem);
        float4 xv = *(const float4 *)(x + b * 32 + elem);
        float q0 = (float)(signed char)(lo & 0xff);
        float q1 = (float)(signed char)(lo >> 8);
        float q2 = (float)(signed char)(hi & 0xff);
        float q3 = (float)(signed char)(hi >> 8);
        acc += d * (q0 * xv.x + q1 * xv.y + q2 * xv.z + q3 * xv.w);
    }
    return warp_sum(acc);
}

// F16 row part: each lane converts 8 consecutive halves from one 16-byte load
// (exact in F32), so a warp step covers 256 values. width % 256 == 0.
__device__ __forceinline__ float f16_wide_row_part(const u8 *w, const float *x, u64 offset,
                                                   int width, int lane, int part, int warps) {
    const uint4 *row = (const uint4 *)(w + offset * 2);
    float acc = 0.0f;
    for (int i = part * 256 + lane * 8; i < width; i += warps * 256) {
        uint4 packed = __ldg(row + i / 8);
        float4 xa = *(const float4 *)(x + i);
        float4 xb = *(const float4 *)(x + i + 4);
        unsigned int words[4] = {packed.x, packed.y, packed.z, packed.w};
        float xs[8] = {xa.x, xa.y, xa.z, xa.w, xb.x, xb.y, xb.z, xb.w};
#pragma unroll
        for (int j = 0; j < 4; j++) {
            acc += half_value_bits(words[j] & 0xffff) * xs[2 * j];
            acc += half_value_bits(words[j] >> 16) * xs[2 * j + 1];
        }
    }
    return warp_sum(acc);
}

// F32 row part: float4 vectorized loads (16 bytes/lane = 128 floats/warp step). width % 128 == 0.
__device__ __forceinline__ float f32_wide_row_part(const u8 *w, const float *x, u64 offset,
                                                   int width, int lane, int part, int warps) {
    const float4 *row = (const float4 *)((const float *)w + offset);
    const float4 *xv = (const float4 *)x;
    float acc = 0.0f;
    int n4 = width >> 2;
    for (int i = part * 32 + lane; i < n4; i += warps * 32) {
        float4 wv = __ldg(row + i);
        float4 iv = __ldg(xv + i);
        acc = fmaf(wv.x, iv.x, acc);
        acc = fmaf(wv.y, iv.y, acc);
        acc = fmaf(wv.z, iv.z, acc);
        acc = fmaf(wv.w, iv.w, acc);
    }
    return warp_sum(acc);
}

// block=(256, 1, 1). F32 arithmetic; only the reduction order differs from matvec.
extern "C" __global__ void matvec_packed_rows(const u8 *w, int kind, const float *x, float *y,
                                             int width, int rows, const int *ids,
                                             int moe, int batched) {
    __shared__ float partial[8];
    int lane = threadIdx.x & 31, warp = threadIdx.x / 32, selected = blockIdx.y;
    int warps = matvec_warps_per_row(kind, rows, gridDim.y, width);
    int part = warp % warps, row = blockIdx.x * (8 / warps) + warp / warps;
    float sum = 0.0f;
    if (row < rows) {
        int expert = moe ? ids[selected] : 0;
        u64 offset = ((u64)expert * rows + row) * width;
        const float *input = x + (batched ? (u64)selected * width : 0);
        if (kind == 12 && width % 256 == 0) {
            sum = q4k_wide_row_part(w, input, offset, width, lane, part, warps);
        } else if (kind == 13 && width % 256 == 0) {
            sum = q5k_wide_row_part(w, input, offset, width, lane, part, warps);
        } else if (kind == 14 && width % 256 == 0) {
            sum = q6k_wide_row_part(w, input, offset, width, lane, part, warps);
        } else if (kind == 8 && width % 32 == 0) {
            sum = q8_0_row_part(w, input, offset, width, lane, part, warps);
        } else if (kind == 1 && width % 256 == 0) {
            sum = f16_wide_row_part(w, input, offset, width, lane, part, warps);
        } else if (kind == 0 && width % 128 == 0) {
            sum = f32_wide_row_part(w, input, offset, width, lane, part, warps);
        } else {
            switch (kind) {
#define SCALAR_ROW(K) warp_scalar_row<K>(w, input, offset, width, lane, part, warps)
                case 0: sum = SCALAR_ROW(0); break;
                case 1: sum = SCALAR_ROW(1); break;
                case 6: sum = SCALAR_ROW(6); break;
                case 8: sum = SCALAR_ROW(8); break;
                case 12: sum = SCALAR_ROW(12); break;
                case 13: sum = SCALAR_ROW(13); break;
                case 14: sum = SCALAR_ROW(14); break;
#undef SCALAR_ROW
                default: sum = __int_as_float(0x7fffffff); break;
            }
        }
    }
    if (warps > 1) {
        if (lane == 0) partial[warp] = sum;
        __syncthreads();
        if (part != 0) return;
        for (int i = 1; i < warps; i++) sum += partial[warp + i];
    }
    if (lane == 0 && row < rows) y[(u64)selected * rows + row] = sum;
}

// Drop-in RMSNorm (grid=rows, block=32..1024 threads, multiple of 32): warp shuffles
// plus one barrier instead of the shared-memory tree. F32; only the reduction order differs.
extern "C" __global__ void normalize_warp(const float *x, const u8 *w, float *y, int n,
                                          float epsilon) {
    __shared__ float partial[32];
    x += (u64)blockIdx.x * n;
    y += (u64)blockIdx.x * n;
    float sum = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) sum += x[i] * x[i];
    sum = warp_sum(sum);
    if ((threadIdx.x & 31) == 0) partial[threadIdx.x / 32] = sum;
    __syncthreads();
    float total = 0.0f;
    for (int i = 0; i < (int)blockDim.x / 32; i++) total += partial[i];
    float inverse = 1.0f / sqrtf(total / (float)n + epsilon);
    for (int i = threadIdx.x; i < n; i += blockDim.x) y[i] = x[i] * inverse * ((const float *)w)[i];
}

// Packs 3 tapped hidden states [t0, t1, t2] of shape [n, width] into [n, 3 * width] in 1 launch.
// Grid: (ceil(width / 256), n, 1), Block: 256 threads.
extern "C" __global__ void eagle3_concat3(const float *t0, const float *t1, const float *t2,
                                          float *out, int width) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= width) return;
    int tok = blockIdx.y;
    u64 src = (u64)tok * width + i;
    u64 dst = (u64)tok * (3 * width) + i;
    out[dst] = t0[src];
    out[dst + width] = t1[src];
    out[dst + 2 * width] = t2[src];
}

// Fuses input_layernorm(emb[tok]) and hidden_norm(g[tok]) directly into
// inp_cat[tok * 2 * width .. (tok + 1) * 2 * width] in 1 launch (grid = (n, 2, 1)).
extern "C" __global__ void eagle3_norm_cat(const float *emb, const float *g,
                                           const u8 *w_emb, const u8 *w_g,
                                           float *inp_cat, int width, float epsilon) {
    __shared__ float partial[32];
    int tok = blockIdx.x;
    int which = blockIdx.y;
    const float *x = (which == 0 ? emb : g) + (u64)tok * width;
    const float *w = (const float *)(which == 0 ? w_emb : w_g);
    float *y = inp_cat + (u64)tok * (2 * width) + (u64)which * width;
    float sum = 0.0f;
    for (int i = threadIdx.x; i < width; i += blockDim.x) sum += x[i] * x[i];
    sum = warp_sum(sum);
    if ((threadIdx.x & 31) == 0) partial[threadIdx.x / 32] = sum;
    __syncthreads();
    float total = 0.0f;
    for (int i = 0; i < (int)blockDim.x / 32; i++) total += partial[i];
    float inverse = rsqrtf(total / (float)width + epsilon);
    for (int i = threadIdx.x; i < width; i += blockDim.x) y[i] = x[i] * inverse * w[i];
}

// Fused residual add + unit-variance RMSNorm for PLT1 folded-RMSNorm layers:
//   x[tok, i] += delta[tok, i]
//   norm_out[tok, i] = x[tok, i] * rsqrt(mean(x[tok]^2) + eps)
// Grid: (n, 1, 1), Block: up to 1024 threads. Uses float4 + register caching when aligned.
extern "C" __global__ void residual_unit_norm(float *x, const float *delta, float *norm_out,
                                              int width, float epsilon) {
    __shared__ float partial[32];
    int tok = blockIdx.x;
    float *xt = x + (u64)tok * width;
    const float *dt = delta + (u64)tok * width;
    float *yt = norm_out + (u64)tok * width;
    int tid = threadIdx.x;
    int warp = tid >> 5;
    int lane = tid & 31;
    int nwarps = blockDim.x >> 5;

    if ((width & 3) == 0 && width <= (int)blockDim.x * 4) {
        int n4 = width >> 2;
        float4 v = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        float sum = 0.0f;
        if (tid < n4) {
            float4 xv = reinterpret_cast<const float4 *>(xt)[tid];
            float4 dv = reinterpret_cast<const float4 *>(dt)[tid];
            v = make_float4(xv.x + dv.x, xv.y + dv.y, xv.z + dv.z, xv.w + dv.w);
            reinterpret_cast<float4 *>(xt)[tid] = v;
            sum = v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
        }
        sum = warp_sum(sum);
        if (lane == 0) partial[warp] = sum;
        __syncthreads();
        float total = (lane < nwarps) ? partial[lane] : 0.0f;
        total = warp_sum(total);
        float inverse = rsqrtf(total / (float)width + epsilon);
        if (tid < n4) {
            reinterpret_cast<float4 *>(yt)[tid] =
                make_float4(v.x * inverse, v.y * inverse, v.z * inverse, v.w * inverse);
        }
        return;
    }

    float sum = 0.0f;
    for (int i = tid; i < width; i += blockDim.x) {
        float v = xt[i] + dt[i];
        xt[i] = v;
        sum += v * v;
    }
    sum = warp_sum(sum);
    if (lane == 0) partial[warp] = sum;
    __syncthreads();
    float total = (lane < nwarps) ? partial[lane] : 0.0f;
    total = warp_sum(total);
    float inverse = rsqrtf(total / (float)width + epsilon);
    for (int i = tid; i < width; i += blockDim.x) {
        yt[i] = xt[i] * inverse;
    }
}

// Fused MoE combine + unit-variance RMSNorm for PLT1 folded-RMSNorm MoE layers.
// Grid: (1, 1, 1), Block: up to 1024 threads; float4 vectorized when aligned.
extern "C" __global__ void combine_unit_norm(float *x, const float *experts,
                                             const float *weights, float *norm_out,
                                             int width, int used, float epsilon) {
    __shared__ float partial[32];
    int tid = threadIdx.x;
    int warp = tid >> 5;
    int lane = tid & 31;
    int nwarps = blockDim.x >> 5;

    if ((width & 3) == 0 && width <= (int)blockDim.x * 4) {
        int n4 = width >> 2;
        float4 v = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        float sum = 0.0f;
        if (tid < n4) {
            float4 xv = reinterpret_cast<const float4 *>(x)[tid];
            float c0 = 0.0f, c1 = 0.0f, c2 = 0.0f, c3 = 0.0f;
            const float4 *e4 = reinterpret_cast<const float4 *>(experts);
            for (int s = 0; s < used; s++) {
                float w = __ldg(weights + s);
                float4 ev = __ldg(e4 + (u64)s * n4 + tid);
                c0 = fmaf(w, ev.x, c0);
                c1 = fmaf(w, ev.y, c1);
                c2 = fmaf(w, ev.z, c2);
                c3 = fmaf(w, ev.w, c3);
            }
            v = make_float4(xv.x + c0, xv.y + c1, xv.z + c2, xv.w + c3);
            reinterpret_cast<float4 *>(x)[tid] = v;
            sum = v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
        }
        sum = warp_sum(sum);
        if (lane == 0) partial[warp] = sum;
        __syncthreads();
        float total = (lane < nwarps) ? partial[lane] : 0.0f;
        total = warp_sum(total);
        float inverse = rsqrtf(total / (float)width + epsilon);
        if (tid < n4) {
            reinterpret_cast<float4 *>(norm_out)[tid] =
                make_float4(v.x * inverse, v.y * inverse, v.z * inverse, v.w * inverse);
        }
        return;
    }

    float sum = 0.0f;
    for (int i = tid; i < width; i += blockDim.x) {
        float c = 0.0f;
        for (int s = 0; s < used; s++) c += weights[s] * experts[(u64)s * width + i];
        float v = x[i] + c;
        x[i] = v;
        sum += v * v;
    }
    sum = warp_sum(sum);
    if (lane == 0) partial[warp] = sum;
    __syncthreads();
    float total = (lane < nwarps) ? partial[lane] : 0.0f;
    total = warp_sum(total);
    float inverse = rsqrtf(total / (float)width + epsilon);
    for (int i = tid; i < width; i += blockDim.x) {
        norm_out[i] = x[i] * inverse;
    }
}

template<int Kind>
__device__ __forceinline__ float packed_row_part(const u8 *w, const float *x, u64 offset,
                                                 int width, int lane, int part, int warps) {
    if (Kind == 12) return q4k_wide_row_part(w, x, offset, width, lane, part, warps);
    if (Kind == 14) return q6k_wide_row_part(w, x, offset, width, lane, part, warps);
    return warp_sum(packed_lane_span<Kind>(w, x, offset, width, lane, part * 256, warps * 256));
}

// Q, K and V projections of one input in a single launch: rows [0, q) go to yq,
// [q, q + k) to yk and the rest to yv. All three must be block-aligned packed
// Q4_K/Q6_K with the same width. Grid as matvec_packed_rows over q + k + v rows.
extern "C" __global__ void matvec_packed_qkv_rows(
    const u8 *wq, int q_kind, int q_rows, float *yq,
    const u8 *wk, int k_kind, int k_rows, float *yk,
    const u8 *wv, int v_kind, int v_rows, float *yv,
    const float *x, int width) {
    __shared__ float partial[8];
    int total = q_rows + k_rows + v_rows;
    int lane = threadIdx.x & 31, warp = threadIdx.x / 32;
    int warps = matvec_warps_per_row(q_kind, total, 1, width);
    int part = warp % warps, row = blockIdx.x * (8 / warps) + warp / warps;
    const u8 *w = wq;
    int kind = q_kind, local = row;
    float *y = yq;
    if (row >= q_rows + k_rows) {
        w = wv; kind = v_kind; y = yv; local = row - q_rows - k_rows;
    } else if (row >= q_rows) {
        w = wk; kind = k_kind; y = yk; local = row - q_rows;
    }
    float sum = 0.0f;
    if (row < total) {
        u64 offset = (u64)local * width;
        sum = kind == 12 ? packed_row_part<12>(w, x, offset, width, lane, part, warps)
            : kind == 13 ? q5k_wide_row_part(w, x, offset, width, lane, part, warps)
                         : packed_row_part<14>(w, x, offset, width, lane, part, warps);
    }
    if (warps > 1) {
        if (lane == 0) partial[warp] = sum;
        __syncthreads();
        if (part != 0) return;
        for (int i = 1; i < warps; i++) sum += partial[warp + i];
    }
    if (lane == 0 && row < total) y[local] = sum;
}

// Fused gate/up projections plus SwiGLU for packed Q4_K/Q5_K/Q6_K dense or MoE
// weights with block-aligned rows. Same grid as matvec_packed_rows; reads x once
// per row pair and drops the separate swiglu launch. y = silu(gate . x) * (up . x).
extern "C" __global__ void matvec_packed_gate_up_rows(
    const u8 *gate, int gate_kind, const u8 *up, int up_kind, const float *x, float *y,
    int width, int rows, const int *ids, int moe) {
    __shared__ float partial[2][8];
    int lane = threadIdx.x & 31, warp = threadIdx.x / 32, selected = blockIdx.y;
    int warps = matvec_warps_per_row(gate_kind, rows, gridDim.y, width);
    int part = warp % warps, row = blockIdx.x * (8 / warps) + warp / warps;
    float g = 0.0f, u = 0.0f;
    if (row < rows) {
        int expert = moe ? ids[selected] : 0;
        u64 offset = ((u64)expert * rows + row) * width;
        g = gate_kind == 12 ? packed_row_part<12>(gate, x, offset, width, lane, part, warps)
          : gate_kind == 13 ? q5k_wide_row_part(gate, x, offset, width, lane, part, warps)
                            : packed_row_part<14>(gate, x, offset, width, lane, part, warps);
        u = up_kind == 12 ? packed_row_part<12>(up, x, offset, width, lane, part, warps)
          : up_kind == 13 ? q5k_wide_row_part(up, x, offset, width, lane, part, warps)
                          : packed_row_part<14>(up, x, offset, width, lane, part, warps);
    }
    if (warps > 1) {
        if (lane == 0) {
            partial[0][warp] = g;
            partial[1][warp] = u;
        }
        __syncthreads();
        if (part != 0) return;
        for (int i = 1; i < warps; i++) {
            g += partial[0][warp + i];
            u += partial[1][warp + i];
        }
    }
    if (lane == 0 && row < rows) y[(u64)selected * rows + row] = g / (1.0f + expf(-g)) * u;
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

#if defined(CUDA_M8_PROTOTYPES) || defined(CUDA_DECODE_FUSION_RESEARCH)
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
    float sum = kind == 12 ? packed_matrix_row<12>(w, x, offset, width)
                           : packed_matrix_row<14>(w, x, offset, width);
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

template<bool Partial = false>
__device__ __forceinline__ void rotary_impl(float *x, int n, int dim, int position, float base, int neox,
                                           const float *factors, float magnitude, int rotary_dim = 0) {
    x += (u64)blockIdx.y * n;
    position += blockIdx.y;
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    int rotated = Partial ? rotary_dim : dim;
    if (p >= (Partial ? n / dim * (rotated / 2) : n / 2)) return;
    int pair = p % (rotated / 2), head = p / (rotated / 2);
    int a = head * dim + (neox ? pair : 2 * pair);
    int b = neox ? a + rotated / 2 : a + 1;
    float angle = (float)position * powf(base, -(float)(2 * pair) / (float)rotated);
    if (factors) angle /= factors[pair];
    float sine = sinf(angle) * magnitude, cosine = cosf(angle) * magnitude;
    float u = x[a], v = x[b];
    x[a] = u * cosine - v * sine;
    x[b] = u * sine + v * cosine;
}

extern "C" __global__ void rotary(float *x, int n, int dim, int position, float base, int neox) {
    rotary_impl(x, n, dim, position, base, neox, nullptr, 1.0f);
}

extern "C" __global__ void rotary_partial_neox(float *x, int n, int dim, int position,
                                               float base, int rotary_dim) {
    rotary_impl<true>(x, n, dim, position, base, 1, nullptr, 1.0f, rotary_dim);
}

extern "C" __global__ void rotary_factored(float *x, int n, int dim, int position, float base,
                                           int neox, const float *factors, float magnitude) {
    rotary_impl(x, n, dim, position, base, neox, factors, magnitude);
}

extern "C" __global__ void rotary_tree(float *x, int n, int dim, const int *positions,
                                       float base, int neox) {
    int pos = positions[blockIdx.y];
    rotary_impl(x, n, dim, pos - (int)blockIdx.y, base, neox, nullptr, 1.0f);
}

extern "C" __global__ void rotary_factored_tree(float *x, int n, int dim, const int *positions,
                                                float base, int neox, const float *factors,
                                                float magnitude) {
    int pos = positions[blockIdx.y];
    rotary_impl(x, n, dim, pos - (int)blockIdx.y, base, neox, factors, magnitude);
}

extern "C" __global__ void rotary_decode(float *x, int n, int dim, const int *control,
                                         float base, int neox) {
    rotary_impl(x, n, dim, control[1], base, neox, nullptr, 1.0f);
}

extern "C" __global__ void rotary_factored_decode(float *x, int n, int dim, const int *control,
                                                  float base, int neox, const float *factors,
                                                  float magnitude) {
    rotary_impl(x, n, dim, control[1], base, neox, factors, magnitude);
}

// Fused factored RoPE (Q and K) + KV cache store for decode graphs (e.g. Phi-3-128k, Llama 3.2).
// Grid: ceil((q_size/2 + k_size) / 256) blocks, 256 threads.
template<typename KV>
__device__ __forceinline__ void rotary_factored_cache_kv_impl(
    float *q, int q_size, float *k, const float *v, KV *keys, KV *values, int k_size, int dim,
    int position, float base, int neox, const float *factors, float magnitude) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int q_pairs = q_size / 2;
    int k_pairs = k_size / 2;
    if (tid < q_pairs) {
        int pair = tid % (dim / 2), head = tid / (dim / 2);
        int a = head * dim + (neox ? pair : 2 * pair);
        int b = neox ? a + dim / 2 : a + 1;
        float angle = (float)position * powf(base, -(float)(2 * pair) / (float)dim);
        if (factors) angle /= factors[pair];
        float sine = sinf(angle) * magnitude, cosine = cosf(angle) * magnitude;
        float u = q[a], w = q[b];
        q[a] = u * cosine - w * sine;
        q[b] = u * sine + w * cosine;
    } else if (tid < q_pairs + k_pairs) {
        int p = tid - q_pairs;
        int pair = p % (dim / 2), head = p / (dim / 2);
        int a = head * dim + (neox ? pair : 2 * pair);
        int b = neox ? a + dim / 2 : a + 1;
        float angle = (float)position * powf(base, -(float)(2 * pair) / (float)dim);
        if (factors) angle /= factors[pair];
        float sine = sinf(angle) * magnitude, cosine = cosf(angle) * magnitude;
        float u = k[a], w = k[b];
        float ra = u * cosine - w * sine;
        float rb = u * sine + w * cosine;
        k[a] = ra;
        k[b] = rb;
        u64 base_kv = (u64)position * k_size;
        kv_store(keys, base_kv + a, ra);
        kv_store(keys, base_kv + b, rb);
        kv_store(values, base_kv + a, v[a]);
        kv_store(values, base_kv + b, v[b]);
    }
}

extern "C" __global__ void rotary_factored_cache_kv_decode(
    float *q, int q_size, float *k, const float *v, float *keys, float *values, int k_size,
    int dim, const int *control, float base, int neox, const float *factors, float magnitude) {
    rotary_factored_cache_kv_impl(q, q_size, k, v, keys, values, k_size, dim, control[1], base,
                                  neox, factors, magnitude);
}

extern "C" __global__ void rotary_factored_cache_kv_batch(
    float *q, int q_size, float *k, const float *v, float *keys, float *values, int k_size,
    int dim, int position, float base, int neox, const float *factors, float magnitude) {
    int tok = blockIdx.y;
    rotary_factored_cache_kv_impl(q + (u64)tok * q_size, q_size, k + (u64)tok * k_size,
                                  v + (u64)tok * k_size, keys, values, k_size, dim,
                                  position + tok, base, neox, factors, magnitude);
}

extern "C" __global__ void rotary_factored_cache_kv_batch_graph(
    float *q, int q_size, float *k, const float *v, float *keys, float *values, int k_size,
    int dim, const int *control, float base, int neox, const float *factors, float magnitude) {
    int tok = blockIdx.y;
    rotary_factored_cache_kv_impl(q + (u64)tok * q_size, q_size, k + (u64)tok * k_size,
                                  v + (u64)tok * k_size, keys, values, k_size, dim,
                                  control[1] + tok, base, neox, factors, magnitude);
}

#ifdef CUDA_KV_F16
extern "C" __global__ void rotary_factored_cache_kv_decode_f16(
    float *q, int q_size, float *k, const float *v, kv_f16 *keys, kv_f16 *values, int k_size,
    int dim, const int *control, float base, int neox, const float *factors, float magnitude) {
    rotary_factored_cache_kv_impl(q, q_size, k, v, keys, values, k_size, dim, control[1], base,
                                  neox, factors, magnitude);
}

extern "C" __global__ void rotary_factored_cache_kv_batch_f16(
    float *q, int q_size, float *k, const float *v, kv_f16 *keys, kv_f16 *values, int k_size,
    int dim, int position, float base, int neox, const float *factors, float magnitude) {
    int tok = blockIdx.y;
    rotary_factored_cache_kv_impl(q + (u64)tok * q_size, q_size, k + (u64)tok * k_size,
                                  v + (u64)tok * k_size, keys, values, k_size, dim,
                                  position + tok, base, neox, factors, magnitude);
}
#endif

#ifdef CUDA_PACKED_DECODE
// Fused full-vector Q/K RMSNorm + RoPE + KV cache store for OLMoE decode (grid=2, block=1024).
// Block 0 handles Q; block 1 handles K and stores K and V into the KV cache.
template<typename KV>
__device__ __forceinline__ void olmoe_qk_norm_rope_cache_impl(
    float *q, const u8 *qn, int q_size, float *k, const u8 *kn, const float *v, KV *keys,
    KV *values, int k_size, int dim, int position, float base, int neox, float epsilon) {
    __shared__ float partial[32];
    __shared__ float shared_vec[4096];
    bool is_q = blockIdx.x == 0;
    float *vec = is_q ? q : k;
    const float *nw = (const float *)(is_q ? qn : kn);
    int size = is_q ? q_size : k_size;
    float sum = 0.0f;
    for (int i = threadIdx.x; i < size; i += blockDim.x) {
        float val = vec[i];
        shared_vec[i] = val;
        sum += val * val;
    }
    sum = warp_sum(sum);
    if ((threadIdx.x & 31) == 0) partial[threadIdx.x / 32] = sum;
    __syncthreads();
    float total = 0.0f;
    for (int i = 0; i < (int)blockDim.x / 32; i++) total += partial[i];
    float inverse = 1.0f / sqrtf(total / (float)size + epsilon);
    for (int i = threadIdx.x; i < size; i += blockDim.x)
        shared_vec[i] = shared_vec[i] * inverse * nw[i];
    __syncthreads();
    int pairs = size / 2;
    u64 kv_base = (u64)position * k_size;
    for (int p = threadIdx.x; p < pairs; p += blockDim.x) {
        int pair = p % (dim / 2), head = p / (dim / 2);
        int a = head * dim + (neox ? pair : 2 * pair);
        int b = neox ? a + dim / 2 : a + 1;
        float angle = (float)position * powf(base, -(float)(2 * pair) / (float)dim);
        float sine = sinf(angle), cosine = cosf(angle);
        float u = shared_vec[a], w = shared_vec[b];
        float ra = u * cosine - w * sine;
        float rb = u * sine + w * cosine;
        vec[a] = ra;
        vec[b] = rb;
        if (!is_q) {
            kv_store(keys, kv_base + a, ra);
            kv_store(keys, kv_base + b, rb);
            kv_store(values, kv_base + a, v[a]);
            kv_store(values, kv_base + b, v[b]);
        }
    }
}

extern "C" __global__ void olmoe_qk_norm_rope_cache_decode(
    float *q, const u8 *qn, int q_size, float *k, const u8 *kn, const float *v, float *keys,
    float *values, int k_size, int dim, const int *control, float base, int neox, float epsilon) {
    olmoe_qk_norm_rope_cache_impl(q, qn, q_size, k, kn, v, keys, values, k_size, dim, control[1],
                                  base, neox, epsilon);
}

#ifdef CUDA_KV_F16
extern "C" __global__ void olmoe_qk_norm_rope_cache_decode_f16(
    float *q, const u8 *qn, int q_size, float *k, const u8 *kn, const float *v, kv_f16 *keys,
    kv_f16 *values, int k_size, int dim, const int *control, float base, int neox, float epsilon) {
    olmoe_qk_norm_rope_cache_impl(q, qn, q_size, k, kn, v, keys, values, k_size, dim, control[1],
                                  base, neox, epsilon);
}
#endif
#endif

template<typename KV>
__device__ __forceinline__ void cache_kv_impl(const float *k, const float *v,
                                              KV *keys, KV *values, int width, int position) {
    k += (u64)blockIdx.y * width;
    v += (u64)blockIdx.y * width;
    position += blockIdx.y;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < width) {
        kv_store(keys, (u64)position * width + i, k[i]);
        kv_store(values, (u64)position * width + i, v[i]);
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

#ifdef CUDA_KV_F16
extern "C" __global__ void cache_kv_f16(const float *k, const float *v, kv_f16 *keys,
                                       kv_f16 *values, int width, int position) {
    cache_kv_impl(k, v, keys, values, width, position);
}

extern "C" __global__ void cache_kv_decode_f16(const float *k, const float *v,
                                              kv_f16 *keys, kv_f16 *values, int width,
                                              const int *control) {
    cache_kv_impl(k, v, keys, values, width, control[1]);
}
#endif

template<bool Gemma, bool Alibi = false, typename KV = float>
__device__ __forceinline__ void attention_scores_impl(const float *q, const KV *keys,
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
             * kv_load(keys, ((u64)time * kv_heads + kv_head) * dim + i);
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

extern "C" __global__ void alibi_attention_scores(const float *q, const float *keys, float *scores,
    int dim, int heads, int kv_heads, int context, int count, float max_bias) {
    attention_scores_impl<false, true>(q, keys, scores, dim, heads, kv_heads,
                                       context, count, 0, max_bias, 0.0f);
}

extern "C" __global__ void gemma_attention_scores(const float *q, const float *keys, float *scores,
    int dim, int heads, int kv_heads, int context, int count, int window, float scale, float softcap) {
    attention_scores_impl<true>(q, keys, scores, dim, heads, kv_heads, context, count, window, scale, softcap);
}

template<typename KV = float>
__device__ __forceinline__ void gemma_attention_scores_decode_impl(const float *q, const KV *keys,
    float *scores, int dim, int heads, int kv_heads, int context, const int *control, int window,
    float scale, float softcap) {
    int head = blockIdx.x;
    int count = control[1] + 1;
    int start = (window > 0 && count > window) ? (count - window) : 0;
    int kv_head = head / (heads / kv_heads);
    for (int time = start + blockIdx.y; time < count; time += gridDim.y) {
        float sum = 0.0f;
        for (int i = threadIdx.x; i < dim; i += blockDim.x)
            sum += (q[(u64)head * dim + i] * scale)
                 * kv_load(keys, ((u64)time * kv_heads + kv_head) * dim + i);
        sum = sum_block(sum);
        if (threadIdx.x == 0) {
            float score = sum;
            if (softcap > 0.0f)
                score = isfinite(score) ? softcap * tanhf(score * (1.0f / softcap))
                                        : __int_as_float(0x7fffffff);
            scores[(u64)head * context + time] = score;
        }
        __syncthreads();
    }
}

extern "C" __global__ void gemma_attention_scores_decode(const float *q, const float *keys,
    float *scores, int dim, int heads, int kv_heads, int context, const int *control, int window,
    float scale, float softcap) {
    gemma_attention_scores_decode_impl(q, keys, scores, dim, heads, kv_heads, context, control,
                                       window, scale, softcap);
}

template<bool Gemma, typename KV = float>
__device__ __forceinline__ void attention_values_impl(float *scores, const KV *values,
    float *output, int dim, int heads, int kv_heads, int context, int count, int window) {
    count += blockIdx.y;
    int start = Gemma && window > 0 && count > window ? count - window : 0;
    scores += (u64)blockIdx.y * heads * context;
    output += (u64)blockIdx.y * heads * dim;
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
    int kv_head = head / (heads / kv_heads);
    for (int i = lane; i < dim; i += blockDim.x) {
        float value = 0.0f;
        for (int t = start; t < count; t++)
            value += s[t] / total * kv_load(values, ((u64)t * kv_heads + kv_head) * dim + i);
        output[(u64)head * dim + i] = value;
    }
}

extern "C" __global__ void attention_values(float *scores, const float *values, float *output,
                                            int dim, int heads, int kv_heads, int context, int count) {
    attention_values_impl<false>(scores, values, output, dim, heads, kv_heads, context, count, 0);
}

extern "C" __global__ void gemma_attention_values(float *scores, const float *values, float *output,
    int dim, int heads, int kv_heads, int context, int count, int window) {
    attention_values_impl<true>(scores, values, output, dim, heads, kv_heads, context, count, window);
}

extern "C" __global__ void gemma_attention_values_decode(float *scores, const float *values,
    float *output, int dim, int heads, int kv_heads, int context, const int *control, int window) {
    attention_values_impl<true>(scores, values, output, dim, heads, kv_heads, context,
                                control[1] + 1, window);
}

#ifdef CUDA_KV_F16
extern "C" __global__ void attention_scores_f16(const float *q, const kv_f16 *keys,
    float *scores, int dim, int heads, int kv_heads, int context, int count) {
    attention_scores_impl<false, false>(q, keys, scores, dim, heads, kv_heads, context, count, 0,
                                        1.0f, 0.0f);
}

extern "C" __global__ void attention_values_f16(float *scores, const kv_f16 *values,
    float *output, int dim, int heads, int kv_heads, int context, int count) {
    attention_values_impl<false>(scores, values, output, dim, heads, kv_heads, context, count, 0);
}

extern "C" __global__ void gemma_attention_scores_f16(const float *q, const kv_f16 *keys,
    float *scores, int dim, int heads, int kv_heads, int context, int count, int window,
    float scale, float softcap) {
    attention_scores_impl<true>(q, keys, scores, dim, heads, kv_heads, context, count, window,
                                scale, softcap);
}

extern "C" __global__ void gemma_attention_values_f16(float *scores, const kv_f16 *values,
    float *output, int dim, int heads, int kv_heads, int context, int count, int window) {
    attention_values_impl<true>(scores, values, output, dim, heads, kv_heads, context, count, window);
}

extern "C" __global__ void gemma_attention_scores_decode_f16(const float *q, const kv_f16 *keys,
    float *scores, int dim, int heads, int kv_heads, int context, const int *control, int window,
    float scale, float softcap) {
    gemma_attention_scores_decode_impl(q, keys, scores, dim, heads, kv_heads, context, control,
                                       window, scale, softcap);
}

extern "C" __global__ void gemma_attention_values_decode_f16(float *scores, const kv_f16 *values,
    float *output, int dim, int heads, int kv_heads, int context, const int *control, int window) {
    attention_values_impl<true>(scores, values, output, dim, heads, kv_heads, context,
                                control[1] + 1, window);
}
#endif

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

#ifdef CUDA_PACKED_DECODE
// One-warp full-softmax top-k router for 64 experts (OLMoE): 2 experts per lane.
extern "C" __global__ void router_warp64_full_softmax(const float *scores, int used,
                                                      int *ids, float *weights, int *status) {
    int lane = threadIdx.x;
    float v[2] = {scores[lane], scores[lane + 32]};
    if (!isfinite(v[0]) || !isfinite(v[1])) atomicExch(status, 1);
    float maximum = fmaxf(v[0], v[1]);
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2)
        maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffff, maximum, offset));
    float e0 = expf(v[0] - maximum), e1 = expf(v[1] - maximum);
    float denom = warp_sum(e0 + e1);
    int idx[2] = {lane, lane + 32};
    for (int s = 0; s < used; s++) {
        float bv = -__int_as_float(0x7f800000);
        int bi = 64;
        if (v[0] > bv || (v[0] == bv && idx[0] < bi)) { bv = v[0]; bi = idx[0]; }
        if (v[1] > bv || (v[1] == bv && idx[1] < bi)) { bv = v[1]; bi = idx[1]; }
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            float ov = __shfl_xor_sync(0xffffffff, bv, offset);
            int oi = __shfl_xor_sync(0xffffffff, bi, offset);
            if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
        }
        if (lane == 0) {
            ids[s] = bi;
            weights[s] = expf(bv - maximum) / denom;
        }
        if (idx[0] == bi) v[0] = -__int_as_float(0x7f800000);
        if (idx[1] == bi) v[1] = -__int_as_float(0x7f800000);
    }
}
#endif
#if defined(CUDA_M8_PROTOTYPES) || defined(CUDA_DECODE_FUSION_RESEARCH)
#define CUDA_ROUTER_WARP128 1
#endif
#if defined(CUDA_ROUTER_WARP128) || defined(CUDA_PACKED_DECODE)
// Same selection, tie-breaking and weight summation order as router(); one warp.
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
    const float neg_inf = -__int_as_float(0x7f800000);
    int my_id = 0;
    float my_score = 0.0f;
    float maximum = 0.0f;
    for (int s = 0; s < used; s++) {
        int best_id = -1;
        float best_score = neg_inf;
        #pragma unroll
        for (int k = 0; k < 4; k++) {
            int idx = lane + k * 32;
            float sc = val[k];
            if (sc > best_score || (sc == best_score && best_id >= 0 && idx < best_id)
                || (best_id < 0 && sc != neg_inf)) {
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
        if ((win_id & 31) == lane) {
            val[win_id >> 5] = neg_inf;
        }
        if (s == 0) maximum = win_score;
        if (lane == s) {
            my_id = win_id;
            my_score = win_score;
        }
    }
    float my_weight = (lane < used) ? expf(my_score - maximum) : 0.0f;
    float total = warp_sum(my_weight);
    if (lane < used) {
        ids[lane] = my_id;
        weights[lane] = my_weight / total;
    }
}
#endif

#if defined(CUDA_M8_PROTOTYPES) || defined(CUDA_DECODE_FUSION_RESEARCH) || defined(CUDA_PACKED_DECODE)
// Equals head_norm (256-lane tree) + rotary + cache_kv for dim-128 heads with
// full, unscaled RoPE; grid=(heads + kv_heads), block=256.
template<typename KV>
__device__ __forceinline__ void qkv_head_norm_rope_cache_impl(
    float *q, const u8 *q_norm, int heads,
    float *k, const u8 *k_norm, const float *v,
    KV *keys, KV *values, int kv_heads,
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
            kv_store(keys, kv_base + a, ra);
            kv_store(keys, kv_base + b_idx, rb);
        }
    }
    if (!is_q && tid >= 128) {
        int i = tid - 128;
        u64 kv_base = ((u64)position * kv_heads + head) * 128;
        kv_store(values, kv_base + i, v[(u64)head * 128 + i]);
    }
}

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

extern "C" __global__ void qkv_head_norm_rope_cache_batch(
    float *q, const u8 *q_norm, int heads,
    float *k, const u8 *k_norm, const float *v,
    float *keys, float *values, int kv_heads,
    float epsilon, int position, float base, int neox) {
    int tok = blockIdx.y;
    qkv_head_norm_rope_cache_impl(
        q + (u64)tok * heads * 128, q_norm, heads,
        k + (u64)tok * kv_heads * 128, k_norm,
        v + (u64)tok * kv_heads * 128,
        keys, values, kv_heads, epsilon, position + tok, base, neox);
}

extern "C" __global__ void qkv_head_norm_rope_cache_batch_graph(
    float *q, const u8 *q_norm, int heads,
    float *k, const u8 *k_norm, const float *v,
    float *keys, float *values, int kv_heads,
    float epsilon, const int *control, float base, int neox) {
    int tok = blockIdx.y;
    qkv_head_norm_rope_cache_impl(
        q + (u64)tok * heads * 128, q_norm, heads,
        k + (u64)tok * kv_heads * 128, k_norm,
        v + (u64)tok * kv_heads * 128,
        keys, values, kv_heads, epsilon, control[1] + tok, base, neox);
}

#ifdef CUDA_KV_F16
extern "C" __global__ void qkv_head_norm_rope_cache_f16(
    float *q, const u8 *q_norm, int heads,
    float *k, const u8 *k_norm, const float *v,
    kv_f16 *keys, kv_f16 *values, int kv_heads,
    float epsilon, int position, float base, int neox) {
    qkv_head_norm_rope_cache_impl(
        q, q_norm, heads, k, k_norm, v, keys, values, kv_heads, epsilon, position, base, neox);
}

extern "C" __global__ void qkv_head_norm_rope_cache_decode_f16(
    float *q, const u8 *q_norm, int heads,
    float *k, const u8 *k_norm, const float *v,
    kv_f16 *keys, kv_f16 *values, int kv_heads,
    float epsilon, const int *control, float base, int neox) {
    qkv_head_norm_rope_cache_impl(
        q, q_norm, heads, k, k_norm, v, keys, values, kv_heads, epsilon, control[1], base, neox);
}
#endif
#endif

#ifdef CUDA_M8_PROTOTYPES
extern "C" __global__ void residual_normalize(float *x, const float *projected,
    const u8 *w, float *y, int n, float epsilon) {
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

#if defined(CUDA_PACKED_DECODE)
#define NORM_ITEMS 4
// Single-block RMSNorm tail: v holds this thread's updated values (element
// threadIdx.x + k * blockDim.x) and nw the matching norm weights, both loaded up
// front so global latency overlaps. y = v * inverse * w, as normalize_warp.
__device__ __forceinline__ void block_rmsnorm(const float *v, const float *nw, float *y,
                                              int n, float epsilon) {
    __shared__ float partial[32];
    float sum = 0.0f;
    #pragma unroll
    for (int k = 0; k < NORM_ITEMS; k++)
        if (threadIdx.x + k * blockDim.x < n) sum += v[k] * v[k];
    sum = warp_sum(sum);
    if ((threadIdx.x & 31) == 0) partial[threadIdx.x / 32] = sum;
    __syncthreads();
    float total = 0.0f;
    for (int i = 0; i < (int)blockDim.x / 32; i++) total += partial[i];
    float inverse = 1.0f / sqrtf(total / (float)n + epsilon);
    #pragma unroll
    for (int k = 0; k < NORM_ITEMS; k++) {
        int i = threadIdx.x + k * blockDim.x;
        if (i < n) y[i] = v[k] * inverse * nw[k];
    }
}

// x += projected, then y = RMSNorm(x) * w. grid=1, block=1024, n <= 4 * 1024.
extern "C" __global__ void residual_rmsnorm(float *x, const float *projected, const u8 *w,
                                            float *y, int n, float epsilon) {
    float v[NORM_ITEMS], nw[NORM_ITEMS];
    #pragma unroll
    for (int k = 0; k < NORM_ITEMS; k++) {
        int i = threadIdx.x + k * blockDim.x;
        if (i < n) {
            v[k] = x[i] + projected[i];
            nw[k] = ((const float *)w)[i];
            x[i] = v[k];
        }
    }
    block_rmsnorm(v, nw, y, n, epsilon);
}

// MoE combine (x += sum_s weights[s] * experts[s]) fused with the next RMSNorm.
// grid=1, block=1024, n <= 4 * 1024.
extern "C" __global__ void combine_rmsnorm(float *x, const float *experts,
                                           const float *weights, int n, int used,
                                           const u8 *w, float *y, float epsilon) {
    float v[NORM_ITEMS], nw[NORM_ITEMS];
    #pragma unroll
    for (int k = 0; k < NORM_ITEMS; k++) {
        int i = threadIdx.x + k * blockDim.x;
        if (i < n) {
            float combined = 0.0f;
            for (int s = 0; s < used; s++) combined += weights[s] * experts[(u64)s * n + i];
            v[k] = x[i] + combined;
            nw[k] = ((const float *)w)[i];
            x[i] = v[k];
        }
    }
    block_rmsnorm(v, nw, y, n, epsilon);
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
