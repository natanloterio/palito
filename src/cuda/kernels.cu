typedef unsigned char u8;
typedef unsigned long long u64;

__device__ float half_value(const u8 *p) {
    unsigned int h = (unsigned int)p[0] | ((unsigned int)p[1] << 8);
    unsigned int sign = (h & 0x8000) << 16;
    unsigned int exponent = (h >> 10) & 31;
    unsigned int fraction = h & 1023;
    if (exponent == 0) {
        float value = (float)fraction * (1.0f / 16777216.0f);
        return (sign != 0) ? -value : value;
    }
    return __uint_as_float(sign | ((exponent + 112) << 23) | (fraction << 13));
}

__device__ __forceinline__ float weight(const u8 *data, int kind, u64 index) {
    if (kind == 0) return ((const float *)data)[index];
    if (kind == 1) return half_value(data + index * 2);
    int block = (kind == 6 || kind == 8) ? 32 : 256;
    int size = kind == 6 ? 22 : kind == 8 ? 34 : kind == 12 ? 144 : 210;
    const u8 *p = data + (index / block) * size;
    int i = index % block;
    if (kind == 6) {
        unsigned int high = (unsigned int)p[2] | ((unsigned int)p[3] << 8)
            | ((unsigned int)p[4] << 16) | ((unsigned int)p[5] << 24);
        int q = ((p[6 + i % 16] >> (4 * (i / 16))) & 15) + ((high >> i) & 1) * 16 - 16;
        return half_value(p) * (float)q;
    }
    if (kind == 8) return half_value(p) * (float)((const signed char *)p)[2 + i];
    if (kind == 12) {
        const u8 *s = p + 4;
        int g = i / 32;
        int scale = g < 4 ? s[g] & 63 : (s[g + 4] & 15) | ((s[g - 4] >> 6) << 4);
        int minimum = g < 4 ? s[g + 4] & 63 : (s[g + 4] >> 4) | ((s[g] >> 6) << 4);
        int q = (p[16 + (i / 64) * 32 + i % 32] >> (4 * (g % 2))) & 15;
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

__device__ float sum_block(float value) {
    __shared__ float partial[256];
    int lane = threadIdx.x;
    partial[lane] = value;
    __syncthreads();
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
        case 6: sum = matrix_row<6>(w, input, offset, width); break;
        case 8: sum = matrix_row<8>(w, input, offset, width); break;
        case 12: sum = matrix_row<12>(w, input, offset, width); break;
        case 14: sum = matrix_row<14>(w, input, offset, width); break;
        default: sum = __int_as_float(0x7fffffff); break;
    }
    if (threadIdx.x == 0) y[(u64)selected * rows + row] = sum;
}

#ifdef CUDA_M8_PROTOTYPES
// One warp owns a complete 256-value block, with eight independent K chains.
// Q4_K blocks are 16-byte aligned; Q6_K's 210-byte stride permits only u16
// loads. Metadata stays local to the lane, without warp-wide broadcasts.
template<int Kind>
__device__ __forceinline__ float packed_matrix_row(const u8 *w, const float *x,
                                                   u64 offset, int width) {
    float parts[8] = {};
    int lane = threadIdx.x & 31;
    for (int first = (threadIdx.x / 32) * 256; first < width; first += 2048) {
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
    return sum_block(parts[0] + parts[1]);
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
            case 14: sum = matrix_row<14>(w, input, offset, width); break;
            default: sum = __int_as_float(0x7fffffff); break;
        }
    }
    if (threadIdx.x == 0) y[(u64)selected * rows + row] = sum;
}
#endif

extern "C" __global__ void normalize(const float *x, const u8 *w, float *y, int n, float epsilon) {
    x += (u64)blockIdx.x * n;
    y += (u64)blockIdx.x * n;
    float sum = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) sum += x[i] * x[i];
    float inverse = 1.0f / sqrtf(sum_block(sum) / (float)n + epsilon);
    for (int i = threadIdx.x; i < n; i += blockDim.x) y[i] = x[i] * inverse * ((const float *)w)[i];
}

extern "C" __global__ void head_norm(float *x, const u8 *w, int dim, float epsilon) {
    x += (u64)blockIdx.x * dim;
    float sum = 0.0f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) sum += x[i] * x[i];
    float inverse = 1.0f / sqrtf(sum_block(sum) / (float)dim + epsilon);
    for (int i = threadIdx.x; i < dim; i += blockDim.x) x[i] *= inverse * ((const float *)w)[i];
}

__device__ __forceinline__ void rotary_impl(float *x, int n, int dim, int position, float base, int neox) {
    x += (u64)blockIdx.y * n;
    position += blockIdx.y;
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= n / 2) return;
    int pair = p % (dim / 2), head = p / (dim / 2);
    int a = head * dim + (neox ? pair : 2 * pair);
    int b = neox ? a + dim / 2 : a + 1;
    float angle = (float)position * powf(base, -(float)(2 * pair) / (float)dim);
    float sine = sinf(angle), cosine = cosf(angle);
    float u = x[a], v = x[b];
    x[a] = u * cosine - v * sine;
    x[b] = u * sine + v * cosine;
}

extern "C" __global__ void rotary(float *x, int n, int dim, int position, float base, int neox) {
    rotary_impl(x, n, dim, position, base, neox);
}

extern "C" __global__ void rotary_decode(float *x, int n, int dim, const int *control,
                                         float base, int neox) {
    rotary_impl(x, n, dim, control[1], base, neox);
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

extern "C" __global__ void attention_scores(const float *q, const float *keys, float *scores,
                                            int dim, int heads, int kv_heads, int context, int count) {
    int head = blockIdx.x, time = blockIdx.y;
    count += blockIdx.z;
    if (time >= count) return;
    q += (u64)blockIdx.z * heads * dim;
    scores += (u64)blockIdx.z * heads * context;
    int kv_head = head / (heads / kv_heads);
    float sum = 0.0f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x)
        sum += q[(u64)head * dim + i] * keys[((u64)time * kv_heads + kv_head) * dim + i];
    sum = sum_block(sum);
    if (threadIdx.x == 0) scores[(u64)head * context + time] = sum / sqrtf((float)dim);
}

extern "C" __global__ void attention_values(float *scores, const float *values, float *output,
                                            int dim, int heads, int kv_heads, int context, int count) {
    count += blockIdx.y;
    scores += (u64)blockIdx.y * heads * context;
    output += (u64)blockIdx.y * heads * dim;
    __shared__ float maxima[256];
    int head = blockIdx.x, lane = threadIdx.x;
    float *s = scores + (u64)head * context;
    float maximum = -__int_as_float(0x7f800000);
    for (int t = lane; t < count; t += blockDim.x) maximum = fmaxf(maximum, s[t]);
    maxima[lane] = maximum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (lane < stride) maxima[lane] = fmaxf(maxima[lane], maxima[lane + stride]);
        __syncthreads();
    }
    maximum = maxima[0];
    float total = 0.0f;
    for (int t = lane; t < count; t += blockDim.x) {
        s[t] = expf(s[t] - maximum);
        total += s[t];
    }
    total = sum_block(total);
    int kv_head = head / (heads / kv_heads);
    for (int i = lane; i < dim; i += blockDim.x) {
        float value = 0.0f;
        for (int t = 0; t < count; t++)
            value += s[t] / total * values[((u64)t * kv_heads + kv_head) * dim + i];
        output[(u64)head * dim + i] = value;
    }
}

extern "C" __global__ void router(const float *scores, int count, int used,
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
    for (int s = lane; s < used; s += blockDim.x) weights[s] /= total;
}

extern "C" __global__ void swiglu(float *gate, const float *up, int count) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) gate[i] = gate[i] / (1.0f + expf(-gate[i])) * up[i];
}

extern "C" __global__ void residual(float *x, const float *projected, int width) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < width) x[i] += projected[i];
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
