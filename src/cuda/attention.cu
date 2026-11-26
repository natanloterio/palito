// One block per query/head keeps bounded scores on chip. Match the original
// 256-lane dot and softmax trees rather than changing numerical precision.
template<typename KV>
__device__ __forceinline__ void attention_fused_impl(const float *q, const KV *keys,
                                                    const KV *values, float *output,
                                                    int dim, int heads, int kv_heads, int count) {
    __shared__ float scores[1024];
    __shared__ float maxima[256];
    int head = blockIdx.x, token = blockIdx.y, thread = threadIdx.x;
    int lane = thread % 32, warp = thread / 32;
    int kv_head = head / (heads / kv_heads);
    count += token;
    q += ((u64)token * heads + head) * dim;
#ifdef CUDA_ATTENTION_REUSE_RESEARCH
    float query[8] = {};
    #pragma unroll
    for (int p = 0; p < 8; p++) {
        int i = lane + p * 32;
        if (i < dim) query[p] = q[i];
    }
#endif
    for (int time = warp; time < count; time += 8) {
        float parts[8] = {};
        #pragma unroll
        for (int p = 0; p < 8; p++) {
            int i = lane + p * 32;
            if (i < dim)
#ifdef CUDA_ATTENTION_REUSE_RESEARCH
                parts[p] = query[p] * kv_load(keys, ((u64)time * kv_heads + kv_head) * dim + i);
#else
                parts[p] = q[i] * kv_load(keys, ((u64)time * kv_heads + kv_head) * dim + i);
#endif
        }

        parts[0] += parts[4];
        parts[1] += parts[5];
        parts[2] += parts[6];
        parts[3] += parts[7];
        parts[0] += parts[2];
        parts[1] += parts[3];
        float sum = parts[0] + parts[1];
        for (int stride = 16; stride > 0; stride /= 2)
            sum += __shfl_down_sync(0xffffffff, sum, stride);
        if (lane == 0) scores[time] = sum / sqrtf((float)dim);
    }
    __syncthreads();
    float maximum = -__int_as_float(0x7f800000);
    for (int time = thread; time < count; time += 256) maximum = fmaxf(maximum, scores[time]);
    maxima[thread] = maximum;
    __syncthreads();
#ifdef CUDA_ATTENTION_REUSE_RESEARCH
    __shared__ float reduced_maximum;
    if (thread < 32) {
        float a = fmaxf(maximum, maxima[thread + 128]);
        float b = fmaxf(maxima[thread + 64], maxima[thread + 192]);
        float c = fmaxf(maxima[thread + 32], maxima[thread + 160]);
        float d = fmaxf(maxima[thread + 96], maxima[thread + 224]);
        maximum = fmaxf(fmaxf(a, b), fmaxf(c, d));
        #pragma unroll
        for (int stride = 16; stride > 0; stride /= 2)
            maximum = fmaxf(maximum, __shfl_down_sync(0xffffffff, maximum, stride));
        if (thread == 0) reduced_maximum = maximum;
    }
    __syncthreads();
    maximum = reduced_maximum;
#else
    for (int stride = 128; stride > 0; stride /= 2) {
        if (thread < stride) maxima[thread] = fmaxf(maxima[thread], maxima[thread + stride]);
        __syncthreads();
    }
    maximum = maxima[0];
#endif
    float total = 0.0f;
    for (int time = thread; time < count; time += 256) {
        scores[time] = expf(scores[time] - maximum);
        total += scores[time];
    }
    total = sum_block(total);
#ifdef CUDA_NORMALIZED_ATTENTION
    for (int time = thread; time < count; time += 256) scores[time] /= total;
    __syncthreads();
#endif
    for (int i = thread; i < dim; i += 256) {
        float value = 0.0f;
        for (int time = 0; time < count; time++)
#ifdef CUDA_NORMALIZED_ATTENTION
            value += scores[time] * kv_load(values, ((u64)time * kv_heads + kv_head) * dim + i);
#else
            value += scores[time] / total
                * kv_load(values, ((u64)time * kv_heads + kv_head) * dim + i);
#endif
        output[((u64)token * heads + head) * dim + i] = value;
    }
}

extern "C" __global__ void attention_fused(const float *q, const float *keys,
                                          const float *values, float *output,
                                          int dim, int heads, int kv_heads, int count) {
    attention_fused_impl(q, keys, values, output, dim, heads, kv_heads, count);
}

// Fast 1-warp-per-(head, token) causal FlashAttention for small-batch verification
// and EAGLE-3 drafting (dim == 128, 1 <= n <= 16, zero __syncthreads(), float4 loads).
// Grid: (heads, n, 1), Block: 32 threads.
template<typename KV>
__device__ __forceinline__ void attention_warp128_impl(
    const float *q, const KV *keys, const KV *values, float *output,
    int heads, int kv_heads, int count) {
    int head = blockIdx.x, token = blockIdx.y, lane = threadIdx.x;
    int kv_head = head / (heads / kv_heads);
    int limit = count + token;
    const float4 qv = __ldg((const float4 *)(q + ((u64)token * heads + head) * 128) + lane);
    const float scale = 0.0883883476f; // 1 / sqrt(128)

    float m = -1.0e30f;
    float l = 0.0f;
    float o0 = 0.0f, o1 = 0.0f, o2 = 0.0f, o3 = 0.0f;

    for (int t = 0; t < limit; ++t) {
        u64 kv_off = ((u64)t * kv_heads + kv_head) * 128 + lane * 4;
        float k0, k1, k2, k3;
        if constexpr (sizeof(KV) == sizeof(float)) {
            float4 kv = __ldg((const float4 *)((const float *)keys + kv_off));
            k0 = kv.x; k1 = kv.y; k2 = kv.z; k3 = kv.w;
        } else {
            k0 = kv_load(keys, kv_off);
            k1 = kv_load(keys, kv_off + 1);
            k2 = kv_load(keys, kv_off + 2);
            k3 = kv_load(keys, kv_off + 3);
        }
        float dot = qv.x * k0 + qv.y * k1 + qv.z * k2 + qv.w * k3;
        #pragma unroll
        for (int s = 16; s > 0; s >>= 1) dot += __shfl_xor_sync(0xffffffffu, dot, s);
        float score = dot * scale;
        float m_new = fmaxf(m, score);
        float alpha = __expf(m - m_new);
        float p = __expf(score - m_new);
        float v0, v1, v2, v3;
        if constexpr (sizeof(KV) == sizeof(float)) {
            float4 vv = __ldg((const float4 *)((const float *)values + kv_off));
            v0 = vv.x; v1 = vv.y; v2 = vv.z; v3 = vv.w;
        } else {
            v0 = kv_load(values, kv_off);
            v1 = kv_load(values, kv_off + 1);
            v2 = kv_load(values, kv_off + 2);
            v3 = kv_load(values, kv_off + 3);
        }
        o0 = fmaf(o0, alpha, p * v0);
        o1 = fmaf(o1, alpha, p * v1);
        o2 = fmaf(o2, alpha, p * v2);
        o3 = fmaf(o3, alpha, p * v3);
        l = fmaf(l, alpha, p);
        m = m_new;
    }
    float inv_l = 1.0f / l;
    float4 *out4 = (float4 *)(output + ((u64)token * heads + head) * 128);
    out4[lane] = make_float4(o0 * inv_l, o1 * inv_l, o2 * inv_l, o3 * inv_l);
}

extern "C" __global__ void attention_warp128(const float *q, const float *keys,
                                             const float *values, float *output,
                                             int heads, int kv_heads, int count) {
    attention_warp128_impl(q, keys, values, output, heads, kv_heads, count);
}

#ifdef CUDA_KV_F16
extern "C" __global__ void attention_warp128_f16(const float *q, const kv_f16 *keys,
                                                 const kv_f16 *values, float *output,
                                                 int heads, int kv_heads, int count) {
    attention_warp128_impl(q, keys, values, output, heads, kv_heads, count);
}
#endif

// Vectorized tree-masked causal attention for speculative tree verification (tree_len <= 64).
// Uses warp-parallel 8-way dot products matching attention_fused_impl (4x-6x faster than scalar).
// Grid: (heads, tree_len, 1), Block: 256 threads (8 warps).
extern "C" __global__ void attention_tree(const float *q, const float *keys,
                                          const float *values, float *output,
                                          int dim, int heads, int kv_heads,
                                          int prefix_len, int tree_len,
                                          const unsigned long long *tree_mask) {
    extern __shared__ float scores[];
    __shared__ float maxima[256];
    int head = blockIdx.x, token = blockIdx.y, thread = threadIdx.x;
    int lane = thread & 31, warp = thread >> 5;
    int kv_head = head / (heads / kv_heads);
    int total_kv = prefix_len + tree_len;
    unsigned long long mask = tree_mask[token];
    const float *query = q + ((u64)token * heads + head) * dim;
    float inv_sqrt_d = rsqrtf((float)dim);

    float qreg[8] = {};
    #pragma unroll
    for (int p = 0; p < 8; p++) {
        int i = lane + p * 32;
        if (i < dim) qreg[p] = query[i];
    }

    for (int time = warp; time < total_kv; time += 8) {
        bool visible = (time < prefix_len) || (((mask >> (time - prefix_len)) & 1ULL) != 0ULL);
        if (!visible) {
            if (lane == 0) scores[time] = -1.0e30f;
            continue;
        }
        const float *key = keys + ((u64)time * kv_heads + kv_head) * dim;
        float parts[8] = {};
        #pragma unroll
        for (int p = 0; p < 8; p++) {
            int i = lane + p * 32;
            if (i < dim) parts[p] = qreg[p] * key[i];
        }
        parts[0] += parts[4];
        parts[1] += parts[5];
        parts[2] += parts[6];
        parts[3] += parts[7];
        parts[0] += parts[2];
        parts[1] += parts[3];
        float sum = parts[0] + parts[1];
        for (int stride = 16; stride > 0; stride >>= 1)
            sum += __shfl_down_sync(0xffffffffu, sum, stride);
        if (lane == 0) scores[time] = sum * inv_sqrt_d;
    }
    __syncthreads();

    float maximum = -1.0e30f;
    for (int time = thread; time < total_kv; time += 256) {
        if (scores[time] > maximum) maximum = scores[time];
    }
    maxima[thread] = maximum;
    __syncthreads();
    for (int stride = 128; stride > 0; stride >>= 1) {
        if (thread < stride && maxima[thread + stride] > maxima[thread])
            maxima[thread] = maxima[thread + stride];
        __syncthreads();
    }
    maximum = maxima[0];

    float total = 0.0f;
    for (int time = thread; time < total_kv; time += 256) {
        bool visible = (time < prefix_len) || (((mask >> (time - prefix_len)) & 1ULL) != 0ULL);
        float w = visible ? expf(scores[time] - maximum) : 0.0f;
        scores[time] = w;
        total += w;
    }
    total = sum_block(total);
    float inv_total = 1.0f / total;
    for (int i = thread; i < dim; i += 256) {
        float value = 0.0f;
        for (int time = 0; time < total_kv; time++) {
            float w = scores[time];
            if (w != 0.0f) {
                value += w * values[((u64)time * kv_heads + kv_head) * dim + i];
            }
        }
        output[((u64)token * heads + head) * dim + i] = value * inv_total;
    }
}

extern "C" __global__ void attention_fused_decode(const float *q, const float *keys,
                                                 const float *values, float *output,
                                                 int dim, int heads, int kv_heads,
                                                 const int *control) {
    attention_fused_impl(q, keys, values, output, dim, heads, kv_heads, control[1] + 1);
}

#ifdef CUDA_KV_F16
extern "C" __global__ void attention_fused_f16(const float *q, const kv_f16 *keys,
                                              const kv_f16 *values, float *output,
                                              int dim, int heads, int kv_heads, int count) {
    attention_fused_impl(q, keys, values, output, dim, heads, kv_heads, count);
}

extern "C" __global__ void attention_fused_decode_f16(const float *q, const kv_f16 *keys,
                                                     const kv_f16 *values, float *output,
                                                     int dim, int heads, int kv_heads,
                                                     const int *control) {
    attention_fused_impl(q, keys, values, output, dim, heads, kv_heads, control[1] + 1);
}
#endif

#if defined(CUDA_M8_PROTOTYPES) || defined(CUDA_ATTENTION_QUERY_PAIR_RESEARCH)
// Launch with even query count, 128-wide heads and 256 threads.
extern "C" __global__ void attention_fused_query_pair(const float *q, const float *keys,
    const float *values, float *output, int heads, int kv_heads, int count) {
    __shared__ float scores[2][1024], maxima[2][256], peak[2];
    int head = blockIdx.x, token = blockIdx.y * 2, tid = threadIdx.x;
    int lane = tid % 32, warp = tid / 32, kv_head = head / (heads / kv_heads);
    int limit = count + token;
    u64 query_offset = ((u64)token * heads + head) * 128;
    float query[2][4];
    #pragma unroll
    for (int p = 0; p < 4; ++p) {
        int i = lane + p * 32;
        query[0][p] = q[query_offset + i];
        query[1][p] = q[query_offset + heads * 128 + i];
    }
    for (int time = warp; time <= limit; time += 8) {
        float products[2][4];
        #pragma unroll
        for (int p = 0; p < 4; ++p) {
            float key = keys[((u64)time * kv_heads + kv_head) * 128 + lane + p * 32];
            products[0][p] = query[0][p] * key;
            products[1][p] = query[1][p] * key;
        }
        #pragma unroll
        for (int j = 0; j < 2; ++j) {
            if (time >= limit + j) continue;
            // The original 256-lane tree includes four inactive products.
            float sum = (__fadd_rn(products[j][0], 0.0f) + __fadd_rn(products[j][2], 0.0f))
                      + (__fadd_rn(products[j][1], 0.0f) + __fadd_rn(products[j][3], 0.0f));
            #pragma unroll
            for (int stride = 16; stride > 0; stride /= 2)
                sum += __shfl_down_sync(0xffffffff, sum, stride);
            if (lane == 0) scores[j][time] = sum / sqrtf(128.0f);
        }
    }
    __syncthreads();
    #pragma unroll
    for (int j = 0; j < 2; ++j) {
        float max_score = -__int_as_float(0x7f800000);
        for (int time = tid; time < limit + j; time += 256)
            max_score = fmaxf(max_score, scores[j][time]);
        maxima[j][tid] = max_score;
    }
    __syncthreads();
    if (tid < 64) {
        int j = tid / 32;
        const float *m = maxima[j] + lane;
        float maximum = fmaxf(fmaxf(fmaxf(m[0], m[128]), fmaxf(m[64], m[192])),
                              fmaxf(fmaxf(m[32], m[160]), fmaxf(m[96], m[224])));
        #pragma unroll
        for (int stride = 16; stride > 0; stride /= 2)
            maximum = fmaxf(maximum, __shfl_down_sync(0xffffffff, maximum, stride));
        if (lane == 0) peak[j] = maximum;
    }
    __syncthreads();
    #pragma unroll
    for (int j = 0; j < 2; ++j) {
        float total = 0.0f;
        for (int time = tid; time < limit + j; time += 256) {
            scores[j][time] = expf(scores[j][time] - peak[j]);
            total += scores[j][time];
        }
        total = sum_block(total);
        for (int time = tid; time < limit + j; time += 256) scores[j][time] /= total;
        __syncthreads();
    }
    if (tid < 128) {
        float a = 0.0f, b = 0.0f;
        for (int time = 0; time < limit; ++time) {
            float value = values[((u64)time * kv_heads + kv_head) * 128 + tid];
            a += scores[0][time] * value;
            b += scores[1][time] * value;
        }
        b += scores[1][limit] * values[((u64)limit * kv_heads + kv_head) * 128 + tid];
        output[query_offset + tid] = a;
        output[query_offset + heads * 128 + tid] = b;
    }
}
#endif

// Split-K (flash-decoding) attention for one decode query, any context length.
// Pass 1: grid=(heads, splits), block=256. Split j covers keys
// [j * ATTENTION_SPLIT, (j + 1) * ATTENTION_SPLIT) and writes, per head/split,
// unnormalized o[dim], its max m and its exp-sum l into
// partial[(head * splits + j) * (dim + 2)]. Splits past `count` exit, so one
// captured grid serves every decode position. F32; requires dim <= 256.
#define ATTENTION_SPLIT 16
template<typename KV>
__device__ __forceinline__ void attention_split_impl(const float *q, const KV *keys,
                                                    const KV *values, float *partial,
                                                    int dim, int heads, int kv_heads, int count) {
    __shared__ float scores[ATTENTION_SPLIT];
    __shared__ float stats[2];
    int head = blockIdx.x, split = blockIdx.y, splits = gridDim.y, thread = threadIdx.x;
    int lane = thread % 32, warp = thread / 32, kv_head = head / (heads / kv_heads);
    int first = split * ATTENTION_SPLIT;
    if (first >= count) return;
    int tile = min(ATTENTION_SPLIT, count - first);
    q += (u64)head * dim;
    for (int t = warp; t < tile; t += 8) {
        const KV *key = keys + ((u64)(first + t) * kv_heads + kv_head) * dim;
        float sum = 0.0f;
        for (int i = lane; i < dim; i += 32) sum += q[i] * kv_load(key, i);
        #pragma unroll
        for (int stride = 16; stride > 0; stride /= 2)
            sum += __shfl_xor_sync(0xffffffff, sum, stride);
        if (lane == 0) scores[t] = sum / sqrtf((float)dim);
    }
    __syncthreads();
    if (warp == 0) {
        float maximum = -__int_as_float(0x7f800000);
        for (int t = lane; t < tile; t += 32) maximum = fmaxf(maximum, scores[t]);
        #pragma unroll
        for (int stride = 16; stride > 0; stride /= 2)
            maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffff, maximum, stride));
        float total = 0.0f;
        for (int t = lane; t < tile; t += 32) {
            float p = expf(scores[t] - maximum);
            scores[t] = p;
            total += p;
        }
        #pragma unroll
        for (int stride = 16; stride > 0; stride /= 2)
            total += __shfl_xor_sync(0xffffffff, total, stride);
        if (lane == 0) {
            stats[0] = maximum;
            stats[1] = total;
        }
    }
    __syncthreads();
    float *out = partial + ((u64)head * splits + split) * (dim + 2);
    for (int i = thread; i < dim; i += 256) {
        float value = 0.0f;
        for (int t = 0; t < tile; t++)
            value += scores[t] * kv_load(values, ((u64)(first + t) * kv_heads + kv_head) * dim + i);
        out[i] = value;
    }
    if (thread == 0) {
        out[dim] = stats[0];
        out[dim + 1] = stats[1];
    }
}

extern "C" __global__ void attention_split(const float *q, const float *keys,
                                          const float *values, float *partial,
                                          int dim, int heads, int kv_heads, int count) {
    attention_split_impl(q, keys, values, partial, dim, heads, kv_heads, count);
}

extern "C" __global__ void attention_split_decode(const float *q, const float *keys,
                                                 const float *values, float *partial,
                                                 int dim, int heads, int kv_heads,
                                                 const int *control) {
    attention_split_impl(q, keys, values, partial, dim, heads, kv_heads, control[1] + 1);
}

#ifdef CUDA_KV_F16
extern "C" __global__ void attention_split_f16(const float *q, const kv_f16 *keys,
                                              const kv_f16 *values, float *partial,
                                              int dim, int heads, int kv_heads, int count) {
    attention_split_impl(q, keys, values, partial, dim, heads, kv_heads, count);
}

extern "C" __global__ void attention_split_decode_f16(const float *q, const kv_f16 *keys,
                                                     const kv_f16 *values, float *partial,
                                                     int dim, int heads, int kv_heads,
                                                     const int *control) {
    attention_split_impl(q, keys, values, partial, dim, heads, kv_heads, control[1] + 1);
}
#endif

// Pass 2: grid=(heads), block=256. Rescales every active split to the global max.
__device__ __forceinline__ void attention_combine_impl(const float *partial, float *output,
                                                      int dim, int splits, int count) {
    int head = blockIdx.x, active = (count + ATTENTION_SPLIT - 1) / ATTENTION_SPLIT;
    const float *base = partial + (u64)head * splits * (dim + 2);
    float maximum = -__int_as_float(0x7f800000);
    for (int j = 0; j < active; j++) maximum = fmaxf(maximum, base[(u64)j * (dim + 2) + dim]);
    float total = 0.0f;
    for (int j = 0; j < active; j++) {
        const float *stats = base + (u64)j * (dim + 2) + dim;
        total += expf(stats[0] - maximum) * stats[1];
    }
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        float value = 0.0f;
        for (int j = 0; j < active; j++) {
            const float *split = base + (u64)j * (dim + 2);
            value += expf(split[dim] - maximum) * split[i];
        }
        output[(u64)head * dim + i] = value / total;
    }
}

extern "C" __global__ void attention_combine(const float *partial, float *output,
                                            int dim, int splits, int count) {
    attention_combine_impl(partial, output, dim, splits, count);
}

extern "C" __global__ void attention_combine_decode(const float *partial, float *output,
                                                   int dim, int splits, const int *control) {
    attention_combine_impl(partial, output, dim, splits, control[1] + 1);
}

// Flash-decoding split pass: grid=(kv_heads * parts, chunks), block=256. Block
// (x, c) serves heads / kv_heads / parts (<= 8) query heads of one KV head over the
// contiguous key range c * chunk .. (c + 1) * chunk as a single tile. The chunk is
// sized on the device from the live count (attention_chunk), so short contexts use
// 16-key chunks and long ones fewer, larger chunks; the host sizes the grid so a
// chunk never exceeds ATTENTION_TILE_MAX. Writes (o, max, sum) to
// partial[(head * chunks + c) * (dim + 2)] for attention_combine_chunks.
// Requires dim <= 256 and 256 % dim == 0.
#define ATTENTION_GROUP_MAX 8
#define ATTENTION_TILE_MAX 128
__device__ __forceinline__ int attention_chunk(int count, int chunks,
                                               int granule = ATTENTION_SPLIT) {
    int per = (count + chunks - 1) / chunks;
    return (per + granule - 1) / granule * granule;
}

// Chunk granularity of attention_split_group: fine, so short contexts still spread
// over many blocks.
#define ATTENTION_GROUP_GRANULE 4

template<typename KV>
__device__ __forceinline__ void attention_split_gqa_impl(
    const float *q, const KV *keys, const KV *values, float *partial, int dim,
    int heads, int kv_heads, int count) {
    __shared__ float scores[ATTENTION_GROUP_MAX][ATTENTION_TILE_MAX];
    __shared__ float stats[ATTENTION_GROUP_MAX][2];
    int parts = gridDim.x / kv_heads, kv_head = blockIdx.x / parts;
    int group = heads / kv_heads / parts, thread = threadIdx.x;
    int lane = thread % 32, warp = thread / 32;
    int first_head = kv_head * group * parts + (blockIdx.x % parts) * group;
    int chunks = gridDim.y, chunk = attention_chunk(count, chunks);
    int first = blockIdx.y * chunk;
    if (first >= count) return;
    int tile = min(chunk, count - first);
    for (int pair = warp; pair < group * tile; pair += 8) {
        int h = pair / tile, t = pair % tile;
        const KV *key = keys + ((u64)(first + t) * kv_heads + kv_head) * dim;
        const float *query = q + (u64)(first_head + h) * dim;
        float sum = 0.0f;
        for (int j = lane; j < dim; j += 32) sum += query[j] * kv_load(key, j);
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2)
            sum += __shfl_xor_sync(0xffffffff, sum, offset);
        if (lane == 0) scores[h][t] = sum / sqrtf((float)dim);
    }
    __syncthreads();
    if (warp < group) {
        float maximum = -__int_as_float(0x7f800000);
        for (int t = lane; t < tile; t += 32) maximum = fmaxf(maximum, scores[warp][t]);
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2)
            maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffff, maximum, offset));
        float total = 0.0f;
        for (int t = lane; t < tile; t += 32) {
            float p = expf(scores[warp][t] - maximum);
            scores[warp][t] = p;
            total += p;
        }
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2)
            total += __shfl_xor_sync(0xffffffff, total, offset);
        if (lane == 0) {
            stats[warp][0] = maximum;
            stats[warp][1] = total;
        }
    }
    __syncthreads();
    int i = thread % dim;
    for (int h = thread / dim; h < group; h += blockDim.x / dim) {
        float value = 0.0f;
        for (int t = 0; t < tile; t++)
            value += scores[h][t] * kv_load(values, ((u64)(first + t) * kv_heads + kv_head) * dim + i);
        float *out = partial + ((u64)(first_head + h) * chunks + blockIdx.y) * (dim + 2);
        out[i] = value;
        if (i == 0) {
            out[dim] = stats[h][0];
            out[dim + 1] = stats[h][1];
        }
    }
}

extern "C" __global__ void attention_split_gqa(const float *q, const float *keys,
                                              const float *values, float *partial, int dim,
                                              int heads, int kv_heads, int count) {
    attention_split_gqa_impl(q, keys, values, partial, dim, heads, kv_heads, count);
}

extern "C" __global__ void attention_split_gqa_decode(const float *q, const float *keys,
                                                     const float *values, float *partial,
                                                     int dim, int heads, int kv_heads,
                                                     const int *control) {
    attention_split_gqa_impl(q, keys, values, partial, dim, heads, kv_heads, control[1] + 1);
}

#ifdef CUDA_KV_F16
extern "C" __global__ void attention_split_gqa_f16(const float *q, const kv_f16 *keys,
                                                  const kv_f16 *values, float *partial, int dim,
                                                  int heads, int kv_heads, int count) {
    attention_split_gqa_impl(q, keys, values, partial, dim, heads, kv_heads, count);
}

extern "C" __global__ void attention_split_gqa_decode_f16(const float *q, const kv_f16 *keys,
                                                         const kv_f16 *values, float *partial,
                                                         int dim, int heads, int kv_heads,
                                                         const int *control) {
    attention_split_gqa_impl(q, keys, values, partial, dim, heads, kv_heads, control[1] + 1);
}
#endif

// Pass 2 for attention_split_gqa: grid=(heads), block=256, over the active chunks.
__device__ __forceinline__ void attention_combine_chunks_impl(const float *partial,
                                                             float *output, int dim,
                                                             int chunks, int count) {
    int chunk = attention_chunk(count, chunks), active = (count + chunk - 1) / chunk;
    const float *base = partial + (u64)blockIdx.x * chunks * (dim + 2);
    float maximum = -__int_as_float(0x7f800000);
    for (int j = 0; j < active; j++) maximum = fmaxf(maximum, base[(u64)j * (dim + 2) + dim]);
    float total = 0.0f;
    for (int j = 0; j < active; j++) {
        const float *stats = base + (u64)j * (dim + 2) + dim;
        total += expf(stats[0] - maximum) * stats[1];
    }
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        float value = 0.0f;
        for (int j = 0; j < active; j++) {
            const float *split = base + (u64)j * (dim + 2);
            value += expf(split[dim] - maximum) * split[i];
        }
        output[(u64)blockIdx.x * dim + i] = value / total;
    }
}

extern "C" __global__ void attention_combine_chunks(const float *partial, float *output,
                                                   int dim, int chunks, int count) {
    attention_combine_chunks_impl(partial, output, dim, chunks, count);
}

extern "C" __global__ void attention_combine_chunks_decode(const float *partial,
                                                          float *output, int dim,
                                                          int chunks, const int *control) {
    attention_combine_chunks_impl(partial, output, dim, chunks, control[1] + 1);
}

#define ATTENTION_KV_DIM 128

#if defined(CUDA_PACKED_DECODE)
// Single-head split pass with vector loads: grid=(heads, chunks), block=256,
// dim % 4 == 0 and dim <= 128 (lane l owns dims 4l..4l+3). Chunk c covers keys
// c * chunk .. (c + 1) * chunk (attention_chunk, <= ATTENTION_TILE_MAX). Warp w
// scores keys w, w + 8, ... and later accumulates their values; the 8 warp
// partials are reduced in shared memory. Same partial layout and combine as
// attention_split_gqa. F32 arithmetic; summation order differs.
template<typename KV>
__device__ __forceinline__ void attention_split_head_impl(
    const float *q, const KV *keys, const KV *values, float *partial, int dim,
    int heads, int kv_heads, int count) {
    __shared__ float scores[ATTENTION_TILE_MAX];
    __shared__ float reduce[8][ATTENTION_KV_DIM];
    __shared__ float stats[2];
    int head = blockIdx.x, kv_head = head / (heads / kv_heads), thread = threadIdx.x;
    int lane = thread % 32, warp = thread / 32;
    int chunks = gridDim.y, chunk = attention_chunk(count, chunks);
    int first = blockIdx.y * chunk;
    if (first >= count) return;
    int tile = min(chunk, count - first);
    bool active = 4 * lane < dim;
    float4 query = active ? *(const float4 *)(q + (u64)head * dim + 4 * lane)
                          : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float scale = sqrtf((float)dim);
    // Four keys per warp step so their loads are in flight together.
    for (int base = warp; base < tile; base += 32) {
        float4 k[4];
        #pragma unroll
        for (int u = 0; u < 4; u++) {
            int t = base + 8 * u;
            k[u] = active && t < tile
                ? kv_load4(keys, ((u64)(first + t) * kv_heads + kv_head) * dim + 4 * lane)
                : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        }
        #pragma unroll
        for (int u = 0; u < 4; u++) {
            float sum = warp_sum(query.x * k[u].x + query.y * k[u].y + query.z * k[u].z
                                 + query.w * k[u].w);
            int t = base + 8 * u;
            if (lane == 0 && t < tile) scores[t] = sum / scale;
        }
    }
    __syncthreads();
    if (warp == 0) {
        float maximum = -__int_as_float(0x7f800000);
        for (int t = lane; t < tile; t += 32) maximum = fmaxf(maximum, scores[t]);
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2)
            maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffff, maximum, offset));
        float total = 0.0f;
        for (int t = lane; t < tile; t += 32) {
            float p = expf(scores[t] - maximum);
            scores[t] = p;
            total += p;
        }
        total = warp_sum(total);
        if (lane == 0) {
            stats[0] = maximum;
            stats[1] = total;
        }
    }
    __syncthreads();
    float4 acc = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    if (active) {
        for (int base = warp; base < tile; base += 32) {
            float4 v[4];
            #pragma unroll
            for (int u = 0; u < 4; u++) {
                int t = base + 8 * u;
                v[u] = t < tile
                    ? kv_load4(values, ((u64)(first + t) * kv_heads + kv_head) * dim + 4 * lane)
                    : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            }
            #pragma unroll
            for (int u = 0; u < 4; u++) {
                int t = base + 8 * u;
                float p = t < tile ? scores[t] : 0.0f;
                acc.x += p * v[u].x;
                acc.y += p * v[u].y;
                acc.z += p * v[u].z;
                acc.w += p * v[u].w;
            }
        }
        reduce[warp][4 * lane] = acc.x;
        reduce[warp][4 * lane + 1] = acc.y;
        reduce[warp][4 * lane + 2] = acc.z;
        reduce[warp][4 * lane + 3] = acc.w;
    }
    __syncthreads();
    float *out = partial + ((u64)head * chunks + blockIdx.y) * (dim + 2);
    if (thread < dim) {
        float value = 0.0f;
        #pragma unroll
        for (int w = 0; w < 8; w++) value += reduce[w][thread];
        out[thread] = value;
    }
    if (thread == 0) {
        out[dim] = stats[0];
        out[dim + 1] = stats[1];
    }
}

extern "C" __global__ void attention_split_head(const float *q, const float *keys,
                                               const float *values, float *partial, int dim,
                                               int heads, int kv_heads, int count) {
    attention_split_head_impl(q, keys, values, partial, dim, heads, kv_heads, count);
}

extern "C" __global__ void attention_split_head_decode(const float *q, const float *keys,
                                                      const float *values, float *partial,
                                                      int dim, int heads, int kv_heads,
                                                      const int *control) {
    attention_split_head_impl(q, keys, values, partial, dim, heads, kv_heads, control[1] + 1);
}

#ifdef CUDA_KV_F16
extern "C" __global__ void attention_split_head_f16(const float *q, const kv_f16 *keys,
                                                   const kv_f16 *values, float *partial,
                                                   int dim, int heads, int kv_heads, int count) {
    attention_split_head_impl(q, keys, values, partial, dim, heads, kv_heads, count);
}

extern "C" __global__ void attention_split_head_decode_f16(const float *q, const kv_f16 *keys,
                                                          const kv_f16 *values, float *partial,
                                                          int dim, int heads, int kv_heads,
                                                          const int *control) {
    attention_split_head_impl(q, keys, values, partial, dim, heads, kv_heads, control[1] + 1);
}
#endif

// Grouped split pass, one warp per query head: grid=(kv_heads, chunks), block =
// 32 * (heads / kv_heads) threads (<= 256), dim % 4 == 0 and dim <= 128 (lane l owns
// dims 4l..4l+3). Every warp of a block reads the same key/value rows, so L1 serves
// the group's reuse and L2 sees each row once per block. Warp-local online softmax
// over 32-key steps; no block barriers. Same partial layout and combine as
// attention_split_head.
template<typename KV>
__device__ __forceinline__ void attention_split_group_impl(
    const float *q, const KV *keys, const KV *values, float *partial, int dim,
    int heads, int kv_heads, int count) {
    int kv_head = blockIdx.x, group = heads / kv_heads;
    int lane = threadIdx.x % 32, head = kv_head * group + threadIdx.x / 32;
    int chunks = gridDim.y, chunk = attention_chunk(count, chunks, ATTENTION_GROUP_GRANULE);
    int begin = blockIdx.y * chunk, end = min(count, begin + chunk);
    if (begin >= end) return;
    bool active = 4 * lane < dim;
    float4 query = active ? *(const float4 *)(q + (u64)head * dim + 4 * lane)
                          : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float scale = sqrtf((float)dim), maximum = -__int_as_float(0x7f800000), total = 0.0f;
    float4 acc = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    for (int base = begin; base < end; base += 32) {
        int n = min(32, end - base);
        float mine = -__int_as_float(0x7f800000);
        for (int u = 0; u < n; u += 4) {
            float4 k[4];
            #pragma unroll
            for (int j = 0; j < 4; j++)
                k[j] = active && u + j < n
                    ? kv_load4(keys, ((u64)(base + u + j) * kv_heads + kv_head) * dim + 4 * lane)
                    : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            #pragma unroll
            for (int j = 0; j < 4; j++) {
                float sum = warp_sum(query.x * k[j].x + query.y * k[j].y + query.z * k[j].z
                                     + query.w * k[j].w);
                if (lane == u + j && u + j < n) mine = sum / scale;
            }
        }
        float step = mine;
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2)
            step = fmaxf(step, __shfl_xor_sync(0xffffffff, step, offset));
        float updated = fmaxf(maximum, step), rescale = expf(maximum - updated);
        float p = lane < n ? expf(mine - updated) : 0.0f;
        total = total * rescale + warp_sum(p);
        maximum = updated;
        acc.x *= rescale;
        acc.y *= rescale;
        acc.z *= rescale;
        acc.w *= rescale;
        for (int u = 0; u < n; u += 4) {
            float4 v[4];
            #pragma unroll
            for (int j = 0; j < 4; j++)
                v[j] = active && u + j < n
                    ? kv_load4(values, ((u64)(base + u + j) * kv_heads + kv_head) * dim + 4 * lane)
                    : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            #pragma unroll
            for (int j = 0; j < 4; j++) {
                float pj = __shfl_sync(0xffffffff, p, u + j);
                acc.x += pj * v[j].x;
                acc.y += pj * v[j].y;
                acc.z += pj * v[j].z;
                acc.w += pj * v[j].w;
            }
        }
    }
    float *out = partial + ((u64)head * chunks + blockIdx.y) * (dim + 2);
    if (active) {
        out[4 * lane] = acc.x;
        out[4 * lane + 1] = acc.y;
        out[4 * lane + 2] = acc.z;
        out[4 * lane + 3] = acc.w;
    }
    if (lane == 0) {
        out[dim] = maximum;
        out[dim + 1] = total;
    }
}

extern "C" __global__ void attention_split_group(const float *q, const float *keys,
                                                const float *values, float *partial, int dim,
                                                int heads, int kv_heads, int count) {
    attention_split_group_impl(q, keys, values, partial, dim, heads, kv_heads, count);
}

extern "C" __global__ void attention_split_group_decode(const float *q, const float *keys,
                                                       const float *values, float *partial,
                                                       int dim, int heads, int kv_heads,
                                                       const int *control) {
    attention_split_group_impl(q, keys, values, partial, dim, heads, kv_heads, control[1] + 1);
}

// Batched chain/tree variant of attention_split_group (grid = (kv_heads, chunks, tokens)).
// Bit-exact with attention_split_group for every token in a speculative chain or tree.
template<typename KV>
__device__ __forceinline__ void attention_split_group_batch_impl(
    const float *q, const KV *keys, const KV *values, float *partial, int dim,
    int heads, int kv_heads, int prefix_len, const unsigned long long *tree_mask) {
    int kv_head = blockIdx.x, tok = blockIdx.z, group = heads / kv_heads;
    int lane = threadIdx.x % 32, head = kv_head * group + threadIdx.x / 32;
    int anc[16];
    int count;
    if (tree_mask != nullptr) {
        unsigned long long m = tree_mask[tok];
        int n_anc = 0;
        while (m != 0ULL && n_anc < 16) {
            int b = __ffsll((long long)m) - 1;
            anc[n_anc++] = b;
            m &= m - 1ULL;
        }
        count = prefix_len + n_anc;
    } else {
        count = prefix_len + tok + 1;
    }
    int chunks = gridDim.y, chunk = attention_chunk(count, chunks, ATTENTION_GROUP_GRANULE);
    int begin = blockIdx.y * chunk, end = min(count, begin + chunk);
    if (begin >= end) return;
    const float *q_tok = q + ((u64)tok * heads + head) * dim;
    bool active = 4 * lane < dim;
    float4 query = active ? *(const float4 *)(q_tok + 4 * lane)
                          : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float scale = sqrtf((float)dim), maximum = -__int_as_float(0x7f800000), total = 0.0f;
    float4 acc = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    for (int base = begin; base < end; base += 32) {
        int n = min(32, end - base);
        float mine = -__int_as_float(0x7f800000);
        for (int u = 0; u < n; u += 4) {
            float4 k[4];
            #pragma unroll
            for (int j = 0; j < 4; j++) {
                if (active && u + j < n) {
                    int idx = base + u + j;
                    int row = (tree_mask != nullptr && idx >= prefix_len)
                        ? (prefix_len + anc[idx - prefix_len]) : idx;
                    k[j] = kv_load4(keys, ((u64)row * kv_heads + kv_head) * dim + 4 * lane);
                } else {
                    k[j] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                }
            }
            #pragma unroll
            for (int j = 0; j < 4; j++) {
                float sum = warp_sum(query.x * k[j].x + query.y * k[j].y + query.z * k[j].z
                                     + query.w * k[j].w);
                if (lane == u + j && u + j < n) mine = sum / scale;
            }
        }
        float step = mine;
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2)
            step = fmaxf(step, __shfl_xor_sync(0xffffffff, step, offset));
        float updated = fmaxf(maximum, step), rescale = expf(maximum - updated);
        float p = lane < n ? expf(mine - updated) : 0.0f;
        total = total * rescale + warp_sum(p);
        maximum = updated;
        acc.x *= rescale;
        acc.y *= rescale;
        acc.z *= rescale;
        acc.w *= rescale;
        for (int u = 0; u < n; u += 4) {
            float4 v[4];
            #pragma unroll
            for (int j = 0; j < 4; j++) {
                if (active && u + j < n) {
                    int idx = base + u + j;
                    int row = (tree_mask != nullptr && idx >= prefix_len)
                        ? (prefix_len + anc[idx - prefix_len]) : idx;
                    v[j] = kv_load4(values, ((u64)row * kv_heads + kv_head) * dim + 4 * lane);
                } else {
                    v[j] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                }
            }
            #pragma unroll
            for (int j = 0; j < 4; j++) {
                float pj = __shfl_sync(0xffffffff, p, u + j);
                acc.x += pj * v[j].x;
                acc.y += pj * v[j].y;
                acc.z += pj * v[j].z;
                acc.w += pj * v[j].w;
            }
        }
    }
    float *out = partial + (((u64)tok * heads + head) * chunks + blockIdx.y) * (dim + 2);
    if (active) {
        out[4 * lane] = acc.x;
        out[4 * lane + 1] = acc.y;
        out[4 * lane + 2] = acc.z;
        out[4 * lane + 3] = acc.w;
    }
    if (lane == 0) {
        out[dim] = maximum;
        out[dim + 1] = total;
    }
}

extern "C" __global__ void attention_split_group_batch(
    const float *q, const float *keys, const float *values, float *partial, int dim,
    int heads, int kv_heads, int prefix_len, const unsigned long long *tree_mask) {
    attention_split_group_batch_impl(q, keys, values, partial, dim, heads, kv_heads, prefix_len,
                                     tree_mask);
}

extern "C" __global__ void attention_split_group_batch_graph(
    const float *q, const float *keys, const float *values, float *partial, int dim,
    int heads, int kv_heads, const int *control, const unsigned long long *tree_mask) {
    attention_split_group_batch_impl(q, keys, values, partial, dim, heads, kv_heads, control[1],
                                     tree_mask);
}

#ifdef CUDA_KV_F16
extern "C" __global__ void attention_split_group_f16(const float *q, const kv_f16 *keys,
                                                    const kv_f16 *values, float *partial,
                                                    int dim, int heads, int kv_heads, int count) {
    attention_split_group_impl(q, keys, values, partial, dim, heads, kv_heads, count);
}

extern "C" __global__ void attention_split_group_decode_f16(const float *q, const kv_f16 *keys,
                                                           const kv_f16 *values, float *partial,
                                                           int dim, int heads, int kv_heads,
                                                           const int *control) {
    attention_split_group_impl(q, keys, values, partial, dim, heads, kv_heads, control[1] + 1);
}

extern "C" __global__ void attention_split_group_batch_f16(
    const float *q, const kv_f16 *keys, const kv_f16 *values, float *partial, int dim,
    int heads, int kv_heads, int prefix_len, const unsigned long long *tree_mask) {
    attention_split_group_batch_impl(q, keys, values, partial, dim, heads, kv_heads, prefix_len,
                                     tree_mask);
}

extern "C" __global__ void attention_split_group_batch_graph_f16(
    const float *q, const kv_f16 *keys, const kv_f16 *values, float *partial, int dim,
    int heads, int kv_heads, const int *control, const unsigned long long *tree_mask) {
    attention_split_group_batch_impl(q, keys, values, partial, dim, heads, kv_heads, control[1],
                                     tree_mask);
}
#endif

// Causal prefill attention (flash style) for dim == 128 and GQA groups <= 8:
// grid = (kv_heads, ceil(tokens / PREFILL_TOKENS)), block = 32 * PREFILL_TOKENS.
// Warp w owns token blockIdx.y * PREFILL_TOKENS + w and all `group` query heads of
// the block's KV head, so every K/V tile staged in shared memory serves
// PREFILL_TOKENS * group query rows. Scores: lane j scores key j of a 32-key tile
// for all rows; online softmax per row; values: lane l owns dims 4l..4l+3. Token t
// sees count + t keys, matching attention_fused.
#define PREFILL_TOKENS 4
#define PREFILL_KEYS 32
#define PREFILL_STRIDE 132

// Warps = PREFILL_TOKENS * Split: Split warps share a token, each owning
// ceil(group / Split) of its query rows (fewer rows per warp, more warps per SM).
template<int Split, typename KV>
__device__ __forceinline__ void attention_prefill_flash_impl(
    const float *q, const KV *keys, const KV *values, float *output, int dim, int heads,
    int kv_heads, int tokens, int count) {
    constexpr int ROWS = ATTENTION_GROUP_MAX / Split;
    __shared__ __align__(16) float tile[PREFILL_KEYS * PREFILL_STRIDE];
    __shared__ __align__(16) float query[PREFILL_TOKENS][ATTENTION_GROUP_MAX][ATTENTION_KV_DIM];
    __shared__ float probability[PREFILL_TOKENS][ATTENTION_GROUP_MAX][PREFILL_KEYS];
    int kv_head = blockIdx.x, group = heads / kv_heads;
    int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    int local = warp / Split, part = warp % Split;
    int per = (group + Split - 1) / Split, row0 = part * per;
    int first = blockIdx.y * PREFILL_TOKENS, token = first + local;
    bool active = token < tokens;
    int limit = active ? count + token : 0;
    int last = min(tokens - 1, first + PREFILL_TOKENS - 1);
    int span = count + last;
    int dim4 = dim / 4;
    for (int i = threadIdx.x; i < PREFILL_TOKENS * group * dim4; i += blockDim.x) {
        int t = i / (group * dim4), rest = i % (group * dim4);
        int r = rest / dim4, c = rest % dim4;
        float4 v = first + t < tokens
            ? *(const float4 *)(q + ((u64)(first + t) * heads + kv_head * group + r) * dim + 4 * c)
            : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        *(float4 *)&query[t][r][4 * c] = v;
    }
    float maximum[ROWS], total[ROWS];
    float4 acc[ROWS];
    #pragma unroll
    for (int r = 0; r < ROWS; r++) {
        maximum[r] = -__int_as_float(0x7f800000);
        total[r] = 0.0f;
        acc[r] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    }
    float scale = sqrtf((float)dim);
    for (int base = 0; base < span; base += PREFILL_KEYS) {
        int n = min(PREFILL_KEYS, span - base);
        __syncthreads();
        for (int i = threadIdx.x; i < PREFILL_KEYS * dim4; i += blockDim.x) {
            int key = i / dim4, c = i % dim4;
            float4 v = key < n ? kv_load4(keys, ((u64)(base + key) * kv_heads + kv_head) * dim
                                                    + 4 * c)
                               : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            *(float4 *)&tile[key * PREFILL_STRIDE + 4 * c] = v;
        }
        __syncthreads();
        if (base < limit) {
            float score[ROWS];
            #pragma unroll
            for (int r = 0; r < ROWS; r++) score[r] = 0.0f;
            for (int d = 0; d < dim; d += 4) {
                float4 k = *(const float4 *)&tile[lane * PREFILL_STRIDE + d];
                #pragma unroll
                for (int r = 0; r < ROWS; r++) {
                    if (r < per && row0 + r < group) {
                        float4 x = *(const float4 *)&query[local][row0 + r][d];
                        score[r] += x.x * k.x + x.y * k.y + x.z * k.z + x.w * k.w;
                    }
                }
            }
            bool valid = base + lane < limit;
            #pragma unroll
            for (int r = 0; r < ROWS; r++) {
                if (r < per && row0 + r < group) {
                    float s = valid ? score[r] / scale : -__int_as_float(0x7f800000);
                    float step = s;
                    #pragma unroll
                    for (int offset = 16; offset > 0; offset /= 2)
                        step = fmaxf(step, __shfl_xor_sync(0xffffffff, step, offset));
                    float updated = fmaxf(maximum[r], step);
                    float rescale = expf(maximum[r] - updated);
                    float p = valid ? expf(s - updated) : 0.0f;
                    total[r] = total[r] * rescale + warp_sum(p);
                    maximum[r] = updated;
                    acc[r].x *= rescale;
                    acc[r].y *= rescale;
                    acc[r].z *= rescale;
                    acc[r].w *= rescale;
                    probability[local][row0 + r][lane] = p;
                }
            }
        }
        __syncthreads();
        for (int i = threadIdx.x; i < PREFILL_KEYS * dim4; i += blockDim.x) {
            int key = i / dim4, c = i % dim4;
            float4 v = key < n ? kv_load4(values, ((u64)(base + key) * kv_heads + kv_head) * dim
                                                    + 4 * c)
                               : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            *(float4 *)&tile[key * PREFILL_STRIDE + 4 * c] = v;
        }
        __syncthreads();
        if (base < limit && 4 * lane < dim) {
            int keys_here = min(n, limit - base);
            for (int j = 0; j < keys_here; j++) {
                float4 v = *(const float4 *)&tile[j * PREFILL_STRIDE + 4 * lane];
                #pragma unroll
                for (int r = 0; r < ROWS; r++) {
                    if (r < per && row0 + r < group) {
                        float p = probability[local][row0 + r][j];
                        acc[r].x += p * v.x;
                        acc[r].y += p * v.y;
                        acc[r].z += p * v.z;
                        acc[r].w += p * v.w;
                    }
                }
            }
        }
    }
    if (!active || 4 * lane >= dim) return;
    #pragma unroll
    for (int r = 0; r < ROWS; r++) {
        if (r < per && row0 + r < group) {
            float4 v = make_float4(acc[r].x / total[r], acc[r].y / total[r], acc[r].z / total[r],
                                   acc[r].w / total[r]);
            *(float4 *)(output + ((u64)token * heads + kv_head * group + row0 + r) * dim
                        + 4 * lane) = v;
        }
    }
}

extern "C" __global__ void __launch_bounds__(128 * PREFILL_TOKENS)
attention_prefill_flash4(const float *q, const float *keys, const float *values, float *output,
                         int heads, int kv_heads, int tokens, int count) {
    attention_prefill_flash_impl<4>(q, keys, values, output, ATTENTION_KV_DIM, heads, kv_heads,
                                    tokens, count);
}

extern "C" __global__ void __launch_bounds__(128 * PREFILL_TOKENS)
attention_prefill_flash4_dim(const float *q, const float *keys, const float *values, float *output,
                             int dim, int heads, int kv_heads, int tokens, int count) {
    attention_prefill_flash_impl<4>(q, keys, values, output, dim, heads, kv_heads, tokens, count);
}

#ifdef CUDA_KV_F16
extern "C" __global__ void __launch_bounds__(128 * PREFILL_TOKENS)
attention_prefill_flash4_f16(const float *q, const kv_f16 *keys, const kv_f16 *values,
                             float *output, int heads, int kv_heads, int tokens, int count) {
    attention_prefill_flash_impl<4>(q, keys, values, output, ATTENTION_KV_DIM, heads, kv_heads,
                                    tokens, count);
}

extern "C" __global__ void __launch_bounds__(128 * PREFILL_TOKENS)
attention_prefill_flash4_dim_f16(const float *q, const kv_f16 *keys, const kv_f16 *values,
                                 float *output, int dim, int heads, int kv_heads, int tokens,
                                 int count) {
    attention_prefill_flash_impl<4>(q, keys, values, output, dim, heads, kv_heads, tokens, count);
}
#endif

// Combine for the chunked split passes, parallel over chunks: grid=(heads),
// block=256, dim % 4 == 0 and dim <= 128. Warp w accumulates chunks w, w + 8, ...
// with float4 loads; partials are reduced in shared memory.
__device__ __forceinline__ void attention_combine_wide_impl(const float *partial,
                                                           float *output, int dim,
                                                           int chunks, int count,
                                                           int granule) {
    __shared__ float weights[1024];
    __shared__ float reduce[8][ATTENTION_KV_DIM];
    __shared__ float denominator;
    int chunk = attention_chunk(count, chunks, granule);
    int active = (count + chunk - 1) / chunk;
    int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    const float *base = partial + (u64)blockIdx.x * chunks * (dim + 2);
    if (warp == 0) {
        float maximum = -__int_as_float(0x7f800000);
        for (int j = lane; j < active; j += 32)
            maximum = fmaxf(maximum, base[(u64)j * (dim + 2) + dim]);
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2)
            maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffff, maximum, offset));
        float total = 0.0f;
        for (int j = lane; j < active; j += 32) {
            const float *stats = base + (u64)j * (dim + 2) + dim;
            float weight = expf(stats[0] - maximum);
            weights[j] = weight;
            total += weight * stats[1];
        }
        total = warp_sum(total);
        if (lane == 0) denominator = total;
    }
    __syncthreads();
    if (4 * lane < dim) {
        float4 acc = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        for (int j = warp; j < active; j += 8) {
            const float *split = base + (u64)j * (dim + 2) + 4 * lane;
            float weight = weights[j];
            // dim + 2 is even, so split rows are 8-byte aligned: two float2 loads.
            float2 a = *(const float2 *)split, b = *(const float2 *)(split + 2);
            acc.x += weight * a.x;
            acc.y += weight * a.y;
            acc.z += weight * b.x;
            acc.w += weight * b.y;
        }
        reduce[warp][4 * lane] = acc.x;
        reduce[warp][4 * lane + 1] = acc.y;
        reduce[warp][4 * lane + 2] = acc.z;
        reduce[warp][4 * lane + 3] = acc.w;
    }
    __syncthreads();
    if (threadIdx.x < dim) {
        float value = 0.0f;
        #pragma unroll
        for (int w = 0; w < 8; w++) value += reduce[w][threadIdx.x];
        output[(u64)blockIdx.x * dim + threadIdx.x] = value / denominator;
    }
}

extern "C" __global__ void attention_combine_wide(const float *partial, float *output,
                                                 int dim, int chunks, int count) {
    attention_combine_wide_impl(partial, output, dim, chunks, count, ATTENTION_SPLIT);
}

extern "C" __global__ void attention_combine_wide_decode(const float *partial,
                                                        float *output, int dim, int chunks,
                                                        const int *control) {
    attention_combine_wide_impl(partial, output, dim, chunks, control[1] + 1, ATTENTION_SPLIT);
}

// Combine for attention_split_group (chunks of ATTENTION_GROUP_GRANULE keys).
extern "C" __global__ void attention_combine_group(const float *partial, float *output,
                                                  int dim, int chunks, int count) {
    attention_combine_wide_impl(partial, output, dim, chunks, count, ATTENTION_GROUP_GRANULE);
}

extern "C" __global__ void attention_combine_group_decode(const float *partial,
                                                         float *output, int dim, int chunks,
                                                         const int *control) {
    attention_combine_wide_impl(partial, output, dim, chunks, control[1] + 1,
                                ATTENTION_GROUP_GRANULE);
}

extern "C" __global__ void attention_combine_group_batch(const float *partial,
                                                         float *output, int dim, int heads,
                                                         int chunks, int prefix_len,
                                                         const unsigned long long *tree_mask) {
    int tok = blockIdx.z;
    int count = (tree_mask != nullptr) ? (prefix_len + __popcll(tree_mask[tok]))
                                       : (prefix_len + tok + 1);
    const float *p_tok = partial + (u64)tok * heads * chunks * (dim + 2);
    float *out_tok = output + (u64)tok * heads * dim;
    attention_combine_wide_impl(p_tok, out_tok, dim, chunks, count, ATTENTION_GROUP_GRANULE);
}

extern "C" __global__ void attention_combine_group_batch_graph(
    const float *partial, float *output, int dim, int heads, int chunks, const int *control,
    const unsigned long long *tree_mask) {
    int prefix_len = control[1];
    int tok = blockIdx.z;
    int count = (tree_mask != nullptr) ? (prefix_len + __popcll(tree_mask[tok]))
                                       : (prefix_len + tok + 1);
    const float *p_tok = partial + (u64)tok * heads * chunks * (dim + 2);
    float *out_tok = output + (u64)tok * heads * dim;
    attention_combine_wide_impl(p_tok, out_tok, dim, chunks, count, ATTENTION_GROUP_GRANULE);
}
#endif

#if defined(CUDA_PACKED_DECODE) && defined(CUDA_KV_F16)
// Keys per pipeline stage of attention_decode_mma (attention_mma.cu); its warp
// chunks are multiples of it.
#define ATTENTION_ROW_BATCH 32

__device__ __forceinline__ int attention_row_chunk(int count, int chunks) {
    int per = (count + chunks - 1) / chunks;
    return max(ATTENTION_ROW_BATCH,
               (per + ATTENTION_ROW_BATCH - 1) / ATTENTION_ROW_BATCH * ATTENTION_ROW_BATCH);
}

__device__ __forceinline__ void kv_copy_async(void *shared, const void *global) {
    unsigned int address = (unsigned int)__cvta_generic_to_shared(shared);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(address), "l"(global));
}

// Combine for attention_decode_mma: grid = (heads), block = 1024, dim == 128.
// Warp w accumulates active chunks w, w + 32, ... (32 in flight per head).
__device__ __forceinline__ void attention_combine_rows_impl(const float *partial, float *output,
                                                           int chunks, int count) {
    __shared__ float weights[1024];
    __shared__ float reduce[32][ATTENTION_KV_DIM];
    __shared__ float denominator;
    const int dim = ATTENTION_KV_DIM;
    int chunk = attention_row_chunk(count, chunks), active = (count + chunk - 1) / chunk;
    int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    const float *base = partial + (u64)blockIdx.x * chunks * (dim + 2);
    if (warp == 0) {
        float maximum = -__int_as_float(0x7f800000);
        for (int j = lane; j < active; j += 32)
            maximum = fmaxf(maximum, base[(u64)j * (dim + 2) + dim]);
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2)
            maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffff, maximum, offset));
        float total = 0.0f;
        for (int j = lane; j < active; j += 32) {
            const float *stats = base + (u64)j * (dim + 2) + dim;
            float weight = expf(stats[0] - maximum);
            weights[j] = weight;
            total += weight * stats[1];
        }
        total = warp_sum(total);
        if (lane == 0) denominator = total;
    }
    __syncthreads();
    float4 acc = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    for (int j = warp; j < active; j += 32) {
        const float *split = base + (u64)j * (dim + 2) + 4 * lane;
        float weight = weights[j];
        float2 a = *(const float2 *)split, b = *(const float2 *)(split + 2);
        acc.x += weight * a.x;
        acc.y += weight * a.y;
        acc.z += weight * b.x;
        acc.w += weight * b.y;
    }
    *(float4 *)&reduce[warp][4 * lane] = acc;
    __syncthreads();
    if (threadIdx.x < dim) {
        float value = 0.0f;
        #pragma unroll
        for (int w = 0; w < 32; w++) value += reduce[w][threadIdx.x];
        output[(u64)blockIdx.x * dim + threadIdx.x] = value / denominator;
    }
}

extern "C" __global__ void __launch_bounds__(1024)
attention_combine_rows(const float *partial, float *output, int dim, int chunks, int count) {
    attention_combine_rows_impl(partial, output, chunks, count);
}

extern "C" __global__ void __launch_bounds__(1024)
attention_combine_rows_decode(const float *partial, float *output, int dim, int chunks,
                              const int *control) {
    attention_combine_rows_impl(partial, output, chunks, control[1] + 1);
}
#endif
