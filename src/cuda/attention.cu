// One block per query/head keeps bounded scores on chip. Match the original
// 256-lane dot and softmax trees rather than changing numerical precision.
__device__ __forceinline__ void attention_fused_impl(const float *q, const float *keys,
                                                    const float *values, float *output,
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
                parts[p] = query[p] * keys[((u64)time * kv_heads + kv_head) * dim + i];
#else
                parts[p] = q[i] * keys[((u64)time * kv_heads + kv_head) * dim + i];
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
            value += scores[time] * values[((u64)time * kv_heads + kv_head) * dim + i];
#else
            value += scores[time] / total * values[((u64)time * kv_heads + kv_head) * dim + i];
#endif
        output[((u64)token * heads + head) * dim + i] = value;
    }
}

extern "C" __global__ void attention_fused(const float *q, const float *keys,
                                          const float *values, float *output,
                                          int dim, int heads, int kv_heads, int count) {
    attention_fused_impl(q, keys, values, output, dim, heads, kv_heads, count);
}

extern "C" __global__ void attention_fused_decode(const float *q, const float *keys,
                                                 const float *values, float *output,
                                                 int dim, int heads, int kv_heads,
                                                 const int *control) {
    attention_fused_impl(q, keys, values, output, dim, heads, kv_heads, control[1] + 1);
}

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
#define ATTENTION_SPLIT 64
__device__ __forceinline__ void attention_split_impl(const float *q, const float *keys,
                                                    const float *values, float *partial,
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
        const float *key = keys + ((u64)(first + t) * kv_heads + kv_head) * dim;
        float sum = 0.0f;
        for (int i = lane; i < dim; i += 32) sum += q[i] * key[i];
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
            value += scores[t] * values[((u64)(first + t) * kv_heads + kv_head) * dim + i];
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
