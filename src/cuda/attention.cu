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
