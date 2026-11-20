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

#if defined(CUDA_QWEN3_SPLIT_DECODE_ATTENTION) || defined(CUDA_QWEN35_SPLIT_DECODE_ATTENTION) || defined(CUDA_LLAMA_SPLIT_DECODE_ATTENTION)
template<int dim>
__device__ __forceinline__ void split_decode_partials(const float *q, const float *keys,
    const float *values, float *output, float *partials, int heads, int kv_heads,
    unsigned tile_size, int count) {
    constexpr int warps = 8;
    __shared__ float scores[256], numerators[warps][dim];
    int tid = threadIdx.x, lane = tid % 32, warp = tid / 32;
    int start = blockIdx.y * tile_size;
    if (start >= count) return;
    if (dim != 64 && count <= tile_size) {
        attention_fused_impl(q, keys, values, output, dim, heads, kv_heads, count);
        return;
    }

    int kv_head = blockIdx.x / (heads / kv_heads);
    int local_count = min(static_cast<int>(tile_size), count - start);
    int stride = kv_heads * dim;
    keys += ((u64)start * kv_heads + kv_head) * dim;
    values += ((u64)start * kv_heads + kv_head) * dim;
    q += blockIdx.x * dim;
    float query[dim / 32];
    #pragma unroll
    for (int i = 0; i < dim / 32; ++i) query[i] = q[lane + 32 * i];
    for (int time = warp; time < local_count; time += warps) {
        const float *key = keys + (u64)time * stride + lane;
        float score;
        if constexpr (dim == 128) {
            score = (query[0] * key[0] + query[2] * key[64])
                  + (query[1] * key[32] + query[3] * key[96]);
        } else {
            score = 0.0f;
            #pragma unroll
            for (int i = 0; i < dim / 32; ++i) score += query[i] * key[32 * i];
        }
        for (int shift = 16; shift > 0; shift /= 2)
            score += __shfl_down_sync(0xffffffff, score, shift);
        if (lane == 0) scores[time] = score / sqrtf((float)dim);
    }
    __syncthreads();
    float score = tid < local_count ? scores[tid] : -__int_as_float(0x7f800000);
    float maximum = score;
    for (int shift = 16; shift > 0; shift /= 2)
        maximum = fmaxf(maximum, __shfl_down_sync(0xffffffff, maximum, shift));
    if (lane == 0) numerators[warp][0] = maximum;
    __syncthreads();
    maximum = numerators[0][0];
    #pragma unroll
    for (int i = 1; i < warps; ++i) maximum = fmaxf(maximum, numerators[i][0]);
    float weight = expf(score - maximum);
    float total = sum_block(weight);
    scores[tid] = weight;
    __syncthreads();

    #pragma unroll
    for (int i = 0; i < dim / 32; ++i) {
        int col = lane + i * 32;
        float sum = 0.0f;
        for (int time = warp; time < local_count; time += warps)
            sum += scores[time] * values[(u64)time * stride + col];
        numerators[warp][col] = sum;
    }
    __syncthreads();
    float *dst = partials + (blockIdx.x * gridDim.y + blockIdx.y) * (dim + 2);
    if (tid < dim) {
        float sum = 0.0f;
        #pragma unroll
        for (int i = 0; i < warps; ++i) sum += numerators[i][tid];
        if (dim == 64 && count == tile_size)
            output[blockIdx.x * dim + tid] = sum / total;
        else
            dst[tid] = sum;
    }
    if (tid == 0 && !(dim == 64 && count == tile_size)) {
        dst[dim] = maximum;
        dst[dim + 1] = total;
    }
}

template<int dim>
__device__ __forceinline__ void split_decode_merge(const float *partials, float *output,
    unsigned tiles, unsigned tile_size, unsigned count) {
    if (count <= tile_size) return;
    unsigned live_tiles = (count + tile_size - 1) / tile_size;
    const float *src = partials + blockIdx.x * tiles * (dim + 2);
    float peak = -__int_as_float(0x7f800000);
    for (unsigned tile = 0; tile < live_tiles; ++tile)
        peak = fmaxf(peak, src[tile * (dim + 2) + dim]);
    float numerator = 0.0f, denominator = 0.0f;
    for (unsigned tile = 0; tile < live_tiles; ++tile) {
        const float *part = src + tile * (dim + 2);
        float scale = expf(part[dim] - peak);
        numerator += part[threadIdx.x] * scale;
        denominator += part[dim + 1] * scale;
    }
    output[blockIdx.x * dim + threadIdx.x] = numerator / denominator;
}

#ifdef CUDA_QWEN3_SPLIT_DECODE_ATTENTION
extern "C" __global__ void attention_decode_partials(const float *q, const float *keys,
    const float *values, float *output, float *partials, int heads, int kv_heads,
    unsigned tile_size, const int *control) {
    split_decode_partials<128>(q, keys, values, output, partials, heads, kv_heads,
                               tile_size, control[1] + 1);
}
extern "C" __global__ void attention_decode_merge(const float *partials, float *output,
    unsigned tiles, unsigned tile_size, const int *control) {
    split_decode_merge<128>(partials, output, tiles, tile_size, control[1] + 1);
}
#endif

#ifdef CUDA_LLAMA_SPLIT_DECODE_ATTENTION
extern "C" __global__ void llama_split_decode_partials(const float *q, const float *keys,
    const float *values, float *output, float *partials, int heads, int kv_heads,
    unsigned tile_size, int count) {
    split_decode_partials<64>(q, keys, values, output, partials, heads, kv_heads, tile_size, count);
}
extern "C" __global__ void llama_split_decode_merge(const float *partials, float *output,
    unsigned tiles, unsigned tile_size, int count) {
    split_decode_merge<64>(partials, output, tiles, tile_size, count);
}
#endif

#ifdef CUDA_QWEN35_SPLIT_DECODE_ATTENTION
extern "C" __global__ void qwen35_split_decode_partials(const float *q, const float *keys,
    const float *values, float *output, float *partials, int heads, int kv_heads,
    unsigned tile_size, int count, const int *control) {
    split_decode_partials<256>(q, keys, values, output, partials, heads, kv_heads,
                               tile_size, control ? control[1] + 1 : count);
}
extern "C" __global__ void qwen35_split_decode_merge(const float *partials, float *output,
    unsigned tiles, unsigned tile_size, int count, const int *control) {
    split_decode_merge<256>(partials, output, tiles, tile_size, control ? control[1] + 1 : count);
}
#endif
#endif

#if defined(CUDA_M8_PROTOTYPES) || defined(CUDA_ATTENTION_QUERY_PAIR_RESEARCH) || defined(CUDA_LLAMA_PREFILL_QUERY_PAIR) || defined(CUDA_QWEN35_PREFILL_QUERY_PAIR)
template<int Dim>
__device__ __forceinline__ void attention_query_pair(const float *q, const float *keys,
    const float *values, float *output, int heads, int kv_heads, int count) {
    __shared__ float scores[2][1024], maxima[2][256], peak[2];
    int head = blockIdx.x, token = blockIdx.y * 2, tid = threadIdx.x;
    int lane = tid % 32, warp = tid / 32, kv_head = head / (heads / kv_heads);
    int limit = count + token;
    u64 query_offset = ((u64)token * heads + head) * Dim;
    constexpr int parts = Dim == 256 ? 8 : 4;
    float query[2][parts];
    #pragma unroll
    for (int p = 0; p < Dim / 32; ++p) {
        int i = lane + p * 32;
        query[0][p] = q[query_offset + i];
        query[1][p] = q[query_offset + heads * Dim + i];
    }
    for (int time = warp; time <= limit; time += 8) {
        float products[2][parts] = {};
        #pragma unroll
        for (int p = 0; p < Dim / 32; ++p) {
            float key = keys[((u64)time * kv_heads + kv_head) * Dim + lane + p * 32];
            products[0][p] = query[0][p] * key;
            products[1][p] = query[1][p] * key;
        }
        #pragma unroll
        for (int j = 0; j < 2; ++j) {
            if (time >= limit + j) continue;
            // The original 256-lane tree includes four inactive products.
            float sum;
            if constexpr (Dim == 256) {
                const float *p = products[j];
                sum = ((p[0] + p[4]) + (p[2] + p[6]))
                    + ((p[1] + p[5]) + (p[3] + p[7]));
            } else {
                sum = (__fadd_rn(products[j][0], 0.0f) + __fadd_rn(products[j][2], 0.0f))
                    + (__fadd_rn(products[j][1], 0.0f) + __fadd_rn(products[j][3], 0.0f));
            }
            #pragma unroll
            for (int stride = 16; stride > 0; stride /= 2)
                sum += __shfl_down_sync(0xffffffff, sum, stride);
            if (lane == 0) scores[j][time] = sum / sqrtf((float)Dim);
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
    if (tid < Dim) {
        float a = 0.0f, b = 0.0f;
        for (int time = 0; time < limit; ++time) {
            float value = values[((u64)time * kv_heads + kv_head) * Dim + tid];
            a += scores[0][time] * value;
            b += scores[1][time] * value;
        }
        b += scores[1][limit] * values[((u64)limit * kv_heads + kv_head) * Dim + tid];
        output[query_offset + tid] = a;
        output[query_offset + heads * Dim + tid] = b;
    }
}
#if defined(CUDA_QWEN3_BATCHED_ATTENTION) || defined(CUDA_LLAMA_BATCHED_ATTENTION)
template<int Group = 1>
__device__ void batched_causal_softmax(float *scores, int n) {
#ifdef CUDA_BATCHED_ATTENTION_WARP_SOFTMAX
    const int lane = threadIdx.x % 32, query = blockIdx.x * 8 + threadIdx.x / 32;
    if (query >= n) return;
    const int row_idx = Group > 1 ? (blockIdx.y / Group * n + query) * Group + blockIdx.y % Group
                                : blockIdx.y * n + query;
    float *row = scores + row_idx * n;
    float x[16], peak = -__int_as_float(0x7f800000);
    #pragma unroll
    for (int i = 0; i < 16; ++i) {
        const int col = i * 32 + lane;
        x[i] = col <= query ? row[col] : -__int_as_float(0x7f800000);
        peak = fmaxf(peak, x[i]);
    }
    for (int offset = 16; offset; offset /= 2)
        peak = fmaxf(peak, __shfl_xor_sync(0xffffffff, peak, offset));
    float sum = 0.0f;
    #pragma unroll
    for (int i = 0; i < 16; ++i) {
        x[i] = expf(x[i] - peak);
        sum += x[i];
    }
    for (int offset = 16; offset; offset /= 2)
        sum += __shfl_xor_sync(0xffffffff, sum, offset);
    #pragma unroll
    for (int i = 0; i < 16; ++i) {
        const int col = i * 32 + lane;
        if (col < n) row[col] = x[i] / sum;
    }
#else
    const int tid = threadIdx.x, query = blockIdx.x;
    const int row_idx = Group > 1 ? (blockIdx.y / Group * n + query) * Group + blockIdx.y % Group
                                : blockIdx.y * n + query;
    float *row = scores + row_idx * n;
    __shared__ float maxima[256];
    float peak = -__int_as_float(0x7f800000);
    for (int i = tid; i <= query; i += 256) peak = fmaxf(peak, row[i]);
    maxima[tid] = peak;
    __syncthreads();
    for (int s = 128; s; s /= 2) {
        if (tid < s) maxima[tid] = fmaxf(maxima[tid], maxima[tid + s]);
        __syncthreads();
    }
    peak = maxima[0];
    float total = 0.0f;
    for (int i = tid; i <= query; i += 256) {
        row[i] = expf(row[i] - peak);
        total += row[i];
    }
    total = sum_block(total);
    for (int i = tid; i < n; i += 256) row[i] = i <= query ? row[i] / total : 0.0f;
#endif
}
extern "C" __global__ void qwen3_batched_causal_softmax(float *scores, int n) {
    batched_causal_softmax(scores, n);
}
#ifdef CUDA_LLAMA_GROUPED_PREFILL_ATTENTION
extern "C" __global__ void llama_grouped_causal_softmax(float *scores, int n) {
    batched_causal_softmax<4>(scores, n);
}
template<bool Pack>
__device__ void llama_grouped_layout(const float *src, float *dst, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int token = i / 2048, group = i / 256 % 8;
    const int j = (group * n + token) * 256 + i % 256;
    if constexpr (Pack) dst[j] = src[i];
    else dst[i] = src[j];
}
extern "C" __global__ void llama_grouped_q_pack(const float *src, float *dst, int n) {
    llama_grouped_layout<true>(src, dst, n);
}
extern "C" __global__ void llama_grouped_out_unpack(const float *src, float *dst, int n) {
    llama_grouped_layout<false>(src, dst, n);
}
#endif
#ifdef CUDA_QWEN3_GROUPED_PREFILL_ATTENTION
extern "C" __global__ void qwen3_grouped_causal_softmax(float *scores, int n) {
    batched_causal_softmax<2>(scores, n);
}
#endif
#endif
#if defined(CUDA_M8_PROTOTYPES) || defined(CUDA_ATTENTION_QUERY_PAIR_RESEARCH)
extern "C" __global__ void attention_fused_query_pair(const float *q, const float *keys,
    const float *values, float *output, int heads, int kv_heads, int count) {
    attention_query_pair<128>(q, keys, values, output, heads, kv_heads, count);
}
#endif
#ifdef CUDA_LLAMA_PREFILL_QUERY_PAIR
extern "C" __global__ void attention_fused_query_pair_64(const float *q, const float *keys,
    const float *values, float *output, int heads, int kv_heads, int count) {
    attention_query_pair<64>(q, keys, values, output, heads, kv_heads, count);
}
#endif
#ifdef CUDA_QWEN35_PREFILL_QUERY_PAIR
extern "C" __global__ void attention_fused_query_pair_256(const float *q, const float *keys,
    const float *values, float *output, int heads, int kv_heads, int count) {
    attention_query_pair<256>(q, keys, values, output, heads, kv_heads, count);
}
#endif
#endif
