#ifdef LOCAL_DECODE_BENCH
template<int Kind>
__device__ float decode_row(const u8 *w, const float *x, u64 offset, int width) {
    float parts[8] = {};
    int lane = threadIdx.x % 32;
    for (int first = lane; first < width; first += 256) {
        #pragma unroll
        for (int p = 0; p < 8; p++) {
            int i = first + p * 32;
            if (i < width) parts[p] += weight(w, Kind, offset + i) * x[i];
        }
    }
    parts[0] += parts[4];
    parts[1] += parts[5];
    parts[2] += parts[6];
    parts[3] += parts[7];
    parts[0] += parts[2];
    parts[1] += parts[3];
    float sum = parts[0] + parts[1];
    for (int stride = 16; stride; stride /= 2)
        sum += __shfl_down_sync(0xffffffff, sum, stride);
    return sum;
}

extern "C" __global__ void matvec_decode(const u8 *w, int kind, const float *x, float *y,
                                         int width, int rows, const int *ids, int moe, int batched) {
    int row = blockIdx.x * 8 + threadIdx.x / 32, selected = blockIdx.y;
    if (row >= rows) return;
    int expert = moe ? ids[selected] : 0;
    u64 offset = ((u64)expert * rows + row) * width;
    x += batched ? (u64)selected * width : 0;
    float sum;
    switch (kind) {
        case 0: sum = decode_row<0>(w, x, offset, width); break;
        case 1: sum = decode_row<1>(w, x, offset, width); break;
        case 6: sum = decode_row<6>(w, x, offset, width); break;
        case 8: sum = decode_row<8>(w, x, offset, width); break;
        case 12: sum = decode_row<12>(w, x, offset, width); break;
        case 14: sum = decode_row<14>(w, x, offset, width); break;
        default: sum = __int_as_float(0x7fffffff); break;
    }
    if (threadIdx.x % 32 == 0) y[(u64)selected * rows + row] = sum;
}

#endif

__device__ bool better_token(float value, int id, float best, int best_id) {
    return id >= 0 && (best_id < 0 || value > best || (value == best && id < best_id));
}

extern "C" __global__ void greedy_partials(const float *logits, int count,
                                           float *scores, int *ids, int *status) {
    __shared__ float values[256];
    __shared__ int indices[256];
    int lane = threadIdx.x, index = blockIdx.x * 256 + lane;
    float value = index < count ? logits[index] : -__int_as_float(0x7f800000);
    if (index < count && !isfinite(value)) atomicExch(status, 1);
    values[lane] = value;
    indices[lane] = index < count ? index : -1;
    __syncthreads();
    for (int stride = 128; stride; stride /= 2) {
        if (lane < stride && better_token(values[lane + stride], indices[lane + stride],
                                         values[lane], indices[lane])) {
            values[lane] = values[lane + stride];
            indices[lane] = indices[lane + stride];
        }
        __syncthreads();
    }
    if (lane == 0) { scores[blockIdx.x] = values[0]; ids[blockIdx.x] = indices[0]; }
}

extern "C" __global__ void greedy_finish(const float *scores, const int *ids, int count,
                                         int *control, int *tokens, int *status) {
    __shared__ float values[256];
    __shared__ int indices[256];
    int lane = threadIdx.x, best = -1;
    float value = -__int_as_float(0x7f800000);
    for (int i = lane; i < count; i += 256) {
        if (better_token(scores[i], ids[i], value, best)) { value = scores[i]; best = ids[i]; }
    }
    values[lane] = value;
    indices[lane] = best;
    __syncthreads();
    for (int stride = 128; stride; stride /= 2) {
        if (lane < stride && better_token(values[lane + stride], indices[lane + stride],
                                         values[lane], indices[lane])) {
            values[lane] = values[lane + stride];
            indices[lane] = indices[lane + stride];
        }
        __syncthreads();
    }
    if (lane == 0) {
        int slot = control[2];
        if (slot < 0 || slot >= 8 || indices[0] < 0) atomicExch(status, 1);
        else tokens[slot] = indices[0];
        tokens[8] = *status;
        control[0] = indices[0];
        control[1]++;
        control[2]++;
    }
}

// Single-launch batched greedy argmax for speculative verification (1 <= n <= 16).
// Grid: (n, 1, 1), Block: 256 threads. Writes argmax of row `blockIdx.x` to `out[blockIdx.x]`.
extern "C" __global__ void greedy_batch_argmax(const float *logits, int vocab, int *out,
                                               int *status) {
    __shared__ float values[256];
    __shared__ int indices[256];
    int row = blockIdx.x;
    int lane = threadIdx.x;
    const float *base = logits + (u64)row * vocab;
    int best = -1;
    float value = -__int_as_float(0x7f800000);
    int vec_n = vocab >> 2;
    const float4 *base4 = (const float4 *)base;
    for (int i = lane; i < vec_n; i += 256) {
        float4 v = __ldg(base4 + i);
        int idx = i << 2;
        if (better_token(v.x, idx, value, best)) { value = v.x; best = idx; }
        if (better_token(v.y, idx + 1, value, best)) { value = v.y; best = idx + 1; }
        if (better_token(v.z, idx + 2, value, best)) { value = v.z; best = idx + 2; }
        if (better_token(v.w, idx + 3, value, best)) { value = v.w; best = idx + 3; }
    }
    for (int i = (vec_n << 2) + lane; i < vocab; i += 256) {
        float v = __ldg(base + i);
        if (better_token(v, i, value, best)) { value = v; best = i; }
    }
    values[lane] = value;
    indices[lane] = best;
    __syncthreads();
    for (int stride = 128; stride; stride >>= 1) {
        if (lane < stride && better_token(values[lane + stride], indices[lane + stride],
                                         values[lane], indices[lane])) {
            values[lane] = values[lane + stride];
            indices[lane] = indices[lane + stride];
        }
        __syncthreads();
    }
    if (lane == 0) {
        if (indices[0] < 0) atomicExch(status, 1);
        out[row] = indices[0];
    }
}

// Single-launch EAGLE-3 argmax + d2t mapping on GPU (zero CPU sync between draft steps!).
// Grid: (1, 1, 1), Block: 256 threads. Writes `draft_id + d2t[draft_id]` to `out[slot]`.
extern "C" __global__ void eagle3_argmax_d2t(const float *logits, int vocab, const int *d2t,
                                             int *out, int slot, int *status) {
    __shared__ float values[256];
    __shared__ int indices[256];
    int lane = threadIdx.x;
    int best = -1;
    float value = -__int_as_float(0x7f800000);
    int vec_n = vocab >> 2;
    const float4 *base4 = (const float4 *)logits;
    for (int i = lane; i < vec_n; i += 256) {
        float4 v = __ldg(base4 + i);
        int idx = i << 2;
        if (better_token(v.x, idx, value, best)) { value = v.x; best = idx; }
        if (better_token(v.y, idx + 1, value, best)) { value = v.y; best = idx + 1; }
        if (better_token(v.z, idx + 2, value, best)) { value = v.z; best = idx + 2; }
        if (better_token(v.w, idx + 3, value, best)) { value = v.w; best = idx + 3; }
    }
    for (int i = (vec_n << 2) + lane; i < vocab; i += 256) {
        float v = __ldg(logits + i);
        if (better_token(v, i, value, best)) { value = v; best = i; }
    }
    values[lane] = value;
    indices[lane] = best;
    __syncthreads();
    for (int stride = 128; stride; stride >>= 1) {
        if (lane < stride && better_token(values[lane + stride], indices[lane + stride],
                                         values[lane], indices[lane])) {
            values[lane] = values[lane + stride];
            indices[lane] = indices[lane + stride];
        }
        __syncthreads();
    }
    if (lane == 0) {
        int id = indices[0];
        if (id < 0) atomicExch(status, 1);
        else out[slot] = id + d2t[id];
    }
}
