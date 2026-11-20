// Q, K, and V are the convolved projections, with Q and K already L2-normalized.
// One thread owns each state row, so no synchronization is needed between rows.
__device__ __forceinline__ float qwen35_recurrent_mul_add(float a, float b, float c) {
#ifdef CUDA_QWEN35_RECURRENT_FMA
    return __fmaf_rn(a, b, c);
#else
    return c + a * b;
#endif
}

template<bool Transposed>
__device__ __forceinline__ void qwen35_update(const float *qkv, const float *decay,
                                             const float *beta, float *state, float *output) {
    constexpr int dim = 128;
    constexpr int inner = 16 * dim;
    int h = blockIdx.x;
    int j = threadIdx.x;
    const float *q = qkv + h * dim;
    const float *k = q + inner;
    const float *v = k + inner;
    float *row = state + h * dim * dim + (Transposed ? j : j * dim);
    constexpr int stride = Transposed ? dim : 1;

    float predicted = 0.0f;
    for (int i = 0; i < dim; ++i) {
        row[i * stride] *= decay[h];
        predicted = qwen35_recurrent_mul_add(row[i * stride], k[i], predicted);
    }

    float delta = (v[j] - predicted) * beta[h];
    float mixed = 0.0f;
    for (int i = 0; i < dim; ++i) {
        row[i * stride] = qwen35_recurrent_mul_add(delta, k[i], row[i * stride]);
        mixed = qwen35_recurrent_mul_add(row[i * stride], q[i], mixed);
    }
    output[h * dim + j] = mixed / sqrtf(float(dim));
}

extern "C" __global__ void qwen35_delta_step(const float *qkv, const float *decay,
                                             const float *beta, float *state, float *output) {
    qwen35_update<false>(qkv, decay, beta, state, output);
}

#ifdef QWEN35_HYBRID
    extern "C" __global__ void qwen35_delta_batch(const float *qkv, const float *decay,
          const float *beta, float *state, float *output, int tokens) {
        // Adjacent lanes own adjacent state elements at each reduction index.
        for (int t = 0; t < tokens; ++t)
            qwen35_update<true>(qkv + t * 6144, decay + t * 16, beta + t * 16,
                               state, output + t * 2048);
    }

    template<int lanes>
    __device__ __forceinline__ void qwen35_delta_panel_impl(const float *qkv, const float *decay,
          const float *beta, float *state, float *output, int tokens) {
        constexpr int dim = 128, inner = 2048;
        const int h = blockIdx.x, j = blockIdx.y * lanes + threadIdx.x;
        float *global_row = state + h * dim * dim + j;
        float panel[dim];
#ifdef CUDA_QWEN35_RECURRENT_STAGE_QK
        __shared__ float qs[dim], ks[dim];
#endif
        #pragma unroll
        for (int i = 0; i < dim; ++i)
            panel[i] = global_row[i * dim];
        for (int t = 0; t < tokens; ++t) {
            const float *q = qkv + t * 3 * inner + h * dim;
            const float *k = q + inner;
            const float value = k[inner + j];
#ifdef CUDA_QWEN35_RECURRENT_STAGE_QK
            #pragma unroll
            for (int i = threadIdx.x; i < dim; i += lanes) {
                qs[i] = q[i];
                ks[i] = k[i];
            }
            if constexpr (lanes == 32) __syncwarp();
            else __syncthreads();
            q = qs;
            k = ks;
#endif
            const float d = decay[t * 16 + h];
            float predicted = 0.0f;
            #pragma unroll
            for (int i = 0; i < dim; ++i) {
                panel[i] *= d;
                predicted = qwen35_recurrent_mul_add(panel[i], k[i], predicted);
            }
            const float delta = (value - predicted) * beta[t * 16 + h];
            float mixed = 0.0f;
            #pragma unroll
            for (int i = 0; i < dim; ++i) {
                panel[i] = qwen35_recurrent_mul_add(delta, k[i], panel[i]);
                mixed = qwen35_recurrent_mul_add(panel[i], q[i], mixed);
            }
            output[t * inner + h * dim + j] = mixed / sqrtf(float(dim));
#ifdef CUDA_QWEN35_RECURRENT_STAGE_QK
            if constexpr (lanes == 32) __syncwarp();
            else __syncthreads();
#endif
        }
        #pragma unroll
        for (int i = 0; i < dim; ++i)
            global_row[i * dim] = panel[i];
    }

    extern "C" __global__ void qwen35_delta_panel(const float *qkv, const float *decay,
          const float *beta, float *state, float *output, int tokens) {
        qwen35_delta_panel_impl<32>(qkv, decay, beta, state, output, tokens);
    }

#ifdef CUDA_QWEN35_RECURRENT_PANEL64
    extern "C" __global__ void qwen35_delta_panel64(const float *qkv, const float *decay,
          const float *beta, float *state, float *output, int tokens) {
        qwen35_delta_panel_impl<64>(qkv, decay, beta, state, output, tokens);
    }
#endif

#ifdef CUDA_QWEN35_RECURRENT_WARP
    extern "C" __global__ void qwen35_delta_warp(const float *qkv, const float *decay,
          const float *beta, float *state, float *output, int tokens) {
        constexpr int dim = 128, inner = 2048;
#ifdef CUDA_QWEN35_RECURRENT_EIGHT_SEGMENTS
        constexpr int rows = 4;
#else
        constexpr int rows = 8;
#endif
        constexpr int segment = dim * rows / 32;
        const int h = blockIdx.x, j = blockIdx.y * rows + (threadIdx.x % rows);
        const int start = (threadIdx.x / rows) * segment;
        float *global = state + (h * dim + start) * dim + j;
        float panel[segment];
#ifndef CUDA_QWEN35_RECURRENT_DIRECT_QK
        __shared__ float qs[dim], ks[dim];
#endif
        #pragma unroll
        for (int i = 0; i < segment; ++i)
            panel[i] = global[i * dim];
        for (int t = 0; t < tokens; ++t) {
            const float *q = qkv + t * 3 * inner + h * dim;
            const float *k = q + inner;
#ifdef CUDA_QWEN35_RECURRENT_DIRECT_QK
            const float *qs = q, *ks = k;
#else
            #pragma unroll
            for (int i = threadIdx.x; i < dim; i += 32) {
                qs[i] = q[i];
                ks[i] = k[i];
            }
            __syncwarp();
#endif
            const float d = decay[t * 16 + h];
            float predicted = 0.0f;
#ifdef CUDA_QWEN35_RECURRENT_FOUR_CHAINS
            float pred[4] = {};
#endif
            #pragma unroll
            for (int i = 0; i < segment; ++i) {
                panel[i] *= d;
#ifdef CUDA_QWEN35_RECURRENT_FOUR_CHAINS
                pred[i % 4] = qwen35_recurrent_mul_add(panel[i], ks[start + i], pred[i % 4]);
#else
                predicted = qwen35_recurrent_mul_add(panel[i], ks[start + i], predicted);
#endif
            }
#ifdef CUDA_QWEN35_RECURRENT_FOUR_CHAINS
            predicted = ((pred[0] + pred[1]) + pred[2]) + pred[3];
#endif
            #pragma unroll
            for (int stride = rows; stride < 32; stride *= 2)
                predicted += __shfl_xor_sync(0xffffffff, predicted, stride);
            // All segments of a row use the same delta.
            predicted = __shfl_sync(0xffffffff, predicted, threadIdx.x % rows);
            const float delta = (k[inner + j] - predicted) * beta[t * 16 + h];
            float mixed = 0.0f;
#ifdef CUDA_QWEN35_RECURRENT_FOUR_CHAINS
            float mix[4] = {};
#endif
            #pragma unroll
            for (int i = 0; i < segment; ++i) {
                panel[i] = qwen35_recurrent_mul_add(delta, ks[start + i], panel[i]);
#ifdef CUDA_QWEN35_RECURRENT_FOUR_CHAINS
                mix[i % 4] = qwen35_recurrent_mul_add(panel[i], qs[start + i], mix[i % 4]);
#else
                mixed = qwen35_recurrent_mul_add(panel[i], qs[start + i], mixed);
#endif
            }
#ifdef CUDA_QWEN35_RECURRENT_FOUR_CHAINS
            mixed = ((mix[0] + mix[1]) + mix[2]) + mix[3];
#endif
            #pragma unroll
            for (int stride = rows; stride < 32; stride *= 2)
                mixed += __shfl_xor_sync(0xffffffff, mixed, stride);
            if (start == 0)
                output[t * inner + h * dim + j] = mixed / sqrtf(float(dim));
#ifndef CUDA_QWEN35_RECURRENT_DIRECT_QK
            __syncwarp();
#endif
        }
        #pragma unroll
        for (int i = 0; i < segment; ++i)
            global[i * dim] = panel[i];
    }
#endif

    extern "C" __global__ void qwen35_conv(float *qkv, float *history, const u8 *weights, int tokens) {
        constexpr int channels = 6144;
        int channel = blockIdx.x * blockDim.x + threadIdx.x;
        if (channel >= channels) return;
        const float *w = reinterpret_cast<const float *>(weights) + channel * 4;
        float *h = history + channel * 3;
        for (int t = 0; t < tokens; ++t) {
            float &value = qkv[t * channels + channel];
            float x = value;
            float y = h[0] * w[0] + h[1] * w[1] + h[2] * w[2] + x * w[3];
            h[0] = h[1];
            h[1] = h[2];
            h[2] = x;
            value = y / (1.0f + expf(-y));
        }
    }

#ifdef CUDA_QWEN35_CONV_TILE
    constexpr int conv_channels = 6144, conv_tile = 16;

    extern "C" __global__ void qwen35_conv_save(const float *qkv, float *history,
                                              float *halo, int tokens) {
        int c = blockIdx.x * blockDim.x + threadIdx.x;
        if (c >= conv_channels) return;
        int start = blockIdx.y * conv_tile;
        #pragma unroll
        for (int i = 0; i < 3; ++i) {
            halo[(blockIdx.y * 3 + i) * conv_channels + c] = start
                ? qkv[(start - 3 + i) * conv_channels + c] : history[c * 3 + i];
            if (!start)
                history[c * 3 + i] = qkv[(tokens - 3 + i) * conv_channels + c];
        }
    }

    extern "C" __global__ void qwen35_conv_tile(float *qkv, const float *halo,
                                               const u8 *weights, int tokens) {
        int c = blockIdx.x * blockDim.x + threadIdx.x;
        if (c >= conv_channels) return;
        int start = blockIdx.y * conv_tile;
        int count = min(conv_tile, tokens - start);
        float *tile = qkv + start * conv_channels + c;
        const float *h = halo + blockIdx.y * 3 * conv_channels + c;
        const float *w = reinterpret_cast<const float *>(weights) + c * 4;
        // Descending writes leave every within-tile predecessor unmodified.
        for (int t = count - 1; t >= 0; --t) {
            float a = t >= 3 ? tile[(t - 3) * conv_channels] : h[t * conv_channels];
            float b = t >= 2 ? tile[(t - 2) * conv_channels] : h[(t + 1) * conv_channels];
            float d = t >= 1 ? tile[(t - 1) * conv_channels] : h[(t + 2) * conv_channels];
            float &value = tile[t * conv_channels];
            float y = a * w[0] + b * w[1] + d * w[2] + value * w[3];
            value = y / (1.0f + expf(-y));
        }
    }
#endif

    extern "C" __global__ void qwen35_l2(float *qkv, float epsilon) {
        float *head = qkv + blockIdx.y * 6144 + blockIdx.x * 128;
        int i = threadIdx.x;
        float x = head[i];
        head[i] = x / sqrtf(sum_block(x * x) + epsilon);
    }

    extern "C" __global__ void qwen35_delta_gates(float *alpha, float *beta,
                                                  const u8 *dt, const u8 *a) {
        int i = threadIdx.x;
        alpha += blockIdx.x * 16;
        beta += blockIdx.x * 16;
        float x = alpha[i] + reinterpret_cast<const float *>(dt)[i];
        float softplus = x > 20.0f ? x : log1pf(expf(x));
        alpha[i] = expf(softplus * reinterpret_cast<const float *>(a)[i]);
        beta[i] = 1.0f / (1.0f + expf(-beta[i]));
    }

    extern "C" __global__ void qwen35_split_qg(const float *qg, float *q, float *g) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        int j = (i / 256) * 512 + i % 256;
        q[blockIdx.y * 2048 + i] = qg[blockIdx.y * 4096 + j];
        g[blockIdx.y * 2048 + i] = qg[blockIdx.y * 4096 + j + 256];
    }

    extern "C" __global__ void qwen35_gate(float *x, const float *gate, int silu) {
        int i = blockIdx.y * 2048 + blockIdx.x * blockDim.x + threadIdx.x;
        float g = gate[i];
        x[i] *= (silu ? g : 1.0f) / (1.0f + expf(-g));
    }

    __device__ __forceinline__ float qwen35_delta_norm_value(float v, const u8 *weights, float epsilon) {
        float rms = sqrtf(sum_block(v * v) / 128.0f + epsilon);
        return v / rms * reinterpret_cast<const float *>(weights)[threadIdx.x];
    }

    extern "C" __global__ void qwen35_delta_norm(float *x, const u8 *weights, float epsilon) {
        x += blockIdx.x * 128;
        int i = threadIdx.x;
        x[i] = qwen35_delta_norm_value(x[i], weights, epsilon);
    }

#ifdef CUDA_QWEN35_RECURRENT_NORM_GATE
    extern "C" __global__ void qwen35_delta_norm_gate(float *x, const float *gate, const u8 *weights, float epsilon) {
        int i = blockIdx.x * 128 + threadIdx.x;
        float v = qwen35_delta_norm_value(x[i], weights, epsilon);
        float g = gate[i];
        x[i] = v * (g / (1.0f + expf(-g)));
    }
#endif
#endif
