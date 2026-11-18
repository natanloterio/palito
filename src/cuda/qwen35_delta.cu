// Q, K, and V are the convolved projections, with Q and K already L2-normalized.
// One thread owns each state row, so no synchronization is needed between rows.
extern "C" __global__ void qwen35_delta_step(const float *qkv, const float *decay,
                                             const float *beta, float *state, float *output) {
    constexpr int dim = 128;
    constexpr int inner = 16 * dim;
    int h = blockIdx.x;
    int j = threadIdx.x;
    const float *q = qkv + h * dim;
    const float *k = q + inner;
    const float *v = k + inner;
    float *row = state + (h * dim + j) * dim;

    float predicted = 0.0f;
    for (int i = 0; i < dim; ++i) {
        row[i] *= decay[h];
        predicted += row[i] * k[i];
    }

    float delta = (v[j] - predicted) * beta[h];
    float mixed = 0.0f;
    for (int i = 0; i < dim; ++i) {
        row[i] += delta * k[i];
        mixed += row[i] * q[i];
    }
    output[h * dim + j] = mixed / sqrtf(float(dim));
}

#ifdef QWEN35_HYBRID
    extern "C" __global__ void qwen35_conv(float *qkv, float *history, const u8 *weights) {
        constexpr int channels = 6144;
        int channel = blockIdx.x * blockDim.x + threadIdx.x;
        if (channel >= channels) return;
        const float *w = reinterpret_cast<const float *>(weights) + channel * 4;
        float *h = history + channel * 3;
        float x = qkv[channel];
        float y = h[0] * w[0] + h[1] * w[1] + h[2] * w[2] + x * w[3];
        h[0] = h[1];
        h[1] = h[2];
        h[2] = x;
        qkv[channel] = y / (1.0f + expf(-y));
    }

    extern "C" __global__ void qwen35_l2(float *qkv, float epsilon) {
        float *head = qkv + blockIdx.x * 128;
        int i = threadIdx.x;
        float x = head[i];
        head[i] = x / sqrtf(sum_block(x * x) + epsilon);
    }

    extern "C" __global__ void qwen35_delta_gates(float *alpha, float *beta,
                                                  const u8 *dt, const u8 *a) {
        int i = threadIdx.x;
        float x = alpha[i] + reinterpret_cast<const float *>(dt)[i];
        float softplus = x > 20.0f ? x : log1pf(expf(x));
        alpha[i] = expf(softplus * reinterpret_cast<const float *>(a)[i]);
        beta[i] = 1.0f / (1.0f + expf(-beta[i]));
    }

    extern "C" __global__ void qwen35_split_qg(const float *qg, float *q, float *g) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        int j = (i / 256) * 512 + i % 256;
        q[i] = qg[j];
        g[i] = qg[j + 256];
    }

    extern "C" __global__ void qwen35_gate(float *x, const float *gate, int silu) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        float g = gate[i];
        x[i] *= (silu ? g : 1.0f) / (1.0f + expf(-g));
    }

    extern "C" __global__ void qwen35_delta_norm(float *x, const u8 *weights, float epsilon) {
        x += blockIdx.x * 128;
        int i = threadIdx.x;
        float v = x[i];
        float rms = sqrtf(sum_block(v * v) / 128.0f + epsilon);
        x[i] = v / rms * reinterpret_cast<const float *>(weights)[i];
    }
#endif
