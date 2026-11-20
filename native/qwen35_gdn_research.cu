#include "gated-delta-net-mma.cuh"

namespace {
constexpr int heads = 16, dim = 128, tokens = 512;
constexpr int inner = heads * dim, qkv_width = 3 * inner;
constexpr int state_size = heads * dim * dim, threads = 256;

__global__ void log_decay(const float *decay, float *logs) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    logs[i] = logf(decay[i]);
}

__global__ void transpose_state(const float *src, float *dst) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    dst[i] = src[i / (dim * dim) * dim * dim + i % dim * dim + i / dim % dim];
}
}

extern "C" int li_qwen35_gdn_512(
    cudaStream_t stream, const float *qkv, const float *decay, const float *beta,
    float *state, float *out, float *logs, float *transposed_in, float *transposed_out) {
    int device;
    if (const auto status = cudaGetDevice(&device); status != cudaSuccess) return status;
    log_decay<<<tokens * heads / threads, threads, 0, stream>>>(decay, logs);
    transpose_state<<<state_size / threads, threads, 0, stream>>>(state, transposed_in);
    if (const auto status = cudaGetLastError(); status != cudaSuccess) return status;
    ggml_cuda_gdn_mma_args args{};
    args.q = qkv;
    args.k = qkv + inner;
    args.v = qkv + 2 * inner;
    args.g = logs;
    args.beta = beta;
    args.state = transposed_in;
    args.dst = out;
    args.state_out = transposed_out;
    args.H = args.H_k = heads;
    args.n_tokens = tokens;
    args.n_seqs = args.rq3 = 1;
    args.sq1 = args.sv1 = dim;
    args.sq2 = args.sv2 = qkv_width;
    args.sb1 = 1;
    args.sb2 = heads;
    args.scale = 1.0f / sqrtf(float(dim));
    args.eligible = true;
    if (!ggml_cuda_gdn_mma_launch(device, args, stream)) return cudaErrorNotSupported;
    transpose_state<<<state_size / threads, threads, 0, stream>>>(transposed_out, state);
    return cudaGetLastError();
}
