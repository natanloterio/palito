#if __CUDA_ARCH__ >= 800
__device__ unsigned int as_tf32(float value) {
    unsigned int result;
    asm("cvt.rn.tf32.f32 %0, %1;" : "=r"(result) : "f"(value));
    return result;
}

template<int Kind, bool Grouped, bool Accurate = false>
__device__ void tensor_core(const u8 *w, const float *x, float *y,
                           int width, int rows, int tokens, const int *groups,
                           int used, int batched) {
    int lane = threadIdx.x % 32, group = lane / 4, thread = lane % 4;
    // 16 rows per warp (blockDim.x / 32 warps; 256 threads = 128 rows).
    int row = blockIdx.x * (blockDim.x / 2) + (threadIdx.x / 32) * 16 + group, tile = blockIdx.y;
    int expert = Grouped ? groups[tile * 5] : 0;
    if (expert < 0) return;
    int assignment = Grouped ? (group < 4 ? groups[tile * 5 + group + 1] : -1)
                             : (tile * 8 + group < tokens ? tile * 8 + group : -1);
    int input = Grouped && !batched && assignment >= 0 ? assignment / used : assignment;
    float c0 = 0.0f, c1 = 0.0f, c2 = 0.0f, c3 = 0.0f;
    for (int k = 0; k < width; k += 8) {
        unsigned int a[4], residual[4];
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            int r = row + (i % 2) * 8, column = k + thread + (i / 2) * 4;
            float value = r < rows && column < width
                ? weight(w, Kind, ((u64)expert * rows + r) * width + column) : 0.0f;
            a[i] = as_tf32(value);
            if (Accurate) residual[i] = as_tf32(value - __uint_as_float(a[i]));
        }
        float x0 = input >= 0 && k + thread < width ? x[(u64)input * width + k + thread] : 0.0f;
        float x1 = input >= 0 && k + thread + 4 < width ? x[(u64)input * width + k + thread + 4] : 0.0f;
        unsigned int b0 = as_tf32(x0), b1 = as_tf32(x1);
        // A: 16x8 row-major, B: 8x8 column-major, C: 16x8 F32.
        // Every lane participates; fragment coordinates follow PTX m16n8k8.
        asm volatile(
            "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
            : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
            : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
        if (Accurate) {
            unsigned int low0 = as_tf32(x0 - __uint_as_float(b0));
            unsigned int low1 = as_tf32(x1 - __uint_as_float(b1));
            asm volatile(
                "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
                "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
                : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(low0), "r"(low1));
            asm volatile(
                "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
                "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
                : "r"(residual[0]), "r"(residual[1]), "r"(residual[2]), "r"(residual[3]), "r"(b0), "r"(b1));
        }
    }
    float result[4] = {c0, c1, c2, c3};
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        int r = row + (i / 2) * 8, column = thread * 2 + i % 2;
        int assignment = Grouped ? (column < 4 ? groups[tile * 5 + column + 1] : -1)
                                 : (tile * 8 + column < tokens ? tile * 8 + column : -1);
        if (r < rows && assignment >= 0) y[(u64)assignment * rows + r] = result[i];
    }
}
#endif

extern "C" __global__ void prefill_matmul_tf32(const u8 *w, int kind, const float *x,
                                             float *y, int width, int rows, int tokens) {
#if __CUDA_ARCH__ >= 800
    switch (kind) {
        case 0: tensor_core<0, false>(w, x, y, width, rows, tokens, nullptr, 1, 0); break;
        case 1: tensor_core<1, false>(w, x, y, width, rows, tokens, nullptr, 1, 0); break;
        case 6: tensor_core<6, false>(w, x, y, width, rows, tokens, nullptr, 1, 0); break;
        case 8: tensor_core<8, false>(w, x, y, width, rows, tokens, nullptr, 1, 0); break;
        case 12: tensor_core<12, false>(w, x, y, width, rows, tokens, nullptr, 1, 0); break;
        case 14: tensor_core<14, false>(w, x, y, width, rows, tokens, nullptr, 1, 0); break;
    }
#else
    asm("trap;");
#endif
}

extern "C" __global__ void grouped_matmul_tf32(const u8 *w, int kind, const float *x,
                                             float *y, int width, int rows, const int *groups,
                                             int used, int batched) {
#if __CUDA_ARCH__ >= 800
    switch (kind) {
        case 0: tensor_core<0, true>(w, x, y, width, rows, 0, groups, used, batched); break;
        case 1: tensor_core<1, true>(w, x, y, width, rows, 0, groups, used, batched); break;
        case 6: tensor_core<6, true>(w, x, y, width, rows, 0, groups, used, batched); break;
        case 8: tensor_core<8, true>(w, x, y, width, rows, 0, groups, used, batched); break;
        case 12: tensor_core<12, true>(w, x, y, width, rows, 0, groups, used, batched); break;
        case 14: tensor_core<14, true>(w, x, y, width, rows, 0, groups, used, batched); break;
    }
#else
    asm("trap;");
#endif
}

extern "C" __global__ void prefill_matmul_tf32x3(const u8 *w, int kind, const float *x,
                                               float *y, int width, int rows, int tokens) {
#if __CUDA_ARCH__ >= 800
    switch (kind) {
        case 0: tensor_core<0, false, true>(w, x, y, width, rows, tokens, nullptr, 1, 0); break;
        case 1: tensor_core<1, false, true>(w, x, y, width, rows, tokens, nullptr, 1, 0); break;
        case 6: tensor_core<6, false, true>(w, x, y, width, rows, tokens, nullptr, 1, 0); break;
        case 8: tensor_core<8, false, true>(w, x, y, width, rows, tokens, nullptr, 1, 0); break;
        case 12: tensor_core<12, false, true>(w, x, y, width, rows, tokens, nullptr, 1, 0); break;
        case 14: tensor_core<14, false, true>(w, x, y, width, rows, tokens, nullptr, 1, 0); break;
    }
#else
    asm("trap;");
#endif
}

extern "C" __global__ void grouped_matmul_tf32x3(const u8 *w, int kind, const float *x,
                                               float *y, int width, int rows, const int *groups,
                                               int used, int batched) {
#if __CUDA_ARCH__ >= 800
    switch (kind) {
        case 0: tensor_core<0, true, true>(w, x, y, width, rows, 0, groups, used, batched); break;
        case 1: tensor_core<1, true, true>(w, x, y, width, rows, 0, groups, used, batched); break;
        case 6: tensor_core<6, true, true>(w, x, y, width, rows, 0, groups, used, batched); break;
        case 8: tensor_core<8, true, true>(w, x, y, width, rows, 0, groups, used, batched); break;
        case 12: tensor_core<12, true, true>(w, x, y, width, rows, 0, groups, used, batched); break;
        case 14: tensor_core<14, true, true>(w, x, y, width, rows, 0, groups, used, batched); break;
    }
#else
    asm("trap;");
#endif
}

#if __CUDA_ARCH__ >= 800
// Eight warps cover 128 rows and all eight assignment columns. Operands are
// cooperatively staged once per K=32 slab, rather than loaded by every warp.
template<int Kind, bool Grouped>
__device__ void tensor_core_tiled(const u8 *w, const float *x, float *y,
                                  int width, int rows, int tokens, const int *groups,
                                  int used, int batched, float weights[][33],
                                  float activations[][8]) {
    int expert = Grouped ? groups[blockIdx.y * 9] : 0;
    if (expert < 0) return;
    int lane = threadIdx.x % 32, group = lane / 4, thread = lane % 4;
    int first_row = blockIdx.x * 128;
    int row = (threadIdx.x / 32) * 16 + group;
    float c0 = 0.0f, c1 = 0.0f, c2 = 0.0f, c3 = 0.0f;
    for (int first_k = 0; first_k < width; first_k += 32) {
        projection_weights<Kind, 128, 32>(w, weights, width, rows, expert, first_row, first_k);
        projection_inputs<Grouped, 32>(x, activations, width, tokens, groups, used, batched, first_k);
        __syncthreads();
        #pragma unroll
        for (int k = 0; k < 32; k += 8) {
            unsigned int a[4], residual[4];
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                float value = weights[row + (i % 2) * 8][k + thread + (i / 2) * 4];
                a[i] = as_tf32(value);
                residual[i] = as_tf32(value - __uint_as_float(a[i]));
            }
            float x0 = activations[k + thread][group];
            float x1 = activations[k + thread + 4][group];
            unsigned int b0 = as_tf32(x0), b1 = as_tf32(x1);
            unsigned int low0 = as_tf32(x0 - __uint_as_float(b0));
            unsigned int low1 = as_tf32(x1 - __uint_as_float(b1));
            asm volatile(
                "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
                "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
                : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
            asm volatile(
                "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
                "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
                : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(low0), "r"(low1));
            asm volatile(
                "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
                "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
                : "r"(residual[0]), "r"(residual[1]), "r"(residual[2]), "r"(residual[3]), "r"(b0), "r"(b1));
        }
        __syncthreads();
    }
    float result[4] = {c0, c1, c2, c3};
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        int r = first_row + row + (i / 2) * 8;
        int assignment = projection_assignment<Grouped>(groups, blockIdx.y, thread * 2 + i % 2, tokens);
        if (r < rows && assignment >= 0) y[(u64)assignment * rows + r] = result[i];
    }
}
#endif

// Launch 256 threads, grid=(ceil(rows/128), ceil(tokens/8), 1), no dynamic scratch.
extern "C" __global__ void prefill_matmul_tf32x3_tiled(const u8 *w, int kind, const float *x,
                                                      float *y, int width, int rows, int tokens) {
#if __CUDA_ARCH__ >= 800
    __shared__ float weights[128][33], activations[32][8];
    switch (kind) {
        case 0: tensor_core_tiled<0, false>(w, x, y, width, rows, tokens, nullptr, 1, 0, weights, activations); break;
        case 1: tensor_core_tiled<1, false>(w, x, y, width, rows, tokens, nullptr, 1, 0, weights, activations); break;
        case 6: tensor_core_tiled<6, false>(w, x, y, width, rows, tokens, nullptr, 1, 0, weights, activations); break;
        case 8: tensor_core_tiled<8, false>(w, x, y, width, rows, tokens, nullptr, 1, 0, weights, activations); break;
        case 12: tensor_core_tiled<12, false>(w, x, y, width, rows, tokens, nullptr, 1, 0, weights, activations); break;
        case 14: tensor_core_tiled<14, false>(w, x, y, width, rows, tokens, nullptr, 1, 0, weights, activations); break;
    }
#else
    asm("trap;");
#endif
}

#ifdef CUDA_M8_PROTOTYPES
#if __CUDA_ARCH__ >= 800
// Diagnostic only: keep each correction out of the large high*high running sum.
template<int Kind>
__device__ void tensor_core_tf32x4_control(const u8 *w, const float *x, float *y,
                                          int width, int rows, int tokens,
                                          float weights[][33], float activations[][8]) {
    int lane = threadIdx.x % 32, group = lane / 4, thread = lane % 4;
    int first_row = blockIdx.x * 128;
    int row = (threadIdx.x / 32) * 16 + group;
    float hh[4] = {}, hl[4] = {}, lh[4] = {}, ll[4] = {};
    for (int first_k = 0; first_k < width; first_k += 32) {
        projection_weights<Kind, 128, 32>(w, weights, width, rows, 0, first_row, first_k);
        projection_inputs<false, 32>(x, activations, width, tokens, nullptr, 1, 0, first_k);
        __syncthreads();
        #pragma unroll
        for (int k = 0; k < 32; k += 8) {
            unsigned int a[4], residual[4];
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                float value = weights[row + (i % 2) * 8][k + thread + (i / 2) * 4];
                a[i] = as_tf32(value);
                residual[i] = as_tf32(value - __uint_as_float(a[i]));
            }
            float x0 = activations[k + thread][group];
            float x1 = activations[k + thread + 4][group];
            unsigned int b0 = as_tf32(x0), b1 = as_tf32(x1);
            unsigned int low0 = as_tf32(x0 - __uint_as_float(b0));
            unsigned int low1 = as_tf32(x1 - __uint_as_float(b1));
            asm volatile(
                "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
                "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                : "+f"(hh[0]), "+f"(hh[1]), "+f"(hh[2]), "+f"(hh[3])
                : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
            asm volatile(
                "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
                "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                : "+f"(hl[0]), "+f"(hl[1]), "+f"(hl[2]), "+f"(hl[3])
                : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(low0), "r"(low1));
            asm volatile(
                "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
                "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                : "+f"(lh[0]), "+f"(lh[1]), "+f"(lh[2]), "+f"(lh[3])
                : "r"(residual[0]), "r"(residual[1]), "r"(residual[2]), "r"(residual[3]), "r"(b0), "r"(b1));
            asm volatile(
                "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
                "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                : "+f"(ll[0]), "+f"(ll[1]), "+f"(ll[2]), "+f"(ll[3])
                : "r"(residual[0]), "r"(residual[1]), "r"(residual[2]), "r"(residual[3]), "r"(low0), "r"(low1));
        }
        __syncthreads();
    }
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        int r = first_row + row + (i / 2) * 8;
        int assignment = projection_assignment<false>(nullptr, blockIdx.y, thread * 2 + i % 2, tokens);
        if (r < rows && assignment >= 0)
            y[(u64)assignment * rows + r] = hh[i] + (hl[i] + (lh[i] + ll[i]));
    }
}
#endif

// Same geometry as tf32x3_tiled: 256 threads, ceil(rows/128) x ceil(tokens/8),
// 17,920 bytes static shared memory, no dynamic scratch; diagnostic SM80+ only.
extern "C" __global__ void prefill_matmul_tf32x4_control(const u8 *w, int kind, const float *x,
                                                       float *y, int width, int rows, int tokens) {
#if __CUDA_ARCH__ >= 800
    __shared__ float weights[128][33], activations[32][8];
    switch (kind) {
        case 0: tensor_core_tf32x4_control<0>(w, x, y, width, rows, tokens, weights, activations); break;
        case 1: tensor_core_tf32x4_control<1>(w, x, y, width, rows, tokens, weights, activations); break;
        case 6: tensor_core_tf32x4_control<6>(w, x, y, width, rows, tokens, weights, activations); break;
        case 8: tensor_core_tf32x4_control<8>(w, x, y, width, rows, tokens, weights, activations); break;
        case 12: tensor_core_tf32x4_control<12>(w, x, y, width, rows, tokens, weights, activations); break;
        case 14: tensor_core_tf32x4_control<14>(w, x, y, width, rows, tokens, weights, activations); break;
    }
#else
    asm("trap;");
#endif
}
#endif

// Grouped grid.y=slots; only the new nine-int grouping layout is accepted.
extern "C" __global__ void grouped_matmul_tf32x3_tiled(const u8 *w, int kind, const float *x,
                                                      float *y, int width, int rows, const int *groups,
                                                      int used, int batched) {
#if __CUDA_ARCH__ >= 800
    __shared__ float weights[128][33], activations[32][8];
    switch (kind) {
        case 0: tensor_core_tiled<0, true>(w, x, y, width, rows, 0, groups, used, batched, weights, activations); break;
        case 1: tensor_core_tiled<1, true>(w, x, y, width, rows, 0, groups, used, batched, weights, activations); break;
        case 6: tensor_core_tiled<6, true>(w, x, y, width, rows, 0, groups, used, batched, weights, activations); break;
        case 8: tensor_core_tiled<8, true>(w, x, y, width, rows, 0, groups, used, batched, weights, activations); break;
        case 12: tensor_core_tiled<12, true>(w, x, y, width, rows, 0, groups, used, batched, weights, activations); break;
        case 14: tensor_core_tiled<14, true>(w, x, y, width, rows, 0, groups, used, batched, weights, activations); break;
    }
#else
    asm("trap;");
#endif
}

// int8-fast MoE router for F32 weights: prefill_matmul_tf32<0> per warp tile (same
// m16n8k8 order over K, so identical results) with (16 WR) rows x (8 WT) tokens per
// (32 WR WT)-thread block and both operands staged by a two-stage cp.async pipeline.
// grid = (ceil(rows / (16 WR)), ceil(tokens / (8 WT))); width % 32 == 0.
#if __CUDA_ARCH__ >= 800
__device__ __forceinline__ void router_copy16(float *dst, const float *src, bool valid) {
    unsigned int d = (unsigned int)__cvta_generic_to_shared(dst);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" :: "r"(d), "l"(src), "r"(valid ? 16 : 0) : "memory");
}

template<int WR, int WT, int STAGES>
__device__ __forceinline__ void router_tf32_impl(const float *w, const float *x, float *y,
                                                 int width, int rows, int tokens) {
    constexpr int KS = 32, STRIDE = KS + 4, R = 16 * WR, T = 8 * WT, THREADS = 32 * WR * WT;
    __shared__ __align__(16) float a[STAGES][R][STRIDE], b[STAGES][T][STRIDE];
    int tid = threadIdx.x, warp = tid / 32, lane = tid % 32, group = lane / 4, thread = lane % 4;
    int first_row = blockIdx.x * R, first_token = blockIdx.y * T;
    int warp_row = (warp % WR) * 16, warp_token = (warp / WR) * 8;
    int steps = width / KS;
    // Every call commits a group (empty past the end) so wait_group counts stay fixed.
    auto issue = [&](int step) {
        if (step < steps) {
            int stage = step % STAGES, k = step * KS;
            #pragma unroll
            for (int i = tid; i < R * (KS / 4); i += THREADS) {
                int r = i / (KS / 4), c = i % (KS / 4);
                bool valid = first_row + r < rows;
                router_copy16(&a[stage][r][c * 4], w + (u64)(valid ? first_row + r : 0) * width + k + c * 4, valid);
            }
            #pragma unroll
            for (int i = tid; i < T * (KS / 4); i += THREADS) {
                int t = i / (KS / 4), c = i % (KS / 4);
                bool valid = first_token + t < tokens;
                router_copy16(&b[stage][t][c * 4], x + (u64)(valid ? first_token + t : 0) * width + k + c * 4, valid);
            }
        }
        asm volatile("cp.async.commit_group;" ::: "memory");
    };
    float c0 = 0.0f, c1 = 0.0f, c2 = 0.0f, c3 = 0.0f;
    #pragma unroll
    for (int s = 0; s < STAGES - 1; s++) issue(s);
    for (int step = 0; step < steps; step++) {
        int stage = step % STAGES;
        asm volatile("cp.async.wait_group %0;" :: "n"(STAGES - 2) : "memory");
        __syncthreads();
        // The stage refilled here was consumed in the previous step (before the barrier).
        issue(step + STAGES - 1);
        #pragma unroll
        for (int kk = 0; kk < KS; kk += 8) {
            unsigned int f[4];
            #pragma unroll
            for (int i = 0; i < 4; i++)
                f[i] = as_tf32(a[stage][warp_row + group + (i % 2) * 8][kk + thread + (i / 2) * 4]);
            unsigned int b0 = as_tf32(b[stage][warp_token + group][kk + thread]);
            unsigned int b1 = as_tf32(b[stage][warp_token + group][kk + thread + 4]);
            asm volatile(
                "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
                "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
                : "r"(f[0]), "r"(f[1]), "r"(f[2]), "r"(f[3]), "r"(b0), "r"(b1));
        }
    }
    float result[4] = {c0, c1, c2, c3};
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        int r = first_row + warp_row + group + (i / 2) * 8;
        int token = first_token + warp_token + thread * 2 + i % 2;
        if (r < rows && token < tokens) y[(u64)token * rows + r] = result[i];
    }
}
#endif

// 64 rows x 16 tokens, 256 threads.
extern "C" __global__ void __launch_bounds__(256)
router_tf32_tiled(const float *w, int kind, const float *x, float *y, int width, int rows, int tokens) {
#if __CUDA_ARCH__ >= 800
    router_tf32_impl<4, 2, 2>(w, x, y, width, rows, tokens);
#else
    asm("trap;");
#endif
}

#define ROUTER_SMALL_STAGES 6
// 32 rows x 8 tokens, 64 threads, ROUTER_SMALL_STAGES-deep pipeline: 8x the blocks
// of router_tf32_tiled for short prompts.
extern "C" __global__ void __launch_bounds__(64)
router_tf32_small(const float *w, int kind, const float *x, float *y, int width, int rows, int tokens) {
#if __CUDA_ARCH__ >= 800
    router_tf32_impl<2, 1, ROUTER_SMALL_STAGES>(w, x, y, width, rows, tokens);
#else
    asm("trap;");
#endif
}
