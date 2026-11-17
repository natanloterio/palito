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
    int row = blockIdx.x * 128 + (threadIdx.x / 32) * 16 + group, tile = blockIdx.y;
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
