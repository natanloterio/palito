#if __CUDA_ARCH__ >= 800
__device__ __forceinline__ unsigned int fp16_pair(float lo, float hi) {
    unsigned int pair;
    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(pair) : "f"(hi), "f"(lo));
    return pair;
}

template<int Kind, bool Packed>
__device__ void fp16_projection_tile(const u8 *weights, const float *input, float *output,
    int width, int rows, int tokens, const int *groups, int used, int grouped,
    int assignment_major, int *status) {
    int expert = grouped ? groups[blockIdx.y * 33] : 0;
    if (expert < 0) return;
    int tid = threadIdx.x, warp = tid / 32, lane = tid % 32;
    int first_row = blockIdx.x * 32, col_base = (warp % 2) * 16;
    int r = (warp / 2) * 16 + lane / 4, pair = lane % 4;
    __shared__ unsigned int a[32][17], b[32][17];
    __shared__ int destinations[32];
    if (tid < 32) {
        int assignment = grouped ? groups[blockIdx.y * 33 + 1 + tid] : blockIdx.y * 32 + tid;
        destinations[tid] = grouped || assignment < tokens ? assignment : -1;
    }
    __syncthreads();
    float c[2][4] = {};
    for (int k = 0; k < width; k += 32) {
        if (Packed) {
            #pragma unroll
            for (int i = tid; i < 256; i += 128) {
                int row = i / 8, j = (i % 8) * 4;
                bool valid = first_row + row < rows;
                const u8 *p = weights + (((u64)expert * rows + first_row + row) * width + k)
                    / 256 * (Kind == 12 ? 144 : 210);
                float a0 = 0.0f, a1 = 0.0f, minimum = 0.0f;
                if (valid && j == 0) residual_parameters<Kind>(p, k % 256, a0, a1, minimum);
                a0 = __shfl_sync(0xffffffff, a0, 0, 8);
                a1 = __shfl_sync(0xffffffff, a1, 0, 8);
                minimum = __shfl_sync(0xffffffff, minimum, 0, 8);
                unsigned int q = valid ? residual_packed_quant<Kind>(p, k % 256 + j) : 0;
                float scale = Kind == 14 && j >= 16 ? a1 : a0;
                int assignment = destinations[row];
                int source = grouped && !assignment_major && assignment >= 0 ? assignment / used : assignment;
                #pragma unroll
                for (int f = 0; f < 4; f += 2) {
                    int q0 = (signed char)(q >> (f * 8));
                    int q1 = (signed char)(q >> ((f + 1) * 8));
                    u64 x = (u64)(source < 0 ? 0 : source) * width + k + j + f;
                    a[row][(j + f) / 2] = fp16_pair(scale * (float)q0 - minimum,
                        scale * (float)q1 - minimum);
                    b[row][(j + f) / 2] = source >= 0
                        ? fp16_pair(input[x], input[x + 1]) : 0;
                }
            }
        } else {
        #pragma unroll
        for (int i = tid; i < 512; i += 128) {
            int row = i / 16, j = (i % 16) * 2;
            float w0 = 0.0f, w1 = 0.0f, x0 = 0.0f, x1 = 0.0f;
            if (first_row + row < rows) {
                u64 index = ((u64)expert * rows + first_row + row) * width + k + j;
                w0 = weight(weights, Kind, index);
                w1 = weight(weights, Kind, index + 1);
            }
            int assignment = destinations[row];
            if (assignment >= 0) {
                int source = grouped && !assignment_major ? assignment / used : assignment;
                u64 index = (u64)source * width + k + j;
                x0 = input[index];
                x1 = input[index + 1];
            }
            a[row][j / 2] = fp16_pair(w0, w1);
            b[row][j / 2] = fp16_pair(x0, x1);
        }
        }
        __syncthreads();
        #pragma unroll
        for (int h = 0; h < 16; h += 8) {
            unsigned int a0 = a[r][h + pair], a1 = a[r + 8][h + pair];
            unsigned int a2 = a[r][h + pair + 4], a3 = a[r + 8][h + pair + 4];
            #pragma unroll
            for (int t = 0; t < 2; t++) {
                int col = col_base + t * 8 + lane / 4;
                unsigned int b0 = b[col][h + pair], b1 = b[col][h + pair + 4];
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                    : "+f"(c[t][0]), "+f"(c[t][1]), "+f"(c[t][2]), "+f"(c[t][3])
                    : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
            }
        }
        __syncthreads();
    }
    #pragma unroll
    for (int t = 0; t < 2; t++) {
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            int row = first_row + r + (i / 2) * 8;
            int assignment = destinations[col_base + t * 8 + pair * 2 + i % 2];
            if (row < rows && assignment >= 0) {
                if (!isfinite(c[t][i])) atomicOr(status, 4);
                output[(u64)assignment * rows + row] = c[t][i];
            }
        }
    }
}
#endif

template<bool Packed>
__device__ __forceinline__ void project_fp16(const u8 *weights, int kind,
    const float *input, float *output, int width, int rows, int tokens, const int *groups,
    int used, int grouped, int assignment_major, int *status) {
#if __CUDA_ARCH__ >= 800
    if (kind == 12) fp16_projection_tile<12, Packed>(weights, input, output, width, rows, tokens,
        groups, used, grouped, assignment_major, status);
    else if (kind == 14) fp16_projection_tile<14, Packed>(weights, input, output, width, rows, tokens,
        groups, used, grouped, assignment_major, status);
    else if (threadIdx.x == 0) atomicOr(status, 8);
#else
    if (threadIdx.x == 0) atomicOr(status, 16);
#endif
}

#define FP16_CONTROL_ENTRY(Name, Packed) \
extern "C" __global__ void Name(const u8 *weights, int kind, const float *input, \
    float *output, int width, int rows, int tokens, const int *groups, int used, \
    int grouped, int assignment_major, int *status) { \
    project_fp16<Packed>(weights, kind, input, output, width, rows, tokens, groups, \
        used, grouped, assignment_major, status); \
}
FP16_CONTROL_ENTRY(projection_fp16_control, false)
FP16_CONTROL_ENTRY(projection_fp16_packed, true)
#undef FP16_CONTROL_ENTRY
