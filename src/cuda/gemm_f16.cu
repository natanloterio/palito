// F16-weight prefill projection on FP16 tensor cores for the relaxed profiles
// (int8-prefill, int8-fast); same ABI as prefill_matmul, grid
// (ceil(rows / 128), ceil(tokens / 128)), 256 threads, 40 KB static shared memory.
// y[t * rows + r] = sum_k w[r][k] * x[t][k] with F16 weights read as stored, the
// F32 activations rounded to F16 as they are staged (as llama.cpp's cuBLAS F16
// path), and F32 accumulation. Requires width % 32 == 0.
// Each block covers 128 rows x 128 tokens in K steps of 32: weights arrive by
// cp.async into the other buffer while the current one computes, activations by
// register prefetch. 8 warps (2 along rows x 4 along tokens) each own 64 x 32
// outputs: 4 x 4 m16n8k16 MMAs per 16 columns, fragments loaded with ldmatrix.
#define GF16_BM 128
#define GF16_BN 128
#define GF16_BK 32
// Halves per shared row: 80 bytes keeps ldmatrix rows on distinct banks.
#define GF16_STRIDE 40

#if __CUDA_ARCH__ >= 800
__device__ __forceinline__ unsigned int gf16_pair(float lo, float hi) {
    unsigned int pair;
    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(pair) : "f"(hi), "f"(lo));
    return pair;
}

__device__ __forceinline__ unsigned int gf16_shared(const void *p) {
    return (unsigned int)__cvta_generic_to_shared(p);
}

__device__ __forceinline__ void gf16_ldmatrix(unsigned int (&r)[4], const unsigned short *p) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
        : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(gf16_shared(p)));
}
#endif

extern "C" __global__ void __launch_bounds__(256)
gemm_f16(const u8 *w, int kind, const float *x, float *y, int width, int rows, int tokens) {
#if __CUDA_ARCH__ >= 800
    __shared__ __align__(16) unsigned short a_tile[2][GF16_BM][GF16_STRIDE];
    __shared__ __align__(16) unsigned short b_tile[2][GF16_BN][GF16_STRIDE];
    const int tid = threadIdx.x, warp = tid / 32, lane = tid % 32;
    const int group = lane / 4, thread = lane % 4;
    const int row0 = blockIdx.x * GF16_BM, token0 = blockIdx.y * GF16_BN;
    const int warp_row = (warp % 2) * 64, warp_token = (warp / 2) * 32;
    const unsigned short *weights = (const unsigned short *)w;

    // 128 rows x 32 halves = 512 16-byte chunks, two per thread; rows past the
    // matrix are zero-filled (source size 0).
    auto load_weights = [&](int k, int buffer) {
        #pragma unroll
        for (int j = 0; j < 2; j++) {
            int chunk = tid + j * 256, row = chunk / 4, column = (chunk % 4) * 8;
            bool valid = row0 + row < rows;
            const unsigned short *source =
                valid ? weights + (u64)(row0 + row) * width + k + column : weights;
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;"
                :: "r"(gf16_shared(&a_tile[buffer][row][column])), "l"(source),
                   "r"(valid ? 16 : 0) : "memory");
        }
        asm volatile("cp.async.commit_group;" ::: "memory");
    };
    // 128 tokens x 32 floats = 1024 float4, four per thread.
    float4 staged[4];
    auto fetch_inputs = [&](int k) {
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            int item = tid + j * 256, token = item / 8, column = (item % 8) * 4;
            staged[j] = token0 + token < tokens
                ? *(const float4 *)(x + (u64)(token0 + token) * width + k + column)
                : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        }
    };
    auto store_inputs = [&](int buffer) {
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            int item = tid + j * 256, token = item / 8, column = (item % 8) * 4;
            *(uint2 *)&b_tile[buffer][token][column] = make_uint2(
                gf16_pair(staged[j].x, staged[j].y), gf16_pair(staged[j].z, staged[j].w));
        }
    };

    float acc[4][4][4];
    #pragma unroll
    for (int m = 0; m < 4; m++)
        #pragma unroll
        for (int n = 0; n < 4; n++)
            #pragma unroll
            for (int i = 0; i < 4; i++) acc[m][n][i] = 0.0f;

    load_weights(0, 0);
    fetch_inputs(0);
    store_inputs(0);
    asm volatile("cp.async.wait_group 0;" ::: "memory");
    __syncthreads();
    for (int k = 0; k < width; k += GF16_BK) {
        const int current = (k / GF16_BK) % 2;
        const bool more = k + GF16_BK < width;
        if (more) {
            load_weights(k + GF16_BK, current ^ 1);
            fetch_inputs(k + GF16_BK);
        }
        #pragma unroll
        for (int kk = 0; kk < GF16_BK; kk += 16) {
            unsigned int a[4][4], b[2][4];
            #pragma unroll
            for (int m = 0; m < 4; m++)
                gf16_ldmatrix(a[m], &a_tile[current][warp_row + m * 16 + lane % 16][kk + (lane / 16) * 8]);
            // Each x4 load holds two n8 tiles: {b0, b1} of tile 2p, then of 2p + 1.
            #pragma unroll
            for (int p = 0; p < 2; p++)
                gf16_ldmatrix(b[p], &b_tile[current][warp_token + p * 16 + (lane / 16) * 8 + lane % 8]
                                           [kk + ((lane / 8) % 2) * 8]);
            #pragma unroll
            for (int m = 0; m < 4; m++)
                #pragma unroll
                for (int n = 0; n < 4; n++) {
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                            : "+f"(acc[m][n][0]), "+f"(acc[m][n][1]), "+f"(acc[m][n][2]), "+f"(acc[m][n][3])
                            : "r"(a[m][0]), "r"(a[m][1]), "r"(a[m][2]), "r"(a[m][3]),
                              "r"(b[n / 2][(n % 2) * 2]), "r"(b[n / 2][(n % 2) * 2 + 1]));
                }
        }
        if (more) store_inputs(current ^ 1);
        asm volatile("cp.async.wait_group 0;" ::: "memory");
        __syncthreads();
    }
    #pragma unroll
    for (int m = 0; m < 4; m++) {
        #pragma unroll
        for (int n = 0; n < 4; n++) {
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                int row = row0 + warp_row + m * 16 + group + (i / 2) * 8;
                int token = token0 + warp_token + n * 8 + thread * 2 + i % 2;
                if (row < rows && token < tokens) y[(u64)token * rows + row] = acc[m][n][i];
            }
        }
    }
#endif
}

