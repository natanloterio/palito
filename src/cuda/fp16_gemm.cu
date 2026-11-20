#ifdef CUDA_FP16_MERGE_REPLAY
extern "C" __global__ void merge_prefill_fp16(float *base, const float *a,
                                             const float *b, int count) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) base[i] = (base[i] + a[i]) + b[i];
}
#endif

extern "C" __global__ void expand_prefill_fp16(const u8 *src, int kind,
                                               unsigned short *dst, int count, int residual) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) {
        const float value = weight(src, kind, i);
        unsigned short bits;
        asm("cvt.rn.f16.f32 %0, %1;" : "=h"(bits) : "f"(value));
        if (residual) {
            float high;
            asm("cvt.f32.f16 %0, %1;" : "=f"(high) : "h"(bits));
            const float low = (value - high) * 1024.0f;
            asm("cvt.rn.f16.f32 %0, %1;" : "=h"(bits) : "f"(low));
        }

        dst[i] = bits;
    }
}

#ifdef CUDA_FP32_GEMM_REPLAY
extern "C" __global__ void expand_prefill_f32(const u8 *src, int kind,
                                              float *dst, int count) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) dst[i] = weight(src, kind, i);
}
#endif

#ifdef CUDA_FP16_NN_REPLAY
extern "C" __global__ void transpose_prefill_fp16(const unsigned short *src,
    unsigned short *dst, int width, int rows) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < width * rows) dst[(i % width) * rows + i / width] = src[i];
}
#endif
#ifdef CUDA_FP16_TILE_REPLAY
__device__ __forceinline__ void corrected_mma(float (&c)[4],
    unsigned a0, unsigned a1, unsigned a2, unsigned a3, unsigned b0, unsigned b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ void corrected_copy(unsigned *dst, const unsigned *src, int bytes) {
    const unsigned addr = (unsigned)__cvta_generic_to_shared(dst);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;"
        :: "r"(addr), "l"(src), "r"(bytes) : "memory");
}

__device__ __forceinline__ int corrected_swizzle(int row, int col) {
    return col ^ ((row & 3) * 4);
}

template<int TileRows, bool Async = false, int Depth = 16, int TileTokens = 32,
         bool WeightFirst = false, bool Split = false>
__device__ __forceinline__ void corrected_fp16_tile(const unsigned *wh, const unsigned *wl,
    const unsigned *xh, const unsigned *xl, float *output, int width, int rows, int tokens) {
    static_assert(!Async || TileRows == 32 || TileRows == 64);
    static_assert(Async || TileTokens == 32);
    constexpr int slots = Async ? 2 : 1, pitch = Async ? Depth : 17;
    constexpr int groups = TileTokens / 16;
    __shared__ __align__(16) unsigned a[slots][2][TileRows][pitch], b[slots][2][TileTokens][pitch];
    constexpr int threads = TileRows * 4;
    const int tid = threadIdx.x, warp = tid / 32, lane = tid % 32;
    const int tile = blockIdx.y * gridDim.x + blockIdx.x;
    const int m = (WeightFirst ? tile / gridDim.y : blockIdx.x) * TileRows;
    const int n = (WeightFirst ? tile % gridDim.y : blockIdx.y) * TileTokens;
    const int r = (warp / 2) * 16 + lane / 4, c = (warp % 2) * groups * 8, pair = lane % 4;
    const int stride = width / 2;
    const int first = Split ? blockIdx.z * (stride / 2) : 0;
    const int end = Split ? first + stride / 2 : stride;
    if constexpr (Split) output += (u64)blockIdx.z * rows * tokens;
    float high[groups][4] = {}, corr_x[groups][4] = {}, corr_w[groups][4] = {};
    auto stage = [&](int slot, int k) {
        const int row = tid / 4;
        const bool valid_w = m + row < rows;
        #pragma unroll
        for (int j = (tid % 4) * 4; j < Depth; j += 16) {
            const u64 wi = valid_w ? (u64)(m + row) * stride + k + j : 0;
            const int s = corrected_swizzle(row, j);
            corrected_copy(&a[slot][0][row][s], wh + wi, valid_w ? 16 : 0);
            corrected_copy(&a[slot][1][row][s], wl + wi, valid_w ? 16 : 0);
            #pragma unroll
            for (int xr = row; xr < TileTokens; xr += TileRows) {
                const bool valid_x = n + xr < tokens;
                const u64 xi = valid_x ? (u64)(n + xr) * stride + k + j : 0;
                corrected_copy(&b[slot][0][xr][s], xh + xi, valid_x ? 16 : 0);
                corrected_copy(&b[slot][1][xr][s], xl + xi, valid_x ? 16 : 0);
            }
        }
        asm volatile("cp.async.commit_group;" ::: "memory");
    };
    if constexpr (Async) stage(0, first);
    for (int k = first; k < end; k += Depth) {
        const int slot = Async ? ((k - first) / Depth) & 1 : 0;
        if constexpr (Async) {
            asm volatile("cp.async.wait_group 0;" ::: "memory");
            __syncthreads();
            if (k + Depth < end) stage(slot ^ 1, k + Depth);
        } else {
        #pragma unroll
        for (int i = tid; i < TileRows * 16; i += threads) {
            const int row = i / 16, j = i % 16;
            const u64 wi = (u64)(m + row) * stride + k + j;
            const u64 xi = (u64)(n + row) * stride + k + j;
            a[0][0][row][j] = m + row < rows ? wh[wi] : 0;
            a[0][1][row][j] = m + row < rows ? wl[wi] : 0;
            if (row < 32) {
                b[0][0][row][j] = n + row < tokens ? xh[xi] : 0;
                b[0][1][row][j] = n + row < tokens ? xl[xi] : 0;
            }
        }
        __syncthreads();
        }
        #pragma unroll
        for (int h = 0; h < Depth; h += 8) {
            const int p = h + pair;
            const int ar = Async ? corrected_swizzle(r, p) : p;
            const unsigned ah0 = a[slot][0][r][ar], ah1 = a[slot][0][r + 8][ar];
            const unsigned ah2 = a[slot][0][r][Async ? ar ^ 4 : ar + 4], ah3 = a[slot][0][r + 8][Async ? ar ^ 4 : ar + 4];
            const unsigned al0 = a[slot][1][r][ar], al1 = a[slot][1][r + 8][ar];
            const unsigned al2 = a[slot][1][r][Async ? ar ^ 4 : ar + 4], al3 = a[slot][1][r + 8][Async ? ar ^ 4 : ar + 4];
            #pragma unroll
            for (int t = 0; t < groups; ++t) {
                const int col = c + 8 * t + lane / 4;
                const int br = Async ? corrected_swizzle(col, p) : p;
                const int br2 = Async ? br ^ 4 : br + 4;
                const unsigned bh0 = b[slot][0][col][br], bh1 = b[slot][0][col][br2];
                corrected_mma(high[t], ah0, ah1, ah2, ah3, bh0, bh1);
                corrected_mma(corr_x[t], ah0, ah1, ah2, ah3, b[slot][1][col][br], b[slot][1][col][br2]);
                corrected_mma(corr_w[t], al0, al1, al2, al3, bh0, bh1);
            }
        }
        if constexpr (!Async) __syncthreads();
    }
    #pragma unroll
    for (int t = 0; t < groups; ++t) {
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int row = m + r + (i / 2) * 8;
            const int col = n + c + t * 8 + pair * 2 + i % 2;
            if (row < rows && col < tokens)
                output[(u64)col * rows + row] =
                    (high[t][i] + corr_x[t][i] / 1024.0f) + corr_w[t][i] / 1024.0f;
        }
    }
}
extern "C" __global__ void corrected_fp16_tile32(const unsigned *wh, const unsigned *wl,
    const unsigned *xh, const unsigned *xl, float *output, int width, int rows, int tokens) {
    corrected_fp16_tile<32>(wh, wl, xh, xl, output, width, rows, tokens);
}
extern "C" __global__ void corrected_fp16_tile64(const unsigned *wh, const unsigned *wl,
    const unsigned *xh, const unsigned *xl, float *output, int width, int rows, int tokens) {
    corrected_fp16_tile<64>(wh, wl, xh, xl, output, width, rows, tokens);
}
extern "C" __global__ void corrected_fp16_tile_async(const unsigned *wh, const unsigned *wl,
    const unsigned *xh, const unsigned *xl, float *output, int width, int rows, int tokens) {
    corrected_fp16_tile<32, true>(wh, wl, xh, xl, output, width, rows, tokens);
}
extern "C" __global__ void corrected_fp16_tile_async_k64(const unsigned *wh, const unsigned *wl,
    const unsigned *xh, const unsigned *xl, float *output, int width, int rows, int tokens) {
    corrected_fp16_tile<32, true, 32>(wh, wl, xh, xl, output, width, rows, tokens);
}
extern "C" __global__ void corrected_fp16_tile_async_n64(const unsigned *wh, const unsigned *wl,
    const unsigned *xh, const unsigned *xl, float *output, int width, int rows, int tokens) {
    corrected_fp16_tile<32, true, 32, 64>(wh, wl, xh, xl, output, width, rows, tokens);
}
extern "C" __global__ void corrected_fp16_tile_async_n128(const unsigned *wh, const unsigned *wl,
    const unsigned *xh, const unsigned *xl, float *output, int width, int rows, int tokens) {
    corrected_fp16_tile<32, true, 16, 128>(wh, wl, xh, xl, output, width, rows, tokens);
}
extern "C" __global__ void corrected_fp16_tile_async_weight_first(const unsigned *wh, const unsigned *wl,
    const unsigned *xh, const unsigned *xl, float *output, int width, int rows, int tokens) {
    corrected_fp16_tile<32, true, 32, 64, true>(wh, wl, xh, xl, output, width, rows, tokens);
}
extern "C" __global__ void corrected_fp16_tile_async_split2(const unsigned *wh, const unsigned *wl,
    const unsigned *xh, const unsigned *xl, float *output, int width, int rows, int tokens) {
    corrected_fp16_tile<32, true, 32, 64, true, true>(wh, wl, xh, xl, output, width, rows, tokens);
}
extern "C" __global__ void corrected_fp16_tile_async_wide(const unsigned *wh, const unsigned *wl,
    const unsigned *xh, const unsigned *xl, float *output, int width, int rows, int tokens) {
    corrected_fp16_tile<64, true, 32, 64, true>(wh, wl, xh, xl, output, width, rows, tokens);
}
extern "C" __global__ void corrected_fp16_split2_fixup(float *output, const float *partials, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) output[i] = partials[i] + partials[size + i];
}
#endif

