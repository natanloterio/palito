// QTIP prototype: batch-1 matrix-vector product with packed trellis tiles
// (layout in src/qtip/packed.rs) and the Walsh-Hadamard transforms around it.
// Research only (feature qtip-research); compiled by examples/qtip_matvec_bench.rs.
//
// y = sigma * W' x' with W'[r][c] = one_mad(state of weight t) and the state the
// QTIP_L-bit window at bit QTIP_K * t of the tile's MSB-first stream.

#ifndef QTIP_L
#define QTIP_L 12
#endif
#ifndef QTIP_K
#define QTIP_K 3
#endif
#define QTIP_TILE 16
#define QTIP_WEIGHTS (QTIP_TILE * QTIP_TILE)
#define QTIP_WORDS ((QTIP_L + QTIP_K * (QTIP_WEIGHTS - 1) + 31) / 32)
// Each lane decodes 8 consecutive weights: half a tile row.
#define QTIP_PER_LANE 8
#define QTIP_MAX_WARPS 32

static_assert(QTIP_WORDS <= 32, "a tile must fit one word per lane");
static_assert(QTIP_K * (QTIP_PER_LANE - 1) + QTIP_L + 31 <= 64,
              "a lane's bit window must fit two words at any alignment");

#ifndef QTIP_UNROLL
#define QTIP_UNROLL 4
#endif
// 2^23 + 510: OR-ing a byte sum (< 2^23) into the mantissa of 2^23 gives
// 2^23 + sum exactly, so subtracting this is an exact int-to-float without I2F.
#define QTIP_MAGIC_OFFSET 8389118.0f

// 1MAD without its normalization: (byte sum of s * a + b) - 510, exact in f32.
// The caller folds 1 / 147.80054 into the scale.
__device__ __forceinline__ float qtip_code(unsigned state) {
    const unsigned mixed = state * 34038481u + 76625530u;
    // dp4a's accumulator adds the exponent bits for free: sum < 2^23, so
    // 0x4B000000 + sum is the float 2^23 + sum.
    return __int_as_float(__dp4a(mixed, 0x01010101u, 0x4B000000u)) - QTIP_MAGIC_OFFSET;
}

// grid (rows / 16, splits), block = 32 * warps threads, dynamic shared =
// tiles_per_split * 16 floats. y must be zeroed: blocks of one row group add
// their split's partial sums with atomicAdd. scale = sigma / 147.80054.
extern "C" __global__ void qtip_matvec(const unsigned* __restrict__ tiles,
                                       const float* __restrict__ x,
                                       float* __restrict__ y,
                                       int cols, int tiles_per_split, float scale) {
    extern __shared__ float4 shared_x4[];
    float* xs = reinterpret_cast<float*>(shared_x4);
    __shared__ float partial[QTIP_MAX_WARPS][QTIP_TILE];
    const int group = blockIdx.x;
    const int col_tiles = cols / QTIP_TILE;
    const int k0 = blockIdx.y * tiles_per_split;
    const int k1 = min(k0 + tiles_per_split, col_tiles);
    const float4* x4 = reinterpret_cast<const float4*>(x + k0 * QTIP_TILE);
    for (int i = threadIdx.x; i < (k1 - k0) * QTIP_TILE / 4; i += blockDim.x) {
        shared_x4[i] = x4[i];
    }
    __syncthreads();

    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32, warps = blockDim.x / 32;
    const int first = lane * QTIP_PER_LANE;
    const int row_in_tile = first / QTIP_TILE;
    const int col_in_tile = first % QTIP_TILE;
    const int bit = first * QTIP_K;
    const int hi_word = bit / 32, shift = bit % 32;
    const int lo_word = min(hi_word + 1, 31);
    const unsigned* base = tiles + (size_t)group * col_tiles * QTIP_WORDS;
    float acc = 0.0f;
    // QTIP_UNROLL tiles per step: issue all their loads before decoding.
    for (int k = k0 + warp; k < k1; k += warps * QTIP_UNROLL) {
        unsigned words[QTIP_UNROLL];
#pragma unroll
        for (int u = 0; u < QTIP_UNROLL; ++u) {
            const int kk = k + u * warps;
            words[u] = (lane < QTIP_WORDS && kk < k1) ? __ldg(base + (size_t)kk * QTIP_WORDS + lane) : 0u;
        }
#pragma unroll
        for (int u = 0; u < QTIP_UNROLL; ++u) {
            const int kk = k + u * warps;
            const unsigned hi = __shfl_sync(0xffffffffu, words[u], hi_word);
            const unsigned lo = __shfl_sync(0xffffffffu, words[u], lo_word);
            if (kk >= k1) {
                break;
            }
            // Top 32 bits of the window and the next 32: bits [shift, shift + 64) of hi:lo:0.
            const unsigned top = __funnelshift_l(lo, hi, shift);
            const unsigned low = lo << shift;
            const float* xv = xs + (kk - k0) * QTIP_TILE + col_in_tile;
            const float4 xa = *reinterpret_cast<const float4*>(xv);
            const float4 xb = *reinterpret_cast<const float4*>(xv + 4);
            const float xx[QTIP_PER_LANE] = {xa.x, xa.y, xa.z, xa.w, xb.x, xb.y, xb.z, xb.w};
#pragma unroll
            for (int j = 0; j < QTIP_PER_LANE; ++j) {
                // State j = window bits [K j, K j + L), from the top of top:low.
                const int end = QTIP_K * j + QTIP_L;
                const unsigned field = end <= 32 ? (top >> (32 - end))
                                                 : __funnelshift_l(low, top, end - 32);
                acc = fmaf(qtip_code(field & ((1u << QTIP_L) - 1u)), xx[j], acc);
            }
        }
    }
    // Lanes 2r and 2r + 1 hold the two halves of tile row r.
    acc += __shfl_xor_sync(0xffffffffu, acc, 1);
    if ((lane & 1) == 0) {
        partial[warp][row_in_tile] = acc;
    }
    __syncthreads();
    if (threadIdx.x < QTIP_TILE) {
        float sum = 0.0f;
        for (int w = 0; w < warps; ++w) {
            sum += partial[w][threadIdx.x];
        }
        atomicAdd(&y[group * QTIP_TILE + threadIdx.x], sum * scale);
    }
}

// Normalized Walsh-Hadamard transform of one vector of length n (power of
// two), one block, dynamic shared = n floats. signs_first: out = H (S in);
// otherwise out = scale * S (H in).
extern "C" __global__ void qtip_fwht(const float* __restrict__ in, float* __restrict__ out,
                                     const float* __restrict__ signs, int n, int signs_first,
                                     float scale) {
    extern __shared__ float v[];
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        v[i] = signs_first ? in[i] * signs[i] : in[i];
    }
    __syncthreads();
    for (int span = 1; span < n; span <<= 1) {
        for (int p = threadIdx.x; p < n / 2; p += blockDim.x) {
            const int i = (p / span) * 2 * span + (p % span);
            const float a = v[i], b = v[i + span];
            v[i] = a + b;
            v[i + span] = a - b;
        }
        __syncthreads();
    }
    const float norm = 1.0f / sqrtf((float)n);
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const float r = v[i] * norm;
        out[i] = signs_first ? r : r * signs[i] * scale;
    }
}

__device__ __forceinline__ float qtip_half(unsigned short bits) {
    float value;
    asm("cvt.f32.f16 %0, %1;" : "=f"(value) : "h"(bits));
    return value;
}

// Bandwidth reference: y = W x for row-major F16 W, one warp per row,
// 16-byte loads. grid rows / 8, block 256. cols must be a multiple of 256.
extern "C" __global__ void f16_matvec(const uint4* __restrict__ w, const float* __restrict__ x,
                                      float* __restrict__ y, int rows, int cols) {
    const int row = blockIdx.x * 8 + threadIdx.x / 32, lane = threadIdx.x % 32;
    if (row >= rows) {
        return;
    }
    const uint4* line = w + (size_t)row * cols / 8;
    float acc = 0.0f;
    for (int i = lane; i < cols / 8; i += 32) {
        const uint4 packed = __ldg(line + i);
        const unsigned parts[4] = {packed.x, packed.y, packed.z, packed.w};
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            acc = fmaf(qtip_half((unsigned short)(parts[j] & 0xffffu)), x[i * 8 + 2 * j], acc);
            acc = fmaf(qtip_half((unsigned short)(parts[j] >> 16)), x[i * 8 + 2 * j + 1], acc);
        }
    }
    for (int offset = 16; offset > 0; offset /= 2) {
        acc += __shfl_xor_sync(0xffffffffu, acc, offset);
    }
    if (lane == 0) {
        y[row] = acc;
    }
}

// Fast normalized Walsh-Hadamard transform for 1024 <= n <= 32768: one block
// of 1024 threads, E = n / 1024 values per thread. Index bits: low 5 (b_low),
// middle 5 (b_mid), high log2(E) (b_high). Layout A (thread = b_low | b_mid << 5,
// register = b_high) does the b_high stages in registers and the b_low stages
// with lane shuffles; one padded shared-memory transpose gives layout B
// (lane = b_mid, warp = b_low), where the b_mid stages are lane shuffles too.
// Dynamic shared = (n + n / 32) floats. Same contract as qtip_fwht.
#define QTIP_FWHT_MAX_E 32
__device__ __forceinline__ int qtip_pad(int i) { return i + (i >> 5); }

template <int E>
__device__ __forceinline__ void qtip_fwht_fast_body(const float* __restrict__ in, float* __restrict__ out,
                                                    const float* __restrict__ signs, int n, int signs_first,
                                                    float scale, float* sh) {
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    float v[E];
#pragma unroll
    for (int e = 0; e < E; ++e) {
        const int i = t + 1024 * e;
        v[e] = signs_first ? in[i] * signs[i] : in[i];
    }
#pragma unroll
    for (int span = 1; span < E; span <<= 1) {
#pragma unroll
        for (int e = 0; e < E; ++e) {
            if ((e & span) == 0) {
                const float a = v[e], b = v[e + span];
                v[e] = a + b;
                v[e + span] = a - b;
            }
        }
    }
    // b_low stages: the index's low 5 bits are this thread's lane.
#pragma unroll
    for (int span = 1; span < 32; span <<= 1) {
#pragma unroll
        for (int e = 0; e < E; ++e) {
            const float other = __shfl_xor_sync(0xffffffffu, v[e], span);
            v[e] = (lane & span) ? other - v[e] : v[e] + other;
        }
    }
#pragma unroll
    for (int e = 0; e < E; ++e) {
        sh[qtip_pad(t + 1024 * e)] = v[e];
    }
    __syncthreads();
    // Layout B: this thread holds index warp | lane << 5 | e << 10.
#pragma unroll
    for (int e = 0; e < E; ++e) {
        v[e] = sh[qtip_pad(warp | (lane << 5) | (e << 10))];
    }
#pragma unroll
    for (int span = 1; span < 32; span <<= 1) {
#pragma unroll
        for (int e = 0; e < E; ++e) {
            const float other = __shfl_xor_sync(0xffffffffu, v[e], span);
            v[e] = (lane & span) ? other - v[e] : v[e] + other;
        }
    }
    __syncthreads();
#pragma unroll
    for (int e = 0; e < E; ++e) {
        sh[qtip_pad(warp | (lane << 5) | (e << 10))] = v[e];
    }
    __syncthreads();
    const float norm = 1.0f / sqrtf((float)n);
#pragma unroll
    for (int e = 0; e < E; ++e) {
        const int i = t + 1024 * e;
        const float r = sh[qtip_pad(i)] * norm;
        out[i] = signs_first ? r : r * signs[i] * scale;
    }
}

extern "C" __global__ void __launch_bounds__(1024)
qtip_fwht_fast(const float* __restrict__ in, float* __restrict__ out, const float* __restrict__ signs,
               int n, int signs_first, float scale) {
    extern __shared__ float sh[];
    switch (n >> 10) {
        case 1: qtip_fwht_fast_body<1>(in, out, signs, n, signs_first, scale, sh); break;
        case 2: qtip_fwht_fast_body<2>(in, out, signs, n, signs_first, scale, sh); break;
        case 4: qtip_fwht_fast_body<4>(in, out, signs, n, signs_first, scale, sh); break;
        case 8: qtip_fwht_fast_body<8>(in, out, signs, n, signs_first, scale, sh); break;
        case 16: qtip_fwht_fast_body<16>(in, out, signs, n, signs_first, scale, sh); break;
        case 32: qtip_fwht_fast_body<32>(in, out, signs, n, signs_first, scale, sh); break;
        default: break;
    }
}

// Fused product with block-diagonal Hadamards of 256 (Incoherence::with_block):
//   y = scale * S_r (I (x) H_256) W' (I (x) H_256) S_c x
// with W' the decoded tiles. A block owns 256 rows (16 row groups, one warp
// each, so blockDim = 512) and the column tiles [k0, k1) of its split. It
// rotates the 256-column chunks its range touches in shared memory, and
// rotates its 256 partial sums before adding them to y (zeroed): the output
// Hadamard is linear, so rotating each split's partial sum is exact.
// tiles_per_split * 16 must divide 256 or be a multiple of 256 so a split
// touches whole chunks only once. Dynamic shared = chunks * 256 floats with
// chunks = ceil over the split. scale folds sigma / 147.80054 and both 1/16 norms.
#define QTIP_CHUNK 256
#define QTIP_FUSED_WARPS 16

// Unnormalized H_256 of v[lane + 32 e], e < 8, held by one warp: index bits
// 5..7 are the register, bits 0..4 the lane.
__device__ __forceinline__ void qtip_fwht256_warp(float (&v)[8]) {
    const int lane = threadIdx.x & 31;
#pragma unroll
    for (int span = 1; span < 8; span <<= 1) {
#pragma unroll
        for (int e = 0; e < 8; ++e) {
            if ((e & span) == 0) {
                const float a = v[e], b = v[e + span];
                v[e] = a + b;
                v[e + span] = a - b;
            }
        }
    }
#pragma unroll
    for (int span = 1; span < 32; span <<= 1) {
#pragma unroll
        for (int e = 0; e < 8; ++e) {
            const float other = __shfl_xor_sync(0xffffffffu, v[e], span);
            v[e] = (lane & span) ? other - v[e] : v[e] + other;
        }
    }
}

// One block of the fused product: row block `row_block` (256 rows) and
// column split `split`. With x_up, the input is SwiGLU(x, x_up) =
// x / (1 + exp(-x)) * x_up, computed while loading (Palito's swiglu kernel).
__device__ __forceinline__ void qtip_fused_block(
    const unsigned* __restrict__ tiles, const float* __restrict__ x,
    const float* __restrict__ x_up, const float* __restrict__ col_signs,
    const float* __restrict__ row_signs, float* __restrict__ y, int cols,
    int tiles_per_split, float scale, int row_block, int split, float* xs, float* partial) {
    const int col_tiles = cols / QTIP_TILE;
    const int k0 = split * tiles_per_split;
    const int k1 = min(k0 + tiles_per_split, col_tiles);
    // Whole 256-column chunks covering [k0 * 16, k1 * 16).
    const int c_first = (k0 * QTIP_TILE) / QTIP_CHUNK * QTIP_CHUNK;
    const int chunks = (k1 * QTIP_TILE - c_first + QTIP_CHUNK - 1) / QTIP_CHUNK;
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    // One warp per 256-column chunk: load, sign, rotate in registers, store.
    for (int chunk = warp; chunk < chunks; chunk += QTIP_FUSED_WARPS) {
        const int c = c_first + chunk * QTIP_CHUNK;
        // Issue every load of the chunk before using any: a branch between
        // them would serialize eight memory latencies.
        float v[8], sign[8];
#pragma unroll
        for (int e = 0; e < 8; ++e) {
            const int i = c + lane + 32 * e;
            // __ldg: pointers from the launch struct lose __restrict__.
            v[e] = __ldg(x + i);
            sign[e] = __ldg(col_signs + i);
        }
        if (x_up != nullptr) {
            float up[8];
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                up[e] = __ldg(x_up + c + lane + 32 * e);
            }
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                v[e] = v[e] / (1.0f + expf(-v[e])) * up[e];
            }
        }
#pragma unroll
        for (int e = 0; e < 8; ++e) {
            v[e] *= sign[e];
        }
        qtip_fwht256_warp(v);
#pragma unroll
        for (int e = 0; e < 8; ++e) {
            xs[chunk * QTIP_CHUNK + lane + 32 * e] = v[e];
        }
    }
    __syncthreads();

    const int group = row_block * QTIP_FUSED_WARPS + warp;
    const int first = lane * QTIP_PER_LANE;
    const int row_in_tile = first / QTIP_TILE;
    const int col_in_tile = first % QTIP_TILE;
    const int bit = first * QTIP_K;
    const int hi_word = bit / 32, shift = bit % 32;
    const int lo_word = min(hi_word + 1, 31);
    const unsigned* base = tiles + (size_t)group * col_tiles * QTIP_WORDS;
    const float* xbase = xs + (k0 * QTIP_TILE - c_first);
    float acc = 0.0f;
    for (int k = k0; k < k1; k += QTIP_UNROLL) {
        unsigned words[QTIP_UNROLL];
#pragma unroll
        for (int u = 0; u < QTIP_UNROLL; ++u) {
            const int kk = k + u;
            words[u] = (lane < QTIP_WORDS && kk < k1) ? __ldg(base + (size_t)kk * QTIP_WORDS + lane) : 0u;
        }
#pragma unroll
        for (int u = 0; u < QTIP_UNROLL; ++u) {
            const int kk = k + u;
            const unsigned hi = __shfl_sync(0xffffffffu, words[u], hi_word);
            const unsigned lo = __shfl_sync(0xffffffffu, words[u], lo_word);
            if (kk >= k1) {
                break;
            }
            const unsigned top = __funnelshift_l(lo, hi, shift);
            const unsigned low = lo << shift;
            const float* xv = xbase + (kk - k0) * QTIP_TILE + col_in_tile;
            const float4 xa = *reinterpret_cast<const float4*>(xv);
            const float4 xb = *reinterpret_cast<const float4*>(xv + 4);
            const float xx[QTIP_PER_LANE] = {xa.x, xa.y, xa.z, xa.w, xb.x, xb.y, xb.z, xb.w};
#pragma unroll
            for (int j = 0; j < QTIP_PER_LANE; ++j) {
                const int end = QTIP_K * j + QTIP_L;
                const unsigned field = end <= 32 ? (top >> (32 - end))
                                                 : __funnelshift_l(low, top, end - 32);
                acc = fmaf(qtip_code(field & ((1u << QTIP_L) - 1u)), xx[j], acc);
            }
        }
    }
    acc += __shfl_xor_sync(0xffffffffu, acc, 1);
    if ((lane & 1) == 0) {
        partial[warp * QTIP_TILE + row_in_tile] = acc;
    }
    __syncthreads();
    if (warp == 0) {
        float v[8];
#pragma unroll
        for (int e = 0; e < 8; ++e) {
            v[e] = partial[lane + 32 * e];
        }
        qtip_fwht256_warp(v);
#pragma unroll
        for (int e = 0; e < 8; ++e) {
            const int row = row_block * QTIP_CHUNK + lane + 32 * e;
            atomicAdd(&y[row], v[e] * row_signs[row] * scale);
        }
    }
}

extern "C" __global__ void __launch_bounds__(512)
qtip_matvec_fused(const unsigned* __restrict__ tiles, const float* __restrict__ x,
                  const float* __restrict__ col_signs, const float* __restrict__ row_signs,
                  float* __restrict__ y, int cols, int tiles_per_split, float scale) {
    extern __shared__ float4 shared_x4[];
    __shared__ float partial[QTIP_CHUNK];
    qtip_fused_block(tiles, x, nullptr, col_signs, row_signs, y, cols, tiles_per_split, scale,
                     blockIdx.x, blockIdx.y, reinterpret_cast<float*>(shared_x4), partial);
}

// Up to three matrices (count) with the same input in one launch (q/k/v,
// gate/up): row blocks [0, rb0) are matrix 0, the next rb1 matrix 1, the rest
// matrix 2; all share cols and tiles_per_split. z0..z2 (n0..n2 floats) are
// buffers of a later projection, zeroed on the way. Scalar __restrict__
// parameters: pointers read from a struct lose them, which cost ~1 us here.
#define QTIP_MULTI_PARAMS                                                                  \
    const unsigned* __restrict__ t0, const unsigned* __restrict__ t1,                       \
        const unsigned* __restrict__ t2, const float* __restrict__ c0,                      \
        const float* __restrict__ c1, const float* __restrict__ c2,                         \
        const float* __restrict__ r0, const float* __restrict__ r1,                         \
        const float* __restrict__ r2, float* __restrict__ y0, float* __restrict__ y1,       \
        float* __restrict__ y2, float s0, float s1, float s2, int rb0, int rb1,             \
        const float* __restrict__ x, const float* __restrict__ x_up, float* __restrict__ z0, \
        float* __restrict__ z1, float* __restrict__ z2, int n0, int n1, int n2, int cols,   \
        int tiles_per_split

template <bool SwiGLU>
__device__ __forceinline__ void qtip_multi_body(QTIP_MULTI_PARAMS) {
    extern __shared__ float4 shared_x4[];
    __shared__ float partial[QTIP_CHUNK];
    const int threads = gridDim.x * gridDim.y * blockDim.x;
    const int id = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
    for (int i = id; i < n0; i += threads) z0[i] = 0.0f;
    for (int i = id; i < n1; i += threads) z1[i] = 0.0f;
    for (int i = id; i < n2; i += threads) z2[i] = 0.0f;
    int row_block = blockIdx.x;
    const unsigned* tiles = t0;
    const float *col_signs = c0, *row_signs = r0;
    float* y = y0;
    float scale = s0;
    if (row_block >= rb0) {
        row_block -= rb0;
        tiles = t1, col_signs = c1, row_signs = r1, y = y1, scale = s1;
        if (row_block >= rb1) {
            row_block -= rb1;
            tiles = t2, col_signs = c2, row_signs = r2, y = y2, scale = s2;
        }
    }
    qtip_fused_block(tiles, x, SwiGLU ? x_up : nullptr, col_signs, row_signs, y, cols,
                     tiles_per_split, scale, row_block, blockIdx.y,
                     reinterpret_cast<float*>(shared_x4), partial);
}

#define QTIP_MULTI_ARGS                                                                    \
    t0, t1, t2, c0, c1, c2, r0, r1, r2, y0, y1, y2, s0, s1, s2, rb0, rb1, x, x_up, z0, z1, \
        z2, n0, n1, n2, cols, tiles_per_split

extern "C" __global__ void __launch_bounds__(512) qtip_matvec_multi(QTIP_MULTI_PARAMS) {
    qtip_multi_body<false>(QTIP_MULTI_ARGS);
}

// qtip_matvec_multi on SwiGLU(x, x_up).
extern "C" __global__ void __launch_bounds__(512) qtip_matvec_multi_swiglu(QTIP_MULTI_PARAMS) {
    qtip_multi_body<true>(QTIP_MULTI_ARGS);
}

// Fused small-batch (2 <= tokens <= 8) QTIP Tensor-Core verification kernel:
// 1 single launch per projection (12 KB static SMEM => 100% occupancy on SM_120),
// fused input+output 256-block FWHT, and direct register-to-Tensor-Core
// mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 (1 MMA computes all 8 tokens!).
// Grid: (rows / 256, splits), Block: 512 threads (16 warps).
// `chunks_per_split` is in 256-column chunks (16 tiles per chunk).
// `y` must be zeroed before launch (atomicAdd across splits).
extern "C" __global__ void __launch_bounds__(512)
qtip_matvec_small_k(const unsigned* __restrict__ tiles,
                    const float* __restrict__ x,
                    const float* __restrict__ col_signs,
                    const float* __restrict__ row_signs,
                    float* __restrict__ y,
                    int rows,
                    int cols,
                    int tokens,
                    int chunks_per_split,
                    float scale) {
    __shared__ unsigned xs_h2[8][QTIP_CHUNK / 2]; // 4 KB: 128 half2 per token
    __shared__ float partial[8][QTIP_CHUNK];      // 8 KB: scratch + output partial

    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x & 31;
    const int group_id = lane >> 2;   // 0..7
    const int tid = lane & 3;         // 0..3
    const int col_tiles = cols / QTIP_TILE;
    const int total_chunks = cols / QTIP_CHUNK;
    const int ch0 = blockIdx.y * chunks_per_split;
    const int ch1 = min(ch0 + chunks_per_split, total_chunks);

    const int group = blockIdx.x * QTIP_FUSED_WARPS + warp;
    const unsigned* base = tiles + (size_t)group * col_tiles * QTIP_WORDS;

    // Precompute the 4 (hi_word, lo_word, shift) tuples for this lane's MMA A-fragment pairs.
    const int t_idx[4] = {
        group_id * QTIP_TILE + 2 * tid,
        (group_id + 8) * QTIP_TILE + 2 * tid,
        group_id * QTIP_TILE + 2 * tid + 8,
        (group_id + 8) * QTIP_TILE + 2 * tid + 8,
    };
    int hi_w[4], lo_w[4], sh[4];
#pragma unroll
    for (int p = 0; p < 4; ++p) {
        const int bit = t_idx[p] * QTIP_K;
        hi_w[p] = bit / 32;
        lo_w[p] = min(hi_w[p] + 1, 31);
        sh[p] = bit & 31;
    }

    float d0 = 0.0f, d1 = 0.0f, d2 = 0.0f, d3 = 0.0f;

    for (int ch = ch0; ch < ch1; ++ch) {
        const int c = ch * QTIP_CHUNK;
        // One warp per token rotates this 256-column chunk in parallel and packs to half2.
        if (warp < tokens) {
            float v[8], sign[8];
            const float* xt = x + (size_t)warp * cols + c;
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                const int idx = lane + 32 * e;
                v[e] = __ldg(xt + idx);
                sign[e] = __ldg(col_signs + c + idx);
            }
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                v[e] *= sign[e];
            }
            qtip_fwht256_warp(v);
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                partial[warp][lane + 32 * e] = v[e];
            }
            __syncwarp();
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                const int j = lane + 32 * e;
                const float x0 = partial[warp][2 * j];
                const float x1 = partial[warp][2 * j + 1];
                unsigned h2;
                asm("cvt.rn.f16x2.f32 %0, %2, %1;" : "=r"(h2) : "f"(x0), "f"(x1));
                xs_h2[warp][j] = h2;
            }
        }
        __syncthreads();

        const int k0 = ch * 16;
        for (int tk = 0; tk < 16; tk += QTIP_UNROLL) {
            unsigned words[QTIP_UNROLL];
#pragma unroll
            for (int u = 0; u < QTIP_UNROLL; ++u) {
                const int kk = k0 + tk + u;
                words[u] = (lane < QTIP_WORDS) ? __ldg(base + (size_t)kk * QTIP_WORDS + lane) : 0u;
            }
#pragma unroll
            for (int u = 0; u < QTIP_UNROLL; ++u) {
                unsigned a[4];
#pragma unroll
                for (int p = 0; p < 4; ++p) {
                    const unsigned hi = __shfl_sync(0xffffffffu, words[u], hi_w[p]);
                    const unsigned lo = __shfl_sync(0xffffffffu, words[u], lo_w[p]);
                    const unsigned top = __funnelshift_l(lo, hi, sh[p]);
                    const unsigned s0 = top >> (32 - QTIP_L);
                    const unsigned s1 = (top >> (32 - QTIP_L - QTIP_K)) & ((1u << QTIP_L) - 1u);
                    const float w0 = qtip_code(s0);
                    const float w1 = qtip_code(s1);
                    asm("cvt.rn.f16x2.f32 %0, %2, %1;" : "=r"(a[p]) : "f"(w0), "f"(w1));
                }
                const int xoff = (tk + u) * 8 + tid;
                const unsigned b0 = (group_id < tokens) ? xs_h2[group_id][xoff] : 0u;
                const unsigned b1 = (group_id < tokens) ? xs_h2[group_id][xoff + 4] : 0u;
                asm volatile(
                    "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                    "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                    : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
                    : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
            }
        }
        __syncthreads();
    }

    const int tok0 = 2 * tid;
    const int tok1 = tok0 + 1;
    if (tok0 < tokens) {
        partial[tok0][warp * QTIP_TILE + group_id] = d0;
        partial[tok0][warp * QTIP_TILE + group_id + 8] = d2;
    }
    if (tok1 < tokens) {
        partial[tok1][warp * QTIP_TILE + group_id] = d1;
        partial[tok1][warp * QTIP_TILE + group_id + 8] = d3;
    }
    __syncthreads();

    // One warp per token rotates the 256-row output chunk in parallel.
    if (warp < tokens) {
        float v[8];
#pragma unroll
        for (int e = 0; e < 8; ++e) {
            v[e] = partial[warp][lane + 32 * e];
        }
        qtip_fwht256_warp(v);
        float* yt = y + (size_t)warp * rows;
#pragma unroll
        for (int e = 0; e < 8; ++e) {
            const int row = blockIdx.x * QTIP_CHUNK + lane + 32 * e;
            atomicAdd(&yt[row], v[e] * __ldg(row_signs + row) * scale);
        }
    }
}
