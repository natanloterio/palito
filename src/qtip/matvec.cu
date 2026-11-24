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
