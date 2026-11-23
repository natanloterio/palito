// Palito native Metal compute shaders (MSL) for macOS Apple Silicon.
// Uses hardware simd_sum() (32-lane SIMD-group reductions) and native half.
// Supports F32 (0), F16 (1), Q5_0 (6), Q8_0 (8), Q4_K (12), Q5_K (13), Q6_K (14).

#include <metal_stdlib>
using namespace metal;

struct Params {
    uint p[8];
    float f[4];
};

static inline uint read_u16(device const uchar *p) {
    return uint(p[0]) | (uint(p[1]) << 8u);
}

static inline float half_val(device const uchar *p) {
    return float(as_type<half>(ushort(read_u16(p))));
}

static inline float decode_weight(device const uchar *w, uint kind, ulong elem) {
    if (kind == 0u) {
        return reinterpret_cast<device const float *>(w)[elem];
    }
    if (kind == 1u) {
        return half_val(w + elem * 2ul);
    }
    if (kind == 6u) {
        ulong blk = elem >> 5ul;
        uint i = uint(elem & 31ul);
        device const uchar *p = w + blk * 22ul;
        uint high = uint(p[2]) | (uint(p[3]) << 8u) | (uint(p[4]) << 16u) | (uint(p[5]) << 24u);
        uint low = (uint(p[6u + (i & 15u)]) >> (4u * (i >> 4u))) & 15u;
        int q = int(low | (((high >> i) & 1u) << 4u)) - 16;
        return half_val(p) * float(q);
    }
    if (kind == 8u) {
        ulong blk = elem >> 5ul;
        uint i = uint(elem & 31ul);
        device const uchar *p = w + blk * 34ul;
        return half_val(p) * float(reinterpret_cast<device const char *>(p)[2u + i]);
    }
    if (kind == 12u || kind == 13u) {
        ulong blk = elem >> 8ul;
        uint i = uint(elem & 255ul);
        ulong stride = (kind == 12u) ? 144ul : 176ul;
        device const uchar *p = w + blk * stride;
        float d = half_val(p);
        float dm = half_val(p + 2u);
        device const uchar *s = p + 4u;
        uint g = i >> 5u;
        uint sc = (g < 4u) ? (uint(s[g]) & 63u)
                           : ((uint(s[g + 4u]) & 15u) | ((uint(s[g - 4u]) >> 6u) << 4u));
        uint mn = (g < 4u) ? (uint(s[g + 4u]) & 63u)
                           : ((uint(s[g + 4u]) >> 4u) | ((uint(s[g]) >> 6u) << 4u));
        uint low_off = (kind == 12u) ? 16u : 48u;
        uint q = (uint(p[low_off + (i >> 6u) * 32u + (i & 31u)]) >> (4u * (g & 1u))) & 15u;
        if (kind == 13u) {
            q |= ((uint(p[16u + (i & 31u)]) >> g) & 1u) << 4u;
        }
        return d * float(sc) * float(q) - dm * float(mn);
    }
    // Q6_K (kind == 14)
    ulong blk = elem >> 8ul;
    uint i = uint(elem & 255ul);
    device const uchar *p = w + blk * 210ul;
    uint h = i >> 7u;
    uint lane = i & 31u;
    uint quarter = (i & 127u) >> 5u;
    uint low = (uint(p[h * 64u + lane + (quarter & 1u) * 32u]) >> (4u * (quarter >> 1u))) & 15u;
    uint high = (uint(p[128u + h * 32u + lane]) >> (2u * quarter)) & 3u;
    int q = int(low | (high << 4u)) - 32;
    int sc = int(reinterpret_cast<device const char *>(p)[192u + (i >> 4u)]);
    return half_val(p + 208u) * float(sc) * float(q);
}

static inline float tg_sum(uint tid, float val, threadgroup float *sh) {
    float s = simd_sum(val);
    if ((tid & 31u) == 0u) sh[tid >> 5u] = s;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total = (tid < 8u) ? sh[tid] : 0.0f;
    total = simd_sum(total);
    if (tid == 0u) sh[0] = total;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return sh[0];
}

// Fast 32-lane SIMD-group packed row dot product.
static inline float packed_row_dot(device const uchar *w, device const float *x,
                                   uint kind, ulong row_elem, uint width, uint lane) {
    float acc = 0.0f;
    if (kind == 12u && (width & 255u) == 0u) {
        uint chunk = lane >> 3u, sub = lane & 7u;
        uint g_lo = chunk * 2u, g_hi = g_lo + 1u;
        uint num_blocks = width >> 8u;
        ulong row_blk = row_elem >> 8ul;
        for (uint b = 0u; b < num_blocks; b++) {
            device const uchar *p = w + (row_blk + b) * 144ul;
            float d = half_val(p), dm = half_val(p + 2u);
            device const uchar *s = p + 4u;
            uint sc0 = (g_lo < 4u) ? (uint(s[g_lo]) & 63u) : ((uint(s[g_lo + 4u]) & 15u) | ((uint(s[g_lo - 4u]) >> 6u) << 4u));
            uint mn0 = (g_lo < 4u) ? (uint(s[g_lo + 4u]) & 63u) : ((uint(s[g_lo + 4u]) >> 4u) | ((uint(s[g_lo]) >> 6u) << 4u));
            uint sc1 = (g_hi < 4u) ? (uint(s[g_hi]) & 63u) : ((uint(s[g_hi + 4u]) & 15u) | ((uint(s[g_hi - 4u]) >> 6u) << 4u));
            uint mn1 = (g_hi < 4u) ? (uint(s[g_hi + 4u]) & 63u) : ((uint(s[g_hi + 4u]) >> 4u) | ((uint(s[g_hi]) >> 6u) << 4u));
            float ds0 = d * float(sc0), ms0 = dm * float(mn0);
            float ds1 = d * float(sc1), ms1 = dm * float(mn1);
            device const uchar *qptr = p + 16u + chunk * 32u + sub * 4u;
            uint xb0 = (b << 8u) + g_lo * 32u + sub * 4u;
            uint xb1 = (b << 8u) + g_hi * 32u + sub * 4u;
            for (uint k = 0u; k < 4u; k++) {
                uint bv = uint(qptr[k]);
                acc += (ds0 * float(bv & 15u) - ms0) * x[xb0 + k];
                acc += (ds1 * float(bv >> 4u) - ms1) * x[xb1 + k];
            }
        }
        return simd_sum(acc);
    }
    if (kind == 13u && (width & 255u) == 0u) {
        uint chunk = lane >> 3u, sub = lane & 7u;
        uint g_lo = chunk * 2u, g_hi = g_lo + 1u;
        uint num_blocks = width >> 8u;
        ulong row_blk = row_elem >> 8ul;
        for (uint b = 0u; b < num_blocks; b++) {
            device const uchar *p = w + (row_blk + b) * 176ul;
            float d = half_val(p), dm = half_val(p + 2u);
            device const uchar *s = p + 4u;
            uint sc0 = (g_lo < 4u) ? (uint(s[g_lo]) & 63u) : ((uint(s[g_lo + 4u]) & 15u) | ((uint(s[g_lo - 4u]) >> 6u) << 4u));
            uint mn0 = (g_lo < 4u) ? (uint(s[g_lo + 4u]) & 63u) : ((uint(s[g_lo + 4u]) >> 4u) | ((uint(s[g_lo]) >> 6u) << 4u));
            uint sc1 = (g_hi < 4u) ? (uint(s[g_hi]) & 63u) : ((uint(s[g_hi + 4u]) & 15u) | ((uint(s[g_hi - 4u]) >> 6u) << 4u));
            uint mn1 = (g_hi < 4u) ? (uint(s[g_hi + 4u]) & 63u) : ((uint(s[g_hi + 4u]) >> 4u) | ((uint(s[g_hi]) >> 6u) << 4u));
            float ds0 = d * float(sc0), ms0 = dm * float(mn0);
            float ds1 = d * float(sc1), ms1 = dm * float(mn1);
            device const uchar *qptr = p + 48u + chunk * 32u + sub * 4u;
            device const uchar *qhptr = p + 16u + sub * 4u;
            uint xb0 = (b << 8u) + g_lo * 32u + sub * 4u;
            uint xb1 = (b << 8u) + g_hi * 32u + sub * 4u;
            for (uint k = 0u; k < 4u; k++) {
                uint bv = uint(qptr[k]), hb = uint(qhptr[k]);
                uint q0 = (bv & 15u) | (((hb >> g_lo) & 1u) << 4u);
                uint q1 = (bv >> 4u) | (((hb >> g_hi) & 1u) << 4u);
                acc += (ds0 * float(q0) - ms0) * x[xb0 + k];
                acc += (ds1 * float(q1) - ms1) * x[xb1 + k];
            }
        }
        return simd_sum(acc);
    }
    if (kind == 14u && (width & 255u) == 0u) {
        uint h = lane >> 4u, col = (lane & 15u) * 2u;
        uint num_blocks = width >> 8u;
        ulong row_blk = row_elem >> 8ul;
        for (uint b = 0u; b < num_blocks; b++) {
            device const uchar *p = w + (row_blk + b) * 210ul;
            float d = half_val(p + 208u);
            uint lo0 = read_u16(p + h * 64u + col);
            uint lo1 = read_u16(p + h * 64u + 32u + col);
            uint hi = read_u16(p + 128u + h * 32u + col);
            uint xb = (b << 8u) + h * 128u + col;
            for (uint quarter = 0u; quarter < 4u; quarter++) {
                uint low = (quarter & 1u) ? lo1 : lo0;
                uint grp = h * 8u + quarter * 2u + (col >> 4u);
                float ds = d * float(reinterpret_cast<device const char *>(p)[192u + grp]);
                uint lsh = (quarter >> 1u) * 4u, hsh = quarter * 2u;
                int q0 = int(((low >> lsh) & 15u) | (((hi >> hsh) & 3u) << 4u)) - 32;
                int q1 = int(((low >> (8u + lsh)) & 15u) | (((hi >> (8u + hsh)) & 3u) << 4u)) - 32;
                acc += ds * float(q0) * x[xb + quarter * 32u];
                acc += ds * float(q1) * x[xb + quarter * 32u + 1u];
            }
        }
        return simd_sum(acc);
    }
    if (kind == 8u && (width & 31u) == 0u) {
        uint num_blocks = width >> 5u;
        ulong row_blk = row_elem >> 5ul;
        uint sub = lane >> 3u, elem = (lane & 7u) * 4u;
        for (uint b = sub; b < num_blocks; b += 4u) {
            device const uchar *p = w + (row_blk + b) * 34ul;
            float d = half_val(p);
            device const char *qs = reinterpret_cast<device const char *>(p + 2u + elem);
            uint xb = (b << 5u) + elem;
            acc += d * (float(qs[0]) * x[xb] + float(qs[1]) * x[xb + 1u]
                      + float(qs[2]) * x[xb + 2u] + float(qs[3]) * x[xb + 3u]);
        }
        return simd_sum(acc);
    }
    for (uint i = lane; i < width; i += 32u) {
        acc += decode_weight(w, kind, row_elem + i) * x[i];
    }
    return simd_sum(acc);
}

kernel void embedding(
    device const uchar *buf_w [[buffer(0)]],
    device float *buf_c [[buffer(3)]],
    constant Params &pc [[buffer(6)]],
    uint gid [[thread_position_in_grid]])
{
    uint width = pc.p[1];
    if (gid < width) {
        buf_c[gid] = decode_weight(buf_w, pc.p[2], ulong(pc.p[0]) * width + gid);
    }
}

kernel void embedding_batch(
    device const uchar *buf_w [[buffer(0)]],
    device float *buf_c [[buffer(3)]],
    device const int *buf_i [[buffer(5)]],
    constant Params &pc [[buffer(6)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint width = pc.p[0];
    if (gid.x < width) {
        uint tok = uint(buf_i[gid.y]);
        buf_c[ulong(gid.y) * width + gid.x] = decode_weight(buf_w, pc.p[1], ulong(tok) * width + gid.x);
    }
}

kernel void rmsnorm(
    device const float *buf_w [[buffer(0)]],
    device const float *buf_a [[buffer(1)]],
    device float *buf_c [[buffer(3)]],
    constant Params &pc [[buffer(6)]],
    uint tid [[thread_position_in_threadgroup]])
{
    threadgroup float sh[8];
    uint width = pc.p[0];
    float sum = 0.0f;
    for (uint i = tid; i < width; i += 256u) {
        float v = buf_a[i];
        sum += v * v;
    }
    float total = tg_sum(tid, sum, sh);
    float inv = rsqrt(total / float(width) + pc.f[0]);
    for (uint i = tid; i < width; i += 256u) {
        buf_c[i] = buf_a[i] * inv * buf_w[i];
    }
}

kernel void rmsnorm_batch(
    device const float *buf_w [[buffer(0)]],
    device const float *buf_a [[buffer(1)]],
    device float *buf_c [[buffer(3)]],
    constant Params &pc [[buffer(6)]],
    uint wg [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]])
{
    threadgroup float sh[8];
    uint width = pc.p[0];
    ulong base = ulong(wg) * width;
    float sum = 0.0f;
    for (uint i = tid; i < width; i += 256u) {
        float v = buf_a[base + i];
        sum += v * v;
    }
    float total = tg_sum(tid, sum, sh);
    float inv = rsqrt(total / float(width) + pc.f[0]);
    for (uint i = tid; i < width; i += 256u) {
        buf_c[base + i] = buf_a[base + i] * inv * buf_w[i];
    }
}

kernel void matvec(
    device const uchar *buf_w [[buffer(0)]],
    device const float *buf_a [[buffer(1)]],
    device float *buf_c [[buffer(3)]],
    device const int *buf_i [[buffer(5)]],
    constant Params &pc [[buffer(6)]],
    uint2 wg [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]])
{
    uint warp = tid >> 5u, lane = tid & 31u;
    uint row = wg.x * 8u + warp;
    uint selected = wg.y;
    uint kind = pc.p[0], width = pc.p[1], rows = pc.p[2];
    if (row >= rows) return;
    uint expert = pc.p[3] ? uint(buf_i[selected]) : 0u;
    ulong row_elem = (ulong(expert) * rows + row) * width;
    device const float *x = buf_a + (pc.p[4] ? ulong(selected) * width : 0ul);
    float total = packed_row_dot(buf_w, x, kind, row_elem, width, lane);
    if (lane == 0u) {
        buf_c[ulong(selected) * rows + row] = total;
    }
}

kernel void matmul_batch(
    device const uchar *buf_w [[buffer(0)]],
    device const float *buf_a [[buffer(1)]],
    device float *buf_c [[buffer(3)]],
    constant Params &pc [[buffer(6)]],
    uint2 wg [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]])
{
    uint warp = tid >> 5u, lane = tid & 31u;
    uint row = wg.x * 8u + warp;
    uint t0 = wg.y * 4u;
    uint kind = pc.p[0], width = pc.p[1], rows = pc.p[2], tokens = pc.p[3];
    if (row >= rows) return;
    uint n = min(4u, tokens - t0);
    ulong row_elem = ulong(row) * width;
    device const float *x0 = buf_a + ulong(t0) * width;
    device const float *x1 = (n > 1u) ? (buf_a + ulong(t0 + 1u) * width) : x0;
    device const float *x2 = (n > 2u) ? (buf_a + ulong(t0 + 2u) * width) : x0;
    device const float *x3 = (n > 3u) ? (buf_a + ulong(t0 + 3u) * width) : x0;
    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
    if (kind == 12u && (width & 255u) == 0u) {
        uint chunk = lane >> 3u, sub = lane & 7u;
        uint g_lo = chunk * 2u, g_hi = g_lo + 1u;
        uint num_blocks = width >> 8u;
        ulong row_blk = row_elem >> 8ul;
        for (uint b = 0u; b < num_blocks; b++) {
            device const uchar *p = buf_w + (row_blk + b) * 144ul;
            float d = half_val(p), dm = half_val(p + 2u);
            device const uchar *s = p + 4u;
            uint sc0 = (g_lo < 4u) ? (uint(s[g_lo]) & 63u) : ((uint(s[g_lo + 4u]) & 15u) | ((uint(s[g_lo - 4u]) >> 6u) << 4u));
            uint mn0 = (g_lo < 4u) ? (uint(s[g_lo + 4u]) & 63u) : ((uint(s[g_lo + 4u]) >> 4u) | ((uint(s[g_lo]) >> 6u) << 4u));
            uint sc1 = (g_hi < 4u) ? (uint(s[g_hi]) & 63u) : ((uint(s[g_hi + 4u]) & 15u) | ((uint(s[g_hi - 4u]) >> 6u) << 4u));
            uint mn1 = (g_hi < 4u) ? (uint(s[g_hi + 4u]) & 63u) : ((uint(s[g_hi + 4u]) >> 4u) | ((uint(s[g_hi]) >> 6u) << 4u));
            float ds0 = d * float(sc0), ms0 = dm * float(mn0);
            float ds1 = d * float(sc1), ms1 = dm * float(mn1);
            device const uchar *qptr = p + 16u + chunk * 32u + sub * 4u;
            uint off0 = (b << 8u) + g_lo * 32u + sub * 4u;
            uint off1 = (b << 8u) + g_hi * 32u + sub * 4u;
            for (uint k = 0u; k < 4u; k++) {
                uint bv = uint(qptr[k]);
                float w0 = ds0 * float(bv & 15u) - ms0;
                float w1 = ds1 * float(bv >> 4u) - ms1;
                a0 += w0 * x0[off0 + k] + w1 * x0[off1 + k];
                a1 += w0 * x1[off0 + k] + w1 * x1[off1 + k];
                a2 += w0 * x2[off0 + k] + w1 * x2[off1 + k];
                a3 += w0 * x3[off0 + k] + w1 * x3[off1 + k];
            }
        }
    } else {
        for (uint i = lane; i < width; i += 32u) {
            float wv = decode_weight(buf_w, kind, row_elem + i);
            a0 += wv * x0[i];
            a1 += wv * x1[i];
            a2 += wv * x2[i];
            a3 += wv * x3[i];
        }
    }
    float s0 = simd_sum(a0), s1 = simd_sum(a1), s2 = simd_sum(a2), s3 = simd_sum(a3);
    if (lane == 0u) {
        buf_c[ulong(t0) * rows + row] = s0;
        if (n > 1u) buf_c[ulong(t0 + 1u) * rows + row] = s1;
        if (n > 2u) buf_c[ulong(t0 + 2u) * rows + row] = s2;
        if (n > 3u) buf_c[ulong(t0 + 3u) * rows + row] = s3;
    }
}

kernel void add_row_bias(
    device const float *buf_w [[buffer(0)]],
    device float *buf_c [[buffer(3)]],
    constant Params &pc [[buffer(6)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < pc.p[0]) buf_c[gid] += buf_w[gid];
}

kernel void add_row_bias_batch(
    device const float *buf_w [[buffer(0)]],
    device float *buf_c [[buffer(3)]],
    constant Params &pc [[buffer(6)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint w = pc.p[0];
    if (gid.x < w) buf_c[ulong(gid.y) * w + gid.x] += buf_w[gid.x];
}

kernel void head_norm(
    device const float *buf_w [[buffer(0)]],
    device float *buf_c [[buffer(3)]],
    constant Params &pc [[buffer(6)]],
    uint wg [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]])
{
    threadgroup float sh[8];
    uint dim = pc.p[0];
    ulong base = ulong(wg) * dim;
    float sum = 0.0f;
    for (uint i = tid; i < dim; i += 256u) {
        float v = buf_c[base + i];
        sum += v * v;
    }
    float total = tg_sum(tid, sum, sh);
    float inv = rsqrt(total / float(dim) + pc.f[0]);
    for (uint i = tid; i < dim; i += 256u) {
        buf_c[base + i] = buf_c[base + i] * inv * buf_w[i];
    }
}

kernel void head_norm_batch(
    device const float *buf_w [[buffer(0)]],
    device float *buf_c [[buffer(3)]],
    constant Params &pc [[buffer(6)]],
    uint2 wg [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]])
{
    threadgroup float sh[8];
    uint dim = pc.p[0];
    ulong base = ulong(wg.y) * pc.p[1] + ulong(wg.x) * dim;
    float sum = 0.0f;
    for (uint i = tid; i < dim; i += 256u) {
        float v = buf_c[base + i];
        sum += v * v;
    }
    float total = tg_sum(tid, sum, sh);
    float inv = rsqrt(total / float(dim) + pc.f[0]);
    for (uint i = tid; i < dim; i += 256u) {
        buf_c[base + i] = buf_c[base + i] * inv * buf_w[i];
    }
}

kernel void rotary(
    device const float *buf_b [[buffer(2)]],
    device float *buf_c [[buffer(3)]],
    constant Params &pc [[buffer(6)]],
    uint gid [[thread_position_in_grid]])
{
    uint size = pc.p[0], dim = pc.p[1], rotated = pc.p[4];
    uint half_d = rotated >> 1u, num_heads = size / dim;
    if (gid >= num_heads * half_d) return;
    uint pair = gid % half_d, head = gid / half_d;
    uint a = head * dim + (pc.p[3] ? pair : 2u * pair);
    uint b = pc.p[3] ? (a + half_d) : (a + 1u);
    float angle = float(pc.p[2]) * pow(pc.f[0], -float(2u * pair) / float(rotated));
    if (pc.p[5]) angle /= buf_b[pair];
    float s = sin(angle) * pc.f[1], c = cos(angle) * pc.f[1];
    float u = buf_c[a], v = buf_c[b];
    buf_c[a] = u * c - v * s;
    buf_c[b] = u * s + v * c;
}

kernel void rotary_batch(
    device const float *buf_b [[buffer(2)]],
    device float *buf_c [[buffer(3)]],
    constant Params &pc [[buffer(6)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint size = pc.p[0], dim = pc.p[1], rotated = pc.p[4];
    uint half_d = rotated >> 1u, num_heads = size / dim;
    if (gid.x >= num_heads * half_d) return;
    uint pair = gid.x % half_d, head = gid.x / half_d;
    ulong base = ulong(gid.y) * size + head * dim;
    ulong a = base + (pc.p[3] ? pair : 2u * pair);
    ulong b = pc.p[3] ? (a + half_d) : (a + 1u);
    float angle = float(pc.p[2] + gid.y) * pow(pc.f[0], -float(2u * pair) / float(rotated));
    if (pc.p[5]) angle /= buf_b[pair];
    float s = sin(angle) * pc.f[1], c = cos(angle) * pc.f[1];
    float u = buf_c[a], v = buf_c[b];
    buf_c[a] = u * c - v * s;
    buf_c[b] = u * s + v * c;
}

kernel void cache_kv(
    device const float *buf_a [[buffer(1)]],
    device const float *buf_b [[buffer(2)]],
    device float *buf_c [[buffer(3)]],
    device float *buf_d [[buffer(4)]],
    constant Params &pc [[buffer(6)]],
    uint gid [[thread_position_in_grid]])
{
    uint kw = pc.p[0];
    if (gid < kw) {
        ulong off = ulong(pc.p[1]) * kw + gid;
        buf_c[off] = buf_a[gid];
        buf_d[off] = buf_b[gid];
    }
}

kernel void cache_kv_batch(
    device const float *buf_a [[buffer(1)]],
    device const float *buf_b [[buffer(2)]],
    device float *buf_c [[buffer(3)]],
    device float *buf_d [[buffer(4)]],
    constant Params &pc [[buffer(6)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint kw = pc.p[0];
    if (gid.x < kw) {
        ulong src = ulong(gid.y) * kw + gid.x;
        ulong dst = ulong(pc.p[1] + gid.y) * kw + gid.x;
        buf_c[dst] = buf_a[src];
        buf_d[dst] = buf_b[src];
    }
}

kernel void attention(
    device const float *values [[buffer(0)]],
    device const float *q [[buffer(1)]],
    device const float *keys [[buffer(2)]],
    device float *attended [[buffer(3)]],
    constant Params &pc [[buffer(6)]],
    uint wg [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]])
{
    threadgroup float sh[8];
    uint dim = pc.p[0], heads = pc.p[1], kv_heads = pc.p[2], count = pc.p[3], window = pc.p[4];
    uint kv_head = wg / (heads / kv_heads);
    float q_val = (tid < dim) ? (q[ulong(wg) * dim + tid] * pc.f[0]) : 0.0f;
    uint start = (window > 0u && count > window) ? (count - window) : 0u;
    float max_score = -3.402823466e+38f, sum_exp = 0.0f, acc = 0.0f;
    for (uint t = start; t < count; t++) {
        ulong kv_base = (ulong(t) * kv_heads + kv_head) * dim;
        float k_val = (tid < dim) ? keys[kv_base + tid] : 0.0f;
        float score = tg_sum(tid, q_val * k_val, sh);
        if (pc.f[1] > 0.0f) score = pc.f[1] * tanh(score / pc.f[1]);
        float new_max = max(max_score, score);
        float rescale = exp(max_score - new_max);
        float p = exp(score - new_max);
        sum_exp = sum_exp * rescale + p;
        float v_val = (tid < dim) ? values[kv_base + tid] : 0.0f;
        acc = acc * rescale + p * v_val;
        max_score = new_max;
    }
    if (tid < dim) attended[ulong(wg) * dim + tid] = acc / sum_exp;
}

kernel void attention_batch(
    device const float *values [[buffer(0)]],
    device const float *q [[buffer(1)]],
    device const float *keys [[buffer(2)]],
    device float *attended [[buffer(3)]],
    constant Params &pc [[buffer(6)]],
    uint2 wg [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]])
{
    threadgroup float sh[8];
    uint head = wg.x, tok = wg.y;
    uint dim = pc.p[0], heads = pc.p[1], kv_heads = pc.p[2];
    uint count = pc.p[3] + tok, window = pc.p[4];
    uint kv_head = head / (heads / kv_heads);
    ulong q_off = (ulong(tok) * heads + head) * dim;
    float q_val = (tid < dim) ? (q[q_off + tid] * pc.f[0]) : 0.0f;
    uint start = (window > 0u && count > window) ? (count - window) : 0u;
    float max_score = -3.402823466e+38f, sum_exp = 0.0f, acc = 0.0f;
    for (uint t = start; t < count; t++) {
        ulong kv_base = (ulong(t) * kv_heads + kv_head) * dim;
        float k_val = (tid < dim) ? keys[kv_base + tid] : 0.0f;
        float score = tg_sum(tid, q_val * k_val, sh);
        if (pc.f[1] > 0.0f) score = pc.f[1] * tanh(score / pc.f[1]);
        float new_max = max(max_score, score);
        float rescale = exp(max_score - new_max);
        float p = exp(score - new_max);
        sum_exp = sum_exp * rescale + p;
        float v_val = (tid < dim) ? values[kv_base + tid] : 0.0f;
        acc = acc * rescale + p * v_val;
        max_score = new_max;
    }
    if (tid < dim) attended[q_off + tid] = acc / sum_exp;
}

kernel void swiglu(
    device const float *buf_a [[buffer(1)]],
    device float *buf_c [[buffer(3)]],
    constant Params &pc [[buffer(6)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < pc.p[0]) {
        float g = buf_c[gid];
        float act = pc.p[1]
            ? (0.5f * g * (1.0f + tanh(0.7978846f * g * (1.0f + 0.044715f * g * g))))
            : (g / (1.0f + exp(-g)));
        buf_c[gid] = act * buf_a[gid];
    }
}

kernel void residual(
    device const float *buf_a [[buffer(1)]],
    device float *buf_c [[buffer(3)]],
    constant Params &pc [[buffer(6)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < pc.p[0]) buf_c[gid] += buf_a[gid];
}

kernel void scale_vec(
    device float *buf_c [[buffer(3)]],
    constant Params &pc [[buffer(6)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < pc.p[0]) buf_c[gid] *= pc.f[0];
}

kernel void softcap(
    device float *buf_c [[buffer(3)]],
    constant Params &pc [[buffer(6)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < pc.p[0]) buf_c[gid] = pc.f[0] * tanh(buf_c[gid] / pc.f[0]);
}

kernel void copy_last_token(
    device const float *buf_a [[buffer(1)]],
    device float *buf_c [[buffer(3)]],
    constant Params &pc [[buffer(6)]],
    uint gid [[thread_position_in_grid]])
{
    uint w = pc.p[0];
    if (gid < w) buf_c[gid] = buf_a[ulong(pc.p[1]) * w + gid];
}

kernel void router(
    device const float *buf_a [[buffer(1)]],
    device float *buf_c [[buffer(3)]],
    device int *buf_i [[buffer(5)]],
    constant Params &pc [[buffer(6)]],
    uint tid [[thread_position_in_threadgroup]])
{
    if (tid != 0u) return;
    uint experts = pc.p[0], used = pc.p[1];
    bool full_softmax = pc.p[2] != 0u;
    float max_val = -3.402823466e+38f;
    for (uint e = 0u; e < experts; e++) max_val = max(max_val, buf_a[e]);
    float full_denom = 0.0f;
    if (full_softmax) {
        for (uint e = 0u; e < experts; e++) full_denom += exp(buf_a[e] - max_val);
    }
    for (uint s = 0u; s < used; s++) {
        float best_v = -3.402823466e+38f;
        uint best_i = 0u;
        for (uint e = 0u; e < experts; e++) {
            bool taken = false;
            for (uint p = 0u; p < s; p++) if (uint(buf_i[p]) == e) taken = true;
            if (!taken && buf_a[e] > best_v) { best_v = buf_a[e]; best_i = e; }
        }
        buf_i[s] = int(best_i);
        buf_c[s] = best_v;
    }
    if (full_softmax) {
        for (uint s = 0u; s < used; s++) buf_c[s] = exp(buf_c[s] - max_val) / full_denom;
    } else {
        float top_max = buf_c[0];
        for (uint s = 1u; s < used; s++) top_max = max(top_max, buf_c[s]);
        float denom = 0.0f;
        for (uint s = 0u; s < used; s++) { buf_c[s] = exp(buf_c[s] - top_max); denom += buf_c[s]; }
        for (uint s = 0u; s < used; s++) buf_c[s] /= denom;
    }
}

kernel void combine_experts(
    device const float *buf_a [[buffer(1)]],
    device const float *buf_b [[buffer(2)]],
    device float *buf_c [[buffer(3)]],
    constant Params &pc [[buffer(6)]],
    uint gid [[thread_position_in_grid]])
{
    uint w = pc.p[0];
    if (gid < w) {
        float sum = 0.0f;
        for (uint s = 0u; s < pc.p[1]; s++) sum += buf_b[s] * buf_a[ulong(s) * w + gid];
        buf_c[gid] += sum;
    }
}
