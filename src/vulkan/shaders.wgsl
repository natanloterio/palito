// Palito Vulkan compute shaders (WGSL -> SPIR-V via naga).
// Supports F32 (0), F16 (1), Q8_0 (8), Q4_K (12), Q5_K (13), Q6_K (14).

struct Params {
    p0: u32,
    p1: u32,
    p2: u32,
    p3: u32,
    p4: u32,
    p5: u32,
    p6: u32,
    p7: u32,
    f0: f32,
    f1: f32,
    f2: f32,
    f3: f32,
}

var<push_constant> pc: Params;

@group(0) @binding(0) var<storage, read> buf_w: array<u32>;
@group(0) @binding(1) var<storage, read> buf_a: array<f32>;
@group(0) @binding(2) var<storage, read> buf_b: array<f32>;
@group(0) @binding(3) var<storage, read_write> buf_c: array<f32>;
@group(0) @binding(4) var<storage, read_write> buf_d: array<f32>;
@group(0) @binding(5) var<storage, read_write> buf_i: array<i32>;

var<workgroup> shared_f32: array<f32, 256>;
var<workgroup> shared_i32: array<i32, 256>;

fn read_u8(byte_off: u32) -> u32 {
    let word = buf_w[byte_off >> 2u];
    return (word >> ((byte_off & 3u) * 8u)) & 0xffu;
}

fn read_i8(byte_off: u32) -> i32 {
    let v = read_u8(byte_off);
    return select(i32(v), i32(v) - 256, v >= 128u);
}

fn read_u16(byte_off: u32) -> u32 {
    let lo = read_u8(byte_off);
    let hi = read_u8(byte_off + 1u);
    return lo | (hi << 8u);
}

fn half_to_f32(bits: u32) -> f32 {
    return unpack2x16float(bits & 0xffffu).x;
}

fn decode_weight(kind: u32, elem: u32) -> f32 {
    if (kind == 0u) {
        return bitcast<f32>(buf_w[elem]);
    }
    if (kind == 1u) {
        return half_to_f32(read_u16(elem * 2u));
    }
    if (kind == 6u) {
        // Q5_0: 22 bytes per 32 elements (2-byte f16 scale + 4-byte qh + 16-byte qs).
        let blk = elem >> 5u;
        let i = elem & 31u;
        let base = blk * 22u;
        let high = read_u8(base + 2u) | (read_u8(base + 3u) << 8u)
            | (read_u8(base + 4u) << 16u) | (read_u8(base + 5u) << 24u);
        let low = (read_u8(base + 6u + (i & 15u)) >> (4u * (i >> 4u))) & 15u;
        let q = i32(low | (((high >> i) & 1u) << 4u)) - 16;
        return half_to_f32(read_u16(base)) * f32(q);
    }
    if (kind == 8u) {
        // Q8_0: 34 bytes per 32 elements (2-byte f16 scale + 32 i8 values).
        let blk = elem >> 5u;
        let i = elem & 31u;
        let base = blk * 34u;
        let d = half_to_f32(read_u16(base));
        return d * f32(read_i8(base + 2u + i));
    }
    if (kind == 12u || kind == 13u) {
        // Q4_K (144 B) or Q5_K (176 B) per 256 elements.
        let blk = elem >> 8u;
        let i = elem & 255u;
        let stride = select(176u, 144u, kind == 12u);
        let base = blk * stride;
        let d = half_to_f32(read_u16(base));
        let dm = half_to_f32(read_u16(base + 2u));
        let s_base = base + 4u;
        let g = i >> 5u;
        var scale: u32;
        var minimum: u32;
        if (g < 4u) {
            scale = read_u8(s_base + g) & 63u;
            minimum = read_u8(s_base + g + 4u) & 63u;
        } else {
            let s_hi = read_u8(s_base + g + 4u);
            let s_lo = read_u8(s_base + g - 4u);
            let m_lo = read_u8(s_base + g);
            scale = (s_hi & 15u) | ((s_lo >> 6u) << 4u);
            minimum = (s_hi >> 4u) | ((m_lo >> 6u) << 4u);
        }
        let low_off = select(48u, 16u, kind == 12u);
        let q_byte = read_u8(base + low_off + (i >> 6u) * 32u + (i & 31u));
        var q = (q_byte >> (4u * (g & 1u))) & 15u;
        if (kind == 13u) {
            let qh = (read_u8(base + 16u + (i & 31u)) >> g) & 1u;
            q = q | (qh << 4u);
        }
        return d * f32(scale) * f32(q) - dm * f32(minimum);
    }
    // Q6_K (kind == 14): 210 bytes per 256 elements.
    let blk = elem >> 8u;
    let i = elem & 255u;
    let base = blk * 210u;
    let h = i >> 7u;
    let lane = i & 31u;
    let quarter = (i & 127u) >> 5u;
    let low_byte = read_u8(base + h * 64u + lane + (quarter & 1u) * 32u);
    let low = (low_byte >> (4u * (quarter >> 1u))) & 15u;
    let hi_byte = read_u8(base + 128u + h * 32u + lane);
    let high = (hi_byte >> (2u * quarter)) & 3u;
    let q = i32(low | (high << 4u)) - 32;
    let sc = read_i8(base + 192u + (i >> 4u));
    let d = half_to_f32(read_u16(base + 208u));
    return d * f32(sc) * f32(q);
}

fn workgroup_sum(tid: u32, val: f32) -> f32 {
    shared_f32[tid] = val;
    workgroupBarrier();
    var stride = 128u;
    loop {
        if (stride == 0u) { break; }
        if (tid < stride) {
            shared_f32[tid] = shared_f32[tid] + shared_f32[tid + stride];
        }
        workgroupBarrier();
        stride = stride >> 1u;
    }
    return shared_f32[0];
}

// 32-lane sub-group sum inside a 256-thread workgroup (8 independent rows).
fn lane32_sum(tid: u32, lane: u32, val: f32) -> f32 {
    shared_f32[tid] = val;
    workgroupBarrier();
    if (lane < 16u) { shared_f32[tid] = shared_f32[tid] + shared_f32[tid + 16u]; }
    workgroupBarrier();
    if (lane < 8u) { shared_f32[tid] = shared_f32[tid] + shared_f32[tid + 8u]; }
    workgroupBarrier();
    if (lane < 4u) { shared_f32[tid] = shared_f32[tid] + shared_f32[tid + 4u]; }
    workgroupBarrier();
    if (lane < 2u) { shared_f32[tid] = shared_f32[tid] + shared_f32[tid + 2u]; }
    workgroupBarrier();
    if (lane == 0u) { shared_f32[tid] = shared_f32[tid] + shared_f32[tid + 1u]; }
    workgroupBarrier();
    return shared_f32[tid & ~31u];
}

fn read_u32_unaligned(byte_off: u32) -> u32 {
    let shift = (byte_off & 3u) * 8u;
    let idx = byte_off >> 2u;
    if (shift == 0u) {
        return buf_w[idx];
    }
    return (buf_w[idx] >> shift) | (buf_w[idx + 1u] << (32u - shift));
}

// Fast packed row dot product for 32 lanes (1 warp) over a single row.
fn packed_row_dot(kind: u32, row_elem: u32, width: u32, x_base: u32, lane: u32) -> f32 {
    var acc = 0.0;
    if (kind == 12u && (width & 255u) == 0u) {
        // Q4_K: 32 lanes process one 256-element block per step (8 elements/lane).
        let chunk = lane >> 3u;
        let sub = lane & 7u;
        let g_lo = chunk * 2u;
        let g_hi = g_lo + 1u;
        let num_blocks = width >> 8u;
        let row_blk = row_elem >> 8u;
        for (var b = 0u; b < num_blocks; b = b + 1u) {
            let base = (row_blk + b) * 144u;
            let dd = buf_w[base >> 2u];
            let d = half_to_f32(dd & 0xffffu);
            let dm = half_to_f32(dd >> 16u);
            let s0 = buf_w[(base >> 2u) + 1u];
            let s1 = buf_w[(base >> 2u) + 2u];
            let s2 = buf_w[(base >> 2u) + 3u];
            let sh_lo = (g_lo & 3u) * 8u;
            let lo0 = (s0 >> sh_lo) & 255u;
            let mid0 = (s1 >> sh_lo) & 255u;
            let hi0 = (s2 >> sh_lo) & 255u;
            let sc0 = select((hi0 & 15u) | ((lo0 >> 6u) << 4u), lo0 & 63u, g_lo < 4u);
            let mn0 = select((hi0 >> 4u) | ((mid0 >> 6u) << 4u), mid0 & 63u, g_lo < 4u);
            let sh_hi = (g_hi & 3u) * 8u;
            let lo1 = (s0 >> sh_hi) & 255u;
            let mid1 = (s1 >> sh_hi) & 255u;
            let hi1 = (s2 >> sh_hi) & 255u;
            let sc1 = select((hi1 & 15u) | ((lo1 >> 6u) << 4u), lo1 & 63u, g_hi < 4u);
            let mn1 = select((hi1 >> 4u) | ((mid1 >> 6u) << 4u), mid1 & 63u, g_hi < 4u);
            let ds0 = d * f32(sc0);
            let ms0 = dm * f32(mn0);
            let ds1 = d * f32(sc1);
            let ms1 = dm * f32(mn1);
            let qw = buf_w[(base + 16u + chunk * 32u + sub * 4u) >> 2u];
            let xb0 = x_base + (b << 8u) + g_lo * 32u + sub * 4u;
            let xb1 = x_base + (b << 8u) + g_hi * 32u + sub * 4u;
            for (var k = 0u; k < 4u; k = k + 1u) {
                let byte_v = (qw >> (k * 8u)) & 255u;
                acc = acc + (ds0 * f32(byte_v & 15u) - ms0) * buf_a[xb0 + k];
                acc = acc + (ds1 * f32(byte_v >> 4u) - ms1) * buf_a[xb1 + k];
            }
        }
        return acc;
    }
    if (kind == 13u && (width & 255u) == 0u) {
        // Q5_K: 32 lanes process one 256-element block per step (8 elements/lane).
        let chunk = lane >> 3u;
        let sub = lane & 7u;
        let g_lo = chunk * 2u;
        let g_hi = g_lo + 1u;
        let num_blocks = width >> 8u;
        let row_blk = row_elem >> 8u;
        for (var b = 0u; b < num_blocks; b = b + 1u) {
            let base = (row_blk + b) * 176u;
            let dd = buf_w[base >> 2u];
            let d = half_to_f32(dd & 0xffffu);
            let dm = half_to_f32(dd >> 16u);
            let s0 = buf_w[(base >> 2u) + 1u];
            let s1 = buf_w[(base >> 2u) + 2u];
            let s2 = buf_w[(base >> 2u) + 3u];
            let sh_lo = (g_lo & 3u) * 8u;
            let lo0 = (s0 >> sh_lo) & 255u;
            let mid0 = (s1 >> sh_lo) & 255u;
            let hi0 = (s2 >> sh_lo) & 255u;
            let sc0 = select((hi0 & 15u) | ((lo0 >> 6u) << 4u), lo0 & 63u, g_lo < 4u);
            let mn0 = select((hi0 >> 4u) | ((mid0 >> 6u) << 4u), mid0 & 63u, g_lo < 4u);
            let sh_hi = (g_hi & 3u) * 8u;
            let lo1 = (s0 >> sh_hi) & 255u;
            let mid1 = (s1 >> sh_hi) & 255u;
            let hi1 = (s2 >> sh_hi) & 255u;
            let sc1 = select((hi1 & 15u) | ((lo1 >> 6u) << 4u), lo1 & 63u, g_hi < 4u);
            let mn1 = select((hi1 >> 4u) | ((mid1 >> 6u) << 4u), mid1 & 63u, g_hi < 4u);
            let ds0 = d * f32(sc0);
            let ms0 = dm * f32(mn0);
            let ds1 = d * f32(sc1);
            let ms1 = dm * f32(mn1);
            let qw = buf_w[(base + 48u + chunk * 32u + sub * 4u) >> 2u];
            let qhw = buf_w[(base + 16u + sub * 4u) >> 2u];
            let xb0 = x_base + (b << 8u) + g_lo * 32u + sub * 4u;
            let xb1 = x_base + (b << 8u) + g_hi * 32u + sub * 4u;
            for (var k = 0u; k < 4u; k = k + 1u) {
                let byte_v = (qw >> (k * 8u)) & 255u;
                let h_byte = (qhw >> (k * 8u)) & 255u;
                let q0 = (byte_v & 15u) | (((h_byte >> g_lo) & 1u) << 4u);
                let q1 = (byte_v >> 4u) | (((h_byte >> g_hi) & 1u) << 4u);
                acc = acc + (ds0 * f32(q0) - ms0) * buf_a[xb0 + k];
                acc = acc + (ds1 * f32(q1) - ms1) * buf_a[xb1 + k];
            }
        }
        return acc;
    }
    if (kind == 14u && (width & 255u) == 0u) {
        // Q6_K: 32 lanes process one 256-element block per step (8 elements/lane).
        let h = lane >> 4u;
        let col = (lane & 15u) * 2u;
        let num_blocks = width >> 8u;
        let row_blk = row_elem >> 8u;
        for (var b = 0u; b < num_blocks; b = b + 1u) {
            let base = (row_blk + b) * 210u;
            let d = half_to_f32(read_u16(base + 208u));
            let lo0 = read_u16(base + h * 64u + col);
            let lo1 = read_u16(base + h * 64u + 32u + col);
            let hi = read_u16(base + 128u + h * 32u + col);
            let xb = x_base + (b << 8u) + h * 128u + col;
            for (var quarter = 0u; quarter < 4u; quarter = quarter + 1u) {
                let low = select(lo0, lo1, (quarter & 1u) != 0u);
                let grp = h * 8u + quarter * 2u + (col >> 4u);
                let ds = d * f32(read_i8(base + 192u + grp));
                let lsh = (quarter >> 1u) * 4u;
                let hsh = quarter * 2u;
                let q0 = i32(((low >> lsh) & 15u) | (((hi >> hsh) & 3u) << 4u)) - 32;
                let q1 = i32(((low >> (8u + lsh)) & 15u) | (((hi >> (8u + hsh)) & 3u) << 4u)) - 32;
                acc = acc + ds * f32(q0) * buf_a[xb + quarter * 32u];
                acc = acc + ds * f32(q1) * buf_a[xb + quarter * 32u + 1u];
            }
        }
        return acc;
    }
    if (kind == 8u && (width & 31u) == 0u) {
        // Q8_0: 8 lanes per 32-element block (4 values/lane), so 32 lanes process
        // 4 blocks (128 elements) per step.
        let num_blocks = width >> 5u;
        let row_blk = row_elem >> 5u;
        let sub = lane >> 3u;
        let elem = (lane & 7u) * 4u;
        var b = sub;
        loop {
            if (b >= num_blocks) { break; }
            let base = (row_blk + b) * 34u;
            let d = half_to_f32(read_u16(base));
            let xb = x_base + (b << 5u) + elem;
            let q0 = read_i8(base + 2u + elem);
            let q1 = read_i8(base + 3u + elem);
            let q2 = read_i8(base + 4u + elem);
            let q3 = read_i8(base + 5u + elem);
            acc = acc + d * (
                f32(q0) * buf_a[xb]
                + f32(q1) * buf_a[xb + 1u]
                + f32(q2) * buf_a[xb + 2u]
                + f32(q3) * buf_a[xb + 3u]
            );
            b = b + 4u;
        }
        return acc;
    }
    var i = lane;
    loop {
        if (i >= width) { break; }
        acc = acc + decode_weight(kind, row_elem + i) * buf_a[x_base + i];
        i = i + 32u;
    }
    return acc;
}

// p0 = token_id, p1 = width, p2 = kind
@compute @workgroup_size(256)
fn embedding(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let width = pc.p1;
    if (i < width) {
        buf_c[i] = decode_weight(pc.p2, pc.p0 * width + i);
    }
}

// buf_a = x, buf_w = norm weight (f32), buf_c = y; p0 = width, f0 = epsilon
@compute @workgroup_size(256)
fn rmsnorm(@builtin(local_invocation_id) lid: vec3<u32>) {
    let tid = lid.x;
    let width = pc.p0;
    var sum = 0.0;
    var i = tid;
    loop {
        if (i >= width) { break; }
        let v = buf_a[i];
        sum = sum + v * v;
        i = i + 256u;
    }
    let total = workgroup_sum(tid, sum);
    let inv = inverseSqrt(total / f32(width) + pc.f0);
    i = tid;
    loop {
        if (i >= width) { break; }
        buf_c[i] = buf_a[i] * inv * bitcast<f32>(buf_w[i]);
        i = i + 256u;
    }
}

// buf_w = weight, buf_a = x, buf_c = y, buf_i = expert ids
// 8 rows per 256-thread workgroup (32 lanes per row).
// workgroup_id.x = row / 8, workgroup_id.y = selected
// p0 = kind, p1 = width, p2 = rows, p3 = moe, p4 = batched
@compute @workgroup_size(256)
fn matvec(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let tid = lid.x;
    let warp = tid >> 5u;
    let lane = tid & 31u;
    let row = wg.x * 8u + warp;
    let selected = wg.y;
    let kind = pc.p0;
    let width = pc.p1;
    let rows = pc.p2;
    let expert = select(0u, u32(buf_i[selected]), pc.p3 != 0u);
    let x_base = select(0u, selected * width, pc.p4 != 0u);
    var sum = 0.0;
    if (row < rows) {
        let row_elem = (expert * rows + row) * width;
        sum = packed_row_dot(kind, row_elem, width, x_base, lane);
    }
    let total = lane32_sum(tid, lane, sum);
    if (lane == 0u && row < rows) {
        buf_c[selected * rows + row] = total;
    }
}

// buf_c += bitcast<f32>(buf_w); p0 = width
@compute @workgroup_size(256)
fn add_row_bias(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i < pc.p0) {
        buf_c[i] = buf_c[i] + bitcast<f32>(buf_w[i]);
    }
}

// In-place per-head (or full-vector) RMSNorm on buf_c using buf_w (f32).
// workgroup_id.x = head; p0 = norm_width, f0 = epsilon
@compute @workgroup_size(256)
fn head_norm(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let tid = lid.x;
    let dim = pc.p0;
    let base = wg.x * dim;
    var sum = 0.0;
    var i = tid;
    loop {
        if (i >= dim) { break; }
        let v = buf_c[base + i];
        sum = sum + v * v;
        i = i + 256u;
    }
    let total = workgroup_sum(tid, sum);
    let inv = inverseSqrt(total / f32(dim) + pc.f0);
    i = tid;
    loop {
        if (i >= dim) { break; }
        buf_c[base + i] = buf_c[base + i] * inv * bitcast<f32>(buf_w[i]);
        i = i + 256u;
    }
}

// In-place RoPE on buf_c (optional factors in buf_b when p5 != 0).
// p0 = size, p1 = dim, p2 = position, p3 = neox, p4 = rotary_dim, p5 = has_factors
// f0 = rope_base, f1 = attn_factor
@compute @workgroup_size(256)
fn rotary(@builtin(global_invocation_id) gid: vec3<u32>) {
    let p = gid.x;
    let size = pc.p0;
    let dim = pc.p1;
    let rotated = pc.p4;
    let half = rotated >> 1u;
    let num_heads = size / dim;
    if (p >= num_heads * half) { return; }
    let pair = p % half;
    let head = p / half;
    let a = head * dim + select(2u * pair, pair, pc.p3 != 0u);
    let b = select(a + 1u, a + half, pc.p3 != 0u);
    var angle = f32(pc.p2) * pow(pc.f0, -f32(2u * pair) / f32(rotated));
    if (pc.p5 != 0u) {
        angle = angle / buf_b[pair];
    }
    let s = sin(angle) * pc.f1;
    let c = cos(angle) * pc.f1;
    let u = buf_c[a];
    let v = buf_c[b];
    buf_c[a] = u * c - v * s;
    buf_c[b] = u * s + v * c;
}

// buf_a = k, buf_b = v, buf_c = keys_cache, buf_d = values_cache
// p0 = kv_width, p1 = position
@compute @workgroup_size(256)
fn cache_kv(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let kw = pc.p0;
    if (i < kw) {
        let off = pc.p1 * kw + i;
        buf_c[off] = buf_a[i];
        buf_d[off] = buf_b[i];
    }
}

// Fused causal GQA/MHA attention with online softmax (arbitrary context, dim <= 256).
// buf_a = q, buf_b = keys_cache, buf_w = values_cache (bitcast f32), buf_c = attended
// workgroup_id.x = head; p0 = dim, p1 = heads, p2 = kv_heads, p3 = count, p4 = window
// f0 = scale (1/sqrt(dim) or gemma scale), f1 = softcap (0 if none)
@compute @workgroup_size(256)
fn attention(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let head = wg.x;
    let tid = lid.x;
    let dim = pc.p0;
    let heads = pc.p1;
    let kv_heads = pc.p2;
    let count = pc.p3;
    let window = pc.p4;
    let kv_head = head / (heads / kv_heads);
    let q_val = select(0.0, buf_a[head * dim + tid] * pc.f0, tid < dim);
    let start = select(0u, count - window, window > 0u && count > window);

    var max_score = -3.402823466e+38;
    var sum_exp = 0.0;
    var acc = 0.0;

    var t = start;
    loop {
        if (t >= count) { break; }
        let kv_base = (t * kv_heads + kv_head) * dim;
        let k_val = select(0.0, buf_b[kv_base + tid], tid < dim);
        var score = workgroup_sum(tid, q_val * k_val);
        if (pc.f1 > 0.0) {
            score = pc.f1 * tanh(score / pc.f1);
        }
        let new_max = max(max_score, score);
        let rescale = exp(max_score - new_max);
        let p = exp(score - new_max);
        sum_exp = sum_exp * rescale + p;
        let v_val = select(0.0, bitcast<f32>(buf_w[kv_base + tid]), tid < dim);
        acc = acc * rescale + p * v_val;
        max_score = new_max;
        workgroupBarrier();
        t = t + 1u;
    }
    if (tid < dim) {
        buf_c[head * dim + tid] = acc / sum_exp;
    }
}

// buf_c = silu(buf_c) * buf_a (or geglu when p1 != 0); p0 = count, p1 = geglu
@compute @workgroup_size(256)
fn swiglu(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i < pc.p0) {
        let g = buf_c[i];
        var act: f32;
        if (pc.p1 != 0u) {
            act = 0.5 * g * (1.0 + tanh(0.7978846 * g * (1.0 + 0.044715 * g * g)));
        } else {
            act = g / (1.0 + exp(-g));
        }
        buf_c[i] = act * buf_a[i];
    }
}

// buf_c += buf_a; p0 = count
@compute @workgroup_size(256)
fn residual(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i < pc.p0) {
        buf_c[i] = buf_c[i] + buf_a[i];
    }
}

// buf_c *= f0; p0 = count, f0 = scale (Gemma embedding scale)
@compute @workgroup_size(256)
fn scale_vec(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i < pc.p0) {
        buf_c[i] = buf_c[i] * pc.f0;
    }
}

// buf_c = f0 * tanh(buf_c / f0); p0 = count, f0 = cap (Gemma final softcap)
@compute @workgroup_size(256)
fn softcap(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i < pc.p0) {
        buf_c[i] = pc.f0 * tanh(buf_c[i] / pc.f0);
    }
}

// Top-k MoE router: buf_a = router logits, buf_i = selected ids, buf_c = probabilities
// p0 = experts, p1 = used, p2 = full_softmax
@compute @workgroup_size(256)
fn router(@builtin(local_invocation_id) lid: vec3<u32>) {
    let tid = lid.x;
    let experts = pc.p0;
    let used = pc.p1;
    let full_softmax = pc.p2 != 0u;
    if (tid != 0u) { return; }

    var max_val = -3.402823466e+38;
    for (var e = 0u; e < experts; e = e + 1u) {
        max_val = max(max_val, buf_a[e]);
    }
    var full_denom = 0.0;
    if (full_softmax) {
        for (var e = 0u; e < experts; e = e + 1u) {
            full_denom = full_denom + exp(buf_a[e] - max_val);
        }
    }
    for (var s = 0u; s < used; s = s + 1u) {
        var best_v = -3.402823466e+38;
        var best_i = 0u;
        for (var e = 0u; e < experts; e = e + 1u) {
            var taken = false;
            for (var p = 0u; p < s; p = p + 1u) {
                if (u32(buf_i[p]) == e) { taken = true; }
            }
            if (!taken) {
                let v = buf_a[e];
                if (v > best_v) {
                    best_v = v;
                    best_i = e;
                }
            }
        }
        buf_i[s] = i32(best_i);
        buf_c[s] = best_v;
    }
    if (full_softmax) {
        for (var s = 0u; s < used; s = s + 1u) {
            buf_c[s] = exp(buf_c[s] - max_val) / full_denom;
        }
    } else {
        var top_max = buf_c[0];
        for (var s = 1u; s < used; s = s + 1u) {
            top_max = max(top_max, buf_c[s]);
        }
        var denom = 0.0;
        for (var s = 0u; s < used; s = s + 1u) {
            let e = exp(buf_c[s] - top_max);
            buf_c[s] = e;
            denom = denom + e;
        }
        for (var s = 0u; s < used; s = s + 1u) {
            buf_c[s] = buf_c[s] / denom;
        }
    }
}

// Batched token embedding: buf_i[0..n] = token IDs, buf_c[t * width + i] = embedding.
// grid = (ceil(width / 256), n, 1); p0 = width, p1 = kind
@compute @workgroup_size(256)
fn embedding_batch(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let t = gid.y;
    let width = pc.p0;
    if (i < width) {
        let tok = u32(buf_i[t]);
        buf_c[t * width + i] = decode_weight(pc.p1, tok * width + i);
    }
}

// Batched RMSNorm: grid = (n, 1, 1); p0 = width, f0 = epsilon
@compute @workgroup_size(256)
fn rmsnorm_batch(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let tid = lid.x;
    let width = pc.p0;
    let base = wg.x * width;
    var sum = 0.0;
    var i = tid;
    loop {
        if (i >= width) { break; }
        let v = buf_a[base + i];
        sum = sum + v * v;
        i = i + 256u;
    }
    let total = workgroup_sum(tid, sum);
    let inv = inverseSqrt(total / f32(width) + pc.f0);
    i = tid;
    loop {
        if (i >= width) { break; }
        buf_c[base + i] = buf_a[base + i] * inv * bitcast<f32>(buf_w[i]);
        i = i + 256u;
    }
}

// Batched matmul: 8 rows x 4 tokens per workgroup (reads each weight block ONCE for 4 tokens).
// grid = (ceil(rows / 8), ceil(tokens / 4), 1)
// p0 = kind, p1 = width, p2 = rows, p3 = tokens
@compute @workgroup_size(256)
fn matmul_batch(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let tid = lid.x;
    let warp = tid >> 5u;
    let lane = tid & 31u;
    let row = wg.x * 8u + warp;
    let t0 = wg.y * 4u;
    let kind = pc.p0;
    let width = pc.p1;
    let rows = pc.p2;
    let tokens = pc.p3;
    let n = min(4u, tokens - t0);
    var acc0 = 0.0;
    var acc1 = 0.0;
    var acc2 = 0.0;
    var acc3 = 0.0;
    if (row < rows) {
        let row_elem = row * width;
        let x0 = t0 * width;
        let x1 = select(x0, (t0 + 1u) * width, n > 1u);
        let x2 = select(x0, (t0 + 2u) * width, n > 2u);
        let x3 = select(x0, (t0 + 3u) * width, n > 3u);
        if (kind == 12u && (width & 255u) == 0u) {
            let chunk = lane >> 3u;
            let sub = lane & 7u;
            let g_lo = chunk * 2u;
            let g_hi = g_lo + 1u;
            let num_blocks = width >> 8u;
            let row_blk = row_elem >> 8u;
            for (var b = 0u; b < num_blocks; b = b + 1u) {
                let base = (row_blk + b) * 144u;
                let dd = buf_w[base >> 2u];
                let d = half_to_f32(dd & 0xffffu);
                let dm = half_to_f32(dd >> 16u);
                let s0 = buf_w[(base >> 2u) + 1u];
                let s1 = buf_w[(base >> 2u) + 2u];
                let s2 = buf_w[(base >> 2u) + 3u];
                let sh_lo = (g_lo & 3u) * 8u;
                let lo0 = (s0 >> sh_lo) & 255u;
                let mid0 = (s1 >> sh_lo) & 255u;
                let hi0 = (s2 >> sh_lo) & 255u;
                let sc0 = select((hi0 & 15u) | ((lo0 >> 6u) << 4u), lo0 & 63u, g_lo < 4u);
                let mn0 = select((hi0 >> 4u) | ((mid0 >> 6u) << 4u), mid0 & 63u, g_lo < 4u);
                let sh_hi = (g_hi & 3u) * 8u;
                let lo1 = (s0 >> sh_hi) & 255u;
                let mid1 = (s1 >> sh_hi) & 255u;
                let hi1 = (s2 >> sh_hi) & 255u;
                let sc1 = select((hi1 & 15u) | ((lo1 >> 6u) << 4u), lo1 & 63u, g_hi < 4u);
                let mn1 = select((hi1 >> 4u) | ((mid1 >> 6u) << 4u), mid1 & 63u, g_hi < 4u);
                let ds0 = d * f32(sc0);
                let ms0 = dm * f32(mn0);
                let ds1 = d * f32(sc1);
                let ms1 = dm * f32(mn1);
                let qw = buf_w[(base + 16u + chunk * 32u + sub * 4u) >> 2u];
                let off0 = (b << 8u) + g_lo * 32u + sub * 4u;
                let off1 = (b << 8u) + g_hi * 32u + sub * 4u;
                for (var k = 0u; k < 4u; k = k + 1u) {
                    let byte_v = (qw >> (k * 8u)) & 255u;
                    let w0 = ds0 * f32(byte_v & 15u) - ms0;
                    let w1 = ds1 * f32(byte_v >> 4u) - ms1;
                    acc0 = acc0 + w0 * buf_a[x0 + off0 + k] + w1 * buf_a[x0 + off1 + k];
                    acc1 = acc1 + w0 * buf_a[x1 + off0 + k] + w1 * buf_a[x1 + off1 + k];
                    acc2 = acc2 + w0 * buf_a[x2 + off0 + k] + w1 * buf_a[x2 + off1 + k];
                    acc3 = acc3 + w0 * buf_a[x3 + off0 + k] + w1 * buf_a[x3 + off1 + k];
                }
            }
        } else {
            var i = lane;
            loop {
                if (i >= width) { break; }
                let wv = decode_weight(kind, row_elem + i);
                acc0 = acc0 + wv * buf_a[x0 + i];
                acc1 = acc1 + wv * buf_a[x1 + i];
                acc2 = acc2 + wv * buf_a[x2 + i];
                acc3 = acc3 + wv * buf_a[x3 + i];
                i = i + 32u;
            }
        }
    }
    let s0 = lane32_sum(tid, lane, acc0);
    let s1 = lane32_sum(tid, lane, acc1);
    let s2 = lane32_sum(tid, lane, acc2);
    let s3 = lane32_sum(tid, lane, acc3);
    if (lane == 0u && row < rows) {
        buf_c[t0 * rows + row] = s0;
        if (n > 1u) { buf_c[(t0 + 1u) * rows + row] = s1; }
        if (n > 2u) { buf_c[(t0 + 2u) * rows + row] = s2; }
        if (n > 3u) { buf_c[(t0 + 3u) * rows + row] = s3; }
    }
}

// Batched row bias: grid = (ceil(width / 256), n, 1); p0 = width
@compute @workgroup_size(256)
fn add_row_bias_batch(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let t = gid.y;
    let w = pc.p0;
    if (i < w) {
        buf_c[t * w + i] = buf_c[t * w + i] + bitcast<f32>(buf_w[i]);
    }
}

// Batched head_norm: grid = (heads_per_token, n, 1); p0 = norm_width, p1 = total_size
@compute @workgroup_size(256)
fn head_norm_batch(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let tid = lid.x;
    let dim = pc.p0;
    let base = wg.y * pc.p1 + wg.x * dim;
    var sum = 0.0;
    var i = tid;
    loop {
        if (i >= dim) { break; }
        let v = buf_c[base + i];
        sum = sum + v * v;
        i = i + 256u;
    }
    let total = workgroup_sum(tid, sum);
    let inv = inverseSqrt(total / f32(dim) + pc.f0);
    i = tid;
    loop {
        if (i >= dim) { break; }
        buf_c[base + i] = buf_c[base + i] * inv * bitcast<f32>(buf_w[i]);
        i = i + 256u;
    }
}

// Batched RoPE: grid = (ceil(pairs / 256), n, 1)
// p0 = size, p1 = dim, p2 = base_position, p3 = neox, p4 = rotary_dim, p5 = has_factors
@compute @workgroup_size(256)
fn rotary_batch(@builtin(global_invocation_id) gid: vec3<u32>) {
    let p = gid.x;
    let t = gid.y;
    let size = pc.p0;
    let dim = pc.p1;
    let rotated = pc.p4;
    let half = rotated >> 1u;
    let num_heads = size / dim;
    if (p >= num_heads * half) { return; }
    let pair = p % half;
    let head = p / half;
    let base = t * size + head * dim;
    let a = base + select(2u * pair, pair, pc.p3 != 0u);
    let b = select(a + 1u, a + half, pc.p3 != 0u);
    var angle = f32(pc.p2 + t) * pow(pc.f0, -f32(2u * pair) / f32(rotated));
    if (pc.p5 != 0u) {
        angle = angle / buf_b[pair];
    }
    let s = sin(angle) * pc.f1;
    let c = cos(angle) * pc.f1;
    let u = buf_c[a];
    let v = buf_c[b];
    buf_c[a] = u * c - v * s;
    buf_c[b] = u * s + v * c;
}

// Batched KV cache write: grid = (ceil(kw / 256), n, 1); p0 = kw, p1 = base_position
@compute @workgroup_size(256)
fn cache_kv_batch(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let t = gid.y;
    let kw = pc.p0;
    if (i < kw) {
        let src = t * kw + i;
        let dst = (pc.p1 + t) * kw + i;
        buf_c[dst] = buf_a[src];
        buf_d[dst] = buf_b[src];
    }
}

// Batched causal attention: grid = (heads, n, 1)
// p0 = dim, p1 = heads, p2 = kv_heads, p3 = base_count, p4 = window
@compute @workgroup_size(256)
fn attention_batch(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let head = wg.x;
    let tok = wg.y;
    let tid = lid.x;
    let dim = pc.p0;
    let heads = pc.p1;
    let kv_heads = pc.p2;
    let count = pc.p3 + tok;
    let window = pc.p4;
    let kv_head = head / (heads / kv_heads);
    let q_off = (tok * heads + head) * dim;
    let q_val = select(0.0, buf_a[q_off + tid] * pc.f0, tid < dim);
    let start = select(0u, count - window, window > 0u && count > window);

    var max_score = -3.402823466e+38;
    var sum_exp = 0.0;
    var acc = 0.0;
    var t = start;
    loop {
        if (t >= count) { break; }
        let kv_base = (t * kv_heads + kv_head) * dim;
        let k_val = select(0.0, buf_b[kv_base + tid], tid < dim);
        var score = workgroup_sum(tid, q_val * k_val);
        if (pc.f1 > 0.0) {
            score = pc.f1 * tanh(score / pc.f1);
        }
        let new_max = max(max_score, score);
        let rescale = exp(max_score - new_max);
        let p = exp(score - new_max);
        sum_exp = sum_exp * rescale + p;
        let v_val = select(0.0, bitcast<f32>(buf_w[kv_base + tid]), tid < dim);
        acc = acc * rescale + p * v_val;
        max_score = new_max;
        workgroupBarrier();
        t = t + 1u;
    }
    if (tid < dim) {
        buf_c[q_off + tid] = acc / sum_exp;
    }
}

// Copy last token's hidden state buf_a[(n-1)*width + i] -> buf_c[i]; p0 = width, p1 = n - 1
@compute @workgroup_size(256)
fn copy_last_token(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let w = pc.p0;
    if (i < w) {
        buf_c[i] = buf_a[pc.p1 * w + i];
    }
}

// buf_c[i] += sum_s(buf_b[s] * buf_a[s * width + i]); p0 = width, p1 = used
@compute @workgroup_size(256)
fn combine_experts(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let width = pc.p0;
    if (i < width) {
        var sum = 0.0;
        for (var s = 0u; s < pc.p1; s = s + 1u) {
            sum = sum + buf_b[s] * buf_a[s * width + i];
        }
        buf_c[i] = buf_c[i] + sum;
    }
}
