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
// workgroup_id.x = row, workgroup_id.y = selected
// p0 = kind, p1 = width, p2 = rows, p3 = moe, p4 = batched
@compute @workgroup_size(256)
fn matvec(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let row = wg.x;
    let selected = wg.y;
    let tid = lid.x;
    let kind = pc.p0;
    let width = pc.p1;
    let rows = pc.p2;
    if (row >= rows) { return; }
    let expert = select(0u, u32(buf_i[selected]), pc.p3 != 0u);
    let row_elem = (expert * rows + row) * width;
    let x_base = select(0u, selected * width, pc.p4 != 0u);
    var sum = 0.0;
    var i = tid;
    loop {
        if (i >= width) { break; }
        sum = sum + decode_weight(kind, row_elem + i) * buf_a[x_base + i];
        i = i + 256u;
    }
    let total = workgroup_sum(tid, sum);
    if (tid == 0u) {
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
