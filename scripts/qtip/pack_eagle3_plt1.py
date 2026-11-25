#!/usr/bin/env python3
"""Pack a HuggingFace EAGLE-3 checkpoint (e.g. yuhuili/EAGLE3-LLaMA3.1-Instruct-8B)
into a self-contained PLT1 (.palito) binary container with Q8_0 weights (GGUF kind 8)
for fast, lossless GPU drafting in Palito.

Q8_0 block layout (34 bytes per 32 elements):
  - f16 scale d = max(|x|) / 127.0  (2 bytes)
  - 32 x int8 qs = round(x / d)     (32 bytes)
Matches Palito's vectorized `q8_0_row_part` kernel bit-for-bit.
"""

import argparse
import json
import struct
from pathlib import Path
import numpy as np
import torch

PLT1_MAGIC = b"PALITO1\0"
PLT1_VERSION = 1
HEADER_ALIGN = 4096
PAYLOAD_ALIGN = 128

SEC_METADATA_JSON = 1
SEC_TENSOR_DIRECTORY = 2
SEC_LAYER_SLABS = 4
SEC_EAGLE3_SPEC_HEAD = 7

CODEC_F16 = 0
CODEC_Q8_0 = 1
CODEC_Q4_K = 3
FLAG_HAS_EAGLE3 = 1 << 4


def align_up(v: int, a: int) -> int:
    return (v + a - 1) & ~(a - 1)


def quantize_q8_0(t: torch.Tensor) -> bytes:
    """Vectorized Q8_0 quantization: [rows, cols] -> bytes (34 bytes per 32 weights)."""
    arr = t.detach().float().numpy()
    assert arr.ndim == 2 and arr.shape[1] % 32 == 0, f"bad shape {arr.shape}"
    rows, cols = arr.shape
    blocks = arr.reshape(-1, 32)
    amax = np.max(np.abs(blocks), axis=1, keepdims=True)
    scales = (amax / 127.0).astype(np.float16)
    inv = np.where(scales.astype(np.float32) > 0, 1.0 / scales.astype(np.float32), 0.0)
    qs = np.clip(np.rint(blocks * inv), -127, 127).astype(np.int8)

    out = np.empty((blocks.shape[0], 34), dtype=np.uint8)
    out[:, :2] = scales.view(np.uint8).reshape(-1, 2)
    out[:, 2:] = qs.view(np.uint8)
    return out.tobytes()


def quantize_q4_k(t: torch.Tensor) -> bytes:
    """Vectorized Q4_K quantization: [rows, cols] -> bytes (144 bytes per 256 weights).
    Matches Palito's `q4k_wide_block` kernel bit-for-bit."""
    arr = t.detach().float().numpy()
    assert arr.ndim == 2 and arr.shape[1] % 256 == 0, f"bad shape {arr.shape}"
    # Reshape to [num_superblocks, 8 groups, 32 elements]
    sb = arr.reshape(-1, 8, 32)
    n_sb = sb.shape[0]

    # Per-group min (as positive offset m = max(0, -min)) and scale s = (max + m) / 15
    g_min = np.minimum(np.min(sb, axis=2), 0.0)   # <= 0
    g_max = np.max(sb, axis=2)
    m_raw = -g_min                                # >= 0, shape [n_sb, 8]
    s_raw = (g_max - g_min) / 15.0                # >= 0, shape [n_sb, 8]

    # Superblock f16 scales d, dm quantizing s_raw, m_raw into 6-bit [0..63]
    d = (np.max(s_raw, axis=1, keepdims=True) / 63.0).astype(np.float16)
    dm = (np.max(m_raw, axis=1, keepdims=True) / 63.0).astype(np.float16)
    d_f32 = d.astype(np.float32)
    dm_f32 = dm.astype(np.float32)

    sc = np.clip(np.rint(np.where(d_f32 > 0, s_raw / d_f32, 0.0)), 0, 63).astype(np.uint8)
    mn = np.clip(np.rint(np.where(dm_f32 > 0, m_raw / dm_f32, 0.0)), 0, 63).astype(np.uint8)

    # Quantize elements using reconstructed ds = d * sc, ms = dm * mn
    ds = (d_f32 * sc.astype(np.float32))[:, :, None]
    ms = (dm_f32 * mn.astype(np.float32))[:, :, None]
    q = np.clip(np.rint(np.where(ds > 0, (sb + ms) / ds, 0.0)), 0, 15).astype(np.uint8)

    out = np.zeros((n_sb, 144), dtype=np.uint8)
    out[:, 0:2] = d.view(np.uint8).reshape(n_sb, 2)
    out[:, 2:4] = dm.view(np.uint8).reshape(n_sb, 2)

    # Pack 12 scale bytes matching q4k_wide_block:
    # bytes 0..4: sc[0..4] | ((sc[4..8] >> 4) << 6)
    # bytes 4..8: mn[0..4] | ((mn[4..8] >> 4) << 6)
    # bytes 8..12: (sc[4..8] & 15) | ((mn[4..8] & 15) << 4)
    out[:, 4:8] = (sc[:, 0:4] & 63) | ((sc[:, 4:8] >> 4) << 6)
    out[:, 8:12] = (mn[:, 0:4] & 63) | ((mn[:, 4:8] >> 4) << 6)
    out[:, 12:16] = (sc[:, 4:8] & 15) | ((mn[:, 4:8] & 15) << 4)

    # Pack 128 nibble bytes: 4 pairs of 32 bytes (low=group 2*p, high=group 2*p+1)
    for pair in range(4):
        out[:, 16 + pair * 32 : 16 + (pair + 1) * 32] = (
            (q[:, 2 * pair, :] & 15) | ((q[:, 2 * pair + 1, :] & 15) << 4)
        )
    return out.tobytes()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True, help="Path to pytorch_model.bin")
    ap.add_argument("--config", type=Path, required=True, help="Path to config.json")
    ap.add_argument("--output", type=Path, required=True, help="Output .palito file")
    ap.add_argument("--tap-layers", type=str, default="2,16,29")
    ap.add_argument("--quant", type=str, default="q4_k", choices=["q8_0", "q4_k"])
    args = ap.parse_args()

    cfg = json.loads(args.config.read_text())
    if args.input.suffix == ".safetensors":
        from safetensors.torch import load_file

        sd = load_file(str(args.input))
    else:
        sd = torch.load(args.input, map_location="cpu", weights_only=False)
    tap_layers = [int(x) for x in args.tap_layers.split(",")]
    assert len(tap_layers) == 3

    d2t = sd["d2t"].detach().to(torch.int32).numpy()
    draft_vocab = int(d2t.shape[0])
    hidden = int(cfg["hidden_size"])
    inter = int(cfg["intermediate_size"])
    vocab = int(cfg["vocab_size"])

    # Pack EAGLE3 metadata + d2t into SEC_EAGLE3_SPEC_HEAD.
    # Layout: [tap0:u32, tap1:u32, tap2:u32, draft_vocab:u32, rope_theta:f32, eps:f32, d2t:i32[draft_vocab]]
    rope_theta = float(cfg.get("rope_theta", 10000.0))
    eps = float(cfg.get("rms_norm_eps", 1e-5))
    eagle_sec = struct.pack("<IIIIff", tap_layers[0], tap_layers[1], tap_layers[2], draft_vocab, rope_theta, eps)
    eagle_sec += d2t.tobytes()

    meta_json = json.dumps({
        "kind": "eagle3",
        "hidden_size": hidden,
        "intermediate_size": inter,
        "vocab_size": vocab,
        "draft_vocab_size": draft_vocab,
        "tap_layers": tap_layers,
        "codec": "q8_0",
    }).encode("utf-8")

    # Tensors to pack: 2D projections in Q8_0, 1D norms in F32 (stored as CODEC_F16=0? Wait: store F32 bytes, rows=1, cols=H).
    q8_names = [
        "fc.weight",
        "midlayer.self_attn.q_proj.weight",
        "midlayer.self_attn.k_proj.weight",
        "midlayer.self_attn.v_proj.weight",
        "midlayer.self_attn.o_proj.weight",
        "midlayer.mlp.gate_proj.weight",
        "midlayer.mlp.up_proj.weight",
        "midlayer.mlp.down_proj.weight",
        "lm_head.weight",
    ]
    norm_names = [
        "midlayer.input_layernorm.weight",
        "midlayer.hidden_norm.weight",
        "midlayer.post_attention_layernorm.weight",
        "norm.weight",
    ]

    tensor_blobs = []
    q_fn, q_codec = (quantize_q4_k, CODEC_Q4_K) if args.quant == "q4_k" else (quantize_q8_0, CODEC_Q8_0)
    for name in q8_names:
        t = sd[name]
        rows, cols = int(t.shape[0]), int(t.shape[1])
        blob = q_fn(t)
        tensor_blobs.append((name, rows, cols, q_codec, blob))
        print(f"  {args.quant.upper():4} {name:42} {rows:6}x{cols:<6} -> {len(blob)/1e6:6.2f} MB")

    for name in norm_names:
        t = sd[name].detach().float().numpy()
        cols = int(t.shape[0])
        blob = t.astype(np.float32).tobytes()
        tensor_blobs.append((name, 1, cols, CODEC_F16, blob))  # F32 norm vector

    # Write PLT1 container matching src/plt1.rs bit-for-bit.
    buf = bytearray(HEADER_ALIGN)
    cursor = HEADER_ALIGN
    sections = []

    def append_pad(align: int):
        nonlocal cursor
        target = align_up(cursor, align)
        if target > cursor:
            buf.extend(b"\x00" * (target - cursor))
            cursor = target

    for kind, data in [(SEC_METADATA_JSON, meta_json), (SEC_EAGLE3_SPEC_HEAD, eagle_sec)]:
        append_pad(PAYLOAD_ALIGN)
        off = cursor
        buf.extend(data)
        cursor += len(data)
        sections.append((kind, 0, off, len(data)))

    append_pad(PAYLOAD_ALIGN)
    slabs_start = cursor
    descs = []
    for name, rows, cols, codec, data in tensor_blobs:
        append_pad(PAYLOAD_ALIGN)
        off = cursor
        buf.extend(data)
        cursor += len(data)
        descs.append((name, rows, cols, codec, off, len(data)))
    sections.append((SEC_LAYER_SLABS, 0, slabs_start, cursor - slabs_start))

    # Build TensorDirectory matching src/plt1.rs.
    dir_bytes = bytearray(struct.pack("<I", len(descs)))
    for name, rows, cols, codec, off, size in descs:
        nb = name.encode("utf-8")
        dir_bytes.extend(struct.pack("<H", len(nb)))
        dir_bytes.extend(nb)
        dir_bytes.extend(struct.pack("<IIHHBBf4fQQQ",
            rows, cols, codec, 0, 0, 0, 1.0,
            0.0, 0.0, 0.0, 0.0,
            off, size, 0))
    append_pad(PAYLOAD_ALIGN)
    dir_off = cursor
    buf.extend(dir_bytes)
    cursor += len(dir_bytes)
    sections.append((SEC_TENSOR_DIRECTORY, 0, dir_off, len(dir_bytes)))

    # Write 64B header + section directory at offset 0.
    hdr = struct.pack(
        "<8sIIIIQIIII16s",
        PLT1_MAGIC,
        PLT1_VERSION,
        1200,
        FLAG_HAS_EAGLE3,
        len(sections),
        64,
        1,
        hidden,
        inter,
        vocab,
        b"\x00" * 16,
    )
    buf[0:64] = hdr
    pos = 64
    for kind, sflags, off, size in sections:
        buf[pos:pos + 24] = struct.pack("<IIQQ", kind, sflags, off, size)
        pos += 24

    args.output.write_bytes(buf)
    print(f"Wrote {args.output} ({len(buf)/1e6:.2f} MB, {len(descs)} tensors)")


if __name__ == "__main__":
    main()
