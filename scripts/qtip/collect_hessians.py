#!/usr/bin/env python3
"""Collect input Hessians H = X^T X / n for every linear layer of a Llama GGUF.

Research tool for the QTIP prototype (feature `qtip-research`). The model is
loaded with transformers (which dequantizes the GGUF) and run on CUDA over
`--chunks` chunks of `--context` tokens, each starting with BOS, as
`llama-perplexity` and `llama-imatrix` do. A pre-forward hook on each linear
layer adds X^T X to an f64 accumulator. Layers that read the same input share
one file: attn_q/k/v, attn_output, ffn_gate/up, ffn_down.

Output directory:
  manifest.json         {"tokens": n, "tensors": {gguf name: {"file", "dim"}}}
  blk.<i>.<group>.f32   row-major dim x dim little-endian f32 of H / n

Before collecting, `--eval-chunks` chunks of `--eval-text` are scored (second
half of each chunk) as a check that the torch model matches the GGUF.

    python3 scripts/qtip/collect_hessians.py \
        --model models/compat-llama32/Llama-3.2-1B-Instruct-f16.gguf \
        --text models/wikitext-2-raw/wiki.train.raw \
        --eval-text models/wikitext-2-raw/wiki.test.raw \
        --output models/compat-llama32/hessians-l32-1b
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import gguf
import torch
from tokenizers import Regex, pre_tokenizers
from transformers import AutoModelForCausalLM, AutoTokenizer

LLAMA3_SPLIT = (
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}"
    r"| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
)

# Group name -> (module path inside a decoder layer, GGUF tensors that read that input).
GROUPS = {
    "attn_qkv": ("self_attn.q_proj", ["attn_q", "attn_k", "attn_v"]),
    "attn_output": ("self_attn.o_proj", ["attn_output"]),
    "ffn_gate_up": ("mlp.gate_proj", ["ffn_gate", "ffn_up"]),
    "ffn_down": ("mlp.down_proj", ["ffn_down"]),
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--text", type=Path, required=True, help="calibration text")
    parser.add_argument("--output", type=Path, required=True, help="new directory")
    parser.add_argument("--context", type=int, default=512)
    parser.add_argument("--chunks", type=int, default=256)
    parser.add_argument("--eval-text", type=Path)
    parser.add_argument("--eval-chunks", type=int, default=8)
    return parser.parse_args()


def load(path: Path):
    if not path.is_file():
        sys.exit(f"model not found: {path}")
    reader = gguf.GGUFReader(path)
    kwargs = {"gguf_file": path.name, "dtype": torch.float32}
    tokenizer = AutoTokenizer.from_pretrained(path.parent, gguf_file=path.name)
    fix_pre_tokenizer(tokenizer, reader)
    model = AutoModelForCausalLM.from_pretrained(path.parent, **kwargs).cuda().eval()
    apply_rope_factors(model, reader)
    return tokenizer, model


def fix_pre_tokenizer(tokenizer, reader):
    """transformers builds a GPT-2 pre-tokenizer from a `llama-bpe` GGUF, which
    splits digits differently ("2|000" instead of "200|0"). Use the Llama 3
    split, which reproduces llama.cpp's token ids exactly on wikitext-2."""
    field = reader.fields.get("tokenizer.ggml.pre")
    if field is None or bytes(field.parts[field.data[0]]).decode() != "llama-bpe":
        return
    tokenizer.backend_tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Split(Regex(LLAMA3_SPLIT), behavior="isolated", invert=False),
        pre_tokenizers.ByteLevel(add_prefix_space=False, trim_offsets=True, use_regex=False),
    ])


def apply_rope_factors(model, reader):
    """Llama 3 GGUFs keep RoPE scaling as `rope_freqs.weight` divisors, which
    transformers drops (rope_scaling comes back None). Divide the inverse
    frequencies by them, as llama.cpp does with its freq_factors."""
    tensor = next((t for t in reader.tensors if t.name == "rope_freqs.weight"), None)
    if tensor is None:
        return
    rotary = model.model.rotary_emb
    factors = torch.tensor(tensor.data, dtype=rotary.inv_freq.dtype, device=rotary.inv_freq.device)
    if factors.shape != rotary.inv_freq.shape:
        sys.exit(f"rope_freqs has shape {tuple(factors.shape)}, expected {tuple(rotary.inv_freq.shape)}")
    rotary.inv_freq = rotary.inv_freq / factors
    rotary.original_inv_freq = rotary.inv_freq


def chunks(tokenizer, text_path: Path, context: int, limit: int):
    """Consecutive `context`-token windows with BOS in front, as a (count, context) tensor."""
    if not text_path.is_file():
        sys.exit(f"text not found: {text_path}")
    ids = tokenizer(text_path.read_text(), add_special_tokens=False)["input_ids"]
    count = min(limit, len(ids) // context)
    if count == 0:
        sys.exit(f"{text_path} is shorter than one chunk of {context} tokens")
    windows = torch.tensor(ids[: count * context]).view(count, context)
    windows[:, 0] = tokenizer.bos_token_id
    return windows


@torch.no_grad()
def perplexity(model, windows) -> float:
    total, count = 0.0, 0
    half = windows.shape[1] // 2
    for window in windows:
        logits = model(window[None].cuda()).logits[0].double()
        log_probs = torch.log_softmax(logits[half - 1 : -1], dim=-1)
        targets = window[half:].cuda()
        total -= log_probs.gather(1, targets[:, None]).sum().item()
        count += targets.numel()
    return math.exp(total / count)


@torch.no_grad()
def collect(model, windows):
    torch.backends.cuda.matmul.allow_tf32 = False
    sums, handles = {}, []
    for index, layer in enumerate(model.model.layers):
        for group, (path, _) in GROUPS.items():
            module = layer.get_submodule(path)
            key = f"blk.{index}.{group}"
            sums[key] = torch.zeros(module.in_features, module.in_features, dtype=torch.float64, device="cuda")

            def hook(_module, inputs, key=key):
                x = inputs[0].reshape(-1, inputs[0].shape[-1]).float()
                sums[key] += (x.T @ x).double()

            handles.append(module.register_forward_pre_hook(hook))
    started = time.time()
    try:
        for done, window in enumerate(windows, 1):
            model(window[None].cuda())
            if done % 32 == 0 or done == len(windows):
                print(f"[{done}/{len(windows)}] {time.time() - started:.0f}s", flush=True)
    finally:
        for handle in handles:
            handle.remove()
    return sums


def save(sums, tokens: int, layers: int, output: Path):
    output.mkdir(parents=True)
    tensors = {}
    for key, total in sums.items():
        mean = (total / tokens).float()
        if not torch.isfinite(mean).all():
            sys.exit(f"{key}: Hessian has non-finite entries")
        file = f"{key}.f32"
        mean.cpu().numpy().astype("<f4").tofile(output / file)
        layer, group = key.split(".")[1], key.split(".", 2)[2]
        for name in GROUPS[group][1]:
            tensors[f"blk.{layer}.{name}.weight"] = {"file": file, "dim": mean.shape[0]}
    manifest = {"tokens": tokens, "layers": layers, "tensors": tensors}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=1, sort_keys=True))


def main():
    args = parse_args()
    if args.output.exists():
        sys.exit(f"{args.output} already exists")
    tokenizer, model = load(args.model)
    if args.eval_text:
        windows = chunks(tokenizer, args.eval_text, args.context, args.eval_chunks)
        print(f"check: perplexity {perplexity(model, windows):.4f} over {len(windows)} chunks", flush=True)
    windows = chunks(tokenizer, args.text, args.context, args.chunks)
    sums = collect(model, windows)
    save(sums, windows.numel(), len(model.model.layers), args.output)
    print(f"wrote {len(sums)} Hessians from {windows.numel()} tokens to {args.output}")


if __name__ == "__main__":
    main()
