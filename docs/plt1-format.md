# Palito `.palito` (`PLT1`) Format & Technical Guide

This guide covers the technical details behind Palito's GPU-native **`.palito` (`PLT1`)** model format, how to convert GGUF models to `PLT1`, how to bundle an `EAGLE-3` draft head, and all advanced CLI, server, and C API options.

---

## 1. How `.palito` (`PLT1`) Works

Standard GGUF files were designed for CPU SIMD, which forces GPUs to waste cycles unpacking unaligned bitstreams and running extra memory passes. `PLT1` is designed from the ground up for modern NVIDIA GPUs (Blackwell `SM_120` / Ampere+):

1. **Zero-SMEM Warp-Register LUT (`HYB Q=5`, `3.0 bpw`):**
   - Each `16×16` weight tile is encoded via a 2-pass circular (tail-biting) `L=10, B=6, V=2` bitshift trellis into **96 bytes** (`3.0` bits/weight) — fitting Llama 3.1 8B in **3.5 GiB** (`PPL = 7.64`), Qwen 3 8B in **3.7 GiB**, and Qwen3-30B-A3B MoE in **11.1 GiB** with coherent output.
   - Instead of expensive integer arithmetic (`1MAD`) or shared-memory lookup tables that reduce SM occupancy, the 32-entry `half2` codebook lives directly in warp registers (`__shfl_sync` + dual sign-flip `0x80008000` = 128 2D Gaussian centroids, **0 KB SMEM**, 100% SM occupancy).
   - *Note on `2.0 bpw` (`L=12, B=4, Q=5`):* Although the container format supports 2.0 bpw, post-training quantization at 2.0 bpw without blockwise fine-tuning degrades quality on 8B models (causing repetitive/degenerate output). Always use the default `3.0 bpw` (`L=10, B=6, Q=5`) preset for production models.
2. **Offline Rotation & `RMSNorm` Folding (`FOLDED_RMSNORM | FOLDED_RESIDUAL_RHT`):**
   - Layer `RMSNorm` weights ($\gamma$) and a shared global residual-stream Hadamard rotation $R_1$ are folded into the projection matrices offline during conversion.
   - At runtime, `hyb_r1_enter` / `hyb_r1_exit` run once per step instead of 14 transforms per layer, and `residual_unit_norm` fuses residual addition + unit-variance `RMSNorm` (eliminating 63 kernel launches per step).
3. **Pre-Swizzled Tensor-Core Tiles + Native Speculative Decoding (`EAGLE-3` + `SuffixDecoding`):**
   - Tiles are pre-swizzled offline into exact `mma.sync.aligned.m16n8k16` A-fragment register order, allowing Palito to verify up to **8–16 draft tokens** in nearly the same time as 1 token on RTX 5090.
   - **Zero-sync GPU-chained `EAGLE-3` (`Q4_K`, 239 MB)** drafts 5–7 tokens ahead in a single uninterrupted GPU chain (`eagle3_argmax_d2t`), paired with zero-weight **`SuffixDecoding`** for instant drafts on code, refactoring, and RAG.
   - Output is **100% lossless** (`exact_parity=true` against single-token greedy decoding).

---

## 2. Step-by-Step: Converting & Running Models in `.palito` Format

### Prerequisites
- Linux x86_64 with an NVIDIA GPU and a CUDA 13 driver.
- `palito` with `convert`: the [Linux bundle](../README.md#download) includes it. From source, build with `--features cuda,qtip-research` (see [building from source](building.md)).

### Step 1 — Convert Any GGUF Model to `.palito` in One Command

`palito convert` automatically collects calibration Hessians on CUDA, quantizes all layers (`3.0 bpw` `HYB Q=5` with GPU Viterbi), strips the base GGUF to a compact skeleton, optionally embeds an `EAGLE-3` head, and writes **one** self-contained `.palito` file:

```sh
# One-command conversion (uses built-in calibration corpus automatically):
palito convert \
  --input models/Qwen3-8B-Q6_K.gguf \
  --output models/qwen3-8b.palito

# With full WikiText-2 calibration and an EAGLE-3 draft head:
palito convert \
  --input models/Qwen3-8B-Q6_K.gguf \
  --calibration-text models/wikitext-2-raw/wiki.train.raw \
  --eagle3 models/eagle3/eagle3-q4_k.palito \
  --output models/qwen3-8b.palito
```

*(To pack an `EAGLE-3` checkpoint from Hugging Face first:)*

```sh
hf download AngelSlim/Qwen3-8B_eagle3 --local-dir models/eagle3
python3 scripts/qtip/pack_eagle3_plt1.py \
  --input models/eagle3/pytorch_model.bin \
  --config models/eagle3/config.json \
  --output models/eagle3/eagle3-q4_k.palito \
  --quant q4_k --tap-layers 2,18,33
```

### Step 2 — Run with One Command (Auto CUDA + Speculative Decoding)

```sh
palito chat --model models/qwen3-8b.palito --stats
# Or OpenAI-compatible server:
palito serve --model models/qwen3-8b.palito
```

`convert` refuses to overwrite an existing `--output` unless you pass `-f` / `--force`. If it fails or you press Ctrl-C, it removes its temporary files (`*.hessians.tmp`, `*.layers.palito.tmp`) and the partial output; `--keep-temp` keeps the temporaries. Run `palito convert --help` for every option (`--bits`, `--hessians`, `--no-hessians`, `--chunks`, `--max-layers`, ...).

*(For step-by-step control or microbenchmarking, the standalone `collect-hessians-cuda`, `gguf-to-plt1`, `pack-palito`, `eagle3-spec-bench`, and `suffix-spec-bench` binaries remain available from a source build.)*

---

## 3. CLI & Server Reference

Every command documents its options: `palito <command> -h` shows the common ones and `palito <command> --help` all of them, with examples. `palito --version` lists the backends and features the binary was built with, and `palito completions <bash|zsh|fish|powershell>` prints a shell completion script.

Sizes such as `--gpu-memory-limit-bytes`, `--memory-limit-bytes` and `--prefix-cache-bytes` accept units (`20GiB`, `512MB`) and also have shorter names without `-bytes`.

### `palito chat` Options

| Option | What it does |
|---|---|
| `--no-thinking` | Turns off Qwen3's reasoning (`<think>`), for short, direct answers |
| `--system "You are..."` | Sets a system message |
| `--stats` | Prints tokens, time to first token, and tok/s after each reply |
| `--tokens N` | Caps each reply at N tokens (default: until stop or full context) |
| `--context N` | Uses N context tokens instead of the model's full context |
| `--no-stream` | Prints each reply only once it is complete |
| `--temperature T` | Sampling temperature (default 0: greedy, repeatable) |
| `--top-p P`, `--top-k K`, `--seed S` | Top-p / top-k sampling and deterministic seed |
| `--min-p P` | Drops tokens below `P * p_max` (default 0: off; llama.cpp uses 0.05) |
| `--repeat-penalty R`, `--repeat-last-n N` | Repetition penalty over last N tokens (default 64) |
| `--presence-penalty P`, `--frequency-penalty F` | OpenAI-style penalties |
| `--stop TEXT` | Stops generation at TEXT (repeatable) |
| `--no-prefix-cache` | Disables GPU prefix caching between chat turns |
| `--chat-template-file FILE` | Uses a Jinja template instead of the model's chat template |
| `--backend cpu\|cuda` | Picks the backend (default: `cuda` for `.palito`, `cpu` for GGUF) |
| `--gpu-device N` | GPU index (`--cuda-device` still works) |

Inside the chat, `/help` lists the commands, `/reset` clears the conversation and `/exit` (or Ctrl-D) quits. Ctrl-C stops the reply being generated; at the prompt it quits. To send text that starts with `/`, type it as `//text`. With input piped in (`echo question | palito chat ...`), a failed reply makes `palito` exit with status 1.

### `palito serve` Details
- **Options:** `--host` (default `127.0.0.1`), `--port` (default `8080`), `--backend`, `--context`, `--model-name` (the id `/v1/models` reports), `--parallel` / `-np`, `--queue`, `--api-key`, `--no-prefix-cache`, `--gpu-device`, `--gpu-memory-limit-bytes`, and the sampling defaults of `chat` (`--temperature`, `--top-p`, ...) for requests that omit them.
- **Startup:** the port is opened before the model loads, so a busy port fails at once; on a terminal, `serve` prints the Palito banner.
- **Endpoints:** `POST /v1/chat/completions`, `POST /v1/completions` (streaming supported), `GET /v1/models`, `GET /health`.
- **Reasoning:** Qwen3 `<think>` blocks are returned in `reasoning_content`; send `"reasoning_format": "none"` to inline or `"chat_template_kwargs": {"enable_thinking": false}` to disable.
- **Structured JSON:** `"response_format": {"type": "json_object"}` or `{"type": "json_schema", ...}` uses constrained decoding (works with greedy and sampling).
- **Tool Calling:** Supports `tools`, `tool_choice` (`auto`, `required`, `none`, or named function), `parallel_tool_calls`, and `role: "tool"` history for Qwen2.5/Qwen3 templates.
- **Concurrency:** `--parallel N` (`-np N`, GPU only) serves up to N concurrent requests with `--queue` (default 16) and shared `--context` split across slots.
- **Security:** Bound to `127.0.0.1` by default. For network access use `--host 0.0.0.0 --api-key YOUR_KEY` (or `PALITO_API_KEY`).

### C Library (`libpalito.so`)

See [`include/palito.h`](../include/palito.h) and [`examples/c/chat.c`](../examples/c/chat.c):

```c
palito_engine *engine = palito_open("model.gguf", NULL, &error);
palito_message messages[] = {{"user", "Hello!"}};
palito_result result;
palito_chat(engine, messages, 1, NULL, print_text, NULL, &result, &error);
palito_result_free(&result);
palito_close(engine);
```

```sh
cc examples/c/chat.c -Iinclude -Llib -lpalito -Wl,-rpath,"$PWD/lib" -o chat
./chat models/Qwen3-30B-A3B-Q4_K_M.gguf "Hello!"
```

---

## 4. Troubleshooting

- **First run is slow to start:** Palito compiles its GPU kernels once and caches them in `~/.cache/palito/kernels`.
- **`context N tokens: M do not fit in device memory`:** Pass a smaller `--context N` or close other GPU apps.
- **`libcuda.so.1` or `libnvrtc.so.13` not found:** Update to an NVIDIA driver supporting CUDA 13; on source builds, set `LD_LIBRARY_PATH` to your CUDA 13 `lib64` directory.
- **Shared GPU:** Use `--gpu-memory-limit 20GiB` (or `--gpu-memory-limit-bytes`) to cap Palito's VRAM usage.
- **`error: port 8080 is already in use`:** Another server holds the port; pick another with `--port`.
