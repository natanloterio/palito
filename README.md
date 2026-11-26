<p align="center">
  <img src="docs/palito_logo.png" alt="Palito: fast local LLM inference" width="640">
</p>

**Fast local LLM inference in a single Rust binary** — custom CUDA, Vulkan, and Metal kernels, GPU-native `.palito` (`PLT1`, `3.0 bpw`) quantization, and built-in `EAGLE-3` + `SuffixDecoding` speculative decoding: **1,017–1,045 tok/s on Llama 3.1 8B**, **1,361 tok/s on Llama 3.2 1B**, **726 tok/s on Qwen 3 8B**, and **557 tok/s on Qwen3-30B-A3B MoE** (up to **1.7× faster than `llama.cpp`** with up to 40% less GPU memory) — no Python or `llama.cpp` required.

---

## Why Palito? Meet the `.palito` (`PLT1`) Model Format

Most local AI tools use the GGUF format, which was originally designed for CPUs. When you load a GGUF file on a modern NVIDIA GPU, the GPU spends a large chunk of its time unpacking data instead of generating text.

Palito introduces **`.palito` (`PLT1`)** — a single-file model format built specifically for GPUs (dense and MoE):

- **Compact, high quality (`3.0 bpw`):** Fits a complete **Qwen 3 8B** in **3.7 GiB**, **Llama 3.1 8B** in **3.5 GiB** (`PPL = 7.64`), and **Qwen3-30B-A3B MoE** (128 experts) in **11.1 GiB** (down from 18.6 GiB `Q4_K_M`).
- **One self-contained file, zero flags:** Tokenizer, GPU-native weights, and the built-in `EAGLE-3` predictor live in one `.palito` file — just run `palito chat --model qwen3-8b.palito`.
- **GPU-ready on disk:** Weights are pre-arranged for NVIDIA Tensor Cores (`mma.m16n8k16`) with zero-SMEM warp-register lookup tables so the GPU streams and computes directly with minimal overhead.
- **Generates multiple tokens at once (100% identical answers):** Bundles `EAGLE-3` and zero-weight `SuffixDecoding` that draft up to 7 tokens ahead and verify them in a single pass — giving **1.3×–2.6× speedups** over single-token decode with **zero quality loss** (`exact_parity=true`).
- **Works with GGUF too:** Already have GGUF models? Palito runs them out of the box with its own custom CUDA, Vulkan, and Metal kernels.

---

## Speed vs `llama.cpp` (NVIDIA RTX 5090)

| Model | File Size | Task | Palito | `llama.cpp` | Speedup |
| :--- | :---: | :--- | :---: | :---: | :---: |
| **Llama 3.1 8B** (`llama-3.1-8b.palito`, `3.0 bpw`) | **3.5 GiB** | General / Coding / Refactor | **417 / 445 / 1,017 tok/s** | 394 / 390 / 608 tok/s (`IQ3_XXS + EAGLE-3`) | **1.06× / 1.14× / 1.67×** |
| **Qwen 3 8B** (`qwen3-8b.palito`, `3.0 bpw`) | **3.7 GiB** | General / Coding / Refactor | **296 / 379 / 726 tok/s** | 306 / 310 / 459 tok/s (`Q3_K_M + EAGLE-3`) | **0.97× / 1.22× / 1.58×** |
| **Qwen3-30B-A3B MoE** (`qwen3-30b-a3b.palito`, `3.0 bpw`) | **11.1 GiB** | Single-stream / Refactor | **349 / 557 tok/s** | 340 tok/s (`Q4_K_M`, 18.6 GiB) | **1.64× refactor, 40% smaller** |
| **Llama 3.2 1B** (`QTIP3`) | **0.9 GiB** | Single-stream decode | **1,361 tok/s** | 1,053 tok/s (`IQ3_XXS`) | **1.29× faster** |

*All speculative runs are verified 100% bit-exact (`exact_parity=true`) against standard token-by-token generation and checked for coherent, non-degenerate output. See [reproducible benchmarks](benchmarks/vs-llama-cpp/README.md).*

---

## Quick Start (3 Simple Steps)

### 1. Install Palito

Download the latest Linux release from the [Releases page](https://github.com/natanloterio/palito/releases):

```sh
tar xzf palito-0.15.0-linux-x86_64.tar.gz
mkdir -p ~/.local/bin
ln -sf "$PWD/palito-0.15.0-linux-x86_64/palito" ~/.local/bin/palito
palito --version
```

*(Uses your NVIDIA GPU with CUDA 13 drivers automatically; CPU mode also works for small models.)*

### 2. Download a Model

```sh
# Fast 30B MoE model (requires Hugging Face CLI: pip install huggingface_hub)
hf download Qwen/Qwen3-30B-A3B-GGUF Qwen3-30B-A3B-Q4_K_M.gguf --local-dir models

# Or a tiny 105 MB model that runs even on a CPU
hf download bartowski/SmolLM2-135M-Instruct-GGUF SmolLM2-135M-Instruct-Q4_K_M.gguf --local-dir models
```

### 3. Chat or Start a Local API Server

**Chat in your terminal:**

```sh
palito chat --model models/Qwen3-30B-A3B-Q4_K_M.gguf --stats
```

**Or start an OpenAI-compatible server** (works with Open WebUI, Cursor, Continue, and any OpenAI SDK):

```sh
palito serve --model models/Qwen3-30B-A3B-Q4_K_M.gguf
# OpenAI API ready at http://127.0.0.1:8080/v1
```

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="unused")
reply = client.chat.completions.create(
    model="palito",
    messages=[{"role": "user", "content": "Hello!"}],
)
print(reply.choices[0].message.content)
```

---

## How to Run Single-File `.palito` Models

A single `.palito` file bundles everything — tokenizer, GPU-native model weights (dense or MoE), and the built-in `EAGLE-3` predictor — so you just pass it directly to `palito chat` or `palito serve` with **zero extra flags** (CUDA and speculative decoding turn on automatically):

```sh
# Chat directly with a single .palito file (Qwen 3 8B, Llama 3.1 8B, or Qwen3-30B-A3B MoE)
palito chat --model models/qwen3-8b.palito --stats

# Or serve it over an OpenAI-compatible API
palito serve --model models/qwen3-30b-a3b.palito
```

To convert any GGUF model into a single `.palito` file in one command (with automatic CUDA Hessian calibration and CUDA Viterbi quantization):

```sh
palito convert --input models/Qwen3-8B-Q6_K.gguf --output models/qwen3-8b.palito
```

See the **[`.palito` (`PLT1`) Format Guide](docs/plt1-format.md)** for optional `EAGLE-3` bundling and advanced options.

---

## Use Palito in Your Language

Palito ships ready-made bindings that run at full native speed in-process (no server needed):

| Language | Install | One-Line Example |
| :--- | :--- | :--- |
| **Python** | `pip install palito` | `palito.Palito("model.gguf").chat([{"role": "user", "content": "Hi"}])` |
| **Node.js / TS** | `npm install palito` | `await (await Palito.open("model.gguf")).chat([{ role: "user", content: "Hi" }])` |
| **Java 22+** | Maven (`io.github.natanloterio:palito`) | `Palito.open(Path.of("model.gguf"), new Options()).chat(...)` |
| **C# / .NET** | `dotnet add package Palito` | `PalitoEngine.Open("model.gguf").Chat(new[] { Message.User("Hi") })` |
| **Go** | `go get github.com/natanloterio/palito/bindings/go` | `engine.Chat(ctx, []palito.Message{{Role: "user", Content: "Hi"}}, nil)` |
| **Rust / C** | `cargo add palito` / `libpalito.so` | See [`examples/c/chat.c`](examples/c/chat.c) |

---

## Supported Models

Works out of the box with **Llama 3 / 3.1 / 3.2**, **Qwen 2.5 / 3 / 3.5** (dense and MoE), **Gemma 2 / 3**, **SmolLM2**, **Phi-3**, **StarCoder2**, **OLMo / OLMo 2 / OLMoE**, and more.

---

## Deep-Dive Documentation

- **[`.palito` (`PLT1`) Format Guide](docs/plt1-format.md)** — How `PLT1` works under the hood, converting GGUF → `.palito`, building from source, full CLI/server flags, C API, and troubleshooting.
- **[Benchmarks vs `llama.cpp`](benchmarks/vs-llama-cpp/README.md)** — Reproducible benchmark scripts and raw measurements.
- **[Language Bindings](bindings/README.md)** — Guides for Python, Node.js, Java, C#, and Go.
- **[Technical Reference & Architecture](docs/technical-reference.md)** — Kernel architecture, tested models, and long-context numbers.
- **[Architecture](docs/architecture.md)** | **[Releasing](docs/releasing.md)** | **[Portuguese CUDA report](docs/relatorio-desempenho-decode-cuda.md)**

License: MIT (see [LICENSE](LICENSE)).
