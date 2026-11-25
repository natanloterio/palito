<p align="center">
  <img src="docs/palito_logo.png" alt="Palito: fast local LLM inference" width="640">
</p>

**Run AI models on your own computer at over 1,000 tokens per second** — nearly **3× faster than `llama.cpp`**, using half the GPU memory, in a single fast Rust binary with no Python or `llama.cpp` required.

---

## Why Palito? Meet the `.palito` (`PLT1`) Model Format

Most local AI tools use the GGUF format, which was originally designed for CPUs. When you load a GGUF file on a modern NVIDIA GPU, the GPU spends a large chunk of its time unpacking data instead of generating text.

Palito introduces **`.palito` (`PLT1`)** — a model format built specifically for GPUs:

- **Half the size:** Shrinks an 8B model's layers down to **1.75 GB** (~2 bits per weight) so it streams off GPU memory much faster.
- **GPU-ready on disk:** Weights are pre-arranged for NVIDIA Tensor Cores so the GPU reads and computes directly with minimal overhead.
- **Generates multiple tokens at once (100% identical answers):** Bundles a tiny built-in predictor (`EAGLE-3`) and smart text lookup (`SuffixDecoding`) that draft several tokens ahead and verify them in a single pass — giving **2×–3× speedups** with **zero quality loss**.
- **Works with GGUF too:** Already have GGUF models? Palito runs them out of the box with its own custom CUDA, Vulkan, and Metal kernels.

---

## Speed vs `llama.cpp` (NVIDIA RTX 5090)

| Model | Task | Palito | `llama.cpp` | Speedup |
| :--- | :--- | :---: | :---: | :---: |
| **Llama 3.1 8B** (`.palito` format) | Coding | **1,002.6 tok/s** | 341 tok/s (`IQ3_XXS`) | **2.94× faster** |
| **Llama 3.1 8B** (`.palito` format) | Code refactoring | **896.6 tok/s** | 341 tok/s (`IQ3_XXS`) | **2.63× faster** |
| **Llama 3.1 8B** (`.palito` format) | General chat & writing | **571–757 tok/s** | 341 tok/s (`IQ3_XXS`) | **1.7×–2.2× faster** |
| **Llama 3.2 1B** (`QTIP3`) | Single-stream decode | **1,361 tok/s** | 1,053 tok/s (`IQ3_XXS`) | **1.29× faster** |
| **Qwen3-30B-A3B** (`Q4_K_M` MoE) | Long prompt (2.9k tok) | **164 ms** first token / **362 tok/s** | 214 ms / 340 tok/s | **1.30× faster TTFT** |

*All speculative runs are verified 100% bit-exact (`exact_parity=true`) against standard token-by-token generation. See [reproducible benchmarks](benchmarks/vs-llama-cpp/README.md).*

---

## Quick Start (3 Simple Steps)

### 1. Install Palito

Download the latest Linux release from the [Releases page](https://github.com/natanloterio/palito/releases):

```sh
tar xzf palito-0.13.0-linux-x86_64.tar.gz
mkdir -p ~/.local/bin
ln -sf "$PWD/palito-0.13.0-linux-x86_64/palito" ~/.local/bin/palito
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

## How to Run Models at 1,000+ tok/s (`.palito` Format)

To hit **1,000+ tok/s** on Llama 3.1 8B, you run a `.palito` (`PLT1`) model together with its compact `EAGLE-3` predictor:

1. **Convert your model once** into a 1.75 GB `.palito` file (`gguf-to-plt1`).
2. **Pack the `EAGLE-3` predictor** (`239 MB`) into `.palito` format (`pack_eagle3_plt1.py`).
3. **Run in hybrid mode** (`EAGLE-3` + `SuffixDecoding`) for up to **1,002+ tok/s**:

```sh
./target/release/examples/eagle3-spec-bench \
  --model models/Llama-3.1-8B-Instruct-Q6_K.gguf \
  --plt1 models/plt1-hyb-q5-2bpw.palito \
  --eagle3 models/eagle3/eagle3-q4_k.palito \
  --tokens 128 --draft-len 7 --hybrid
```

👉 **Want the exact build & conversion commands?** See the **[`.palito` (`PLT1`) Format & 1,000+ tok/s Guide](docs/plt1-format.md)**.

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

- **[`.palito` (`PLT1`) Format & 1,000+ tok/s Guide](docs/plt1-format.md)** — How `PLT1` works under the hood, converting GGUF → `.palito`, building from source, full CLI/server flags, C API, and troubleshooting.
- **[Benchmarks vs `llama.cpp`](benchmarks/vs-llama-cpp/README.md)** — Reproducible benchmark scripts and raw measurements.
- **[Language Bindings](bindings/README.md)** — Guides for Python, Node.js, Java, C#, and Go.
- **[Technical Reference & Architecture](docs/technical-reference.md)** — Kernel architecture, tested models, and long-context numbers.
- **[Architecture](docs/architecture.md)** | **[Releasing](docs/releasing.md)** | **[Portuguese CUDA report](docs/relatorio-desempenho-decode-cuda.md)**

License: MIT (see [LICENSE](LICENSE)).
