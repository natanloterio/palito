<p align="center">
  <img src="docs/palito_logo.png" alt="Palito: fast local LLM inference" width="640">
</p>

Fast local LLM inference for GGUF models, written in Rust, with its own CUDA kernels.

- **Fast:** on an RTX 5090 with Qwen3-30B-A3B (Q4_K_M), the first token of a 2.9k-token prompt arrives in ~0.16 s (llama.cpp: 0.18 s), and decoding runs at ~300–350 tok/s.
- **Uses the model's full context:** `chat` opens with the model's trained context (Qwen3-30B-A3B: 40,960 tokens) and replies run until the model finishes.
- **One binary:** no Python, no server, no llama.cpp. The Linux bundle ships the CUDA compiler it needs.

## Requirements

- Linux x86_64.
- For the GPU: an NVIDIA GPU with a driver that supports CUDA 13. Ampere (RTX 30xx, A100) or newer gets the fast path; older GPUs work, but more slowly.
- Enough GPU memory for the model: Qwen3-30B-A3B Q4_K_M needs ~18 GB for the weights, plus ~0.2 MB per context token.

You can also run without a GPU on the CPU backend (small models only, see below).

## Install

### From a release bundle

Download `palito-<version>-linux-x86_64.tar.gz` from the [releases page](https://github.com/natanloterio/palito/releases), then:

```sh
tar xzf palito-<version>-linux-x86_64.tar.gz
mkdir -p ~/.local/bin
ln -sf "$PWD/palito-<version>-linux-x86_64/palito" ~/.local/bin/palito
palito --version
```

The `palito` launcher in the bundle uses the GPU by default and keeps compiled kernels in `~/.cache/palito/kernels`.

### From source

You need Rust 1.85 or newer.

```sh
# GPU build, packaged and installed as ~/.local/bin/palito.
# The script needs libnvrtc.so.13; point NVRTC_LIB_DIR at a CUDA 13 lib folder.
NVRTC_LIB_DIR=/usr/local/cuda-13/lib64 scripts/package-linux.sh --install

# CPU-only build
cargo build --release
./target/release/palito --help
```

## Get a model

Any supported GGUF file works. For example, with the Hugging Face CLI (`pip install -U huggingface_hub`):

```sh
# Qwen3-30B-A3B (MoE, 18 GB): the model Palito is tuned for
hf download Qwen/Qwen3-30B-A3B-GGUF Qwen3-30B-A3B-Q4_K_M.gguf --local-dir models

# SmolLM2-135M (105 MB): tiny, good for a quick test, runs on CPU
hf download bartowski/SmolLM2-135M-Instruct-GGUF SmolLM2-135M-Instruct-Q4_K_M.gguf --local-dir models
```

## Use

### Chat

```sh
palito chat --model models/Qwen3-30B-A3B-Q4_K_M.gguf
```

Type a message and press Enter. The reply appears token by token as it is generated. `/reset` clears the conversation and `/exit` quits.

Useful options:

| Option | What it does |
|---|---|
| `--no-thinking` | Turns off Qwen3's reasoning, for short, direct answers |
| `--system "You are..."` | Sets a system message |
| `--stats` | Prints tokens, time to first token and tok/s after each reply |
| `--tokens N` | Caps each reply at N tokens (default: until the model stops or the context is full) |
| `--context N` | Uses N context tokens instead of the model's full context |
| `--no-stream` | Prints each reply only once it is complete |

Set `PALITO_MODEL` once to skip `--model`:

```sh
export PALITO_MODEL=$PWD/models/Qwen3-30B-A3B-Q4_K_M.gguf
palito chat
```

### Send a prompt from a file or a script

`chat` reads from standard input, so you can pipe in a prompt, even a very long one:

```sh
echo "Explain photosynthesis in three sentences." | palito chat --no-thinking
palito chat --no-thinking < long-document-and-question.txt
```

### Plain text completion

`generate` continues a text without a chat template. `--stream` prints the text as it is generated; without it you get one JSON object (text, token IDs, timings) at the end:

```sh
palito generate --model models/SmolLM2-135M-Instruct-Q4_K_M.gguf \
  --prompt "The capital of France is" --tokens 16 --stream

palito generate --model models/SmolLM2-135M-Instruct-Q4_K_M.gguf \
  --prompt "The capital of France is" --tokens 16 | jq -r .text
```

### CPU only

```sh
palito chat --backend cpu --model models/SmolLM2-135M-Instruct-Q4_K_M.gguf --tokens 300
```

## Use Palito from your own program

Palito is also a library, `libpalito.so`, with a C API in [`include/palito.h`](include/palito.h). Your program loads it and calls it directly, in the same process: no server, no subprocess. Any language that can call C can use it.

The release bundle contains `lib/libpalito.so`, `include/palito.h` and two examples. Keep the bundle's `lib/` on the library path, since the GPU kernel compiler loads from there.

**C** ([`examples/c/chat.c`](examples/c/chat.c)):

```c
palito_engine *engine = palito_open("model.gguf", NULL, &error);   // NULL: default options
palito_message messages[] = {{"user", "Hello!"}};
palito_result result;
palito_chat(engine, messages, 1, NULL, print_text, NULL, &result, &error);
// print_text(text, length, user_data) receives the reply as it is generated;
// result.text holds the full reply, result.tokens_per_second the speed.
palito_result_free(&result);
palito_close(engine);
```

```sh
cc examples/c/chat.c -Iinclude -Llib -lpalito -o chat
LD_LIBRARY_PATH=lib ./chat models/Qwen3-30B-A3B-Q4_K_M.gguf "Hello!"
```

**Python** ([`examples/python/palito.py`](examples/python/palito.py), standard library only):

```python
from palito import Palito

with Palito("models/Qwen3-30B-A3B-Q4_K_M.gguf") as llm:
    reply = llm.chat([{"role": "user", "content": "Hello!"}],
                     on_text=lambda text: print(text, end="", flush=True))
    print(reply["finish_reason"], reply["tokens_per_second"])
```

Run it with `LD_LIBRARY_PATH=lib PALITO_LIBRARY=lib/libpalito.so`.

How it works:
- Open a model once, then call `palito_chat` or `palito_complete` as many times as you like. The defaults are the same as `palito chat`: full context, fastest GPU profile.
- Each call is stateless: pass the whole conversation every time.
- The callback gets the text as it is generated; return nonzero to stop.
- Errors come back as `-1` plus a message.
- Use one engine per thread, or guard it with a lock.

Rust programs can use the crate directly: `palito::engine::Engine`.

## Supported models

Tested on the GPU with the fast path: **Qwen3** (dense and MoE, e.g. Qwen3-0.6B and Qwen3-30B-A3B), **SmolLM2 / Llama**, **Gemma 3**.

Also supported, using the slower reference path, which Palito picks automatically: Llama 3.2, Qwen2.5, Qwen3.5, Gemma 2, Phi-3 (4K and 128K), StarCoder2, GPT-2, Bloom, Pythia, OLMo, OLMo 2 and OLMoE.

The CPU backend runs Llama-family models such as SmolLM2. See [docs/technical-reference.md](docs/technical-reference.md) for the exact list of tested files and quantizations.

## Speed

Qwen3-30B-A3B Q4_K_M on an RTX 5090:

| Prompt length | Time to first token | Decode speed |
|---|---|---|
| 0.5k tokens | 0.04 s | ~340 tok/s |
| 2.9k tokens | 0.16 s | ~300 tok/s |
| 10k tokens | 0.75 s | 211 tok/s |
| 40k tokens | 6.4 s | 103 tok/s |

## Troubleshooting

- **The first run is slow to start.** Palito compiles its GPU kernels the first time and caches them. Later runs start faster.
- **`context N tokens: M do not fit in device memory`.** The full context did not fit in free GPU memory, so Palito used a smaller one. Close other GPU programs, or pass `--context` with a smaller value.
- **`reply stopped: the N-token context is full`.** The conversation filled the context. Use `/reset`, or start with a shorter prompt.
- **`libcuda.so.1` or `libnvrtc.so.13` not found.** Install or update the NVIDIA driver (CUDA 13 support). From a source build, put the CUDA 13 libraries on `LD_LIBRARY_PATH`.
- **The GPU is shared with something else.** `--gpu-memory-limit-bytes N` caps how much GPU memory Palito uses.

## More

- [Technical reference](docs/technical-reference.md): precision profiles (`--cuda-matmul`), validation, benchmarks against llama.cpp, tests.
- [Architecture](docs/architecture.md)
- [Performance report, in Portuguese](docs/relatorio-desempenho-decode-cuda.md)

License: MIT.
