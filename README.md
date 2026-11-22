<p align="center">
  <img src="docs/palito_logo.png" alt="Palito: fast local LLM inference" width="640">
</p>

Fast local LLM inference for GGUF models, written in Rust, with its own CUDA kernels.

- **Fast:** on an RTX 5090 with Qwen3-30B-A3B (Q4_K_M), the first token of a 2.9k-token prompt arrives in ~0.16 s (llama.cpp: 0.18 s), and decoding runs at ~300–350 tok/s.
- **Uses the model's full context:** `chat` opens with the model's trained context (Qwen3-30B-A3B: 40,960 tokens) and replies run until the model finishes.
- **One binary:** chat in the terminal, an OpenAI-compatible server (`palito serve`), or a C library for your own program. No Python and no llama.cpp; the Linux bundle ships the CUDA compiler it needs.

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

You need Rust 1.87 or newer.

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
| `--temperature T` | Samples instead of always picking the most likely token (default 0: greedy, the same reply every time) |
| `--top-p P`, `--top-k K`, `--seed S` | Narrow the sampling to the most likely tokens; a seed makes it repeatable |
| `--stop TEXT` | Ends the reply at TEXT, which is not printed (repeat for several) |
| `--no-prefix-cache` | Re-reads the whole conversation every turn instead of reusing the previous turn's work (GPU) |

For Qwen3, its authors suggest `--temperature 0.6 --top-p 0.95 --top-k 20` with reasoning on, and `--temperature 0.7 --top-p 0.8 --top-k 20` with `--no-thinking`. Greedy decoding works well for short answers, but long reasoning can start repeating itself.

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

## Run as a server (OpenAI-compatible API)

```sh
palito serve --model models/Qwen3-30B-A3B-Q4_K_M.gguf
# Serving the OpenAI API at http://127.0.0.1:8080/v1
```

Any OpenAI client works: point it at `http://127.0.0.1:8080/v1`. That includes the official SDKs, Open WebUI and editor plugins. The API key can be any text unless you set one.

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="unused")
for chunk in client.chat.completions.create(
        model="palito", stream=True,
        messages=[{"role": "user", "content": "Hello!"}]):
    print(chunk.choices[0].delta.content or "", end="", flush=True)
```

```sh
curl http://127.0.0.1:8080/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "Hello!"}]}'
```

- **Endpoints:** `POST /v1/chat/completions`, `POST /v1/completions` (both support `"stream": true`), `GET /v1/models`, `GET /health`.
- **Qwen3 reasoning:** the `<think>` part comes back separately as `reasoning_content` (as with llama.cpp and vLLM), and `content` holds only the answer. Send `"reasoning_format": "none"` to keep it inside `content`, or `"chat_template_kwargs": {"enable_thinking": false}` to turn reasoning off.
- **Sampling:** `temperature`, `top_p`, `top_k` and `seed` work as usual. When a request leaves them out, the server uses its own defaults: greedy, unless you start it with e.g. `--temperature 0.6 --top-p 0.95 --top-k 20`. Greedy runs about 20% faster (~370 vs ~300 tok/s on Qwen3-30B).
- **Stop sequences:** `stop` (a string or up to 16 strings) ends the reply; the stop text is not returned. It applies to everything the model writes, reasoning included, so with reasoning on avoid stops like `"\n"`.
- **JSON output:** `"response_format": {"type": "json_object"}` or `{"type": "json_schema", "json_schema": {"name": "...", "schema": {...}}}` makes the reply JSON that always parses and matches the schema (constrained decoding, also with greedy). Reasoning is turned off for these requests. A schema keyword Palito cannot enforce (e.g. `uniqueItems`, `not`, `if`) returns 400 instead of being ignored. Qwen3-30B decodes at ~327 tok/s greedy (~300 sampled) with a schema.
- **Not supported:** `n` > 1 is rejected.
- **One request at a time:** the model runs one request at a time, and others wait in a queue (`--queue`, default 16; beyond that the server returns 503). If a client disconnects, its generation stops.
- **Local by default:** the server only accepts connections from this machine. To serve your network, use `--host 0.0.0.0 --api-key YOUR_KEY` (or set `PALITO_API_KEY`). Clients then send `Authorization: Bearer YOUR_KEY`.
- **Conversations are cheap to continue:** on the GPU, a request that starts with the previous request's prompt (a chat resent with one more turn) skips most of the shared part. Replies are the same as without it. `--no-prefix-cache` turns it off.
- **Other options:** `--port`, `--context`, `--backend cpu`, `--gpu-memory-limit-bytes`, `--model-name`.

## Use Palito from your own program

Palito is also a library, `libpalito.so`, with a C API in [`include/palito.h`](include/palito.h). Your program loads it and calls it directly, in the same process: no server, no subprocess. Ready-made bindings for **Python, Node.js, Java, C# and Go** are in [`bindings/`](bindings/README.md), and in the release bundle. They run at the same speed as `palito chat`.

Every binding finds the library through the installed `palito` command, so once Palito is installed there is nothing to configure. Elsewhere, set `PALITO_LIBRARY=/path/to/libpalito.so`.

| Language | Install | Hello |
|---|---|---|
| Python | `pip install ./bindings/python` | `palito.Palito("model.gguf").chat([{"role": "user", "content": "Hi"}])` |
| Node.js | `npm install ./bindings/node` | `await (await Palito.open("model.gguf")).chat([{ role: "user", content: "Hi" }])` |
| Java 22+ | `mvn -f bindings/java/pom.xml install` | `Palito.open(Path.of("model.gguf"), new Options()).chat(List.of(Message.user("Hi")), new Generation())` |
| C# | `dotnet add reference bindings/csharp/Palito/Palito.csproj` | `PalitoEngine.Open("model.gguf").Chat(new[] { Message.User("Hi") })` |
| Go | `go get github.com/natanloterio/palito/bindings/go` | `engine.Chat(ctx, []palito.Message{{Role: "user", Content: "Hi"}}, nil)` |

Each binding streams the reply as it is generated, can stop early, and takes the same settings: token cap, reasoning, sampling and stop sequences. See its README for an example.

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
cc examples/c/chat.c -Iinclude -Llib -lpalito -Wl,-rpath,"$PWD/lib" -o chat
./chat models/Qwen3-30B-A3B-Q4_K_M.gguf "Hello!"
```

How it works:
- Open a model once, then call `palito_chat` or `palito_complete` as many times as you like. The defaults are the same as `palito chat`: full context, fastest GPU profile.
- Each call is stateless: pass the whole conversation every time.
- The callback gets the text as it is generated; return nonzero to stop.
- `palito_generation` sets the token cap, reasoning, sampling (`temperature`, `top_p`, `top_k`, `seed`) and stop sequences. Fill it with `palito_generation_init` first.
- `palito_chat_json` and `palito_complete_json` take an extra JSON Schema string; the output is JSON matching it. The struct layouts never change, so programs built against an older header keep working.
- `libpalito.so` finds the NVRTC bundled next to it, so `LD_LIBRARY_PATH` is not needed.
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

Continuing a conversation is faster: with ~9.7k tokens of history, the next turn's first token comes after 0.18 s instead of 0.77 s, because the history is not read again.

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
