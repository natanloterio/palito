<p align="center">
  <img src="docs/palito_logo.png" alt="Palito: fast local LLM inference" width="480">
</p>

# Palito

Fast local LLM inference in a single Rust binary. Runs GGUF models and its own GPU-native `.palito` format on CUDA, Vulkan, Metal or CPU, with an OpenAI-compatible server. No Python, no `llama.cpp`.

## Download

### Linux (NVIDIA GPU or CPU)

```sh
curl -L https://github.com/natanloterio/palito/releases/download/v0.16.1/palito-0.16.1-linux-x86_64.tar.gz | tar xz
mkdir -p ~/.local/bin
ln -sf "$PWD/palito-0.16.1-linux-x86_64/palito" ~/.local/bin/palito
```

The bundle carries the CUDA 13 runtime compiler; only the NVIDIA driver must be installed. Without a GPU it runs on the CPU.

### macOS (Apple Silicon)

```sh
curl -L https://github.com/natanloterio/palito/releases/download/v0.16.1/palito-0.16.1-macos-arm64.tar.gz | tar xz
mkdir -p ~/.local/bin
ln -sf "$PWD/palito-0.16.1-macos-arm64/palito" ~/.local/bin/palito
```

If `palito` is not found, add `export PATH="$HOME/.local/bin:$PATH"` to `~/.zshrc`. GGUF models run on the GPU with `--backend metal`; `.palito` models run on the CPU (their GPU kernels are CUDA-only for now).

### Windows (x86_64)

Download [palito-0.16.1-windows-x86_64.zip](https://github.com/natanloterio/palito/releases/download/v0.16.1/palito-0.16.1-windows-x86_64.zip), extract it and run `palito.exe` from PowerShell in that folder (or add the folder to `PATH`). GGUF models run on the GPU with `--backend vulkan` (AMD, Intel and NVIDIA drivers include Vulkan); `.palito` models run on the CPU.

The Linux bundle is for x86_64 only: on a Mac or on Windows, use the bundles above. Other platforms and GPUs: install with Cargo (`cargo install palito --features vulkan`), see [building from source](docs/building.md).

### Libraries

[Python](bindings/README.md) · [Node.js](bindings/README.md) · [Java](bindings/README.md) · [C#](bindings/README.md) · [Go](bindings/README.md) · [Rust](https://crates.io/crates/palito) · [C](examples/c/chat.c)

## Get started

Download a ready-made `.palito` model and chat with it (NVIDIA GPU):

```sh
palito pull qwen3-8b
palito chat --model qwen3-8b
```

`palito list --available` shows every model you can pull (today `qwen3-8b` and `llama-3.1-8b`), `palito list` the ones you have, and `palito rm` deletes one. Models are stored in `~/.local/share/palito/models` (set `PALITO_MODELS` to use another disk); downloads resume where they stopped and are checked against their sha256. `palito pull user/repo:file.gguf` downloads any file from Hugging Face, using the token from `hf auth login` (or `HF_TOKEN`) for gated repos. The Llama model is Built with Llama and comes with the Llama 3.1 Community License.

Or run any GGUF file:

```sh
hf download Qwen/Qwen3-30B-A3B-GGUF Qwen3-30B-A3B-Q4_K_M.gguf --local-dir models
palito chat --model models/Qwen3-30B-A3B-Q4_K_M.gguf
```

No GPU? Try a 105 MB model that runs on any CPU:

```sh
hf download bartowski/SmolLM2-135M-Instruct-GGUF SmolLM2-135M-Instruct-Q4_K_M.gguf --local-dir models
palito chat --model models/SmolLM2-135M-Instruct-Q4_K_M.gguf
```

In the chat, `/help` lists the commands and Ctrl-C stops a reply.

## OpenAI-compatible API

```sh
palito serve --model models/Qwen3-30B-A3B-Q4_K_M.gguf
```

```sh
curl http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "Why is the sky blue?"}]}'
```

Any OpenAI client works by pointing its base URL at the server:

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="unused")
reply = client.chat.completions.create(
    model="palito",
    messages=[{"role": "user", "content": "Why is the sky blue?"}],
)
print(reply.choices[0].message.content)
```

Endpoints: `/v1/chat/completions` and `/v1/completions` (both with streaming), `/v1/models` and `/health`. Tool calling, JSON schemas, stop sequences and parallel slots (`-np`) are supported. See the [server reference](docs/plt1-format.md).

## The `.palito` format

GGUF was designed for CPUs; on a GPU much of the time goes to unpacking weights. A `.palito` file stores weights already laid out for NVIDIA tensor cores, at 3.0 bits per weight, in one self-contained file with the tokenizer and an `EAGLE-3` draft head:

```sh
palito convert -i models/Qwen3-8B-Q6_K.gguf -o models/qwen3-8b.palito
palito chat --model models/qwen3-8b.palito
```

Qwen 3 8B fits in 3.7 GiB and Qwen3-30B-A3B in 11.1 GiB (down from 18.6 GiB as `Q4_K_M`). Speculative decoding (`EAGLE-3` + `SuffixDecoding`) turns on automatically and gives identical output to token-by-token decoding. See the [format guide](docs/plt1-format.md).

## Performance

Pure decode on an RTX 5090, against `llama.cpp` CUDA:

<p align="center">
  <img src="docs/charts/decode-vs-llama-cpp.svg" alt="Decode speed on RTX 5090: Qwen3-30B-A3B MoE 412 vs 340 tok/s, Llama 3.1 8B 347 vs 341, Llama 3.2 1B 1,361 vs 1,053 (Palito vs llama.cpp)" width="720">
</p>

Long-context decode on `.palito` with `--cuda-matmul int8-fast` (FP16 KV cache and Tensor Core attention, half the KV memory of F32):

<p align="center">
  <img src="docs/charts/long-context-decode.svg" alt="Long-context decode, FP16 vs F32 KV cache: Qwen3-30B-A3B 328.9 vs 246.7 tok/s at 9.3k tokens and 242.3 vs 141.0 at 22.9k; Llama 3.1 8B 253.9 vs 220.8 and 207.8 vs 135.5; Qwen 3 8B 241.5 vs 200.2 and 190.3 vs 120.0" width="720">
</p>

With speculative decoding (automatically gated above 4k context so long-context decode stays at full CUDA Graph speed), Llama 3.1 8B reaches **1,119 tok/s** on code refactoring, Qwen 3 8B reaches **763 tok/s**, and Qwen3-30B-A3B reaches **590 tok/s** (`exact_parity=true`). Full tables and methodology: [performance](docs/performance.md) and [reproducible benchmarks](benchmarks/vs-llama-cpp/README.md).

## Supported models

Llama 3 / 3.1 / 3.2, Qwen 2.5 / 3 / 3.5 (dense and MoE), Gemma 2 / 3, SmolLM2, Phi-3, StarCoder2, OLMo / OLMo 2 / OLMoE, and more. See the [tested models](docs/technical-reference.md).

## Documentation

- [`.palito` format, CLI and server reference](docs/plt1-format.md)
- [Performance](docs/performance.md) and [benchmarks vs `llama.cpp`](benchmarks/vs-llama-cpp/README.md)
- [Language bindings](bindings/README.md)
- [Building from source](docs/building.md)
- [Technical reference](docs/technical-reference.md) and [architecture](docs/architecture.md)
- [Releasing](docs/releasing.md)

## Works with OpenAI-compatible apps

`palito serve` speaks the OpenAI API, so apps and libraries that let you set a custom OpenAI base URL can use it. Set the base URL to `http://127.0.0.1:8080/v1` and the API key to any value (or the one given to `--api-key`).

Tested: the [OpenAI Python SDK](https://github.com/openai/openai-python), [LangChain](https://python.langchain.com/docs/integrations/chat/openai/) (`ChatOpenAI`) and [LiteLLM](https://github.com/BerriAI/litellm) (`openai/` models). The rest of this list offers an OpenAI-compatible provider but has not been tested with Palito yet; reports and pull requests are welcome.

Palito does not serve `/v1/embeddings`, so apps that index documents (RAG) need a separate embedding provider. Agents that rely on tool calls need a model with Qwen's `<tool_call>` format, such as Qwen 2.5 or Qwen 3.

### Chat interfaces

- [Open WebUI](https://github.com/open-webui/open-webui)
- [LibreChat](https://github.com/danny-avila/LibreChat)
- [Lobe Chat](https://github.com/lobehub/lobe-chat)
- [NextChat](https://github.com/ChatGPTNextWeb/ChatGPT-Next-Web)
- [big-AGI](https://github.com/enricoros/big-AGI)
- [AnythingLLM](https://github.com/Mintplex-Labs/anything-llm)
- [Chatbox](https://github.com/Bin-Huang/Chatbox)
- [Cherry Studio](https://github.com/kangfenmao/cherry-studio)
- [Dify](https://github.com/langgenius/dify)
- [Perplexica](https://github.com/ItzCrazyKns/Perplexica)

### Code editors

- [Continue](https://github.com/continuedev/continue)
- [Cline](https://github.com/cline/cline)
- [Void](https://github.com/voideditor/void)
- [gptel](https://github.com/karthink/gptel) (Emacs)
- [Open Interpreter](https://github.com/OpenInterpreter/open-interpreter)

### Libraries and frameworks

- [LangChain](https://python.langchain.com/docs/integrations/chat/openai/) and [LangChain.js](https://js.langchain.com/docs/integrations/chat/openai/)
- [LlamaIndex](https://docs.llamaindex.ai/en/stable/examples/llm/openai_like/) (`OpenAILike`)
- [LiteLLM](https://github.com/BerriAI/litellm)
- [Haystack](https://github.com/deepset-ai/haystack)
- [Semantic Kernel](https://github.com/microsoft/semantic-kernel)
- [Spring AI](https://github.com/spring-projects/spring-ai)
- [LangChain4j](https://github.com/langchain4j/langchain4j)
- [crewAI](https://github.com/crewAIInc/crewAI)

### Terminal

- [aichat](https://github.com/sigoden/aichat)
- [llm](https://github.com/simonw/llm)

### Observability

- [Langfuse](https://github.com/langfuse/langfuse)
- [OpenLIT](https://github.com/openlit/openlit)

## License

MIT. See [LICENSE](LICENSE).
