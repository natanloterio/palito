# Performance vs `llama.cpp` (NVIDIA RTX 5090)

## Pure Single-Token Decode (No Speculative Decoding)

Measured token-by-token (`PALITO_NO_EAGLE3=1`, `f32-tiled`, CUDA Graph, `context=2048`, `tokens=128`, median of 3 runs) against `llama.cpp` CUDA (`-fa on`, no speculation):

| Model | Format / Size | Palito (pure decode) | `llama.cpp` (pure decode) | Ratio |
| :--- | :---: | :---: | :---: | :---: |
| **Qwen3-30B-A3B MoE** | `.palito` (`3.0 bpw`, **11.1 GiB**) | **412–414 tok/s** | 340 tok/s (`Q4_K_M`, 18.6 GiB) | **1.21× faster, 40% smaller** |
| **Qwen3-30B-A3B MoE** | `Q4_K_M` GGUF (18.6 GiB, same file) | **374–380 tok/s** | 340 tok/s (`Q4_K_M`, 18.6 GiB) | **1.10–1.12× faster** |
| **Llama 3.1 8B** | `.palito` (`3.0 bpw`, **3.5 GiB**) | **347–355 tok/s** | 341 tok/s (`IQ3_XXS`) | **1.02–1.04×** |
| **Qwen 3 8B** | `.palito` (`3.0 bpw`, **3.7 GiB**) | **324–329 tok/s** | — | — |
| **Llama 3.2 1B** | `QTIP3` (**0.9 GiB**) | **1,361 tok/s** | 1,053 tok/s (`IQ3_XXS`) | **1.29×** |

### Long-Context Pure Decode (`.palito` `3.0 bpw`, FP16 KV Cache + Tensor Core Attention)

With `--cuda-matmul int8-fast` (default in `palito chat`/`serve`, FP16 KV cache + `attention_decode_mma_decode_f16`; automatic EAGLE-3 is gated above 4k context so long-context decode stays at full CUDA Graph speed with identical greedy output):

| Model (`.palito`, `3.0 bpw`) | Prompt Length | KV F32 (`f32-tiled`) | KV FP16 (`int8-fast`) | Speedup & KV VRAM |
| :--- | :---: | :---: | :---: | :---: |
| **Qwen3-30B-A3B MoE** | 9,274 / 22,939 tok | 246.7 / 141.0 tok/s | **328.9 / 242.3 tok/s** | **+33% / +72%**, 50% less KV VRAM (`1.12 / 2.62 GiB`) |
| **Llama 3.1 8B** | 8,998 / 22,156 tok | 220.8 / 135.5 tok/s | **253.9 / 207.8 tok/s** | **+15% / +53%**, 50% less KV VRAM (`1.50 / 3.50 GiB`) |
| **Qwen 3 8B** | 9,274 / 22,939 tok | 200.2 / 120.0 tok/s | **241.5 / 190.3 tok/s** | **+21% / +59%**, 50% less KV VRAM (`1.69 / 3.94 GiB`) |

## With Speculative Decoding (`EAGLE-3` + `SuffixDecoding`)

Both Palito and `llama.cpp` support speculative decoding (`EAGLE-3` and n-gram/lookup). In the table below (`f32-tiled`, `ctx=2048`), Palito's Refactor column uses `EAGLE-3 + SuffixDecoding` while `llama.cpp` was measured with `EAGLE-3` only (without `--spec-type ngram-simple` / `ngram-mod`), and Qwen3-30B-A3B's `llama.cpp` baseline is non-speculative `Q4_K_M`:

| Model | File Size | Task | Palito (`EAGLE-3` + `Suffix`) | `llama.cpp` | Notes |
| :--- | :---: | :--- | :---: | :---: | :--- |
| **Llama 3.1 8B** (`3.0 bpw`) | **3.5 GiB** | General / Coding / Refactor | **431 / 473 / 1,119 tok/s** | 394 / 390 / 608 tok/s (`IQ3_XXS + EAGLE-3`) | **1.09× / 1.21×** (`EAGLE-3` vs `EAGLE-3`); Refactor adds `SuffixDecoding` |
| **Qwen 3 8B** (`3.0 bpw`) | **3.7 GiB** | General / Coding / Refactor | **297 / 368 / 763 tok/s** | 306 / 310 / 459 tok/s (`Q3_K_M + EAGLE-3`) | **0.97× / 1.19×** (`EAGLE-3` vs `EAGLE-3`); Refactor adds `SuffixDecoding` |
| **Qwen3-30B-A3B MoE** (`3.0 bpw`) | **11.1 GiB** | Refactor (`SuffixDecoding`) | **590 tok/s** | 340 tok/s (`Q4_K_M`, no spec) | Speculative vs non-speculative baseline |

*All speculative runs are verified 100% bit-exact (`exact_parity=true`) against standard token-by-token generation and checked for coherent, non-degenerate output. See [reproducible benchmarks](../benchmarks/vs-llama-cpp/README.md) for identical-GGUF HTTP streaming benchmarks (including long context up to 40k tokens).*
