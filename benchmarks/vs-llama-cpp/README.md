# Palito 0.9.0 vs llama.cpp (RTX 5090)

Reproducible end-to-end HTTP streaming benchmarks comparing `palito serve` (`0.9.0`, default `int8-fast` profile) against `llama-server` (`1.0.0-dev`, build 12434, commit `6b01a4767`, `-ngl 99 -np 1 --jinja --reasoning-format none -fa on`).

## Methodology

- **Identical measurement boundary:** both servers are queried through the same OpenAI streaming `/v1/chat/completions` client in [scripts/bench_long_context.py](../../scripts/bench_long_context.py) (`--palito-mode serve`), with greedy decoding (`temperature: 0`), fixed reply budget (`256` tokens), and prompt caching disabled (`--no-prefix-cache` / `cache_prompt: false`).
- **Metrics:** client-side wall-clock **TTFT** (request send to first non-empty content/reasoning SSE chunk) and **decode tok/s** (`(completion_tokens - 1) / (t_done - t_first_token)`).
- **Samples:** 1 discarded warm-up + 3 measured runs per prompt size; tables report medians. All raw samples and environment metadata are committed alongside this file.
- **Hardware:** NVIDIA GeForce RTX 5090 (32 GB VRAM, driver `615.71.09`, CUDA 13).

## Qwen3-30B-A3B Q4_K_M (18 GB MoE, context 40,960)

Raw samples: [qwen3-30b-a3b.json](./qwen3-30b-a3b.json)

| Prompt tokens | Palito 0.9.0 TTFT ms | llama.cpp `fa-on` TTFT ms | Palito 0.9.0 decode tok/s | llama.cpp `fa-on` decode tok/s | llama.cpp `fa-on-q8kv` decode tok/s |
|---:|---:|---:|---:|---:|---:|
| 507 | 42 | **37** | **362.1** (+6.4%) | 340.4 | 314.5 |
| 2,907 | **164** (-23%) | 214 | **343.0** (+12.4%) | 305.2 | 274.7 |
| 10,003 | **700** (-16%) | 831 | **293.4** (+5.3%) | 278.6 | 230.4 |
| 40,001 | **5,204** (-5.3%) | 5,495 | **191.8** (+6.3%) | 180.4 | 125.0 |

## OLMoE-1B-7B Q4_K_M (4.2 GB MoE, context 4,096)

Raw samples: [olmoe-1b-7b.json](./olmoe-1b-7b.json)

| Prompt tokens | Palito 0.9.0 TTFT ms | llama.cpp `fa-on` TTFT ms | Palito 0.9.0 decode tok/s | llama.cpp `fa-on` decode tok/s |
|---:|---:|---:|---:|---:|
| 512 | 18 | **14** | 785.2 | **924.1** |
| 2,912 | 188 | **75** | 678.1 | **756.3** |

## Phi-3-mini-128k Q4_K_M (2.4 GB dense, context 4,096)

Raw samples: [phi3-mini-128k.json](./phi3-mini-128k.json)

| Prompt tokens | Palito 0.9.0 TTFT ms | llama.cpp `fa-on` TTFT ms | Palito 0.9.0 decode tok/s | llama.cpp `fa-on` decode tok/s |
|---:|---:|---:|---:|---:|
| 501 | 479 | **24** | 197.6 | **395.2** |
| 2,901 | 8,437 | **136** | 174.8 | **314.7** |

*Note:* Phi-3-mini uses `head_dim = 96` (so prefill falls back to quadratic non-flash attention, since Palito's flash prefill currently targets `head_dim = 128`) and mixes `Q5_K` tensors (which do not yet have MMQ prefill or wide packed decode kernels).

## Reproduce

```sh
python3 scripts/bench_long_context.py \
  --palito target/dist/palito-0.9.0-linux-x86_64/palito \
  --model models/benchmark-qwen3/gguf/Qwen3-30B-A3B-Q4_K_M.gguf \
  --sizes 500 2900 10000 40000 \
  --context 40960 --tokens 256 --runs 3 \
  --llama-configs fa-on fa-on-q8kv \
  --palito-mode serve
```
