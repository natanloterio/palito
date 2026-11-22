# Palito: independent Rust GGUF engine

The default runtime is now an independent Rust numerical engine. Cargo does not
build, load, or link llama.cpp, GGML, or the native bridge. llama.cpp is only an
external benchmark reference. The earlier shared-kernel control-plane code is
retained under [benchmarks/legacy](../benchmarks/legacy), not compiled. Its historical
performance results do not measure this independent engine.

See the [architecture reference](architecture.md) for execution flows,
ownership/resource boundaries, performance evidence, and design extension points.
The [standalone master Mermaid chart](architecture.mmd) is editable.

## Default CPU execution scope

- One dense, decoder-only `llama` architecture model, one sequence, CPU execution.
- GGUF v3 with an explicit tensor allowlist: F32, F16, Q5_0, Q8_0, Q4_K, Q5_K, Q6_K.
- Own GGUF reader, tensor dequantization, RMSNorm, adjacent-pair RoPE, causal
  grouped-query attention, SwiGLU, residuals, and tied/separate output weights.
- Packed GGUF matrices remain packed. Our kernels decode bounded blocks on demand
  and reuse them across a batch; activations and KV remain F32. Runtime-guarded
  AVX2 kernels accelerate Q5_0, Q8_0, Q4_K, and Q6_K on supported x86-64 CPUs;
  other CPUs use the portable implementation. No global `target-cpu=native`,
  external BLAS, or llama.cpp/GGML numerical code is required.
- Layer-wise batched matrix projections and causal prefill, with reusable workspace.
  Attention reads only the visible prefix, even when future KV rows in the batch
  have already been written. The output projection is needed only for the last
  prompt token. Single-token decode uses the same weight representation.
- Context-capacity and optional numerical-memory admission checks; counted
  weight/KV/workspace payload allocations, separate from live KV rows and RSS.
- Opt-in bounded immutable-prefix reuse and transactional, greedy-exact n-gram
  speculation. Both are disabled by default.
- Greedy generation; the executor can stop at EOG. Benchmarks intentionally generate
  a fixed number of tokens, including EOG, to keep workloads identical.
- Explicit token IDs or text using our Rust byte-level BPE tokenizer from the
  GGUF's embedded vocabulary, merge ranks, token types, and BOS/EOS settings.
  Text supports `gpt2`/`smollm` and `gpt2`/`qwen2` metadata; unsupported tokenizers
  return explicit errors while raw-ID inference remains available.
- No continuous batching or MoE in the default CPU backend. Independent CUDA/MoE
  execution is opt-in and described below.

The optional KTransformers request-level adapter is described below; its server
can run its own supported heterogeneous MoE configuration.

The default model's name is `Q4_K_M`, but its actual tensors include Q5_0, Q8_0,
Q4_K, Q6_K, and F32. Capabilities are checked from the tensor directory, not inferred
from the filename.

## Native model compatibility

This initial matrix uses llama.cpp revision
[`6b01a476`](https://github.com/ggml-org/llama.cpp/tree/6b01a4767f8d3d0c416c410a49861333965c0982)
as a test oracle only. Status applies to the named fixture, tokenizer and
mode, not to an entire architecture or to performance parity.

| GGUF architecture / fixture | Weight label / tokenizer | Mode | Default CPU | CPU reference | CUDA |
|---|---|---|---|---|---|
| `llama` / SmolLM2-135M-Instruct | Q4_K_M / `smollm` | Causal text | Validated | Validated | Validated |
| `llama` / SmolLM2-135M-Instruct | Q4_1, Q5_1 / `smollm` | Causal text | Validated (short prompts, GGUF EOS) | Validated (short prompts, GGUF EOS) | Validated (short prompts, GGUF EOS; `f32`, no graph) |
| `llama` / SmolLM2-135M-Instruct | IQ4_NL + Q3_K / `smollm` | Causal text | Validated (short prompts, GGUF EOS) | Validated (short prompts, GGUF EOS) | Validated (short prompts, GGUF EOS; `f32`, no graph) |
| `llama` / Llama 3.2 1B Instruct | Q8_0 / `llama-bpe` | Causal text | Not supported | Validated | Validated (`f32`, no graph) |
| `qwen3` / Qwen3-0.6B | F16, Q4_K_M / `qwen2` | Causal text | Not supported | Validated | Validated |
| `qwen3` / Qwen3-0.6B | BF16 / `qwen2` | Causal text | Not supported | Validated | Validated (`f32`, no graph) |
| `qwen3` / Qwen3-0.6B | Pure Q2_K / `qwen2` | Causal text | Not supported | Validated (short prompts) | Validated (short prompts; `f32`, no graph) |
| `qwen3moe` / Qwen3-30B-A3B | Q4_K_M / `qwen2` | Causal text | Not supported | Validated | Validated |
| `qwen2` / Qwen2.5 0.5B Instruct | Q8_0, Q4_K_M and Q4_0 / `qwen2` | Causal text | Not supported | Validated | Validated (`f32`, no graph) |
| `gemma2` / Gemma 2 2B IT | Q8_0 / embedded SentencePiece | Causal text | Not supported | Validated | Validated (`f32`, no graph) |
| `gemma3` / Gemma 3 270M IT | Q8_0 / embedded SentencePiece | Causal text | Not supported | Validated | Validated |
| `phi3` / Phi-3 Mini 4K Instruct | Q8_0 / embedded SentencePiece | Causal text | Not supported | Validated | Validated (`f32`, no graph) |
| `phi3` / Phi-3 Mini 4K Instruct | Q4_K_M / embedded SentencePiece | Causal text | Not supported | Validated | Validated (`f32`, no graph) |
| `phi3` / Phi-3 Mini 128K Instruct | Q4_K_M / embedded SentencePiece | Causal text | Not supported | Both LongRoPE selections validated with short inputs | Both LongRoPE selections validated with short inputs (`f32`, no graph) |
| `starcoder2` / StarCoder2-3B | Q4_K_M / `starcoder` | Causal text | Not supported | Validated (short prompts) | Validated (short prompts; `f32`, no graph) |
| `gpt2` / GPT-2 small | Q8_0 / `gpt-2` | Causal text | Not supported | Validated (short prompts and 1024-token boundary) | Validated (same; `f32`, no graph) |
| `bloom` / Bloom-560M | Q4_K_M / `bloom` | Causal text with ALiBi | Not supported | Validated (short prompts) | Validated (same; `f32`, no graph) |
| `gptneox` / Pythia-70M | Q8_0 / `mpt` | Causal text with partial RoPE and parallel residuals | Not supported | Validated (short prompts) | Validated (same; `f32`, no graph) |
| `olmo` / OLMo-1B-hf | Q4_K_M / `olmo` | Causal text, non-affine LayerNorm, no QKV clamp | Not supported | Validated (short prompts) | Validated (same; `f32`, no graph) |
| `olmo2` / OLMo-2-0425-1B | Q4_K_M / `dbrx` | Causal text, full attention and post-RMSNorm | Not supported | Validated (short prompts) | Validated (same; `f32`, no graph) |
| `olmoe` / OLMoE-1B-7B-0924-Instruct | Q4_K_M / `olmo` | Causal text, top-8 of 64 experts without selected-weight renormalization | Not supported | Validated (short prompts) | Validated (same; `f32`, no graph) |
| `qwen35` / Qwen3.5-0.8B | Q4_K_M / `qwen35` | Causal text, gated full attention and Gated DeltaNet | Not supported | Validated (short prompts, 32-token configured boundary, and recurrent reset/replay) | Validated (same plus 128/140-token fixed-ID checks; batched prefill, `f32`, no graph) |
| `mamba` / Mamba-130M-hf | Q4_K_M / `olmo` | Causal text, Mamba-1 selective scan without dt/B/C RMSNorm | Not supported | Validated (short prompts and recurrent reset/replay) | Not supported |
| Other variants of the architectures above | Other layouts, formats or modes | Varies | Not validated | Not validated | Not validated |
| Other model architectures in the pinned registry | All | All | Not supported | Not supported | Not supported |

To list all architecture names and backend statuses in the pinned registry,
run `bash scripts/model_inventory.sh /path/to/llama.cpp`. The inventory
reports architecture-level coverage, not validation of individual modes.

The shared loader supports Llama's shared `rope_freqs.weight` divisors and
Phi-3's shared short/long divisors with an attention magnitude.
Other scaled or partial RoPE, shared experts, and unrecognized tensor layouts
are rejected. Sliding-window attention is implemented for Gemma 2 and Gemma 3
on the CPU reference and CUDA; attention logit softcapping is supported for
the validated Gemma 2 layout.
The Qwen3 entry covers separate Q/K/V projections and either tied or
separate output weights. It does not cover embeddings, reranking, chat templates,
or CPU/GPU offload. Existing `qwen2` pretokenization applies NFC normalization;
the parity check below uses NFC inputs and does not establish parity for
decomposed Unicode.

The [Llama 3.2 fixture](https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/blob/067b946cf014b7c697f3654f621d577a3e3afd1c/Llama-3.2-1B-Instruct-Q8_0.gguf)
uses F32 and Q8_0 tensors. The previous loader rejected its frequency tensor.
CPU reference and RTX 5090 now pass all 128,256 logits at five checkpoints
(1, 6, 7, 10 and 16 tokens) against the pinned F32-activation oracle,
with maximum absolute errors of `2.15e-5` and `3.86e-6`, respectively.
Exact text, BOS/header/turn IDs, four separately generated greedy IDs and
bitwise reset/replay were checked. This is short-context evidence, not a
validation of the advertised 128K context, other quants, or decode graphs.

To reproduce the dense compatibility checkpoint, from the repository root:

```sh
git --no-pager -C ../llama.cpp rev-parse HEAD  # must match the pinned revision above
git --no-pager -C ../llama.cpp status --short  # must be clean
export CARGO_TARGET_DIR=target/compat-native
cargo build --release --locked --features cuda
mkdir -p results/compat-native
python3 scripts/build_f32_oracle.py --output-dir results/compat-native/oracle-build
export LLAMA32_GGUF=models/compat-llama32/Llama-3.2-1B-Instruct-Q8_0.gguf
python3 scripts/validate_dense_compat.py --fixture llama32 --backend reference \
  --model "$LLAMA32_GGUF" --rust "$CARGO_TARGET_DIR/release/palito" \
  --oracle target/llama-f32-oracle/bin/logits-oracle \
  --output results/compat-native/llama32-reference-new.json
cargo test --release --locked --features cuda --test dense_compat \
  llama32_reference -- --ignored --test-threads=1
```

For GPU validation, configure CUDA 13 NVRTC as below, verify the GPU is idle,
then run the validator with `--backend cuda` and a new output file, and the same
test command with `llama32_cuda`. The validator records exact IDs, per-checkpoint
metrics and input/binary checksums. The existing arithmetic gate is unchanged:
maximum absolute error <= 0.01, cosine >= 0.99999, identical argmax.

[Phi-3 Mini 4K Instruct Q8_0](https://huggingface.co/bartowski/Phi-3-mini-4k-instruct-GGUF/blob/e1447f6da0be91f91683c5d19f938d4f51122d88/Phi-3-mini-4k-instruct-Q8_0.gguf)
is the smallest Microsoft Phi-3 text variant, using the 4K version to avoid
LongRoPE. The loader splits fused QKV and gate/up by stored row, without
dequantizing or requantizing them. It reuses the existing projection and SwiGLU
operators with split-half RoPE. The tokenizer uses the GGUF's SentencePiece
scores, byte fallback, Phi-3 prefix/whitespace rules, and turn/end markers;
it does not apply the chat template.

On the Q8_0 file, all 32,064 logits at each of five checkpoints (1, 6, 7, 10 and
15 tokens) passed, with maximum errors of `4.01e-4` on the CPU reference and
`1.59e-5` on RTX 5090. Both matched all four greedy IDs and passed bitwise
reset/replay. Reproduce using `--fixture phi3-mini-4k`, set `PHI3_GGUF` to the
fixture path, and use the `phi3_reference` and `phi3_cuda` test filters. This
does not validate the full 4K context, longer variants, or non-`f32` profiles.

The [Q4_K_M fixture](https://huggingface.co/bartowski/Phi-3-mini-4k-instruct-GGUF/blob/e1447f6da0be91f91683c5d19f938d4f51122d88/Phi-3-mini-4k-instruct-Q4_K_M.gguf)
also passes the same five full-logit, exact-ID, separate greedy and reset
checks. It contains 32 Q5_K tensors alongside Q4_K, Q6_K and F32.
The portable block decoder and dense CUDA F32 kernels now handle Q5_K without
expanding whole matrices. Maximum absolute error was `2.67e-4` on the CPU
reference and `1.02e-5` on RTX 5090. Reuse the Phi-3 commands with the Q4_K_M
fixture path and a new output file. CUDA MoE, graphs and other matrix profiles
with Q5_K are explicitly rejected.

The shared compatibility changes are in [the transformer loader](../src/transformer.rs),
[GGUF parsing](../src/gguf.rs), [the tokenizer](../src/tokenizer.rs),
[SentencePiece](../src/tokenizer/sentencepiece.rs), [the CPU reference](../src/cuda/reference.rs),
and [CUDA decode](../src/cuda.rs), [prefill](../src/cuda/batch.rs) and [kernels](../src/cuda/kernels.cu).
[The external oracle](../native/logits_oracle.cpp) adds a vocabulary-only tokenizer
and a separate greedy run. [The lifecycle tests](../tests/dense_compat.rs) verify
prefill, explicit continuation, stop markers, stale-logit rejection, and reset.

The official [Qwen2.5 0.5B Instruct fixtures](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/tree/9217f5db79a29953eb74d5343926648285ec7e67)
use separate Q/K/V projections and biases. The native loader applies the biases
before split-half RoPE on Q/K and before caching V. Each fixture passed all
151,936 logits at five checkpoints (1, 5, 6, 9 and 15 tokens), exact NFC-input
tokenization including control tokens, and separate four-token greedy and bitwise
reset/replay checks. Maximum absolute error on CPU reference / RTX 5090 was
`5.67e-5` / `2.27e-5` for Q8_0, `4.68e-5` / `3.85e-5` for Q4_K_M, and
`5.61e-4` / `3.20e-5` for Q4_0. The Q4_K_M file contains Q5_0, Q8_0, Q4_K,
Q6_K and F32 tensors; the Q4_0 file contains 169 Q4_0, 121 F32 and one Q8_0
tensor. Q4_0 uses the portable block decoder and the dense F32 CUDA
projection path; CUDA graphs, MoE and other CUDA math profiles with Q4_0
are rejected. Reproduce
with `--fixture qwen25`, `QWEN25_GGUF` set to the selected fixture, and
the `qwen25_reference` and `qwen25_cuda` lifecycle test filters. This does
not cover decomposed Unicode, automatic chat templating, or full context length.

The [Gemma 2 2B IT Q8_0 fixture](https://huggingface.co/bartowski/gemma-2-2b-it-GGUF/blob/855f67caed130e1befc571b52bd181be2e858883/gemma-2-2b-it-Q8_0.gguf)
uses tied output weights, split-half RoPE, alternating local/global layers,
and softcaps on the scaled attention scores and output logits. Against the
pinned F32-GELU control, all 256,000 logits passed at input lengths 1, 6, 7,
10 and 15, with maximum absolute error `1.72e-5` on CPU reference and
`8.56e-6` on RTX 5090. Exact tokenization (including turn markers), separate
four-token greedy generation, and prefill/decode/reset lifecycle also passed.
A copy changing only `gemma2.attention.sliding_window` from 4096 to 8 passed
the same checks, crossing the masking boundary at lengths 10 and 15
(`2.06e-5` CPU / `9.51e-6` CUDA maximum absolute error). This is a controlled
masking test, not validation of the original 4096-token window or 8192-token
context. Other Gemma 2 sizes are not covered.

Reproduce with `scripts/validate_dense_compat.py --fixture gemma2` using the
F32-GELU oracle described below. For lifecycle checks, set `GEMMA2_GGUF` to the
original file and run
`cargo test --release --locked --features cuda --test dense_compat gemma2 -- --ignored --test-threads=1`.
Eight full-logit comparisons with pre-change binaries (Gemma 3 Q8_0 and
Qwen2.5 Q8_0, both backends, prefill chunks 1 and 4) were bitwise identical.

The pinned inventory has 174 entries: nine have partial causal-text coverage
and 165 have no native path. Other dense layouts
and formats, shared experts/other MoE (including DeepSeek and GPT-OSS),
Qwen3.5 hybrid recurrence and Mamba/Mamba2 remain open. CPU/GPU offload is a
separate capability. Embeddings/pooling, encoder-decoder and multimodal modes
are not covered by any causal-text row.

The [Phi-3 Mini 128K Instruct Q4_K_M fixture](https://huggingface.co/QuantFactory/Phi-3-mini-128k-instruct-GGUF/blob/0cc851bb3014ecd709dbe418684c8d66baa35263/Phi-3-mini-128k-instruct.Q4_K_M.gguf)
previously rejected `rope.scaling.attn_factor`. The loader now validates both
F32 divisor tensors and selects one at model open based on the configured
single-sequence context (padded to 256 by the oracle), not the current token
position. It applies the GGUF's magnitude to the sine and cosine used for
both Q and K. The CPU reference and RTX 5090 passed five full-vocabulary
checkpoints each at contexts 4096 (short) and 4097 (long), exact tokenization,
separate greedy generation, and bitwise reset/replay. Worst absolute error
was `8.40e-4` on CPU reference and `4.44e-5` on CUDA. Switching contexts with
identical IDs changed logits by up to `0.488`, so using the wrong factors would
not pass the gate. This validates selection from position zero and the
configured-context boundary, not sequences of 4096+ tokens or the advertised
131072-token context.

Reproduce with `scripts/validate_dense_compat.py --fixture phi3-mini-128k`
and `--context 4096` or `--context 4097`, using the pinned F32-GELU oracle
built with `scripts/build_f32_oracle.py --f32-gelu`. Set `PHI3_128K_GGUF`
for `dense_compat` test filters `phi3_longrope_reference_lifecycle` and
`phi3_longrope_cuda_lifecycle`. Use distinct output paths for each run.

[SmolLM2-135M-Instruct Q4_1 and Q5_1](https://huggingface.co/QuantFactory/SmolLM2-135M-Instruct-GGUF/tree/476854d00ede130660aba430d15f9347ad2e7d0e)
use packed affine blocks with an F16 scale and offset. Both passed the
F32-GELU gate for all 49,152 logits at five checkpoints, exact tokenization,
four separately generated greedy IDs, and reset/replay on default CPU,
CPU reference and RTX 5090. Worst errors on CPU (both paths) / CUDA were
`3.56e-5` / `2.16e-5` for Q4_1 and `6.49e-5` / `1.29e-5` for Q5_1.
These checks do not establish parity for the oracle's extra EOG tokens:
the default CPU path stops only at the GGUF's EOS ID; exact tokenization
of those extra markers is covered. CUDA rejects MoE, graphs and non-F32
profiles with Q4_1 or Q5_1. Reproduce with `--fixture smollm2` and backend
`cpu`, `reference` or `cuda`. Set `SMOLLM2_Q4_1_GGUF` / `SMOLLM2_Q5_1_GGUF`
for `dense_compat` filters `smollm2_q4_1` / `smollm2_q5_1` and run the CPU
and CUDA tests separately.

The QuantFactory file named `SmolLM2-135M-Instruct.Q2_K.gguf` in that same
revision actually contains 180 IQ4_NL and 30 Q3_K tensors, plus Q8_0 and
F32; it contains no Q2_K tensors. The same SmolLM2 gate and lifecycle checks
passed on both CPUs and CUDA, with worst absolute errors `4.11e-5` and
`1.94e-5`, respectively. Reproduce with `--fixture smollm2` and
`SMOLLM2_IQ4_NL_GGUF` for the `smollm2_iq4_nl` test filters. This does not
validate actual Q2_K tensors.

A separate Qwen3-0.6B fixture, prepared from the pinned F16 model using
`llama-quantize --pure INPUT.gguf OUTPUT.gguf Q2_K 4` at `6b01a476`,
contains 198 actual Q2_K tensors. It passed all 151,936 logits at five
checkpoints, exact IDs, separate greedy and reset/replay. Maximum errors
were `7.54e-4` on the CPU reference and `3.40e-4` on CUDA. Reproduce with
`--fixture qwen3`, `QWEN3_Q2_K_GGUF`, and `qwen3_q2_k` lifecycle filters.

[StarCoder2-3B Q4_K_M](https://huggingface.co/QuantFactory/starcoder2-3b-GGUF/blob/1408543fccd214a5980b16d6d01200001d82193a/starcoder2-3b.Q4_K_M.gguf)
uses affine LayerNorm, biased Q/K/V and attention output, and a biased
ungated GELU FFN. The pinned F32-GELU gate passed all 49,152 logits at five
checkpoints for a code prompt and fixed continuation. Worst absolute error
was `2.01e-5` on the CPU reference and `2.68e-5` on RTX 5090. Both matched
four separate greedy IDs and passed reset/replay, and tokenization matched
including FIM and EOG markers. Gemma 2 and SmolLM2 logits stayed bitwise
identical for prefill chunks 1 and 4 on both backends. Reproduce with
`--fixture starcoder2` and `STARCODER2_GGUF` for the
`starcoder2_reference_lifecycle` and `starcoder2_cuda_lifecycle` filters.
This does not validate the full 16K context or other StarCoder2 variants.

[GPT-2 small Q8_0](https://huggingface.co/igorbkz/gpt2-Q8_0-GGUF/blob/a1a3a46a01c0478277d60919465688e6f4a64350/gpt2.Q8_0.gguf)
reuses the affine GELU transformer path, with split packed QKV rows and
learned absolute position embeddings in place of RoPE. The pinned F32-GELU
gate passed all 50,257 logits at five checkpoints; worst errors were
`6.16e-5` on the CPU reference and `3.11e-5` on CUDA. Exact tokenizer IDs,
separate greedy and reset/replay passed. Use `--fixture gpt2` and `GPT2_GGUF`
for `gpt2_reference_lifecycle` and `gpt2_cuda_lifecycle`.
The same gate also passed at 1023 and 1024 input tokens with a fixed
repeated token sequence, covering the final learned position.

Qwen3-0.6B fixtures:
[ggml-org F16](https://huggingface.co/ggml-org/Qwen3-0.6B-GGUF/blob/afe8ad3577aa22500b3cd1c880f153b3b0e1cae8/Qwen3-0.6B-f16.gguf)
(F16/F32 tensors) and
[Unsloth Q4_K_M](https://huggingface.co/unsloth/Qwen3-0.6B-GGUF/blob/50968a4468ef4233ed78cd7c3de230dd1d61a56b/Qwen3-0.6B-Q4_K_M.gguf)
(Q4_K/Q6_K/F32 tensors). Each was compared against the pinned oracle using
the same file in both engines: exact tokenizer IDs (including Qwen control
tokens), 151,936 logits at three fixed-continuation checkpoints, and four
greedy tokens in a separate generation run. The logit checkpoints were
`The capital of France is` plus 0, 1, or 4 tokens from ` Paris. It is`.
The largest observed absolute logit difference was `8.11e-5` for CPU
reference and `1.06e-5` for CUDA. Logits used the matched-F32
activation oracle described below (max absolute error <= 0.01, cosine >=
0.99999, equal argmax). This does not claim bitwise equality with llama.cpp
or compare its optimized activation-quantized path.

The [ggml-org BF16 fixture](https://huggingface.co/ggml-org/Qwen3-0.6B-GGUF/blob/afe8ad3577aa22500b3cd1c880f153b3b0e1cae8/Qwen3-0.6B-BF16.gguf)
contains 198 BF16 and 113 F32 tensors. BF16 weights are decoded to F32 for
arithmetic, not run through a BF16 tensor-core profile. The same 151,936 logits
passed at lengths 1, 5, 6, 9 and 15 with maximum absolute error `1.60e-4` on
CPU reference and `1.82e-5` on RTX 5090. NFC tokenizer IDs including special
tokens, separate greedy generation and the prefill/decode/reset lifecycle
also passed. Reproduce with `--fixture qwen3`, `QWEN3_BF16_GGUF` and the
`qwen3_bf16` lifecycle test filter. CUDA graphs, MoE and other CUDA math
profiles with BF16 weights are not supported.

```sh
cargo build --release --locked --features cuda
target/release/palito generate --backend reference \
  --model models/compat-qwen3/Qwen3-0.6B-Q4_K_M.gguf \
  --prompt 'The capital of France is' --tokens 4
# Use --backend cuda to run the same file on a configured CUDA machine.
```

The reference backend requires the `cuda` Cargo feature to expose its CLI entry,
but runs entirely on CPU and does not require CUDA libraries or hardware.

For Gemma 3, the CPU reference and CUDA implement alternating local/global layers,
per-layer RoPE bases, embedding scaling, post-attention and post-FFN RMSNorm,
and F32 GELU. Both accept explicit IDs and plain text.
The embedded SentencePiece tokenizer preserves literal whitespace, supports UTF-8
byte fallback, and uses the configured BOS and Gemma turn markers. `generate`
stops on EOS or end-of-turn; no chat template is applied automatically.
CUDA uses the unfused score-buffer path, supports only `--cuda-matmul f32`
and rejects decode graphs. On the
[270M IT Q8_0 fixture](https://huggingface.co/ggml-org/gemma-3-270m-it-GGUF/blob/e7647be17ae1108f2f605ed061ca0608b171afff/gemma-3-270m-it-Q8_0.gguf),
all 262,144 logits pass the same `0.01` absolute error and `0.99999` cosine
gates, with identical argmax across the 512-token window boundary. The largest
absolute error across five checkpoints was `4.56e-4` on the CPU reference
and `4.41e-4` on an RTX 5090. This uses
the same Q8_0 file in both engines and the pinned correctness oracle with
its F16 GELU lookup disabled. That variant does not replace the ordinary
oracle and must not be used as a speed baseline or as evidence of parity
with the unmodified llama.cpp GELU arithmetic.
Tokenizer IDs match the pinned tokenizer for whitespace, Unicode, byte fallback
and explicit turn markers. Two independent greedy continuations matched across
the CPU reference, CUDA, and the F32-GELU oracle on this fixture.

```sh
python3 scripts/build_f32_oracle.py --f32-gelu --output-dir results/gemma3-oracle-NEW
python3 scripts/validate_gemma3_logits.py \
  --model models/compat-gemma3/gemma-3-270m-it-Q8_0.gguf \
  --rust target/release/palito \
  --oracle target/llama-f32-gelu-oracle/bin/logits-oracle
# Add --backend cuda to validate CUDA on a configured machine.
```

## Independent CUDA and Qwen3

The optional `cuda` feature uses our [CUDA kernels](../src/cuda/kernels.cu), not
llama.cpp, GGML, cuBLAS, Torch, or KTransformers numerical code. `cudarc` provides
only dynamically loaded CUDA driver/NVRTC bindings. Cargo builds need no CUDA
toolkit or compiler; execution currently requires Linux, an NVIDIA GPU/driver,
and CUDA 13 NVRTC libraries on `LD_LIBRARY_PATH`.

- `generate`, `bench`, and `logits` accept `--backend cuda`; the default remains
  `--backend cpu`. Missing hardware/libraries and unsupported options are errors,
  never an implicit CPU fallback.
- Supports the dense `llama` layout used by SmolLM2 and dense Qwen3. Qwen3 and
  Qwen3-MoE share per-head Q/K RMSNorm, support for wider queries, and
  NeoX/split-half RoPE. Qwen3-MoE additionally uses rank-three expert matrices.
- Router top-k selection uses lowest-index tie breaking and selected-probability
  renormalization. Only selected expert gate/up/down matrices execute; **all expert
  weights remain GPU-resident**. There is no independent CPU-expert offload yet.
- The Qwen2 tokenizer applies NFC normalization before ordinary BPE pieces,
  matching the verified HF tokenizer; SmolLM input remains unnormalized. Unused
  vocabulary padding is never recognized as a special token and decoding it is
  an explicit error. Qwen's control-token end markers `<|im_end|>` and
  `<|endoftext|>` both stop until-EOG generation; fixed-length benchmarks ignore EOG.
- Weights stay packed in VRAM; activations, KV, reductions, and logits are F32.
  Attention is causal GQA. Each `evaluate` completes device work and downloads
  logits before returning. Opt-in graph generation instead returns device-sampled
  token IDs, while preserving synchronized request timing and the full-logit API.
- Device workspaces and KV are allocated once. `--gpu-memory-limit-bytes` defaults
  to 24 GiB of owned buffers; admission also preserves 2 GiB of currently free GPU
  memory. It does not reserve the GPU against other users or count driver/JIT
  overhead, including driver-owned graph allocations. Graph control and sampling
  buffers are counted. `memory.allocation_domain` distinguishes `"cuda"` from `"host"`;
  CUDA's report excludes host tokenizer/model-loading copies and host logits.
- Dense `llama`, dense Qwen3 and Qwen3-MoE prefill are layer-wise batched: packed projections
  reuse weights across prompt tokens, and each query sees only its causal KV prefix.
  Internal chunks contain at most 4096 tokens (512 before 0.2.3); larger public batches
  are split. Per-token score rows are only reserved for non-flash attention (BLOOM, Gemma).
  Default MoE assignments are grouped by expert in four-assignment tiles, then
  scattered back and combined in the original top-k order. Explicit
  `--cuda-matmul f32-tiled` uses 32-assignment register expert tiles for chunks
  of at least 256 tokens at the measured width-2048/hidden-768, 128-expert/top-8
  shape, when all three expert projections in the layer are Q4_K/Q6_K. Other
  chunks, shapes, and formats retain staged eight-assignment expert tiles.
  Register experts compute 8, 16, or 32 assignments according to each descriptor's
  actual count, without changing its 32-assignment layout or allocations.
  Their K=32 tiles stage contiguous input columns and reuse the existing
  warp-broadcast packed decoder, preserving increasing-K F32 accumulation.
  Eight-way inner-loop unrolling retains that per-output accumulation order.
  At chunks >=256, Q4_K/Q6_K dense projections with input/output widths
  2048/4096, 4096/2048, or 2048/512 use reference-order 2-row/16-token tiles;
  other dense projections retain their existing implementation. Packed-load F32
  Q4_K/Q6_K matrix-vector kernels serve decode and final logits; other formats
  retain the scalar decoder. Packed decode changes reduction order, not precision;
  activations are not quantized. Default `f32` keeps its existing kernels.
  Fused attention keeps scores in shared memory when the
  configured context is <=1024 and head dimension <=256; larger configurations
  retain the original score-buffer implementation. Persistent batched activations,
  grouping, routing, and fallback score buffers are included in device admission.
  Lower `--batch` to reduce workspace, not KV capacity. There is no silent
  workspace reduction or lower-precision fallback when a budget is exceeded.
  Prefix caching, speculation, CPU policy switches, and `--memory-limit-bytes`
  are rejected for CUDA rather than silently ignored.
- `--cuda-decode-graph` explicitly enables one bounded captured decode graph,
  device-side greedy sampling, and up to eight replayed steps per fixed-length
  generation group. Until-EOG generation samples one step at a time. It requires
  context <=1024 and head dimension <=256; unsupported configurations fail.
  Full-logit calls still download logits; token-only calls do not expose stale
  host logits. Reset clears sampling state, and numerical failures require reset.
  This is not a universal decode speedup; MoE remains limited by device execution.
- `--cuda-matmul tf32x3-tiled` is a separate approximate expert-only experiment.
  It **fails** the strict real-model MoE logit gate (0.08683 maximum error versus
  the 0.01 limit), despite matching bounded generated IDs. Do not treat it as
  quality-equivalent or select it solely for its faster prefill timing.
- `--cuda-matmul residual-int8-research` is an explicitly unadmitted arithmetic
  experiment, not a recommended inference profile. At chunks >=256, supported
  block-aligned Q4_K/Q6_K projections use two-component activation quantization
  and shared-memory DP4A tiles. Expert gate/up/down switch together only when
  all three qualify. Smaller chunks and unsupported projections explicitly keep
  the `f32-tiled` path; decode, router, final logits and KV storage remain F32.
  One peak-sized quantization workspace is reused across stages and layers, with
  Q/K/V and gate/up each sharing prepared components. Its two byte arrays and
  component scales/sums are included in admission: **7,864,320 extra bytes**
  for Qwen3 batch512, not four retained stage allocations. Nonfinite conversion
  or reconstruction/dot overflow poisons the request and requires reset.
  Existing `f32`/`f32-tiled` compilation inputs and numerical behavior are
  unchanged. See the [full-model quality evidence](../results/cuda-runtime-m8-int8-model/quality-summary.json)
  before considering this research flag; matrix-corpus speed is not quality
  or end-to-end admission.
  **Rejected by full-model quality gates:** 330 of 462 vocabulary comparisons
  fail; maximum error **5.963800** exceeds 0.01 and minimum cosine **0.986610**
  is below 0.99999. All 14 fixed 32-token generated sequences match, which does
  not override the failed logits. The 132 short-chunk fallback comparisons
  remain bit-exact. No performance acceptance or promotion follows this result.
- `--backend reference` is a slow, independently computed packed CPU oracle for
  these model layouts, including MoE. It is not the optimized default CPU engine
  and does not provide its numerical-memory budget, prefix cache, or speculation.
- `--cuda-kernel-cache PATH` opts into a persistent compilation cache in a trusted,
  private directory. Identity includes all compiled CUDA source, target architecture,
  compilation options, NVRTC-reported version and Cargo lockfile; PTX is checksummed
  and published atomically. Invalid entries or I/O errors fail explicitly rather
  than silently recompiling; remove the named corrupt entry to rebuild. CUDA/NVRTC
  libraries are still required on cache hits. The cache is disabled by default.

With an existing CUDA installation and already provisioned model:

```sh
cargo build --release --locked --features cuda,ktransformers -j 4
export LD_LIBRARY_PATH="$CUDA_ROOT/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
target/release/palito generate --backend cuda \
  --model models/benchmark-qwen3/gguf/Qwen3-30B-A3B-Q4_K_M.gguf \
  --cuda-kernel-cache target/cuda-kernel-cache \
  --prompt 'The capital of France is' --tokens 16

# Independent full-logit, exact-context, reset, and HF-tokenizer comparisons:
python3 scripts/validate_cuda.py --cuda-root "$CUDA_ROOT" \
  --output-dir results/rust-cuda-validation-NEW_RUN

# Own CUDA versus CPU quantizers/expert offsets/routing and tiny MoE forward pass:
cargo test --release --locked --features cuda --lib -j 4 \
  cuda_quantizers_expert_offsets_and_router_match_independent_cpu_kernels \
  -- --ignored --test-threads=1
cargo test --release --locked --features cuda --test cuda -j 4 \
  -- --ignored --test-threads=1
```

The real-model validation script additionally needs the existing offline HF
Qwen3 assets and isolated Transformers environment. It preserves complete logit
vectors and raw commands, and requires maximum absolute error <= 0.01, cosine
similarity >= 0.99999, identical argmax, and exact generated IDs across CPU/GPU,
chunk sizes, repeated reset, and exact-context admission.

On the RTX 5090, the [completed validation](../results/rust-cuda-validation-run2/validation.json)
passed six full-logit comparisons: 49,152 logits per dense case and 151,936 per
MoE case. Maximum absolute errors were 0.000290 (dense) and 0.000049 (MoE);
greedy IDs matched. All 21 Qwen tokenizer cases matched HF IDs and decoded text,
including composed/decomposed Unicode. Exact-context generation, chunk changes,
and repeated reset also matched the independent CPU outputs.

KTransformers remains a distinct whole-request adapter; SGLang/KT, not this
Rust CUDA backend, owns its kernels and KV. Both are available from the same CLI:

```sh
target/release/palito remote \
  --endpoint http://127.0.0.1:30171 --model qwen3-moe-benchmark \
  --kt-method LLAMAFILE --prompt 'The capital of France is' --tokens 16

# Runs engines sequentially and releases each owned service before the next:
python3 scripts/benchmark_moe.py --cuda-root "$CUDA_ROOT" \
  --include-rust-cuda --output-dir results/cuda-moe-comparison-NEW_RUN
```

The optional comparison includes independent all-resident Rust CUDA, KT with
CPU experts, llama.cpp with CPU experts, and an additional all-GPU llama.cpp
baseline. Reports distinguish residency, F32 versus F16 KV, direct synchronized
numerical time versus HTTP request latency, and sequential versus batched prefill.
The existing standalone `ktransformers` CLI and its guarded benchmark remain
supported. No service is launched implicitly by `generate` or `remote`.

The [completed four-deployment comparison](../results/rust-cuda-moe-vs-kt-llama-run1/comparison.json)
uses Qwen3-30B-A3B, 32 generated tokens, one warmup, and five measured samples:

| Input tokens | Rust CUDA, ms | llama.cpp all-GPU, ms | KT CPU experts, ms | llama.cpp CPU experts, ms |
|---:|---:|---:|---:|---:|
| 46 | 1356.45 | 104.20 | 1467.63 | 1588.58 |
| 256 | 5602.28 | 138.11 | 2970.59 | 2543.89 |

These are medians, excluding model loading. Rust measures synchronized numerical
generation plus host sampling; the other columns include HTTP/service overhead.
They are not identical timing scopes. Rust uses F32 KV; the service deployments
use F16. Rust and all-GPU llama.cpp produced identical generated IDs in both
cases, but **Rust is substantially slower**. Rust decode alone measured
54.39 and 45.80 tokens/s. GPU-batched prefill and kernel/launch optimization
remain necessary; the working CUDA baseline is not a performance-parity claim.
All owned benchmark services were stopped after measurement.

The subsequent [CUDA optimization milestone](../benchmarks/cuda-optimization.md)
removed intermediate prompt output projections/readback and parallelized stable
MoE routing. On the same RTX 5090, dense 46/256-token prefill improved 1.34x/1.35x,
MoE improved 1.67x/1.59x, and MoE decode rose from 58.72 to 88.44 tokens/s.
Numerical reservations stayed unchanged; independent CPU-logit gates and exact
old/new 512-token logits passed. The [checkpoint](../results/cuda-optimization-m1/checkpoint.json)
combines seven completed cases with an isolated retry after an unresolved
kernel-blocked process-lifecycle failure. Rust still does **not** beat the tuned
all-GPU llama.cpp baseline; the report preserves variability and cleanup limits.

The [second kernel checkpoint](../results/cuda-optimization-m2/checkpoint.json)
specializes packed matrix products and removes redundant reduction barriers.
Fresh paired measurements improved MoE decode from 88.73 to 102.73 tokens/s
and short/256-token prefill latency by approximately 10-12%, with unchanged
numerical reservations and exact old/new logits. Reverse-order confirmation
passed, but the strict 10% prefill threshold is borderline under pooling;
dense 512-token results remain variable. See the
[experiment log](../benchmarks/cuda-optimization.md) for all outcomes and limitations.

The [multi-token matrix prototype](../results/cuda-prefill-prototype-m3/checkpoint.json)
reuses packed weight decoding across four prompt tokens. Isolated 16/64-token
products measured 2.43-2.71x faster than an independent-row batched GPU baseline,
with exact GPU outputs and clean memory/race/synchronization checks. M3 kept it
test-only because single-token latency regressed. The subsequent dense prefill
integration uses it only for multi-token tiles and preserves decode; see the
[integration gates and results](../benchmarks/cuda-optimization.md).
The [clean-boot real-model controls](../results/cuda-prefill-prototype-m3/clean-boot-control/summary.json)
refresh both Rust and pinned llama.cpp measurements after recovery; Rust still
trails, and a pre-existing MoE decode token mismatch remains disclosed.

The [dense batched-prefill checkpoint](../results/cuda-dense-prefill-m4/checkpoint.json)
integrates the tile into actual inference: SmolLM2 46/256/512-token prefill
improved **8.62x / 11.46x / 13.01x** versus a fresh sequential Rust control.
All eight regression sequences and both 512-token logit vectors matched exactly.
The tradeoffs are **7.8 MiB extra dense workspace** at context 1024 / batch 128
and roughly 80 ms more dense startup/JIT time. Decode stays within the 5%
regression allowance; MoE was still sequential at M4. Fresh tuned llama.cpp controls
are still faster. See the [experiment log](../benchmarks/cuda-optimization.md).

The [M5 grouped-prefill and fused-attention iteration](../benchmarks/cuda-runtime-m5.md)
adds expert grouping, exact-arithmetic on-chip attention, and opt-in kernel caching.
The final fresh matrix improves MoE 46/256/512-token prefill **1.91x / 2.70x / 3.25x**
over M4, and dense 256/512-token prefill **1.19x / 1.39x**. Seven-trial dense
startup falls from **544 to 184 ms** with a warm compilation cache. Ten actual GPU
tests, independent numerical/tokenizer gates, all three sanitizers and exact
old/new 512-token logits pass. Eight-token projection tiles were slower and remain
test-only. **Not all M5 performance targets passed**: short MoE missed 2x, and
decode-heavy latency exceeded 5% in fixed order but passed in reverse order.
At M5 the external MoE numerical discrepancy remained unresolved; llama.cpp remains
faster. See the [checkpoint](../results/cuda-runtime-m5/checkpoint.json) for all evidence.

The [M6 profiling and tensor-core iteration](../benchmarks/cuda-runtime-m6.md)
adds paired benchmark controls, calibrated layer/router tracing, and an explicit
`--cuda-matmul tf32x3` experiment. On the measured 256/512-token MoE workload it
delivers **1.18-1.19x prefill throughput**, passing 12 bounded full-logit gates
(worst error 0.000508). It remains approximate and experimental, **not recommended
for dense or tiny prompts**; default F32 and its CUDA sources remain unchanged.
Slower two-row projections and regressive decode fusion were rejected.
Single-product `tf32` fails quality gates. All 12 GPU tests and three sanitizers
pass; external MoE equivalence and the broad performance target remained unmet at M6.
See the [M6 checkpoint](../results/cuda-runtime-m6/checkpoint.json).

The [M7 staged-expert and graph iteration](../benchmarks/cuda-runtime-m7.md) validates
an explicit `f32-tiled`, batch/chunk 512, graph-enabled profile. Paired MoE
256/512-token prefill improves **2.00x / 2.13x** versus M6; short-context decode
improves from **105.97 to 110.15 tok/s**. It uses 64.73 MiB more MoE workspace,
and dense decode after 256 input tokens regresses about 5.3% even though total
request latency improves. Full-vector gates, exact generated IDs, 19 GPU tests,
and three sanitizers pass. The faster `tf32x3-tiled` experiment **fails quality**.
Retuned llama.cpp remains substantially faster; defaults are unchanged.

The [M8 implementation plan](../benchmarks/cuda-runtime-m8-plan.md) separates
F32-quality structural optimization from experimental arithmetic. Packed F32
decode, measured-shape reference-order dense tiles, and count-aware grouped
register prefill are enabled only by the existing opt-in `f32-tiled` profile.
The earlier serial-K dense register candidate failed full-model quality and
remains test-only, as do four-product TF32 controls. There are no new production
precision modes.
The [first-wave paired results](../results/cuda-runtime-m8-wave1/paired-public-profile/paired.json)
show Qwen3 short-context decode improving from about **110 to 165 tok/s** with
graphs, without activation quantization or an owned-buffer increase. That wave
left prefill-512 at about 893 ms. Full-vector/ID checks, held-out prompts, and all three
sanitizers pass; default F32 kernels remain unchanged. This is not llama.cpp parity.

The [second-wave paired results](../results/cuda-runtime-m8-wave2/paired-public-profile/paired.json)
reduce MoE 256/512-token prefill from **470/894 ms to 383/682 ms**
(**1.23x/1.31x**) against that accepted first wave. Decode stays about 165 tok/s;
dense performance is unchanged. All 88 full-vector quality comparisons are
bitwise equal, generated IDs match, and all three sanitizers pass.
The 512-token MoE profile reserves **10,656 extra bytes**, counted in admission
and allocation. Its profile still attributed 92.5% of kernel time to projections.

The [third-wave checkpoint](../results/cuda-runtime-m8-wave3/checkpoint.json) accepts
reference-order dense tiles and count-aware expert computation.
[Fresh paired results](../results/cuda-runtime-m8-wave3/paired-public-profile/paired.json)
reduce MoE 256/512-token prefill from **384/683 ms to 317/562 ms**
(**1.21x/1.21x**) against the accepted second wave. Decode stays about 165 tok/s;
owned-buffer memory is unchanged. All 128 full-vector comparisons are bitwise
equal, IDs match, and all three sanitizers pass. The new dense tiles preserve the
legacy accumulation/reduction order rather than relaxing numerical gates.
Gate/up fusion was bit-exact but slower and remains prototype-only.
Its profile attributed 91.1% of kernel time to projections, including
298 ms of grouped experts and 203 ms of tiled dense projections. Expert
decode/reuse and resource-balanced fusion were the next experiments.

The [fourth-wave checkpoint](../results/cuda-runtime-m8-wave4/checkpoint.json) accepts
coalesced K=32 expert tiles. Fresh public ABBA reduces MoE 256/512-token prefill
from **317/564 ms to 291/528 ms** (**1.089x/1.068x**) against accepted wave3.
All **152 full-vector comparisons are bitwise identical**, generated IDs match,
memory is unchanged, and decode remains about 165 tok/s. Tests and all three
sanitizers pass; default F32 PTX is unchanged. Smaller fused gate/up tiles are
correct and faster than the old fusion, but still slower than the new separate
projections, so fusion remains disabled. A fresh profile reduces expert kernels
from 298 to 259 ms; projections still account for 90.4% of kernel time.

The [fifth-wave checkpoint](../results/cuda-runtime-m8-wave5/checkpoint.json) accepts
eight-way expert-loop unrolling, not the wider or fully active tile prototypes.
Fresh ABBA reduces MoE 256/512-token prefill from **292/528 ms to 277/507 ms**
(**1.053x/1.041x**) against accepted wave4. All **176 full-vector comparisons
are bitwise identical**, IDs match, memory is unchanged, and short-context
decode remains about 165 tok/s. Tests and all three sanitizers pass.
Expert kernel time falls from 259 to 238 ms; dense tiles remain about 204 ms.
The planned 450-ms prefill goal remains unmet; there is no new llama.cpp
comparison or parity claim.

The [sixth-wave checkpoint](../results/cuda-runtime-m8-wave6c/checkpoint.json) accepts
bit-exact native half-scale conversion in `f32-tiled`, not the experimental dense
schedules. All **65,536 half-bit patterns** preserve legacy conversion bits;
**200 full-vector comparisons** are bitwise identical. Fresh public ABBA reduces
MoE 256/512-token prefill from **278/508 ms to 272/498 ms** and improves
short-context decode from **164.39 to 170.49 tok/s**, with unchanged owned memory.
Tests and all three sanitizers pass; default F32 PTX is unchanged. Dense reuse,
compact reductions, fixed widths and interleaving were rejected rather than
promoted on synthetic timings. The 450-ms goal remains about 48 ms away.

For bounded, local GPU prefill diagnostics:

```sh
cargo build --release --locked --features cuda --example cuda-capture -j 4
target/release/examples/cuda-capture \
  --model models/benchmark-qwen3/gguf/Qwen3-30B-A3B-Q4_K_M.gguf \
  --ids prompt-ids.json --layers 0,15 --positions 0,127 \
  --batch 128 --chunk 128 --matmul f32 --output capture.jsonl
```

The input is a JSON token-ID array; `--cases` also accepts named precision cases
and creates a new output directory with one JSONL file per case.

`--stages` optionally limits projection stages; routing is always retained.
Omitting it preserves the original all-stage capture. Unknown/duplicate stages
are errors, and stage-selection storage counts against the capture budget.

For research-only replay of actual Qwen3-30B-A3B packed weights and captured
operands, capture **every** position of one 256/512-token chunk, for example:

```sh
target/release/examples/cuda-capture \
  --model models/benchmark-qwen3/gguf/Qwen3-30B-A3B-Q4_K_M.gguf \
  --ids prompt512-ids.json --layers 0 --positions "$(seq -s, 0 511)" \
  --stages attn_norm,Qproj,Kproj,Vproj,attn_attended,attn_output,ffn_norm,ffn_gate,ffn_up,ffn_down_inp,ffn_down \
  --batch 512 --chunk 512 --matmul f32-tiled --max-bytes 134217728 \
  --output model-capture.jsonl

M8_REPLAY_MODEL=models/benchmark-qwen3/gguf/Qwen3-30B-A3B-Q4_K_M.gguf \
M8_REPLAY_MODEL_SHA256=44a80615da6ba15e6ad960b411aaab164c499564b8e2322fe5daf1a3c5e9ae15 \
M8_REPLAY_CAPTURE=model-capture.jsonl M8_REPLAY_LAYER=0 \
cargo test --release --locked --features cuda --lib \
  benchmark_model_projection_replay -- --ignored --nocapture --test-threads=1
```

Run GPU work serially with the CUDA 13 library path configured as above. The
ignored replay test is deliberately shape-bounded to this model's seven
Q/K/V/output and expert gate/up/down projections, and to captures starting at
position 0. It rejects missing positions or wrong widths, validates routing
descriptors, and requires every structural candidate output bit to match its
captured production result before reporting CUDA-event samples. Model, capture,
packed tensor, input, output and source hashes identify the corpus. Captures are
bounded to 512MiB files. The existing GGUF parser loads the full packed model on
the host (transient memory can approach twice its file size); only the selected
layer's seven tensors are retained, and one tensor at a time is uploaded.
Model loading, parsing, hashing, copies and correctness checks are outside kernel
timings; grouping is reported separately. This diagnoses real-operand execution,
not end-to-end speed or acceptance of any approximate arithmetic. Production
defaults, allocations and numerical kernels are unchanged.

With the same `M8_REPLAY_*` environment, replace the replay test name with
`benchmark_residual_model_primitives` to run bounded arithmetic research.
It prepares two signed INT8 activation components per 32 values, once per input
stage (shared across Q/K/V and across gate/up). GPU components, scales and sums
must match the CPU quantization oracle bit-for-bit. Packed Q4_K DP4A includes
minima/activation-sum corrections; Q6_K retains signed scales. Zero/subnormal
blocks are supported; nonfinite inputs and reconstruction/dot overflows fail
explicitly. Signed-zero reconstruction is not guaranteed by this approximation.
The benchmark checks the full production projection against its capture before
sampling eight output rows for every token/routed assignment, with 16 independent
CPU F64/affine oracle dots per projection. It separately attributes activation
reconstruction, decoder rearrangement and integer-merge rounding errors.
A scalar four-product TF32 control measures accuracy, not tensor-core throughput.
CUDA-event timings include both component preparations or both products and
corrections, respectively; allocation, copies and CPU worklists are excluded.
These sampled dot timings cannot establish the required production-GEMM speedup
or full-logit quality gates. Primitive diagnostic controls remain test-only.
Only the explicit `residual-int8-research` profile loads the shared conversion
and matrix kernels; existing profiles do not include them. See the
[primitive results](../results/cuda-runtime-m8-residual-primitives/summary.json)
and [research checkpoint](../results/cuda-runtime-m8-residual-primitives/checkpoint.json).

`benchmark_residual_model_matrices`, with the same environment, evaluates a
research 32-row/32-assignment DP4A tile against the production dense and expert
kernels. It stages packed integer weights and both activation components in
shared memory, retains Q4_K minima and Q6_K signed corrections, and writes the
original token/assignment-major output layout. Weight decoding, activation
gather and output scatter occur inside the measured kernel; no separate layout
conversion is omitted. All output rows are compared, not sampled; NaN-initialized
interiors detect unwritten outputs and surrounding guards detect overruns.
Preparation remains shared once per input stage. Aggregate comparisons count
each preparation once, and the same expert-grouping cost on both paths.
Repeated isolated projection timings are not end-to-end inference measurements.
The benchmark does not establish full-logit/exact-ID acceptance.
The separately opt-in research integration is evaluated by the ignored
`cuda::model_quality::residual_full_model_quality` test. Its `M8_QUALITY_MODEL`,
`M8_QUALITY_CASES`, `M8_QUALITY_OUTPUT` and `M8_QUALITY_MODE` select the model,
bounded case map, new output directory and profile. `M8_QUALITY_TEACHER` optionally
supplies the baseline's saved teacher schedule. It writes complete F32 vocabulary
vectors, compares greedy generation separately through the decode graph, and
checks workspace reuse, exact context capacity, invalid admission and sticky
quantization failure/reset. This is diagnostic work, not inference timing.
A subsequent test-only three-component FMA-residual quantizer reduces worst
component-reconstruction error to **1.3514e-8** across3,670,016 first-layer captured
values. All bytes/scales/sums match an independent CPU oracle;68 ordinary tests
and three scoped sanitizers pass. Native integer tensor-core instruction mappings
also match8,192 CPU integer outputs. These are
[primitive results](../results/cuda-runtime-m8-precise3-primitives/checkpoint.json),
not final-logit or end-to-end admission. Full three-component IMMA matrices are
evaluated separately in the [MMA corpus](../results/cuda-runtime-m8-mma3-matrices-v3/summary.json).
All84 projections win, with **2.5715x aggregate speedup including conversion and
grouping**. This passes the matrix gate, not the full-model gate.
The explicit `--cuda-matmul residual-mma3-research` mode now connects that frozen
kernel to prefill batches of at least32 tokens (initially256); shorter or unsupported
projections, including decode projections, retain the accepted F32-tiled path. It requires compute
capability8.0 and adds a separately accounted, reusable11.25MiB workspace at
batch512 on the MoE model, without allocating the rejected two-component
workspace. The [full-model gate](../results/cuda-runtime-m8-mma3-model/quality-summary.json)
passes all594 full-vocabulary comparisons across18 cases, including four
predeclared broader heldouts: worst error0.001194, minimum cosine0.9999999989,
exact teacher-forced argmax and32-token generated IDs. Capacity/error/reset,
workspace reuse,68 ordinary tests, strict Clippy/fmt and three scoped real-model
sanitizers pass.
The [public ABBA](../results/cuda-runtime-m8-mma3-acceptance/performance-summary.json)
reduces256-token prefill from272.25ms to115.15ms (**2.36x**) and512-token prefill
from498.34ms to200.94ms (**2.48x**). Targeted complete requests improve1.83x/2.48x;
decode and46-token prefill remain effectively unchanged. All measured IDs match
the accepted runtime, with no phase regression above5%. This is scoped opt-in
validation, not a production/default promotion.
The subsequent [attention improvement](../results/cuda-runtime-m8-attention-normalized/performance-summary.json)
normalizes each shared score once instead of repeating its division for every
output dimension, only in the MMA3 profile. All594 vocabulary vectors and all
18 generated sequences remain bit-exact to admitted MMA3; four F32/F32-tiled
graph/non-graph PTX comparisons are byte-identical. No workspace is added.
ABBA improves512-token prefill another2.6% to195.55ms and256-token prefill to113.23ms.
Decode rises from170.45 to176.27token/s on the1/32 case,153.26 to168.75 after46
prompt tokens, and93.93 to132.84 after256. The latter complete requests improve14.3%.
See the [completed admission](../results/cuda-runtime-m8-attention-normalized/public-checkpoint.json)
and [latest comparison](../results/cuda-runtime-m8-attention-normalized/comparison.json):
llama.cpp still leads by4.31x/5.99x prefill and approximately1.96x short decode.
The [fresh head-to-head control](../results/cuda-runtime-m8-head-to-head/comparison.json)
still shows llama.cpp ahead; matrix improvements do not establish otherwise.
The [new post-MMA comparison](../results/cuda-runtime-m8-mma3-acceptance/comparison.json)
narrows the prefill gap to4.43x/6.14x at256/512 tokens versus llama.cpp F32 KV;
short decode is170.42 versus344.42token/s. The engines still differ in generated
IDs for the one-token/32-generated case, and llama controls are not ABBA.

The [small-prefix extension](../results/cuda-runtime-m8-mma3-small-acceptance/checkpoint.json)
lowers only MMA3 eligibility to32 tokens; the rejected two-component mode stays
at256 and defaults remain unchanged. Actual32/46/128-token matrices pass the
1.5x conversion/grouping-inclusive gate in both development and frozen-binary
heldout runs. The baseline uses its real small-prefix dense kernel and tile8
expert grouping, not the larger-prefix kernels or candidate tile32 grouping.
All [990 vocabulary comparisons](../results/cuda-runtime-m8-mma3-small-model/quality-summary.json)
across30 cases and all30 generated32-token sequences pass unchanged quality
gates. The18 prior cases, including four broader prompts, are retained;12 new
boundary cases reuse existing prompts rather than claiming new unseen content.
All528 unaffected prior snapshots and132 below-threshold fallback comparisons
are bit-exact. Actual batch capacities31/32/46/128/255 pass memory, full-context,
error/reset and exact-ID checks; three bounded sanitizers test the newly allocated
batch32 workspace. At batch32 the accounted increase is737,280 quantization bytes
plus12,096 group bytes; batch512 adds nothing over admitted MMA3.
[Public ABBA](../results/cuda-runtime-m8-mma3-small-acceptance/performance-summary.json)
cuts46-token prefill128.87ms ->63.21ms (**2.04x**) and improves its complete
request **1.63x**, exceeding the predeclared10% gate in both orders. All measured
IDs match; no phase regresses more than5%. The256/512-token prefills remain
113.25/195.78ms and short decode176.30token/s. The
[fresh comparison](../results/cuda-runtime-m8-mma3-small-acceptance/comparison.json)
still favors llama.cpp:4.76x/4.33x/5.99x prefill at46/256/512 tokens and roughly
1.95x short decode. These are scoped research results, not production/default
promotion or cross-engine numerical equivalence.

The subsequent [projection pipeline study](../results/cuda-runtime-m8-mma3-projection-profile/study-summary.json)
is **not admitted**. A test-only double-buffered `cp.async` kernel preserves all
three activation components and the exact floating-point accumulation order.
The cached-index revision improves complete matrix-panel costs1.439x/1.378x/
1.255x/1.054x at32/46/128/512 tokens, including preparation and grouping.
All28 projections /25,733,120 outputs per revision match frozen MMA3 bits;
24 synthetic tail/routing/layout fixtures,68 ordinary tests, strict Clippy/fmt
and three scoped sanitizers pass. But all three revisions fail the predeclared
10% improvement at every panel length; the final occupancy-bound revision also
fails the individual-projection5% regression bound. No threshold is narrowed
after failure, and no heldout/full-model/public admission is claimed.
The inference executable remains byte-identical to accepted small-prefix MMA3.

Hardware-counter profiling is blocked by `ERR_NVGPUCTRPERM`, after a retained-ABI
replay resolved an older Nsight/CUDA13 injection mismatch. No GPU permissions,
clocks or power configuration were changed. Driver resource queries show cached
indices raise registers66 ->76 and theoretical resident blocks7 ->6; forcing7
blocks introduces16 local bytes/thread and does not rescue the speed gate.
These are static resources, not measured warp stalls or achieved occupancy.
The validated cached-index version remains research-only; larger-prefix
projection cost still requires a materially different improvement.

The [block-affine arithmetic screen](../results/cuda-runtime-m8-mma3-block-affine-development/summary.json)
is also **rejected**, with no inference integration. It retains three activation
components but shares their scales across256 values, folds Q4/Q6 subgroup
coefficients into two exact signed-byte planes, and applies floating-point
corrections once per GGUF block. Complete development-panel speedups versus
admitted MMA3 are only0.879x/0.875x/0.866x/0.850x at32/46/128/512 tokens:
all fail the predeclared1.5x gate, with an individual-projection regression too.
All28 actual matrices /25,733,120 values pass the matrix accuracy screen
(maximum absolute error0.00009919, minimum cosine0.9999999999992476); this is
not full-vocabulary or full-model quality admission. Independent CPU oracles,
24 synthetic matrix fixtures,71 ordinary tests, strict Clippy/fmt and three
sanitizers covering both the quantizer and matrices pass. The accepted executable
remains byte-identical.

**Model provenance limitation:** the current local GGUF's whole-file SHA256
differs from both the historical corpus pin and the download lock. All21 packed
projection tensors across layers0/37/47 match the historical corpus exactly.
The [explicit user-approved amendment](../results/cuda-runtime-m8-mma3-block-affine-development/provenance-amendment.json)
permits this matrix-only screen with unchanged tensor, capture and admitted
output hash checks; it does not establish metadata-only differences or
whole-model equivalence. No model files or historical results were modified, and
no heldout/full-model/public run was used to excuse the block-affine rejection.
The current file (`398bf479...6aa86`) was subsequently approved for fresh
same-file comparisons only.

[Parallel expert grouping](../results/cuda-runtime-m8-parallel-grouping/paired/paired.json)
is available with `--features cuda-parallel-grouping-research` and
`--cuda-matmul residual-mma3-research`. With 512 extra bytes of workspace on
Qwen3, it matches all 990 full-vocabulary snapshots and 30 independent
generations in the fresh quality run. It speeds up 512-token prefill by
2.7-3.0% across both ABBA orders, below the 10% package gate. The default
grouping path is unchanged. The final feature-gated build also matched the
measured candidate's 46-token smoke-test logits exactly.

The optional `cuda-packed-reduction-research` feature reduces packed F32
matvec synchronization in the same MMA3 profile without reordering additions.
In the [short-context decode ABBA](../results/cuda-runtime-m8-packed-decode-trace/decode-abba-summary.json)
it improves throughput from about 176 to 210 tokens/s with no added reservation.
All 990 full-vocabulary snapshots and 30 independent generations match the
fresh Rust baseline. The accepted executable remains unchanged.

`cuda-packed-down-research` is an additional, unadmitted opt-in experiment for
width-768 Q4_K/Q6_K expert down projections in single-token MMA3 execution. It
maps one warp to each output row while preserving the original packed reduction
tree. Other shapes and default builds retain their existing dispatch. Replay
results are diagnostic, not full-model admission or llama.cpp parity.

`cuda-packed-gate-up-research` is a separate opt-in experiment for width-2048
Q4_K/Q6_K expert gate and up projections in single-token MMA3 execution. It
preserves each packed reduction tree and writes the SwiGLU activation directly.
Other shapes and default builds retain separate launches.

`cuda-decode-fusion-research` combines bitwise-preserving single-launch Q/K/V
packed decode matvecs, fused head RMSNorm/RoPE/KV-cache decode for 128-wide
heads, and single-warp register top-k routing for 128 experts in MMA3 mode,
without changing numerical reservation or default dispatch.

`cuda-mma3-wide-rows-research` is an opt-in 64-row MMA3 tile for grouped
expert projections at exactly 512 prefill tokens. It shares activation staging
across more output rows without changing the K accumulation order. Dense
projections, smaller chunks, and default builds retain their existing dispatch.
The separate `cuda-mma3-dense-wide-rows-research` feature also enables the
tile for eligible dense MMA3 projections at 512 tokens only.

`cuda-mma3-splitk-research` splits K across power-of-two CTA slices (minimum
128 K elements, up to 1024 CTAs) for low-CTA dense MMA3 prefill projections and
reduces slices in deterministic increasing order inside existing workspace
scratch, adding no device reservation.

`cuda-attention-query-pair-research` pairs adjacent causal queries for
512-token MMA3 prefill with 128-wide heads and bounded fused attention.
`cuda-attention-query-pair-even-research` extends this bitwise-preserving pair
kernel to even prefill lengths of at least 32 tokens.
Each query keeps its original score and value reduction order. Other attention
shapes, including single-token decode, retain their prior dispatch.

Two additional MMA3 opt-in features preserve the original reduction order:
`cuda-block-reduction-research` for shared 256-thread sums and
`cuda-attention-reuse-research` for fused attention query reuse and softmax
maximum reduction. With both enabled alongside parallel grouping and packed
matvec reduction, all 990 full-vocabulary snapshots and 30 independent
generations match the fresh Rust baseline. Neither feature changes the
default dispatch.

See the [full-matrix results](../results/cuda-runtime-m8-residual-matrices/summary.json)
and [checkpoint](../results/cuda-runtime-m8-residual-matrices/checkpoint.json).

Choose positions present in the prompt. Captures contain selected projection
inputs/outputs and routing records, with a default 64 MiB retained-record budget
(`--max-bytes`). JSON encoding and transient host staging are not included in that
budget. Output files are created exclusively, never overwritten. Capture requires
non-singleton prefill chunks and rejects decode graphs; reset cancels a capture.
It intentionally synchronizes/copies GPU data and must not be used for inference
timings. Without an enabled capture, no diagnostic device copies occur.

Replay actual routing through the grouped microbenchmark:

```sh
M8_ROUTING_FIXTURE=capture.jsonl cargo test --release --locked --features cuda \
  --lib -j 4 benchmark_grouped_routing_replay \
  -- --ignored --test-threads=1 --nocapture
```

Replay validates expert IDs, counts, and assignment ordering before initializing
CUDA. Input files are limited to 64 MiB and 128 routing records; use selected
layers or a routing-only subset of larger combined projection dumps.
Its matrix operands are deterministic fixtures: captured routing is real,
but replay timings are not full-model performance measurements.

The subsequent [matched-precision MoE investigation](../benchmarks/moe-numerical-fix.md)
resolves the numerical diagnosis: the original comparison mixed Rust's F32
activations with the external engine's packed-product activation conversion.
An isolated pinned external graph with row-wise weight dequantization and F32
activations passes all 151,936 logits on five prefixes, including the original
divergence and 256-token prefill, against both Rust CPU and CUDA. A controlled
packed-product build restores the discrepancy. Rust precision and dependencies
are unchanged. Optimized llama.cpp still uses different arithmetic and can select
different tokens; the correctness oracle is **not a performance baseline**.

## Build and first benchmark

Rust/Cargo 1.85+ is sufficient for the runtime, without CMake, C++, a service, or
a reference checkout. The first Cargo build downloads Rust crates. Model download
is explicit and separate; inference makes no model-network requests.

Run from this directory:

```sh
python3 scripts/download_model.py
cargo build --release --locked -j 4
# input.json contains nonnegative token IDs from this GGUF's vocabulary.
target/release/palito bench \
  --model models/SmolLM2-135M-Instruct-Q4_K_M.gguf \
  --input-tokens-file input.json --tokens 8 --warmups 1 --repetitions 3
```

The downloader fetches the public Apache-2.0 SmolLM2-135M-Instruct GGUF
(105,454,432 bytes) at the immutable revision in
[fixtures/model.json](../fixtures/model.json), then verifies its size and SHA-256.
Existing files with an incorrect checksum are rejected, not overwritten.

Only the explicit external reference build below requires CMake and C++17.
It is not a Cargo build step or a runtime dependency.

## What is compared

The independent Rust engine and an external llama.cpp control process receive
identical explicit token IDs and greedy generation counts. The reference
tokenizes a seed prompt once outside measurement; our runtime never calls it.
Rust retains packed weights and uses F32 activations/KV; llama.cpp uses its
quantized activation kernels and F16 KV.
This is not a same-kernel or bitwise-equivalent comparison.

Every default measured request starts with a full active KV/state reset and no
prefix reuse. OS page caches remain
warm. Model/context load time is reported separately. Tokenization and reset are
outside the phase timers:

- `prefill_ns`: chunked prompt evaluation, synchronized before returning.
- `ttft_ns`: prefill plus first greedy sample, not client/end-to-end TTFT.
- `decode_ns`: the remaining **N-1** output steps. When explicitly enabled,
  this includes all draft verification work, including rejected candidates.
- For one generated token, decode steps/time are zero and no decode ratio exists.

The harness requires repeatability within each engine and Rust invariance across
tested chunk/thread changes. Cross-engine token equality is reported, not assumed:
different activation arithmetic can change greedy choices. Incomplete results,
within-engine mismatches, subprocess failures and invalid timings fail explicitly.

Results preserve raw process output, commands, per-request timings, model/prompt/
binary/shared-library/source hashes, linkage, the Cargo lockfile hash, Rust version,
the llama.cpp revision and dirty status, CPU model/affinity, and
system load. Existing report paths are rejected to preserve the baseline.

```sh
cmake -S native -B target/external-reference \
  -DLLAMA_SOURCE_DIR="$PWD/../llama.cpp" -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CUDA=OFF -DGGML_NATIVE=ON -DGGML_OPENMP=OFF -DGGML_BLAS=OFF
cmake --build target/external-reference --parallel 4 --target li-reference
python3 scripts/benchmark_rust.py \
  --reference target/external-reference/bin/li-reference \
  --output-dir results/rust-independent-NEW_RUN
```

The suite covers one generated token, a normal prompt, chunks 1/7/64, one/four
threads, 96-token prefill, 32-token decode, and the exact context boundary
(125 prompt + 4 generated - 1 = 128 evaluated tokens). Each process warms up once
and produces three measured samples. Engine order alternates by case; no exclusive
hardware reservation or clock control is assumed. The old shared-kernel CLI in
[scripts/benchmark.py](../scripts/benchmark.py) is retired.

## Run inference

```sh
cargo run --release --locked -- bench \
  --model models/SmolLM2-135M-Instruct-Q4_K_M.gguf \
  --input-tokens-file input.json --tokens 32
```

For text completion, use exactly one of `--prompt`, `--prompt-file`, or
`--input-tokens-file`. `generate` stops at the model's EOG token and returns JSON
with decoded text and generated IDs:

```sh
target/release/palito generate \
  --model models/SmolLM2-135M-Instruct-Q4_K_M.gguf \
  --prompt 'The capital of France is' --tokens 16

# These commands read metadata without expanding model weights:
target/release/palito tokenize \
  --model models/SmolLM2-135M-Instruct-Q4_K_M.gguf --prompt 'Hello, world!'
target/release/palito detokenize \
  --model models/SmolLM2-135M-Instruct-Q4_K_M.gguf --input-tokens-file input.json
```

No chat template is applied automatically. Embedded control tokens such as
`<|im_start|>` are recognized in input and omitted from decoded output.
User-defined tokens are retained. The whole generated sequence is decoded
together so UTF-8 bytes split across tokens are reassembled; incomplete/invalid
UTF-8 uses the standard replacement character. Some SmolLM byte symbols are
pruned from its vocabulary; text requiring an absent byte returns an explicit
error, not an unknown-token fallback. Regex failures also propagate as errors.

`bench` still generates fixed-length token-ID reports; it also accepts text,
with tokenization outside the numerical phase timers. `logits` accepts the same
input choices and emits the complete next-token F32 logit vector. No reference
process, network request, or additional tokenizer file is used by these commands.

## Tests

```sh
# Synthetic GGUF, executor, numerical operation, and local HTTP fixture tests:
cargo test --locked --all-features -j 4
python3 -m unittest discover -s scripts -p 'test_*.py' -v

# Real-model full-logit reproducibility across threads, chunks, and state reset:
LI_TEST_MODEL="$PWD/models/SmolLM2-135M-Instruct-Q4_K_M.gguf" \
  cargo test --release --locked --test cpu -j 4 -- --ignored --test-threads=1

# All Rust builds are independent of llama.cpp:
cargo test --locked --no-default-features --lib -j 4

cargo fmt --check
cargo clippy --locked --all-features --all-targets -j 4 -- -D warnings

# External-only tokenizer oracle and text/explicit-ID generation equivalence:
cmake --build target/external-reference --parallel 4 --target li-reference
python3 scripts/validate_rust_tokenizer.py \
  --reference target/external-reference/bin/li-reference \
  --output-dir results/rust-tokenizer-NEW_RUN
```

The verified SmolLM2 tokenizer run matched external llama.cpp token IDs and
decoded bytes in all 21 cases, including empty input, multilingual text, emoji,
numbers, whitespace, special tokens, and a long word. Text and explicit-ID
generation produced identical output tokens. See the
[text validation report](../results/rust-tokenizer-smollm2-run1/validation.json).

Real-model tests run serially to avoid competing with each other.
Default benchmarks use no persistent prefix reuse; dedicated cache tests and
explicitly labeled warm-cache experiments exercise the opt-in path.

## Numerical memory, prefix reuse, and speculation

`bench` and `generate` emit `memory` with counted numerical payload capacities.
The following budget/cache/speculation controls describe the default CPU backend:
weights, active KV allocation, reusable workspace, logits, token history, and
the current immutable prefix snapshot. `kv_live_bytes` describes occupied rows;
reset/rollback makes rows unreachable but does **not** free their allocation.
`weight_bytes` in model metadata still means GGUF file size, not resident weights.
`expanded_weight_bytes` is `null` unless `--f32-weights` is selected.

`--memory-limit-bytes` admits the fixed numerical allocations, bounded worker
block/dot scratch, and the configured prefix capacity before allocating KV and
workspace. The prefix reservation is capped at the largest possible snapshot.
This is **not a process RSS limit**: model loading/file copies, tokenizer metadata,
container descriptors, allocator overhead, general thread stacks, and caller-owned
request/output vectors are excluded. Weights and metadata are already loaded when
the numerical budget is checked.

```sh
# Optional optimizations; defaults remain cold-cache, non-speculative execution:
target/release/palito bench \
  --model models/SmolLM2-135M-Instruct-Q4_K_M.gguf \
  --prompt 'The capital of France is' --tokens 16 \
  --memory-limit-bytes 150000000 \
  --prefix-cache-bytes 8388608 --speculative-window 4

# Independent expanded/scalar/sequential control for differential validation:
target/release/palito bench \
  --model models/SmolLM2-135M-Instruct-Q4_K_M.gguf \
  --input-tokens-file input.json --tokens 8 \
  --f32-weights --sequential-prefill --scalar-kernels
```

The one-entry prefix cache is local to an immutable backend/model/configuration;
it cannot be restored into a different model instance. Exact token IDs must match
the stored prefix. Each snapshot contains its own KV/logits, is bounded by the
payload budget, and survives active-request reset only when explicitly enabled.
Oversized prompts are not cached; an existing fitting snapshot can still be reused.
Replacement evicts the old entry before allocating a new one. Reports expose
`cached_prompt_tokens`; the default comparison harness rejects unexpected reuse.
Snapshot creation/restoration is included in prefill time. Warm-cache latency is
not a cold-inference speedup or a fair comparison with an uncached reference.

`CpuBackend` exposes single-use append transactions with commit/rollback. Nested,
foreign, and reset-invalidated handles are rejected. Rollback restores the prior
logits/history/visible position; stale rejected KV cannot affect later attention.
`verify_draft` evaluates a proposed token batch causally and commits only its
greedy-matching prefix, including the correct logits at that boundary.

`--speculative-window N` enables a bounded local n-gram proposer (no extra model
download), with `N <= batch`. Acceptance, proposal counts, and verification batches
are reported per request. Generation preserves greedy output and leaves the final
sampled token unevaluated, including at the exact context boundary. Verification
can cost more than it saves; it remains opt-in. Chunk/thread determinism is tested
within a selected kernel path; scalar versus vector reductions can differ slightly
and are checked against the numerical oracle rather than claimed bitwise equal.

### Numerical oracle

An explicit, validation-only F32 copy isolates forward-pass correctness from
llama.cpp's quantized activation arithmetic. Performance tests still use the
original GGUF:

```sh
cmake --build target/external-reference --parallel 4 --target llama-quantize
target/external-reference/bin/llama-quantize --allow-requantize \
  models/SmolLM2-135M-Instruct-Q4_K_M.gguf target/validation-f32.gguf F32
python3 scripts/validate_rust_logits.py \
  --model models/SmolLM2-135M-Instruct-Q4_K_M.gguf \
  --f32-reference-model target/validation-f32.gguf \
  --reference target/external-reference/bin/li-reference \
  --benchmark-dir results/rust-independent-NEW_RUN \
  --output-dir results/rust-logits-NEW_RUN
```

The checker compares all 49,152 logits for short, decode-heavy, near-context and
(if applicable) first-divergence inputs. It requires cosine >= 0.9999,
RMSE <= 0.01, maximum absolute error <= 0.05, and identical top token. Raw vectors,
commands, hashes and metrics are saved even on threshold failure. The external
oracle still has F16 KV; this is not bitwise equality.

The original expanded-F32 SmolLM2 run on the i9-14900K completed all eight scenarios,
with matching cross-engine token sequences in seven. The 32-token decode case
diverged under the quantized llama.cpp reference; the explicit F32 oracle agreed
with Rust's top token at that divergence. Across all four numerical checks,
cosine similarity exceeded 0.9999999 and maximum absolute logit error was below
0.009. See the [performance report](../results/rust-independent-smollm2-run2/comparison.json)
and [numerical validation](../results/rust-independent-logits-run2/validation.json).

That historical correctness-first baseline was not a speedup: the four-thread short case
measured 693.92 ms Rust prefill versus 55.45 ms llama.cpp, and 60.63 versus 199.22
decode tokens/s.

### Packed CPU results and measured policies

The optimized run again completed all eight reference scenarios, with identical
greedy output in seven. The known long-decode quantized-reference divergence
remains; all four full-vocabulary checks against the preserved F32 oracle pass.
The selectable scalar/F32 control also reproduces the previous Rust logits
exactly. All measured cache/speculation variants preserve cold Rust output.

Median of three samples after one warmup, four CPU threads, 45 input/8 output
tokens on the shared i9-14900K:

| Engine/path | Prefill ms | Decode tokens/s | Weight payload bytes |
| --- | ---: | ---: | ---: |
| Rust scalar F32, sequential control | 663.40 | 66.32 | 538,060,032 |
| Rust packed AVX2, batched prefill | 104.27 | 87.52 | 103,668,480 |
| External llama.cpp | 51.97 | 196.03 | Not instrumented by this harness |

Packed storage reduces the Rust weight payload by **80.7%**. The measured short
case has 6.36x faster prefill and 1.32x higher decode throughput than the new
binary's scalar/sequential control, but remains slower than llama.cpp. These are
configuration-specific shared-machine measurements, not general speedup claims.
The initial portable packed kernel regressed decode; that intermediate report is
retained, not used as the final result.

Chunk 64 beat chunks 7/1 for the tested short prefill; no automatic general-purpose
policy was inferred. Warm-prefix reuse restored all 45 prompt tokens. N-gram
speculation accepted 7/15 proposals in the decode-heavy case but did not improve
its decode throughput, and combining it with prefix reuse hurt the short case.
Therefore neither optimization is silently enabled.

See the [packed comparison](../results/rust-packed-smollm2-run2/comparison.json) and
[policy/numerical/memory validation](../results/rust-cpu-policies-run1/validation.json).
Raw commands, vectors, sample timings, allocations, and hashes are retained.

```sh
python3 scripts/benchmark_rust.py \
  --reference target/external-reference/bin/li-reference \
  --output-dir results/rust-packed-NEW_RUN
python3 scripts/validate_cpu_optimizations.py \
  --benchmark-dir results/rust-packed-NEW_RUN \
  --oracle-dir results/rust-independent-logits-run2 \
  --output-dir results/rust-policies-NEW_RUN
```

The policy checker requires matching model/binary provenance, verifies eight
full-logit comparisons against the preserved successful oracle, measures control
and opt-in paths, and tests exact numerical-budget admission and rejection one
byte below the required reservation. Existing output directories are rejected.

## Backend boundary and KTransformers

[Backend](../src/backend.rs) contains no llama.cpp/GGML types. The generic
[Executor](../src/lib.rs) owns policy, request admission, prefill chunking, and token
positions. The [Rust CPU backend](../src/cpu.rs) implements numerical token stepping.
There is no `llama-cpp` Cargo feature or native build script.

The optional [KTransformers adapter](../src/ktransformers.rs) targets the **current
kt-kernel + SGLang local server**, not the archived standalone Python engine.
It implements the separate `RequestBackend` interface: SGLang owns token stepping,
batching, and KV lifetimes. It deliberately does not implement `Backend`, and
cannot be passed to the token-step `Executor`. Its advertised token-step and
Rust-controlled chunked-prefill capabilities are false.

The client uses `/server_info`, `/model_info`, and non-streaming `/generate`.
The contract was inspected against KTransformers revision
`ca198c50de70abb6827ba6f573bb6671e54436e2` and its SGLang submodule revision
`2c4085a01cf75cde927d35a259b4adaa7b07b3ab`. These identify the inspected API,
not the version of a user's running server. Runtime reports retain the version
and selected configuration actually returned by that server.

### Build without llama.cpp

```sh
cargo build --release --locked --no-default-features \
  --features ktransformers --bin ktransformers -j 4
```

No Python dependencies are needed for this Rust client. The server must already
be installed and running in its own compatible environment. This project neither
installs nor starts/stops a server, downloads its model, or changes global Python
packages. See the upstream [KT-Kernel setup guide](https://github.com/kvcache-ai/ktransformers/blob/ca198c50de70abb6827ba6f573bb6671e54436e2/kt-kernel/README.md)
for compatible dependencies/hardware.

A **launch template**, with real compatible model paths substituted:

```sh
python -m sglang.launch_server \
  --host 127.0.0.1 --port 30000 \
  --model-path /path/to/gpu-model --served-model-name local-moe \
  --kt-method LLAMAFILE --kt-weight-path /path/to/cpu-gguf-directory \
  --kt-cpuinfer 4 --kt-threadpool-count 1 --kt-num-gpu-experts 0 \
  --kt-max-deferred-experts-per-token 0 \
  --context-length 2048 --disable-radix-cache --disable-shared-experts-fusion
```

**This is not a way to load the dense SmolLM2 fixture into KT-Kernel.** Current
KTransformers CPU kernels target supported MoE models. The LLAMAFILE path requires
compatible CPU expert GGUF weights plus the GPU-side model weights. AMX paths
use differently prepared CPU weights. Consult upstream model-specific launch
instructions; the adapter never converts or requantizes weights.

### Connect and generate

```sh
target/release/ktransformers --model local-moe --kt-method LLAMAFILE info
target/release/ktransformers --model local-moe --kt-method LLAMAFILE \
  run --prompt 'Explain memory bandwidth briefly.' --tokens 32
target/release/ktransformers --model local-moe --kt-method LLAMAFILE \
  bench --prompt-file fixtures/prompt.txt --tokens 32 \
  --warmups 1 --repetitions 3 --output results/kt-request-baseline.json
```

For an exact common input across backends, replace `--prompt-file` with
`--input-tokens-file` pointing to a JSON array of token IDs. These options are
mutually exclusive; the report records the selected input.

Use `--endpoint http://127.0.0.1:PORT` before the subcommand for a different port.
Only plain HTTP loopback origins are accepted; proxies, redirects, embedded
credentials, and remote hosts are disabled. The initial adapter does not support
authenticated servers. Every HTTP call has a configurable `--timeout-seconds`
(default 300), and responses are bounded to 8 MiB. Generation is never
automatically retried after a network or server error.

The handshake checks the served-model name, explicit KT method, CPU weight path,
positive CPU thread count, model identity, generation support and initialized
tokenizer. It rejects ordinary SGLang servers without a reported KT configuration.
Only selected identity/configuration fields are exposed; server API keys and raw
server configuration are not copied into reports. This verifies reported
configuration, not that every expert ran on the CPU or a cryptographic weight
identity.

Expert deferral must be explicitly configured by the server, with zero used for
the baseline. A positive value requires client `--allow-expert-deferral`.
Selecting an AMX or other method explicitly does not establish numerical parity
with GGUF/llama.cpp. Actual tensor types and weight/KV allocation bytes are not
exposed by this API, so the adapter does not invent them.

The library accepts either raw text or explicit input token IDs. Text uses the
server tokenizer without a client chat template. Known token counts are checked
against the context capacity when the server reports it; text tokenization and
remaining admission checks belong to the server. Missing context capacity is
reported as `null`, not guessed.

Responses must contain complete output token IDs and consistent prompt/completion/
cache accounting, a normal finish reason, and the same reported weight version.
Aborts, incomplete fixed-length results, silent length truncation, changed
versions, malformed JSON, HTTP errors, and timeouts surface as errors.

### Request-level benchmarks and tests

The KT benchmark reports only `elapsed_ns`: HTTP transport, server scheduling,
tokenization/generation, and response parsing/validation. Connection-handshake and
model-load time are excluded; the model is already loaded by the service.
There are **no invented prefill/decode or TTFT values**, and this report is not
directly comparable with the in-process llama.cpp phase benchmark. A kernel-level
performance comparison would require a matching direct KT/SGLang baseline.

Benchmark mode requires disabled radix caching and zero reported cached tokens,
checks greedy token repeatability across warmups/measured requests, and refuses
to overwrite reports. It never invokes a global cache flush, creates sessions,
or mutates server settings. An HTTP timeout closes the client request; it is not
a guarantee of immediate server-side cancellation or GPU memory reclamation.

```sh
# Local HTTP fixtures; no Python, models, GPU, or llama.cpp needed:
cargo test --locked --no-default-features --features ktransformers -j 4
cargo clippy --locked --no-default-features --features ktransformers \
  --all-targets -j 4 -- -D warnings

# Opt-in real-service test; no server or model is automatically provisioned:
LI_KT_ENDPOINT=http://127.0.0.1:30000 LI_KT_MODEL=local-moe LI_KT_METHOD=LLAMAFILE \
  cargo test --locked --no-default-features --features ktransformers \
  --test ktransformers -- --ignored --test-threads=1
```

Protocol/CLI tests exercise real local HTTP connections with deterministic fixture
responses, not KT kernels. A live-server test is provided separately and ignored
unless explicitly requested. The interface does not promise portable KV pages,
mutable layer placement, speculative rollback, or client-owned cache reset.

### Provisioned Qwen3 comparison

The optional comparison setup is separate from the small CPU baseline. It uses
the documented Qwen3-30B-A3B checkpoint, original HF weights (about 61 GB), and
the official Q4_K_M GGUF (about 18.6 GB). Do not run these downloads or GPU
services on a shared machine without approval.

```sh
uv venv --python /path/to/python3.11 .venv-ktransformers-bench
uv pip install --python .venv-ktransformers-bench/bin/python \
  -r benchmarks/ktransformers/requirements.txt
python3 scripts/download_moe_models.py
```

Build CUDA `llama-server` under `target/benchmark-llama-cuda13` using a coherent
toolkit. The following optional isolated toolkit avoids changing system CUDA.
The architecture below targets this benchmark machine's RTX 5090:

```sh
uv venv --python /path/to/python3.11 .venv-cuda-bench
uv pip install --python .venv-cuda-bench/bin/python \
  -r benchmarks/ktransformers/cuda-requirements.txt
CUDA_ROOT="$(.venv-cuda-bench/bin/python -c \
  'import sysconfig; print(sysconfig.get_paths()["purelib"] + "/nvidia/cu13")')"
cmake -E create_symlink lib "$CUDA_ROOT/lib64"
cmake -E create_symlink libcudart.so.13 "$CUDA_ROOT/lib/libcudart.so"
cmake -S ../llama.cpp -B target/benchmark-llama-cuda13 \
  -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120 \
  -DCMAKE_CUDA_COMPILER="$CUDA_ROOT/bin/nvcc" -DCUDAToolkit_ROOT="$CUDA_ROOT" \
  -DCUDA_CUDART="$CUDA_ROOT/lib/libcudart.so.13" \
  -DCUDA_cudart_LIBRARY="$CUDA_ROOT/lib/libcudart.so.13" \
  -DCUDA_cublas_LIBRARY="$CUDA_ROOT/lib/libcublas.so.13" \
  -DCUDA_cublasLt_LIBRARY="$CUDA_ROOT/lib/libcublasLt.so.13" \
  -DCMAKE_BUILD_RPATH="$CUDA_ROOT/lib" \
  -DCMAKE_EXE_LINKER_FLAGS="-Wl,-rpath-link,$CUDA_ROOT/lib" \
  -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_APP=OFF \
  -DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF
cmake --build target/benchmark-llama-cuda13 --parallel 4 --target llama-server
python3 scripts/benchmark_moe.py --cuda-root "$CUDA_ROOT" \
  --output-dir results/qwen3-comparison
```

The development symlinks are confined to the isolated toolkit and provide the
standard layout required by SGLang's JIT linker. Install the complete pinned
`ktransformers[sglang]` distribution: the CUDA kernel payload is split across
its companion wheels, so installing only `kt-kernel` and `sglang-kt` is insufficient.

The downloader pins both model revisions, verifies weight SHA-256 values, and
retains resumable partial downloads. The driver requires the completed downloads,
the release Rust adapter, and the CUDA `llama-server` build. It starts services
sequentially, bounds CPU expert threads to four, uses identical explicit input
IDs (short prompt and 256-token prompt), disables prefix reuse, performs a warmup
and five measured requests per case, and cleans up its own server processes.
It preserves server logs, raw responses, package/model provenance and comparison
JSON. It does not stop or change existing servers.

This compares deployment latency, not equivalent numerical implementations.
CPU experts use the same GGUF, but KT uses original non-expert weights cast to
F16 while llama.cpp uses the GGUF's non-expert tensor types. Attention kernels,
graph policies, and client implementations also differ. Generated tokens per
second includes prompt processing and must not be called isolated decode speed.

Verified local run on an Intel i9-14900K and RTX 5090 (four CPU expert threads,
32 generated tokens, one warmup and five measured requests per case):

| Input tokens | KT median request latency | llama.cpp median request latency |
| --- | --- | --- |
| 46 | 1456.38 ms | 1628.97 ms |
| 256 (repeated short input) | 2923.91 ms | 2481.01 ms |

KT had 10.6% lower latency on the short input; on the repeated 256-token input,
KT had 17.9% higher latency. Each engine's output was repeatable, but output
tokens differed between engines. These are configuration-specific deployment
measurements, not numerical parity or a general engine ranking.
The [comparison report](../results/kt-vs-llama-qwen3-run4/comparison.json) has the
controls and limitations; adjacent files retain raw samples, launch commands,
model hashes, software versions and server logs.

## Implementation stages and remaining research

The CPU sequence now has packed quantized kernels, measured batched prefill,
counted numerical reservations, chunk/thread policy measurements, bounded immutable
prefix reuse, and transactional speculation with commit/rollback tests. Specialized
AVX2 kernels are runtime-dispatched and have a tested portable fallback. The
optional request-level KTransformers backend remains available and separate.

The optional independent CUDA path now executes dense and Qwen3-MoE models;
KTransformers remains request-level. Further expansion still requires work:
GPU-batched prefill, additional tokenizer/model architectures, continuous
batching, and independent heterogeneous MoE residency policies. These are not
implied by the CPU results or by the KT HTTP adapter.

See the [literature review](../../literature_review/review.md) for the evidence and
limitations behind this order. A successful smoke benchmark establishes a
baseline; it does not establish a speedup or generalize to larger models.

## Embedding API

`palito::engine::Engine` opens a model with the `chat` defaults: the GGUF context, capped at 65,535 and fitted to free device memory on CUDA. It then tries the CUDA profiles in this order and uses the first that opens: `int8-fast` with the decode graph, `f32-tiled` with and then without the graph, then `f32`. It runs chat (the model's template, `enable_thinking` per call) or plain completion with streamed text and a finish reason (stop, length, context full, cancelled). `src/ffi.rs` wraps it as the C API declared in `include/palito.h`, built as `target/release/libpalito.so` (`crate-type = ["rlib", "cdylib"]`). Panics are caught at the boundary. Strings returned by the library are freed by the library. Tests: `cargo test --release --test engine -- --include-ignored` (needs the SmolLM2 fixture; the CUDA case needs `--features cuda`).

## Linux CLI bundle

`scripts/package-linux.sh` builds `target/dist/palito-<version>-linux-x86_64.tar.gz`: the CUDA binary, bundled NVRTC 13 and a wrapper that defaults to `--backend cuda` with a kernel cache in `~/.cache/palito/kernels`. With `--install`, the script also links the wrapper as `~/.local/bin/palito`. The host needs an NVIDIA driver with CUDA 13 support.

```sh
scripts/package-linux.sh --install
palito chat --model /path/to/model.gguf
```

`chat` streams each reply to stdout token by token (`--no-stream` prints it when complete; the text is identical either way). With the decode graph the tokens are read back while the next steps already run on the GPU, so streaming does not slow decoding (Qwen3-30B-A3B, 1,014-token reply: 338-341 tok/s streamed, 337-338 tok/s without). `generate --stream` prints plain text as it is generated instead of the final JSON object. Text is emitted once it is stable: a token that ends inside a UTF-8 character is held until the character completes (`palito::stream::TextStream`, used with `Executor::generate_streaming`).

`chat` uses the model's trained context by default (GGUF `context_length`; Qwen3-30B-A3B: 40,960 tokens). On CUDA it is capped at 65,535 and shrunk, in 256-token steps, to the largest context whose buffers fit in free device memory minus 2 GiB, with a notice. Replies run until end of turn or until the context is full, which is reported (`--tokens N` caps a reply; small models stuck in a greedy loop otherwise run to the full context). Older turns are dropped while a message plus a 2048-token reply reserve exceeds the context; a single message that fits alone keeps the remaining context for its reply. Measured on Qwen3-30B-A3B Q4_K_M (RTX 5090, `int8-fast`): a 40.7k-token prompt reaches the first token in 6.4 s and decodes at 103 tok/s (211 tok/s at 9.9k, ~335 tok/s near empty); a 39.7k-token needle-in-a-haystack prompt recalls the needle at 5%, 50% and 95% depth with `int8-fast` and `int8-prefill`; generating the full 40,960-token context runs at 152 tok/s on average. Other commands keep `--context 512` unless given. Without `--gpu-memory-limit-bytes` the CUDA budget is free device memory minus 2 GiB (previously a fixed 24 GiB).

`PALITO_BACKEND`, `PALITO_MODEL` and `PALITO_CUDA_KERNEL_CACHE` set the defaults, and flags take precedence.

`chat` on CUDA defaults to `--cuda-matmul int8-fast` (below) with the decode graph and prefill chunks of up to 4096 tokens; GPUs below sm_80 fall back to `f32-tiled`, and models whose weights only the reference path supports (Q5_K, e.g. Phi-3-mini-128k Q4_K_M) to `f32` without the decode graph, each with a notice. `--cuda-matmul int8-prefill` keeps the more precise tensor-core prefill: INT8 MMA projections, FP16 hi/lo attention over K/V converted once per layer; attention projections use a two-plane residual pair with an integer factor, FFN/expert projections three residual components; decode is the `f32-tiled` path. Its logits pass a relaxed gate against the F32 oracle (same argmax, cos >= 0.99999, max_abs <= 0.15; 0.09 on a 256-token real-text prompt); Qwen3-30B-A3B 2.9k prompt on an RTX 5090: 0.35 s to the first token. Pass `--cuda-matmul f32-tiled` for F32 prefill.

`--cuda-matmul int8-fast` (chat default) runs prefill at llama.cpp's precision: one 8-bit plane per 32 activations (as `q8_1`), single FP16 attention and a TF32 MoE router. Q4_K projections run in cp.async double-buffered INT8 MMA tiles, and each expert's assignments fill whole 128-token tiles plus a 64- and/or 32-token remainder tile. On Qwen3-30B-A3B Q4_K_M (RTX 5090) it reaches the first token of a 2.9k prompt in ~0.16 s (llama.cpp 0.183 s; `int8-prefill` 0.35 s), 1.4k in ~0.086 s (llama.cpp 0.090 s), 0.55k in ~0.043 s (llama.cpp 0.045 s). Its logits are about 1 off F32 (max_abs 0.86-0.97, cos ~0.99966 on a 256-token real-text prompt), so generated text can differ from the F32 profiles.
