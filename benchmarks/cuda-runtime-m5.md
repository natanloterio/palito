# M5: grouped MoE prefill, attention, and startup

This bounded iteration starts from the retained M4 runtime and identical local
GGUF files. It does not claim general superiority over llama.cpp.

## Gates declared before performance changes

- Grouped MoE prefill: at least 2x lower median prefill latency at 46/256/512
  tokens; no more than 5% decode-heavy request regression. Added numerical
  workspace must be at most 64 MiB. Preserve the existing GPU addition order:
  require identical generated sequences and bitwise-equal full logits against M4.
- Attention prototype: causal GQA with F32 KV, nonzero positions, reset and
  chunk boundaries. Kernel error <= `1e-4 + 1e-5 * abs(reference)`.
  Full-model max logit error <= 0.01, cosine >= 0.99999 and unchanged argmax on
  the measured cases. Require at least 10% end-to-end dense prefill improvement
  at both 256 and 512 tokens, with the same decode regression allowance.
- Matrix prototype: preserve bitwise outputs and the existing single-token path.
  Require at least 10% model-level prefill improvement, not just hot-cache
  synthetic speedup.
- Compilation cache: identical loaded kernels and inference results; complete
  source/compiler/options/architecture identity, tested invalidation and
  corruption handling. Measure uncached, cold-cache and warm-cache process startup
  separately; target >=25% lower dense startup latency with a warm cache.
- Use seven measured requests after warmup, preserve individual samples, and
  run GPU experiments sequentially. Compare the complete eight-case matrix with
  retained M4 and fresh pinned llama.cpp F32/F16 controls where feasible.

## First-divergence investigation

At the previously divergent 19-token common prefix, the independent Rust CPU
and CUDA paths differ by at most 0.00001097 (RMSE 0.00000374), and both choose
7269. The pinned external CUDA oracle chooses 614, with a top-two margin of
0.09889; Rust's margin is 0.06649. External CPU chooses 7269.

Rust versus external GPU max error is 0.42057, RMSE 0.08860 and cosine
0.9997397. Versus external CPU: max error 0.65495, RMSE 0.12737 and cosine
0.9994753. These do **not** establish tight external numerical equivalence.
llama.cpp's quantized CUDA matrix-vector implementation quantizes activations
to Q8_1; Rust uses F32 activations. Different numerical paths are established,
but their contribution versus other graph/kernel differences is not isolated.
No inference bug or quality equivalence is inferred from sampled text alone.

Full vectors and input IDs are retained in `results/cuda-runtime-m5/`.
`native/logits_oracle.cpp` is a standalone external-only executable against
the pinned llama.cpp; it is never linked into Rust. It uses original packed
weights and F32 KV, with GPU flash attention enabled and CPU offload disabled
for the respective runs. This numerical oracle is not a performance baseline.

Reproduce the external oracle from the runtime directory:

```sh
g++ -O3 -std=c++17 -I../llama.cpp/include -I../llama.cpp/ggml/include \
  native/logits_oracle.cpp -Ltarget/benchmark-llama-cuda13/bin \
  -Wl,-rpath,'$ORIGIN' -lllama -lggml -lggml-base \
  -o target/benchmark-llama-cuda13/bin/logits-oracle
# Set the same CUDA library path as the main CUDA benchmark procedure.
taskset -c 0-3 target/benchmark-llama-cuda13/bin/logits-oracle \
  models/benchmark-qwen3/gguf/Qwen3-30B-A3B-Q4_K_M.gguf \
  results/cuda-runtime-m5/divergence-input.txt -1
```

## Retained implementations

- Token-major shared projections, Q/K head norm, RoPE and causal attention now
  serve both architectures. Routing remains stable per token. A device grouping
  pass constructs deterministic expert/four-assignment descriptors; projections
  reuse a packed weight decode across assignments and scatter into original
  token/top-k slots. Combination retains the old selected-expert summation order.
  All expert weights stay resident. There is no CPU-expert offload or host
  readback between expert stages.
- Fused attention computes scores cooperatively in warps, preserves the original
  256-lane dot and softmax trees, and stores scores on chip. Unlike the originally
  proposed online-softmax direction, this bounded implementation does not change
  accumulation order. It supports configured context <=1024 and head dimension
  <=256; other shapes fall back explicitly. Single-token decode stays unchanged.
- Batched scores no longer need a global allocation on the fused path. At
  context1024/batch128, dense workspace shrinks from 8,258,064 to **3,577,352 bytes**;
  MoE grows from 307,780 to **22,757,360 bytes** (+21.41 MiB, below the 64 MiB limit).
  Weights, KV and vocabulary reservations are unchanged.
- `--cuda-kernel-cache PATH` is opt-in. Keys hash the complete concatenated CUDA
  sources, all `CompileOptions` fields including architecture, NVRTC major/minor,
  and Cargo lockfile. PTX identity/checksum validation precedes loading; a miss
  compiles normally and atomically renames a private temporary file. Cache errors
  are explicit, not broad fallback. Use a trusted private directory. The cache is
  not an authenticity mechanism and does not remove the NVRTC dependency.
- Benchmark provenance now includes **all** CUDA sources, not just `kernels.cu`.
  The eight-token projection experiment is test-only and is not compiled into
  production.

## Correctness and rejected experiment

- **54 ordinary tests** and the default CPU-only suite pass, as do strict
  all-target/all-feature Clippy, formatting and the locked CUDA/KT release build.
- **Ten actual GPU tests** pass serially: six-format quantizers, expert offsets,
  stable routing, four/eight-token projection tails and guards, grouped expert
  assignment preservation including >256 experts, causal fused attention through
  the 1024 boundary, original attention fallback, multi-layer dense/MoE chunks,
  nonzero positions, 128+1 internal split, reset, exact/one-under budgets, and
  sticky earlier-token overflow.
- Memcheck, racecheck and synccheck over both GPU test executables report
  **zero errors/hazards**. The fused and grouped kernels match the old GPU
  addition order bitwise, stronger than the initial numerical tolerances.
- Six independent full-logit gates, exact-context/chunk/reset generation, and
  21 offline HF tokenizer cases pass. Maximum CPU/CUDA logit errors are
  0.0002893 dense and 0.00004814 MoE. All eight real-model regression sequences
  and both 512-token full-logit vectors match the retained M4 runtime exactly.
- Eight-token projection tiles pass correctness but are **19-38% slower** than
  four-token tiles for measured 16/64-token Q4_K/Q6_K products. Small-batch
  regressions are larger. They were rejected before model integration; raw
  CUDA-event samples are in `wider-tile-microbenchmark.log`. This is a hot-cache
  synthetic comparison, not a model throughput claim.

## Measurements

The final matrix uses identical artifacts/IDs, context1024, batch/chunk128,
four threads pinned to CPUs0-3, F32 Rust KV, one warmup, seven trials and
concurrency1. Measurements exclude loading. No concurrent performance experiments
or system policy changes were made. Whole-device memory polling is not a
per-process peak guarantee. Seven-sample p95/p99 are descriptive only.

| Case | Fresh M4 prefill ms | M5 ms | Speedup | Fresh llama F32 KV ms |
|---|---:|---:|---:|---:|
| dense-decode (1 input) | 1.192 | 1.249 | 0.954x | 1.105 |
| dense-short46 | 5.599 | 5.935 | 0.943x | 3.937 |
| dense-prefill256 | 37.553 | 31.565 | 1.190x | 11.138 |
| dense-long512 | 104.773 | 75.385 | 1.390x | 22.538 |
| moe-decode (1 input) | 9.485 | 10.730 | 0.884x | 3.548 |
| moe-short46 | 425.093 | 222.055 | 1.914x | 15.745 |
| moe-prefill256 | 2940.524 | 1091.075 | 2.695x | 48.463 |
| moe-long512 | 7168.084 | 2202.400 | 3.255x | 91.079 |

Pinned all-GPU llama.cpp remains faster. Its internal phase boundaries differ
from synchronized Rust phase timers; this is not a same-kernel comparison.
Fresh tuned F16 KV controls and HTTP wall times are preserved alongside F32.

Earlier staged matrices are also retained, not substituted for the final one:
grouping alone measured 2.00x/2.74x/3.15x MoE prefill speedup against M4; attention
fusion then improved dense256/512 another 1.35x/1.58x against the grouped control.

**The complete declared M5 target is not met.** Final short-MoE speedup is 1.914x,
below 2x. Fixed-order decode-heavy request latency rises **9.35% dense / 8.56% MoE**,
above 5%. An explicit reverse-order confirmation finds **+2.12% / -0.56%**,
within the allowance, while the M4 control itself slows between windows.
This establishes timing sensitivity, not proof of no regression. Both complete
sample sets are preserved; no unconditional <=5% claim or universal win is made.
The inherited benchmark `milestone_passed` field is M1-specific; the separate M5
checkpoint evaluates M5's actual gates and also reports failure.

### Startup cache

Seven fresh-process samples per mode rotate disabled/cold/warm ordering.
Median dense loading is **544.15 ms disabled**, **557.28 ms cold** and
**184.32 ms warm**, a **66.13% warm-cache reduction**. Full logits match exactly.
Cold compilation remains slower; a cache is not free first-run speedup.
Other experiment windows show startup variability (the final matrix disables
the cache), so these are paired cache-experiment measurements, not guaranteed
startup bounds or loading claims for the 18.6 GB MoE model.

The measured executable is retained as `optimized-runtime` with SHA256
`954d4ae23caa332dedf690a89dfe5585bc8f9feb0e562ee5820fc6126de0c0b9`.
A subsequent Cargo rebuild produced a byte-different executable; rather than
assuming equivalence, a separate complete eight-case, three-trial check verifies
all generated IDs and both full 512-token vectors exactly. That executable is
retained as `rebuilt-runtime`; its measurements are preserved in `rebuilt-control/`.
The seven-trial final matrix and startup numbers above belong to the original
measured executable, not silently to the rebuild.

### Remaining bottlenecks and limits

The final dense profile uses warmup plus one 256-input/4-output request (518
evaluated tokens). Packed batched products consume **57.7%** of kernel time,
fused prompt attention **11.8%**, and original decode attention-values **20.0%**.
The next bounded experiments should profile grouped MoE projection costs and
test fused decode attention separately. Merely widening a tile has already
failed; avoid assuming tensor-core/reduced-precision arithmetic preserves quality.
Current external MoE quality equivalence, training-limit contexts, concurrency,
continuous batching, CPU expert offload and per-request energy remain unverified
or unsupported. No external computational backend was added to Rust.

## Reproduction

From the runtime directory, with `CUDA_ROOT` pointing at the isolated CUDA13
toolkit and its `lib` directory on `LD_LIBRARY_PATH`:

```sh
cargo build --release --locked --features cuda,ktransformers
cargo test --release --locked --features cuda --lib --test cuda \
  -- --ignored --skip benchmark_packed_prefill_weight_reuse --test-threads=1
cargo test --release --locked --features cuda --lib \
  benchmark_packed_prefill_weight_reuse -- --ignored --nocapture --test-threads=1
python3 scripts/benchmark_cuda_iteration.py --output-dir results/M5-NEW \
  --label m4 --binary results/cuda-dense-prefill-m4/optimized-runtime \
  --cuda-root "$CUDA_ROOT"
python3 scripts/benchmark_cuda_iteration.py --output-dir results/M5-NEW \
  --label candidate --binary results/cuda-runtime-m5/optimized-runtime \
  --cuda-root "$CUDA_ROOT" --compare-to results/M5-NEW/m4 --llama
```

`results/cuda-runtime-m5/checkpoint.json` indexes commands, hashes, raw samples,
startup measurements, oracle vectors, numerical validation, sanitizer output,
profiles, opposite-order decode runs and the retained executable/source archive.
Result directories are create-new, so reproduction never overwrites prior runs.
