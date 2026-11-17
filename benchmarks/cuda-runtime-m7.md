# M7: staged expert projections and bounded graph decode

## Outcome

The validated F32 hybrid profile improves Qwen3-30B-A3B 256/512-token prefill
**2.00x / 2.13x** over immutable M6, with exact generated IDs and unchanged
full-vector quality gates. Short-context MoE decode improves only **1.04x**.
**It does not beat retuned llama.cpp.** The faster approximate tiled TF32x3
experiment fails quality and is not accepted as quality-equivalent.

The preserved runtime is
[`accepted-runtime`](../results/cuda-runtime-m7/accepted-runtime), SHA256
`0168b1ad6f4920e4077eaf2a72ec20174089bd85b0fa5b1b05f751194659ca22`.
Default matmul remains F32, graph decode remains disabled, and the independent
runtime has no llama.cpp/GGML numerical dependency.

## Implemented policy

- Repair redundant shared-memory allocation in grouped prefill.
- Opt-in `f32-tiled`: staged F32 expert products with eight assignments per tile;
  ordinary dense projections and scalar single-token products remain F32.
  Reduction order can differ, so this is not a bitwise-logit guarantee.
- Opt-in `tf32x3-tiled`: the same expert-only dispatch with approximate
  three-product TF32 arithmetic. It remains an explicitly failed-quality experiment.
- Bounded internal prefill capacity rises from 128 to 512 tokens. Batch/chunk
  controls still determine workspace and actual work; no larger default is forced.
- Opt-in graph decode uses device token/position state and greedy sampling, with
  at most eight fixed-length replay steps before synchronized token readback.
  Until-EOG generation remains single-step. Full-logit evaluation still downloads
  logits. Context >1024 or head dimension >256 is rejected for graph mode.
- Reset and sticky numerical-failure semantics are preserved; token-only
  evaluation cannot expose stale host logits.

Rejected optimizations are not silently enabled: staged dense projections
regressed, globally broadcast packed metadata regressed MoE decode to about
76 tok/s, and the warp-per-row decode prototype was not consistently faster.
Scalar production matvec decoding was restored. The warp metadata helper is
retained only for staged loading; the alternate decode prototype is test-only.

## Controls and chunk sweep

Hardware: RTX 5090, i9-14900K, four pinned CPU threads, concurrency one, context
1024. Models and their hashes are unchanged from M6. No downloads, dependency
installations, clock/power changes, or exclusive-GPU claim were made.

The exploratory sweep used one warmup and three repetitions for both real models
at batch/chunk 64/128/256/512. Each configuration retained six complete logit
vectors and generated-ID sequences. These sequential sweep timings are not the
final paired acceptance measurements.

MoE prefill, milliseconds:

| Expert mode | Chunk | 256 input | 512 input |
|---|---:|---:|---:|
| F32 | 64 | 832.18 | 1701.11 |
| F32 | 128 | 763.34 | 1555.74 |
| F32 | 256 | 734.56 | 1497.32 |
| F32 | 512 | 735.99 | 1466.04 |
| F32 tiled | 64 | 646.99 | 1316.50 |
| F32 tiled | 128 | 533.36 | 1073.07 |
| F32 tiled | 256 | 472.64 | 953.85 |
| F32 tiled | 512 | 472.66 | 892.69 |
| TF32x3 tiled, **quality failed** | 512 | 439.26 | 804.41 |

All modes keep dense projections F32 in this hybrid policy. The complete
[sweep and quality results](../results/cuda-runtime-m7/v3-sweep-quality.json)
include all 24 configurations and the rejected numerical comparisons.

## Final opposite-order measurements

Order A1/B1/B2/A2, three warmups and seven measurements per window:

- A: immutable M6, F32, batch/chunk 128, graph off.
- B: M7, `f32-tiled`, batch/chunk 512, graph on.
- Same prompt IDs, output lengths, context, affinity, and model identities.
- All measured generated IDs match across both orders and all repetitions.

The following are pooled medians across the two windows per profile. This is a
tuned-profile comparison, not isolated attribution to one kernel or graph replay.

| Workload | M6 prefill ms | M7 prefill ms | Prefill speedup, each order | Total request speedup |
|---|---:|---:|---|---:|
| Dense 46 input / 8 output | 5.20 | 4.56 | 1.140x / 1.143x | 1.076x |
| Dense 256 / 8 | 27.80 | 21.70 | 1.278x / 1.284x | 1.090x |
| Dense 512 / 1 | 65.50 | 46.62 | 1.406x / 1.403x | 1.405x |
| MoE 46 / 8 | 192.10 | 130.49 | 1.468x / 1.473x | 1.322x |
| MoE 256 / 8 | 942.35 | 470.52 | 1.999x / 2.006x | 1.833x |
| MoE 512 / 1 | 1904.64 | 892.45 | 2.134x / 2.132x | 2.134x |

For 1-input/32-output generation, dense decode rises from **720.10 to 791.29
tok/s**; MoE rises from **105.97 to 110.15 tok/s**. Graph replay does not remove
MoE's matrix execution bottleneck. Dense decode after a 256-token prompt is
approximately 5.3% slower (247.81 to 235.48 tok/s), despite a 9.0% total-request
speedup. Do not characterize this profile as an across-the-board decode win.

Numerical reservation increases by **10,623,024 bytes (10.13 MiB)** for dense
and **67,876,016 bytes (64.73 MiB)** for MoE. The latter rises from
18,775,408,112 to 18,843,284,128 bytes. These counts include owned graph control
and sampling buffers, but exclude driver-owned graph/JIT allocations and host
model/tokenizer copies. The paired runner explicitly allows at most 68 MiB extra;
the baseline guard is unchanged.

Raw records, commands, binary/model/source hashes, and GPU telemetry:
[paired report](../results/cuda-runtime-m7/paired-f32-tiled-graph512/paired.json).
Two opposite-order windows are not a confidence interval. Short-request telemetry
is sampled at 200 ms and is not a per-token trace.

## Quality and validation

No numerical gate was relaxed: finite full vectors, max absolute error <=0.01,
cosine >=0.99999, identical argmax, and exact generated IDs where compared.

- F32 and F32 tiled pass all six prefixes on both real models at all four swept
  chunk sizes. Dense outputs are bitwise equal to M6; worst MoE error is
  **0.00003076**.
- The final preserved precision binary passes 12 full-vector M6 comparisons and
  five exact matched-F32 external prefixes (1/7/19/46/256 tokens). Worst external
  error is **0.00005722**. The previously divergent 19-token input is included.
- Final ABBA 512-token vectors also pass: dense bitwise equal, MoE max error
  **0.00001073**. These bounded fixed-prefix checks are not broad task evaluation.
- TF32x3 tiled fails the 256-token MoE vector at all four chunk sizes:
  **0.08682251** versus M6, **0.08683252** versus the matched external oracle.
  Matching generated IDs and high cosine do not override the failed error gate.
  CLI help and README explicitly disclose the failure.
- Strict Clippy, default tests, all-feature ordinary tests, and all 16 Python
  benchmark tests pass.
- CUDA regression suites: **11 primitive + 8 integration tests pass**.
- Compute Sanitizer memcheck, racecheck, and synccheck each pass three staged
  projection tests and two captured-decode integration tests, with zero reported
  errors/hazards. Sanitized code is identical to final execution code; the only
  later source change marks the approximate mode's failed gate in CLI help.

See [accepted comparisons](../results/cuda-runtime-m7/accepted-comparison.json),
[graph/oracle experiments](../results/cuda-runtime-m7/v4-quality.json), and
[final checkpoint](../results/cuda-runtime-m7/checkpoint.json).

## Fair external controls

Pinned optimized llama.cpp revision:
`6b01a4767f8d3d0c416c410a49861333965c0982`.
Fresh all-GPU controls use both F32/F16 KV and batch/chunk 128 and 512, flash
attention, no prefix reuse, the same prompt IDs, three warmups and seven samples.
The larger-batch retune matters; retaining only batch 128 would handicap it.

For MoE with F16 KV at batch/chunk 512, llama.cpp measures **25.99 / 32.52 ms**
prefill for 256/512 tokens and **364.49 tok/s** short-context decode. M7 measures
470.52 / 892.45 ms and 110.15 tok/s. Rust remains about **18x / 27x slower in
prefill** and **3.3x slower in decode** on these workloads.

Rust reports synchronized numerical timing; external controls report server
internal timing and HTTP latency separately. Optimized external packed arithmetic
quantizes activations and is not the independent engine's F32 arithmetic. The
matched-F32 CPU oracle is correctness-only, never a speed baseline.

Both control sizes and raw responses are retained in
[A2](../results/cuda-runtime-m7/paired-f32-tiled-graph512/a2/summary.json) and
[retuned 512](../results/cuda-runtime-m7/external-retuned512/summary.json).

## Reproduce the accepted profile

With the already provisioned model and CUDA libraries:

```sh
export LD_LIBRARY_PATH="$CUDA_ROOT/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
results/cuda-runtime-m7/accepted-runtime generate --backend cuda \
  --model models/benchmark-qwen3/gguf/Qwen3-30B-A3B-Q4_K_M.gguf \
  --context 1024 --batch 512 --prefill-chunk 512 \
  --cuda-matmul f32-tiled --cuda-decode-graph \
  --prompt 'The capital of France is' --tokens 16

python3 scripts/benchmark_cuda_paired.py \
  --baseline results/cuda-runtime-m6/accepted-runtime \
  --candidate results/cuda-runtime-m7/accepted-runtime \
  --candidate-matmul f32-tiled --candidate-decode-graph \
  --candidate-batch 512 --candidate-chunk 512 \
  --allowed-reservation-increase 71303168 --llama \
  --cuda-root "$CUDA_ROOT" --warmups 3 --repetitions 7 \
  --output-dir results/cuda-runtime-m7/paired-NEW_RUN
```
