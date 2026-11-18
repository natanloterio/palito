# Local inference Rust: independent GGUF engine

The default runtime is now an independent Rust numerical engine. Cargo does not
build, load, or link llama.cpp, GGML, or the native bridge. llama.cpp is only an
external benchmark reference. The earlier shared-kernel control-plane code is
retained under [benchmarks/legacy](benchmarks/legacy), not compiled. Its historical
performance results do not measure this independent engine.

See the [architecture reference](../docs/architecture.md) for execution flows,
ownership/resource boundaries, performance evidence, and design extension points.
The [standalone master Mermaid chart](../docs/architecture.mmd) is editable.

## Default CPU execution scope

- One dense, decoder-only `llama` architecture model, one sequence, CPU execution.
- GGUF v3 with an explicit tensor allowlist: F32, F16, Q5_0, Q8_0, Q4_K, Q6_K.
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

## Independent CUDA and Qwen3-MoE

The optional `cuda` feature uses our [CUDA kernels](src/cuda/kernels.cu), not
llama.cpp, GGML, cuBLAS, Torch, or KTransformers numerical code. `cudarc` provides
only dynamically loaded CUDA driver/NVRTC bindings. Cargo builds need no CUDA
toolkit or compiler; execution currently requires Linux, an NVIDIA GPU/driver,
and CUDA 13 NVRTC libraries on `LD_LIBRARY_PATH`.

- `generate`, `bench`, and `logits` accept `--backend cuda`; the default remains
  `--backend cpu`. Missing hardware/libraries and unsupported options are errors,
  never an implicit CPU fallback.
- Supports the dense `llama` layout used by SmolLM2, plus Qwen3-MoE's wider queries,
  per-head Q/K RMSNorm, NeoX/split-half RoPE, and rank-three expert matrices.
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
- Dense `llama` and Qwen3-MoE prefill are layer-wise batched: packed projections
  reuse weights across prompt tokens, and each query sees only its causal KV prefix.
  Internal chunks contain at most 512 tokens; larger public batches are split.
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
  unchanged. See the [full-model quality evidence](results/cuda-runtime-m8-int8-model/quality-summary.json)
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
target/release/local-inference generate --backend cuda \
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

On the RTX 5090, the [completed validation](results/rust-cuda-validation-run2/validation.json)
passed six full-logit comparisons: 49,152 logits per dense case and 151,936 per
MoE case. Maximum absolute errors were 0.000290 (dense) and 0.000049 (MoE);
greedy IDs matched. All 21 Qwen tokenizer cases matched HF IDs and decoded text,
including composed/decomposed Unicode. Exact-context generation, chunk changes,
and repeated reset also matched the independent CPU outputs.

KTransformers remains a distinct whole-request adapter; SGLang/KT, not this
Rust CUDA backend, owns its kernels and KV. Both are available from the same CLI:

```sh
target/release/local-inference remote \
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

The [completed four-deployment comparison](results/rust-cuda-moe-vs-kt-llama-run1/comparison.json)
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

The subsequent [CUDA optimization milestone](benchmarks/cuda-optimization.md)
removed intermediate prompt output projections/readback and parallelized stable
MoE routing. On the same RTX 5090, dense 46/256-token prefill improved 1.34x/1.35x,
MoE improved 1.67x/1.59x, and MoE decode rose from 58.72 to 88.44 tokens/s.
Numerical reservations stayed unchanged; independent CPU-logit gates and exact
old/new 512-token logits passed. The [checkpoint](results/cuda-optimization-m1/checkpoint.json)
combines seven completed cases with an isolated retry after an unresolved
kernel-blocked process-lifecycle failure. Rust still does **not** beat the tuned
all-GPU llama.cpp baseline; the report preserves variability and cleanup limits.

The [second kernel checkpoint](results/cuda-optimization-m2/checkpoint.json)
specializes packed matrix products and removes redundant reduction barriers.
Fresh paired measurements improved MoE decode from 88.73 to 102.73 tokens/s
and short/256-token prefill latency by approximately 10-12%, with unchanged
numerical reservations and exact old/new logits. Reverse-order confirmation
passed, but the strict 10% prefill threshold is borderline under pooling;
dense 512-token results remain variable. See the
[experiment log](benchmarks/cuda-optimization.md) for all outcomes and limitations.

The [multi-token matrix prototype](results/cuda-prefill-prototype-m3/checkpoint.json)
reuses packed weight decoding across four prompt tokens. Isolated 16/64-token
products measured 2.43-2.71x faster than an independent-row batched GPU baseline,
with exact GPU outputs and clean memory/race/synchronization checks. M3 kept it
test-only because single-token latency regressed. The subsequent dense prefill
integration uses it only for multi-token tiles and preserves decode; see the
[integration gates and results](benchmarks/cuda-optimization.md).
The [clean-boot real-model controls](results/cuda-prefill-prototype-m3/clean-boot-control/summary.json)
refresh both Rust and pinned llama.cpp measurements after recovery; Rust still
trails, and a pre-existing MoE decode token mismatch remains disclosed.

The [dense batched-prefill checkpoint](results/cuda-dense-prefill-m4/checkpoint.json)
integrates the tile into actual inference: SmolLM2 46/256/512-token prefill
improved **8.62x / 11.46x / 13.01x** versus a fresh sequential Rust control.
All eight regression sequences and both 512-token logit vectors matched exactly.
The tradeoffs are **7.8 MiB extra dense workspace** at context 1024 / batch 128
and roughly 80 ms more dense startup/JIT time. Decode stays within the 5%
regression allowance; MoE was still sequential at M4. Fresh tuned llama.cpp controls
are still faster. See the [experiment log](benchmarks/cuda-optimization.md).

The [M5 grouped-prefill and fused-attention iteration](benchmarks/cuda-runtime-m5.md)
adds expert grouping, exact-arithmetic on-chip attention, and opt-in kernel caching.
The final fresh matrix improves MoE 46/256/512-token prefill **1.91x / 2.70x / 3.25x**
over M4, and dense 256/512-token prefill **1.19x / 1.39x**. Seven-trial dense
startup falls from **544 to 184 ms** with a warm compilation cache. Ten actual GPU
tests, independent numerical/tokenizer gates, all three sanitizers and exact
old/new 512-token logits pass. Eight-token projection tiles were slower and remain
test-only. **Not all M5 performance targets passed**: short MoE missed 2x, and
decode-heavy latency exceeded 5% in fixed order but passed in reverse order.
At M5 the external MoE numerical discrepancy remained unresolved; llama.cpp remains
faster. See the [checkpoint](results/cuda-runtime-m5/checkpoint.json) for all evidence.

The [M6 profiling and tensor-core iteration](benchmarks/cuda-runtime-m6.md)
adds paired benchmark controls, calibrated layer/router tracing, and an explicit
`--cuda-matmul tf32x3` experiment. On the measured 256/512-token MoE workload it
delivers **1.18-1.19x prefill throughput**, passing 12 bounded full-logit gates
(worst error 0.000508). It remains approximate and experimental, **not recommended
for dense or tiny prompts**; default F32 and its CUDA sources remain unchanged.
Slower two-row projections and regressive decode fusion were rejected.
Single-product `tf32` fails quality gates. All 12 GPU tests and three sanitizers
pass; external MoE equivalence and the broad performance target remained unmet at M6.
See the [M6 checkpoint](results/cuda-runtime-m6/checkpoint.json).

The [M7 staged-expert and graph iteration](benchmarks/cuda-runtime-m7.md) validates
an explicit `f32-tiled`, batch/chunk 512, graph-enabled profile. Paired MoE
256/512-token prefill improves **2.00x / 2.13x** versus M6; short-context decode
improves from **105.97 to 110.15 tok/s**. It uses 64.73 MiB more MoE workspace,
and dense decode after 256 input tokens regresses about 5.3% even though total
request latency improves. Full-vector gates, exact generated IDs, 19 GPU tests,
and three sanitizers pass. The faster `tf32x3-tiled` experiment **fails quality**.
Retuned llama.cpp remains substantially faster; defaults are unchanged.

The [M8 implementation plan](benchmarks/cuda-runtime-m8-plan.md) separates
F32-quality structural optimization from experimental arithmetic. Packed F32
decode, measured-shape reference-order dense tiles, and count-aware grouped
register prefill are enabled only by the existing opt-in `f32-tiled` profile.
The earlier serial-K dense register candidate failed full-model quality and
remains test-only, as do four-product TF32 controls. There are no new production
precision modes.
The [first-wave paired results](results/cuda-runtime-m8-wave1/paired-public-profile/paired.json)
show Qwen3 short-context decode improving from about **110 to 165 tok/s** with
graphs, without activation quantization or an owned-buffer increase. That wave
left prefill-512 at about 893 ms. Full-vector/ID checks, held-out prompts, and all three
sanitizers pass; default F32 kernels remain unchanged. This is not llama.cpp parity.

The [second-wave paired results](results/cuda-runtime-m8-wave2/paired-public-profile/paired.json)
reduce MoE 256/512-token prefill from **470/894 ms to 383/682 ms**
(**1.23x/1.31x**) against that accepted first wave. Decode stays about 165 tok/s;
dense performance is unchanged. All 88 full-vector quality comparisons are
bitwise equal, generated IDs match, and all three sanitizers pass.
The 512-token MoE profile reserves **10,656 extra bytes**, counted in admission
and allocation. Its profile still attributed 92.5% of kernel time to projections.

The [third-wave checkpoint](results/cuda-runtime-m8-wave3/checkpoint.json) accepts
reference-order dense tiles and count-aware expert computation.
[Fresh paired results](results/cuda-runtime-m8-wave3/paired-public-profile/paired.json)
reduce MoE 256/512-token prefill from **384/683 ms to 317/562 ms**
(**1.21x/1.21x**) against the accepted second wave. Decode stays about 165 tok/s;
owned-buffer memory is unchanged. All 128 full-vector comparisons are bitwise
equal, IDs match, and all three sanitizers pass. The new dense tiles preserve the
legacy accumulation/reduction order rather than relaxing numerical gates.
Gate/up fusion was bit-exact but slower and remains prototype-only.
Its profile attributed 91.1% of kernel time to projections, including
298 ms of grouped experts and 203 ms of tiled dense projections. Expert
decode/reuse and resource-balanced fusion were the next experiments.

The [fourth-wave checkpoint](results/cuda-runtime-m8-wave4/checkpoint.json) accepts
coalesced K=32 expert tiles. Fresh public ABBA reduces MoE 256/512-token prefill
from **317/564 ms to 291/528 ms** (**1.089x/1.068x**) against accepted wave3.
All **152 full-vector comparisons are bitwise identical**, generated IDs match,
memory is unchanged, and decode remains about 165 tok/s. Tests and all three
sanitizers pass; default F32 PTX is unchanged. Smaller fused gate/up tiles are
correct and faster than the old fusion, but still slower than the new separate
projections, so fusion remains disabled. A fresh profile reduces expert kernels
from 298 to 259 ms; projections still account for 90.4% of kernel time.

The [fifth-wave checkpoint](results/cuda-runtime-m8-wave5/checkpoint.json) accepts
eight-way expert-loop unrolling, not the wider or fully active tile prototypes.
Fresh ABBA reduces MoE 256/512-token prefill from **292/528 ms to 277/507 ms**
(**1.053x/1.041x**) against accepted wave4. All **176 full-vector comparisons
are bitwise identical**, IDs match, memory is unchanged, and short-context
decode remains about 165 tok/s. Tests and all three sanitizers pass.
Expert kernel time falls from 259 to 238 ms; dense tiles remain about 204 ms.
The planned 450-ms prefill goal remains unmet; there is no new llama.cpp
comparison or parity claim.

The [sixth-wave checkpoint](results/cuda-runtime-m8-wave6c/checkpoint.json) accepts
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
[primitive results](results/cuda-runtime-m8-residual-primitives/summary.json)
and [research checkpoint](results/cuda-runtime-m8-residual-primitives/checkpoint.json).

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
[primitive results](results/cuda-runtime-m8-precise3-primitives/checkpoint.json),
not final-logit or end-to-end admission. Full three-component IMMA matrices are
evaluated separately in the [MMA corpus](results/cuda-runtime-m8-mma3-matrices-v3/summary.json).
All84 projections win, with **2.5715x aggregate speedup including conversion and
grouping**. This passes the matrix gate, not the full-model gate.
The explicit `--cuda-matmul residual-mma3-research` mode now connects that frozen
kernel to prefill batches of at least32 tokens (initially256); shorter or unsupported
projections, including decode projections, retain the accepted F32-tiled path. It requires compute
capability8.0 and adds a separately accounted, reusable11.25MiB workspace at
batch512 on the MoE model, without allocating the rejected two-component
workspace. The [full-model gate](results/cuda-runtime-m8-mma3-model/quality-summary.json)
passes all594 full-vocabulary comparisons across18 cases, including four
predeclared broader heldouts: worst error0.001194, minimum cosine0.9999999989,
exact teacher-forced argmax and32-token generated IDs. Capacity/error/reset,
workspace reuse,68 ordinary tests, strict Clippy/fmt and three scoped real-model
sanitizers pass.
The [public ABBA](results/cuda-runtime-m8-mma3-acceptance/performance-summary.json)
reduces256-token prefill from272.25ms to115.15ms (**2.36x**) and512-token prefill
from498.34ms to200.94ms (**2.48x**). Targeted complete requests improve1.83x/2.48x;
decode and46-token prefill remain effectively unchanged. All measured IDs match
the accepted runtime, with no phase regression above5%. This is scoped opt-in
validation, not a production/default promotion.
The subsequent [attention improvement](results/cuda-runtime-m8-attention-normalized/performance-summary.json)
normalizes each shared score once instead of repeating its division for every
output dimension, only in the MMA3 profile. All594 vocabulary vectors and all
18 generated sequences remain bit-exact to admitted MMA3; four F32/F32-tiled
graph/non-graph PTX comparisons are byte-identical. No workspace is added.
ABBA improves512-token prefill another2.6% to195.55ms and256-token prefill to113.23ms.
Decode rises from170.45 to176.27token/s on the1/32 case,153.26 to168.75 after46
prompt tokens, and93.93 to132.84 after256. The latter complete requests improve14.3%.
See the [completed admission](results/cuda-runtime-m8-attention-normalized/public-checkpoint.json)
and [latest comparison](results/cuda-runtime-m8-attention-normalized/comparison.json):
llama.cpp still leads by4.31x/5.99x prefill and approximately1.96x short decode.
The [fresh head-to-head control](results/cuda-runtime-m8-head-to-head/comparison.json)
still shows llama.cpp ahead; matrix improvements do not establish otherwise.
The [new post-MMA comparison](results/cuda-runtime-m8-mma3-acceptance/comparison.json)
narrows the prefill gap to4.43x/6.14x at256/512 tokens versus llama.cpp F32 KV;
short decode is170.42 versus344.42token/s. The engines still differ in generated
IDs for the one-token/32-generated case, and llama controls are not ABBA.

The [small-prefix extension](results/cuda-runtime-m8-mma3-small-acceptance/checkpoint.json)
lowers only MMA3 eligibility to32 tokens; the rejected two-component mode stays
at256 and defaults remain unchanged. Actual32/46/128-token matrices pass the
1.5x conversion/grouping-inclusive gate in both development and frozen-binary
heldout runs. The baseline uses its real small-prefix dense kernel and tile8
expert grouping, not the larger-prefix kernels or candidate tile32 grouping.
All [990 vocabulary comparisons](results/cuda-runtime-m8-mma3-small-model/quality-summary.json)
across30 cases and all30 generated32-token sequences pass unchanged quality
gates. The18 prior cases, including four broader prompts, are retained;12 new
boundary cases reuse existing prompts rather than claiming new unseen content.
All528 unaffected prior snapshots and132 below-threshold fallback comparisons
are bit-exact. Actual batch capacities31/32/46/128/255 pass memory, full-context,
error/reset and exact-ID checks; three bounded sanitizers test the newly allocated
batch32 workspace. At batch32 the accounted increase is737,280 quantization bytes
plus12,096 group bytes; batch512 adds nothing over admitted MMA3.
[Public ABBA](results/cuda-runtime-m8-mma3-small-acceptance/performance-summary.json)
cuts46-token prefill128.87ms ->63.21ms (**2.04x**) and improves its complete
request **1.63x**, exceeding the predeclared10% gate in both orders. All measured
IDs match; no phase regresses more than5%. The256/512-token prefills remain
113.25/195.78ms and short decode176.30token/s. The
[fresh comparison](results/cuda-runtime-m8-mma3-small-acceptance/comparison.json)
still favors llama.cpp:4.76x/4.33x/5.99x prefill at46/256/512 tokens and roughly
1.95x short decode. These are scoped research results, not production/default
promotion or cross-engine numerical equivalence.

The subsequent [projection pipeline study](results/cuda-runtime-m8-mma3-projection-profile/study-summary.json)
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

The [block-affine arithmetic screen](results/cuda-runtime-m8-mma3-block-affine-development/summary.json)
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
The [explicit user-approved amendment](results/cuda-runtime-m8-mma3-block-affine-development/provenance-amendment.json)
permits this matrix-only screen with unchanged tensor, capture and admitted
output hash checks; it does not establish metadata-only differences or
whole-model equivalence. No model files or historical results were modified, and
no heldout/full-model/public run was used to excuse the block-affine rejection.
The current file (`398bf479...6aa86`) was subsequently approved for fresh
same-file comparisons only.

[Parallel expert grouping](results/cuda-runtime-m8-parallel-grouping/paired/paired.json)
is available with `--features cuda-parallel-grouping-research` and
`--cuda-matmul residual-mma3-research`. With 512 extra bytes of workspace on
Qwen3, it matches all 990 full-vocabulary snapshots and 30 independent
generations in the fresh quality run. It speeds up 512-token prefill by
2.7-3.0% across both ABBA orders, below the 10% package gate. The default
grouping path is unchanged. The final feature-gated build also matched the
measured candidate's 46-token smoke-test logits exactly.

The optional `cuda-packed-reduction-research` feature reduces packed F32
matvec synchronization in the same MMA3 profile without reordering additions.
In the [short-context decode ABBA](results/cuda-runtime-m8-packed-decode-trace/decode-abba-summary.json)
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

`cuda-mma3-wide-rows-research` is an opt-in 64-row MMA3 tile for grouped
expert projections at exactly 512 prefill tokens. It shares activation staging
across more output rows without changing the K accumulation order. Dense
projections, smaller chunks, and default builds retain their existing dispatch.
The separate `cuda-mma3-dense-wide-rows-research` feature also enables the
tile for eligible dense MMA3 projections at 512 tokens only.

`cuda-attention-query-pair-research` pairs adjacent causal queries for
512-token MMA3 prefill with 128-wide heads and bounded fused attention.
Each query keeps its original score and value reduction order. Other attention
shapes, including single-token decode, retain their prior dispatch.

Two additional MMA3 opt-in features preserve the original reduction order:
`cuda-block-reduction-research` for shared 256-thread sums and
`cuda-attention-reuse-research` for fused attention query reuse and softmax
maximum reduction. With both enabled alongside parallel grouping and packed
matvec reduction, all 990 full-vocabulary snapshots and 30 independent
generations match the fresh Rust baseline. Neither feature changes the
default dispatch.

See the [full-matrix results](results/cuda-runtime-m8-residual-matrices/summary.json)
and [checkpoint](results/cuda-runtime-m8-residual-matrices/checkpoint.json).

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

The subsequent [matched-precision MoE investigation](benchmarks/moe-numerical-fix.md)
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
target/release/local-inference bench \
  --model models/SmolLM2-135M-Instruct-Q4_K_M.gguf \
  --input-tokens-file input.json --tokens 8 --warmups 1 --repetitions 3
```

The downloader fetches the public Apache-2.0 SmolLM2-135M-Instruct GGUF
(105,454,432 bytes) at the immutable revision in
[fixtures/model.json](fixtures/model.json), then verifies its size and SHA-256.
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
[scripts/benchmark.py](scripts/benchmark.py) is retired.

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
target/release/local-inference generate \
  --model models/SmolLM2-135M-Instruct-Q4_K_M.gguf \
  --prompt 'The capital of France is' --tokens 16

# These commands read metadata without expanding model weights:
target/release/local-inference tokenize \
  --model models/SmolLM2-135M-Instruct-Q4_K_M.gguf --prompt 'Hello, world!'
target/release/local-inference detokenize \
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
[text validation report](results/rust-tokenizer-smollm2-run1/validation.json).

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
target/release/local-inference bench \
  --model models/SmolLM2-135M-Instruct-Q4_K_M.gguf \
  --prompt 'The capital of France is' --tokens 16 \
  --memory-limit-bytes 150000000 \
  --prefix-cache-bytes 8388608 --speculative-window 4

# Independent expanded/scalar/sequential control for differential validation:
target/release/local-inference bench \
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
0.009. See the [performance report](results/rust-independent-smollm2-run2/comparison.json)
and [numerical validation](results/rust-independent-logits-run2/validation.json).

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

See the [packed comparison](results/rust-packed-smollm2-run2/comparison.json) and
[policy/numerical/memory validation](results/rust-cpu-policies-run1/validation.json).
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

[Backend](src/backend.rs) contains no llama.cpp/GGML types. The generic
[Executor](src/lib.rs) owns policy, request admission, prefill chunking, and token
positions. The [Rust CPU backend](src/cpu.rs) implements numerical token stepping.
There is no `llama-cpp` Cargo feature or native build script.

The optional [KTransformers adapter](src/ktransformers.rs) targets the **current
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
The [comparison report](results/kt-vs-llama-qwen3-run4/comparison.json) has the
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

See the [literature review](../literature_review/review.md) for the evidence and
limitations behind this order. A successful smoke benchmark establishes a
baseline; it does not establish a speedup or generalize to larger models.
