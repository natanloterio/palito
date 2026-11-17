# M8 plan: closing the llama.cpp execution gap

Status: implementation proposal, not a performance result. No runtime or precision
changes are included in this document.

## 1. Recommendation

Build a small, shape-aware matrix engine around the existing independent GGUF
decoders, rather than incrementally widening the current scalar tiles.

Maintain two separate tracks:

1. **F32-quality structural optimization:** register tiling, packed-load reuse,
   count-aware expert execution, gate/up fusion, and specialized decode. This is
   the production-oriented track.
2. **Higher-throughput arithmetic research:** diagnose the failed TF32x3 path,
   then test residual-corrected INT8 against a higher-component floating-point
   control. This remains experimental unless all numerical and performance
   gates pass.

Reprofile between stages. Matrix execution is the first priority, but attention
and grouping must improve before a credible prefill-parity claim is possible.
Do not promise that F32 SIMT arithmetic alone will beat llama.cpp's INT8 path.

## 2. Evidence and realistic ceilings

Baselines:

- [Accepted M7 report](cuda-runtime-m7.md).
- [Accepted comparison](../results/cuda-runtime-m7/accepted-comparison.json).
- [Fresh profiling analysis](../results/llama-gap-investigation/analysis.json),
  with raw Nsight reports and SQLite exports in the same directory.
- External revision: `6b01a4767f8d3d0c416c410a49861333965c0982`.
- Hardware: RTX 5090, SM120; Qwen3-30B-A3B Q4_K_M and SmolLM2-135M Q4_K_M.

Unprofiled Qwen3 measurements:

| Workload | Accepted M7 | Retuned llama.cpp |
|---|---:|---:|
| Prefill 256 | 470.52 ms | 25.99 ms |
| Prefill 512 | 892.45 ms | 32.52 ms |
| Short-context decode | 110.15 tok/s | 364.49 tok/s |

These are different activation-arithmetic paths. The external F32-KV control
does not make llama.cpp's Q8 activation arithmetic F32. Retain the report's
timing-boundary caveats; do not use the correctness-only CPU oracle as a speed
baseline.

Fresh M7 prefill-512 profile, per prefill:

| Component | GPU kernel time | Share |
|---|---:|---:|
| Grouped expert projections | 546.73 ms | 61.38% |
| Dense projections | 294.01 ms | 33.01% |
| Attention | 36.11 ms | 4.05% |
| Expert grouping | 9.27 ms | 1.04% |
| Other | 4.55 ms | 0.51% |

The profiled execution is almost completely serial. Holding non-matrix work
fixed gives the following illustrative Amdahl calculation:

| Matrix improvement | Remaining serialized GPU work |
|---|---:|
| 2x | 470.29 ms |
| 4x | 260.11 ms |
| 10x | 134.00 ms |
| Infinitely fast matrices | 49.92 ms |

These are kernel-time estimates, not measured request latencies or predictions.
Attention alone currently takes more GPU time than the external engine's entire
unprofiled prefill request. Matrix optimization is necessary but insufficient.

Decode findings:

- Our matrix-vector kernels account for 7.48 ms/model step, 81.62% of kernel time.
- Graph replay is already working; weights are already resident on the GPU.
- llama.cpp uses packed integer dot products and overlaps independent work.
- It launches about 1,110 kernels/step versus our 965. Reducing total kernel
  count is not, by itself, the explanation or the goal.
- External summed kernel time exceeds active interval union because of overlap.
  Evaluate critical-path time, not just summed kernel durations.

## 3. What to borrow, and where we might do better

| Design direction | Useful lesson | Workload-specific opportunity and limitation |
|---|---|---|
| llama.cpp MMQ/MMVQ | Integer tensor-core/DP4A arithmetic, packed scales, shared activation conversion, fusion, QKV concurrency | Reproduce the execution principles independently. Its single Q8 activation approximation is not automatically acceptable under our quality policy. |
| Marlin-style weight-only GEMM | Register reuse, interleaved unpack/compute, pipelined loads, hardware-friendly weight layout | Apply these ideas to actual Q4_K/Q6_K semantics and F32 reconstruction. Marlin's formats, FP16 inputs, and some reduction choices are not drop-in substitutes. |
| Shape-specialized F32 kernels | Avoid activation quantization and exploit independent accumulators | Could be competitive at small batches where conversion and generic scheduling matter. Whether it beats packed INT8 is unproven. |
| Count-aware expert work | Match tiles to actual per-expert assignment counts | Our model has fixed dimensions and 128 experts, but routing remains dynamic. Specialize dimensions, never assume fixed expert choices. |
| Residual-corrected INT8 | Use fast integer products for multiple activation components | Potentially a better speed/accuracy tradeoff than either scalar F32 or single-pass Q8. More products and correction traffic may erase the speed advantage. |
| Multi-component floating-point MMA | Recover more input information than a single low-precision conversion | A control for the INT8 experiment, not evidence that the existing TF32x3 path is safe or that tensor-core accumulation is bitwise F32. |

Fusion and shape specialization are not novel inventions, and llama.cpp already
uses several of them. The opportunity is a better measured design for these
shapes, this GPU, and our stricter numerical requirements, not a claim that the
reference engine overlooked all these techniques.

## 4. Nonnegotiable constraints

- Independently written Rust/CUDA numerical implementation. No llama.cpp/GGML
  build, link, or runtime dependency.
- Preserve default F32 behavior and explicit experimental-mode selection.
- Preserve GGUF Q4_K/Q6_K reconstruction, including scale/min corrections and
  rounding. Do not silently reinterpret Q4_K as native FP4.
- Keep `fmad: Some(false)` initially. Contraction, different reduction orders,
  and operand approximation must be identified and tested separately.
- F32-quality means passing the declared numerical contract, not necessarily
  bitwise-identical reductions. Reduced-precision operands remain explicitly
  identified even if a finite validation corpus passes.
- Keep graph/reset/error-poisoning behavior, format fallbacks, supported context
  limits, and full-logit versus sampling APIs intact.
- Preserve the 24 GiB owned-device limit and 2 GiB free-headroom policy. Initially
  budget at most 64 MiB additional persistent scratch over the chosen M7 profile;
  larger representations require a separate explicit memory decision.
- Count workspaces, repacked weights, graph controls, and concurrent buffers.
  Continue disclosing excluded driver/JIT/graph-internal allocations.
- No clock/power changes, GPU resets, unrelated process termination, dependency
  installation, model downloads, or assumptions of exclusive machine access.

### Actual SM120 resources

Read-only CUDA driver queries on the target device returned:

| Resource | Value |
|---|---:|
| SM count | 170 |
| Resident threads/SM | 1,536 = 48 warps |
| Registers/SM | 65,536 32-bit registers |
| Shared memory/SM | 102,400 bytes = 100 KiB |
| Opt-in shared memory/block | 101,376 bytes = 99 KiB |
| Resident blocks/SM | 24 |

The current generic Blackwell tuning guide lists 128 KiB shared memory/SM for
CC12.0, unlike this driver's 100 KiB. Use queried target limits for resource
calculations and save them in benchmark provenance. Static shared memory above
48 KiB is not a portable assumption; explicitly opt in when required.

SM120 does support TMA-based designs: NVIDIA's SM120 grouped example uses TMA
and device-side descriptor modification. However, that example is native FP4,
not GGUF Q4_K. Conversely, NVIDIA's BF16x9 example targets SM100a. Do not transplant
its `tcgen05`, two-SM scheduling, or other architecture-specific machinery into
the SM120 runtime. Require documentation, successful target compilation, and a
correctness smoke test for every new instruction path.

## 5. Implementation stages and dependencies

### M8.0: representative inputs and numerical diagnosis

**Changes**

- Extend [grouped_bench.rs](../src/cuda/grouped_bench.rs) to include 512 tokens,
  dense projection shapes, and replayable real routing distributions.
- Reuse [prefill.rs](../src/cuda/prefill.rs) and its existing `Harness`, packed
  fixtures, decoder checks, guards, tails, and CPU comparisons.
- Add an opt-in diagnostic capture to the CUDA backend, aligned with the stages
  exposed by [reference.rs](../src/cuda/reference.rs). Capture selected layers,
  positions, projection inputs/outputs, routing logits, and expert IDs.
- Store bounded, local binary fixtures plus tensor metadata, shapes, hashes,
  baseline mode, and routing counts. Do not retain full-model activation dumps.
  Diagnostic copies/synchronization must never run during timed benchmarks.

**Questions to settle**

1. What are the per-layer expert-count histograms, empty/tiny groups, upper
   quantiles, padding waste, and row-tile work counts at 64/128/256/512 tokens?
   The mean at 512 is 32 assignments/expert, not evidence of uniform routing.
2. Does the failed TF32x3 tiled mode first diverge at unpacking, a projection,
   accumulation, normalization, or routing? Compare identical captured inputs,
   not hidden states that have already diverged.
3. Which matrix shapes dominate actual time, including Q, attention output,
   gate/up/down, and the final vocabulary projection?

**Exit:** reproduce immutable M7 results, produce a fixed shape-weighted corpus,
and record the first observed approximate-path divergence. This diagnosis may
continue alongside structural kernel work; it does not block the F32 track.
Do not simply assume that the omitted residual product explains the failure.

### M8.1: replace weak F32 tile execution

**Depends on:** representative fixtures from M8.0.

**Files:** [prefill.cu](../src/cuda/prefill.cu),
[kernels.cu](../src/cuda/kernels.cu), [batch.rs](../src/cuda/batch.rs),
[prefill.rs](../src/cuda/prefill.rs).

Start with Q4_K/Q6_K fast paths and existing fallback kernels for other supported
formats. Reuse decoder helpers rather than creating a second GGUF interpretation.

1. Implement register-tiled F32 output accumulation: several output rows and
   tokens per thread, multiple independent accumulators, and reusable fragments.
   The present one-output/one-accumulator serial chain is the baseline to replace.
2. Start with three bounded candidate CTA shapes, expressed as
   `(output rows, tokens, K tile)`:
   - `(64, 16, 64)`, 128 threads, eight output accumulators/thread.
   - `(32, 32, 64)`, 128 threads, eight output accumulators/thread.
   - `(64, 32, 32)`, 256 threads, eight output accumulators/thread.
   These are experiments, not predetermined winners. Include the current kernel
   in every comparison; test actual live registers and shared allocation.
3. Initially keep per-output K traversal and dequantization arithmetic unchanged
   where feasible. This isolates reuse/ILP from reduction-order changes.
4. Compare shared F32 weight tiles against packed shared staging followed by
   register unpacking. Vectorize aligned raw loads while retaining safe tails.
5. Add double buffering only to the winning layout. First prove an ordinary
   load pipeline; use `cp.async` or TMA only when target support, alignment,
   packed layout, and measured transfer cost justify their complexity.
6. If long reductions still dominate, test a fixed-order parallel-K variant
   separately. Avoid atomic floating-point accumulation and nondeterministic
   split-K output merging.

Do not begin with a new generic GEMM framework, broad autotuner, or whole-model
F32 expansion. Dense tiled kernels already regressed once; enable by measured
shape class, not by a global assumption that tiled must mean faster.

**Decision gate:** seek at least 1.5x improvement in the captured, time-weighted
matrix corpus, with numerical gates passing. If the bounded initial candidates
do not reach 1.2x, stop geometry tuning and investigate instruction/decode/reuse
cost before adding variants. Promote only shape classes that win end to end.

### M8.2: grouped experts and fused gate/up

**Depends on:** M8.0 routing distributions and a validated M8.1 tile core.

**Files:** [batch.rs](../src/cuda/batch.rs),
[prefill.cu](../src/cuda/prefill.cu), [cuda.rs](../src/cuda.rs),
[grouped_bench.rs](../src/cuda/grouped_bench.rs).

1. Map the proven compute core onto expert assignment lists. Keep an explicit
   assignment ID so scatter and the existing top-k combination order survive
   any scheduling changes.
2. Compare the current grouping with count/prefix/scatter construction. If it
   wins, build compact GPU work descriptors containing expert, assignment range,
   and output-row tile. No host routing readback or per-layer synchronization.
3. Compare a simple fixed assignment tile with a small count-aware policy:
   tiny groups use a narrow path; larger groups use the winning register tile.
   Measure padding and bucket-launch overhead, not just useful FLOPs.
4. Only if skew leaves a meaningful execution tail, test a bounded device work
   queue. It must read GPU-generated work counts without CPU launch decisions.
   Do not add a persistent scheduler merely because it sounds more advanced.
5. Fuse gate and up projections for matching expert/row tiles, sharing activation
   loads. Compute SwiGLU in the epilogue and avoid writing both intermediate
   tensors when downstream consumers do not require them.
6. Measure fusion separately: two accumulator sets may reduce occupancy enough
   to lose. Preserve the existing activation function and F32 rounding behavior;
   do not substitute a faster approximate sigmoid.

**Decision gate:** retain each scheduling/fusion change only with a real-model
benefit; target at least 15% further MoE prefill improvement over M8.1. A primitive
win that disappears after grouping, scatter, or graph overhead is not sufficient.

### M8.3: dedicated F32 decode, then concurrency

**Depends on:** M8.0 decode fixtures. Can proceed independently of M8.2.

**Files:** [kernels.cu](../src/cuda/kernels.cu),
[decode.rs](../src/cuda/decode.rs), [cuda.rs](../src/cuda.rs).

1. Build Q4_K/Q6_K single-token kernels around vectorized packed block loads,
   register unpacking, adjacent-row reuse, and independent partial sums. Keep
   actual decoded values F32. Test quantized metadata placement and broadcasts
   locally; the previously regressing global metadata-broadcast change is not
   a valid shortcut.
2. Benchmark dense and expert shapes separately. Do not force a prefill CTA
   through the single-token path or infer decode speed from GEMM results.
3. Test joint gate/up decode while retaining F32 expert-combination ordering.
4. After individual kernels improve, compare sequential QKV against explicit
   dependency-aware concurrency. Q/K/V share the normalized input, but attention
   must wait for all required outputs; consumers and buffer reuse need events.
   Alternatively, benchmark one combined QKV projection. Choose from evidence,
   not a permanent menu of equivalent modes.
5. Compare ordinary streams and captured graph dependencies. Preserve allocation
   lifetimes through replay and destruction, reset behavior, error reporting,
   poisoning, and memory accounting. Capture must not conceal missing dependencies.

**Decision gate:** target at least 15% higher short-context model decode throughput
than the current accepted profile, with no hidden longer-context phase regression.
Judge overlap from the active timeline and wall time, not kernel sums. This work
must remain safe to fall back to the existing graph/non-graph implementations.

### M8.R: bounded tensor-core accuracy research

**Depends on:** M8.0 captures and numerical diagnosis. It is not a prerequisite
for shipping validated structural improvements.

**Files:** [tensorcore.cu](../src/cuda/tensorcore.cu),
[prefill.rs](../src/cuda/prefill.rs),
[grouped_bench.rs](../src/cuda/grouped_bench.rs).

#### Candidate A: residual-corrected integer products

For each activation block, investigate a decomposition of the form
`x ~= s0*q0 + s1*q1`, where `q1` quantizes the residual after reconstructing `q0`.
Use integer dot products for both components and combine their scaled results
in F32. Prepare components once per reusable input, not independently for every
selected expert or gate/up branch.

Proof obligations before a production-sized kernel:

- Derive the actual Q4_K affine dot product, including group scales, minima,
  activation sums, and correction terms. Q6_K needs its own signed/group-scale
  treatment. Model the existing decoder's rounding as well as real-number algebra.
- Bound integer accumulators and choose safe scaling, saturation, zero-block,
  subnormal, and nonfinite handling consistent with existing error semantics.
- Measure reconstruction and dot-product error on captured inputs and cancellation
  fixtures before testing a model.
- Count conversion, scatter, second dot product, metadata, correction, and F32
  merge costs. Two fast products are not automatically faster than a good F32
  kernel, and may be slower than llama.cpp's single-component path.

Multiple components are still approximate. A routing-margin check cannot certify
final logits, nor can keeping the router itself F32 undo prior hidden-state errors.
Do not use heuristic selective correction as an unannounced correctness fallback.

#### Candidate B: higher-component floating-point control

First use the existing TF32 machinery to compare its current three-product
evaluation with a four-product control including residual-times-residual and
separately accumulated correction terms. This is a diagnostic experiment, not
an assumption that a fourth product fixes all errors.

If needed, evaluate a small BF16 multi-component dot-product prototype based on
published decomposition work, using only instructions verified for SM120.
The SM100a CUTLASS BF16x9 example is conceptual evidence, not a compatible kernel.
Operand reconstruction and full dot-product rounding must be evaluated separately;
even exact input decomposition does not imply bitwise-equivalent F32 accumulation.

**Experiment limit:** test reconstruction/dot primitives before implementing
complete grouped kernels. Compare at most two new arithmetic designs initially;
advance only the best measured accuracy/cost tradeoff.

**Promotion gate:** unchanged full-model quality gates on development and held-out
cases, at least 1.5x matrix-corpus speedup over the improved F32 implementation,
and a meaningful end-to-end benefit (initial target: 10%). Otherwise retain a
documented failed experiment, not a new default or weaker acceptance threshold.

### M8.4: attention and the remaining non-matrix floor

**Depends on:** a new profile after matrix improvements; no need to wait for M8.R.

**Files:** [prefill.cu](../src/cuda/prefill.cu),
[kernels.cu](../src/cuda/kernels.cu), [batch.rs](../src/cuda/batch.rs),
[decode.rs](../src/cuda/decode.rs).

- Reuse K/V tiles across adjacent prefill queries and GQA query heads; test
  query/key tiling rather than repeating scalar cache reads.
- Investigate F32 online-softmax accumulation while preserving causal masking,
  position offsets, partial tiles, and the existing F32 KV storage.
- For decode, reuse each K/V read across the associated query heads and consider
  compatible normalization/RoPE/cache-write fusion. Do not assume a matrix tile
  contains a complete head if the proposed epilogue needs head-wide reductions.
- Retain current fallbacks for unsupported dimensions/contexts.
- Revisit grouping/normalization/sampling only at their newly measured shares.

**Decision gate:** target at least 2x attention-kernel improvement at 512 tokens
and 10% end-to-end prefill improvement when attention has become a major share.
Do not lower KV precision merely to improve the comparison.

## 6. Layout changes and memory economics

A lossless packed-layout permutation is worth a bounded experiment after the
compute access pattern is known:

- Preserve original integer codes and scale/min bit patterns.
- Verify reconstructed F32 weights bitwise against the existing decoder,
  including extreme half scales and boundary blocks.
- If worthwhile, replace the device representation at upload rather than keeping
  both full layouts. Keep model files immutable and CPU reference data independent.
- Measure upload/repack time, host peak memory, device padding, and steady-state
  bandwidth effects. Do not claim a throughput win without disclosing startup cost.

Predecoding scales or expanding Q4 codes to INT8 increases memory traffic/storage
and needs a complete byte calculation. F16 expansion is not lossless F32 decoding.

Full F32 expansion of just the expert gate/up/down weights requires:

`48 * 128 * 3 * 2048 * 768 * 4 = 115,964,116,992 bytes`

That is about 108 GiB, before attention, embeddings, KV, and workspaces. Reject it
as a whole-model strategy. A bounded hot-expert cache is deferred until routing
reuse and admission/eviction costs justify it; eight selected experts/token is
not evidence of a useful cross-layer cache hit rate.

## 7. Acceptance protocol

### Numerical and behavioral gates

Freeze the validation manifest before tuning:

- Existing M6/M7 comparison cases, the divergent 19-token prefix, and matched-F32
  external-oracle prefixes remain mandatory.
- Include lengths around tile/chunk boundaries and 46/256/512-token workloads.
- Add a small development corpus and at least four separately held-out prompts
  covering prose, code, multilingual text, and a route-sensitive/long input.
  Select teacher-forced positions in advance and include at least 32 generated
  tokens per case where context/EOS permits. Keep deterministic synthetic routing
  and cancellation fixtures in primitive tests, not as substitutes for model tests.
- Compare full-vocabulary logits at declared positions, not just the sampled ID.
  Preserve `max_abs <= 0.01`, cosine `>= 0.99999`, identical argmax, and exact
  generated IDs on declared cases. Report worst cases and route mismatches.
- Compare optimized F32 against immutable accepted baselines and the independent
  packed CPU reference; compare new arithmetic against both baseline and oracle.
  Passing against a changed implementation alone is circular.
- Preserve all six supported primitive formats/fallbacks, nondivisible tails,
  empty/tiny/skewed expert groups, sentinel guards, and assignment indexing.
- Exercise full-logit/sampling APIs, reset, graph replay, failure/poisoning,
  memory-budget rejection, and unsupported-shape fallbacks.
- Run serial GPU tests and memcheck/racecheck/synccheck for new memory-sharing,
  pipelining, fused, or concurrent paths. Run related ordinary tests, strict
  Clippy, and all-feature build before acceptance.

Identical IDs on six prefixes did not rescue TF32x3's 0.0868 maximum logit error;
they must not rescue a future failing mode either. Held-out success is evidence,
not a universal mathematical guarantee.

### Performance and attribution

Reuse [benchmark_cuda_paired.py](../scripts/benchmark_cuda_paired.py) and
[benchmark_cuda_iteration.py](../scripts/benchmark_cuda_iteration.py).
Use the [M7 reproduction procedure](cuda-runtime-m7.md) for existing commands.

- Preserve immutable M7 binaries, source manifests, model/prompt hashes, exact
  commands, device attributes, compiler/NVRTC identity, telemetry, and raw records.
- First time isolated candidates against their immediate parent; accept them only
  after real-model comparison against immutable M7.
- Run shape-weighted primitives and all dense/MoE 46/256/512 workloads. Include
  short decode and decode after the longer prompts, where M7 already had a phase
  regression. Report prefill, decode, first-token, and complete request separately.
- Use sequential GPU runs, at least three warmups and seven samples/window,
  balanced A1/B1/B2/A2 order, and separate profiling runs. Contention/thermal drift
  requires transparent reruns, not clock changes or selecting favorable samples.
- For promotion, require a repeatable improvement in both order contrasts.
  Investigate any greater-than-5% untargeted phase regression; reject it or retain
  an explicitly measured shape dispatch rather than hiding it in request averages.
- Reprofile after each accepted stage. Save summed time, active union, span,
  copies, node identities, and useful/padded work counts.
- No achieved-bandwidth, occupancy, or stall claims from Nsight Systems alone.
  The current Nsight Compute probe fails on `cuCtxGetDevice_v2`; this is a tool
  compatibility problem, not established counter-permission denial. Obtain a
  compatible profiler only through the normal approved tooling process.
- Retune fresh external controls for relevant batch/chunk sizes and disclose
  activation arithmetic, KV type, graph state, input IDs, generated work, and
  timing boundaries. If results approach parity, require matched scopes before
  a headline win; an external-only native measurement driver is preferable to
  quietly comparing different HTTP/CLI boundaries. Never link it into our runtime.
- Preserve losing experiments and quality failures alongside winners.

### Planning targets, not forecasts

First integrated checkpoint: aim for MoE prefill-512 at or below 450 ms and
short-context decode at or above 130 tok/s, without changing quality policy.
Missing these goals triggers reanalysis, not a weaker numerical contract.

Subsequent research targets are sub-100-ms prefill and 250 tok/s decode, followed
by the fresh external engine's measured envelope. None is established by current
data. A claim of beating llama.cpp requires repeated independent balanced runs
under declared, comparable timing and quality conditions, not crossing one old
median once.

## 8. First implementation slice

The first change set should contain only:

1. Bounded replay fixtures and real expert-count summaries in the existing harness.
2. One register-tiled Q4_K dense kernel, with the three candidate geometries
   benchmark-selectable in the harness, not three new permanent public CLI modes.
3. Decoder/tail/guard tests and a time-weighted comparison against the legacy path.
4. A real-model trial restricted to proven winning shapes, preserving the old path
   for everything else.

Q6_K, grouped integration, fusion, and arithmetic research follow evidence from
that slice. Do not start by globally enabling a larger tile, changing `fmad`,
loosening the precision gate, rewriting graph orchestration, or introducing a
new numerical dependency.

Dependency outline:

```text
M8.0 fixtures --------> M8.1 F32 tiles ----> M8.2 grouped/fused experts --+
       |                                                             |
       +-------------> M8.3 F32 decode/concurrency -------------------+--> profile
       |                                                             |       |
       +-------------> M8.R arithmetic research (optional promotion) -+       v
                                                                     M8.4 attention
                                                                            |
                                                                            v
                                                                   final paired gates
```

GPU experiments remain sequential even if implementation tasks are independent.
Archive each accepted stage and keep the last accepted dispatch available until
the complete validation is finished.

### First-wave work partition

The first implementation wave runs five independent work packages, with a
separate integration/validation owner:

| Package | Owned implementation | Integration condition |
|---|---|---|
| Register-tiled prefill | Dense CUDA prototypes and the existing projection harness | Primitive correctness and shape-weighted timings before dispatch changes |
| Packed F32 decode | Additive matvec prototype and focused decode tests | Preserve the old matvec until model-level validation |
| Routing replay | Grouped benchmark validation, 512-token/skewed/captured cases | Replay the diagnostic capture schema without host changes in timed inference |
| Bounded diagnostics | Opt-in prefill capture, capture example, explicit budgets | No disabled-path copies; graph/single-token restrictions must be explicit |
| Arithmetic control | Four-product TF32 diagnostic and numerical tests | No public mode or automatic precision changes |

Production dispatch and shared module wiring are integrated centrally. Build,
GPU correctness, sanitizer, and performance runs are serialized. Work-package
completion alone does not mean that its kernel has been accepted or enabled.
Starting sources and measurement records are preserved under
[the first-wave results directory](../results/cuda-runtime-m8-wave1).

#### Initial implementation evidence

The five packages now have implementations. Kernel prototypes are guarded by
`CUDA_M8_PROTOTYPES`, enabled only in the test harness; production dispatch and
the public precision-mode list are unchanged. The bounded `cuda-capture` example
is usable independently of these prototype kernels.

Initial measurements, before any production promotion:

- Q4_K dense 512-token projections: the register32x32 prototype is about 2.60x
  faster for 2048-to-4096 and 2.15x for 4096-to-2048 than the existing prompt
  kernel. Tiny batches regress substantially, so global replacement is rejected.
- Packed F32 decode primitives improve about 1.15x to 1.89x for the measured
  Q4_K shapes. Q6_K gains range from about 1.06x to 1.78x.
- A follow-up grouped register wrapper, using 32-assignment descriptors and the
  same compute core, improves about 1.55x to 1.87x over M7 expert tiles across
  captured routing from layers 0, 37, and 47. This is captured routing with
  synthetic matrix operands, not an accepted full-model prefill result.
- Single-expert small-shape measurements did not predict these grouped results:
  they underfill the GPU. Keep realistic expert concurrency in the harness.

See the [isolated kernel summary](../results/cuda-runtime-m8-wave1/microbench-summary.json)
and [routing-replay summary](../results/cuda-runtime-m8-wave1/routing-replay-summary.json).

The first full 512-token routing capture also establishes:

- Per-layer hottest-expert counts range from 173 to 417 assignments; median
  empty-expert count is 32. The mean of 32 assignments/expert hides major skew.
- At the selected projection positions, the first observed F32/TF32x3-tiled
  difference is layer 0's expert gate, approximately 0.0000231 at position 18.
- The first routing-order change is layer 1, but its selected expert set is
  unchanged. The first expert-set change is layer 25 for one token, while the
  maximum router-logit discrepancy there is only about 0.000423.
- Layer 26 then shows a much larger router discrepancy. This supports propagated
  approximation plus a routing-boundary crossing, not a transfer or graph problem.
  Later-layer comparisons do not isolate local kernel error.
- The four-product control reduces the realistic synthetic Q6_K dot-product
  maximum error from about 0.02490 to 0.00861 versus the F64 dot of F32 operands.
  This is a primitive diagnostic, not model-level precision acceptance.

The [capture comparison](../results/cuda-runtime-m8-wave1/capture-comparison.json)
distinguishes top-k reordering from actual expert-set changes. Full routing was
captured at all 512 positions, while projection vectors were sampled at positions
18, 255, and 511. Each capture retains about 49.7 MB of records; JSON encoding
expands the on-disk files to about 135 MB, as explicitly excluded from the budget.

Full-model candidate trials, final lint/build checks, and sanitizer validation
must finish before declaring the first wave accepted or changing kernel dispatch.

## 9. Public research references

These are sources of design principles, not new runtime dependencies or copied
implementations.

- [NVIDIA Blackwell tuning guide](https://docs.nvidia.com/cuda/blackwell-tuning-guide/index.html):
  architecture/resource guidance; prefer queried limits where this device differs.
- [CUTLASS SM120 grouped GEMM example](https://github.com/NVIDIA/cutlass/blob/b2f249acd20392beeb16f5e5a31f4f1a4e38b3bd/examples/79_blackwell_geforce_gemm/79d_blackwell_geforce_nvfp4_grouped_gemm.cu):
  TMA, device-side group scheduling, native FP4, fixed single-CTA cluster shape.
- [CUTLASS BF16x9 example](https://github.com/NVIDIA/cutlass/blob/b2f249acd20392beeb16f5e5a31f4f1a4e38b3bd/examples/78_blackwell_emulated_bf16x9_gemm/78_blackwell_emulated_bf16x9_gemm.cu):
  high-component arithmetic; explicitly SM100a, not a ready-made SM120 solution.
- [CUTLASS SM100 functionality](https://github.com/NVIDIA/cutlass/blob/b2f249acd20392beeb16f5e5a31f4f1a4e38b3bd/media/docs/cpp/blackwell_functionality.md):
  avoid confusing SM100 `tcgen05` features with GeForce support.
- [Marlin](https://github.com/IST-DASLab/marlin):
  weight-only register reuse, layout/pipeline co-design, and workload-dependent
  scheduling; its published speedups do not establish results for this runtime.
- [Henry, Tang, Heinecke: higher-precision computations with BF16](https://arxiv.org/abs/1904.06376):
  multi-component operand decomposition and accuracy/performance tradeoffs.
- [PyTorch float32 matmul precision documentation](https://docs.pytorch.org/docs/2.15/generated/torch.set_float32_matmul_precision.html):
  distinguishes output dtype from internal arithmetic and explains dropped-product
  BF16 methods. An F32 output tensor is not proof of F32-quality multiplication.

The [local profiling analysis](../results/llama-gap-investigation/analysis.json)
contains pinned llama.cpp source locations for MMQ, MMVQ, fusion, and concurrency.
