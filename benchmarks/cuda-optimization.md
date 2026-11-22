# CUDA optimization experiments

## Target and constraints

Hardware: RTX 5090 (32,607 MiB), i9-14900K, one NUMA node, 125 GiB RAM.
This is shared hardware: no clock/power changes, exclusive reservation, global
package changes, parallel competing experiments, or new model downloads.
Only existing local model artifacts are used. Own device buffers remain capped
at 24 GiB with at least 2 GiB free headroom at admission.

Longer-term bounded competition target: on this hardware, exceed the pinned
all-GPU llama.cpp baseline by at least 10% in single-request decode throughput
for both SmolLM2-135M and Qwen3-30B-A3B, with identical GGUF files, explicit token
IDs, context and KV precision; no >5% prefill regression or increased peak memory.
This is not a universal engine claim. That target is **not currently reached**.
No concurrency claim is possible with the current single-owner numerical backend.

## Milestone 1: eliminate redundant prefill work

Predeclared acceptance before performance edits:

- At least 15% lower median prefill latency on prompts of 46 and 256 tokens for
  both models versus the preserved pre-change Rust executable.
- No more than 5% median decode regression on a decode-heavy workload.
- Exact old/new logits and IDs for arithmetic-preserving scheduling changes;
  independent CPU comparisons also require max absolute error <= 0.01,
  cosine similarity >= 0.99999, and identical argmax.
- Existing state/admission/tokenization/quantizer tests remain passing. Include
  a 512-token numerical case, different chunk sizes, and exact-context boundaries.
- Numerical reservations do not increase. Measure process peak RSS and sampled
  GPU memory separately; numerical reservations are not process peak memory.
- Keep or discard based on the full matrix, not its fastest individual case.

Representative matrix per model (context 1024, batch/chunk 128, four CPU threads,
one warmup and seven measured repetitions, cold logical KV/no prefix reuse):

| Case | Prompt tokens | Generated tokens | Purpose |
|---|---:|---:|---|
| decode | 1 | 32 | Decode and single-token path |
| short | 46 | 8 | Typical short prompt |
| prefill | 256 | 8 | Prefill scaling |
| long | 512 | 1 | Longer causal prefill |

Use the already verified prompt IDs, repeated/truncated deterministically to
the requested length. Use fixed, disclosed engine order for this first checkpoint;
do not infer small changes are significant without an interleaved rerun. Preserve samples,
commands, input IDs, hashes, compiler/build/hardware data, load time and RSS.
Report median, range, and descriptive p95/p99; seven repetitions do not support
strong tail-latency or statistical-significance claims.

The external baseline is llama.cpp revision
`6b01a4767f8d3d0c416c410a49861333965c0982`, Release `-O3 -DNDEBUG`,
native CPU dispatch, CUDA SM120, all layers/experts on GPU, flash attention,
CUDA graphs enabled, F32 KV, context 1024, batch/microbatch 128, four threads.
Never use the historical CPU-only reference bridge as the GPU baseline.
Record llama.cpp's internal prefill/decode times separately from HTTP wall time;
Rust's direct synchronized timers are not end-to-end service measurements.
Model load/startup, tokenization and reset are excluded from engine phase timers.
Include a tuned F16-KV llama.cpp deployment separately if comparing memory/speed
tradeoffs; do not silently substitute it for the precision-matched control.

## Profiling and experiment log

Nsight Systems 2024.6 and Nsight Compute 2025.1 are installed. Kernel perf paranoia
is 4: CPU perf sampling is unavailable without changing system policy, which this
work will not do. Try CUDA tracing with CPU sampling/context switching disabled;
record any driver/tool incompatibility rather than treating it as a valid trace.

Initial hypothesis: every internal prompt token currently evaluates the vocabulary
projection, copies full logits to the host, copies router status, and synchronizes.
These outputs are needed only at the end of `evaluate`. Eliminate redundant
projection/readback, then synchronize once at the public boundary. Sticky
nonfinite/error status must preserve detection of failures in skipped positions.
This is a small prototype, not a pretext for a transformer rewrite.

Experiment 1 (scheduling) passed exact 512-token old/new logits and all generated
IDs. Dense prefill improved 1.29x/1.46x for 46/256 tokens; MoE improved only
1.05x/1.06x. Decode differences were within 2%. Reservations were unchanged.
**The full milestone threshold failed**, so this is not recorded as success.
The failed initial profiling invocation also remains recorded: it accidentally
used Qwen token IDs with SmolLM2, failed admission, and is not performance evidence.

The follow-up valid Nsight MoE trace found matrix kernels at 51.4% of GPU time
and the single-threaded router at **38.9%**, averaging 131.5 microseconds across
4,704 layer invocations. There were 16 device-to-host copies for eight public
evaluation calls, confirming that intermediate readback was removed.

Experiment 2 hypothesis: parallel candidate scans and stable block top-k
reductions remove the router's serial expert/selected-expert nested loops.
Keep the same tie order, sequential selected-exponential sum, F32 probability
normalization, full residency, and all numerical gates. Reuse the predeclared
matrix/thresholds. Add tied scores, signed zero, >256 experts, nonfinite scores,
and intermediate-token overflow/reset tests. Do not optimize unrelated kernels
until the result is measured.

### Final checkpoint: two experiments, not performance parity

The [consolidated checkpoint](../results/cuda-optimization-m1/checkpoint.json)
validates all eight workloads, model hashes, CPU affinity, configurations, exact
generated IDs, and both 512-token full-logit vectors. It combines seven completed
`current` cases with the successful isolated `current-long-retry/moe-long` case.
This is **not one uninterrupted successful run**: the original final case had a
process-lifecycle failure, retained in
[failure.json](../results/cuda-optimization-m1/current/failure.json).
The retry used the identical engine binary; only harness timeout/subset handling
changed. Baseline and final binaries and CUDA source snapshots are preserved.

Median prefill latency, milliseconds (seven measured samples per cell):

| Model | Prompt tokens | Original Rust | Final Rust | Rust speedup | llama.cpp F32 KV |
|---|---:|---:|---:|---:|---:|
| SmolLM2-135M | 46 | 70.151 | 52.517 | 1.34x | 3.300 |
| SmolLM2-135M | 256 | 684.542 | 505.281 | 1.35x | 9.729 |
| SmolLM2-135M | 512 | 2095.964 | 1527.202 | 1.37x | 19.519 |
| Qwen3-30B-A3B | 46 | 789.812 | 474.268 | 1.67x | 13.662 |
| Qwen3-30B-A3B | 256 | 4923.436 | 3104.740 | 1.59x | 42.754 |
| Qwen3-30B-A3B | 512 | 11090.036 | 7448.563 | 1.49x | 80.230 |

The llama.cpp column is its internal prefill timer, captured in the original
baseline window, **not a fresh final rerun**. Rust uses direct synchronized
numerical timers, not HTTP wall time. Both use the identical model artifacts
and F32 KV; the external engine performs real batched prefill. Complete F32/F16
controls, HTTP times, raw samples, ranges, TTFT, load times, and request means
remain in the linked checkpoint and source records.

For the one-input/32-output decode workload, Qwen improved from **58.72 to
88.44 tokens/s** (1.51x). Dense decode was effectively unchanged:
676.97 versus 677.55 tokens/s. Rust Qwen decode took 350.53 ms for 31 token
steps; the external baseline's internal decode timer was 91.87 ms. These are
disclosed, different timer boundaries, not a service-level speed comparison.
**The bounded target of beating tuned all-GPU llama.cpp remains unmet.**

All four predeclared short/prefill latency reductions exceed 15%, decode stays
within its regression allowance, and numerical reservations are unchanged.
These performance/numerical gates pass on the combined evidence; lifecycle
reliability remains unresolved. There is also an observed **7.7% slower dense
256-token median versus experiment 1**, although those sample ranges overlap
(463.73-506.89 versus 463.86-508.18 ms). Do not attribute that difference to
either the router or noise without an interleaved rerun.

The final [independent numerical validation](../results/cuda-optimization-m1/final-correctness/validation.json)
passed six complete CPU/CUDA logit comparisons, exact-context generation,
chunk/reset checks, and all 21 offline HF tokenizer cases. Maximum absolute
error was 0.000290 for dense and 0.000049 for MoE, below the 0.01 tolerance;
cosines exceeded 0.99999 and argmax matched. Both 512-token vectors also matched
the original Rust executable exactly. Validation additionally passed 52 regular
Rust tests, three explicitly executed GPU tests, the default CPU suite, strict
all-feature/all-target Clippy, release build, formatting, and 16 Python tests.

Numerical reservations remain 151,115,536 bytes for dense and 18,752,958,532
bytes for MoE at context 1024. Process peak RSS was approximately 419 MiB and
34.8 GiB respectively, essentially unchanged; host model-loading copies remain
a separate limitation. Sampled whole-device MoE peaks were 19,968 MiB before,
19,964 MiB in the completed final cases, and 20,469 MiB in the isolated retry.
That shared-device observation is not an engine-owned peak-memory guarantee.

The final [diagnostic profile](../results/cuda-optimization-m1/parallel-moe-profile-stats.csv)
repeated the 98-token MoE trace: average router time fell from 131.5 to
7.54 microseconds over 4,704 calls (about 17.4x); its share of GPU kernel time
fell from 38.9% to 3.5%. Matrix-vector kernels now occupy **81.4%**. Profile
timings are excluded from the benchmark table.

Decision: retain deferred output/readback and the parallel stable router, with
unchanged precision and full expert residency. The next highest-value bounded
prototype is packed quantized batched matrix products and grouped MoE prefill.
Validate isolated kernel correctness/scaling before a larger scheduler rewrite;
do not begin an execution-graph or offload rewrite without measured benefit.

### Lifecycle incident and cleanup limits

The failed child left owned PID 224611 / thread 224674 blocked in kernel wait
`vma_start_exclude_readers`, with SIGKILL pending even after targeted TERM/KILL.
The cause is unresolved; there is no evidence establishing a router-kernel
cause. The benchmark controller was stopped, and the missing workload,
independent validation, and final profile subsequently completed normally.
Harness timeout cleanup now has bounded process-group communication and an
explicit incomplete-cleanup result, covered by a mocked regression test.

At the final check the blocked task still existed. No benchmark service listened
on ports 30171/30172; the GPU reported 0% utilization and 2,100 MiB used, with
only the unrelated desktop applications listed. No driver reset, reboot, global
policy change, or unrelated process termination was attempted. This checkpoint
does not claim complete process cleanup or production lifecycle reliability.

#### Follow-up diagnosis: host-kernel memory-reclaim faults

The [read-only incident investigation](../results/kernel-lifecycle-investigation/diagnosis.json)
found direct journal evidence that changes the earlier unknown-cause assessment:

- At 12:59:54 UTC, `kswapd0` (PID 240) suffered a general-protection fault in
  `refill_obj_stock+0x92/0x260` during slab/buffer-head reclaim. The kernel
  reported exit with `preempt_count 1`; that reclaim worker was absent when
  inspected.
- At 15:52:31 UTC, `palito` PID 224611 suffered another general-
  protection fault in the same function while an anonymous-memory page fault
  entered direct reclaim. Its stack includes `try_to_free_pages`,
  `try_to_free_buffers`, and `kmem_cache_free`.
- The surviving thread remains in uninterruptible VMA exclusion wait with
  SIGKILL pending. A stranded VMA reference/lock after the fault is consistent
  with this observation, but the full thread stack could not be read without
  additional privileges. The exact deadlock mechanism remains unproven.

The failure layer is now established as a **host-kernel fault during memory
reclaim**, not merely slow CUDA shutdown. The origin of the corrupt kernel
pointers is still unknown: these traces alone cannot distinguish a kernel bug,
driver interaction, hardware instability, or prior corruption. A responsive GPU
does not establish that every driver is fault-free; neither trace establishes
the CUDA router as the cause.

No Rust change or repeated SIGKILL can be relied upon to recover this damaged
kernel state. Pause heavyweight experiments and coordinate an administrator-
approved maintenance reboot after saving work and draining shared users.
No reboot, GPU reset, kernel change, or global policy change was performed.
After recovery, confirm the old task is gone, reclaim workers exist, and kernel
logs remain clean before rerunning correctness and performance checks. An
installed older boot image is a possible administrator-controlled comparison,
not a verified fix.

**Evidence qualification:** both M1 and M2 were measured after the first
reclaim-worker oops. Their raw results remain preserved observations, but
performance and stability conclusions require confirmation on a clean boot.
The incident report retains both exact-time kernel traces and task observations.

### Reproduction

From the runtime directory, with `CUDA_ROOT` pointing to the existing coherent
CUDA 13 installation:

```sh
cargo build --release --locked --features cuda,ktransformers -j 4
mkdir results/cuda-optimization-NEW_RUN
cp target/release/palito results/cuda-optimization-NEW_RUN/baseline-runtime
python3 scripts/benchmark_cuda_iteration.py \
  --output-dir results/cuda-optimization-NEW_RUN --label baseline \
  --cuda-root "$CUDA_ROOT" --llama \
  --binary results/cuda-optimization-NEW_RUN/baseline-runtime

# After a candidate change (never overwrite the baseline executable/results):
cargo build --release --locked --features cuda,ktransformers -j 4
python3 scripts/benchmark_cuda_iteration.py \
  --output-dir results/cuda-optimization-NEW_RUN --label candidate \
  --cuda-root "$CUDA_ROOT" --llama \
  --compare-to results/cuda-optimization-NEW_RUN/baseline

python3 scripts/validate_cuda.py --cuda-root "$CUDA_ROOT" \
  --output-dir results/cuda-validation-NEW_RUN
```

For this checkpoint, `baseline-runtime` is the actual pre-change executable and
`optimized-runtime` is the retained final executable. Building today's source
and calling it a pre-change baseline will not reproduce the historical speedup.
The consolidated report records the selected source paths and hashes; it leaves
both the incomplete original run and the diagnostic subset's standalone
`milestone_passed=false` flags intact.

The script preserves raw process output and `/usr/bin/time -v` resource reports,
samples whole-device GPU memory every 200 ms, pins each engine to the same four
available logical CPUs, and stops only the explicitly owned llama-server group.
It rejects altered input IDs, non-repeatable outputs, cached work, old/new
generation mismatches, and any old/new difference in the two 512-token logit
vectors. It reports the predeclared performance threshold separately from those
mandatory correctness checks, so a failed performance hypothesis stays readable.
The checkpoint includes both precision-matched F32 KV and tuned F16 KV controls.
The Rust binary never links or invokes llama.cpp during inference.
`--models moe --cases long` selects a diagnostic subset; a subset cannot satisfy
the full-matrix milestone flag. Each Rust process has a 180-second timeout and
bounded TERM/KILL cleanup; unresolved kernel-blocked descendants are explicitly
reported rather than silently treated as completed.

Profiles are diagnostic and excluded from performance tables. The dense profile
includes 98 evaluated tokens (one warmup and one measured 46+4-token generation,
where only three generated tokens are evaluated per request); the MoE follow-up
uses the same count. Nsight traces include model loading, so HtoD time is not
decode overhead. Raw device API time and kernel time can overlap; do not add them
to estimate total wall time.

Further experiments depend on evidence: launch overhead, warp/block reductions,
format-specialized quantized kernels, truly batched prefill/grouped experts.
KTransformers remains an independently measured request-level service, not a
numerical dependency or a substitute for our CUDA implementation.

## Milestone 2: specialize packed matrix-vector execution

Predeclared before this prototype: the final M1 trace attributes 81.4% of GPU
kernel time to matrix-vector products. The current inner loop dispatches on the
weight format and uses a generic shared-memory reduction with nine block
barriers. A format-specialized inner loop and a three-barrier reduction whose
last warp uses shuffle instructions can preserve the original addition tree.
This tests a small kernel change before the larger batched-prefill prototype;
it does not claim to implement batched prefill.

Acceptance versus the preserved M1 optimized executable:

- At least 10% lower median latency on Qwen's 46- and 256-token prefill cases.
- No more than 5% median decode latency regression for either model's
  one-input/32-output workload; disclose all other measured regressions.
- Identical generated IDs in all eight matrix cases and exact old/new full
  512-token vectors. Independent CPU/CUDA logit tolerances remain unchanged.
- No numerical reservation increase, unchanged F32 activations/KV, selected
  expert semantics, and supported formats. Test widths below, equal to, and
  above one block, including a non-power-of-two width and batched expert inputs.
- Same eight-workload controls as M1, seven measured repetitions, and a fresh
  sequential M1-binary control on shared hardware. Preserve failed experiments
  and distinguish historical external llama.cpp controls from fresh timings.

The old kernel-blocked task remains unresolved; no driver reset or global
configuration changes are authorized by this experiment. Run no competing GPU
experiments. Stop at this bounded prototype and its measured keep/discard decision.

### M2 outcome: retain a measured, limited kernel improvement

The [M2 checkpoint](../results/cuda-optimization-m2/checkpoint.json) preserves
complete sequential eight-case baseline/candidate runs, followed by candidate-
then-baseline confirmation of MoE short/prefill and dense long prefill. Each
process uses one warmup and seven measured repetitions. No case needed a retry.
The baseline is the preserved M1 optimized binary, not a reconstruction from
today's source. Its embedded CUDA source is represented by the M1 source
snapshots; fresh baseline provenance also hashes the current working tree, which
must not be mistaken for that older binary's build sources.

Implementation: specialize the packed weight loop for each of the six admitted
formats, dispatch once per row, and replace the final shared-memory reduction
stages with warp shuffles. The fixed 256-thread launch and original addition
tree remain unchanged. Other reductions, KV precision, layouts, expert selection,
and allocation sizes are untouched. This is **not batched GPU prefill**.

First complete matrix, median milliseconds:

| Model | Prompt tokens | Fresh M1 baseline | M2 candidate | Prefill speedup |
|---|---:|---:|---:|---:|
| SmolLM2-135M | 46 | 52.466 | 48.601 | 1.08x |
| SmolLM2-135M | 256 | 449.444 | 421.012 | 1.07x |
| SmolLM2-135M | 512 | 1329.266 | 1389.079 | 0.96x |
| Qwen3-30B-A3B | 46 | 463.406 | 409.239 | 1.13x |
| Qwen3-30B-A3B | 256 | 3108.336 | 2797.355 | 1.11x |
| Qwen3-30B-A3B | 512 | 7453.568 | 6838.371 | 1.09x |

On the one-input/32-output workload, dense decode rose from **675.51 to 722.45
tokens/s**, and MoE from **88.73 to 102.73 tokens/s** (1.16x).
These compare fresh sequential measurements, not medians from different
milestones. There was no final external llama.cpp rerun; historical precision-
matched and tuned controls remain in the M1 report. Performance parity with
all-GPU llama.cpp remains unachieved.

The reverse-order confirmation measured MoE 46-token prefill at 468.965 versus
414.836 ms, and 256-token prefill at 3114.376 versus 2802.541 ms. Both pairs
clear the predeclared 10% latency-reduction threshold, but only narrowly at 256
tokens. Pooling all fourteen samples per binary gives **9.91%** lower latency
there. The checkpoint explicitly records this pooled gate as false: useful
improvement is reproducible, but a robust guaranteed >=10% reduction is not
established. The generic harness's inherited `milestone_passed=false` tests
M1's different 15%-on-both-models target and is preserved, not rewritten.

The initial dense 512-token result regressed **4.5%**. In the reverse-order run,
the candidate instead took 1195.421 ms versus 1433.661 ms. Initial ranges were
1237.21-1431.20 ms (baseline) and 1173.31-1409.72 ms (candidate); confirmation
ranges also overlapped. Pooling gives only a 3.8% latency reduction. Preserve
both outcomes; do not claim a stable long-prefill win or infer a causal
regression from the first median alone.

Validation passed:

- All eight workloads' generated IDs and both full 512-token logit vectors
  matched M1 exactly.
- The final [independent CPU/CUDA validation](../results/cuda-optimization-m2/final-correctness/validation.json)
  passed six full-logit cases, chunk/reset/exact-context generation, and all 21
  HF tokenizer cases. Maximum absolute errors remained <=0.000290 dense and
  <=0.000049 MoE, with cosine >=0.99999 and matching argmax.
- Expanded GPU kernel checks cover six formats at one storage block, 512, 768,
  and 2048 input elements, including selected-expert offsets and distinct
  per-expert inputs. Three actual GPU tests, all 52 regular Rust tests, strict
  all-feature/all-target Clippy, release build, and formatting passed.
- Numerical reservations were identical. Whole-device sampled peaks were equal
  within each original pair: 2741 MiB dense and 20469 MiB MoE. Process RSS
  remained approximately 419 MiB / 34.8 GiB. Sampling and driver/JIT limitations
  still apply. Dense short-case load time rose from 301.85 to 310.83 ms; MoE
  load remained roughly 12.7-12.9 seconds. No startup improvement is claimed.

The [candidate diagnostic trace](../results/cuda-optimization-m2/specialized-moe-profile-stats.csv)
used the same 98-token workload as M1. Across 37,640 matvec calls, total matrix
kernel time fell from 830.95 to 703.57 ms, and mean time from 22.08 to 18.69
microseconds (15.3% lower). These are separate diagnostic windows, not timing
samples for the table or an isolated kernel microbenchmark. Matrix products
still occupy **78.7%** of GPU kernel time.

Decision: retain this small improvement, explicitly noting the borderline
threshold and variable dense long-context result. Next prototype: tiled
multi-token packed matrix multiplication that reuses weight decoding across
prompt rows, then causal layer-wise prefill. Validate scaling and numerical
behavior before integrating grouped MoE dispatch or rewriting the scheduler.

At final cleanup the old M1 kernel-blocked task still existed. No benchmark
service listened on ports 30171/30172, and whole-device GPU usage had returned
to approximately 2100 MiB at 1% utilization. No shared driver reset or unrelated
process termination was performed.

Reproduce the paired M2 measurement using its retained binary:

```sh
python3 scripts/benchmark_cuda_iteration.py \
  --output-dir results/cuda-m2-NEW_RUN --label baseline \
  --cuda-root "$CUDA_ROOT" \
  --binary results/cuda-optimization-m1/optimized-runtime
python3 scripts/benchmark_cuda_iteration.py \
  --output-dir results/cuda-m2-NEW_RUN --label candidate \
  --cuda-root "$CUDA_ROOT" \
  --binary results/cuda-optimization-m2/optimized-runtime \
  --compare-to results/cuda-m2-NEW_RUN/baseline
```

For reverse-order confirmation, run the candidate first, then the baseline,
using unique labels and `--models moe --cases short prefill`, followed by
`--models dense --cases long`. All selected inputs, raw process output,
resource reports, provenance, source snapshots, and final executable remain
alongside the checkpoint.

## Milestone 3 prototype: reuse packed weights across prompt tokens

The host reboot at 16:52:50 UTC on November 16 cleared the old blocked task.
`kswapd0` was present, the GPU responded, and all three existing GPU correctness
smoke tests passed without new matching kernel faults. This establishes recovery
of the immediate state, not a fix for the underlying kernel fault. Historical
M1/M2 performance still requires clean-boot confirmation.

Predeclared prototype scope: an isolated four-token matrix tile, compiling only
in explicitly executed GPU tests, not in the production backend. Each lane
decodes one packed weight and reuses it across up to four input tokens, keeping
the current per-token accumulation order and F32 arithmetic. Compare both
sequential matvec launches and a stronger single-launch independent-row batch;
otherwise launch-count savings could be mistaken for weight-reuse gains.

Acceptance before scheduler integration:

- All six supported formats match the independent packed CPU implementation
  within absolute 0.0001 plus relative 0.00001 error, and match the current GPU
  matvec exactly. Cover batches 1, 3, 4, 5, 17, widths below/at/above 256, and
  partial final token tiles. Check output guard regions.
- On isolated Q4_K and Q6_K products with widths/rows representative of dense
  projections and expert FFNs, target at least 1.5x median throughput versus
  the stronger single-launch baseline at 16/64 tokens. Report every tested
  shape, including batch-one regressions; no hidden fallback or precision change.
- Record synchronized wall and CUDA-event timings, warmups, alternating order,
  at least seven samples, allocations, GPU/kernel/compiler/source identity.
  Model loading and transfers stay outside timers; reused weights mean hot-cache
  kernel measurements, not end-to-end inference or llama.cpp comparisons.
- Stop after this isolated prototype. Retain for integration only if evidence
  supports it; no layer scheduler rewrite or default-path change in this step.

### M3 outcome: weight-reuse prototype passes its isolated gates

The [prototype checkpoint](../results/cuda-prefill-prototype-m3/checkpoint.json)
records all 24 benchmark shapes, raw samples, source/tool hashes, memory
accounting, and regressions. At M3, the four-token tile was in a separate CUDA
source compiled only by the explicit GPU tests, with production inference still
using the existing matrix-vector path. That prototype did not implement full batched
prefill, grouped prompt experts, or a new numerical dependency.

CUDA-event median speedup versus the stronger **single-launch independent-row
baseline**, not the slower one-launch-per-token baseline:

| Format | Input width | Output rows | 16 tokens | 64 tokens |
|---|---:|---:|---:|---:|
| Q4_K | 2048 | 768 | 2.48x | 2.66x |
| Q4_K | 768 | 2048 | 2.46x | 2.56x |
| Q4_K | 2048 | 4096 | 2.69x | 2.71x |
| Q6_K | 2048 | 768 | 2.43x | 2.50x |
| Q6_K | 768 | 2048 | 2.57x | 2.69x |
| Q6_K | 2048 | 4096 | 2.57x | 2.61x |

All twelve large-batch cases exceed the predeclared 1.5x target. Four-token
speedups range from 1.72x to 2.54x, but **one-token latency regresses 14.5-28.4%**.
Keep the current decode kernel; do not enable the prototype unconditionally.
Weights are synthetic and repeatedly reused, with uploads and CPU reference
checks outside timers. Event intervals can include host launch gaps; synchronized
wall timings are recorded separately. These are isolated hot-cache matrix
measurements, **not end-to-end inference or llama.cpp speedups**.

Correctness passed 120 fixture cases across six formats, storage-block/256/768/
2048 widths, and token counts 1/3/4/5/17, including partial tiles and output guard
regions. Some widths repeat for block-256 formats. All three GPU variants
produced bitwise-identical outputs. The independent CPU decoder with F64 dot
accumulation passed the unchanged absolute-plus-relative tolerance. The 24
benchmark shapes also pass this CPU/GPU check before timing.

The first attempt used sequential F32 CPU accumulation and failed even for the
**existing GPU baseline**, on Q6_K width 2048 / 17 tokens: 0.73262024 GPU versus
0.732491 CPU at output 40. Switching only the reference dot sum to F64 resolved
the cancellation-sensitive comparison without relaxing its tolerance or changing
the GPU kernel. This failed attempt is retained in the checkpoint.

Compute Sanitizer memcheck, racecheck, and synccheck all passed: **zero errors or
hazards**. The three existing GPU correctness tests also passed after extracting
their unchanged packed fixture into a shared test helper. All 52 regular Rust
tests, strict all-feature/all-target Clippy, release build, and formatting passed.
No new dependency, installed package, or system configuration change was needed.

### Clean-boot real-model control

The [clean-boot control](../results/cuda-prefill-prototype-m3/clean-boot-control/summary.json)
reran the complete eight-workload matrix with the preserved M2 executable and
fresh pinned all-GPU llama.cpp F32/F16 deployments. Both 512-token vectors and
all generated Rust IDs matched the preserved pre-reboot M2 results exactly.
This confirms execution on the recovered boot, not superiority or permanent
resolution of the earlier kernel fault.

The rebuilt production executable has a different binary hash from retained M2,
so its behavior was separately checked with another complete
[eight-case control](../results/cuda-prefill-prototype-m3/rebuilt-control/summary.json).
All generated Rust IDs, both full 512-token vectors, execution configuration,
and numerical reservations matched the clean-boot M2 control exactly. This
verifies preservation rather than assuming it from the test-only prototype scope.

| Model | Prompt tokens | Rust prefill, ms | llama.cpp F32 prefill, ms |
|---|---:|---:|---:|
| SmolLM2-135M | 46 | 48.650 | 3.621 |
| SmolLM2-135M | 256 | 443.669 | 10.313 |
| Qwen3-30B-A3B | 46 | 430.783 | 13.121 |
| Qwen3-30B-A3B | 256 | 2845.186 | 41.416 |

Rust timings are synchronized numerical phase timers; llama.cpp timings are
internal service phase timers, not HTTP wall times. Rust one-input/32-output
decode measured 720.06 tokens/s dense and 99.54 tokens/s MoE. Full measurements,
including external HTTP times and F16 controls, remain in the control report.
The prototype has **not** accelerated these production measurements.

External generated IDs match in all tested cases except the MoE one-input/
32-output workload, where both F32 and F16 llama.cpp diverge at zero-based
output index 18. This mismatch was already recorded in the original M1 baseline,
not introduced by this prototype. Do not equate exact internal Rust regression
results with full quality equivalence to llama.cpp; that remains unestablished.

No new matching kernel fault appeared after the full control matrix. The reclaim
worker remained present, benchmark servers were stopped, and GPU memory returned
to desktop levels. Continue monitoring; a clean run is not proof of a kernel fix.

Decision: retain the isolated tile as a demonstrated building block, not a
production fast path. Next implement layer-wise dense causal prefill with
token-major workspace and explicit budget accounting, preserving existing
single-token decode. Gate integration on nonzero-position/chunk/KV-causality
tests, full logits, and real-model comparisons before attempting MoE grouping.

Reproduce the isolated prototype:

```sh
export LD_LIBRARY_PATH="$CUDA_ROOT/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
cargo test --release --locked --features cuda,ktransformers --lib -j 4 \
  packed_prefill_tiles_match_cpu_and_gpu_with_token_tails \
  -- --ignored --test-threads=1
cargo test --release --locked --features cuda,ktransformers --lib -j 4 \
  benchmark_packed_prefill_weight_reuse \
  -- --ignored --test-threads=1 --nocapture

# Fresh real-model and tuned external controls, sequentially:
python3 scripts/benchmark_cuda_iteration.py \
  --output-dir results/cuda-prefill-NEW_RUN --label clean-boot-control \
  --cuda-root "$CUDA_ROOT" --llama \
  --binary results/cuda-optimization-m2/optimized-runtime \
  --compare-to results/cuda-optimization-m2/candidate
```

## Milestone 4: integrate dense causal GPU prefill

Predeclared integration scope: dense `llama` layout only, F32 activations/KV,
unchanged packed weights, four-token projection tiles, token-major reusable
workspace, and causal attention over per-layer KV. Process at most 128 tokens
per internal tile, even when a public batch is larger. Single-token calls and
Qwen3-MoE retain their current numerical paths. No grouped-expert integration
or new backend dependency in this milestone.

Acceptance before retaining the default dense fast path:

- At least 2x lower median dense prefill latency at 46 and 256 tokens versus a
  fresh sequential run of the retained M2 executable, seven measured samples.
  Report the full eight-workload matrix, not only dense wins.
- At most 5% median decode latency regression on either one-input/32-output case;
  investigate borderline results with reverse-order targeted runs.
- Exact old/new generated IDs and both full 512-token vectors; independent
  CPU full-logit thresholds remain max abs <=0.01, cosine >=0.99999, equal argmax.
- Focused synthetic tests cover multi-layer GQA, nonzero positions, token-tile
  tails, mixed single/batch calls, capacity boundaries, reset with stale future
  KV, and sticky numerical failure. GPU memory/race/synchronization checks pass.
- Admission counts all new allocations before upload. For dense context 1024 /
  batch 128, additional numerical reservation must be <=32 MiB. This is an
  explicit workspace/speed tradeoff, not an unchanged-memory claim. MoE owned
  buffer reservations must remain unchanged.

### M4 outcome: dense batched prefill retained

The [integration checkpoint](../results/cuda-dense-prefill-m4/checkpoint.json)
passes all of the predeclared M4 gates. Layer-wise prefill is now the default
for dense multi-token calls. Packed projections use the validated four-token
tile; normalization, RoPE, cache writes, attention, and FFN operate over
token-major rows. Attention masks each query to its own prefix even though
future rows in the same internal batch have already been written into KV.
Only the final prompt row projects to vocabulary logits, while every row is
checked for nonfinite hidden state. Public calls larger than 128 tokens are
split internally without committing visible request state before completion.

Fresh sequential full matrix, seven measured repetitions, context 1024 and
batch/chunk 128. Baseline is the retained M2 executable, not rebuilt current
source. All workloads use identical GGUF files and explicit prompt IDs:

| Model | Input tokens | Sequential Rust prefill, ms | Batched candidate, ms | Speedup | Fresh llama.cpp F32, ms |
|---|---:|---:|---:|---:|---:|
| SmolLM2-135M | 1 | 1.170 | 1.191 | 0.98x | 0.981 |
| SmolLM2-135M | 46 | 48.552 | 5.635 | **8.62x** | 3.429 |
| SmolLM2-135M | 256 | 431.638 | 37.671 | **11.46x** | 10.091 |
| SmolLM2-135M | 512 | 1361.039 | 104.609 | **13.01x** | 20.233 |
| Qwen3-30B-A3B | 1 | 9.175 | 9.211 | 1.00x | 2.983 |
| Qwen3-30B-A3B | 46 | 415.868 | 414.497 | 1.00x | 13.130 |
| Qwen3-30B-A3B | 256 | 2802.260 | 2799.039 | 1.00x | 42.141 |
| Qwen3-30B-A3B | 512 | 6840.625 | 6831.164 | 1.00x | 82.795 |

Rust columns are direct synchronized engine phase timers; llama.cpp uses its
internal service phase timer. HTTP/service timings, separate F16 controls,
samples, descriptive percentiles, ranges, and commands remain in the report.
These are not interchangeable end-to-end timing scopes. **Rust still does not
beat tuned all-GPU llama.cpp.** MoE intentionally remains sequential; its
near-1.00x differences are not evidence of improvement.

Single-token decode kernels and the MoE numerical path remain in place.
Decode-heavy latency regressed 1.0% dense and 0.65% MoE, inside the 5% allowance.
Candidate throughput was 704.51 and 102.86 tokens/s respectively. Dense
short-case decode latency was 2.3% slower; dense 256-token decode was 1.5%
faster. These small shared-host differences are disclosed, not claimed as wins.

The deliberate memory cost is **8,193,536 bytes (7.814 MiB)** of extra dense
workspace at context 1024 / batch 128. Total numerical reservation rises from
151,115,536 to 159,309,072 bytes; KV and packed weights do not grow. This
workspace is allocated at model open even for a decode-heavy request with that
configuration. MoE reservation remains exactly 18,752,958,532 bytes. Reported
process RSS was approximately 419 MiB dense and 34.8 GiB MoE. Sampled whole-device
peaks were identical within the pairs, but polling/driver allocation granularity
does not negate the explicitly counted workspace increase.

Startup regressed: dense short-case load/JIT time rose from **296.85 to 380.36
ms**, with the dense cases showing approximately 77-88 ms extra. MoE loading
remained about 12.5-12.7 seconds. Engine timers exclude startup; a cold one-shot
short request is not necessarily faster overall. No startup or energy win is
claimed.

Correctness and safety:

- All eight generated sequences and both complete 512-token vectors matched
  the old CUDA executable exactly, with no case retry.
- The [independent validation](../results/cuda-dense-prefill-m4/independent-validation/validation.json)
  passed six CPU/CUDA logit cases, chunk/reset/exact-context generation, and
  all 21 offline HF tokenizer cases under the unchanged numerical thresholds.
- New synthetic GPU tests cover two-layer GQA, mixed 1/5/2 and 3/1/4 chunks,
  full batches and tails, nonzero positions, exact-context rejection, stale
  future KV after reset, and a 129-token public call split across the internal
  128-token boundary. Counted device budget accepts the exact reservation and
  rejects one byte less.
- The overflow fixture uses one layer so the first token can fail while the
  final token has finite hidden state/logits. The batch still fails, stays
  poisoned, and recovers only after explicit reset.
- Memcheck, racecheck, and synccheck reported zero errors/hazards. Six actual
  GPU correctness tests, all 52 regular all-feature Rust tests, the default
  CPU-only suite, strict all-feature/all-target Clippy, release build, and
  formatting passed.

The inherited benchmark harness's `milestone_passed=false` is preserved: it
checks M1's unchanged-reservation rule, not this milestone's predeclared <=32 MiB
workspace tradeoff. The explicit M4 gates in the consolidated checkpoint pass.
That report also hashes both CUDA source files, including the newly integrated
prefill source, and preserves the exact candidate binary/source snapshots.

The [final dense trace](../results/cuda-dense-prefill-m4/dense-profile-stats.csv)
profiles one warmup plus one 256-input/4-output request: 518 evaluated tokens.
Batch matrix products take 46.2% of GPU kernel time; attention scores and values
take 23.0% and 22.8%. Profiling is excluded from the performance table.
Fused causal attention is now a substantial dense opportunity, but MoE still
has the larger structural prefill gap. Next bounded experiment: grouped
MoE prefill using the existing matvec-dominated MoE evidence, without assuming
the dense speedup transfers automatically.

Fresh F32/F16 llama.cpp controls retain the pre-existing MoE decode-heavy token
divergence; other tested cases match generated IDs. Internal exact-regression
gates are not a claim of full llama.cpp quality equivalence or training-limit
long-context coverage. No new matching kernel fault appeared after this matrix,
and no shared hardware settings were changed.

Reproduce the full comparison using preserved executables:

```sh
python3 scripts/benchmark_cuda_iteration.py \
  --output-dir results/cuda-dense-NEW_RUN --label baseline \
  --cuda-root "$CUDA_ROOT" \
  --binary results/cuda-optimization-m2/optimized-runtime
python3 scripts/benchmark_cuda_iteration.py \
  --output-dir results/cuda-dense-NEW_RUN --label candidate \
  --cuda-root "$CUDA_ROOT" --llama \
  --binary results/cuda-dense-prefill-m4/optimized-runtime \
  --compare-to results/cuda-dense-NEW_RUN/baseline
```

The existing ignored CUDA integration-test command also runs the new dense
state/budget tests. To exercise only those tests, use the `gpu_dense_prefill`
filter with `--ignored --test-threads=1` in a release CUDA-enabled test run.

## Milestone 6: measured experiments, explicit tensor cores

The [M6 report](cuda-runtime-m6.md) records all six follow-up investigations.
MoE projections account for 93.5% of GPU kernel time. Two-row activation reuse
was slower; fused decode regressed dense 256-input requests in both run orders,
so neither is production-integrated. The default F32 CUDA compilation inputs
remain byte-identical to M5.

Explicit experimental `--cuda-matmul tf32x3` uses three native TF32 MMA products
with F32 accumulation. Long MoE prefill is 1.18-1.19x faster in opposite-order
prototype runs and the accepted-binary confirmation. All 12 bounded model
logit gates pass (max error 0.000508); dense/tiny prompts regress substantially,
so this is not a general default or broad quality certification. Single TF32
fails the quality gate. Default F32, six quantization formats, numerical
reservation and original decode remain supported.

The [checkpoint](../results/cuda-runtime-m6/checkpoint.json) indexes exact
executables, source snapshots, full matrices, raw samples, fresh pinned llama.cpp
controls, resource profiles, failures and validation. External traces localize
differences before expert membership changes, but perturb the external logits;
the discrepancy is not fixed or conclusively explained. The complete performance
target remains unmet, and llama.cpp is still faster. The next bounded experiment
is shared-reduction-workspace reuse across format specializations, not an
unmeasured rewrite.

## Known gaps

- M5 subsequently implements grouped MoE prefill, bounded on-chip fused attention,
  and an opt-in compilation cache. See the
  [new experiment log](cuda-runtime-m5.md) and
  [checkpoint](../results/cuda-runtime-m5/checkpoint.json).
  Dense/MoE 512-token prefill improves another 1.39x/3.25x over M4 in its fresh
  matrix, but the complete performance target is **not** met: fixed-order decode
  regressions and the short-MoE threshold failure are retained, not hidden.
- Fused attention is limited to configured context <=1024 and head dimension
  <=256; larger configurations use the original materialized score buffer.
  No continuous batching, Rust HTTP service, or
  independent CPU-expert offload. Concurrency throughput and comparable streaming
  HTTP TTFT remain unmeasured, not zero.
- GPU memory polling can miss transient peaks. Driver/JIT allocations and model
  loading copies are outside numerical accounting.
- No reliable per-request energy accounting; do not infer energy from TDP.
- Long-context validation here means 512 tokens, not the model's training limit.
- Current tokenizer/architecture support remains explicitly limited; no model
  compatibility claims beyond the tested layouts.
