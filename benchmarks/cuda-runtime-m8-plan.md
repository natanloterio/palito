# M8 plan: closing the llama.cpp execution gap

Status: six accepted F32 structural waves, followed by scoped opt-in MMA3,
bit-exact attention reuse and small-prefix admission. Latest measurements are
63.21/113.25/195.78ms prefill at46/256/512 tokens and176.30token/s short decode.
The450-ms prefill checkpoint is reached by the explicitly approximate research
mode, not by unchanged F32 defaults. Serial-K dense register prefill and the
two-component INT8 experiment remain rejected. llama.cpp still leads.

### Next execution plan: representative operands before arithmetic changes

1. **Model-backed replay (implement first).** Load actual packed GGUF projection
   tensors and contiguous captured activations, expected outputs and routing.
   Cover Q/K/V/output and expert gate/up/down, layers 0/37/47, prefixes 256/512,
   with separate development and held-out prompts. Record model, capture, tensor,
   operand and source hashes. Reject incomplete captures, duplicate positions,
   inconsistent routing, unsupported shapes and nonfinite values explicitly.
   Check every replayed output bit against its captured production result before
   timing. Use the existing GPU harness, rotating samples and CUDA events; exclude
   loading, parsing and validation from kernel timing. Report grouping/conversion
   preparation separately. Replay is diagnostic, never end-to-end acceptance.
2. **Bounded arithmetic prototypes.** First measure two-component residual INT8
   activation reconstruction on these operands, then Q4_K affine dot products
   including minima/activation-sum corrections, Q6_K signed scales and decoder
   rounding. Compare against one higher-component floating-point control, at most
   two new designs. Include zero/subnormal/nonfinite handling, accumulator bounds
   and cancellation cases. Count preparation, both products and correction costs;
   do not infer performance from theoretical tensor-core throughput.
3. **Integrate only a demonstrated winner.** Require at least 1.5x matrix-corpus
   speedup including amortized preparation before a full-model research branch.
   Then require unchanged maximum logit error <=0.01, cosine >=0.99999, exact IDs,
   development/held-out prompts, teacher forcing, capacity/memory/error tests,
   three CUDA sanitizers and fresh public ABBA with a meaningful end-to-end win
   (initial target 10%). Structural changes remain bit-exact; arithmetic research
   may change accumulation but never relaxes the original quality gates.
   Residual INT8 remains approximate and research-only; F32 production
   activations and defaults remain unchanged until explicit promotion.
4. **Decode and attention after attribution.** Benchmark joint gate/up and
   dependency-safe QKV overlap against current decode. Reprofile before choosing
   attention tiling: the latest 36-ms attention alone exceeds the historical
   llama.cpp whole-prefill control, so matrix improvements alone cannot establish
   parity. Refresh external controls before making any parity claim.

Stop a candidate when model-backed replay or public gates falsify its benefit;
preserve the failed evidence rather than expand another synthetic tile sweep.
The first deliverable is a reproducible projection corpus and replay baseline,
not a promise that residual correction will meet quality or performance gates.

#### First deliverable implemented: actual-model projection replay

The ignored `benchmark_model_projection_replay` test loads validated GGUF tensors,
not `packed_fixture`. Optional stage-filtered diagnostics capture all positions
without retaining unrelated projection stages. The loader requires complete
position-zero 256/512-token chunks and the Qwen3 model's actual projection shapes;
it rejects missing/duplicate/nonfinite operands and inconsistent routing.
GPU-generated descriptors must match the independent existing CPU builder.

The initial corpus fixes one existing development prompt and one different
existing held-out prompt before arithmetic research. It covers layers 0/37/47,
prefixes 256/512, four dense and three expert matrices per capture:
**12 captures, 84 projections, 252 structural kernel replays**. All replays match
the captured production output bits and guards, covering 165,150,720 captured
output values. This is self-consistency against production, not independent
proof that a new arithmetic design meets full-model quality gates.

| Candidate relative to accepted production kernel | Actual-model cases | Median speedup | Wins |
|---|---:|---:|---:|
| Compact dense 2x16 reduction | 48 | 0.951x | 0 |
| Interleaved dense 2x16 rows | 48 | 0.991x | 1 |
| Expert coalesced K32 without unroll8 | 36 | 0.917x | 0 |
| Expert unroll4 instead of unroll8 | 36 | 0.991x | 2 |

The real-operand corpus reproduces the rejection of synthetic dense winners.
Both prompt splits favor the retained baseline overall; a marginal single-case
win is not a promotion. Changing operands reverses the dense rankings; this
experiment does not isolate a particular metadata value or GPU stall as the
cause. Source hashes verify production kernels, dense/expert dispatch and
compilation options remain unchanged from accepted wave6c.

Each replay reports seven rotating CUDA-event samples after three warmups.
Loading, host hashing, JSON parsing, copies and bitwise validation are excluded;
grouping timings are separate. Repeated isolated projections can keep weights
hot and do not reproduce the full model's cache pressure or launch dependencies;
real operands are necessary evidence, not sufficient end-to-end acceptance.
Actual packed weights are used without uploading
the whole model: the existing GGUF parser transiently loads the packed host
model, retains only seven selected-layer tensors and uploads one at a time.
Capture files are individually bounded to 512MiB; the initial corpus occupies
about 2.73GB of JSONL. It is intentionally not a new inference benchmark.

See the [replay checkpoint](../results/cuda-runtime-m8-model-replay/checkpoint.json),
[corpus and operand provenance](../results/cuda-runtime-m8-model-replay/corpus.json),
[case-level results](../results/cuda-runtime-m8-model-replay/summary.json), and
[capture/replay commands](../results/cuda-runtime-m8-model-replay/commands.json).
Usage and memory constraints are documented in the [runtime README](../README.md).
The bounded residual primitive phase is implemented below. Production arithmetic,
decode concurrency and attention improvements remain pending.

#### Second deliverable implemented: residual INT8 arithmetic primitives

The ignored `benchmark_residual_model_primitives` test reuses the same validated
loader, model/capture hashes and routing. Its CUDA source is included only by the
test harness. There is no production INT8 mode or activation conversion.

Two symmetric signed-INT8 components represent each 32-value activation block.
The first scale is the block maximum divided by 127; the second quantizes the
first component's residual, both using nearest-even rounding. Nonzero subnormal
scales are floored to the smallest positive F32. Zero blocks avoid division by
zero; signed-zero preservation is not promised. Device flags reject nonfinite
inputs, reconstruction overflow and nonfinite dot results.

Four input stages are prepared once per capture, sharing Q/K/V and gate/up
components. Q4_K DP4A explicitly subtracts minima times the component's integer
activation sum; Q6_K uses signed quants and signed 16-value scales. Integer
accumulators reset per metadata group: Q4 dot bound 60,960, Q6 bound 65,024,
activation-sum bound 4,064. Separate component products, corrections and F32
merges are included in dot timings. The floating control is four scalar TF32
products, **not** an MMA throughput measurement.

The design was frozen before all six development captures and then all six
fixed held-out captures were evaluated without changing source. Across
**84 projections**, each candidate measured **1,032,192 sampled dots**: eight
output rows for every token/routed assignment. All 165,150,720 full production
outputs were first rebound bit-for-bit to their captures. All 66,060,288
activation values' GPU component arrays, scales and sums matched the independent
CPU quantizer exactly. There are also 1,344 CPU F64/affine dot oracles.

| Diagnostic relative to captured projection outputs | Maximum absolute error | Minimum cosine | Median sampled-dot speed ratio |
|---|---:|---:|---:|
| Independent GPU F32 reduction control | 0.00002099 | 0.99999999999964 | 1.000x |
| Two-component INT8 affine/signed DP4A | 0.00107432 | 0.99999999658693 | 1.012x |
| Scalar TF32x4 accuracy control | 0.00002194 | 0.99999999999964 | 0.763x |

Ratios compare **sampled dot primitives**, not production GEMM, and include the
INT8 path's full-input preparation apportioned across its three/two/one reuse
branches. They exclude allocation, copies, CPU worklists and host loading.
INT8 wins 45/84 such comparisons, ranging from 0.684x to 1.490x. This does not
demonstrate the required 1.5x production matrix-corpus speedup. Median preparation
times across captures range from 5.84 to 11.54 microseconds per input stage at
256 tokens and 7.86 to 18.10 microseconds at 512 tokens.

The second component reduces maximum activation reconstruction error from
1.53769 to 0.005922. In the smaller independent CPU dot subset, maximum absolute
activation-reconstruction, affine-rearrangement and integer-merge errors are
0.00017330, about 1.34e-14 and 0.000000921, respectively. The tiny observed affine
difference is not a general proof of decoder-rearrangement equivalence.
Both approximate candidates satisfy the original logit-error/cosine thresholds
when applied diagnostically to all 84 sampled intermediate projections; **this
is not full-logit, exact-ID or full-model arithmetic admission**.

Conclusion: the residual decomposition is numerically promising on real operands,
but this scalar/DP4A dot mapping is not a demonstrated production speed winner.
The following full-matrix experiment addresses the next arithmetic task,
reusing these components and corrections against production on the same corpus.
Do not promote it into inference without the existing matrix and unchanged
full-model quality gates.
Decode overlap and attention remain separate pending tasks.

Results and commands are preserved in the
[primitive summary](../results/cuda-runtime-m8-residual-primitives/summary.json),
[corpus](../results/cuda-runtime-m8-residual-primitives/corpus.json) and
[checkpoint](../results/cuda-runtime-m8-residual-primitives/checkpoint.json).
Production remains accepted wave6c; there is no new end-to-end or llama.cpp claim.

#### Third deliverable implemented: full-matrix shared INT8 tile

`benchmark_residual_model_matrices` evaluates one frozen 32-row/32-assignment,
K32 tile using 128 threads and 4,480 bytes of static shared memory. Packed
integer weights are decoded once per tile/metadata group and reused across
assignments; both activation components are gathered into shared memory and
reused across rows. The kernel applies Q4_K minima/activation-sum corrections
and both signed 16-value Q6_K scale groups before merging components in F32.
Original GPU expert descriptors select shared-token or assignment-major inputs
and scatter outputs back into the captured layout. All this work is inside
the timed kernel; no separate weight conversion or activation scatter is hidden.

The same six development then six fixed held-out captures were run without
changing the design. This time **all 165,150,720 outputs across 84 projections**
are compared, rather than eight sampled rows. Production results must first
match their captures bit-for-bit. NaN-initialized candidate interiors require
every output to be written and finite; prefix/suffix guards must remain intact.
The four full input preparations per capture are shared across Q/K/V and gate/up,
with all component bytes, scales and sums checked against the CPU quantizer.

| Corpus split | Speedup including preparation | Including common expert grouping |
|---|---:|---:|
| Development | 1.758x | 1.731x |
| Fixed held-out | 1.770x | 1.742x |
| Combined | **1.764x** | **1.736x** |

These are ratios of sums of per-projection seven-sample CUDA-event medians,
including each of the four preparations once per capture. The grouping column
adds the same one-per-capture grouping cost to both paths. Across all captures,
production projections total 82.298ms, candidate projections 46.177ms, preparation
0.488ms, and common grouping 1.731ms. Individual capture speedups including
preparation/grouping range from 1.551x to 1.888x, exceeding the 1.5x research
matrix-corpus target on every capture. The tile wins only 68/84 individual
projection comparisons; its 0.528x worst ratio is retained, not hidden by a
shape-specific fallback or a retuned held-out dispatch.

Maximum full-projection absolute error is **0.00566865**, minimum cosine
**0.99999999666628**; all 84 intermediate projections satisfy the numerical
thresholds diagnostically. The worst error is in layer47 expert down, larger
than the earlier sampled-dot maximum: complete output coverage matters.
This is still not full-model logit/exact-ID admission. Some extreme synthetic
Q6_K intermediate fixtures exceed the 0.01 surrogate, despite matching their
independently computed affine arithmetic. Their errors are recorded rather
than mistaken for final-logit acceptance or suppressed by changing inputs.

Correctness fixtures cover 24 full matrices across Q4_K/Q6_K, widths
256/768/2048/4096, dense/shared-token/assignment-major paths, sparse/unused expert
descriptors, row/token tails and overflow. Every synthetic output is checked
against the independent CPU affine oracle, not merely sampled. The existing
ordinary tests and focused CUDA sanitizer runs are recorded in the
[checkpoint](../results/cuda-runtime-m8-residual-matrices/checkpoint.json).
See the [summary](../results/cuda-runtime-m8-residual-matrices/summary.json),
[full corpus](../results/cuda-runtime-m8-residual-matrices/corpus.json) and
[commands](../results/cuda-runtime-m8-residual-matrices/commands.json).

The following deliverable evaluates the next gate: a separately opt-in full-model
research integration, including
accounted/reusable quantization workspace and unchanged final-logit <=0.01,
cosine >=0.99999, exact-ID, teacher-forced, held-out, memory/error and sanitizer
checks. Only after that, run fresh public ABBA and require a meaningful end-to-end
win. Isolated matrices can keep weights hot and omit full-model dependencies;
the measured 1.736x is not an inference speedup or a new llama.cpp comparison.
Production arithmetic, dispatch, compilation options and defaults remain wave6c.

#### Fourth deliverable implemented: full-model integration, quality rejected

The explicit `--cuda-matmul residual-int8-research` flag now exercises the
shared-memory integer candidate in actual inference. It is **not quality-admitted**
and must not replace the accepted `f32-tiled` profile. Only eligible Q4_K/Q6_K
projections with block-aligned widths and chunks >=256 use quantized activations;
expert gate/up/down qualify together. Other projections and smaller chunks retain
the F32Tiled path. Router, output head, decode, attention and KV precision remain
unchanged. Ordinary profiles do not compile the research kernels.

One reusable peak-sized allocation holds both byte components and their scales
and sums, sharing preparation across Q/K/V and across gate/up. It adds
**7,864,320 bytes (7.5 MiB)** at Qwen3 batch512. Measured workspace increases from
90,639,232 to 98,503,552 bytes and numerical reservation from 18,843,294,784 to
18,851,159,104 bytes. The allocation remains stable across stage/layer reuse,
short/long request reuse and full context. No per-stage retained cache is hidden.

The frozen full-model corpus contains both existing prompt splits at lengths
1/255/256/257/511/512, plus each 512-token prompt chunked at256: **14 cases**.
Every case compares complete 151,936-value logits at prefill and32 baseline
teacher-forced continuation steps: **462 comparisons, 70,194,432 values**.
It also runs separate actual graph-enabled32-token greedy generations.
The same fixed prompts are not four new unseen held-out cases.

**Result: reject this arithmetic candidate under the unchanged gates.**

- **330/462** comparisons fail. All approximate eligible cases fail the0.01
  maximum-error gate; only the132 short-chunk F32 fallback comparisons pass.
- Worst maximum absolute error is **5.963799953460693** and minimum cosine is
  **0.9866099091495659**, versus limits0.01 and0.99999.
- All462 teacher-forced argmax IDs and all14 actual32-token generation sequences
  match. These successes do **not** substitute for full-vocabulary quality.
- All132 fallback comparisons are bit-exact. A first-layer runtime binding probe
  also matches both frozen normalized inputs and all Q/K/V candidate outputs
  bitwise for the256-token development and held-out cases. This checks that the
  runtime shared workspace reproduces the matrix candidate for these branches;
  it does not certify every layer or identify the cause of later divergence.
- Exact1024-token context capacity works; one-over rejection preserves valid
  logits. A deliberately injected nonfinite quantization status poisons inference
  until reset, and reset recovers identical one-token results. Admission arithmetic
  is checked at its derived bound and one byte below; it is not a second physical
  model open at a near-exhausted device budget.

See the [quality summary](../results/cuda-runtime-m8-int8-model/quality-summary.json),
[binding check](../results/cuda-runtime-m8-int8-model/integration-bindings.json)
and [checkpoint](../results/cuda-runtime-m8-int8-model/checkpoint.json).
Validation includes ordinary tests, strict Clippy and bounded real-model
memcheck/racecheck/synccheck scopes; sanitizer logs declare the instrumented
kernel launch limits rather than implying whole-model instrumentation.

**Decision:** keep the rejected opt-in research result, unchanged thresholds and
accepted F32 profile. Do not run performance ABBA to excuse failed quality.
Matrix speed1.736x remains a matrix-only result. A new accuracy/cost iteration
must pass the arithmetic and full-model gates before end-to-end speed admission;
fresh llama.cpp controls and broader unseen quality cases remain necessary.

#### Fifth deliverable: higher-accuracy decomposition and native integer MMA

The rejected two-component experiment is preserved, not reclassified as a win.
A test-only continuation of the same residual design uses three components and
explicit FMA residual formation (`fmaf` / CPU `mul_add`). This deliberately
accounts for product rounding while computing the remainder; it is not merely
an extra component appended to the previous separate-operation quantizer.
Ordinary inference profiles do not load this new quantizer.

On both first-layer256-token captures, all component bytes, scales and sums
match the CPU oracle for **3,670,016 activation values**. Worst real-valued
component-reconstruction error falls from approximately3.54e-6 after two
components to **1.3514e-8** after three. This is activation reconstruction, not
matrix or model-quality admission. Zero/subnormal blocks, nonfinite input,
overflow and guard regions are checked. The frozen
[primitive checkpoint](../results/cuda-runtime-m8-precise3-primitives/checkpoint.json)
records68 ordinary tests, strict Clippy and three clean scoped sanitizers.

An independent SM120 instruction-mapping proof compiles both
`mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32` and itsK32 counterpart to native
IMMA instructions. All **8,192 integer outputs across64 fixtures** match the CPU.
K32 is appropriate for Q4_K's32-value metadata group; Q6_K needs separate K16
products because its coefficients change every16 values. This proves fragment
mapping, not a useful inference speedup.

The matrix research engine combines these primitives in a32-row/32-assignment
shared-memory tile. It preserves integer accumulator boundaries and original
assignment destinations, reuses the existing descriptor format, and supports
dense/shared-token/assignment-major paths. Packed codes use aligned u32 Q4_K
loads and two aligned u16 loads per word for Q6_K's210-byte stride. An exact
packed-versus-scalar code check guards that lossless loading change.
All24 synthetic matrices /53,176 outputs match the independent CPU affine
oracle within its explicit rounding bound, including sparse experts and tails.
The legacy two-component matrix fixture suite also passes.

The [complete MMA corpus](../results/cuda-runtime-m8-mma3-matrices-v3/summary.json)
evaluates all84 actual-model projections (165,150,720 outputs), reaching
**2.5715x including conversion and grouping** (development2.5672x,
heldout2.5759x). All84 individual projections win; the smallest speedup is1.1402x.
Worst projection error is0.000183106 and minimum cosine0.9999999999993507.
These are matrix diagnostics, not final-logit admission. All68 ordinary tests,
strict Clippy/fmt and three scoped matrix sanitizers pass.
The first collection attempt is preserved as a
[runner framing failure](../results/cuda-runtime-m8-mma3-matrices/runner-failure.json):
Rust prefixes its first record with a test name. Only the collector framing
changed for the separate retries; no arithmetic or numerical gate was relaxed.
The [second collector failure](../results/cuda-runtime-m8-mma3-matrices-v2/runner-failure.json)
preserves the incomplete delimiter fix. The final collector is checked against
actual prefixed records; completed GPU logs are reused only after exact source
and executable hash checks, not rerun or selectively pooled for better timing.
The explicit `residual-mma3-research` runtime now uses a separate, accounted
11.25MiB three-component workspace at batch512. QKV and gate/up reuse each input
preparation; the active input count determines component strides. The old
two-component allocation is absent in this mode. Capability8.0 is required.
Full-model quality passes: the existing14 boundary/chunking cases are
extended by four predeclared512-token prose, Rust, Portuguese and mixed-format
heldouts, all with32 teacher-forced steps and independent32-token generation.
The0.01 maximum error,0.99999 cosine and exact-ID gates are unchanged.
All594 full-vocabulary comparisons (90,249,984 logits) pass: worst error
0.001193524, minimum cosine0.999999998922753, zero argmax or generation mismatches.
All132 fallback comparisons remain bit-exact. Exact1024 context, one-over rejection,
sticky numerical error/reset and11,796,480-byte payload accounting pass.
All68 ordinary tests, strict Clippy/fmt and three scoped real-model sanitizers pass.
The two first-layer captures also bit-match the frozen MMA3 normalized input and
Q/K/V outputs (eight checks /3,670,016 floats), not a claim about all runtime bindings.
See the [full-model quality evidence](../results/cuda-runtime-m8-mma3-model/checkpoint.json).

Public ABBA now passes the predeclared targeted10% improvement and5% regression
gates. At256 tokens, prefill is272.250ms ->115.151ms (**2.3643x**); at512,
498.343ms ->200.940ms (**2.4801x**). Both orders improve; complete targeted requests
improve1.8289x/2.4801x. All measured IDs match. Decode is effectively unchanged
(170.26 ->170.42token/s), as is46-token prefill (128.90 ->128.69ms).
This is scoped MoE opt-in admission; defaults and production promotion are unchanged.
See the [ABBA report](../results/cuda-runtime-m8-mma3-acceptance/performance-summary.json).

The [new512-token profile](../results/cuda-runtime-m8-mma3-acceptance/profile-summary.json)
shows198.604ms GPU kernel sum: projections147.272ms (including143.802ms MMA),
attention36.076ms, routing/grouping/combine10.162ms, preparation2.278ms.
Attention is now18.2%, but eliminating it entirely cannot close the remaining gap.
Matrices remain the dominant floor. The next practical matrix experiment is
small-prefix MMA3 on fixed32/46/128-token development inputs; keep the validated
>=256 policy until actual-matrix, full-model and public gates pass for the extension.
This also targets the46-token cliff: it currently takes longer than256 tokens.

#### Sixth deliverable: bit-exact attention probability reuse

The post-MMA profile exposed a directly removable cost: the fused attention
value loop repeats the same rounded `scores[time] / total` for every output
dimension. The MMA3-only variant computes each probability once in the existing
shared score array, synchronizes, then preserves the original ordered value sum.
It adds no workspace and changes neither precision nor the softmax/dot trees.

All48 causal/context fixtures and2,097,152 Qwen-shaped outputs match the original
GPU reference bitwise; alternating isolated samples measure approximately1.16x.
More importantly, all594 complete vocabulary vectors (90,249,984 logits) across
the same18 full-model cases are **byte-for-byte identical** to admitted MMA3,
with exact32-token generation, unchanged memory and passing error/reset/capacity.
The original0.01/cosine/exact-ID admission therefore remains intact, not relaxed.
Four default-F32/F32-tiled, graph/non-graph NVRTC PTX comparisons are byte-identical.
All68 ordinary tests, strict Clippy/fmt and three bounded attention sanitizers pass.

[Public ABBA](../results/cuda-runtime-m8-attention-normalized/performance-summary.json)
against admitted MMA3 improves both targeted prefill orders:
115.183ms ->113.230ms at256 tokens;200.692ms ->195.553ms at512.
Decode improves170.45 ->176.27token/s for1/32,153.26 ->168.75 after46 prompt tokens,
and93.93 ->132.84 after256. The256/8 complete request improves14.3%; the512/1
request improves2.6%. All IDs match and all phases remain within the5% bound.
This structural stage requires bit-exactness and improvement in both orders;
it does not weaken the earlier arithmetic engine's10% admission requirement.

The initial diagnostic profile collector accidentally selected F32-tiled instead
of the ABBA MMA3 baseline. Those traces and the exact collector are retained;
the corrected collector derives both profiles from `paired.json` and reruns
matched diagnostics. No quality, ABBA or llama timings were rerun or pooled.
See the [disclosure](../results/cuda-runtime-m8-attention-normalized/profile-mode-correction.json).
At this checkpoint the small-prefix experiment was next and policy remained>=256;
the separately admitted extension below changes only the MMA3 threshold.

#### Seventh deliverable: small-prefix MMA3 without a misleading baseline

The actual32/46/128-token baseline uses `prefill_matmul` for dense projections
and `grouped_matmul_tiled` with tile8 expert descriptors, unlike the>=256
register kernels. Replay now binds every production output bitwise and measures
each side's actual GPU grouping cost separately. Candidate tile32 grouping and
all amortized quantization preparation are included, never borrowed from baseline.

| Prefix | Development speedup | Frozen-binary heldout speedup |
|---|---:|---:|
|32|1.5974x|1.6344x|
|46|1.9014x|1.9236x|
|128|3.1345x|3.1661x|

Each split covers63 projections and22,149,120 outputs at layers0/37/47.
All predeclared per-length1.5x gates pass, with maximum projection error
0.0001220703125. Heldout execution verifies exact development sources and
binaries before collection; no threshold is chosen from heldout timing.
See the [development summary](../results/cuda-runtime-m8-mma3-small-matrices/summary.json)
and [heldout summary](../results/cuda-runtime-m8-mma3-small-heldout/summary.json).
These matrix diagnostics alone did not authorize runtime admission.

Only MMA3 now uses a shared32-token minimum for projection dispatch, quantization
allocation and tile32 group reservation. Legacy two-component INT8 remains>=256;
F32 defaults and below32-token projection fallbacks remain unchanged.
At batch512 there is no memory increase over admitted MMA3. At actual batch32,
737,280 quantization bytes and12,096 extra descriptor bytes are reserved before
execution. Actual capacities31/32/46/128/255 verify both memory deltas, stable
reuse, exact1024 context, preserved logits on one-over rejection and applicable
poison/reset recovery. Batch31 allocates neither new workspace.

Full-model quality retains all18 prior cases, including four broader prompts,
and adds31/32/33/46/127/128 boundaries on both existing replay prompts.
These12 prefix cases are not new unseen content. All **990/990** complete
vocabulary comparisons (150,416,640 values),32-step teacher forcing and all30
independent32-token generations pass unchanged0.01/cosine0.99999/exact-ID gates.
Worst error remains0.0011935234; minimum cosine0.999999998922753.
All528 unaffected old snapshots and132 fallback comparisons are bit-exact.
Four default-F32/F32-tiled PTX comparisons,68 ordinary tests, strict Clippy/fmt
and three scoped sanitizers pass. The sanitizer scope uses the physically
smallest eligible batch32 workspace: first12 selected quantizer/matrix launches
for memcheck, first7 MMA launches for racecheck/synccheck, not whole-model
instrumentation. See the [quality checkpoint](../results/cuda-runtime-m8-mma3-small-model/checkpoint.json)
and [small-capacity accounting](../results/cuda-runtime-m8-mma3-small-model/small-capacity-summary.json).

Public ABBA against immutable admitted attention MMA3 passes the predeclared
**10% improvement in both orders for short prefill and complete requests**:
46-token prefill128.865ms ->63.205ms (**2.0388x**), complete request **1.6263x**.
All IDs match, no measured phase exceeds5% regression and batch512 memory is
identical. Longer prefills remain113.252ms/195.785ms at256/512 tokens, and short
decode176.30token/s. The matched46-token diagnostic profile attributes
59.52ms of62.93ms kernel time to projections (58.71ms MMA), versus125.40ms of
128.41ms before extension. This is diagnostic, not timing admission.
See the [completed admission](../results/cuda-runtime-m8-mma3-small-acceptance/checkpoint.json)
and [performance report](../results/cuda-runtime-m8-mma3-small-acceptance/performance-summary.json).

The next matrix goal is reducing the native MMA kernel's remaining projection
cost, not another threshold change or attention-layout guess. Keep the frozen
candidate and real-operand baselines; require conversion/grouping-inclusive
matrix wins and the same full-model/public gates for any new arithmetic.
Decode still needs an independently attributed improvement to close its~1.95x
gap. No default or production promotion is implied by these research admissions.

#### Eighth deliverable: bounded asynchronous projection pipeline study

Hardware profiling was attempted before selecting a pipeline design. Installed
Nsight Compute2025.1.1 cannot load the CUDA13 cudarc context entry point
`cuCtxGetDevice_v2`. A test-only exporter now freezes actual packed weights,
three prepared activation components, scales, sums, group descriptors and
expected output. A retained-driver-ABI standalone replay matches all188,416
dense46-token outputs and guards bitwise, resolving that compatibility issue.
Nsight then reports **ERR_NVGPUCTRPERM**. No administrative permission, driver,
clock or power changes were attempted; no hardware-counter utilization or stall
attribution is claimed. Both failures and exact replay sources are preserved in
the [diagnosis](../results/cuda-runtime-m8-mma3-projection-profile/counter-diagnosis.json).

The test-only kernel stages two32-value groups, asynchronously copies activation
components/metadata while computing the current tile, and retains the admitted
integer group boundaries and exact floating-point correction/accumulation order.
It reuses the admitted packed decoders and MMA primitive. The short arithmetic
loop is intentionally retained separately rather than refactoring the admitted
kernel during a speculative experiment. No production source includes the new
kernel. A second revision caches invariant source/expert row indices and offsets
outside the K loop; a third tests a measured register/occupancy constraint.

Before each development run, the same gates were frozen: **>=1.10x complete
matrix-panel speedup at every32/46/128/512 length**, with no individual projection
slower by more than5%, and bit-exact equality to frozen admitted MMA3 outputs.
Each panel contains seven real projections: layer0 at32/46/512, layer47 at128.
Preparation and real tile32 grouping are measured and counted once on both sides.
These are structural baseline-relative gates, not a relaxation of arithmetic
quality or a claim that the original1.5x arithmetic-engine gate has changed.

| Development revision |32 tokens|46 tokens|128 tokens|512 tokens|Decision|
|---|---:|---:|---:|---:|---|
|Async double buffer|1.2068x|1.1790x|1.1405x|1.0211x|Reject:512 target missed|
|Cached indices/offsets|1.4394x|1.3781x|1.2553x|1.0542x|Reject:512 target missed|
|Cached indices, seven-block launch bound|1.4093x|1.3650x|1.2351x|1.0391x|Reject:512 target and individual regression bound missed|

Every revision checks **28 actual projections /25,733,120 output values** against
the old frozen candidate hashes, not merely a newly rebuilt reference. These are
the same development operands across revisions, not84 independent heldout cases.
All outputs match bitwise. Each also passes the24 synthetic Q4/Q6, dense/shared/
assignment-major, sparse/tail fixtures,68 ordinary tests and strict Clippy/fmt.
Memcheck/racecheck/synccheck instrument all24 synthetic pipeline launches per
revision, not whole-model inference. Static shared memory rises5,888 ->11,776
bytes per block; no inference allocation or budget is changed.

Driver-reported resources provide a narrower fact than unavailable counters:
baseline66 registers/thread and maximum7 resident128-thread blocks/SM;
cached-index pipeline76 registers and maximum6 blocks, both with zero local
bytes/thread. A seven-block compiler bound reaches72 registers and7 blocks but
introduces16 local bytes/thread and fails to improve the gate. This does not
prove occupancy or any particular stall is the achieved-performance bottleneck.

**Decision: no runtime integration or promotion.** The accepted executable is
SHA-identical, and no heldout timing selection, full-model arithmetic admission
or public ABBA is run to excuse a failed development speed gate. Frozen archives
retain all three designs. The workspace keeps the validated cached-index
research version; its attractive short-prefix results do not authorize a
post-hoc narrowed admission policy. See the [complete study](../results/cuda-runtime-m8-mma3-projection-profile/study-summary.json)
and [cached-index evidence](../results/cuda-runtime-m8-mma3-pipeline-development-index-reuse/checkpoint.json).
The next substantive matrix design should target repeated per-group affine/scale
work rather than continue an unbounded pipeline/occupancy parameter sweep.
Hardware-counter access would improve attribution but requires administrative
authorization on this shared host.

#### Short-prefill negative controls

- Native 8-assignment expert descriptors and a single compact MMA3 launch per
  projection were bitwise exact. Complete-panel speedup was 1.299x on L0/T46
  but only 1.006x on held-out L47/T46, below the 1.10x gate.
- Dense warp-register MMA3 eliminated shared staging and preserved frozen
  L0/T46 output bits. Including quantization, Q/K/V/O reached
  0.809x/0.797x/1.167x/0.793x. The design failed the per-projection gate.
- Dense Q4_K planar weight staging kept the packed byte count and produced
  identical L0/T46 output bits. Including quantization, Q/K/O reached
  1.061x/1.059x/1.050x, below the per-projection gate.

These remain test-only. See the [parity checkpoint](../results/llama-parity-agent/state.json)
for raw logs and build identities.

#### Short-context decode follow-up

- Packed expert gate/up/SwiGLU fusion was bitwise exact in all four frozen
  replays and sped up the complete replacement path by 1.351x to 1.461x.
  The opt-in package matched all 990 full-vocabulary digests and 30 free-greedy
  sequences. Both Rust ABBA orders missed the 1.10x package gate
  (1.087x and 1.085x decode); it is not the admitted comparator.
- Width-2048 retained-input normalization matched the captured outputs bitwise,
  but one held-out position reached only 1.048x. It stays test-only.

#### Grouped MMA3 follow-up

- Synchronous K64 staging and uniform mixed-width expert tiling both passed the
  L0/T46 replay, but delivered only 1.036x and 1.027x respectively on held-out
  L47/T46. Neither changed dispatch.
- Extending the 64-row tile to 256 tokens preserved bits and passed L0 and L47,
  but L37 reached 1.087x for the complete panel, below 1.10x.
- Sharing one weight tile across two same-expert descriptors preserved bits
  but achieved 0.782x at L0/T256. Driver-predicted residency fell from eight
  to four blocks per SM, and register allocation rose from 64 to 102 per
  thread. The measurements do not identify a stall cause.

#### Ninth deliverable: block-affine arithmetic derivation and rejected screen

The bounded next arithmetic design is implemented only in the test harness.
It retains all three activation components, with one scale per component
per256 inputs and sums still per32. For Q4, form the integer coefficient
`a[k] = scale6[k/32] * code4[k]`, retaining `minimum6[k/32]` separately.
For Q6, form `a[k] = signed_scale8[k/16] * (code6[k] - 32)`, preserving every
16-value metadata change and its sign. Split either coefficient exactly as
`a = low + 128*high`, with `low` in0..127 and `high` in-32..32; all planes
therefore fit signed bytes.

Within each256-value block and activation component, integer MMA produces
`I = sum(a*q)`. Q4 also accumulates `J = sum(minimum6*q)` using the32-value
activation sums. Only then apply `(d_weight*float(I) - d_min*float(J))*d_activation`
(Q6 omits the minimum). Reset integer accumulators at every GGUF block, accumulate
components separately in block order, then combine `(total0+total1)+total2`.
Absolute int32 bounds are4,129,024 for the low-plane dot,1,040,384 for the high
plane,133,169,152 for the combined Q6 dot and2,048,256 for Q4 minima. This
changes floating-point rounding and activation decomposition; it is not a
bit-exact structural optimization.

Before CUDA implementation, exhaustive signed coefficient splits and257 packed
metadata blocks per format passed an independent `Tensor::row` oracle.
Zero/subnormal/outlier/error and three-component reconstruction tests pass.
The CUDA quantizer matches CPU bits, including explicit nonfinite/overflow
status and guards. All24 synthetic Q4/Q6 dense/shared-token/assignment-major,
sparse-routing and tail matrices pass the CPU affine oracle. Three sanitizers
independently cover seven quantizer launches and all24 matrix launches.
All71 ordinary tests, strict all-feature/all-target Clippy and fmt pass; the
original32-value quantizer and admitted MMA/pipeline GPU tests also pass.

The [design](../results/cuda-runtime-m8-mma3-block-affine-development/design.json)
declared before timing requires at least1.5x complete panel speedup at **each**
32/46/128/512 length and no individual projection slower by more than5%.
Candidate and baseline quantization are measured separately and charged once
per each of four cached input stages; grouping is charged once per side.
Actual packed tensor, capture and admitted output hashes match the frozen
corpus. The28 repeated development projections contain25,733,120 output values.

| Tokens | Admitted MMA3 complete panel | Block-affine complete panel | Speedup |
|---|---:|---:|---:|
|32|1.245472ms|1.417472ms|0.878657x|
|46|1.318880ms|1.507168ms|0.875072x|
|128|1.556000ms|1.797152ms|0.865814x|
|512|3.430848ms|4.037920ms|0.849657x|

**Reject:** all per-length speed gates fail, and the worst individual speedup
is0.707805x. All matrices pass the diagnostic accuracy screen versus captured
F32 (maximum absolute error0.0000991821, minimum cosine0.9999999999992476), but
intermediate matrices do not substitute for full-vocabulary teacher forcing.
No heldout selection, full-model/public admission or fresh llama controls were
run after the failed development screen. Fewer floating-point corrections do
not establish a faster complete kernel; hardware counters remain unavailable,
so no measured stall/utilization cause is asserted.

The whole-file model hash check initially stopped the runner before timing.
The local GGUF hashes to `398bf479...6aa86`, not the historical corpus pin
`44a80615...9ae15` or download lock `0d003f66...aa8f5`. A read-only investigation
verified all21 packed projection tensors across layers0/37/47 match the frozen
corpus. With explicit user approval, a separate
[provenance amendment](../results/cuda-runtime-m8-mma3-block-affine-development/provenance-amendment.json)
pins the current file for this matrix-only study and preserves original
tensor/capture/output checks. The cause of the whole-file mismatch is unknown:
this is not proof of metadata-only differences or whole-model equivalence.
Whole-model/public admission remains independently blocked until resolved.
No model files or prior result roots were changed.

The [summary](../results/cuda-runtime-m8-mma3-block-affine-development/summary.json),
[validation checkpoint](../results/cuda-runtime-m8-mma3-block-affine-development/checkpoint.json),
raw logs, exact test executable and source archive retain the rejected evidence.
Accepted inference remains SHA-identical to small-prefix MMA3, and no inference
mode includes or dispatches the block-affine kernel. Do not narrow the panel or
retune this candidate after failure; resolve model provenance before any future
whole-model experiment.

#### What still accounts for the llama.cpp gap

A read-only pass over the saved CUDA traces attributes the current Rust prefill
cost by projection. Each trace has seven MMA3 launches per layer across all
48 layers; launch order and grid shapes agree with Q/K/V/O/gate/up/down.

| Rust GPU kernel time | 46 tokens | 512 tokens |
|---|---:|---:|
| Dense Q/K/V/O MMA3 | 23.76 ms | 46.86 ms |
| Expert gate/up/down MMA3 | 34.95 ms | 96.79 ms |
| Attention | 0.56 ms | 31.05 ms |
| Expert grouping alone | 0.90 ms | 9.18 ms |
| Activation quantization | 0.45 ms | 2.28 ms |
| All kernels | 62.93 ms | 193.44 ms |

Kernel intervals fill all but about 0.33 ms of each Rust GPU kernel span. This
does not measure host work outside that span, but idle gaps between GPU kernels
cannot explain the observed prefill gap. The quantizer itself is also a small
cost. The expensive part is what the matrix kernel does with its output.

The [historical llama CUDA trace](../results/llama-gap-investigation/analysis.json)
is a different 512-token workload with F16 KV, not a phase-matched control.
It nevertheless verifies the pinned engine dispatch: 22.10 ms in Q4_K/Q6_K
MMQ, plus 1.61 ms for Stream-K fixup; 47 of those MMQ launches are fused
gate/up/GLU (10.53 ms). Observed dense tiles use 128 token columns and routed
tiles use 32; non-fused MMQ can split the K dimension across blocks. Rust uses
three activation components and applies per-group corrections to each, with
32-row by 32-column blocks taking the full K dimension and separate
gate/up/activation launches. llama's traced MMQ uses a single Q8_1 activation
representation, not native Blackwell FP4 or equivalent arithmetic.

On the frozen 46-token layer-0 routing, 368 assignments fill only 13.5% of
2,720 column slots in populated 32-wide Rust expert tiles. At 512 tokens on
layer 0, 4,096 assignments fill 71.1% of 5,760 slots. This excludes empty
descriptors whose blocks return early. Padded columns still participate in
MMA in nonempty tiles. As a geometry-only check on the 46-token route,
8-column tiles would have 784 slots, but would create 98 populated tiles
instead of 85 and change weight reuse. It is not a measured speedup.

Long prefill has two other substantial costs: one-block-per-query/head F32
attention without cross-query tiling, and single-block expert grouping that
scans assignments twice per expert. The historical llama trace shows about
2.27 ms including FlashAttention fixup and 1.28 ms for its expert-index helper.
The KV formats, workloads, and launch counts differ; these are path comparisons,
not exact causal shares. Decode is a separate path: Rust unpacks to F32 for
its matvecs, while llama uses packed Q8_1 dot products and fusion. The old Rust
decode trace predates current packed F32 decode, so its per-kernel timings must
not be treated as today's decode breakdown.

The next controlled diagnostic should use the same frozen weights, activations
and routing for one dense projection and the routed expert path at 46/512
tokens. Compare admitted MMA3, a one-component timing-only control using the
same tile geometry, and the pinned llama MMQ path. Include preparation and
grouping, and report matrix error. This separates the cost of the precision
choice from tiling and fusion before designing another candidate. A
one-component control is not an admission candidate. Do not start a fresh
whole-model comparison under the historical GGUF pin while that file is unavailable.

#### Parallel expert grouping: validated component, not a package admission

The user explicitly approved fresh whole-model baselines on the current GGUF
(`398bf479...6aa86`), using the same bytes in both engines. This does not resolve
the difference from the historical GGUF. See the [fresh validation basis](../results/cuda-runtime-m8-parallel-grouping/fresh-validation.json).

Two kernels replace the single-block assignment scan: parallel expert counts,
then stable ballot compaction with exactly the original expert and assignment
order. The 512-token routing case went from 193.98 to 31.26 microseconds
including both launches (6.20x). The four frozen routing cases match both the
CPU descriptors and the old GPU descriptors, including unused slots and guards.
Synthetic cases pass memcheck, racecheck and synccheck.

The integrated candidate adds 512 bytes of accounted scratch on this model.
All 990 full-vocabulary snapshots and 30 independent generations match a freshly
run Rust baseline bit for bit. Fresh [Rust ABBA](../results/cuda-runtime-m8-parallel-grouping/paired/paired.json)
shows 2.7-3.0% speedup at 512-token prefill and 3.6-3.7% at 256 tokens, with
no other measured phase regressing by more than 5%. This is below the 10%
package gate. Keep it available for combination rather than replace the
accepted binary: build with `--features cuda-parallel-grouping-research`
and select `residual-mma3-research`. Builds without the feature do not
dispatch the parallel grouping path.

Fresh [llama.cpp controls](../results/cuda-runtime-m8-parallel-grouping/paired/a2/summary.json)
still lead materially. The component result confirms that grouping is worth
keeping but has limited leverage on the remaining gap. The next experiment
should isolate projection precision cost on identical operands, then screen
an expert projection that avoids work on padded output columns. Long-context
attention and decode require independent improvements.

#### Matched projection controls and packed decode

The [layer-0 replay](../results/cuda-runtime-m8-matched-projections/summary.json)
uses identical weights, activations, routes and 32-by-32 tile geometry. Including
preparation and grouping, one component is 1.35x/1.40x faster at 46/512 tokens,
but maximum absolute error against captured F32 grows to 0.0110 (over 1,000x
the three-component error on both panels). This is a timing diagnostic, not a
candidate for inference. Skipping fully empty eight-column MMA operations
preserves output bits but is slower overall (0.95x/0.98x). The grouped down
projection improves 1.08x at 46 tokens; gate/up do not improve. This control
does not include a same-operand llama.cpp kernel measurement.

A [current packed-decode trace](../results/cuda-runtime-m8-packed-decode-trace/kernels.csv)
puts 74.1% of kernel time in F32 packed matvec, of which about 85 ms is in
expert-shaped launches. Replacing the packed reduction's repeated block
barriers with one barrier and warp shuffles preserves the addition tree.
The [decode ABBA](../results/cuda-runtime-m8-packed-decode-trace/decode-abba-summary.json)
shows 1.193x speedup in both orders (about 176 to 210 tokens/s), with unchanged
reserved bytes and exact generated IDs. The decode-only candidate matches
990 full-vocabulary snapshots and 30 independent generations on the approved
GGUF. It is selectable with `cuda-packed-reduction-research` and the MMA3
profile. The next decode target is the expert matvec itself, rather than
further tuning the now-shorter reduction.

The [combined ABBA](../results/cuda-runtime-m8-packed-decode-trace/combined-paired/paired.json)
keeps generated IDs exact on all four public cases and all 151,936 logits
exact at the end of the 512-token case. Short-context decode and complete
request improve 1.195x-1.198x in both orders, but the 512-token complete request
improves only 1.029x-1.031x. The combined build also matches all 990
full-vocabulary snapshot digests and 30 independent generations. The package
therefore remains opt-in.
[Fresh pinned llama.cpp controls](../results/cuda-runtime-m8-packed-decode-trace/combined-paired/a2/summary.json)
on the same GGUF with F32 KV are still 4.67x/4.16x/5.79x faster at
46/256/512-token prefill and 1.62x faster at short-context decode (31 timed
steps). These are separate engine timing boundaries, not a same-kernel test.

#### Narrow expert MMA3 screen

Replay the frozen layer-0 46/512-token routes with 32-row by 8- or 16-column
blocks (two warps) for descriptors that fit, and the original 32-column
arithmetic for fuller descriptors. Both passes read the original descriptors;
no assignments are split or reordered. Time both launches, including their
descriptor checks, and add common grouping and reused activation preparation
once per layer. Before runtime integration, require bit-exact expert outputs,
at least 1.15x on the 46-token expert panel, and no more than 5% regression at
512 tokens (or restrict dispatch to the short-prompt path and remeasure).
Keep FP16 arithmetic and compact repacking as separate experiments.

The [development replay](../results/cuda-runtime-m8-narrow-expert/summary.json)
and [held-out replay](../results/cuda-runtime-m8-narrow-expert/heldout-summary.json)
were bit-exact for both variants across 18,235,392 outputs per variant. The
16-column variant improved the layer-0 46-token expert panel by 1.21x, but
regressed at 512 tokens (0.92x) and on held-out 32- and 46-token routes
(0.83x and 0.68x). The 8-column variant did not improve any panel. Reject
this two-pass implementation, including short-only dispatch. The experiment
does not isolate the cost of the second launch and descriptor sweep from
the cost of narrower staging. Neither variant is wired into the runtime.

#### FP16 projection control

The [FP16 replay](../results/cuda-runtime-m8-fp16-control/packed/summary.json)
keeps the original packed GGUF layout and 32-by-32 geometry. Both operands
are converted to FP16 on the fly, with FP32 tensor-core accumulation. Loading
four packed weights at a time and sharing metadata in eight-lane groups is
bit-exact against scalar FP16 staging and reduces complete projection-panel
time from 1.329 to 0.984 ms at 46 tokens and from 3.425 to 2.087 ms at 512
tokens versus MMA3, including grouping and any required preparation.
Maximum projection error against captured F32 is 0.000811 across the replay.

The [whole-model quality check](../results/cuda-runtime-m8-fp16-control/quality-summary.json)
rejects the packed FP16 variant: 785 of 990 vocabulary snapshots fail the
unchanged gate, with worst absolute error 10.1221, minimum cosine 0.87251,
two teacher-forced argmax changes, and two independent generation changes.
The quality-only overlay keeps the existing MMA3 workspace and preparation;
it is not a runtime timing result. Do not promote the variant or use its
projection speedup as a model speedup. Test packed weight layout or fusion
with the accepted MMA3 arithmetic next, separately from reduced precision.

#### Refreshed llama.cpp target, not a superiority claim

Fresh serialized controls use the immutable accepted F32Tiled implementation,
batch/chunk512, context1024, decode graphs, three warmups and seven repetitions.
Both engines use the same Qwen3 GGUF and prompt IDs.

| Workload | Rust accepted F32Tiled | llama.cpp, F32 KV |
|---|---:|---:|
|256-token prefill|272.900ms|26.384ms|
|512-token prefill|499.638ms|32.783ms|
|1-token prompt /32 generated, decode|169.96token/s|343.96token/s|

The gaps are still approximately **10.34x /15.24x prefill** and **2.02x decode**.
F32 here labels llama.cpp's KV storage, not identical arithmetic. Its one-token
decode case still generates different IDs; the other three measured cases match.
Engine-only Rust timing and internal llama.cpp timing have different boundaries.
This fixed-order refresh is not ABBA, universal performance evidence, or numerical
equivalence. See the [complete comparison](../results/cuda-runtime-m8-head-to-head/comparison.json).
After MMA3, fresh F32-KV llama controls measure25.999ms/32.740ms at256/512 tokens,
versus Rust115.151ms/200.940ms: **4.43x/6.14x still to close**.
Decode remains170.42 versus344.42token/s. Short prefill is128.69 versus13.215ms.
The Rust A/B portion is ABBA; llama follows the final baseline, with the same
one-token decode-ID mismatch caveat. See the
[post-MMA comparison](../results/cuda-runtime-m8-mma3-acceptance/comparison.json).
After structural attention reuse, fresh F32-KV llama controls are26.294ms/32.672ms,
versus Rust113.230ms/195.553ms: **4.31x/5.99x prefill still to close**.
Short decode is176.27 versus344.69token/s. The same cross-engine decode-ID and
timing-boundary caveats apply; the project does not yet outperform llama.cpp.
See the [latest comparison](../results/cuda-runtime-m8-attention-normalized/comparison.json).
After small-prefix admission, fresh F32-KV llama controls are13.270/26.183/32.697ms
at46/256/512 tokens versus Rust63.205/113.252/195.785ms: **4.76x/4.33x/5.99x**
still to close. Short decode is176.30 versus344.64token/s. The same fixed-order
external-control, timing-boundary and decode-ID caveats apply. See the
[small-prefix comparison](../results/cuda-runtime-m8-mma3-small-acceptance/comparison.json).
Project-level success still requires quality-preserving, measured head-to-head
wins on the declared workloads, not a fast matrix microbenchmark.

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
The paired runner now accepts explicit `--baseline-matmul`,
`--baseline-decode-graph`, `--baseline-batch`, and `--baseline-chunk` controls.
For like-for-like M7 comparisons, select `f32-tiled`, graphs, and 512/512 on both
sides. Historical baseline defaults remain F32, graph-off, and 128/128.

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

The five packages now have implementations. At the end of the first wave,
register-prefill, grouped-register, and arithmetic-control prototypes were enabled
only in the test harness through `CUDA_M8_PROTOTYPES`. The packed F32 decoder
passed acceptance and is enabled only in the existing `f32-tiled` profile through
`CUDA_PACKED_DECODE`. The second-wave section below records the later grouped
register promotion. Default F32
and the public precision-mode list are unchanged. The bounded `cuda-capture`
example is usable independently of the prototype kernels.

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

#### Accepted packed-decode profile

The existing opt-in `--cuda-matmul f32-tiled --cuda-decode-graph` profile now uses
the packed-load Q4_K/Q6_K F32 matrix-vector kernel. Other supported formats retain
the scalar helper. Activations remain F32; the change is unpack/load reuse and
reduction structure, not activation quantization. No workspace reservation increase
is needed. Register-tiled prefill and grouped kernels are not yet enabled.

The final public-CLI comparison used immutable M7 versus the new executable,
both at batch/chunk 512 with graphs, in A1/B1/B2/A2 order, three warmups and seven
samples per window:

| Workload | M7 decode | New decode | Complete-request speedup |
|---|---:|---:|---:|
| MoE, 1-token prompt / 32 generated | about 110 tok/s | about 165 tok/s | 1.50x |
| MoE, 46-token prompt / 8 generated | about 103 tok/s | about 149 tok/s | 1.12x |
| MoE, 256-token prompt / 8 generated | about 72 tok/s | about 92 tok/s | 1.04x |
| Dense, 1-token prompt / 32 generated | about 793 tok/s | about 810 tok/s | 1.02x |

MoE prefill-512 remains about 893 ms: this decoder change does not solve the
prefill bottleneck. Dense prefill is essentially unchanged. No untargeted phase
regression exceeds 5% in these runs, and all generated IDs are exact.

Validation:

- 1,110 full-vector comparisons across MoE, dense, route-sensitive prefixes, and
  four held-out prompts per model; exact argmax/IDs, worst maximum error
  `0.00008678436279296875`, minimum cosine about `0.9999999999939`.
- Final public-CLI 512-token vectors: dense bitwise equal; MoE maximum difference
  `0.000003814697265625`.
- 61 ordinary all-feature tests and 12 focused Python harness tests pass.
- Focused primitive tests and all eight GPU integration tests pass.
- Memcheck/racecheck/synccheck report zero errors/hazards for the new primitive
  kernels, and separately for graph generation/reset/poisoning tests including
  the opt-in profile.
- Strict Clippy, formatting, and release builds pass.
- Compiling default-F32 CUDA sources before/after produces byte-identical PTX
  with and without decode graphs. This does not claim that the opt-in decoder's
  different reduction order is bitwise identical.

Results and raw records:

- [Final public-CLI ABBA](../results/cuda-runtime-m8-wave1/paired-public-profile/paired.json).
- [MoE and held-out quality](../results/cuda-runtime-m8-wave1/decode-quality-summary.json).
- [Dense quality](../results/cuda-runtime-m8-wave1/dense-quality-summary.json).
- [Default-F32 PTX equivalence](../results/cuda-runtime-m8-wave1/default-f32-ptx-equivalence.json).

This completed the first parallel wave, not all of M8. Its next step was to gate
the measured winning dense/grouped prefill shapes, then pursue gate/up fusion,
count-aware scheduling, and attention. The four-product control is still only
a numerical diagnostic; no approximate arithmetic mode has been promoted.

### Accepted second wave: grouped F32 prefill

Only the grouped register candidate passed model-level acceptance. It is compiled
through `CUDA_REGISTER_PREFILL` and selected exclusively by `f32-tiled`:

- At least 256 tokens in the actual internal chunk.
- Width 2048, expert hidden width 768, 128 experts, top 8.
- Gate, up, and down must all be Q4_K/Q6_K for that layer.
- All three projections share 32-assignment descriptors and 128-thread,
  32-row/32-token register geometry. Assignment IDs and top-k combination order
  are preserved.

Other layers, shapes, formats, and smaller chunks retain staged eight-assignment
experts. The grouping kernel still uses 256 threads. Existing launch sites
continue to default to 256; only the selected grouped projection uses 128.
Default F32 and the approximate modes keep their previous kernels and dispatch.

Admission and allocation use one workspace-sizing helper, reserving the maximum
of tile-8 and tile-32 descriptors at capacity. This supports mixed chunks and
per-layer format fallbacks without allocating during evaluation. At batch 512,
the MoE reservation increases by exactly **10,656 bytes**, to 18,843,294,784 bytes;
workspace is 90,639,232 bytes. Dense reservation is unchanged. The existing
24-GiB owned-buffer limit, 2-GiB free headroom, and driver/JIT/graph-internal
exclusions remain unchanged.

#### Dense candidate rejected, not hidden by exact IDs

The combined dense/grouped candidate used register dense tiles for Q4_K/Q6_K
2048-to-4096 and 4096-to-2048 projections at 256 or more tokens. On the 256-token
MoE prefix, maximum logit error was **0.032967567443847656**, above the unchanged
0.01 limit; the 257-token prefix also failed at 0.015839815139770508.
Generated IDs still matched, which did not rescue the candidate.

The dense core replaces the legacy 256-lane reduction tree with increasing-K
accumulation. Removing only its production selection restored bitwise equality
on all 19 initial full-model cases. Those serial-K dense entrypoints remain
prototype-only; ordinary dense prefill was unchanged in this second wave.
The rejected executable, precision driver, outputs, and source snapshot are
preserved alongside the accepted results. That snapshot also contains a
subsequently added test-only workspace test; production code matches the rejected
candidate. This wave did not capture its per-layer routing differences.

#### Model quality and regression gates

- **88 full-vocabulary vector comparisons, all bitwise equal**, against the
  immutable accepted first-wave precision driver; exact argmax and generated IDs.
- Original and earlier held-out cases, four newly frozen long prompts, lengths
  255/256/257/511/512, tight capacities and mixed chunks, including a final singleton
  and transitions between register and staged descriptors.
- Twenty teacher-forced checkpoints extend the new prompts with baseline IDs,
  not candidate IDs, at steps 1/2/7/15/31.
- CPU tests cover capacity bounds for both descriptor formats, unchanged fallback
  allocation, and the exact batch-512 reservation increase.
- 62 ordinary all-feature tests, eight GPU integration tests, and four focused
  primitive/workspace tests pass. Existing register primitives compare with the
  independent packed CPU implementation and check tails, assignment layout, and
  guards.
- Memcheck/racecheck/synccheck are clean for register primitives and separately
  for graph generation/reset/poisoning tests including `f32-tiled`.
- Strict Clippy, formatting, and release builds pass. Default-F32 PTX is
  byte-identical to the accepted first wave with and without decode graphs.

#### Paired public performance and fresh attribution

Both binaries use `f32-tiled`, graphs, context 1024, batch/chunk 512, no prefix
cache, three warmups and seven samples per A1/B1/B2/A2 window. Pooled phase medians:

| Workload | Accepted first wave | Second wave | Improvement |
|---|---:|---:|---:|
| MoE prefill, 256 tokens | 470.16 ms | 382.94 ms | 1.228x |
| MoE prefill, 512 tokens | 893.89 ms | 682.42 ms | 1.310x |
| MoE short-context decode | 164.54 tok/s | 164.65 tok/s | essentially unchanged |

Both order contrasts improve prefill: 1.226x/1.228x at 256, and 1.312x/1.305x
at 512. Complete 256-token/8-generated request speedup is 1.190x. Dense performance
is essentially unchanged; no measured request, prefill, or decode phase regresses
by more than 5%. Public long-prefix vectors are also bitwise equal and IDs match.
The 16-KiB benchmark allowance is an explicit ceiling; measured model reservation
increases are checked exactly in the precision runs, not assumed equal to it.

Separate single-evaluation Nsight Systems profiles use the same frozen 512 IDs
and controls. They are attribution traces, not warmed performance acceptance:

| Kernel family | First wave | Second wave |
|---|---:|---:|
| All kernels, summed | 886.02 ms | 670.79 ms |
| Grouped experts, 144 launches | 544.36 ms | 319.16 ms |
| Legacy dense projections, 240 launches | 292.30 ms | 301.06 ms |
| Attention, 48 launches | 35.92 ms | 37.00 ms |
| Grouping, 48 launches | 9.22 ms | 9.28 ms |

Both traces contain 1,012 kernels: this gain comes from matrix execution, not
launch removal. Active union equals summed time; kernel spans are 886.30/671.07 ms.
After the first kernel, each trace records only two D2H copies totaling 607,748
bytes (status and logits), not recurring weight uploads. Projection kernels
still consume **92.5%** of candidate kernel time. These traces do not measure
achieved bandwidth, occupancy, stalls, or padded work counts.

The 450-ms prefill planning goal remains unmet; the decode goal of 130 tok/s is
met. Reanalysis therefore prioritizes **dense tiling that preserves the legacy
accumulation/reduction order**, plus further packed expert decoding/reuse work.
Benchmark such candidates on the measured shape corpus before enabling them.
Attention alone, only 5.5% of this trace, cannot close the remaining gap.
Fusion/count-aware scheduling remain research tasks, not accepted improvements.
There are no fresh external controls or claims of llama.cpp parity in this wave.

Evidence:

- [Full-vector and chunk quality](../results/cuda-runtime-m8-wave2/quality-acceptance.json).
- [Rejected dense candidate](../results/cuda-runtime-m8-wave2/rejected-dense-grouped-quality.json).
- [Public ABBA](../results/cuda-runtime-m8-wave2/paired-public-profile/paired.json).
- [Phase and planning gates](../results/cuda-runtime-m8-wave2/performance-acceptance.json).
- [Fresh kernel attribution](../results/cuda-runtime-m8-wave2/profile-summary.json).
- [Sanitizers](../results/cuda-runtime-m8-wave2/sanitizer-summary.json).
- [Accepted source/binary checkpoint](../results/cuda-runtime-m8-wave2/checkpoint.json).

### Accepted third wave: reference-order dense and counted experts

This wave promotes two structural changes only in `f32-tiled`. Default F32 PTX
remains byte-identical with and without decode graphs.

**Dense reference-order tiles.** For actual chunks >=256, Q4_K/Q6_K projections
with input/output widths 2048/4096, 4096/2048, or 2048/512 select
`prefill_matmul_ref2x16`: two rows, sixteen tokens, 256 threads and 16,384 shared
bytes. Each physical lane independently accumulates two of the legacy 256
lane streams, combining them only at the original stride-128 reduction point.
The remaining tree and F32 arithmetic are unchanged, with contraction disabled.
This fixes the second wave's numerical failure without relaxing its gates.
Routers, other shapes/formats, and smaller chunks retain legacy dense projection.
The 2x8 and 4x16 alternatives remain prototype-only; measured 2x16 microbench
speedups span 1.25-1.51x against legacy across the selected shapes and formats.

**Count-aware expert computation.** Existing eligibility, 33-int descriptors,
launch grid, grouping order, 128 projection threads, and workspace are unchanged.
`grouped_matmul_reg32_counted` computes eight assignments when at most eight
are valid, sixteen when at most sixteen are valid, otherwise thirty-two.
All threads still participate in every barrier. This is narrower computation,
not a compact queue or host-side scheduling change. At the captured 512-token
layouts, padded assignments fall from 5760/6016/5760 to 4616/4552/4712 for
layers 0/37/47; useful work rises from 68-71% to 87-90%. Routing replay improves
1.021-1.090x across 24 measured shape/kind/layout comparisons.

**Fusion rejected on performance.** The same-kind Q4_K/Q6_K gate/up/SwiGLU
prototype is bit-exact against separate projections and the existing epilogue.
It is nevertheless slower: separate/fused speedup is 0.881-0.886x for Q4_K and
0.800-0.808x for Q6_K on the three captured routing layers. It stays behind
`CUDA_M8_PROTOTYPES`, absent from production loading and dispatch.
Its sixteen accumulators/thread and 25,088 shared bytes, versus eight and 16,768
for one unfused projection, suggest a resource-pressure hypothesis; occupancy
and stalls were not measured. A resource-balanced smaller-row fusion is a future
experiment, not an accepted optimization.

#### Unchanged numerical and memory gates

- **128 full-vocabulary comparisons are F32-bit identical** to immutable wave2,
  including signed-zero checks; maximum error zero, cosine one, exact IDs.
- Original, held-out, and four newly frozen long prompts; boundary lengths
  255/256/257/511/512; tight capacities and mixed chunks; forty teacher-forced
  checkpoints use baseline continuation IDs.
- Six additional public long-prefix vector comparisons are bitwise equal.
- Entire reported memory objects are equal, not merely within an allowance.
  MoE batch512 reservation remains **18,843,294,784 bytes**, including
  **90,639,232 bytes** workspace; wave3 adds zero owned bytes.
- 63 ordinary all-feature tests, eight GPU integrations, six selected GPU
  primitives, and a final synthetic routing replay pass. New primitives cover
  packed-block/row/token tails, descriptor layouts, independent CPU products,
  legacy bits, and guards. The final replay explicitly compares F32 bits.
- Memcheck/racecheck/synccheck are clean for four primitive tests and separately
  for graph generation/reset/poisoning tests. Formatting, strict Clippy and
  release builds pass, including the final test-only bit-comparison edit.

#### Fresh public ABBA and bottleneck attribution

The baseline is accepted wave2, not the older tagged wave1 executable. Controls
remain graphs, `f32-tiled`, context1024, batch/chunk512, no prefix cache, three
warmups and seven samples per A1/B1/B2/A2 window; memory allowance is zero.

| Workload | Accepted second wave, fresh run | Third wave | Improvement |
|---|---:|---:|---:|
| MoE prefill, 256 tokens | 384.10 ms | 317.19 ms | 1.211x |
| MoE prefill, 512 tokens | 683.12 ms | 562.30 ms | 1.215x |
| MoE short-context decode | 164.43 tok/s | 164.54 tok/s | essentially unchanged |

Both order contrasts improve prefill: 1.215x/1.208x at 256 and 1.219x/1.214x at
512. Complete 256-token/eight-generated request speedup is 1.171x. Unselected
dense-model and small-chunk cases are essentially unchanged. No measured pooled
or order-specific request/prefill/decode phase regresses by more than 5%;
the minimum observed ratio is 0.996.

Separate single-evaluation, unwarmed Nsight Systems attribution uses the same
frozen 512 IDs and controls:

| Kernel family | Second wave | Third wave |
|---|---:|---:|
| All kernels, summed | 667.42 ms | 553.49 ms |
| Grouped experts, 144 launches | 317.60 ms | 298.09 ms |
| All dense prefill projections, 240 launches | 299.46 ms | 205.99 ms |
| Attention, 48 launches | 36.85 ms | 36.07 ms |
| Grouping, 48 launches | 9.23 ms | 9.19 ms |

Of the candidate dense launches, 192 use reference tiles (202.78 ms); the
remaining 48 are routers (3.21 ms). Both traces still contain 1,012 kernels,
with summed time equal to active union and spans 667.70/553.77 ms. After the first
kernel, each records only two D2H copies totaling 607,748 bytes, not recurring
weight uploads. Projection work still consumes **91.1%** of candidate kernel
time. These are attribution traces, not warmed acceptance or measurements of
bandwidth, occupancy, or stalls.

The 450-ms prefill goal is still unmet; decode exceeds its 130-tok/s goal.
Next priorities are packed expert decoding/reuse and resource-balanced fusion,
then further specialized decode/QKV concurrency. Residual-corrected INT8 stays
research behind unchanged quality gates; attention follows as matrices improve.
No new arithmetic mode, external control, or llama.cpp parity claim is introduced.

Evidence:

- [Accepted source/binary checkpoint](../results/cuda-runtime-m8-wave3/checkpoint.json).
- [128-vector quality](../results/cuda-runtime-m8-wave3/quality-acceptance.json).
- [Public ABBA](../results/cuda-runtime-m8-wave3/paired-public-profile/paired.json).
- [Performance, memory and public bitwise gates](../results/cuda-runtime-m8-wave3/performance-acceptance.json).
- [Fresh attribution](../results/cuda-runtime-m8-wave3/profile-summary.json).
- [Sanitizers](../results/cuda-runtime-m8-wave3/sanitizer-summary.json).
- [Labeled rejected fusion measurements](../results/cuda-runtime-m8-wave3/fusion-labeled-summary.json).
- [Default-F32 PTX equivalence](../results/cuda-runtime-m8-wave3/default-f32-ptx-equivalence.json).

### Accepted fourth wave: coalesced K=32 expert tiles

The only promoted kernel is `grouped_matmul_coalesced_k32`, replacing counted
K=64 computation under the existing `f32-tiled` expert guards. Actual chunks
must still be >=256, with width2048/hidden768, 128 experts/top8, and all three
expert projections Q4_K/Q6_K. Other modes, shapes, formats and smaller chunks
retain their previous dispatch. Dense reference tiles and packed decode are
unchanged.

Two changes improve expert execution without changing its arithmetic:

- Input staging maps neighboring lanes to contiguous K columns, rather than
  distant token rows. A same-K=64 control isolates this staging change.
- K=32 uses the existing block-aligned `warp_weight` fast path and reduces
  static shared storage from 16,768 to **8,448 bytes**. Every valid output still
  accumulates products in increasing K order; contraction remains disabled.

The existing 33-int descriptors, count-aware 8/16/32 assignment selection,
128-thread blocks, grid, routing/top-k order and persistent allocations remain
unchanged. No compact queues, additional launches, host scheduling decisions,
activation quantization, or new precision mode are introduced.

#### Replay selection and fusion against the best separate kernels

Rotating three-warmup/seven-sample CUDA-event replay uses captured routing at
layers 0/37/47 and deterministic packed operands, not captured model weights.
All candidates compare bitwise with the accepted ordered register reference,
with independent CPU checks on small fixtures.

| Candidate | Speedup against accepted counted K=64 |
|---|---:|
| Coalesced K=32, Q4_K | 1.116-1.162x |
| Coalesced K=32, Q6_K | 1.301-1.339x |
| Coalesced K=64, both formats | 1.012-1.049x |
| Coalesced K=128, both formats | 0.583-0.651x |

These ranges cover 24 shape/kind/layout comparisons at input/output widths
2048/768 and 768/2048, with both shared-token and assignment-major inputs.
Only K=32 is promoted; K=64/K=128 stay prototype-only.

Driver queries report 48 registers/thread and zero local bytes/thread for all
three separate candidates. Predicted maximum resident 128-thread blocks/SM
are 5/10/2 for accepted K=64/coalesced K=32/coalesced K=128 respectively.
These are resource-based predictions, **not measured achieved occupancy**.
There are no bandwidth or stall-counter claims.

Resource-balanced gate/up/SwiGLU fusion was also implemented with sixteen rows,
count-aware token tiles, and both K=64 and K=32. All are bit-exact, including
different gate/up weights and row/token/packed-block tails. Although smaller
fusion beats the old separate K=64 kernels, it does not beat the best separate
K=32 kernels: separate/fused speedup for K=32 fusion is **0.987-0.991x Q4_K**
and **0.965-0.974x Q6_K**. Its 8,448 shared bytes match the separate kernel,
but 64 registers/thread yield a driver-predicted eight blocks/SM rather than ten.
This does not prove the cause of the timing difference. All fusion remains
`CUDA_M8_PROTOTYPES` only; no production loading, dispatch, or diagnostic changes
were needed.

#### Full-model and regression acceptance

- **152 full-vocabulary comparisons are F32-bit identical** to immutable wave3:
  zero error, cosine one, exact generated IDs, and signed-zero-preserving checks.
- Existing cases plus four newly frozen prompts at 256/257/511/512 tokens;
  sixty teacher-forced checkpoints use baseline continuation IDs. Tight
  capacities, boundary chunks, mixed chunks and fallback transitions remain covered.
- Six additional public long-prefix vector comparisons are bitwise equal.
- Entire reported memory objects are equal, with zero benchmark allowance:
  MoE batch512 reservation **18,843,294,784 bytes**, workspace **90,639,232 bytes**.
- 63 ordinary all-feature tests, eight GPU integrations and six selected
  primitive/replay tests pass. Odd-width packed tests use complete storage,
  including width257/rows256; row tails are covered separately.
- Memcheck/racecheck/synccheck are clean for the two changed primitive tests and
  separately for graph generation/reset/poisoning tests.
- Formatting, strict Clippy and release builds pass. Default-F32 numerical PTX
  remains byte-identical with and without decode graphs.

Fresh public A1/B1/B2/A2 uses accepted wave3, graphs, `f32-tiled`, context1024,
batch/chunk512, no prefix cache, three warmups and seven samples per window:

| Workload | Accepted third wave, fresh run | Fourth wave | Improvement |
|---|---:|---:|---:|
| MoE prefill, 256 tokens | 317.32 ms | 291.39 ms | 1.089x |
| MoE prefill, 512 tokens | 563.75 ms | 527.78 ms | 1.068x |
| MoE short-context decode | 164.46 tok/s | 164.59 tok/s | essentially unchanged |

Both order contrasts improve prefill: 1.091x/1.089x at 256 and 1.072x/1.065x at
512. Complete 256-token/eight-generated request speedup is 1.070x. No pooled or
order-specific measured request/prefill/decode phase regresses by more than 5%;
the minimum ratio is 0.9967. Unselected dense/small-chunk cases are unchanged
within measurement noise.

Separate single-evaluation, unwarmed Nsight Systems attribution uses the same
frozen 512 IDs and controls:

| Kernel family | Third wave | Fourth wave |
|---|---:|---:|
| All kernels, summed | 552.78 ms | 516.10 ms |
| Grouped experts, 144 launches | 297.82 ms | 258.79 ms |
| Reference dense tiles, 192 launches | 202.41 ms | 204.37 ms |
| Attention, 48 launches | 36.01 ms | 36.38 ms |
| Grouping, 48 launches | 9.18 ms | 9.19 ms |

Both traces still contain 1,012 kernels, with summed time equal to active union
and spans 553.06/516.38 ms. Each has only two D2H copies after the first kernel,
totaling 607,748 bytes. The gain is expert execution, not launch removal or
recurring upload elimination. Projections still consume **90.4%** of candidate
kernel time. These traces are attribution, not warmed acceptance.

The 450-ms prefill goal remains unmet, while decode exceeds 130 tok/s. Further
expert/dense reuse and tile geometry remain higher-leverage than attention
alone. Specialized decode/QKV concurrency and residual-corrected INT8 research
remain pending; experimental arithmetic must retain the unchanged quality gates.
No fresh external control or llama.cpp parity claim is introduced.

Evidence:

- [Accepted source/binary checkpoint](../results/cuda-runtime-m8-wave4/checkpoint.json).
- [152-vector quality](../results/cuda-runtime-m8-wave4/quality-acceptance.json).
- [Replay and resource measurements](../results/cuda-runtime-m8-wave4/prototype-summary.json).
- [Fusion against the best separate kernels](../results/cuda-runtime-m8-wave4/fusion-k32-summary.json).
- [Public ABBA](../results/cuda-runtime-m8-wave4/paired-public-profile/paired.json).
- [Performance, memory and public bitwise gates](../results/cuda-runtime-m8-wave4/performance-acceptance.json).
- [Fresh attribution](../results/cuda-runtime-m8-wave4/profile-summary.json).
- [Sanitizers](../results/cuda-runtime-m8-wave4/sanitizer-summary.json).
- [Default-F32 PTX equivalence](../results/cuda-runtime-m8-wave4/default-f32-ptx-equivalence.json).

### Accepted fifth wave: ordered eight-way expert-loop unrolling

The proposed wider and fully active expert tiles were benchmarked first, not
assumed faster. None provided consistent gains. The promoted change is instead
`grouped_matmul_unroll8`: the accepted 32-row/K=32/coalesced/count-aware geometry
with eight-way inner-loop unrolling. Per-output products and additions still
follow increasing K, with contraction disabled.

Dispatch guards, descriptors, 128 threads, grid, routing/top-k order, memory
admission, persistent workspace, dense tiles, and decode are unchanged.
Only existing `f32-tiled` eligible expert projections select the new kernel;
all smaller chunks, unsupported shapes/formats and other modes retain their
previous paths. No new arithmetic mode or runtime tuning option is introduced.

#### Rejected geometry and completed unroll sweep

The fixed routing corpus contains layers 0/37/47 at both 256 and 512 tokens.
The 256-token records are exact ID prefixes of the existing 512-token captures;
they are not new model captures. Representative replay covers 48 combinations
of prefix, layer, Q4_K/Q6_K, input/output widths 2048/768 or 768/2048, and shared
token versus assignment-major inputs. Operands remain deterministic packed
fixtures. Timings use three warmups and seven rotating samples.

All wider/balanced candidates are bitwise equal to the ordered GPU reference
and pass independent small-shape CPU, guard and tail checks:

| Candidate | Speedup range against wave4 | Wins / 48 |
|---|---:|---:|
| Balanced 32 rows, 128 threads | 0.964-1.042x | 5 |
| Wide 64 rows, 256 threads | 0.885-0.962x | 0 |
| Balanced 64 rows, 128 threads | 0.717-1.036x | 6 |
| Balanced 128 rows, 256 threads | 0.694-1.112x | 7 |

Balanced mappings redistribute rows/tokens per thread so all threads own
outputs even for eight/sixteen-assignment computation. These layouts remain
prototype-only: avoiding inactive threads is not itself a performance win.

The subsequent unroll sweep preserves the accepted geometry:

| K-loop unroll | Speedup range | Median speedup | Wins / 48 |
|---|---:|---:|---:|
| 2 | 0.999-1.093x | 1.042x | 47 |
| 4 | 1.028-1.115x | 1.077x | 48 |
| **8** | **1.039-1.123x** | **1.083x** | **48** |
| 16 | 0.943-0.995x | 0.957x | 0 |
| 32 | 0.921-0.969x | 0.944x | 0 |

Only factor eight is promoted. The final full-sweep records are authoritative;
the earlier 2/4/8 sweep is preserved separately. Driver queries report 56
registers/thread, 8,448 static shared bytes, zero local bytes/thread and nine
predicted resident blocks/SM for unroll8, versus 48 registers and ten predicted
blocks for wave4. These are resource limits, not achieved occupancy; no measured
bandwidth, instruction-cache or stall explanation is asserted.

#### Full-model and regression gates

- **176 full-vocabulary comparisons are F32-bit identical** to immutable wave4:
  zero error, cosine one, exact IDs, and signed-zero-preserving checks.
- Existing cases plus four newly frozen 256/257/511/512-token prompts and eighty
  baseline-teacher-forced checkpoints. Tight capacities, mixed chunks and
  transitions between optimized and fallback kernels remain covered.
- Six additional public long-prefix vector comparisons are bitwise equal.
- Entire memory objects remain equal with zero benchmark allowance. MoE batch512
  reservation stays **18,843,294,784 bytes**, workspace **90,639,232 bytes**.
- 63 ordinary all-feature tests, eight GPU integrations and seven focused
  primitive/replay tests pass. The selected unroll8 path has its own focused
  test for CPU products, assignment/row/K tails and guards.
- Memcheck/racecheck/synccheck are clean for selected unroll8 and counted
  projections, and separately for graph generation/reset/poisoning tests.
- Formatting, strict Clippy and release builds pass. The final test-only
  sanitizer selector leaves production binary hashes unchanged. Default-F32 PTX
  remains byte-identical with and without decode graphs.

Fresh public A1/B1/B2/A2 uses accepted wave4, graphs, `f32-tiled`, context1024,
batch/chunk512, no prefix cache, three warmups and seven samples per window:

| Workload | Accepted fourth wave, fresh run | Fifth wave | Improvement |
|---|---:|---:|---:|
| MoE prefill, 256 tokens | 291.90 ms | 277.09 ms | 1.053x |
| MoE prefill, 512 tokens | 527.90 ms | 507.01 ms | 1.041x |
| MoE short-context decode | 164.40 tok/s | 164.61 tok/s | essentially unchanged |

Both order contrasts improve prefill: 1.054x/1.051x at 256 and 1.042x/1.040x at
512. Complete 256-token/eight-generated request speedup is 1.042x. No pooled or
order-specific measured request/prefill/decode phase regresses by more than 5%;
the minimum ratio is 0.9987.

Separate single-evaluation, unwarmed Nsight Systems attribution uses the same
frozen 512 IDs and controls:

| Kernel family | Fourth wave | Fifth wave |
|---|---:|---:|
| All kernels, summed | 516.13 ms | 495.76 ms |
| Grouped experts, 144 launches | 258.73 ms | 238.39 ms |
| Reference dense tiles, 192 launches | 204.44 ms | 204.39 ms |
| Attention, 48 launches | 36.39 ms | 36.41 ms |
| Grouping, 48 launches | 9.19 ms | 9.19 ms |

Both traces contain 1,012 kernels, summed time equals active union, and spans are
516.41/496.04 ms. Each has two D2H copies after the first kernel, totaling
607,748 bytes. This is an expert-execution improvement, not launch removal or
upload elimination. Projections still consume **90.0%** of candidate kernel
time. Attribution traces are not warmed performance acceptance.

The 450-ms prefill goal remains about 57 ms away; decode exceeds 130 tok/s.
Expert and reference-order dense reuse remain the main opportunities. Wider
geometry must be measured rather than promoted on resource predictions alone.
Decode/QKV concurrency and residual-corrected INT8 research remain pending;
attention follows matrix improvements. No fresh external control or llama.cpp
parity claim is introduced.

Evidence:

- [Accepted source/binary checkpoint](../results/cuda-runtime-m8-wave5/checkpoint.json).
- [176-vector quality](../results/cuda-runtime-m8-wave5/quality-acceptance.json).
- [Wider/balanced geometry replay](../results/cuda-runtime-m8-wave5/prototype-summary.json).
- [Completed unroll sweep and resources](../results/cuda-runtime-m8-wave5/unroll-full-summary.json).
- [Frozen routing prefixes](../results/cuda-runtime-m8-wave5/routing-prefixes.json).
- [Public ABBA](../results/cuda-runtime-m8-wave5/paired-public-profile/paired.json).
- [Performance, memory and public bitwise gates](../results/cuda-runtime-m8-wave5/performance-acceptance.json).
- [Fresh attribution](../results/cuda-runtime-m8-wave5/profile-summary.json).
- [Sanitizers](../results/cuda-runtime-m8-wave5/sanitizer-summary.json).
- [Default-F32 PTX equivalence](../results/cuda-runtime-m8-wave5/default-f32-ptx-equivalence.json).

### Accepted sixth wave: bit-exact native half-scale conversion

Dense scheduling experiments did not deliver a consistent real-model win:

- Register reuse across two/four output rows and packed metadata variants lost
  their twelve target shape/format/prefix comparisons.
- Ordered dense-loop unroll factors 2/4/8/16 were slower.
- Compact reductions and 64/32-lane ownership preserved the reference tree.
  A shape/format policy won synthetic replay, but public ABBA regressed
  256-token prefill from 277.55 to 280.13 ms. Its 512-token result was essentially
  unchanged and did not improve in both orders. Its 200 bitwise comparisons
  passed; numerical correctness alone did not authorize promotion.
- Fixed widths increased registers to 99/103 and lost all twelve comparisons.
- Half-warp row interleaving won all twelve synthetic cases at tile16, but real
  512-token prefill again did not improve in both ABBA orders.

The rejected experiments and source snapshots are preserved in
[wave6](../results/cuda-runtime-m8-wave6/checkpoint.json) and
[wave6b](../results/cuda-runtime-m8-wave6b/checkpoint.json). Synthetic fixtures
use constant normal half scales and deterministic operands, not captured model
operands. These timings are not sufficient acceptance evidence.

The promoted change instead replaces finite half-scale reconstruction with
native `cvt.f32.f16` under the internal `CUDA_NATIVE_HALF` define, enabled only
by the existing opt-in `f32-tiled` profile. Reserved-exponent inputs retain
legacy handling. An exhaustive GPU/CPU-reference comparison checks all 65,536
half encodings, including subnormals, both signed zeros and reserved encodings.
This is not a new precision mode, activation quantization or a reduction-order
change. No dependency or public tuning option is added.

Production dense dispatch is byte-identical to accepted wave5. Its original
2-row/16-token/256-thread kernel and legacy reduction implementation are retained;
all new dense geometry, unrolling, width and interleaving kernels remain
`CUDA_M8_PROTOTYPES` only. Expert dispatch, K32/unroll8/counting, graph policy,
admission limits and allocations are unchanged.

Quality and regression acceptance:

- **200 full-vocabulary comparisons** against immutable wave5, all F32-bit
  identical, maximum absolute error zero, cosine one and exact generated IDs.
- Four new frozen texts at prefixes 256/257/511/512; 100 total baseline-teacher-
  forced checkpoints; boundary chunks 255/256/257/511; dense fallback cases.
- All compared memory objects equal. Batch512 still owns 18,843,294,784 bytes
  including 90,639,232 workspace bytes; zero extra device reservation.
- Six additional public long-vector comparisons are bitwise identical.
- 63 ordinary tests, eight focused GPU tests, eight GPU integrations, strict
  Clippy/format checks, and six clean sanitizer groups covering memcheck,
  racecheck and synccheck on new primitives and graph regressions.
- Default-F32 numerical PTX is byte-identical to archived wave5 with and without
  graphs. Owned accounting still excludes driver/JIT/graph-internal allocations.

Fresh public ABBA, three warmups and seven repetitions per phase:

| Workload | Accepted fifth wave, fresh run | Sixth wave | Improvement |
|---|---:|---:|---:|
| MoE prefill, 256 tokens | 278.00 ms | 271.91 ms | 1.022x |
| MoE prefill, 512 tokens | 507.91 ms | 498.40 ms | 1.019x |
| MoE short-context decode | 164.39 tok/s | 170.49 tok/s | 1.037x |

Both prefill order contrasts improve: 1.023x/1.021x at 256 and 1.021x/1.018x
at 512. The minimum pooled/order-specific measured request, prefill or decode
ratio is 1.0025; no measured phase exceeds the 5% regression limit.

Separate single-evaluation, unwarmed Nsight attribution uses the same frozen
512 IDs, context1024 and batch/chunk512:

| Kernel family | Fifth wave | Sixth wave |
|---|---:|---:|
| All kernels, summed | 496.01 ms | 485.93 ms |
| Grouped experts, 144 launches | 238.52 ms | 233.88 ms |
| Reference dense tiles, 192 launches | 204.52 ms | 199.12 ms |
| Attention, 48 launches | 36.41 ms | 36.38 ms |
| Grouping, 48 launches | 9.18 ms | 9.18 ms |

Both traces contain 1,012 kernels, summed time equals active union, and spans are
496.29/486.21 ms. Each has two D2H copies after the first kernel, totaling
607,748 bytes; there are no recurring weight uploads. Projections still consume
89.8% of candidate kernel time. These traces are attribution, not warmed
acceptance, and establish no bandwidth, achieved occupancy or stall claims.

The 450-ms prefill goal remains about 48 ms away; decode exceeds 130 tok/s.
Further dense scheduling needs representative model-backed measurements before
promotion. Specialized decode/QKV concurrency and residual-corrected INT8
research remain pending under unchanged quality gates. No fresh llama.cpp
control or parity claim is introduced.

Evidence:

- [Accepted source/binary checkpoint](../results/cuda-runtime-m8-wave6c/checkpoint.json).
- [200-vector quality](../results/cuda-runtime-m8-wave6c/quality-acceptance.json).
- [Exhaustive half-bit comparison](../results/cuda-runtime-m8-wave6c/half-correctness.log).
- [Public ABBA](../results/cuda-runtime-m8-wave6c/paired-public-profile/paired.json).
- [Performance, memory and public bitwise gates](../results/cuda-runtime-m8-wave6c/performance-acceptance.json).
- [Fresh attribution](../results/cuda-runtime-m8-wave6c/profile-summary.json).
- [Sanitizers](../results/cuda-runtime-m8-wave6c/sanitizer-summary.json).
- [Default-F32 PTX equivalence](../results/cuda-runtime-m8-wave6c/default-f32-ptx-equivalence.json).

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
