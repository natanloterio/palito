# M6: measured kernels, decode fusion, numerical tracing and tensor cores

## Gates declared before implementation

- Preserve M5's executable and source archive. Default F32 optimizations must
  preserve bitwise logits, routing, state/reset/chunks, budgets and supported
  fallbacks. Require >=10% end-to-end improvement in the targeted phase and <=5%
  decode-heavy request regression; evaluate opposite-order runs, not a selected
  favorable window.
- Use recorded GPU clocks, memory clock, utilization, temperature and power;
  configurable warmup applies identically to controls/candidates. Run GPU work
  sequentially. Do not lock clocks, alter power policy or claim exclusivity.
- Profile grouped MoE before changing its projections. Prototype activation reuse
  across output rows, not another unmeasured increase in token tile size.
- Evaluate fused decode attention separately from matrix changes.
- Trace the exact previous 19-token divergent prefix, comparing intermediate
  layers, router choices and expert combination against the pinned external
  oracle. Tracing is diagnostic only and never a benchmark.
- Tensor-core execution is an explicitly selected experiment, never a silent
  default precision reduction. TF32 operands retain F32 range and use F32
  accumulation, but round both decoded weights and activations. Kernel tolerance:
  `0.005 + 0.003 * abs(reference)`. Model gate: finite logits, max absolute
  error <=0.05, cosine >=0.9999, and matching argmax on teacher-forced cases.
  Greedy equality alone is not quality evidence. Record failures; do not promote
  the mode as quality-equivalent without these gates and broader evaluation.
- Preserve commands, artifacts, source/binary identity, raw samples, rejected
  approaches and full numerical evidence. A bounded experiment may fail.

## Initial profile

On M5's 256-input/4-output MoE diagnostic (one warmup plus one request),
`grouped_matmul` consumes **76.3%** of GPU kernel time and shared
`prefill_matmul` another **17.2%**. Group construction is **0.5%**.
This supports prioritizing projection execution, not rewriting the router or
grouping algorithm. Full Nsight data is in `results/cuda-runtime-m6/`.

## Tensor-core refinement declared after the first experiment

Single-product TF32 measured about 1.24x faster on long MoE prefill but failed
the original quality gate (max errors 0.289/0.484 at 512/256 tokens). Dense
seven-token error also exceeded 0.05. Generated text happened to match; this
does not override the failed quality gate.

One bounded corrective experiment uses high/residual TF32 operands and three
products: `Ah*Bh + Ah*Bl + Al*Bh`, with F32 accumulation. The omitted `Al*Bl`
term and residual rounding still make this approximate. It must pass the
**same** model quality gates; no thresholds are relaxed. Measure its extra
instructions rather than assuming it retains the single-product speedup.

## Outcomes

| Work | Decision |
|---|---|
| Measurement controls | Added sequential ABBA runner, immutable executables, consistent warmups, GPU clocks/utilization/temperature/power, telemetry timeout and identity/affinity/precision/memory guards. |
| MoE profiling | Grouped and shared projections account for 93.5% of kernel time; grouping is not the bottleneck. |
| Two-output-row activation reuse | Bitwise correct, but representative grouped products are 15-29% slower. Test-only, not production-integrated. |
| Single-token attention fusion | Bitwise correct, but misses the 10% target and regresses dense 256-input/8-output requests in both orders. Production dispatch rejected. |
| External numerical tracing | Raw projections, head-normalized/RoPE outputs and routing localized; no proven architecture bug or equivalence fix. |
| Tensor cores | Single TF32 fails quality. Three-product TF32 passes the bounded model gates and improves long MoE prefill; explicit experimental opt-in only. |

The accepted executable is
[`accepted-runtime`](../results/cuda-runtime-m6/accepted-runtime), SHA256
`fc7cf496876e8cf74730c44b8cba9d145369497ad3de724b988ba6f06f5c46c5`.
Its default remains **F32**. Both approximate modes require compute capability
8.0 or newer; unsupported opt-in and non-CUDA use fail explicitly. Router
projections, single-token projections, activations and KV remain F32. Upstream
approximation can nevertheless change routing: unchanged router arithmetic is
not a guarantee of unchanged expert choices.

Default F32 does not compile/load the tensor-core module. The rejected row-reuse
template is isolated in test-only `rows2.cu`; all three production F32 CUDA
sources are byte-identical to M5. This deliberately avoids retaining a production
code-generation refactor for a rejected optimization. No numerical reservation
increase is observed: dense 154,628,360 bytes, MoE 18,775,408,112 bytes in either
selected mode. These counts exclude driver/JIT and host allocations.

## Measurements and controls

Hardware and original model hashes are unchanged from M5: i9-14900K, RTX 5090,
the same SmolLM2 Q4_K_M and Qwen3-30B-A3B Q4_K_M artifacts. Pinned llama.cpp is
`6b01a4767f8d3d0c416c410a49861333965c0982`; fresh matched F32-KV and tuned F16-KV
all-GPU controls are retained in
[`paired-f32/a2`](../results/cuda-runtime-m6/paired-f32/a2/summary.json).
All variants use identical prompt IDs, context 1024, batch/chunk 128,
four pinned CPU threads, concurrency one, three warmups and seven samples.
There is no prefix reuse, clock locking or exclusive hardware claim.

### Opposite-order prototype evidence

Each variant has two separate process windows in order A1/B1/B2/A2. Report both
contrasts; do not treat two windows as a confidence interval.

| Prototype / workload | A1/B1 speedup | A2/B2 speedup |
|---|---:|---:|
| Fused decode: dense 1-input/32-output request | 0.9963x | 1.0126x |
| Fused decode: MoE 1-input/32-output request | 0.9871x | 1.0182x |
| Fused decode: dense 256-input/8-output request | 0.9271x | 0.9412x |
| TF32x3: MoE 256-token prefill | 1.1829x | 1.1861x |
| TF32x3: MoE 512-token prefill | 1.1868x | 1.1842x |
| TF32x3: MoE 256-input/8-output request | 1.1606x | 1.1629x |

Full results, including unfavorable cases:
[`paired-f32`](../results/cuda-runtime-m6/paired-f32/paired.json) and
[`paired-tf32x3`](../results/cuda-runtime-m6/paired-tf32x3/paired.json).
These runs measured preserved **experimental v4**, not the accepted executable:
v4 still enabled fused decode and the generalized scalar template. Its TF32x3
46-token request speedups are 0.9747x/0.9483x. Generated IDs matched throughout;
default F32 full 512-token vectors matched bitwise. Approximate long-prefix
vectors also passed the existing, stricter CUDA numerical comparator.

### Accepted-binary confirmation

After rejecting fused decode and isolating row reuse, reran the complete F32
matrix and the selected MoE mode. These are **single-order confirmation runs**,
not another ABBA pair and not pooled with earlier builds.

| Model | Input/output | F32 prefill ms | TF32x3 prefill ms | Prefill speedup | Request speedup |
|---|---:|---:|---:|---:|---:|
| Dense | 1/32 | 1.191 | not targeted | - | - |
| Dense | 46/8 | 5.434 | not targeted | - | - |
| Dense | 256/8 | 28.696 | not targeted | - | - |
| Dense | 512/1 | 65.643 | not targeted | - | - |
| MoE | 1/32 | 9.173 | 9.216 | 0.9953x | 1.0074x |
| MoE | 46/8 | 194.673 | 203.727 | 0.9556x | 0.9731x |
| MoE | 256/8 | 963.413 | 812.866 | 1.1852x | 1.1632x |
| MoE | 512/1 | 1940.075 | 1638.583 | 1.1840x | 1.1840x |

See [`accepted-comparison.json`](../results/cuda-runtime-m6/accepted-comparison.json),
[`accepted-f32`](../results/cuda-runtime-m6/accepted-f32/summary.json) and
[`accepted-tf32x3`](../results/cuda-runtime-m6/accepted-tf32x3/summary.json).
The accepted F32 request ratios against the last M5 control span
0.9742x-1.0208x; this is preservation, not a new default speedup claim.
Both full 512-token F32 vectors and all eight generation sequences match M5.
Intermediate builds and their regressions remain under `final-*`, rather than
being overwritten or substituted for the accepted measurements.

TF32x3 is **not recommended for dense or tiny prompts**. The resident-model
quality helper also measures those cases: dense 7/19/46/256/512-token prefill
ratios are approximately 0.082x/0.164x/0.273x/0.701x/0.734x. MoE 7/19-token
ratios are approximately 0.452x/0.733x. These helper timings are unpaired
diagnostics, not additional acceptance trials. A one-token prompt bypasses
tensor cores entirely; timing variation there is not a matrix-kernel gain.

Fresh tuned llama.cpp remains substantially faster: its F16-KV 256/512-token
MoE internal prefill medians are 41.841/79.216 ms. Timing boundaries differ
from Rust's synchronized engine measurements; the pre-existing MoE decode
output mismatch also remains. **No superiority or external numerical
equivalence claim is made. The broad performance target is not met.**

## Correctness and failed approaches

- Accepted TF32x3 passes all 12 teacher-forced full-vector cases: both models
  at lengths 1, 7, 19, 46, 256 and 512. The 19-token MoE case is the exact
  previously divergent prefix. Worst absolute error is **0.000507355**;
  all cosine/argmax/finite gates pass and all measured generation sequences
  match F32. This is bounded logit evidence, **not perplexity or broad quality
  certification**, especially for discontinuous expert routing.
- The single-product TF32 experiment fails: MoE 256/512-token errors are
  0.484464/0.288729 and dense seven-token error is 0.096639. Matching generated
  text does not rescue it. `--cuda-matmul tf32` remains explicitly experimental
  and is not recommended.
- Six-format dense and grouped tensor-core checks cover row/token/storage tails,
  guarded output buffers and both shared-token/per-assignment expert inputs.
  TF32 is compared with independently rounded CPU operands; TF32x3 with
  unrounded CPU operands. An additional stricter cancellation fixture failed
  at Q6_K/2048 (-2.0302322 versus -2.0284257); its log is retained. Final kernel
  checks use the originally declared `0.005 + 0.003*abs(reference)`, not a
  relaxed model gate.
- 55 all-feature ordinary tests, 29 default CPU-only tests, 17 Python tests,
  strict all-target Clippy and 12 actual release GPU tests pass. Independent
  CPU/CUDA full logits, reset/chunks and offline HF tokenization pass.
- Memcheck, racecheck and synccheck pass on the full pre-isolation GPU suite;
  after final isolation all 12 GPU tests were rerun, plus all three sanitizers
  on the affected packed/grouped tests, including the test-only two-row paths.
  Single-token fused attention is independently covered at causal/context
  boundaries but is not production-dispatched.
- Static NVCC resource evidence shows grouped row reuse doubles shared memory
  from 24 to 48 KiB, with 40 versus 48 registers and no spills. This supports
  investigating occupancy, not a proven counter-based explanation. Nsight
  Compute 2025.1.1 cannot instrument this cudarc/CUDA 13 binary: its injection
  shim lacks `cuCtxGetDevice_v2`. No counters were acquired and no permissions
  or global settings were changed. Nsight Systems timings and static resource
  evidence remain available.

## Numerical diagnosis: calibrated, not declared fixed

The reference and external GPU traces have 672 matched exact stage keys.
Auxiliary `(view)` names are excluded from comparisons. At layer zero:

| Stage | Maximum absolute difference |
|---|---:|
| Attention normalization | 0.00000036 |
| Raw Q/K/V projection | 0.00029454 / 0.00023348 / 0.00019864 |
| Q/K after head normalization and RoPE | 0.04509092 / 0.05684281 |
| Router logits | 0.01223231 |
| Layer output | 0.00145423 |

The first **ordered** top-eight difference is layer 2, but its expert set is
unchanged. First membership change is layer 9: expert 60 versus 23. The
eighth/ninth raw-score margins are 0.0136991 (reference) and 0.0220759
(external, opposite order). Expert IDs are categorical, not numerical errors.

Crucially, enabling external tracing itself changes final logits by
**0.28134012** versus the uninstrumented external oracle. Rust reference tracing
changes them by zero. Therefore these traces localize *instrumented*
differences before MoE routing; they do not conclusively explain the original
untraced divergence. No architecture/routing bug is claimed fixed. Evidence:
[`trace-comparison-v2`](../results/cuda-runtime-m6/trace-comparison-v2.json) and
[`trace-calibration`](../results/cuda-runtime-m6/trace-calibration.json).

## Reproduction

From the runtime directory, using the already provisioned models/toolkit:

```sh
export CUDA_ROOT="$PWD/.venv-cuda-bench/lib/python3.11/site-packages/nvidia/cu13"
export LD_LIBRARY_PATH="$CUDA_ROOT/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
cargo build --release --locked --features cuda,ktransformers --examples --bin palito

# Select only for the measured long-MoE workload; F32 is still the default.
target/release/palito bench --backend cuda \
  --model models/benchmark-qwen3/gguf/Qwen3-30B-A3B-Q4_K_M.gguf \
  --input-tokens-file results/cuda-runtime-m6/accepted-f32/moe-prefill-input.json \
  --context 1024 --batch 128 --prefill-chunk 128 --threads 4 \
  --tokens 8 --warmups 3 --repetitions 7 --cuda-matmul tf32x3

# Fresh ABBA: output directory must not already exist. Never overwrite controls.
python3 scripts/benchmark_cuda_paired.py \
  --baseline results/cuda-runtime-m5/optimized-runtime \
  --candidate results/cuda-runtime-m6/accepted-runtime \
  --candidate-matmul tf32x3 --models moe \
  --cuda-root "$CUDA_ROOT" --output-dir results/cuda-m6-NEW-PAIR

# Full teacher-forced vectors plus three-warmup/seven-sample resident-model trials.
cargo run --release --features cuda,ktransformers --example cuda-precision -- \
  --model models/benchmark-qwen3/gguf/Qwen3-30B-A3B-Q4_K_M.gguf \
  --cases results/cuda-runtime-m6/moe-precision-cases.json \
  --matmul tf32x3 --output results/cuda-m6-NEW-PRECISION.json

cargo test --all-features --lib --tests
cargo test --no-default-features
cargo clippy --all-features --all-targets -- -D warnings
python3 -m unittest discover -s scripts -p 'test_*.py'
cargo test --release --features cuda --lib --test cuda -- \
  --ignored --skip benchmark_ --test-threads=1
python3 scripts/validate_cuda.py --cuda-root "$CUDA_ROOT" \
  --binary results/cuda-runtime-m6/accepted-runtime \
  --output-dir results/cuda-m6-NEW-VALIDATION
```

Cargo integration-test builds can replace `target/release/palito` with
a different feature-unification build. Always select the immutable executable
for validation; the validator records its hash before/after and all CUDA sources.

`examples/trace_reference.rs` and the external-only `native/logits_oracle.cpp`
produce create-new JSONL traces. The external build requires `-lggml-base` in
addition to `-lllama -lggml`; see M5's reproduction command. It is never linked
into the Rust numerical runtime. Actual commands, model/binary/source hashes,
raw samples, process resources, telemetry and source snapshots are indexed by
the [checkpoint](../results/cuda-runtime-m6/checkpoint.json).

## Remaining bottleneck and next bounded experiment

The scalar/grouped projection path still dominates. Next, prototype hoisting
the shared reduction workspace out of the six mutually exclusive format
specializations: the static report reserves 24 KiB where one four-token
reduction needs 4 KiB. Test actual NVRTC resources, exact numerical output and
the full paired matrix before any promotion. This is a hypothesis, not a
measured gain. A subsequent tensor-core experiment can use eight expert
assignments per group: current four-assignment groups mask half of the MMA's
eight columns. Neither experiment is smuggled into this checkpoint.

Broader quality evaluation, unperturbed external numerical diagnosis, actual
per-token latency distributions, concurrency and longer-context performance
remain unverified. Shared-device telemetry samples whole processes every
200 ms and cannot attribute a clock or power reading to each short request.
No per-request energy inference is made.
