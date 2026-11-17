# MoE numerical discrepancy: matched-precision diagnosis and validation fix

## Outcome

The outstanding discrepancy was a mismatched matrix-arithmetic comparison, not
evidence of an incorrect Rust Qwen3-MoE graph. The unchanged, preserved M6 Rust
executable agrees with a separate pinned external graph when matrix products use
dequantized weights and F32 activations.

The fix is to the external correctness oracle and validation protocol.
**No Rust numerical code, activation precision, quality threshold, dependency,
or performance baseline was changed.** This does not make optimized llama.cpp's
outputs identical to Rust's: those backends deliberately use different arithmetic.

## Causal evidence

Model: original `Qwen3-30B-A3B-Q4_K_M.gguf`, not a separately requantized model.
External source: `6b01a4767f8d3d0c416c410a49861333965c0982`.
The original [19-token shared prefix](../results/cuda-runtime-m5/divergence-input.json)
is tested without tensor callbacks. F32 KV, context 1024, batch/microbatch 128,
four CPU threads, affinity 0-3; no expanded F32 model is generated.

The external packed CPU products convert activations to their vector-dot type
(Q8 for quantized weights; other storage types can also convert activations).
The external CUDA packed paths also quantize activations. Rust instead multiplies
dequantized weight values by F32 activations. Small projection differences can
affect head normalization and MoE expert membership.

Two builds use the **same pinned graph, patch, compiler configuration, disabled
fusion/repacking, and CPU execution**. Only `GGML_F32_REFERENCE_SOURCE` differs:

- [F32 helper](../native/f32_oracle.c): row-wise weight dequantization, untouched
  F32 activations, F64 dot accumulation, F32 output.
- [Packed negative control](../native/packed_oracle_control.c): returns false
  from the interception hook, retaining original GGML matrix products.

| Comparison at original divergent prefix | Max absolute error | RMSE | Argmax |
|---|---:|---:|---|
| Rust CUDA vs original optimized external GPU | 0.420568 | 0.088601 | 7269 vs 614 |
| Rust CUDA vs controlled packed external CPU | 0.657070 | 0.139573 | 7269 vs 614 |
| Rust CUDA vs matched-F32 external CPU | 0.00000524526 | 0.000000863393 | 7269 vs 7269 |
| Rust CPU vs matched-F32 external CPU | 0.0000114638 | 0.00000372138 | 7269 vs 7269 |

Replacing packed products alone removes the large error and token disagreement;
restoring them restores both. This isolates the matrix-product arithmetic as the
cause without relying on perturbed layer traces. Activation conversion is the
principal precision mismatch; the oracle is not a bitwise reproduction of
optimized packed reductions. The previously observed tracing perturbation is
not used as proof, and its historical evidence remains untouched.

Full vectors, commands, timings and hashes:

- [Matched validation](../results/moe-numerical-fix/matched-validation/validation.json).
- [Input/model/executable provenance](../results/moe-numerical-fix/matched-validation/provenance.json).
- [Controlled arithmetic comparison](../results/moe-numerical-fix/controlled-arithmetic-comparison.json).
- [Packed control build/run records](../results/moe-numerical-fix/controlled-packed/).
- [F32 build manifest](../results/moe-numerical-fix/reproducible-build/manifest.json).

## Regression gates

All **151,936 logits**, not merely top tokens, pass on five teacher-forced
prefixes. Each case checks independent Rust CPU and default-F32 Rust CUDA
against the untraced external graph.

| Prefix length | Rust CPU max error | Rust CUDA max error | Equal argmax |
|---:|---:|---:|---|
| 1 | 0.0000603046 | 0.0000121739 | Yes |
| 7 | 0.0000200309 | 0.0000102516 | Yes |
| 19, original divergence | 0.0000114638 | 0.00000524526 | Yes |
| 46, canonical prompt | 0.0000243164 | 0.0000143055 | Yes |
| 256, multi-chunk prefill | 0.0000591244 | 0.0000572170 | Yes |

Gates: finite vectors, equal shape and argmax, cosine >= 0.99999, max absolute
error <= 0.01, RMSE <= 0.01. These retain the existing CUDA gate and are stricter
than the old external max-error/cosine limits; none were relaxed. Worst RMSE is
0.0000115480; minimum cosine exceeds 0.999999999997.

The [native graph test](../native/test_f32_oracle.cpp) additionally passes **48**
cases: F32/F16/Q5_0/Q8_0/Q4_K/Q6_K weights; ordinary products, broadcast batches,
expert gate/up and expert-down layouts; strided/offset activations; seven tokens,
five output rows, three experts and two selected experts; one/three CPU threads.
Non-F32-weight cases exactly match direct dequantized-weight/F64-dot expectations.
F32-weight products retain GGML's implementation and allow 0.00002 reduction error.
The same executable rejects the original packed backend, rather than silently
accepting lower-precision products.

`--f32-reference` checks a CPU backend marker before model loading and rejects
the packed control. Default external CLI usage and optional tracing remain
supported. Python syntax checks pass for both new scripts.

## Reproduce

From the runtime directory, with the existing reference checkout and model:

```sh
python3 scripts/build_f32_oracle.py \
  --output-dir results/f32-oracle-build-NEW

CUDA_ROOT="$PWD/.venv-cuda-bench/lib/python3.11/site-packages/nvidia/cu13"
python3 scripts/validate_moe_f32_oracle.py \
  --output-dir results/f32-oracle-validation-NEW \
  --binary results/cuda-runtime-m6/accepted-runtime \
  --reference target/llama-f32-oracle/bin/logits-oracle \
  --cuda-root "$CUDA_ROOT"
```

The builder creates a detached pinned worktree, applies the
[external-only patch](../native/f32-oracle.patch), disables CUDA, repacking,
BLAS, OpenMP and llamafile products, builds the external tool, runs its graph
tests, and records source/library/executable hashes. Unexpected worktree changes
are rejected, not overwritten. The Rust executable and model are hashed before
and after validation, as are the external executable and libraries.

For the causal negative control, replay the recorded
[configure](../results/moe-numerical-fix/controlled-packed/configure.json),
[build](../results/moe-numerical-fix/controlled-packed/build.json),
[link](../results/moe-numerical-fix/controlled-packed/link.json) and
[run](../results/moe-numerical-fix/controlled-packed/divergence.json) commands.
The control differs from the F32 build only in the helper source and build/output
directory. Both builds use `use_ref=true`; this is not a comparison of tracing,
fusion, graph architecture or performance settings.

## Boundaries

- The helper allocates one dequantized weight row per CPU thread, not a roughly
  120 GB expanded model. Original packed model storage and graph buffers remain.
- Rust never links GGML/llama.cpp or invokes this oracle for inference.
- These are correctness runs, not speed benchmarks. The original optimized
  llama.cpp binaries remain the performance baseline and still outperform Rust.
- This establishes the bounded matched-precision cases above, not bitwise
  equivalence, broad task-quality parity, all possible expert selections, or
  token equality with lower-precision optimized backends.
- No GPU reset, clock/power/kernel setting change, download or unrelated process
  termination was used. No owned inference process remained after validation;
  GPU returned to 0% utilization and desktop-only memory use, with swap unused.
