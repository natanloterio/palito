"""Measure CPU execution policies and validate optimized math against a preserved float oracle."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess

from benchmark import ROOT, digest, positive, run
from benchmark_rust import summarize, validate
from validate_rust_logits import compare_logits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-dir", type=Path, required=True)
    parser.add_argument("--oracle-dir", type=Path, required=True,
                        help="Previously verified F32-oracle report with raw logits and provenance")
    parser.add_argument("--model", type=Path, default=ROOT / "models/SmolLM2-135M-Instruct-Q4_K_M.gguf")
    parser.add_argument("--rust", type=Path, default=ROOT / "target/release/local-inference")
    parser.add_argument("--repetitions", type=positive, default=3)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    model, rust, benchmark, oracle = (p.resolve(strict=True) for p in
                                     (args.model, args.rust, args.benchmark_dir, args.oracle_dir))
    directory = args.output_dir.resolve()
    directory.mkdir(parents=True, exist_ok=False)

    def save(name, value):
        with (directory / name).open("x") as output:
            json.dump(value, output, indent=2, allow_nan=False)
            output.write("\n")

    def read_record(path):
        return json.loads(json.loads(path.read_text())["stdout"])

    provenance = json.loads((benchmark / "provenance.json").read_text())
    oracle_provenance = json.loads((oracle / "provenance.json").read_text())
    if not json.loads((oracle / "validation.json").read_text())["passed"]:
        raise ValueError("oracle report did not pass")
    if (digest(rust) != provenance["rust_sha256"]
            or digest(model) != provenance["model_sha256"]
            or digest(model) != oracle_provenance["model_sha256"]):
        raise ValueError("benchmark binary or model provenance does not match")
    save("provenance.json", {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "model_sha256": digest(model), "rust_sha256": digest(rust),
        "benchmark_provenance_sha256": digest(benchmark / "provenance.json"),
        "oracle_provenance_sha256": digest(oracle / "provenance.json"),
        "oracle_validation_sha256": digest(oracle / "validation.json"),
        "oracle_report": str(oracle), "benchmark_report": str(benchmark),
        "source_sha256": {str(p.relative_to(ROOT)): digest(p) for p in sorted((ROOT / "src").rglob("*.rs"))},
        "script_sha256": digest(Path(__file__)),
        "affinity": sorted(os.sched_getaffinity(0)), "load_before": os.getloadavg(),
        "controls": "Sequential processes; four CPU threads; one warmup; no exclusive hardware reservation",
    })
    numerical = {}
    for name in ("short", "decode-heavy", "exact-context", "first-divergence"):
        token_file = oracle / f"{name}-input.json"
        expected = read_record(oracle / f"{name}-reference.json")["logits"]
        old_rust = read_record(oracle / f"{name}-rust.json")["logits"]
        numerical[name] = {}
        for variant, flags in [
            ("packed", []),
            ("scalar-f32", ["--f32-weights", "--sequential-prefill", "--scalar-kernels"]),
        ]:
            record = run([rust, "logits", "--model", model, "--input-tokens-file", token_file, *flags])
            save(f"{name}-{variant}-logits.json", record)
            logits = json.loads(record["stdout"])["logits"]
            metrics = compare_logits(logits, expected)
            metrics["against_previous_rust"] = compare_logits(logits, old_rust)
            metrics["passed"] = (
                metrics["cosine_similarity"] >= 0.9999 and metrics["rmse"] <= 0.01
                and metrics["max_absolute_error"] <= 0.05
                and metrics["rust_top_token"] == metrics["reference_top_token"]
                and (variant != "scalar-f32" or logits == old_rust)
            )
            numerical[name][variant] = metrics
            print("numerical", name, variant, metrics["passed"], flush=True)

    suite = json.loads((benchmark / "comparison.json").read_text())["summary"]
    results = {}
    variants = [
        ("f32-reference", ["short", "prefill-heavy", "decode-heavy"],
         ["--f32-weights", "--sequential-prefill", "--scalar-kernels"]),
        ("f32-batched", ["short", "prefill-heavy", "decode-heavy"], ["--f32-weights"]),
        ("packed-sequential", ["short", "prefill-heavy", "decode-heavy"], ["--sequential-prefill"]),
        ("packed-portable", ["short"], ["--scalar-kernels"]),
        ("prefix-cache", ["short"], ["--prefix-cache-bytes", 8 * 1024 * 1024]),
        ("speculation", ["decode-heavy", "prefill-heavy"], ["--speculative-window", 4]),
        ("prefix-and-speculation", ["short"],
         ["--prefix-cache-bytes", 8 * 1024 * 1024, "--speculative-window", 4]),
    ]
    for variant, cases, flags in variants:
        for name in cases:
            case = suite[name]
            token_file = benchmark / f"{name}-input.json"
            prompt = json.loads(token_file.read_text())
            expected = read_record(benchmark / f"{name}-rust.json")["samples"][0]["generated_tokens"]
            output = run([
                rust, "bench", "--model", model, "--input-tokens-file", token_file,
                "--context", case["context"], "--batch", 128, "--prefill-chunk", 64,
                "--threads", 4, "--tokens", case["generated_tokens"],
                "--warmups", 1, "--repetitions", args.repetitions, *flags,
            ])
            save(f"{variant}-{name}.json", output)
            record = json.loads(output["stdout"])
            cached = "prefix" in variant
            validate(record, prompt, case["generated_tokens"], args.repetitions, allow_cached=cached)
            same = all(sample["generated_tokens"] == expected for sample in record["samples"])
            cache_valid = all(sample["cached_prompt_tokens"] == (len(prompt) if cached else 0)
                              for sample in record["samples"])
            stats = record["samples"][0]["speculation"]
            stats_valid = stats is None or (
                0 <= stats["accepted_tokens"] <= stats["proposed_tokens"]
                and stats["verification_batches"] >= 0)
            result = {
                **summarize(record), "memory": record["memory"],
                "cached_prompt_tokens": record["samples"][0]["cached_prompt_tokens"],
                "speculation": stats, "tokens_equal_to_cold_packed": same,
                "passed": same and cache_valid and stats_valid,
            }
            results.setdefault(variant, {})[name] = result
            print(variant, name, json.dumps(result), flush=True)

    short = read_record(benchmark / "short-rust.json")
    reservation = short["memory"]["numerical_reservation_bytes"]
    budget_checks = {}
    for name, limit, success in [("exact-budget", reservation, True),
                                 ("one-byte-under", reservation - 1, False)]:
        command = [str(x) for x in [
            rust, "bench", "--model", model, "--input-tokens-file", benchmark / "short-input.json",
            "--tokens", 8, "--warmups", 0, "--repetitions", 1, "--memory-limit-bytes", limit]]
        output = subprocess.run(command, capture_output=True, text=True, timeout=300, check=False)
        save(f"{name}.json", {"command": command, "returncode": output.returncode,
                              "stdout": output.stdout, "stderr": output.stderr})
        budget_checks[name] = ((output.returncode == 0) if success else
                               (output.returncode != 0 and "exceeds memory limit" in output.stderr))
    prefill_cases = {key: suite[key]["rust"]["prefill_median_ms"]
                     for key in ("short", "chunk-one", "chunk-seven")}
    packed_memory = short["memory"]["weights_allocated_bytes"]
    f32_memory = results["f32-reference"]["short"]["memory"]["weights_allocated_bytes"]
    verified = sum(result["speculation"]["verification_batches"]
                   for cases in results.values() for result in cases.values()
                   if result["speculation"] is not None)
    passed = (all(check["passed"] for checks in numerical.values() for check in checks.values())
              and all(result["passed"] for cases in results.values() for result in cases.values())
              and all(budget_checks.values()) and verified > 0 and packed_memory < f32_memory / 3)
    save("validation.json", {
        "finished_at": datetime.now(timezone.utc).isoformat(), "passed": passed,
        "numerical": numerical, "variants": results, "budget_checks": budget_checks,
        "weights": {"packed_bytes": packed_memory, "f32_bytes": f32_memory,
                    "reduction_fraction": 1 - packed_memory / f32_memory},
        "policy_observation": {
            "short_prefill_ms_by_case": prefill_cases,
            "fastest_measured_case": min(prefill_cases, key=prefill_cases.get),
            "scope": "Observed medians on this shared machine; no automatic policy change or general optimality claim",
        },
        "limitations": "Prefix runs intentionally reuse warmed cache and are not cold-inference speed comparisons; speculation is opt-in and may be slower.",
    })
    if not passed:
        raise ValueError("CPU optimization validation failed; see saved reports")


if __name__ == "__main__":
    main()
