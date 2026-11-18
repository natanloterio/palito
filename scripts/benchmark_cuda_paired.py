"""Sequential ABBA controls with immutable executables, full outputs and GPU telemetry."""
import argparse
import json
from pathlib import Path
import shutil
import statistics
import subprocess
import sys

from benchmark import digest
from benchmark_cuda_iteration import CASES, MODELS, ROOT
from benchmark_moe import write_new
from validate_cuda import compare_logits


def validate_controls(reports, provenance, candidate_matmul, *, candidate_decode_graph=False,
                      candidate_batch=None, candidate_chunk=None, allowed_reservation_increase=0,
                      baseline_matmul="f32", baseline_decode_graph=False):
    if allowed_reservation_increase < 0:
        raise ValueError("reservation allowance must be nonnegative")
    baseline = provenance["a1"]
    expected = reports["a1"]["controls"]
    for label, report in reports.items():
        if (provenance[label]["models"] != baseline["models"]
                or provenance[label]["affinity"] != baseline["affinity"]):
            raise ValueError(f"{label}: model identity or CPU affinity changed")
        controls = dict(report["controls"])
        mode = controls.pop("matmul", "f32")
        graph = controls.pop("decode_graph", False)
        if graph != (candidate_decode_graph if label[0] == "b" else baseline_decode_graph):
            raise ValueError(f"{label}: wrong decode graph policy")
        for field, selected in [("batch", candidate_batch), ("chunk", candidate_chunk)]:
            if selected is not None and label[0] == "b":
                if controls.get(field) != selected:
                    raise ValueError(f"{label}: wrong {field}")
                controls[field] = expected[field]
        if mode != (candidate_matmul if label[0] == "b" else baseline_matmul):
            raise ValueError(f"{label}: wrong precision mode")
        if controls != {k: v for k, v in expected.items() if k not in ("matmul", "decode_graph")}:
            raise ValueError(f"{label}: benchmark controls changed")
        for name, result in report["results"].items():
            allowance = allowed_reservation_increase if label[0] == "b" else 0
            if result["memory"]["numerical_reservation_bytes"] > reports["a1"]["results"][name]["memory"]["numerical_reservation_bytes"] + allowance:
                raise ValueError(f"{label}/{name}: numerical reservation increased")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--baseline-matmul", choices=["f32", "tf32", "tf32x3", "f32-tiled", "tf32x3-tiled", "residual-mma3-research"], default="f32")
    parser.add_argument("--baseline-decode-graph", action="store_true")
    parser.add_argument("--baseline-batch", type=int, default=128)
    parser.add_argument("--baseline-chunk", type=int, default=128)
    parser.add_argument("--candidate-matmul", choices=["f32", "tf32", "tf32x3", "f32-tiled", "tf32x3-tiled", "residual-mma3-research"], default="f32")
    parser.add_argument("--candidate-decode-graph", action="store_true")
    parser.add_argument("--candidate-batch", type=int, default=128)
    parser.add_argument("--candidate-chunk", type=int, default=128)
    parser.add_argument("--allowed-reservation-increase", type=int, default=0,
                        help="Explicit bounded candidate workspace allowance in bytes; baseline remains exact")
    parser.add_argument("--llama", action="store_true", help="Run pinned F32/F16 llama controls after the final baseline run")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cuda-root", type=Path, required=True)
    parser.add_argument("--models", nargs="+", choices=list(MODELS), default=list(MODELS))
    parser.add_argument("--cases", nargs="+", choices=list(CASES), default=list(CASES))
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--repetitions", type=int, default=7)
    args = parser.parse_args()
    if args.warmups < 1 or args.repetitions < 3 or args.allowed_reservation_increase < 0:
        parser.error("at least one warmup and three repetitions required")
    if not 1 <= args.candidate_chunk <= args.candidate_batch <= 512:
        parser.error("require 1 <= candidate chunk <= candidate batch <= 512")
    if not 1 <= args.baseline_chunk <= args.baseline_batch <= 512:
        parser.error("require 1 <= baseline chunk <= baseline batch <= 512")
    root = args.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=False)
    binaries = {}
    for label, path in [("a", args.baseline), ("b", args.candidate)]:
        destination = root / f"{label}-runtime"
        shutil.copy2(path.resolve(strict=True), destination)
        binaries[label] = destination
    order = ["a1", "b1", "b2", "a2"]
    commands = []
    for label in order:
        command = [
            sys.executable, str(ROOT / "scripts/benchmark_cuda_iteration.py"),
            "--output-dir", str(root), "--label", label, "--binary", str(binaries[label[0]]),
            "--cuda-root", str(args.cuda_root.resolve(strict=True)),
            "--warmups", str(args.warmups), "--repetitions", str(args.repetitions),
            "--models", *args.models, "--cases", *args.cases,
        ]
        commands.append(command)
        if label[0] == "b":
            command.extend(["--cuda-matmul", args.candidate_matmul,
                            "--batch", str(args.candidate_batch), "--chunk", str(args.candidate_chunk)])
            if args.candidate_decode_graph:
                command.append("--cuda-decode-graph")
        else:
            command.extend(["--cuda-matmul", args.baseline_matmul,
                            "--batch", str(args.baseline_batch), "--chunk", str(args.baseline_chunk)])
            if args.baseline_decode_graph:
                command.append("--cuda-decode-graph")
        if label == "a2" and args.llama:
            command.append("--llama")
        write_new(root / "commands.json" if label == "a1" else root / f"{label}-command.json",
                  {"order": order, "command": command})
        with (root / f"{label}.log").open("x") as log:
            subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
    reports = {label: json.loads((root / label / "summary.json").read_text()) for label in order}
    provenance = {label: json.loads((root / label / "provenance.json").read_text()) for label in order}
    validate_controls(reports, provenance, args.candidate_matmul,
                      candidate_decode_graph=args.candidate_decode_graph,
                      candidate_batch=args.candidate_batch, candidate_chunk=args.candidate_chunk,
                      allowed_reservation_increase=args.allowed_reservation_increase,
                      baseline_matmul=args.baseline_matmul,
                      baseline_decode_graph=args.baseline_decode_graph)
    results = {}
    for name in reports["a1"]["results"]:
        samples = {}
        records = {}
        for label in order:
            record = json.loads((root / label / f"{name}.json").read_text())
            records[label] = record
            samples[label] = [(s["ttft_ns"] + s["decode_ns"]) / 1e6 for s in record["samples"]]
            if record["prompt_token_ids"] != records["a1"]["prompt_token_ids"]:
                raise ValueError(f"{name}: prompt IDs changed")
            if any(s["generated_tokens"] != records["a1"]["samples"][0]["generated_tokens"]
                   for s in record["samples"]):
                raise ValueError(f"{name}: arithmetic-preserving candidate changed generated IDs")
        results[name] = {
            "request_speedup_by_order": [
                statistics.median(samples[a]) / statistics.median(samples[b])
                for a, b in [("a1", "b1"), ("a2", "b2")]
            ],
            "pooled_request_speedup": statistics.median(samples["a1"] + samples["a2"])
            / statistics.median(samples["b1"] + samples["b2"]),
            "prefill_speedup_by_order": [
                reports[a]["results"][name]["prefill_ms"]["median"]
                / reports[b]["results"][name]["prefill_ms"]["median"]
                for a, b in [("a1", "b1"), ("a2", "b2")]
            ],
            "decode_speedup_by_order": [
                reports[a]["results"][name]["decode_ms"]["median"]
                / reports[b]["results"][name]["decode_ms"]["median"]
                if reports[b]["results"][name]["decode_ms"]["median"] else None
                for a, b in [("a1", "b1"), ("a2", "b2")]
            ],
            "all_ids_exact": True,
            "raw_request_ms": samples,
        }
    long_logits = {}
    if "long" in args.cases:
        for model in args.models:
            expected = None
            for label in order:
                process = json.loads((root / label / f"{model}-long-logits-process.json").read_text())
                actual = json.loads(process["stdout"])["logits"]
                if expected is not None:
                    if args.candidate_matmul == args.baseline_matmul == "f32" and actual != expected:
                        raise ValueError(f"{model}: long-prefix logits changed")
                    long_logits[f"{model}-{label}"] = compare_logits(expected, actual)
                if expected is None:
                    expected = actual
    report = {
        "order": order, "warmups": args.warmups, "repetitions_per_run": args.repetitions,
        "binary_sha256": {label: digest(path) for label, path in binaries.items()},
        "results": results, "commands": commands, "long_logits": long_logits,
        "candidate_matmul": args.candidate_matmul,
        "baseline_matmul": args.baseline_matmul,
        "baseline_decode_graph": args.baseline_decode_graph,
        "baseline_batch": args.baseline_batch, "baseline_chunk": args.baseline_chunk,
        "candidate_decode_graph": args.candidate_decode_graph,
        "candidate_batch": args.candidate_batch, "candidate_chunk": args.candidate_chunk,
        "allowed_reservation_increase": args.allowed_reservation_increase,
        "limitations": [
            "Two opposite-order contrasts are not a confidence interval or proof of statistical significance.",
            "Shared hardware, no clock normalization or exclusive window; inspect recorded telemetry.",
            "Telemetry polls every 200ms and cannot establish a clock/utilization trace for each short request.",
        ],
    }
    write_new(root / "paired.json", report)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
