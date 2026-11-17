"""Historical shared-kernel benchmark helpers; use benchmark_rust.py for the independent engine."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def run(command, timeout=300):
    completed = subprocess.run(
        [str(arg) for arg in command], capture_output=True, text=True, timeout=timeout, check=False
    )
    if completed.returncode:
        raise RuntimeError(
            f"Command failed ({completed.returncode}): {command}\n"
            f"{completed.stderr}\n{completed.stdout}"
        )
    return {
        "command": [str(arg) for arg in command],
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def digest(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def native_build_info(reference):
    directory = reference.parent
    settings = {}
    for line in (directory.parent / "CMakeCache.txt").read_text().splitlines():
        if "=" not in line or line.startswith(("//", "#")):
            continue
        key, value = line.split("=", 1)
        name = key.split(":", 1)[0]
        if name.startswith(("CMAKE_CXX_FLAGS", "GGML_")) or name in (
            "CMAKE_CXX_COMPILER", "CMAKE_BUILD_TYPE", "BUILD_SHARED_LIBS"
        ):
            settings[name] = value
    return {
        "cmake_settings": settings,
        "compiler_version": run([settings["CMAKE_CXX_COMPILER"], "--version"])["stdout"],
        "shared_library_sha256": {
            path.name: digest(path) for path in sorted(directory.glob("lib*.so*")) if path.is_file()
        },
    }


def validate_records(records, repetitions, generated_count):
    if not records or {record["runner"] for record in records} != {"rust", "cpp-reference"}:
        raise ValueError("Both Rust and C++ reference results are required")
    reference = None
    prompt_size = None
    prompt_ids = None
    for record in records:
        if record.get("schema_version") != 1:
            raise ValueError("Unsupported runner result schema")
        count = record["prompt_tokens"]
        if type(count) is not int or count <= 0:
            raise ValueError("Invalid prompt token count")
        if prompt_size is not None and count != prompt_size:
            raise ValueError("Prompt token counts differ")
        prompt_size = count
        ids = record["prompt_token_ids"]
        if (len(ids) != count or any(type(token) is not int or token < 0 for token in ids)
                or (prompt_ids is not None and ids != prompt_ids)):
            raise ValueError("Prompt token IDs differ or are invalid")
        prompt_ids = ids
        if len(record["samples"]) != repetitions:
            raise ValueError("Runner returned an incomplete sample set")
        for sample in record["samples"]:
            tokens = sample["generated_tokens"]
            if (len(tokens) != generated_count
                    or any(type(token) is not int or token < 0 for token in tokens)
                    or sample["decode_steps"] != generated_count - 1):
                raise ValueError("Incomplete generation or invalid token IDs")
            for name in ("prefill_ns", "ttft_ns", "decode_ns"):
                value = sample[name]
                if type(value) is not int or value < 0:
                    raise ValueError(f"Invalid timing: {name}")
            if (sample["prefill_ns"] <= 0 or sample["ttft_ns"] < sample["prefill_ns"]
                    or (generated_count > 1 and sample["decode_ns"] <= 0)
                    or (generated_count == 1 and sample["decode_ns"] != 0)):
                raise ValueError("Inconsistent phase timing")
            if reference is not None and tokens != reference:
                raise ValueError("Greedy token mismatch across runners or repeated requests")
            reference = tokens


def summarize(records):
    result = {}
    for runner in ("cpp-reference", "rust"):
        samples = [s for record in records if record["runner"] == runner for s in record["samples"]]
        prompt_count = next(record["prompt_tokens"] for record in records if record["runner"] == runner)
        result[runner] = {
            "samples": len(samples),
            "prefill_median_ms": statistics.median(s["prefill_ns"] for s in samples) / 1e6,
            "ttft_median_ms": statistics.median(s["ttft_ns"] for s in samples) / 1e6,
            "decode_median_ms": statistics.median(s["decode_ns"] for s in samples) / 1e6,
            "prefill_median_tokens_per_second": statistics.median(
                prompt_count * 1e9 / s["prefill_ns"] for s in samples),
            "decode_median_tokens_per_second": statistics.median(
                s["decode_steps"] * 1e9 / s["decode_ns"] if s["decode_steps"] else 0
                for s in samples),
        }
    result["rust_over_cpp_latency_ratio"] = {
        phase: result["rust"][f"{phase}_median_ms"] / result["cpp-reference"][f"{phase}_median_ms"]
        for phase in ("prefill", "ttft", "decode")
        if result["cpp-reference"][f"{phase}_median_ms"] > 0
    }
    return result


def positive(value):
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return result


def slowdown(value):
    result = float(value)
    if not math.isfinite(result) or result < 1:
        raise argparse.ArgumentTypeError("must be finite and at least 1")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=ROOT / "models/SmolLM2-135M-Instruct-Q4_K_M.gguf")
    parser.add_argument("--binary", type=Path, default=ROOT / "target/release/local-inference")
    parser.add_argument("--prompt-file", type=Path, default=ROOT / "fixtures/prompt.txt")
    parser.add_argument("--output", type=Path, default=ROOT / "results/baseline.json")
    parser.add_argument("--context", type=positive, default=512)
    parser.add_argument("--batch", type=positive, default=128)
    parser.add_argument("--prefill-chunk", type=positive, default=64)
    parser.add_argument("--threads", type=positive, default=4)
    parser.add_argument("--tokens", type=positive, default=32)
    parser.add_argument("--warmups", type=positive, default=1)
    parser.add_argument("--repetitions", type=positive, default=3)
    parser.add_argument("--rounds", type=positive, default=2)
    parser.add_argument("--max-slowdown", type=slowdown,
                        help="Optional noisy-machine-sensitive Rust/C++ latency gate, e.g. 1.20")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists; choose a new --output to preserve the baseline")
    if not 0 < args.prefill_chunk <= args.batch <= args.context <= 2**31 - 1:
        parser.error("require 0 < chunk <= batch <= context <= i32::MAX")
    for name in ("model", "binary", "prompt_file"):
        path = getattr(args, name).resolve(strict=True)
        if not path.is_file():
            parser.error(f"{name} must be a regular file")
        setattr(args, name, path)
    paths = json.loads(run([args.binary, "paths"])["stdout"])
    if paths.get("build_profile") != "release":
        parser.error("benchmark requires a release Rust binary; run cargo build --release")
    cpp = [paths["native_reference"], args.model, args.prompt_file, args.context,
           args.batch, args.prefill_chunk, args.threads, args.tokens, args.warmups, args.repetitions, "v1"]
    rust = [args.binary, "bench", "--model", args.model, "--prompt-file", args.prompt_file,
            "--context", args.context, "--batch", args.batch, "--prefill-chunk", args.prefill_chunk,
            "--threads", args.threads, "--tokens", args.tokens, "--warmups", args.warmups,
            "--repetitions", args.repetitions]
    invocations = []
    records = []
    started = datetime.now(timezone.utc).isoformat()
    load_start = os.getloadavg()
    for round_index in range(args.rounds):
        # Alternate whole-process order to reduce systematic warm-cache/thermal bias.
        commands = [cpp, rust] if round_index % 2 == 0 else [rust, cpp]
        for command in commands:
            invocation = run(command)
            records.append(json.loads(invocation["stdout"]))
            invocations.append(invocation)
    official = run([
        paths["llama_bench"], "-m", args.model, "-p", args.batch, "-n", args.tokens,
        "-b", args.batch, "-ub", args.batch, "-t", args.threads, "-ngl", "0",
        "-fa", "0", "-r", args.repetitions, "-o", "json",
    ])
    official["results"] = json.loads(official["stdout"])
    failure = None
    summary = None
    try:
        validate_records(records, args.repetitions, args.tokens)
        summary = summarize(records)
        if args.max_slowdown is not None and any(
            value > args.max_slowdown
            for value in summary["rust_over_cpp_latency_ratio"].values()
        ):
            failure = f"Rust/C++ phase latency exceeds --max-slowdown={args.max_slowdown}"
    except ValueError as error:
        failure = str(error)
    source = ROOT.parent / "llama.cpp"
    report = {
        "schema_version": 1,
        "started_at": started,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "passed": failure is None,
        "failure": failure,
        "protocol": {
            "sampling": "greedy, fixed-length including EOG; no chat template",
            "timing": "synchronized prefill and decode; reset/load/tokenization excluded from phase timings",
            "ttft": "prefill + first sample, not client/network/end-to-end TTFT",
            "decode": "N-1 evaluations/samples after first token",
            "cache": "full request KV reset before every warmup and measured request; OS pages remain warm",
            "comparison": "C++ and Rust control loops use the same minimal C bridge and native kernels",
            "official_baseline": "unmodified llama-bench uses synthetic tokens, not the fixed prompt; no direct ratio",
        },
        "settings": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "provenance": {
            "model_sha256": digest(args.model),
            "model_size_bytes": args.model.stat().st_size,
            "prompt_sha256": digest(args.prompt_file),
            "rust_binary_sha256": digest(args.binary),
            "cpp_binary_sha256": digest(Path(paths["native_reference"])),
            "cargo_lock_sha256": digest(ROOT / "Cargo.lock"),
            "rustc_version": run(["rustc", "--version"])["stdout"].strip(),
            "native_artifacts": native_build_info(Path(paths["native_reference"])),
            "llama_revision": paths["llama_revision"],
            "llama_worktree": run(["git", "-C", source, "status", "--porcelain"])["stdout"],
            "platform": platform.platform(),
            "cpu_model": next(line.split(":", 1)[1].strip()
                              for line in Path("/proc/cpuinfo").read_text().splitlines()
                              if line.startswith("model name")),
            "cpu_affinity": sorted(os.sched_getaffinity(0)),
            "load_average_start": load_start,
            "load_average_end": os.getloadavg(),
            "native_build": "Release, CPU, GGML_NATIVE=ON, CUDA/BLAS/OpenMP=OFF",
        },
        "summary": summary,
        "records": records,
        "invocations": invocations,
        "official_llama_bench": official,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=args.output.parent, delete=False) as output:
            temporary = Path(output.name)
            json.dump(report, output, indent=2, allow_nan=False)
            output.write("\n")
        temporary.replace(args.output)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    print(json.dumps({"passed": report["passed"], "failure": failure, "summary": summary}, indent=2))
    print(f"Full report: {args.output.resolve()}")
    return 1 if failure else 0


if __name__ == "__main__":
    raise SystemExit(
        "The shared-kernel benchmark is retired. Use scripts/benchmark_rust.py "
        "--reference /path/to/external/li-reference --output-dir results/NEW_RUN."
    )
