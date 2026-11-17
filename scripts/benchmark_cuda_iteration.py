"""Reproducible, sequential single-request CUDA optimization checkpoints."""
import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import signal
import statistics
import subprocess
import threading
import time

from benchmark import digest
from benchmark_moe import ROOT, gpu_state, http_json, server, validate_llama_sample, write_new
from benchmark_rust import validate

CASES = {"decode": (1, 32), "short": (46, 8), "prefill": (256, 8), "long": (512, 1)}
MODELS = {
    "dense": ROOT / "models/SmolLM2-135M-Instruct-Q4_K_M.gguf",
    "moe": ROOT / "models/benchmark-qwen3/gguf/Qwen3-30B-A3B-Q4_K_M.gguf",
}

def capture_process(command, environment, timeout=600, cleanup_timeout=5):
    process = subprocess.Popen(command, env=environment, stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, text=True, start_new_session=True)
    timed_out = False
    cleanup_incomplete = False
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired as deadline:
        timed_out = True
        for sig in [signal.SIGTERM, signal.SIGKILL]:
            try:
                os.killpg(process.pid, sig)
            except ProcessLookupError:
                pass  # The explicitly owned group already exited.
            try:
                stdout, stderr = process.communicate(timeout=cleanup_timeout)
                break
            except subprocess.TimeoutExpired as expired:
                deadline = expired
        else:
            cleanup_incomplete = True
            decode = lambda value: value.decode(errors="replace") if isinstance(value, bytes) else value or ""
            stdout, stderr = decode(deadline.stdout), decode(deadline.stderr)
            # A kernel-blocked descendant may retain pipes even after the parent dies.
            process.stdout.close()
            process.stderr.close()
    return {"pid": process.pid, "returncode": process.poll(), "stdout": stdout, "stderr": stderr,
            "timed_out": timed_out, "cleanup_incomplete": cleanup_incomplete}


def distribution(values):
    if not values or any(not isinstance(x, (int, float)) or not 0 <= x < float("inf") for x in values):
        raise ValueError("invalid or empty measurements")
    ordered = sorted(values)
    def percentile(q):
        position = (len(ordered) - 1) * q
        low = int(position)
        return ordered[low] + (ordered[min(low + 1, len(ordered) - 1)] - ordered[low]) * (position - low)
    return {"median": statistics.median(values), "min": min(values), "max": max(values),
            "p95_descriptive": percentile(.95), "p99_descriptive": percentile(.99)}


def summarize(record):
    samples = record["samples"]
    return {
        **{f"{phase}_ms": distribution([s[f"{phase}_ns"] / 1e6 for s in samples])
           for phase in ["prefill", "ttft", "decode"]},
        "synchronized_request_ms": distribution([(s["ttft_ns"] + s["decode_ns"]) / 1e6 for s in samples]),
        "prefill_tokens_per_second": distribution([record["prompt_tokens"] * 1e9 / s["prefill_ns"] for s in samples]),
        "decode_tokens_per_second": distribution([s["decode_steps"] * 1e9 / s["decode_ns"] for s in samples])
        if samples[0]["decode_steps"] else None,
        "mean_inter_token_ms_per_request": distribution([s["decode_ns"] / s["decode_steps"] / 1e6 for s in samples])
        if samples[0]["decode_steps"] else None,
        "load_ms": record["load_ns"] / 1e6, "memory": record["memory"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--binary", type=Path, default=ROOT / "target/release/local-inference")
    parser.add_argument("--cuda-root", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--compare-to", type=Path)
    parser.add_argument("--llama", action="store_true", help="Run pinned all-GPU llama.cpp F32 and F16 KV controls")
    parser.add_argument("--repetitions", type=int, default=7)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--cuda-matmul", choices=["f32", "tf32", "tf32x3", "f32-tiled", "tf32x3-tiled"], default="f32")
    parser.add_argument("--cuda-decode-graph", action="store_true")
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--chunk", type=int, default=128)
    parser.add_argument("--models", nargs="+", choices=list(MODELS), default=list(MODELS))
    parser.add_argument("--cases", nargs="+", choices=list(CASES), default=list(CASES))
    parser.add_argument("--process-timeout", type=int, default=180)
    args = parser.parse_args()
    precision = [] if args.cuda_matmul == "f32" else ["--cuda-matmul", args.cuda_matmul]
    if args.cuda_decode_graph:
        precision.append("--cuda-decode-graph")
    if args.repetitions < 3 or args.warmups < 1 or args.process_timeout <= 0:
        parser.error("at least three repetitions, one warmup and a positive process timeout required")
    if not 1 <= args.chunk <= args.batch <= 512:
        parser.error("require 1 <= chunk <= batch <= 512")
    root = args.output_dir.resolve()
    directory = root / args.label
    directory.mkdir(parents=True, exist_ok=False)
    binary = args.binary.resolve(strict=True)
    cuda = args.cuda_root.resolve(strict=True)
    environment = dict(os.environ, OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4", MKL_NUM_THREADS="4")
    environment["LD_LIBRARY_PATH"] = os.pathsep.join(filter(None, [str(cuda / "lib"), environment.get("LD_LIBRARY_PATH")]))
    affinity = sorted(os.sched_getaffinity(0))[:4]
    pin = ["taskset", "--cpu-list", ",".join(map(str, affinity))]
    sources = [ROOT / "Cargo.toml", ROOT / "Cargo.lock", Path(__file__),
               *sorted((ROOT / "src").rglob("*.rs")), *sorted((ROOT / "src").rglob("*.cu"))]
    provenance = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "binary": str(binary), "binary_sha256": digest(binary),
        "models": {name: {"path": str(path), "sha256": digest(path)} for name, path in MODELS.items()},
        "source_sha256": {str(p.relative_to(ROOT)): digest(p) for p in sources},
        "affinity": affinity, "gpu_before": gpu_state(), "load_before": os.getloadavg(),
        "platform": platform.platform(),
        "rustc": subprocess.check_output(["rustc", "-Vv"], text=True),
        "cuda": subprocess.check_output([str(cuda / "bin/nvcc"), "--version"], text=True),
        "llama_revision": subprocess.check_output(["git", "-C", str(ROOT.parent / "llama.cpp"), "rev-parse", "HEAD"], text=True).strip(),
        "llama_dirty": subprocess.check_output(["git", "-C", str(ROOT.parent / "llama.cpp"), "status", "--short"], text=True),
        "linkage": subprocess.check_output(["ldd", str(binary)], text=True),
        "cpu": subprocess.check_output(["lscpu"], text=True),
    }
    if any(name in provenance["linkage"] for name in ["libllama", "libggml", "libli_bridge"]):
        raise ValueError("Rust executable depends on forbidden reference engine")
    write_new(directory / "provenance.json", provenance)
    gpu_samples, sampling_errors = [], []
    stop = threading.Event()
    def monitor():
        while not stop.is_set():
            try:
                result = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu,clocks.current.sm,clocks.current.memory,temperature.gpu,power.draw",
                                         "--format=csv,noheader,nounits"],
                                        text=True, capture_output=True, timeout=5)
            except subprocess.TimeoutExpired as error:
                sampling_errors.append(f"GPU telemetry timed out: {error}")
                return
            if result.returncode:
                sampling_errors.append(result.stderr)
                return
            try:
                for row in csv.reader(result.stdout.splitlines()):
                    if len(row) != 7:
                        raise ValueError(f"expected seven GPU telemetry fields, got {row}")
                    index, memory, utilization, sm, memory_clock, temperature, power = map(float, row)
                    gpu_samples.append({
                        "monotonic_ns": time.monotonic_ns(), "device": int(index),
                        "used_mib": memory, "utilization_percent": utilization,
                        "sm_clock_mhz": sm, "memory_clock_mhz": memory_clock,
                        "temperature_c": temperature, "power_w": power,
                    })
            except ValueError as error:
                sampling_errors.append(f"invalid or unsupported GPU telemetry: {error}")
                return
            stop.wait(.2)

    def run(name, command):
        timer = directory / f"{name}-resources.txt"
        command = ["/usr/bin/time", "-v", "-o", str(timer), *pin, *map(str, command)]
        thread = threading.Thread(target=monitor)
        stop.clear()
        start_index = len(gpu_samples)
        thread.start()
        started = time.monotonic_ns()
        try:
            result = capture_process(command, environment, timeout=args.process_timeout)
        finally:
            stop.set()
            thread.join()
        raw = {"command": command, "wall_ns": time.monotonic_ns() - started,
               **result,
               "gpu_samples": gpu_samples[start_index:], "gpu_sampling_errors": sampling_errors[:]}
        write_new(directory / f"{name}-process.json", raw)
        if result["timed_out"]:
            raise TimeoutError(f"{name}: owned process group {result['pid']} timed out; cleanup_incomplete={result['cleanup_incomplete']}")
        if result["returncode"] != 0:
            raise subprocess.CalledProcessError(result["returncode"], command, result["stdout"], result["stderr"])
        if sampling_errors:
            raise RuntimeError(f"GPU polling failed: {sampling_errors}")
        return json.loads(result["stdout"])

    results = {}
    for model_name, model in MODELS.items():
        if model_name not in args.models:
            continue
        tokenizer = run(f"{model_name}-tokenizer", [binary, "tokenize", "--model", model, "--prompt-file", ROOT / "fixtures/prompt.txt"])
        seed = tokenizer["prompt_token_ids"]
        if not seed:
            raise ValueError("empty benchmark seed")
        for case, (length, generated) in CASES.items():
            if case not in args.cases:
                continue
            ids = (seed * (length // len(seed) + 1))[:length]
            input_path = directory / f"{model_name}-{case}-input.json"
            write_new(input_path, ids)
            record = run(f"{model_name}-{case}", [
                binary, "bench", "--backend", "cuda", "--model", model, "--input-tokens-file", input_path,
                "--context", 1024, "--batch", args.batch, "--prefill-chunk", args.chunk, "--threads", 4,
                "--tokens", generated, "--warmups", args.warmups, "--repetitions", args.repetitions, *precision])
            validate(record, ids, generated, args.repetitions)
            name = f"{model_name}-{case}"
            write_new(directory / f"{name}.json", record)
            results[name] = summarize(record)
            print(name, json.dumps(results[name]), flush=True)
        # Preserve a longer full-logit vector, rather than judging quality from output text.
        if "long" in args.cases:
            run(f"{model_name}-long-logits", [
                binary, "logits", "--backend", "cuda", "--model", model,
                "--input-tokens-file", directory / f"{model_name}-long-input.json",
                "--context", 1024, "--batch", args.batch, "--prefill-chunk", args.chunk, *precision])

    external = {}
    if args.llama:
        llama = ROOT / "target/benchmark-llama-cuda13/bin/llama-server"
        provenance["llama_binary_sha256"] = digest(llama)
        provenance["llama_cmake_cache"] = (llama.parent.parent / "CMakeCache.txt").read_text()
        write_new(directory / "llama-build.json", provenance)
        endpoint = "http://127.0.0.1:30172"
        for model_name, model in MODELS.items():
            if model_name not in args.models:
                continue
            for precision in ["f32", "f16"]:
                command = [*pin, llama, "--host", "127.0.0.1", "--port", "30172", "--model", model,
                           "--ctx-size", "1024", "--batch-size", str(args.batch), "--ubatch-size", str(args.chunk),
                           "--threads", "4", "--threads-batch", "4", "--parallel", "1",
                           "--gpu-layers", "all", "--flash-attn", "on", "--cache-type-k", precision,
                           "--cache-type-v", precision, "--no-context-shift", "--fit", "off", "--poll", "0"]
                label = f"llama-{model_name}-{precision}"
                write_new(directory / f"{label}-command.json", [str(x) for x in command])
                with server(command, endpoint, directory / f"{label}.log", environment):
                    for case, (length, generated) in CASES.items():
                        if case not in args.cases:
                            continue
                        ids = json.loads((directory / f"{model_name}-{case}-input.json").read_text())
                        samples = []
                        for i in range(args.repetitions + args.warmups):
                            start = time.monotonic_ns()
                            response = http_json(endpoint, "/completion", {
                                "prompt": ids, "n_predict": generated, "temperature": 0.0, "top_k": 1,
                                "top_p": 1.0, "min_p": 0.0, "repeat_penalty": 1.0, "seed": 0,
                                "ignore_eos": True, "cache_prompt": False, "return_tokens": True})
                            elapsed = time.monotonic_ns() - start
                            validate_llama_sample(response, ids, generated)
                            sample = {"http_ns": elapsed, "response": response}
                            write_new(directory / f"{label}-{case}-{i}.json", {"warmup": i < args.warmups, **sample})
                            if i >= args.warmups:
                                samples.append(sample)
                        external[f"{model_name}-{precision}-{case}"] = {
                            "http_ms": distribution([s["http_ns"] / 1e6 for s in samples]),
                            "internal_prefill_ms": distribution([s["response"]["timings"]["prompt_ms"] for s in samples]),
                            "internal_decode_ms": distribution([s["response"]["timings"]["predicted_ms"] for s in samples]),
                            "raw_timings": [s["response"]["timings"] for s in samples],
                            "output_token_match": samples[0]["response"]["tokens"] ==
                            json.loads((directory / f"{model_name}-{case}.json").read_text())["samples"][0]["generated_tokens"],
                        }
    comparison = {}
    if args.compare_to:
        previous = args.compare_to.resolve(strict=True)
        old_report = json.loads((previous / "summary.json").read_text())
        if old_report["controls"]["warmups"] != args.warmups:
            raise ValueError("changed warmup count between baseline and candidate")
        if old_report["controls"].get("matmul", "f32") != args.cuda_matmul:
            raise ValueError("arithmetic-preserving comparison cannot change matmul precision")
        if (old_report["controls"]["batch"] != args.batch
                or old_report["controls"]["chunk"] != args.chunk
                or old_report["controls"].get("decode_graph", False) != args.cuda_decode_graph):
            raise ValueError("single-order comparison cannot change batch, chunk, or decode policy")
        old_provenance = json.loads((previous / "provenance.json").read_text())
        if old_provenance["models"] != provenance["models"] or old_provenance["affinity"] != affinity:
            raise ValueError("changed model identity or CPU affinity")
        for name, current in results.items():
            old = old_report["results"][name]
            old_record = json.loads((previous / f"{name}.json").read_text())
            new_record = json.loads((directory / f"{name}.json").read_text())
            if old_record["prompt_token_ids"] != new_record["prompt_token_ids"]:
                raise ValueError("input IDs changed")
            if old_record["samples"][0]["generated_tokens"] != new_record["samples"][0]["generated_tokens"]:
                raise ValueError(f"arithmetic-preserving optimization changed output: {name}")
            comparison[name] = {
                "prefill_speedup": old["prefill_ms"]["median"] / current["prefill_ms"]["median"],
                "decode_speedup": old["decode_ms"]["median"] / current["decode_ms"]["median"] if current["decode_ms"]["median"] else None,
                "reservation_not_increased": current["memory"]["numerical_reservation_bytes"] <= old["memory"]["numerical_reservation_bytes"],
            }
        for model in MODELS:
            if model not in args.models or "long" not in args.cases:
                continue
            old = json.loads((previous / f"{model}-long-logits-process.json").read_text())
            new = json.loads((directory / f"{model}-long-logits-process.json").read_text())
            if json.loads(old["stdout"])["logits"] != json.loads(new["stdout"])["logits"]:
                raise ValueError(f"long-context logits changed: {model}")
        full_matrix = set(args.models) == set(MODELS) and set(args.cases) == set(CASES)
        comparison["milestone_passed"] = full_matrix and all(
            comparison[f"{model}-{case}"]["prefill_speedup"] >= 1 / .85
            for model in MODELS for case in ["short", "prefill"]
        ) and all(comparison[f"{model}-decode"]["decode_speedup"] >= 1 / 1.05 for model in MODELS
                  ) and all(comparison[name]["reservation_not_increased"] for name in results)
    report = {
        "schema_version": 1, "finished_at": datetime.now(timezone.utc).isoformat(),
        "results": results, "external": external, "comparison": comparison,
        "controls": {"context": 1024, "batch": args.batch, "chunk": args.chunk, "threads": 4,
                     "warmups": args.warmups, "repetitions": args.repetitions, "concurrency": 1,
                     "prefix_reuse": False, "rust_kv": "f32", "cpu_affinity": affinity,
                     "matmul": args.cuda_matmul, "decode_graph": args.cuda_decode_graph},
        "limitations": [
            "Rust engine-only timings differ from HTTP wall time and llama.cpp internal timing boundaries.",
            "Descriptive p95/p99 from seven samples are not robust tail-latency estimates.",
            "Per-request mean inter-token latency is not a measured per-token latency distribution.",
            "Concurrency and per-request energy are not measured; no universal superiority claim.",
            "GPU polling measures whole-device usage and can miss transient peaks.",
            "Model/engine order is fixed across processes; no exclusive hardware or clock control.",
        ],
        "gpu_after": gpu_state(), "load_after": os.getloadavg(),
    }
    write_new(directory / "summary.json", report)
    print(json.dumps({"comparison": comparison, "external": external}, indent=2))


if __name__ == "__main__":
    main()
