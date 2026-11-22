"""Sequential KT/llama service comparison, optionally including independent Rust CUDA."""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import ipaddress
import json
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from benchmark import digest
from benchmark_rust import summarize as summarize_phases, validate as validate_rust

ROOT = Path(__file__).resolve().parents[1]
THREADS = 4
CONTEXT = 512
GENERATED = 32
WARMUPS = 1
REPETITIONS = 5


def http_json(endpoint, path, payload=None, timeout=300):
    url = urllib.parse.urlsplit(endpoint)
    if url.scheme != "http" or not ipaddress.ip_address(url.hostname).is_loopback:
        raise ValueError("benchmark endpoints must be literal loopback HTTP origins")
    data = None if payload is None else json.dumps(payload).encode()
    request = urllib.request.Request(
        endpoint + path, data=data, headers={"Content-Type": "application/json"}
    )
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(request, timeout=timeout) as response:
        body = response.read(8 * 1024 * 1024 + 1)
        if len(body) > 8 * 1024 * 1024:
            raise ValueError("response exceeds 8 MiB")
        return json.loads(body)


def validate_llama_sample(response, prompt, generated):
    tokens = response["tokens"]
    # tokens_cached is the retained slot size, not the number of prefix hits.
    timings = response["timings"]
    if (len(tokens) != generated or any(type(token) is not int or token < 0 for token in tokens)
            or response["stop_type"] != "limit" or response["truncated"]
            or response["tokens_evaluated"] != len(prompt)
            or timings["prompt_n"] != len(prompt) or timings["cache_n"] != 0):
        raise ValueError("llama.cpp returned an incomplete, truncated, or cached workload")


def summarize(samples, count):
    durations = [sample["elapsed_ns"] for sample in samples]
    if not durations or any(type(value) is not int or value <= 0 for value in durations):
        raise ValueError("invalid or empty timing samples")
    return {
        "samples": len(durations),
        "median_ms": statistics.median(durations) / 1e6,
        "min_ms": min(durations) / 1e6,
        "max_ms": max(durations) / 1e6,
        "generated_tokens_per_second_including_prompt": statistics.median(
            count * 1e9 / value for value in durations),
    }


def write_new(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as output:
        json.dump(value, output, indent=2, allow_nan=False)
        output.write("\n")


def gpu_state():
    return subprocess.check_output([
        "nvidia-smi", "--query-gpu=name,driver_version,memory.used,utilization.gpu",
        "--format=csv,noheader",
    ], text=True).strip()


@contextmanager
def server(command, endpoint, log_path, environment):
    with log_path.open("x") as log:
        process = subprocess.Popen(
            [str(value) for value in command], stdout=log, stderr=subprocess.STDOUT,
            env=environment, start_new_session=True,
        )
        try:
            deadline = time.monotonic() + 900
            while True:
                if process.poll() is not None:
                    tail = "\n".join(log_path.read_text(errors="replace").splitlines()[-65:])
                    raise RuntimeError(f"server exited ({process.returncode}):\n{tail}\nLog: {log_path}")
                try:
                    http_json(endpoint, "/health", timeout=2)
                    break
                except json.JSONDecodeError:
                    # Health responses may have an empty or plain-text 200 body.
                    break
                except (urllib.error.URLError, TimeoutError, ConnectionError):
                    if time.monotonic() >= deadline:
                        raise TimeoutError(f"server readiness timed out; inspect {log_path}")
                    time.sleep(2)
            yield process
        finally:
            # Only terminate this benchmark's explicitly created process group.
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass  # The entire owned process group already exited.
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait(timeout=20)


def prepare_inputs(python, model, directory):
    source = ROOT / "fixtures/prompt.txt"
    code = (
        "import json,sys; from transformers import AutoTokenizer; "
        "t=AutoTokenizer.from_pretrained(sys.argv[1],local_files_only=True,trust_remote_code=False); "
        "print(json.dumps(t.encode(open(sys.argv[2]).read(),add_special_tokens=False)))"
    )
    short = json.loads(subprocess.check_output(
        [python, "-c", code, str(model), str(source)], text=True, timeout=120
    ))
    if not short or any(type(token) is not int or token < 0 for token in short):
        raise ValueError("tokenizer produced invalid benchmark input")
    long = (short * (256 // len(short) + 1))[:256]
    cases = {"short": short, "long": long}
    for name, tokens in cases.items():
        write_new(directory / f"{name}-input.json", tokens)
    return cases


def llama_cases(endpoint, cases, directory, label):
    results = {}
    for name, prompt in cases.items():
        samples = []
        reference = None
        for index in range(WARMUPS + REPETITIONS):
            start = time.perf_counter_ns()
            response = http_json(endpoint, "/completion", {
                "prompt": prompt, "n_predict": GENERATED, "temperature": 0.0,
                "top_k": 1, "top_p": 1.0, "min_p": 0.0, "repeat_penalty": 1.0,
                "seed": 0, "ignore_eos": True, "cache_prompt": False, "return_tokens": True,
            })
            elapsed = time.perf_counter_ns() - start
            write_new(directory / f"{label}-{name}-{index}.json", {
                "warmup": index < WARMUPS, "elapsed_ns": elapsed, "response": response,
            })
            validate_llama_sample(response, prompt, GENERATED)
            if reference is not None and response["tokens"] != reference:
                raise ValueError("llama.cpp greedy output changed between repetitions")
            reference = response["tokens"]
            if index >= WARMUPS:
                samples.append({"elapsed_ns": elapsed, "response": response})
        results[name] = {"input": prompt, "samples": samples}
        write_new(directory / f"{label}-{name}.json", results[name])
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cuda-root", type=Path, required=True)
    parser.add_argument("--include-rust-cuda", action="store_true",
                        help="Also measure our all-resident CUDA kernels and an all-GPU llama.cpp deployment")
    args = parser.parse_args()
    directory = args.output_dir.resolve()
    directory.mkdir(parents=True, exist_ok=False)
    python = ROOT / ".venv-ktransformers-bench/bin/python"
    model = ROOT / "models/benchmark-qwen3/hf"
    gguf = ROOT / "models/benchmark-qwen3/gguf"
    llama = ROOT / "target/benchmark-llama-cuda13/bin/llama-server"
    adapter = ROOT / "target/release/ktransformers"
    rust = ROOT / "target/release/palito"
    manifest = json.loads((model.parent / "models-lock.json").read_text())
    for source in manifest:
        folder = model if source["repository"].endswith("A3B") else gguf
        for entry in source["files"]:
            if (folder / entry["rfilename"]).stat().st_size != entry["size"]:
                raise ValueError(f"missing or incomplete model file: {entry['rfilename']}")
    cases = prepare_inputs(str(python), model, directory)
    environment = dict(os.environ, OMP_NUM_THREADS=str(THREADS), MKL_NUM_THREADS=str(THREADS),
                       OPENBLAS_NUM_THREADS=str(THREADS), CUDA_VISIBLE_DEVICES="0",
                       HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
    cuda = str(args.cuda_root.resolve(strict=True))
    environment["CUDA_HOME"] = cuda
    environment["CUDACXX"] = cuda + "/bin/nvcc"
    environment["PATH"] = cuda + "/bin:" + environment.get("PATH", "")
    environment["LD_LIBRARY_PATH"] = os.pathsep.join(filter(None, [
        cuda + "/lib", environment.get("LD_LIBRARY_PATH"),
    ]))
    kt_endpoint, llama_endpoint = "http://127.0.0.1:30171", "http://127.0.0.1:30172"
    kt_command = [
        python, "-m", "sglang.launch_server", "--host", "127.0.0.1", "--port", "30171",
        "--model-path", model, "--served-model-name", "qwen3-moe-benchmark",
        "--kt-method", "LLAMAFILE", "--kt-weight-path", gguf, "--kt-cpuinfer", str(THREADS),
        "--kt-threadpool-count", "1", "--kt-num-gpu-experts", "0",
        "--kt-max-deferred-experts-per-token", "0", "--context-length", str(CONTEXT),
        "--disable-radix-cache", "--disable-shared-experts-fusion",
        "--max-running-requests", "1", "--max-total-tokens", "1024",
        "--chunked-prefill-size", "128", "--mem-fraction-static", "0.25",
        "--attention-backend", "triton", "--sampling-backend", "pytorch",
        "--disable-cuda-graph", "--disable-overlap-schedule", "--dtype", "float16",
    ]
    llama_command = [
        llama, "--host", "127.0.0.1", "--port", "30172",
        "--model", gguf / "Qwen3-30B-A3B-Q4_K_M.gguf",
        "--alias", "qwen3-moe-benchmark", "--ctx-size", str(CONTEXT),
        "--threads", str(THREADS), "--threads-batch", str(THREADS), "--parallel", "1",
        "--batch-size", "128", "--ubatch-size", "128", "--gpu-layers", "all", "--cpu-moe",
        "--flash-attn", "on", "--cache-type-k", "f16", "--cache-type-v", "f16", "--poll", "0",
        "--no-context-shift", "--fit", "off",
    ]
    provenance = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "gpu_before": gpu_state(), "load_average_before": os.getloadavg(),
        "cpu_model_names": sorted({
            line.split(":", 1)[1].strip()
            for line in Path("/proc/cpuinfo").read_text().splitlines()
            if line.startswith("model name")
        }),
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "model_manifest": manifest,
        "kt_server_command": [str(value) for value in kt_command],
        "llama_server_command": [str(value) for value in llama_command],
        "requirements": (ROOT / "benchmarks/ktransformers/requirements.txt").read_text(),
        "cuda_compiler": subprocess.check_output([cuda + "/bin/nvcc", "--version"], text=True),
        "cuda_root": cuda,
        "cuda_requirements": (ROOT / "benchmarks/ktransformers/cuda-requirements.txt").read_text(),
        "torch": json.loads(subprocess.check_output([
            python, "-c", "import json,torch; print(json.dumps("
            "{'version':torch.__version__,'cuda':torch.version.cuda}))",
        ], text=True)),
        "environment_freeze": subprocess.check_output(
            ["uv", "pip", "freeze", "--python", str(python)], text=True),
        "llama_revision": subprocess.check_output(
            ["git", "-C", str(ROOT.parent / "llama.cpp"), "rev-parse", "HEAD"], text=True).strip(),
        "prompt_sha256": hashlib.sha256((ROOT / "fixtures/prompt.txt").read_bytes()).hexdigest(),
    }
    if args.include_rust_cuda:
        provenance["rust_binary_sha256"] = digest(rust)
        provenance["gguf_sha256"] = digest(gguf / "Qwen3-30B-A3B-Q4_K_M.gguf")
        provenance["rust_source_sha256"] = {
            str(path.relative_to(ROOT)): digest(path) for path in [
                ROOT / "Cargo.toml", ROOT / "Cargo.lock", Path(__file__),
                *sorted((ROOT / "src").rglob("*.rs")), ROOT / "src/cuda/kernels.cu"]
        }
        provenance["llama_all_gpu_command"] = [str(value) for value in llama_command if value != "--cpu-moe"]
    write_new(directory / "provenance.json", provenance)
    cuda_results = {}
    if args.include_rust_cuda:
        for name, prompt in cases.items():
            command = [
                rust, "bench", "--backend", "cuda", "--model", gguf / "Qwen3-30B-A3B-Q4_K_M.gguf",
                "--input-tokens-file", directory / f"{name}-input.json", "--context", str(CONTEXT),
                "--batch", "128", "--prefill-chunk", "128", "--tokens", str(GENERATED),
                "--warmups", str(WARMUPS), "--repetitions", str(REPETITIONS),
            ]
            result = subprocess.run([str(x) for x in command], env=environment,
                                    capture_output=True, text=True, timeout=900)
            write_new(directory / f"rust-cuda-{name}-process.json", {
                "command": [str(x) for x in command], "returncode": result.returncode,
                "stdout": result.stdout, "stderr": result.stderr})
            result.check_returncode()
            record = json.loads(result.stdout)
            validate_rust(record, prompt, GENERATED, REPETITIONS)
            cuda_results[name] = record
            write_new(directory / f"rust-cuda-{name}.json", record)
    kt_results = {}
    with server(kt_command, kt_endpoint, directory / "ktransformers-server.log", environment):
        for name in cases:
            path = directory / f"ktransformers-{name}.json"
            command = [
                adapter, "--endpoint", kt_endpoint, "--model", "qwen3-moe-benchmark",
                "--kt-method", "LLAMAFILE", "bench", "--input-tokens-file", directory / f"{name}-input.json",
                "--tokens", str(GENERATED), "--warmups", str(WARMUPS),
                "--repetitions", str(REPETITIONS), "--output", path,
            ]
            subprocess.run([str(value) for value in command], check=True, timeout=900)
            kt_results[name] = json.loads(path.read_text())
        if args.include_rust_cuda:
            result = subprocess.run([
                str(rust), "remote", "--endpoint", kt_endpoint, "--model", "qwen3-moe-benchmark",
                "--kt-method", "LLAMAFILE", "--input-tokens-file", str(directory / "short-input.json"),
                "--tokens", str(GENERATED), "--fixed-length",
            ], capture_output=True, text=True, check=True, timeout=300)
            remote = json.loads(result.stdout)
            if remote["completion"]["generated_tokens"] != kt_results["short"]["samples"][0]["generated_tokens"]:
                raise ValueError("unified CLI differs from the KTransformers adapter")
            write_new(directory / "unified-ktransformers-remote.json", remote)
    llama_results = {}
    with server(llama_command, llama_endpoint, directory / "llama-server.log", environment):
        tokenized = http_json(llama_endpoint, "/tokenize", {
            "content": (ROOT / "fixtures/prompt.txt").read_text(), "add_special": False, "parse_special": True,
        })["tokens"]
        if tokenized != cases["short"]:
            raise ValueError("HF and GGUF tokenizers disagree on the benchmark prompt")
        llama_results = llama_cases(llama_endpoint, cases, directory, "llama")
    all_gpu_results = {}
    if args.include_rust_cuda:
        command = [value for value in llama_command if value != "--cpu-moe"]
        with server(command, llama_endpoint, directory / "llama-all-gpu-server.log", environment):
            all_gpu_results = llama_cases(llama_endpoint, cases, directory, "llama-all-gpu")
    summary = {}
    for name, prompt in cases.items():
        kt = kt_results[name]
        if kt["input"] != {"kind": "token_ids", "value": prompt}:
            raise ValueError("adapter did not benchmark the identical explicit input IDs")
        kt_summary = summarize(kt["samples"], GENERATED)
        llama_summary = summarize(llama_results[name]["samples"], GENERATED)
        summary[name] = {
            "prompt_tokens": len(prompt), "generated_tokens": GENERATED,
            "ktransformers": kt_summary, "llama.cpp": llama_summary,
            "kt_over_llama_latency_ratio": kt_summary["median_ms"] / llama_summary["median_ms"],
            "observed_output_token_match": (
                kt["samples"][0]["generated_tokens"] == llama_results[name]["samples"][0]["response"]["tokens"]
            ),
        }
        if args.include_rust_cuda:
            summary[name]["rust_cuda"] = {
                **summarize_phases(cuda_results[name]),
                "synchronized_generation_ms": statistics.median(
                    (s["ttft_ns"] + s["decode_ns"]) / 1e6 for s in cuda_results[name]["samples"]),
                "timing_scope": "direct numerical work and host sampling; excludes load/reset/tokenization/HTTP",
                "all_experts_gpu_resident": True,
            }
            summary[name]["llama.cpp_all_gpu"] = summarize(all_gpu_results[name]["samples"], GENERATED)
            summary[name]["rust_cuda_matches_llama_all_gpu_tokens"] = (
                cuda_results[name]["samples"][0]["generated_tokens"]
                == all_gpu_results[name]["samples"][0]["response"]["tokens"])
    report = {
        "schema_version": 1, "finished_at": datetime.now(timezone.utc).isoformat(),
        "comparison": "Sequential single-request hybrid deployments on the same host/GPU; four CPU expert threads",
        "controls": {"context_tokens": CONTEXT, "batch_and_chunk_tokens": 128,
                     "warmups_per_case": WARMUPS, "measured_repetitions": REPETITIONS,
                     "kv_dtype": "f16", "prefix_reuse": False, "speculation": False},
        "limitations": [
            "Same trained checkpoint and CPU expert GGUF, but KT uses original non-expert weights cast to F16; llama.cpp uses GGUF non-expert quantization.",
            "KT disables CUDA graphs and uses Triton attention; llama.cpp uses its native execution policy and flash attention.",
            "KT is measured through the Rust adapter, llama.cpp through a Python HTTP client; timings include transport and all server work.",
            "Engine-specific ignore-EOS settings may differ in logit masking; no cross-engine exact-distribution claim.",
            "Backend order is not randomized, and the shared machine has no exclusive CPU or GPU reservation.",
            "Generated tokens/second includes prompt processing; it is not isolated decode throughput.",
        ],
        "gpu_after": gpu_state(), "load_average_after": os.getloadavg(), "summary": summary,
    }
    if args.include_rust_cuda:
        report["comparison"] = "Sequential independent Rust CUDA, KT hybrid, llama.cpp hybrid, and llama.cpp all-GPU"
        report["controls"]["kv_dtype"] = "Rust CUDA F32; KT and both llama.cpp deployments F16"
        report["limitations"].extend([
            "Rust CUDA keeps all experts on GPU; compare residency with llama.cpp_all_gpu, not the hybrid CPU-expert services.",
            "Rust CUDA currently evaluates prefill token-by-token; llama.cpp/KT use batched prefill.",
            "Rust direct numerical generation timing excludes HTTP/reset/tokenization; HTTP service latency is not an identical timing scope.",
        ])
    write_new(directory / "comparison.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
