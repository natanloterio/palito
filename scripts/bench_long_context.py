#!/usr/bin/env python3
"""Long-context TTFT and decode benchmark: Palito `chat` versus llama-server.

Both engines get the same single user message (pinned repository text, newlines
flattened, truncated to a target token count) through the model's chat template,
greedy, a fixed reply budget and no prompt cache. Each engine loads the model once
per configuration, runs one discarded warm-up request per prompt size and then
`--runs` measured requests; medians are reported.

Every phase that loads a model holds `/tmp/palito-gpu.lock` (flock(2), the same
lock `flock /tmp/palito-gpu.lock <cmd>` takes) and the engine process is stopped
before the lock is released, so the two engines are never loaded at once.
"""

import argparse
import contextlib
import datetime
import fcntl
import json
import os
import re
import signal
import socket
import statistics
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def asset_root():
    """Main checkout (models and the llama.cpp build are untracked, so not in worktrees)."""
    common = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--path-format=absolute", "--git-common-dir"],
                            check=True, capture_output=True, text=True).stdout.strip()
    return Path(common).parent


ASSETS = asset_root()
LOCK_PATH = "/tmp/palito-gpu.lock"
CORPUS_REV = "2ded846"
CORPUS_PATHS = (
    "docs/technical-reference.md",
    "docs/architecture.md",
    "benchmarks/cuda-runtime-m8-plan.md",
    "benchmarks/cuda-optimization.md",
)
INSTRUCTION = "Read the following technical document and then explain in detail what it describes. Document: "
DEFAULT_SIZES = (500, 2900, 10000, 40000)
STATS_LINE = re.compile(r"\[(\d+) tokens · first token ([\d.]+) ms · ([\d.]+) tok/s\]")
LLAMA_BASE_FLAGS = ("-ngl", "99", "-np", "1", "--jinja", "--reasoning-format", "none")
LLAMA_CONFIGS = {
    "fa-off": ("-fa", "off"),
    "fa-on": ("-fa", "on"),
    "fa-on-q8kv": ("-fa", "on", "-ctk", "q8_0", "-ctv", "q8_0"),
}


def log(message):
    print(message, file=sys.stderr, flush=True)


@contextlib.contextmanager
def gpu_lock():
    with open(LOCK_PATH, "a") as handle:
        log(f"waiting for {LOCK_PATH}")
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def corpus_text():
    parts = [
        subprocess.run(["git", "-C", str(ROOT), "show", f"{CORPUS_REV}:{path}"],
                       check=True, capture_output=True, text=True).stdout
        for path in CORPUS_PATHS
    ]
    return flatten("\n".join(parts))


def flatten(text):
    """`palito chat` reads one message per line, so the prompt has no newlines."""
    return " ".join(text.split())


def tokenize(palito, model, text):
    with tempfile.NamedTemporaryFile("w", suffix=".txt") as handle:
        handle.write(text)
        handle.flush()
        out = subprocess.run([palito, "tokenize", "--model", model, "--prompt-file", handle.name],
                             check=True, capture_output=True, text=True).stdout
    return json.loads(out)["prompt_token_ids"]


def detokenize(palito, model, ids):
    with tempfile.NamedTemporaryFile("w", suffix=".json") as handle:
        json.dump(ids, handle)
        handle.flush()
        out = subprocess.run([palito, "detokenize", "--model", model, "--input-tokens-file", handle.name],
                             check=True, capture_output=True, text=True).stdout
    return json.loads(out)["text"]


def build_prompts(palito, model, sizes):
    """Return {target: prompt}; `target` counts message tokens before the chat template."""
    instruction = len(tokenize(palito, model, INSTRUCTION))
    body = tokenize(palito, model, corpus_text())
    prompts = {}
    for target in sizes:
        need = target - instruction
        if need <= 0 or need > len(body):
            raise ValueError(f"target {target} outside 1..{len(body) + instruction} tokens")
        prompts[target] = flatten(INSTRUCTION + detokenize(palito, model, body[:need]))
    return prompts


def parse_palito_stats(stderr):
    samples = [
        {"generated": int(n), "ttft_ms": float(ttft), "decode_tps": float(rate)}
        for n, ttft, rate in STATS_LINE.findall(stderr)
    ]
    if not samples:
        raise ValueError(f"no stats lines in palito output:\n{stderr[-2000:]}")
    return samples


def palito_input(prompt, requests):
    return "".join(f"{prompt}\n/reset\n" for _ in range(requests))


def run_palito(args, prompts):
    results = {}
    for target, prompt in prompts.items():
        command = [args.palito, "chat", "--model", args.model, "--backend", "cuda",
                   "--tokens", str(args.tokens), "--stats", "--no-stream", "--no-prefix-cache"]
        with gpu_lock():
            log(f"palito: {target} tokens")
            done = subprocess.run(command, input=palito_input(prompt, args.runs + 1),
                                  capture_output=True, text=True)
        if done.returncode != 0:
            raise RuntimeError(f"palito failed ({done.returncode}):\n{done.stderr[-2000:]}")
        samples = parse_palito_stats(done.stderr)
        if len(samples) != args.runs + 1:
            raise RuntimeError(f"palito: expected {args.runs + 1} replies, got {len(samples)}")
        results[target] = samples[1:]
    return results


def llama_env(args):
    env = dict(os.environ)
    binary_dir = str(Path(args.llama_server).resolve().parent)
    env["LD_LIBRARY_PATH"] = ":".join(p for p in (binary_dir, args.cuda_lib, env.get("LD_LIBRARY_PATH")) if p)
    return env


def post_json(url, body, timeout):
    request = urllib.request.Request(url, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def wait_healthy(base, process, timeout=600):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"llama-server exited with {process.returncode}")
        try:
            with urllib.request.urlopen(f"{base}/health", timeout=5) as response:
                if response.status == 200:
                    return
        except (urllib.error.URLError, ConnectionError, TimeoutError):
            pass
        time.sleep(1)
    raise TimeoutError("llama-server did not become healthy")


def llama_request(base, prompt, tokens):
    body = {
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": tokens, "temperature": 0, "top_k": 1,
        "cache_prompt": False, "seed": 0,
    }
    reply = post_json(f"{base}/v1/chat/completions", body, timeout=900)
    timings = reply["timings"]
    return {
        "generated": timings["predicted_n"], "prompt_tokens": timings["prompt_n"],
        "ttft_ms": timings["prompt_ms"], "decode_tps": timings["predicted_per_second"],
    }


@contextlib.contextmanager
def llama_server(args, flags, log_path):
    assert_port_free(args.port)
    command = [args.llama_server, "-m", args.model, "-c", str(args.context), "--port", str(args.port),
               *LLAMA_BASE_FLAGS, *flags]
    with open(log_path, "w") as server_log:
        process = subprocess.Popen(command, env=llama_env(args), stdout=server_log,
                                   stderr=subprocess.STDOUT, start_new_session=True)
    try:
        wait_healthy(f"http://127.0.0.1:{args.port}", process)
        yield f"http://127.0.0.1:{args.port}"
    finally:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=60)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()


def parse_sse_metrics(lines, started, timestamps=None):
    """Parse OpenAI SSE lines and return (first_token_s, end_s, prompt_tokens, generated)."""
    first = None
    end = None
    prompt_tokens = None
    generated = None
    chunks = 0
    for idx, raw in enumerate(lines):
        line = raw.strip()
        if not line.startswith("data: "):
            continue
        payload = line[6:]
        now = timestamps[idx] if timestamps is not None else time.perf_counter()
        if payload == "[DONE]":
            end = now
            break
        event = json.loads(payload)
        usage = event.get("usage")
        if isinstance(usage, dict):
            if usage.get("prompt_tokens") is not None:
                prompt_tokens = int(usage["prompt_tokens"])
            if usage.get("completion_tokens") is not None:
                generated = int(usage["completion_tokens"])
        choices = event.get("choices") or []
        if choices:
            delta = choices[0].get("delta") or {}
            text = delta.get("content") or delta.get("reasoning_content")
            if text:
                if first is None:
                    first = now
                chunks += 1
    if first is None or end is None:
        raise RuntimeError("incomplete SSE stream (missing first token or [DONE])")
    if generated is None:
        generated = chunks
    ttft_ms = (first - started) * 1000.0
    decode_s = end - first
    decode_tps = (max(generated - 1, 1) / decode_s) if decode_s > 0 else 0.0
    sample = {"generated": generated, "ttft_ms": round(ttft_ms, 2), "decode_tps": round(decode_tps, 2)}
    if prompt_tokens is not None:
        sample["prompt_tokens"] = prompt_tokens
    return sample


def stream_chat_request(base, prompt, tokens, extra=None):
    body = {
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": tokens,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if extra:
        body.update(extra)
    request = urllib.request.Request(
        f"{base}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    started = time.perf_counter()
    lines = []
    timestamps = []
    with urllib.request.urlopen(request, timeout=900) as response:
        for raw in response:
            lines.append(raw.decode("utf-8", errors="replace"))
            timestamps.append(time.perf_counter())
    return parse_sse_metrics(lines, started, timestamps)


def assert_port_free(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        if sock.connect_ex(("127.0.0.1", port)) == 0:
            raise RuntimeError(f"port {port} is already in use; stop the existing server before benchmarking")


@contextlib.contextmanager
def palito_server(args, log_path):
    assert_port_free(args.port)
    command = [
        args.palito, "serve",
        "--model", args.model,
        "--backend", "cuda",
        "--context", str(args.context),
        "--port", str(args.port),
        "--no-prefix-cache",
    ]
    with open(log_path, "w") as server_log:
        process = subprocess.Popen(command, stdout=server_log,
                                   stderr=subprocess.STDOUT, start_new_session=True)
    try:
        wait_healthy(f"http://127.0.0.1:{args.port}", process)
        yield f"http://127.0.0.1:{args.port}"
    finally:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=60)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()


def run_palito_serve(args, prompts):
    results = {}
    with gpu_lock():
        log_path = Path(args.out_dir) / "palito-serve.log"
        with palito_server(args, log_path) as base:
            for target, prompt in prompts.items():
                log(f"palito serve: {target} tokens")
                samples = [stream_chat_request(base, prompt, args.tokens) for _ in range(args.runs + 1)]
                results[target] = samples[1:]
    return results


def run_llama(args, prompts, name, sizes):
    results = {}
    with gpu_lock():
        log_path = Path(args.out_dir) / f"llama-server-{name}.log"
        with llama_server(args, LLAMA_CONFIGS[name], log_path) as base:
            for target in sizes:
                log(f"llama.cpp {name}: {target} tokens")
                if args.palito_mode == "serve":
                    extra = {"top_k": 1, "cache_prompt": False, "seed": 0}
                    samples = [stream_chat_request(base, prompts[target], args.tokens, extra)
                               for _ in range(args.runs + 1)]
                else:
                    samples = [llama_request(base, prompts[target], args.tokens)
                               for _ in range(args.runs + 1)]
                results[target] = samples[1:]
    return results


def probe_cmd(cmd, env=None):
    try:
        out = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=15)
        text = (out.stdout or out.stderr).strip()
        return text.splitlines()[0] if text else ""
    except (OSError, subprocess.SubprocessError):
        return ""


def collect_metadata(args):
    model_path = Path(args.model)
    return {
        "date_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "gpu": probe_cmd(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"]),
        "palito_version": probe_cmd([args.palito, "--version"]),
        "palito_git_sha": probe_cmd(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"]),
        "llama_cpp_version": probe_cmd([args.llama_server, "--version"], env=llama_env(args)),
        "model": model_path.name,
        "model_bytes": model_path.stat().st_size if model_path.exists() else None,
        "corpus_rev": CORPUS_REV,
        "palito_mode": args.palito_mode,
        "context": args.context,
        "tokens": args.tokens,
        "runs": args.runs,
        "sizes": args.sizes,
        "llama_configs": args.llama_configs,
    }


def medians(samples):
    return {
        "ttft_ms": statistics.median(s["ttft_ms"] for s in samples),
        "decode_tps": statistics.median(s["decode_tps"] for s in samples),
        "generated": statistics.median(s["generated"] for s in samples),
    }


def table(results, prompt_tokens):
    """Markdown table; `results` is {engine: {target: [samples]}}."""
    engines = list(results)
    head = "| Prompt tokens | " + " | ".join(f"{e} TTFT ms | {e} decode tok/s" for e in engines) + " |"
    rows = [head, "|" + "---|" * (1 + 2 * len(engines))]
    for target in sorted(prompt_tokens):
        cells = []
        for engine in engines:
            if target in results[engine]:
                m = medians(results[engine][target])
                cells += [f"{m['ttft_ms']:.0f}", f"{m['decode_tps']:.1f}"]
            else:
                cells += ["–", "–"]
        rows.append(f"| {prompt_tokens[target]:,} | " + " | ".join(cells) + " |")
    return "\n".join(rows)


def parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--palito", default=os.path.expanduser("~/.local/bin/palito"))
    parser.add_argument("--model", default=str(ASSETS / "models/benchmark-qwen3/gguf/Qwen3-30B-A3B-Q4_K_M.gguf"))
    parser.add_argument("--llama-server", default=str(ASSETS / "target/benchmark-llama-cuda13/bin/llama-server"))
    parser.add_argument("--cuda-lib", default=str(
        ASSETS / ".venv-cuda-bench/lib/python3.11/site-packages/nvidia/cu13/lib"))
    parser.add_argument("--sizes", type=int, nargs="+", default=list(DEFAULT_SIZES))
    parser.add_argument("--llama-configs", nargs="+", default=["fa-off", "fa-on", "fa-on-q8kv"],
                        choices=sorted(LLAMA_CONFIGS))
    parser.add_argument("--tokens", type=int, default=256, help="reply budget per request")
    parser.add_argument("--runs", type=int, default=3, help="measured runs after one warm-up")
    parser.add_argument("--context", type=int, default=40960, help="llama-server -c")
    parser.add_argument("--port", type=int, default=18093)
    parser.add_argument("--out-dir", default=str(ROOT / "target/bench-long-context"))
    parser.add_argument("--palito-mode", choices=["serve", "chat"], default="serve",
                        help="serve measures both engines via identical HTTP SSE streaming; chat uses palito chat CLI")
    parser.add_argument("--skip-palito", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if max(args.sizes) + args.tokens > args.context:
        raise SystemExit(f"--context {args.context} too small for {max(args.sizes)} + {args.tokens} tokens")
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    prompts = build_prompts(args.palito, args.model, args.sizes)
    results = {}
    if not args.skip_palito:
        results["Palito int8-fast"] = (
            run_palito_serve(args, prompts) if args.palito_mode == "serve" else run_palito(args, prompts)
        )
    for name in args.llama_configs:
        results[f"llama.cpp {name}"] = run_llama(args, prompts, name, args.sizes)
    first_with_counts = next(
        (r for r in results.values() if r and "prompt_tokens" in r[args.sizes[0]][0]),
        None,
    )
    prompt_tokens = {t: (first_with_counts[t][0]["prompt_tokens"] if first_with_counts else t) for t in args.sizes}
    raw = Path(args.out_dir) / "results.json"
    payload = {
        "metadata": collect_metadata(args),
        "prompt_tokens": prompt_tokens,
        "results": results,
    }
    raw.write_text(json.dumps(payload, indent=2) + "\n")
    print(table(results, prompt_tokens))
    log(f"raw samples: {raw}")


if __name__ == "__main__":
    main()
