"""Compare the independent Rust engine with an external llama.cpp control process."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import statistics

from benchmark import ROOT, digest, positive, run


def validate(record, prompt, count, repetitions, allow_cached=False):
    if record["prompt_token_ids"] != prompt or record["prompt_tokens"] != len(prompt):
        raise ValueError("runner changed input token IDs")
    samples = record["samples"]
    if len(samples) != repetitions:
        raise ValueError("runner returned an incomplete sample set")
    reference = None
    for sample in samples:
        cached = sample.get("cached_prompt_tokens", 0)
        if type(cached) is not int or not 0 <= cached <= len(prompt) or (cached and not allow_cached):
            raise ValueError("unexpected or invalid prefix reuse")
        tokens = sample["generated_tokens"]
        if (len(tokens) != count or any(type(t) is not int or t < 0 for t in tokens)
                or sample["decode_steps"] != count - 1
                or (reference is not None and tokens != reference)):
            raise ValueError("invalid or non-repeatable generated output")
        for field in ("prefill_ns", "ttft_ns", "decode_ns"):
            if type(sample[field]) is not int or sample[field] < 0:
                raise ValueError(f"invalid {field}")
        if (sample["prefill_ns"] <= 0 or sample["ttft_ns"] < sample["prefill_ns"]
                or (count == 1 and sample["decode_ns"] != 0)
                or (count > 1 and sample["decode_ns"] == 0)):
            raise ValueError("inconsistent phase timing")
        reference = tokens


def summarize(record):
    samples = record["samples"]
    return {
        "samples": len(samples),
        **{f"{phase}_median_ms": statistics.median(s[f"{phase}_ns"] for s in samples) / 1e6
           for phase in ("prefill", "ttft", "decode")},
        "decode_tokens_per_second": (
            statistics.median(s["decode_steps"] * 1e9 / s["decode_ns"] for s in samples)
            if samples[0]["decode_steps"] else None
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True,
                        help="Separately built native/bin/li-reference; never linked into Rust")
    parser.add_argument("--model", type=Path, default=ROOT / "models/SmolLM2-135M-Instruct-Q4_K_M.gguf")
    parser.add_argument("--rust", type=Path, default=ROOT / "target/release/local-inference")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repetitions", type=positive, default=3)
    args = parser.parse_args()
    directory = args.output_dir.resolve()
    directory.mkdir(parents=True, exist_ok=False)
    reference, rust, model = (p.resolve(strict=True) for p in (args.reference, args.rust, args.model))

    def save(name, value):
        with (directory / name).open("x") as output:
            json.dump(value, output, indent=2, allow_nan=False)
            output.write("\n")

    seed = run([reference, model, ROOT / "fixtures/prompt.txt",
                512, 128, 64, 4, 1, 1, 1, "v1"])
    save("tokenization-reference.json", seed)
    prompt = json.loads(seed["stdout"])["prompt_token_ids"]
    if not prompt:
        raise ValueError("empty reference prompt")
    repeated = (prompt * (128 // len(prompt) + 1))[:125]
    cases = [
        ("single-token", prompt[:8], 1, 4, 3, 128),
        ("short", prompt, 8, 4, 64, 512),
        ("chunk-one", prompt, 8, 4, 1, 512),
        ("chunk-seven", prompt, 8, 4, 7, 512),
        ("one-thread", prompt, 8, 1, 64, 512),
        ("prefill-heavy", repeated[:96], 4, 4, 64, 128),
        ("decode-heavy", prompt[:8], 32, 4, 64, 128),
        ("exact-context", repeated, 4, 4, 64, 128),
    ]
    provenance = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "model": str(model), "model_sha256": digest(model),
        "rust_binary": str(rust), "rust_sha256": digest(rust),
        "reference_binary": str(reference), "reference_sha256": digest(reference),
        "cargo_lock_sha256": digest(ROOT / "Cargo.lock"),
        "cpu_model_names": sorted({line.split(":", 1)[1].strip()
                                  for line in Path("/proc/cpuinfo").read_text().splitlines()
                                  if line.startswith("model name")}),
        "affinity": sorted(os.sched_getaffinity(0)), "load_before": os.getloadavg(),
        "native_libraries": {p.name: digest(p) for p in sorted(reference.parent.glob("*.so*")) if p.is_file()},
        "rust_linkage": run(["ldd", rust]),
        "reference_linkage": run(["ldd", reference]),
        "rust_version": run(["rustc", "--version"])["stdout"].strip(),
        "reference_revision": run(["git", "-C", ROOT.parent / "llama.cpp", "rev-parse", "HEAD"])["stdout"].strip(),
        "reference_dirty_status": run(["git", "-C", ROOT.parent / "llama.cpp", "status", "--short"])["stdout"],
        "rust_source_sha256": {str(p.relative_to(ROOT)): digest(p) for p in sorted((ROOT / "src").rglob("*.rs"))},
        "reference_source_sha256": {p.name: digest(p) for p in sorted((ROOT / "native").glob("*")) if p.is_file()},
        "benchmark_script_sha256": digest(Path(__file__)),
    }
    for line in provenance["reference_linkage"]["stdout"].splitlines():
        if "=>" in line:
            path = Path(line.split("=>", 1)[1].strip().split()[0])
            if path.is_file() and any(name in path.name for name in ("llama", "ggml", "li_bridge")):
                provenance["native_libraries"][str(path.resolve())] = digest(path)
    if any(name in provenance["rust_linkage"]["stdout"] for name in ("libllama", "libggml", "libli_bridge")):
        raise ValueError("Rust executable links a forbidden native inference library")
    save("provenance.json", provenance)
    summaries = {}
    outputs = {}
    for index, (name, ids, count, threads, chunk, context) in enumerate(cases):
        json_path = directory / f"{name}-input.json"
        text_path = directory / f"{name}-input.txt"
        save(json_path.name, ids)
        text_path.write_text(" ".join(map(str, ids)) + "\n")
        commands = {
            "rust": [rust, "bench", "--model", model, "--input-tokens-file", json_path,
                     "--context", context, "--batch", 128, "--prefill-chunk", chunk,
                     "--threads", threads, "--tokens", count, "--warmups", 1,
                     "--repetitions", args.repetitions],
            "llama.cpp": [reference, model, text_path, context, 128, chunk, threads,
                          count, 1, args.repetitions, "v2-token-ids"],
        }
        records = {}
        order = ["rust", "llama.cpp"] if index % 2 == 0 else ["llama.cpp", "rust"]
        for engine in order:
            raw = run(commands[engine], timeout=900)
            save(f"{name}-{engine}.json", raw)
            records[engine] = json.loads(raw["stdout"])
            validate(records[engine], ids, count, args.repetitions)
        rust_summary, cpp_summary = (summarize(records[e]) for e in ("rust", "llama.cpp"))
        summaries[name] = {
            "prompt_tokens": len(ids), "generated_tokens": count, "threads": threads,
            "chunk": chunk, "context": context, "engine_order": order,
            "rust": rust_summary, "llama.cpp": cpp_summary,
            "rust_over_llama_latency_ratio": {
                phase: rust_summary[f"{phase}_median_ms"] / cpp_summary[f"{phase}_median_ms"]
                for phase in ("prefill", "decode") if cpp_summary[f"{phase}_median_ms"] > 0
            },
            "output_token_match": records["rust"]["samples"][0]["generated_tokens"] ==
                                  records["llama.cpp"]["samples"][0]["generated_tokens"],
        }
        outputs[name] = {engine: record["samples"][0]["generated_tokens"] for engine, record in records.items()}
        print(name, json.dumps(summaries[name]), flush=True)
    for engine in ("rust", "llama.cpp"):
        for case in ("chunk-one", "chunk-seven", "one-thread"):
            if outputs[case][engine] != outputs["short"][engine]:
                raise ValueError(f"{engine} output changed across chunks or threads: {case}")
    report = {
        "schema_version": 1, "finished_at": datetime.now(timezone.utc).isoformat(),
        "implementation": "Independent Rust GGUF loader, F32 CPU kernels, attention and KV; no native inference linkage",
        "limitations": [
            "Rust expands weights to F32 at load and uses F32 KV; llama.cpp uses quantized kernels and F16 KV.",
            "Cross-engine token equality is observed, not required: arithmetic and activation precision differ.",
            "Both process phase timers exclude model loading, tokenization and state reset.",
            "Rust prefill is sequential token stepping, not an optimized matrix-matrix kernel.",
            "Warm caches, one warmup/process, no exclusive hardware reservation; per-case engine order alternates.",
        ],
        "summary": summaries, "load_after": os.getloadavg(),
    }
    save("comparison.json", report)


if __name__ == "__main__":
    main()
