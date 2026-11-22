"""Numerical checks against an explicitly dequantized, external llama.cpp oracle."""
import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path

from benchmark import ROOT, digest, run


def compare_logits(actual, expected):
    if (not actual or len(actual) != len(expected)
            or any(not math.isfinite(x) for x in actual + expected)):
        raise ValueError("invalid logit vectors")
    dot = sum(x * y for x, y in zip(actual, expected))
    denominator = math.sqrt(sum(x * x for x in actual) * sum(y * y for y in expected))
    if not denominator:
        raise ValueError("zero-norm logits")
    result = {
        "vocab_tokens": len(actual),
        "cosine_similarity": dot / denominator,
        "rmse": math.sqrt(sum((x - y) ** 2 for x, y in zip(actual, expected)) / len(actual)),
        "max_absolute_error": max(abs(x - y) for x, y in zip(actual, expected)),
        "rust_top_token": max(range(len(actual)), key=actual.__getitem__),
        "reference_top_token": max(range(len(expected)), key=expected.__getitem__),
    }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="Original quantized GGUF")
    parser.add_argument("--f32-reference-model", type=Path, required=True,
                        help="Explicit F32 dequantization of exactly the same GGUF")
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--benchmark-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    directory = args.output_dir.resolve()
    directory.mkdir(parents=True, exist_ok=False)
    model, f32_model, reference = (p.resolve(strict=True) for p in
                                  (args.model, args.f32_reference_model, args.reference))
    benchmark = args.benchmark_dir.resolve(strict=True)

    def save(name, value):
        with (directory / name).open("x") as output:
            json.dump(value, output, indent=2, allow_nan=False)
            output.write("\n")

    cases = {name: json.loads((benchmark / f"{name}-input.json").read_text())
             for name in ("short", "decode-heavy", "exact-context")}
    rust_decode = json.loads(json.loads((benchmark / "decode-heavy-rust.json").read_text())["stdout"])
    cpp_decode = json.loads(json.loads((benchmark / "decode-heavy-llama.cpp.json").read_text())["stdout"])
    a = rust_decode["samples"][0]["generated_tokens"]
    b = cpp_decode["samples"][0]["generated_tokens"]
    different = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), None)
    if different is not None:
        cases["first-divergence"] = cases["decode-heavy"] + a[:different]
    limits = {"minimum_cosine": 0.9999, "maximum_rmse": 0.01, "maximum_absolute_error": 0.05}
    save("provenance.json", {
        "model_sha256": digest(model), "reference_f32_sha256": digest(f32_model),
        "reference_sha256": digest(reference),
        "rust_sha256": digest(ROOT / "target/release/palito"),
        "thresholds": limits,
        "scope": "Rust loads original quantized GGUF; external oracle loads explicit F32 dequantization; reference KV remains F16",
    })
    checks = {}
    for name, tokens in cases.items():
        save(f"{name}-input.json", tokens)
        text_path = directory / f"{name}-input.txt"
        text_path.write_text(" ".join(map(str, tokens)) + "\n")
        rust = run([ROOT / "target/release/palito", "logits", "--model", model,
                    "--input-tokens-file", directory / f"{name}-input.json",
                    "--context", 512, "--batch", 128, "--prefill-chunk", 64])
        cpp = run([reference, f32_model, text_path, 512, 128, 64, 4, 1, 1, 1, "v3-logits"])
        save(f"{name}-rust.json", rust)
        save(f"{name}-reference.json", cpp)
        metrics = compare_logits(json.loads(rust["stdout"])["logits"], json.loads(cpp["stdout"])["logits"])
        metrics["passed"] = (
            metrics["cosine_similarity"] >= limits["minimum_cosine"]
            and metrics["rmse"] <= limits["maximum_rmse"]
            and metrics["max_absolute_error"] <= limits["maximum_absolute_error"]
            and metrics["rust_top_token"] == metrics["reference_top_token"]
        )
        checks[name] = metrics
        print(name, json.dumps(metrics), flush=True)
    passed = all(check["passed"] for check in checks.values())
    save("validation.json", {"finished_at": datetime.now(timezone.utc).isoformat(),
                            "thresholds": limits, "passed": passed, "checks": checks})
    if not passed:
        raise ValueError("numerical oracle validation failed; see saved metrics")


if __name__ == "__main__":
    main()
