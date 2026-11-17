"""Compare independent Rust CPU/CUDA with a bounded-memory, matched-precision external graph."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

from benchmark import ROOT, digest
from benchmark_moe import write_new
from validate_cuda import compare_logits as check_cuda_logits
from validate_rust_logits import compare_logits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--cuda-root", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    rust, reference = (p.resolve(strict=True) for p in (args.binary, args.reference))
    model = ROOT / "models/benchmark-qwen3/gguf/Qwen3-30B-A3B-Q4_K_M.gguf"
    files = [rust, reference, model, Path(__file__), ROOT / "scripts/validate_cuda.py",
             ROOT / "scripts/validate_rust_logits.py",
             *sorted({p.resolve() for p in reference.parent.glob("*.so*")})]
    hashes = {str(p): digest(p) for p in files}
    environment = dict(os.environ)
    environment["LD_LIBRARY_PATH"] = os.pathsep.join(filter(None, [
        str(reference.parent), str(args.cuda_root.resolve(strict=True) / "lib"),
        environment.get("LD_LIBRARY_PATH")]))
    cases = {"one": [1], "seven": list(range(1, 8))}
    cases["divergence"] = json.loads(
        (ROOT / "results/cuda-runtime-m5/divergence-input.json").read_text())
    for name, source in [("prompt46", "moe-short"), ("prompt256", "moe-prefill")]:
        cases[name] = json.loads((ROOT / "results/cuda-runtime-m6/accepted-f32"
                                 / f"{source}-input.json").read_text())
    write_new(output / "provenance.json", {"sha256_before": hashes, "cases": cases,
                                         "arithmetic": "local-f32-activations-v1"})
    reports = {}

    def run(label, command):
        command = list(map(str, ["taskset", "-c", "0-3", *command]))
        start = time.monotonic()
        result = subprocess.run(command, env=environment, capture_output=True, text=True, timeout=600)
        write_new(output / f"{label}.json", {
            "command": command, "elapsed_seconds": time.monotonic() - start,
            "returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr})
        result.check_returncode()
        return json.loads(result.stdout)

    for name, ids in cases.items():
        token_json, token_text = output / f"{name}-input.json", output / f"{name}-input.txt"
        write_new(token_json, ids)
        with token_text.open("x") as file:
            file.write(" ".join(map(str, ids)) + "\n")
        expected = run(f"{name}-external", [reference, model, token_text, "0", "--f32-reference"])
        if expected["arithmetic"] != "local-f32-activations-v1":
            raise ValueError("external oracle did not preserve F32 activations")
        reports[name] = {}
        for backend in ("reference", "cuda"):
            actual = run(f"{name}-{backend}", [
                rust, "logits", "--backend", backend, "--model", model,
                "--input-tokens-file", token_json, "--context", "1024", "--batch", "128",
                "--prefill-chunk", "128", "--threads", "4"])
            if actual["prompt_token_ids"] != ids:
                raise ValueError(f"changed token input: {name}/{backend}")
            metrics = compare_logits(actual["logits"], expected["logits"])
            # Keep both the original external quality limits and the stricter CUDA gate.
            check_cuda_logits(expected["logits"], actual["logits"])
            if metrics["rmse"] > 0.01:
                raise ValueError(f"RMSE exceeds original external gate: {name}/{backend}")
            reports[name][backend] = metrics
        write_new(output / f"{name}-comparison.json", reports[name])
    after = {str(p): digest(p) for p in files}
    if after != hashes:
        raise ValueError("model, executable, library, or validation source changed during validation")
    write_new(output / "validation.json", {
        "passed": True, "comparisons": reports, "sha256_after": after,
        "thresholds": {"max_absolute_error": 0.01, "minimum_cosine": 0.99999,
                       "maximum_rmse": 0.01, "equal_argmax": True},
        "limitations": [
            "Optimized llama.cpp activation-quantized arithmetic remains different.",
            "The external correctness build must never be used as a performance baseline.",
            "Rust does not link to or call this external oracle."]})
    print(json.dumps(reports, indent=2))


if __name__ == "__main__":
    main()
