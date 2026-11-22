"""Full-logit CUDA/independent CPU comparisons and offline Qwen tokenizer checks."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess

from benchmark_moe import ROOT, gpu_state, write_new
from validate_rust_tokenizer import CASES


def compare_logits(expected, actual):
    if not expected or len(expected) != len(actual):
        raise ValueError("logit vector shape mismatch")
    if any(not math.isfinite(x) for x in expected + actual):
        raise ValueError("nonfinite logits")
    maximum = max(abs(a - b) for a, b in zip(expected, actual))
    dot = sum(a * b for a, b in zip(expected, actual))
    norms = math.sqrt(sum(x * x for x in expected) * sum(x * x for x in actual))
    cosine = dot / norms if norms else float("nan")
    argmax = lambda values: max(range(len(values)), key=values.__getitem__)
    if maximum > 0.01 or not cosine >= 0.99999 or argmax(expected) != argmax(actual):
        raise ValueError(f"CUDA numerical mismatch: max_abs={maximum}, cosine={cosine}")
    return {"logits": len(actual), "maximum_absolute_error": maximum,
            "cosine_similarity": cosine, "greedy_token": argmax(actual)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cuda-root", type=Path, required=True)
    parser.add_argument("--binary", type=Path, default=ROOT / "target/release/palito")
    args = parser.parse_args()
    directory = args.output_dir.resolve()
    directory.mkdir(parents=True, exist_ok=False)
    rust = args.binary.resolve(strict=True)
    binary_hash = hashlib.sha256(rust.read_bytes()).hexdigest()
    environment = dict(os.environ)
    environment["LD_LIBRARY_PATH"] = os.pathsep.join(filter(None, [
        str(args.cuda_root.resolve(strict=True) / "lib"), environment.get("LD_LIBRARY_PATH")]))
    reports = {}

    def run(label, command):
        command = [str(x) for x in command]
        result = subprocess.run(command, env=environment, capture_output=True, text=True, timeout=600)
        write_new(directory / f"{label}-process.json", {
            "command": command, "returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr})
        result.check_returncode()
        return json.loads(result.stdout)

    for name, model, cpu in [
        ("dense", ROOT / "models/SmolLM2-135M-Instruct-Q4_K_M.gguf", "cpu"),
        ("moe", ROOT / "models/benchmark-qwen3/gguf/Qwen3-30B-A3B-Q4_K_M.gguf", "reference"),
    ]:
        cases = {"one": [1], "short": [1, 2, 3], "longer": [1, 2, 3, 4, 5, 6, 7]}
        reports[name] = {}
        for case, ids in cases.items():
            path = directory / f"{name}-{case}-input.json"
            write_new(path, ids)
            common = ["--model", model, "--input-tokens-file", path,
                      "--context", "16", "--batch", "8", "--prefill-chunk", "3", "--threads", "4"]
            oracle = run(f"{name}-{case}-cpu", [rust, "logits", "--backend", cpu, *common])
            actual = run(f"{name}-{case}-cuda", [rust, "logits", "--backend", "cuda", *common])
            reports[name][case] = compare_logits(oracle["logits"], actual["logits"])
            if actual["prompt_token_ids"] != ids or oracle["prompt_token_ids"] != ids:
                raise ValueError("input IDs changed")
        # Reset and chunk invariance are checked across all repetitions, not just the first sample.
        path = directory / f"{name}-short-input.json"
        outputs = []
        for backend, chunk in [("cuda", 1), ("cuda", 3), (cpu, 3)]:
            result = run(f"{name}-{backend}-chunk{chunk}-bench", [
                rust, "bench", "--backend", backend, "--model", model,
                "--input-tokens-file", path, "--context", "8", "--batch", "3",
                "--prefill-chunk", chunk, "--tokens", "6", "--warmups", "1", "--repetitions", "2"])
            for sample in result["samples"]:
                if len(sample["generated_tokens"]) != 6 or sample["cached_prompt_tokens"] != 0:
                    raise ValueError("incomplete or cached generation")
                outputs.append(sample["generated_tokens"])
        if any(output != outputs[0] for output in outputs):
            raise ValueError("generation differs across CPU/GPU, chunks, or reset")
        reports[name]["generation"] = {"generated_tokens": outputs[0], "exact_context_boundary": True}

    texts = dict(CASES)
    texts["benchmark-prompt"] = (ROOT / "fixtures/prompt.txt").read_text()
    hf = ROOT / "models/benchmark-qwen3/hf"
    python = ROOT / ".venv-ktransformers-bench/bin/python"
    code = (
        "import json,sys; from transformers import AutoTokenizer; "
        "t=AutoTokenizer.from_pretrained(sys.argv[1],local_files_only=True,trust_remote_code=False); "
        "texts=json.load(sys.stdin); ids={k:t.encode(v,add_special_tokens=False) for k,v in texts.items()}; "
        "print(json.dumps({k:{'ids':v,'decoded':t.decode(v,skip_special_tokens=True,clean_up_tokenization_spaces=False)} for k,v in ids.items()}))")
    expected = json.loads(subprocess.check_output([str(python), "-c", code, str(hf)],
                                                input=json.dumps(texts), text=True, timeout=120))
    tokenizer = {}
    for name, text in texts.items():
        prompt = directory / f"qwen-{name}-prompt.txt"
        with prompt.open("x") as output:
            output.write(text)
        actual = run(f"qwen-tokenizer-{name}", [
            rust, "tokenize", "--model", ROOT / "models/benchmark-qwen3/gguf/Qwen3-30B-A3B-Q4_K_M.gguf",
            "--prompt-file", prompt])
        if actual["prompt_token_ids"] != expected[name]["ids"] or actual["decoded_text"] != expected[name]["decoded"]:
            raise ValueError(f"Qwen tokenizer differs from HF: {name}")
        tokenizer[name] = {"prompt_token_ids": actual["prompt_token_ids"], "decoded_text": actual["decoded_text"]}
    sources = [ROOT / "Cargo.toml", ROOT / "Cargo.lock", Path(__file__), *sorted((ROOT / "src").rglob("*.rs")),
               *sorted((ROOT / "src").rglob("*.cu"))]
    if hashlib.sha256(rust.read_bytes()).hexdigest() != binary_hash:
        raise ValueError("validation executable changed during the run")
    report = {
        "schema_version": 1, "full_logit_validation": reports, "qwen_tokenizer": tokenizer,
        "binary": str(rust), "binary_sha256": binary_hash,
        "source_sha256": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources},
        "gpu_after": gpu_state(),
        "limitations": ["CPU and CUDA share the strict GGUF parser and model specification, not numerical kernels.",
                        "Qwen reference stays packed; no 120 GB expanded model is allocated.",
                        "This is numerical/state validation, not a cross-engine speed claim."],
    }
    write_new(directory / "validation.json", report)
    print(json.dumps({"numerical": reports, "qwen_tokenizer_cases": len(tokenizer)}, indent=2))


if __name__ == "__main__":
    main()
