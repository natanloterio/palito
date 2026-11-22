"""Verify embedded GGUF text support against an explicit external tokenizer oracle."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from benchmark import ROOT, digest, run


CASES = {
    "empty": "",
    "ascii": "Hello, world!",
    "contractions": "I'm ready; she's here. We've said don't, can't, I'll.",
    "uppercase-contractions": "I'M WE'RE DON'T He'S",
    "numbers": "0 1234567890 3.14159 -42 1e10",
    "unicode-numbers": "\u0661\u0662\u0663 \uff11\uff12\uff13 \u00bd\u2167",
    "leading-spaces": "   leading  middle   trailing   ",
    "whitespace-only": " \t  \r\n\n\t ",
    "line-endings": "one\r\ntwo\n\nthree\tfour",
    "punctuation": "foo_bar != (x+y); https://example.test/a?b=1 ...",
    "accented": "caf\u00e9 na\u00efve Stra\u00dfe \u20ac \u00a3",
    "combining": "cafe\u0301 a\u0308 \u0301",
    "multilingual": "\u4f60\u597d \u65e5\u672c\u8a9e \u0645\u0631\u062d\u0628\u0627 \u0928\u092e\u0938\u094d\u0924\u0947",
    "emoji": "\U0001f642 \U0001f469\u200d\U0001f4bb \u2764\ufe0f",
    "unicode-spaces": "a\u00a0b\u2003c\u2028d",
    "nul": "before\0after",
    "special-adjacent": "<|im_start|><|im_end|><|endoftext|>",
    "special-prefix": "<|im_sta <|im_start|> <|im_end|>suffix",
    "formatted-chat": "<|im_start|>user\nSay hello.<|im_end|>\n<|im_start|>assistant\n",
    "long-word": "ab" * 2048,
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=ROOT / "models/SmolLM2-135M-Instruct-Q4_K_M.gguf")
    parser.add_argument("--rust", type=Path, default=ROOT / "target/release/palito")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    model, rust, reference = (p.resolve(strict=True) for p in (args.model, args.rust, args.reference))
    directory = args.output_dir.resolve()
    directory.mkdir(parents=True, exist_ok=False)

    def save(name, value):
        with (directory / name).open("x") as output:
            json.dump(value, output, indent=2, allow_nan=False)
            output.write("\n")

    linkage = run(["ldd", rust])
    if any(name in linkage["stdout"] for name in ("libllama", "libggml", "libli_bridge")):
        raise ValueError("Rust executable links a forbidden native inference library")
    save("provenance.json", {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "model_sha256": digest(model),
        "rust_sha256": digest(rust),
        "reference_sha256": digest(reference),
        "rust_linkage": linkage,
        "reference_linkage": run(["ldd", reference]),
        "reference_libraries": {p.name: digest(p) for p in sorted(reference.parent.glob("*.so*"))
                                if p.is_file()},
        "cargo_lock_sha256": digest(ROOT / "Cargo.lock"),
        "source_sha256": {str(p.relative_to(ROOT)): digest(p)
                         for p in sorted((ROOT / "src").rglob("*.rs"))},
        "oracle_source_sha256": {p.name: digest(p) for p in sorted((ROOT / "native").glob("*"))
                                if p.is_file()},
        "script_sha256": digest(Path(__file__)),
        "scope": "Exact token IDs and decoded UTF-8 bytes; no inference kernels are shared",
    })
    checks = {}
    cases = {**CASES, "benchmark-prompt": (ROOT / "fixtures/prompt.txt").read_text()}
    for name, text in cases.items():
        path = directory / f"{name}-input.txt"
        path.write_bytes(text.encode("utf-8"))
        actual = run([rust, "tokenize", "--model", model, "--prompt-file", path])
        expected = run([reference, model, path, 512, 128, 64, 4, 1, 1, 1, "v4-tokenize"])
        save(f"{name}-rust.json", actual)
        save(f"{name}-reference.json", expected)
        a, b = json.loads(actual["stdout"]), json.loads(expected["stdout"])
        checks[name] = {
            "prompt_tokens": len(a["prompt_token_ids"]),
            "tokens_equal": a["prompt_token_ids"] == b["prompt_token_ids"],
            "decoded_equal": list(a["decoded_text"].encode("utf-8")) == b["decoded_bytes"],
        }
        checks[name]["passed"] = checks[name]["tokens_equal"] and checks[name]["decoded_equal"]
        print(name, json.dumps(checks[name]), flush=True)

    prompt = directory / "benchmark-prompt-input.txt"
    ids = json.loads(json.loads((directory / "benchmark-prompt-rust.json").read_text())["stdout"])[
        "prompt_token_ids"]
    token_file = directory / "generation-input.json"
    save(token_file.name, ids)
    generated = run([rust, "generate", "--model", model, "--prompt-file", prompt, "--tokens", 8])
    save("generation-text.json", generated)
    text_record = json.loads(generated["stdout"])
    text_ids = text_record["sample"]["generated_tokens"]
    explicit = run([rust, "bench", "--model", model, "--input-tokens-file", token_file,
                    "--tokens", len(text_ids), "--warmups", 0, "--repetitions", 1])
    save("generation-explicit-ids.json", explicit)
    detokenize_file = directory / "generation-output.json"
    save(detokenize_file.name, text_ids)
    decoded = run([rust, "detokenize", "--model", model, "--input-tokens-file", detokenize_file])
    save("generation-decoded.json", decoded)
    id_record = json.loads(explicit["stdout"])
    checks["generation"] = {
        "prompt_tokens_equal": text_record["prompt_token_ids"] == id_record["prompt_token_ids"] == ids,
        "generated_tokens_equal": text_ids == id_record["samples"][0]["generated_tokens"],
        "decoded_text_equal": text_record["text"] == json.loads(decoded["stdout"])["text"],
        "text": text_record["text"],
    }
    checks["generation"]["passed"] = all(checks["generation"][key] for key in
                                        ("prompt_tokens_equal", "generated_tokens_equal", "decoded_text_equal"))
    passed = all(check["passed"] for check in checks.values())
    save("validation.json", {"finished_at": datetime.now(timezone.utc).isoformat(),
                            "passed": passed, "checks": checks})
    if not passed:
        raise ValueError("text validation failed; see saved per-case outputs")


if __name__ == "__main__":
    main()
