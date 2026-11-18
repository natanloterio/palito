"""Check exact token IDs, full-vocabulary logits and separate greedy generation."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory

from validate_cuda import compare_logits


FIXTURES = {
    "llama32": (128256, 128000, {
        "<|start_header_id|>user<|end_header_id|>\n\nHi<|eot_id|>": [128006, 128007, 128009],
        "<|eom_id|><|end_of_text|>": [128008, 128001],
    }),
    "phi3-mini-4k": (32064, 1, {
        "<|user|>\nHi<|end|>\n<|assistant|>": [32010, 32007, 32001],
        "<|system|>Test<|endoftext|></s>": [32006, 32000, 2],
    }),
    "qwen25": (151936, None, {
        "<|im_start|>user\nHi<|im_end|>": [151644, 151645],
        "<|endoftext|><|im_end|>": [151643, 151645],
    }),
    "gemma2": (256000, 2, {
        "<start_of_turn>user\nHi<end_of_turn>": [106, 107],
        "<end_of_turn><eos>": [107, 1],
    }),
    "smollm2": (49152, None, {
        "<|im_start|>user\nHi<|im_end|>": [1, 2],
        "<|endoftext|><reponame>": [0, 4],
    }),
    "starcoder2": (49152, None, {
        "<fim_prefix>def add(a, b):\n<fim_suffix>\n<fim_middle>": [1, 3, 2],
        "<|endoftext|><fim_pad>": [0, 4],
    }),
    "gpt2": (50257, None, {"<|endoftext|>": [50256]}),
    "bloom": (250880, None, {"<s>Bonjour</s>": [1, 2], "<unk><pad>": [3, 0]}),
    "pythia": (50304, None, {"<|endoftext|>": [0]}),
    "olmo": (50304, None, {"<|endoftext|>": [50279]}),
    "olmo2": (100352, None, {"<|endoftext|><|im_end|>": [100257, 100265]}),
    "olmoe": (50304, None, {"<|endoftext|>": [50279]}),
    "mamba": (50280, None, {"<|endoftext|>": [0]}),
    "qwen35": (248320, None, {
        "<|endoftext|><|im_end|><|fim_pad|><|repo_name|><|file_sep|>":
            [248044, 248046, 248063, 248064, 248065],
    }),
}
FIXTURES["qwen3"] = FIXTURES["qwen25"]
FIXTURES["phi3-mini-128k"] = FIXTURES["phi3-mini-4k"]
PROMPT = "The capital of France is"
CONTINUATION = " Paris. It is located on the Seine."


def run(*command):
    p = subprocess.run(list(map(str, command)), capture_output=True, text=True, timeout=600)
    if p.returncode:
        raise RuntimeError(f"{command}: {p.stderr}")
    return json.loads(p.stdout)


def sha256(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", choices=FIXTURES, required=True)
    parser.add_argument("--backend", choices=("cpu", "reference", "cuda"), required=True)
    parser.add_argument("--context", type=int, help="Configure the same context on both runtimes")
    for arg in ("model", "rust", "oracle", "output"):
        parser.add_argument(f"--{arg}", type=Path, required=True)
    args = parser.parse_args()
    if args.context is not None and args.context <= 0:
        parser.error("--context must be positive")
    if args.fixture == "phi3-mini-128k" and args.context is None:
        parser.error("Phi-3 Mini 128K requires explicit --context to identify factor selection")
    vocab, bos, specials = FIXTURES[args.fixture]
    model, rust, oracle = (p.resolve(strict=True) for p in (args.model, args.rust, args.oracle))
    report = {"fixture": args.fixture, "backend": args.backend,
              "sha256": {p.name: sha256(p) for p in (model, rust, oracle)},
              "tokenizer": {}, "checkpoints": []}
    if args.context is not None:
        report["context_tokens"] = args.context
    with TemporaryDirectory() as d:
        text, ids, raw = (Path(d) / name for name in ("prompt.txt", "ids.txt", "ids.json"))

        def oracle_run(path, *flags):
            context = (f"--context={args.context}",) if args.context and "--tokenize" not in flags else ()
            return run(oracle, model, path, 0, "--f32-reference", *context, *flags)

        def encode(prompt, required=()):
            text.write_bytes(prompt.encode("utf-8"))
            expected = oracle_run(text, "--tokenize")["token_ids"]
            actual = run(rust, "tokenize", "--model", model, "--prompt-file", text)["prompt_token_ids"]
            if expected != actual or (bos is not None and actual[:1] != [bos]) or not set(required) <= set(actual):
                raise ValueError(f"tokenizer mismatch for {prompt!r}: {expected} != {actual}")
            report["tokenizer"][prompt] = actual
            return actual

        def save_ids(tokens):
            ids.write_text(" ".join(map(str, tokens)))
            raw.write_text(json.dumps(tokens))

        prompt_text, continuation = (
            ("def square(x):", "\n    return x * x\n\nprint(square(5))\n")
            if args.fixture == "starcoder2" else (PROMPT, CONTINUATION)
        )
        prompt = encode(prompt_text)
        sequence = encode(prompt_text + continuation)
        if sequence[:len(prompt)] != prompt or len(sequence) < len(prompt) + 4:
            raise ValueError("the prompt is not a prefix of the fixed continuation")
        accent = "caf\u00e9" if args.fixture in ("qwen25", "qwen3", "smollm2") else "cafe\u0301"
        for case in ("", f" I'M 1234567890\r\n{accent}, \u4f60\u597d \U0001f642\t ", "a\0b"):
            encode(case)
        for case, required in specials.items():
            encode(case, required)
        common = ("--backend", args.backend, "--model", model, "--prefill-chunk", 4)
        if args.context is not None:
            common += ("--context", args.context)
        for count in (1, len(prompt), len(prompt) + 1, len(prompt) + 4, len(sequence)):
            tokens = sequence[:count]
            save_ids(tokens)
            expected = oracle_run(ids)
            actual = run(rust, "logits", *common, "--input-tokens-file", raw)
            if actual["prompt_token_ids"] != tokens or len(expected["logits"]) != vocab:
                raise ValueError("IDs or logit vocabulary changed")
            checkpoint = {"input_ids": tokens, **compare_logits(expected["logits"], actual["logits"])}
            report["checkpoints"].append(checkpoint)
            print(json.dumps(checkpoint), flush=True)

        save_ids(prompt)
        greedy = oracle_run(ids, "--greedy=4")["token_ids"]
        actual = run(rust, "generate", *common, "--prompt", prompt_text, "--tokens", 4)
        if actual["prompt_token_ids"] != prompt or actual["sample"]["generated_tokens"] != greedy:
            raise ValueError(f"greedy mismatch: expected {greedy}, got {actual['sample']}")
        report["greedy_ids"] = greedy

    with args.output.open("x") as output:
        json.dump(report, output, indent=2, allow_nan=False)
        output.write("\n")


if __name__ == "__main__":
    main()
