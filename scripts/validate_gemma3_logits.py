"""Compare Gemma 3 270M raw-ID logits with the pinned F32-GELU oracle."""

import argparse
import json
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory

from validate_cuda import compare_logits


def run_json(*command):
    return json.loads(subprocess.run(
        list(map(str, command)), check=True, stdout=subprocess.PIPE,
        text=True, timeout=240,
    ).stdout)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--rust", type=Path, required=True)
    parser.add_argument("--backend", choices=("reference", "cuda"), default="reference")
    parser.add_argument("--oracle", type=Path, required=True,
                        help="logits-oracle built with build_f32_oracle.py --f32-gelu")
    args = parser.parse_args()

    sequence = [2] + [198, 201, 202, 203] * 128 + [198, 201]
    with TemporaryDirectory() as directory:
        raw = Path(directory) / "ids.json"
        text = Path(directory) / "ids.txt"
        for count in (1, 3, 511, 512, 515):
            ids = sequence[:count]
            raw.write_text(json.dumps(ids), encoding="utf-8")
            text.write_text(" ".join(map(str, ids)), encoding="utf-8")
            expected = run_json(args.oracle.resolve(), args.model, text, "0", "--f32-reference")
            actual = run_json(
                args.rust.resolve(), "logits", "--backend", args.backend,
                "--model", args.model, "--context", "1024", "--input-tokens-file", raw,
            )
            if actual["prompt_token_ids"] != ids or len(expected["logits"]) != 262144:
                raise ValueError(f"input IDs or vocabulary changed at {count} tokens")
            print(json.dumps({"input_tokens": count, **compare_logits(
                expected["logits"], actual["logits"],
            )}), flush=True)


if __name__ == "__main__":
    main()
