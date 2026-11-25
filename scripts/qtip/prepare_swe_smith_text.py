#!/usr/bin/env python3
"""Render SWE-smith agent trajectories (SWE-bench/SWE-smith-trajectories, MIT)
as plain text for Hessian calibration and perplexity evaluation.

Each trajectory becomes "### <role>\\n<content>\\n\\n" per message, cut
at `--max-trajectory-chars`. The
calibration file comes from `--calib` shards and the evaluation file from
`--eval` shards, keeping only evaluation trajectories whose repository
(the instance_id before the first '.') never appears in calibration.
Needs pyarrow.

    python3 scripts/qtip/prepare_swe_smith_text.py \\
        --calib models/swe-smith-trajectories/data/ticks-00000-of-00008.parquet \\
        --eval models/swe-smith-trajectories/data/ticks-00001-of-00008.parquet \\
        --out-calib models/swe-smith-trajectories/calib.txt \\
        --out-eval models/swe-smith-trajectories/eval.txt
"""

import argparse
import json
import sys
from pathlib import Path

import pyarrow.parquet as pq


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--calib", type=Path, nargs="+", required=True)
    parser.add_argument("--eval", type=Path, nargs="+", required=True)
    parser.add_argument("--out-calib", type=Path, required=True)
    parser.add_argument("--out-eval", type=Path, required=True)
    parser.add_argument("--max-chars", type=int, default=8_000_000, help="per output file")
    parser.add_argument("--max-trajectory-chars", type=int, default=16_000,
                        help="truncate each trajectory, for more repositories per file")
    return parser.parse_args()


def repository(instance_id: str) -> str:
    return instance_id.split(".", 1)[0]


def render(messages_json: str) -> str | None:
    try:
        messages = json.loads(messages_json)
    except json.JSONDecodeError:
        return None
    parts = []
    for message in messages:
        content = message.get("content")
        if isinstance(content, list):  # [{"type": "text", "text": ...}, ...]
            content = "".join(c.get("text", "") for c in content if isinstance(c, dict))
        if not isinstance(content, str) or not content.strip():
            continue
        parts.append(f"### {message.get('role', 'unknown')}\n{content.strip()}\n\n")
    return "".join(parts) or None


def rows(paths):
    for path in paths:
        if not path.is_file():
            sys.exit(f"not found: {path}")
        table = pq.read_table(path, columns=["messages", "instance_id"])
        yield from zip(table.column("instance_id").to_pylist(), table.column("messages").to_pylist())


def write(paths, output: Path, max_chars: int, max_trajectory: int, skip_repos=frozenset()):
    if output.exists():
        sys.exit(f"{output} already exists")
    repos, total, count = set(), 0, 0
    with output.open("w") as out:
        for instance_id, messages in rows(paths):
            if repository(instance_id) in skip_repos:
                continue
            text = render(messages)
            if text is None:
                continue
            text = text[:max_trajectory] + "\n\n"
            out.write(text)
            repos.add(repository(instance_id))
            total += len(text)
            count += 1
            if total >= max_chars:
                break
    print(f"{output}: {count} trajectories, {len(repos)} repositories, {total} chars")
    return repos


def main():
    args = parse_args()
    calib_repos = write(args.calib, args.out_calib, args.max_chars, args.max_trajectory_chars)
    write(args.eval, args.out_eval, args.max_chars, args.max_trajectory_chars, frozenset(calib_repos))


if __name__ == "__main__":
    main()
