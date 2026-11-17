"""Fetch pinned official Qwen3 weights for the KT/llama.cpp comparison."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
SOURCES = [
    ("Qwen/Qwen3-30B-A3B", "ad44e777bcd18fa416d9da3bd8f70d33ebb85d39", "hf"),
    ("Qwen/Qwen3-30B-A3B-GGUF", "e4d4bafdfb96a411a163846265362aceb0b9c63a", "gguf"),
]


def fetch(item):
    repo, revision, entry, directory = item
    path = directory / entry["rfilename"]
    path.parent.mkdir(parents=True, exist_ok=True)
    expected_size = entry["size"]
    expected_hash = entry.get("lfs", {}).get("sha256")
    if path.exists():
        if path.stat().st_size != expected_size:
            raise RuntimeError(f"Existing file has incorrect size: {path}")
        if expected_hash:
            with path.open("rb") as source:
                if hashlib.file_digest(source, "sha256").hexdigest() != expected_hash:
                    raise RuntimeError(f"Existing file has incorrect checksum: {path}")
        print(f"Verified {path.name}", flush=True)
        return
    partial = path.with_name(path.name + ".partial")
    for attempt in range(5):
        offset = partial.stat().st_size if partial.exists() else 0
        if offset > expected_size:
            raise RuntimeError(f"Partial download is larger than expected: {partial}")
        if offset == expected_size:
            break
        url = f"https://huggingface.co/{repo}/resolve/{revision}/{entry['rfilename']}"
        request = urllib.request.Request(url, headers={"Range": f"bytes={offset}-"} if offset else {})
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                if offset and response.status != 206:
                    raise RuntimeError(f"Server did not honor resume request for {path.name}")
                if offset and not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-"):
                    raise RuntimeError(f"Incorrect resume range for {path.name}")
                with partial.open("ab" if offset else "wb") as output:
                    while chunk := response.read(8 * 1024 * 1024):
                        if output.tell() + len(chunk) > expected_size:
                            raise RuntimeError(f"Download exceeds expected size: {path.name}")
                        output.write(chunk)
            break
        except (OSError, TimeoutError) as error:
            print(f"Download attempt {attempt + 1} failed for {path.name}: {error}", flush=True)
            if attempt == 4:
                raise
            time.sleep(2 * (attempt + 1))
    if partial.stat().st_size != expected_size:
        raise RuntimeError(f"Incomplete download: {path.name}")
    if expected_hash:
        with partial.open("rb") as source:
            if hashlib.file_digest(source, "sha256").hexdigest() != expected_hash:
                raise RuntimeError(f"SHA-256 mismatch: {path.name}")
    partial.replace(path)
    print(f"Downloaded and verified {path.name} ({expected_size} bytes)", flush=True)


def main():
    work = []
    lock = []
    for repo, revision, kind in SOURCES:
        with urllib.request.urlopen(
            f"https://huggingface.co/api/models/{repo}/revision/{revision}?blobs=true", timeout=60
        ) as response:
            metadata = json.load(response)
        if metadata["sha"] != revision:
            raise RuntimeError("Model revision mismatch")
        entries = [
            entry for entry in metadata["siblings"]
            if (kind == "hf" and entry["rfilename"].endswith((".safetensors", ".json", ".txt", ".model")))
            or (kind == "gguf" and entry["rfilename"] == "Qwen3-30B-A3B-Q4_K_M.gguf")
        ]
        if not entries:
            raise RuntimeError(f"No weights found for {repo}")
        directory = ROOT / "models" / "benchmark-qwen3" / kind
        work.extend((repo, revision, entry, directory) for entry in entries)
        lock.append({"repository": repo, "revision": revision, "files": entries})
    manifest = ROOT / "models/benchmark-qwen3/models-lock.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(lock, indent=2) + "\n")
    print(f"Fetching {sum(item[2]['size'] for item in work):,} bytes with four workers", flush=True)
    with ThreadPoolExecutor(max_workers=4) as executor:
        list(executor.map(fetch, work))
    print(f"Model manifest: {manifest}", flush=True)


if __name__ == "__main__":
    main()
