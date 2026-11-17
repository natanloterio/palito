"""Download only the pinned public benchmark model; never install Python packages."""
import hashlib
import json
from pathlib import Path
import tempfile
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def main():
    manifest = json.loads((ROOT / "fixtures/model.json").read_text())
    destination = ROOT / "models" / manifest["filename"]
    destination.parent.mkdir(exist_ok=True)
    if destination.exists():
        with destination.open("rb") as source:
            checksum = hashlib.file_digest(source, "sha256").hexdigest()
        if destination.stat().st_size != manifest["size_bytes"] or checksum != manifest["sha256"]:
            raise RuntimeError(f"Existing model fails size/checksum validation: {destination}")
        print(f"Verified existing model: {destination}")
        return
    url = (f'https://huggingface.co/{manifest["repository"]}/resolve/'
           f'{manifest["revision"]}/{manifest["filename"]}')
    temporary = None
    try:
        digest = hashlib.sha256()
        size = 0
        with tempfile.NamedTemporaryFile(dir=destination.parent, suffix=".part", delete=False) as output:
            temporary = Path(output.name)
            with urllib.request.urlopen(url, timeout=60) as response:
                while chunk := response.read(1024 * 1024):
                    size += len(chunk)
                    if size > manifest["size_bytes"]:
                        raise RuntimeError("Download exceeds pinned file size")
                    digest.update(chunk)
                    output.write(chunk)
        if size != manifest["size_bytes"] or digest.hexdigest() != manifest["sha256"]:
            raise RuntimeError("Downloaded model fails pinned size/SHA-256 validation")
        temporary.replace(destination)
        print(f"Downloaded and verified {size} bytes: {destination}")
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
