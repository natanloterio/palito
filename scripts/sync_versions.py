#!/usr/bin/env python3
"""Keeps the language bindings at the crate version (Cargo.toml [package]).

    python3 scripts/sync_versions.py                  # rewrite stale manifests
    python3 scripts/sync_versions.py --check          # exit 1 if any is stale
    python3 scripts/sync_versions.py --check --tag v0.7.0   # also match a tag

Only the package's own version is touched: the first match of each pattern,
which in every manifest comes before any dependency version.
"""
import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Manifest:
    path: str
    pattern: str  # group 1: text before the version, group 2: the version, group 3: after

    def find(self, text):
        match = re.search(self.pattern, text, re.MULTILINE)
        if match is None:
            raise ValueError(f"{self.path}: no version found")
        return match

    def version(self, text):
        return self.find(text).group(2)

    def with_version(self, text, version):
        match = self.find(text)
        return text[:match.start(2)] + version + text[match.end(2):]


MANIFESTS = (
    Manifest("bindings/python/pyproject.toml", r'^(version = ")([^"]+)(")'),
    Manifest("bindings/node/package.json", r'^(  "version": ")([^"]+)(")'),
    Manifest("bindings/csharp/Palito/Palito.csproj", r"(<Version>)([^<]+)(</Version>)"),
    # The project's own <version> is the first one at two-space indent.
    Manifest("bindings/java/pom.xml", r"^(  <version>)([^<]+)(</version>)"),
    # The Maven dependency snippet users copy; it lags otherwise.
    Manifest("bindings/java/README.md", r"^(  <version>)([^<]+)(</version>)"),
)


def cargo_version(root):
    text = (Path(root) / "Cargo.toml").read_text()
    match = re.search(r'^\[package\][^\[]*?^version = "([^"]+)"', text, re.MULTILINE | re.DOTALL)
    if match is None:
        raise ValueError("Cargo.toml: no [package] version")
    return match.group(1)


def check(root, version):
    """Paths of the manifests whose version differs from `version`."""
    return [m.path for m in MANIFESTS if m.version((Path(root) / m.path).read_text()) != version]


def sync(root, version):
    """Rewrites the stale manifests; returns their paths."""
    stale = check(root, version)
    for manifest in MANIFESTS:
        if manifest.path in stale:
            file = Path(root) / manifest.path
            file.write_text(manifest.with_version(file.read_text(), version))
    return stale


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", default=Path(__file__).resolve().parent.parent)
    parser.add_argument("--check", action="store_true", help="report instead of rewriting")
    parser.add_argument("--tag", help="release tag (vX.Y.Z or refs/tags/vX.Y.Z) that must match")
    args = parser.parse_args(argv)
    try:
        version = cargo_version(args.root)
        if args.tag is not None:
            tag = args.tag.removeprefix("refs/tags/")
            if tag != f"v{version}":
                print(f"tag {tag} does not match Cargo.toml version {version}", file=sys.stderr)
                return 1
        if args.check:
            stale = check(args.root, version)
            for path in stale:
                print(f"{path}: not at {version} (run scripts/sync_versions.py)", file=sys.stderr)
            return 1 if stale else 0
        for path in sync(args.root, version):
            print(f"{path} -> {version}")
        return 0
    except (OSError, ValueError) as error:
        print(f"sync_versions: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
