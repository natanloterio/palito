#!/usr/bin/env bash
# Packages the CLI and C library for macOS (Apple Silicon, Metal) or Windows
# (x86_64, Vulkan) after `cargo build --release --lib --bin palito --features <gpu>`:
#   <target>/dist/palito-<version>-<platform>/  and  .tar.gz (macOS) / .zip (Windows)
# The layout matches the Linux bundle: the command at the top and the library in
# lib/, where the language bindings look for it. CARGO_TARGET_DIR is honored.
# Used by .github/workflows/cli-bundles.yml (the Linux CUDA bundle is
# scripts/package-linux.sh, built locally).
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"

usage() { echo "usage: $0 macos-arm64|windows-x86_64" >&2; exit 2; }
[[ $# -eq 1 ]] || usage
platform="$1"
case "$platform" in
    macos-arm64) exe=palito; lib=libpalito.dylib; gpu=metal ;;
    windows-x86_64) exe=palito.exe; lib=palito.dll; gpu=vulkan ;;
    *) usage ;;
esac

version="$(sed -n 's/^version = "\(.*\)"/\1/p' Cargo.toml | head -n1)"
target="${CARGO_TARGET_DIR:-$root/target}"
release="$target/release"
name="palito-${version}-${platform}"
dist="$target/dist/$name"

for file in "$release/$exe" "$release/$lib"; do
    if [[ ! -f "$file" ]]; then
        echo "missing $file; build it first: cargo build --release --lib --bin palito --features $gpu" >&2
        exit 1
    fi
done

rm -rf "$dist"
mkdir -p "$dist/lib" "$dist/include" "$dist/examples"
cp "$release/$exe" "$dist/$exe"
chmod 755 "$dist/$exe"
cp "$release/$lib" "$dist/lib/$lib"
cp include/palito.h "$dist/include/palito.h"
cp -R examples/c "$dist/examples/"
cp LICENSE "$dist/LICENSE"
# Binding sources only: no dependencies, build output or caches.
tar -cf - --exclude node_modules --exclude package-lock.json --exclude target --exclude bin \
    --exclude obj --exclude __pycache__ --exclude '*.sln' bindings | tar -C "$dist" -xf -

common_readme() {
    cat <<README
Ready-made models (downloaded from Hugging Face, checked by sha256):
  $1 pull qwen3-8b
  $1 chat --model qwen3-8b
  $1 list --available

Any GGUF file, on the GPU:
  $1 chat --model $2 --backend $gpu
  $1 serve --model $2 --backend $gpu     # OpenAI API on http://127.0.0.1:8080/v1

.palito models run on the CPU here (their GPU kernels are CUDA-only for now);
GGUF models run on the CPU unless --backend $gpu is given (or PALITO_BACKEND=$gpu).
Models are stored in $3 (set PALITO_MODELS to use another disk).

Library (embed Palito in your program, no server):
  include/palito.h, lib/$lib, examples/c/chat.c
Language bindings in bindings/ find lib/$lib through the palito command on
your PATH, or set PALITO_LIBRARY. See https://github.com/natanloterio/palito
README
}

case "$platform" in
    macos-arm64)
        {
            cat <<README
palito ${version} (macOS, Apple Silicon)

Install:
  mkdir -p ~/.local/bin && ln -sf "\$PWD/palito" ~/.local/bin/palito
  (add ~/.local/bin to PATH in ~/.zshrc if 'palito' is not found)
If the archive was downloaded with a browser, clear the quarantine flag first:
  xattr -dr com.apple.quarantine .

README
            common_readme palito /path/to/model.gguf "~/.local/share/palito/models"
        } > "$dist/README.txt"
        tar -C "$target/dist" -czf "$target/dist/$name.tar.gz" "$name"
        archive="$target/dist/$name.tar.gz"
        ;;
    windows-x86_64)
        {
            cat <<README
palito ${version} (Windows x86_64, Vulkan)

Requires a GPU driver with Vulkan for --backend vulkan (AMD, Intel and NVIDIA
drivers ship it). Run from PowerShell or cmd in this folder, or add the folder
to PATH.

README
            common_readme palito.exe 'C:\models\model.gguf' '%USERPROFILE%\.local\share\palito\models'
        } | sed 's/$/\r/' > "$dist/README.txt"
        python="$(command -v python3 || command -v python)"
        (cd "$target/dist" && "$python" -m zipfile -c "$name.zip" "$name")
        archive="$target/dist/$name.zip"
        ;;
esac

echo "bundle: $dist"
echo "archive: $archive"
