#!/usr/bin/env bash
# Builds and packages the palito CLI and C library for one platform:
#   target/dist/palito-<version>-<platform>/  and its .tar.gz (.zip on Windows)
#
#   linux-x86_64    CUDA CLI with `convert`, NVRTC 13 bundled, wrapper script
#   macos-arm64     Metal
#   windows-x86_64  Vulkan
#
# Every bundle has the command at the top and the library in lib/, where the
# language bindings look for it, plus the header, the C example, the binding
# sources, LICENSE and a README.
#
# Options:
#   --no-build  package what is already in target/release
#   --install   (Linux, macOS) copy to ~/.local/share/palito/<version> and link
#               ~/.local/bin/palito
# Environment: NVRTC_LIB_DIR (Linux: where libnvrtc.so.13 is taken from; default
# the project's CUDA 13 benchmark venv), CARGO_TARGET_DIR.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"

usage() { echo "usage: $0 linux-x86_64|macos-arm64|windows-x86_64 [--no-build] [--install]" >&2; exit 2; }
platform=""
build=1
install=0
for arg in "$@"; do
    case "$arg" in
        --no-build) build=0 ;;
        --install) install=1 ;;
        linux-x86_64 | macos-arm64 | windows-x86_64) [[ -z "$platform" ]] || usage; platform="$arg" ;;
        *) usage ;;
    esac
done
case "$platform" in
    linux-x86_64) exe=palito; lib=libpalito.so; features=cuda,qtip-research; gpu=cuda ;;
    macos-arm64) exe=palito; lib=libpalito.dylib; features=metal; gpu=metal ;;
    windows-x86_64) exe=palito.exe; lib=palito.dll; features=vulkan; gpu=vulkan ;;
    *) usage ;;
esac
if [[ $install -eq 1 && "$platform" == windows-x86_64 ]]; then
    echo "--install is for Linux and macOS; on Windows add the bundle folder to PATH" >&2
    exit 2
fi

version="$(sed -n 's/^version = "\(.*\)"/\1/p' Cargo.toml | head -n1)"
target="${CARGO_TARGET_DIR:-$root/target}"
release="$target/release"
name="palito-${version}-${platform}"
dist="$target/dist/$name"
python="$(command -v python3 || command -v python)"
build_command="cargo build --release --lib --bin palito --features $features"

if [[ "$platform" == linux-x86_64 ]]; then
    nvrtc="${NVRTC_LIB_DIR:-$root/.venv-cuda-bench/lib/python3.11/site-packages/nvidia/cu13/lib}"
    [[ -f "$nvrtc/libnvrtc.so.13" ]] || { echo "libnvrtc.so.13 not found in $nvrtc (set NVRTC_LIB_DIR)" >&2; exit 1; }
fi

# The bindings must carry the same version as the bundle.
"$python" scripts/sync_versions.py --check

if [[ $build -eq 1 ]]; then
    $build_command
fi
for file in "$release/$exe" "$release/$lib"; do
    [[ -f "$file" ]] || { echo "missing $file; build it first: $build_command" >&2; exit 1; }
done

rm -rf "$dist"
mkdir -p "$dist/lib" "$dist/include" "$dist/examples"
cp "$release/$lib" "$dist/lib/$lib"
cp include/palito.h "$dist/include/palito.h"
cp -R examples/c "$dist/examples/"
cp LICENSE "$dist/LICENSE"
# Binding sources only: no dependencies, build output or caches.
tar -cf - --exclude node_modules --exclude package-lock.json --exclude target --exclude bin \
    --exclude obj --exclude __pycache__ --exclude '*.sln' bindings | tar -C "$dist" -xf -

if [[ "$platform" == linux-x86_64 ]]; then
    # The real binary goes to bin/; the top-level palito is a wrapper.
    mkdir -p "$dist/bin"
    cp "$release/$exe" "$dist/bin/palito"
    chmod 755 "$dist/bin/palito"
    strip "$dist/bin/palito" "$dist/lib/$lib" 2>/dev/null || true
    cp -L "$nvrtc/libnvrtc.so.13" "$dist/lib/"
    cp -L "$nvrtc"/libnvrtc-builtins.so.13* "$dist/lib/"
    # cudarc probes the unversioned name first; keep it pointing at the bundled 13.x.
    ln -s libnvrtc.so.13 "$dist/lib/libnvrtc.so"
    cat > "$dist/palito" <<'WRAPPER'
#!/usr/bin/env bash
# Runs the bundled CLI with its NVRTC, the CUDA backend and a persistent kernel
# cache by default. Every default can be overridden by flag or environment.
set -euo pipefail
here="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)"
export LD_LIBRARY_PATH="$here/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PALITO_BACKEND="${PALITO_BACKEND:-cuda}"
cache="${PALITO_CUDA_KERNEL_CACHE:-${XDG_CACHE_HOME:-$HOME/.cache}/palito/kernels}"
mkdir -p -m 700 "$cache"
export PALITO_CUDA_KERNEL_CACHE="$cache"
exec "$here/bin/palito" "$@"
WRAPPER
else
    cp "$release/$exe" "$dist/$exe"
fi
chmod 755 "$dist/$exe"

models_readme() {
    cat <<README
Ready-made models (downloaded from Hugging Face, checked by sha256):
  $1 pull qwen3-8b
  $1 chat --model qwen3-8b
  $1 list --available

README
}

library_readme() {
    cat <<README
Library (embed Palito in your program, same process, no server):
  include/palito.h, lib/$lib, examples/c/chat.c

Language bindings in bindings/ (each finds lib/$lib through the
installed palito command, or set PALITO_LIBRARY):
  Python   bindings/python   pip install ./bindings/python
  Node.js  bindings/node     npm install ./bindings/node   (uses koffi)
  Java 22+ bindings/java     mvn -f bindings/java/pom.xml install
  C#       bindings/csharp   dotnet add reference bindings/csharp/Palito/Palito.csproj
  Go       bindings/go       go get github.com/natanloterio/palito/bindings/go

https://github.com/natanloterio/palito
README
}

other_gpu_readme() {
    cat <<README
Any GGUF file, on the GPU:
  $1 chat --model $2 --backend $gpu
  $1 serve --model $2 --backend $gpu     # OpenAI API on http://127.0.0.1:8080/v1

.palito models run on the CPU here (their GPU kernels are CUDA-only for now);
GGUF models run on the CPU unless --backend $gpu is given (or PALITO_BACKEND=$gpu).
Models are stored in $3 (set PALITO_MODELS to use another disk).

README
}

readme() {
    case "$platform" in
        linux-x86_64)
            cat <<README
palito ${version} (Linux x86_64, CUDA)

Requires an NVIDIA driver with CUDA 13 support. NVRTC 13 is bundled in lib/.

README
            models_readme ./palito
            cat <<README
Any GGUF file:
  ./palito chat --model /path/to/model.gguf
  ./palito serve --model /path/to/model.gguf     # OpenAI API on http://127.0.0.1:8080/v1
  ./palito --help

Defaults set by the wrapper (override with flags or by exporting them):
  PALITO_BACKEND=cuda
  PALITO_CUDA_KERNEL_CACHE=~/.cache/palito/kernels
  PALITO_MODEL      (unset; export it to skip --model)
Models are stored in ~/.local/share/palito/models (set PALITO_MODELS to use
another disk).

chat on CUDA uses the int8-fast profile (tensor-core prefill at llama.cpp
precision, F32 decode; falls back to f32-tiled below sm_80), the decode graph and
prefill chunks of up to 4096 tokens (capped by --context).
chat uses the model's trained context unless --context is given (CUDA: at most
65535, shrunk to what fits in free GPU memory minus 2 GiB). Replies run until end
of turn or a full context (reported); --tokens N caps a reply.
Replies stream token by token (--no-stream prints them when complete).
Decoding is greedy by default; --temperature T (with --top-p, --top-k, --seed)
samples instead, and --stop TEXT ends a reply at TEXT.
Use --cuda-matmul int8-prefill for the more precise tensor-core prefill,
--cuda-matmul f32-tiled for F32 prefill or --cuda-matmul f32 for the reference
profile.
Qwen3 reasoning is returned by serve as reasoning_content.

C example: cc examples/c/chat.c -Iinclude -Llib -lpalito -Wl,-rpath,"\$PWD/lib" -o chat

README
            ;;
        macos-arm64)
            cat <<README
palito ${version} (macOS, Apple Silicon)

Install:
  mkdir -p ~/.local/bin && ln -sf "\$PWD/palito" ~/.local/bin/palito
  (add ~/.local/bin to PATH in ~/.zshrc if 'palito' is not found)
If the archive was downloaded with a browser, clear the quarantine flag first:
  xattr -dr com.apple.quarantine .

README
            models_readme palito
            other_gpu_readme palito /path/to/model.gguf "~/.local/share/palito/models"
            ;;
        windows-x86_64)
            cat <<README
palito ${version} (Windows x86_64, Vulkan)

Requires a GPU driver with Vulkan for --backend vulkan (AMD, Intel and NVIDIA
drivers ship it). Run from PowerShell or cmd in this folder, or add the folder
to PATH.

README
            models_readme palito.exe
            other_gpu_readme palito.exe 'C:\models\model.gguf' '%USERPROFILE%\.local\share\palito\models'
            ;;
    esac
    library_readme
}

if [[ "$platform" == windows-x86_64 ]]; then
    readme | sed 's/$/\r/' > "$dist/README.txt"
    archive="$target/dist/$name.zip"
    rm -f "$archive"
    (cd "$target/dist" && "$python" -m zipfile -c "$name.zip" "$name")
else
    readme > "$dist/README.txt"
    archive="$target/dist/$name.tar.gz"
    tar -C "$target/dist" -czf "$archive" "$name"
fi
echo "bundle: $dist"
echo "archive: $archive"

if [[ $install -eq 1 ]]; then
    prefix="$HOME/.local/share/palito/$version"
    rm -rf "$prefix"
    mkdir -p "$(dirname "$prefix")" "$HOME/.local/bin"
    cp -R "$dist" "$prefix"
    ln -sfn "$prefix/palito" "$HOME/.local/bin/palito"
    echo "installed: $HOME/.local/bin/palito -> $prefix/palito"
fi
