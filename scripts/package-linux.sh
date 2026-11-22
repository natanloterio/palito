#!/usr/bin/env bash
# Builds a self-contained Linux x86_64 bundle of the CUDA CLI:
#   target/dist/palito-<version>-linux-x86_64{/,.tar.gz}
# The bundle carries NVRTC 13 (runtime kernel compiler); the NVIDIA driver
# (libcuda) must come from the host. With --install it is copied to
# ~/.local/share/palito/<version> and linked as ~/.local/bin/palito.
#
# NVRTC_LIB_DIR overrides where libnvrtc.so.13 is taken from (default: the
# project's CUDA 13 benchmark venv).
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"
install=0
for arg in "$@"; do
    case "$arg" in
        --install) install=1 ;;
        *) echo "usage: $0 [--install]" >&2; exit 2 ;;
    esac
done

version="$(sed -n 's/^version = "\(.*\)"/\1/p' Cargo.toml | head -n1)"
name="palito-${version}-linux-x86_64"
nvrtc="${NVRTC_LIB_DIR:-$root/.venv-cuda-bench/lib/python3.11/site-packages/nvidia/cu13/lib}"
[[ -f "$nvrtc/libnvrtc.so.13" ]] || { echo "libnvrtc.so.13 not found in $nvrtc (set NVRTC_LIB_DIR)" >&2; exit 1; }

cargo build --release --features cuda

dist="$root/target/dist/$name"
rm -rf "$dist"
mkdir -p "$dist/bin" "$dist/lib"
install -m 755 target/release/palito "$dist/bin/palito"
strip "$dist/bin/palito" 2>/dev/null || true
cp -L "$nvrtc/libnvrtc.so.13" "$dist/lib/"
cp -L "$nvrtc"/libnvrtc-builtins.so.13* "$dist/lib/"
# cudarc probes the unversioned name first; keep it pointing at the bundled 13.x.
ln -s libnvrtc.so.13 "$dist/lib/libnvrtc.so"
# C API: library, header and examples (C and Python ctypes).
install -m 755 target/release/libpalito.so "$dist/lib/libpalito.so"
strip "$dist/lib/libpalito.so" 2>/dev/null || true
mkdir -p "$dist/include" "$dist/examples"
install -m 644 include/palito.h "$dist/include/palito.h"
cp -r examples/c examples/python "$dist/examples/"
find "$dist/examples" -name __pycache__ -prune -exec rm -rf {} +

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
chmod 755 "$dist/palito"

cat > "$dist/README.txt" <<README
palito ${version} (Linux x86_64, CUDA)

Requires an NVIDIA driver with CUDA 13 support. NVRTC 13 is bundled in lib/.

  ./palito chat --model /path/to/model.gguf
  ./palito --help

Defaults set by the wrapper (override with flags or by exporting them):
  PALITO_BACKEND=cuda
  PALITO_CUDA_KERNEL_CACHE=~/.cache/palito/kernels
  PALITO_MODEL      (unset; export it to skip --model)

chat on CUDA uses the int8-fast profile (tensor-core prefill at llama.cpp
precision, F32 decode; falls back to f32-tiled below sm_80), the decode graph and
prefill chunks of up to 4096 tokens (capped by --context).
chat uses the model's trained context unless --context is given (CUDA: at most
65535, shrunk to what fits in free GPU memory minus 2 GiB). Replies run until end
of turn or a full context (reported); --tokens N caps a reply.
Replies stream token by token (--no-stream prints them when complete).
Use --cuda-matmul int8-prefill for the more precise tensor-core prefill,
--cuda-matmul f32-tiled for F32 prefill or --cuda-matmul f32 for the reference
profile.

Library (embed Palito in your program, same process, no server):
  include/palito.h, lib/libpalito.so (needs lib/ on the library path for NVRTC)
  cc examples/c/chat.c -Iinclude -Llib -lpalito -o chat
  LD_LIBRARY_PATH=lib ./chat /path/to/model.gguf "Hello!"
  LD_LIBRARY_PATH=lib PALITO_LIBRARY=lib/libpalito.so \
    python3 examples/python/palito.py /path/to/model.gguf "Hello!"
README

tar -C "$root/target/dist" -czf "$root/target/dist/$name.tar.gz" "$name"
echo "bundle: $dist"
echo "archive: $root/target/dist/$name.tar.gz"

if [[ $install -eq 1 ]]; then
    prefix="$HOME/.local/share/palito/$version"
    rm -rf "$prefix"
    mkdir -p "$(dirname "$prefix")" "$HOME/.local/bin"
    cp -a "$dist" "$prefix"
    ln -sfn "$prefix/palito" "$HOME/.local/bin/palito"
    echo "installed: $HOME/.local/bin/palito -> $prefix/palito"
fi
