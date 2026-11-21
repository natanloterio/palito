#!/usr/bin/env bash
# Builds a self-contained Linux x86_64 bundle of the CUDA CLI:
#   target/dist/local-inference-<version>-linux-x86_64{/,.tar.gz}
# The bundle carries NVRTC 13 (runtime kernel compiler); the NVIDIA driver
# (libcuda) must come from the host. With --install it is copied to
# ~/.local/share/local-inference/<version> and linked as ~/.local/bin/local-inference.
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
name="local-inference-${version}-linux-x86_64"
nvrtc="${NVRTC_LIB_DIR:-$root/.venv-cuda-bench/lib/python3.11/site-packages/nvidia/cu13/lib}"
[[ -f "$nvrtc/libnvrtc.so.13" ]] || { echo "libnvrtc.so.13 not found in $nvrtc (set NVRTC_LIB_DIR)" >&2; exit 1; }

cargo build --release --features cuda

dist="$root/target/dist/$name"
rm -rf "$dist"
mkdir -p "$dist/bin" "$dist/lib"
install -m 755 target/release/local-inference "$dist/bin/local-inference"
strip "$dist/bin/local-inference" 2>/dev/null || true
cp -L "$nvrtc/libnvrtc.so.13" "$dist/lib/"
cp -L "$nvrtc"/libnvrtc-builtins.so.13* "$dist/lib/"
# cudarc probes the unversioned name first; keep it pointing at the bundled 13.x.
ln -s libnvrtc.so.13 "$dist/lib/libnvrtc.so"

cat > "$dist/local-inference" <<'WRAPPER'
#!/usr/bin/env bash
# Runs the bundled CLI with its NVRTC, the CUDA backend and a persistent kernel
# cache by default. Every default can be overridden by flag or environment.
set -euo pipefail
here="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)"
export LD_LIBRARY_PATH="$here/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export LOCAL_INFERENCE_BACKEND="${LOCAL_INFERENCE_BACKEND:-cuda}"
cache="${LOCAL_INFERENCE_CUDA_KERNEL_CACHE:-${XDG_CACHE_HOME:-$HOME/.cache}/local-inference/kernels}"
mkdir -p -m 700 "$cache"
export LOCAL_INFERENCE_CUDA_KERNEL_CACHE="$cache"
exec "$here/bin/local-inference" "$@"
WRAPPER
chmod 755 "$dist/local-inference"

cat > "$dist/README.txt" <<README
local-inference ${version} (Linux x86_64, CUDA)

Requires an NVIDIA driver with CUDA 13 support. NVRTC 13 is bundled in lib/.

  ./local-inference chat --model /path/to/model.gguf --context 4096
  ./local-inference --help

Defaults set by the wrapper (override with flags or by exporting them):
  LOCAL_INFERENCE_BACKEND=cuda
  LOCAL_INFERENCE_CUDA_KERNEL_CACHE=~/.cache/local-inference/kernels
  LOCAL_INFERENCE_MODEL      (unset; export it to skip --model)

chat on CUDA uses the int8-prefill profile (tensor-core prefill, F32 decode; falls
back to f32-tiled below sm_80), the decode graph and prefill chunks of up to 4096
tokens (capped by --context).
Use --cuda-matmul f32-tiled for F32 prefill or --cuda-matmul f32 for the
reference profile.
README

tar -C "$root/target/dist" -czf "$root/target/dist/$name.tar.gz" "$name"
echo "bundle: $dist"
echo "archive: $root/target/dist/$name.tar.gz"

if [[ $install -eq 1 ]]; then
    prefix="$HOME/.local/share/local-inference/$version"
    rm -rf "$prefix"
    mkdir -p "$(dirname "$prefix")" "$HOME/.local/bin"
    cp -a "$dist" "$prefix"
    ln -sfn "$prefix/local-inference" "$HOME/.local/bin/local-inference"
    echo "installed: $HOME/.local/bin/local-inference -> $prefix/local-inference"
fi
