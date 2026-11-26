# Building from source

The [Linux bundle](../README.md#download) is the quickest way to run Palito on Linux. Build from source on macOS, Windows, AMD or Intel GPUs, or to work on Palito itself.

You need a recent stable Rust toolchain ([rustup](https://rustup.rs)).

## Install with Cargo

```sh
cargo install palito                      # CPU only
cargo install palito --features cuda      # NVIDIA
cargo install palito --features vulkan    # AMD, Intel or NVIDIA through Vulkan
cargo install palito --features metal     # Apple Silicon
```

Add `qtip-research` to the features to get `palito convert` (GGUF to `.palito`), for example `--features cuda,qtip-research`.

### CUDA requirements

The CUDA backend compiles its kernels at run time with NVRTC, so the machine needs the NVIDIA driver and the CUDA 13 toolkit. If NVRTC is not on the default library path, point Palito at it:

```sh
export NVRTC_LIB_DIR=/usr/local/cuda-13/lib64
export LD_LIBRARY_PATH="$NVRTC_LIB_DIR:$LD_LIBRARY_PATH"
```

The Linux bundle carries NVRTC, so it only needs the driver.

## Build from a clone

```sh
git clone https://github.com/natanloterio/palito
cd palito
cargo build --release --features cuda,qtip-research --bin palito
./target/release/palito --version
```

`palito --version` lists the backends and features the binary was built with.

| Feature | Adds |
| :--- | :--- |
| `cuda` | NVIDIA backend |
| `vulkan` | Vulkan backend |
| `metal` | Metal backend (macOS) |
| `qtip-research` | `palito convert` and the `.palito` quantizer |
| `ktransformers` | `palito remote`, a client for a KTransformers server |

## Build the Linux bundle

`scripts/package-linux.sh` builds the CUDA CLI with `convert` and packs it with NVRTC into `target/dist/palito-<version>-linux-x86_64.tar.gz`. With `--install` it also links it as `~/.local/bin/palito`:

```sh
NVRTC_LIB_DIR=/usr/local/cuda-13/lib64 scripts/package-linux.sh --install
```

## Run the tests

```sh
cargo test                                   # CPU: unit, CLI and fixture tests
cargo test --features cuda,qtip-research     # also the CUDA tests
```

Tests that need real models are `#[ignore]`d and say which file they expect, for example `models/SmolLM2-135M-Instruct-Q4_K_M.gguf`. Run them with `-- --ignored`. The [technical reference](technical-reference.md) lists the model-specific validation runs.

## Language bindings

The bindings load the `libpalito` shared library at run time. Any release build also produces it (`target/release/libpalito.so`, `.dylib` on macOS, `palito.dll` on Windows):

```sh
cargo build --release --features cuda --lib
export PALITO_LIBRARY="$PWD/target/release/libpalito.so"
```

[Bindings](../bindings/README.md) explains how each binding finds the library, and [releasing](releasing.md) how versions and packages are published.
