"""Build an isolated pinned F32-activation correctness oracle, never a speed baseline."""
import argparse
import json
from pathlib import Path
import subprocess

from benchmark import ROOT, digest
from benchmark_moe import write_new

REVISION = "6b01a4767f8d3d0c416c410a49861333965c0982"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--f32-gelu", action="store_true",
                        help="Build a separate oracle without the F16 GELU lookup table")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    variant = "llama-f32-gelu-oracle" if args.f32_gelu else "llama-f32-oracle"
    source = ROOT / f"target/{variant}-source"
    build = ROOT / f"target/{variant}"
    patches = [ROOT / "native/f32-oracle.patch"]
    if args.f32_gelu:
        patches.append(ROOT / "native/f32-gelu-oracle.patch")
    commands = []

    def run(label, command):
        command = list(map(str, command))
        result = subprocess.run(command, capture_output=True, text=True, timeout=600)
        write_new(output / f"{label}.json", {
            "command": command, "returncode": result.returncode,
            "stdout": result.stdout, "stderr": result.stderr})
        commands.append(command)
        result.check_returncode()
        return result.stdout

    if not source.exists():
        run("worktree", ["git", "-C", ROOT.parent / "llama.cpp", "worktree",
                         "add", "--detach", source, REVISION])
    revision = run("revision", ["git", "-C", source, "rev-parse", "HEAD"]).strip()
    if revision != REVISION:
        raise ValueError(f"unexpected oracle source revision: {revision}")
    diff_command = ["git", "-C", source, "--no-pager", "diff", "--no-ext-diff", "--no-color", "HEAD"]
    changes = run("source-diff", diff_command)
    if not changes:
        run("patch-check", ["git", "-C", source, "apply", "--check", *patches])
        run("patch", ["git", "-C", source, "apply", *patches])
    elif changes != "".join(patch.read_text() for patch in patches):
        raise ValueError("oracle worktree has unexpected changes; refusing to overwrite")
    run("configure", ["cmake", "-S", source, "-B", build,
                     "-DCMAKE_BUILD_TYPE=Release", "-DGGML_CUDA=OFF", "-DGGML_NATIVE=ON",
                     "-DGGML_OPENMP=OFF", "-DGGML_BLAS=OFF", "-DGGML_LLAMAFILE=OFF",
                     "-DGGML_CPU_REPACK=OFF", "-DLLAMA_BUILD_TESTS=OFF",
                     "-DLLAMA_BUILD_EXAMPLES=OFF", "-DLLAMA_BUILD_TOOLS=OFF",
                     *(["-DLLAMA_BUILD_COMMON=OFF"] if args.f32_gelu else []),
                     f"-DGGML_F32_REFERENCE_SOURCE={ROOT / 'native/f32_oracle.c'}"])
    run("build", ["cmake", "--build", build, "--target", "llama", "-j", "4"])
    binaries = build / "bin"
    for name, file, libraries in [
        ("logits-oracle", "logits_oracle.cpp", ["-lllama", "-lggml", "-lggml-base"]),
        ("test-f32-oracle", "test_f32_oracle.cpp", ["-lggml-cpu", "-lggml-base"]),
    ]:
        run(name, ["g++", "-O3", "-std=c++17", f"-I{source / 'include'}",
                   f"-I{source / 'ggml/include'}", ROOT / "native" / file,
                   f"-L{binaries}", "-Wl,-rpath,$ORIGIN", *libraries,
                   "-o", binaries / name])
    tests = json.loads(run("graph-tests", [binaries / "test-f32-oracle"]))
    if tests["passed"] != 48:
        raise ValueError("incomplete external graph tests")
    files = [*patches, ROOT / "native/f32_oracle.c", ROOT / "native/logits_oracle.cpp",
             ROOT / "native/test_f32_oracle.cpp", Path(__file__),
             *sorted({p.resolve() for p in binaries.glob("*.so*")}),
             binaries / "logits-oracle", binaries / "test-f32-oracle"]
    write_new(output / "manifest.json", {
        "revision": REVISION, "purpose": __doc__, "commands": commands, "f32_gelu": args.f32_gelu,
        "graph_tests": tests, "sha256": {str(p): digest(p) for p in files}})
    print(json.dumps({"reference": str(binaries / "logits-oracle"), "graph_tests": tests}))


if __name__ == "__main__":
    main()
