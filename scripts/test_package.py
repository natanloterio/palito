"""Tests for scripts/package.sh with fake build outputs (--no-build)."""

import subprocess
import tarfile
import tempfile
import unittest
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "package.sh"


def cargo_version():
    for line in (ROOT / "Cargo.toml").read_text().splitlines():
        if line.startswith("version = "):
            return line.split('"')[1]
    raise AssertionError("no version in Cargo.toml")


class PackageTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.target = Path(self.dir.name) / "target"
        (self.target / "release").mkdir(parents=True)
        self.nvrtc = Path(self.dir.name) / "nvrtc"
        self.nvrtc.mkdir()
        for lib in ["libnvrtc.so.13", "libnvrtc-builtins.so.13.0"]:
            (self.nvrtc / lib).write_bytes(b"fake")
        self.home = Path(self.dir.name) / "home"
        self.version = cargo_version()

    def tearDown(self):
        self.dir.cleanup()

    def run_script(self, *args):
        env = {"CARGO_TARGET_DIR": str(self.target), "NVRTC_LIB_DIR": str(self.nvrtc),
               "HOME": str(self.home), "PATH": "/usr/bin:/bin"}
        return subprocess.run(["bash", str(SCRIPT), *args], env=env, capture_output=True, text=True)

    def fake_build(self, *names):
        for name in names:
            (self.target / "release" / name).write_bytes(b"fake " + name.encode())

    def tar_members(self, name):
        with tarfile.open(self.target / "dist" / f"{name}.tar.gz") as tar:
            return {m.name: m for m in tar.getmembers()}

    def assert_common(self, name, paths, lib):
        for path in ["include/palito.h", "examples/c/chat.c", "bindings/python/pyproject.toml",
                     "README.txt", "LICENSE", f"lib/{lib}"]:
            self.assertIn(f"{name}/{path}", paths)
        self.assertFalse([p for p in paths if "node_modules" in p or "__pycache__" in p])

    def test_linux_bundle_has_wrapper_binary_and_nvrtc(self):
        self.fake_build("palito", "libpalito.so")
        out = self.run_script("linux-x86_64", "--no-build")
        self.assertEqual(out.returncode, 0, out.stderr)
        name = f"palito-{self.version}-linux-x86_64"
        members = self.tar_members(name)
        self.assert_common(name, members, "libpalito.so")
        for path in ["palito", "bin/palito", "lib/libnvrtc.so.13", "lib/libnvrtc-builtins.so.13.0"]:
            self.assertIn(f"{name}/{path}", members)
        self.assertTrue(members[f"{name}/lib/libnvrtc.so"].issym())
        wrapper = (self.target / "dist" / name / "palito").read_text()
        self.assertIn('PALITO_BACKEND="${PALITO_BACKEND:-cuda}"', wrapper)
        self.assertTrue(members[f"{name}/palito"].mode & 0o111)
        readme = (self.target / "dist" / name / "README.txt").read_text()
        self.assertIn(f"palito {self.version} (Linux x86_64, CUDA)", readme)
        self.assertIn("./palito pull qwen3-8b", readme)

    def test_linux_needs_nvrtc(self):
        self.fake_build("palito", "libpalito.so")
        for lib in self.nvrtc.iterdir():
            lib.unlink()
        out = self.run_script("linux-x86_64", "--no-build")
        self.assertEqual(out.returncode, 1)
        self.assertIn("set NVRTC_LIB_DIR", out.stderr)

    def test_macos_bundle_layout_and_install(self):
        self.fake_build("palito", "libpalito.dylib")
        out = self.run_script("macos-arm64", "--no-build", "--install")
        self.assertEqual(out.returncode, 0, out.stderr)
        name = f"palito-{self.version}-macos-arm64"
        members = self.tar_members(name)
        self.assert_common(name, members, "libpalito.dylib")
        self.assertTrue(members[f"{name}/palito"].mode & 0o111, "command is executable")
        self.assertNotIn(f"{name}/bin/palito", members, "no wrapper on macOS")
        readme = (self.target / "dist" / name / "README.txt").read_text()
        self.assertIn(f"palito {self.version} (macOS, Apple Silicon)", readme)
        self.assertIn("--backend metal", readme)
        link = self.home / ".local" / "bin" / "palito"
        # resolve() both sides: macOS temp dirs live under /var -> /private/var.
        self.assertEqual(link.resolve(), (self.home / ".local/share/palito" / self.version / "palito").resolve())

    def test_windows_bundle_layout(self):
        self.fake_build("palito.exe", "palito.dll")
        out = self.run_script("windows-x86_64", "--no-build")
        self.assertEqual(out.returncode, 0, out.stderr)
        name = f"palito-{self.version}-windows-x86_64"
        with zipfile.ZipFile(self.target / "dist" / f"{name}.zip") as archive:
            names = set(archive.namelist())
        self.assert_common(name, names, "palito.dll")
        self.assertIn(f"{name}/palito.exe", names)
        readme = (self.target / "dist" / name / "README.txt").read_bytes()
        self.assertIn(b"--backend vulkan", readme)
        self.assertIn(b"\r\n", readme, "Notepad-friendly line endings")

    def test_windows_has_no_install(self):
        out = self.run_script("windows-x86_64", "--no-build", "--install")
        self.assertEqual(out.returncode, 2)
        self.assertIn("add the bundle folder to PATH", out.stderr)

    def test_missing_build_output_names_the_build_command(self):
        out = self.run_script("macos-arm64", "--no-build")
        self.assertEqual(out.returncode, 1)
        self.assertIn("cargo build --release --lib --bin palito --features metal", out.stderr)

    def test_bad_arguments_are_rejected(self):
        for args in [(), ("solaris",), ("macos-arm64", "linux-x86_64"), ("macos-arm64", "--fast")]:
            out = self.run_script(*args)
            self.assertEqual(out.returncode, 2, args)
            self.assertIn("usage", out.stderr)


if __name__ == "__main__":
    unittest.main()
