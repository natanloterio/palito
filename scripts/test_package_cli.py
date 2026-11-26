"""Tests for scripts/package-cli.sh with fake build outputs (no cargo build)."""

import subprocess
import tarfile
import tempfile
import unittest
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "package-cli.sh"


def cargo_version():
    for line in (ROOT / "Cargo.toml").read_text().splitlines():
        if line.startswith("version = "):
            return line.split('"')[1]
    raise AssertionError("no version in Cargo.toml")


class PackageCliTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.target = Path(self.dir.name)
        (self.target / "release").mkdir()
        self.version = cargo_version()

    def tearDown(self):
        self.dir.cleanup()

    def run_script(self, *args):
        return subprocess.run(
            ["bash", str(SCRIPT), *args],
            env={"CARGO_TARGET_DIR": str(self.target), "PATH": "/usr/bin:/bin"},
            capture_output=True,
            text=True,
        )

    def fake_build(self, *names):
        for name in names:
            (self.target / "release" / name).write_bytes(b"fake " + name.encode())

    def test_macos_bundle_layout(self):
        self.fake_build("palito", "libpalito.dylib")
        out = self.run_script("macos-arm64")
        self.assertEqual(out.returncode, 0, out.stderr)
        name = f"palito-{self.version}-macos-arm64"
        archive = self.target / "dist" / f"{name}.tar.gz"
        with tarfile.open(archive) as tar:
            members = {m.name: m for m in tar.getmembers()}
        for path in ["palito", "lib/libpalito.dylib", "include/palito.h", "examples/c/chat.c",
                     "bindings/python/pyproject.toml", "README.txt", "LICENSE"]:
            self.assertIn(f"{name}/{path}", members)
        self.assertTrue(members[f"{name}/palito"].mode & 0o111, "command is executable")
        self.assertFalse([m for m in members if "node_modules" in m or "__pycache__" in m])
        readme = (self.target / "dist" / name / "README.txt").read_text()
        self.assertIn(f"palito {self.version} (macOS, Apple Silicon)", readme)
        self.assertIn("--backend metal", readme)
        self.assertIn("palito pull qwen3-8b", readme)

    def test_windows_bundle_layout(self):
        self.fake_build("palito.exe", "palito.dll")
        out = self.run_script("windows-x86_64")
        self.assertEqual(out.returncode, 0, out.stderr)
        name = f"palito-{self.version}-windows-x86_64"
        with zipfile.ZipFile(self.target / "dist" / f"{name}.zip") as archive:
            names = set(archive.namelist())
        for path in ["palito.exe", "lib/palito.dll", "include/palito.h", "README.txt", "LICENSE"]:
            self.assertIn(f"{name}/{path}", names)
        readme = (self.target / "dist" / name / "README.txt").read_bytes()
        self.assertIn(b"--backend vulkan", readme)
        self.assertIn(b"\r\n", readme, "Notepad-friendly line endings")

    def test_missing_build_output_fails_with_the_build_command(self):
        out = self.run_script("macos-arm64")
        self.assertNotEqual(out.returncode, 0)
        self.assertIn("cargo build --release --lib --bin palito --features metal", out.stderr)

    def test_unknown_platform_is_rejected(self):
        out = self.run_script("solaris")
        self.assertEqual(out.returncode, 2)
        self.assertIn("usage", out.stderr)


if __name__ == "__main__":
    unittest.main()
