"""Tests for install.sh against a fake release served from file:// URLs."""

import hashlib
import io
import json
import os
import subprocess
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "install.sh"
VERSION = "9.8.7"
PLATFORM = "linux-x86_64"
NAME = f"palito-{VERSION}-{PLATFORM}"


def bundle() -> bytes:
    """A tar.gz with an executable `palito` that prints its version."""
    data = io.BytesIO()
    with tarfile.open(fileobj=data, mode="w:gz") as tar:
        script = f"#!/bin/sh\necho 'palito {VERSION}'\n".encode()
        info = tarfile.TarInfo(f"{NAME}/palito")
        info.size, info.mode = len(script), 0o755
        tar.addfile(info, io.BytesIO(script))
    return data.getvalue()


class InstallTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        base = Path(self.dir.name)
        self.home = base / "home"
        self.home.mkdir()
        self.releases = base / "releases"
        self.api = base / "api"
        archive = self.releases / "download" / f"v{VERSION}" / f"{NAME}.tar.gz"
        archive.parent.mkdir(parents=True)
        self.archive_bytes = bundle()
        archive.write_bytes(self.archive_bytes)
        self.write_api(hashlib.sha256(self.archive_bytes).hexdigest())
        # Stands in for /opt/homebrew/bin: writable, on PATH only when a test says so.
        self.shared = base / "brew-bin"
        self.shared.mkdir()

    def tearDown(self):
        self.dir.cleanup()

    def write_api(self, sha):
        release = self.api / "releases" / "tags" / f"v{VERSION}"
        release.parent.mkdir(parents=True, exist_ok=True)
        assets = [
            {"name": f"palito-{VERSION}-macos-arm64.tar.gz", "digest": "sha256:" + "1" * 64},
            {"name": f"{NAME}.tar.gz", "size": 1, "digest": f"sha256:{sha}"},
        ]
        release.write_text(json.dumps({"tag_name": f"v{VERSION}", "assets": assets}, indent=2))

    def run_install(self, shell="/bin/zsh", **env):
        base = {
            "HOME": str(self.home),
            "PATH": "/usr/bin:/bin",
            "SHELL": shell,
            "PALITO_VERSION": VERSION,
            "PALITO_PLATFORM": PLATFORM,
            "PALITO_RELEASES_URL": self.releases.as_uri(),
            "PALITO_API_URL": self.api.as_uri(),
            "PALITO_SHARED_BIN_DIRS": str(self.shared),
        }
        base.update(env)
        return subprocess.run(["sh", str(SCRIPT)], env=base, capture_output=True, text=True)

    def test_installs_links_and_adds_path_once(self):
        out = self.run_install()
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertIn("sha256 ok", out.stderr)
        link = self.home / ".local" / "bin" / "palito"
        self.assertEqual(link.resolve(), (self.home / ".local/share/palito" / VERSION / "palito").resolve())
        run = subprocess.run([str(link)], capture_output=True, text=True)
        self.assertEqual(run.stdout.strip(), f"palito {VERSION}")
        self.assertIn("palito pull qwen3-8b", out.stderr)

        again = self.run_install()
        self.assertEqual(again.returncode, 0, again.stderr)
        zshrc = (self.home / ".zshrc").read_text()
        self.assertEqual(zshrc.count("added by the palito installer"), 1)
        self.assertIn(f'export PATH="{self.home}/.local/bin:$PATH"', zshrc)

    def test_not_on_path_says_how_to_use_it_in_this_terminal(self):
        out = self.run_install()
        self.assertEqual(out.returncode, 0, out.stderr)
        last = out.stderr.strip().splitlines()[-1]
        self.assertEqual(last.strip(), f'export PATH="{self.home}/.local/bin:$PATH"')

    def test_prefers_a_writable_directory_already_on_path(self):
        out = self.run_install(PATH=f"{self.shared}:/usr/bin:/bin")
        self.assertEqual(out.returncode, 0, out.stderr)
        link = self.shared / "palito"
        self.assertEqual(link.resolve(), (self.home / ".local/share/palito" / VERSION / "palito").resolve())
        self.assertFalse((self.home / ".zshrc").exists(), "no profile change needed")
        self.assertNotIn("does not see palito", out.stderr)
        # Works in the same shell right away.
        run = subprocess.run(["sh", "-c", "palito"], env={"PATH": f"{self.shared}:/usr/bin:/bin"},
                             capture_output=True, text=True)
        self.assertEqual(run.stdout.strip(), f"palito {VERSION}")

    def test_does_not_replace_another_programs_palito(self):
        (self.shared / "palito").write_text("#!/bin/sh\necho other\n")
        out = self.run_install(PATH=f"{self.shared}:/usr/bin:/bin")
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertEqual((self.shared / "palito").read_text(), "#!/bin/sh\necho other\n")
        self.assertTrue((self.home / ".local/bin/palito").is_symlink())

    def test_upgrade_moves_an_older_local_bin_link_too(self):
        self.assertEqual(self.run_install().returncode, 0)  # ~/.local/bin, profile line
        old_target = self.home / ".local/share/palito/1.0.0/palito"
        old_target.parent.mkdir(parents=True)
        old_target.write_text("old")
        local = self.home / ".local/bin/palito"
        local.unlink()
        local.symlink_to(old_target)
        out = self.run_install(PATH=f"{self.shared}:/usr/bin:/bin")
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertEqual(local.resolve(), (self.shared / "palito").resolve())

    def test_bash_and_fish_profiles(self):
        self.assertEqual(self.run_install(shell="/bin/bash").returncode, 0)
        # macOS login shells read ~/.bash_profile, not ~/.bashrc.
        profile = ".bash_profile" if sys.platform == "darwin" else ".bashrc"
        self.assertIn(".local/bin", (self.home / profile).read_text())
        self.assertEqual(self.run_install(shell="/usr/bin/fish").returncode, 0)
        self.assertIn("fish_add_path", (self.home / ".config/fish/conf.d/palito.fish").read_text())

    def test_path_already_set_or_opted_out_leaves_profiles_alone(self):
        bin_dir = self.home / ".local" / "bin"
        out = self.run_install(PATH=f"{bin_dir}:/usr/bin:/bin")
        self.assertEqual(out.returncode, 0, out.stderr)
        out = self.run_install(PALITO_NO_MODIFY_PATH="1")
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertFalse((self.home / ".zshrc").exists())
        self.assertIn("Add", out.stderr)

    def test_checksum_mismatch_installs_nothing(self):
        self.write_api("0" * 64)
        out = self.run_install()
        self.assertEqual(out.returncode, 1)
        self.assertIn("checksum mismatch", out.stderr)
        self.assertFalse((self.home / ".local").exists())

    def test_missing_checksum_warns_and_installs(self):
        (self.api / "releases" / "tags" / f"v{VERSION}").unlink()
        out = self.run_install()
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertIn("skipping verification", out.stderr)

    def test_missing_release_fails_clearly(self):
        out = self.run_install(PALITO_VERSION="1.0.0")
        self.assertEqual(out.returncode, 1)
        self.assertIn("download failed", out.stderr)

    def test_custom_directories(self):
        out = self.run_install(PALITO_INSTALL_DIR=str(self.home / "opt"), PALITO_BIN_DIR=str(self.home / "bin"))
        self.assertEqual(out.returncode, 0, out.stderr)
        # resolve() both sides: macOS temp dirs live under /var -> /private/var.
        self.assertTrue((self.home / "bin" / "palito").resolve().is_relative_to((self.home / "opt").resolve()))

    def test_platform_detection_rejects_unknown_systems(self):
        env = {k: v for k, v in os.environ.items() if k != "PALITO_PLATFORM"}
        fake_bin = Path(self.dir.name) / "fakebin"
        fake_bin.mkdir()
        (fake_bin / "uname").write_text('#!/bin/sh\n[ "$1" = -s ] && echo FreeBSD || echo amd64\n')
        (fake_bin / "uname").chmod(0o755)
        out = subprocess.run(["sh", str(SCRIPT)], capture_output=True, text=True,
                             env={**env, "HOME": str(self.home), "PATH": f"{fake_bin}:/usr/bin:/bin",
                                  "PALITO_VERSION": VERSION})
        self.assertEqual(out.returncode, 1)
        self.assertIn("no prebuilt bundle for FreeBSD amd64", out.stderr)


if __name__ == "__main__":
    unittest.main()
