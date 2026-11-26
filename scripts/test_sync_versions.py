import tempfile
import unittest
from pathlib import Path

from sync_versions import MANIFESTS, cargo_version, check, main, sync


CARGO = '[package]\nname = "palito"\nversion = "0.7.0"\n\n[dependencies]\nserde = { version = "1.0" }\n'
FILES = {
    "bindings/python/pyproject.toml": '[project]\nname = "palito"\nversion = "0.5.0"\nrequires-python = ">=3.9"\n',
    "bindings/node/package.json": '{\n  "name": "palito",\n  "version": "0.5.0",\n  "dependencies": { "koffi": "^3.3.8" }\n}\n',
    "bindings/csharp/Palito/Palito.csproj": "<Project>\n  <PropertyGroup>\n    <Version>0.5.0</Version>\n  </PropertyGroup>\n</Project>\n",
    "bindings/java/pom.xml": "<project>\n  <artifactId>palito</artifactId>\n  <version>0.5.0</version>\n"
                             "  <dependencies><dependency><version>5.11.4</version></dependency></dependencies>\n</project>\n",
    "bindings/java/README.md": "```xml\n<dependency>\n  <groupId>io.github.natanloterio</groupId>\n  <artifactId>palito</artifactId>\n"
                               "  <version>0.5.0</version>\n</dependency>\n```\n",
}


class SyncVersionsTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.root = Path(self.dir.name)
        (self.root / "Cargo.toml").write_text(CARGO)
        for path, text in FILES.items():
            (self.root / path).parent.mkdir(parents=True, exist_ok=True)
            (self.root / path).write_text(text)

    def tearDown(self):
        self.dir.cleanup()

    def test_every_manifest_is_covered(self):
        self.assertEqual(sorted(m.path for m in MANIFESTS), sorted(FILES))

    def test_reads_the_package_version(self):
        self.assertEqual(cargo_version(self.root), "0.7.0")

    def test_check_lists_stale_manifests(self):
        self.assertEqual(sorted(check(self.root, "0.7.0")), sorted(FILES))

    def test_sync_sets_only_the_project_version(self):
        sync(self.root, "0.7.0")
        self.assertEqual(check(self.root, "0.7.0"), [])
        pom = (self.root / "bindings/java/pom.xml").read_text()
        self.assertIn("<version>0.7.0</version>", pom)
        self.assertIn("<version>5.11.4</version>", pom)
        self.assertIn('"koffi": "^3.3.8"', (self.root / "bindings/node/package.json").read_text())

    def test_sync_is_idempotent(self):
        sync(self.root, "0.7.0")
        before = {p: (self.root / p).read_text() for p in FILES}
        sync(self.root, "0.7.0")
        self.assertEqual(before, {p: (self.root / p).read_text() for p in FILES})

    def test_missing_version_is_an_error(self):
        (self.root / "bindings/node/package.json").write_text('{ "name": "palito" }\n')
        with self.assertRaises(ValueError):
            check(self.root, "0.7.0")

    def test_check_with_tag(self):
        sync(self.root, "0.7.0")
        self.assertEqual(main(["--root", str(self.root), "--check", "--tag", "v0.7.0"]), 0)
        self.assertEqual(main(["--root", str(self.root), "--check", "--tag", "v0.8.0"]), 1)
        self.assertEqual(main(["--root", str(self.root), "--check", "--tag", "refs/tags/v0.7.0"]), 0)

    def test_check_fails_on_stale(self):
        self.assertEqual(main(["--root", str(self.root), "--check"]), 1)
        self.assertEqual(main(["--root", str(self.root)]), 0)
        self.assertEqual(main(["--root", str(self.root), "--check"]), 0)


if __name__ == "__main__":
    unittest.main()
