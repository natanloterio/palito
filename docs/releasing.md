# Releasing

One version for everything: `Cargo.toml` `[package] version`. The bindings follow it.

1. Bump `version` in `Cargo.toml`, then align the bindings:

   ```sh
   python3 scripts/sync_versions.py          # rewrites pyproject.toml, package.json, Palito.csproj, pom.xml, the Java README
   python3 scripts/sync_versions.py --check  # exit 1 if any is stale
   ```

   Also update the version in the README's download command (`palito-X.Y.Z-linux-x86_64.tar.gz`, three places).

2. Commit (`chore: release X.Y.Z`), tag `vX.Y.Z` and push the tag. `.github/workflows/release.yml` then publishes:

   | Registry | Package | Credential |
   |---|---|---|
   | crates.io | `palito` | `CARGO_REGISTRY_TOKEN` |
   | PyPI | `palito` | trusted publisher (workflow `release.yml`, environment `pypi`) |
   | npm | `palito` (maintained by the `gaiahub` org) | trusted publishing (workflow `release.yml`), with provenance |
   | NuGet | `Palito` | trusted publishing (policy `palito-release`: workflow `release.yml`, environment `nuget`) |
   | Maven Central | `io.github.natanloterio:palito` | `MAVEN_CENTRAL_USERNAME`, `MAVEN_CENTRAL_PASSWORD`, `GPG_PRIVATE_KEY`, `GPG_PASSPHRASE` |
   | Go | `github.com/natanloterio/palito/bindings/go` | none: the workflow pushes the tag `bindings/go/vX.Y.Z` |

   The workflow stops before uploading anything if a binding or the tag does not match `Cargo.toml`. The Go step runs only after the other registries succeed, and only while the repository is public (proxy.golang.org cannot fetch private modules).

3. Build the CUDA bundle (CLI with `convert`, plus NVRTC 13; it needs a local NVRTC, so it is not in CI). Once every registry job of the release run has succeeded, create the GitHub release with it; the README's download link points at this asset:

   ```sh
   NVRTC_LIB_DIR=/usr/local/cuda-13/lib64 scripts/package-linux.sh
   gh release create vX.Y.Z target/dist/palito-X.Y.Z-linux-x86_64.tar.gz --title "Palito X.Y.Z"
   ```

   Publishing the release starts the `cli-bundles` workflow, which builds the macOS (Apple Silicon, Metal) and Windows (x86_64, Vulkan) bundles with `scripts/package-cli.sh`, smoke-tests them (pull a small GGUF from Hugging Face, generate on the CPU) and attaches them to the release. To add them to an older release: Actions > cli-bundles > Run workflow with its tag. Update the version in the README's download links (they are not synced by `sync_versions.py`).

   Hosted macOS runners are sometimes cancelled for capacity; rerun the failed jobs on the same tag (`gh run rerun <run> --failed`). Maven Central can take 20 minutes or more to validate a deployment.

**Dry run:** Actions > release > Run workflow, with "dry_run" ticked (the default). Every package is built and checked (`cargo publish --dry-run`, `twine check`, `npm pack --dry-run`, `dotnet pack`, `mvn -P release verify`), and nothing is uploaded.

The crates.io package holds only the library, the CLI, the C header and the CUDA kernels (`include` in `Cargo.toml`). The bindings load `libpalito` at run time from `$PALITO_LIBRARY` or the installed bundle, so their packages do not include it.
