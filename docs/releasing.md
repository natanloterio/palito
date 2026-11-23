# Releasing

One version for everything: `Cargo.toml` `[package] version`. The bindings follow it.

1. Bump `version` in `Cargo.toml`, then align the bindings:

   ```sh
   python3 scripts/sync_versions.py          # rewrites pyproject.toml, package.json, Palito.csproj, pom.xml
   python3 scripts/sync_versions.py --check  # exit 1 if any is stale
   ```

2. Commit (`chore: release X.Y.Z`), tag `vX.Y.Z` and push the tag. `.github/workflows/release.yml` then publishes:

   | Registry | Package | Credential |
   |---|---|---|
   | crates.io | `palito` | `CARGO_REGISTRY_TOKEN` |
   | PyPI | `palito` | trusted publisher (workflow `release.yml`, environment `pypi`) |
   | npm | `palito` | `NPM_TOKEN` |
   | NuGet | `Palito` | `NUGET_API_KEY` |
   | Maven Central | `io.github.natanloterio:palito` | `MAVEN_CENTRAL_USERNAME`, `MAVEN_CENTRAL_PASSWORD`, `GPG_PRIVATE_KEY`, `GPG_PASSPHRASE` |
   | Go | `github.com/natanloterio/palito/bindings/go` | none: the workflow pushes the tag `bindings/go/vX.Y.Z` |

   The workflow stops before uploading anything if a binding or the tag does not match `Cargo.toml`. The Go step runs only after the other registries succeed, and only while the repository is public (proxy.golang.org cannot fetch private modules).

3. Build the CUDA bundle and attach it to the GitHub release (it needs a local NVRTC 13, so it is not in CI):

   ```sh
   scripts/package-linux.sh
   gh release create vX.Y.Z target/dist/palito-X.Y.Z-linux-x86_64.tar.gz --title "Palito X.Y.Z"
   ```

**Dry run:** Actions > release > Run workflow, with "dry_run" ticked (the default). Every package is built and checked (`cargo publish --dry-run`, `twine check`, `npm publish --dry-run`, `dotnet pack`, `mvn -P release verify`), and nothing is uploaded.

The crates.io package holds only the library, the CLI, the C header and the CUDA kernels (`include` in `Cargo.toml`). The bindings load `libpalito` at run time from `$PALITO_LIBRARY` or the installed bundle, so their packages do not include it.
