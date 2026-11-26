#!/bin/sh
# Installs the palito CLI from the latest GitHub release (Linux x86_64 with
# CUDA, macOS Apple Silicon with Metal):
#
#   curl -fsSL https://raw.githubusercontent.com/natanloterio/palito/master/install.sh | sh
#
# The bundle goes to ~/.local/share/palito/<version>, the command is linked as
# ~/.local/bin/palito and that directory is added to PATH in your shell profile.
#
# Environment:
#   PALITO_VERSION          version to install (e.g. 0.16.2; default: latest)
#   PALITO_INSTALL_DIR      where bundles go (default ~/.local/share/palito)
#   PALITO_BIN_DIR          where the command is linked (default ~/.local/bin)
#   PALITO_NO_MODIFY_PATH=1 do not touch shell profiles
# Windows: irm https://raw.githubusercontent.com/natanloterio/palito/master/install.ps1 | iex
set -eu

REPO="natanloterio/palito"
RELEASES="${PALITO_RELEASES_URL:-https://github.com/$REPO/releases}"
API="${PALITO_API_URL:-https://api.github.com/repos/$REPO}"

say() { printf '%s\n' "$*" >&2; }
fail() { say "error: $*"; exit 1; }

download() { # url output
    if command -v curl >/dev/null 2>&1; then
        curl -fsSL --retry 3 -o "$2" "$1"
    elif command -v wget >/dev/null 2>&1; then
        wget -q -O "$2" "$1"
    else
        fail "curl or wget is required"
    fi
}

fetch_text() { # url
    if command -v curl >/dev/null 2>&1; then
        curl -fsSL --retry 3 "$1"
    else
        wget -q -O - "$1"
    fi
}

detect_platform() {
    if [ -n "${PALITO_PLATFORM:-}" ]; then
        echo "$PALITO_PLATFORM"
        return
    fi
    os="$(uname -s)"
    arch="$(uname -m)"
    case "$os/$arch" in
        Linux/x86_64 | Linux/amd64) echo linux-x86_64 ;;
        Darwin/arm64 | Darwin/aarch64) echo macos-arm64 ;;
        Darwin/x86_64)
            # Rosetta shells report x86_64 on Apple Silicon.
            if [ "$(sysctl -n hw.optional.arm64 2>/dev/null || echo 0)" = 1 ]; then
                echo macos-arm64
            else
                fail "Intel Macs have no prebuilt bundle; install with: cargo install palito"
            fi
            ;;
        *) fail "no prebuilt bundle for $os $arch; install with: cargo install palito --features vulkan" ;;
    esac
}

latest_version() {
    # The /latest page redirects to /tag/vX.Y.Z; no API call (no rate limit).
    if command -v curl >/dev/null 2>&1; then
        url="$(curl -fsSLI -o /dev/null -w '%{url_effective}' "$RELEASES/latest")"
    else
        url="$(wget -S --spider "$RELEASES/latest" 2>&1 | sed -n 's/^ *Location: *//p' | tail -n1)"
    fi
    version="${url##*/v}"
    case "$version" in
        [0-9]*.[0-9]*.[0-9]*) echo "$version" ;;
        *) fail "could not find the latest release at $RELEASES/latest" ;;
    esac
}

sha256() {
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum "$1" | cut -d' ' -f1
    else
        shasum -a 256 "$1" | cut -d' ' -f1
    fi
}

# Prints the sha256 GitHub recorded for the asset, or nothing if unavailable.
expected_sha256() { # version file
    fetch_text "$API/releases/tags/v$1" 2>/dev/null | tr ',{}' '\n\n\n' | awk -v file="$2" '
        /"name"[ ]*:/ { found = index($0, "\"" file "\"") > 0 }
        found && /"digest"[ ]*:[ ]*"sha256:/ { sub(/.*sha256:/, ""); sub(/".*/, ""); print; exit }'
}

add_to_path() { # bin_dir
    case ":$PATH:" in *":$1:"*) return ;; esac
    if [ "${PALITO_NO_MODIFY_PATH:-0}" = 1 ]; then
        say "Add $1 to PATH to run palito from anywhere."
        return
    fi
    line="export PATH=\"$1:\$PATH\""
    case "$(basename "${SHELL:-sh}")" in
        zsh) profile="$HOME/.zshrc" ;;
        bash) profile="$HOME/.bashrc"; [ "$(uname -s)" = Darwin ] && profile="$HOME/.bash_profile" ;;
        fish)
            mkdir -p "$HOME/.config/fish/conf.d"
            echo "fish_add_path $1" > "$HOME/.config/fish/conf.d/palito.fish"
            say "Added $1 to PATH in ~/.config/fish/conf.d/palito.fish (open a new terminal)"
            return
            ;;
        *) profile="$HOME/.profile" ;;
    esac
    if ! grep -qs "# added by the palito installer" "$profile"; then
        printf '\n%s  # added by the palito installer\n' "$line" >> "$profile"
    fi
    say "Added $1 to PATH in $profile (open a new terminal, or run: $line)"
}

main() {
    platform="$(detect_platform)"
    version="${PALITO_VERSION:-$(latest_version)}"
    version="${version#v}"
    name="palito-$version-$platform"
    file="$name.tar.gz"
    install_dir="${PALITO_INSTALL_DIR:-$HOME/.local/share/palito}"
    bin_dir="${PALITO_BIN_DIR:-$HOME/.local/bin}"

    tmp="$(mktemp -d)"
    trap 'rm -rf "$tmp"' EXIT INT TERM

    say "Downloading palito $version ($platform)..."
    download "$RELEASES/download/v$version/$file" "$tmp/$file" \
        || fail "download failed: $RELEASES/download/v$version/$file"

    expected="$(expected_sha256 "$version" "$file" || true)"
    actual="$(sha256 "$tmp/$file")"
    if [ -z "$expected" ]; then
        say "warning: could not read the release checksum from GitHub; skipping verification"
    elif [ "$expected" != "$actual" ]; then
        fail "checksum mismatch for $file (expected $expected, got $actual)"
    else
        say "sha256 ok"
    fi

    tar -xzf "$tmp/$file" -C "$tmp"
    [ -x "$tmp/$name/palito" ] || fail "unexpected archive layout in $file"
    mkdir -p "$install_dir" "$bin_dir"
    rm -rf "${install_dir:?}/$version"
    mv "$tmp/$name" "$install_dir/$version"
    ln -sfn "$install_dir/$version/palito" "$bin_dir/palito"

    say "Installed palito $version: $bin_dir/palito -> $install_dir/$version/palito"
    add_to_path "$bin_dir"
    say ""
    say "Get started:"
    say "  palito pull qwen3-8b"
    say "  palito chat --model qwen3-8b"
}

main "$@"
