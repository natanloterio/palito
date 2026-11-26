# Installs the palito CLI (Windows x86_64, Vulkan) from the latest GitHub release:
#
#   irm https://raw.githubusercontent.com/natanloterio/palito/master/install.ps1 | iex
#
# The bundle goes to %LOCALAPPDATA%\palito\<version> and that folder is added
# to your user PATH (a new terminal picks it up).
#
# Environment:
#   PALITO_VERSION          version to install (e.g. 0.16.2; default: latest)
#   PALITO_INSTALL_DIR      where bundles go (default %LOCALAPPDATA%\palito)
#   PALITO_NO_MODIFY_PATH=1 do not change the user PATH
# Linux and macOS: curl -fsSL https://raw.githubusercontent.com/natanloterio/palito/master/install.sh | sh

$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'  # Invoke-WebRequest is much faster without the progress bar
# Windows PowerShell 5.1 on older systems defaults to TLS 1.0, which GitHub refuses.
[Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12

function Install-Palito {
    $repo = 'natanloterio/palito'
    $releases = if ($env:PALITO_RELEASES_URL) { $env:PALITO_RELEASES_URL } else { "https://github.com/$repo/releases" }
    $api = if ($env:PALITO_API_URL) { $env:PALITO_API_URL } else { "https://api.github.com/repos/$repo" }

    $arch = if ($env:PROCESSOR_ARCHITEW6432) { $env:PROCESSOR_ARCHITEW6432 } else { $env:PROCESSOR_ARCHITECTURE }
    if ($arch -ne 'AMD64') {
        throw "no prebuilt bundle for Windows $arch; install with: cargo install palito --features vulkan"
    }
    $platform = 'windows-x86_64'

    $version = $env:PALITO_VERSION
    if (-not $version) {
        # The /latest page redirects to /tag/vX.Y.Z; no API call (no rate limit).
        $response = Invoke-WebRequest -Uri "$releases/latest" -UseBasicParsing -Method Head
        $final = $response.BaseResponse.ResponseUri.AbsoluteUri
        if (-not $final) { $final = $response.BaseResponse.RequestMessage.RequestUri.AbsoluteUri }
        $version = ($final -split '/v')[-1]
        if ($version -notmatch '^\d+\.\d+\.\d+$') { throw "could not find the latest release at $releases/latest" }
    }
    $version = $version.TrimStart('v')
    $name = "palito-$version-$platform"
    $file = "$name.zip"

    $tmp = Join-Path ([IO.Path]::GetTempPath()) ("palito-install-" + [Guid]::NewGuid())
    New-Item -ItemType Directory -Path $tmp | Out-Null
    try {
        Write-Host "Downloading palito $version ($platform)..."
        $archive = Join-Path $tmp $file
        try {
            Invoke-WebRequest -Uri "$releases/download/v$version/$file" -OutFile $archive -UseBasicParsing
        } catch {
            throw "download failed: $releases/download/v$version/$file"
        }

        $expected = $null
        try {
            $release = Invoke-RestMethod -Uri "$api/releases/tags/v$version" -UseBasicParsing
            $asset = $release.assets | Where-Object { $_.name -eq $file } | Select-Object -First 1
            if ($asset.digest -like 'sha256:*') { $expected = $asset.digest.Substring(7).ToLower() }
        } catch {}
        $actual = (Get-FileHash -Algorithm SHA256 -Path $archive).Hash.ToLower()
        if (-not $expected) {
            Write-Warning 'could not read the release checksum from GitHub; skipping verification'
        } elseif ($expected -ne $actual) {
            throw "checksum mismatch for $file (expected $expected, got $actual)"
        } else {
            Write-Host 'sha256 ok'
        }

        Expand-Archive -Path $archive -DestinationPath $tmp -Force
        $source = Join-Path $tmp $name
        if (-not (Test-Path (Join-Path $source 'palito.exe'))) { throw "unexpected archive layout in $file" }

        $installRoot = if ($env:PALITO_INSTALL_DIR) { $env:PALITO_INSTALL_DIR } else { Join-Path $env:LOCALAPPDATA 'palito' }
        $target = Join-Path $installRoot $version
        New-Item -ItemType Directory -Path $installRoot -Force | Out-Null
        if (Test-Path $target) { Remove-Item -Recurse -Force $target }
        Move-Item -Path $source -Destination $target
    } finally {
        Remove-Item -Recurse -Force $tmp -ErrorAction SilentlyContinue
    }

    Write-Host "Installed palito $version in $target"

    # Point the user PATH at this version, replacing older palito folders.
    if ($env:PALITO_NO_MODIFY_PATH -eq '1') {
        Write-Host "Add $target to PATH to run palito from anywhere."
    } else {
        $userPath = [Environment]::GetEnvironmentVariable('Path', 'User')
        $parts = @($userPath -split ';' | Where-Object { $_ -and -not $_.StartsWith($installRoot, [StringComparison]::OrdinalIgnoreCase) })
        [Environment]::SetEnvironmentVariable('Path', (@($target) + $parts) -join ';', 'User')
        $env:Path = "$target;" + (($env:Path -split ';' | Where-Object { $_ -and -not $_.StartsWith($installRoot, [StringComparison]::OrdinalIgnoreCase) }) -join ';')
        Write-Host "Added $target to your user PATH (new terminals pick it up; this one already has it)."
    }

    Write-Host ''
    Write-Host 'Get started:'
    Write-Host '  palito pull qwen3-8b'
    Write-Host '  palito chat --model qwen3-8b'
}

Install-Palito
