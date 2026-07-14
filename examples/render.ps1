#!/usr/bin/env pwsh
# Render a trd JSONL frame-parameter file to an animated GIF/WebP (PowerShell 7).
#
# PowerShell port of examples/render.sh with the same behaviour:
#   JSONL --(duckdb: cast to FLOAT[2], Arrow IPC)--> trd --(tensors)--> ffmpeg
#
# Unlike render.sh (which pipes everything with no intermediate files), Windows
# DuckDB cannot write to '/dev/stdout' and PowerShell pipelines are not
# binary-safe, so the Arrow IPC stages are handed off through temporary files
# (created in a temp dir and auto-removed). The produced GIF/WebP is identical.
#
# Usage:
#   examples/render.ps1 [-InputPath INPUT.jsonl] [-Output OUTPUT.gif|.webp] `
#                       [-Width 256] [-Height 256] [-Fps 30] [-Native]
#   examples/render.ps1 INPUT.jsonl OUTPUT.gif 256 256 30   # positional
# Defaults: examples/frames.jsonl  out.gif  256 256 30
#
# By default the frame stream is rendered to a GIF/WebP via the headless trd-cli.
# With -Native (alias -App) it is played live in the interactive trd-app window
# (trd-native); -Output is then ignored and neither uv nor ffmpeg are needed.
#
# On Windows this auto-sources scripts\dev-env.ps1 (the flake.nix devShell
# counterpart; see README "Windows setup (without Nix)" for the one-time
# prerequisites) to put cargo, the MSVC linker, ffmpeg, duckdb and uv on PATH;
# set $env:TRD_SKIP_DEV_ENV = '1' to manage the environment yourself. On
# Linux/macOS run inside `nix develop`. If uv is unavailable the encode step
# falls back to a system `python` that already has pyarrow + numpy.
# On WSL, set $env:WGPU_BACKEND = 'gl' first for GPU rendering (else software).

[CmdletBinding()]
param(
    [Parameter(Position = 0)][string]$InputPath,
    [Parameter(Position = 1)][string]$Output = 'out.gif',
    [Parameter(Position = 2)][int]$Width = 256,
    [Parameter(Position = 3)][int]$Height = 256,
    [Parameter(Position = 4)][int]$Fps = 30,
    [Alias('App')][switch]$Native
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$root = Split-Path -Parent $PSScriptRoot
if (-not $InputPath) { $InputPath = Join-Path $PSScriptRoot 'frames.jsonl' }

# Make the trd toolchain available the way `nix develop` does on Linux.
$devEnv = Join-Path $root 'scripts\dev-env.ps1'
if ((Test-Path $devEnv) -and -not $env:TRD_SKIP_DEV_ENV) {
    . $devEnv -Quiet -NoInstall
}

# Fail early if a base tool is missing. cargo is always required; the GIF path
# also needs ffmpeg. duckdb is optional -- if its 'arrow' community extension
# can't load, the frame stream is built with pyarrow instead.
$required = if ($Native) { @('cargo') } else { @('cargo', 'ffmpeg') }
foreach ($tool in $required) {
    if (-not (Get-Command $tool -ErrorAction SilentlyContinue)) {
        Write-Error "error: $tool not found on PATH`nOn Windows run '. scripts\dev-env.ps1' first (the flake.nix devShell counterpart); on Linux/macOS use 'nix develop'."
    }
}

# Probe the optional Python-based producers/encoders once (numpy is only needed
# to encode). uv, when present, supplies pyarrow/numpy on demand.
$uvOk = [bool](Get-Command uv -ErrorAction SilentlyContinue)
$pyarrowOk = $false
$pyNumpyOk = $false
if (Get-Command python -ErrorAction SilentlyContinue) {
    try { & python -c 'import pyarrow' 2>$null } catch { }
    $pyarrowOk = ($LASTEXITCODE -eq 0)
    try { & python -c 'import pyarrow, numpy' 2>$null } catch { }
    $pyNumpyOk = ($LASTEXITCODE -eq 0)
}

# Choose a frame producer: DuckDB (via its 'arrow' community extension) if that
# extension actually loads, otherwise scripts\jsonl_to_arrow.py via uv/python.
$jsonlToArrow = Join-Path $root 'scripts/jsonl_to_arrow.py'
$producer = $null
if (Get-Command duckdb -ErrorAction SilentlyContinue) {
    try { & duckdb -c 'INSTALL arrow FROM community; LOAD arrow;' 2>$null | Out-Null } catch { }
    if ($LASTEXITCODE -eq 0) { $producer = 'duckdb' }
    else { Write-Warning "duckdb 'arrow' extension unavailable; building the frame stream with pyarrow instead." }
}
if (-not $producer) {
    if ($uvOk) { $producer = 'uv' }
    elseif ($pyarrowOk) { $producer = 'python' }
    else {
        Write-Error "error: need duckdb (with the 'arrow' community extension) or uv/python with pyarrow to build the Arrow frame stream.`nrun '. scripts\dev-env.ps1', or 'pip install pyarrow'."
    }
}

# encode.py needs pyarrow + numpy. Prefer `uv run` (as render.sh does); fall
# back to a system `python` that already has both. Skipped for the native viewer.
if (-not $Native) {
    $encodePy = Join-Path $root 'scripts/encode.py'
    if ($uvOk) {
        $encoderFile = 'uv'
        $encoderArgs = @('run', '--with', 'pyarrow', '--with', 'numpy', $encodePy, '--fps', $Fps, '-o', $Output)
    }
    elseif ($pyNumpyOk) {
        $encoderFile = 'python'
        $encoderArgs = @($encodePy, '--fps', $Fps, '-o', $Output)
    }
    else {
        Write-Error "error: need 'uv' (preferred) or a 'python' with pyarrow + numpy to encode.`nrun '. scripts\dev-env.ps1' to install uv, or 'pip install pyarrow numpy'."
    }
}

# DuckDB SQL string literals escape a single quote by doubling it; forward
# slashes work on every platform, so normalise Windows backslashes.
function ConvertTo-SqlPath([string]$p) { ($p -replace "'", "''") -replace '\\', '/' }

$work = (New-Item -ItemType Directory -Path (Join-Path ([System.IO.Path]::GetTempPath()) "trd-render-$([guid]::NewGuid())")).FullName
$framesArrow = Join-Path $work 'frames.arrows'
$imagesArrow = Join-Path $work 'images.arrows'
try {
    # 1. Build a streaming Arrow IPC file of frame params (center/size as
    #    FixedSizeList<f32>[2], theta as f32) from the JSONL. DuckDB does the
    #    cast when its 'arrow' extension is available; otherwise pyarrow does.
    if ($producer -eq 'duckdb') {
        $sql = @"
INSTALL arrow FROM community; LOAD arrow;
COPY (
  SELECT center::FLOAT[2] AS center, size::FLOAT[2] AS size, theta::FLOAT AS theta
  FROM read_json_auto('$(ConvertTo-SqlPath $InputPath)')
) TO '$(ConvertTo-SqlPath $framesArrow)' (FORMAT arrows);
"@
        & duckdb -c $sql
        if ($LASTEXITCODE -ne 0) { throw "duckdb failed (exit $LASTEXITCODE)" }
    }
    else {
        # $producer is 'uv' or 'python': run jsonl_to_arrow.py to the temp file.
        if ($producer -eq 'uv') {
            $genArgs = @('run', '--with', 'pyarrow', $jsonlToArrow, $InputPath, '-o', $framesArrow)
        }
        else {
            $genArgs = @($jsonlToArrow, $InputPath, '-o', $framesArrow)
        }
        $gen = Start-Process -FilePath $producer -NoNewWindow -Wait -PassThru -ArgumentList $genArgs
        if ($gen.ExitCode -ne 0) { throw "jsonl_to_arrow ($producer) failed (exit $($gen.ExitCode))" }
    }

    if ($Native) {
        # 2. Play the frame stream live in the interactive trd-app window
        #    (trd-native). It reads the same Arrow stream trd-cli consumes.
        $appArgs = @(
            'run', '--manifest-path', (Join-Path $root 'Cargo.toml'),
            '-q', '-p', 'trd-app', '--', '--width', $Width, '--height', $Height, '--fps', $Fps
        )
        $app = Start-Process -FilePath 'cargo' -NoNewWindow -Wait -PassThru `
            -ArgumentList $appArgs `
            -RedirectStandardInput $framesArrow
        if ($app.ExitCode -ne 0) { throw "trd-app failed (exit $($app.ExitCode))" }
    }
    else {
        # 2. trd renders each row to r,g,b,a fixed_shape_tensor<u8> channels. The
        #    Arrow streams are redirected via files so the bytes stay intact.
        $trdArgs = @(
            'run', '--manifest-path', (Join-Path $root 'Cargo.toml'),
            '-q', '-p', 'trd-cli', '--', '--width', $Width, '--height', $Height
        )
        $trd = Start-Process -FilePath 'cargo' -NoNewWindow -Wait -PassThru `
            -ArgumentList $trdArgs `
            -RedirectStandardInput $framesArrow -RedirectStandardOutput $imagesArrow
        if ($trd.ExitCode -ne 0) { throw "trd failed (exit $($trd.ExitCode))" }

        # 3. encode.py decodes the tensors and pipes RGBA frames to ffmpeg
        #    (.gif or .webp by output extension).
        $enc = Start-Process -FilePath $encoderFile -NoNewWindow -Wait -PassThru `
            -ArgumentList $encoderArgs `
            -RedirectStandardInput $imagesArrow
        if ($enc.ExitCode -ne 0) { throw "encode ($encoderFile) failed (exit $($enc.ExitCode))" }
    }
}
finally {
    Remove-Item -Recurse -Force $work -ErrorAction SilentlyContinue
}

if ($Native) {
    Write-Host "streamed $InputPath to the trd-app window (${Width}x${Height}, ${Fps}fps)"
}
else {
    Write-Host "wrote $Output (${Width}x${Height}, ${Fps}fps) from $InputPath"
}
