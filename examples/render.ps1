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
#                       [-Width 256] [-Height 256] [-Fps 30] [-Native] [-Web]
#   examples/render.ps1 INPUT.jsonl OUTPUT.gif 256 256 30   # positional
# Defaults: examples/frames.jsonl  output/out.gif  256 256 30
#
# By default the frame stream is rendered to a GIF/WebP via the headless trd-cli.
# With -Native (alias -App) it is played live in the interactive trd-app window
# (trd-native); -Output is then ignored and neither uv nor ffmpeg are needed.
# With -Web (alias -Wasm) it builds the browser (wasm) bundle with wasm-pack + bun
# and serves web/dist, printing the machine URLs and an SSH-tunnel command. The web
# demo generates its own Arrow frame stream in-browser, so all positional arguments
# are ignored. Override the port with $env:PORT (default 8088); binds all interfaces.
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
    [Parameter(Position = 1)][string]$Output = 'output/out.gif',
    [Parameter(Position = 2)][int]$Width = 256,
    [Parameter(Position = 3)][int]$Height = 256,
    [Parameter(Position = 4)][int]$Fps = 30,
    [Alias('App')][switch]$Native,
    [Alias('Wasm')][switch]$Web
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

# --- -Web/-Wasm: build the browser (wasm) bundle and serve it -----------------
# Mirrors the --web/--wasm mode of render.sh, but Windows-native (no Nix): the
# bundle is built with wasm-pack + bun (web/'s `bun run build`) and served from
# web/dist by a small Bun static server. The demo generates its own Arrow frame
# stream in-browser, so InputPath/Output and the other arguments are ignored.
if ($Web) {
    foreach ($tool in @('cargo', 'wasm-pack', 'bun')) {
        if (-not (Get-Command $tool -ErrorAction SilentlyContinue)) {
            Write-Error "error: $tool not found on PATH`n-Web needs cargo, wasm-pack and bun. On Windows run '. scripts\dev-env.ps1' first (and install wasm-pack + bun); on Linux/macOS use 'nix develop'."
        }
    }

    $port = if ($env:PORT) { $env:PORT } else { '8088' }
    $webDir = Join-Path $root 'web'
    Write-Host 'building trd web (wasm) bundle (wasm-pack + bun)...'
    Push-Location $webDir
    try {
        & bun run build
        if ($LASTEXITCODE -ne 0) { throw "web build failed (exit $LASTEXITCODE)" }
    }
    finally {
        Pop-Location
    }

    $user = if ($env:USERNAME) { $env:USERNAME } else { 'user' }
    # First non-loopback IPv4 of this host (for the direct / SSH-tunnel URLs).
    $ip = $null
    try {
        $ip = Get-NetIPAddress -AddressFamily IPv4 -ErrorAction Stop |
            Where-Object { $_.IPAddress -ne '127.0.0.1' -and $_.PrefixOrigin -ne 'WellKnown' } |
            Select-Object -First 1 -ExpandProperty IPAddress
    }
    catch { }
    if (-not $ip) { $ip = '<server-ip>' }

    $distDir = Join-Path $webDir 'dist'
    # Small Bun static file server (the no-Nix counterpart of static-web-server,
    # which nix's `.#web` app uses). Bun.file sets content-types, incl. wasm.
    $serveScript = Join-Path ([System.IO.Path]::GetTempPath()) "trd-web-serve-$([guid]::NewGuid()).ts"
    @'
const root = process.argv[2];
const port = Number(Bun.env.PORT ?? 8088);
Bun.serve({
  port,
  hostname: "0.0.0.0",
  async fetch(req) {
    let path = decodeURIComponent(new URL(req.url).pathname);
    if (path.endsWith("/")) path += "index.html";
    const asset = Bun.file(root + path);
    return (await asset.exists())
      ? new Response(asset)
      : new Response("404 Not Found", { status: 404 });
  },
});
'@ | Set-Content -Path $serveScript -Encoding utf8

    Write-Host ''
    Write-Host "trd web (wasm) server - port $port  (press Ctrl-C to stop)"
    Write-Host ''
    Write-Host "  On this machine:        http://localhost:$port"
    Write-Host "  Direct (same network):  http://${ip}:$port"
    Write-Host ''
    Write-Host '  SSH tunnel (recommended if the port is not directly reachable):'
    Write-Host "    ssh -L ${port}:localhost:$port $user@$ip"
    Write-Host '  then open in a WebGPU browser (Chrome/Edge):'
    Write-Host "                          http://localhost:$port"
    Write-Host ''

    try {
        $env:PORT = $port
        & bun $serveScript $distDir
    }
    finally {
        Remove-Item -Force $serveScript -ErrorAction SilentlyContinue
    }
    exit 0
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
    $outDir = Split-Path -Parent $Output
    if ($outDir -and -not (Test-Path $outDir)) {
        New-Item -ItemType Directory -Path $outDir -Force | Out-Null
    }
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
