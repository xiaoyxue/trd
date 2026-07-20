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
#   examples/render.ps1 [-CLI | -Native | -Web [-ArrowRenderer|-CanvasRenderer]] `
#                       [-Mesh OBJ]... [-Wireframe] [-Aabb] [-Axes] `
#                       [-InputPath INPUT.jsonl] [-Output OUTPUT.gif|.webp] `
#                       [-Width 256] [-Height 256] [-Fps 30]
#   examples/render.ps1 INPUT.jsonl OUTPUT.gif 256 256 30   # positional
# Defaults: examples/frames.0.0.2.jsonl  output/out.gif  256 256 30
# Run with no arguments (or -Help) to print the flag guidance and exit; pass -CLI
# to render the default demo.
#
# By default (or with -CLI, alias -Headless) the frame stream is rendered to a
# GIF/WebP via the headless trd-cli.
# With -Native (alias -App) it is played live in the interactive trd-app window
# (trd-native); -Output is then ignored and neither uv nor ffmpeg are needed.
# The content flags below (-Mesh/-Wireframe/-Aabb/-Axes) apply to both -CLI and
# -Native (trd-cli and trd-app share trd-core's mesh Scene renderer).
# With -Mesh OBJ the input is a protocol 0.0.3 stream: a leading mesh table
# (scripts\obj_to_arrow.py encodes the OBJ) concatenated with the params stream,
# so trd renders the loaded mesh (centered + uniformly scaled to fit) driven by
# InputPath. Try: examples\render.ps1 -CLI -Mesh assets\meshes\bunny.obj `
# examples\frames.turntable.jsonl output\bunny.gif. -Mesh is repeatable: pass it
# several times to load several meshes (one table row each, in order); a frame's
# `draws` list then references them by 0-based index. Two-mesh demo:
# examples\render.ps1 -CLI -Wireframe -Mesh assets\meshes\bunny.obj `
# -Mesh examples\cube.obj examples\frames.multimesh.jsonl output\scene.gif.
# (-Mesh needs pyarrow via uv/python and is ignored by -Web.)
# With -Wireframe trd draws mesh edges as a line list instead of filled triangles
# (protocol #38); combine with -Mesh for a wireframe asset.
# With -Aabb trd overlays each drawn mesh's axis-aligned bounding box as a green
# wireframe box (#42); combine with -Mesh (e.g. add -Aabb to the bunny turntable
# to see its box track the rotation).
# With -Axes trd overlays a coordinate-axes gizmo (X=red, Y=green, Z=blue) at the
# world origin (#42), marking the world frame the camera looks at.
#
# Dolly-camera capstone (#49): examples\bunny_dolly.py authors the same 45°
# bird's-eye dolly camera twice - CG (eye/target/fovy) and CV (K + pose) - as two
# JSONL streams that render identically (verified to <0.01% pixels). render.ps1
# runs this producer automatically: pass frames.bunny_dolly.cg.jsonl (or
# .cv.jsonl) as InputPath and, if it is missing, it is generated on the fly - no
# manual pre-step. Compare the two camera forms:
#   examples\render.ps1 -CLI -Wireframe -Aabb -Axes -Mesh assets\meshes\bunny.obj `
#     examples\frames.bunny_dolly.cg.jsonl output\bunny_dolly_cg.gif 1024 1024 24
#   examples\render.ps1 -CLI -Wireframe -Aabb -Axes -Mesh assets\meshes\bunny.obj `
#     examples\frames.bunny_dolly.cv.jsonl output\bunny_dolly_cv.gif 1024 1024 24
# With -Web (alias -Wasm) it builds the browser (wasm) bundle with wasm-pack + bun
# and serves web/dist, printing the machine URLs and an SSH-tunnel command. The web
# demo generates its own Arrow frame stream in-browser, so all positional arguments
# are ignored. Two in-browser renderers share the bundle: -ArrowRenderer (default)
# runs the offscreen output-stream smoke (the browser counterpart of the CLI);
# -CanvasRenderer runs the on-screen canvas demo. Override the port with $env:PORT
# (default 8088); binds all interfaces.
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
    [Alias('Headless')][switch]$CLI,
    [Alias('App')][switch]$Native,
    [Alias('Wasm')][switch]$Web,
    [switch]$ArrowRenderer,
    [switch]$CanvasRenderer,
    [switch]$Wireframe,
    [switch]$Aabb,
    [switch]$Axes,
    [switch]$Help,
    # Repeatable -Mesh <obj> flags land here (PowerShell can't bind a named
    # parameter more than once); they are extracted into $meshes below. Leaving
    # -Mesh out of the formal parameters keeps positional InputPath/Output/Width/
    # Height/Fps binding intact when -Mesh flags are interleaved.
    [Parameter(ValueFromRemainingArguments = $true)][string[]]$Rest
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

# Print flag guidance (shown for a bare invocation or -Help).
function Show-RenderUsage {
    Write-Host @'
render.ps1 - render a trd JSONL frame-parameter file to a GIF/WebP (or play/serve it). PowerShell 7.

Usage:
  examples\render.ps1 [MODE] [CONTENT FLAGS] [-InputPath INPUT.jsonl] [-Output OUTPUT.gif|.webp] `
                      [-Width 256] [-Height 256] [-Fps 30]
  examples\render.ps1 INPUT.jsonl OUTPUT.gif 256 256 30   # positional form

Defaults: InputPath=examples\frames.0.0.2.jsonl  Output=output\out.gif  Width=256  Height=256  Fps=30

MODE (pick one; default -CLI):
  -CLI, -Headless   Render to a GIF/WebP via the headless trd-cli (default).
  -Native, -App     Play live in the interactive trd-app window (-Output ignored).
  -Web, -Wasm       Build the browser (wasm) bundle and serve it (positional args ignored).
                      -ArrowRenderer   offscreen output-stream smoke (default)
                      -CanvasRenderer  on-screen canvas demo

CONTENT FLAGS (apply to -CLI and -Native):
  -Mesh OBJ         Load OBJ as a protocol 0.0.3 mesh (centered + scaled to fit).
                    Repeatable: pass several times to load several meshes (row 0,
                    1, ...); a frame's `draws` list references them by index.
  -Wireframe        Draw mesh edges as a line list instead of filled triangles (#38).
  -Aabb             Overlay each mesh's axis-aligned bounding box as a green box (#42).
  -Axes             Overlay a coordinate-axes gizmo (X=red, Y=green, Z=blue) at the origin (#42).

  -Help             Show this guidance and exit.

Examples:
  examples\render.ps1 -CLI                                    # default demo -> output\out.gif
  examples\render.ps1 -Native                                # play the default demo live
  examples\render.ps1 -Native -Mesh assets\meshes\bunny.obj `
    examples\frames.bunny_dolly.cg.jsonl _ 1024 1024 24      # live dolly capstone in a window
  examples\render.ps1 -CLI -Aabb -Mesh assets\meshes\bunny.obj `
    examples\frames.turntable.jsonl output\bunny.gif 1024 1024 24
  examples\render.ps1 -CLI -Wireframe -Aabb `
    -Mesh assets\meshes\bunny.obj -Mesh examples\cube.obj `
    examples\frames.multimesh.jsonl output\scene.gif 1024 1024 24
  examples\render.ps1 -CLI -Wireframe -Axes -Aabb -Mesh assets\meshes\bunny.obj `
    examples\frames.bunny_dolly.cg.jsonl output\bunny_dolly.gif 1024 1024 24  # dolly capstone (#49; auto-generates the frames)
  examples\render.ps1 -Web                                    # build + serve the wasm demo

On Windows this auto-sources scripts\dev-env.ps1; on Linux/macOS run inside `nix develop`.
'@
}

# A bare invocation (no arguments at all), or -Help, prints the flag guidance and
# exits rather than silently rendering the default demo -- pass -CLI to run it.
if ($Help -or $PSBoundParameters.Count -eq 0) {
    Show-RenderUsage
    exit 0
}

# --- Mode selection & validation ---------------------------------------------
# The top-level modes are mutually exclusive: the default headless render
# (explicit alias -CLI/-Headless), the live -Native window, and the browser
# -Web/-Wasm bundle. -ArrowRenderer / -CanvasRenderer sub-select the in-browser
# renderer and therefore apply only to -Web.
$modeCount = @($CLI, $Native, $Web).Where({ $_ }).Count
if ($modeCount -gt 1) { Write-Error 'error: choose only one of -CLI, -Native, -Web.' }
if ($ArrowRenderer -and $CanvasRenderer) { Write-Error 'error: -ArrowRenderer and -CanvasRenderer are mutually exclusive.' }
if (($ArrowRenderer -or $CanvasRenderer) -and -not $Web) { Write-Error 'error: -ArrowRenderer / -CanvasRenderer apply only to -Web/-Wasm.' }

# --- Repeatable -Mesh <obj> extraction ---------------------------------------
# PowerShell can't bind a named parameter more than once, so the repeatable
# -Mesh flag (parity with render.sh's `--mesh`) is captured by
# ValueFromRemainingArguments into $Rest and unpacked here, preserving order
# (mesh 0 = first -Mesh). Each mesh becomes one row of the leading 0.0.3 mesh
# table (scripts\obj_to_arrow.py); a frame's `draws` list references them by
# 0-based index. Also accepts the -Mesh=OBJ / -Mesh:OBJ forms. Anything else in
# $Rest is an unrecognised argument.
$meshes = @()
if ($Rest) {
    for ($i = 0; $i -lt $Rest.Count; $i++) {
        $tok = $Rest[$i]
        if ($tok -ieq '-Mesh') {
            $i++
            if ($i -ge $Rest.Count) { Write-Error 'error: -Mesh requires an OBJ path.' }
            $meshes += $Rest[$i]
        }
        elseif ($tok -like '-Mesh=*' -or $tok -like '-Mesh:*') {
            $meshes += $tok.Substring(6)
        }
        else {
            Write-Error "error: unexpected argument '$tok' (use -Mesh <obj>; content flags are -Wireframe/-Aabb/-Axes)."
        }
    }
}

$root = Split-Path -Parent $PSScriptRoot
if (-not $InputPath) { $InputPath = Join-Path $PSScriptRoot 'frames.0.0.2.jsonl' }

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

    # Both browser renderers ship in one bundle; web/src/main.ts routes on the
    # `arrow-smoke` query param, so only the opened URL differs. Default (or
    # -ArrowRenderer) = the offscreen ArrowRenderer output-stream smoke (the
    # browser counterpart of -CLI); -CanvasRenderer = the on-screen canvas demo.
    if ($CanvasRenderer) {
        $demoQuery = ''
        $rendererLabel = 'CanvasRenderer (on-screen canvas demo)'
    }
    else {
        $demoQuery = '?arrow-smoke'
        $rendererLabel = 'ArrowRenderer (offscreen output-stream smoke)'
    }

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
    Write-Host "  renderer: $rendererLabel"
    Write-Host ''
    Write-Host "  On this machine:        http://localhost:$port$demoQuery"
    Write-Host "  Direct (same network):  http://${ip}:$port$demoQuery"
    Write-Host ''
    Write-Host '  SSH tunnel (recommended if the port is not directly reachable):'
    Write-Host "    ssh -L ${port}:localhost:$port $user@$ip"
    Write-Host '  then open in a WebGPU browser (Chrome/Edge):'
    Write-Host "                          http://localhost:$port$demoQuery"
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
$pythonOk = [bool](Get-Command python -ErrorAction SilentlyContinue)
$pyarrowOk = $false
$pyNumpyOk = $false
if ($pythonOk) {
    try { & python -c 'import pyarrow' 2>$null } catch { }
    $pyarrowOk = ($LASTEXITCODE -eq 0)
    try { & python -c 'import pyarrow, numpy' 2>$null } catch { }
    $pyNumpyOk = ($LASTEXITCODE -eq 0)
}

# Dolly-camera capstone (#49): examples\bunny_dolly.py authors the 45° bird's-eye
# dolly camera as two JSONL streams - CG (eye/target/fovy) and CV (K + pose) -
# that render identically. If the requested InputPath is one of its outputs
# (frames.bunny_dolly.{cg,cv}.jsonl) and it is not present yet, generate it now
# via the (pure-stdlib) producer so the demo renders without a manual pre-step.
if ($InputPath -match 'frames\.bunny_dolly\.(cg|cv)\.jsonl$' -and -not (Test-Path $InputPath)) {
    $prefix = $InputPath -replace '\.(cg|cv)\.jsonl$', ''
    $dollyPy = Join-Path $root 'examples/bunny_dolly.py'
    Write-Host "generating dolly frames via examples/bunny_dolly.py (--out-prefix $prefix)..."
    if ($pythonOk) {
        $dollyGen = Start-Process -FilePath 'python' -NoNewWindow -Wait -PassThru -ArgumentList @($dollyPy, '--out-prefix', $prefix)
    }
    elseif ($uvOk) {
        $dollyGen = Start-Process -FilePath 'uv' -NoNewWindow -Wait -PassThru -ArgumentList @('run', '--python', '3.12', $dollyPy, '--out-prefix', $prefix)
    }
    else {
        Write-Error 'error: need python (or uv) to run examples/bunny_dolly.py.'
    }
    if ($dollyGen.ExitCode -ne 0) { throw "bunny_dolly.py failed (exit $($dollyGen.ExitCode))" }
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

# DuckDB's producer only understands the 0.0.1/0.0.2 columns (center/size/theta/
# model); its SQL silently DROPS the additive 0.0.3 camera (eye/target/direction/
# up/k/pose/fovy/aspect/znear/zfar) and instanced draw-list (draws) columns. If
# the input carries any of those, fall back to the pyarrow producer (which emits
# them) so the camera/draw data actually reaches trd - otherwise an authored
# camera is lost and trd renders with the identity camera (z-clipping).
if ($producer -eq 'duckdb' -and
    (Select-String -Path $InputPath -Pattern '"(eye|target|direction|up|k|pose|fovy|aspect|znear|zfar|draws)"\s*:' -Quiet)) {
    if ($uvOk) { $producer = 'uv' }
    elseif ($pyarrowOk) { $producer = 'python' }
    else {
        Write-Error "error: '$InputPath' carries 0.0.3 camera/draw columns that DuckDB cannot emit;`ninstall uv or a python with pyarrow to render it (run '. scripts\dev-env.ps1')."
    }
}

# -Mesh (repeatable) encodes the leading 0.0.3 mesh table via
# scripts\obj_to_arrow.py (one row per OBJ, in order). DuckDB cannot author the
# nested-list mesh table, so this always needs a pyarrow-capable Python.
$objToArrow = Join-Path $root 'scripts/obj_to_arrow.py'
$meshProducer = $null
if ($meshes.Count -gt 0) {
    if ($uvOk) { $meshProducer = 'uv' }
    elseif ($pyarrowOk) { $meshProducer = 'python' }
    else {
        Write-Error "error: -Mesh needs uv or a python with pyarrow to encode $($meshes -join ', ').`nrun '. scripts\dev-env.ps1', or 'pip install pyarrow'."
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

# Binary-safe concatenation of Arrow IPC files into one stream. render.sh pipes
# `obj_to_arrow.py` then the params producer into a single trd stdin; on Windows
# (no binary-safe pipes) we stage each stage to a temp file and concatenate the
# bytes here, reproducing the exact [mesh][params] byte order trd reads.
function Join-Files([string[]]$Parts, [string]$Dest) {
    $out = [System.IO.File]::Create($Dest)
    try {
        foreach ($p in $Parts) {
            $in = [System.IO.File]::OpenRead($p)
            try { $in.CopyTo($out) } finally { $in.Dispose() }
        }
    }
    finally { $out.Dispose() }
}

$work = (New-Item -ItemType Directory -Path (Join-Path ([System.IO.Path]::GetTempPath()) "trd-render-$([guid]::NewGuid())")).FullName
$framesArrow = Join-Path $work 'frames.arrows'
$meshArrow = Join-Path $work 'mesh.arrows'
$streamArrow = Join-Path $work 'stream.arrows'
$imagesArrow = Join-Path $work 'images.arrows'
try {
    # 1. Build a streaming Arrow IPC file of frame params from the JSONL: the
    #    required 0.0.1 columns (center/size as FixedSizeList<f32>[2], theta as
    #    f32, defaulting to the identity when absent) plus the additive 0.0.2
    #    `model` column (FixedSizeList<f32>[16], column-major) - used verbatim if
    #    present, else synthesized to match scripts/jsonl_to_arrow.py. DuckDB does
    #    the cast when its 'arrow' extension is available; otherwise pyarrow does.
    if ($producer -eq 'duckdb') {
        $sql = @"
INSTALL arrow FROM community; LOAD arrow;
COPY (
  WITH raw AS (
    SELECT
      COALESCE(center, [0.0, 0.0]) AS c,
      COALESCE(size, [1.0, 1.0]) AS s,
      COALESCE(theta, 0.0) AS th,
      model AS m
    FROM read_json('$(ConvertTo-SqlPath $InputPath)',
      format = 'newline_delimited',
      columns = {center: 'DOUBLE[]', size: 'DOUBLE[]', theta: 'DOUBLE', model: 'DOUBLE[]'})
  )
  SELECT
    c::FLOAT[2] AS center,
    s::FLOAT[2] AS size,
    th::FLOAT AS theta,
    COALESCE(m, [
      s[1] * cos(th), s[1] * sin(th), 0.0, 0.0,
      -s[2] * sin(th), s[2] * cos(th), 0.0, 0.0,
      0.0, 0.0, 1.0, 0.0,
      c[1], c[2], 0.0, 1.0
    ])::FLOAT[16] AS model
  FROM raw
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

    # 1b. -Mesh: encode the OBJ(s) into a leading 0.0.3 mesh table (one row per
    #     -Mesh, in order) and concatenate it *before* the params so trd reads
    #     [mesh][params]. A frame's `draws` list references these meshes by
    #     0-based index (mesh 0 = first -Mesh). Without -Mesh, trd reads the
    #     params stream directly.
    if ($meshes.Count -gt 0) {
        if ($meshProducer -eq 'uv') {
            $meshArgs = @('run', '--with', 'pyarrow', $objToArrow) + $meshes + @('-o', $meshArrow)
        }
        else {
            $meshArgs = @($objToArrow) + $meshes + @('-o', $meshArrow)
        }
        $meshGen = Start-Process -FilePath $meshProducer -NoNewWindow -Wait -PassThru -ArgumentList $meshArgs
        if ($meshGen.ExitCode -ne 0) { throw "obj_to_arrow ($meshProducer) failed (exit $($meshGen.ExitCode))" }
        Join-Files -Parts @($meshArrow, $framesArrow) -Dest $streamArrow
        $trdInput = $streamArrow
    }
    else {
        $trdInput = $framesArrow
    }

    if ($Native) {
        # 2. Play the frame stream live in the interactive trd-app window
        #    (trd-native). It reads the same [mesh][params] stream trd-cli
        #    consumes and renders the Scene (meshes + overlays) via trd-core.
        #    -Wireframe/-Aabb/-Axes pass through to trd-app (#38, #42).
        $appArgs = @(
            'run', '--manifest-path', (Join-Path $root 'Cargo.toml'),
            '-q', '-p', 'trd-app', '--', '--width', $Width, '--height', $Height, '--fps', $Fps
        )
        if ($Wireframe) { $appArgs += '--wireframe' }
        if ($Aabb) { $appArgs += '--aabb' }
        if ($Axes) { $appArgs += '--axes' }
        $app = Start-Process -FilePath 'cargo' -NoNewWindow -Wait -PassThru `
            -ArgumentList $appArgs `
            -RedirectStandardInput $trdInput
        if ($app.ExitCode -ne 0) { throw "trd-app failed (exit $($app.ExitCode))" }
    }
    else {
        # 2. trd renders each row to r,g,b,a fixed_shape_tensor<u8> channels. The
        #    Arrow streams are redirected via files so the bytes stay intact.
        #    -Wireframe/-Aabb/-Axes pass through to trd-cli (#38, #42).
        $trdArgs = @(
            'run', '--manifest-path', (Join-Path $root 'Cargo.toml'),
            '-q', '-p', 'trd-cli', '--', '--width', $Width, '--height', $Height
        )
        if ($Wireframe) { $trdArgs += '--wireframe' }
        if ($Aabb) { $trdArgs += '--aabb' }
        if ($Axes) { $trdArgs += '--axes' }
        $trd = Start-Process -FilePath 'cargo' -NoNewWindow -Wait -PassThru `
            -ArgumentList $trdArgs `
            -RedirectStandardInput $trdInput -RedirectStandardOutput $imagesArrow
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
