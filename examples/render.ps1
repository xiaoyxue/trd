#!/usr/bin/env pwsh
# Render a trd JSONL frame-parameter file to an animated GIF/WebP, play it live in
# the native window, or replay it in a WebGPU browser (PowerShell 7).
#
# Windows-native port of examples/render.sh with the SAME behaviour and flags:
#   JSONL --(duckdb/pyarrow: Arrow IPC)--> trd --(tensors)--> ffmpeg   (-CLI)
#   JSONL --(pyarrow: Arrow IPC)--------> trd-app window              (-Native)
#   JSONL --(pyarrow: Arrow IPC)--------> stream.arrow + config.json  (-Web)
#                                         served to the generic web renderer
#
# Unlike render.sh (which pipes everything with no intermediate files), Windows
# DuckDB cannot write to '/dev/stdout' and PowerShell pipelines are not
# binary-safe, so the Arrow IPC stages are handed off through temporary files
# (created in a temp dir and auto-removed). The produced GIF/WebP is identical.
#
# Everything runs Windows-native: no WSL, no Nix. -Web builds the wasm bundle with
# wasm-pack + bun (the counterpart of `nix build .#web`) and serves it with a small
# Bun static server (the counterpart of static-web-server), so the placement /
# frame-plane / generic-web checks that render.sh runs under nix run here too.
#
# Usage:
#   examples/render.ps1 [-CLI | -Native | -Web [-CanvasRenderer|-OffscreenRenderer]] `
#                       [-Mesh OBJ]... [-Texture IMG] [-Wireframe] [-Aabb] [-Axes] `
#                       [-AxesLocal] [-PlacementQuad] [-PlacementQuadColor "R G B"] `
#                       [-FramesBase DIR] [-InputPath INPUT.jsonl] `
#                       [-Output OUTPUT.gif|.webp] [-Width 256] [-Height 256] [-Fps 30]
#   examples/render.ps1 INPUT.jsonl OUTPUT.gif 256 256 30   # positional
# Defaults: examples/frames.0.0.2.jsonl  output/out.gif  256 256 30
# Run with no arguments (or -Help) to print the flag guidance and exit; pass -CLI
# to render the default demo.
#
# By default (or with -CLI, alias -Headless) the frame stream is rendered to a
# GIF/WebP via the headless trd-cli.
# With -Native (alias -App) it is played live in the interactive trd-app window
# (trd-native); -Output is then ignored and neither uv nor ffmpeg are needed.
# With -Web (alias -Wasm) it renders the SAME scene as -CLI, but in a WebGPU
# browser: it builds the config-driven web bundle, writes stream.arrow (the
# identical bytes trd-cli reads on stdin) + config.json (renderer target + scene
# flags + baked resolution + default fps) + the -FramesBase stills into web/dist,
# and serves it so the browser replays exactly what -CLI would render.
# The content flags below (-Mesh/-Texture/-Wireframe/-Aabb/-Axes/-AxesLocal/
# -PlacementQuad/-FramesBase) and the positional Width/Height apply to all three
# modes (trd-cli, trd-app and the web renderer share trd-core). Only the playback
# rate is a live URL param for -Web: append ?fps=N.
#
# With -Mesh OBJ the input is a protocol 0.0.3 stream: a leading mesh table
# (scripts\obj_to_arrow.py encodes the OBJ) concatenated with the params stream,
# so trd renders the loaded mesh (centered + uniformly scaled to fit) driven by
# InputPath. Try: examples\render.ps1 -CLI -Mesh assets\meshes\bunny.obj `
# examples\frames.turntable.jsonl output\bunny.gif. -Mesh is repeatable: pass it
# several times to load several meshes (one table row each, in order); a frame's
# `draws` list then references them by 0-based index. Two-mesh demo:
# examples\render.ps1 -CLI -Wireframe -Mesh assets\meshes\bunny.obj `
# -Mesh examples\cube.obj examples\frames.multimesh.jsonl output\scene.gif.
# (-Mesh needs pyarrow via uv/python.)
# With -Texture IMG trd binds IMG as a 0.0.4 texture table (sampled albedo) and
# renders textured, sampling it at each vertex UV (#20). Requires -Mesh (with
# UVs); mutually exclusive with -Wireframe. Needs pyarrow + pillow + numpy;
# downscaled to 2048 (portable limit).
# With -Wireframe trd draws mesh edges as a line list instead of filled triangles
# (protocol #38); combine with -Mesh for a wireframe asset.
# With -Aabb trd overlays each drawn mesh's axis-aligned bounding box as a green
# wireframe box (#42).
# With -Axes trd overlays a coordinate-axes gizmo (X=red, Y=green, Z=blue) at the
# world origin (#42), marking the world frame the camera looks at.
# With -AxesLocal trd overlays a coordinate-axes gizmo at EACH drawn object's own
# local frame (its per-draw model), so each placed mesh shows its own axes (#77).
# With -PlacementQuad trd appends a canonical colored quad mesh (origin-centred,
# extent 2, corners +/-1) as the LAST -Mesh, so a stream can draw the reconstructed
# placement quad as a wireframe overlay (a debug check that it matches the filmed
# poster) — author its per-frame draw with placement_quad_by_local_coord.py.
# -PlacementQuadColor "R G B" (0..1 floats) tints it (default cyan) and implies
# -PlacementQuad. The quad rides the mesh table, so it needs the same pyarrow
# producer as -Mesh.
# With -FramesBase DIR trd composites each frame's 0.0.5 background still (its
# `frame_path`, relative to DIR) BENEATH the scene via a FramePlane (#63). Extract
# the stills first, at the render height so each per-frame decode stays cheap:
#   uv run --with pyarrow scripts\extract_frames.py `
#     assets\videos\cornellbox\CameraMovement.mp4 --format jpg --height 540 -o output\cornellbox
#
# Dolly-camera capstone (#49): examples\bunny_dolly.py authors the same 45°
# bird's-eye dolly camera twice - CG (eye/target/fovy) and CV (K + pose) - as two
# JSONL streams that render identically (verified to <0.01% pixels). render.ps1
# runs this producer automatically: pass frames.bunny_dolly.cg.jsonl (or
# .cv.jsonl) as InputPath and, if it is missing, it is generated on the fly - no
# manual pre-step. The CV stream's K is baked for 1024x1024 (render it at that
# resolution); the CG stream is resolution-independent. Compare the two forms:
#   examples\render.ps1 -CLI -Wireframe -Aabb -Axes -Mesh assets\meshes\bunny.obj `
#     examples\frames.bunny_dolly.cg.jsonl output\bunny_dolly_cg.gif 1024 1024 24
#   examples\render.ps1 -CLI -Wireframe -Aabb -Axes -Mesh assets\meshes\bunny.obj `
#     examples\frames.bunny_dolly.cv.jsonl output\bunny_dolly_cv.gif 1024 1024 24
#
# -Web replays any -CLI scene in the browser (same flags + positional W H FPS).
# Two in-browser renderers share the bundle: -CanvasRenderer (default) draws to
# the on-screen WebGPU CanvasRenderer; -OffscreenRenderer (alias -ArrowRenderer)
# draws to an offscreen ArrowRenderer texture read back to a 2D canvas (the browser
# twin of the CLI output stream). Override the port with $env:PORT (default 8080);
# binds all interfaces. e.g.:
#   examples\render.ps1 -Web -CanvasRenderer -PlacementQuad -AxesLocal `
#     -FramesBase output\cornellbox examples\frames.cornellbox.stage1.jsonl '' 960 540 25
#
# On Windows this auto-sources scripts\dev-env.ps1 (the flake.nix devShell
# counterpart; see README "Windows setup (without Nix)" for the one-time
# prerequisites) to put cargo, the MSVC linker, ffmpeg, duckdb and uv on PATH;
# set $env:TRD_SKIP_DEV_ENV = '1' to manage the environment yourself. On
# Linux/macOS run inside `nix develop`. If uv is unavailable the encode step
# falls back to a system `python` that already has pyarrow + numpy.

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
    [switch]$CanvasRenderer,
    [Alias('ArrowRenderer')][switch]$OffscreenRenderer,
    [switch]$Wireframe,
    [switch]$Aabb,
    [switch]$Axes,
    [switch]$AxesLocal,
    [switch]$PlacementQuad,
    [string]$PlacementQuadColor,
    [string]$Texture,
    [string]$FramesBase,
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
  -Web, -Wasm       Build the wasm bundle and serve the SAME scene as -CLI in a
                    WebGPU browser (generates stream.arrow + config.json).
                      -CanvasRenderer    on-screen WebGPU surface (default)
                      -OffscreenRenderer offscreen texture -> RGBA readback -> 2D canvas
                                         (alias -ArrowRenderer; browser twin of -CLI)

BROWSER QUERY PARAM (-Web; append to the URL, no rebuild):
  ?fps=N            Override the playback rate (the resolution is baked into the
                    stream, so it is a positional Width/Height argument).

CONTENT FLAGS (apply to -CLI, -Native and -Web):
  -Mesh OBJ         Load OBJ as a protocol 0.0.3 mesh (centered + scaled to fit).
                    Repeatable: pass several times to load several meshes (row 0,
                    1, ...); a frame's `draws` list references them by index.
  -Texture IMG      Bind IMG as a 0.0.4 texture and render textured - sampling it
                    at each vertex UV (#20). Requires -Mesh (with UVs); mutually
                    exclusive with -Wireframe.
  -Wireframe        Draw mesh edges as a line list instead of filled triangles (#38).
  -Aabb             Overlay each mesh's axis-aligned bounding box as a green box (#42).
  -Axes             Overlay a coordinate-axes gizmo (X=red, Y=green, Z=blue) at the origin (#42).
  -AxesLocal        Overlay a coordinate-axes gizmo at EACH drawn object's own local frame (#77).
  -PlacementQuad    Append a canonical colored quad mesh (origin-centred, extent 2)
                    as the last -Mesh, drawn as a wireframe placement overlay (#77).
  -PlacementQuadColor "R G B"
                    Tint the placement quad (0..1 floats; default cyan). Implies -PlacementQuad.
  -FramesBase DIR   Composite each frame's 0.0.5 background still (`frame_path`,
                    relative to DIR) beneath the scene via a FramePlane (#63).

  -Help             Show this guidance and exit.

Examples:
  examples\render.ps1 -CLI                                    # default demo -> output\out.gif
  examples\render.ps1 -Native                                # play the default demo live
  examples\render.ps1 -CLI -Aabb -Mesh assets\meshes\bunny.obj `
    examples\frames.turntable.jsonl output\bunny.gif 1024 1024 24
  examples\render.ps1 -CLI -Mesh assets\meshes\bunny_with_texture\bunny.obj `
    -Texture assets\meshes\bunny_with_texture\bunny_uv_map1.jpg `
    examples\frames.bunny_dolly.cv.jsonl output\bunny_textured.gif 1024 1024 24  # textured dolly (#20)
  examples\render.ps1 -CLI -Wireframe -Axes -Aabb -Mesh assets\meshes\bunny.obj `
    examples\frames.bunny_dolly.cg.jsonl output\bunny_dolly.gif 1024 1024 24  # dolly capstone (#49; auto-generates the frames)
  # Two-stage placement-quad pipeline (#77): extract stills once, then render:
  #   uv run --with pyarrow scripts\extract_frames.py `
  #     assets\videos\cornellbox\CameraMovement.mp4 --format jpg -o output\cornellbox
  examples\render.ps1 -CLI -PlacementQuad -AxesLocal -FramesBase output\cornellbox `
    examples\frames.cornellbox.stage1.jsonl output\cornellbox_stage1.gif 960 540 25  # stage 1: quad + local axes
  examples\render.ps1 -CLI -PlacementQuad -AxesLocal -Aabb `
    -Mesh assets\meshes\bunny_with_texture\bunny.obj `
    -Texture assets\meshes\bunny_with_texture\bunny_uv_map1.jpg `
    -FramesBase output\cornellbox `
    examples\frames.cornellbox.stage2.jsonl output\cornellbox_stage2.gif 960 540 25  # stage 2: placed bunny
  examples\render.ps1 -Web -CanvasRenderer -PlacementQuad -AxesLocal `
    -FramesBase output\cornellbox examples\frames.cornellbox.stage1.jsonl '' 960 540 25  # replay stage 1 in the browser
  #   then open http://localhost:8080  (append ?fps=N to tune playback)

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
# -Web/-Wasm bundle. -CanvasRenderer / -OffscreenRenderer sub-select the
# in-browser renderer and therefore apply only to -Web.
$modeCount = @($CLI, $Native, $Web).Where({ $_ }).Count
if ($modeCount -gt 1) { Write-Error 'error: choose only one of -CLI, -Native, -Web.' }
$rendererCount = @($CanvasRenderer, $OffscreenRenderer).Where({ $_ }).Count
if ($rendererCount -gt 1) { Write-Error 'error: choose only one of -CanvasRenderer, -OffscreenRenderer.' }
if ($rendererCount -ge 1 -and -not $Web) { Write-Error 'error: -CanvasRenderer / -OffscreenRenderer apply only to -Web/-Wasm.' }

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
            Write-Error "error: unexpected argument '$tok' (use -Mesh <obj>; content flags are -Wireframe/-Aabb/-Axes/-AxesLocal/-PlacementQuad)."
        }
    }
}

$root = Split-Path -Parent $PSScriptRoot
if (-not $InputPath) { $InputPath = Join-Path $PSScriptRoot 'frames.0.0.2.jsonl' }

# -PlacementQuadColor implies -PlacementQuad (matches render.sh's
# --placement-quad-color).
$quad = [bool]$PlacementQuad -or [bool]$PlacementQuadColor

# -Texture binds a 0.0.4 texture table (sampled albedo) and renders textured. It
# needs a real -Mesh (UVs to sample; the placement quad is added later and does
# not count) and is mutually exclusive with -Wireframe.
if ($Texture) {
    if ($meshes.Count -eq 0) {
        Write-Error 'error: -Texture requires at least one -Mesh (with UVs to sample).'
    }
    if ($Wireframe) {
        Write-Error 'error: -Texture and -Wireframe are mutually exclusive.'
    }
}

# Make the trd toolchain available the way `nix develop` does on Linux.
$devEnv = Join-Path $root 'scripts\dev-env.ps1'
if ((Test-Path $devEnv) -and -not $env:TRD_SKIP_DEV_ENV) {
    . $devEnv -Quiet -NoInstall
}

# Binary-safe concatenation of Arrow IPC files into one stream. render.sh pipes
# the mesh/texture/params producers into a single trd stdin; on Windows (no
# binary-safe pipes) we stage each stage to a temp file and concatenate the bytes
# here, reproducing the exact [mesh][texture][params] byte order trd reads.
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

# DuckDB SQL string literals escape a single quote by doubling it; forward
# slashes work on every platform, so normalise Windows backslashes.
function ConvertTo-SqlPath([string]$p) { ($p -replace "'", "''") -replace '\\', '/' }

$serve = $null
$work = (New-Item -ItemType Directory -Path (Join-Path ([System.IO.Path]::GetTempPath()) "trd-render-$([guid]::NewGuid())")).FullName
try {
    # --- -PlacementQuad: append a canonical colored quad mesh -----------------
    # render.sh's --placement-quad adds an origin-centred, extent-2 unit square
    # (corners +/-1) as the LAST -Mesh, so a stream can draw the reconstructed
    # placement quad as a wireframe overlay. Its +/-1 corners map straight to the
    # camera-space quad; the producer emits a per-frame `draws` entry
    # {mesh: idx, mode: "wireframe"} placing it. -PlacementQuadColor "R G B"
    # (0..1 floats) bakes the outline color into the vertices (wireframe uses them).
    if ($quad) {
        $qr, $qg, $qb = 0, 1, 1
        if ($PlacementQuadColor) {
            $parts = $PlacementQuadColor -split '[,\s]+' | Where-Object { $_ -ne '' }
            if ($parts.Count -ne 3) {
                Write-Error 'error: -PlacementQuadColor expects "R G B" (three 0..1 floats).'
            }
            $qr, $qg, $qb = $parts
        }
        $quadObj = Join-Path $work 'placement_quad.obj'
        @"
# canonical placement-quad overlay (render.ps1 -PlacementQuad): centred, extent 2, corners +/-1.
# 'v x y z r g b' bakes the outline color into the vertices (wireframe uses them).
v -1 -1 0 $qr $qg $qb
v 1 -1 0 $qr $qg $qb
v 1 1 0 $qr $qg $qb
v -1 1 0 $qr $qg $qb
f 1 2 3
f 1 3 4
"@ | Set-Content -Path $quadObj -Encoding ascii
        $meshes += $quadObj
    }

    # --- Fail early if a base tool is missing ---------------------------------
    # cargo is always required. -Web additionally needs wasm-pack + bun to build
    # the bundle; -CLI needs ffmpeg to encode the GIF/WebP. duckdb is optional --
    # if its 'arrow' community extension can't load, pyarrow builds the stream.
    if ($Web) { $required = @('cargo', 'wasm-pack', 'bun') }
    elseif ($Native) { $required = @('cargo') }
    else { $required = @('cargo', 'ffmpeg') }
    foreach ($tool in $required) {
        if (-not (Get-Command $tool -ErrorAction SilentlyContinue)) {
            Write-Error "error: $tool not found on PATH`nOn Windows run '. scripts\dev-env.ps1' first (the flake.nix devShell counterpart); on Linux/macOS use 'nix develop'. -Web also needs wasm-pack + bun."
        }
    }

    # Probe the optional Python-based producers/encoders once (numpy is only
    # needed to encode). uv, when present, supplies pyarrow/numpy on demand.
    $uvOk = [bool](Get-Command uv -ErrorAction SilentlyContinue)
    $pythonOk = [bool](Get-Command python -ErrorAction SilentlyContinue)
    $pyarrowOk = $false
    $pyNumpyOk = $false
    $pyTextureOk = $false
    if ($pythonOk) {
        try { & python -c 'import pyarrow' 2>$null } catch { }
        $pyarrowOk = ($LASTEXITCODE -eq 0)
        try { & python -c 'import pyarrow, numpy' 2>$null } catch { }
        $pyNumpyOk = ($LASTEXITCODE -eq 0)
        try { & python -c 'import pyarrow, PIL, numpy' 2>$null } catch { }
        $pyTextureOk = ($LASTEXITCODE -eq 0)
    }

    # Dolly-camera capstone (#49): examples\bunny_dolly.py authors the 45°
    # bird's-eye dolly camera as two JSONL streams - CG (eye/target/fovy) and CV
    # (K + pose) - that render identically. If the requested InputPath is one of
    # its outputs (frames.bunny_dolly.{cg,cv}.jsonl) and it is not present yet,
    # generate it now via the (pure-stdlib) producer so the demo renders without
    # a manual pre-step.
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
    # up/k/pose/fovy/aspect/znear/zfar) and instanced draw-list (draws) columns, as
    # well as the 0.0.5 background frame reference (frame_path/frame_url). If the
    # input carries any of those, fall back to the pyarrow producer (which emits
    # them) so the camera/draw/frame data actually reaches trd - otherwise an
    # authored camera is lost (identity-camera z-clipping) or the background frame
    # plane never appears.
    if ($producer -eq 'duckdb' -and
        (Select-String -Path $InputPath -Pattern '"(eye|target|direction|up|k|pose|fovy|aspect|znear|zfar|draws|frame_path|frame_url)"\s*:' -Quiet)) {
        if ($uvOk) { $producer = 'uv' }
        elseif ($pyarrowOk) { $producer = 'python' }
        else {
            Write-Error "error: '$InputPath' carries 0.0.3+ camera/draw/frame columns that DuckDB cannot emit;`ninstall uv or a python with pyarrow to render it (run '. scripts\dev-env.ps1')."
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
            Write-Error "error: -Mesh/-PlacementQuad needs uv or a python with pyarrow to encode $($meshes -join ', ').`nrun '. scripts\dev-env.ps1', or 'pip install pyarrow'."
        }
    }

    # -Texture encodes the image into a 0.0.4 texture table via
    # scripts\texture_to_arrow.py, concatenated between the mesh table and the
    # params ([mesh][texture][params]). Needs pyarrow + pillow + numpy; downscaled
    # to --max-size 2048 to stay within the portable (downlevel/WebGL2) limit.
    $textureToArrow = Join-Path $root 'scripts/texture_to_arrow.py'
    $textureProducer = $null
    if ($Texture) {
        if ($uvOk) { $textureProducer = 'uv' }
        elseif ($pyTextureOk) { $textureProducer = 'python' }
        else {
            Write-Error "error: -Texture needs uv or a python with pyarrow + pillow + numpy to encode $Texture.`nrun '. scripts\dev-env.ps1', or 'pip install pyarrow pillow numpy'."
        }
    }

    # encode.py needs pyarrow + numpy. Prefer `uv run` (as render.sh does); fall
    # back to a system `python` that already has both. Only -CLI encodes a GIF.
    if (-not $Native -and -not $Web) {
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

    # --- Build the trd input stream: [mesh?][texture?][params] ----------------
    $framesArrow = Join-Path $work 'frames.arrows'
    $meshArrow = Join-Path $work 'mesh.arrows'
    $textureArrow = Join-Path $work 'texture.arrows'
    $streamArrow = Join-Path $work 'stream.arrows'
    $imagesArrow = Join-Path $work 'images.arrows'

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
    #     [mesh][params] (or [mesh][texture][params] with -Texture). A frame's
    #     `draws` list references these meshes by 0-based index. Without -Mesh,
    #     trd reads the params stream directly.
    if ($meshes.Count -gt 0) {
        if ($meshProducer -eq 'uv') {
            $meshArgs = @('run', '--with', 'pyarrow', $objToArrow) + $meshes + @('-o', $meshArrow)
        }
        else {
            $meshArgs = @($objToArrow) + $meshes + @('-o', $meshArrow)
        }
        $meshGen = Start-Process -FilePath $meshProducer -NoNewWindow -Wait -PassThru -ArgumentList $meshArgs
        if ($meshGen.ExitCode -ne 0) { throw "obj_to_arrow ($meshProducer) failed (exit $($meshGen.ExitCode))" }

        # 1c. -Texture: encode the image into a 0.0.4 texture table and splice it
        #     between the mesh table and the params ([mesh][texture][params]).
        if ($Texture) {
            if ($textureProducer -eq 'uv') {
                $textureArgs = @('run', '--with', 'pyarrow', '--with', 'pillow', '--with', 'numpy', $textureToArrow, $Texture, '--max-size', '2048', '-o', $textureArrow)
            }
            else {
                $textureArgs = @($textureToArrow, $Texture, '--max-size', '2048', '-o', $textureArrow)
            }
            $texGen = Start-Process -FilePath $textureProducer -NoNewWindow -Wait -PassThru -ArgumentList $textureArgs
            if ($texGen.ExitCode -ne 0) { throw "texture_to_arrow ($textureProducer) failed (exit $($texGen.ExitCode))" }
            Join-Files -Parts @($meshArrow, $textureArrow, $framesArrow) -Dest $streamArrow
        }
        else {
            Join-Files -Parts @($meshArrow, $framesArrow) -Dest $streamArrow
        }
        $trdInput = $streamArrow
    }
    else {
        $trdInput = $framesArrow
    }

    # --- Appearance flags (pass through to trd-cli/trd-app and config.json) ----
    $sceneArgs = @()
    if ($Wireframe) { $sceneArgs += '--wireframe' }
    if ($Texture) { $sceneArgs += '--textured' }
    if ($Aabb) { $sceneArgs += '--aabb' }
    if ($Axes) { $sceneArgs += '--axes' }
    if ($AxesLocal) { $sceneArgs += '--axes-local' }
    if ($FramesBase) { $sceneArgs += @('--frames-base', $FramesBase) }

    if ($Web) {
        # --- -Web: build the wasm bundle and serve the SAME scene as -CLI ------
        # Windows-native counterpart of render.sh --web (which builds `nix .#web`
        # and serves it with static-web-server). Build the config-driven bundle
        # with wasm-pack + bun, then drop the runtime inputs the generic renderer
        # (web/src/generic-renderer.ts) fetches at load into web/dist:
        #   stream.arrow  — the identical bytes trd-cli reads on stdin
        #   config.json   — target renderer + scene flags + baked resolution + fps
        #   frames/…      — the 0.0.5 background stills (copied from -FramesBase)
        # A small Bun static server then serves the directory; only ?fps is a live
        # URL override (the resolution is baked into the CV `k`, a positional arg).
        $webDir = Join-Path $root 'web'
        $distDir = Join-Path $webDir 'dist'
        $port = if ($env:PORT) { $env:PORT } else { '8080' }

        # Renderer target: on-screen canvas (default) vs. offscreen texture readback.
        if ($OffscreenRenderer) {
            $target = 'offscreen'
            $rendererLabel = 'ArrowRenderer (offscreen texture -> RGBA readback -> 2D canvas)'
        }
        else {
            $target = 'canvas'
            $rendererLabel = 'CanvasRenderer (on-screen WebGPU surface)'
        }

        # Base mesh mode mirrors the -CLI precedence: textured > wireframe > filled.
        if ($Texture) { $mode = 'textured' }
        elseif ($Wireframe) { $mode = 'wireframe' }
        else { $mode = 'filled' }

        Write-Host 'building trd web (wasm) bundle (wasm-pack + bun)...'
        Push-Location $webDir
        try {
            & bun run build
            if ($LASTEXITCODE -ne 0) { throw "web build failed (exit $LASTEXITCODE)" }
        }
        finally {
            Pop-Location
        }

        Write-Host 'writing web stream.arrow + config.json (same producers as -CLI)...'
        Copy-Item -LiteralPath $trdInput -Destination (Join-Path $distDir 'stream.arrow') -Force
        $config = [ordered]@{
            target        = $target
            mode          = $mode
            showAabb      = [bool]$Aabb
            showAxes      = [bool]$Axes
            showLocalAxes = [bool]$AxesLocal
            background    = [bool]$FramesBase
            width         = $Width
            height        = $Height
            fps           = $Fps
        }
        $config | ConvertTo-Json | Set-Content -Path (Join-Path $distDir 'config.json') -Encoding utf8

        # Background stills: copy the -FramesBase tree into web/dist so each frame's
        # `frame_path` ("frames/frame_xxxxxx.jpg", relative to it) resolves under
        # the served root.
        if ($FramesBase) {
            if (-not (Test-Path $FramesBase)) {
                Write-Error "error: -FramesBase '$FramesBase' not found. Extract the stills first, e.g.`n  uv run --with pyarrow scripts\extract_frames.py <video> --format jpg -o $FramesBase"
            }
            Write-Host "copying background stills from $FramesBase..."
            Copy-Item -Path (Join-Path $FramesBase '*') -Destination $distDir -Recurse -Force
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

        # Small Bun static file server (the no-Nix counterpart of static-web-server,
        # which nix's `.#web` app uses). Bun.file sets content-types, incl. wasm.
        $serveScript = Join-Path $work 'serve.ts'
        @'
const root = process.argv[2];
const port = Number(Bun.env.PORT ?? 8080);
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
        Write-Host "  scene:    mode=$mode aabb=$([bool]$Aabb) axes=$([bool]$Axes) axes-local=$([bool]$AxesLocal) background=$([bool]$FramesBase)"
        Write-Host "  stream:   ${Width}x${Height}, default ${Fps}fps  (override live with ?fps=N)"
        Write-Host ''
        Write-Host "  On this machine:        http://localhost:$port"
        Write-Host "  Direct (same network):  http://${ip}:$port"
        Write-Host ''
        Write-Host '  SSH tunnel (recommended if the port is not directly reachable):'
        Write-Host "    ssh -L ${port}:localhost:$port $user@$ip"
        Write-Host '  then open in a WebGPU browser (Chrome/Edge):'
        Write-Host "                          http://localhost:$port"
        Write-Host ''
        Write-Host '  WebGPU needs a secure context, so open http://localhost:PORT (localhost'
        Write-Host '  qualifies); after a rebuild hard-refresh (Ctrl+Shift+R) to drop the'
        Write-Host '  cached bundle.'
        Write-Host ''

        $env:PORT = $port
        & bun $serveScript $distDir
    }
    elseif ($Native) {
        # Play the frame stream live in the interactive trd-app window
        # (trd-native). It reads the same [mesh][texture][params] stream trd-cli
        # consumes and renders the Scene (meshes + overlays) via trd-core. The
        # appearance flags pass through to trd-app too.
        $appArgs = @(
            'run', '--manifest-path', (Join-Path $root 'Cargo.toml'),
            '-q', '-p', 'trd-app', '--', '--width', $Width, '--height', $Height, '--fps', $Fps
        ) + $sceneArgs
        $app = Start-Process -FilePath 'cargo' -NoNewWindow -Wait -PassThru `
            -ArgumentList $appArgs `
            -RedirectStandardInput $trdInput
        if ($app.ExitCode -ne 0) { throw "trd-app failed (exit $($app.ExitCode))" }
    }
    else {
        # trd renders each row to r,g,b,a fixed_shape_tensor<u8> channels. The
        # Arrow streams are redirected via files so the bytes stay intact.
        $trdArgs = @(
            'run', '--manifest-path', (Join-Path $root 'Cargo.toml'),
            '-q', '-p', 'trd-cli', '--', '--width', $Width, '--height', $Height
        ) + $sceneArgs
        $trd = Start-Process -FilePath 'cargo' -NoNewWindow -Wait -PassThru `
            -ArgumentList $trdArgs `
            -RedirectStandardInput $trdInput -RedirectStandardOutput $imagesArrow
        if ($trd.ExitCode -ne 0) { throw "trd failed (exit $($trd.ExitCode))" }

        # encode.py decodes the tensors and pipes RGBA frames to ffmpeg
        # (.gif or .webp by output extension).
        $enc = Start-Process -FilePath $encoderFile -NoNewWindow -Wait -PassThru `
            -ArgumentList $encoderArgs `
            -RedirectStandardInput $imagesArrow
        if ($enc.ExitCode -ne 0) { throw "encode ($encoderFile) failed (exit $($enc.ExitCode))" }
    }
}
finally {
    Remove-Item -Recurse -Force $work -ErrorAction SilentlyContinue
    if ($serve) { Remove-Item -Recurse -Force $serve -ErrorAction SilentlyContinue }
}

if ($Web) {
    # served synchronously above; nothing to print here.
}
elseif ($Native) {
    Write-Host "streamed $InputPath to the trd-app window (${Width}x${Height}, ${Fps}fps)"
}
else {
    Write-Host "wrote $Output (${Width}x${Height}, ${Fps}fps) from $InputPath"
}
