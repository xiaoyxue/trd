#!/usr/bin/env pwsh
# Render a trd JSONL frame-parameter file to an animated GIF/WebP, play it live in
# the native window, or replay it in a WebGPU browser (PowerShell 7).
#
# Windows-native port of examples/render.sh with the SAME behaviour and flags:
#   JSONL --(pyarrow: Arrow IPC)--------> trd --(tensors)--> ffmpeg   (-CLI)
#   JSONL --(pyarrow: Arrow IPC)--------> trd-app window              (-Native)
#   JSONL --(pyarrow: Arrow IPC)--------> stream.arrow + config.json  (-Web)
#                                         served to the generic web renderer
#
# Unlike render.sh (which pipes everything with no intermediate files), Windows
# PowerShell pipelines are not binary-safe, so the Arrow IPC stages are handed off
# through temporary files (created in a temp dir and auto-removed). The produced
# GIF/WebP is identical.
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
#                       [-FramesBase DIR] [-FramesTable FILE] [-InputPath INPUT.jsonl] `
#                       [-Output OUTPUT.gif|.webp] [-Width 256] [-Height 256] [-Fps 30]
#   examples/render.ps1 INPUT.jsonl OUTPUT.gif 256 256 30   # positional
# Defaults: examples/frames.bunny_dolly.cg.jsonl (renders assets/meshes/bunny.obj)  output/out.gif  256 256 30
# Run with no arguments (or -Help) to print the flag guidance and exit; pass -CLI
# to render the default demo (the bunny dolly camera capstone).
#
# By default (or with -CLI, alias -Headless) the frame stream is rendered to a
# GIF/WebP via the headless trd-cli.
# With -Native (alias -App) it is played live in the interactive trd-app window
# (trd-native); -Output is then ignored and neither uv nor ffmpeg are needed.
# With -Web (alias -Wasm) it renders the SAME scene as -CLI, but in a WebGPU
# browser: it builds the config-driven web bundle, writes stream.arrow (the
# identical bytes trd-cli reads on stdin) + config.json (renderer target + scene
# flags + baked resolution + default fps) + the -FramesBase stills into
# web/viewer/dist,
# and serves it so the browser replays exactly what -CLI would render.
# The content flags below (-Mesh/-Texture/-Wireframe/-Aabb/-Axes/-AxesLocal/
# -PlacementQuad/-FramesBase) and the positional Width/Height apply to all three
# modes (trd-cli, trd-app and the web renderer share trd-core). Only the playback
# rate is a live URL param for -Web: append ?fps=N.
#
# The input document is params-first with self-contained GLB resources.
# scripts\scene_to_arrow.py converts OBJ/albedo offline, retaining the old
# preview fit without changing the camera/model params. When no -Mesh is given, the
# bunny (assets\meshes\bunny.obj) is loaded as the default demo object. Try:
# examples\render.ps1 -CLI -Mesh assets\meshes\bunny.obj `
# examples\frames.turntable.jsonl output\bunny.gif. -Mesh is repeatable: pass it
# several times to load several meshes (one table row each, in order); a frame's
# `draws` list then references them by 0-based index. Two-mesh demo:
# examples\render.ps1 -CLI -Wireframe -Mesh assets\meshes\bunny.obj `
# -Mesh examples\cube.obj examples\frames.multimesh.jsonl output\scene.gif.
# (-Mesh needs pyarrow, numpy and Pillow via uv/python.)
# With -Texture IMG the converted GLB embeds IMG as its albedo and
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
# With -FramesBase DIR trd composites each external background still (its
# `frame_path`, relative to DIR) BENEATH the scene via a FramePlane (#63), decoded
# at full resolution and sampled down to the surface. Extract the stills first:
#   uv run --with pyarrow scripts\extract_frames.py `
#     assets\videos\cornellbox\CameraMovement.mp4 --format jpg -o output\cornellbox
# (optionally add --height H to extract smaller stills and save memory).
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
# prerequisites) to put cargo, the MSVC linker, ffmpeg and uv on PATH;
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
    # --- Disney PBR shading (-CLI and -Native; mutually exclusive with -Wireframe/
    # -Textured at the trd-cli/trd-app layer). Numeric values pass straight through
    # to trd's --metallic/--roughness/… f32 flags; defaults mirror render.sh.
    [switch]$Pbr,
    [string]$Env,
    [switch]$EnvBackground,
    [string]$EnvBackgroundBlur = '0.0',
    [string]$Metallic = '0.0',
    [string]$Roughness = '0.35',
    [string]$EnvIntensity = '1.0',
    [string]$Exposure = '1.2',
    [string]$Ambient = '0.12',
    [string]$Specular = '0.5',
    [string]$Clearcoat = '0.0',
    [ValidateSet('reinhard', 'aces')][string]$Tonemap = 'reinhard',
    # --- Local coordinate-plane grid (#110/#114), scoped to wireframe draws.
    [ValidateSet('xy', 'xz', 'yz')][string]$GridLocal,
    [string]$GridMesh,
    [string]$FramesBase,
    [string]$FramesTable,
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

Defaults: InputPath=examples\frames.bunny_dolly.cg.jsonl (renders assets\meshes\bunny.obj)  Output=output\out.gif  Width=256  Height=256  Fps=30

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
  -Mesh OBJ         Load OBJ as a mesh table entry (centered + scaled to fit).
                    Repeatable: pass several times to load several meshes (row 0,
                    1, ...); a frame's `draws` list references them by index.
                    Defaults to assets\meshes\bunny.obj when no mesh is given.
  -Texture IMG      Embed IMG in the first GLB and render textured - sampling it
                    at each vertex UV (#20). Requires -Mesh (with UVs); mutually
                    exclusive with -Wireframe.
  -Wireframe        Draw mesh edges as a line list instead of filled triangles (#38).
  -Aabb             Overlay each mesh's axis-aligned bounding box as a green box (#42).
  -Axes             Overlay a coordinate-axes gizmo (X=red, Y=green, Z=blue) at the origin (#42).
  -AxesLocal        Overlay a coordinate-axes gizmo at EACH drawn object's own local frame (#77).
  -GridLocal PLANE  Overlay a coordinate-plane grid lattice (PLANE = xy|xz|yz) on each
                    WIREFRAME drawn object's own local frame (#110). Scoped to wireframe
                    draws, so a filled/textured mesh gets no grid.
  -GridMesh ID      Narrow -GridLocal to draws of mesh ID only (e.g. the placement quad),
                    so a wireframe content mesh doesn't also pick up a floor grid (#114).
                    Ignored without -GridLocal.
  -PlacementQuad    Append a canonical colored quad mesh (origin-centred, extent 2)
                    as the last -Mesh, drawn as a wireframe placement overlay (#77).
  -PlacementQuadColor "R G B"
                    Tint the placement quad (0..1 floats; default cyan). Implies -PlacementQuad.
  -FramesBase DIR   Resolve each external background still (`frame_path`,
                    relative to DIR) beneath the scene via a FramePlane (#63).
  -FramesTable FILE Retired. Use external frame_path/frame_url and -FramesBase.

PBR SHADING (-CLI and -Native; the Disney principled BRDF, #112):
  -Pbr              Shade the bound albedo with the Disney BRDF instead of flat texturing.
                    Replaces -Textured (mutually exclusive); still needs a -Texture for
                    the albedo. Use -Metallic 1 -Roughness 0.3 for a shiny metal look.
  -Env HDR          Equirectangular HDR environment map (.hdr) reflected by metallic
                    surfaces (decoded here; downscaled to 2048px). Only used with -Pbr.
  -EnvBackground    Also draw the -Env probe as the frame's background sky, behind every
                    primitive (tone-mapped with -Exposure/-Tonemap). Needs -Env.
  -EnvBackgroundBlur N
                    Blur of the -EnvBackground sky, 0 = sharp … 1 = fully blurred (default 0.0)
  -Metallic N       0 = dielectric, 1 = metal          (default 0.0)
  -Roughness N      0 = mirror, 1 = fully rough         (default 0.35)
  -EnvIntensity N   Env-map reflection gain (0 = off)   (default 1.0)
  -Exposure N       Tone-map exposure                   (default 1.2)
  -Ambient N        Constant ambient fill (× base)      (default 0.12)
  -Specular N       Dielectric specular strength        (default 0.5)
  -Clearcoat N      Clearcoat lobe strength             (default 0.0)
  -Tonemap OP       Tone-map operator: reinhard (default) | aces (filmic, softer highlight
                    roll-off + better hue retention on bright albedo, #116).

  -Help             Show this guidance and exit.

Examples:
  examples\render.ps1 -CLI                                    # default demo -> output\out.gif
  examples\render.ps1 -Native                                # play the default demo live
  examples\render.ps1 -CLI -Aabb -Mesh assets\meshes\bunny.obj `
    examples\frames.turntable.jsonl output\bunny.gif 1024 1024 24
  examples\render.ps1 -CLI -Mesh assets\meshes\bunny_with_texture\bunny.obj `
    -Texture assets\meshes\bunny_with_texture\bunny_uv_map1.jpg `
    examples\frames.bunny_dolly.cv.jsonl output\bunny_textured.gif 1024 1024 24  # textured dolly (#20)
  examples\render.ps1 -CLI -Pbr -Tonemap aces -Metallic 1.0 -Roughness 0.30 `
    -Env assets\envmap\uffizi-large.hdr `
    -Mesh assets\meshes\can\coke.obj -Texture assets\meshes\can\can_around.jpg `
    examples\frames.bunny_dolly.cg.jsonl output\can_pbr_aces.gif 512 512 24  # metallic can, ACES tone-map (#116)
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
# (mesh 0 = first -Mesh). Each converted GLB becomes one resource row;
# a CG frame's `draws` list references them by
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
if (-not $InputPath) { $InputPath = Join-Path $PSScriptRoot 'frames.bunny_dolly.cg.jsonl' }

# -PlacementQuadColor implies -PlacementQuad (matches render.sh's
# --placement-quad-color).
$quad = [bool]$PlacementQuad -or [bool]$PlacementQuadColor

# -Texture embeds sampled albedo in the first converted GLB. It
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

# -Pbr renders the bound albedo with the Disney principled BRDF (a virtual light
# rig + smooth normals + optional -Env HDR reflection). It needs a -Texture (the
# albedo) and is mutually exclusive with -Wireframe/-Textured (parity with
# render.sh's --pbr guard).
if ($Pbr) {
    if (-not $Texture) {
        Write-Error 'error: -Pbr requires a -Texture (the albedo to shade).'
    }
    if ($Wireframe) {
        Write-Error 'error: -Pbr and -Wireframe are mutually exclusive.'
    }
}

# Make the trd toolchain available the way `nix develop` does on Linux.
$devEnv = Join-Path $root 'scripts\dev-env.ps1'
if ((Test-Path $devEnv) -and -not $env:TRD_SKIP_DEV_ENV) {
    . $devEnv -Quiet -NoInstall
}

if ($FramesTable) {
    throw '-FramesTable is retired. Use frame_path/frame_url params and -FramesBase for external images.'
}

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

    # Keep the demo's default bunny, converted offline to a self-contained GLB.
    if ($meshes.Count -eq 0) {
        $meshes += (Join-Path $root 'assets/meshes/bunny.obj')
    }

    # --- Fail early if a base tool is missing ---------------------------------
    # cargo is always required. -Web additionally needs wasm-pack + bun to build
    # the bundle; -CLI needs ffmpeg to encode the GIF/WebP. pyarrow (via uv/python)
    # builds the Arrow stream.
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

    # Params stay unchanged; the bundler converts OBJ/albedo into GLB resources.
    $jsonlToArrow = Join-Path $root 'scripts/jsonl_to_arrow.py'
    $producer = $null
    if ($uvOk) { $producer = 'uv' }
    elseif ($pyTextureOk) { $producer = 'python' }
    else {
        Write-Error "error: need uv or python with pyarrow, numpy and Pillow to build params/GLB input."
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

    # --- Build the trd input document: [params][mesh] ------------------------
    $framesArrow = Join-Path $work 'frames.arrows'
    $streamArrow = Join-Path $work 'stream.arrows'
    $imagesArrow = Join-Path $work 'images.arrows'

    # 1. Build a streaming Arrow IPC file of frame params from the JSONL via
    #    scripts\jsonl_to_arrow.py (pyarrow): the always-present `model` column
    #    (FixedSizeList<f32>[16], column-major - a row's explicit matrix, else
    #    identity) plus optional camera / draws / frame_path columns.
    if ($producer -eq 'uv') {
        $genArgs = @('run', '--with', 'pyarrow', $jsonlToArrow, $InputPath, '-o', $framesArrow)
    }
    else {
        $genArgs = @($jsonlToArrow, $InputPath, '-o', $framesArrow)
    }
    $genArgs += @('--fps', $Fps)
    & $producer @genArgs
    if ($LASTEXITCODE -ne 0) { throw "jsonl_to_arrow ($producer) failed (exit $LASTEXITCODE)" }
    $sceneToArrow = Join-Path $root 'scripts\scene_to_arrow.py'
    $bundleArgs = if ($uvOk) {
        @('run', '--with', 'pyarrow', '--with', 'numpy', '--with', 'pillow', $sceneToArrow)
    } else {
        @($sceneToArrow)
    }
    $bundleArgs += @('--params', $framesArrow, '-o', $streamArrow)
    foreach ($mesh in $meshes) { $bundleArgs += @('--mesh', $mesh) }
    if ($Texture) { $bundleArgs += @('--texture', $Texture) }
    & $producer @bundleArgs
    if ($LASTEXITCODE -ne 0) { throw "scene_to_arrow ($producer) failed (exit $LASTEXITCODE)" }
    $trdInput = $streamArrow

    # --- Appearance flags (pass through to trd-cli/trd-app and config.json) ----
    $sceneArgs = @()
    if ($Wireframe) { $sceneArgs += '--wireframe' }
    # --pbr shades the same bound albedo with the Disney BRDF; it replaces
    # --textured (mutually exclusive at the trd-cli/trd-app layer) and forwards the
    # material + optional HDR env probe. The albedo is embedded in the GLB.
    if ($Texture -and -not $Pbr) { $sceneArgs += '--textured' }
    if ($Pbr) {
        $sceneArgs += @(
            '--pbr',
            '--metallic', $Metallic,
            '--roughness', $Roughness,
            '--env-intensity', $EnvIntensity,
            '--exposure', $Exposure,
            '--ambient', $Ambient,
            '--specular', $Specular,
            '--clearcoat', $Clearcoat,
            '--tonemap', $Tonemap
        )
        if ($Env) { $sceneArgs += @('--env', $Env) }
    }
    # The HDR sky is a scene background, not a material, so it is forwarded on
    # its own — with or without -Pbr (#235 R2). It needs -Env to show anything.
    if ($EnvBackground) {
        $sceneArgs += @('--env-background', '--env-background-blur', $EnvBackgroundBlur)
        if (-not $Pbr -and $Env) {
            $sceneArgs += @('--env', $Env, '--exposure', $Exposure, '--tonemap', $Tonemap)
        }
    }
    if ($Aabb) { $sceneArgs += '--aabb' }
    if ($Axes) { $sceneArgs += '--axes' }
    if ($AxesLocal) { $sceneArgs += '--axes-local' }
    if ($GridLocal) { $sceneArgs += @('--grid-local', $GridLocal) }
    if ($GridMesh) { $sceneArgs += @('--grid-mesh', $GridMesh) }
    if ($FramesBase) { $sceneArgs += @('--frames-base', $FramesBase) }

    if ($Web) {
        # --- -Web: build the wasm bundle and serve the SAME scene as -CLI ------
        # Windows-native counterpart of render.sh --web (which builds `nix .#web`
        # and serves it with static-web-server). Build the config-driven bundle
        # with wasm-pack + bun, then drop the runtime inputs the generic viewer
        # (web/viewer/src/viewer.ts) fetches at load into web/viewer/dist:
        #   stream.arrow  — the identical bytes trd-cli reads on stdin
        #   config.json   — target renderer + scene flags + baked resolution + fps
        #   frames/…      — external background stills (copied from -FramesBase)
        # A small Bun static server then serves the directory; only ?fps is a live
        # URL override (the resolution is baked into the CV `k`, a positional arg).
        $webDir = Join-Path $root 'web'
        $distDir = Join-Path $webDir 'viewer\dist'
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

        # Base mesh mode mirrors the -CLI precedence: pbr > textured > wireframe
        # > filled. -Pbr shades the bound albedo with the Disney BRDF (same as
        # trd-cli/trd-app --pbr), so it takes precedence over plain texturing.
        if ($Pbr) { $mode = 'pbr' }
        elseif ($Texture) { $mode = 'textured' }
        elseif ($Wireframe) { $mode = 'wireframe' }
        else { $mode = 'filled' }

        Write-Host 'building trd web (wasm) bundle (wasm-pack + bun)...'
        Push-Location $webDir
        try {
            & bun run build:viewer
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
        # -Pbr: forward the Disney material (byte-identical to the trd-cli/trd-app
        # --pbr flags) and, if -Env is set, copy the .hdr probe into the served
        # root so the browser fetches + decodes it in-wasm (trd-core does no I/O).
        if ($Pbr) {
            $config['pbr'] = [ordered]@{
                metallic     = [double]$Metallic
                roughness    = [double]$Roughness
                specular     = [double]$Specular
                clearcoat    = [double]$Clearcoat
                envIntensity = [double]$EnvIntensity
                exposure     = [double]$Exposure
                ambient      = [double]$Ambient
                tonemap      = $Tonemap
            }
            if ($Env) {
                Copy-Item -LiteralPath $Env -Destination (Join-Path $distDir 'env.hdr') -Force
                $config['env'] = 'env.hdr'
            }
        }
        # The sky follows the material's tone mapping but is a scene background
        # (#235 R2); the browser loads the probe as part of the PBR setup, so it
        # needs -Pbr -Env here.
        if ($EnvBackground) {
            if ($Pbr -and $Env) {
                $config['envBackground'] = [ordered]@{ blur = [double]$EnvBackgroundBlur }
            }
            else {
                Write-Warning '-EnvBackground needs -Pbr -Env in -Web mode; skipping the sky'
            }
        }
        $config | ConvertTo-Json -Depth 5 | Set-Content -Path (Join-Path $distDir 'config.json') -Encoding utf8

        # Background stills: copy the -FramesBase tree into web/viewer/dist so each frame's
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
        Write-Host "  scene:    mode=$mode aabb=$([bool]$Aabb) axes=$([bool]$Axes) axes-local=$([bool]$AxesLocal) external-background=$([bool]$FramesBase)"
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
        # (trd-native). It reads the same [params][mesh] document trd-cli
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
