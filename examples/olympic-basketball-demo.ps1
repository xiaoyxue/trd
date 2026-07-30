#!/usr/bin/env pwsh
# olympic-basketball-demo.ps1 — Windows PowerShell port of
# examples/olympic-basketball-demo.sh (the FIBA / Paris-2024 basketball AR
# can-placement demo).
#
# Renders the "two cans on the court" advertising demo over a real broadcast shot
# of the 2024 Olympic basketball final (France vs USA), for any of three drink
# cans, using the Disney PBR path with the ACES filmic tone-map. It is the
# Windows twin of the .sh: same committed scenes (examples\frames.fiba.stage2.*)
# and helpers (examples\olympic\*.py), same outputs — it just renders through
# examples\render.ps1 on the native Windows GPU (no nixGL; nixGL is Linux-only).
#
# Outputs (in -Outdir, default output\), where <NAME> = heineken | coca | qd:
#   reveal_scan (1920x1080, 24 fps, 576 frames / 24 s = 12 s scan intro + 12 s base):
#     fiba_stage2_<NAME>_pbr_duo_reveal_scan_uffizi_large.mp4            (plain)
#     fiba_stage2_<NAME>_pbr_duo_reveal_scan_fade_uffizi_large.mp4       (fade)
#     fiba_stage2_<NAME>_pbr_duo_reveal_scan_wireframe_uffizi_large.mp4  (wireframe)
#     fiba_stage2_<NAME>_pbr_duo_reveal_scan_wireframe_fade_uffizi_large.mp4
#   dolly (512x512, 24 fps):
#     <NAME>_pbr_dolly_aces.gif
#
# In every reveal_scan variant the lower (sideline) can — which overlaps a player
# — disappears at base frame 91 (12 + 3.8 = 15.8 s) and stays gone: the *fade*
# variants ramp it out over 6 frames, the *plain*/wireframe variants hard-cut it.
#
# Requirements
# ------------
#   * PowerShell 7. On Windows this auto-sources scripts\dev-env.ps1 (the flake.nix
#     devShell counterpart), which resolves/installs cargo, ffmpeg and uv. Set
#     $env:TRD_SKIP_DEV_ENV = '1' to manage the environment yourself.
#   * A GPU that trd can render on (the native Windows path uses it directly).
#   * uv on PATH (dev-env.ps1 installs it via winget) for the pillow/numpy/pyarrow
#     compositing helpers; without uv it falls back to a system `python` that must
#     already have pillow + numpy + pyarrow.
#   * The FIBA background frames. They are NOT vendored (copyrighted footage): the
#     script extracts them from your local shot_0001.mp4 via -Source, into
#     -FramesBase (default output\fiba), unless they already exist there. The
#     dolly output needs no background frames (-What dolly needs no -Source).
#
# Examples
#   pwsh examples\olympic-basketball-demo.ps1 -Can heineken -Source C:\videos\shot_0001.mp4
#   pwsh examples\olympic-basketball-demo.ps1 -Can coca -What dolly
[CmdletBinding()]
param(
    [ValidateSet('heineken', 'coca', 'qd')][string]$Can = 'heineken',
    [ValidateSet('all', 'reveal', 'dolly')][string]$What = 'all',
    [string]$Source = '',
    [string]$FramesBase = 'output/fiba',
    [Alias('Env')][string]$EnvMap = 'assets/envmap/uffizi-large.hdr',
    [string]$Outdir = 'output',
    # Material / lighting overrides (empty => per-can preset / shared ACES default).
    [string]$Metallic = '',
    [string]$Roughness = '',
    [string]$Exposure = '',
    [string]$EnvIntensity = '',
    [string]$Ambient = '',
    [string]$Specular = '',
    [ValidateSet('reinhard', 'aces')][string]$Tonemap = '',
    [switch]$KeepWork,
    [switch]$Help
)

function Show-DemoUsage {
    Write-Host @'
olympic-basketball-demo.ps1 — FIBA basketball AR can-placement demo (Windows).

Usage:
  pwsh examples\olympic-basketball-demo.ps1 [options]

Options:
  -Can NAME           heineken | coca | qd            (default: heineken)
  -What WHAT          all | reveal | dolly            (default: all)
  -Source FILE        shot_0001.mp4 to extract FIBA background frames from
                      (only used if -FramesBase has no frames yet)
  -FramesBase DIR     extracted background plates dir  (default: output\fiba)
  -Env HDR            IBL environment map              (default: assets\envmap\uffizi-large.hdr)
  -Outdir DIR         output directory                 (default: output)
  -Metallic / -Roughness / -Exposure / -EnvIntensity / -Ambient /
  -Specular / -Tonemap VALUE
                      override the can's PBR/lighting preset
  -KeepWork           keep the per-can intermediate work dir
  -Help               show this help

Per-can presets (metallic / roughness); shared ACES lighting is
env-intensity 0.90, exposure 0.45, ambient 0.03, specular 0.6, tonemap aces:
  heineken  0.7 / 0.25      coca  1.0 / 0.30      qd  0.0 / 0.30
'@
}

# Bare invocation (or -Help) -> print guidance and exit 0 (repo convention).
if ($Help -or $PSBoundParameters.Count -eq 0) {
    Show-DemoUsage
    exit 0
}

$ErrorActionPreference = 'Stop'

$S = $PSScriptRoot                          # examples\  (committed scene dir)
$root = Split-Path $S -Parent               # repo / worktree root
Set-Location $root

# Auto-source the Windows dev environment (cargo + ffmpeg + uv), like render.ps1.
$devEnv = Join-Path $root 'scripts\dev-env.ps1'
if ((Test-Path $devEnv) -and -not $env:TRD_SKIP_DEV_ENV) {
    . $devEnv
}

# ---- resolve the can preset ---------------------------------------------
switch ($Can) {
    'heineken' {
        $Name = 'heineken'
        $Mesh = 'assets/meshes/can_hei/source/3d66.com_JCI54557823712.obj'
        $Tex = 'assets/meshes/can_hei/textures/3d66-export-JCI54557823712-003.jpg'
        $DefMetal = '0.7'; $DefRough = '0.25'
    }
    'coca' {
        $Name = 'coca'
        $Mesh = 'assets/meshes/can/coke.obj'
        $Tex = 'assets/meshes/can/can_around.jpg'
        $DefMetal = '1.0'; $DefRough = '0.30'
    }
    'qd' {
        $Name = 'qd'
        $Mesh = 'assets/meshes/qd_beer/source/3d66.com_JDH5455878326.obj'
        $Tex = 'assets/meshes/qd_beer/textures/3d66-export-JDH5455878326-001.jpg'
        $DefMetal = '0.0'; $DefRough = '0.30'
    }
}

# Per-can material + shared ACES lighting (from can_hei_pbr_dolly_aces), overridable.
if (-not $Metallic) { $Metallic = $DefMetal }
if (-not $Roughness) { $Roughness = $DefRough }
if (-not $Exposure) { $Exposure = '0.45' }
if (-not $EnvIntensity) { $EnvIntensity = '0.90' }
if (-not $Ambient) { $Ambient = '0.03' }
if (-not $Specular) { $Specular = '0.6' }
if (-not $Tonemap) { $Tonemap = 'aces' }

# ---- tool + asset checks -------------------------------------------------
$renderPs1 = Join-Path $S 'render.ps1'
foreach ($t in @($renderPs1, $Mesh, $Tex, $EnvMap)) {
    if (-not (Test-Path $t)) { throw "error: missing asset/tool: $t" }
}
foreach ($t in 'cargo', 'ffmpeg') {
    if (-not (Get-Command $t -ErrorAction SilentlyContinue)) {
        throw "error: '$t' not found on PATH. Run '. scripts\dev-env.ps1' first (or install it)."
    }
}
$uv = (Get-Command uv -ErrorAction SilentlyContinue)?.Source
$python = (Get-Command python -ErrorAction SilentlyContinue)?.Source
if (-not $python) { $python = (Get-Command python3 -ErrorAction SilentlyContinue)?.Source }
if (-not $uv -and -not $python) {
    throw "error: need 'uv' (preferred) or a 'python' with pillow + numpy + pyarrow for the compositing helpers."
}

# Run a native command (ffmpeg/uv/python) and fail on a non-zero exit code.
function Invoke-Native {
    param([Parameter(Mandatory)][string]$Exe, [Parameter(ValueFromRemainingArguments)][string[]]$CmdArgs)
    & $Exe @CmdArgs
    if ($LASTEXITCODE -ne 0) { throw "error: $Exe exited $LASTEXITCODE (args: $($CmdArgs -join ' '))" }
}

# Run a python helper with pillow/numpy/pyarrow available (uv preferred).
function Invoke-Py {
    param([Parameter(ValueFromRemainingArguments)][string[]]$PyArgs)
    if ($uv) {
        Invoke-Native $uv run --with pillow --with numpy --with pyarrow python @PyArgs
    }
    else {
        Invoke-Native $python @PyArgs
    }
}

# ---- low-level render: <in> <out> <W> <H> [extra render.ps1 flags...] ----
function Invoke-Render {
    param(
        [Parameter(Mandatory)][string]$InPath,
        [Parameter(Mandatory)][string]$OutPath,
        [Parameter(Mandatory)][int]$W,
        [Parameter(Mandatory)][int]$H,
        [string[]]$Extra = @()
    )
    $renderArgs = @(
        '-CLI', '-Pbr',
        '-Texture', $Tex, '-Env', $EnvMap,
        '-Metallic', $Metallic, '-Roughness', $Roughness, '-EnvIntensity', $EnvIntensity,
        '-Exposure', $Exposure, '-Ambient', $Ambient, '-Specular', $Specular, '-Tonemap', $Tonemap
    ) + $Extra + @('-Mesh', $Mesh, $InPath, $OutPath, "$W", "$H", '24')
    # Invoke render.ps1 as a child `pwsh -File` (command-line parsing), NOT the
    # in-process `& $script @array` call operator: array splatting mis-binds the
    # repeatable -Mesh flag (captured by render.ps1's ValueFromRemainingArguments)
    # into the positional Width. The child inherits our dev-env PATH.
    & pwsh -NoProfile -File $renderPs1 @renderArgs
    if ($LASTEXITCODE -ne 0) { throw "error: render.ps1 failed (exit $LASTEXITCODE) for $OutPath" }
}

$Work = Join-Path $Outdir "_olympic_work/$Name"
New-Item -ItemType Directory -Force -Path $Outdir | Out-Null

function Get-BackgroundFrames {
    $first = Join-Path $FramesBase 'frames/frame_000000.jpg'
    if (Test-Path $first) { return }
    if (-not $Source) {
        throw "error: no frames at $FramesBase\frames\ and no -Source given.`n" +
        "       Provide the broadcast clip: -Source C:\path\to\shot_0001.mp4"
    }
    Write-Host "### extracting FIBA background frames from $Source -> $FramesBase"
    Invoke-Py (Join-Path $root 'scripts\extract_frames.py') $Source -o $FramesBase --format jpg --no-arrow
    if (-not (Test-Path $first)) {
        throw "error: frame extraction did not produce $first"
    }
}

function Invoke-Encode {
    # encode a f%04d.png sequence (start 0) into an H.264 mp4.
    param([string]$Dir, [string]$OutMp4)
    Invoke-Native ffmpeg -y -framerate 24 -start_number 0 -i (Join-Path $Dir 'f%04d.png') `
        -c:v libx264 -pix_fmt yuv420p -crf 18 $OutMp4
}

function Invoke-Concat {
    # concatenate two mp4s (same size) into one.
    param([string]$A, [string]$B, [string]$OutMp4)
    Invoke-Native ffmpeg -y -i $A -i $B -filter_complex '[0:v][1:v]concat=n=2:v=1[v]' `
        -map '[v]' -c:v libx264 -pix_fmt yuv420p -crf 18 $OutMp4
}

function Invoke-RevealScan {
    Get-BackgroundFrames
    $orig = Join-Path $FramesBase 'frames/frame_000000.jpg'
    $base = Join-Path $Work 'duo_base.mp4'
    $dc = Join-Path $S 'frames.fiba.stage2.can_duo.jsonl'
    if (Test-Path $Work) { Remove-Item -Recurse -Force $Work }
    New-Item -ItemType Directory -Force -Path $Work | Out-Null

    Write-Host "### [$Name] 1/6 clean duo base (288f, both cans)"
    Invoke-Render $dc $base 1920 1080 -Extra @('-FramesBase', $FramesBase)

    Write-Host "### [$Name] 2/6 upper-only base (lower can removed)"
    Invoke-Py (Join-Path $S 'olympic\upper_only.py') $dc (Join-Path $Work 'duo_upper.jsonl')
    Invoke-Render (Join-Path $Work 'duo_upper.jsonl') (Join-Path $Work 'duo_upper.mp4') 1920 1080 -Extra @('-FramesBase', $FramesBase)

    Write-Host "### [$Name] 3/6 gizmo stills (solid + wireframe, frame 0)"
    Invoke-Render (Join-Path $S 'frames.fiba.stage2.can_duo.gizmo_f0.jsonl') (Join-Path $Work 'gizmo.mp4') `
        1920 1080 -Extra @('-FramesBase', $FramesBase, '-PlacementQuad', '-Aabb', '-AxesLocal', '-GridLocal', 'xy')
    Invoke-Render (Join-Path $S 'frames.fiba.stage2.can_duo.gizmo_wire_f0.jsonl') (Join-Path $Work 'gizmowire.mp4') `
        1920 1080 -Extra @('-FramesBase', $FramesBase, '-PlacementQuad', '-Aabb', '-AxesLocal', '-GridLocal', 'xy', '-GridMesh', '1')
    Invoke-Native ffmpeg -y -i (Join-Path $Work 'gizmo.mp4') -frames:v 1 (Join-Path $Work 'gizmo0.png')
    Invoke-Native ffmpeg -y -i (Join-Path $Work 'gizmowire.mp4') -frames:v 1 (Join-Path $Work 'gizmowire0.png')

    Write-Host "### [$Name] 4/6 composite fade + cut bases (lower can gone @ frame 91)"
    foreach ($d in 'cb', 'up', 'fadeb', 'cutb') {
        $p = Join-Path $Work $d
        if (Test-Path $p) { Remove-Item -Recurse -Force $p }
        New-Item -ItemType Directory -Force -Path $p | Out-Null
    }
    Invoke-Native ffmpeg -y -i $base -start_number 0 (Join-Path $Work 'cb/c%04d.png')
    Invoke-Native ffmpeg -y -i (Join-Path $Work 'duo_upper.mp4') -start_number 0 (Join-Path $Work 'up/u%04d.png')
    Invoke-Py (Join-Path $S 'olympic\composite_bases.py') (Join-Path $Work 'cb') (Join-Path $Work 'up') (Join-Path $Work 'fadeb') (Join-Path $Work 'cutb')
    Invoke-Encode (Join-Path $Work 'fadeb') (Join-Path $Work 'fade_base.mp4')
    Invoke-Encode (Join-Path $Work 'cutb') (Join-Path $Work 'cut_base.mp4')

    Write-Host "### [$Name] 5/6 scan intros (solid + wireframe, 288f each)"
    Copy-Item (Join-Path $Work 'cb/c0000.png') (Join-Path $Work 'base0.png') -Force
    foreach ($d in 'intro', 'introwire') {
        $p = Join-Path $Work $d
        if (Test-Path $p) { Remove-Item -Recurse -Force $p }
    }
    Invoke-Py (Join-Path $S 'olympic\scan_intro.py') $orig (Join-Path $Work 'base0.png') (Join-Path $Work 'gizmo0.png') (Join-Path $Work 'intro') 72 12 68 6
    Invoke-Py (Join-Path $S 'olympic\scan_intro.py') $orig (Join-Path $Work 'base0.png') (Join-Path $Work 'gizmowire0.png') (Join-Path $Work 'introwire') 72 12 68 6
    Invoke-Native ffmpeg -y -framerate 24 -i (Join-Path $Work 'intro/f%04d.png') -c:v libx264 -pix_fmt yuv420p -crf 18 (Join-Path $Work 'intro.mp4')
    Invoke-Native ffmpeg -y -framerate 24 -i (Join-Path $Work 'introwire/f%04d.png') -c:v libx264 -pix_fmt yuv420p -crf 18 (Join-Path $Work 'introwire.mp4')

    Write-Host "### [$Name] 6/6 concat 4 finals"
    $pre = Join-Path $Outdir "fiba_stage2_${Name}_pbr_duo_reveal_scan"
    Invoke-Concat (Join-Path $Work 'intro.mp4') (Join-Path $Work 'cut_base.mp4') "${pre}_uffizi_large.mp4"
    Invoke-Concat (Join-Path $Work 'intro.mp4') (Join-Path $Work 'fade_base.mp4') "${pre}_fade_uffizi_large.mp4"
    Invoke-Concat (Join-Path $Work 'introwire.mp4') (Join-Path $Work 'cut_base.mp4') "${pre}_wireframe_uffizi_large.mp4"
    Invoke-Concat (Join-Path $Work 'introwire.mp4') (Join-Path $Work 'fade_base.mp4') "${pre}_wireframe_fade_uffizi_large.mp4"
}

function Invoke-Dolly {
    Write-Host "### [$Name] dolly ACES turntable (512x512)"
    Invoke-Render (Join-Path $S 'frames.bunny_dolly.cg.jsonl') (Join-Path $Outdir "${Name}_pbr_dolly_aces.gif") 512 512
}

Write-Host "== olympic-basketball-demo: can=$Name  what=$What  env=$([System.IO.Path]::GetFileName($EnvMap))"
Write-Host "   material: metallic=$Metallic roughness=$Roughness exposure=$Exposure"
Write-Host "             env-intensity=$EnvIntensity ambient=$Ambient specular=$Specular tonemap=$Tonemap"

switch ($What) {
    'all' { Invoke-RevealScan; Invoke-Dolly }
    'reveal' { Invoke-RevealScan }
    'dolly' { Invoke-Dolly }
}

if (-not $KeepWork -and (Test-Path $Work)) { Remove-Item -Recurse -Force $Work }

Write-Host "== done. outputs in $Outdir\:"
Get-ChildItem -Path (Join-Path $Outdir "fiba_stage2_${Name}_pbr_duo_reveal_scan*_uffizi_large.mp4"), `
    (Join-Path $Outdir "${Name}_pbr_dolly_aces.gif") -ErrorAction SilentlyContinue |
    Select-Object -ExpandProperty Name
