#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Prepare a Windows development environment for trd - the counterpart to
    flake.nix's devShell on Linux/macOS.

.DESCRIPTION
    Dot-source this script to set up the CURRENT PowerShell 7 session so the
    trd toolchain works without Nix:

        . .\scripts\dev-env.ps1

    Like `nix develop`, it makes the toolchain discoverable on PATH for the
    shell you dot-source it into. It handles:

      * cargo / rustc      - rustup's ~\.cargo\bin, pinned to the MSVC host
                             (wgpu's raw-dylib deps crash on the -gnu host).
      * the MSVC toolchain - cl.exe / link.exe, imported from vcvars64.bat, so
                             cargo can link native binaries.
      * ffmpeg, uv         - the extra tools examples\render.ps1 needs.

    Running it directly (`.\scripts\dev-env.ps1`) only validates/reports and
    will try to install missing tools; PATH changes will NOT persist unless the
    script is dot-sourced.

.PARAMETER Quiet
    Suppress the informational summary (used when render.ps1 sources this).

.PARAMETER NoInstall
    Only discover tools already present; never attempt to install anything.

.EXAMPLE
    . .\scripts\dev-env.ps1
    cargo build -p trd-cli
    examples\render.ps1
#>
[CmdletBinding()]
param(
    [switch]$Quiet,
    [switch]$NoInstall
)

Set-StrictMode -Version Latest
# Probe native tools by exit code below; don't let non-zero exits throw.
$PSNativeCommandUseErrorActionPreference = $false

function Write-DevInfo([string]$m) { if (-not $Quiet) { Write-Host "trd dev-env: $m" } }
function Write-DevWarn([string]$m) { Write-Warning "trd dev-env: $m" }

function Test-Tool([string]$name) { [bool](Get-Command $name -ErrorAction SilentlyContinue) }

function Add-PathPrefix([string]$dir) {
    if ($dir -and (Test-Path $dir) -and (($env:Path -split ';') -notcontains $dir)) {
        $env:Path = "$dir;$env:Path"
    }
}

# Resolve a tool: return $true if on PATH, else prepend the first existing
# candidate's directory and re-check.
function Resolve-Tool {
    param([string]$Name, [string[]]$Candidates)
    if (Test-Tool $Name) { return $true }
    foreach ($exe in $Candidates) {
        if ($exe -and (Test-Path $exe)) {
            Add-PathPrefix (Split-Path -Parent $exe)
            if (Test-Tool $Name) { return $true }
        }
    }
    return (Test-Tool $Name)
}

$cargoBin    = Join-Path $env:USERPROFILE '.cargo\bin'
$wingetLinks = Join-Path $env:LOCALAPPDATA 'Microsoft\WinGet\Links'
$scoopShims  = Join-Path $env:USERPROFILE 'scoop\shims'
$localBin    = Join-Path $env:USERPROFILE '.local\bin'

# --- Rust (cargo / rustc) -------------------------------------------------
Resolve-Tool 'cargo'  @((Join-Path $cargoBin 'cargo.exe'))  | Out-Null
Resolve-Tool 'rustup' @((Join-Path $cargoBin 'rustup.exe')) | Out-Null

if (Test-Tool 'rustup') {
    # rust-toolchain.toml pins a bare "stable" channel, so the host defaults to
    # whatever rustup was installed with. Force the MSVC host: the -gnu host
    # needs dlltool for wgpu's raw-dylib deps and crashes at runtime (0xC0000005).
    $activeToolchain = (& rustup show active-toolchain) 2>$null
    if ($activeToolchain -notmatch 'msvc') {
        Write-DevInfo 'pinning rustup default host to x86_64-pc-windows-msvc'
        & rustup set default-host x86_64-pc-windows-msvc | Out-Null
    }
}

# --- MSVC toolchain (cl.exe / link.exe via vcvars64) ----------------------
if (Test-Tool 'cl') {
    Write-DevInfo 'MSVC toolchain already on PATH'
}
else {
    $installer = 'C:\Program Files (x86)\Microsoft Visual Studio\Installer'
    $vswhere = Join-Path $installer 'vswhere.exe'
    $vcvars = $null
    if (Test-Path $vswhere) {
        # -requires filters out non-C++ VS shells (e.g. SSMS) that -latest may pick.
        $vsPath = & $vswhere -latest -products * `
            -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 `
            -property installationPath 2>$null
        if ($vsPath) {
            $candidate = Join-Path $vsPath 'VC\Auxiliary\Build\vcvars64.bat'
            if (Test-Path $candidate) { $vcvars = $candidate }
        }
    }
    if (-not $vcvars) {
        $vcvars = Get-ChildItem `
            'C:\Program Files\Microsoft Visual Studio', `
            'C:\Program Files (x86)\Microsoft Visual Studio' `
            -Filter vcvars64.bat -Recurse -ErrorAction SilentlyContinue |
        Select-Object -First 1 -ExpandProperty FullName
    }
    if ($vcvars) {
        Write-DevInfo "importing MSVC environment from $vcvars"
        # vcvars64.bat calls vswhere, which lives in the Installer dir; prepend
        # it to PATH first, then import the resulting environment (PATH, INCLUDE,
        # LIB, ...) back into this session.
        $lines = & $env:ComSpec /c "set `"PATH=$installer;%PATH%`" && call `"$vcvars`" >nul 2>&1 && set"
        foreach ($line in $lines) {
            if ($line -match '^([^=]+)=(.*)$') {
                Set-Item -Path "Env:$($matches[1])" -Value $matches[2]
            }
        }
    }
    else {
        Write-DevWarn "MSVC C++ build tools not found; cargo cannot link native binaries. Install 'Desktop development with C++' via the Visual Studio Installer."
    }
}

# --- Extra tools for examples\render.ps1 ----------------------------------
Resolve-Tool 'ffmpeg' @(
    'C:\Tools\ffmpeg\bin\ffmpeg.exe',
    (Join-Path $wingetLinks 'ffmpeg.exe'),
    (Join-Path $scoopShims 'ffmpeg.exe')
) | Out-Null

# Honor uv's own UV_INSTALL_DIR (its official custom install-dir env var) first,
# so a uv installed off the default paths (e.g. on another drive) is still found.
$uvInstallDir = $env:UV_INSTALL_DIR
$uvOk = Resolve-Tool 'uv' @(
    ($(if ($uvInstallDir) { Join-Path $uvInstallDir 'uv.exe' })),
    (Join-Path $localBin 'uv.exe'),
    (Join-Path $cargoBin 'uv.exe'),
    (Join-Path $wingetLinks 'uv.exe'),
    (Join-Path $scoopShims 'uv.exe')
)
if (-not $uvOk -and -not $NoInstall -and (Test-Tool 'winget')) {
    Write-DevInfo 'uv not found; installing via winget'
    & winget install --id astral-sh.uv -e --source winget `
        --accept-source-agreements --accept-package-agreements --disable-interactivity 2>&1 | Out-Null
    $uvOk = Resolve-Tool 'uv' @(
        (Join-Path $localBin 'uv.exe'),
        (Join-Path $wingetLinks 'uv.exe')
    )
}

# --- Report ---------------------------------------------------------------
$hints = @{
    cargo  = 'install rustup from https://rustup.rs'
    ffmpeg = 'winget install --id Gyan.FFmpeg -e'
}
foreach ($t in 'cargo', 'ffmpeg') {
    if (-not (Test-Tool $t)) { Write-DevWarn "$t not found. Install: $($hints[$t])" }
}
if (-not (Test-Tool 'uv')) {
    Write-DevWarn "uv not found. Install: winget install --id astral-sh.uv -e (or: pip install uv). examples\render.ps1 falls back to a system 'python' with pyarrow + numpy."
}

if (-not $Quiet) {
    $rustVer = if (Test-Tool 'rustc') { (& rustc --version) } else { '(missing)' }
    Write-Host "trd dev-env ready: $rustVer"
    foreach ($t in 'cargo', 'cl', 'ffmpeg', 'uv') {
        $cmd = Get-Command $t -ErrorAction SilentlyContinue
        $src = if ($cmd) { $cmd.Source } else { '(missing)' }
        Write-Host ('  {0,-8} {1}' -f $t, $src)
    }
    if ($MyInvocation.InvocationName -ne '.') {
        Write-Warning 'Not dot-sourced: PATH changes will not persist. Re-run as:  . .\scripts\dev-env.ps1'
    }
}
