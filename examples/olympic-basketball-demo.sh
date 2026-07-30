#!/usr/bin/env bash
# olympic-basketball-demo.sh — FIBA / Paris-2024 basketball AR can-placement demo.
#
# Renders the "two cans on the court" advertising demo over a real broadcast shot
# of the 2024 Olympic basketball final (France vs USA), for any of three drink
# cans, using the Disney PBR path with the ACES filmic tone-map.
#
# Outputs (in --outdir, default output/), where <NAME> = heineken | coca | qd:
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
# The two-can placement is committed (examples/frames.fiba.stage2.can_duo*.jsonl);
# meshes are normalized on load, so the same placement is reused for every can.
#
# Requirements
# ------------
#   * Run inside `nix develop` (needs render.sh, ffmpeg, the Rust toolchain):
#         nix develop -c bash examples/olympic-basketball-demo.sh --can heineken ...
#   * GPU: on non-NixOS Linux the renders are auto-wrapped with nixGL (needs
#     network on first use). Override the wrapper with TRD_NIXGL_CMD="..." or
#     disable it with --no-nixgl (NixOS / WSL-gl).
#   * uv (for the pillow/numpy compositing helpers) on PATH or at ~/.nix-profile.
#   * The FIBA background frames. They are NOT vendored (copyrighted footage): the
#     script extracts them from your local shot_0001.mp4 via --source, into
#     --frames-base (default output/fiba), unless they already exist there.
#
# Examples
#   nix develop -c bash examples/olympic-basketball-demo.sh \
#       --can heineken --source ~/videos/shot_0001.mp4
#   nix develop -c bash examples/olympic-basketball-demo.sh --can coca --what dolly
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # repo/worktree root
cd "$HERE"

# ---- defaults ------------------------------------------------------------
CAN=heineken
SOURCE=""
FRAMES_BASE=output/fiba
ENVMAP=assets/envmap/uffizi-large.hdr
OUTDIR=output
WHAT=all                       # all | reveal | dolly
KEEP_WORK=0
USE_NIXGL=auto                 # auto | on | off
# material overrides (empty => per-can preset / shared ACES default)
METALLIC="" ROUGHNESS="" EXPOSURE="" ENVINT="" AMBIENT="" SPECULAR="" TONEMAP=""

usage() {
  cat <<'EOF'
olympic-basketball-demo.sh — FIBA basketball AR can-placement demo

Usage:
  nix develop -c bash examples/olympic-basketball-demo.sh [options]

Options:
  --can NAME          heineken | coca | qd            (default: heineken)
  --what WHAT         all | reveal | dolly            (default: all)
  --source FILE       shot_0001.mp4 to extract FIBA background frames from
                      (only used if --frames-base has no frames yet)
  --frames-base DIR   extracted background plates dir  (default: output/fiba)
  --env HDR           IBL environment map              (default: assets/envmap/uffizi-large.hdr)
  --outdir DIR        output directory                 (default: output)
  --metallic / --roughness / --exposure / --env-intensity / --ambient /
  --specular / --tonemap VALUE
                      override the can's PBR/lighting preset
  --no-nixgl          do not wrap renders with nixGL (NixOS / WGPU_BACKEND=gl)
  --keep-work         keep the per-can intermediate work dir
  -h, --help          show this help

Per-can presets (metallic / roughness); shared ACES lighting is
env-intensity 0.90, exposure 0.45, ambient 0.03, specular 0.6, tonemap aces:
  heineken  0.7 / 0.25      coca  1.0 / 0.30      qd  0.0 / 0.30
EOF
}

# bare invocation -> print guidance and exit 0 (repo convention)
[ $# -eq 0 ] && { usage; exit 0; }

while [ $# -gt 0 ]; do
  case "$1" in
    --can) CAN="$2"; shift 2 ;;
    --what) WHAT="$2"; shift 2 ;;
    --source) SOURCE="$2"; shift 2 ;;
    --frames-base) FRAMES_BASE="$2"; shift 2 ;;
    --env) ENVMAP="$2"; shift 2 ;;
    --outdir) OUTDIR="$2"; shift 2 ;;
    --metallic) METALLIC="$2"; shift 2 ;;
    --roughness) ROUGHNESS="$2"; shift 2 ;;
    --exposure) EXPOSURE="$2"; shift 2 ;;
    --env-intensity) ENVINT="$2"; shift 2 ;;
    --ambient) AMBIENT="$2"; shift 2 ;;
    --specular) SPECULAR="$2"; shift 2 ;;
    --tonemap) TONEMAP="$2"; shift 2 ;;
    --no-nixgl) USE_NIXGL=off; shift ;;
    --keep-work) KEEP_WORK=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; usage; exit 2 ;;
  esac
done

# ---- resolve the can preset ---------------------------------------------
case "$(printf '%s' "$CAN" | tr '[:upper:]' '[:lower:]')" in
  heineken|hei|beer_hes|hes)
    NAME=heineken
    MESH=assets/meshes/can_hei/source/3d66.com_JCI54557823712.obj
    TEX=assets/meshes/can_hei/textures/3d66-export-JCI54557823712-003.jpg
    DEF_METAL=0.7; DEF_ROUGH=0.25 ;;
  coca|coke|cocacola|can)
    NAME=coca
    MESH=assets/meshes/can/coke.obj
    TEX=assets/meshes/can/can_around.jpg
    DEF_METAL=1.0; DEF_ROUGH=0.30 ;;
  qd|qd_beer|qingdao)
    NAME=qd
    MESH=assets/meshes/qd_beer/source/3d66.com_JDH5455878326.obj
    TEX=assets/meshes/qd_beer/textures/3d66-export-JDH5455878326-001.jpg
    DEF_METAL=0.0; DEF_ROUGH=0.30 ;;
  *) echo "unknown --can '$CAN' (want: heineken | coca | qd)" >&2; exit 2 ;;
esac
# per-can material + shared ACES lighting (from can_hei_pbr_dolly_aces), overridable
METALLIC="${METALLIC:-$DEF_METAL}"; ROUGHNESS="${ROUGHNESS:-$DEF_ROUGH}"
EXPOSURE="${EXPOSURE:-0.45}"; ENVINT="${ENVINT:-0.90}"
AMBIENT="${AMBIENT:-0.03}"; SPECULAR="${SPECULAR:-0.6}"; TONEMAP="${TONEMAP:-aces}"

# ---- environment ---------------------------------------------------------
[ -n "${IN_NIX_SHELL:-}" ] || {
  echo "error: run inside 'nix develop' (render.sh + ffmpeg need it), e.g.:" >&2
  echo "  nix develop -c bash examples/olympic-basketball-demo.sh $*" >&2
  exit 1; }
for t in "$MESH" "$TEX" "$ENVMAP"; do
  [ -f "$t" ] || { echo "error: missing asset: $t" >&2; exit 1; }
done
UV="$(command -v uv || echo "$HOME/.nix-profile/bin/uv")"
[ -x "$UV" ] || { echo "error: 'uv' not found (needed for pillow/numpy helpers)" >&2; exit 1; }

# nixGL wrapper (auto on non-NixOS Linux)
NIXGL=()
if [ "$USE_NIXGL" != off ]; then
  if [ -n "${TRD_NIXGL_CMD:-}" ]; then
    # shellcheck disable=SC2206
    NIXGL=($TRD_NIXGL_CMD)
  elif [ ! -e /run/opengl-driver ]; then
    NIXGL=(env NIXPKGS_ALLOW_UNFREE=1 nix run --impure github:nix-community/nixGL#nixGLNvidia --)
  fi
fi

py()  { "$UV" run --with pillow --with numpy --with pyarrow python3 "$@"; }
tailf() { tail -1; }

S=examples                                  # committed scene dir
WORK="$OUTDIR/_olympic_work/$NAME"
mkdir -p "$OUTDIR"

# low-level render: <in> <out> <W> <H> [extra render flags...]
_render() {
  local inp="$1" out="$2" w="$3" h="$4"; shift 4
  "${NIXGL[@]}" bash "$S/render.sh" --cli --pbr \
    --mesh "$MESH" --texture "$TEX" --env "$ENVMAP" \
    --metallic "$METALLIC" --roughness "$ROUGHNESS" --env-intensity "$ENVINT" \
    --exposure "$EXPOSURE" --ambient "$AMBIENT" --specular "$SPECULAR" --tonemap "$TONEMAP" \
    "$@" "$inp" "$out" "$w" "$h" 24 2>&1 | tailf
}
concat() { ffmpeg -y -i "$1" -i "$2" -filter_complex '[0:v][1:v]concat=n=2:v=1[v]' \
             -map '[v]' -c:v libx264 -pix_fmt yuv420p -crf 18 "$3" 2>&1 | tailf; }
encode() { ffmpeg -y -framerate 24 -start_number 0 -i "$1/f%04d.png" \
             -c:v libx264 -pix_fmt yuv420p -crf 18 "$2" 2>&1 | tailf; }

ensure_frames() {
  [ -f "$FRAMES_BASE/frames/frame_000000.jpg" ] && return
  [ -n "$SOURCE" ] || {
    echo "error: no frames at $FRAMES_BASE/frames/ and no --source given." >&2
    echo "       Provide the broadcast clip: --source /path/to/shot_0001.mp4" >&2
    exit 1; }
  echo "### extracting FIBA background frames from $SOURCE -> $FRAMES_BASE"
  py scripts/extract_frames.py "$SOURCE" -o "$FRAMES_BASE" --format jpg --no-arrow 2>&1 | tailf
  [ -f "$FRAMES_BASE/frames/frame_000000.jpg" ] || {
    echo "error: frame extraction did not produce $FRAMES_BASE/frames/frame_000000.jpg" >&2
    exit 1; }
}

reveal_scan() {
  ensure_frames
  local orig="$FRAMES_BASE/frames/frame_000000.jpg"
  local base="$WORK/duo_base.mp4"
  local dc="$S/frames.fiba.stage2.can_duo.jsonl"
  rm -rf "$WORK"; mkdir -p "$WORK"

  echo "### [$NAME] 1/6 clean duo base (288f, both cans)"
  _render "$dc" "$base" 1920 1080 --frames-base "$FRAMES_BASE"

  echo "### [$NAME] 2/6 upper-only base (lower can removed)"
  py "$S/olympic/upper_only.py" "$dc" "$WORK/duo_upper.jsonl" | tailf
  _render "$WORK/duo_upper.jsonl" "$WORK/duo_upper.mp4" 1920 1080 --frames-base "$FRAMES_BASE"

  echo "### [$NAME] 3/6 gizmo stills (solid + wireframe, frame 0)"
  _render "$S/frames.fiba.stage2.can_duo.gizmo_f0.jsonl"      "$WORK/gizmo.mp4" \
          1920 1080 --frames-base "$FRAMES_BASE" --placement-quad --aabb --axes-local --grid-local xy
  _render "$S/frames.fiba.stage2.can_duo.gizmo_wire_f0.jsonl" "$WORK/gizmowire.mp4" \
          1920 1080 --frames-base "$FRAMES_BASE" --placement-quad --aabb --axes-local --grid-local xy --grid-mesh 1
  ffmpeg -y -i "$WORK/gizmo.mp4"     -frames:v 1 "$WORK/gizmo0.png"     2>&1 | tailf
  ffmpeg -y -i "$WORK/gizmowire.mp4" -frames:v 1 "$WORK/gizmowire0.png" 2>&1 | tailf

  echo "### [$NAME] 4/6 composite fade + cut bases (lower can gone @ frame 91)"
  rm -rf "$WORK/cb" "$WORK/up" "$WORK/fadeb" "$WORK/cutb"
  mkdir -p "$WORK/cb" "$WORK/up" "$WORK/fadeb" "$WORK/cutb"
  ffmpeg -y -i "$base"            -start_number 0 "$WORK/cb/c%04d.png" 2>&1 | tailf
  ffmpeg -y -i "$WORK/duo_upper.mp4" -start_number 0 "$WORK/up/u%04d.png" 2>&1 | tailf
  py "$S/olympic/composite_bases.py" "$WORK/cb" "$WORK/up" "$WORK/fadeb" "$WORK/cutb" | tailf
  encode "$WORK/fadeb" "$WORK/fade_base.mp4"
  encode "$WORK/cutb"  "$WORK/cut_base.mp4"

  echo "### [$NAME] 5/6 scan intros (solid + wireframe, 288f each)"
  ffmpeg -y -i "$base" -vf 'select=eq(n\,0)' -vsync 0 -frames:v 1 "$WORK/base0.png" 2>&1 | tailf
  rm -rf "$WORK/intro" "$WORK/introwire"
  py "$S/olympic/scan_intro.py" "$orig" "$WORK/base0.png" "$WORK/gizmo0.png"     "$WORK/intro"     72 12 68 6 | tailf
  py "$S/olympic/scan_intro.py" "$orig" "$WORK/base0.png" "$WORK/gizmowire0.png" "$WORK/introwire" 72 12 68 6 | tailf
  ffmpeg -y -framerate 24 -i "$WORK/intro/f%04d.png"     -c:v libx264 -pix_fmt yuv420p -crf 18 "$WORK/intro.mp4"     2>&1 | tailf
  ffmpeg -y -framerate 24 -i "$WORK/introwire/f%04d.png" -c:v libx264 -pix_fmt yuv420p -crf 18 "$WORK/introwire.mp4" 2>&1 | tailf

  echo "### [$NAME] 6/6 concat 4 finals"
  local pre="$OUTDIR/fiba_stage2_${NAME}_pbr_duo_reveal_scan"
  concat "$WORK/intro.mp4"     "$WORK/cut_base.mp4"  "${pre}_uffizi_large.mp4"
  concat "$WORK/intro.mp4"     "$WORK/fade_base.mp4" "${pre}_fade_uffizi_large.mp4"
  concat "$WORK/introwire.mp4" "$WORK/cut_base.mp4"  "${pre}_wireframe_uffizi_large.mp4"
  concat "$WORK/introwire.mp4" "$WORK/fade_base.mp4" "${pre}_wireframe_fade_uffizi_large.mp4"
}

dolly() {
  echo "### [$NAME] dolly ACES turntable (512x512)"
  _render "$S/frames.bunny_dolly.cg.jsonl" "$OUTDIR/${NAME}_pbr_dolly_aces.gif" 512 512
}

echo "== olympic-basketball-demo: can=$NAME  what=$WHAT  env=$(basename "$ENVMAP")"
echo "   material: metallic=$METALLIC roughness=$ROUGHNESS exposure=$EXPOSURE"
echo "             env-intensity=$ENVINT ambient=$AMBIENT specular=$SPECULAR tonemap=$TONEMAP"
[ ${#NIXGL[@]} -gt 0 ] && echo "   gpu wrapper: ${NIXGL[*]}"

case "$WHAT" in
  all)    reveal_scan; dolly ;;
  reveal) reveal_scan ;;
  dolly)  dolly ;;
  *) echo "unknown --what '$WHAT' (want: all | reveal | dolly)" >&2; exit 2 ;;
esac

[ "$KEEP_WORK" -eq 1 ] || rm -rf "$WORK"

echo "== done. outputs in $OUTDIR/:"
ls -1 "$OUTDIR"/fiba_stage2_${NAME}_pbr_duo_reveal_scan*_uffizi_large.mp4 \
      "$OUTDIR/${NAME}_pbr_dolly_aces.gif" 2>/dev/null || true
