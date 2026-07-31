#!/usr/bin/env bash
# Render a trd JSONL frame-parameter file to an animated GIF/WebP.
#
# Pipeline (fully piped, no intermediate files):
#   JSONL --(Arrow IPC: pyarrow)--> trd --(tensors)--> ffmpeg
#
# Usage:
#   examples/render.sh [--cli | --native | --web [--canvas-renderer|--offscreen-renderer]] \
#                      [--mesh OBJ]... [--texture IMG] [--wireframe] [--aabb] [--axes] [--frames-base DIR] [INPUT.jsonl] [OUTPUT.gif|.webp] [WIDTH] [HEIGHT] [FPS]
# Defaults: examples/frames.bunny_dolly.cg.jsonl (rendering assets/meshes/bunny.obj)  output/out.gif  256 256 30
# Run with no arguments (or -h/--help) to print the flag guidance and exit; pass
# --cli to render the default demo (the bunny dolly camera capstone).
#
# By default (or with --cli, alias --headless) the frame stream is rendered to a
# GIF/WebP via the headless trd-cli.
# With --native (alias --app) it is played live in the interactive trd-app window
# (trd-native); OUTPUT is then ignored and neither uv nor ffmpeg are needed.
# The stream protocol is 0.0.5-only and mesh-first: every stream begins with a
# mesh table (scripts/obj_to_arrow.py encodes the OBJ) concatenated with the
# params stream, so trd renders the loaded mesh (centered + uniformly scaled to
# fit) driven by INPUT.jsonl. When no --mesh (and no --placement-quad) is given,
# the bunny (assets/meshes/bunny.obj) is loaded as the default demo object. Try:
# examples/render.sh --mesh assets/meshes/bunny.obj \
# examples/frames.turntable.jsonl output/bunny.gif. --mesh is repeatable: pass it
# several times to load several meshes (one table row each, in order); a frame's
# `draws` list then references them by 0-based index. Try the two-mesh demo:
# examples/render.sh --cli --wireframe --mesh assets/meshes/bunny.obj \
# --mesh examples/cube.obj examples/frames.multimesh.jsonl output/scene.gif.
# (--mesh needs pyarrow via uv/python3; it applies to --web too.)
# The appearance flags below (--wireframe/--aabb/--axes/--axes-local) apply to
# both --cli and --native: trd-cli and trd-app share trd-core's mesh Scene renderer.
# With --wireframe trd draws mesh edges as a line list instead of filled
# triangles (protocol #38); combine with --mesh for a wireframe asset.
# With --aabb trd overlays each drawn mesh's axis-aligned bounding box as a green
# wireframe box (#42); combine with --mesh (e.g. add --aabb to the bunny
# turntable to see its box track the rotation).
# With --axes trd overlays a coordinate-axes gizmo (X=red, Y=green, Z=blue) at
# the world origin (#42), marking the world frame the camera looks at.
# With --axes-local trd overlays a coordinate-axes gizmo at EACH drawn object's
# local frame (its own `model`) — its model-space X/Y/Z axes as placed. This
# visualizes e.g. #77's (e1,e2,e3) quad frame the mesh is anchored in
# (pair it with examples/placement_quad_by_local_coord.py, ideally --turns 0 to freeze
# the spin so the axes show the fixed placement frame).
# With --placement-quad trd draws the reconstructed placement quad itself as a
# colored wireframe outline (--placement-quad-color "R G B", default cyan) — a
# debug check that the surface the mesh is anchored to matches the filmed poster.
# --placement-quad appends a canonical quad mesh that a per-frame draw references;
# author it with examples/placement_quad_by_local_coord.py --placement-quad (which emits the
# {mesh:idx, mode:"wireframe"} draw). The quad geometry travels in the Arrow mesh
# table and its placement in the per-frame draw list — no hardcoded gizmo.
#
# Dolly-camera capstone (#49): examples/bunny_dolly.py authors the same 45°
# bird's-eye *dolly* camera twice — CG (eye/target/fovy) and CV (K + pose) — as
# two JSONL streams that render identically (verified to <0.01% pixels).
# render.sh runs this producer automatically: pass frames.bunny_dolly.cg.jsonl
# (or .cv.jsonl) as INPUT and, if it is missing, it is generated on the fly — no
# manual pre-step. Render with --wireframe --aabb --axes --mesh assets/meshes/
# bunny.obj and compare the two forms:
#   examples/render.sh --cli --wireframe --aabb --axes --mesh assets/meshes/bunny.obj \
#     examples/frames.bunny_dolly.cg.jsonl output/bunny_dolly_cg.gif 1024 1024 24
#   examples/render.sh --cli --wireframe --aabb --axes --mesh assets/meshes/bunny.obj \
#     examples/frames.bunny_dolly.cv.jsonl output/bunny_dolly_cv.gif 1024 1024 24
# With --web (alias --wasm) it renders the SAME scene as --cli, but in a WebGPU
# browser. It builds the config-driven web bundle via nix (.#web), copies it to a
# writable serve dir, and drops in the runtime inputs the generic viewer
# (web/src/viewer.ts) fetches at load: `stream.arrow` (the identical
# mesh++texture++params bytes trd-cli reads on stdin, from the same producers),
# `config.json` (the chosen renderer target + scene flags + baked resolution +
# default fps), and — when --frames-base is set — the background stills, so the
# browser replays exactly what --cli would render. static-web-server then serves
# the dir (override the port with PORT, default 8080; it binds all interfaces).
# Two in-browser targets share the bundle: --canvas-renderer (default) draws to
# the on-screen WebGPU CanvasRenderer; --offscreen-renderer (alias --arrow-
# renderer) draws to an offscreen ArrowRenderer texture read back to a 2D canvas
# (the browser twin of the CLI output stream). All the content flags below
# (--mesh/--texture/--wireframe/--aabb/--axes/--axes-local/--placement-quad/
# --frames-base) and the positional WIDTH/HEIGHT apply to --web exactly as to
# --cli; only the playback rate is a live URL param:
#   ?fps=N    playback frame rate (1..240; default = the FPS positional arg)
# e.g. examples/render.sh --web --canvas-renderer --placement-quad --axes-local \
#        --frames-base output/cornellbox examples/frames.cornellbox.stage1.jsonl \
#        '' 960 540 25   then open http://localhost:8080/?fps=30
# WebGPU needs a secure context, so open http://localhost:<port> (an SSH tunnel
# makes a remote machine's origin "localhost" too); a bare http://<ip> is NOT a
# secure context.
#
# Run from `nix develop`. The Arrow frame stream is built with pyarrow (via
# uv/python3).
# On WSL, prefix with WGPU_BACKEND=gl for GPU rendering (else it uses software).
set -euo pipefail

# Print flag guidance (shown for a bare invocation or -h/--help).
usage() {
  cat <<'USAGE'
render.sh — render a trd JSONL frame-parameter file to a GIF/WebP (or play/serve it).

Usage:
  examples/render.sh [MODE] [CONTENT FLAGS] [INPUT.jsonl] [OUTPUT.gif|.webp] [WIDTH] [HEIGHT] [FPS]

Defaults: INPUT=examples/frames.bunny_dolly.cg.jsonl (renders assets/meshes/bunny.obj)  OUTPUT=output/out.gif  WIDTH=256  HEIGHT=256  FPS=30

MODE (pick one; default --cli):
  --cli, --headless   Render to a GIF/WebP via the headless trd-cli (default).
  --native, --app     Play live in the interactive trd-app window (OUTPUT ignored).
  --web, --wasm       Build the web (wasm) bundle and serve the SAME scene as --cli
                      in a WebGPU browser (generates stream.arrow + config.json).
                        --canvas-renderer     on-screen WebGPU surface (default)
                        --offscreen-renderer  offscreen texture read back to a canvas
                                              (alias --arrow-renderer)

BROWSER QUERY PARAM (--web; append to the URL, no rebuild):
  ?fps=N              playback frame rate (1..240; default = the FPS positional arg).
                      The render resolution is baked into the stream's CV `k`, so it
                      is the WIDTH/HEIGHT positional args (not a URL param).
                      e.g. http://localhost:PORT/?fps=30
  Open http://localhost:PORT (WebGPU needs this secure context; a bare IP is not
  one — use the printed SSH tunnel for a remote browser).

CONTENT FLAGS (--cli and --native):
  --mesh OBJ          Load OBJ as a mesh table entry (centered + scaled to fit).
                      Repeatable: pass several times to load several meshes (row 0,
                      1, …); a frame's `draws` list references them by index.
                      Defaults to assets/meshes/bunny.obj when no mesh is given.
  --texture IMG       Bind IMG as a texture table and render textured — sampling it
                      at each vertex UV (#20). Requires --mesh (with UVs); mutually
                      exclusive with --wireframe.
  --wireframe         Draw mesh edges as a line list instead of filled triangles (#38).
  --aabb              Overlay each mesh's axis-aligned bounding box as a green box (#42).
  --axes              Overlay a coordinate-axes gizmo (X=red, Y=green, Z=blue) at the origin (#42).
  --axes-local        Overlay a coordinate-axes gizmo at EACH drawn object's own local
                      frame (its model), e.g. #77's (e1,e2,e3) quad placement frame.
  --grid-local PLANE  Overlay a coordinate-plane grid lattice (PLANE = xy|xz|yz) on each
                      WIREFRAME drawn object's own local frame — e.g. --grid-local xy tiles
                      a grid across the placement quad's local floor (#110). Scoped to
                      wireframe draws, so a filled/textured mesh (the bunny) gets no grid.
  --grid-mesh ID      Narrow --grid-local to draws of mesh ID only (the placement quad), so a
                      content mesh drawn WIREFRAME (e.g. a wireframe-reveal intro) doesn't also
                      pick up a floor grid. Ignored without --grid-local (#114).
  --placement-quad    Draw the reconstructed placement quad as a colored wireframe outline
                      (debug check vs. the filmed poster). Appends a canonical quad mesh;
                      author its per-frame draw with placement_quad_by_local_coord.py --placement-quad.
  --placement-quad-color "R G B"
                      Placement-quad outline color, 0..1 floats (default: "0 1 1" cyan).
                      Implies --placement-quad's mesh.
  --frames-base DIR   Composite each frame's 0.0.5 background still (its `frame_path`,
                      resolved relative to DIR) *beneath* the scene via a FramePlane (#63).
                      Stills are decoded at full resolution; extract them with
                      scripts/extract_frames.py <video> --format jpg (add --height H to
                      extract smaller stills and save memory).

  -h, --help          Show this guidance and exit.

Examples:
  examples/render.sh --cli                                   # default demo → output/out.gif
  examples/render.sh --native                                # play the default demo live
  examples/render.sh --native --wireframe --aabb --axes --mesh assets/meshes/bunny.obj \
    examples/frames.bunny_dolly.cg.jsonl '' 1024 1024 24     # live dolly capstone in a window
  examples/render.sh --cli --aabb --mesh assets/meshes/bunny.obj \
    examples/frames.turntable.jsonl output/bunny.gif 1024 1024 24
  examples/render.sh --cli --mesh assets/meshes/bunny_with_texture/bunny.obj \
    --texture assets/meshes/bunny_with_texture/bunny_uv_map1.jpg \
    examples/frames.bunny_dolly.cg.jsonl output/bunny_textured.gif 512 512 20  # textured bunny (#20)
  examples/render.sh --cli --wireframe --aabb \
    --mesh assets/meshes/bunny.obj --mesh examples/cube.obj \
    examples/frames.multimesh.jsonl output/scene.gif 1024 1024 24
  examples/render.sh --cli --wireframe --axes --aabb --mesh assets/meshes/bunny.obj \
    examples/frames.bunny_dolly.cg.jsonl output/bunny_dolly.gif 1024 1024 24  # dolly capstone (#49; auto-generates the frames)
  # Two-stage placement-quad pipeline (#77): stage 1 = placement quad + local frame
  # (before placing the mesh); stage 2 = mesh anchored on it, with AABB + local frame.
  #   uv run --with pyarrow --with numpy scripts/perception_to_arrow.py \
  #     --assets assets/videos/cornellbox -o examples/frames.cornellbox.perception.arrow
  #   uv run --with pyarrow --with numpy examples/placement_quad_by_local_coord.py --from-perception \
  #     examples/frames.cornellbox.perception.arrow --no-place-mesh --placement-quad \
  #     --placement-quad-mesh-index 0 -o examples/frames.cornellbox.stage1.jsonl
  examples/render.sh --cli --placement-quad --axes-local --frames-base output/cornellbox \
    examples/frames.cornellbox.stage1.jsonl output/cornellbox_stage1.gif 960 540 25  # stage 1: placement quad only
  #   uv run --with pyarrow --with numpy examples/placement_quad_by_local_coord.py --from-perception \
  #     examples/frames.cornellbox.perception.arrow --placement-quad \
  #     -o examples/frames.cornellbox.stage2.jsonl
  examples/render.sh --cli --placement-quad --axes-local --aabb \
    --mesh assets/meshes/bunny_with_texture/bunny.obj \
    --texture assets/meshes/bunny_with_texture/bunny_uv_map1.jpg \
    --frames-base output/cornellbox \
    examples/frames.cornellbox.stage2.jsonl output/cornellbox_stage2.gif 960 540 25  # stage 2: mesh placed
  # --web replays any --cli scene in the browser (same flags + positional W H FPS):
  examples/render.sh --web --canvas-renderer --placement-quad --axes-local \
    --frames-base output/cornellbox \
    examples/frames.cornellbox.stage1.jsonl '' 960 540 25   # stage 1 on-screen (WebGPU canvas)
  #   then open http://localhost:8080/?fps=30                (fps tuned live; size baked)
  examples/render.sh --web --offscreen-renderer --placement-quad --axes-local --aabb \
    --mesh assets/meshes/bunny_with_texture/bunny.obj \
    --texture assets/meshes/bunny_with_texture/bunny_uv_map1.jpg \
    --frames-base output/cornellbox \
    examples/frames.cornellbox.stage2.jsonl '' 960 540 25   # stage 2 via offscreen texture readback

Run from `nix develop`. On WSL, prefix with WGPU_BACKEND=gl for GPU rendering.
USAGE
}

# A bare invocation (no arguments at all) prints the flag guidance and exits,
# rather than silently rendering the default demo — pass --cli to run it.
if [ $# -eq 0 ]; then
  usage
  exit 0
fi

# Extract the optional mode flags (--cli/--native/--web), the --web renderer
# sub-flags (--canvas-renderer/--offscreen-renderer), and repeatable --mesh <obj>
# flags that prepend a mesh Arrow stream (0.0.5 [mesh][params]); the rest are
# positional.
cli=0
native=0
web=0
offscreen_renderer=0
canvas_renderer=0
wireframe=0
pbr=0
env=""
metallic="0.0"
roughness="0.35"
env_intensity="1.0"
exposure="1.2"
ambient="0.12"
specular="0.5"
clearcoat="0.0"
tonemap="reinhard"
aabb=0
axes=0
axes_local=0
grid_local=""
grid_mesh=""
quad=0
quad_color="0 1 1"
meshes=()
texture=""
frames_base=""
positional=()
while [ $# -gt 0 ]; do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    --cli|--headless) cli=1 ;;
    --native|--app) native=1 ;;
    --web|--wasm) web=1 ;;
    --offscreen-renderer|--arrow-renderer) offscreen_renderer=1 ;;
    --canvas-renderer) canvas_renderer=1 ;;
    --wireframe) wireframe=1 ;;
    --pbr) pbr=1 ;;
    --env) shift; env="${1:?--env requires an .hdr path}" ;;
    --env=*) env="${1#--env=}" ;;
    --metallic) shift; metallic="${1:?--metallic requires a float}" ;;
    --metallic=*) metallic="${1#--metallic=}" ;;
    --roughness) shift; roughness="${1:?--roughness requires a float}" ;;
    --roughness=*) roughness="${1#--roughness=}" ;;
    --env-intensity) shift; env_intensity="${1:?--env-intensity requires a float}" ;;
    --env-intensity=*) env_intensity="${1#--env-intensity=}" ;;
    --exposure) shift; exposure="${1:?--exposure requires a float}" ;;
    --exposure=*) exposure="${1#--exposure=}" ;;
    --ambient) shift; ambient="${1:?--ambient requires a float}" ;;
    --ambient=*) ambient="${1#--ambient=}" ;;
    --specular) shift; specular="${1:?--specular requires a float}" ;;
    --specular=*) specular="${1#--specular=}" ;;
    --clearcoat) shift; clearcoat="${1:?--clearcoat requires a float}" ;;
    --clearcoat=*) clearcoat="${1#--clearcoat=}" ;;
    --tonemap) shift; tonemap="${1:?--tonemap requires reinhard|aces}" ;;
    --tonemap=*) tonemap="${1#--tonemap=}" ;;
    --aabb) aabb=1 ;;
    --axes) axes=1 ;;
    --axes-local) axes_local=1 ;;
    --grid-local) shift; grid_local="${1:?--grid-local requires a plane: xy|xz|yz}" ;;
    --grid-local=*) grid_local="${1#--grid-local=}" ;;
    --grid-mesh) shift; grid_mesh="${1:?--grid-mesh requires a mesh id (integer)}" ;;
    --grid-mesh=*) grid_mesh="${1#--grid-mesh=}" ;;
    --placement-quad) quad=1 ;;
    --placement-quad-color) shift; quad=1; quad_color="${1:?--placement-quad-color requires \"R G B\" (0..1 floats)}" ;;
    --placement-quad-color=*) quad=1; quad_color="${1#--placement-quad-color=}" ;;
    --mesh) shift; meshes+=("${1:?--mesh requires an OBJ path}") ;;
    --mesh=*) meshes+=("${1#--mesh=}") ;;
    --texture) shift; texture="${1:?--texture requires an image path}" ;;
    --texture=*) texture="${1#--texture=}" ;;
    --frames-base) shift; frames_base="${1:?--frames-base requires a directory}" ;;
    --frames-base=*) frames_base="${1#--frames-base=}" ;;
    *) positional+=("$1") ;;
  esac
  shift
done
if [ ${#positional[@]} -gt 0 ]; then set -- "${positional[@]}"; else set --; fi

# Modes are mutually exclusive; the renderer sub-flags apply only to --web.
if [ $((cli + native + web)) -gt 1 ]; then
  echo "error: choose only one of --cli, --native, --web" >&2
  exit 1
fi
if [ $((offscreen_renderer + canvas_renderer)) -gt 1 ]; then
  echo "error: choose only one of --canvas-renderer, --offscreen-renderer" >&2
  exit 1
fi
if [ $((offscreen_renderer + canvas_renderer)) -ge 1 ] && [ "$web" -ne 1 ]; then
  echo "error: --canvas-renderer / --offscreen-renderer apply only to --web/--wasm" >&2
  exit 1
fi

# --texture provides a 0.0.4 texture table (bound as the sampled albedo) and
# renders textured. It needs a --mesh (UVs to sample the texture) and is
# mutually exclusive with --wireframe. It applies to --web too (the browser
# renderer replays the same generated [mesh][texture][params] stream).
if [ -n "$texture" ]; then
  if [ ${#meshes[@]} -eq 0 ]; then
    echo "error: --texture requires at least one --mesh (with UVs to sample)" >&2
    exit 1
  fi
  if [ "$wireframe" -eq 1 ]; then
    echo "error: --texture and --wireframe are mutually exclusive" >&2
    exit 1
  fi
fi

# --pbr renders the bound albedo with the Disney principled BRDF (a virtual
# light rig + smooth normals + optional --env HDR reflection). It needs a
# --texture (the albedo) and is mutually exclusive with --wireframe/--textured.
if [ "$pbr" -eq 1 ]; then
  if [ -z "$texture" ]; then
    echo "error: --pbr requires a --texture (the albedo to shade)" >&2
    exit 1
  fi
  if [ "$wireframe" -eq 1 ]; then
    echo "error: --pbr and --wireframe are mutually exclusive" >&2
    exit 1
  fi
fi

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
input="${1:-$root/examples/frames.bunny_dolly.cg.jsonl}"
output="${2:-output/out.gif}"
width="${3:-256}"
height="${4:-256}"
fps="${5:-30}"

# --placement-quad appends a canonical colored quad mesh (an origin-centred,
# extent-2 unit square, corners ±1) as the LAST --mesh, so a stream can draw the
# reconstructed placement quad as a wireframe overlay (a debug check that it
# matches the filmed poster). Its ±1 corners map straight to the camera-space quad
# — the producer (examples/placement_quad_by_local_coord.py --placement-quad) emits a per-frame
# `draws` entry {mesh: idx, mode: "wireframe"} placing it. --placement-quad-color
# "R G B" (0..1 floats) tints it (default cyan). The quad rides the mesh table
# (Arrow), so it needs the same pyarrow producer as --mesh.
quad_obj=""
if [ "$quad" -eq 1 ]; then
  read -r qr qg qb <<QC
$quad_color
QC
  : "${qr:=0}" "${qg:=1}" "${qb:=1}"
  quad_obj="$(mktemp --suffix=.obj)"
  trap 'rm -f "$quad_obj"' EXIT
  cat > "$quad_obj" <<QUAD
# canonical placement-quad overlay (render.sh --placement-quad): centred, extent 2, corners ±1.
# 'v x y z r g b' bakes the outline color into the vertices (wireframe uses them).
v -1 -1 0 $qr $qg $qb
v 1 -1 0 $qr $qg $qb
v 1 1 0 $qr $qg $qb
v -1 1 0 $qr $qg $qb
f 1 2 3
f 1 3 4
QUAD
  meshes+=("$quad_obj")
fi

# The stream protocol is mesh-first (0.0.5 requires a leading [mesh] table; there
# is no params-only fallback). When neither --mesh nor --placement-quad supplied a
# mesh, load the bunny as the default demo object so the stream is a valid
# [mesh][params] and the default INPUT (frames.bunny_dolly.cg.jsonl) has something
# to place.
if [ ${#meshes[@]} -eq 0 ]; then
  meshes+=("$root/assets/meshes/bunny.obj")
fi

# examples/bunny_dolly.py authors the 45° bird's-eye *dolly* camera capstone (#49)
# as two JSONL streams — CG (eye/target/fovy) and CV (K + pose) — that render
# identically. If the requested INPUT is one of its outputs
# (frames.bunny_dolly.{cg,cv}.jsonl) and it is not present yet, generate it now
# via the (pure-stdlib) producer so the demo renders without a manual pre-step.
case "$input" in
  *frames.bunny_dolly.cg.jsonl|*frames.bunny_dolly.cv.jsonl)
    if [ ! -f "$input" ]; then
      prefix=${input%.cg.jsonl}; prefix=${prefix%.cv.jsonl}
      echo "generating dolly frames via examples/bunny_dolly.py (--out-prefix $prefix)…" >&2
      if command -v python3 >/dev/null 2>&1; then
        python3 "$root/examples/bunny_dolly.py" --out-prefix "$prefix" >&2
      elif command -v uv >/dev/null 2>&1; then
        uv run --python 3.12 "$root/examples/bunny_dolly.py" --out-prefix "$prefix" >&2
      else
        echo "error: need python3 (or uv) to run examples/bunny_dolly.py" >&2
        exit 1
      fi
    fi
    ;;
esac

# The web path builds/serves via nix; native needs only cargo; the GIF path also
# needs ffmpeg (and uv for encoding).
if [ "$web" -eq 1 ]; then
  tools="nix"
elif [ "$native" -eq 1 ]; then
  tools="cargo"
else
  tools="cargo ffmpeg uv"
fi
for tool in $tools; do
  if ! command -v "$tool" >/dev/null 2>&1; then
    echo "error: $tool not found on PATH" >&2
    echo "run this example inside 'nix develop'" >&2
    exit 1
  fi
done

# Choose a frame producer for the params stream: scripts/jsonl_to_arrow.py via
# uv (or python3 with pyarrow). The stream protocol is 0.0.5-only and mesh-first,
# so the params batch carries the `model`/camera/`draws`/`frame_path` columns the
# pyarrow producer emits (the old DuckDB `arrow` path only understood the retired
# 0.0.1/0.0.2 center/size/theta/model columns and is gone).
if command -v uv >/dev/null 2>&1; then
  producer=uv
elif command -v python3 >/dev/null 2>&1 && python3 -c 'import pyarrow' >/dev/null 2>&1; then
  producer=python3
else
  echo "error: need uv or python3 with pyarrow to build the Arrow frame stream" >&2
  exit 1
fi

# Emit the Arrow IPC frame stream on stdout via the chosen producer.
frames() {
  case "$producer" in
    uv) uv run --with pyarrow "$root/scripts/jsonl_to_arrow.py" "$input" ;;
    python3) python3 "$root/scripts/jsonl_to_arrow.py" "$input" ;;
  esac
}

# When rendering loaded meshes (--mesh, repeatable), pick a pyarrow-capable
# Python to encode the OBJ(s) into the leading mesh Arrow stream.
# scripts/obj_to_arrow.py emits a mesh table with **one row per --mesh** (in the
# order given); it is concatenated *before* the params stream so trd reads
# [mesh][params]. A frame's `draws` list references these meshes by 0-based index
# (mesh 0 = first --mesh).
mesh_producer=""
if [ ${#meshes[@]} -gt 0 ]; then
  if command -v uv >/dev/null 2>&1; then
    mesh_producer=uv
  elif command -v python3 >/dev/null 2>&1 && python3 -c 'import pyarrow' >/dev/null 2>&1; then
    mesh_producer=python3
  else
    echo "error: --mesh needs uv or python3 with pyarrow to encode ${meshes[*]}" >&2
    exit 1
  fi
fi

# The optional texture table (0.0.4): scripts/texture_to_arrow.py encodes the
# image into a one-row `rgba` fixed_shape_tensor<u8>[H,W,4] Arrow stream,
# concatenated *between* the mesh table and the params so trd reads
# [mesh][texture][params] and binds it as the sampled albedo. Downscaled to
# --max-size 2048 to stay within the renderer's portable (downlevel/WebGL2)
# 2048² texture limit. Needs pyarrow + pillow + numpy.
texture_producer=""
if [ -n "$texture" ]; then
  if command -v uv >/dev/null 2>&1; then
    texture_producer=uv
  elif command -v python3 >/dev/null 2>&1 \
    && python3 -c 'import pyarrow, PIL, numpy' >/dev/null 2>&1; then
    texture_producer=python3
  else
    echo "error: --texture needs uv or python3 with pyarrow/pillow/numpy to encode $texture" >&2
    exit 1
  fi
fi

# The full trd input stream: the optional leading mesh table, the optional
# texture table, then the params.
stream() {
  case "$mesh_producer" in
    uv) uv run --with pyarrow "$root/scripts/obj_to_arrow.py" "${meshes[@]}" ;;
    python3) python3 "$root/scripts/obj_to_arrow.py" "${meshes[@]}" ;;
  esac
  case "$texture_producer" in
    uv) uv run --with pyarrow --with pillow --with numpy "$root/scripts/texture_to_arrow.py" "$texture" --max-size 2048 ;;
    python3) python3 "$root/scripts/texture_to_arrow.py" "$texture" --max-size 2048 ;;
  esac
  frames
}

# Appearance flags pass through to both trd-cli (--cli) and trd-app (--native);
# trd-core's shared mesh Scene renderer honours them either way.
wireframe_flag=()
[ "$wireframe" -eq 1 ] && wireframe_flag=(--wireframe)
textured_flag=()
[ -n "$texture" ] && textured_flag=(--textured)
# --pbr shades the same bound albedo with the Disney BRDF; it replaces
# --textured (the two are mutually exclusive at the trd-cli layer) and forwards
# the material + optional HDR env probe.
pbr_flag=()
if [ "$pbr" -eq 1 ]; then
  textured_flag=()
  pbr_flag=(--pbr --metallic "$metallic" --roughness "$roughness" --env-intensity "$env_intensity" --exposure "$exposure" --ambient "$ambient" --specular "$specular" --clearcoat "$clearcoat" --tonemap "$tonemap")
  [ -n "$env" ] && pbr_flag+=(--env "$env")
fi
aabb_flag=()
[ "$aabb" -eq 1 ] && aabb_flag=(--aabb)
axes_flag=()
[ "$axes" -eq 1 ] && axes_flag=(--axes)
axes_local_flag=()
[ "$axes_local" -eq 1 ] && axes_local_flag=(--axes-local)
grid_local_flag=()
[ -n "$grid_local" ] && grid_local_flag=(--grid-local "$grid_local")
grid_mesh_flag=()
[ -n "$grid_mesh" ] && grid_mesh_flag=(--grid-mesh "$grid_mesh")
# --frames-base resolves each frame's 0.0.5 `frame_path` (relative to this dir)
# to the still image trd composites *beneath* the scene via a FramePlane (#63).
frames_base_flag=()
[ -n "$frames_base" ] && frames_base_flag=(--frames-base "$frames_base")

# --web/--wasm: replay the SAME stream + scene flags as --cli, but in the browser.
# Build the config-driven web bundle (nix .#web) once, copy it to a writable serve
# dir, then drop in the runtime inputs the generic viewer (web/src/viewer.ts)
# fetches at load:
#   stream.arrow  — mesh++texture++params, the identical bytes trd-cli reads on stdin
#   config.json   — target renderer + scene flags + baked resolution + default fps
#   frames/…      — the 0.0.5 background stills (copied from --frames-base) so each
#                   frame's `frame_path` resolves under the served root
# static-web-server serves the directory; only ?fps is a live URL override (the
# render resolution is baked into the CV `k`, so it is a render.sh positional arg).
# --canvas-renderer (default) draws to the on-screen WebGPU CanvasRenderer;
# --offscreen-renderer draws to an offscreen ArrowRenderer texture read back to a
# 2D canvas (the browser twin of the CLI output stream).
if [ "$web" -eq 1 ]; then
  port="${PORT:-8080}"
  user="$(whoami)"
  # First non-loopback IPv4 of this host (for the direct / SSH-tunnel URLs).
  ip="$(hostname -I 2>/dev/null | awk '{print $1; exit}')"
  [ -n "$ip" ] || ip="<server-ip>"

  # Renderer target: on-screen canvas (default) vs. offscreen texture readback.
  if [ "$offscreen_renderer" -eq 1 ]; then
    target="offscreen"
    renderer_label="ArrowRenderer (offscreen texture -> RGBA readback -> 2D canvas)"
  else
    target="canvas"
    renderer_label="CanvasRenderer (on-screen WebGPU surface)"
  fi

  # Base mesh mode mirrors the --cli precedence: pbr > textured > wireframe >
  # filled. --pbr shades the bound albedo with the Disney BRDF (same as
  # trd-cli/trd-app --pbr), so it takes precedence over plain texturing.
  if [ "$pbr" -eq 1 ]; then
    mode="pbr"
  elif [ -n "$texture" ]; then
    mode="textured"
  elif [ "$wireframe" -eq 1 ]; then
    mode="wireframe"
  else
    mode="filled"
  fi

  # Render an int flag as a JSON boolean for config.json.
  bool() { if [ "$1" -eq 1 ]; then printf true; else printf false; fi; }
  background=false
  [ -n "$frames_base" ] && background=true

  echo "building trd web (wasm) bundle via nix (.#web)…" >&2
  dist="$(cd "$root" && nix build --no-link --print-out-paths ".#web")"

  serve="$(mktemp -d)"
  trap 'rm -rf "$serve"' EXIT
  # nix store outputs are read-only; copy the bundle out and make it writable so we
  # can drop the generated stream/config/frames next to index.html.
  cp -RL "$dist"/. "$serve"/
  chmod -R u+w "$serve"

  echo "generating web stream.arrow + config.json (same producers as --cli)…" >&2
  stream > "$serve/stream.arrow"

  # --pbr: forward the Disney material (byte-identical to the trd-cli/trd-app
  # --pbr flags) and, if --env is set, copy the .hdr probe into the served root
  # so the browser fetches + decodes it in-wasm (trd-core does no file/codec I/O).
  pbr_json=""
  if [ "$pbr" -eq 1 ]; then
    env_json=""
    if [ -n "$env" ]; then
      cp -L "$env" "$serve/env.hdr"
      chmod u+w "$serve/env.hdr"
      env_json=",
  \"env\": \"env.hdr\""
    fi
    pbr_json=",
  \"pbr\": {
    \"metallic\": $metallic,
    \"roughness\": $roughness,
    \"specular\": $specular,
    \"clearcoat\": $clearcoat,
    \"envIntensity\": $env_intensity,
    \"exposure\": $exposure,
    \"ambient\": $ambient,
    \"tonemap\": \"$tonemap\"
  }$env_json"
  fi

  cat > "$serve/config.json" <<CFG
{
  "target": "$target",
  "mode": "$mode",
  "showAabb": $(bool "$aabb"),
  "showAxes": $(bool "$axes"),
  "showLocalAxes": $(bool "$axes_local"),
  "background": $background,
  "width": $width,
  "height": $height,
  "fps": $fps$pbr_json
}
CFG

  # Background stills: copy the --frames-base tree in so each frame's `frame_path`
  # ("frames/frame_xxxxxx.jpg", relative to it) resolves under the served root.
  if [ -n "$frames_base" ]; then
    echo "copying background stills from $frames_base…" >&2
    cp -RL "$frames_base"/. "$serve"/
    chmod -R u+w "$serve"
  fi

  cat <<EOF

trd web (wasm) server — port $port  (press Ctrl-C to stop)
  renderer: $renderer_label
  scene:    mode=$mode aabb=$(bool "$aabb") axes=$(bool "$axes") axes-local=$(bool "$axes_local") background=$background
  stream:   ${width}x${height}, default ${fps}fps  (override live with ?fps=N)

  On this machine:        http://localhost:$port
  Direct (same network):  http://$ip:$port

  SSH tunnel (recommended if the port is not directly reachable):
    ssh -L $port:localhost:$port $user@$ip
  then open in a WebGPU browser (Chrome/Edge):
                          http://localhost:$port

  WebGPU needs a secure context, so open http://localhost:$port (an SSH tunnel
  makes a remote machine's origin "localhost" too); a bare http://<ip> is NOT
  a secure context.

EOF

  echo "serving $serve on port $port…" >&2
  exec nix run nixpkgs#static-web-server -- --root "$serve" --port "$port"
fi

if [ "$native" -eq 1 ]; then
  # Play the frame stream live in the interactive trd-app window (trd-native).
  # The appearance flags pass through to trd-app too (it now renders the mesh
  # Scene via the shared trd-core MeshRenderer, like trd-cli).
  stream \
    | cargo run --manifest-path "$root/Cargo.toml" -q -p trd-app -- --width "$width" --height "$height" --fps "$fps" "${wireframe_flag[@]}" "${textured_flag[@]}" "${pbr_flag[@]}" "${aabb_flag[@]}" "${axes_flag[@]}" "${axes_local_flag[@]}" "${grid_local_flag[@]}" "${grid_mesh_flag[@]}" "${frames_base_flag[@]}"
  echo "streamed $input to the trd-app window (${width}x${height}, ${fps}fps)"
else
  mkdir -p "$(dirname "$output")"
  stream \
    | cargo run --manifest-path "$root/Cargo.toml" -q -p trd-cli -- --width "$width" --height "$height" "${wireframe_flag[@]}" "${textured_flag[@]}" "${pbr_flag[@]}" "${aabb_flag[@]}" "${axes_flag[@]}" "${axes_local_flag[@]}" "${grid_local_flag[@]}" "${grid_mesh_flag[@]}" "${frames_base_flag[@]}" \
    | uv run --with pyarrow --with numpy "$root/scripts/encode.py" --fps "$fps" -o "$output"
  echo "wrote $output (${width}x${height}, ${fps}fps) from $input"
fi
