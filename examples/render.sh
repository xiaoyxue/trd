#!/usr/bin/env bash
# Render a trd JSONL frame-parameter file to an animated GIF/WebP.
#
# Pipeline (fully piped, no intermediate files):
#   JSONL --(Arrow IPC: duckdb or pyarrow)--> trd --(tensors)--> ffmpeg
#
# Usage:
#   examples/render.sh [--cli | --native | --web [--arrow-renderer|--canvas-renderer]] \
#                      [--mesh OBJ]... [--wireframe] [--aabb] [INPUT.jsonl] [OUTPUT.gif|.webp] [WIDTH] [HEIGHT] [FPS]
# Defaults: examples/frames.0.0.2.jsonl  output/out.gif  256 256 30
#
# By default (or with --cli, alias --headless) the frame stream is rendered to a
# GIF/WebP via the headless trd-cli.
# With --native (alias --app) it is played live in the interactive trd-app window
# (trd-native); OUTPUT is then ignored and neither uv nor ffmpeg are needed.
# With --mesh OBJ the input is a protocol 0.0.3 stream: a leading mesh table
# (scripts/obj_to_arrow.py encodes the OBJ) concatenated with the params stream,
# so trd renders the loaded mesh (centered + uniformly scaled to fit) driven by
# INPUT.jsonl. Try: examples/render.sh --mesh assets/meshes/bunny.obj \
# examples/frames.turntable.jsonl output/bunny.gif. --mesh is repeatable: pass it
# several times to load several meshes (one table row each, in order); a frame's
# `draws` list then references them by 0-based index. Try the two-mesh demo:
# examples/render.sh --cli --wireframe --mesh assets/meshes/bunny.obj \
# --mesh examples/cube.obj examples/frames.multimesh.jsonl output/scene.gif.
# (--mesh needs pyarrow via uv/python3 and is ignored by --web.)
# With --wireframe (--cli only) trd draws mesh edges as a line list instead of
# filled triangles (protocol #38); combine with --mesh for a wireframe asset.
# With --aabb (--cli only) trd overlays each drawn mesh's axis-aligned bounding
# box as a green wireframe box (#42); combine with --mesh (e.g. add --aabb to the
# bunny turntable to see its box track the rotation).
# With --web (alias --wasm) it builds the browser (wasm) bundle via nix and serves
# it, printing the URLs and an SSH-tunnel command. The web demo generates its own
# Arrow frame stream in-browser, so all positional arguments are ignored. Two
# in-browser renderers share the bundle: --arrow-renderer (default) runs the
# offscreen output-stream smoke (the browser counterpart of the CLI);
# --canvas-renderer runs the on-screen canvas demo. Override the port with PORT
# (default 8080); the server binds all interfaces.
#
# Run from `nix develop`. The Arrow frame stream is built with duckdb's 'arrow'
# community extension when it loads, otherwise with pyarrow (via uv/python3).
# On WSL, prefix with WGPU_BACKEND=gl for GPU rendering (else it uses software).
set -euo pipefail

# Extract the optional mode flags (--cli/--native/--web), the --web renderer
# sub-flags (--arrow-renderer/--canvas-renderer), and repeatable --mesh <obj>
# flags that prepend a mesh Arrow stream (0.0.3 [mesh][params]); the rest are
# positional.
cli=0
native=0
web=0
arrow_renderer=0
canvas_renderer=0
wireframe=0
aabb=0
meshes=()
positional=()
while [ $# -gt 0 ]; do
  case "$1" in
    --cli|--headless) cli=1 ;;
    --native|--app) native=1 ;;
    --web|--wasm) web=1 ;;
    --arrow-renderer) arrow_renderer=1 ;;
    --canvas-renderer) canvas_renderer=1 ;;
    --wireframe) wireframe=1 ;;
    --aabb) aabb=1 ;;
    --mesh) shift; meshes+=("${1:?--mesh requires an OBJ path}") ;;
    --mesh=*) meshes+=("${1#--mesh=}") ;;
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
if [ "$arrow_renderer" -eq 1 ] && [ "$canvas_renderer" -eq 1 ]; then
  echo "error: --arrow-renderer and --canvas-renderer are mutually exclusive" >&2
  exit 1
fi
if { [ "$arrow_renderer" -eq 1 ] || [ "$canvas_renderer" -eq 1 ]; } && [ "$web" -ne 1 ]; then
  echo "error: --arrow-renderer / --canvas-renderer apply only to --web/--wasm" >&2
  exit 1
fi

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
input="${1:-$root/examples/frames.0.0.2.jsonl}"
output="${2:-output/out.gif}"
width="${3:-256}"
height="${4:-256}"
fps="${5:-30}"
# DuckDB SQL string literals escape a single quote by doubling it.
sql_input=${input//\'/\'\'}

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

# --web/--wasm: build the browser (wasm) bundle and serve it. The demo generates
# its own Arrow frame stream in-browser, so the positional arguments are ignored.
if [ "$web" -eq 1 ]; then
  port="${PORT:-8080}"
  user="$(whoami)"
  # First non-loopback IPv4 of this host (for the direct / SSH-tunnel URLs).
  ip="$(hostname -I 2>/dev/null | awk '{print $1; exit}')"
  [ -n "$ip" ] || ip="<server-ip>"

  # Both browser renderers ship in one bundle; web/src/main.ts routes on the
  # `arrow-smoke` query param, so only the opened URL differs. Default (or
  # --arrow-renderer) = the offscreen ArrowRenderer output-stream smoke (the
  # browser counterpart of --cli); --canvas-renderer = the on-screen canvas demo.
  if [ "$canvas_renderer" -eq 1 ]; then
    demo_query=""
    renderer_label="CanvasRenderer (on-screen canvas demo)"
  else
    demo_query="?arrow-smoke"
    renderer_label="ArrowRenderer (offscreen output-stream smoke)"
  fi

  echo "building trd web (wasm) bundle via nix (.#web)…" >&2
  (cd "$root" && nix build --no-link ".#web")

  cat <<EOF

trd web (wasm) server — port $port  (press Ctrl-C to stop)
  renderer: $renderer_label

  On this machine:        http://localhost:$port$demo_query
  Direct (same network):  http://$ip:$port$demo_query

  SSH tunnel (recommended if the port is not directly reachable):
    ssh -L $port:localhost:$port $user@$ip
  then open in a WebGPU browser (Chrome/Edge):
                          http://localhost:$port$demo_query

EOF

  cd "$root"
  exec env PORT="$port" nix run ".#web"
fi

# Choose a frame producer: DuckDB's 'arrow' community extension if it loads,
# otherwise scripts/jsonl_to_arrow.py via uv (or python3 with pyarrow).
if command -v duckdb >/dev/null 2>&1 && duckdb -c "INSTALL arrow FROM community; LOAD arrow;" >/dev/null 2>&1; then
  producer=duckdb
elif command -v uv >/dev/null 2>&1; then
  producer=uv
elif command -v python3 >/dev/null 2>&1 && python3 -c 'import pyarrow' >/dev/null 2>&1; then
  producer=python3
else
  echo "error: need duckdb (with the 'arrow' community extension) or uv/python3 with pyarrow" >&2
  echo "to build the Arrow frame stream" >&2
  exit 1
fi

# DuckDB's producer only understands the 0.0.1/0.0.2 columns (center/size/theta/
# model); its SQL silently DROPS the additive 0.0.3 camera (eye/target/direction/
# up/k/pose/fovy/aspect/znear/zfar) and instanced draw-list (draws) columns. If
# the input carries any of those, fall back to the pyarrow producer (which emits
# them) so the camera/draw data actually reaches trd — otherwise an authored
# camera is lost and trd renders with the identity camera (z-clipping).
if [ "$producer" = duckdb ] \
  && grep -Eq '"(eye|target|direction|up|k|pose|fovy|aspect|znear|zfar|draws)"[[:space:]]*:' "$input"; then
  if command -v uv >/dev/null 2>&1; then
    producer=uv
  elif command -v python3 >/dev/null 2>&1 && python3 -c 'import pyarrow' >/dev/null 2>&1; then
    producer=python3
  else
    echo "error: '$input' carries 0.0.3 camera/draw columns that DuckDB cannot emit;" >&2
    echo "install uv or python3 with pyarrow to render it" >&2
    exit 1
  fi
fi

# DuckDB reads the JSONL and streams Arrow IPC (FORMAT arrows) to stdout. It emits
# the required 0.0.1 columns (center/size as FixedSizeList<f32>[2], theta as f32,
# defaulting to the identity when a row omits them) plus the additive 0.0.2 `model`
# column (FixedSizeList<f32>[16], column-major): a row's explicit `model` is used
# verbatim, else synthesized as translate(center).rotate_z(theta).scale(size) to
# match scripts/jsonl_to_arrow.py. The explicit `columns=` schema makes every
# column exist (NULL when a row omits it), so one query renders both the 0.0.1
# (center/size/theta) and 0.0.2 (model) example data.
sql="INSTALL arrow FROM community; LOAD arrow;
  COPY (
    WITH raw AS (
      SELECT
        COALESCE(center, [0.0, 0.0]) AS c,
        COALESCE(size, [1.0, 1.0]) AS s,
        COALESCE(theta, 0.0) AS th,
        model AS m
      FROM read_json('$sql_input',
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
  ) TO '/dev/stdout' (FORMAT arrows);"

# Emit the Arrow IPC frame stream on stdout via the chosen producer.
frames() {
  case "$producer" in
    duckdb) duckdb -c "$sql" ;;
    uv) uv run --with pyarrow "$root/scripts/jsonl_to_arrow.py" "$input" ;;
    python3) python3 "$root/scripts/jsonl_to_arrow.py" "$input" ;;
  esac
}

# When rendering loaded meshes (--mesh, repeatable), pick a pyarrow-capable
# Python to encode the OBJ(s) into the leading mesh Arrow stream (duckdb can't
# author the nested-list mesh table). scripts/obj_to_arrow.py emits a 0.0.3 mesh
# table with **one row per --mesh** (in the order given); it is concatenated
# *before* the params stream so trd reads [mesh][params]. A frame's `draws` list
# references these meshes by 0-based index (mesh 0 = first --mesh).
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

# The full trd input stream: the optional leading mesh table, then the params.
stream() {
  case "$mesh_producer" in
    uv) uv run --with pyarrow "$root/scripts/obj_to_arrow.py" "${meshes[@]}" ;;
    python3) python3 "$root/scripts/obj_to_arrow.py" "${meshes[@]}" ;;
  esac
  frames
}

if [ "$native" -eq 1 ]; then
  # Play the frame stream live in the interactive trd-app window (trd-native).
  stream \
    | cargo run --manifest-path "$root/Cargo.toml" -q -p trd-app -- --width "$width" --height "$height" --fps "$fps"
  echo "streamed $input to the trd-app window (${width}x${height}, ${fps}fps)"
else
  mkdir -p "$(dirname "$output")"
  wireframe_flag=()
  [ "$wireframe" -eq 1 ] && wireframe_flag=(--wireframe)
  aabb_flag=()
  [ "$aabb" -eq 1 ] && aabb_flag=(--aabb)
  stream \
    | cargo run --manifest-path "$root/Cargo.toml" -q -p trd-cli -- --width "$width" --height "$height" "${wireframe_flag[@]}" "${aabb_flag[@]}" \
    | uv run --with pyarrow --with numpy "$root/scripts/encode.py" --fps "$fps" -o "$output"
  echo "wrote $output (${width}x${height}, ${fps}fps) from $input"
fi
