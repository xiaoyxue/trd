#!/usr/bin/env bash
# Render a trd JSONL frame-parameter file to an animated GIF/WebP.
#
# Pipeline (fully piped, no intermediate files):
#   JSONL --(Arrow IPC: duckdb or pyarrow)--> trd --(tensors)--> ffmpeg
#
# Usage:
#   examples/render.sh [--native|--app] [INPUT.jsonl] [OUTPUT.gif|.webp] [WIDTH] [HEIGHT] [FPS]
# Defaults: examples/frames.jsonl  out.gif  256 256 30
#
# By default the frame stream is rendered to a GIF/WebP via the headless trd-cli.
# With --native (alias --app) it is played live in the interactive trd-app window
# (trd-native); OUTPUT is then ignored and neither uv nor ffmpeg are needed.
#
# Run from `nix develop`. The Arrow frame stream is built with duckdb's 'arrow'
# community extension when it loads, otherwise with pyarrow (via uv/python3).
# On WSL, prefix with WGPU_BACKEND=gl for GPU rendering (else it uses software).
set -euo pipefail

# Extract the optional --native/--app flag; the rest are positional.
native=0
positional=()
for arg in "$@"; do
  case "$arg" in
    --native|--app) native=1 ;;
    *) positional+=("$arg") ;;
  esac
done
if [ ${#positional[@]} -gt 0 ]; then set -- "${positional[@]}"; else set --; fi

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
input="${1:-$root/examples/frames.jsonl}"
output="${2:-out.gif}"
width="${3:-256}"
height="${4:-256}"
fps="${5:-30}"
# DuckDB SQL string literals escape a single quote by doubling it.
sql_input=${input//\'/\'\'}

# cargo is always required; the GIF path also needs ffmpeg (and uv for encoding).
if [ "$native" -eq 1 ]; then tools="cargo"; else tools="cargo ffmpeg uv"; fi
for tool in $tools; do
  if ! command -v "$tool" >/dev/null 2>&1; then
    echo "error: $tool not found on PATH" >&2
    echo "run this example inside 'nix develop'" >&2
    exit 1
  fi
done

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

# DuckDB reads the JSONL, casts the [x,y] arrays to fixed-size FLOAT[2]
# (Arrow FixedSizeList<f32>[2]), and streams Arrow IPC (FORMAT arrows) to stdout.
sql="INSTALL arrow FROM community; LOAD arrow;
  COPY (
    SELECT center::FLOAT[2] AS center, size::FLOAT[2] AS size, theta::FLOAT AS theta
    FROM read_json_auto('$sql_input')
  ) TO '/dev/stdout' (FORMAT arrows);"

# Emit the Arrow IPC frame stream on stdout via the chosen producer.
frames() {
  case "$producer" in
    duckdb) duckdb -c "$sql" ;;
    uv) uv run --with pyarrow "$root/scripts/jsonl_to_arrow.py" "$input" ;;
    python3) python3 "$root/scripts/jsonl_to_arrow.py" "$input" ;;
  esac
}

if [ "$native" -eq 1 ]; then
  # Play the frame stream live in the interactive trd-app window (trd-native).
  frames \
    | cargo run --manifest-path "$root/Cargo.toml" -q -p trd-app -- --width "$width" --height "$height" --fps "$fps"
  echo "streamed $input to the trd-app window (${width}x${height}, ${fps}fps)"
else
  frames \
    | cargo run --manifest-path "$root/Cargo.toml" -q -p trd-cli -- --width "$width" --height "$height" \
    | uv run --with pyarrow --with numpy "$root/scripts/encode.py" --fps "$fps" -o "$output"
  echo "wrote $output (${width}x${height}, ${fps}fps) from $input"
fi
