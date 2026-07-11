#!/usr/bin/env bash
# Render a trd JSONL frame-parameter file to an animated GIF/WebP.
#
# Pipeline (fully piped, no intermediate files):
#   JSONL --(duckdb: cast to FLOAT[2], Arrow IPC)--> trd --(tensors)--> ffmpeg
#
# Usage:
#   examples/render.sh [INPUT.jsonl] [OUTPUT.gif|.webp] [WIDTH] [HEIGHT] [FPS]
# Defaults: examples/frames.jsonl  out.gif  256 256 30
#
# Run from `nix develop`. Also requires duckdb on PATH (for example, install it
# system-wide or start the shell with a profile that provides it).
# On WSL, prefix with WGPU_BACKEND=gl for GPU rendering (else it uses software).
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
input="${1:-$root/examples/frames.jsonl}"
output="${2:-out.gif}"
width="${3:-256}"
height="${4:-256}"
fps="${5:-30}"
# DuckDB SQL string literals escape a single quote by doubling it.
sql_input=${input//\'/\'\'}

for tool in cargo uv ffmpeg duckdb; do
  if ! command -v "$tool" >/dev/null 2>&1; then
    echo "error: $tool not found on PATH" >&2
    echo "run this example inside 'nix develop'; duckdb is an external dependency" >&2
    exit 1
  fi
done

# DuckDB reads the JSONL, casts the [x,y] arrays to fixed-size FLOAT[2]
# (Arrow FixedSizeList<f32>[2]), and streams Arrow IPC (FORMAT arrows) to stdout.
# INSTALL is idempotent; it only hits the network on first use.
duckdb -c "INSTALL arrow FROM community; LOAD arrow;
  COPY (
    SELECT center::FLOAT[2] AS center, size::FLOAT[2] AS size, theta::FLOAT AS theta
    FROM read_json_auto('$sql_input')
  ) TO '/dev/stdout' (FORMAT arrows);" \
  | cargo run --manifest-path "$root/Cargo.toml" -q -p trd-cli -- --width "$width" --height "$height" \
  | uv run --with pyarrow --with numpy "$root/scripts/encode.py" --fps "$fps" -o "$output"

echo "wrote $output (${width}x${height}, ${fps}fps) from $input"
