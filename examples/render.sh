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
# Requires on PATH: duckdb (e.g. `nix run nixpkgs#duckdb`), cargo, uv, ffmpeg.
# On WSL, prefix with WGPU_BACKEND=gl for GPU rendering (else it uses software).
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
input="${1:-$root/examples/frames.jsonl}"
output="${2:-out.gif}"
width="${3:-256}"
height="${4:-256}"
fps="${5:-30}"

if ! command -v duckdb >/dev/null 2>&1; then
  echo "error: duckdb not found on PATH (try: nix run nixpkgs#duckdb)" >&2
  exit 1
fi

# DuckDB reads the JSONL, casts the [x,y] arrays to fixed-size FLOAT[2]
# (Arrow FixedSizeList<f32>[2]), and streams Arrow IPC (FORMAT arrows) to stdout.
# INSTALL is idempotent; it only hits the network on first use.
duckdb -c "INSTALL arrow FROM community; LOAD arrow;
  COPY (
    SELECT center::FLOAT[2] AS center, size::FLOAT[2] AS size, theta::FLOAT AS theta
    FROM read_json_auto('$input')
  ) TO '/dev/stdout' (FORMAT arrows);" \
  | cargo run --manifest-path "$root/Cargo.toml" -q -p trd-cli -- --width "$width" --height "$height" \
  | uv run --with pyarrow --with numpy "$root/scripts/encode.py" --fps "$fps" -o "$output"

echo "wrote $output (${width}x${height}, ${fps}fps) from $input"
