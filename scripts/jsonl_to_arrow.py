#!/usr/bin/env python3
"""Convert a trd JSONL frame-params file to an Arrow IPC stream.

A dependency-light alternative to the DuckDB `arrow` community extension for
examples/render.sh and examples/render.ps1: it reads one JSON object per line
(`{"center": [x, y], "size": [sx, sy], "theta": t}`).

By default it emits a **protocol 0.0.2** stream: the legacy `center`/`size`/`theta`
columns *plus* an explicit `model` column — the 4x4 triangle transformation matrix
`translate(center) . rotate_z(theta) . scale(size)`, column-major
(`FixedSizeList<f32>[16]`, matching glam `Mat4::from_cols_array`). This threads the
new matrix protocol through the whole stack. Pass `--v1` to emit the legacy 0.0.1
stream (no `model` column) for byte-for-byte regression.

Run via:
  uv run --with pyarrow scripts/jsonl_to_arrow.py frames.jsonl        # -> stdout
  python scripts/jsonl_to_arrow.py frames.jsonl -o frames.arrows      # -> file
  python scripts/jsonl_to_arrow.py --v1 frames.jsonl                  # legacy 0.0.1
"""
import argparse
import json
import math
import sys

import pyarrow as pa
from pyarrow import ipc

PROTOCOL_VERSION_KEY = b"trd.protocol.version"
FRAME_RATE_KEY = b"trd.stream.frame_rate"


def model_matrix(center, size, theta):
    """Column-major 4x4 `translate(center) . rotate_z(theta) . scale(size)`.

    Matches trd-core's `model_from_2d_affine` (glam), so the 0.0.2 `model` path
    renders identically to the 0.0.1 `center/size/theta` path.
    """
    c, s = math.cos(theta), math.sin(theta)
    sx, sy = size[0], size[1]
    tx, ty = center[0], center[1]
    # Columns of T*R*S (each group of 4 is one column):
    return [
        sx * c, sx * s, 0.0, 0.0,   # col 0
        -sy * s, sy * c, 0.0, 0.0,  # col 1
        0.0, 0.0, 1.0, 0.0,         # col 2
        tx, ty, 0.0, 1.0,           # col 3
    ]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("input", help="JSONL frame-params file")
    ap.add_argument("-o", "--output", default="-", help="output path ('-' = stdout)")
    ap.add_argument(
        "--v1",
        action="store_true",
        help="emit legacy protocol 0.0.1 (center/size/theta only, no model column)",
    )
    ap.add_argument(
        "--fps",
        type=float,
        default=None,
        help="declare trd.stream.frame_rate metadata (playback fps)",
    )
    args = ap.parse_args()

    with open(args.input, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]

    f32 = pa.float32()
    fsl2 = pa.list_(f32, 2)  # FixedSizeList<f32>[2]

    metadata = {}
    if args.fps and args.fps > 0:
        metadata[FRAME_RATE_KEY] = str(args.fps).encode()

    columns = [
        pa.array([r["center"] for r in rows], type=fsl2),
        pa.array([r["size"] for r in rows], type=fsl2),
        pa.array([r["theta"] for r in rows], type=f32),
    ]
    fields = [("center", fsl2), ("size", fsl2), ("theta", f32)]

    if args.v1:
        metadata[PROTOCOL_VERSION_KEY] = b"0.0.1"
    else:
        metadata[PROTOCOL_VERSION_KEY] = b"0.0.2"
        fsl16 = pa.list_(f32, 16)  # FixedSizeList<f32>[16] = column-major Mat4
        # A `model` row is the explicit matrix if provided, else synthesized.
        model_rows = [
            r["model"] if "model" in r else model_matrix(r["center"], r["size"], r["theta"])
            for r in rows
        ]
        columns.append(pa.array(model_rows, type=fsl16))
        fields.append(("model", fsl16))

    schema = pa.schema(fields, metadata=metadata)
    batch = pa.record_batch(columns, schema=schema)

    sink = sys.stdout.buffer if args.output == "-" else open(args.output, "wb")
    try:
        with ipc.new_stream(sink, schema) as writer:
            writer.write_batch(batch)
    finally:
        if sink is not sys.stdout.buffer:
            sink.close()


if __name__ == "__main__":
    main()
