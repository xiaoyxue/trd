#!/usr/bin/env python3
"""Convert a trd JSONL frame-params file to an Arrow IPC stream.

A dependency-light alternative to the DuckDB `arrow` community extension for
examples/render.sh and examples/render.ps1. Each JSON line is one frame and may
carry either the legacy 0.0.1 params or the 0.0.2 matrix directly:

  0.0.1:  {"center": [x, y], "size": [sx, sy], "theta": t}
  0.0.2:  {"model": [16 floats]}   # column-major 4x4 triangle transform

`--version` selects the wire protocol (default: the latest, `0.0.2`):

  * `0.0.2` emits `center`/`size`/`theta` **plus** an explicit `model` column
    (`FixedSizeList<f32>[16]`, column-major, matching glam `Mat4::from_cols_array`).
    A row's `model` is used verbatim if present, else synthesized as
    `translate(center) . rotate_z(theta) . scale(size)`. Missing `center`/`size`/
    `theta` default to the identity (`[0,0]`/`[1,1]`/`0`).
  * `0.0.1` emits only `center`/`size`/`theta` (legacy; no `model` column).

Run via:
  uv run --with pyarrow scripts/jsonl_to_arrow.py examples/frames.0.0.2.jsonl   # -> stdout (0.0.2)
  python scripts/jsonl_to_arrow.py --version 0.0.1 examples/frames.0.0.1.jsonl  # legacy 0.0.1
  python scripts/jsonl_to_arrow.py frames.jsonl -o frames.arrows                # -> file
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
        "--version",
        choices=["0.0.1", "0.0.2"],
        default="0.0.2",
        help="wire protocol version to emit (default: latest, 0.0.2)",
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

    # Rows may carry either 0.0.1 params or a 0.0.2 model matrix; fill the
    # required legacy columns with the identity when only a `model` is given.
    def center(r):
        return r.get("center", [0.0, 0.0])

    def size(r):
        return r.get("size", [1.0, 1.0])

    def theta(r):
        return r.get("theta", 0.0)

    f32 = pa.float32()
    fsl2 = pa.list_(f32, 2)  # FixedSizeList<f32>[2]

    metadata = {PROTOCOL_VERSION_KEY: args.version.encode()}
    if args.fps and args.fps > 0:
        metadata[FRAME_RATE_KEY] = str(args.fps).encode()

    columns = [
        pa.array([center(r) for r in rows], type=fsl2),
        pa.array([size(r) for r in rows], type=fsl2),
        pa.array([theta(r) for r in rows], type=f32),
    ]
    fields = [("center", fsl2), ("size", fsl2), ("theta", f32)]

    if args.version == "0.0.2":
        fsl16 = pa.list_(f32, 16)  # FixedSizeList<f32>[16] = column-major Mat4
        # A `model` row is the explicit matrix if provided, else synthesized.
        model_rows = [
            r["model"] if "model" in r else model_matrix(center(r), size(r), theta(r))
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
