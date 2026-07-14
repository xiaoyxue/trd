#!/usr/bin/env python3
"""Convert a trd JSONL frame-params file to an Arrow IPC stream.

A dependency-light alternative to the DuckDB `arrow` community extension for
examples/render.sh and examples/render.ps1: it reads one JSON object per line
(`{"center": [x, y], "size": [sx, sy], "theta": t}`) and writes the trd stream
protocol 0.0.1 input columns as an Arrow IPC stream -- `center`/`size` as
`FixedSizeList<f32>[2]` and `theta` as `f32`. Run via:

  uv run --with pyarrow scripts/jsonl_to_arrow.py frames.jsonl        # -> stdout
  python scripts/jsonl_to_arrow.py frames.jsonl -o frames.arrows      # -> file

The protocol-version metadata is optional (trd-core accepts its absence), so the
stream is consumed as-is by trd-cli and trd-app.
"""
import argparse
import json
import sys

import pyarrow as pa
from pyarrow import ipc


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("input", help="JSONL frame-params file")
    ap.add_argument("-o", "--output", default="-", help="output path ('-' = stdout)")
    args = ap.parse_args()

    with open(args.input, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]

    f32 = pa.float32()
    fsl = pa.list_(f32, 2)  # FixedSizeList<f32>[2]
    schema = pa.schema([("center", fsl), ("size", fsl), ("theta", f32)])
    batch = pa.record_batch(
        [
            pa.array([r["center"] for r in rows], type=fsl),
            pa.array([r["size"] for r in rows], type=fsl),
            pa.array([r["theta"] for r in rows], type=f32),
        ],
        schema=schema,
    )

    sink = sys.stdout.buffer if args.output == "-" else open(args.output, "wb")
    try:
        with ipc.new_stream(sink, schema) as writer:
            writer.write_batch(batch)
    finally:
        if sink is not sys.stdout.buffer:
            sink.close()


if __name__ == "__main__":
    main()
