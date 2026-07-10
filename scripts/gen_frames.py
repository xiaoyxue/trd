#!/usr/bin/env python3
"""Generate a trd protocol 0.0.1 input Arrow IPC stream on stdout.

Sweeps `theta` from 0 to 2*pi over N frames with a fixed center and size.
Run via: uv run --with pyarrow --with numpy scripts/gen_frames.py --frames 60
"""
import argparse
import math
import sys

import numpy as np
import pyarrow as pa
from pyarrow import ipc


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=60)
    ap.add_argument("--size", type=float, default=0.6)
    args = ap.parse_args()

    n = args.frames
    center = pa.FixedSizeListArray.from_arrays(
        pa.array([0.0, 0.0] * n, pa.float32()), 2
    )
    size = pa.FixedSizeListArray.from_arrays(
        pa.array([args.size, args.size] * n, pa.float32()), 2
    )
    theta = pa.array([i * 2 * math.pi / n for i in range(n)], pa.float32())
    schema = pa.schema(
        [
            pa.field("center", center.type),
            pa.field("size", size.type),
            pa.field("theta", pa.float32()),
        ],
        metadata={b"trd.protocol.version": b"0.0.1"},
    )
    batch = pa.record_batch([center, size, theta], schema=schema)
    with ipc.new_stream(sys.stdout.buffer, schema) as writer:
        writer.write_batch(batch)


if __name__ == "__main__":
    main()
