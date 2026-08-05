#!/usr/bin/env python3
"""Simulate the **upstream perception stage** output as an Arrow IPC stream.

In the cornellbox AR pipeline the *upstream* stage is a perception/tracker that,
per video frame, emits only what it observed — **no scene, no placement**:

  * ``k``              — the camera intrinsics ``K`` (a 3×3 matrix, row-major, in the
                         source video's pixel space), as ``FixedSizeList<Float32>[9]``.
  * ``placement_quad`` — the 4 image-space quad corner points it detected on the
                         filmed poster ``[x0,y0, x1,y1, x2,y2, x3,y3]`` (source pixels),
                         as ``FixedSizeList<Float32>[8]``. This quad is the surface the
                         downstream stage places the model mesh onto.
  * ``frame_path``     — the background still for that frame (relative to a
                         ``--frames-base`` the renderer resolves), as ``Utf8``.

This is deliberately *placement-free*: it carries no ``model``/``draws`` columns.
The **downstream** stage (examples/placement_quad_by_local_coord.py --from-perception)
consumes this stream, runs the #77 single-view reconstruction (K + quad → the
plane's ``(e1,e2,e3)`` frame in camera E³), anchors the mesh there, and emits a
render-ready 0.0.6 params stream. Splitting the two stages this way lets the
perception output be produced/checked independently of the renderer.

Here we *simulate* that upstream by reading the recorded ``K.txt`` /
``QuadImagePoints.txt`` fixtures and packing them into the Arrow stream (applying
``--step``/``--limit`` so the upstream owns frame selection).

Run via:
  uv run --with pyarrow --with numpy scripts/perception_to_arrow.py \
    --assets assets/videos/cornellbox -o examples/frames.cornellbox.perception.arrow
"""
import argparse
import os
import re
import sys

import numpy as np
import pyarrow as pa
from pyarrow import ipc

# Arrow metadata key marking this as the perception-stage stream (mirrors the
# `trd.protocol.version` convention of the params producers).
PERCEPTION_STAGE_KEY = b"trd.pipeline.stage"
PERCEPTION_STAGE_VALUE = b"perception"


def parse_k(path):
    """Parse the 3×3 OpenCV intrinsics from ``K.txt`` (skips ``#`` comments)."""
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            rows.append([float(x) for x in line.split()])
    k = np.array(rows, dtype=np.float64)
    if k.shape != (3, 3):
        raise SystemExit(f"error: K.txt is not 3×3: {k.shape}")
    return k


def parse_quads(path):
    """Parse per-frame 4 image-space quad points from ``QuadImagePoints.txt``."""
    pt = re.compile(r"\(\s*([-\d.eE+]+)\s*,\s*([-\d.eE+]+)\s*\)")
    quads = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip().startswith("frame"):
                continue
            pts = [(float(a), float(b)) for a, b in pt.findall(line)]
            if len(pts) != 4:
                raise SystemExit(f"error: expected 4 quad points, got {len(pts)}: {line!r}")
            quads.append(np.array(pts, dtype=np.float64))
    return quads


def main():
    ap = argparse.ArgumentParser(
        description="Simulate the upstream perception stage (K + placement_quad + frame) as an Arrow stream.")
    ap.add_argument("--assets", default="assets/videos/cornellbox",
                    help="dir with K.txt / QuadImagePoints.txt")
    ap.add_argument("--frame-ext", default="jpg", help="still extension (png|jpg)")
    ap.add_argument("--frame-rel", default="frames",
                    help="frame_path prefix relative to the renderer's --frames-base")
    ap.add_argument("--step", type=int, default=2, help="emit every Nth frame")
    ap.add_argument("--limit", type=int, default=None, help="cap number of emitted frames")
    ap.add_argument("-o", "--output", default="-",
                    help="output Arrow IPC path (default: stdout)")
    args = ap.parse_args()

    K = parse_k(os.path.join(args.assets, "K.txt"))
    quads = parse_quads(os.path.join(args.assets, "QuadImagePoints.txt"))
    print(f"parsed {len(quads)} quads; K fx={K[0,0]:.1f} fy={K[1,1]:.1f} "
          f"cx={K[0,2]:.1f} cy={K[1,2]:.1f}", file=sys.stderr)

    idx = list(range(0, len(quads), args.step))
    if args.limit is not None:
        idx = idx[: args.limit]

    # K is constant across frames in the fixture, but the perception stream carries
    # it per-row so downstream stays general (a real tracker may refine K per frame).
    k_row = [float(x) for x in K.flatten(order="C")]  # row-major 3×3
    k_col, quad_col, frame_col = [], [], []
    for f in idx:
        k_col.append(k_row)
        quad_col.append([float(x) for x in quads[f].flatten(order="C")])  # x0,y0,...,x3,y3
        frame_col.append(f"{args.frame_rel}/frame_{f:06d}.{args.frame_ext}")

    schema = pa.schema(
        [
            ("k", pa.list_(pa.float32(), 9)),
            ("placement_quad", pa.list_(pa.float32(), 8)),
            ("frame_path", pa.utf8()),
        ],
        metadata={PERCEPTION_STAGE_KEY: PERCEPTION_STAGE_VALUE},
    )
    batch = pa.record_batch(
        [
            pa.array(k_col, type=pa.list_(pa.float32(), 9)),
            pa.array(quad_col, type=pa.list_(pa.float32(), 8)),
            pa.array(frame_col, type=pa.utf8()),
        ],
        schema=schema,
    )

    if args.output == "-":
        sink = sys.stdout.buffer
        with ipc.new_stream(sink, schema) as writer:
            writer.write_batch(batch)
    else:
        with open(args.output, "wb") as fh, ipc.new_stream(fh, schema) as writer:
            writer.write_batch(batch)
    dest = "stdout" if args.output == "-" else args.output
    print(f"wrote {len(idx)} perception rows (k+placement_quad+frame) to {dest} "
          f"(step {args.step})", file=sys.stderr)


if __name__ == "__main__":
    main()
