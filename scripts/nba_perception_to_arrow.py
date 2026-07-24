#!/usr/bin/env python3
"""Adapt the NBA court-calibration parquet into the trd **perception** Arrow stream.

The upstream research dataset ``per_frame_KVP_cube.parquet`` (VideoAnalysis#1133;
see ``~/Asset/nba-short/``) already solves the hard part of issue #95: per broadcast
frame it carries the camera intrinsics ``K`` and a tracked planar floor quad
(``ad_quad`` — the "ad-unit" rectangle the reference AR cube stands on). That is
*exactly* the input trd's single-view placement stage
(``examples/placement_quad_by_local_coord.py --from-perception``) consumes, so this
adapter is a thin repack: it emits the **same** perception schema as
``scripts/perception_to_arrow.py`` (``k`` + ``placement_quad`` + ``frame_path``),
just sourced from the parquet instead of the cornellbox ``K.txt`` /
``QuadImagePoints.txt`` fixture.

Mapping (parquet → perception stream), per ``(shot, method, present_index)`` row:

  * ``K`` (3×3, **row-major** ``[f,0,cx, 0,f,cy, 0,0,1]``) → ``k``  — already the
    row-major layout the downstream reader (`read_perception_records`) expects.
  * ``ad_quad`` (4 floor corners ``UL,UR,LR,LL`` as ``[x0,y0,…,x3,y3]``, source
    pixels) → ``placement_quad``  — same ring ordering the unit-square DLT wants.
  * ``present_index`` (0-based frame index into ``NBA.mp4``) → ``frame_path``
    ``"<frame-rel>/frame_%06d.<ext>"``  — the still the renderer composites under.

All image quantities are in **source-video pixels** (1920×1080); the downstream
stage scales ``K`` to the render resolution (pass its ``--src-width 1920
--src-height 1080``). Use a ``BA_*`` method — the trustworthy motion-BA focal
(``BA_2511`` for shot 2, ``BA_2568`` for shot 7); ``2VP_*`` is degenerate.

Run (from ``nix develop``)::

    uv run --with pyarrow scripts/nba_perception_to_arrow.py \
        --parquet ~/Asset/nba-short/per_frame_KVP_cube.parquet \
        --shot 2 --method BA_2511 \
        -o examples/frames.nba.perception.arrow

It prints the emitted ``present_index`` range to stderr so the offline frame
extractor knows which frames of ``NBA.mp4`` to pull (they must land at
``<frames-base>/<frame-rel>/frame_<present_index>.<ext>``).
"""
import argparse
import sys

import pyarrow as pa
import pyarrow.parquet as pq
from pyarrow import ipc

# Mirror scripts/perception_to_arrow.py so the stream is indistinguishable from
# the cornellbox one to the downstream stage.
PERCEPTION_STAGE_KEY = b"trd.pipeline.stage"
PERCEPTION_STAGE_VALUE = b"perception"


def main():
    ap = argparse.ArgumentParser(
        description="Repack the NBA K/ad_quad parquet into the trd perception Arrow stream (#95).")
    ap.add_argument("--parquet", required=True,
                    help="per_frame_KVP_cube.parquet (VideoAnalysis#1133 NBA dataset)")
    ap.add_argument("--shot", type=int, default=2, help="NBA shot id (2 or 7)")
    ap.add_argument("--method", default="BA_2511",
                    help="K-estimation method (use a BA_* focal; BA_2511=shot2, BA_2568=shot7)")
    ap.add_argument("--frame-rel", default="frames",
                    help="frame_path prefix relative to the renderer's --frames-base")
    ap.add_argument("--frame-ext", default="jpg", help="still extension (jpg|png)")
    ap.add_argument("-o", "--output", default="-",
                    help="output Arrow IPC path (default: stdout)")
    args = ap.parse_args()

    table = pq.read_table(args.parquet)

    cols = table.column_names
    for need in ("shot", "method", "present_index", "K", "ad_quad"):
        if need not in cols:
            raise SystemExit(f"error: parquet missing '{need}' column (have {cols})")

    rows = table.to_pylist()
    picked = [r for r in rows if r["shot"] == args.shot and r["method"] == args.method]
    picked.sort(key=lambda r: r["present_index"])
    if not picked:
        methods = sorted({r["method"] for r in rows if r["shot"] == args.shot})
        raise SystemExit(
            f"error: no rows for shot={args.shot} method={args.method}; "
            f"available methods for that shot: {methods}")

    k_col, quad_col, frame_col = [], [], []
    for r in picked:
        K = [float(x) for x in r["K"]]            # row-major 3×3, already 9 floats
        quad = [float(x) for x in r["ad_quad"]]   # x0,y0,…,x3,y3 (UL,UR,LR,LL)
        if len(K) != 9:
            raise SystemExit(f"error: K has {len(K)} elems (want 9)")
        if len(quad) != 8:
            raise SystemExit(f"error: ad_quad has {len(quad)} elems (want 8)")
        pi = int(r["present_index"])
        k_col.append(K)
        quad_col.append(quad)
        frame_col.append(f"{args.frame_rel}/frame_{pi:06d}.{args.frame_ext}")

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
        with ipc.new_stream(sys.stdout.buffer, schema) as writer:
            writer.write_batch(batch)
    else:
        with open(args.output, "wb") as fh, ipc.new_stream(fh, schema) as writer:
            writer.write_batch(batch)

    pmin = picked[0]["present_index"]
    pmax = picked[-1]["present_index"]
    f = float(picked[0]["K"][0])
    dest = "stdout" if args.output == "-" else args.output
    print(
        f"wrote {len(picked)} perception rows (shot {args.shot}, {args.method}, "
        f"f={f:.0f}px) to {dest}; present_index {pmin}..{pmax} "
        f"(extract these frames of NBA.mp4 to <frames-base>/{args.frame_rel}/"
        f"frame_%06d.{args.frame_ext})",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
