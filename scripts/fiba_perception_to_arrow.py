#!/usr/bin/env python3
"""Adapt the FIBA court-calibration parquet into the trd **perception** Arrow stream.

FIBA twin of ``scripts/nba_perception_to_arrow.py`` (#95): the upstream research
dataset ``per_frame_KVP_cube_best.parquet`` (VideoAnalysis#1133; vendored under
``assets/videos/fiba/`` — see that dir's ``DATASET.md``) already solves the hard
part of the FIBA court AR demo (#110): per broadcast frame it carries the camera
intrinsics ``K`` and a tracked planar floor quad (``ad_quad`` — the "ad-unit"
rectangle the reference AR cube stands on). That is *exactly* the input trd's
single-view placement stage (``examples/placement_quad_by_local_coord.py
--from-perception``) consumes, so this adapter is a thin repack: it emits the
**same** perception schema as ``scripts/perception_to_arrow.py`` (``k`` +
``placement_quad`` + ``frame_path``), just sourced from the parquet.

Difference from the NBA adapter — the ``*_best.parquet`` keeps the whole 288-frame
timeline but stores **only the frame identity on untracked frames** (geometry
columns ``K``/``ad_quad``/… are ``null`` once the ad quad leaves the frame). This
adapter therefore **skips untracked / null-geometry rows** and emits only the
tracked frames (``present_index`` 0..221 for shot 1).

Mapping (parquet → perception stream), per tracked ``(shot, method, present_index)``
row:

  * ``K`` (3×3, **row-major** ``[f,0,cx, 0,f,cy, 0,0,1]``) → ``k``.
  * ``ad_quad`` (4 floor corners ``UL,UR,LR,LL`` as ``[x0,y0,…,x3,y3]``, source
    pixels) → ``placement_quad``.
  * ``present_index`` (0-based frame index into ``shot_0001.mp4``) → ``frame_path``
    ``"<frame-rel>/frame_%06d.<ext>"``.

All image quantities are in **source-video pixels** (1920×1080); the downstream
stage scales ``K`` to the render resolution (pass its ``--src-width 1920
--src-height 1080``). Use the trustworthy focal method — ``2VP_4510`` (best
held-out accuracy) or its corroborator ``1circle_4252``; ``BA`` is zoom-degraded
on this footage (the inverse of nba-short).

Run (from ``nix develop``)::

    uv run --with pyarrow scripts/fiba_perception_to_arrow.py \
        --method 2VP_4510 \
        -o examples/frames.fiba.perception.arrow

The calibration parquet is vendored at
``assets/videos/fiba/per_frame_KVP_cube_best.parquet`` (the ``--parquet``
default), so no external dataset is needed. It prints the emitted
``present_index`` range to stderr so the offline frame extractor knows which
frames of the (un-vendored, copyrighted) ``shot_0001.mp4`` to pull — they must
land at ``<frames-base>/<frame-rel>/frame_<present_index>.<ext>``.
"""
import argparse
import sys

import pyarrow as pa
import pyarrow.parquet as pq
from pyarrow import ipc

# Mirror scripts/perception_to_arrow.py so the stream is indistinguishable from
# the cornellbox / nba one to the downstream stage.
PERCEPTION_STAGE_KEY = b"trd.pipeline.stage"
PERCEPTION_STAGE_VALUE = b"perception"


def main():
    ap = argparse.ArgumentParser(
        description="Repack the FIBA K/ad_quad parquet into the trd perception Arrow stream (#110).")
    ap.add_argument("--parquet", default="assets/videos/fiba/per_frame_KVP_cube_best.parquet",
                    help="per_frame_KVP_cube_best.parquet (VideoAnalysis#1133 FIBA dataset; "
                         "the calibration is vendored under assets/videos/fiba/)")
    ap.add_argument("--shot", type=int, default=1, help="FIBA shot id (only 1 in this dataset)")
    ap.add_argument("--method", default="2VP_4510",
                    help="K-estimation method (2VP_4510=best, 1circle_4252=corroborator)")
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

    # The *_best.parquet keeps untracked frames as frame-id-only rows (geometry
    # null); skip them so the perception stream carries only drawable frames.
    tracked = [r for r in picked if r.get("K") is not None and r.get("ad_quad") is not None]
    skipped = len(picked) - len(tracked)
    if not tracked:
        raise SystemExit(
            f"error: all {len(picked)} rows for shot={args.shot} method={args.method} "
            "have null geometry (no tracked frames)")

    k_col, quad_col, frame_col = [], [], []
    for r in tracked:
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

    pmin = tracked[0]["present_index"]
    pmax = tracked[-1]["present_index"]
    f = float(tracked[0]["K"][0])
    dest = "stdout" if args.output == "-" else args.output
    print(
        f"wrote {len(tracked)} perception rows (shot {args.shot}, {args.method}, "
        f"f={f:.0f}px) to {dest}; skipped {skipped} untracked frame(s); "
        f"present_index {pmin}..{pmax} (extract these frames of shot_0001.mp4 to "
        f"<frames-base>/{args.frame_rel}/frame_%06d.{args.frame_ext})",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
