#!/usr/bin/env python3
"""Generate `examples/frames.0.0.2.jsonl` from `examples/frames.0.0.1.jsonl`.

The 0.0.1 example authors each frame as `{center, size, theta}`. This script
converts every frame into a **protocol 0.0.2** row that carries the triangle's
4x4 transformation matrix *directly*:

    {"model": [16 floats]}   # column-major translate(center) . rotate_z(theta) . scale(size)

matching `trd-core`'s `model_from_2d_affine` (glam). The 0.0.2 example is thus the
same animation as 0.0.1, expressed purely as transformation matrices.

Run (from the repo root, inside `nix develop` or with a pyarrow-free Python):
    python scripts/gen_frames.py
    python scripts/gen_frames.py -i examples/frames.0.0.1.jsonl -o examples/frames.0.0.2.jsonl
"""
import argparse
import json
import math
import os


def model_matrix(center, size, theta):
    """Column-major 4x4 `translate(center) . rotate_z(theta) . scale(size)`."""
    c, s = math.cos(theta), math.sin(theta)
    sx, sy = size[0], size[1]
    tx, ty = center[0], center[1]
    return [
        sx * c, sx * s, 0.0, 0.0,   # col 0
        -sy * s, sy * c, 0.0, 0.0,  # col 1
        0.0, 0.0, 1.0, 0.0,         # col 2
        tx, ty, 0.0, 1.0,           # col 3
    ]


def main() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(here)
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "-i", "--input", default=os.path.join(root, "examples", "frames.0.0.1.jsonl")
    )
    ap.add_argument(
        "-o", "--output", default=os.path.join(root, "examples", "frames.0.0.2.jsonl")
    )
    args = ap.parse_args()

    with open(args.input, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]

    with open(args.output, "w", encoding="utf-8", newline="\n") as out:
        for r in rows:
            model = model_matrix(r["center"], r["size"], r["theta"])
            out.write(json.dumps({"model": [round(x, 6) for x in model]}) + "\n")

    print(f"wrote {len(rows)} rows -> {args.output}")


if __name__ == "__main__":
    main()
