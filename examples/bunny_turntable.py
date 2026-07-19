#!/usr/bin/env python3
"""Author ``examples/frames.turntable.jsonl`` — a fixed-camera turntable spin.

This is the *params* half of the bunny turntable demo (#44). Each frame carries
only a **model matrix** that rotates the mesh about its +Y axis by an even
fraction of a full turn:

    {"model": [16 floats]}   # column-major rotate_y(2π·i/N)

No camera columns are emitted, so ``trd-core`` uses its **default AABB-fit
fixed camera**: the camera stays put and frames the mesh's bounding box while
the mesh spins in place — a classic turntable. The mesh itself (the Stanford
bunny, a cube, …) is supplied separately as the *leading mesh table* of the
``[mesh][params]`` stream, e.g. via ``examples/render.sh --mesh``:

    examples/render.sh --cli --aabb --axes \\
        --mesh assets/meshes/bunny.obj \\
        examples/frames.turntable.jsonl output/bunny_turntable.gif

The same params also drive the wasm canvas demo (``web/src/canvas-demo.ts``),
which authors its own cube mesh table in TypeScript and streams these frames.

Run (from the repo root, inside ``nix develop`` or any plain Python 3):
    python examples/bunny_turntable.py
    python examples/bunny_turntable.py --frames 72 --out examples/frames.turntable.jsonl
"""
import argparse
import json
import math
import os


def rotate_y(theta):
    """Column-major 4×4 ``rotate_y(theta)`` matching glam ``Mat4::from_rotation_y``.

    Columns: ``(c,0,-s,0) (0,1,0,0) (s,0,c,0) (0,0,0,1)`` — a right-handed
    rotation about +Y, identical to the native ``rotate_y`` used by the other
    example producers.
    """
    c, s = math.cos(theta), math.sin(theta)
    return [
        c, 0.0, -s, 0.0,
        0.0, 1.0, 0.0, 0.0,
        s, 0.0, c, 0.0,
        0.0, 0.0, 0.0, 1.0,
    ]


def main() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--frames", type=int, default=48, help="frame count / full-turn steps (default 48)"
    )
    ap.add_argument(
        "--out",
        default=os.path.join(here, "frames.turntable.jsonl"),
        help="output JSONL path (default examples/frames.turntable.jsonl)",
    )
    args = ap.parse_args()

    n = max(args.frames, 1)

    def r6(xs):
        return [round(x, 6) for x in xs]

    rows = [{"model": r6(rotate_y(2.0 * math.pi * i / n))} for i in range(n)]

    with open(args.out, "w", encoding="utf-8", newline="\n") as out:
        for r in rows:
            out.write(json.dumps(r) + "\n")

    print(f"wrote {n} turntable frames -> {args.out} (rotate_y, {360.0 / n:.3g}°/frame)")


if __name__ == "__main__":
    main()
