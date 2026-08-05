#!/usr/bin/env python3
"""Author the **frame-plane** demo (#63): a folder of animated background stills
plus a turntable JSONL that references them per frame via ``frame_path``.

This is the end-to-end scenario for the background frame-compositing slice: each
frame carries

    {"model": [16 floats],            # column-major rotate_y(2π·i/N)
     "frame_path": "frames/frame_000000.png"}

so ``scripts/jsonl_to_arrow.py`` emits a protocol **0.0.6** stream whose per-frame
``frame_path`` column names a still image. ``trd --frames-base <dir>`` loads each
still at the boundary and composites it *beneath* the spinning bunny via a
``FramePlane`` (the mesh + axes/AABB gizmos draw on top). The backgrounds are an
animated hue-shifting gradient with a bar sweeping left→right, so the rendered GIF
visibly proves the plane texture updates every frame (one reused GPU texture).

The stills are written with a tiny **stdlib-only** PNG encoder (no Pillow), so
this producer runs under any plain Python 3 like the other ``examples/*.py``.
Output lands under ``output/`` (gitignored) — nothing here is committed.

Run (from the repo root, inside ``nix develop`` or any plain Python 3):
    python examples/bunny_frameplane.py --out-dir output/fp_demo
    # then render the composited turntable GIF:
    examples/render.sh --cli --wireframe --axes --aabb \\
        --mesh assets/meshes/bunny.obj \\
        --frames-base output/fp_demo \\
        output/fp_demo/turntable_fp.jsonl output/fp_demo.gif 512 512 24
"""
import argparse
import json
import math
import os
import struct
import zlib


def rotate_y(theta):
    """Column-major 4×4 ``rotate_y(theta)`` matching glam ``Mat4::from_rotation_y``
    (identical to examples/bunny_turntable.py)."""
    c, s = math.cos(theta), math.sin(theta)
    return [
        c, 0.0, -s, 0.0,
        0.0, 1.0, 0.0, 0.0,
        s, 0.0, c, 0.0,
        0.0, 0.0, 0.0, 1.0,
    ]


def write_png_rgb(path, width, height, rgb_rows):
    """Encode an RGB image (list of ``height`` rows, each ``width`` ``(r,g,b)``
    tuples) as a PNG using only the standard library (zlib + struct)."""
    def chunk(tag, data):
        return (
            struct.pack(">I", len(data))
            + tag
            + data
            + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
        )

    raw = bytearray()
    for row in rgb_rows:
        raw.append(0)  # filter type 0 (None) per scanline
        for r, g, b in row:
            raw += bytes((r & 0xFF, g & 0xFF, b & 0xFF))
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)  # 8-bit RGB
    with open(path, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n")
        f.write(chunk(b"IHDR", ihdr))
        f.write(chunk(b"IDAT", zlib.compress(bytes(raw), 9)))
        f.write(chunk(b"IEND", b""))


def background_rows(width, height, phase):
    """An animated hue-shifting vertical gradient with a bright bar sweeping
    left→right by ``phase`` ∈ [0, 1) — an unmistakably per-frame background."""
    bar_x = int(phase * width)
    bar_w = max(6, width // 64)
    rows = []
    for y in range(height):
        t = y / height
        r = int(90 + 90 * math.sin(2 * math.pi * (phase + t)))
        g = int(70 + 60 * math.sin(2 * math.pi * (phase + 0.33 + t)))
        b = int(120 + 90 * math.sin(2 * math.pi * (phase + 0.66 + t)))
        base = (max(0, min(255, r)), max(0, min(255, g)), max(0, min(255, b)))
        row = [base] * width
        for x in range(bar_x, min(width, bar_x + bar_w)):
            row[x] = (255, 240, 40)
        rows.append(row)
    return rows


def main() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(here)
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--out-dir",
        default=os.path.join(root, "output", "fp_demo"),
        help="output directory for frames/ and turntable_fp.jsonl (default output/fp_demo)",
    )
    ap.add_argument(
        "--frames", type=int, default=24, help="frame count / full-turn steps (default 24)"
    )
    ap.add_argument(
        "--size", type=int, default=512, help="square background size in px (default 512)"
    )
    args = ap.parse_args()

    n = max(args.frames, 1)
    frames_dir = os.path.join(args.out_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)

    def r6(xs):
        return [round(x, 6) for x in xs]

    rows = []
    for i in range(n):
        rel = os.path.join("frames", f"frame_{i:06d}.png")
        write_png_rgb(
            os.path.join(args.out_dir, rel),
            args.size,
            args.size,
            background_rows(args.size, args.size, i / n),
        )
        rows.append(
            {"model": r6(rotate_y(2.0 * math.pi * i / n)), "frame_path": rel.replace("\\", "/")}
        )

    jsonl = os.path.join(args.out_dir, "turntable_fp.jsonl")
    with open(jsonl, "w", encoding="utf-8", newline="\n") as out:
        for r in rows:
            out.write(json.dumps(r) + "\n")

    print(f"wrote {n} background stills -> {frames_dir} and {jsonl}")


if __name__ == "__main__":
    main()
