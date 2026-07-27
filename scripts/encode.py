#!/usr/bin/env python3
"""Encode a trd protocol 0.0.1/0.0.2 output Arrow IPC stream (stdin) to GIF/WebP/MP4.

Decodes the r,g,b,a fixed_shape_tensor channels and streams raw RGBA frames to
ffmpeg (no intermediate files). The playback rate defaults to the stream's
`trd.stream.frame_rate` schema metadata (or 30 if absent), so the output plays at
the same speed as the live front-ends; pass --fps to override. Run via:
  uv run --with pyarrow --with numpy scripts/encode.py -o output/out.gif

The output format is chosen from the `-o` extension: `.webp` → animated WebP,
`.mp4`/`.m4v`/`.mov` → H.264 (the practical choice at 1080p/4K, where GIF balloons
to hundreds of MB), anything else → GIF.
"""
import argparse
import subprocess
import sys

import numpy as np
import pyarrow as pa
from pyarrow import ipc

FRAME_RATE_KEY = b"trd.stream.frame_rate"
DEFAULT_FRAME_RATE = 30.0


def stream_frame_rate(schema) -> float:
    """The stream's declared playback rate, or the default when absent/invalid."""
    meta = schema.metadata or {}
    raw = meta.get(FRAME_RATE_KEY)
    if raw is None:
        return DEFAULT_FRAME_RATE
    try:
        rate = float(raw)
    except ValueError:
        return DEFAULT_FRAME_RATE
    return rate if rate > 0 and np.isfinite(rate) else DEFAULT_FRAME_RATE


def ffmpeg_cmd(fmt: str, width: int, height: int, fps: float, out: str) -> list[str]:
    base = [
        "ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "rgba",
        "-s", f"{width}x{height}", "-r", str(fps), "-i", "pipe:0",
    ]
    if fmt == "gif":
        vf = "split[a][b];[a]palettegen=stats_mode=full[p];[b][p]paletteuse=dither=bayer"
        return base + ["-vf", vf, out]
    if fmt == "mp4":
        # H.264: yuv420p + even dims for broad player compatibility; the practical
        # codec at 1080p/4K where an animated GIF would be hundreds of MB.
        return base + [
            "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
            "-movflags", "+faststart", out,
        ]
    # animated webp
    return base + ["-c:v", "libwebp_anim", "-loop", "0", "-pix_fmt", "yuva420p", out]


def output_format(path: str) -> str:
    """Pick the encoder from the output extension."""
    low = path.lower()
    if low.endswith(".webp"):
        return "webp"
    if low.endswith((".mp4", ".m4v", ".mov")):
        return "mp4"
    return "gif"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--output", required=True)
    ap.add_argument(
        "--fps",
        type=float,
        default=None,
        help="playback rate override; default = stream's trd.stream.frame_rate or 30",
    )
    args = ap.parse_args()
    fmt = output_format(args.output)

    reader = ipc.open_stream(sys.stdin.buffer)
    fps = args.fps if args.fps and args.fps > 0 else stream_frame_rate(reader.schema)
    proc = None
    width = height = None
    for batch in reader:
        chans = []
        for name in ("r", "g", "b", "a"):
            arr = batch.column(name)  # FixedShapeTensorArray
            chans.append(arr.to_numpy_ndarray())  # (rows, H, W)
        stacked = np.stack(chans, axis=-1)  # (rows, H, W, 4)
        if proc is None:
            _, height, width, _ = stacked.shape
            proc = subprocess.Popen(
                ffmpeg_cmd(fmt, width, height, fps, args.output),
                stdin=subprocess.PIPE,
            )
        proc.stdin.write(np.ascontiguousarray(stacked, dtype=np.uint8).tobytes())
    if proc is None:
        print("no frames in stream", file=sys.stderr)
        sys.exit(1)
    proc.stdin.close()
    if proc.wait() != 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
