#!/usr/bin/env python3
"""Extract a video into still frames + a frame-to-Arrow-row **mapping manifest**.

This is the **offline / boundary** groundwork for the trd frame-compositing
pipeline (issue #62, prep slice #76). It turns a ``video.mp4`` into per-frame
still images and a small manifest so any producer can stamp a per-frame
``frame_path`` (native) / ``frame_url`` (browser) into the Arrow scene stream,
and the boundary can resolve it back to an image. **No ``trd-core`` code and no
codec are involved** — this is pure boundary tooling (``ffmpeg`` at the edge,
exactly like ``scripts/encode.py`` on egress).

Convention (issue #76)
----------------------
* Layout: ``<out>/frames/frame_000000.png`` … (zero-padded 6-digit index).
* **0-based frame index == Arrow row 0** — output order *is* the scene-stream
  order (#62 D2), so ``row N`` ↔ ``frame N`` is a pure function of the row index.
* Manifest ``<out>/frames.arrow`` (Arrow IPC) — one row per frame, columns:
    - ``row``        — ``uint32`` frame/row index (0-based).
    - ``frame_path`` — ``utf8`` native path, **relative to the manifest dir**
      (e.g. ``frames/frame_000000.png``).
    - ``frame_url``  — ``utf8`` browser URL, relative to a served base
      (e.g. ``frames/frame_000000.png`` under ``--url-base frames``).
  Schema metadata carries ``trd.stream.frame_rate`` (source fps, so display
  sinks play back at the right speed, #62 D2) plus ``trd.frames.width`` /
  ``trd.frames.height`` / ``trd.frames.count``.
* A sidecar ``<out>/frames.json`` mirrors the manifest for human inspection and
  for demos/tools that would rather not read Arrow.

Determinism
-----------
Re-extracting the same clip yields byte-identical frames + manifest: frames come
straight from the decoder with ``-fps_mode passthrough`` (no drop/dup, no rate
conversion), lossless PNG by default, and the manifest is a pure function of the
frame count + probed metadata.

Run
---
    uv run --with pyarrow scripts/extract_frames.py \\
        assets/videos/cornellbox/CameraMovement.mp4 -o output/cornellbox
    # -> output/cornellbox/frames/frame_000000.png … + frames.arrow + frames.json

Needs ``ffmpeg``/``ffprobe`` on PATH (run inside ``nix develop``). ``pyarrow`` is
only needed for the ``frames.arrow`` manifest; pass ``--no-arrow`` to emit just
the JSON sidecar without it.
"""
import argparse
import json
import os
import subprocess
import sys

FRAME_STEM = "frame_"
FRAME_DIGITS = 6
PROTOCOL_VERSION_KEY = b"trd.protocol.version"
PROTOCOL_VERSION = b"0.0.4"
FRAME_RATE_KEY = b"trd.stream.frame_rate"


def usage_guidance() -> str:
    """Flag guidance shown for a bare invocation (repo convention)."""
    return (
        "extract_frames.py — extract a video into still frames + an Arrow "
        "frame-to-row mapping manifest (#76).\n\n"
        "Usage:\n"
        "  scripts/extract_frames.py VIDEO [-o OUT_DIR] [--format png|jpg]\n"
        "                            [--url-base BASE] [--fps N] [--no-arrow]\n\n"
        "Arguments:\n"
        "  VIDEO           input video file (e.g. a .mp4).\n"
        "  -o, --out       output directory (default: output/<video-stem>).\n"
        "  --format        still-image format: png (lossless, default) or jpg.\n"
        "  --url-base      served-base prefix for frame_url (default: frames).\n"
        "  --fps           override the source fps recorded in the manifest\n"
        "                  metadata (does NOT resample; extraction is passthrough).\n"
        "  --no-arrow      skip the frames.arrow manifest (emit frames.json only).\n\n"
        "Emits:\n"
        "  <out>/frames/frame_000000.png …   zero-padded stills (row N == frame N)\n"
        "  <out>/frames.arrow                Arrow IPC manifest (row/path/url + w/h/fps)\n"
        "  <out>/frames.json                 human-readable sidecar\n\n"
        "Example:\n"
        "  scripts/extract_frames.py assets/videos/cornellbox/CameraMovement.mp4 \\\n"
        "    -o output/cornellbox\n\n"
        "Run inside `nix develop` (needs ffmpeg/ffprobe; pyarrow for the .arrow "
        "manifest)."
    )


def run(cmd):
    """Run a subprocess, raising a clean error with stderr on failure."""
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise SystemExit(
            f"error: command failed ({proc.returncode}): {' '.join(cmd)}\n{proc.stderr.strip()}"
        )
    return proc.stdout


def probe_video(video):
    """Return ``(width, height, fps, nb_frames)`` via ffprobe.

    ``fps`` comes from the stream's ``r_frame_rate`` rational (e.g. ``25/1``);
    ``nb_frames`` is best-effort (some containers omit it) — the authoritative
    count is the number of files ffmpeg actually writes.
    """
    out = run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=width,height,r_frame_rate,nb_frames",
            "-of", "json", video,
        ]
    )
    stream = json.loads(out)["streams"][0]
    width = int(stream["width"])
    height = int(stream["height"])
    num, den = stream["r_frame_rate"].split("/")
    fps = float(num) / float(den) if float(den) != 0 else float(num)
    nb_frames = int(stream["nb_frames"]) if stream.get("nb_frames", "N/A").isdigit() else None
    return width, height, fps, nb_frames


def extract(video, frames_dir, fmt):
    """Decode ``video`` into ``frames_dir/frame_%06d.<fmt>`` (0-based, passthrough).

    ``-fps_mode passthrough`` keeps exactly one output image per decoded frame
    (no drop/dup, no rate conversion) so ``row N == frame N`` holds; ``-start_number
    0`` makes the index 0-based to match Arrow row 0.
    """
    os.makedirs(frames_dir, exist_ok=True)
    pattern = os.path.join(frames_dir, f"{FRAME_STEM}%0{FRAME_DIGITS}d.{fmt}")
    cmd = [
        "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
        "-i", video, "-fps_mode", "passthrough", "-start_number", "0",
    ]
    if fmt == "png":
        cmd += ["-pix_fmt", "rgb24"]  # deterministic, lossless
    else:
        cmd += ["-q:v", "2"]
    cmd.append(pattern)
    run(cmd)


def frame_files(frames_dir, fmt):
    """Sorted list of extracted frame filenames (basenames), 0-based order."""
    names = [
        n for n in os.listdir(frames_dir)
        if n.startswith(FRAME_STEM) and n.endswith(f".{fmt}")
    ]
    return sorted(names)


def build_rows(names, url_base):
    """Build the row→(path, url) mapping. ``frame_path`` is relative to the
    manifest dir (``frames/<name>``); ``frame_url`` prepends ``url_base``."""
    rows = []
    for i, name in enumerate(names):
        frame_path = f"frames/{name}"
        frame_url = f"{url_base.rstrip('/')}/{name}" if url_base else f"frames/{name}"
        rows.append((i, frame_path, frame_url))
    return rows


def write_json_manifest(path, rows, width, height, fps):
    manifest = {
        "width": width,
        "height": height,
        "fps": fps,
        "count": len(rows),
        "frames": [
            {"row": r, "frame_path": p, "frame_url": u} for (r, p, u) in rows
        ],
    }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
        fh.write("\n")


def write_arrow_manifest(path, rows, width, height, fps):
    """Emit the Arrow IPC manifest (row/frame_path/frame_url + w/h/fps metadata)."""
    import pyarrow as pa
    from pyarrow import ipc

    row_arr = pa.array([r for (r, _, _) in rows], type=pa.uint32())
    path_arr = pa.array([p for (_, p, _) in rows], type=pa.utf8())
    url_arr = pa.array([u for (_, _, u) in rows], type=pa.utf8())
    schema = pa.schema(
        [
            pa.field("row", pa.uint32(), nullable=False),
            pa.field("frame_path", pa.utf8(), nullable=False),
            pa.field("frame_url", pa.utf8(), nullable=False),
        ],
        metadata={
            PROTOCOL_VERSION_KEY: PROTOCOL_VERSION,
            FRAME_RATE_KEY: str(fps).encode(),
            b"trd.frames.width": str(width).encode(),
            b"trd.frames.height": str(height).encode(),
            b"trd.frames.count": str(len(rows)).encode(),
        },
    )
    batch = pa.record_batch([row_arr, path_arr, url_arr], schema=schema)
    with open(path, "wb") as sink:
        with ipc.new_stream(sink, schema) as writer:
            writer.write_batch(batch)


def main() -> None:
    if len(sys.argv) == 1:
        print(usage_guidance())
        raise SystemExit(0)

    ap = argparse.ArgumentParser(
        description="Extract a video into still frames + an Arrow frame-to-row manifest (#76).",
        add_help=True,
    )
    ap.add_argument("video", help="input video file (e.g. a .mp4)")
    ap.add_argument("-o", "--out", default=None, help="output directory (default: output/<stem>)")
    ap.add_argument("--format", choices=["png", "jpg"], default="png", help="still format")
    ap.add_argument("--url-base", default="frames", help="served-base prefix for frame_url")
    ap.add_argument("--fps", type=float, default=None, help="override source fps in the manifest")
    ap.add_argument("--no-arrow", action="store_true", help="skip frames.arrow (JSON only)")
    args = ap.parse_args()

    if not os.path.isfile(args.video):
        raise SystemExit(f"error: video not found: {args.video}")

    stem = os.path.splitext(os.path.basename(args.video))[0]
    out_dir = args.out or os.path.join("output", stem)
    frames_dir = os.path.join(out_dir, "frames")
    os.makedirs(out_dir, exist_ok=True)

    width, height, probed_fps, nb_frames = probe_video(args.video)
    fps = args.fps if args.fps is not None else probed_fps

    print(f"probed: {width}x{height} @ {probed_fps:g} fps"
          + (f", {nb_frames} frames" if nb_frames else "") + f" -> {frames_dir}",
          file=sys.stderr)
    extract(args.video, frames_dir, args.format)

    names = frame_files(frames_dir, args.format)
    if not names:
        raise SystemExit("error: ffmpeg produced no frames")
    rows = build_rows(names, args.url_base)

    json_path = os.path.join(out_dir, "frames.json")
    write_json_manifest(json_path, rows, width, height, fps)
    emitted = [json_path]
    if not args.no_arrow:
        arrow_path = os.path.join(out_dir, "frames.arrow")
        write_arrow_manifest(arrow_path, rows, width, height, fps)
        emitted.append(arrow_path)

    print(f"extracted {len(rows)} frames ({args.format}, {width}x{height} @ {fps:g} fps)",
          file=sys.stderr)
    print("wrote " + " + ".join(emitted), file=sys.stderr)


if __name__ == "__main__":
    main()
