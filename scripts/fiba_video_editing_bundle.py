#!/usr/bin/env python3
"""Build the FIBA video-editing Arrow timeline (document ``0.2.0``).

The document is **sparse**: it carries only the frames that have an ad-placement
quad, since those are the frames the editor can do anything with. Everything else
is ordinary video, and the editor plays it as such (#264). The MP4 stays
external; the first emitted row embeds an encoded JPEG poster.
"""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import pyarrow as pa
import pyarrow.parquet as pq
from pyarrow import ipc

VERSION = "0.2.0"


def command_json(*args):
    result = subprocess.run(args, check=True, capture_output=True, text=True)
    return json.loads(result.stdout)


def probe_video(path):
    data = command_json(
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=codec_name,width,height,r_frame_rate,avg_frame_rate,nb_frames,duration",
        "-of",
        "json",
        str(path),
    )
    stream = data["streams"][0]
    rate = stream.get("avg_frame_rate") or stream["r_frame_rate"]
    fps_num, fps_den = (int(value) for value in rate.split("/"))
    return {
        "codec": stream["codec_name"],
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "fps_num": fps_num,
        "fps_den": fps_den,
        "frame_count": int(stream["nb_frames"]),
        "duration_us": round(float(stream["duration"]) * 1_000_000),
    }


def first_frame_jpeg(path):
    return subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(path),
            "-vf",
            "select=eq(n\\,0)",
            "-frames:v",
            "1",
            "-f",
            "image2pipe",
            "-vcodec",
            "mjpeg",
            "-q:v",
            "2",
            "-",
        ],
        check=True,
        capture_output=True,
    ).stdout


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument(
        "--calibration",
        default="assets/videos/fiba/per_frame_KVP_cube_best.parquet",
    )
    parser.add_argument("--method", default="2VP_4510")
    parser.add_argument("-o", "--output", required=True)
    args = parser.parse_args()

    video = Path(args.video)
    info = probe_video(video)
    digest = hashlib.sha256(video.read_bytes()).hexdigest()
    poster = first_frame_jpeg(video)

    rows = [
        row
        for row in pq.read_table(args.calibration).to_pylist()
        if row["shot"] == 1 and row["method"] == args.method
    ]
    rows.sort(key=lambda row: int(row["present_index"]))
    if not rows:
        raise SystemExit(f"no calibration rows for method {args.method}")
    if len(rows) != info["frame_count"]:
        raise SystemExit(
            f"calibration has {len(rows)} rows, video has {info['frame_count']} frames"
        )
    present_indices = [int(row["present_index"]) for row in rows]
    expected = list(range(len(rows)))
    if present_indices != expected:
        raise SystemExit("simple FIBA bundle requires present_index contiguous from zero")

    video_frame_indices = expected
    # Sparse by construction: a row exists only where the calibration has both a
    # K and an ad quad. The frames in between are plain video, and saying so by
    # *omission* is the point of `0.2.0` (#264).
    annotated = [
        index
        for index, row in zip(video_frame_indices, rows)
        if row.get("K") is not None and row.get("ad_quad") is not None
    ]
    if not annotated:
        raise SystemExit("no annotated frames: every row lacks K or ad_quad")
    annotated_rows = [
        row
        for row in rows
        if row.get("K") is not None and row.get("ad_quad") is not None
    ]
    timestamps_us = [
        round(index * info["fps_den"] * 1_000_000 / info["fps_num"]) for index in annotated
    ]
    tracked = [True] * len(annotated)
    ks = [[float(value) for value in row["K"]] for row in annotated_rows]
    quads = [[float(value) for value in row["ad_quad"]] for row in annotated_rows]
    posters = [poster] + [None] * (len(annotated) - 1)

    metadata = {
        b"trd.video_edit.version": VERSION.encode(),
        b"trd.video_edit.table.kind": b"timeline",
        b"trd.video.source_name": video.name.encode(),
        b"trd.video.mime": b"video/mp4",
        b"trd.video.codec": info["codec"].encode(),
        b"trd.video.sha256": digest.encode(),
        b"trd.video.byte_length": str(video.stat().st_size).encode(),
        b"trd.video.width": str(info["width"]).encode(),
        b"trd.video.height": str(info["height"]).encode(),
        b"trd.video.fps_num": str(info["fps_num"]).encode(),
        b"trd.video.fps_den": str(info["fps_den"]).encode(),
        b"trd.video.frame_count": str(info["frame_count"]).encode(),
        b"trd.video.duration_us": str(info["duration_us"]).encode(),
    }
    schema = pa.schema(
        [
            ("video_frame_index", pa.uint32()),
            ("present_index", pa.uint32()),
            ("timestamp_us", pa.int64()),
            ("k", pa.list_(pa.float32(), 9)),
            ("placement_quad", pa.list_(pa.float32(), 8)),
            ("tracked", pa.bool_()),
            ("poster_bytes", pa.binary()),
        ],
        metadata=metadata,
    )
    batch = pa.record_batch(
        [
            pa.array(annotated, type=pa.uint32()),
            pa.array(annotated, type=pa.uint32()),
            pa.array(timestamps_us, type=pa.int64()),
            pa.array(ks, type=pa.list_(pa.float32(), 9)),
            pa.array(quads, type=pa.list_(pa.float32(), 8)),
            pa.array(tracked, type=pa.bool_()),
            pa.array(posters, type=pa.binary()),
        ],
        schema=schema,
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as file, ipc.new_stream(file, schema) as writer:
        writer.write_batch(batch)
    shots = []
    for index in annotated:
        if shots and index == shots[-1][1] + 1:
            shots[-1][1] = index
        else:
            shots.append([index, index])
    print(
        f"wrote {len(annotated)} annotated rows of {info['frame_count']} frames, "
        f"poster={len(poster)} bytes, shots="
        + ", ".join(f"{start}-{end}" for start, end in shots)
        + f" to {output}"
    )


if __name__ == "__main__":
    main()
