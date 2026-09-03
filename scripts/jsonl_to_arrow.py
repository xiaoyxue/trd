#!/usr/bin/env python3
"""Convert a trd JSONL frame-params file to a protocol 0.0.6 Arrow IPC stream.

A dependency-light pyarrow producer for examples/render.sh and render.ps1. The
stream protocol is **0.0.6-only** (mesh-first; older wire formats are
retired and no longer produced or accepted). Each JSON line is one frame.

Emitted params columns (all optional except `model`, which is always emitted):

  * `video_frame_index` (`UInt32`, all-or-nothing): sidecar-video frame key for
    sparse scene rows. Values must be strictly increasing.
  * `model` (`FixedSizeList<f32>[16]`, column-major, matching glam
    `Mat4::from_cols_array`): the per-frame object transform. A row's explicit
    `"model"` is used verbatim; a row without one gets the identity.
  * **Camera** (per-frame, all-or-nothing) in either the CV form (`"k"`: 9 floats,
    `"pose"`: 16 floats, camera-to-world) or the CG form (`"eye"`/`"target"`/
    `"direction"`/`"up"`: 3 floats each; `"fovy"`/`"aspect"`/`"znear"`/`"zfar"`:
    scalars). Each camera column is emitted only when *every* row provides it,
    matching trd-core's non-null column requirement. An omitted camera column
    decodes to the identity view/projection.
  * **Multi-mesh draw list** (per-frame, all-or-nothing):
    `"draws": [{"mesh": i, "model": [16 floats], "mode": "wireframe"?}, ...]` places
    several instances of the stream's meshes. Emitted as `draw_mesh`
    (`List<UInt32>`) + `draw_model` (`List<FixedSizeList<f32>[16]>`) only when
    *every* row provides `"draws"`. When absent, one instance of mesh 0 is placed
    by each frame's own `model`. Each draw may carry an optional `"mode"`
    (`"filled"`/`"wireframe"`/`"textured"`/`"shadow"`) render-mode override; when
    *any* draw names one, the per-draw `draw_mode` (`List<UInt8>`) column is emitted
    (`0`=filled, `1`=wireframe, `2`=textured, `3`=shadow grounding blob,
    `255`=inherit the front-end's global mode). This lets one frame mix e.g. a
    textured mesh with a wireframe overlay and a grounding shadow.
  * **Background frame reference** (per-frame): `frame_path` (native filesystem
    path) and/or `frame_url` (browser URL) name the still image the shell loads and
    composites *beneath* the scene via a `FramePlane`. Each is emitted when *any*
    row provides it; a row without one renders with no background (null).
  * **Inline background reference** (per-frame): `frame_id` (`UInt32`) indexes
    a row in the optional preceding frames resource table. It is mutually
    exclusive with `frame_path` / `frame_url` on the same row.

Run via:
  uv run --with pyarrow scripts/jsonl_to_arrow.py examples/frames.turntable.jsonl  # -> stdout
  python scripts/jsonl_to_arrow.py frames.jsonl -o frames.arrows                   # -> file
"""
import argparse
import json
import sys

import pyarrow as pa
from pyarrow import ipc

PROTOCOL_VERSION = "0.0.6"
PROTOCOL_VERSION_KEY = b"trd.protocol.version"
TABLE_KIND_KEY = b"trd.table.kind"
FRAME_RATE_KEY = b"trd.stream.frame_rate"

# Optional camera columns: (json key, fixed-size-list length).
CAMERA_VEC = [("eye", 3), ("target", 3), ("direction", 3), ("up", 3), ("k", 9), ("pose", 16)]
CAMERA_SCALAR = ["fovy", "aspect", "znear", "zfar"]

# Column-major identity 4x4 (glam Mat4::IDENTITY) for rows without an explicit model.
IDENTITY_MODEL = [
    1.0, 0.0, 0.0, 0.0,
    0.0, 1.0, 0.0, 0.0,
    0.0, 0.0, 1.0, 0.0,
    0.0, 0.0, 0.0, 1.0,
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("input", help="JSONL frame-params file")
    ap.add_argument("-o", "--output", default="-", help="output path ('-' = stdout)")
    ap.add_argument(
        "--fps",
        type=float,
        default=None,
        help="declare trd.stream.frame_rate metadata (playback fps)",
    )
    args = ap.parse_args()

    with open(args.input, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]

    f32 = pa.float32()
    fsl16 = pa.list_(f32, 16)  # FixedSizeList<f32>[16] = column-major Mat4

    metadata = {
        PROTOCOL_VERSION_KEY: PROTOCOL_VERSION.encode(),
        TABLE_KIND_KEY: b"params",
    }
    if args.fps and args.fps > 0:
        metadata[FRAME_RATE_KEY] = str(args.fps).encode()

    # `model` is the one always-present column: a row's explicit matrix, else identity.
    model_rows = [r.get("model", IDENTITY_MODEL) for r in rows]
    columns = [pa.array(model_rows, type=fsl16)]
    fields = [("model", fsl16)]

    if all("video_frame_index" in row for row in rows):
        indices = [int(row["video_frame_index"]) for row in rows]
        if any(current <= previous for previous, current in zip(indices, indices[1:])):
            raise SystemExit("error: video_frame_index must be strictly increasing")
        columns.insert(0, pa.array(indices, type=pa.uint32()))
        fields.insert(0, ("video_frame_index", pa.uint32()))

    # Camera columns are all-or-nothing (every row must provide them).
    for name, length in CAMERA_VEC:
        if all(name in r for r in rows):
            typ = pa.list_(f32, length)
            columns.append(pa.array([r[name] for r in rows], type=typ))
            fields.append((name, typ))
    for name in CAMERA_SCALAR:
        if all(name in r for r in rows):
            columns.append(pa.array([r[name] for r in rows], type=f32))
            fields.append((name, f32))

    # Per-frame instanced draw list (all-or-nothing).
    if all("draws" in r for r in rows):
        mesh_ids_type = pa.list_(pa.uint32())
        models_type = pa.list_(fsl16)
        mesh_ids = [[int(d["mesh"]) for d in r["draws"]] for r in rows]
        models = [[d["model"] for d in r["draws"]] for r in rows]
        columns.append(pa.array(mesh_ids, type=mesh_ids_type))
        fields.append(("draw_mesh", mesh_ids_type))
        columns.append(pa.array(models, type=models_type))
        fields.append(("draw_model", models_type))

        # Optional per-draw render-mode override. Emitted (as `draw_mode`,
        # List<UInt8>) when *any* draw names a "mode"; draws without one get
        # 255 = "inherit the front-end's global mode". So a stream can flip
        # just an overlay quad to wireframe while every other draw follows
        # the renderer's `--wireframe`/`--textured`/default mode.
        mode_wire = {"filled": 0, "wireframe": 1, "textured": 2, "shadow": 3, "inherit": 255}
        if any("mode" in d for r in rows for d in r["draws"]):
            def to_mode_byte(d):
                m = d.get("mode", "inherit")
                if isinstance(m, int):
                    return m
                if m not in mode_wire:
                    raise SystemExit(
                        f"error: draw mode {m!r} must be one of "
                        f"{sorted(mode_wire)} or an int 0/1/2/3/255"
                    )
                return mode_wire[m]

            modes_type = pa.list_(pa.uint8())
            modes = [[to_mode_byte(d) for d in r["draws"]] for r in rows]
            columns.append(pa.array(modes, type=modes_type))
            fields.append(("draw_mode", modes_type))

    # Per-frame background frame reference: the still image the shell loads and
    # composites beneath the scene via a FramePlane. Emitted when *any* row names
    # one; a row without it decodes to "no background" (null). `frame_path` (native
    # filesystem path) and `frame_url` (browser URL) are independent.
    utf8 = pa.utf8()
    for name in ("frame_path", "frame_url"):
        if any(name in r for r in rows):
            columns.append(pa.array([r.get(name) for r in rows], type=utf8))
            fields.append((name, utf8))

    if any("frame_id" in r for r in rows):
        conflicts = [
            i
            for i, row in enumerate(rows)
            if row.get("frame_id") is not None
            and (row.get("frame_path") or row.get("frame_url"))
        ]
        if conflicts:
            raise SystemExit(
                "error: frame_id is mutually exclusive with frame_path/frame_url "
                f"(conflict at row {conflicts[0]})"
            )
        columns.append(
            pa.array([r.get("frame_id") for r in rows], type=pa.uint32())
        )
        fields.append(("frame_id", pa.uint32()))

    schema = pa.schema(fields, metadata=metadata)
    batch = pa.record_batch(columns, schema=schema)

    sink = sys.stdout.buffer if args.output == "-" else open(args.output, "wb")
    try:
        with ipc.new_stream(sink, schema) as writer:
            writer.write_batch(batch)
    finally:
        if sink is not sys.stdout.buffer:
            sink.close()


if __name__ == "__main__":
    main()
