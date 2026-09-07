#!/usr/bin/env python3
"""Convert an existing Arrow video-edit timeline into editable params.

The input and video stay unchanged. Camera/quad representation and present_index
are explicitly converted; other arrays, sparse rows and metadata are retained.
No FPS-derived PTS or decoded video images are generated.

uv run --with pyarrow python scripts/timeline_to_params.py INPUT -o OUTPUT \
  [--glb MODEL.glb] [--mesh-id UUID] [--identity-model]
"""

from __future__ import annotations

import argparse
import hashlib
import math
import os
from pathlib import Path
import struct
import tempfile
import uuid

import pyarrow as pa

from protocol_schema import PROTOCOL_VERSION

SOURCE_VERSION_KEY = b"trd.video_edit.version"
SOURCE_KIND_KEY = b"trd.video_edit.table.kind"


def read_source(path: Path) -> tuple[pa.Schema, list[pa.RecordBatch]]:
    with path.open("rb") as source:
        if source.read(4) != b"\xff" * 4:
            raise ValueError("input must be an Arrow IPC stream; Parquet is not supported")
        source.seek(0)
        reader = pa.ipc.open_stream(source)
        schema = reader.schema
        batches = list(reader)
        if source.read(1):
            raise ValueError("expected exactly one annotation stream")
    metadata = schema.metadata or {}
    if metadata.get(SOURCE_VERSION_KEY) != b"0.2.0":
        raise ValueError("source must explicitly declare video-edit version 0.2.0")
    if metadata.get(SOURCE_KIND_KEY) != b"timeline":
        raise ValueError("source must be a timeline table")
    if len(set(schema.names)) != len(schema):
        raise ValueError("source column names must be unique")
    for name, data_type in [
        ("k", pa.list_(pa.float32(), 9)),
        ("placement_quad", pa.list_(pa.float32(), 8)),
        ("video_frame_index", pa.uint32()),
        ("present_index", pa.uint32()),
    ]:
        if name not in schema.names or schema.field(name).type != data_type:
            raise ValueError(f"source {name} must have type {data_type}")
    for name in ["bottom_quads", "model", "mesh_id", "draw_model", "draw_mesh"]:
        if name in schema.names:
            raise ValueError(f"source already contains reserved destination field {name}")
    return schema, batches


def positive_metadata(metadata: dict[bytes, bytes], key: bytes) -> int:
    value = int(metadata[key])
    if value <= 0:
        raise ValueError(f"{key.decode()} must be positive")
    return value


def fhc_quad(values: list[float], width: int, height: int) -> list[dict[str, float]]:
    if len(values) != 8 or not all(math.isfinite(value) for value in values):
        raise ValueError("placement_quad must have eight finite pixel coordinates")
    points = list(zip(values[::2], values[1::2], strict=True))
    crosses = []
    for i in range(4):
        a, b, c = points[i], points[(i + 1) % 4], points[(i + 2) % 4]
        crosses.append((b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0]))
    if not all(value > 0 for value in crosses):
        raise ValueError("placement_quad must already be clockwise, convex and nondegenerate")
    # The reference normalizes the first edge: even a cyclic rotation changes
    # the placement's basis and units, so the source corner order is semantic.
    return [
        {"x": 2 * x - width, "y": 2 * y - height, "w": float(width), "h": float(height)}
        for x, y in points
    ]


def fhc_intrinsics(values: list[float], width: int, height: int) -> dict[str, float]:
    if len(values) != 9 or not all(math.isfinite(value) for value in values):
        raise ValueError("k must contain nine finite values")
    if values[0] <= 0 or values[4] <= 0 or any(abs(values[i]) > 1e-6 for i in [3, 6, 7]):
        raise ValueError("source k must be row-major pinhole intrinsics with positive focal lengths")
    if abs(values[8] - 1) > 1e-6:
        raise ValueError("source k must have homogeneous scale 1")
    return {
        "fx": 2 * values[0], "fy": 2 * values[4],
        "cx": 2 * values[2] - width, "cy": 2 * values[5] - height,
        "skew": 2 * values[1], "w": float(width), "h": float(height),
    }


def convert(
    input_path: Path, output_path: Path, glb_path: Path | None,
    mesh_id: str | None, identity_model: bool,
) -> tuple[int, int]:
    if input_path.resolve() == output_path.resolve():
        raise ValueError("output must not overwrite the original annotation")
    schema, batches = read_source(input_path)
    metadata = dict(schema.metadata or {})
    width = positive_metadata(metadata, b"trd.video.width")
    height = positive_metadata(metadata, b"trd.video.height")
    frame_count = positive_metadata(metadata, b"trd.video.frame_count")
    fps_num = positive_metadata(metadata, b"trd.video.fps_num")
    fps_den = positive_metadata(metadata, b"trd.video.fps_den")
    glb = glb_path.read_bytes() if glb_path else None
    asset_id = None
    if glb is not None:
        if len(glb) < 12 or glb[:4] != b"glTF" or struct.unpack_from("<II", glb, 4) != (2, len(glb)):
            raise ValueError("mesh input must be a complete binary GLB version 2")
        asset_id = uuid.UUID(mesh_id) if mesh_id else uuid.uuid5(
            uuid.NAMESPACE_URL, "trd:glb:sha256:" + hashlib.sha256(glb).hexdigest(),
        )
    elif mesh_id:
        raise ValueError("--mesh-id requires --glb")
    point = pa.struct([pa.field(name, pa.float32(), nullable=False) for name in ["x", "y", "w", "h"]])
    quads_type = pa.list_(pa.field("item", pa.list_(pa.field("item", point, nullable=False), 4), nullable=False))
    intrinsics_type = pa.struct([
        pa.field(name, pa.float32(), nullable=False) for name in ["fx", "fy", "cx", "cy", "skew", "w", "h"]
    ])
    matrix_type = pa.list_(pa.field("item", pa.list_(
        pa.field("item", pa.list_(pa.field("item", pa.float32(), nullable=False), 4), nullable=False), 4,
    ), nullable=False))
    fields = []
    for field in schema:
        if field.name == "k":
            fields.append(pa.field("k", intrinsics_type, nullable=False, metadata=field.metadata))
        elif field.name == "present_index":
            fields.append(pa.field("present_index", pa.int64(), nullable=False, metadata=field.metadata))
        else:
            fields.append(field)
    fields.append(pa.field("bottom_quads", quads_type, nullable=False))
    if identity_model:
        fields.append(pa.field("model", matrix_type, nullable=False))
    if asset_id:
        fields.append(pa.field("mesh_id", pa.list_(pa.field("item", pa.uuid(), nullable=False)), nullable=False))
    del metadata[SOURCE_VERSION_KEY]
    del metadata[SOURCE_KIND_KEY]
    metadata[b"trd.protocol.version"] = PROTOCOL_VERSION.encode()
    metadata[b"trd.table.kind"] = b"params"
    metadata[b"trd.stream.frame_rate"] = str(fps_num / fps_den).encode()
    metadata[b"trd.source.conversion"] = b"video_edit_0.2.0_to_fhc_params"
    output_schema = pa.schema(fields, metadata=metadata)
    result, previous, total = [], None, 0
    identity = [[float(row == column) for column in range(4)] for row in range(4)]
    for batch in batches:
        intrinsics, quads = [], []
        for row in range(batch.num_rows):
            index = batch.column("video_frame_index")[row].as_py()
            source_index = batch.column("present_index")[row].as_py()
            k = batch.column("k")[row].as_py()
            quad = batch.column("placement_quad")[row].as_py()
            if index is None or source_index is None or index != source_index:
                raise ValueError("this conversion requires the timeline's explicit 1:1 video/present index mapping")
            if index >= frame_count or (previous is not None and index <= previous):
                raise ValueError("source frame keys must be increasing and within the video's range")
            if k is None or quad is None:
                raise ValueError("conversion requires tracked geometry; no source rows are silently dropped")
            if "tracked" in schema.names and batch.column("tracked")[row].as_py() is not True:
                raise ValueError("conversion requires tracked rows; an untracked row is not a guessed placement")
            intrinsics.append(fhc_intrinsics(k, width, height))
            quads.append([fhc_quad(quad, width, height)])
            previous = index
        arrays = []
        for field, array in zip(schema, batch.columns, strict=True):
            if field.name == "k":
                arrays.append(pa.array(intrinsics, type=intrinsics_type))
            elif field.name == "present_index":
                arrays.append(array.cast(pa.int64()))
            else:
                arrays.append(array)
        arrays.append(pa.array(quads, type=quads_type))
        if identity_model:
            arrays.append(pa.array([[identity] for _ in range(batch.num_rows)], type=matrix_type))
        if asset_id:
            ids = pa.ExtensionArray.from_storage(
                pa.uuid(), pa.array([asset_id.bytes] * batch.num_rows, type=pa.binary(16)),
            )
            arrays.append(pa.ListArray.from_arrays(
                pa.array(range(batch.num_rows + 1), type=pa.int32()), ids,
                type=output_schema.field("mesh_id").type,
            ))
        result.append(pa.RecordBatch.from_arrays(arrays, schema=output_schema))
        total += batch.num_rows
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=output_path.parent, delete=False) as output:
            temporary = Path(output.name)
            with pa.ipc.new_stream(output, output_schema) as writer:
                for batch in result:
                    writer.write_batch(batch)
            if asset_id:
                mesh_schema = pa.schema([
                    pa.field("mesh_id", pa.uuid(), nullable=False),
                    pa.field("glb", pa.large_binary(), nullable=False),
                ], metadata={
                    b"trd.protocol.version": PROTOCOL_VERSION.encode(), b"trd.table.kind": b"mesh",
                })
                mesh_batch = pa.RecordBatch.from_arrays([
                    pa.ExtensionArray.from_storage(
                        pa.uuid(), pa.array([asset_id.bytes], type=pa.binary(16)),
                    ),
                    pa.array([glb], type=pa.large_binary()),
                ], schema=mesh_schema)
                with pa.ipc.new_stream(output, mesh_schema) as writer:
                    writer.write_batch(mesh_batch)
        os.replace(temporary, output_path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return total, len(result)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("-o", "--output", type=Path, required=True)
    parser.add_argument("--glb", type=Path)
    parser.add_argument("--mesh-id")
    parser.add_argument("--identity-model", action="store_true")
    args = parser.parse_args()
    rows, batches = convert(args.input, args.output, args.glb, args.mesh_id, args.identity_model)
    print(f"wrote {rows} sparse params rows in {batches} batch(es) to {args.output}")


if __name__ == "__main__":
    main()
