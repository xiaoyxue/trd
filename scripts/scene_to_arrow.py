#!/usr/bin/env python3
"""Bundle current params and offline-converted OBJ assets as [params][mesh].

GLB resources contain geometry, material and textures. No legacy Arrow mesh,
texture or inline-frame stream is emitted. External frame references stay in
the original params batches. Run with pyarrow, numpy and Pillow.
"""

import argparse
import hashlib
import sys
import uuid
from pathlib import Path

import pyarrow as pa
from PIL import Image

from glb_assets import glb_bytes, png_bytes
from obj_to_arrow import parse_obj
from protocol_schema import PROTOCOL_VERSION
from texture_to_arrow import image_to_rgba


def bundle(params: bytes, paths: list[Path], texture_path: Path | None) -> bytes:
    source = pa.BufferReader(params)
    reader = pa.ipc.open_stream(source)
    schema, batches = reader.schema, list(reader)
    metadata = schema.metadata or {}
    if source.tell() != len(params):
        raise ValueError("expected one params stream")
    if metadata.get(b"trd.table.kind") != b"params" or metadata.get(b"trd.protocol.version") != PROTOCOL_VERSION.encode():
        raise ValueError("params must declare the current protocol version and table kind")
    if "frame_id" in schema.names:
        raise ValueError("inline frame_id is retired; use frame_path/frame_url and --frames-base")
    if texture_path and not paths:
        raise ValueError("a texture requires an OBJ mesh")
    texture = None
    if texture_path:
        height, width, flat = image_to_rgba(str(texture_path), 2048)
        texture = png_bytes(Image.fromarray(flat.reshape(height, width, 4)))
    resources, ids = [], []
    for index, path in enumerate(paths):
        if path.suffix.lower() != ".obj":
            raise ValueError(f"offline conversion expects an OBJ file: {path}")
        positions, colors, uvs, indices = parse_obj(path.read_text(encoding="utf-8"))
        mesh = {"position": positions, "index": indices}
        if colors:
            mesh["color"] = colors
        if uvs:
            mesh["uv"] = uvs
        glb = glb_bytes(mesh, texture if index == 0 else None)
        resources.append(glb)
        ids.append(uuid.uuid5(uuid.NAMESPACE_URL, f"trd:obj:{index}:{hashlib.sha256(glb).hexdigest()}").bytes)
    output = pa.BufferOutputStream()
    with pa.ipc.new_stream(output, schema) as writer:
        for batch in batches:
            writer.write_batch(batch)
    if resources:
        mesh_schema = pa.schema(
            [pa.field("mesh_id", pa.uuid(), nullable=False), pa.field("glb", pa.large_binary(), nullable=False)],
            metadata={b"trd.protocol.version": PROTOCOL_VERSION.encode(), b"trd.table.kind": b"mesh"},
        )
        batch = pa.RecordBatch.from_arrays(
            [pa.array(ids, type=pa.uuid()), pa.array(resources, type=pa.large_binary())],
            schema=mesh_schema,
        )
        with pa.ipc.new_stream(output, mesh_schema) as writer:
            writer.write_batch(batch)
    return output.getvalue().to_pybytes()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--params", default="-", help="params Arrow stream; '-' reads stdin")
    parser.add_argument("--mesh", action="append", default=[], type=Path, help="offline OBJ input")
    parser.add_argument("--texture", type=Path, help="albedo for the first OBJ")
    parser.add_argument("-o", "--output", default="-")
    args = parser.parse_args()
    params = sys.stdin.buffer.read() if args.params == "-" else Path(args.params).read_bytes()
    data = bundle(params, args.mesh, args.texture)
    if args.output == "-":
        sys.stdout.buffer.write(data)
    else:
        Path(args.output).write_bytes(data)


if __name__ == "__main__":
    main()
