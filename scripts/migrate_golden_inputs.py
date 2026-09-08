#!/usr/bin/env python3
"""Explicitly migrate the two legacy golden inputs to params + GLB resources.

This is offline fixture tooling, not a legacy-format fallback in an application.
Camera/draw arrays and expected PNGs are untouched. Old implicit mesh preview
transforms are baked into the fixture GLB geometry so gizmos stay equivalent.

Run with the repository's existing pyarrow/numpy/Pillow environment:
  uv run --with pyarrow --with numpy --with pillow scripts/migrate_golden_inputs.py \
    --source-dir <old-golden-directory>
"""

from __future__ import annotations

import argparse
import json
import uuid
from pathlib import Path

import numpy as np
import pyarrow as pa
from PIL import Image

from glb_assets import glb_bytes, png_bytes
from protocol_schema import PROTOCOL_VERSION

ROOT = Path(__file__).resolve().parent.parent
GOLDEN = ROOT / "crates" / "trd-core" / "tests" / "golden"


def tables(data: bytes) -> dict[str, list[pa.RecordBatch]]:
    source = pa.BufferReader(data)
    result = {}
    while source.tell() < len(data):
        reader = pa.ipc.open_stream(source)
        metadata = reader.schema.metadata or {}
        kind = metadata.get(b"trd.table.kind", b"").decode()
        if kind in result or kind not in {"mesh", "texture", "frames", "params"}:
            raise ValueError(f"unsupported/duplicate legacy table {kind!r}")
        if metadata.get(b"trd.protocol.version") != b"0.0.6":
            raise ValueError("migration input must explicitly be protocol 0.0.6")
        result[kind] = list(reader)
    if not result.get("mesh") or not result.get("params"):
        raise ValueError("legacy fixture needs mesh and params tables")
    return result


def rgba_image(batch: pa.RecordBatch, column: str, row: int) -> Image.Image:
    array = batch.column(batch.schema.get_field_index(column))
    if isinstance(array, pa.ExtensionArray):
        shape = array.type.shape
        values = array.storage[row].as_py()
    else:
        metadata = batch.schema.field(column).metadata or {}
        shape = json.loads(metadata[b"ARROW:extension:metadata"])["shape"]
        values = array[row].as_py()
    pixels = np.asarray(values, dtype=np.uint8).reshape(shape)
    return Image.fromarray(pixels)


def migrate(source: Path, destination: Path) -> None:
    old = tables(source.read_bytes())
    textures = {}
    for batch in old.get("texture", []):
        if "rgba" in batch.schema.names:
            textures[0] = png_bytes(rgba_image(batch, "rgba", 0))
        else:
            raise ValueError("unexpected texture layout in golden fixture")
    resources = []
    for batch in old["mesh"]:
        for mesh in batch.to_pylist():
            if mesh.get("gltf_path") or mesh.get("gltf_url") or mesh.get("material"):
                raise ValueError("this migration is restricted to the original golden meshes")
            glb = glb_bytes(mesh, textures.get(len(resources)))
            resources.append(glb)
    paths = []
    for batch in old.get("frames", []):
        for row in range(batch.num_rows):
            payload = batch.column("frame_bytes")[row].as_py() if "frame_bytes" in batch.schema.names else None
            if payload is not None:
                suffix = ".jpg" if payload.startswith(b"\xff\xd8") else ".png"
            else:
                payload = png_bytes(rgba_image(batch, "frame_pixels", row))
                suffix = ".png"
            relative = Path("frames") / f"{destination.stem}-resource-{len(paths)}{suffix}"
            target = destination.parent / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(payload)
            paths.append(relative.as_posix())

    result = []
    for batch in old["params"]:
        fields, arrays = [], []
        if "draw_mesh" not in batch.schema.names or "draw_model" not in batch.schema.names:
            raise ValueError("golden params must have explicit draw lists")
        for field, array in zip(batch.schema, batch.columns, strict=True):
            if field.name == "frame_id":
                fields.append(pa.field("frame_path", pa.string(), nullable=field.nullable))
                arrays.append(pa.array([None if index is None else paths[index]
                                        for index in array.to_pylist()], type=pa.string()))
            else:
                fields.append(field)
                arrays.append(array)
        metadata = dict(batch.schema.metadata or {})
        metadata[b"trd.protocol.version"] = PROTOCOL_VERSION.encode()
        schema = pa.schema(fields, metadata=metadata)
        result.append(pa.RecordBatch.from_arrays(arrays, schema=schema))
    mesh_schema = pa.schema([
        pa.field("mesh_id", pa.uuid(), nullable=False),
        pa.field("glb", pa.large_binary(), nullable=False),
    ], metadata={b"trd.protocol.version": PROTOCOL_VERSION.encode(), b"trd.table.kind": b"mesh"})
    ids = [uuid.uuid5(uuid.NAMESPACE_URL, f"trd:golden:{destination.stem}:{i}").bytes
           for i in range(len(resources))]
    meshes = pa.RecordBatch.from_arrays(
        [pa.array(ids, type=pa.uuid()), pa.array(resources, type=pa.large_binary())],
        schema=mesh_schema,
    )
    output = pa.BufferOutputStream()
    with pa.ipc.new_stream(output, result[0].schema) as writer:
        for batch in result:
            writer.write_batch(batch)
    with pa.ipc.new_stream(output, mesh_schema) as writer:
        writer.write_batch(meshes)
    destination.write_bytes(output.getvalue().to_pybytes())
    print(f"migrated {source.name}: camera/draw arrays unchanged; {len(resources)} GLBs, {len(paths)} backgrounds")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    args = parser.parse_args()
    for name in ["stage1.arrow", "stage2.arrow"]:
        migrate(args.source_dir / name, GOLDEN / name)


if __name__ == "__main__":
    main()
