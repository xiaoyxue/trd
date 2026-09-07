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
import io
import json
import struct
import uuid
from pathlib import Path

import numpy as np
import pyarrow as pa
from PIL import Image

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


def png_bytes(image: Image.Image) -> bytes:
    output = io.BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


def glb_bytes(mesh: dict, texture: bytes | None) -> bytes:
    positions = np.asarray(mesh["position"], dtype="<f4")
    if positions.ndim != 2 or positions.shape[1] != 3 or not len(positions):
        raise ValueError("expected nonempty legacy positions")
    minimum, maximum = positions.min(axis=0), positions.max(axis=0)
    extent = float((maximum - minimum).max())
    scale = np.float32(2.0 / extent if extent > 1e-6 else 1.0)
    positions = (positions - (minimum + maximum) * np.float32(0.5)) * scale
    binary = bytearray()
    views, accessors = [], []

    def view(payload: bytes) -> int:
        while len(binary) % 4:
            binary.append(0)
        index = len(views)
        views.append({"buffer": 0, "byteOffset": len(binary), "byteLength": len(payload)})
        binary.extend(payload)
        return index

    def accessor(values, kind: str, component: int = 5126) -> int:
        values = np.asarray(values, dtype="<u4" if component == 5125 else "<f4")
        item = {"bufferView": view(values.tobytes()), "componentType": component,
                "count": len(values), "type": kind}
        if kind == "VEC3":
            item["min"] = values.min(axis=0).tolist()
            item["max"] = values.max(axis=0).tolist()
        accessors.append(item)
        return len(accessors) - 1

    attributes = {"POSITION": accessor(positions, "VEC3")}
    for name, attribute, kind in [("color", "COLOR_0", "VEC3"), ("uv", "TEXCOORD_0", "VEC2")]:
        if mesh.get(name) is not None:
            attributes[attribute] = accessor(mesh[name], kind)
    indices = mesh.get("index")
    if indices is None:
        indices = np.arange(len(positions), dtype=np.uint32)
    primitive = {"attributes": attributes, "indices": accessor(indices, "SCALAR", 5125),
                 "material": 0}
    pbr = {"baseColorFactor": [1.0, 1.0, 1.0, 1.0],
           "metallicFactor": 0.0, "roughnessFactor": 0.5}
    document = {
        "asset": {"version": "2.0"},
        "bufferViews": views, "accessors": accessors,
        "meshes": [{"primitives": [primitive]}],
        "nodes": [{"mesh": 0}], "scenes": [{"nodes": [0]}], "scene": 0,
        "materials": [{"pbrMetallicRoughness": pbr}],
    }
    if texture is not None:
        document["images"] = [{"bufferView": view(texture), "mimeType": "image/png"}]
        document["textures"] = [{"source": 0}]
        pbr["baseColorTexture"] = {"index": 0}
    document["buffers"] = [{"byteLength": len(binary)}]
    while len(binary) % 4:
        binary.append(0)
    encoded = json.dumps(document, separators=(",", ":")).encode()
    encoded += b" " * (-len(encoded) % 4)
    size = 12 + 8 + len(encoded) + 8 + len(binary)
    glb = (b"glTF" + struct.pack("<II", 2, size)
           + struct.pack("<I4s", len(encoded), b"JSON") + encoded
           + struct.pack("<I4s", len(binary), b"BIN\0") + binary)
    return bytes(glb)


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
