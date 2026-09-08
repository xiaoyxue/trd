"""Self-contained GLB assets for offline OBJ conversion and fixture migration."""

import io
import json
import struct

import numpy as np
from PIL import Image


def png_bytes(image: Image.Image) -> bytes:
    output = io.BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


def glb_bytes(mesh: dict, texture: bytes | None) -> bytes:
    positions = np.asarray(mesh["position"], dtype="<f4")
    if positions.ndim != 2 or positions.shape[1] != 3 or not len(positions) or not np.isfinite(positions).all():
        raise ValueError("expected nonempty finite positions")
    minimum, maximum = positions.min(axis=0), positions.max(axis=0)
    extent = float((maximum - minimum).max())
    scale = np.float32(2.0 / extent if extent > 1e-6 else 1.0)
    # OBJ demos historically used the renderer's extent-2 preview base. Baking
    # it into the converted asset leaves source camera/model arrays unchanged.
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
