#!/usr/bin/env python3
"""Convert images to a keyed trd **texture** Arrow IPC stream (0.0.6).

Each row binds one base-color image to a dense mesh-table ``mesh_id`` and carries
``width``, ``height``, and variable-length tightly packed ``rgba_bytes``. Rows
may therefore use different dimensions. The Rust decoder still accepts the
legacy one-row fixed-shape ``rgba`` table as an implicit texture for mesh 0.
Keyed output includes a row-0 legacy ``rgba`` compatibility alias.

Run via:
  uv run --with pyarrow --with pillow --with numpy \\
      scripts/texture_to_arrow.py albedo.png -o albedo.tex.arrows
  uv run --with pyarrow --with pillow --with numpy \\
      scripts/texture_to_arrow.py photo.jpg --max-size 2048           # -> stdout
  uv run --with pyarrow --with pillow --with numpy \\
      scripts/texture_to_arrow.py mesh0.png mesh1.png --mesh-id 0 --mesh-id 1
"""
import argparse
import sys

import numpy as np
import pyarrow as pa
from pyarrow import ipc
from PIL import Image

PROTOCOL_VERSION_KEY = b"trd.protocol.version"
PROTOCOL_VERSION = b"0.0.6"
TABLE_KIND_KEY = b"trd.table.kind"

MESH_ID_COLUMN = "mesh_id"
WIDTH_COLUMN = "width"
HEIGHT_COLUMN = "height"
RGBA_BYTES_COLUMN = "rgba_bytes"
LEGACY_RGBA_COLUMN = "rgba"


def image_to_rgba(path, max_size=None):
    """Load ``path`` as RGBA and return ``(height, width, flat)``.

    ``flat`` is a 1-D ``uint8`` array of ``height * width * 4`` interleaved RGBA
    bytes, row-major with a **top-left origin** (PIL's default). ``max_size``,
    when set, downscales the image (preserving aspect ratio, ``Image.thumbnail``)
    so ``max(height, width) <= max_size`` — GPUs cap texture dimensions, so large
    source art must be shrunk before upload.
    """
    with Image.open(path) as image:
        image = image.convert("RGBA")
        if max_size is not None and max(image.size) > max_size:
            # Scales in place, preserving aspect ratio (longest side -> max_size).
            image.thumbnail((max_size, max_size), Image.LANCZOS)
        width, height = image.size
        rgba = np.asarray(image, dtype=np.uint8)  # (H, W, 4), row-major top-left
    return height, width, np.ascontiguousarray(rgba).reshape(-1)


def texture_batch(rows):
    """Build a keyed multi-model texture ``RecordBatch``."""
    rows = sorted(rows, key=lambda row: row[0])
    for mesh_id, height, width, flat in rows:
        expected = height * width * 4
        if flat.size != expected:
            raise ValueError(
                f"mesh {mesh_id}: expected {expected} RGBA bytes, got {flat.size}"
            )
    compatibility = next(
        ((height, width, flat) for mesh_id, height, width, flat in rows if mesh_id == 0),
        (1, 1, np.array([255, 255, 255, 255], dtype=np.uint8)),
    )
    compatibility_height, compatibility_width, compatibility_flat = compatibility
    compatibility_size = compatibility_height * compatibility_width * 4
    tensor_type = pa.fixed_shape_tensor(
        pa.uint8(),
        [compatibility_height, compatibility_width, 4],
        dim_names=["height", "width", "channel"],
    )
    legacy_storage = pa.FixedSizeListArray.from_arrays(
        pa.array(compatibility_flat, type=pa.uint8()),
        compatibility_size,
    )
    legacy_array = pa.ExtensionArray.from_storage(tensor_type, legacy_storage)
    schema = pa.schema(
        [
            pa.field(LEGACY_RGBA_COLUMN, tensor_type, nullable=False),
            pa.field(MESH_ID_COLUMN, pa.list_(pa.uint32()), nullable=False),
            pa.field(WIDTH_COLUMN, pa.list_(pa.uint32()), nullable=False),
            pa.field(HEIGHT_COLUMN, pa.list_(pa.uint32()), nullable=False),
            pa.field(RGBA_BYTES_COLUMN, pa.list_(pa.binary()), nullable=False),
        ],
        metadata={
            PROTOCOL_VERSION_KEY: PROTOCOL_VERSION,
            TABLE_KIND_KEY: b"texture",
        },
    )
    return schema, pa.record_batch(
        [
            legacy_array,
            pa.array([[row[0] for row in rows]], type=pa.list_(pa.uint32())),
            pa.array([[row[2] for row in rows]], type=pa.list_(pa.uint32())),
            pa.array([[row[1] for row in rows]], type=pa.list_(pa.uint32())),
            pa.array(
                [[row[3].tobytes() for row in rows]],
                type=pa.list_(pa.binary()),
            ),
        ],
        schema=schema,
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Convert images to a keyed trd texture Arrow IPC stream (0.0.6)."
    )
    ap.add_argument("input", nargs="+", help="input image file(s) (JPEG/PNG/...)")
    ap.add_argument(
        "--mesh-id",
        action="append",
        type=int,
        dest="mesh_ids",
        help="mesh id for each input; defaults to positional 0,1,...",
    )
    ap.add_argument("-o", "--output", default="-", help="output path ('-' = stdout)")
    ap.add_argument(
        "--max-size",
        type=int,
        default=None,
        metavar="N",
        help="downscale (preserving aspect) so max(H, W) <= N; GPUs cap texture size",
    )
    args = ap.parse_args()

    mesh_ids = args.mesh_ids if args.mesh_ids is not None else list(range(len(args.input)))
    if len(mesh_ids) != len(args.input):
        raise SystemExit("error: --mesh-id count must match the number of input images")
    if any(mesh_id < 0 for mesh_id in mesh_ids):
        raise SystemExit("error: mesh ids must be non-negative")
    if len(set(mesh_ids)) != len(mesh_ids):
        raise SystemExit("error: mesh ids must be unique")
    rows = [
        (mesh_id, *image_to_rgba(path, args.max_size))
        for mesh_id, path in zip(mesh_ids, args.input)
    ]
    schema, batch = texture_batch(rows)

    sink = sys.stdout.buffer if args.output == "-" else open(args.output, "wb")
    try:
        with ipc.new_stream(sink, schema) as writer:
            writer.write_batch(batch)
    finally:
        if sink is not sys.stdout.buffer:
            sink.close()


if __name__ == "__main__":
    main()
