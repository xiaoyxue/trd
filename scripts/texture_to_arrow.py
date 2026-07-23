#!/usr/bin/env python3
"""Convert an image (JPEG/PNG/...) to a trd **texture** Arrow IPC stream (0.0.5).

A trd texture table is **one row = one image**, carried in a single column
(matching ``trd_core::texture::ImageTexture::from_arrow``):

  * ``rgba`` — a canonical ``arrow.fixed_shape_tensor<uint8>`` of shape
    ``[H, W, 4]``: interleaved RGBA8, row-major, **top-left origin**. Concretely
    the storage is ``FixedSizeList<UInt8>[H*W*4]`` and the field carries the
    ``ARROW:extension:name = arrow.fixed_shape_tensor`` /
    ``ARROW:extension:metadata`` (a JSON ``{"shape":[H,W,4], ...}``)
    canonical-extension metadata that makes the ``[H, W, 4]`` self-describing —
    height/width come from that shape, not a separate column.

This is the *same* ``fixed_shape_tensor<u8>`` canonical extension family trd
already emits on its output (a rendered frame, ``output.rs``), so a texture input
is symmetric with a rendered frame — only the layout differs (a texture is one
interleaved ``[H, W, 4]`` tensor; the output is per-channel planar ``[H, W]``
tensors). ``ImageTexture::from_arrow`` reads row 0's ``H*W*4`` bytes as row-major
RGBA and takes the height/width from the tensor shape.

Run via:
  uv run --with pyarrow --with pillow --with numpy \\
      scripts/texture_to_arrow.py albedo.png -o albedo.tex.arrows
  uv run --with pyarrow --with pillow --with numpy \\
      scripts/texture_to_arrow.py photo.jpg --max-size 2048           # -> stdout
"""
import argparse
import sys

import numpy as np
import pyarrow as pa
from pyarrow import ipc
from PIL import Image

PROTOCOL_VERSION_KEY = b"trd.protocol.version"
PROTOCOL_VERSION = b"0.0.5"

# The single image column of a trd texture table (see `trd_core::texture`).
TEXTURE_COLUMN = "rgba"


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


def texture_batch(height, width, flat):
    """Build a one-row texture ``RecordBatch`` from flat row-major RGBA bytes.

    The ``rgba`` column is a canonical ``arrow.fixed_shape_tensor<uint8>`` of
    shape ``[H, W, 4]`` (storage ``FixedSizeList<UInt8>[H*W*4]``); the field's
    canonical-extension metadata makes the shape self-describing, exactly as the
    Rust decoder (``ImageTexture::from_arrow``) expects.
    """
    n = height * width * 4
    if flat.size != n:
        raise ValueError(f"expected {n} RGBA bytes, got {flat.size}")
    # Canonical fixed_shape_tensor: storage is FixedSizeList<uint8>[H*W*4] and the
    # field gains `ARROW:extension:name`/`ARROW:extension:metadata` on IPC. Shape
    # is [height, width, 4] to match the Rust decoder's `[H, W, 4]` match.
    tensor_type = pa.fixed_shape_tensor(
        pa.uint8(), [height, width, 4], dim_names=["height", "width", "channel"]
    )
    storage = pa.FixedSizeListArray.from_arrays(pa.array(flat, type=pa.uint8()), n)
    array = pa.ExtensionArray.from_storage(tensor_type, storage)
    field = pa.field(TEXTURE_COLUMN, tensor_type, nullable=False)
    schema = pa.schema([field], metadata={PROTOCOL_VERSION_KEY: PROTOCOL_VERSION})
    return schema, pa.record_batch([array], schema=schema)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Convert an image to a trd texture Arrow IPC stream (0.0.5)."
    )
    ap.add_argument("input", help="input image file (JPEG/PNG/...)")
    ap.add_argument("-o", "--output", default="-", help="output path ('-' = stdout)")
    ap.add_argument(
        "--max-size",
        type=int,
        default=None,
        metavar="N",
        help="downscale (preserving aspect) so max(H, W) <= N; GPUs cap texture size",
    )
    args = ap.parse_args()

    height, width, flat = image_to_rgba(args.input, args.max_size)
    schema, batch = texture_batch(height, width, flat)

    sink = sys.stdout.buffer if args.output == "-" else open(args.output, "wb")
    try:
        with ipc.new_stream(sink, schema) as writer:
            writer.write_batch(batch)
    finally:
        if sink is not sys.stdout.buffer:
            sink.close()


if __name__ == "__main__":
    main()
