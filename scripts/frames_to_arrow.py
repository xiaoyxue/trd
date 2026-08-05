#!/usr/bin/env python3
"""Pack image files into a protocol 0.0.6 inline ``frames`` resource table.

One input image becomes one table row; its 0-based row ordinal is the
``frame_id`` referenced by the terminal params table.

Storage modes:

* ``bytes``: ``frame_bytes: Binary`` containing the original PNG/JPEG bytes.
  This is the recommended clip representation because data stays compressed
  until a frame is selected.
* ``pixels``: ``frame_pixels: arrow.fixed_shape_tensor<uint8>[H,W,4]``.
  Every image must have the same dimensions; pixels are row-major RGBA with a
  top-left origin.

Examples:

  uv run --with pyarrow scripts/frames_to_arrow.py --storage bytes \
      frames/frame_000000.jpg frames/frame_000001.jpg -o frames.arrow

  uv run --with pyarrow --with pillow --with numpy \
      scripts/frames_to_arrow.py --storage pixels frames/*.png -o frames.arrow
"""

import argparse
from pathlib import Path
import sys

import pyarrow as pa
from pyarrow import ipc

PROTOCOL_VERSION_KEY = b"trd.protocol.version"
PROTOCOL_VERSION = b"0.0.6"
TABLE_KIND_KEY = b"trd.table.kind"


def frames_batch(paths: list[Path], storage: str):
    if not paths:
        raise ValueError("frames table needs at least one image")

    metadata = {
        PROTOCOL_VERSION_KEY: PROTOCOL_VERSION,
        TABLE_KIND_KEY: b"frames",
    }
    if storage == "bytes":
        rows = []
        for path in paths:
            if path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
                raise ValueError(f"{path}: encoded frames must be PNG or JPEG")
            payload = path.read_bytes()
            if not payload:
                raise ValueError(f"{path}: encoded frame is empty")
            rows.append(payload)
        field = pa.field("frame_bytes", pa.binary(), nullable=False)
        schema = pa.schema([field], metadata=metadata)
        return schema, pa.record_batch(
            [pa.array(rows, type=pa.binary())], schema=schema
        )

    import numpy as np
    from PIL import Image

    images = []
    size = None
    for path in paths:
        with Image.open(path) as image:
            rgba = np.asarray(image.convert("RGBA"), dtype=np.uint8)
        height, width, channels = rgba.shape
        if channels != 4:
            raise ValueError(f"{path}: expected RGBA pixels")
        if size is None:
            size = (height, width)
        elif size != (height, width):
            raise ValueError(
                f"{path}: dimensions {width}x{height} differ from "
                f"{size[1]}x{size[0]}"
            )
        images.append(np.ascontiguousarray(rgba))

    height, width = size
    row_size = height * width * 4
    tensor_type = pa.fixed_shape_tensor(
        pa.uint8(),
        [height, width, 4],
        dim_names=["height", "width", "channel"],
    )
    flat = np.stack(images).reshape(-1)
    storage_array = pa.FixedSizeListArray.from_arrays(
        pa.array(flat, type=pa.uint8()), row_size
    )
    array = pa.ExtensionArray.from_storage(tensor_type, storage_array)
    field = pa.field("frame_pixels", tensor_type, nullable=False)
    schema = pa.schema([field], metadata=metadata)
    return schema, pa.record_batch([array], schema=schema)


def write_frames_stream(paths: list[Path], output, storage: str) -> None:
    schema, batch = frames_batch(paths, storage)
    with ipc.new_stream(output, schema) as writer:
        writer.write_batch(batch)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pack PNG/JPEG images into a trd 0.0.6 frames table."
    )
    parser.add_argument("images", nargs="+", type=Path, help="images in frame_id order")
    parser.add_argument(
        "--storage",
        choices=["bytes", "pixels"],
        default="bytes",
        help="encoded Binary (default) or raw fixed-shape RGBA tensor",
    )
    parser.add_argument("-o", "--output", default="-", help="output path ('-' = stdout)")
    args = parser.parse_args()

    sink = sys.stdout.buffer if args.output == "-" else open(args.output, "wb")
    try:
        write_frames_stream(args.images, sink, args.storage)
    except ValueError as error:
        raise SystemExit(f"error: {error}") from error
    finally:
        if sink is not sys.stdout.buffer:
            sink.close()


if __name__ == "__main__":
    main()
