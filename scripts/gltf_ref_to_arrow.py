#!/usr/bin/env python3
"""Write one reference-only GLB/glTF 2.0 mesh row for protocol 0.0.6."""

import argparse
import sys

import pyarrow as pa
from pyarrow import ipc

PROTOCOL_VERSION_KEY = b"trd.protocol.version"
PROTOCOL_VERSION = b"0.0.6"
TABLE_KIND_KEY = b"trd.table.kind"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", help="native/local GLB or glTF path")
    parser.add_argument("--url", help="HTTP(S) or browser-resolvable GLB/glTF URL")
    parser.add_argument("-o", "--output", default="-", help="output path ('-' = stdout)")
    args = parser.parse_args()
    if not args.path and not args.url:
        parser.error("at least one of --path or --url is required")

    f32 = pa.float32()
    vec3 = pa.list_(f32, 3)
    vec2 = pa.list_(f32, 2)
    fields = [
        pa.field("position", pa.list_(vec3), nullable=True),
        pa.field("color", pa.list_(vec3), nullable=True),
        pa.field("uv", pa.list_(vec2), nullable=True),
        pa.field("index", pa.list_(pa.uint32()), nullable=True),
        pa.field("gltf_path", pa.utf8(), nullable=True),
        pa.field("gltf_url", pa.utf8(), nullable=True),
        pa.field("material", pa.utf8(), nullable=True),
    ]
    schema = pa.schema(
        fields,
        metadata={
            PROTOCOL_VERSION_KEY: PROTOCOL_VERSION,
            TABLE_KIND_KEY: b"mesh",
        },
    )
    columns = [
        pa.array([None], type=field.type)
        for field in fields[:4]
    ] + [
        pa.array([args.path], type=pa.utf8()),
        pa.array([args.url], type=pa.utf8()),
        pa.array([None], type=pa.utf8()),
    ]
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
