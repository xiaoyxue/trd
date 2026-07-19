#!/usr/bin/env python3
"""Convert one or more Wavefront ``.obj`` meshes to a trd **mesh** Arrow IPC
stream (0.0.3).

A trd 0.0.3 input stream is two concatenated Arrow IPC streams — ``[mesh][params]``
— so a mesh table is authored by this script and the per-frame params by
``scripts/jsonl_to_arrow.py``; ``examples/render.sh --mesh`` concatenates them.

Because Arrow requires every column of a record batch to have the same length —
while a mesh has a different number of vertices and indices — each mesh travels as
**one row = one mesh**, with the per-vertex/per-index data nested in list columns
(matching ``trd_core::Mesh::from_arrow_all``):

  * ``position`` — ``List<FixedSizeList<Float32>[3]>`` (x, y, z), required.
  * ``color``    — ``List<FixedSizeList<Float32>[3]>``, optional; emitted only when
                   *every* mesh carries the ``v x y z r g b`` vertex-color extension.
  * ``index``    — ``List<UInt32>`` triangle list (faces triangulated by a fan).

Passing several ``.obj`` files produces a **multi-row** mesh table (one row per
file, in order); a stream's per-frame ``draw_mesh`` list references these meshes
by 0-based row index. Only ``v`` (positions, with optional trailing r g b) and
``f`` (faces) are read; normals/texcoords are ignored. Face vertex references use
the position index only (``a/b/c`` → ``a``); negative indices are relative to the
current vertex count.

Run via:
  uv run --with pyarrow scripts/obj_to_arrow.py assets/meshes/bunny.obj      # -> stdout
  python scripts/obj_to_arrow.py a.obj b.obj -o scene.mesh.arrows            # 2-mesh table
"""
import argparse
import sys

import pyarrow as pa
from pyarrow import ipc

PROTOCOL_VERSION_KEY = b"trd.protocol.version"
PROTOCOL_VERSION = b"0.0.3"


def parse_obj(text):
    """Parse OBJ text into (positions, colors, indices).

    ``positions`` is a list of ``[x, y, z]``; ``colors`` is a list of ``[r, g, b]``
    (empty unless *every* vertex carried a color); ``indices`` is a flat triangle
    list of 0-based ``uint32`` vertex indices.
    """
    positions = []
    colors = []
    indices = []
    for line in text.splitlines():
        parts = line.split()
        if not parts:
            continue
        tag = parts[0]
        if tag == "v":
            coords = [float(v) for v in parts[1:]]
            positions.append(coords[0:3])
            if len(coords) >= 6:
                colors.append(coords[3:6])
        elif tag == "f":
            # Resolve each face-vertex reference to a 0-based position index,
            # then fan-triangulate the (possibly n-gon) polygon.
            verts = []
            for token in parts[1:]:
                raw = int(token.split("/")[0])
                verts.append(raw - 1 if raw > 0 else len(positions) + raw)
            for i in range(1, len(verts) - 1):
                indices.extend((verts[0], verts[i], verts[i + 1]))

    if len(colors) != len(positions):
        # Vertex colors are all-or-nothing; drop partial colors (default white).
        colors = []
    return positions, colors, indices


def mesh_batch(meshes):
    """Build the nested-list mesh ``RecordBatch`` — one row per mesh.

    ``meshes`` is a list of ``(positions, colors, indices)`` tuples. The optional
    ``color`` column is emitted only when *every* mesh carries vertex colors.
    """
    f32 = pa.float32()
    vec3 = pa.list_(f32, 3)  # FixedSizeList<Float32>[3]
    geom_type = pa.list_(vec3)  # List<FixedSizeList<Float32>[3]>
    index_type = pa.list_(pa.uint32())  # List<UInt32>

    columns = [pa.array([m[0] for m in meshes], type=geom_type)]
    fields = [("position", geom_type)]
    if all(m[1] for m in meshes):
        columns.append(pa.array([m[1] for m in meshes], type=geom_type))
        fields.append(("color", geom_type))
    columns.append(pa.array([m[2] for m in meshes], type=index_type))
    fields.append(("index", index_type))

    schema = pa.schema(fields, metadata={PROTOCOL_VERSION_KEY: PROTOCOL_VERSION})
    return schema, pa.record_batch(columns, schema=schema)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("input", nargs="+", help="Wavefront .obj mesh file(s), one row per file")
    ap.add_argument("-o", "--output", default="-", help="output path ('-' = stdout)")
    args = ap.parse_args()

    meshes = []
    for path in args.input:
        with open(path, encoding="utf-8") as f:
            positions, colors, indices = parse_obj(f.read())
        if not positions or not indices:
            sys.exit(f"error: {path} has no triangles (need `v` and `f` lines)")
        meshes.append((positions, colors, indices))

    schema, batch = mesh_batch(meshes)

    sink = sys.stdout.buffer if args.output == "-" else open(args.output, "wb")
    try:
        with ipc.new_stream(sink, schema) as writer:
            writer.write_batch(batch)
    finally:
        if sink is not sys.stdout.buffer:
            sink.close()


if __name__ == "__main__":
    main()
