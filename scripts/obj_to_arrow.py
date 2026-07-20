#!/usr/bin/env python3
"""Convert one or more Wavefront ``.obj`` meshes to a trd **mesh** Arrow IPC
stream (0.0.4).

A trd input stream is two concatenated Arrow IPC streams — ``[mesh][params]``
— so a mesh table is authored by this script and the per-frame params by
``scripts/jsonl_to_arrow.py``; ``examples/render.sh --mesh`` concatenates them.

Because Arrow requires every column of a record batch to have the same length —
while a mesh has a different number of vertices and indices — each mesh travels as
**one row = one mesh**, with the per-vertex/per-index data nested in list columns
(matching ``trd_core::Mesh::from_arrow_all``):

  * ``position`` — ``List<FixedSizeList<Float32>[3]>`` (x, y, z), required.
  * ``color``    — ``List<FixedSizeList<Float32>[3]>``, optional; emitted only when
                   *every* mesh carries the ``v x y z r g b`` vertex-color extension.
  * ``uv``       — ``List<FixedSizeList<Float32>[2]>``, optional (0.0.4); emitted
                   when *any* mesh carries ``vt`` texcoords. One uv per position
                   vertex, parallel to ``position``, already V-flipped to the
                   top-left texel origin (see below). Meshes lacking texcoords get
                   all-zero uvs so every column stays equal length.
  * ``index``    — ``List<UInt32>`` triangle list (faces triangulated by a fan).

Passing several ``.obj`` files produces a **multi-row** mesh table (one row per
file, in order); a stream's per-frame ``draw_mesh`` list references these meshes
by 0-based row index. ``v`` (positions, with optional trailing r g b), ``vt``
(texcoords) and ``f`` (faces) are read; normals are ignored. Face vertex
references use the position index (``a/vt/vn`` → ``a``) and, when present, the
texcoord index (the middle field). Negative indices are relative to the current
vertex/texcoord count.

Because OBJ allows a position vertex to pair with different texcoords on different
faces (while trd carries **one uv per position vertex**), each position is
assigned the uv of the *last* face corner that references it — a close match to
the tobj ``single_index`` expansion used by ``trd_core::Mesh::from_obj``. OBJ
``vt`` v runs bottom-up, so uv is emitted V-flipped as ``[u, 1.0 - v]`` to match
that loader's top-left texel origin, keeping Arrow-loaded and OBJ-loaded meshes
in agreement.

Run via:
  uv run --with pyarrow scripts/obj_to_arrow.py assets/meshes/bunny.obj      # -> stdout
  python scripts/obj_to_arrow.py a.obj b.obj -o scene.mesh.arrows            # 2-mesh table
"""
import argparse
import sys

import pyarrow as pa
from pyarrow import ipc

PROTOCOL_VERSION_KEY = b"trd.protocol.version"
PROTOCOL_VERSION = b"0.0.4"


def parse_obj(text):
    """Parse OBJ text into (positions, colors, uvs, indices).

    ``positions`` is a list of ``[x, y, z]``; ``colors`` is a list of ``[r, g, b]``
    (empty unless *every* vertex carried a color); ``uvs`` is a list of ``[u, v]``
    parallel to ``positions`` (empty unless the mesh carried ``vt`` texcoords),
    already V-flipped to the top-left texel origin; ``indices`` is a flat triangle
    list of 0-based ``uint32`` vertex indices.
    """
    positions = []
    colors = []
    texcoords = []  # raw `vt u v`, bottom-up (OBJ origin)
    pos_texcoord = {}  # position index -> texcoord index (last face corner wins)
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
        elif tag == "vt":
            uv = [float(v) for v in parts[1:3]]
            if len(uv) == 2:
                texcoords.append(uv)
        elif tag == "f":
            # Resolve each face-vertex reference to a 0-based position index (and
            # optional texcoord index), then fan-triangulate the (possibly n-gon)
            # polygon.
            verts = []
            for token in parts[1:]:
                fields = token.split("/")
                raw = int(fields[0])
                pi = raw - 1 if raw > 0 else len(positions) + raw
                verts.append(pi)
                if len(fields) >= 2 and fields[1] != "":
                    traw = int(fields[1])
                    ti = traw - 1 if traw > 0 else len(texcoords) + traw
                    # The last face corner referencing this position wins — a
                    # close match to tobj's single_index expansion.
                    pos_texcoord[pi] = ti
            for i in range(1, len(verts) - 1):
                indices.extend((verts[0], verts[i], verts[i + 1]))

    if len(colors) != len(positions):
        # Vertex colors are all-or-nothing; drop partial colors (default white).
        colors = []

    # Build a `uv` parallel to `positions`, V-flipped to match the Rust OBJ loader
    # (mesh.rs `from_obj`: `[u, 1.0 - v]`). Positions with no texcoord reference
    # default to [0, 0]. Emitted only when the mesh carried `vt` lines.
    uvs = []
    if texcoords:
        for pi in range(len(positions)):
            ti = pos_texcoord.get(pi)
            if ti is not None and 0 <= ti < len(texcoords):
                u, v = texcoords[ti]
                uvs.append([u, 1.0 - v])
            else:
                uvs.append([0.0, 0.0])
    return positions, colors, uvs, indices


def mesh_batch(meshes):
    """Build the nested-list mesh ``RecordBatch`` — one row per mesh.

    ``meshes`` is a list of ``(positions, colors, uvs, indices)`` tuples. The
    optional ``color`` column is emitted only when *every* mesh carries vertex
    colors; the optional ``uv`` column is emitted when *any* mesh carries
    texcoords (meshes lacking them get all-zero uvs so every column stays equal
    length).
    """
    f32 = pa.float32()
    vec3 = pa.list_(f32, 3)  # FixedSizeList<Float32>[3]
    vec2 = pa.list_(f32, 2)  # FixedSizeList<Float32>[2]
    geom_type = pa.list_(vec3)  # List<FixedSizeList<Float32>[3]>
    uv_type = pa.list_(vec2)  # List<FixedSizeList<Float32>[2]>
    index_type = pa.list_(pa.uint32())  # List<UInt32>

    columns = [pa.array([m[0] for m in meshes], type=geom_type)]
    fields = [("position", geom_type)]
    if all(m[1] for m in meshes):
        columns.append(pa.array([m[1] for m in meshes], type=geom_type))
        fields.append(("color", geom_type))
    if any(m[2] for m in meshes):
        # A mesh without texcoords gets one all-zero uv per position so the `uv`
        # column stays parallel to `position` (equal-length Arrow columns).
        uv_rows = [m[2] if m[2] else [[0.0, 0.0]] * len(m[0]) for m in meshes]
        columns.append(pa.array(uv_rows, type=uv_type))
        fields.append(("uv", uv_type))
    columns.append(pa.array([m[3] for m in meshes], type=index_type))
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
            positions, colors, uvs, indices = parse_obj(f.read())
        if not positions or not indices:
            sys.exit(f"error: {path} has no triangles (need `v` and `f` lines)")
        meshes.append((positions, colors, uvs, indices))

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
