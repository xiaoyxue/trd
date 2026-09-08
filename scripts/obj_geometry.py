#!/usr/bin/env python3
"""Device-free OBJ geometry parsing for offline conversion to self-contained GLB."""


def parse_obj(text):
    """Parse OBJ text into (positions, colors, uvs, indices).

    ``positions`` is a list of ``[x, y, z]``; ``colors`` is a list of ``[r, g, b]``
    (empty unless *every* vertex carried a color); ``uvs`` is a list of ``[u, v]``
    parallel to ``positions`` (empty unless the mesh carried ``vt`` texcoords),
    already V-flipped to the top-left texel origin; ``indices`` is a flat triangle
    list of 0-based ``uint32`` vertex indices.

    OBJ indexes positions and texcoords **independently** (``f v/vt/vn``), so one
    position can pair with *different* texcoords on different faces — exactly what
    happens at every UV-unwrap seam (where atlas islands meet). trd carries **one
    uv per vertex**, so each unique ``(position, texcoord)`` corner is emitted as
    its own output vertex (the standard OBJ→GPU de-duplication, matching tobj's
    ``single_index`` expansion). Collapsing to one uv per *position* instead would
    stretch seam triangles across the whole atlas (sampling the wrong islands, the
    background, even a watermark banner) — visible as garbled seams (#20).
    """
    positions = []  # raw `v x y z`
    colors = []  # raw `v ... r g b` (parallel to positions when present)
    texcoords = []  # raw `vt u v`, bottom-up (OBJ origin)
    out_positions = []  # split: one entry per unique (position, texcoord) corner
    out_pos_index = []  # source position index for each split vertex (for colors)
    out_uvs = []  # split, V-flipped to the top-left texel origin
    indices = []
    corner_map = {}  # (position index, texcoord index|None) -> split vertex index
    have_texcoords = False

    def corner_vertex(pi, ti):
        """Return the split-vertex index for the ``(pi, ti)`` face corner."""
        key = (pi, ti)
        idx = corner_map.get(key)
        if idx is None:
            idx = len(out_positions)
            corner_map[key] = idx
            out_positions.append(positions[pi])
            out_pos_index.append(pi)
            if ti is not None and 0 <= ti < len(texcoords):
                u, v = texcoords[ti]
                out_uvs.append([u, 1.0 - v])
            else:
                out_uvs.append([0.0, 0.0])
        return idx

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
                have_texcoords = True
        elif tag == "f":
            # Resolve each face-vertex reference to a 0-based position index (and
            # optional texcoord index), split it into a unique corner vertex, then
            # fan-triangulate the (possibly n-gon) polygon.
            verts = []
            for token in parts[1:]:
                fields = token.split("/")
                raw = int(fields[0])
                pi = raw - 1 if raw > 0 else len(positions) + raw
                ti = None
                if len(fields) >= 2 and fields[1] != "":
                    traw = int(fields[1])
                    ti = traw - 1 if traw > 0 else len(texcoords) + traw
                verts.append(corner_vertex(pi, ti))
            for i in range(1, len(verts) - 1):
                indices.extend((verts[0], verts[i], verts[i + 1]))

    # Vertex colors are all-or-nothing; keep them only when every position carried
    # one, expanding per split vertex (duplicated corners share their source color).
    out_colors = []
    if colors and len(colors) == len(positions):
        out_colors = [colors[pi] for pi in out_pos_index]

    # `uv` is emitted only when the mesh carried `vt` lines (V-flipped to the
    # top-left texel origin, matching mesh.rs `from_obj`: `[u, 1.0 - v]`).
    uvs = out_uvs if have_texcoords else []
    return out_positions, out_colors, uvs, indices

