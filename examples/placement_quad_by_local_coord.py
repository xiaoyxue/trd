#!/usr/bin/env python3
"""Find a **placement quad**'s local coordinate frame from ``K`` and place a mesh on it.

Given, per video frame, the camera intrinsics ``K`` and the 4 image points of a
planar **placement quad** (a quad in the scene that positions the mesh — e.g. a
poster on the floor), this recovers the quad's plane frame ``(e1,e2,e3)`` directly
in the **camera E³** frame — a single-view projective-metrology reconstruction
(#77 / VideoAnalysis#1206, via vanishing points), using **only ``K`` + the quad**
(*no camera extrinsics* / ``Pose``). A model mesh (e.g. the bunny) is then anchored
on that frame and ``K`` projects it, so it stays glued to the placement quad as the
real camera dollies. This is video-agnostic; ``cornellbox`` is just the sample clip
(pass ``--assets`` / ``--from-perception`` for another).

The tool runs in two stages, mirroring the pipeline:

  * **Stage 1 — before placing the mesh** (``--placement-quad --no-place-mesh``):
    reconstruct the local frame and emit only the placement-quad draw. Rendered
    with ``--placement-quad --axes-local`` this shows the quad + its local coords.
  * **Stage 2 — after placing the mesh** (``--place-mesh --placement-quad``):
    additionally anchor the mesh on the frame. Rendered with
    ``--placement-quad --axes-local --aabb`` this adds the mesh with its AABB and
    local coords, on top of stage 1's quad.

The upstream ``K + placement-quad points + frames`` are consumed as an Arrow
stream via ``--from-perception`` (see scripts/perception_to_arrow.py); the
``--assets`` fixture path (``K.txt`` / ``QuadImagePoints.txt``) is the cornellbox
convenience used to *simulate* that upstream and for ``--validate``. This Python
reconstruction is the reference for a future Rust port ("local frame from K +
quads").

Why no Pose is needed (and the render is still exact)
-----------------------------------------------------
trd's CV camera form accepts ``k`` **without** ``pose`` (``check_camera_form``:
``k.is_some()`` alone is a valid CV form), in which case the view matrix is the
identity and the projection is ``projection_from_intrinsics(k)``. So we render in
the **camera frame**: the clip transform is ``P · I · model``. We build ``model``
in the OpenGL camera frame as ``C4 · m_cam`` (``C4 = diag(1,-1,-1,1)`` flips the
OpenCV camera axes Y-down/Z-forward to GL Y-up/Z-back), where ``m_cam`` is the
placement in the OpenCV camera E³. This yields pixel-identical output to lifting
the mesh into world space with the true ``pose`` and rendering with ``pose⁻¹`` as
the view — the ``pose``/``pose⁻¹`` pair cancels — so dropping Pose changes nothing
on screen while removing the extrinsics requirement entirely.

``K`` is in **pixels** (resolution-specific): authored for a source resolution
(``--src-width``/``--src-height``, default 1920×1080) and scaled linearly to the
render resolution so the projected mesh lines up with the stretched background plate.

Run (from ``nix develop``) — simulate the upstream perception stream, then this stage::

    uv run --with pyarrow --with numpy scripts/perception_to_arrow.py \
        --assets assets/videos/cornellbox -o examples/frames.cornellbox.perception.arrow
    # stage 1 (placement quad only) and stage 2 (mesh placed):
    uv run --with pyarrow --with numpy examples/placement_quad_by_local_coord.py \
        --from-perception examples/frames.cornellbox.perception.arrow \
        --placement-quad --no-place-mesh --placement-quad-mesh-index 0 \
        -o examples/frames.cornellbox.stage1.jsonl
    uv run --with pyarrow --with numpy examples/placement_quad_by_local_coord.py \
        --from-perception examples/frames.cornellbox.perception.arrow \
        --placement-quad --place-mesh \
        -o examples/frames.cornellbox.stage2.jsonl

    # Full 250-frame bunny-only variant for an inline frames resource table:
    uv run --with numpy examples/placement_quad_by_local_coord.py \
        --assets assets/videos/cornellbox --step 1 --inline-frames \
        --width 1920 --height 1080 \
        -o examples/frames.cornellbox.inline.jsonl

then render each stage's GIF via trd-cli (wrap with nixGL on native GPU boxes)::

    examples/render.sh --cli --placement-quad --axes-local \
        --frames-base output/cornellbox \
        examples/frames.cornellbox.stage1.jsonl output/cornellbox_stage1.gif 960 540 25
    examples/render.sh --cli --placement-quad --axes-local --aabb \
        --mesh assets/meshes/bunny_with_texture/bunny.obj \
        --texture assets/meshes/bunny_with_texture/bunny_uv_map1.jpg \
        --frames-base output/cornellbox \
        examples/frames.cornellbox.stage2.jsonl output/cornellbox_stage2.gif 960 540 25

Optionally cross-check this single-view frame against the multi-view ground truth
(``cornellbox_gt.py``, which *does* use ``Pose.txt``)::

    uv run --with numpy examples/placement_quad_by_local_coord.py --validate \
        --frames-list 0 60 125 190 249
"""
import argparse
import json
import os
import re
import sys

import numpy as np

DEFAULT_SRC_W, DEFAULT_SRC_H = 1920, 1080  # resolution K.txt is authored for


# --------------------------------------------------------------------------- #
# Inputs the *method* needs: K + the per-frame quad image points (no Pose).
# --------------------------------------------------------------------------- #
def parse_k(path):
    """Parse the 3×3 OpenCV intrinsics from ``K.txt`` (skips ``#`` comments)."""
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            rows.append([float(x) for x in line.split()])
    k = np.array(rows, dtype=np.float64)
    if k.shape != (3, 3):
        raise SystemExit(f"error: K.txt is not 3×3: {k.shape}")
    return k


def parse_quads(path):
    """Parse per-frame 4 image-space quad points from ``QuadImagePoints.txt``."""
    pt = re.compile(r"\(\s*([-\d.eE+]+)\s*,\s*([-\d.eE+]+)\s*\)")
    quads = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip().startswith("frame"):
                continue
            pts = [(float(a), float(b)) for a, b in pt.findall(line)]
            if len(pts) != 4:
                raise SystemExit(f"error: expected 4 quad points, got {len(pts)}: {line!r}")
            quads.append(np.array(pts, dtype=np.float64))
    return quads


def colmajor(m):
    """Flatten a 4×4 numpy matrix to a 16-float column-major list (glam layout)."""
    return [float(x) for x in m.flatten(order="F")]


def _n(v):
    return v / np.linalg.norm(v)


def _line(p, q):
    """Homogeneous image line through two 2-D points."""
    return np.cross([p[0], p[1], 1.0], [q[0], q[1], 1.0])


# --------------------------------------------------------------------------- #
# #77 single-view projective metrology (K + one quad → plane frame in camera E³)
# --------------------------------------------------------------------------- #
def homography_unit_square_to_quad(quad):
    """DLT homography mapping the unit square (0,0)(1,0)(1,1)(0,1) → quad (px)."""
    src = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
    A = []
    for (x, y), (u, v) in zip(src, quad):
        A.append([-x, -y, -1, 0, 0, 0, u * x, u * y, u])
        A.append([0, 0, 0, -x, -y, -1, v * x, v * y, v])
    _, _, vt = np.linalg.svd(np.array(A))
    return vt[-1].reshape(3, 3)


def pose_from_quad(quad, K):
    """Zhang homography decomposition: unit-square→quad ⇒ (r1, r2, t) in camera E³.

    ``H ≅ K·[r1 r2 t]`` ⇒ ``K⁻¹H = [r1 r2 t]`` up to the scale ``λ`` that makes
    ``r1`` a unit vector. Sign fixed so the plane is in front (``t_z > 0``).
    """
    H = homography_unit_square_to_quad(quad)
    B = np.linalg.inv(K) @ H
    lam = 1.0 / np.linalg.norm(B[:, 0])
    r1, r2, t = lam * B[:, 0], lam * B[:, 1], lam * B[:, 2]
    if t[2] < 0:
        r1, r2, t = -r1, -r2, -t
    return r1, r2, t


def normal_basis_from_quad(quad, K):
    """#77 / VideoAnalysis#1206: local frame ``(e1,e2,e3)`` on the quad's plane.

    Returns ``(origin3d, e (3×3 rows e1,e2,e3), origin_px, lam)`` in the camera E³
    frame, or ``None`` for a degenerate (near-parallel-edge) quad.
    """
    p1, p2, p3, p4 = [np.asarray(p, float) for p in quad]
    Kinv = np.linalg.inv(K)

    # (1) plane normal from the two base vanishing points.
    v1 = np.cross(_line(p1, p2), _line(p3, p4))  # VP of edges p1p2 ∥ p3p4
    v2 = np.cross(_line(p2, p3), _line(p4, p1))  # VP of edges p2p3 ∥ p4p1
    d1, d2 = Kinv @ v1, Kinv @ v2
    cr = np.cross(d1, d2)
    if np.linalg.norm(cr) < 1e-9:
        return None  # near-parallel edges → degenerate
    n = _n(cr)
    z = _n(Kinv @ (K @ n))  # == n; back-project the VP *direction* (step-1 gotcha)

    # (2) in-plane axes: seed e1 from the quad edge, Gram–Schmidt against z.
    r1, r2, t = pose_from_quad(quad, K)
    x = _n(r1 - np.dot(r1, z) * z)
    y = _n(np.cross(z, x))

    # (4) anchor at the quad centre (unit-square (0.5,0.5)).
    origin3d = 0.5 * r1 + 0.5 * r2 + t
    lam = 0.5 * np.linalg.norm(r1)

    def proj(X):
        p = K @ X
        return p[:2] / p[2]

    o_px = proj(origin3d)

    # (3) orientation convention: z tip must be *above* the origin in the image
    # (image +v is down), else flip z (and recompute y to stay right-handed).
    if proj(origin3d + lam * z)[1] > o_px[1]:
        z = -z
        y = _n(np.cross(z, x))

    return origin3d, np.array([x, y, z]), o_px, lam


def rotate_y(theta):
    """4×4 rotate about local +Y (glam ``from_rotation_y``)."""
    c, s = np.cos(theta), np.sin(theta)
    return np.array(
        [
            [c, 0.0, s, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [-s, 0.0, c, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )


def ang(a, b):
    """Angle (deg) between two vectors, sign-folded (axes are ± ambiguous)."""
    c = abs(np.dot(_n(a), _n(b)))
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


# --------------------------------------------------------------------------- #
# Author the scene (Pose-free): render in the camera frame, view = identity.
# --------------------------------------------------------------------------- #
def k_render_columns(K, width, height, src_w, src_h):
    """K scaled to the render resolution, as trd's column-major ``k`` (9 floats)."""
    sx, sy = width / src_w, height / src_h
    return [
        K[0, 0] * sx, 0.0, 0.0,
        K[0, 1] * sx, K[1, 1] * sy, 0.0,
        K[0, 2] * sx, K[1, 2] * sy, 1.0,
    ]


def emit_placement_jsonl(K, quads, args):
    """Emit a 0.0.6 JSONL that places the bunny **per frame via #77's single-view
    basis**, using only ``K`` + the quad — **no ``pose`` column**.

    Reads the recorded ``K.txt``/``QuadImagePoints.txt`` fixture directly: builds
    the per-frame ``(K, quad, frame_path)`` records (applying ``--step``/
    ``--limit`` and synthesizing ``frames/frame_%06d``), then hands them to the
    shared :func:`emit_records` reconstructor.
    """
    idx = list(range(0, len(quads), args.step))
    if args.limit is not None:
        idx = idx[: args.limit]
    records = [
        (K, quads[f], f"{args.frame_rel}/frame_{f:06d}.{args.frame_ext}") for f in idx
    ]
    emit_records(records, args)


def frame_reference(frame_path, inline_frames):
    """Return the params field selecting this record's background resource."""
    if not inline_frames:
        return {"frame_path": frame_path}
    match = re.search(r"(?:^|[/\\])frame_(\d+)\.[^./\\]+$", frame_path)
    if match is None:
        raise SystemExit(
            "error: --inline-frames requires frame paths ending in "
            f"frame_NNNNNN.ext, got {frame_path!r}"
        )
    return {"frame_id": int(match.group(1))}


def read_perception_records(path):
    """Read the upstream perception Arrow stream (scripts/perception_to_arrow.py).

    Returns the per-frame ``(K (3×3), quad (4×2), frame_path)`` records it carries
    — the *input to this stage*. ``k`` is the row-major 3×3 intrinsics and
    ``placement_quad`` the 4 image points ``[x0,y0,…,x3,y3]``, both in the source
    video's pixel space. A row whose ``k``/``placement_quad`` are **null** (an
    untracked frame — the geometry left the frame; e.g. the FIBA clip's tail) is
    returned as ``(None, None, frame_path)`` so the downstream stage renders it as
    a background-plate-only frame instead of dropping it.
    """
    import pyarrow as pa  # lazy: only the --from-perception path needs Arrow
    from pyarrow import ipc

    src = sys.stdin.buffer if path == "-" else open(path, "rb")
    try:
        table = ipc.open_stream(src).read_all()
    finally:
        if src is not sys.stdin.buffer:
            src.close()
    for col in ("k", "placement_quad", "frame_path"):
        if col not in table.column_names:
            raise SystemExit(f"error: perception stream missing '{col}' column")
    ks = table.column("k").to_pylist()
    quads = table.column("placement_quad").to_pylist()
    frames = table.column("frame_path").to_pylist()
    records = []
    for k_flat, q_flat, frame_path in zip(ks, quads, frames):
        if k_flat is None or q_flat is None:
            # Untracked frame: geometry is null → background plate only.
            records.append((None, None, frame_path))
            continue
        K = np.array(k_flat, dtype=np.float64).reshape(3, 3)  # row-major
        quad = np.array(q_flat, dtype=np.float64).reshape(4, 2)
        records.append((K, quad, frame_path))
    return records


def emit_records(records, args):
    """Reconstruct + emit the 0.0.6 render stream for ``(K, quad, frame_path)`` records.

    The placement-quad's local frame is reconstructed from ``K`` + the quad points.
    With ``--place-mesh`` (default) the model mesh is anchored on it (stage 2); with
    ``--placement-quad`` the quad itself is emitted as an overlay draw. Emitting only
    the quad (``--no-place-mesh --placement-quad``) is stage 1 (before placing the
    mesh). Placements are built in the OpenCV camera E³, flipped OpenCV→GL
    (``C4 = diag(1,-1,-1,1)``), and emitted as per-frame ``model`` with **no camera
    pose**, so trd renders in camera space (view = identity, projection from ``k``).
    ``k`` is the record's (possibly per-frame) intrinsics scaled to the render
    resolution.

    A record with **null geometry** (``K``/``quad`` is ``None`` — an untracked
    frame) is emitted as a **background-plate-only** row: an explicit *empty*
    ``draws`` list (trd-core draws no mesh for it, per
    ``DecodedFrame::resolved_draws``) plus its external ``frame_path`` or inline
    ``frame_id``. It carries a
    placeholder ``k`` (the last tracked frame's render-K) only so
    ``scripts/jsonl_to_arrow.py``'s all-or-nothing ``k`` column stays present for
    the tracked frames; the frame plane ignores the camera, so it is never
    projected. The mesh spin phase is normalized over the **tracked** frames, so
    the placed animation is identical whether or not the untracked tail is kept.
    """
    C4 = np.diag([1.0, -1.0, -1.0, 1.0])  # OpenCV camera (Y-down,Z-fwd) → GL (Y-up,Z-back)
    out = sys.stdout if args.output == "-" else open(args.output, "w", encoding="utf-8")
    written = 0

    def render_k(K):
        return [float(x) for x in k_render_columns(K, args.width, args.height,
                                                   args.src_width, args.src_height)]

    tracked = [(K, quad) for (K, quad, _fp) in records if K is not None and quad is not None]
    n_tracked = max(1, len(tracked))
    # One in-plane anchor per mesh copy. --place-offset (repeatable) wins; otherwise
    # the single --place-offset-e1/e2 pair (backward compatible).
    placements = (args.place_offset if args.place_offset
                  else [(args.place_offset_e1, args.place_offset_e2)])
    # Placeholder render-K for untracked (background-only) rows; seed with the
    # first tracked frame's so a leading untracked run still has one, then hold
    # the last tracked frame's as playback advances.
    last_render_k = render_k(tracked[0][0]) if tracked else None
    placed_i = 0
    for K, quad, frame_path in records:
        frame_ref = frame_reference(frame_path, args.inline_frames)
        if K is None or quad is None:
            # Untracked frame → just the video still: empty draw list = no mesh.
            row = {"draws": [], **frame_ref}
            if last_render_k is not None:
                row["k"] = last_render_k
            out.write(json.dumps(row) + "\n")
            written += 1
            continue
        nb = normal_basis_from_quad(quad, K)
        if nb is None:
            continue
        o3d, e, _o_px, lam = nb
        e1, e2, e3 = e
        # The placement quad's gizmo axes (what --axes-local draws): red = r1,
        # green = r2, i.e. the two quad half-edges r1/2, r2/2. In-plane offsets
        # are expressed in *these* axes (not the orthonormalised e1/e2), so a
        # request like "move along −green" maps directly to −r2 as seen on screen.
        r1, r2, t = pose_from_quad(quad, K)
        draws = []
        if args.place_mesh:
            size = lam * args.size_factor  # mesh half-extent in the (scaled) reconstruction
            # One or more in-plane anchors: each (e1_off, e2_off) drops a copy of
            # the mesh at a different spot on the P² plane (a --place-offset repeated
            # for a row of cans; falls back to the single --place-offset-e1/e2). A
            # per-copy phase offset spins them out of lockstep so a row doesn't look
            # mirror-identical.
            for copy_i, (oe1, oe2) in enumerate(placements):
                theta = (2.0 * np.pi * args.turns * (placed_i / n_tracked)
                         + 2.0 * np.pi * args.copy_phase * copy_i)
                # In-plane translation stays in the P² local frame: shift the anchor
                # off the quad centre along the quad's own gizmo axes — red (r1) and
                # green (r2) — in quad half-edge units (±1 ≈ a quad edge), then lift
                # along the normal e3 so the feet rest on the plane. Used to move the
                # mesh clear of the active players without leaving the quad plane
                # (e.g. −green/−r2 pushes it down-court, off the mid-court action).
                anchor = (o3d
                          + oe1 * (r1 / 2.0)
                          + oe2 * (r2 / 2.0))
                # OpenCV camera-frame placement: mesh +X→e1, +Y(up)→e3, +Z→e1×e3 (=−e2);
                # centre lifted half a height along e3 so the feet rest on the plane.
                r_place = np.eye(4)
                r_place[:3, 0] = e1
                r_place[:3, 1] = e3
                r_place[:3, 2] = -e2
                trans = np.eye(4)
                trans[:3, 3] = anchor + args.lift * size * e3
                s_mat = np.diag([size, size, size, 1.0])
                m_cam = trans @ r_place @ rotate_y(theta) @ s_mat
                model = C4 @ m_cam  # camera frame → GL camera frame (view = identity)
                draws.append({"mesh": args.mesh_index, "model": colmajor(model)})
                if args.shadow:
                    # Contact / grounding blob shadow on the P² plane, at the mesh's
                    # ground anchor: a flat quad spanning the plane's orthonormal e1/e2
                    # axes, sized to the mesh footprint (--shadow-scale × half-extent).
                    # Stays in the P² local frame so it tracks the recovered floor as
                    # the camera dollies; the renderer feathers a soft dark alpha from
                    # the quad radius (mode "shadow" → DrawableObject::BlobShadow), so
                    # the mesh reads as resting on the court rather than floating.
                    shadow_r = size * args.shadow_scale
                    m_sh = np.eye(4)
                    m_sh[:3, 0] = e1 * shadow_r
                    m_sh[:3, 1] = e2 * shadow_r
                    m_sh[:3, 2] = e3 * shadow_r  # flat quad (local z=0); keeps M invertible
                    m_sh[:3, 3] = anchor
                    shadow_model = C4 @ m_sh
                    draws.insert(0, {
                        "mesh": args.mesh_index,
                        "model": colmajor(shadow_model),
                        "mode": "shadow",
                    })
        if args.placement_quad:
            # Draw the reconstructed placement quad itself as an overlay so it can be
            # checked against the filmed poster. A canonical origin-centred, extent-2
            # square mesh (corners ±1) has an identity preview base, so this model
            # alone maps its ±1 corners onto the camera-space quad corners
            # `a·r1 + b·r2 + t` (unit-square (a,b)), i.e. exactly the poster
            # (H = K·[r1 r2 t] reprojects onto it).
            nrm = _n(np.cross(r1, r2))  # plane normal (z=0 column: keeps the model invertible)
            m_quad = np.eye(4)
            m_quad[:3, 0] = r1 / 2.0
            m_quad[:3, 1] = r2 / 2.0
            # Scale the normal to the in-plane half-edge (|r1/2|) so the --axes-local
            # gizmo's blue Z arm reads the same length as the red X arm instead of the
            # raw unit normal (which drew ~2× longer). The quad mesh is flat (z=0), so
            # this only sizes the gizmo arm — the drawn quad outline is unchanged.
            m_quad[:3, 2] = nrm * 0.5
            m_quad[:3, 3] = 0.5 * (r1 + r2) + t
            quad_model = C4 @ m_quad
            draws.append({
                "mesh": args.placement_quad_mesh_index,
                "model": colmajor(quad_model),
                "mode": args.placement_quad_mode,
            })
        if not draws:
            continue
        last_render_k = render_k(K)
        row = {
            "k": last_render_k,
            "draws": draws,
            **frame_ref,
        }
        out.write(json.dumps(row) + "\n")
        written += 1
        placed_i += 1
    if out is not sys.stdout:
        out.close()
    dest = "stdout" if args.output == "-" else args.output
    print(f"wrote {written} single-view (#77, Pose-free) placements to {dest} "
          f"({args.width}×{args.height}); {placed_i} placed + "
          f"{written - placed_i} background-only frame(s)", file=sys.stderr)


# --------------------------------------------------------------------------- #
# Optional cross-check vs. the multi-view ground truth (uses Pose.txt).
# --------------------------------------------------------------------------- #
def ground_truth_basis(corners_world, R, t):
    """Ground-truth (triangulated) frame in the OpenCV camera E³ of a frame."""
    Xc = (R @ corners_world.T).T + t  # (4,3) camera-frame corners
    o = Xc.mean(axis=0)
    ez = _n(np.cross(Xc[1] - Xc[0], Xc[3] - Xc[0]))
    if np.dot(ez, o) > 0:  # camera at origin, plane in +Z: normal points back to it
        ez = -ez
    ex = _n(Xc[1] - Xc[0])
    ex = _n(ex - np.dot(ex, ez) * ez)
    ey = np.cross(ez, ex)
    return o, np.array([ex, ey, ez]), Xc


def run_validation(K, quads, args):
    """Compare #77's single-view basis to the multi-view ground truth per frame."""
    import cornellbox_gt as gt  # lazy: only the --validate path touches Pose.txt

    poses = gt.parse_poses(os.path.join(args.assets, "Pose.txt"))
    n = min(len(poses), len(quads))
    poses = poses[:n]

    corners = np.array([gt.triangulate_corner(K, poses, quads, j)[0] for j in range(4)])
    world_center = corners.mean(axis=0)
    print(f"triangulated world center = "
          f"({world_center[0]:.3f}, {world_center[1]:.3f}, {world_center[2]:.3f})\n")

    print(f"{'frame':>5} | {'e3 err':>7} | {'e1 err':>7} | {'RH det':>7} | "
          f"{'orig dir':>8} | {'quad reproj':>11} | world origin (via GT pose)")
    print("-" * 96)
    prev_z, jit = None, []
    for f in args.frames_list:
        if f >= n:
            continue
        R, t = poses[f]
        nb = normal_basis_from_quad(quads[f], K)
        if nb is None:
            print(f"{f:>5} |  degenerate quad → None")
            continue
        o3d, e, _o_px, _lam = nb
        o_gt, e_gt, _Xc = ground_truth_basis(corners, R, t)

        e3_err = ang(e[2], e_gt[2])
        e1_err = ang(e[0], e_gt[0])
        det = float(np.linalg.det(e))
        orig_dir = ang(o3d, o_gt)
        r1q, r2q, tq = pose_from_quad(quads[f], K)
        reproj = 0.0
        for (a, b), obs in zip([(0, 0), (1, 0), (1, 1), (0, 1)], quads[f]):
            X = a * r1q + b * r2q + tq
            p = K @ X
            reproj = max(reproj, float(np.hypot(*(p[:2] / p[2] - obs))))
        world_o = R.T @ (o3d - t)  # GT lift for reporting only
        if prev_z is not None:
            jit.append(ang(e[2], prev_z))
        prev_z = e[2]
        print(f"{f:>5} | {e3_err:6.3f}° | {e1_err:6.3f}° | {det:+.4f} | "
              f"{orig_dir:7.3f}° | {reproj:8.2e}px | "
              f"({world_o[0]:+.3f}, {world_o[1]:+.3f}, {world_o[2]:+.3f})")
    if jit:
        print(f"\ne3 frame-to-frame jitter (deg): mean {np.mean(jit):.3f}, max {np.max(jit):.3f} "
              f"(spec warns e3 is a sensitive nonlinear fn of the quad)")


def main():
    ap = argparse.ArgumentParser(
        description="Find a placement quad's local frame from K + quad points and place a mesh (Pose-free, #77).")
    ap.add_argument("--assets", default="assets/videos/cornellbox",
                    help="dir with K.txt / QuadImagePoints.txt for the sample clip (Pose.txt only for "
                         "--validate). Ignored when --from-perception is given.")
    ap.add_argument("--frame-ext", default="jpg", help="still extension (png|jpg)")
    ap.add_argument("--frame-rel", default="frames",
                    help="frame_path prefix relative to --frames-base (default: frames)")
    ap.add_argument("--inline-frames", action="store_true",
                    help="emit frame_id parsed from each frame_NNNNNN path instead of "
                         "frame_path; pair with scripts/extract_frames.py --embed")
    ap.add_argument("--width", type=int, default=960, help="render width")
    ap.add_argument("--height", type=int, default=540, help="render height")
    ap.add_argument("--src-width", type=int, default=DEFAULT_SRC_W,
                    help=f"resolution K/quad points are authored for (default: {DEFAULT_SRC_W})")
    ap.add_argument("--src-height", type=int, default=DEFAULT_SRC_H,
                    help=f"resolution K/quad points are authored for (default: {DEFAULT_SRC_H})")
    ap.add_argument("--step", type=int, default=2, help="use every Nth frame (fixture path only)")
    ap.add_argument("--limit", type=int, default=None, help="cap number of emitted frames")
    ap.add_argument("--size-factor", type=float, default=1.0,
                    help="mesh size vs. the reconstructed quad edge")
    ap.add_argument("--turns", type=float, default=1.0, help="mesh spins this many turns over the clip")
    ap.add_argument("--lift", type=float, default=1.0,
                    help="fraction of the mesh half-extent lifted along the plane normal (feet ≈ 1.0)")
    ap.add_argument("--place-offset-e1", type=float, default=0.0,
                    help="shift the placed mesh in the placement-quad plane along the quad's "
                         "RED gizmo axis (r1 / local X), in quad half-edge units: +1.0 ≈ the "
                         "quad edge. Matches the --axes-local red arm; stays in the P² local "
                         "frame. Default 0 (quad centre).")
    ap.add_argument("--place-offset-e2", type=float, default=0.0,
                    help="shift the placed mesh in the placement-quad plane along the quad's "
                         "GREEN gizmo axis (r2 / local Y), in quad half-edge units. Matches the "
                         "--axes-local green arm; e.g. a negative value moves the mesh down-court "
                         "(−green), off the mid-court action. Default 0 (quad centre).")
    ap.add_argument("--place-offset", type=float, nargs=2, action="append",
                    metavar=("E1", "E2"), default=None,
                    help="place a mesh copy at this (e1, e2) in-plane offset (quad "
                         "half-edge units, same axes as --place-offset-e1/e2). Repeat "
                         "for a row of cans, e.g. --place-offset 1.4 -1.7 --place-offset "
                         "2.1 -1.5 --place-offset 2.8 -1.3. When given, overrides the "
                         "single --place-offset-e1/e2. Each copy gets its own shadow.")
    ap.add_argument("--copy-phase", type=float, default=0.0,
                    help="per-copy spin phase offset (turns) between successive "
                         "--place-offset cans, so a row doesn't look mirror-identical "
                         "(default 0: all copies share the same spin phase).")
    ap.add_argument("--place-mesh", action=argparse.BooleanOptionalAction, default=True,
                    help="anchor the model mesh on the placement-quad frame (stage 2). "
                         "Use --no-place-mesh for stage 1 (placement quad only, before placing the mesh).")
    ap.add_argument("--mesh-index", type=int, default=0,
                    help="mesh-table row index of the placed model mesh (default 0)")
    ap.add_argument("--shadow", action="store_true",
                    help="lay a soft contact / grounding blob shadow on the placement-quad "
                         "plane under the placed mesh (a mode:\"shadow\" draw), so the mesh "
                         "reads as sitting on the recovered floor instead of floating over "
                         "the composited video plate. Requires --place-mesh; stays in the P² "
                         "local frame (spans the quad's r1/r2 axes at the mesh anchor).")
    ap.add_argument("--shadow-scale", type=float, default=1.8,
                    help="blob-shadow radius as a multiple of the placed mesh's half-extent "
                         "(default 1.8: a soft grounding shadow spreading a little past the "
                         "mesh footprint). Larger = a wider, softer shadow.")
    ap.add_argument("--placement-quad", action="store_true",
                    help="emit a per-frame draw of the reconstructed placement quad itself as a "
                         "wireframe overlay (visualizes the local frame / a check that it matches the "
                         "filmed poster). Pair with render.sh --placement-quad, which adds the canonical "
                         "quad mesh.")
    ap.add_argument("--placement-quad-mesh-index", type=int, default=1,
                    help="mesh-table row index of the canonical placement-quad mesh "
                         "(default 1: mesh=0, quad=1; use 0 for stage 1 where the quad is the only mesh)")
    ap.add_argument("--placement-quad-mode", default="wireframe",
                    choices=["wireframe", "filled", "textured"],
                    help="render mode for the placement-quad overlay draw (default: wireframe outline)")
    ap.add_argument("-o", "--output", default="-", help="output JSONL path (default: stdout)")
    ap.add_argument("--from-perception", metavar="ARROW", default=None,
                    help="read the upstream perception Arrow stream (K + placement_quad + frame_path, "
                         "from scripts/perception_to_arrow.py) instead of the K.txt/QuadImagePoints.txt "
                         "fixture. This makes the script the *downstream* stage consuming that stream.")
    ap.add_argument("--validate", action="store_true",
                    help="cross-check the single-view basis vs. multi-view ground truth (reads Pose.txt)")
    ap.add_argument("--frames-list", type=int, nargs="+", default=[0, 60, 125, 190, 249],
                    help="frames to report in --validate")
    args = ap.parse_args()

    if args.from_perception:
        # Downstream stage: consume the upstream perception Arrow stream directly.
        records = read_perception_records(args.from_perception)
        print(f"read {len(records)} perception rows from {args.from_perception}", file=sys.stderr)
        emit_records(records, args)
        return

    K = parse_k(os.path.join(args.assets, "K.txt"))
    quads = parse_quads(os.path.join(args.assets, "QuadImagePoints.txt"))
    print(f"parsed {len(quads)} quads; K fx={K[0,0]:.1f} fy={K[1,1]:.1f} "
          f"cx={K[0,2]:.1f} cy={K[1,2]:.1f}", file=sys.stderr)

    if args.validate:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        run_validation(K, quads, args)
    else:
        emit_placement_jsonl(K, quads, args)


if __name__ == "__main__":
    main()
