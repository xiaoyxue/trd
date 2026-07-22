#!/usr/bin/env python3
"""Author the **cornellbox AR composite** — GROUND TRUTH (GT) placement.

This is the *reference* placement for the cornellbox capstone: it triangulates
the floor quad's world corners from **every frame's ``K`` and ``Pose``** (a
multi-view DLT over the whole clip) and anchors the bunny there. Because it uses
the true extrinsics it is exact, and we treat it as the **ground truth** against
which the single-view method (`placement_quad_by_local_coord.py`, #77 — the method we
actually ship, which needs only ``K`` + the quad) is validated.

This is the real-video capstone for the frame-compositing pipeline (#62 / #63):
a **textured bunny anchored in a filmed scene** while the *real* camera dollies
around it. Each output frame carries

* a **background still** (``frame_path``) — one decoded frame of
  ``assets/videos/cornellbox/CameraMovement.mp4`` (extract with
  ``scripts/extract_frames.py``), composited *beneath* the scene by a
  ``FramePlane`` (#63); and
* a **CV camera** (``k`` + ``pose``) taken straight from the shoot's OpenCV
  intrinsics (``K.txt``) and extrinsics (``Pose.txt``), so the rendered bunny is
  seen from the *same* viewpoint as the plate; and
* a **textured bunny** placed on the marked floor quad and slowly spun.

Where the bunny goes (GROUND TRUTH via multi-view triangulation).
``QuadImagePoints.txt`` gives, per frame, the 2-D image projections of the *same*
four world points (a fixed floor quad). Because we have ``K`` and every frame's
``[R|t]``, we **triangulate** those four world corners once (multi-view DLT over
the whole clip), then sit the bunny at the quad centre, oriented to the quad's
plane normal, scaled to the quad, and spun about the normal. This uses the true
``Pose`` and so is exact — the yardstick for the Pose-free #77 method.

Coordinate conversion (OpenCV → trd/OpenGL)
-------------------------------------------
``Pose.txt`` is **world-to-camera** OpenCV extrinsics (``X_cam = R·X_world + t``)
in the OpenCV camera frame (**+X right, +Y down, +Z forward**). trd's
``Camera::from_cv`` wants a ``pose`` that is **world-from-camera** (camera-to-
world) in the OpenGL camera frame (**+X right, +Y up, +Z backward**; the view is
its inverse). We convert once per frame:

* camera position in world:  ``c = -Rᵀ·t``  (same in both frames);
* the OpenGL camera basis in world is the OpenCV camera basis with the **Y and Z
  axes negated** (down→up, forward→backward). The OpenCV camera axes in world are
  the *rows* of ``R`` (columns of ``Rᵀ``), so
  ``pose = [ R₀ | −R₁ | −R₂ | c ]`` (columns; bottom row ``[0,0,0,1]``).

``K`` is in **pixels** and therefore resolution-specific; it is authored for
1920×1080 and scaled linearly to the render resolution (``fx,cx`` by ``W/1920``,
``fy,cy`` by ``H/1080``) so the projected bunny lines up with the stretched
background plate.

Run (from ``nix develop``)::

    uv run --with numpy examples/cornellbox_gt.py \
        --assets assets/videos/cornellbox \
        --frames output/cornellbox/frames --frame-ext jpg \
        --width 960 --height 540 --step 2 \
        -o examples/frames.cornellbox_gt.jsonl

then render the composite GIF via trd-cli (wrap with nixGL on native GPU boxes)::

    examples/render.sh --cli \
        --mesh assets/meshes/bunny_with_texture/bunny.obj \
        --texture assets/meshes/bunny_with_texture/bunny_uv_map1.jpg \
        --frames-base output/cornellbox \
        examples/frames.cornellbox_gt.jsonl output/cornellbox_gt.gif 960 540 25
"""
import argparse
import json
import os
import re
import sys

import numpy as np

SRC_W, SRC_H = 1920, 1080  # resolution K.txt is authored for


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


def parse_poses(path):
    """Parse per-frame world-to-camera ``[R|t]`` blocks from ``Pose.txt``.

    Returns a list of ``(R (3×3), t (3,))`` numpy tuples, one per frame.
    """
    poses = []
    with open(path, encoding="utf-8") as fh:
        lines = [ln.rstrip("\n") for ln in fh]
    i = 0
    while i < len(lines):
        s = lines[i].strip()
        if s.startswith("frame "):
            block = []
            for j in range(i + 1, i + 4):
                block.append([float(x) for x in lines[j].split()])
            rt = np.array(block, dtype=np.float64)  # 3×4
            poses.append((rt[:, :3].copy(), rt[:, 3].copy()))
            i += 4
        else:
            i += 1
    return poses


def parse_quads(path):
    """Parse per-frame 4 image-space quad points from ``QuadImagePoints.txt``.

    Returns a list of ``(4, 2)`` numpy arrays, one per frame.
    """
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


def triangulate_corner(k, poses, quads, corner, stride=8):
    """Multi-view DLT triangulation of one fixed world ``corner``.

    Uses every ``stride``-th frame's projection ``P = K·[R|t]`` and its observed
    image point for this corner. Returns the world point (3,) and the RMS
    reprojection error in pixels.
    """
    A = []
    Ps = []
    obs = []
    for f in range(0, len(poses), stride):
        R, t = poses[f]
        P = k @ np.hstack([R, t.reshape(3, 1)])  # 3×4
        u, v = quads[f][corner]
        A.append(u * P[2, :] - P[0, :])
        A.append(v * P[2, :] - P[1, :])
        Ps.append(P)
        obs.append((u, v))
    A = np.array(A)
    _, _, vt = np.linalg.svd(A)
    Xh = vt[-1]
    X = Xh[:3] / Xh[3]
    # RMS reprojection error over the frames used.
    errs = []
    for P, (u, v) in zip(Ps, obs):
        p = P @ np.append(X, 1.0)
        errs.append(np.hypot(p[0] / p[2] - u, p[1] / p[2] - v))
    return X, float(np.sqrt(np.mean(np.square(errs))))


def normalize(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v


def rotate_y(theta):
    """Column-major-friendly 4×4 rotate about local +Y (glam ``from_rotation_y``)."""
    c, s = np.cos(theta), np.sin(theta)
    return np.array(
        [
            [c, 0.0, s, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [-s, 0.0, c, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )


def pose_gl(R, t):
    """OpenCV world-to-camera ``[R|t]`` → trd GL camera-to-world 4×4 ``pose``.

    Columns are the OpenGL camera axes in world (right, up, backward) + the
    camera position: ``[R₀ | −R₁ | −R₂ | −Rᵀt]``.
    """
    c = -R.T @ t
    right = R[0, :]
    up = -R[1, :]
    back = -R[2, :]
    m = np.eye(4)
    m[:3, 0] = right
    m[:3, 1] = up
    m[:3, 2] = back
    m[:3, 3] = c
    return m


def colmajor(m):
    """Flatten a 4×4 numpy matrix to a 16-float column-major list (glam layout)."""
    return [float(x) for x in m.flatten(order="F")]


def main():
    ap = argparse.ArgumentParser(description="Author the cornellbox AR composite JSONL (0.0.5).")
    ap.add_argument("--assets", default="assets/videos/cornellbox",
                    help="dir with K.txt / Pose.txt / QuadImagePoints.txt")
    ap.add_argument("--frames", default="output/cornellbox/frames",
                    help="dir of extracted stills (for existence checks + naming)")
    ap.add_argument("--frame-ext", default="jpg", help="still extension (png|jpg)")
    ap.add_argument("--frame-rel", default="frames",
                    help="frame_path prefix relative to --frames-base (default: frames)")
    ap.add_argument("--width", type=int, default=960, help="render width")
    ap.add_argument("--height", type=int, default=540, help="render height")
    ap.add_argument("--step", type=int, default=2, help="use every Nth frame")
    ap.add_argument("--limit", type=int, default=None, help="cap number of emitted frames")
    ap.add_argument("--scale", type=float, default=1.0,
                    help="bunny size vs. quad edge (1.0 = largest extent ≈ quad edge)")
    ap.add_argument("--turns", type=float, default=1.0, help="bunny spins this many turns over the clip")
    ap.add_argument("--lift", type=float, default=1.0,
                    help="fraction of the bunny half-extent lifted along the quad normal (feet on plane ≈ 1.0)")
    ap.add_argument("-o", "--output", default="-", help="output JSONL path (default: stdout)")
    args = ap.parse_args()

    k = parse_k(os.path.join(args.assets, "K.txt"))
    poses = parse_poses(os.path.join(args.assets, "Pose.txt"))
    quads = parse_quads(os.path.join(args.assets, "QuadImagePoints.txt"))
    n = min(len(poses), len(quads))
    if n == 0:
        raise SystemExit("error: no poses/quads parsed")
    poses, quads = poses[:n], quads[:n]
    print(f"parsed {n} frames; K fx={k[0,0]:.1f} fy={k[1,1]:.1f} cx={k[0,2]:.1f} cy={k[1,2]:.1f}",
          file=sys.stderr)

    # Triangulate the four fixed world corners of the floor quad (multi-view DLT).
    corners = np.zeros((4, 3))
    for j in range(4):
        X, rms = triangulate_corner(k, poses, quads, j)
        corners[j] = X
        print(f"corner {j}: world=({X[0]:.3f}, {X[1]:.3f}, {X[2]:.3f})  reproj_rms={rms:.2f}px",
              file=sys.stderr)

    center = corners.mean(axis=0)
    edges = [np.linalg.norm(corners[(j + 1) % 4] - corners[j]) for j in range(4)]
    edge_len = float(np.mean(edges))
    normal = normalize(np.cross(corners[1] - corners[0], corners[3] - corners[0]))
    # Orient the normal toward the cameras (average camera position).
    cams = np.array([-(R.T @ t) for (R, t) in poses])
    if np.dot(normal, cams.mean(axis=0) - center) < 0:
        normal = -normal
    # Local basis: X along the first quad edge (perp. to normal), Y = normal, Z = X×Y.
    x_axis = normalize(corners[1] - corners[0])
    x_axis = normalize(x_axis - np.dot(x_axis, normal) * normal)
    z_axis = np.cross(x_axis, normal)
    r_place = np.eye(4)
    r_place[:3, 0] = x_axis
    r_place[:3, 1] = normal
    r_place[:3, 2] = z_axis

    scale = 0.5 * edge_len * args.scale  # bunny largest extent ≈ edge_len·scale
    lift = args.lift * scale
    print(f"quad: center=({center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f}) "
          f"edge≈{edge_len:.3f} normal=({normal[0]:.2f}, {normal[1]:.2f}, {normal[2]:.2f}) "
          f"bunny_scale={scale:.3f}", file=sys.stderr)

    trans = np.eye(4)
    trans[:3, 3] = center + normal * lift
    s_mat = np.diag([scale, scale, scale, 1.0])

    # K scaled to the render resolution (K is pixel-valued → resolution-specific).
    sx, sy = args.width / SRC_W, args.height / SRC_H
    k_render = [
        k[0, 0] * sx, 0.0, 0.0,
        k[0, 1] * sx, k[1, 1] * sy, 0.0,
        k[0, 2] * sx, k[1, 2] * sy, 1.0,
    ]

    indices = list(range(0, n, args.step))
    if args.limit is not None:
        indices = indices[: args.limit]

    out = sys.stdout if args.output == "-" else open(args.output, "w", encoding="utf-8")
    written = 0
    for out_i, f in enumerate(indices):
        R, t = poses[f]
        pose = pose_gl(R, t)
        theta = 2.0 * np.pi * args.turns * (out_i / max(1, len(indices)))
        model = trans @ r_place @ rotate_y(theta) @ s_mat
        rel = f"{args.frame_rel}/frame_{f:06d}.{args.frame_ext}"
        row = {
            "k": [float(x) for x in k_render],
            "pose": colmajor(pose),
            "draws": [{"mesh": 0, "model": colmajor(model)}],
            "frame_path": rel,
        }
        out.write(json.dumps(row) + "\n")
        written += 1
    if out is not sys.stdout:
        out.close()

    dest = "stdout" if args.output == "-" else args.output
    print(f"wrote {written} frames to {dest} ({args.width}×{args.height}, step {args.step})",
          file=sys.stderr)


if __name__ == "__main__":
    main()
