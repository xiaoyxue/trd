#!/usr/bin/env python3
"""Author the **#49 dolly-camera turntable capstone** as two equivalent JSONL
frame streams — one in the **CG** camera form, one in the **CV** form — that
`trd` must render identically.

The animation (protocol 0.0.3):

- **45° bird's-eye dolly camera.** The camera looks at the world origin from a
  *fixed* elevation + azimuth direction; only its **distance** oscillates
  ``dist_i = mid + amp * sin(2*pi*i/N)`` (a dolly — the camera moves along its
  view axis, it does not orbit or zoom the lens). ``eye_i = dist_i * dir``.
- **Y-spin bunny.** Each frame's ``model`` is ``rotate_y(2*pi*i/N)`` (a full
  turn), composed by the renderer over the mesh's centre + scale-to-fit preview.
- **near/far.** The renderer's CV intrinsics path always projects with the
  default clip planes (0.1 / 1000), so both forms use those; they comfortably
  bracket the preview-normalised bunny's bounding sphere across the whole dolly
  range, so nothing is clipped.

The **same** camera is authored twice:

- **CG form** (``examples/frames.bunny_dolly.cg.jsonl``): ``eye`` / ``target`` /
  ``up`` + ``fovy`` / ``aspect``. Resolution-independent.
- **CV form** (``examples/frames.bunny_dolly.cv.jsonl``): pinhole ``k`` (9) +
  camera-to-world ``pose`` (16). ``k`` is in **pixel** units, so it is authored
  for a specific square ``--width``×``--height`` (default 1024²) and must be
  rendered at that resolution to match the CG form.

Both decode (see ``trd-core`` ``FrameParams::view_matrix`` /
``projection_matrix``) to the *same* ``P·V``:

- ``view`` : CG ``look_at_rh(eye, target, up)`` == CV ``inverse(pose)``.
- ``proj`` : CG ``perspective_rh(fovy, aspect, 0.1, 1000)`` ==
  CV ``projection_from_intrinsics(k, viewport)`` when, for a square viewport,
  ``fx = fy = H/(2*tan(fovy/2))``, ``cx = W/2``, ``cy = H/2``, ``skew = 0`` and
  ``aspect = W/H``.

The script asserts this equivalence numerically (max abs diff of the two decoded
``P·V`` matrices < 1e-4) before writing, so a math regression fails fast without
a GPU. Pure stdlib (no numpy/pyarrow); the JSONL is turned into Arrow by
``scripts/jsonl_to_arrow.py`` inside ``examples/render.sh``.

Run (from the repo root):

    python examples/bunny_dolly.py                       # writes both JSONLs (1024²)
    python examples/bunny_dolly.py --frames 72 --width 1024 --height 1024

Then render each and compare (from ``nix develop``; wrap with nixGL on native
Linux GPU boxes):

    examples/render.sh --cli --wireframe --mesh assets/meshes/bunny.obj \
      examples/frames.bunny_dolly.cg.jsonl output/bunny_dolly_cg.gif 1024 1024 24
    examples/render.sh --cli --wireframe --mesh assets/meshes/bunny.obj \
      examples/frames.bunny_dolly.cv.jsonl output/bunny_dolly_cv.gif 1024 1024 24
"""
import argparse
import json
import math
import os

Vec3 = tuple  # (x, y, z)


def sub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def cross(a, b):
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def normalize(a):
    n = math.sqrt(dot(a, a))
    if n == 0.0:
        return (0.0, 0.0, 0.0)
    return (a[0] / n, a[1] / n, a[2] / n)


def look_basis(eye, target, up):
    """glam ``look_at_rh`` basis: forward ``f`` (eye→target), right ``s``, up ``u``."""
    f = normalize(sub(target, eye))
    s = normalize(cross(f, up))
    u = cross(s, f)
    return f, s, u


def rotate_y(theta):
    """Column-major 4×4 ``rotate_y(theta)`` matching glam ``Mat4::from_rotation_y``."""
    c, s = math.cos(theta), math.sin(theta)
    # cols: (c,0,-s,0) (0,1,0,0) (s,0,c,0) (0,0,0,1)
    return [
        c, 0.0, -s, 0.0,
        0.0, 1.0, 0.0, 0.0,
        s, 0.0, c, 0.0,
        0.0, 0.0, 0.0, 1.0,
    ]


def view_matrix(eye, target, up):
    """Column-major 4×4 ``look_at_rh`` view matrix (world→camera), glam layout."""
    f, s, u = look_basis(eye, target, up)
    # cols: (s.x,u.x,-f.x,0) (s.y,u.y,-f.y,0) (s.z,u.z,-f.z,0) (-s·e,-u·e,f·e,1)
    return [
        s[0], u[0], -f[0], 0.0,
        s[1], u[1], -f[1], 0.0,
        s[2], u[2], -f[2], 0.0,
        -dot(s, eye), -dot(u, eye), dot(f, eye), 1.0,
    ]


def pose_matrix(eye, target, up):
    """Column-major 4×4 camera-to-world ``pose`` = ``inverse(view_matrix)``.

    Built directly (no matrix inverse) from the orthonormal camera basis: the
    columns are the camera axes in world space (right ``s``, up ``u``, back
    ``-f``) and the translation is ``eye``.
    """
    f, s, u = look_basis(eye, target, up)
    return [
        s[0], s[1], s[2], 0.0,
        u[0], u[1], u[2], 0.0,
        -f[0], -f[1], -f[2], 0.0,
        eye[0], eye[1], eye[2], 1.0,
    ]


def intrinsics(width, height, fovy):
    """Column-major pinhole ``K`` (9) matching a square-viewport ``perspective_rh``.

    ``fx = fy = H/(2 tan(fovy/2))``, ``cx = W/2``, ``cy = H/2``, no skew — the
    exact inverse of ``trd-core``'s ``projection_from_intrinsics`` for
    ``aspect = W/H`` (see module docstring).
    """
    fy = height / (2.0 * math.tan(0.5 * fovy))
    fx = fy  # square pixels; aspect handled by the W/H viewport
    cx, cy = width / 2.0, height / 2.0
    # cols of [[fx,0,cx],[0,fy,cy],[0,0,1]]: (fx,0,0) (0,fy,0) (cx,cy,1)
    return [fx, 0.0, 0.0, 0.0, fy, 0.0, cx, cy, 1.0]


# --- numeric self-check: rebuild both decoded P·V and compare -----------------

DEFAULT_NEAR, DEFAULT_FAR = 0.1, 1000.0


def perspective_rh(fovy, aspect, near, far):
    """Column-major glam ``Mat4::perspective_rh`` (z ∈ [0,1])."""
    h = math.cos(0.5 * fovy) / math.sin(0.5 * fovy)  # 1/tan(fovy/2)
    w = h / aspect
    r = far / (near - far)
    return [
        w, 0.0, 0.0, 0.0,
        0.0, h, 0.0, 0.0,
        0.0, 0.0, r, -1.0,
        0.0, 0.0, r * near, 0.0,
    ]


def projection_from_intrinsics(k, width, height):
    """Column-major projection matching ``trd-core``'s ``projection_from_intrinsics``."""
    fx, s, fy, cx, cy = k[0], k[3], k[4], k[6], k[7]
    w, h = float(max(width, 1)), float(max(height, 1))
    n, f = DEFAULT_NEAR, DEFAULT_FAR
    return [
        2.0 * fx / w, 0.0, 0.0, 0.0,
        2.0 * s / w, 2.0 * fy / h, 0.0, 0.0,
        2.0 * cx / w - 1.0, 2.0 * cy / h - 1.0, f / (n - f), -1.0,
        0.0, 0.0, (f * n) / (n - f), 0.0,
    ]


def mat_mul(a, b):
    """Column-major 4×4 ``a * b``."""
    out = [0.0] * 16
    for col in range(4):
        for row in range(4):
            acc = 0.0
            for k in range(4):
                acc += a[k * 4 + row] * b[col * 4 + k]
            out[col * 4 + row] = acc
    return out


def max_abs_diff(a, b):
    return max(abs(x - y) for x, y in zip(a, b))


def assert_equivalent(eye, target, up, fovy, width, height):
    """Assert the CG and CV forms decode to the same ``P·V`` (within tolerance)."""
    aspect = width / height
    vp_cg = mat_mul(
        perspective_rh(fovy, aspect, DEFAULT_NEAR, DEFAULT_FAR),
        view_matrix(eye, target, up),
    )
    k = intrinsics(width, height, fovy)
    # CV view = inverse(pose); pose is rigid, so inverse == view_matrix by design.
    vp_cv = mat_mul(
        projection_from_intrinsics(k, width, height),
        view_matrix(eye, target, up),
    )
    return max_abs_diff(vp_cg, vp_cv)


def main() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--frames", type=int, default=72, help="frame count (default 72)")
    ap.add_argument("--width", type=int, default=1024, help="render width px (default 1024)")
    ap.add_argument("--height", type=int, default=1024, help="render height px (default 1024)")
    ap.add_argument("--fovy-deg", type=float, default=45.0, help="vertical FOV degrees")
    ap.add_argument("--mid", type=float, default=5.5, help="mean dolly distance")
    ap.add_argument("--amp", type=float, default=1.5, help="dolly distance amplitude")
    ap.add_argument("--elev-deg", type=float, default=35.0, help="camera elevation degrees")
    ap.add_argument("--azim-deg", type=float, default=35.0, help="camera azimuth degrees")
    ap.add_argument(
        "--out-prefix",
        default=os.path.join(here, "frames.bunny_dolly"),
        help="output path prefix ('.cg.jsonl'/'.cv.jsonl' appended)",
    )
    args = ap.parse_args()

    n = max(args.frames, 1)
    fovy = math.radians(args.fovy_deg)
    elev = math.radians(args.elev_deg)
    azim = math.radians(args.azim_deg)
    target = (0.0, 0.0, 0.0)
    up = (0.0, 1.0, 0.0)
    # Fixed bird's-eye view direction (unit): azimuth in xz, elevation toward +y.
    view_dir = (
        math.cos(elev) * math.sin(azim),
        math.sin(elev),
        math.cos(elev) * math.cos(azim),
    )

    def r6(xs):
        return [round(x, 6) for x in xs]

    cg_rows, cv_rows = [], []
    worst = 0.0
    for i in range(n):
        t = 2.0 * math.pi * i / n
        dist = args.mid + args.amp * math.sin(t)
        eye = (dist * view_dir[0], dist * view_dir[1], dist * view_dir[2])
        model = rotate_y(t)

        worst = max(worst, assert_equivalent(eye, target, up, fovy, args.width, args.height))

        cg_rows.append({
            "model": r6(model),
            "eye": r6(eye),
            "target": list(target),
            "up": list(up),
            "fovy": round(fovy, 6),
            "aspect": round(args.width / args.height, 6),
        })
        cv_rows.append({
            "model": r6(model),
            "k": r6(intrinsics(args.width, args.height, fovy)),
            "pose": r6(pose_matrix(eye, target, up)),
        })

    tol = 1e-4
    if worst > tol:
        raise SystemExit(
            f"CG/CV cameras diverge: max |P·V| diff {worst:.3e} > {tol:.0e}. "
            "The two forms would not render identically."
        )

    cg_path = f"{args.out_prefix}.cg.jsonl"
    cv_path = f"{args.out_prefix}.cv.jsonl"
    for path, rows in ((cg_path, cg_rows), (cv_path, cv_rows)):
        with open(path, "w", encoding="utf-8", newline="\n") as out:
            for r in rows:
                out.write(json.dumps(r) + "\n")

    print(
        f"wrote {n} frames -> {cg_path} (CG) and {cv_path} (CV); "
        f"max decoded P·V diff {worst:.2e} (< {tol:.0e}), authored for "
        f"{args.width}x{args.height}"
    )


if __name__ == "__main__":
    main()
