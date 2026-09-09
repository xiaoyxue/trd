#!/usr/bin/env python3
"""Direct placement with raw directions and all axis lengths equal to |0.5*r1|.

Read current params-only FHC input and write comparison matrices/vertices, not
an application document. Keep the original placement reference unchanged.
"""

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pyarrow as pa

ROOT = Path(__file__).resolve().parents[4]
CV_TO_GL = np.diag([1.0, -1.0, -1.0, 1.0])


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load reference module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def expand(origin, r1, r2, e3, coordinates):
    u, v, w = coordinates
    length = np.linalg.norm(0.5 * r1)
    return (origin + u * 0.5 * r1 + v * (length / np.linalg.norm(r2)) * r2
            + w * (length / np.linalg.norm(e3)) * e3)


def project(intrinsics, points):
    homogeneous = np.asarray(points) @ intrinsics.T
    if not np.isfinite(homogeneous).all() or (homogeneous[..., 2] <= 0).any():
        raise ValueError("comparison vertices must project finitely in front of the camera")
    return homogeneous[..., :2] / homogeneous[..., 2:3]


def pixel_geometry(row):
    k = row["k"]
    if not isinstance(k, dict):
        raise ValueError("expected the current FHC intrinsics struct")
    values = np.array([k[name] for name in ("fx", "fy", "cx", "cy", "skew", "w", "h")])
    if not np.isfinite(values).all() or any(k[name] <= 0 for name in ("fx", "fy", "w", "h")):
        raise ValueError("camera dimensions/focal lengths must be positive and finite")
    if any(k[name] != int(k[name]) for name in ("w", "h")):
        raise ValueError("camera dimensions must be integers")
    intrinsics = np.array([
        [k["fx"] / 2, k["skew"] / 2, (k["cx"] + k["w"]) / 2],
        [0.0, k["fy"] / 2, (k["cy"] + k["h"]) / 2],
        [0.0, 0.0, 1.0],
    ])
    quads = row["bottom_quads"]
    if not isinstance(quads, list) or len(quads) != 1 or len(quads[0]) != 4:
        raise ValueError("this experiment requires exactly one four-corner quad per row")
    if any(point["w"] != k["w"] or point["h"] != k["h"] for point in quads[0]):
        raise ValueError("quad and camera must use the same FHC dimensions")
    quad = np.array([
        [(point["x"] + point["w"]) / 2, (point["y"] + point["h"]) / 2]
        for point in quads[0]
    ])
    if not np.isfinite(quad).all():
        raise ValueError("quad coordinates must be finite")
    edges = np.roll(quad, -1, axis=0) - quad
    turns = edges[:, 0] * np.roll(edges[:, 1], -1) - edges[:, 1] * np.roll(edges[:, 0], -1)
    if not (turns > 0).all():
        raise ValueError("quad corners must be clockwise, convex and nondegenerate")
    return intrinsics, quad, int(k["w"]), int(k["h"])


def comparison_row(reference, row):
    frame = row["video_frame_index"]
    if frame is None or frame < 0:
        raise ValueError("an explicit nonnegative video_frame_index is required")
    if "tracked" in row and row["tracked"] is not True:
        raise ValueError(f"source frame {frame} is not tracked")
    for name in ("model", "draw_model"):
        if name in row:
            matrices = row[name]
            if matrices is None or len(matrices) != 1:
                raise ValueError(f"source frame {frame}: expected one identity {name}")
            matrix = np.asarray(matrices[0], dtype=np.float64)
            if matrix.size != 16 or not np.array_equal(matrix.reshape(4, 4), np.eye(4)):
                raise ValueError(f"source frame {frame}: this comparison requires identity {name}")
    intrinsics, quad, width, height = pixel_geometry(row)
    normal_frame = reference.normal_basis_from_quad(quad, intrinsics)
    if normal_frame is None:
        raise ValueError(f"source frame {frame} has no finite plane frame")
    origin, orthogonal_axes, _, axis_length = normal_frame
    r1, r2, _ = reference.pose_from_quad(quad, intrinsics)
    e3 = orthogonal_axes[2]
    raw = np.eye(4)
    raw[:3, :3] = np.column_stack((r1, r2, e3))
    raw[:3, 3] = origin
    determinant = float(np.linalg.det(raw[:3, :3]))
    if not np.isfinite(raw).all() or not np.isfinite(determinant) or abs(determinant) <= 1e-12:
        raise ValueError(f"source frame {frame} has a singular direct basis")

    # Mesh +Y is height; account for the displayed triad's normal-flip sign once.
    handedness = 1.0 if determinant > 0 else -1.0
    mesh_to_coefficients = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, -handedness, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ])
    lengths = np.linalg.norm(raw[:3, :3], axis=0)
    unit_length = 0.5 * lengths[0]
    equal_lengths = np.diag([0.5, unit_length / lengths[1], unit_length / lengths[2], 1.0])
    scaled_basis = CV_TO_GL @ raw @ equal_lengths
    placement = scaled_basis @ mesh_to_coefficients
    axis = CV_TO_GL @ raw @ np.diag([0.5, 0.5, axis_length, 1.0])
    grounding = np.eye(4)
    grounding[1, 3] = 0.5
    cube = placement @ grounding
    coefficients = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
                             [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
    vertices = np.array([expand(origin, r1, r2, e3, point) for point in coefficients])
    unit_axes = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
                          [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    return {
        "video_frame_index": frame,
        "present_index": row.get("present_index"),
        "width": width,
        "height": height,
        "k_row_major": intrinsics.flatten().tolist(),
        "quad": quad.tolist(),
        "origin_camera": origin.tolist(),
        "origin_px": project(intrinsics, origin).tolist(),
        "r1": r1.tolist(),
        "r2": r2.tolist(),
        "e3": e3.tolist(),
        "axis_length": float(axis_length),
        "placement_axis_length": float(unit_length),
        "handedness": handedness,
        "raw_basis_model": reference.colmajor(CV_TO_GL @ raw),
        "scaled_basis_model": reference.colmajor(scaled_basis),
        "mesh_to_coefficients": reference.colmajor(mesh_to_coefficients),
        "axis_model": reference.colmajor(axis),
        "placement_model": reference.colmajor(placement),
        "cube_model": reference.colmajor(cube),
        "unit_quad_coefficients": coefficients.tolist(),
        "unit_quad_camera": vertices.tolist(),
        "unit_quad_pixels": project(intrinsics, vertices).tolist(),
        "unit_axes_coefficients": unit_axes.tolist(),
        "unit_axes_camera": [
            expand(origin, r1, r2, e3, point).tolist() for point in unit_axes
        ],
        "unit_axes_pixels": project(intrinsics, [
            expand(origin, r1, r2, e3, point) for point in unit_axes
        ]).tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("params", type=Path)
    parser.add_argument("-o", "--output", type=Path, required=True)
    args = parser.parse_args()
    if args.params.resolve() == args.output.resolve():
        raise ValueError("output must not overwrite the supplied params")
    reference = load_module(
        "original_placement_reference", ROOT / "examples" / "placement_quad_by_local_coord.py"
    )
    version = load_module("scene_protocol_version", ROOT / "scripts" / "protocol_version.py")
    with args.params.open("rb") as source:
        table = pa.ipc.open_stream(source).read_all()
        if source.read(1):
            raise ValueError("use the params-only input, not an appended mesh bundle")
    metadata = table.schema.metadata or {}
    if metadata.get(b"trd.protocol.version") != version.PROTOCOL_VERSION.encode():
        raise ValueError(f"expected protocol {version.PROTOCOL_VERSION} params")
    if metadata.get(b"trd.table.kind") != b"params":
        raise ValueError("expected a params table")
    if len(set(table.column_names)) != len(table.column_names):
        raise ValueError("source columns must have unique names")
    required = {"video_frame_index", "k", "bottom_quads"}
    if missing := required - set(table.column_names):
        raise ValueError(f"missing required columns: {sorted(missing)}")
    if table.num_rows == 0:
        raise ValueError("source params must contain tracked rows")
    rows = [comparison_row(reference, row) for row in table.to_pylist()]
    indices = [row["video_frame_index"] for row in rows]
    if any(right <= left for left, right in zip(indices, indices[1:])):
        raise ValueError("source frame identities must be strictly increasing")
    result = {
        "formula": "P = O + u*0.5*r1 + v*(L/|r2|)*r2 + w*(L/|e3|)*e3; L = |0.5*r1|",
        "matrix_layout": "column-major GL camera models; k_row_major is pixel OpenCV K",
        "source_name": metadata.get(b"trd.video.source_name", b"").decode(),
        "source_sha256": metadata.get(b"trd.video.sha256", b"").decode(),
        "rows": rows,
    }
    encoded = json.dumps(result, indent=2, allow_nan=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(encoded + "\n", encoding="utf-8")
    print(f"wrote {len(rows)} direct-basis Python placement rows to {args.output}")


if __name__ == "__main__":
    main()
