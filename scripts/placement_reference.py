#!/usr/bin/env python3
"""Generate placement parity expectations by calling the existing Python reference.

Read the ORIGINAL pixel-K/quad timeline, not the converted params. This keeps
the expected basis independent of the converter and Rust placement code.
"""

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pyarrow as pa

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_EXTENT = 1.0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("timeline", type=Path)
    parser.add_argument("-o", "--output", type=Path, required=True)
    args = parser.parse_args()
    path = ROOT / "examples" / "placement_quad_by_local_coord.py"
    spec = importlib.util.spec_from_file_location("placement_reference", path)
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    with args.timeline.open("rb") as source:
        table = pa.ipc.open_stream(source).read_all()
    metadata = table.schema.metadata
    if metadata.get(b"trd.video_edit.version") != b"0.2.0":
        raise ValueError("reference input must be the original video-edit 0.2.0 timeline")
    width = int(metadata[b"trd.video.width"])
    height = int(metadata[b"trd.video.height"])
    converted = []
    cv_to_gl = np.diag([1.0, -1.0, -1.0, 1.0])
    for row in table.to_pylist():
        k = np.asarray(row["k"], dtype=np.float64).reshape((3, 3))
        quad = np.asarray(row["placement_quad"], dtype=np.float64).reshape((4, 2))
        basis = reference.normal_basis_from_quad(quad, k)
        if basis is None:
            raise ValueError(f"source frame {row['video_frame_index']} has a degenerate quad")
        origin, axes, origin_px, axis_length = basis
        r1, r2, _ = reference.pose_from_quad(quad, k)
        local = np.eye(4)
        local[:3, 0] = r1 / 2
        local[:3, 1] = r2 / 2
        local[:3, 2] = axes[2] * axis_length
        local[:3, 3] = origin
        cube = np.eye(4)
        cube_size = DEFAULT_MODEL_EXTENT * axis_length
        cube[:3, :3] = np.column_stack([axes[0], axes[2], -axes[1]]) * cube_size
        cube[:3, 3] = origin + 0.5 * cube_size * axes[2]
        converted.append({
            "present_index": row["present_index"],
            "quad": quad.tolist(),
            "origin_camera": origin.tolist(),
            "origin_px": origin_px.tolist(),
            "axis_model": reference.colmajor(cv_to_gl @ local),
            "cube_model": reference.colmajor(cv_to_gl @ cube),
        })
    result = {
        "reference": "examples/placement_quad_by_local_coord.py: normal_basis_from_quad + pose_from_quad",
        "width": width, "height": height,
        "source_name": metadata[b"trd.video.source_name"].decode(),
        "source_sha256": metadata[b"trd.video.sha256"].decode(),
        "default_model_extent": DEFAULT_MODEL_EXTENT,
        "rows": converted,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"wrote {len(converted)} original-quad Python reference rows to {args.output}")


if __name__ == "__main__":
    main()
