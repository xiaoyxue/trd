#!/usr/bin/env python3
"""Generate/check the current 0.0.7 Arrow contract and committed scene fixtures.

There is one scene protocol, not a historical-version selector. The independent
video-edit 0.2.0 JSON describes offline source data and is not generated here.

uv run --with pyarrow python scripts/protocol_schema.py [--check]
"""

import argparse
import json
from pathlib import Path

import pyarrow as pa

from protocol_version import PROTOCOL_VERSION, validate_render_config_metadata
ROOT = Path(__file__).resolve().parent.parent
PROTOCOL_OUTPUT = ROOT / "docs/protocol/0.0.7.schema.json"
PROTOCOL_FIXTURES = [ROOT / f"crates/trd-core/tests/golden/stage{stage}.arrow" for stage in (1, 2)]


def tracked_types():
    nonnull = lambda value: pa.field("item", value, nullable=False)
    point = pa.struct([pa.field(name, pa.float32(), nullable=False) for name in ("x", "y", "w", "h")])
    k = pa.struct([pa.field(name, pa.float32(), nullable=False) for name in ("fx", "fy", "cx", "cy", "skew", "w", "h")])
    return {
        "bottom_quads": pa.list_(nonnull(pa.list_(nonnull(point), 4))),
        "k": k,
        "model": pa.list_(nonnull(pa.list_(nonnull(pa.list_(nonnull(pa.float32()), 4)), 4))),
        "mesh_id": pa.list_(nonnull(pa.uuid())),
        "present_index": pa.int64(), "pts": pa.int64(), "track_id": pa.list_(pa.string()),
    }


def camera_types():
    return {
        "model": pa.list_(pa.float32(), 16),
        "k": pa.list_(pa.float32(), 9), "pose": pa.list_(pa.float32(), 16),
        **{name: pa.list_(pa.float32(), 3) for name in ("eye", "target", "direction", "up")},
        **{name: pa.float32() for name in ("fovy", "aspect", "znear", "zfar")},
        "draw_mesh": pa.list_(pa.uint32()), "draw_model": pa.list_(pa.list_(pa.float32(), 16)),
        "draw_mode": pa.list_(pa.uint8()), "video_frame_index": pa.uint32(),
        "present_index": pa.int64(), "pts": pa.int64(),
        "frame_path": pa.string(), "frame_url": pa.string(), "tonemap": pa.uint8(),
    }


def build():
    def columns(types, required=()):
        return [{"name": name, "arrow_type": str(value), "required": name in required}
                for name, value in types.items()]
    return {
        "protocol_version": PROTOCOL_VERSION,
        "generated_by": "scripts/protocol_schema.py",
        "prose_spec": "docs/protocol/0.0.7.md",
        "source_contract": "assets/schemas/trd-render-sub-schema.md",
        "schema_kind": "Descriptive Arrow contract, not a JSON instance-validation schema",
        "container": "concatenated Apache Arrow IPC streams",
        "input_order": ["params", "mesh?"],
        "compatibility": "none; declared versions other than 0.0.7 are rejected",
        "schema_metadata": {
            "trd.protocol.version": PROTOCOL_VERSION,
            "trd.table.kind": ["params", "mesh"],
            "trd.stream.frame_rate": {"required": False, "type": "positive float string", "default": 30},
            "trd.render.config": {
                "required": False,
                "scope": "params schema; shared by all record batches and sparse rows",
                "type": "JSON object",
                "default": {"shadow": {"enable": True, "shadow_type": "blob"}},
                "shadow": {
                    "enable": {"type": "boolean", "default": True},
                    "shadow_type": {"supported": ["blob"], "reserved_unsupported": ["shadow_map"], "default": "blob"},
                },
                "validation": "Missing fields use defaults; malformed JSON, null/wrong types, unknown fields/types and shadow_map are errors.",
            },
        },
        "source_adapter_exception": "Unversioned FHC source tables use the explicit tracked adapter; declared old versions are never upgraded.",
        "tables": {
            "params": {
                "row_model": "one existing scene sample; sparse video rows are not padded",
                "retain_unknown_columns": True,
                "adapters": {
                    "tracked_fhc": {
                        "matrix_order": "row-major model on wire; column-major internally",
                        "columns": columns(tracked_types(), ("bottom_quads", "k")),
                        "constraints": [
                            "Preserve source corner order/winding and finite off-image coordinates.",
                            "Use the source image dimensions for FHC conversions.",
                            "Per-placement lists align; consumed values are finite and non-null.",
                            "An absent model column means identity per instance.",
                            "UUID bindings are required where implicit correspondence is ambiguous.",
                        ],
                    },
                    "cg_cv": {
                        "matrix_order": "column-major",
                        "columns": columns(camera_types()),
                        "draw_mode": {"0": "filled", "1": "wireframe", "2": "textured", "3": "blob shadow", "4": "shaded/PBR", "255": "inherit"},
                        "tonemap": {"0": "Reinhard", "1": "ACES"},
                        "constraints": [
                            "Preserve existing CG/CV camera defaults and golden camera values.",
                            "Do not mix CG and CV camera forms.",
                            "Draw-list lengths align; indices belong only to this adapter.",
                        ],
                    },
                },
            },
            "mesh": {
                "required": False,
                "allow_extra_columns": False,
                "row_model": "one original self-contained GLB per unique UUID",
                "columns": [
                    {"name": "mesh_id", "arrow_type": str(pa.uuid()), "storage_type": "fixed_size_binary[16]", "nullable": False},
                    {"name": "glb", "arrow_type": str(pa.large_binary()), "nullable": False},
                ],
                "constraints": [
                    "Unique UUIDs and valid self-contained GLB 2.0 payloads.",
                    "No external GLB buffers/images or resource path/URL.",
                    "Resource row order does not change UUID bindings.",
                    "Preserve original bytes, schema metadata and all batch boundaries.",
                ],
            },
        },
        "editing": {
            "scope": "selected object across all corresponding existing sparse rows",
            "stored_model": "adjustment * original_local_model",
            "identity": "track_id across reorder/gaps, otherwise consistent ordered bindings; reject ambiguity",
            "replay": "read saved per-row models without applying authoring adjustments twice",
            "retained": ["unknown columns", "types/nulls", "field/schema metadata", "row order", "params/mesh batch boundaries", "other objects", "UUID/GLB bytes"],
        },
        "media": {"video": "external", "missing_row": "video-only", "pts": "never fabricated from fps", "external_stills": ["frame_path", "frame_url"]},
        "retired_inputs": ["mesh-first envelope", "Arrow OBJ geometry", "texture table", "inline frames table", "frame_id", "gltf_path resource", "gltf_url resource", "Parquet application input", "video-edit 0.2.0 application input"],
        "output": {"row_model": "one image per params row", "batch_boundaries": "mirror params", "channels": ["r", "g", "b", "a"], "arrow_type": "fixed_shape_tensor<uint8>[height,width]"},
    }


def check_fixture(path):
    data = path.read_bytes()
    stream = pa.BufferReader(data)
    kinds, checked = [], 0
    while stream.tell() < len(data):
        reader = pa.ipc.open_stream(stream)
        schema = reader.schema
        metadata = schema.metadata or {}
        if metadata.get(b"trd.protocol.version") != PROTOCOL_VERSION.encode():
            raise ValueError(f"{path.name}: must declare protocol {PROTOCOL_VERSION}")
        kind = metadata.get(b"trd.table.kind", b"").decode()
        kinds.append(kind)
        if kinds not in (["params"], ["params", "mesh"]):
            raise ValueError(f"{path.name}: invalid table order {kinds}")
        if kind == "mesh":
            types = {"mesh_id": pa.uuid(), "glb": pa.large_binary()}
            if schema.names != list(types) or any(field.nullable for field in schema):
                raise ValueError(f"{path.name}: mesh has only non-null mesh_id/glb")
        else:
            validate_render_config_metadata(metadata)
            types = tracked_types() if "bottom_quads" in schema.names else camera_types()
            if "frame_id" in schema.names:
                raise ValueError(f"{path.name}: inline frames are retired")
        for field in schema:
            expected = types.get(field.name)
            if expected is not None and str(field.type).replace(" not null", "") != str(expected).replace(" not null", ""):
                raise ValueError(f"{path.name}: {field.name}: expected {expected}, got {field.type}")
            checked += 1
        for _ in reader:
            pass
    if not kinds:
        raise ValueError(f"{path.name}: empty document")
    return checked


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    rendered = json.dumps(build(), indent=2, ensure_ascii=False) + "\n"
    checked = sum(check_fixture(path) for path in PROTOCOL_FIXTURES)
    if args.check:
        if not PROTOCOL_OUTPUT.exists() or PROTOCOL_OUTPUT.read_text(encoding="utf-8") != rendered:
            raise ValueError("0.0.7.schema.json is stale; run scripts/protocol_schema.py")
    else:
        PROTOCOL_OUTPUT.write_text(rendered, encoding="utf-8")
    print(f"Protocol {PROTOCOL_VERSION}: {checked} fixture columns; current schema {'checked' if args.check else 'written'}")


if __name__ == "__main__":
    main()
