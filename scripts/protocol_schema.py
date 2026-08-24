#!/usr/bin/env python3
"""Emit — and check — the machine-readable schemas under ``docs/``.

Two **independent** versioned schemas live here, and they are deliberately not
tied to each other (see ``AGENTS.md``):

* ``docs/protocol/0.0.6.schema.json`` — the **stream protocol**. Dense: one
  params row is one rendered frame, and the output image stream is 1:1 with it.
* ``docs/video-editing.schema.json`` — the **video-editing document** (0.2.0).
  **Sparse**: a row exists only for a frame that was annotated, so a 288-frame
  video with 222 tracked frames is a 222-row table and the other 66 frames have
  no row at all. ``video_frame_index`` is the join key back to the container.

The prose specs (``docs/protocol/0.0.6.md``, ``docs/video-editing.md``) are what
a human reads; these are what a producer generates against. Keeping them
generated rather than hand-written is what stops the two from drifting the way a
second copy always does: run ``--check`` and a stale file fails instead of
quietly lying.

The declared types are not taken on trust. ``--check`` also opens the Arrow
fixtures and asserts every column they carry matches the type declared here.
The video-editing fixture is generated from an external MP4 and is gitignored,
so it is checked when present and skipped when not — the same rule the Rust
tests use.

    python3 scripts/protocol_schema.py            # rewrite both JSON files
    python3 scripts/protocol_schema.py --check    # fail if stale or contradicted

`pyarrow` is not in the nix dev shell, so inside `nix develop` run it as
``uv run --with pyarrow scripts/protocol_schema.py --check``.
"""

import argparse
import json
import sys
from pathlib import Path

import pyarrow as pa

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
PROTOCOL_OUTPUT = REPOSITORY_ROOT / "docs/protocol/0.0.6.schema.json"
VIDEO_EDIT_OUTPUT = REPOSITORY_ROOT / "docs/video-editing.schema.json"

# Committed fixtures the protocol types are checked against.
PROTOCOL_FIXTURES = (
    REPOSITORY_ROOT / "crates/trd-core/tests/golden/stage1.arrow",
    REPOSITORY_ROOT / "crates/trd-core/tests/golden/stage2.arrow",
)
# Generated from an external MP4 and gitignored: checked when present.
VIDEO_EDIT_FIXTURES = (REPOSITORY_ROOT / "web/gui-video-editing/data/fiba-shot1.arrow",)

PROTOCOL_VERSION = "0.0.6"
VIDEO_EDIT_VERSION = "0.2.0"

# Arrow type strings, spelled the way `str(pyarrow.DataType)` prints them, so a
# producer can compare its own schema against this file without a parser.
F32_3 = "fixed_size_list<item: float>[3]"
F32_9 = "fixed_size_list<item: float>[9]"
F32_16 = "fixed_size_list<item: float>[16]"
LIST_F32_3 = f"list<item: {F32_3}>"
LIST_F32_2 = "list<item: fixed_size_list<item: float>[2]>"
LIST_F32_16 = f"list<item: {F32_16}>"


def column(name, arrow_type, required, doc, **extra):
    return {"name": name, "arrow_type": arrow_type, "required": required, "doc": doc, **extra}


def build_video_edit():
    """The video-editing document (0.2.0) as plain data — the **sparse** one."""
    return {
        "video_edit_version": VIDEO_EDIT_VERSION,
        "generated_by": "scripts/protocol_schema.py",
        "prose_spec": "docs/video-editing.md",
        "independent_of": (
            "trd.protocol.version — the two versions move separately on purpose; "
            "editor state never bumps the stream protocol and vice versa"
        ),
        "containers": ["Arrow IPC stream", "Parquet"],
        "container_note": (
            "both decode through one path; the format is sniffed, not declared. "
            "Parquet compression is limited to uncompressed and snappy, because "
            "zstd/gzip/lz4/brotli are C shims that would break the wasm build"
        ),
        "row_model": {
            "shape": "sparse",
            "rule": (
                "the timeline table stores only frames with ad-placement quads; "
                "everything else is ordinary video and is played as such. A frame "
                "with no row is looked up as None and rendered as plain video"
            ),
            "join_key": "video_frame_index",
            "row_count": "<= trd.video.frame_count, often far below it",
            "worked_example": {
                "document": "web/gui-video-editing/data/fiba-shot1.arrow",
                "container_frame_count": 288,
                "rows": 222,
                "frames_without_a_row": 66,
                "note": (
                    "frames 222-287 carry no row, which is why the editor reports "
                    "`Arrow video_frame_index: none` and `tracking state: none` "
                    "for them. In this document the annotated frames happen to be "
                    "a contiguous prefix; the schema does not require that — gaps "
                    "may fall anywhere, which is what video_frame_index is for"
                ),
            },
        },
        "schema_metadata": {
            "trd.video_edit.version": {"required": True, "value": VIDEO_EDIT_VERSION},
            "trd.video_edit.table.kind": {"required": True, "value": "timeline"},
            "trd.video.source_name": {"required": True, "type": "string", "doc": "source file name, e.g. shot_0001.mp4"},
            "trd.video.mime": {"required": True, "type": "string", "doc": "e.g. video/mp4"},
            "trd.video.codec": {"required": True, "type": "string", "doc": "e.g. h264"},
            "trd.video.sha256": {"required": True, "type": "hex string", "doc": "identifies the exact source the document was authored against"},
            "trd.video.byte_length": {"required": True, "type": "integer"},
            "trd.video.width": {"required": True, "type": "integer"},
            "trd.video.height": {"required": True, "type": "integer"},
            "trd.video.fps_num": {"required": True, "type": "integer", "doc": "rational frame rate numerator; never a rounded float"},
            "trd.video.fps_den": {"required": True, "type": "integer"},
            "trd.video.frame_count": {"required": True, "type": "integer", "doc": "the container's frame count, which the sparse rows index into"},
            "trd.video.duration_us": {"required": True, "type": "integer"},
        },
        "metadata_note": (
            "every key above is required; a missing one is rejected rather than "
            "defaulted. The container's unpresented tail is deliberately NOT "
            "carried here — it is a property of the file, discovered by probing"
        ),
        "table": {
            "trd.video_edit.table.kind": "timeline",
            "row_meaning": "one annotated frame",
            "reference_producer": "scripts/fiba_video_editing_bundle.py",
            "columns": [
                column(
                    "video_frame_index",
                    "uint32",
                    True,
                    "the container frame this row annotates; the join key that makes the table sparse",
                    nullable=False,
                ),
                column("present_index", "uint32", True, "the source parquet row this timeline row copies", nullable=False),
                column("timestamp_us", "int64", True, "deterministic media timestamp, microseconds", nullable=False),
                column(
                    "k",
                    F32_9,
                    False,
                    "row-major OpenCV intrinsics; paired with placement_quad",
                    nullable=True,
                ),
                column(
                    "placement_quad",
                    "fixed_size_list<item: float>[8]",
                    False,
                    "the ad-placement quad as 4 corners x (x, y) in image pixels, ordered TL, TR, BR, BL",
                    nullable=True,
                ),
                column("tracked", "bool", True, "whether the K/quad geometry on this row is valid", nullable=False),
                column(
                    "poster_bytes",
                    "binary",
                    False,
                    "optional encoded JPEG so an editor can show something before the first decode",
                    nullable=True,
                ),
            ],
            "constraints": [
                "video_frame_index is strictly increasing across the whole table, not merely unique",
                "video_frame_index < trd.video.frame_count; a row past the video is rejected",
                "rows may skip frames freely — that is the point of the schema",
                "k and placement_quad are present together or absent together; one without the other is rejected",
                "poster_bytes is non-null on at most one row, and only on the first row",
                "a sparse document need not annotate frame 0, so the poster is optional; the first decoded frame serves instead",
            ],
        },
    }


def build():
    """The whole `0.0.6` contract as plain data."""
    return {
        "protocol_version": PROTOCOL_VERSION,
        "generated_by": "scripts/protocol_schema.py",
        "prose_spec": "docs/protocol/0.0.6.md",
        "row_model": (
            "dense — one params row is one rendered frame, and the output image "
            "stream is 1:1 with the params rows. The sparse, per-annotated-frame "
            "table is a different schema: docs/video-editing.schema.json"
        ),
        "compatibility": (
            "none — the renderer accepts exactly this version and hard-rejects "
            "any other or missing one"
        ),
        "stream": {
            "container": "concatenated Apache Arrow IPC streams on one byte channel",
            "order": ["mesh", "texture?", "frames?", "params"],
            "note": (
                "resource tables precede the terminal params table; duplicate or "
                "out-of-order tables are hard errors"
            ),
        },
        "schema_metadata": {
            "trd.protocol.version": {
                "required": True,
                "value": PROTOCOL_VERSION,
                "doc": "carried by every input sub-stream and by the output stream",
            },
            "trd.table.kind": {
                "required": True,
                "one_of": ["mesh", "texture", "frames", "params"],
                "doc": "input sub-streams declare their kind; schemas are never sniffed from column names",
            },
            "trd.stream.frame_rate": {
                "required": False,
                "type": "float, frames per second",
                "default": 30.0,
                "doc": "playback speed as a property of the data; trd-cli copies it to the output stream",
            },
        },
        "conventions": {
            "matrices": "column-major, right-handed, flattened to 16 floats",
            "clip_space": "wgpu, z in [0, 1]",
            "vertex_chain": "clip = P * V * M * (position, 1)",
            "images": "top-left origin, sRGB byte values",
        },
        "tables": {
            "mesh": {
                "trd.table.kind": "mesh",
                "required": True,
                "row_meaning": "one row is one mesh; its 0-based ordinal is the id params draw lists use",
                "reference_producer": "scripts/obj_to_arrow.py",
                "columns": [
                    column("position", LIST_F32_3, True, "vertex positions"),
                    column("color", LIST_F32_3, False, "vertex RGB; absent means white"),
                    column("uv", LIST_F32_2, False, "top-left-origin texture coordinates"),
                    column("index", "list<item: uint32>", False, "triangle indices; absent means sequential"),
                ],
                "constraints": [
                    "color and uv, when present, carry one value per vertex",
                    "null values in any present column are rejected",
                    "a zero-row table is an error",
                ],
            },
            "texture": {
                "trd.table.kind": "texture",
                "required": False,
                "row_meaning": "one row is the mesh albedo; the first non-empty row is bound",
                "reference_producer": "scripts/texture_to_arrow.py",
                "columns": [
                    column(
                        "rgba",
                        "extension<arrow.fixed_shape_tensor[value_type=uint8, shape=[H,W,4], dim_names=[height,width,channel]]>",
                        True,
                        "row-major RGBA8 albedo",
                        storage_type="fixed_size_list<item: uint8>[H*W*4]",
                        extension_name="arrow.fixed_shape_tensor",
                        nullable=False,
                    ),
                ],
                "constraints": ["shape is field metadata, so every row shares H, W and 4 channels"],
            },
            "frames": {
                "trd.table.kind": "frames",
                "required": False,
                "row_meaning": (
                    "one row is one reusable inline background image; its 0-based ordinal is the "
                    "frame_id params reference, so several params rows may reuse one image"
                ),
                "reference_producer": "scripts/frames_to_arrow.py",
                "columns": [
                    column("frame_bytes", "binary", False, "encoded PNG or JPEG bytes", nullable=True),
                    column(
                        "frame_pixels",
                        "extension<arrow.fixed_shape_tensor[value_type=uint8, shape=[H,W,C], dim_names=[height,width,channel]]>",
                        False,
                        "raw row-major pixels; C is 3 or 4",
                        storage_type="fixed_size_list<item: uint8>[H*W*C]",
                        extension_name="arrow.fixed_shape_tensor",
                        nullable=True,
                    ),
                ],
                "constraints": [
                    "exactly one of the two payloads is non-null per row",
                    "RGB tensors gain opaque alpha; RGBA is preserved byte-for-byte",
                    "frame_pixels rows in one table share their dimensions; frame_bytes rows need not",
                ],
            },
            "params": {
                "trd.table.kind": "params",
                "required": True,
                "terminal": True,
                "row_meaning": "one row is one rendered frame; output is 1:1 with these rows",
                "reference_producer": "scripts/jsonl_to_arrow.py",
                "all_columns_optional": True,
                "columns": [
                    column("model", F32_16, False, "single-object fallback model matrix"),
                    column("k", F32_9, False, "CV camera intrinsics"),
                    column("pose", F32_16, False, "CV camera-to-world pose"),
                    column("eye", F32_3, False, "CG camera position"),
                    column("target", F32_3, False, "CG look-at target"),
                    column("direction", F32_3, False, "CG look direction (alternative to target)"),
                    column("up", F32_3, False, "CG up vector"),
                    column("fovy", "float", False, "CG vertical field of view, radians"),
                    column("aspect", "float", False, "CG aspect ratio"),
                    column("znear", "float", False, "CG near plane"),
                    column("zfar", "float", False, "CG far plane"),
                    column("draw_mesh", "list<item: uint32>", False, "mesh-table row ids, one per draw"),
                    column("draw_model", LIST_F32_16, False, "per-draw model matrices; length equals draw_mesh"),
                    column(
                        "draw_mode",
                        "list<item: uint8>",
                        False,
                        "per-draw render-mode override; length equals draw_mesh",
                        values={
                            "0": "filled",
                            "1": "wireframe",
                            "2": "textured",
                            "3": "shadow",
                            "4": "pbr",
                            "255": "inherit the front-end's global mode",
                        },
                    ),
                    column("frame_path", "string", False, "external background, native filesystem path", nullable=True),
                    column("frame_url", "string", False, "external background, browser URL", nullable=True),
                    column("frame_id", "uint32", False, "inline background, frames-table row id", nullable=True),
                ],
                "constraints": [
                    "draw_mesh and draw_model are present together or not at all; one without the other is an error",
                    "no draw columns renders mesh 0 using model; an explicit empty draw list renders no meshes",
                    "CV (k/pose) and CG (eye/target/direction/up/fovy/aspect/znear/zfar) camera forms are mutually exclusive",
                    "eye requires target or direction, and vice versa",
                    "a row has at most one background source: inline frame_id, or external frame_path/frame_url",
                    "frame_id requires a preceding frames table and must be in range",
                    "frame_path wins over frame_url when both are non-empty on a row",
                ],
            },
        },
        "output": {
            "doc": "the rendered image stream trd-cli writes; one row per params row, batch boundaries mirrored",
            "schema_metadata": {
                "trd.protocol.version": PROTOCOL_VERSION,
                "trd.stream.frame_rate": "copied from the input when it declared one",
            },
            "columns": [
                column(
                    name,
                    "extension<arrow.fixed_shape_tensor[value_type=uint8, shape=[height,width], dim_names=[height,width]]>",
                    True,
                    f"planar {name.upper()} channel, row-major uint8",
                    storage_type="fixed_size_list<item: uint8>[height*width]",
                    extension_name="arrow.fixed_shape_tensor",
                    nullable=False,
                )
                for name in ("r", "g", "b", "a")
            ],
        },
    }


def declared_types(spec):
    """Column name -> declared arrow_type, across every table in a spec."""
    types = {}
    tables = spec.get("tables")
    if tables is None:
        tables = {"timeline": spec["table"]}
    for table in tables.values():
        for col in table["columns"]:
            types[col["name"]] = col["arrow_type"]
    return types


def fixture_columns(path):
    """Every (table_kind, field) an Arrow IPC stream carries, sub-stream by sub-stream."""
    data = path.read_bytes()
    buf = pa.py_buffer(data)
    offset = 0
    while offset < len(data):
        source = pa.BufferReader(buf)
        source.seek(offset)
        try:
            reader = pa.ipc.open_stream(source)
        except pa.ArrowInvalid:
            return
        schema = reader.schema
        for _ in reader:
            pass
        meta = {k.decode(): v.decode() for k, v in (schema.metadata or {}).items()}
        for field in schema:
            yield meta.get("trd.table.kind"), field
        nxt = source.tell()
        if nxt <= offset:
            return
        offset = nxt


def check_against_fixtures(spec, fixtures, required):
    """Fail if a fixture contradicts a declared type.

    `required` is False for generated, gitignored fixtures: absent means skip,
    the same rule the Rust tests use, so a fresh checkout still passes.
    """
    types = declared_types(spec)
    problems = []
    checked = 0
    for path in fixtures:
        if not path.exists():
            if required:
                problems.append(f"{path} is missing")
            else:
                print(f"  skipped {path.name}: not generated in this checkout")
            continue
        for kind, field in fixture_columns(path):
            declared = types.get(field.name)
            if declared is None:
                problems.append(f"{path.name} [{kind}] has undocumented column `{field.name}`")
                continue
            # Tensor columns carry H/W/C placeholders, so compare the stable prefix.
            if declared.startswith("extension<"):
                if not str(field.type).startswith("extension<arrow.fixed_shape_tensor"):
                    problems.append(
                        f"{path.name} [{kind}] `{field.name}`: expected a fixed_shape_tensor, got {field.type}"
                    )
            elif str(field.type) != declared:
                problems.append(
                    f"{path.name} [{kind}] `{field.name}`: declared {declared}, fixture has {field.type}"
                )
            checked += 1
    return checked, problems


def render(spec):
    return json.dumps(spec, indent=2, ensure_ascii=False) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify the committed JSON is current and the fixtures agree; write nothing",
    )
    args = parser.parse_args()

    targets = [
        ("stream protocol", build(), PROTOCOL_OUTPUT, PROTOCOL_FIXTURES, True),
        ("video-editing document", build_video_edit(), VIDEO_EDIT_OUTPUT, VIDEO_EDIT_FIXTURES, False),
    ]

    problems = []
    for label, spec, output, fixtures, required in targets:
        print(f"{label}:")
        checked, found = check_against_fixtures(spec, fixtures, required)
        problems += found
        for problem in found:
            print(f"  error: {problem}", file=sys.stderr)
        print(f"  checked {checked} fixture columns against the declared types")

        rendered = render(spec)
        relative = output.relative_to(REPOSITORY_ROOT)
        if args.check:
            current = output.read_text(encoding="utf-8") if output.exists() else ""
            if current != rendered:
                print(
                    f"  error: {relative} is stale — re-run scripts/protocol_schema.py",
                    file=sys.stderr,
                )
                problems.append(f"{relative} is stale")
            else:
                print(f"  {relative} is up to date")
        else:
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(rendered, encoding="utf-8")
            print(f"  wrote {relative} ({len(rendered)} bytes)")

    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
