# trd rendering sub-schema

**Purpose:** the minimal data trd consumes, not the complete upstream business
schema. Target: the simplified scene input discussed in #367 / #368.

Selected source fields follow the supplied `field_definition_design 2.md`.
That complete upstream business document is not redistributed here. This document
does not change their types or redefine the upstream tables. Unselected source
columns are retained for export, not deleted or validated as rendering inputs.

## 1. Input shape

```text
[params]                    placement coordinates + reference wireframe cube
[params] [mesh: one row]    one model
[params] [mesh: N rows]     multiple models
```

Each bracket is a complete Arrow IPC stream with its own schema. A `mesh` table,
when supplied, follows params; it is not repeated for every params row.
The caller may instead supply GLB bytes or ID-keyed GLB buffers through the
native/wasm API. Both delivery forms produce the same in-memory mesh bindings.

No separate texture or inline-video-frame table is required. Video remains an
external input. Parquet is not an application input container.

## 2. Params: four rendering field groups

For a tracked input, consume only the following source fields when the
corresponding operation needs them:

| Group | Source field | Source type | Needed for |
|---|---|---|---|
| Placement | `bottom_quads` | `list<fixed_size_list<point,4>>` | Reconstruct the local coordinate frame for each placement |
| Camera | `k` | FHC intrinsics struct | Resolve the CV camera and interpret the placement |
| Object transform | `model` | `list<fixed_size_list<fixed_size_list<float32,4>,4>>` | Place/edit loaded models; missing values at the column level start from identity |
| Asset binding | `mesh_id` | `list<uuid>` | Match placed models to supplied assets |

**Column presence is conditional; source nullability is not redefined.**
When a selected source column is present, preserve its declared non-null
column/item/struct constraints. The geometry-only case does not require
`model` or `mesh_id`; exporting a newly model-bound source adds the applicable
binding fields in their source-defined form.

### Selected geometry types

```text
point = struct<
  x: float32 not null,
  y: float32 not null,
  w: float32 not null,
  h: float32 not null
>

bottom_quads: list<fixed_size_list<point not null, 4> not null> not null

k: struct<
  fx: float32 not null,
  fy: float32 not null,
  cx: float32 not null,
  cy: float32 not null,
  skew: float32 not null,
  w: float32 not null,
  h: float32 not null
> not null

model: list<
  fixed_size_list<fixed_size_list<float32 not null, 4> not null, 4> not null
> not null

mesh_id: list<uuid not null> not null
```

Keep the source's clockwise quad ordering, canonical FHC image frame, finite
coordinates and aligned per-placement lists. Do not reinterpret FHC as pixel
coordinates or the source's row-major `model` as column-major data.

Preserve the source's starting corner as well as winding. The placement
reference derives its first basis axis and reconstruction scale from the first
edge; rotating the list to the smallest image y changes both. An upstream table
already using its prescribed first corner keeps it, while converted legacy
FIBA quads keep their original UL/UR/LR/LL order. The renderer must not sort
either input.

The rendering subset also accepts finite, convex tracked quads extending
outside the image (the existing FIBA tracking contains them). Preserve those
coordinates and let rasterization clip the visible result; clamping corners
would change the reconstructed placement. This is a rendering-input allowance,
not a change to the upstream business schema's in-frame annotation constraint.

### Mapping to the existing rendering vocabulary

These are adapter conversions, not new columns added to source data:

| Source | Existing rendering concept | Conversion |
|---|---|---|
| `bottom_quads[i]` | Placement quad / local frame | FHC points to pixels, then the placement calculation |
| FHC `k` | CV `k` / `Camera` | FHC intrinsics to pixel intrinsics |
| `model[i]` | Per-instance `draw_model` / `Draw.model` | Row-major 4x4 to the internal column-major matrix |
| UUID `mesh_id[i]` | `draw_mesh` / renderer mesh index | Explicit UUID-to-mesh-row/slot lookup |

FHC conversions use the source frame, not the current window dimensions:

```text
u_px = (x + w) / 2           v_px = (y + h) / 2
fx_px = k.fx / 2             fy_px = k.fy / 2
cx_px = (k.cx + k.w) / 2     cy_px = (k.cy + k.h) / 2
skew_px = k.skew / 2
```

Identity `model` places the object's AABB bottom center at the placement-local
origin, using the size convention of `placement_quad_by_local_coord.py`:

```text
asset_base = scale_to_extent_1 * translate(-aabb_center_x, -aabb_min_y, -aabb_center_z)
placement_frame = Python placement basis * scale(quad.axis_length)
world_model = placement_frame * object_model * asset_base
```

`asset_base` is derived from the unchanged GLB, not baked back into it or into
the editable `model`. The same deterministic base is applied on load and replay.
The shared placement default fits the longest asset edge to one quad half-edge
unit, independently of the ordinary mesh viewer's preview size. It preserves
proportions and grounds the unedited AABB. The initial quad
placement has zero in-plane offset and no preset lift; later user translations,
rotations and scales can intentionally move the mesh off its initial contact.
Ordinary CG/CV scenes without a quad retain their existing matrix semantics and
do not receive this placement-specific base.

For one mesh and one quad, loading establishes their binding and selects the
quad. Clicking the mesh performs GPU picking and selects both that instance
and its owning quad. Retain the original object editor's Translate, Rotate, Scale,
axis constraints, numeric controls and mouse gestures; a read-only matrix view
is an advanced aid, not a replacement for those controls. Translate offers only
the quad basis: **e1 (quad X)**, **e2 (-quad Z)** and **e3 (normal)** in that order,
without a separate object-local basis selector.

The internal `assets/meshes/cube/cube.obj` is placed by the existing placement
method with the same default extent-1 normalization as imported models: its
**bottom-face center**, not body center, is at the quad origin, and its +Y points
along the plane normal. This reference asset never enters Arrow.
Clicking a quad selects/highlights it and shows its local axes, plane grid and
cube; clicking empty image space deselects it and hides those local references.
Hovering fills the quad without selecting it; moving away clears only the hover
feedback. **Show Coordinate** and **Show Plane** independently control axes and
grid for the selected quad. The reference cube's wireframe is dark blue;
ordinary model AABBs retain their existing green styling.

Rendering resolution is separate from presentation: a 1920x1080 target is
displayed fully inside the available canvas region with its original aspect
ratio, using letterboxing rather than cropping. Hover and picking use the
painted image bounds, not the surrounding black bars.

The source `model` column remains row-major on export; `draw_model` is the
internal correspondence, not a reason to rename that source column.

### CG/CV compatibility

**The existing CG-style camera is retained, including for golden fixtures.**
Keep the existing CG/CV params reader as a separate adapter:

- CV matrix form: `k`, optional `pose`.
- CG form: `eye`, `target` or `direction`, `up`, `fovy`, `aspect`, `znear`, `zfar`.
- Existing `model`, `draw_model`, `draw_mesh` and render-mode fields retain their
  meanings within that adapter.

| CG field | Existing Arrow type | Existing behavior to preserve |
|---|---|---|
| `eye` | `fixed_size_list<float32,3>` | World-space camera position |
| `target` | `fixed_size_list<float32,3>` | Look-at point; takes precedence over `direction` |
| `direction` | `fixed_size_list<float32,3>` | Alternative look direction from `eye` |
| `up` | `fixed_size_list<float32,3>` | Default +Y |
| `fovy` | `float32` | Vertical field of view in radians |
| `aspect` | `float32` | Defaults to the viewport's aspect when perspective is used |
| `znear` | `float32` | Existing perspective default 0.1 |
| `zfar` | `float32` | Existing perspective default 1000 |

These columns remain optional according to the existing camera-form rules.
An eye requires a target or direction, and vice versa. Do not mix the CV and CG
forms in one camera. Missing projection/view halves keep their existing identity
behavior; the new adapter must not invent a different framing camera.

Both adapters resolve a typed camera and draw list. Do not append CG fields to
the tracked source schema or confuse its FHC `k` struct with the matrix-form `k`.
A scene without a placement quad uses its ordinary world coordinate system.

Golden fixtures may continue to use the CG columns directly. Asset/framing
migration must not change their camera values, look-at convention, clip-depth
range or matrix composition. Do not convert those fixtures to FHC or regenerate
golden images to conceal an accidental camera change.

## 3. Optional application helpers, not mandatory rendering fields

| Field/context | When it is useful | Rendering requirement |
|---|---|---|
| `present_index` | Join annotated rows to the selected video's frame identity | Needed by the video adapter, not by geometry/camera rendering |
| `pts` plus the video's time base | Timestamp-based lookup or diagnostics | Optional; never manufacture PTS from FPS |
| `track_id` | Keep an edited instance identifiable when lists reorder or several placements share one asset | Optional editor helper; not an asset ID |
| Existing video metadata | Source validation, playback timing and video selection | Provided by the existing video layer, not required as a complete params schema |

Keep original identity values unchanged. A video adapter must use a valid
packet-to-presented-frame mapping; `present_index` is not implicitly an
FPS-grid frame number. Do not guess cross-frame instance identity from list
position when it can change. If an editing operation needs identity information
that is absent, report that limitation instead of modifying the wrong object.

In particular, **do not require** `vertical_vp`, `height`, `bboxes`, detection
scores, labels, shot/scene results, processing IDs or run-status fields merely
to render a quad or model. The current placement calculation uses the quad and
camera; those extra business/algorithm fields stay as pass-through data.

## 4. Mesh: only an ID and a GLB

The mesh table has exactly two non-null fields:

```text
mesh_id: uuid not null
glb: large_binary not null
```

| Field | Meaning |
|---|---|
| `mesh_id` | The source UUID referenced by params |
| `glb` | The original bytes of one self-contained GLB asset |

One row is one asset, not one frame and not necessarily one glTF primitive.
Store each asset once; multiple placements may reference the same UUID.
IDs must be unique and GLB payloads must be nonempty and valid.

Do not split geometry, materials or textures into Arrow columns. There are no
`position`, `index`, `uv`, `normal`, `material` or texture fields, and no
separate texture table. Those resources remain inside the GLB.

```text
Arrow mesh row -> GLB decoder -> Rust MeshAsset -> renderer-local mesh index
```

Decoding is an in-memory rendering step, not a conversion into another Arrow
mesh representation. Preserve the GLB bytes unchanged when exporting; do not
re-encode GLB or strip fields the current renderer does not consume.

Use indices for internal storage where convenient, with an explicit
UUID-to-index lookup. Do not replace source UUIDs with GPU slot indices.
Self-contained GLBs do not require a separate image/buffer fetch protocol;
unsupported assets or external resource dependencies produce explicit errors.
This does not expand the current GLB importer's rendering capabilities.

There is no OBJ payload/path or standalone material/texture resource input in
this table. **Existing internal OBJ loading and rendering must remain
supported**, including `Mesh::from_obj`, its materials/textures and the existing
non-protocol viewer paths. Removing OBJ from the Arrow input contract is not
permission to remove or change the renderer's ability to draw OBJ-loaded meshes.

## 5. Read subset, write through

**A sub-schema is a consumption contract, not an export projection.**

1. Retain the complete input schemas and RecordBatches.
2. Read/validate only the selected rendering fields and any helper fields an
   enabled operation actually needs.
3. Update the appropriate `model` and binding entries, or add the missing
   model-binding fields when a model is assigned.
4. Reuse every untouched column, including unknown data, original types,
   nulls, field/schema metadata, row order and batch boundaries.
5. Preserve untouched matrices without TRS decomposition or recomposition.

The tracked-source adapter writes source `model`/`mesh_id`; the CG/CV adapter
uses its own existing column names. Neither exports internal GPU indices over
source UUIDs. IPC envelope bytes can differ after editing, but unrelated Arrow
data must not be lost.

## 6. Video is not another Arrow table

Keep the current independent video decoder/player. No frame image, MP4 bytes,
or mandatory image link is required in params. Sparse annotated rows select
overlays for corresponding video frames; other frames stay video-only.

The older CG/CV adapter's optional external still-image references are separate
from video playback. They do not become required fields of this sub-schema.
No full video-metadata schema, inline frames table, or Parquet reader is added
to satisfy rendering.
