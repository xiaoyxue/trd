# Editable protocol 0.0.7 params and GLB scene documents

This page describes the **0.0.7** contract and current document APIs agreed in
#367/#370, based on #368. See the [formal specification](0.0.7.md),
[machine-readable contract](0.0.7.schema.json), and
[atomic implementation migration status](README.md#implementation-migration-status).
The latter tracks remaining old runtime/generator stamps; this is not an old
mesh-first compatibility mode.

Both native and browser video editors expose only this current workflow.
The old annotation/catalog UI, implicit default annotation and runtime
mesh-first/annotation fallback are removed. A video-edit 0.2.0 input is explicitly
rejected with the offline conversion command; original data and conversion
tools remain. Ordinary internal OBJ viewers are unaffected.

## Retained input

`trd_core::SceneDocument` retains the original params schemas/RecordBatches and
optional mesh resources. A mesh row contains exactly `mesh_id` (Arrow UUID) and
`glb` (LargeBinary). The GLB bytes, including their material/texture payloads,
are kept unchanged on export.
Original mesh schema/field metadata and record-batch boundaries, including
empty batches, are retained too. API-supplied resources with no source mesh
stream are emitted in a canonical single batch.

`SceneDocument::read` reads `[params][mesh?]`. A source table using the minimal
FHC `bottom_quads`/`k` schema can be read without inventing trd metadata. When a
stream declares trd protocol metadata, it must match the currently supported
0.0.7 version. A CG/CV params source keeps the existing camera fields and
conventions.

`SceneDocument::frames` exposes decoded rendering views. `frame(row)` decodes
only the requested source row for playback. Neither replaces the original
arrays. `apply_model_edits` edits source row/instance pairs atomically:

- tracked source: row-major nested `model` and UUID bindings;
- CG/CV params: column-major `draw_model`, preserving existing camera columns;
- all untouched columns, metadata, batch boundaries and unedited rows remain.

`bind_glb(bytes)` uses an existing unique source UUID or creates a new UUID.
Sources naming several UUIDs require `bind_meshes` with explicit bindings.
Internal renderer slots never overwrite those UUIDs.

## Browser API

All operations below are implemented in Rust; the browser supplies bytes.

```typescript
const doc = ArrowSceneDocument.fromArrow(arrowBytes);
// Or:
const single = ArrowSceneDocument.fromArrowWithGlb(arrowBytes, glbBytes);
const multiple = ArrowSceneDocument.fromArrowWithGlbs(
  arrowBytes,
  new Map([[meshUuidA, glbA], [meshUuidB, glbB]]),
);

renderer.loadSceneDocument(single); // CanvasRenderer or OffscreenRenderer
const matrix = single.getModel(row, object); // 16 column-major float32 values
matrix[12] += 0.1;
single.setModel(row, object, matrix);
await renderer.renderIndex(row);
const exportedBytes = single.exportArrow(); // params plus unchanged mesh resources
```

`fromArrowWithMesh(arrowBytes, meshUuid, glbBytes)` supplies an explicit ID for
the single-resource case. `meshIds()` exposes generated/preserved bindings.
Both renderers observe edits to the same document handle; they do not re-upload
meshes for matrix-only changes.

The native and wasm video-editor loaders also accept scene documents. A
single-mesh/single-quad input binds and selects its quad on load. Clicking the
mesh uses the same grounded geometry in the GPU pick pass as in rendering and
selects its owning quad. The original **Interaction** and **Transform** sections
provide Translate, Rotate, Scale, axis constraints and mouse gestures. They author an
adjustment to the stored local matrix without decomposing or losing arbitrary
source transforms. Translate offers only the quad basis, in the order **e1 (quad X)**,
**e2 (-quad Z)**, **e3 (normal)**, matching the placement basis rather than
relabeling model XYZ as e1/e2/e3; there is no object-local basis selector.
The reference wireframe cube uses the same
default extent-1 size as an imported model, with its bottom center on the plane.
The read-only **Model matrix** section is an advanced view.
**Export Arrow** writes the retained document. Editor transforms apply to the
selected object across all its existing sparse rows, not only the displayed
row. Each row stores `adjustment * original_local_model`; its K, quad, timing
and other objects remain unchanged. `track_id` identifies instances across
reordering or gaps. Without it, cross-frame editing requires a consistent
ordered object/binding list; ambiguous correspondence is rejected.
Dragging replaces the current adjustment against captured original matrices,
so it does not accumulate numerical drift on every pointer update.
Fresh replay reads those saved per-row models directly, with identity UI
adjustments: seeking or loading must not apply the authoring adjustment twice.
The explicit API `setModel(row, object, ...)` remains a single-cell operation;
`SceneDocument::model_track` supplies the complete target set used by the editor.

The editor has separate **Open Video** and **Load Arrow** buttons, each with
local-file and HTTP(S) URL selection. Loading/replacing Arrow does not reopen
the video. Arrow may also be loaded before a video; it is validated against
the video timeline when one becomes available. Clearing the Arrow selection
and choosing **Unload Arrow** leaves the video running independently.
Playback can resume within the buffered video tail even after the reader reaches
EOF. At the end, Details reports `playing: false` and `ended: true`; seeking
back to a presented frame clears the ended state.

## Native video editor

Convert the real FIBA annotation before running the editor round trip; do not
replace it with a synthetic golden scene:

```powershell
uv run --with pyarrow python scripts\timeline_to_params.py `
  assets\videos\fiba\fiba-shot1.arrow -o output\fiba-roundtrip\fiba.params.arrow
uv run --with pyarrow python scripts\timeline_to_params.py `
  assets\videos\fiba\fiba-shot1.arrow -o output\fiba-roundtrip\fiba.dragon.arrow `
  --glb assets\meshes\glb\Meshy_AI_Dragon_0804104424_texture.glb --identity-model
```

The explicit conversion preserves the sparse source rows and unrelated columns,
converts row-major pixel K to FHC `k`, adds FHC `bottom_quads` without changing
the source corner order, and
widens `present_index` to the source sub-schema's int64. It preserves the original
`timestamp_us`; it does **not** relabel that FPS-derived value as real packet PTS.
The original annotation and MP4 remain unchanged. The same conversion without
`--identity-model` exercises adding a missing model column on edit.
FIBA's finite convex tracking quads may extend outside the video image. The
rendering subset preserves them instead of clamping their corners and changing
the placement reconstruction.

For placement acceptance, regenerate the annotation from the actual calibration
and matching video first; a pre-existing Arrow may name a different source
render. Then compare all rows against the repository's original Python method:

```powershell
uv run --with pyarrow python scripts\fiba_video_editing_bundle.py `
  --video "E:\Asset\Video\shot_0001.mp4" `
  --calibration assets\videos\fiba\per_frame_KVP_cube_best.parquet `
  --method 2VP_4510 -o output\fiba-case1\fiba.source.arrow
uv run --with pyarrow python scripts\timeline_to_params.py `
  output\fiba-case1\fiba.source.arrow -o output\fiba-case1\fiba.params.arrow
uv run --with pyarrow --with numpy python scripts\placement_reference.py `
  output\fiba-case1\fiba.source.arrow -o output\fiba-case1\reference.json
$env:TRD_FIBA_PARAMS = "$PWD\output\fiba-case1\fiba.params.arrow"
$env:TRD_FIBA_REFERENCE = "$PWD\output\fiba-case1\reference.json"
cargo test -p trd-placement --test fiba_placement -- --ignored --nocapture
```

The reference calls `normal_basis_from_quad` and `pose_from_quad` using original
K and quad order. The Rust comparison checks all four reprojected corners, the
origin, gizmo matrix and cube bottom-center anchor for all 222 rows. A cyclic corner
rotation is not harmless: it changes the first edge used to set basis and scale.
In the source editor, start with **Show quad**, then **click the quad** to select
and highlight it. Selection shows the local axes, plane grid and reference cube;
clicking empty image space deselects it. Visibility toggles remain available for
the selected reference. The existing internal `cube.obj` is parsed by the OBJ
loader and uses the same placement method as other Y-up models. Its bottom
center maps to the quad origin and its +Y follows the normal. No GLB or Arrow
OBJ resource is needed for Case 1.

**Show Coordinate** and **Show Plane** are independent controls; hover feedback
is separate from selection. The reference cube has light-blue edges, without
changing the green AABBs of ordinary models. The renderer stays at source
resolution while the full video image fits the available panel with its aspect
ratio intact. Pointer coordinates are mapped from that painted image rectangle,
excluding letterbox bars.

```powershell
cargo run -p trd-gui-video-editing -- `
  --document params.arrow --glb model.glb `
  --video video.mp4 --preview-width 1920

cargo run -p trd-gui-video-editing -- `
  --document params.arrow `
  --glb-mesh 00000000-0000-0000-0000-000000000001=model-a.glb `
  --glb-mesh 00000000-0000-0000-0000-000000000002=model-b.glb `
  --video video.mp4 --preview-width 1920
```

An already bundled `[params][mesh]` needs no `--glb` argument. Supplying both
bundled assets and additional byte resources is an explicit conflict.

Video is still an independent source: no inline video frame or mandatory
image link is added to the document. The existing player handles decoding and
seeking. Arrow source identity must correspond to that selected video; no PTS
is fabricated from FPS. General frame-identity mapping beyond the player's
existing index model remains part of integration acceptance.

Parquet application input has been removed. Offline calibration preparation
can still convert source data to Arrow separately.

## Rendering

`trd_placement::document_scene` resolves both adapters to the existing core
camera and scene types. FHC geometry is converted only in that view. Reference
mode draws a bottom-anchored cube and axes. For a quad-bound GLB, the existing
Python placement convention supplies the basis and half-edge scale; the asset's
AABB is proportionally fit to extent 1 with its bottom center at local zero.
`DEFAULT_PLACEMENT_EXTENT` is shared with the wireframe cube; the ordinary mesh
viewer's preview size is unchanged.
The final transform is `placement * edited_local_model * grounded_asset_base`.
The base is derived from immutable GLB geometry on both authoring and replay,
never accumulated into the exported matrix. `Renderer::with_assets` itself
uses identity bases, so placement is not applied twice. No-quad CG/CV scenes
retain their prior transforms. All placement math stays in `trd-placement`.

`trd-app` uses this same document reader/placement adapter and wakes its window
when input arrives. Both browser renderers expose document background references;
the viewer preloads them and disables compositing on rows with no reference,
rather than retaining the previous still. `examples/render.ps1` and `render.sh`
use `scripts/scene_to_arrow.py` to convert their existing OBJ/albedo demo inputs
offline into `[params][mesh]`. Camera/model arrays are retained; the OBJ preview
normalization is baked into the converted GLB. Inline `-FramesTable` /
`--frames-table` inputs are retired; use external `frame_path`/`frame_url` and
`--frames-base` instead.

The editable document path uses `document_scene_with_overlays` to assemble two
back-to-front scenes for `Renderer::draw_layers`: video plus quad fill, outline,
grid and coordinate axes first; meshes, reference cubes and their AABBs/gizmos
second. Quad guides never tint or cover model content. Meshes still share depth
within the foreground scene, and their AABBs remain on top. This avoids changing
the global primitive order used by ordinary CG/OBJ scenes.
Details is captured from that displayed frame's inspected instance and its
resolved mesh slot, including the actual GPU material/IBL/tone-map settings.
Selecting another instance does not reuse object 0's transform or material.

Internal OBJ loading and the original CG camera path remain intact. The GLB
importer's current capability limits still apply; this work does not silently
claim animation, skinning or arbitrary multi-primitive import support.

## Primary end-to-end cases

Run these on native and Chrome/wasm surfaces on both Windows and Linux:

| Case | Input | Acceptance |
|---|---|---|
| 1 | Params only | Click the quad to show highlight, local axes, plane grid and a cube whose bottom center is at the local origin; click away to deselect. |
| 2 | Params plus one mesh | Edit all corresponding sparse-frame models, play/pause/resume and seek across them; export, close the process and freshly reopen with the same video. Repeat playback/seeks and compare every saved model plus matched-frame rendering. No model appears in the untracked tail. |
| 3 | Params plus multiple mesh rows | Verify one bound model per quad, independent transforms/materials and exact source/resource retention. Play/seek before export and after fresh reopen; no asset swaps or stale selected-instance Details. |

Multiple meshes occupy rows of one `mesh_id`/`glb` table. These cases organize
the feature acceptance; the remaining L3 gates, CG/OBJ regressions and video/
large-file seek coverage still apply. The full planned evidence matrix is
recorded on #367 and #370.

The pixel snapshot gate now includes both crates:

```text
cargo test -p trd-core -p trd-placement --test golden_render -- --ignored
```

The original seven core regressions keep their expected PNGs and camera/draw
values. Three placement snapshots add the cases above, using Uffizi and the
same image-difference tolerance. The single-model case additionally requires
exact same-device pixels across export/reload. This headless gate does not
substitute for the visible native/Chrome round trips.

For the real editor case, open the converted FIBA params with its matching
`shot_0001.mp4` (1920x1080, 24 fps, 288 frames), edit a GLB on a tracked row,
export, and reload in **video-editing**. Compare the edited and reopened
rendering at the same tracked row and inspect the first/middle/last tracked
rows plus the video-only 222-287 tail. Converted source-document editing uses
the video frame's original render resolution, independent of window size/DPI.
