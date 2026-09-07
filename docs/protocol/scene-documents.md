# Editable params and GLB scene documents

This is the in-progress implementation on #370, based on #368. The final wire
version/fixture migration is not complete; the existing stream reader is not
yet retired. The agreed target is the
[render sub-schema](../../assets/schemas/trd-render-sub-schema.md).

## Retained input

`trd_core::SceneDocument` retains the original params schemas/RecordBatches and
optional mesh resources. A mesh row contains exactly `mesh_id` (Arrow UUID) and
`glb` (LargeBinary). The GLB bytes, including their material/texture payloads,
are kept unchanged on export.

`SceneDocument::read` reads `[params][mesh?]`. A source table using the minimal
FHC `bottom_quads`/`k` schema can be read without inventing trd metadata. When a
stream declares trd protocol metadata, it must match the currently supported
version. A legacy CG/CV params source keeps the existing camera fields and
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

The native and wasm video-editor loaders also accept scene documents. In source
mode, the shared **Source models** controls edit the selected row/instance and
**Export Arrow** writes the retained document. These controls do not currently
propagate an edit across a whole track; the UI states the row-local scope.

The editor has separate **Open Video** and **Load Arrow** buttons, each with
local-file and HTTP(S) URL selection. Loading/replacing Arrow does not reopen
the video. Arrow may also be loaded before a video; it is validated against
the video timeline when one becomes available. Clearing the Arrow selection
and choosing **Unload Arrow** leaves the video running independently.

## Native video editor

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
mode draws a centered cube and axes; GLB mode composes the placement origin
with each local model. `Renderer::with_assets` uses identity asset base matrices
so source placement is not silently preview-scaled or re-centered.

Internal OBJ loading and the original CG camera path remain intact. The GLB
importer's current capability limits still apply; this work does not silently
claim animation, skinning or arbitrary multi-primitive import support.

## Primary end-to-end cases

Run these on native and Chrome/wasm surfaces on both Windows and Linux:

| Case | Input | Acceptance |
|---|---|---|
| 1 | Params only | Show the quad outline, local axes and origin-centered wireframe cube together. |
| 2 | Params plus one mesh | Edit the active model matrix, export updated params, reload with the same GLB and compare the edited/reopened rendering at identical frames and camera/lighting settings. Unrelated columns and untouched models survive. |
| 3 | Params plus multiple mesh rows | Verify asset IDs, independent transforms and each GLB's material/textures; no swapped or missing assets. |

Multiple meshes occupy rows of one `mesh_id`/`glb` table. These cases organize
the feature acceptance; the remaining L3 gates, CG/OBJ regressions and video/
large-file seek coverage still apply. The full planned evidence matrix is
recorded on #367 and #370.
