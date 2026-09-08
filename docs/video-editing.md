# Video editing

Native and browser video editing share one Rust-owned **protocol 0.0.7 params/GLB** editor.
There is no old annotation/catalog interface, automatic old-format conversion,
or implicit FIBA document on startup. Open Video and Load Arrow are independent.
The browser always uses [mediabunny]; native uses ffmpeg/ffprobe.

The rendering/editing contract is [protocol 0.0.7](protocol/0.0.7.md), with
[editable document APIs](protocol/scene-documents.md). Remaining runtime/
generator stamps are listed in the
[atomic migration status](protocol/README.md#implementation-migration-status);
the old annotation schema is not the current editor protocol.

## Data set

The FIBA example uses an external `shot_0001.mp4`: 1920x1080, 24/1 fps, 288
frames. The original calibration is
`assets/videos/fiba/per_frame_KVP_cube_best.parquet`, method `2VP_4510`.
Its 222 tracked rows cover 0-221; 222-287 is video-only. The MP4 stays external
and uncommitted. Preserve the original source annotation and calibration.

## Editing timeline

Applications accept `[params][mesh?]`, each bracket a complete Arrow IPC stream.
The optional mesh table has only UUID `mesh_id` and original `glb` bytes.
GLB supplies geometry, textures and material; renderer slots never replace
source UUIDs. Params alone show a grounded reference cube.

### Container sniffing

Read Arrow from its bytes, not its filename. Parquet, legacy
`trd.video_edit.version = 0.2.0` annotation input and old mesh-first scene input
are rejected by the editor. The error for an annotation names the offline
converter; the runtime does not silently migrate or open a different UI.

### Sparse rows

Existing source frame identities select the row for a presented video frame.
A missing row means video only, not a fabricated empty placement.
The tracked-source adapter preserves FHC `bottom_quads`, `k`, optional
`present_index`/`pts`/`track_id`, and row-major nested model matrices.
The separate CG/CV camera adapter keeps its original conventions.
See the [render sub-schema](../assets/schemas/trd-render-sub-schema.md).

The old annotation schema remains documented as **offline source data** in
[`video-editing.schema.json`](video-editing.schema.json); its decoder and
calibration tools do not make it an accepted editor input.

### Selection and placement state

The current controls are **Show quad**, **Show Coordinate**, **Show Plane**
and, for params-only input, **Show reference cube**. Hover/selection feedback
is separate from those toggles. Pick a mesh to select its owning instance.
Translate uses quad directions in order **e1**, **e2**, **e3**; Rotate and Scale
adjust its local model. No catalog or derived-shots panel is presented.

Edits apply to the selected object across all corresponding existing sparse
rows. Each row retains its own camera and quad. `track_id` survives reordered
instances and gaps; without IDs, consistent ordered bindings are required.
Ambiguous correspondence is an error, not permission to edit another object.

## Arrow scene export and round-trip

**Export Arrow** writes the retained params and original GLB resources:

```text
[params] [mesh?]
```

Every affected row stores `adjustment * original_local_model`. Repeated pointer
updates replace that adjustment against captured baselines; they do not multiply
against the previous drag result. Export retains unrelated columns, nulls,
schema/field metadata, row order, params/resource batch boundaries and GLB bytes.
No MVP matrix or decoded video pixels are stored.

Fresh replay renders the saved matrices directly. UI adjustments start at
identity: loading, playing or seeking must not apply the original edit twice.

### Canonical acceptance workflow

1. Load the matching video and current params/GLB input. Edit translation,
   rotation and scale; play, pause/resume and seek through several tracked rows.
2. Export through the actual UI and audit **all corresponding sparse-frame
   matrices**, not just the currently displayed row. Unrelated data must survive.
3. Close the case and verify its owned PIDs/ports are clear. Freshly launch with
   the exported Arrow and the same video.
4. Repeat real playback and seeks, including first/middle/last tracked frames,
   the video-only tail and return to an edited frame. Compare saved matrices,
   matched-frame rendering and fresh Details identity.

For FIBA, the single Dragon edit must persist in all 222 rows. Synthetic
multiple-quad binding fixtures are explicitly separate from that genuine
single-quad example; shifted test quads must not be called original tracking.
Screenshots are only for named visual milestones. Use Copy details and Arrow
audits between them, not screenshots after each action.

## Generate the document

First generate or retain annotation from the matching calibration/video:

```powershell
uv run --with pyarrow python scripts\fiba_video_editing_bundle.py `
  --video "E:\Asset\Video\shot_0001.mp4" `
  --calibration assets\videos\fiba\per_frame_KVP_cube_best.parquet `
  --method 2VP_4510 -o output\fiba.source.arrow
```

Then convert explicitly, without overwriting that source:

```powershell
uv run --with pyarrow python scripts\timeline_to_params.py `
  output\fiba.source.arrow -o output\fiba.params.arrow
uv run --with pyarrow python scripts\timeline_to_params.py `
  output\fiba.source.arrow -o output\fiba.dragon.arrow `
  --glb assets\meshes\glb\Meshy_AI_Dragon_0804104424_texture.glb --identity-model
```

The converter preserves corner order and sparse identities, converts camera
representation explicitly, and does not relabel FPS-grid timestamps as real PTS.

### The Parquet twin, and the parity test

Parquet calibration is offline input, not an alternative application document.
Current decoder parity compares fragmented native `SceneDocument::read_from`
with browser-buffer `SceneDocument::read`. The original schema fixture remains
available for offline conversion; no runtime Parquet compatibility is implied.

## Browser/media boundary

### Reader boundary

The editor has no `<video>` element. [mediabunny] demuxes/decodes behind
`FrameReader`; `MediabunnyReader` is the application adapter, not an opt-in
URL mode. The old mp4box reader is not selected by the editor bootstrap.
Do not extend it for new application media work.

The raw `moov` walk remains because Rust needs the container's rational frame
rate. Byte fetching, decoder configuration, keyframe catch-up and drain belong
behind the reader boundary, not in rendering code.

### Ranged bytes

HTTP video must support byte ranges and CORS. The existing helper streams
responses and logs delivered bytes, not merely announced Content-Length:

```powershell
bun web\gui-video-editing\serve-documents.ts "E:\Asset\Video" --port 8092 --log
```

The acceptance target for a 218 GiB recording is roughly 11 MiB/<2 s to open.
Measure each current implementation rather than assuming that benchmark from a
successful short clip. Native ffmpeg can prefetch more than the browser and
must report its own measured total. Do not mask excess reads or change tolerances.
Servers closing connections must announce `Connection: close`.

### Playback clock

`VideoPlayer` owns the browser pacing clock, pause/resume, lookahead and seek
generation. VideoFrames pass directly to Rust for GPU-to-GPU upload.
Queued tail frames remain playable after the reader reaches EOF. When playback
ends, Details reports playing=false/ended=true; seeking back clears ended.

Rust matches presented video identity to sparse source rows. K and placement are
selected with that same frame; there is no separate Arrow playback timer.
Native playback uses the last **presentable** sample rather than demanding a
discarded trailing container sample. Missing evidence is not guessed; unexpected
early decode termination remains an error.

### Probe page

Start both entrypoints explicitly:

```powershell
cd web\gui-video-editing
$env:BUN_PORT='8085'
bun .\index.html .\probe.html
```

The probe is `/probe`, not `/probe.html`. `seek` is in **seconds**.
Use `?reader=mediabunny&url=...&seek=...&frames=8` for a deep seek and
`?reader=mediabunny&url=...&scrub=t1,t2,t3&overlap=1` for coalescing.
Also exercise the actual editor at `/?document=none&video=...`; the probe alone
does not cover UI frame synchronization.

## Placement

`trd-placement` owns the GPU-free reconstruction and grounding. The object basis:

```text
object X -> e1
object Y -> e3
object Z -> -e2
```

The default asset/reference extent is 1 in quad half-edge units, with the AABB
bottom center at local zero. No Olympic preset offset/lift is added.
Each frame renders:

```text
placement_from_that_frames_camera_and_quad * saved_local_model * grounded_asset_base
```

The asset base is derived, never accumulated into exported models.

## Catalog and lighting

The old fixed catalog is removed from both editor delivery surfaces. Supply
the desired self-contained GLBs in the current document instead. Ordinary
OBJ/internal mesh viewers remain supported outside this editor input contract.
Models use `assets/envmap/uffizi-large.hdr` as the default IBL probe. Imported
GLB maps/materials remain authoritative; Dragon's mapped-material setup uses
zero direct/ambient light. Raw tracking is not temporally smoothed.

## Rendering and visibility

### Layer order

Two shared renderer layers keep depth-disabled guides below content:

1. External video plus quad fill/outline/grid/quad-coordinate axes.
2. Meshes/reference cubes and their own AABBs/gizmos.

Meshes share depth within the foreground. Global primitive ordering for ordinary
viewers is unchanged. Basis labels are Rust-positioned egui text, not another
JS rendering implementation.

### Selection overlays

Hover adds a translucent quad fill; selection highlights its outline. Meshes
remain visible on their tracked rows independent of selection. Missing sparse
rows contain no quad, model or reference gizmos, while the original video plays.
The video-only path must not overwrite any loaded mesh's material.

## Details and diagnostics

Copy details and the inspector share one row formatter. Values describe the
displayed render, not a newer pending request or object 0 when another is selected.

### Inspector sections

#### `[Source]` — which file this is, and whether it is the one expected

Reports source kind/name/bytes, codec, dimensions, rational fps, frame count,
duration, readiness, playing/ended and errors. An unpresented tail includes its
evidence (`AV_PKT_FLAG_DISCARD` or sample tables); unknown is not zero.

#### `[Timeline / synchronization]` — which frame is where

Requested, presented, displayed and rendered identities are deliberately
separate. A settled seek must agree across them with no pending render,
in-flight/coalesced frame or outstanding seek. Source-row identity remains
separate from a video's full container frame count.

#### `[Tracking / quad frame]` — the raw tracking data and the basis built from it

Shows the inspected instance's TL/TR/BR/BL corners, K, reconstructed origin,
half-edge lengths, basis quality and raw pose deltas where available. Camera
and quad values follow the same displayed source row.

#### `[Placement / object]` — what is placed on the quad and where

Names the selected instance, actual resolved asset bounds and displayed draw
model. UI transform values are adjustments; after reopening, identity controls
do not mean the saved Arrow model was reset.

#### `[Material / lighting]` — how it is shaded

Shows the inspected mesh's actual GPU material, imported map availability,
render mode, IBL gain/yaw, direct/ambient scale, exposure, tone map and debug
view. Multiple instances do not inherit object 0's diagnostic values.

#### `[Renderer]` — the device and what it did this frame

Reports adapter/backend/device type, target/picking sizes, MSAA and frame-path
transfer counts, with render/picking errors.

### Frame-path traffic

Counts full-resolution frame upload, readback and UI upload bytes at the actual
transfer sites. Shared-device native composition removes GPU readback/UI upload;
browser VideoFrame upload can remain GPU-to-GPU. Zero bytes is measured, not an
inference from architecture. Uniform/geometry updates are not frame-pixel traffic.

### Stable displayed facts

The displayed snapshot carries the inspected object and source frame. Newer
selection/seek requests cannot relabel an older image. GPU appearance data is
captured from the renderer that produced it; input schema values alone are
not proof of what the frame actually used.

## Build and run

```powershell
cd web\gui-video-editing
bun run build:wasm
$env:BUN_PORT='8085'
bun .\index.html .\probe.html
```

Open `http://localhost:8085/` for the current empty editor. Load Video and Arrow
through separate controls, or use `?document=<params-url>&video=<mp4-url>`.
Linux's headless server is reached through SSH port forwarding. Each package
uses its own copy of the one `trd-wasm` build.

### Native editor

```powershell
cargo run -p trd-gui-video-editing -- --document output\fiba.dragon.arrow `
  --video "E:\Asset\Video\shot_0001.mp4" --preview-width 1920
```

Use `--video-url` for HTTP. `--probe-only --probe-frame N` reports the frame
actually decoded, not just the one requested. ffmpeg streams RGBA without a
temporary frame directory. `--preview-width` scales down only; 1920 is required
for 1080p E2E authoring and replay. Windows Chrome uses native desktop DPI,
not forced CSS viewport/device metrics.

## Source map

| Path | Responsibility |
|---|---|
| `crates/trd-core/src/protocol/scene_document.rs` | current retained params/GLB document |
| `crates/trd-core/src/protocol/document_edit.rs` | atomic model edits and cross-frame identity |
| `crates/trd-core/src/media/video_document/` | offline legacy annotation domain/decoder |
| `crates/trd-core/src/media/mp4_probe/` | container timing and raw `moov` walk |
| `crates/trd-placement/` | reconstruction, grounding and layered scene assembly |
| `crates/trd-gui/src/video_editing/` | shared editor, model controls, export, Details |
| `crates/trd-gui/src/video_editing_renderer.rs` | native/browser composition and picking |
| `crates/trd-wasm/src/gui.rs` | browser delivery ABI |
| `web/gui-video-editing/src/main.ts` | current-only resource/media bootstrap |
| `native/trd-gui-video-editing/` | native ffmpeg/ffprobe delivery |
| `scripts/timeline_to_params.py` | explicit offline legacy annotation conversion |

## Remaining work

Complete current-revision Windows/Linux L3 and the atomic protocol 0.0.7
cutover before merge. Track temporal smoothing and pre-video poster/digest UX
separately. The acceptance matrix and exact handoffs live on #367/#370 and
[AGENTS.md](../AGENTS.md).

[mediabunny]: https://mediabunny.dev/
