# Video editing

`web/gui-video-editing` is the browser editor from #163. It places and edits 3D
objects on a tracked court quad while the external FIBA MP4 plays. Rust owns
Arrow, placement, state, picking, materials, and all WebGPU rendering;
TypeScript only opens/fetches browser resources and copies decoded video pixels.

## Data set

- video: local `shot_0001.mp4` (1920x1080, 24 fps, 288 frames); the
  reference Linux location is
  `/home/xiaoyxue/Asset/fiba-shot1/shot_0001.mp4`, while Windows can use any
  local path;
- calibration:
  `assets/videos/fiba/per_frame_KVP_cube_best.parquet`;
- selected method: `2VP_4510`;
- tracked frames: 0–221;
- video-only frames: 222–287.

The copyrighted MP4 stays local and uncommitted. Numeric calibration and
repository-owned catalog assets are versioned.

## Editing timeline

The editor does **not** read render protocol `0.0.6` directly. It has a separate
authoring document:

```text
trd.video_edit.version = 0.1.0
trd.video_edit.table.kind = timeline
```

Schema metadata records source name, MIME/codec, SHA-256, byte length,
dimensions, frame rate, frame count, and duration. The single timeline table
contains:

| Column | Type | Meaning |
|---|---|---|
| `video_frame_index` | `u32` | contiguous decoded frame index |
| `present_index` | `u32` | source parquet row; equal to frame index here |
| `timestamp_us` | `i64` | deterministic media timestamp |
| `k` | nullable `FixedSizeList<f32>[9]` | row-major OpenCV intrinsics |
| `placement_quad` | nullable `FixedSizeList<f32>[8]` | TL/TR/BR/BL pixels |
| `tracked` | `bool` | whether K/quad geometry is valid |
| `poster_bytes` | nullable `Binary` | encoded JPEG on row 0 only |

Every timeline row directly copies the parquet row with the same
`present_index`; there is no frame-zero propagation. The Rust decoder rejects
unsupported versions/table kinds, partial K/quad geometry, non-contiguous frame
indices, misplaced posters, and metadata/frame-count mismatches.

The initial timeline embeds no mesh/material/edit resources. The fixed catalog
is fetched from repository assets at runtime. Exporting edited state as a normal
`0.0.6` stream is still pending.

## Generate the document

Linux/Nix:

```sh
uv run --with pyarrow scripts/fiba_video_editing_bundle.py \
  --video /home/xiaoyxue/Asset/fiba-shot1/shot_0001.mp4 \
  --calibration assets/videos/fiba/per_frame_KVP_cube_best.parquet \
  --method 2VP_4510 \
  -o web/gui-video-editing/data/fiba-shot1.arrow
```

PowerShell 7 (Windows native):

```powershell
$video = 'C:\path\to\fiba-shot1\shot_0001.mp4'
uv run --with pyarrow scripts\fiba_video_editing_bundle.py `
  --video $video `
  --calibration assets\videos\fiba\per_frame_KVP_cube_best.parquet `
  --method 2VP_4510 `
  -o web\gui-video-editing\data\fiba-shot1.arrow
```

The generated Arrow file is ignored; regenerate it from the local MP4.

## Browser/media boundary

`HTMLVideoElement` owns demux, play/pause, seeking, and the media clock.
`requestVideoFrameCallback` reports the presented `mediaTime`; TypeScript wraps
the current image in a WebCodecs `VideoFrame`, copies RGBA, and sends it to Rust.

Rust validates the local filename/byte length and decoded
dimensions/duration, maps media time to `video_frame_index`, selects the Arrow
row, recomputes placement, and returns a composed RGBA image. HTTP(S) sources
use decoded dimensions/duration validation. There is no independent Arrow timer,
so video pixels and calibration rows do not drift.

The canvas starts blank until a local file or HTTP(S) URL is opened. Playback
then pauses on frame 0. Completing the embedded-poster/digest UX before video
selection remains follow-up work.

## Placement

`crates/trd-placement` is GPU-free and exposes quad reconstruction, placement,
axes, and outline matrices. K is row-major OpenCV; the result is converted to
the renderer's GL camera convention.

The object basis matches the Python/Olympic reference:

```text
object X -> e1
object Y -> e3
object Z -> -e2
```

Default catalog placement:

```text
size_factor = 0.24
offset_e1   = 1.3
offset_e2   = -1.7
lift        = 1.0
```

Each tracked frame computes:

```text
draw_model =
  current_quad_placement
  * persistent_object_local_transform
  * mesh_preview_normalization
```

Edits therefore persist while the object follows the current quad. Translation
can use either fixed quad directions (`e1/e2/e3`) or the rotated object-local
(`X/Y/Z`) basis.

## Catalog and lighting

The first slice contains:

- Coca-Cola can: OBJ + `can_around.jpg`;
- beer can: OBJ + diffuse texture;
- Dragon: embedded-texture GLB.

All PBR assets automatically bind `assets/envmap/uffizi-large.hdr`.
Coke/beer start with `metallic=0`, `roughness=0.35`. Dragon preserves its GLB
base-color, metallic-roughness, and normal maps and uses IBL without
direct/ambient light.

The tracking basis is currently recomputed from raw K/quad rows without
temporal smoothing. High-frequency normal/roughness maps and glossy IBL
reflections can make small pose jitter appear as material flicker, especially on
Dragon. The material is not reloaded per frame; pose smoothing is pending.

## Rendering and visibility

The editor uses `trd-core` `DrawableObject`s and isolated GPU submissions:

1. video `FramePlane` plus the paused quad/grid/axes;
2. placed mesh;
3. optional selection AABB.

GPU ID picking selects the mesh. Shared `trd-gui` controls edit transform,
render mode, Disney material, IBL, tone mapping, and overlays.

All editor gizmos are hidden during playback. The placed object remains visible
on tracked rows. Rows 222–287 hide both quad and object while the original video
continues.

## Build and run

All browser delivery surfaces share the Bun workspace. The GUI viewer and video
editor each generate `trd-gui` wasm into their own package directory; the editor
does not import `web/gui-viewer/pkg`. Generate `fiba-shot1.arrow` first, then run:

```sh
cd web
bun run --cwd viewer build:wasm      # stage the local trd-wasm file dependency
bun run --cwd gui-video-editing build:wasm
bun install --frozen-lockfile
bun run typecheck
bun run check
bun run build
bun run --cwd gui-video-editing dev
```

The same Bun commands run natively in PowerShell 7 on Windows; WSL and Nix are
not involved.

The editor defaults to port 8085. On the headless Linux host:

```sh
ssh -N -L 8085:localhost:8085 xiaoyxue@10.32.84.63
```

Open <http://localhost:8085> in a WebGPU browser and select the local MP4.

## Source map

| Path | Responsibility |
|---|---|
| `crates/trd-core/src/video_editing.rs` | versioned timeline decoder |
| `crates/trd-placement/src/lib.rs` | K/quad frame and placement math |
| `crates/trd-gui/src/video_editing.rs` | editor state, commands, UI |
| `crates/trd-gui/src/video_editing_renderer.rs` | wasm composition and picking |
| `web/gui-video-editing/src/main.ts` | thin video/file/resource byte bridge |
| `web/gui-video-editing/pkg` | editor-owned generated `trd-gui` wasm package |
| `scripts/fiba_video_editing_bundle.py` | timeline generator |

## Remaining work

- export `[mesh][texture?][frames][params]` protocol `0.0.6` with the
  Rust-computed `draw_model`;
- reload that export and prove equivalent placement/transforms;
- add multi-frame Python/Rust matrix and reprojection parity;
- complete pre-video poster/digest UX;
- add temporal pose smoothing;
- add automated browser coverage for local-video playback and editing.

PBR material state remains runtime asset state because render protocol `0.0.6`
does not serialize PBR fields.
