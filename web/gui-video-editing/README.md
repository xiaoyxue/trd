# GUI video editing web

First vertical slice of #163.

Canonical user/developer documentation lives in
[`docs/video-editing.md`](../../docs/video-editing.md); this package README keeps
the local generation and launch recipe close to the bootstrap.

The editor loads an **optional**, separate `trd.video_edit.version = 0.2.0`
authoring document. It is **sparse**: a row exists only for a frame that carries
an ad-placement quad — for FIBA shot 1 that is frames 0-221 of 288, so the
222-287 tail has no rows at all and simply plays. Each row contains:

- `video_frame_index` and source `present_index` (strictly increasing, with gaps);
- `K`, placement quad, and tracked state;
- an optional encoded poster, on the first row only;
- video identity/size/fps/count/digest in schema metadata.

Without a document the editor is a plain player: the timeline comes from the
video container and the placement UI stays inert. With one, the left pane lists
the derived **shots** (runs of consecutive annotated frames), jumps to a shot's
first frame, and offers a **Show placement overlay** toggle that also governs
whether quads are drawn during playback.

Each Arrow line copies `K` and `ad_quad` directly from the parquet row with the
same zero-based `present_index`; no additional quad homography is applied.
Rust renders that row's quad/grid/axes in the GPU background pass using the
shared analytic-AA gizmo pipeline (1.5 px quad stroke); no separate egui
screen-space transform is applied to the overlay.

The initial document contains no 3D model resources. After a user selects a
quad, chooses an asset, and edits it, Rust will compose the final model matrix
and export the normal render protocol `0.0.6` stream:

```text
[mesh] [texture?] [frames] [params]
```

PBR material state remains attached to the imported/catalog asset in this
simple slice because protocol `0.0.6` does not serialize PBR material fields.

## Generate the FIBA document

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

## Run

Requires a **WebGPU-capable browser** (Chrome/Edge 113+, or Firefox with WebGPU
enabled): egui itself now runs on WebGPU through `egui-wgpu`, so there is no
WebGL fallback — a browser without WebGPU fails outright rather than degrading
(#257).

```sh
cd web
bun run --cwd viewer build:wasm  # stage the workspace's trd-wasm file dependency
bun run --cwd gui-video-editing build:wasm
bun install --frozen-lockfile
bun run --cwd gui-video-editing dev
```

These Bun commands run natively in PowerShell 7 on Windows as well as under
Linux/Nix.

The native media/timeline counterpart is:

```sh
cargo run -p trd-gui-video-editing -- \
  --document web/gui-video-editing/data/fiba-shot1.arrow \
  --video /path/to/shot_0001.mp4
```

Open the URL printed by Bun. The canvas stays blank until **Open video...**
selects either a local file or an HTTP(S) URL. Choose the same
`shot_0001.mp4` used to generate the Arrow document (for example the reference
Linux path above or the `$video` path on Windows). It remains paused on frame 0.
Use **Play** or the full-width player timeline below the viewer to move through
the shot. The HTML video element owns playback; each presented frame is copied
through WebCodecs to RGBA and sent to Rust. Rust validates the source dimensions
and duration, maps media time to `video_frame_index`, selects the matching Arrow
row, and updates the frame and quad overlay inside egui.

The editor is locked to **Fit right pane**. It preserves the source video's 16:9
aspect ratio, centers the image with letterboxing when necessary, and resizes the
GPU video/mesh/gizmo composite target to the fitted image dimensions.

Click the green quad to select its Rust-reconstructed local coordinate frame,
then choose Coca-Cola can, beer can, or dragon from the left pane. Object
interaction, numeric transforms, render mode, PBR material, tone mapping, and
overlays use the shared `trd-gui` controls. Catalog meshes are centered and
normalized to the reconstructed quad scale, start at the Olympic-demo anchor
with their base on the plane, and move in quad-local coordinates. Translation
offers exactly one active direction from two mutually exclusive bases: fixed quad
`e1`/`e2`/`e3`, or the object's rotated local `X`/`Y`/`Z`. Click the rendered
object to select it through Rust's GPU ID pass; its selection box and local-axis
gizmo identify the active transform. Typed Rust `InteractionEvent`s update the
selected `ObjectTransform`, and Rust computes
`draw_model = quad_placement * object_transform`; JavaScript never computes
model matrices.

Playback follows the FIBA visibility policy: the complete placement quad is
hidden while playing, together with every object/world AABB, axis, and grid
gizmo. Tracked rows still render the placed object; rows 222–287 have
`tracked=false`, so both quad and object are absent while the original video
frames continue playing.

The initial catalog placement matches the Olympic demo's upper can:
`size_factor=0.24`, `offset_e1=1.3`, `offset_e2=-1.7`, `lift=1.0`. The
`[e1,e3,-e2]` basis/sign convention is identical to
`placement_quad_by_local_coord.py`.

Every catalog object automatically binds
`assets/envmap/uffizi-large.hdr` as its IBL probe. Textured Coke/beer assets
start in PBR with the printed-can preset (`metallic=0`, `roughness=0.35`).
Dragon disables direct/ambient lighting and uses only the Uffizi IBL probe.

Final playback is an explicit two-pass render. Pass 1 updates the current video
background plus standalone placement-quad gizmos; pass 2 loads that color and
renders the mesh. The material and edited object-local transform persist, while
each frame recomputes
`final = current_quad_basis(e1,e2,e3) * object_model * normalized_vertices`
using that Arrow row's calibration.

## WebCodecs decode probe

`probe.html` + `probe.ts` are a standalone check that mp4box.js and
`VideoDecoder` decode this project's MP4s — demux, extract `avcC`, decode, draw
one frame — before the editor's playback path is rebuilt on them (#282). It
imports none of the editor's code, so a failure there costs nothing:

```sh
cd web/gui-video-editing
bun probe.html
```

Pick a local MP4, or decode one from a URL. `serve-documents.ts` serves a
directory with the CORS and `Range` headers a naive static server omits:

```sh
bun web/gui-video-editing/serve-documents.ts /path/to/videos --port 8090
```

The probe prints the track, the `description` (`avcC`) byte count, and the first
decoded frames. Two results worth keeping: an AVC decoder configured **without**
`description` accepts the configuration and then emits neither frames nor an
error, and our MP4s carry `moov` at the *end*, so streaming one needs range
reads that locate `moov` first rather than a plain in-order feed.
