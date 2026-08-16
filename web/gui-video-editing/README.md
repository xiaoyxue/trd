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

It may be **Arrow IPC or Parquet**: the container is sniffed from the bytes
(`PAR1` at both ends versus the Arrow IPC continuation marker), never from the
file name, so a URL without a useful suffix and a mislabelled file both work.
Parquet keeps schema key-value metadata, so the version and table-kind contract
is the same either way.

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

`probe.html` + `src/probe.ts` are a standalone check that the ranged MP4 reader
in `src/media/` opens this project's videos and lands on an exact frame —
locate `moov`, extract `avcC`, seek, decode, draw — before the editor's playback
path is rebuilt on it (#282). It imports none of the editor's code, so a failure
there costs nothing:

```sh
cd web/gui-video-editing
bun probe.html
```

Pick a local MP4, or decode one from a URL. `?url=…&seek=…` runs it without a
click, so a result is a command rather than a click-through.
`serve-documents.ts` serves a directory with the CORS and `Range` headers a
naive static server omits:

```sh
bun web/gui-video-editing/serve-documents.ts /path/to/videos --port 8090
```

The probe prints the track, the `description` (`avcC`) bytes, the frames it
delivered, and — the number to watch — how much of the file was actually
transferred.

### What `src/media/` does, and why

`byte-source.ts` gives random access over either a local `File` or a URL, so
nothing above it knows which it has. `mp4-video.ts` walks the top-level boxes to
find `moov`, reads exactly that one box, hands it to mp4box for the sample
tables, and then feeds `VideoDecoder` from the key frame at or before a target
time. Measured on a range-serving origin:

| Video | `moov` | To open | Seek |
|---|---|---|---|
| FIBA shot 1, 6.36 MiB | 4 KiB, at the **end** | 0.04 MiB (0.56%) | lands on the exact frame |
| 4K60 test clip, 11.79 GiB | 10 MiB, at the **front** | 9.75 MiB (0.08%) | `3600.000s` → pts `3600.0000s`, 12 MiB |

Both open in two requests: the head, then the `moov` its header sized. Results
worth keeping:

- An AVC decoder configured **without** `description` accepts the configuration
  and then emits neither frames nor an error.
- `moov` sits at either end depending on the muxer — the FIBA clip has it last,
  the 4K clip first — so neither "download it all" nor "stream it in order"
  works for both. Range reads driven by the box list do.
- Step through the box list by each box's recorded size. Scanning for the bytes
  `moov` also matches them inside sample data, and `totalSize - moovSize`
  assumes nothing follows `moov`, which is untrue of any file ending in `free`,
  `skip` or `mfra`.
- A seek must be driven by frames *delivered*, not samples *queued*: the decoder
  only reports what it skipped after decoding it, so a key frame seconds ahead
  of the target ends the loop before a single frame comes out.
- Clamp the target, and clamp it to **mp4box's** idea of the end. It refuses to
  seek past a duration taken from the last sample in *decode* order, which with
  B-frames is earlier than the last presented frame, and answers an out-of-range
  request with a meaningless offset rather than an error. Asking a 4727.966s
  video for 5011s then read 256 MiB and returned nothing; clamped, it returns
  the last frame after 0.22 MiB.
- `--virtual-time-budget` starves the WebCodecs output callbacks, so headless
  Chrome cannot check a decode that way — drive a real-time browser instead.
