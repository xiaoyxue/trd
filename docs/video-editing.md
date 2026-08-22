# Video editing

`web/gui-video-editing` is the browser editor from #163. It places and edits 3D
objects on a tracked court quad while the external FIBA MP4 plays. Rust owns
Arrow, placement, state, picking, materials, and all WebGPU rendering;
TypeScript only opens/fetches browser resources and copies decoded video pixels.

## Contents

- [Data set](#data-set)
- [Editing timeline](#editing-timeline)
  - [Container sniffing](#container-sniffing)
  - [Sparse rows](#sparse-rows)
  - [Selection and placement state](#selection-and-placement-state)
- [Generate the document](#generate-the-document)
  - [The Parquet twin, and the parity test](#the-parquet-twin-and-the-parity-test)
- [Browser/media boundary](#browsermedia-boundary)
  - [Reader boundary](#reader-boundary)
  - [Ranged bytes](#ranged-bytes)
  - [Playback clock](#playback-clock)
  - [Probe page](#probe-page)
- [Placement](#placement)
- [Catalog and lighting](#catalog-and-lighting)
- [Rendering and visibility](#rendering-and-visibility)
  - [Layer order](#layer-order)
  - [Selection overlays](#selection-overlays)
- [Details and diagnostics](#details-and-diagnostics)
  - [Inspector sections](#inspector-sections) — **field reference**
  - [Frame-path traffic](#frame-path-traffic)
  - [Stable displayed facts](#stable-displayed-facts)
- [Build and run](#build-and-run)
  - [Native editor](#native-editor)
- [Source map](#source-map)
- [Remaining work](#remaining-work)

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
trd.video_edit.version = 0.2.0
trd.video_edit.table.kind = timeline
```

Schema metadata records source name, MIME/codec, SHA-256, byte length,
dimensions, frame rate, frame count, and duration.

### Container sniffing

**Rule: read Arrow IPC streams or Parquet from the bytes, not from the file
name.** The container is sniffed (`PAR1` at both ends versus the Arrow IPC
continuation marker), never taken from the file name — a URL need not carry a
useful suffix, and a mislabelled file is read for what it is.

Parquet carries schema key-value metadata, so the version and table-kind contract
above is identical either way. Both readers feed one decoder: the same rows in
either container produce the same document.

### Sparse rows

**Rule: the timeline table stores only frames with ad-placement quads.**
Everything else is ordinary video and is played as such:

| Column | Type | Meaning |
|---|---|---|
| `video_frame_index` | `u32` | decoded frame index; **strictly increasing, with gaps** |
| `present_index` | `u32` | source parquet row |
| `timestamp_us` | `i64` | deterministic media timestamp |
| `k` | nullable `FixedSizeList<f32>[9]` | row-major OpenCV intrinsics |
| `placement_quad` | nullable `FixedSizeList<f32>[8]` | TL/TR/BR/BL pixels |
| `tracked` | `bool` | whether K/quad geometry is valid |
| `poster_bytes` | nullable `Binary` | optional encoded JPEG, on the first row only |

Every timeline row directly copies the parquet row with the same
`present_index`; there is no frame-zero propagation. The Rust decoder rejects
unsupported versions and table kinds, partial K/quad geometry, frame indices
outside the video, out-of-order or duplicated indices, and a misplaced poster.
What it deliberately **allows** is a document that annotates almost nothing: a
frame with no row is looked up as `None` and rendered as plain video (#264).

**Shots are derived, not stored.** A shot is a maximal run of consecutive
annotated frames, so the run boundaries can never disagree with the rows. The
editor lists them in the left pane and jumps to a shot's first frame.

### Selection and placement state

**Rule: placement overlays are visible state, not hidden editor state.** Two
independent toggles govern what is drawn over an annotated frame, including
*during playback* — an annotated frame shows its quad as it plays past, and the
toggles are how that is turned off:

- **Show placement quads** — the quad outline itself;
- **Show gizmos** — the quad's local floor grid and basis axes.

They are separate because the questions are: the outline alone judges the quad
against the plate, the gizmos alone read the reconstructed basis. **Show
placement quads** starts on so the editable frames announce themselves; **Show
gizmos** starts off and follows selection, since a basis is only meaningful for a
quad you are working in.

Selecting a quad (clicking it) highlights the outline, washes its face, enables
the catalog and switches **Show gizmos** on; clicking away deselects it and
switches them back off. The toggle is flipped, not overridden, so it still
describes what is drawn and can be set by hand between clicks. Until a quad is
selected the **Object catalog** is greyed out and says why on hover — *"Select a
placement quad first"* (#329).

**A placed object and its quad are bound.** The object is authored in that quad's
frame (`draw_model = quad_placement * object_transform`), so while one is placed
the quad stays selected and its gizmos stay up, and clicks go to the object's id
pass. Editing an object whose basis had silently disappeared would be editing
blind. **Reset all** is what unbinds them.

**The document is optional.** Without one the editor is a plain player: the
timeline comes from the container (ffprobe natively, the `moov` box in the
browser) and the placement UI stays inert.

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

### The Parquet twin, and the parity test

Both containers decode through one code path, and
`the_real_document_decodes_identically_from_both_containers` pins that on the
real 222-row document rather than on hand-built rows. Both fixtures are
generated, so the test is `#[ignore]`d and **skips** when they are absent:

```sh
uv run --with pyarrow scripts/doc_fixtures.py -o /tmp/trd-doc
TRD_DOC_DIR=/tmp/trd-doc cargo test -p trd-core --lib video_editing -- --ignored --nocapture
```

```powershell
uv run --with pyarrow scripts\doc_fixtures.py -o $env:TEMP\trd-doc
$env:TRD_DOC_DIR = "$env:TEMP\trd-doc"
cargo test -p trd-core --lib video_editing -- --ignored --nocapture
```

`scripts/doc_fixtures.py` converts the generated Arrow document into the Parquet
twin plus one copy per codec, creating the output directory. Run the two steps
separately: the first needs the network the first time (`uv` fetches pyarrow),
and the second may be a cold build, so a stall is otherwise hard to attribute.
Success prints two `ok` lines and **no** `skipping:` line — a `skipping:` line
means the test found no fixture and asserted nothing.

`TRD_DOC_DIR` is where the Parquet fixtures are looked for (default: the
platform temp dir); the Arrow fixture defaults to its generated location in the
tree, resolved against the repository root rather than the crate directory a
test binary runs from. `TRD_DOC_ARROW` / `TRD_DOC_PARQUET` override the two
paths individually. The per-codec copies drive
`unsupported_compression_says_so_clearly`,
which pins that `snappy`/uncompressed read and that `zstd`/`gzip` are refused
with parquet's own "Disabled feature at compile time" — those codecs are C shims
and are left out so the crate keeps cross-compiling to wasm32.

## Browser/media boundary

### Reader boundary

**Rule: the browser owns no `<video>` element.** [mediabunny] demuxes and decodes
the MP4 behind the `FrameReader` seam (`src/media/frame-reader.ts`), and
`MediabunnyReader` is **the** browser media adapter.

Range reads, locating `moov`, feeding the demuxer, decoder configuration/reset,
key-frame catch-up and the end-of-stream drain are the library's job, not ours.
Do not extend the hand-written mp4box + `VideoDecoder` reader
(`src/media/mp4-video.ts`) — fix the mediabunny path instead.

The one part deliberately kept ours is the raw `moov` box walk (`locateMoov`),
because Rust reads the frame rate from it as a **rational**, which mediabunny
does not surface.

### Ranged bytes

**Rule: media cost is set by what is watched, not by file size.** Bytes arrive
through a `ByteSource` (`src/media/byte-source.ts`) that reads **ranges**, so a
local file and an HTTP(S) URL behave identically.

Opening a 218 GiB / 694,840-frame 4K MP4 over HTTP costs ~11 MiB and under two
seconds, with each deep seek a further few tens of MiB. A URL source therefore
needs `Accept-Ranges` **and** `Access-Control-Allow-Origin`; `serve-documents.ts`
is the local helper that sends both and streams its responses. Run it with
`--log` to read those numbers off the server rather than out of devtools — one
line per request, and the delivered bytes when each body ends.

The native side reads the same URLs through ffmpeg rather than `fetch`, and it
is stricter about HTTP framing: a server that ends the connection must send
`Connection: close`, or ffmpeg writes its `moov` seek onto the socket the server
already closed and never recovers (#326).

### Playback clock

**Rule: `VideoPlayer` is the media clock.** `VideoPlayer`
(`src/media/player.ts`) drives play/pause/seek over that reader. Each decoded
`VideoFrame` is handed to Rust as-is (`presentVideoFrame`) and copied
**GPU→GPU**, never downloaded to RGBA.

The pixels are already in GPU memory, and at source resolution the round trip
would cost ~99 MB a frame for 4K (#229). Details reports it as
`frame upload: 0 B`. Overlapping seeks — what dragging the scrubber produces —
coalesce to the last target rather than queueing.

Rust validates the local filename/byte length and decoded dimensions/duration,
maps media time to `video_frame_index`, selects the Arrow row, recomputes
placement, and returns a composed RGBA image. HTTP(S) sources use decoded
dimensions/duration validation. There is no independent Arrow timer, so video
pixels and calibration rows do not drift.

### Probe page

**Rule: use `probe.html` to isolate media faults.** `probe.html`
(`?url=&seek=&frames=`, `?scrub=t1,t2,…`, `?overlap=1`, `?reader=mediabunny`)
exercises that layer alone, reporting bytes read and where each seek landed — the
cheapest way to tell a media fault from an editor fault.

The canvas starts blank until a local file or HTTP(S) URL is opened. Playback
then pauses on frame 0. Completing the embedded-poster/digest UX before video
selection remains follow-up work.

[mediabunny]: https://mediabunny.dev/

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

### Layer order

**Rule: the editor renders as isolated `trd-core` submissions.** The layer order
is:

1. the video background frame plane (`Scene::background().frame`) plus the paused
   quad/grid/axes;
2. placed mesh;
3. optional selection AABB.

GPU ID picking selects the mesh. Shared `trd-gui` controls edit transform,
render mode, Disney material, IBL, tone mapping, and overlays. **Reset all** in
the left pane returns the quad selection, placed object, transform, material and
overlay toggles to their opening state while keeping the video and document.

The basis arms are labelled `e1`/`e2`/`e3` at their tips. Those labels are the
one overlay drawn as **egui text over the image** rather than as scene geometry:
`trd-core` draws lines and triangles and has no glyphs, so labelling in the
render pass would mean adding a font atlas.

The positions are still Rust's — each tip is projected through the same `K` the
pass uses — so the text tracks the arm instead of being placed by eye.

### Selection overlays

**Rule: quad overlays follow their own toggles during playback too.** The quad
outline and the gizmos follow their own **Show placement quads** /**Show gizmos**
toggles.

Hovering a quad and selecting it both add a `QuadFill` — a translucent green wash
over the quad's face — and selection additionally turns the outline yellow and
switches **Show gizmos** on. Clicking off the quad deselects it and switches them
back off — unless an object is placed, which binds the two: its quad stays
selected and its basis stays visible while it is edited.

The placed object does not depend on that selection and remains visible on
tracked rows. Rows 222–287 have no annotation, so quad, gizmos and object are all
absent while the original video continues.

## Details and diagnostics

**Rule: Details reports one immutable Rust-calculated snapshot.** The left pane's
collapsed **Details** inspector is shared by browser and native delivery
surfaces. UI code reads one immutable `VideoEditingDiagnostics` snapshot;
tracking and scene facts are calculated in Rust rather than reconstructed in
TypeScript or directly in egui widgets.

### Inspector sections

Six sections, in the order the panel draws them. Every row is a *fact*, not a
control — Details never changes anything.

Values that a document can disagree with the file about are rendered as
`expected … / observed …` followed by **`[MATCH]`** or **`[MISMATCH]`**: the
first comes from the authoring document, the second from probing the file.
Without a document there is nothing to compare, so only the observed value
appears.

#### `[Source]` — which file this is, and whether it is the one expected

| Field | What it says |
|---|---|
| `kind` | `local file` or `URL` — how the video was opened |
| `name` | file name, checked against the document |
| `byte length` | size in bytes, checked against the document |
| `declared MIME` / `declared codec` | what the container declares (e.g. `video/mp4`, `h264`) |
| `dimensions` | the coded size, checked against the document |
| `FPS` | frame rate as an exact **rational** (`24/1`, `30000/1001`) — never a rounded float, because the timeline numbers frames with it |
| `frame count` | samples the video track **stores** |
| `unpresented tail` | samples stored but **never shown** — `none`, `N sample(s) (AV_PKT_FLAG_DISCARD)`, or `unknown`. A recorder stopping mid-interval writes a final sample outside the presentation window; it is counted in `frame count` but no decoder outputs it, so the last timeline index has no picture behind it (#324). `unknown` means *not checked*, which is deliberately distinct from `none` — natively it is not probed for URL sources, because a second remote probe would double the cost of opening one (#326) |
| `duration` | container duration, checked against the document |
| `SHA-256` | the document's digest of the file it was authored against |
| `digest` | whether that digest has been verified here — `not browser-verified yet` is the normal state, not an error |
| `media readyState` | the media element's readiness (`4` = enough data to play through) |
| `loaded` / `playing` / `ended` | player state |
| `error` | the last reported error, prefixed with the path that produced it (`media` / `catalog` / `document` / `render` / `pick`), or `none`. Each path clears only its own, so a decoded frame no longer erases a catalog failure (#329) |

#### `[Timeline / synchronization]` — which frame is where

Four frame identities are reported separately **on purpose**: they diverge while
work is in flight, and collapsing them would hide exactly the faults this panel
exists to catch.

| Field | What it says |
|---|---|
| `media time` | the displayed frame's presentation time, carried with the frame rather than read from the player |
| `frame duration` | how long the displayed frame is shown, as the **container** declares it — `0.042 s` at 24 fps. Variable-rate files differ frame to frame, which is what several seek faults turned on (#317/#319/#322) |
| `requested frame` | what the timeline last asked for |
| `presented frame` | the newest frame the shell has handed over |
| `displayed frame` | the frame whose pixels are on screen |
| `rendered frame` | the frame the last completed render drew |
| `Arrow video_frame_index` / `present_index` / `timestamp_us` | the document row for the displayed frame, or `none` where the sparse document has no row |
| `media/row delta` | disagreement between the frame's own time and its document row |
| `tracking state` | `tracked`, `video-only`, or `none` (no row) |
| `source size` / `render size` | decoded frame size, and the size trd renders at (`--preview-width` scales the native decode down) |
| `source generation` | bumped whenever the video is replaced; stale frames and picks are rejected by it |
| `render revision` | bumped by every scene change, so a completed render for an older scene is discarded |
| `pending render` | a render the scene has asked for but not yet started |
| `in-flight frame` | the frame a running render is drawing |
| `coalesced frame` | a newer frame that arrived while a render was in flight and replaced an older waiting one |
| `last render` | wall time of the last completed render |
| `seek target` / `seek pending` | the seek still waiting for its frame. A seek retires when the frame answering **that seek** arrives, not on a timestamp comparison (#322) |

#### `[Tracking / quad frame]` — the raw tracking data and the basis built from it

| Field | What it says |
|---|---|
| `TL` / `TR` / `BR` / `BL` | the document's placement quad, in pixels, exactly as authored |
| `K (fx, fy, cx, cy)` | camera intrinsics for this frame |
| `origin` / `e1` / `e2` / `e3` | the 3-D quad frame reconstructed from the quad and `K` — origin plus the two in-plane axes and the normal |
| `half-edge lengths` | the quad's half-extents along `e1`/`e2` |
| `axis length` | the gizmo arm length drawn for this basis |
| `\|dot(e1,e2/e3), dot(e2,e3)\|` | orthogonality residuals — all three should be ~0 |
| `handedness determinant` | `+1` for a right-handed basis; a sign flip means the normal inverted |
| `previous tracked frame` | which row the deltas below are measured against |
| `pose translation delta` / `pose rotation delta` | **unsmoothed** motion since that row — the raw tracking signal, so jitter is visible rather than hidden |
| `axis-length ratio` | scale change against the previous row |
| `continuity` | whether the normal's sign stayed continuous |
| `placement error` | why a quad could not be reconstructed, or `none` |
| `tracking smoothing` | `off` — stated so nobody assumes the deltas are filtered |

#### `[Placement / object]` — what is placed on the quad and where

| Field | What it says |
|---|---|
| `selected quad` | whether a placement quad is selected (the **Object catalog** is disabled until it is) |
| `selected object` | index of the selected object, or `none` |
| `catalog asset` / `source format` | which catalog entry is loaded and what it was imported from (`OBJ`, `GLB`) |
| `preview AABB min` / `max` / `preview scale` | the imported mesh's bounds and the scale fitted to the quad |
| `Olympic preset` | the fixed placement preset's parameters |
| `object translation` / `rotation` / `scale` | the local edit applied on top of the tracked pose — persistent across frames |
| `movement basis` | which axes dragging moves along (`quad e1 / quad e2 / quad e3`) |
| `visibility` | why the object is or is not drawn — `tracked`, `no asset`, `untracked tail`, `playing` |
| `draw_model` | the final 4×4 model matrix, copyable |

#### `[Material / lighting]` — how it is shaded

| Field | What it says |
|---|---|
| `render mode` | `filled`, `wireframe`, `textured`, or `PBR` |
| `imported metallic` / `imported roughness` | factors the asset itself declared, kept separate from the sliders so an import can be told from an edit |
| `base-color map` / `metallic-roughness map` / `normal map` | which texture maps the import supplied |
| `metallic` / `roughness` / `specular` / `clearcoat` | the current material |
| `IBL` | the environment map in use, or `none` |
| `IBL gain (object x scene)` | the object's gain times the scene's |
| `environment yaw` | environment rotation — the one value that drives both the visible sky and the reflections (#182) |
| `direct light / ambient` | analytic light and ambient terms |
| `exposure` / `tone map` | tone mapping (`reinhard` or `aces`) |
| `PBR debug` | the debug view (`shaded`, `roughness`, `metallic`, `normal`) |

#### `[Renderer]` — the device and what it did this frame

| Field | What it says |
|---|---|
| `adapter` / `backend` / `device type` | the GPU actually chosen — confirms a discrete card rather than a software fallback |
| `source size` / `render target` | decoded size, and the texture size rendered into |
| `mode` | the render mode the last completed render used |
| `frame-path crossings` | how many times full-resolution image data crossed CPU↔GPU (see below) |
| `frame upload` / `readback` / `ui upload` / `total / frame` | the bytes behind that count |
| `MSAA` | sample count of the mesh pass |
| `drawables (background/foreground/selection)` | how many drawables each layer submitted |
| `frame texture upload` | bytes uploaded for the video frame itself |
| `pick target` / `latest pick` | the pick render target and the last pick result |
| `last render error` / `last pick error` | explicit failures, or `none` |

### Frame-path traffic

**Rule: frame-path traffic counts only full-resolution image data.** The renderer
section reports the frame's **frame-path CPU↔GPU traffic** — `frame-path
crossings` plus the bytes for `frame upload`, `readback`, `ui upload`, and their
total.

The copy count is therefore *observed*, not asserted: each figure is written at
the transfer site itself, and the crossing count is derived from the bytes, so a
path that stops copying reports `0 B` and one crossing fewer because that code
did not run.

The scope is deliberately narrow — full-resolution image data only. Per-frame
uniforms and egui's own tessellated geometry and font atlas still cross the
boundary every frame and are not counted, so a `0` reads as *no frame-sized
buffer crossed*, not as *nothing crossed*. Today's shared path is 3 crossings;
binding the rendered texture directly into egui is what drives `readback`/
`ui upload` to `0 B` (#229).

### Stable displayed facts

**Rule: Details describes the image on screen, even while newer work is in
flight.** Completed renders retain the exact scene/material/asset/renderer facts
used to produce their pixels. While a newer frame or scene revision is in flight,
Details continues to describe the image on screen and separately reports the new
pending/presented identities.

Diagnostics JSON is serialized only when **Copy diagnostics JSON** is pressed.
The Dragon view makes its metallic factor, metallic-roughness and normal maps,
zero direct/ambient light, Uffizi IBL, and unsmoothed tracking inputs visible
without log inspection.

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

### Native editor

The native delivery surface replaces the browser's mediabunny reader with ffprobe
plus an
ffmpeg raw-RGBA stream while hosting the same Rust `VideoEditingApp`:

```sh
cargo run -p trd-gui-video-editing -- \
  --document web/gui-video-editing/data/fiba-shot1.arrow \
  --video /path/to/shot_0001.mp4
```

Use `--video-url https://example.com/shot_0001.mp4` instead of `--video` to
launch directly from HTTP(S); the two options are mutually exclusive.

Add `--probe-only` to validate metadata and decode frame 0 without opening a
native window.

`--preview-width` (default **960**, accepted range **1–1920**) is the width
ffmpeg scales the streamed preview frames to; the height follows from the
source aspect ratio. It is clamped to the source width, so it only ever scales
*down* — passing a value larger than the video does nothing.

This flag is **native-only, and it is a decode-cost lever rather than a
rendering setting.** The browser surface has no equivalent: mediabunny hands
back full-resolution `VideoFrame`s that never leave the GPU, so the two
delivery surfaces show the same scene at different effective video resolutions
unless `--preview-width` is set to the source width. Lower it to make native
playback and seeking cheaper, raise it toward the source width to match what
the browser displays:

```sh
cargo run -p trd-gui-video-editing -- \
  --document web/gui-video-editing/data/fiba-shot1.arrow \
  --video /path/to/shot_0001.mp4 \
  --preview-width 1920
```

```powershell
cargo run -p trd-gui-video-editing -- `
  --document web\gui-video-editing\data\fiba-shot1.arrow `
  --video C:\path\to\shot_0001.mp4
```

The native shell validates filename/size/dimensions/frame count and feeds
decoded frames into the shared editor state. Its panels, timeline, quad
selection, catalog, transforms, PBR/IBL controls, GPU picking, and layered
composition are the same Rust implementation used by the browser. ffmpeg
outputs RGBA directly to a Rust decoder thread/channel; no temporary frame
files are created. Native **Open video** uses the operating-system file picker
for local MP4s; HTTP(S) URLs are passed directly to ffprobe/ffmpeg and validated
against the document's codec, dimensions, frame count (when reported), and
duration before playback.

## Source map

| Path | Responsibility |
|---|---|
| `crates/trd-core/src/media/video_document/` | versioned timeline decoder (`trd.video_edit 0.2.0`, Arrow or Parquet) |
| `crates/trd-core/src/media/video.rs` | `VideoTiming` / `VideoInfo` — what a clip is, from either source |
| `crates/trd-core/src/media/mp4_probe/` | `moov` walk for the container's own timeline (#264) |
| `crates/trd-core/src/media/arrow_columns.rs` | Arrow column/metadata accessors for the document |
| `crates/trd-placement/src/lib.rs` | K/quad frame and placement math |
| `crates/trd-gui/src/video_editing/mod.rs` | editor state and typed scheduler |
| `crates/trd-gui/src/video_editing/editing_ui.rs` | editor panels, quad/catalog wiring, player footer |
| `crates/trd-gui/src/video_editing/diagnostics.rs` | immutable Details snapshot + pure calculations |
| `crates/trd-gui/src/video_editing/details_ui.rs` | Details inspector presentation |
| `crates/trd-gui/src/video_editing_renderer.rs` | shared native/wasm composition and picking |
| `web/gui-video-editing/src/main.ts` | thin video/file/resource byte bridge |
| `web/gui-video-editing/src/media/` | mediabunny reader, ranged byte source, player (the `FrameReader` seam) |
| `crates/trd-wasm/src/gui.rs` | the browser bridge (`VideoEditingHandle`) and JS ABI |
| `web/gui-video-editing/pkg` | editor-owned copy of the generated `trd_wasm` package |
| `native/trd-gui-video-editing` | native ffmpeg-backed host for the shared editor |
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
