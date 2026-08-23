# Architecture

`trd-core` is the *single* rendering core: the exact same Rust/wgpu code renders
in every front-end (headless CLI, native window, browser, interactive viewer) by
drawing into whatever render target each one provides. JavaScript/TypeScript is a
thin bootstrap only — the WebGPU API is never called from JS.

Everything shares **one render function** and one mesh-first **render format**.
The video editor additionally reads a separate versioned authoring timeline and
derives ordinary render scenes from it in Rust:

```
input-stream ─┬─ trd-cli  → trd-core → offscreen readback → image-stream   (headless)
(mesh-first)  ├─ trd-app  → trd-core → window surface                      (native playback)
              ├─ trd-wasm → trd-core → canvas surface                      (browser)
              └─ trd-gui  → trd-core → offscreen → egui image      (interactive, native + browser)

                  image-stream → scripts/encode.py → ffmpeg → GIF / WebP / MP4

video-edit timeline + VideoFrame RGBA
              → trd-placement + trd-gui → trd-core → egui image    (browser editor)
```

## Contents

- [The render core — `trd-core`](#the-render-core--trd-core)
  - [Renderer and render targets](#renderer-and-render-targets)
  - [External video frames](#external-video-frames)
  - [Drawable scene model](#drawable-scene-model)
  - [Scene background](#scene-background)
  - [Typed PBR domains](#typed-pbr-domains)
  - [CPU mesh domain](#cpu-mesh-domain)
  - [Arrow input and output](#arrow-input-and-output)
  - [Video metadata domain](#video-metadata-domain)
  - [Typed math](#typed-math)
- [The front-ends](#the-front-ends)
  - [Front-end roles](#front-end-roles)
  - [Browser delivery surface](#browser-delivery-surface)
- [Source layout](#source-layout)

## The render core — `trd-core`

Platform-agnostic wgpu logic, shared verbatim by every target.

### Renderer and render targets

**Rule: `Renderer` owns behavior; render targets are pure data.**
`Renderer` (`render/renderer.rs`) is the one persistent render harness: it
rasterizes a `Scene` of `DrawableObject`s into *any* `wgpu::TextureView`, owning
the GPU context, pipelines, materials, and mesh store, with the render **target**
passed as a plain per-call argument rather than a type parameter or an owned
field (#203).

`render/render_target.rs` holds only the resources a frame lands in:

| Type | Holds | Used by |
|---|---|---|
| `TextureTarget` | texture + padded staging buffer | `trd-cli`, `trd-gui`, browser `OffscreenRenderer` |
| `SurfaceTarget` | surface + config + sRGB view format | `trd-app`, `trd-wasm`'s `CanvasRenderer` |
| `RenderTarget` | closed enum over the two, holding just the discriminant | the public render entry |

**All** the behaviour is on the renderer, behind one match:
`Renderer::render(camera, scene, &mut RenderTarget)` is the single render entry,
dispatching to a private `render_surface` (acquire → encode through the sRGB view
→ submit → present) or `render_texture` (encode → submit). It is synchronous,
returning `Result<Option<SurfaceRepair>, RenderError>`.

The asymmetric tail is typed, not branched: `read_pixels`, plus the multi-camera
`draw_layers`/`render_layers`/`render_params`, takes the concrete
`&TextureTarget`, so asking a *surface* for pixels is a type error rather than a
runtime arm.

Creating, resizing, reconfiguring and replacing a target are `Renderer`
functions too: `create_texture_target`, `resize_texture_target`,
`resize_surface`, `reconfigure_surface`, and `replace_surface`. The surface ones
are *associated* functions taking a `&wgpu::Device`, because a window is resized
and repaired before the stream's mesh table has arrived to build a `Renderer`
from.

Live-surface shells are **not** renderers. They create the `wgpu::Surface`, own
the resulting `RenderTarget`, and apply their own recovery policy to the
`Ok(Some(SurfaceRepair))` / `Err(RenderError::Surface(_))` the harness reports:
the native window defers to the next redraw; the browser reconfigures or
recreates the surface in-call and retries once.

Gizmo segments use `gizmo_line.wgsl`: the vertex stage expands each model-space
segment to a configurable pixel-width quad and the fragment stage feathers its
rectangle distance, so axes/AABBs/grids remain anti-aliased without MSAA. Axis
cone tips reuse the unlit triangle path.

### External video frames

**Rule: only the copy crosses the browser boundary (#302).** The background frame
plane normally takes bytes (`update_frame_texture_rgba`), and bytes are bytes on
every platform.

A browser frame is the exception: `Queue::copy_external_image_to_texture` is
`#[cfg(web)]` in wgpu, so the *copy* cannot be compiled into a crate that also
builds natively. `trd-core` owns the trait, allocates the destination texture,
and keeps its format/usage invariants (`Renderer::update_frame_texture_external`
→ `FramePlane::copy_external`).

The delivery surface that decoded the frame implements two methods —
`ExternalFrame::{size, copy_into}`. `crates/trd-wasm`'s `BrowserVideoFrame` is
the only implementor, over a WebCodecs `VideoFrame`, and closes it on `Drop`.

So **no shared crate names a browser type**: `trd-core` and `trd-gui` have no
`web-sys` dependency, `FrameSource::External` is a plain enum variant that native
simply cannot construct, and the rule is asserted by
`crates/trd-wasm/tests/wasm_bindgen_containment.rs` rather than remembered.

### Drawable scene model

**Rule: every visible primitive is a placed `DrawableObject` (#41).** A
`DrawableObject` is a small `Copy` struct pairing a `Primitive` with the `model`
that places it, so every primitive can be instanced.

| Primitive variant | Meaning |
|---|---|
| `Mesh { mesh_id, mode }` | mesh geometry, including filled / wireframe / textured / **PBR** modes |
| `AabbBox { mesh_id }` | a mesh-aligned bounding box |
| `CoordinateAxes` | world or local axes |
| `PlaneGrid { plane }` | grid on a selected plane |
| `QuadOutline { selected }` | video-editing placement outline |
| `QuadFill` | translucent placement face fill |
| `BlobShadow` | simple contact shadow |

Geometry is owned once (decode-once mesh store + shared line-quad/arrow buffers).
A drawable is a light handle naming *which* primitive + its per-frame model.

A `Scene` (an object list plus its `Background`) is rebuilt each frame; every
front-end hands it to `Renderer::render` without per-type branching. The render
core walks its objects into a flat list, batches by primitive — a batch key is a
drawable minus its model, so the same taxonomy serves both (#204) — binds the
shared `P·V` camera uniform (plus viewport size for gizmo lines), and records the
draws in `Primitive::sort_key` order, which is the frame's z-order because every
overlay pipeline disables depth.

### Scene background

**Rule: background is a per-frame scene setting, not a primitive (#204).**
`Background { environment: Option<EnvironmentBackground>, frame: Option<FrameFit> }`
holds the camera-centered HDR environment probe and the fullscreen background
frame plane (#63).

The two slots are **independent** — a frame may draw both — and the renderer
always draws the environment first, then the frame plane, then the mesh scene
over them. Appearance (filled / wireframe / textured / **PBR**) is a *mode* of
the mesh drawable, not a separate primitive.

<details>
<summary>Why it works this way</summary>

Both were `DrawableObject` variants, and were the only two carrying no model and
the only two the batcher had to skip; as scene settings they are set once, with
no ordering or duplicate to get wrong.
</details>

### Typed PBR domains

**Rule: PBR concepts live in typed domain modules, not in one catch-all path.**

| Domain | Location |
|---|---|
| Disney surface parameters and preserved glTF auxiliary data | `material/disney.rs` |
| Analytic lights and rig controls | `light.rs` |
| HDR environment data + its CPU precompute | `render/env_map.rs` |
| Environment binding and sky pipeline | `render/environment.rs` |
| Per-object output transform | `render/tonemap.rs` |

Materials and lights sit at the crate root because a material and a light are
universal domain vocabulary (#223). `pbr.rs` contains only the unchanged
shader-uniform packing and smooth-normal derivation. `trd-core`'s boundary-level
`mesh/gltf.rs` parses caller-owned bytes into these types without entering the
render hot path or performing filesystem I/O.

### CPU mesh domain

**Rule: `mesh/` is device-free CPU geometry (#221).** The crate root holds the
universal domain vocabulary (a mesh, a material, a texture, a camera), so the
canonical `Mesh`/`MeshShading` container lives in `mesh/mesh.rs`.

Loaders sit beside it by format:

- `mesh/obj.rs` (#36)
- `mesh/arrow.rs` (#37)
- `mesh/gltf.rs`

The geometry every source shares (`aabb`, `center`, `preview_transform`,
`edge_indices`) lives in `mesh/mod.rs`. A mesh's GPU residency is its face in
`render/mesh_store.rs`, and the `Vertex` layout it is written in stays with the
other `repr(C)` + `Pod` types in `render/gpu_types.rs`.

### Arrow input and output

**Rule: a type that owns a transport is a `*Stream`; one that owns none is a
`*Session` (#296).** The format logic stays in `protocol/`; transport wrappers
live in `io/`.

Input path:

- `protocol/input_session.rs`'s `InputSession` is the **single framing driver**
  (native + wasm) and is deliberately transport-free.
- It feeds byte chunks through `arrow`'s `StreamDecoder`, validates explicit
  `0.0.6` `trd.table.kind` metadata, decodes `[mesh][texture?][frames?][params]`
  via the one column decoder in `protocol/arrow_decode.rs`, and yields one
  `FrameBatch` per params record batch.
- `io/input_stream.rs`'s `InputStream<R: Read>` wraps it for the blocking case:
  it owns the `Read`, exposes the prologue via inherent methods and implements
  `Iterator<Item = Result<FrameBatch, StreamError>>`, so the 64 KiB read loop
  exists once.
- `stream_filter/` drives that for the CLI (`run_stream`) and `native/trd-app`
  for the window.
- Params stay one batch in flight; optional indexed frames resources are retained
  for playback/reuse (encoded Binary stays compressed until selected).

Output path:

- `protocol/output_session.rs` + `io/output_stream.rs` split Arrow IPC *output*
  serialization the same way.
- `OutputSession` writes the `r,g,b,a` `fixed_shape_tensor<u8>` stream
  incrementally and owns no transport, while `OutputStream<W: Write>` owns the
  `Write`.
- `tightly_pack_rgba` (`protocol/image_encode.rs`) strips GPU row padding.
- The path is shared by the CLI and the browser offscreen renderer.

### Video metadata domain

**Rule: video metadata is not the render protocol (#296).** `media/` contains
what trd knows about a *video*, as opposed to the render wire.

One session needs the same handful of facts — size, exact frame rate, frame
count, duration — and there are **two** sources for them:

| Source | Location |
|---|---|
| `0.2.0` authoring document | `media/video_document/`, read from Arrow IPC or Parquet |
| Container metadata | `media/mp4_probe/`, walked for its `moov` box |

They are alternative answers to one question rather than unrelated parsers, so
they share one `VideoTiming` (`media/video.rs`) and the columnar helpers in
`media/arrow_columns.rs`. The document is **optional** (#264): without one the
editor is a player whose timeline comes from the container.

`trd-core` does no codec work — demuxing and decoding belong to the delivery
surfaces (mediabunny in the browser, ffmpeg natively). Deliberately **not** under
`protocol/`: the editor document is independent of the render `PROTOCOL_VERSION`
and must stay that way.

### Typed math

**Rule: math types encode affine constraints that glam cannot.** `math/` is the
typed homogeneous linear-algebra layer over glam:
`Vector`/`Point`/`Normal`/`Matrix`/`Rotation`/`Transform`/`Aabb`.

They are zero-cost `#[repr(transparent)]` newtypes with **private** fields
enforcing affine rules (`point − point → vector`, no `point + point`).
Conventions are column-major, right-handed, clip `z ∈ [0, 1]`.

## The front-ends

Each is a *thin shell* that only supplies a render target and calls the core:

| Front-end | Reads | Renders into | Produces |
|---|---|---|---|
| **`trd-cli`** | Arrow stream (stdin) | offscreen texture → read-back | Arrow image stream (stdout) |
| **`trd-app`** | Arrow stream (stdin) | live window swapchain | frames on screen |
| **`trd-wasm`** | Arrow stream (buffered via `loadIpc`) | live canvas (or offscreen texture) | frames in the browser |
| **`trd-gui`** | a mesh + live gestures | offscreen texture → egui image | an interactive orbit/zoom viewer (native + browser) |
| **video editor** | `0.2.0` timeline + external video | offscreen texture → egui image | quad-local 3D editing over video |

### Front-end roles

**Rule: front-ends are shells, not renderers.** Each supplies a target, owns its
platform lifecycle, and delegates pixels to `trd-core`.

- **`trd-cli`** — headless Arrow filter: renders each frame to an offscreen
  texture and writes the pixels as an Arrow image stream. It does **not** encode
  video; pipe the stream to [`scripts/encode.py`](../scripts/encode.py) (ffmpeg)
  for a GIF/WebP/MP4.
- **`trd-app`** — native window: a background thread reads the mesh-first stream
  from stdin; the window plays it at `--fps`, drawing each frame straight into
  the swapchain surface. No read-back, no file.
- **`trd-gui`** — interactive viewer (native + browser): turns orbit/zoom/pan
  gestures into an updated camera + model matrix and re-renders one mesh through
  `trd-core`, offscreen, shown as an egui image.

### Browser delivery surface

**Rule: `trd-wasm` / `web/` is the only browser delivery surface.** Every
`#[wasm_bindgen]` export in the repo lives in `crates/trd-wasm`, both the
viewer's `CanvasRenderer`/`OffscreenRenderer` and the GUI's `start` /
`startVideoEditing` / `VideoEditingHandle` (`src/gui.rs`, `src/gui_web_app.rs`).

Every other crate — `trd-gui` included — is a plain `rlib` free of
`wasm-bindgen`, so one wasm build produces one JS package (`trd_wasm`) that all
three `web/` packages stage into their own `pkg/` (#180).

It is also the only crate that may name **`web-sys`** (#302): the browser frame
copy reaches `trd-core` through `ExternalFrame`, so the shared crates carry no
browser type and no `cfg` hiding one. Both rules are scanned by
`tests/wasm_bindgen_containment.rs`.

#### Prefer moving a `cfg` into `trd-wasm` over living with one

Reach for `#[cfg(target_arch = "wasm32")]` only when both arms are real — as in
the two `platform.rs` shims — never to hide a browser-only type. That says when a
`cfg` is *permitted*; this says what to try first, because the cheapest `cfg` to
review is the one that was never written.

**The two directions are not the same problem.**

| Written as | Means | Standing |
|---|---|---|
| `#[cfg(target_arch = "wasm32")]` | "this exists **only** in the browser" | Almost always belongs in `trd-wasm` instead. A native build never compiles it, so nothing native checks it — the failure the `web-sys` rule exists to prevent. |
| `#[cfg(not(target_arch = "wasm32"))]` | "this has **no browser meaning at all**" | Legitimate. A browser has no `R: Read` and no blocking executor; `io/mod.rs` and the GPU test harness say so honestly. |

**Work down this order and stop at the first that fits.**

| | Resolution | Precedent |
|---|---|---|
| 1 | **Name a seam in the shared crate; implement it in `trd-wasm`.** The shared crate keeps the type and its invariants; the browser supplies only the part that cannot compile natively. | `ExternalFrame` (#302) — replaced eleven `cfg`s and a `web-sys` dependency across two shared crates |
| 2 | **Two real arms behind one shim**, so no caller sees a `cfg` at all. | the two `platform.rs` files |
| 3 | **Gate the module, not its items** — one `cfg` on the `mod`, none inside. | `render/mod.rs` gating the GPU test harness (#299) |
| 4 | **Last resort: a `cfg` in place**, for something with genuinely no browser meaning. | `cfg(not(target_arch = "wasm32"))` on `InputStream`, the native byte transport |

**A seam must not cost more than the `cfg` it removes.** If option 1 only
relocates the complexity — spreading one call across three crates, or adding a
trait with exactly one implementor and one caller that no third party could ever
use — take a lower row and write the honest `cfg`. The goal is code a reader can
follow **on one platform without mentally compiling the other**, not the smallest
possible `cfg` count.

The one thing that is never a trade-off: option 4 must not hide a **browser-only**
type in a shared crate. That is the line `tests/wasm_bindgen_containment.rs`
draws, and the case a native build cannot check for you.

<details>
<summary>Why — how eleven browser types accumulated unnoticed</summary>

A browser type in a shared crate has to be hidden behind a `cfg` a native build
never compiles, which is how eleven of them accumulated unnoticed; the browser
frame copy now reaches the render core through `trd_core::ExternalFrame`
(`trd-core` owns the trait and the destination texture, `trd-wasm`'s
`BrowserVideoFrame` owns the `copy_external_image_to_texture` call wgpu marks
`#[cfg(web)]`).
</details>

Runtime shape:

- `CanvasRenderer.create(canvas)` holds a persistent `Renderer` + `InputSession`
  and renders the **same** `Scene` as the CLI.
- There is **one** config-driven front-end: `render.sh --web` writes the demo's
  `stream.arrow` + `config.json`, and
  [`web/viewer/src/viewer.ts`](../web/viewer/src/viewer.ts) fetches both and
  replays by index.
- Two targets share the bundle: the on-screen `CanvasRenderer` and the offscreen
  `OffscreenRenderer` (renders to a texture, reads it back, paints a 2D canvas).
- JS only moves Arrow bytes; it never touches WebGPU. Ships as the `trd-wasm` npm
  library.

## Source layout

| Path | What it is |
|---|---|
| `crates/trd-core` | the unified render core (`render/` module tree, `shader/*.wgsl`, `protocol/`, `io/`, `media/`, `stream_filter/`) |
| `crates/trd-cli` | headless CLI: Arrow stream in → Arrow image out |
| `crates/trd-gui` | reusable egui UI, scene/interaction state, and native render backends (a plain `rlib`: every browser entry point moved to `trd-wasm` in #180) |
| `crates/trd-placement` | GPU-free K + image-quad reconstruction and placement matrices |
| `crates/trd-wasm` | the **only** `wasm-bindgen` crate, the only `cdylib`, and the only crate naming `web-sys` (all three guarded by `tests/wasm_bindgen_containment.rs`): viewer bindings (`canvas_renderer`/`offscreen_renderer`) + the GUI entry points (`gui.rs`, `gui_web_app.rs`) + the browser `ExternalFrame` impl (`browser_frame.rs`); the `trd-wasm` npm library |
| `native/trd-app` | native stream-playback window (winit + live wgpu surface) |
| `native/trd-gui-app` | native eframe shell around the reusable `trd-gui` library |
| `native/trd-gui-video-editing` | native ffmpeg-backed video timeline/player shell |
| `web/viewer` | config-driven browser stream player around `trd-wasm` |
| `web/gui-viewer` | browser eframe shell around the `trd_wasm` GUI entry points |
| `web/gui-video-editing` | browser video-editing surface with its own copy of the generated `trd_wasm` package |
| `web/package.json` | shared Bun workspace for all browser delivery surfaces |
| `scripts/fiba_video_editing_bundle.py` | FIBA video/parquet → `0.2.0` timeline document |
| `examples/` | demo streams + `render.sh` / `render.ps1` wrappers + producer scripts |
| `scripts/` | pyarrow producers (`obj`/`texture`/`jsonl`/perception `_to_arrow.py`), `encode.py`, `extract_frames.py`, `dev-env.ps1` |

Tests are placed by **kind, not size**: a unit test sits inline in the module it
pins, as `#[cfg(test)] mod tests`, however long it grows; an integration test
gets its own file in `crates/*/tests/`. There is no `src/**/tests.rs` middle
form — it compiles into the crate like an inline module, so it would only blur
the distinction; the reasoning is in [`AGENTS.md`](../AGENTS.md#where-a-test-lives--by-kind-not-by-size-305).
Contributor/agent conventions (which gates a change owes, PR workflow) live in
[`AGENTS.md`](../AGENTS.md); how to run them, including GPU-adapter selection and
every e2e procedure, is in [`AGENTS.md`](../AGENTS.md#testing).
