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

## The render core — `trd-core`

Platform-agnostic wgpu logic, shared verbatim by every target:

- **`render/` (module tree) + `shader/*.wgsl`** — `Renderer` (`render/renderer.rs`)
  is the one persistent render harness: it rasterizes a `Scene` of
  `DrawableObject`s into *any* `wgpu::TextureView`, owning the GPU context,
  pipelines, materials, and mesh store, with the render **target** passed as a
  plain per-call argument rather than a type parameter or an owned field (#203).
  A target is **pure data** — `render/render_target.rs` holds only the resources
  a frame lands in (`TextureTarget` = texture + padded staging buffer;
  `SurfaceTarget` = surface + config + sRGB view format; `RenderTarget` = the
  closed enum over the two, holding just the discriminant so each variant stays
  the single source of truth for its own size). **All** the behaviour is on the
  renderer, behind one match: `Renderer::render(camera, scene, &mut RenderTarget)`
  is the single render entry, dispatching to a private `render_surface` (acquire →
  encode through the sRGB view → submit → present) or `render_texture` (encode →
  submit). It is synchronous, returning `Result<Option<SurfaceRepair>,
  RenderError>`; the asymmetric tail — `read_pixels`, plus the multi-camera
  `draw_layers`/`render_layers`/`render_params` — takes the concrete
  `&TextureTarget`, so asking a *surface* for pixels is a type error rather than a
  runtime arm. Creating, resizing, reconfiguring and replacing a target are
  `Renderer` functions too (`create_texture_target`, `resize_texture_target`,
  `resize_surface`, `reconfigure_surface`, `replace_surface`); the surface ones are
  *associated* functions taking a `&wgpu::Device`, because a window is resized and
  repaired before the stream's mesh table has arrived to build a `Renderer` from.
  Texture targets serve `trd-cli`, `trd-gui` and the browser `OffscreenRenderer`;
  surface targets serve `trd-app` and `trd-wasm`'s `CanvasRenderer`. Live-surface
  shells are **not** renderers: they create the `wgpu::Surface`, own the resulting
  `RenderTarget`, and apply their own recovery policy to the
  `Ok(Some(SurfaceRepair))` / `Err(RenderError::Surface(_))` the harness reports
  (the native window defers to the next redraw; the browser reconfigures or
  recreates the surface in-call and retries once). Gizmo segments use
  `gizmo_line.wgsl`: the vertex stage expands each model-space segment to a
  configurable pixel-width quad and the fragment stage feathers its rectangle
  distance, so axes/AABBs/grids remain anti-aliased without MSAA. Axis cone tips
  reuse the unlit triangle path.
- **`DrawableObject` + `Primitive` + `Scene` (`render/`)** — the base interface
  for every primitive (#41). A `DrawableObject` is a small `Copy` struct pairing
  a `Primitive` — *what* to draw: `Mesh { mesh_id, mode }`, `AabbBox { mesh_id }`,
  `CoordinateAxes`, `PlaneGrid { plane }`, `QuadOutline { selected }`,
  `BlobShadow` — with the `model` that places it, so every one of them can be
  instanced. Geometry is owned once
  (decode-once mesh store + shared
  line-quad/arrow buffers); a drawable is a light handle naming *which* primitive
  + its per-frame model. A `Scene` (an object list plus its `Background`) is
  rebuilt each frame; every front-end hands it to `Renderer::render` without
  per-type branching.
  The render core walks its objects into a flat list, batches by primitive —
  a batch key is a drawable minus its model, so the same taxonomy serves both
  (#204) — binds
  the shared `P·V` camera uniform (plus viewport size for gizmo lines), and
  records the draws in `Primitive::sort_key` order, which is the frame's z-order
  because every overlay pipeline disables depth.
- **`Scene::background()` — what a frame draws *behind* its primitives (#204).**
  `Background { environment: Option<EnvironmentBackground>, frame: Option<FrameFit> }`
  holds the camera-centered HDR environment probe and the fullscreen background
  frame plane (#63). Both were `DrawableObject` variants, and were the only two
  carrying no model and the only two the batcher had to skip; as scene settings
  they are set once, with no ordering or duplicate to get wrong. The two slots
  are **independent** — a frame may draw both — and the renderer always draws the
  environment first, then the frame plane, then the mesh scene over them. Appearance (filled / wireframe / textured / **PBR**) is a *mode* of
  the mesh drawable, not a separate primitive.
- **Typed PBR domains** — Disney
  surface parameters and preserved glTF auxiliary data live in
  `material/disney.rs`; analytic lights and rig controls in `light.rs` (both at
  the crate root, since a material and a light are universal domain vocabulary —
  #223); HDR environment data + its CPU precompute in `render/env_map.rs`, its
  binding and the sky pipeline in `render/environment.rs`; and the
  per-object output transform in `render/tonemap.rs`. `pbr.rs` contains only the
  unchanged shader-uniform packing and smooth-normal derivation. `trd-core`'s
  boundary-level `mesh/gltf.rs` parses caller-owned bytes into these types without
  entering the render hot path or performing filesystem I/O.
- **`mesh/` — the CPU mesh and its loaders (#221).** The crate root holds the
  universal domain vocabulary (a mesh, a material, a texture, a camera), so the
  canonical `Mesh`/`MeshShading` container lives in `mesh/mesh.rs` with one
  loader per format beside it — `mesh/obj.rs` (#36), `mesh/arrow.rs` (#37) and
  `mesh/gltf.rs` — plus the geometry every source shares (`aabb`, `center`,
  `preview_transform`, `edge_indices`) in `mesh/mod.rs`. The module is
  **device-free**: a mesh's GPU residency is its face in `render/mesh_store.rs`,
  and the `Vertex` layout it is written in stays with the other `repr(C)` + `Pod`
  types in `render/gpu_types.rs`.
- **`stream.rs` + `protocol.rs`** — the Arrow input layer. `protocol.rs`'s
  `InputSession` is the **single framing driver** (native + wasm): it feeds byte
  chunks through `arrow`'s `StreamDecoder`, validates explicit `0.0.6`
  `trd.table.kind` metadata, decodes `[mesh][texture?][frames?][params]`, and
  yields one `FrameBatch` per params record batch. `stream.rs` (`run_stream` for
  the CLI, `read_scene_stream_with_meta` for the window) drives it from a
  blocking `Read`. Params stay one batch in flight; optional indexed frames
  resources are retained for playback/reuse (encoded Binary stays compressed
  until selected).
- **`output.rs`** — the Arrow IPC *output* serialization. `OutputSession` writes
  the `r,g,b,a` `fixed_shape_tensor<u8>` stream incrementally; `tightly_pack_rgba`
  strips GPU row padding. Shared by the CLI and the browser offscreen renderer.
- **`math/`** — the typed homogeneous linear-algebra layer over glam
  (`Vector`/`Point`/`Normal`/`Matrix`/`Rotation`/`Transform`/`Aabb`): zero-cost
  `#[repr(transparent)]` newtypes with **private** fields enforcing affine rules
  glam can't (`point − point → vector`, no `point + point`). Column-major,
  right-handed, clip `z ∈ [0, 1]`.

## The front-ends

Each is a *thin shell* that only supplies a render target and calls the core:

| Front-end | Reads | Renders into | Produces |
|---|---|---|---|
| **`trd-cli`** | Arrow stream (stdin) | offscreen texture → read-back | Arrow image stream (stdout) |
| **`trd-app`** | Arrow stream (stdin) | live window swapchain | frames on screen |
| **`trd-wasm`** | Arrow stream (buffered via `loadIpc`) | live canvas (or offscreen texture) | frames in the browser |
| **`trd-gui`** | a mesh + live gestures | offscreen texture → egui image | an interactive orbit/zoom viewer (native + browser) |
| **video editor** | `0.2.0` timeline + external video | offscreen texture → egui image | quad-local 3D editing over video |

- **`trd-cli`** — headless Arrow filter: renders each frame to an offscreen
  texture and writes the pixels as an Arrow image stream. It does **not** encode
  video; pipe the stream to [`scripts/encode.py`](../scripts/encode.py) (ffmpeg)
  for a GIF/WebP/MP4.
- **`trd-app`** — native window: a background thread reads the mesh-first stream
  from stdin; the window plays it at `--fps`, drawing each frame straight into the
  swapchain surface. No read-back, no file.
- **`trd-wasm` / `web/`** — the **only** browser delivery surface: every
  `#[wasm_bindgen]` export in the repo lives in `crates/trd-wasm`, both the
  viewer's `CanvasRenderer`/`OffscreenRenderer` and the GUI's `start` /
  `startVideoEditing` / `VideoEditingHandle` (`src/gui.rs`, `src/gui_web_app.rs`).
  Every other crate — `trd-gui` included — is a plain `rlib` free of
  `wasm-bindgen`, so one wasm build produces one JS package (`trd_wasm`) that all
  three `web/` packages stage into their own `pkg/` (#180).
  `CanvasRenderer.create(canvas)` holds a
  persistent `Renderer` + `InputSession` and renders the **same** `Scene` as
  the CLI. There is **one** config-driven front-end: `render.sh --web` writes the
  demo's `stream.arrow` + `config.json`, and
  [`web/viewer/src/viewer.ts`](../web/viewer/src/viewer.ts) fetches both and
  replays by index.
  Two targets share the bundle: the on-screen `CanvasRenderer` and the offscreen
  `OffscreenRenderer` (renders to a texture, reads it back, paints a 2D canvas). JS
  only moves Arrow bytes; it never touches WebGPU. Ships as the `trd-wasm` npm
  library.
- **`trd-gui`** — interactive viewer (native + browser): turns orbit/zoom/pan
  gestures into an updated camera + model matrix and re-renders one mesh through
  `trd-core`, offscreen, shown as an egui image.

## Source layout

| Path | What it is |
|---|---|
| `crates/trd-core` | the unified render core (`render/` module tree, `shader/*.wgsl`, `stream.rs`, `protocol.rs`) |
| `crates/trd-cli` | headless CLI: Arrow stream in → Arrow image out |
| `crates/trd-gui` | reusable egui UI, scene/interaction state, native render backends, and browser wasm entry |
| `crates/trd-placement` | GPU-free K + image-quad reconstruction and placement matrices |
| `crates/trd-wasm` | `wasm-bindgen` browser bindings (`canvas_renderer`/`offscreen_renderer`); the `trd-wasm` npm library |
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

Contributor/agent conventions (build system, GPU-adapter selection, testing
policy, PR workflow) live in [`AGENTS.md`](../AGENTS.md).
