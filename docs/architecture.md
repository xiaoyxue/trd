# Architecture

`trd-core` is the *single* rendering core: the exact same Rust/wgpu code renders
in every front-end (headless CLI, native window, browser, interactive viewer) by
drawing into whatever render target each one provides. JavaScript/TypeScript is a
thin bootstrap only — the WebGPU API is never called from JS.

Everything shares **one render function** and **one data format** (a mesh-first
Arrow stream):

```
input-stream ─┬─ trd-cli  → trd-core → offscreen readback → image-stream   (headless)
(mesh-first)  ├─ trd-app  → trd-core → window surface                      (native playback)
              ├─ trd-wasm → trd-core → canvas surface                      (browser)
              └─ trd-gui  → trd-core → offscreen → egui image      (interactive, native + browser)

                  image-stream → scripts/encode.py → ffmpeg → GIF / WebP / MP4
```

## The render core — `trd-core`

Platform-agnostic wgpu logic, shared verbatim by every target:

- **`render/` (module tree) + `*.wgsl` shaders** — `MeshRenderer`
  (`render/mesh_renderer.rs`) rasterizes a `Scene` of `DrawableObject`s into *any*
  `wgpu::TextureView`; that one renderer is why the same code targets an offscreen
  texture, a window swapchain, or a browser canvas. The offscreen render target +
  async pixel read-back is factored into a shared `OffscreenTarget` harness
  (`render/offscreen.rs`), reused by every read-back front-end (`trd-cli`, the
  browser `OffscreenRenderer`, and `trd-gui`). The live-present front-ends
  (`trd-app`, `trd-wasm`'s `CanvasRenderer`) instead own their `wgpu::Surface`
  swapchain directly and draw the same `Scene` into it.
- **`DrawableObject` + `Scene` (`render/scene.rs`)** — the base interface for every
  primitive (#41). It is a small `Copy` enum — `Mesh { mesh_id, model, mode }`,
  `AabbBox { mesh_id, model }`, `CoordinateAxes { model }`, and `FramePlane { fit }`
  (a background still). Geometry is owned once (decode-once mesh store + shared
  gizmo buffers); a drawable is a light handle naming *which* primitive + its
  per-frame model. A `Scene = Vec<DrawableObject>` is rebuilt each frame;
  `MeshRenderer::encode` walks it once, binds the shared `P·V` camera uniform, and
  records the draws with **no per-type branching**. Appearance (filled / wireframe
  / textured / **PBR**) is a *mode* of the mesh drawable, not a separate primitive.
- **`stream.rs` + `protocol.rs`** — the Arrow input layer. `protocol.rs`'s
  `InputSession` is the **single framing driver** (native + wasm): it feeds byte
  chunks through `arrow`'s `StreamDecoder`, validates the schema once (`0.0.5`
  only), and yields one `FrameBatch` per record batch. `stream.rs` (`run_stream`
  for the CLI, `read_scene_stream_with_meta` for the window) drives it from a
  blocking `Read`. Only one record batch is ever in flight, so an animation of any
  length streams in constant memory.
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

- **`trd-cli`** — headless Arrow filter: renders each frame to an offscreen
  texture and writes the pixels as an Arrow image stream. It does **not** encode
  video; pipe the stream to [`scripts/encode.py`](../scripts/encode.py) (ffmpeg)
  for a GIF/WebP/MP4.
- **`trd-app`** — native window: a background thread reads the mesh-first stream
  from stdin; the window plays it at `--fps`, drawing each frame straight into the
  swapchain surface. No read-back, no file.
- **`trd-wasm` / `web/`** — browser: `CanvasRenderer.create(canvas)` holds a
  persistent `MeshRenderer` + `InputSession` and renders the **same** `Scene` as
  the CLI. There is **one** config-driven front-end: `render.sh --web` writes the
  demo's `stream.arrow` + `config.json`, and
  [`web/src/viewer.ts`](../web/src/viewer.ts) fetches both and replays by index.
  Two targets share the bundle: the on-screen `CanvasRenderer` and the offscreen
  `OffscreenRenderer` (renders to a texture, reads it back, paints a 2D canvas). JS
  only moves Arrow bytes; it never touches WebGPU. Ships as the `trd-wasm` npm
  library.
- **`trd-gui`** — interactive viewer (native + browser): turns orbit/zoom/pan
  gestures into an updated camera + model matrix and re-renders one mesh through
  `trd-core`, offscreen, shown as an egui image. `--backend arrow` (or
  `?backend=arrow`) round-trips each frame through the real Arrow wire — the seam
  an external producer would drive. Design notes: [`docs/gui-design.md`](gui-design.md).

## Source layout

| Path | What it is |
|---|---|
| `crates/trd-core` | the unified render core (`render/` module tree, `*.wgsl` shaders, `stream.rs`, `protocol.rs`) |
| `crates/trd-cli` | headless CLI: Arrow stream in → Arrow image out |
| `crates/trd-app` | native interactive window (winit + live wgpu surface) |
| `crates/trd-gui` | interactive egui orbit/zoom viewer (native eframe + browser wasm) |
| `crates/trd-wasm` | `wasm-bindgen` browser bindings (`canvas_renderer`/`offscreen_renderer`); the `trd-wasm` npm library |
| `web/` | bun-managed TypeScript wrapper (`main.ts` → config-driven `viewer.ts`) that loads `trd-wasm` |
| `examples/` | demo streams + `render.sh` / `render.ps1` wrappers + producer scripts |
| `scripts/` | pyarrow producers (`obj`/`texture`/`jsonl`/perception `_to_arrow.py`), `encode.py`, `extract_frames.py`, `dev-env.ps1` |

Contributor/agent conventions (build system, GPU-adapter selection, testing
policy, PR workflow) live in [`AGENTS.md`](../AGENTS.md).
