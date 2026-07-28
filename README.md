# trd

**A tile (relational) oriented renderer, built on Rust + wgpu.**

`trd-core` is the *single* rendering core. The exact same Rust/wgpu code renders
in three places — a headless CLI, a native window, and the browser — by drawing
into whatever render target each one provides. JavaScript/TypeScript is a thin
bootstrap only; the WebGPU API is never called from JS.

## How it fits together

Everything shares **one render function** and **one data format**:

```
input-stream ─┬─ trd-cli  → trd-core → offscreen readback → image-stream   (headless)
(mesh-first)  ├─ trd-app  → trd-core → window surface                      (native playback)
              ├─ trd-wasm → trd-core → canvas surface                      (browser)
              └─ trd-gui  → trd-core → offscreen → egui image      (interactive, native + browser)

                  image-stream → scripts/encode.py → ffmpeg → GIF / WebP
```

### The render core — `trd-core`

Platform-agnostic wgpu logic, shared verbatim by every target:

- **`render/` (module tree) + `mesh.wgsl` / `textured.wgsl` / `frame_plane.wgsl`** —
  `MeshRenderer` (`render/mesh_renderer.rs`) rasterizes a `Scene` of
  `DrawableObject`s into *any* `wgpu::TextureView`. That one renderer is why the
  same code targets an offscreen texture, a window swapchain, or a browser canvas.
  The offscreen render target + async pixel read-back is itself factored into a
  shared `OffscreenTarget` harness (`render/offscreen.rs`) reused by every headless
  renderer — native `BatchRenderer` and the two browser renderers alike.
- **`DrawableObject` + `Scene` (`render/scene.rs`)** — the base interface for every
  primitive the renderer can draw (#41). `DrawableObject` is a small `Copy` enum —
  `Mesh { mesh_id, model, mode }` (filled or **wireframe** mode), `AabbBox {
  mesh_id, model }`, `CoordinateAxes { model }`, and `FramePlane { fit }` (the #63
  background still) — where geometry (GPU buffers) is owned once by the renderer's
  decode-once mesh store and each variant carries only *which* primitive to draw
  plus its per-frame model. A `Scene = Vec<DrawableObject>` is rebuilt each frame;
  `MeshRenderer::encode` walks it once, binds the shared `P·V` camera uniform,
  buckets the drawables (background frame → filled meshes → wireframe meshes → AABB
  boxes → axes) into one instance buffer, and records the draws — with **no
  per-type branching** in any front-end. A single-object frame is the degenerate
  one-element scene, so there is no special case. Wireframe is a *mode* of the mesh
  drawable, not a separate primitive; the AABB box and axes gizmo are core-side
  additions to the scene (not wire columns).
- **`stream.rs`** — the native Arrow IPC filter: `run_stream` is the CLI filter,
  and `read_scene_stream_with_meta` drives the windowed viewer. Both frame the
  input by driving the shared `protocol.rs` `InputSession` from a blocking
  `Read`, so all mesh-first sub-stream sniffing lives in one place. Only one
  record batch is ever in flight, so an animation of any length streams in
  constant memory.
- **`protocol.rs`** — the cross-platform (native + wasm) incremental Arrow IPC
  decoder. `InputSession` feeds arbitrary byte chunks through `arrow`'s
  `StreamDecoder`, validates the protocol schema once (accepts `0.0.5` only), and
  yields one `FrameBatch` (`Vec<DecodedFrame>`) per record batch — the **single
  framing driver** for both the native CLI/window and the browser.
- **`output.rs`** — the cross-platform Arrow IPC *output* serialization.
  `OutputSession` writes the `r,g,b,a` `fixed_shape_tensor<u8>` stream
  incrementally (one output batch per input batch); `tightly_pack_rgba` strips GPU
  row padding. Shared by the native CLI and the browser `OffscreenRenderer`.
- **`math/`** — the typed homogeneous linear-algebra layer over glam:
  `Vector2/3/4`, `Point2/3/4`, `Normal3`, `Matrix3/4`, `Rotation` (unit
  quaternion), `Transform`, and `Aabb2/3`. Zero-cost `#[repr(transparent)]`
  newtypes with **private** inner fields that enforce affine-space rules
  (`point − point → vector`, no `point + point`) the raw glam types can't.
  Column-major, right-handed, clip `z ∈ [0, 1]`; `render/`'s MVP transforms
  are built on it and its `ToWgsl` layout keeps the GPU `Uniform` byte-identical.

### The consumers

Each is a *thin shell* that only supplies a render target and calls the core:

| Target | Reads | Renders into | Produces |
|---|---|---|---|
| **`trd-cli`** | Arrow stream (stdin) | offscreen texture → pixel read-back | Arrow image stream (stdout) |
| **`trd-app`** | Arrow stream (stdin) | live window swapchain | frames on screen |
| **`trd-wasm`** | Arrow stream (buffered via `loadIpc`) | live canvas surface (or offscreen texture) | frames in the browser |
| **`trd-gui`** | a mesh (`--mesh` / `?mesh=`) + live gestures | offscreen texture → egui image (native + browser) | an interactive orbit/zoom viewer |

- **`trd-cli` — headless Arrow filter.** For each input frame it renders to an
  offscreen texture, copies the pixels back (`copy_texture_to_buffer`), and writes
  them as an Arrow image stream. It does **not** encode video itself — piping that
  stream to `scripts/encode.py` (ffmpeg) turns it into a GIF/WebP.
- **`trd-app` — native window.** A background thread reads the mesh-first input
  stream from stdin (via `read_scene_stream_with_meta`); the window plays it at
  `--fps`, drawing each frame **straight into the swapchain surface** and presenting
  it. No read-back, no file — pixels go on screen. With no stdin it renders nothing
  (a black window) until a scene arrives.
- **`trd-wasm` / `web/` — browser.** `CanvasRenderer.create(canvas)` obtains a wgpu
  surface from the `<canvas>` and holds a persistent `MeshRenderer` plus an
  `InputSession`, rendering the **same mesh Scene** as the native CLI through the
  shared `build_scene`/`MeshRenderer` path — no per-frontend branching. There is
  **one** config-driven front-end: `render.sh --web` runs the *same* Arrow producers
  and scene flags as `--cli` and writes `stream.arrow` + `config.json` into the
  served directory; the tiny [`web/src/main.ts`](web/src/main.ts) loads
  [`web/src/viewer.ts`](web/src/viewer.ts), which fetches both,
  decodes the whole stream once with `loadIpc` (buffering every frame), and replays
  it by index with `renderIndex(i)`. Two targets share the bundle — the on-screen
  `CanvasRenderer` (default) and the offscreen `OffscreenRenderer` (renders to a texture,
  reads it back to RGBA, and paints it to a 2D canvas). Modes/overlays come from the
  config via `setWireframe`/`setTextured`/`setShowAabb`/`setShowAxes`/`setShowLocalAxes`,
  and `setCompositeFrame` + `updateFrameTextureRgba` composite each frame's 0.0.5
  background still. JS only moves Arrow bytes and schedules frames; it never touches
  the WebGPU API. The crate root is glue only — the two renderers live in
  `crates/trd-wasm/src/{canvas_renderer,offscreen_renderer}.rs` — and it ships as the
  `trd-wasm` npm library.
- **`trd-gui` — interactive viewer (native + browser).** An egui viewer that turns
  orbit / zoom / pan gestures into an updated camera + model matrix and re-renders
  a single mesh through `trd-core`. It renders **offscreen** to RGBA (via the shared
  `OffscreenTarget` harness) and shows the pixels as an egui image, so egui's own
  toolkit stays independent of `trd-core`'s wgpu. Native (an eframe app over
  `InProcRenderer`) and browser (`web_renderer`, `wasm-bindgen` `start(canvas)`)
  share the same scene + interaction code; `--backend arrow` (or `?backend=arrow`)
  round-trips each frame's params through the real Arrow wire
  (`decode_params_stream`) — the seam an external producer would drive.

### Stream protocol

Frame parameters are plain columnar data, so **any** tool that emits the input
columns as an Arrow IPC stream can drive the renderer. The current — and **only
supported** — version is **0.0.5**: it is **mesh-first** (`[mesh][texture?][params]`)
and is **not** backward-compatible with `0.0.1`–`0.0.4` (older streams are
hard-rejected). See the [no-backward-compat policy](AGENTS.md).

| Direction | Columns | Arrow type |
|---|---|---|
| **Input** (params) | `model` *(opt)* | `FixedSizeList<f32>[16]` (4×4 model matrix) |
| | `k` *(opt)* | `FixedSizeList<f32>[9]` (3×3 camera intrinsics, **CV**) |
| | `pose` *(opt)* | `FixedSizeList<f32>[16]` (4×4 camera-to-world pose, **CV**) |
| | `eye`, `target`, `direction`, `up` *(opt)* | `FixedSizeList<f32>[3]` (**CG** look-at) |
| | `fovy`, `aspect`, `znear`, `zfar` *(opt)* | `f32` (**CG** perspective) |
| | `draw_mesh` *(opt)* | `List<u32>` (per-instance mesh index) |
| | `draw_model` *(opt)* | `List<FixedSizeList<f32>[16]>` (per-instance 4×4 model) |
| | `frame_path` / `frame_url` *(opt)* | `Utf8` (per-frame background image path / URL) |
| **Input** (mesh) | `position`, `color` | `List<FixedSizeList<f32>[3]>` |
| | `uv` *(opt)* | `List<FixedSizeList<f32>[2]>` (per-vertex texture coords) |
| | `index` | `List<u32>` |
| **Input** (texture, *opt*) | `rgba` | `fixed_shape_tensor<u8>` `[H, W, 4]` (interleaved RGBA) |
| **Output** (image) | `r`, `g`, `b`, `a` | `fixed_shape_tensor<u8>` `[H, W]` |

Every stream is **mesh-first**: a leading **mesh** Arrow stream (one row per mesh,
geometry in nested list columns), optionally followed by a **texture** stream, then
the per-frame **params** stream (`[mesh][texture?][params]`). **Both the native path
and the browser `CanvasRenderer`** decode the mesh (`Mesh::from_arrow`), upload it,
and render it **centered and uniformly scaled to fit** (a `base_model` beneath the
per-frame `model`). Encode an OBJ with `scripts/obj_to_arrow.py`;
`examples/render.sh --mesh <obj>` wires it end-to-end. The params columns are all
**optional/additive** and drive the MVP transform `clip = P · V · M · (pos, 1)`.

The params stream carries three optional features on top of the mesh geometry:

- an optional per-frame **camera**, authored either **CV**-style (`k` intrinsics +
  `pose` extrinsics, resolved as `view = inverse(pose)`) or **CG**-style (a look-at
  from `eye` + `target`/`direction` + `up`, with `fovy`/`aspect`/`znear`/`zfar`
  perspective). CV wins per component; absent any camera column the view is
  identity with a default perspective.
- an optional per-frame **draw list** (`draw_mesh` + `draw_model`, equal-length
  list columns) that instances several meshes in one frame — each entry places
  mesh `draw_mesh[i]` under model `draw_model[i]` (composed beneath that mesh's
  preview transform). Absent a draw list, one instance of mesh 0 is placed by the
  frame's own `model`.
- optional **textured rendering**: a **texture** Arrow stream spliced between the
  mesh and params streams (`[mesh][texture][params]`) — one row of interleaved-RGBA
  `rgba` (`fixed_shape_tensor<u8>` `[H, W, 4]`, dimensions self-describing) — plus an
  optional per-vertex **`uv`** mesh column. With `--textured` (`setTextured(true)`)
  meshes sample the bound texture at each UV (`textureSample`, `Rgba8UnormSrgb` linear
  clamp-to-edge); a textured draw with no texture stream samples a default 1×1 white.
  Encode an image with `scripts/texture_to_arrow.py`; `examples/render.sh --texture
  <img>` wires it end-to-end for `--cli` and `--web` alike (`.ps1 -Texture` on Windows).

The params stream also carries an optional per-frame **background frame**: a
`frame_path` (native) or `frame_url` (browser) `Utf8` column naming an image
composited **beneath** the scene by a `FramePlane` drawable (a fullscreen quad
sampling a reused `Rgba8UnormSrgb` texture, depth-write off so the mesh scene +
gizmos draw on top; `Stretch`/`Cover` fit). The core decodes the reference only; the
shell does the image I/O — `trd --frames-base <dir>` / `trd-app --frames-base <dir>`
load the PNG/JPEG (relative to `<dir>`). Produce the stills + manifest with
`scripts/extract_frames.py`. Without `--frames-base`, `frame_path` is ignored and
the scene renders over the black clear.

**Full specification: [`docs/protocol/0.0.5.md`](docs/protocol/0.0.5.md)** — the
single, self-contained schema reference (the protocol is `0.0.5`-only).


## Repository layout

| Path | What it is |
|---|---|
| `crates/trd-core` | the unified render core (`render/` module tree, `*.wgsl` shaders, `stream.rs`, `protocol.rs`) |
| `crates/trd-cli` | headless CLI: Arrow stream in → Arrow image out |
| `crates/trd-app` | native interactive window (winit + live wgpu surface); split into `main`/`cli`/`error`/`renderer`/`stream`/`app` modules |
| `crates/trd-gui` | interactive egui orbit/zoom viewer (native eframe + browser wasm); offscreen-renders one mesh through `trd-core` |
| `crates/trd-wasm` | `wasm-bindgen` entry point (crate-root glue + `canvas_renderer`/`offscreen_renderer` modules); packaged as the `trd-wasm` npm library |
| `web/` | bun-managed thin TypeScript wrapper (`main.ts` → config-driven `viewer.ts`) that loads `trd-wasm` |
| `examples/` | mesh-first demo streams (e.g. `frames.bunny_dolly.cg.jsonl`, `frames.turntable.jsonl`) + `render.sh` / `render.ps1` wrappers |
| `scripts/jsonl_to_arrow.py` | JSONL → Arrow `0.0.5` params stream (pyarrow producer) |
| `scripts/extract_frames.py` | video → still `frames/` + [frame-to-row mapping manifest](docs/frame-extraction.md) (ffmpeg; boundary tooling for the #62 compositing pipeline) |
| `scripts/encode.py` | Arrow image stream → ffmpeg GIF/WebP |
| `scripts/dev-env.ps1` | Windows dev-environment setup (the `nix develop` counterpart) |

## Quick start

**1. Get a dev environment.**

- **Linux / macOS / WSL** — [Nix](https://nixos.org/download) is the build system
  *and* the dev shell (pinned Rust, `bun`, wasm tools, `biome`, ffmpeg, Vulkan…):

  ```sh
  nix develop
  ```

- **Windows** — no Nix; dot-source the setup script instead (one-time installs are
  in [Windows setup](#windows-setup-without-nix)):

  ```powershell
  . .\scripts\dev-env.ps1
  ```

**2. Run the demo** — renders the bunny dolly-camera capstone
([`examples/frames.bunny_dolly.cg.jsonl`](examples/frames.bunny_dolly.cg.jsonl),
loading `assets/meshes/bunny.obj`):

```sh
# Linux / macOS / WSL
examples/render.sh --cli      # render → output/out.gif
examples/render.sh --native   # play live in a window
# (run examples/render.sh with no flags to print the flag guidance)
```

```powershell
# Windows (PowerShell 7)
examples\render.ps1 -CLI      # render → output/out.gif
examples\render.ps1 -Native   # play live in a window
# (run examples\render.ps1 with no arguments to print the flag guidance)
```

> On WSL, prefix GPU commands with `WGPU_BACKEND=gl` (otherwise rendering is
> software). On a **native Linux GPU box that isn't NixOS** (e.g. Ubuntu), the
> `nix develop` Vulkan loader can't reach the host GPU driver, so wrap GPU
> commands with [nixGL](https://github.com/nix-community/nixGL):
>
> ```sh
> NIXPKGS_ALLOW_UNFREE=1 nix run --impure github:nix-community/nixGL#nixGLNvidia -- \
>   examples/render.sh --cli      # or --native / --web; use #nixGLIntel for Intel/Mesa
> ```
>
> NixOS machines don't need this (the driver is on `/run/opengl-driver`).

**3. Try the web build:**

```sh
# Linux / macOS / WSL
examples/render.sh --web   # generate the demo's stream.arrow + config.json, then build + serve
nix run .#web              # serve a prebuilt dist/ (populate it first via render.sh --web)
```

```powershell
# Windows (PowerShell 7)
examples\render.ps1 -Web   # build + serve, printing the URL + SSH-tunnel command
```

Open the printed URL in a WebGPU browser (Chrome/Edge). On a remote (SSH) host,
run the printed tunnel command first, then browse to <http://localhost:8080>.

## Building & running

The Nix flake is the build system — reproducible outputs, no manual toolchain:

```sh
nix build .#trd-cli   # native CLI (Arrow stream filter) + Vulkan/GL runtime libs
nix build .#trd-wasm  # wasm-bindgen JS/TS library package
nix build .#web       # bun-bundled, HTTP-servable dist/  (also plain `nix build`)
nix run   .#trd -- --width 256 --height 256   # params on stdin → images on stdout
nix run   .#web                               # serve dist/  (PORT, default 8080)
nix flake check       # every gate: fmt, clippy (native+wasm32), test, tsc, biome
```

> `nix build` / `nix flake check` only see git-tracked files — `git add` new files
> before building.

For fast iteration use plain `cargo` / `bun` inside `nix develop` (or, on Windows,
after `. .\scripts\dev-env.ps1`). The sections below assume that.

### Native CLI

`trd` (package `trd-cli`) is a pure Arrow filter: params stream in → image stream
out. The `examples/render.*` wrappers build the whole JSONL → GIF pipeline for you:

```sh
examples/render.sh  [MODE] [INPUT.jsonl] [OUT.gif|.webp] [WIDTH] [HEIGHT] [FPS]   # Linux/macOS
examples\render.ps1 [MODE] [-InputPath]  [-Output]       [-Width] [-Height] [-Fps]  # Windows (PS7)
# Defaults: examples/frames.bunny_dolly.cg.jsonl (renders assets/meshes/bunny.obj) → output/out.gif, 256×256 @ 30 fps
# MODE (pick one): --cli/-CLI (default: headless GIF/WebP) · --native/-Native (live window) ·
#   --web/-Wasm (browser; --canvas-renderer default, or --offscreen-renderer/--arrow-renderer)
# Run either wrapper with no arguments (or -h/--help / -Help) to print the flag
# guidance and exit; pass a mode (e.g. --cli) to render the default demo.
```

On Windows the Arrow stages are handed off through a temp dir (PowerShell
pipelines aren't binary-safe); the output is identical.

#### Mesh & render flags (`--cli`)

Beyond the `MODE`, the headless (`--cli`) path takes content/appearance flags that
map straight onto `trd-cli`:

| Flag | Effect |
|---|---|
| `--mesh <obj>` | Prepend a mesh Arrow stream (mesh-first `[mesh][params]`) built from `<obj>` by [`scripts/obj_to_arrow.py`](scripts/obj_to_arrow.py); the mesh renders centered + scaled-to-fit, driven by the params `INPUT.jsonl`. **Repeatable** — pass `--mesh` several times to load several meshes (one table row each, in order); a frame's `draws` list then references them by 0-based index. Needs pyarrow (via uv/python3). |
| `--texture <img>` | Splice a texture Arrow stream (`[mesh][texture][params]`) built from `<img>` by [`scripts/texture_to_arrow.py`](scripts/texture_to_arrow.py) and render **textured** (`trd --textured`, #20): meshes sample the image at each vertex UV. Requires `--mesh` (with UVs); mutually exclusive with `--wireframe`. Downscaled to ≤ 2048² (portable limit); needs pyarrow + pillow + numpy. |
| `--wireframe` | Draw mesh **edges** as a line list (`trd --wireframe`) instead of filled triangles (protocol #38). Reveals topology; on a dense asset (e.g. the ~70k-tri bunny) the edges read as a fine mesh. |
| `--aabb` | Overlay each drawn mesh instance's **axis-aligned bounding box** as a green (`[0, 1, 0]`) wireframe box (`trd --aabb`, #42). The box uses the *same* per-instance model as the mesh, so it tracks the mesh through the preview + per-frame transforms. Combine freely with `--wireframe`. |
| `--axes` | Overlay a **coordinate-axes gizmo** (X=red, Y=green, Z=blue lines) at the world origin (`trd --axes`, #42), under the camera `P·V` with an identity model, marking the world frame the camera looks at. |
| `--frames-base <dir>` | Composite each frame's **background still** beneath the scene via a `FramePlane` (`trd --frames-base <dir>`, #63). A frame's 0.0.5 `frame_path` column (relative to `<dir>`) is loaded + decoded at the boundary into a reused GPU texture; the mesh + gizmos draw on top. Without it, `frame_path` is ignored (no background). |

```sh
# Single bunny turntable, filled, with its bounding box:
examples/render.sh --cli --aabb --mesh assets/meshes/bunny.obj \
  examples/frames.turntable.jsonl output/bunny.gif 1024 1024 24

# Textured bunny (samples a UV-mapped albedo at each vertex UV, #20):
examples/render.sh --cli --mesh assets/meshes/bunny_with_texture/bunny.obj \
  --texture assets/meshes/bunny_with_texture/bunny_uv_map1.jpg \
  examples/frames.bunny_dolly.cg.jsonl output/bunny_textured.gif 512 512 20

# Two-mesh scene (bunny = mesh 0, cube = mesh 1), wireframe + boxes:
examples/render.sh --cli --wireframe --aabb \
  --mesh assets/meshes/bunny.obj --mesh examples/cube.obj \
  examples/frames.multimesh.jsonl output/scene.gif 1024 1024 24
```

These flags are also raw `trd-cli` flags, so any producer pipeline can use them:
`… | trd --width 1024 --height 1024 --wireframe --aabb --axes | …`.

#### Dolly-camera turntable capstone (#49)

[`examples/bunny_dolly.py`](examples/bunny_dolly.py) authors the **same** 45°
bird's-eye *dolly* camera **twice** — once in the **CG** form (`eye`/`target`/
`up` + `fovy`/`aspect`) and once in the **CV** form (pinhole `K` + camera-to-world
`pose`) — as two JSONL streams. The camera looks at the world origin from a fixed
direction while only its **distance** oscillates (`dist = mid + amp·sin(2π·i/N)`,
a dolly, not an orbit/zoom); the bunny Y-spins via each frame's `model`. Both
forms decode to the *same* `P·V` (the script asserts this numerically before
writing) and render **identically** — verified at 1024² to differ by at most
**0.0054 % of pixels** per frame (a thin edge margin from the `f32`
`inverse(pose)` path; the spec permits a tiny tolerance).

```sh
python examples/bunny_dolly.py            # writes frames.bunny_dolly.{cg,cv}.jsonl (1024²)

# Render each form and compare — the two GIFs are visually indistinguishable:
examples/render.sh --cli --wireframe --mesh assets/meshes/bunny.obj \
  examples/frames.bunny_dolly.cg.jsonl output/bunny_dolly_cg.gif 1024 1024 24
examples/render.sh --cli --wireframe --mesh assets/meshes/bunny.obj \
  examples/frames.bunny_dolly.cv.jsonl output/bunny_dolly_cv.gif 1024 1024 24
```

The CV `K` is in **pixel** units, so the CV stream is authored for a specific
square resolution (`--width`/`--height`, default 1024²) and must be rendered at
that resolution to match the CG stream.

#### Background frame plane (#63)

[`examples/bunny_frameplane.py`](examples/bunny_frameplane.py) authors the
end-to-end **frame-compositing** demo: a folder of animated background stills plus
a turntable JSONL whose per-frame `frame_path` column (protocol **0.0.5**) names
each still. `trd --frames-base <dir>` loads each image at the boundary and
composites it *beneath* the spinning bunny via a `FramePlane`; the mesh + axes/AABB
gizmos draw on top. The backgrounds sweep a bright bar left→right over a
hue-shifting gradient, so the GIF visibly proves the plane texture updates every
frame (one reused GPU texture). The stills are written with a stdlib-only PNG
encoder (no Pillow), and land under `output/` (gitignored).

```sh
python examples/bunny_frameplane.py --out-dir output/fp_demo   # 24 stills + turntable_fp.jsonl

# Composite the stills under a wireframe turntable bunny + axes + AABB:
examples/render.sh --cli --wireframe --axes --aabb \
  --mesh assets/meshes/bunny.obj \
  --frames-base output/fp_demo \
  output/fp_demo/turntable_fp.jsonl output/fp_demo.gif 512 512 24
```

#### NBA court AR demo (#95)

[`scripts/nba_perception_to_arrow.py`](scripts/nba_perception_to_arrow.py) drives
the **same** single-view placement pipeline on a **real NBA broadcast clip**: it
repacks a per-frame court-calibration dataset — camera `K` + a tracked planar floor
quad (the "ad-unit" rectangle a reference AR cube stands on), from
VideoAnalysis#1133 — into the perception Arrow stream that
[`examples/placement_quad_by_local_coord.py`](examples/placement_quad_by_local_coord.py)
already consumes. The bunny is anchored to that court quad via the Pose-free #77
reconstruction and **stays glued to the same spot on the floor as the broadcast
camera pans and zooms**. It is the cornellbox demo with only the clip (and its
calibration) swapped — `trd-core` is untouched.

The broadcast video and its extracted frames are **not** vendored (copyrighted); the
demo reads the video from a local file (e.g. `~/Asset/nba-short/NBA.mp4`). The
per-frame **calibration** it depends on *is* vendored — image-free numeric camera `K`
+ floor quads at [`assets/videos/nba/per_frame_KVP_cube.parquet`](assets/videos/nba/)
(see its `DATASET.md`) — so the pipeline runs without the external dataset. Only the
other derived artifacts are committed too: the adapter, the perception stream
(`examples/frames.nba.perception.arrow`) and the placed scene
(`examples/frames.nba.stage2.jsonl`).

```sh
NBA_MP4=~/Asset/nba-short/NBA.mp4     # only the (copyrighted) video is external

# 1. vendored parquet → perception Arrow (shot 2; use a trustworthy BA_* focal).
#    Prints the present_index range to extract next.
uv run --with pyarrow scripts/nba_perception_to_arrow.py \
  --shot 2 --method BA_2511 \
  -o examples/frames.nba.perception.arrow

# 2. extract those broadcast frames (present_index 428..579) as background stills
mkdir -p output/nba/frames
ffmpeg -i "$NBA_MP4" -vf "select='between(n,428,579)',scale=960:540" \
  -vsync 0 -start_number 428 -q:v 3 output/nba/frames/frame_%06d.jpg

# 3. perception → placed scene (bunny on the court quad, Pose-free #77), then render
uv run --with pyarrow --with numpy examples/placement_quad_by_local_coord.py \
  --from-perception examples/frames.nba.perception.arrow --place-mesh --placement-quad \
  --size-factor 0.7 --src-width 1920 --src-height 1080 --width 960 --height 540 \
  -o examples/frames.nba.stage2.jsonl
examples/render.sh --cli --placement-quad --axes-local --aabb \
  --mesh assets/meshes/bunny_with_texture/bunny.obj \
  --texture assets/meshes/bunny_with_texture/bunny_uv_map1.jpg \
  --frames-base output/nba \
  examples/frames.nba.stage2.jsonl output/nba_stage2.gif 960 540 25
```

Like the cornellbox demo this also has a **stage 1** (the anchor before the mesh —
just the reconstructed placement quad + its local axes, no bunny): swap
`--place-mesh` for `--no-place-mesh --placement-quad-mesh-index 0` in step 3 to write
`examples/frames.nba.stage1.jsonl`, then render it with `--placement-quad --axes-local`
(no `--mesh`).

Shot 7 (`--method BA_2568`, `present_index` 1440..1638) is the other calibrated
shot. `K` is authored for 1920×1080, so keep `--src-width 1920 --src-height 1080`
(it is scaled to the render size). On Windows, run the same three steps with
`examples\render.ps1 -CLI` for the final render.

#### FIBA court AR demo — native 1080p (#110)

The same single-view placement pipeline on the **2024 Paris Olympic basketball
final** (France vs USA), rendered at the clip's **native 1920×1080** (instead of
the NBA demo's downscaled 960×540). It reuses the identical stages —
[`scripts/fiba_perception_to_arrow.py`](scripts/fiba_perception_to_arrow.py)
repacks the vendored per-frame calibration
([`assets/videos/fiba/per_frame_KVP_cube_best.parquet`](assets/videos/fiba/), see
its `DATASET.md`) into the perception stream that
[`examples/placement_quad_by_local_coord.py`](examples/placement_quad_by_local_coord.py)
consumes, and the bunny stays glued to the same court spot as the camera pans and
zooms. It is placed at **half scale** and shifted along the placement-quad's local
**+e1** axis onto the open right side of the free-throw key (all still expressed in
the quad's P² local frame), so it clears the active players instead of covering the
shooter at the quad centre. The trustworthy focal here is **`2VP_4510`** (`1circle_4252` corroborates) —
the **inverse of nba-short**, where BA was trusted and 2VP was degenerate; a clean
demonstration of why multiple K-methods are stored (the best flips with the
footage). The parquet tracks the ad quad for the first **222** frames; the
remaining **66** (the tail, where the quad leaves frame) carry null geometry and
are rendered as **plain video plates** (no AR mesh) so the clip stays continuous
with the source rather than being trimmed. Only the image-free calibration is
vendored; the copyrighted broadcast clip stays external.

```sh
FIBA_MP4=~/Asset/fiba-shot1/shot_0001.mp4   # only the (copyrighted) video is external

# 1. vendored parquet → perception Arrow (shot 1, the best-K method 2VP_4510).
#    Emits all 288 frames: 222 tracked (k+quad) + 66 untracked (null geometry →
#    background-plate only). Prints the present_index range to extract next.
uv run --with pyarrow scripts/fiba_perception_to_arrow.py \
  --method 2VP_4510 \
  -o examples/frames.fiba.perception.arrow

# 2. extract the full broadcast span (present_index 0..287) at native 1080p
mkdir -p output/fiba/frames
ffmpeg -i "$FIBA_MP4" -vf "select='between(n,0,287)',scale=1920:1080" \
  -vsync 0 -start_number 0 -q:v 3 output/fiba/frames/frame_%06d.jpg

# 3a. stage 2 — perception → placed scene (bunny on the court quad, Pose-free #77)
#     0.35 scale + a −1.6 half-edge shift along the quad's local −green (−r2)
#     gizmo axis lifts the small bunny up-court, above the players roaming the
#     key, so it never occludes an active player (it stays anchored in the
#     placement-quad's P² local frame). --place-offset-e1 shifts along the red
#     (r1) axis, --place-offset-e2 along the green (r2) axis, in half-edge units.
uv run --with pyarrow --with numpy examples/placement_quad_by_local_coord.py \
  --from-perception examples/frames.fiba.perception.arrow --place-mesh --placement-quad \
  --size-factor 0.35 --place-offset-e1 0.0 --place-offset-e2 -1.6 \
  --src-width 1920 --src-height 1080 --width 1920 --height 1080 \
  -o examples/frames.fiba.stage2.jsonl
examples/render.sh --cli --placement-quad --axes-local --aabb \
  --mesh assets/meshes/bunny_with_texture/bunny.obj \
  --texture assets/meshes/bunny_with_texture/bunny_uv_map1.jpg \
  --frames-base output/fiba \
  examples/frames.fiba.stage2.jsonl output/fiba_stage2.mp4 1920 1080 24

# 3b. stage 1 — the anchor before the mesh (placement quad + local axes, no bunny)
uv run --with pyarrow --with numpy examples/placement_quad_by_local_coord.py \
  --from-perception examples/frames.fiba.perception.arrow --no-place-mesh --placement-quad \
  --placement-quad-mesh-index 0 --src-width 1920 --src-height 1080 --width 1920 --height 1080 \
  -o examples/frames.fiba.stage1.jsonl
examples/render.sh --cli --placement-quad --axes-local \
  --frames-base output/fiba \
  examples/frames.fiba.stage1.jsonl output/fiba_stage1.mp4 1920 1080 24

# 3c. XY floor grid + local axes — a coordinate-plane grid laid in the placement
#     quad's P² local frame (#110), with `--axes-local` drawing that frame's axes
#     (red = r1, green = r2, blue = normal) at the quad origin. `--grid-local xy`
#     overlays a PlaneGrid on each *wireframe* draw's local XY plane; since the
#     placement quad is the wireframe draw, that is exactly one lattice carpeting
#     the recovered court floor (extends ~3× past the quad so the found plane is
#     easy to eyeball) — a filled/textured content mesh (the bunny) gets no stray
#     grid. `xz`/`yz` pick the other coordinate planes. Works on the quad-only
#     stage-1 stream …
examples/render.sh --cli --placement-quad --grid-local xy --axes-local \
  --frames-base output/fiba \
  examples/frames.fiba.stage1.jsonl output/fiba_stage1_grid.mp4 1920 1080 24
# … and on the full bunny + quad stage-2 scene (grid + P² axes on the quad floor,
#     AABB + local axes on the bunny — the grid never walls off the bunny thanks
#     to wireframe scoping):
examples/render.sh --cli --placement-quad --grid-local xy --axes-local --aabb \
  --mesh assets/meshes/bunny_with_texture/bunny.obj \
  --texture assets/meshes/bunny_with_texture/bunny_uv_map1.jpg \
  --frames-base output/fiba \
  examples/frames.fiba.stage2.jsonl output/fiba_stage2_grid.mp4 1920 1080 24
```

At 1080p (and 4K) an animated GIF balloons to hundreds of MB, so the output is
written as **H.264 `.mp4`** — [`scripts/encode.py`](scripts/encode.py) picks the
codec from the `-o` extension (`.mp4`/`.mov` → H.264, `.webp` → animated WebP,
else GIF). To render the NBA-style GIF instead, just name the output
`output/fiba_stage2.gif`. On Windows, run the same steps with
`examples\render.ps1 -CLI`.


#### The render pipeline

Under the hood it is a fully-piped `JSONL → Arrow → trd → ffmpeg` flow, no
intermediate files:

```sh
# producer → renderer → encoder   (mesh-first; pyarrow producers)
uv run --with pyarrow scripts/obj_to_arrow.py assets/meshes/bunny.obj > /tmp/stream.arrow
uv run --with pyarrow scripts/jsonl_to_arrow.py examples/frames.bunny_dolly.cg.jsonl >> /tmp/stream.arrow
cat /tmp/stream.arrow \
  | cargo run -q -p trd-cli -- --width 256 --height 256 \
  | uv run --with pyarrow --with numpy scripts/encode.py --fps 30 -o output/out.gif
```

The stream protocol is **`0.0.5`-only** and **mesh-first**: every input is
`[mesh][texture?][params]` (the leading mesh table authored by
[`scripts/obj_to_arrow.py`](scripts/obj_to_arrow.py), the per-frame params by
[`scripts/jsonl_to_arrow.py`](scripts/jsonl_to_arrow.py)). Each JSONL frame gives
its `model` transform as a 4×4 matrix (defaulting to identity when the frame is
driven entirely by its `draws` list), plus optional per-frame camera / draw-list /
`frame_path` columns. Older wire formats (`0.0.1`–`0.0.4`) are **not** accepted —
a stream declaring any other version is hard-rejected (see
[`docs/protocol/0.0.5.md`](docs/protocol/0.0.5.md) for the full specification).

- **Producer** — emits the input stream via the pyarrow scripts
  [`obj_to_arrow.py`](scripts/obj_to_arrow.py) (mesh),
  [`texture_to_arrow.py`](scripts/texture_to_arrow.py) (optional texture) and
  [`jsonl_to_arrow.py`](scripts/jsonl_to_arrow.py) (params).
- **`trd-cli`** — renders each row to `r,g,b,a` tensors (the output stream).
- **[`scripts/encode.py`](scripts/encode.py)** — pipes RGBA to ffmpeg, producing
  `.gif` or `.webp` by output extension. On WSL, prefix the `cargo` step with
  `WGPU_BACKEND=gl`.

### Native window

`trd-app` opens a window and plays the *same* params stream live (the desktop
counterpart of the browser target). Use the wrappers, or drive it directly:

```sh
examples/render.sh --native            # Linux/macOS
examples\render.ps1 -Native            # Windows (PowerShell 7)

# …or pipe the mesh-first stream straight into trd-app:
cat <(uv run --with pyarrow scripts/obj_to_arrow.py assets/meshes/bunny.obj) \
    <(uv run --with pyarrow scripts/jsonl_to_arrow.py examples/frames.bunny_dolly.cg.jsonl) \
  | cargo run -q -p trd-app -- --fps 30
```

Options: `--width`/`--height` (initial size), `--fps`, `--once` (hold the last
frame instead of looping). Honours `WGPU_BACKEND` / `RUST_LOG`. Close the window
to exit. In `--native` mode the output file is ignored and neither `uv` nor
`ffmpeg` is needed.

### Web (wasm)

```sh
nix build .#web    # Rust core → wasm-bindgen lib → bun dist/  (in ./result)
nix run   .#web    # serve dist/ over HTTP  (PORT, default 8080)
```

The [`examples/render.sh`](examples/render.sh) `--web` (alias `--wasm`) flag makes
the browser the **in-browser twin of `--cli`**: it runs the *same* Arrow producers
(mesh + texture + params) at the *same* scene flags, writes the resulting
`stream.arrow` plus a small `config.json` — and, with `--frames-base`, the
background stills — next to the bundled `index.html`, then serves the directory
with `static-web-server`. Handy for a remote GPU box, it prints the machine URL and
a ready-to-copy SSH-tunnel command first (`PORT` overrides the port, default 8080):

```sh
# On-screen WebGPU canvas (default target); tune fps live, resolution is baked in:
examples/render.sh --web --canvas-renderer --placement-quad --axes-local \
  --frames-base output/cornellbox \
  examples/frames.cornellbox.stage1.jsonl '' 960 540 25   # then open http://localhost:8080/?fps=30

# Offscreen OffscreenRenderer texture read back to a 2D canvas (browser twin of --cli output):
examples/render.sh --web --offscreen-renderer --placement-quad --axes-local --aabb \
  --mesh assets/meshes/bunny_with_texture/bunny.obj \
  --texture assets/meshes/bunny_with_texture/bunny_uv_map1.jpg \
  --frames-base output/cornellbox \
  examples/frames.cornellbox.stage2.jsonl '' 960 540 25

PORT=9000 examples/render.sh --web          # serve on a custom port
```

Every `--cli` content flag applies to `--web` unchanged — `--mesh`, `--texture`,
`--wireframe`, `--aabb`, `--axes`, `--axes-local`, `--placement-quad`,
`--frames-base`, and the positional `WIDTH`/`HEIGHT` (the `--grid-local` floor
grid is native/`--cli` + `--native` only). The render resolution is baked
into the stream's CV `k`, so it is a positional argument, **not** a URL param; the
only live URL param is **`?fps=N`** (1..240, default = the `FPS` positional). Two
render targets share the one bundle: **`--canvas-renderer`** (default) draws to the
on-screen WebGPU `CanvasRenderer`; **`--offscreen-renderer`** draws to an offscreen
`OffscreenRenderer` texture read back to RGBA and painted to a 2D canvas.

The server binds all interfaces, so browse to `http://<host-ip>:PORT` directly,
or forward it — `ssh -L 8080:localhost:8080 <user>@<host>`, then open
<http://localhost:8080>.

> **Windows:** [`examples/render.ps1`](examples/render.ps1) `-Web` (alias `-Wasm`)
> has **not** yet been ported to this config-driven model — it still builds the old
> `wasm-pack` demo bundle. The generic `render.sh --web` flow above is the current
> Nix/Linux path; aligning the PowerShell wrapper with it is a pending Windows
> follow-up.

The wasm core is a standard, TypeScript-typed npm package (`nix build .#trd-wasm`,
built with `wasm-bindgen-cli` + `wasm-opt`); its crate root is glue only, with the
two renderers in `crates/trd-wasm/src/{canvas_renderer,offscreen_renderer}.rs`. The
generic viewer fetches the prebuilt Arrow stream and replays it by index —
decoding it **once** with `loadIpc` (buffering every frame) rather than pushing
frame-by-frame:

```ts
import init, { CanvasRenderer } from "trd-wasm"; // fully typed

await init({ module_or_path: wasmUrl });
const canvas = await CanvasRenderer.create(canvasEl);
const total = canvas.loadIpc(streamBytes); // decode + buffer all frames
canvas.renderIndex(0);                     // draw buffered frame 0
```

`OffscreenRenderer` is the offscreen counterpart: it renders each buffered frame to an
offscreen texture, and its `renderIndex(i)` is **async**, returning that frame's
tightly-packed RGBA `Uint8Array` to paint onto a 2D canvas. Both renderers also
keep the streaming `pushIpc` path (append input / emit output, `finish()` → EOS)
for producer-driven pipelines.

`web/`'s npm dependencies (`apache-arrow` and its tree) are installed offline in
the Nix sandbox via [bun2nix](https://github.com/nix-community/bun2nix):
`web/bun.nix` pins them by hash from `web/bun.lock`. Regenerate it after changing
`web/bun.lock` — see [`AGENTS.md`](AGENTS.md) for the exact command.

For local iteration inside `nix develop`, `web/` uses `wasm-pack` + bun:

```sh
cd web
bun run build      # wasm-pack → pkg, then bun bundles → web/dist
bun run dev        # dev server; open the printed URL in a WebGPU browser
bun run check      # Biome format-check + lint (local @biomejs/biome)
bun run typecheck  # tsc --noEmit
```

(The `web` wasm-bindgen target is used because bun does not instantiate the
`bundler` target's ESM-imported wasm.)

### Windows setup (without Nix)

There is no `nix develop` on Windows. Its counterpart is
[`scripts/dev-env.ps1`](scripts/dev-env.ps1), which prepares the current
PowerShell 7 session the same way the flake's dev shell does.

**1. Install the toolchain and tools (one time).** Rust (`rustup`) and the MSVC
C++ build tools are required to build and link the core; the render-example extras
are optional. `winget` is shown below (scoop or manual installs work too):

```powershell
# Required: Rust + the MSVC C++ build tools (use the MSVC host, not -gnu).
winget install --id Rustlang.Rustup -e
winget install --id Microsoft.VisualStudio.2022.BuildTools -e `
  --override "--quiet --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended"

# Optional render-example tools. ffmpeg is only needed for the GIF/WebP path;
# uv is optional (the wrappers fall back to a system python that
# already has pyarrow + numpy).
winget install --id Gyan.FFmpeg -e
winget install --id astral-sh.uv -e     # optional
```

**2. Prepare each shell.** Dot-source the setup script to put everything on `PATH`
— cargo pinned to the MSVC host, the MSVC linker imported from `vcvars64.bat`, plus
ffmpeg/uv. It installs a missing `uv` via winget automatically (pass
`-NoInstall` to skip, `-Quiet` to hide the summary):

```powershell
. .\scripts\dev-env.ps1     # prints a dev-shell-style tool summary
```

**3. Build and render** with plain `cargo` / the example wrapper:

```powershell
cargo build -p trd-cli      # cargo can now link native binaries
examples\render.ps1 -CLI    # render examples\frames.bunny_dolly.cg.jsonl → output\out.gif
```

Notes:

- Use the **MSVC** Rust host, not `-gnu`: wgpu's raw-dylib dependencies crash at
  runtime on the `-gnu` host. `dev-env.ps1` runs
  `rustup set default-host x86_64-pc-windows-msvc` for you.
- `examples\render.ps1` auto-sources `dev-env.ps1` (with `-NoInstall`), so step 2
  is optional when you only run the wrapper. Set `$env:TRD_SKIP_DEV_ENV = '1'` to
  manage the environment yourself.
- On WSL, set `$env:WGPU_BACKEND = 'gl'` first for GPU rendering (otherwise the
  Vulkan backend falls back to software).
- The **web** wrapper also builds on Windows with just `bun` (no Nix): `cd web;
  bun run build:wasm; bun install; bun run typecheck; bun run check; bun run dev`.
  `@biomejs/biome` is a local dev dependency, so `bun run check` works outside the
  Nix dev shell. `apache-arrow` is fetched by `bun install`.

### Tests

```sh
cargo fmt --all --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace                 # fast; GPU tests are skipped
cargo test --workspace -- --ignored    # GPU-gated render tests
```

The `--ignored` render tests need a real GPU adapter. On a native Linux box that
isn't NixOS, run them through [nixGL](https://github.com/nix-community/nixGL) so
the `nix develop` Vulkan loader finds the host driver:

```sh
NIXPKGS_ALLOW_UNFREE=1 nix run --impure github:nix-community/nixGL#nixGLNvidia -- \
  cargo test --workspace -- --ignored   # #nixGLIntel for Intel/Mesa; WSL uses WGPU_BACKEND=gl
```

The **golden / snapshot render test** (`crates/trd-core/tests/golden_render.rs`,
#88) is one of these `--ignored` GPU tests: it runs committed Arrow fixtures
(`crates/trd-core/tests/golden/stage{1,2}.arrow`) through the real `run_stream`
pipeline and pixel-diffs the frames against committed golden PNGs. Each frame
keeps its `0.0.5` `frame_path`, which the test resolves against
`tests/golden/frames/` (the committed cornellbox stills) and composites the
scene over — an **AR composite over the cornellbox background** (the test-side
equivalent of `--frames-base`). Regenerate after changing the fixtures or making
an intended visual change:

```sh
python3 scripts/golden_fixtures.py                                   # rebuild the .arrow inputs + stills (needs uv + ffmpeg)
TRD_UPDATE_GOLDENS=1 cargo test -p trd-core --test golden_render -- --ignored   # refresh golden PNGs (GPU)
```

The companion `tests/decoder_parity.rs` decodes the same fixtures through both
the native and wasm decoders and asserts identical frames; it needs no GPU and
runs in the normal `cargo test` / `nix flake check` gate.
