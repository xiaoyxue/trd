# trd

**A tile (relational) oriented renderer, built on Rust + wgpu.**

`trd-core` is the *single* rendering core. The exact same Rust/wgpu code renders
everywhere — a headless CLI, a native window, the browser, and an interactive
viewer — by drawing into whatever render target each front-end provides.
JavaScript/TypeScript is a thin bootstrap only; the WebGPU API is never called
from JS.

> Contributor/agent guidance (build system internals, GPU-adapter selection,
> testing policy, PR workflow) lives in **[`AGENTS.md`](AGENTS.md)**. This README
> is the user-facing "what it is / how to run it" guide.

## How it fits together

Everything shares **one render function** and **one data format** (a mesh-first
Arrow stream):

```
input-stream ─┬─ trd-cli  → trd-core → offscreen readback → image-stream   (headless)
(mesh-first)  ├─ trd-app  → trd-core → window surface                      (native playback)
              ├─ trd-wasm → trd-core → canvas surface                      (browser)
              └─ trd-gui  → trd-core → offscreen → egui image      (interactive, native + browser)

                  image-stream → scripts/encode.py → ffmpeg → GIF / WebP / MP4
```

### The render core — `trd-core`

Platform-agnostic wgpu logic, shared verbatim by every target:

- **`render/` (module tree) + `*.wgsl` shaders** — `MeshRenderer`
  (`render/mesh_renderer.rs`) rasterizes a `Scene` of `DrawableObject`s into *any*
  `wgpu::TextureView`; that one renderer is why the same code targets an offscreen
  texture, a window swapchain, or a browser canvas. The offscreen render target +
  async pixel read-back is factored into a shared `OffscreenTarget` harness
  (`render/offscreen.rs`), and the on-screen present path into `OnscreenTarget`
  (`render/onscreen.rs`), each reused by every front-end of its kind.
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

### The front-ends

Each is a *thin shell* that only supplies a render target and calls the core:

| Front-end | Reads | Renders into | Produces |
|---|---|---|---|
| **`trd-cli`** | Arrow stream (stdin) | offscreen texture → read-back | Arrow image stream (stdout) |
| **`trd-app`** | Arrow stream (stdin) | live window swapchain | frames on screen |
| **`trd-wasm`** | Arrow stream (buffered via `loadIpc`) | live canvas (or offscreen texture) | frames in the browser |
| **`trd-gui`** | a mesh + live gestures | offscreen texture → egui image | an interactive orbit/zoom viewer (native + browser) |

- **`trd-cli`** — headless Arrow filter: renders each frame to an offscreen
  texture and writes the pixels as an Arrow image stream. It does **not** encode
  video; pipe the stream to [`scripts/encode.py`](scripts/encode.py) (ffmpeg) for a
  GIF/WebP/MP4.
- **`trd-app`** — native window: a background thread reads the mesh-first stream
  from stdin; the window plays it at `--fps`, drawing each frame straight into the
  swapchain surface. No read-back, no file.
- **`trd-wasm` / `web/`** — browser: `CanvasRenderer.create(canvas)` holds a
  persistent `MeshRenderer` + `InputSession` and renders the **same** `Scene` as
  the CLI. There is **one** config-driven front-end: `render.sh --web` writes the
  demo's `stream.arrow` + `config.json`, and [`web/src/viewer.ts`](web/src/viewer.ts)
  fetches both and replays by index. Two targets share the bundle: the on-screen
  `CanvasRenderer` and the offscreen `OffscreenRenderer` (renders to a texture,
  reads it back, paints a 2D canvas). JS only moves Arrow bytes; it never touches
  WebGPU. Ships as the `trd-wasm` npm library.
- **`trd-gui`** — interactive viewer (native + browser): turns orbit/zoom/pan
  gestures into an updated camera + model matrix and re-renders one mesh through
  `trd-core`, offscreen, shown as an egui image. `--backend arrow` (or
  `?backend=arrow`) round-trips each frame through the real Arrow wire — the seam
  an external producer would drive.

## Stream protocol

Frame parameters are plain columnar data, so **any** tool that emits the input
columns as an Arrow IPC stream can drive the renderer. The current — and **only
supported** — version is **0.0.5**: **mesh-first** (`[mesh][texture?][params]`)
and **not** backward-compatible with `0.0.1`–`0.0.4` (older streams are
hard-rejected).

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

All params columns are **optional/additive** and drive the MVP transform
`clip = P · V · M · (pos, 1)`. The stream carries three optional features on top of
the mesh geometry:

- a per-frame **camera** — **CV** (`k` intrinsics + `pose`, `view = inverse(pose)`)
  or **CG** (a look-at from `eye` + `target`/`direction` + `up`, with
  `fovy`/`aspect`/`znear`/`zfar`). CV wins per component; absent any camera column,
  identity view + default perspective.
- a per-frame **draw list** (`draw_mesh` + `draw_model`, equal-length) that
  instances several meshes in one frame. Absent a draw list, one instance of mesh 0
  is placed by the frame's own `model`.
- a per-frame **background frame** — a `frame_path` (native) / `frame_url`
  (browser) image composited **beneath** the scene by a `FramePlane`. The core
  decodes the reference only; the shell does the image I/O (`--frames-base <dir>`).

**Rendering appearance is a render-time choice, not a wire column** — filled,
wireframe, textured, and **PBR** are CLI/render modes (see
[Rendering appearance](#rendering-appearance)), so the same stream renders any way.

**Full specification: [`docs/protocol/0.0.5.md`](docs/protocol/0.0.5.md).**

## Repository layout

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

## Quick start

**1. Get a dev environment.**

- **Linux / macOS / WSL** — [Nix](https://nixos.org/download) is the build system
  *and* dev shell (pinned Rust, `bun`, wasm tools, `biome`, ffmpeg, Vulkan):

  ```sh
  nix develop
  ```

- **Windows** — no Nix; dot-source the setup script (see [Windows setup](#windows-setup-without-nix)):

  ```powershell
  . .\scripts\dev-env.ps1
  ```

**2. Run the demo** (the bunny dolly-camera capstone, loading `assets/meshes/bunny.obj`):

```sh
examples/render.sh --cli      # Linux/macOS/WSL — render → output/out.gif
examples/render.sh --native   # play live in a window
```

```powershell
examples\render.ps1 -CLI      # Windows (PowerShell 7)
examples\render.ps1 -Native
```

Run either wrapper with no arguments to print its flag guidance.

**3. Try the web build:**

```sh
examples/render.sh --web   # generate the demo stream, build + serve  (Linux/macOS/WSL)
```

Open the printed URL in a WebGPU browser (Chrome/Edge). On a remote (SSH) host, run
the printed tunnel command first, then browse to <http://localhost:8080>.

> **GPU on Linux without NixOS / on WSL.** On WSL prefix GPU commands with
> `WGPU_BACKEND=gl`. On a native (non-NixOS) Linux GPU box, the `nix develop`
> Vulkan loader can't reach the host driver — wrap GPU commands with
> [nixGL](https://github.com/nix-community/nixGL)
> (`NIXPKGS_ALLOW_UNFREE=1 nix run --impure github:nix-community/nixGL#nixGLNvidia -- <cmd>`).
> Full GPU-selection details are in [`AGENTS.md`](AGENTS.md#gpu).

## Building & running

The Nix flake is the build system — reproducible outputs, no manual toolchain:

```sh
nix build .#trd-cli   # native CLI (Arrow stream filter) + Vulkan/GL runtime libs
nix build .#trd-wasm  # wasm-bindgen JS/TS library package
nix build .#web       # bun-bundled, HTTP-servable dist/  (also plain `nix build`)
nix run   .#trd -- --width 256 --height 256   # stream on stdin → images on stdout
nix run   .#web                               # serve dist/  (PORT, default 8080)
nix flake check       # every gate: fmt, clippy (native+wasm32), test, tsc, biome
```

> `nix build` / `nix flake check` only see git-tracked files — `git add` new files
> before building.

For fast iteration use plain `cargo` / `bun` inside `nix develop` (or, on Windows,
after `. .\scripts\dev-env.ps1`).

### Native CLI

`trd` (package `trd-cli`) is a pure Arrow filter: stream in → image stream out. The
`examples/render.*` wrappers build the whole `JSONL → Arrow → trd → ffmpeg`
pipeline for you (no intermediate files):

```sh
examples/render.sh  [MODE] [INPUT.jsonl] [OUT.gif|.webp|.mp4] [WIDTH] [HEIGHT] [FPS]
# MODE (pick one): --cli (headless, default) · --native (live window) · --web (browser)
# Defaults: examples/frames.bunny_dolly.cg.jsonl → output/out.gif, 256×256 @ 30 fps
```

`scripts/encode.py` picks the codec from the `-o` extension (`.mp4`/`.mov` → H.264,
`.webp` → animated WebP, else GIF) — use `.mp4` for 1080p/4K, where a GIF balloons
to hundreds of MB.

The raw producer → renderer → encoder pipeline the wrappers run:

```sh
uv run --with pyarrow scripts/obj_to_arrow.py assets/meshes/bunny.obj  > /tmp/stream.arrow
uv run --with pyarrow scripts/jsonl_to_arrow.py examples/frames.bunny_dolly.cg.jsonl >> /tmp/stream.arrow
cat /tmp/stream.arrow \
  | cargo run -q -p trd-cli -- --width 256 --height 256 \
  | uv run --with pyarrow --with numpy scripts/encode.py --fps 30 -o output/out.gif
```

On Windows the Arrow stages are handed off through a temp dir (PowerShell pipelines
aren't binary-safe); the output is identical.

#### Content & appearance flags

The headless (`--cli`) flags map straight onto `trd-cli`, so any producer pipeline
can use them (`… | trd --width 1024 --wireframe --aabb | …`):

| Flag | Effect |
|---|---|
| `--mesh <obj>` | Prepend a mesh built from `<obj>` by [`obj_to_arrow.py`](scripts/obj_to_arrow.py); renders centered + scaled-to-fit. **Repeatable** — several meshes load in order; a frame's `draws` list references them by 0-based index. |
| `--texture <img>` | Splice a texture stream from `<img>` and render **textured** (samples the image at each vertex UV). Requires `--mesh` with UVs. |
| `--wireframe` | Draw mesh **edges** as a line list instead of filled triangles. |
| `--pbr` | Physically-based **Disney principled BRDF** shading (see [Rendering appearance](#rendering-appearance)). |
| `--aabb` | Overlay each drawn instance's green axis-aligned **bounding box**. |
| `--axes` / `--axes-local` | Overlay a coordinate-axes gizmo at the **world** origin / at **each** drawn object's own model frame. |
| `--frames-base <dir>` | Composite each frame's **background still** (its `frame_path`, relative to `<dir>`) beneath the scene via a `FramePlane`. |
| `--no-msaa` | Disable 4× MSAA (default on the mesh pass); render single-sampled/aliased. |

```sh
# Single bunny turntable, filled, with its bounding box:
examples/render.sh --cli --aabb --mesh assets/meshes/bunny.obj \
  examples/frames.turntable.jsonl output/bunny.gif 1024 1024 24

# Textured bunny (samples a UV-mapped albedo at each vertex UV):
examples/render.sh --cli --mesh assets/meshes/bunny_with_texture/bunny.obj \
  --texture assets/meshes/bunny_with_texture/bunny_uv_map1.jpg \
  examples/frames.bunny_dolly.cg.jsonl output/bunny_textured.gif 512 512 20

# Two-mesh scene (bunny = mesh 0, cube = mesh 1), wireframe + boxes:
examples/render.sh --cli --wireframe --aabb \
  --mesh assets/meshes/bunny.obj --mesh examples/cube.obj \
  examples/frames.multimesh.jsonl output/scene.gif 1024 1024 24
```

Advanced placement/overlay flags (`--placement-quad`, `--grid-local xy|xz|yz`) drive
the AR demos below; run `render.sh` with no arguments for the full list.

#### Rendering appearance

The same stream renders four ways — pick one per render:

- **filled** (default) — per-vertex color.
- **`--wireframe`** — edges as a line list; reveals topology.
- **`--texture <img>`** — samples a bound UV-mapped albedo (`Rgba8UnormSrgb`, linear
  clamp-to-edge). Requires UV-mapped geometry.
- **`--pbr`** — the physically-based **Disney principled BRDF** path with smooth
  shading normals: `--metallic` (0 dielectric → 1 metal), `--roughness` (0 mirror →
  1 rough), `--specular`, `--clearcoat`, plus an optional HDR **environment probe**
  `--env <file.hdr>` (equirectangular Radiance map) reflected by metallic surfaces
  (`--env-intensity`). HDR output is tone-mapped by `--tonemap reinhard|aces`
  (default `reinhard`; `aces` is the filmic curve), with `--exposure` / `--ambient`
  controls. Requires a texture stream for the albedo.

```sh
# Shiny metal bunny under an HDR environment probe, ACES tone-map:
examples/render.sh --cli --pbr --metallic 1 --roughness 0.3 --tonemap aces \
  --env assets/envmap/uffizi-large.hdr \
  --mesh assets/meshes/bunny_with_texture/bunny.obj \
  --texture assets/meshes/bunny_with_texture/bunny_uv_map1.jpg \
  examples/frames.bunny_dolly.cg.jsonl output/bunny_pbr.gif 512 512 20
```

Anti-aliasing: the mesh pass renders at **4× MSAA** by default (resolved before
read-back); `--no-msaa` renders single-sampled.

#### Camera forms & AR demos

- **Dolly-camera capstone** — [`examples/bunny_dolly.py`](examples/bunny_dolly.py)
  authors the *same* camera **twice**, once **CG** (`eye`/`target`/`up` + `fovy`)
  and once **CV** (pinhole `K` + `pose`), as two JSONL streams. Both decode to the
  same `P·V` and render identically (verified to differ by ≤ 0.0054 % of pixels) —
  a proof that the CV and CG paths agree. The CV `K` is in pixel units, so render
  the CV stream at the resolution it was authored for.
- **Background compositing** —
  [`examples/bunny_frameplane.py`](examples/bunny_frameplane.py) authors a folder of
  animated stills + a turntable JSONL whose `frame_path` names each one;
  `--frames-base <dir>` composites them beneath the spinning bunny.
- **Single-view AR placement (real broadcast clips)** — the perception pipeline
  ([`scripts/{nba,fiba}_perception_to_arrow.py`](scripts/) →
  [`examples/placement_quad_by_local_coord.py`](examples/placement_quad_by_local_coord.py))
  anchors a mesh to a tracked planar court quad (Pose-free reconstruction) so it
  **stays glued to the same floor spot as the camera pans and zooms**. Only the
  image-free per-frame calibration is vendored (`assets/videos/{nba,fiba}/…`, see
  each `DATASET.md`); the copyrighted broadcast video stays external. The packaged
  **FIBA / Paris-2024 Olympic** demo (Disney PBR + ACES, native 1080p) is one
  command (`--source` points at the external broadcast clip):

  ```sh
  examples/olympic-basketball-demo.sh --source ~/Asset/fiba-shot1/shot_0001.mp4
  ```

  Run it with no arguments to print its options (drink-can preset, shot, overlays).
  See the script header and `render.sh` with no args for the individual stages and
  the `--placement-quad` / `--axes-local` / `--grid-local` overlays.

### Native window

`trd-app` opens a window and plays the *same* stream live:

```sh
examples/render.sh --native            # Linux/macOS/WSL
examples\render.ps1 -Native            # Windows (PowerShell 7)

# …or pipe a mesh-first stream straight into trd-app:
cat <(uv run --with pyarrow scripts/obj_to_arrow.py assets/meshes/bunny.obj) \
    <(uv run --with pyarrow scripts/jsonl_to_arrow.py examples/frames.bunny_dolly.cg.jsonl) \
  | cargo run -q -p trd-app -- --fps 30
```

Options: `--width`/`--height` (initial size), `--fps`, `--once` (hold the last
frame instead of looping). Honours `WGPU_BACKEND` / `RUST_LOG`. Close the window to
exit; neither `uv` nor `ffmpeg` is needed.

### Web (wasm)

```sh
nix build .#web    # Rust core → wasm-bindgen lib → bun dist/  (in ./result)
nix run   .#web    # serve dist/ over HTTP  (PORT, default 8080)
```

`render.sh --web` (alias `--wasm`) is the **in-browser twin of `--cli`**: it runs the
same Arrow producers at the same scene flags, writes `stream.arrow` + `config.json`
(+ background stills with `--frames-base`) next to the bundled `index.html`, and
serves it — printing the machine URL and a ready-to-copy SSH-tunnel command:

```sh
# On-screen WebGPU canvas (default); tune fps live, resolution is baked in:
examples/render.sh --web --canvas-renderer --placement-quad --axes-local \
  --frames-base output/cornellbox \
  examples/frames.cornellbox.stage1.jsonl '' 960 540 25   # open http://localhost:8080/?fps=30

# Offscreen renderer read back to a 2D canvas (browser twin of --cli output):
examples/render.sh --web --offscreen-renderer --mesh assets/meshes/bunny_with_texture/bunny.obj \
  --texture assets/meshes/bunny_with_texture/bunny_uv_map1.jpg \
  --frames-base output/cornellbox \
  examples/frames.cornellbox.stage2.jsonl '' 960 540 25
```

Every `--cli` content flag applies to `--web` unchanged (`--grid-local` is
native-only). The render resolution is baked into the stream's CV `k`, so it is a
positional argument, **not** a URL param; the only live URL param is **`?fps=N`**.
Two targets share the bundle: **`--canvas-renderer`** (on-screen WebGPU) and
**`--offscreen-renderer`** (offscreen texture → 2D canvas).

The wasm core is a standard, TypeScript-typed npm package. The generic viewer
replays a prebuilt stream by index — decoding it **once** with `loadIpc`:

```ts
import init, { CanvasRenderer } from "trd-wasm"; // fully typed

await init({ module_or_path: wasmUrl });
const canvas = await CanvasRenderer.create(canvasEl);
const total = canvas.loadIpc(streamBytes); // decode + buffer all frames
canvas.renderIndex(0);                     // draw buffered frame 0
```

`OffscreenRenderer` is the offscreen counterpart: `renderIndex(i)` is **async**,
returning that frame's RGBA `Uint8Array` to paint onto a 2D canvas. Both also keep
the streaming `pushIpc` path (append input / emit output, `finish()` → EOS) for
producer-driven pipelines.

For local iteration inside `nix develop`, `web/` uses `wasm-pack` + bun:

```sh
cd web
bun run build      # wasm-pack → pkg, then bun bundles → web/dist
bun run dev        # dev server; open the printed URL in a WebGPU browser
bun run check      # Biome format-check + lint
bun run typecheck  # tsc --noEmit
```

`web/`'s npm deps are installed offline in the Nix sandbox via
[bun2nix](https://github.com/nix-community/bun2nix) (`web/bun.nix` pins them by hash
from `web/bun.lock`); regenerate it after changing `web/bun.lock` — see
[`AGENTS.md`](AGENTS.md) for the exact command.

> **Windows:** `render.ps1 -Web` still builds the older `wasm-pack` demo bundle; the
> config-driven `render.sh --web` flow above is the current Nix/Linux path.

### Windows setup (without Nix)

There is no `nix develop` on Windows. Its counterpart is
[`scripts/dev-env.ps1`](scripts/dev-env.ps1), which prepares the current PowerShell 7
session the same way the flake's dev shell does.

**1. Install the toolchain (one time).** Rust (`rustup`) + the MSVC C++ build tools
are required to build and link the core; the render-example extras are optional:

```powershell
# Required: Rust + MSVC C++ build tools (use the MSVC host, not -gnu).
winget install --id Rustlang.Rustup -e
winget install --id Microsoft.VisualStudio.2022.BuildTools -e `
  --override "--quiet --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended"

# Optional render tools: ffmpeg (GIF/WebP/MP4 path) + uv (pyarrow producers).
winget install --id Gyan.FFmpeg -e
winget install --id astral-sh.uv -e
```

**2. Prepare each shell** — dot-source the setup script to put cargo (pinned to the
MSVC host), the MSVC linker, ffmpeg, and uv on `PATH`:

```powershell
. .\scripts\dev-env.ps1     # prints a dev-shell-style tool summary
```

**3. Build and render:**

```powershell
cargo build -p trd-cli
examples\render.ps1 -CLI    # render → output\out.gif
```

Notes:

- Use the **MSVC** Rust host, not `-gnu` (wgpu's raw-dylib deps crash on `-gnu`);
  `dev-env.ps1` runs `rustup set default-host x86_64-pc-windows-msvc` for you.
- `render.ps1` auto-sources `dev-env.ps1` (`-NoInstall`), so step 2 is optional when
  you only run the wrapper.
- The **web** wrapper builds on Windows with just `bun` (no Nix): `cd web; bun run
  build:wasm; bun install; bun run typecheck; bun run check; bun run dev`.

## Tests

```sh
cargo fmt --all --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test  --workspace                 # fast; GPU tests are skipped
cargo test  --workspace -- --ignored    # GPU-gated render tests (need a real GPU)
```

The **golden / snapshot render test** (`crates/trd-core/tests/golden_render.rs`, #88)
is the pixel-level regression net: it runs committed Arrow fixtures through the real
`run_stream` pipeline and pixel-diffs each frame against committed golden PNGs, at
**both** 4× MSAA and MSAA-off, plus PBR tone-map variants. It is GPU-gated
(`--ignored`); the companion `tests/decoder_parity.rs` needs no GPU and runs in
`nix flake check`.

The full testing policy (required tiers, MSAA-on/off requirement, per-crate e2e,
multi-platform verification + handoff, and golden regeneration) lives in
**[`AGENTS.md`](AGENTS.md#testing--required-before-a-task-is-done)**.
