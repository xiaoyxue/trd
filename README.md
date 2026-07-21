# trd

**A tile (relational) oriented renderer, built on Rust + wgpu.**

`trd-core` is the *single* rendering core. The exact same Rust/wgpu code renders
in three places — a headless CLI, a native window, and the browser — by drawing
into whatever render target each one provides. JavaScript/TypeScript is a thin
bootstrap only; the WebGPU API is never called from JS.

## How it fits together

Everything shares **one render function** and **one data format**:

```
params-stream ─┬─ trd-cli  → trd-core → offscreen readback → image-stream   (headless)
               ├─ trd-app  → trd-core → window surface                      (native GUI)
               └─ trd-wasm → trd-core → canvas surface                      (browser)

                  image-stream → scripts/encode.py → ffmpeg → GIF / WebP
```

### The render core — `trd-core`

Platform-agnostic wgpu logic, shared verbatim by every target:

- **`render.rs` + `triangle.wgsl`** — `render_triangle(device, queue, view, format,
  params)` draws one parametric triangle (`FrameParams` = `center`, `size`,
  `theta`) into *any* `wgpu::TextureView`. That one function is why the same code
  targets an offscreen texture, a window swapchain, or a browser canvas.
- **`DrawableObject` + `Scene` (`render.rs`)** — the base interface for every
  primitive the renderer can draw (#41). `DrawableObject` is a small `Copy` enum —
  `Mesh { mesh_id, model, mode }` (filled or **wireframe** mode), `AabbBox {
  mesh_id, model }`, and `CoordinateAxes { model }` — where geometry (GPU buffers)
  is owned once by the renderer's decode-once mesh store and each variant carries
  only *which* primitive to draw plus its per-frame model. A `Scene = Vec<
  DrawableObject>` is rebuilt each frame; `MeshRenderer::encode` walks it once,
  binds the shared `P·V·M` uniform, buckets the drawables (filled meshes → wireframe
  meshes → AABB boxes → axes) into one instance buffer, and records the draws — with
  **no per-type branching** in any front-end. A single-object frame is the
  degenerate one-element scene, so there is no special case. Wireframe is a *mode*
  of the mesh drawable, not a separate primitive; the AABB box and axes gizmo are
  core-side additions to the scene (not wire columns).
- **`stream.rs`** — the native Arrow IPC filter: `read_frame_stream` decodes the
  input frames; `run_stream` is the CLI filter. Only one record batch is ever in
  flight, so an animation of any length streams in constant memory.
- **`protocol.rs`** — the cross-platform (native + wasm) incremental Arrow IPC
  decoder. `InputSession` feeds arbitrary byte chunks through `arrow`'s
  `StreamDecoder`, validates the protocol schema once (accepts
  `0.0.1`/`0.0.2`/`0.0.3`/`0.0.4`/`0.0.5`), and yields one `FrameBatch` (`Vec<FrameParams>`) per
  record batch — the
  browser's input path.
- **`output.rs`** — the cross-platform Arrow IPC *output* serialization.
  `OutputSession` writes the `r,g,b,a` `fixed_shape_tensor<u8>` stream
  incrementally (one output batch per input batch); `tightly_pack_rgba` strips GPU
  row padding. Shared by the native CLI and the browser `ArrowRenderer`.
- **`math/`** — the typed homogeneous linear-algebra layer over glam:
  `Vector2/3/4`, `Point2/3/4`, `Normal3`, `Matrix3/4`, `Rotation` (unit
  quaternion), `Transform`, and `Aabb2/3`. Zero-cost `#[repr(transparent)]`
  newtypes with **private** inner fields that enforce affine-space rules
  (`point − point → vector`, no `point + point`) the raw glam types can't.
  Column-major, right-handed, clip `z ∈ [0, 1]`; `render.rs`'s MVP transforms
  are built on it and its `ToWgsl` layout keeps the GPU `Uniform` byte-identical.

### The three consumers

Each is a *thin shell* that only supplies a render target and calls the core:

| Target | Reads | Renders into | Produces |
|---|---|---|---|
| **`trd-cli`** | Arrow params stream (stdin) | offscreen texture → pixel read-back | Arrow image stream (stdout) |
| **`trd-app`** | Arrow params stream (stdin) | live window swapchain | frames on screen |
| **`trd-wasm`** | Arrow stream (buffered via `loadIpc`) | live canvas surface (or offscreen texture) | frames in the browser |

- **`trd-cli` — headless Arrow filter.** For each input frame it renders to an
  offscreen texture, copies the pixels back (`copy_texture_to_buffer`), and writes
  them as an Arrow image stream. It does **not** encode video itself — piping that
  stream to `scripts/encode.py` (ffmpeg) turns it into a GIF/WebP.
- **`trd-app` — native window.** A background thread reads the params stream from
  stdin; the window plays it at `--fps`, drawing each frame **straight into the
  swapchain surface** and presenting it. No read-back, no file — pixels go on
  screen. With no stdin it shows the identity triangle.
- **`trd-wasm` / `web/` — browser.** `CanvasRenderer.create(canvas)` obtains a wgpu
  surface from the `<canvas>` and holds a persistent `MeshRenderer` plus an
  `InputSession`, rendering the **same mesh Scene** as the native CLI through the
  shared `build_scene`/`MeshRenderer` path — no per-frontend branching. There is
  **one** config-driven front-end: `render.sh --web` runs the *same* Arrow producers
  and scene flags as `--cli` and writes `stream.arrow` + `config.json` into the
  served directory; the tiny [`web/src/main.ts`](web/src/main.ts) loads
  [`web/src/generic-renderer.ts`](web/src/generic-renderer.ts), which fetches both,
  decodes the whole stream once with `loadIpc` (buffering every frame), and replays
  it by index with `renderIndex(i)`. Two targets share the bundle — the on-screen
  `CanvasRenderer` (default) and the offscreen `ArrowRenderer` (renders to a texture,
  reads it back to RGBA, and paints it to a 2D canvas). Modes/overlays come from the
  config via `setWireframe`/`setTextured`/`setShowAabb`/`setShowAxes`/`setShowLocalAxes`,
  and `setCompositeFrame` + `updateFrameTextureRgba` composite each frame's 0.0.5
  background still. JS only moves Arrow bytes and schedules frames; it never touches
  the WebGPU API. The crate root is glue only — the two renderers live in
  `crates/trd-wasm/src/{canvas_renderer,arrow_renderer}.rs` — and it ships as the
  `trd-wasm` npm library.

### Stream protocol

Frame parameters are plain columnar data, so **any** tool that emits the input
columns as an Arrow IPC stream can drive the renderer. The current version is
**0.0.4**; it is backward-compatible with 0.0.3, 0.0.2 and 0.0.1.

| Direction | Columns | Arrow type |
|---|---|---|
| **Input** (params) | `center`, `size` | `FixedSizeList<f32>[2]` |
| | `theta` | `f32` |
| | `model` *(opt, 0.0.2)* | `FixedSizeList<f32>[16]` (4×4 model matrix) |
| | `k` *(opt, 0.0.2)* | `FixedSizeList<f32>[9]` (3×3 camera intrinsics, **CV**) |
| | `pose` *(opt, 0.0.2)* | `FixedSizeList<f32>[16]` (4×4 camera-to-world pose, **CV**) |
| | `eye`, `target`, `direction`, `up` *(opt, 0.0.3)* | `FixedSizeList<f32>[3]` (**CG** look-at) |
| | `fovy`, `aspect`, `znear`, `zfar` *(opt, 0.0.3)* | `f32` (**CG** perspective) |
| | `draw_mesh` *(opt, 0.0.3)* | `List<u32>` (per-instance mesh index) |
| | `draw_model` *(opt, 0.0.3)* | `List<FixedSizeList<f32>[16]>` (per-instance 4×4 model) |
| | `frame_path` / `frame_url` *(opt, 0.0.5)* | `Utf8` (per-frame background image path / URL) |
| **Input** (mesh, *opt, 0.0.3*) | `position`, `color` | `List<FixedSizeList<f32>[3]>` |
| | `uv` *(opt, 0.0.4)* | `List<FixedSizeList<f32>[2]>` (per-vertex texture coords) |
| | `index` | `List<u32>` |
| **Input** (texture, *opt, 0.0.4*) | `rgba` | `fixed_shape_tensor<u8>` `[H, W, 4]` (interleaved RGBA) |
| **Output** (image) | `r`, `g`, `b`, `a` | `fixed_shape_tensor<u8>` `[H, W]` |

The `0.0.2` matrix columns are **optional/additive** and drive the MVP transform
`clip = P · V · M · (pos, 0, 1)`; a stream with none of them (or identity
matrices) renders identically to `0.0.1`. The protocol-version metadata is
optional, so DuckDB and pyarrow streams are both accepted as-is.

**0.0.3** adds three things on top of the params stream:

- an optional leading **mesh** Arrow stream, concatenated before the params
  stream (`[mesh][params]`): one row per mesh, geometry in nested list columns.
  **Both the native path and the browser `CanvasRenderer`** decode it
  (`Mesh::from_arrow`), upload it, and render it **centered and uniformly scaled
  to fit** (a `base_model` beneath the per-frame `model`), driven by the
  following params. Encode an OBJ with `scripts/obj_to_arrow.py`;
  `examples/render.sh --mesh <obj>` wires it end-to-end.
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

**0.0.4** adds **textured rendering**: an optional **texture** Arrow stream
spliced between the mesh and params streams (`[mesh][texture][params]`) — one row
of interleaved-RGBA `rgba` (`fixed_shape_tensor<u8>` `[H, W, 4]`, dimensions
self-describing) — plus an optional per-vertex **`uv`** mesh column. With
`--textured` (`setTextured(true)`) meshes sample the bound texture at each UV
(`textureSample`, `Rgba8UnormSrgb` linear clamp-to-edge); a textured draw with no
texture stream samples a default 1×1 white. Encode an image with
`scripts/texture_to_arrow.py`; `examples/render.sh --texture <img>` wires it
end-to-end for `--cli` and `--web` alike (`.ps1 -Texture` on Windows).

A params-only stream (no leading mesh) still renders the built-in hello-triangle.

**0.0.5** adds an optional per-frame **background frame**: a `frame_path` (native)
or `frame_url` (browser) `Utf8` params column naming an image composited **beneath**
the scene by a new `FramePlane` drawable (a fullscreen quad sampling a reused
`Rgba8UnormSrgb` texture, depth-write off so the mesh scene + gizmos draw on top;
`Stretch`/`Cover` fit). The core decodes the reference only; the shell does the
image I/O — `trd --frames-base <dir>` / `trd-app --frames-base <dir>` load the
PNG/JPEG (relative to `<dir>`). Produce the stills + manifest with
`scripts/extract_frames.py`. Without `--frames-base`, `frame_path` is ignored and
the scene renders over the black clear.

**Full, versioned specification: [`docs/protocol/`](docs/protocol/)** (per-version
schema reference + [changelog](docs/protocol/CHANGELOG.md)).


## Repository layout

| Path | What it is |
|---|---|
| `crates/trd-core` | the unified render core (`render.rs`, `triangle.wgsl`, `stream.rs`) |
| `crates/trd-cli` | headless CLI: Arrow params in → Arrow image out |
| `crates/trd-app` | native interactive window (winit + live wgpu surface); split into `main`/`cli`/`error`/`renderer`/`stream`/`app` modules |
| `crates/trd-wasm` | `wasm-bindgen` entry point (crate-root glue + `canvas_renderer`/`arrow_renderer` modules); packaged as the `trd-wasm` npm library |
| `web/` | bun-managed thin TypeScript wrapper (`main.ts` → config-driven `generic-renderer.ts`) that loads `trd-wasm` |
| `examples/` | `frames.0.0.2.jsonl` (+ legacy `frames.0.0.1.jsonl`) demo + `render.sh` / `render.ps1` wrappers |
| `scripts/jsonl_to_arrow.py` | JSONL → Arrow params stream (pyarrow; duckdb-free producer) |
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

**2. Run the demo** — renders [`examples/frames.0.0.2.jsonl`](examples/frames.0.0.2.jsonl):

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
after `. .\scripts\dev-env.ps1`). The sections below assume that. DuckDB is
optional — see [the render pipeline](#the-render-pipeline).

### Native CLI

`trd` (package `trd-cli`) is a pure Arrow filter: params stream in → image stream
out. The `examples/render.*` wrappers build the whole JSONL → GIF pipeline for you:

```sh
examples/render.sh  [MODE] [INPUT.jsonl] [OUT.gif|.webp] [WIDTH] [HEIGHT] [FPS]   # Linux/macOS
examples\render.ps1 [MODE] [-InputPath]  [-Output]       [-Width] [-Height] [-Fps]  # Windows (PS7)
# Defaults: examples/frames.0.0.2.jsonl → output/out.gif, 256×256 @ 30 fps
# MODE (pick one): --cli/-CLI (default: headless GIF/WebP) · --native/-Native (live window) ·
#   --web/-Wasm (browser; --canvas-renderer default, or --offscreen-renderer/--arrow-renderer)
# Run either wrapper with no arguments (or -h/--help / -Help) to print the flag
# guidance and exit; pass a mode (e.g. --cli) to render the default demo.
```

On Windows the Arrow stages are handed off through a temp dir (Windows DuckDB
can't write to `/dev/stdout` and PowerShell pipelines aren't binary-safe); the
output is identical.

#### Mesh & render flags (`--cli`)

Beyond the `MODE`, the headless (`--cli`) path takes content/appearance flags that
map straight onto `trd-cli`:

| Flag | Effect |
|---|---|
| `--mesh <obj>` | Prepend a mesh Arrow stream (protocol 0.0.3 `[mesh][params]`) built from `<obj>` by [`scripts/obj_to_arrow.py`](scripts/obj_to_arrow.py); the mesh renders centered + scaled-to-fit, driven by the params `INPUT.jsonl`. **Repeatable** — pass `--mesh` several times to load several meshes (one table row each, in order); a frame's `draws` list then references them by 0-based index. Needs pyarrow (via uv/python3). |
| `--texture <img>` | Splice a texture Arrow stream (protocol 0.0.4 `[mesh][texture][params]`) built from `<img>` by [`scripts/texture_to_arrow.py`](scripts/texture_to_arrow.py) and render **textured** (`trd --textured`, #20): meshes sample the image at each vertex UV. Requires `--mesh` (with UVs); mutually exclusive with `--wireframe`. Downscaled to ≤ 2048² (portable limit); needs pyarrow + pillow + numpy. |
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

#### The render pipeline

Under the hood it is a fully-piped `JSONL → Arrow → trd → ffmpeg` flow, no
intermediate files:

```sh
# producer → renderer → encoder   (duckdb-free; uses pyarrow)
uv run --with pyarrow scripts/jsonl_to_arrow.py examples/frames.0.0.2.jsonl \
  | cargo run -q -p trd-cli -- --width 256 --height 256 \
  | uv run --with pyarrow --with numpy scripts/encode.py --fps 30 -o output/out.gif
```

The producer's `--version` flag selects the input JSONL protocol (default
`0.0.2`). [`examples/frames.0.0.2.jsonl`](examples/frames.0.0.2.jsonl) gives each
frame's `model` transform as a 4×4 matrix directly; the legacy
[`examples/frames.0.0.1.jsonl`](examples/frames.0.0.1.jsonl)
(`--version 0.0.1`) uses `center`/`size`/`theta` fields instead. Regenerate the
0.0.2 file from the 0.0.1 one with `scripts/gen_frames.py`.

- **Producer** — emits the input params stream. The wrappers use `duckdb` when its
  `arrow` community extension loads, else fall back to
  [`scripts/jsonl_to_arrow.py`](scripts/jsonl_to_arrow.py) (pyarrow), so a missing
  or broken duckdb extension never blocks a render.
- **`trd-cli`** — renders each row to `r,g,b,a` tensors (the output stream).
- **[`scripts/encode.py`](scripts/encode.py)** — pipes RGBA to ffmpeg, producing
  `.gif` or `.webp` by output extension. On WSL, prefix the `cargo` step with
  `WGPU_BACKEND=gl`.

<details><summary>DuckDB producer (equivalent first stage)</summary>

```sh
duckdb -c "INSTALL arrow FROM community; LOAD arrow;
  COPY (
    WITH raw AS (
      SELECT
        COALESCE(center, [0.0, 0.0]) AS c,
        COALESCE(size, [1.0, 1.0]) AS s,
        COALESCE(theta, 0.0) AS th,
        model AS m
      FROM read_json('examples/frames.0.0.2.jsonl',
        format = 'newline_delimited',
        columns = {center: 'DOUBLE[]', size: 'DOUBLE[]', theta: 'DOUBLE', model: 'DOUBLE[]'})
    )
    SELECT
      c::FLOAT[2] AS center,
      s::FLOAT[2] AS size,
      th::FLOAT AS theta,
      COALESCE(m, [
        s[1] * cos(th), s[1] * sin(th), 0.0, 0.0,
        -s[2] * sin(th), s[2] * cos(th), 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        c[1], c[2], 0.0, 1.0
      ])::FLOAT[16] AS model
    FROM raw
  ) TO '/dev/stdout' (FORMAT arrows);" \
  | cargo run -q -p trd-cli -- --width 256 --height 256 \
  | uv run --with pyarrow --with numpy scripts/encode.py --fps 30 -o output/out.gif
```

`FORMAT arrows` (plural) is the streaming IPC format. The explicit `columns=`
schema forces every column to exist (`NULL` when a row omits it), so the same
query serves both the 0.0.1 (`center`/`size`/`theta`) and 0.0.2 (`model`) example
data: the required 0.0.1 columns default to the identity and the additive 0.0.2
`model` matrix is used verbatim when present, else synthesized to match
[`scripts/jsonl_to_arrow.py`](scripts/jsonl_to_arrow.py).

</details>

### Native window

`trd-app` opens a window and plays the *same* params stream live (the desktop
counterpart of the browser target). Use the wrappers, or drive it directly:

```sh
examples/render.sh --native            # Linux/macOS
examples\render.ps1 -Native            # Windows (PowerShell 7)

# …or pipe any producer straight into trd-app:
uv run --with pyarrow scripts/jsonl_to_arrow.py examples/frames.0.0.2.jsonl \
  | cargo run -q -p trd-app -- --fps 30
```

Options: `--width`/`--height` (initial size), `--fps`, `--once` (hold the last
frame instead of looping). No stdin → the identity triangle. Honours
`WGPU_BACKEND` / `RUST_LOG`. Close the window to exit. In `--native` mode the
output file is ignored and neither `uv` nor `ffmpeg` is needed.

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

# Offscreen ArrowRenderer texture read back to a 2D canvas (browser twin of --cli output):
examples/render.sh --web --offscreen-renderer --placement-quad --axes-local --aabb \
  --mesh assets/meshes/bunny_with_texture/bunny.obj \
  --texture assets/meshes/bunny_with_texture/bunny_uv_map1.jpg \
  --frames-base output/cornellbox \
  examples/frames.cornellbox.stage2.jsonl '' 960 540 25

PORT=9000 examples/render.sh --web          # serve on a custom port
```

Every `--cli` content flag applies to `--web` unchanged — `--mesh`, `--texture`,
`--wireframe`, `--aabb`, `--axes`, `--axes-local`, `--placement-quad`,
`--frames-base`, and the positional `WIDTH`/`HEIGHT`. The render resolution is baked
into the stream's CV `k`, so it is a positional argument, **not** a URL param; the
only live URL param is **`?fps=N`** (1..240, default = the `FPS` positional). Two
render targets share the one bundle: **`--canvas-renderer`** (default) draws to the
on-screen WebGPU `CanvasRenderer`; **`--offscreen-renderer`** (alias
`--arrow-renderer`) draws to an offscreen `ArrowRenderer` texture read back to RGBA
and painted to a 2D canvas.

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
two renderers in `crates/trd-wasm/src/{canvas_renderer,arrow_renderer}.rs`. The
generic renderer fetches the prebuilt Arrow stream and replays it by index —
decoding it **once** with `loadIpc` (buffering every frame) rather than pushing
frame-by-frame:

```ts
import init, { CanvasRenderer } from "trd-wasm"; // fully typed

await init({ module_or_path: wasmUrl });
const canvas = await CanvasRenderer.create(canvasEl);
const total = canvas.loadIpc(streamBytes); // decode + buffer all frames
canvas.renderIndex(0);                     // draw buffered frame 0
```

`ArrowRenderer` is the offscreen counterpart: it renders each buffered frame to an
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
# duckdb and uv are optional (the wrappers fall back to a system python that
# already has pyarrow + numpy).
winget install --id Gyan.FFmpeg -e
winget install --id astral-sh.uv -e     # optional
winget install --id DuckDB.cli -e       # optional
```

**2. Prepare each shell.** Dot-source the setup script to put everything on `PATH`
— cargo pinned to the MSVC host, the MSVC linker imported from `vcvars64.bat`, plus
ffmpeg/duckdb/uv. It installs a missing `uv` via winget automatically (pass
`-NoInstall` to skip, `-Quiet` to hide the summary):

```powershell
. .\scripts\dev-env.ps1     # prints a dev-shell-style tool summary
```

**3. Build and render** with plain `cargo` / the example wrapper:

```powershell
cargo build -p trd-cli      # cargo can now link native binaries
examples\render.ps1 -CLI    # render examples\frames.0.0.2.jsonl → output\out.gif
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
