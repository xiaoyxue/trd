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
- **`stream.rs`** — the native Arrow IPC filter: `read_frame_stream` decodes the
  input frames; `run_stream` is the CLI filter. Only one record batch is ever in
  flight, so an animation of any length streams in constant memory.
- **`protocol.rs`** — the cross-platform (native + wasm) incremental Arrow IPC
  decoder. `InputSession` feeds arbitrary byte chunks through `arrow`'s
  `StreamDecoder`, validates the protocol schema once (accepts
  `0.0.1`/`0.0.2`/`0.0.3`), and yields one `FrameBatch` (`Vec<FrameParams>`) per
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
| **`trd-wasm`** | Arrow params stream (via `pushIpc`) | live canvas surface | frames in the browser |

- **`trd-cli` — headless Arrow filter.** For each input frame it renders to an
  offscreen texture, copies the pixels back (`copy_texture_to_buffer`), and writes
  them as an Arrow image stream. It does **not** encode video itself — piping that
  stream to `scripts/encode.py` (ffmpeg) turns it into a GIF/WebP.
- **`trd-app` — native window.** A background thread reads the params stream from
  stdin; the window plays it at `--fps`, drawing each frame **straight into the
  swapchain surface** and presenting it. No read-back, no file — pixels go on
  screen. With no stdin it shows the identity triangle.
- **`trd-wasm` / `web/` — browser.** `CanvasRenderer.create(canvas)` obtains a wgpu
  surface from the `<canvas>` and holds a persistent pipeline plus an `InputSession`.
  `web/main.ts` produces a persistent Apache Arrow IPC stream (one one-row batch per
  `requestAnimationFrame`) and pumps its bytes into `canvas.pushIpc(chunk)`; Rust
  decodes and draws each frame straight to the canvas — no pixel read-back. JS only
  moves Arrow bytes and schedules frames; it never touches the WebGPU API. Packaged
  as the `trd-wasm` npm library.

### Stream protocol

Frame parameters are plain columnar data, so **any** tool that emits the input
columns as an Arrow IPC stream can drive the renderer. The current version is
**0.0.3**; it is backward-compatible with 0.0.2 and 0.0.1.

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
| **Input** (mesh, *opt, 0.0.3*) | `position`, `color` | `List<FixedSizeList<f32>[3]>` |
| | `index` | `List<u32>` |
| **Output** (image) | `r`, `g`, `b`, `a` | `fixed_shape_tensor<u8>` `[H, W]` |

The `0.0.2` matrix columns are **optional/additive** and drive the MVP transform
`clip = P · V · M · (pos, 0, 1)`; a stream with none of them (or identity
matrices) renders identically to `0.0.1`. The protocol-version metadata is
optional, so DuckDB and pyarrow streams are both accepted as-is.

**0.0.3** adds three things on top of the params stream:

- an optional leading **mesh** Arrow stream, concatenated before the params
  stream (`[mesh][params]`): one row per mesh, geometry in nested list columns.
  The native path decodes it (`Mesh::from_arrow`), uploads it, and renders it
  **centered and uniformly scaled to fit** (a `base_model` beneath the per-frame
  `model`), driven by the following params. Encode an OBJ with
  `scripts/obj_to_arrow.py`; `examples/render.sh --mesh <obj>` wires it end-to-end.
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

A params-only stream (no leading mesh) still renders the built-in hello-triangle.

**Full, versioned specification: [`docs/protocol/`](docs/protocol/)** (per-version
schema reference + [changelog](docs/protocol/CHANGELOG.md)).


## Repository layout

| Path | What it is |
|---|---|
| `crates/trd-core` | the unified render core (`render.rs`, `triangle.wgsl`, `stream.rs`) |
| `crates/trd-cli` | headless CLI: Arrow params in → Arrow image out |
| `crates/trd-app` | native interactive window (winit + live wgpu surface) |
| `crates/trd-wasm` | `wasm-bindgen` entry point; packaged as the `trd-wasm` npm library |
| `web/` | bun-managed thin TypeScript wrapper that loads `trd-wasm` |
| `examples/` | `frames.0.0.2.jsonl` (+ legacy `frames.0.0.1.jsonl`) demo + `render.sh` / `render.ps1` wrappers |
| `scripts/jsonl_to_arrow.py` | JSONL → Arrow params stream (pyarrow; duckdb-free producer) |
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
examples/render.sh --web   # build + serve, printing the URL + SSH-tunnel command
nix run .#web              # equivalent: build + serve at http://localhost:8080
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
#   --web/-Wasm (browser; --arrow-renderer/-ArrowRenderer default, or --canvas-renderer/-CanvasRenderer)
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
| `--wireframe` | Draw mesh **edges** as a line list (`trd --wireframe`) instead of filled triangles (protocol #38). Reveals topology; on a dense asset (e.g. the ~70k-tri bunny) the edges read as a fine mesh. |
| `--aabb` | Overlay each drawn mesh instance's **axis-aligned bounding box** as a green (`[0, 1, 0]`) wireframe box (`trd --aabb`, #42). The box uses the *same* per-instance model as the mesh, so it tracks the mesh through the preview + per-frame transforms. Combine freely with `--wireframe`. |
| `--axes` | Overlay a **coordinate-axes gizmo** (X=red, Y=green, Z=blue lines) at the world origin (`trd --axes`, #42), under the camera `P·V` with an identity model, marking the world frame the camera looks at. |

```sh
# Single bunny turntable, filled, with its bounding box:
examples/render.sh --cli --aabb --mesh assets/meshes/bunny.obj \
  examples/frames.turntable.jsonl output/bunny.gif 1024 1024 24

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

The [`examples/render.sh`](examples/render.sh) wrapper wraps that build-and-serve
step behind a `--web` (alias `--wasm`) flag, and — handy for a remote GPU box —
prints the machine URL plus a ready-to-copy SSH-tunnel command before serving
(`PORT` overrides the port, default 8080; all positional arguments are ignored,
since the demo generates its own frames in-browser):

```sh
examples/render.sh --web                    # ArrowRenderer (default): offscreen output-stream smoke
examples/render.sh --web --canvas-renderer  # CanvasRenderer: on-screen canvas demo
PORT=9000 examples/render.sh --wasm         # serve on a custom port
```

The server binds all interfaces, so browse to `http://<host-ip>:PORT` directly,
or forward it — `ssh -L 8080:localhost:8080 <user>@<host>`, then open
<http://localhost:8080>.

Both in-browser renderers ship in one bundle ([`web/src/main.ts`](web/src/main.ts)
routes on the `?arrow-smoke` query param), so the flag only changes which URL the
wrapper points you at: `--arrow-renderer` (default) opens the offscreen
`ArrowRenderer` output-stream roundtrip — the browser counterpart of the headless
`--cli` render — while `--canvas-renderer` opens the on-screen `CanvasRenderer` demo.

On Windows (no Nix), [`examples/render.ps1`](examples/render.ps1) exposes the same
flag as `-Web` (alias `-Wasm`): it builds the bundle with `wasm-pack` + `bun`
(`web`'s `bun run build`) and serves `web/dist` with a small Bun static server,
printing the same URLs + SSH-tunnel command (`$env:PORT` overrides the port,
default 8088; positional arguments are ignored):

```powershell
examples\render.ps1 -Web                    # ArrowRenderer (default): offscreen output-stream smoke
examples\render.ps1 -Web -CanvasRenderer    # CanvasRenderer: on-screen canvas demo
$env:PORT = 9000; examples\render.ps1 -Wasm # serve on a custom port
```

The wasm core is a standard, TypeScript-typed npm package (`nix build .#trd-wasm`,
built with `wasm-bindgen-cli` + `wasm-opt`). `web/` imports it and drives it with
Apache Arrow JS — the browser produces the same protocol-0.0.2 IPC stream the CLI
consumes and pumps it into the renderer:

```ts
import init, { CanvasRenderer } from "trd-wasm"; // fully typed

await init({ module_or_path: wasmUrl });
const canvas = await CanvasRenderer.create(canvasEl);
const rendered = canvas.pushIpc(ipcChunk); // rows drawn this chunk
canvas.finish();                           // end of stream
```

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

The demo animates one Arrow one-row batch per frame. Two query flags help testing:
`?smoke=1` renders a single two-row batch then stops (sets
`#trd-status[data-rows-rendered="2"]`); `?benchmarkRate=60` / `?benchmarkRate=120`
drive a fixed-rate run and log p50/p95/p99 timings (Arrow generation, `pushIpc`
total, render-submit, and derived transfer-plus-decode) to the console.

A second browser type, **`ArrowRenderer`**, is the offscreen counterpart of the CLI:
it renders to an offscreen texture and returns the same protocol-0.0.2 Arrow **output**
stream (four `fixed_shape_tensor<u8>` channels `r,g,b,a`) instead of drawing to a canvas.

```ts
import init, { ArrowRenderer } from "trd-wasm";

await init({ module_or_path: wasmUrl });
const arrow = await ArrowRenderer.create(width, height);
const outChunk = await arrow.pushIpc(inputIpcChunk); // new output IPC bytes
const eos = arrow.finish();                           // output EOS
```

Input is one persistent Arrow IPC stream (arbitrary chunk boundaries); `pushIpc`
returns only newly produced output bytes, one output record batch per input batch,
with the schema on the first productive result; `finish()` emits EOS; calls after
`finish()` reject. The `?arrow-smoke` flag runs an in-page roundtrip that feeds a
two-batch input through `ArrowRenderer` and validates the decoded output
(sets `document.body[data-arrow-smoke="pass"]`).

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
