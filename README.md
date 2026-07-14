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
- **`stream.rs`** — the Arrow IPC protocol: `read_frame_stream` decodes the input
  frames; `run_stream` is the CLI filter. Only one record batch is ever in flight,
  so an animation of any length streams in constant memory.

### The three consumers

Each is a *thin shell* that only supplies a render target and calls the core:

| Target | Reads | Renders into | Produces |
|---|---|---|---|
| **`trd-cli`** | Arrow params stream (stdin) | offscreen texture → pixel read-back | Arrow image stream (stdout) |
| **`trd-app`** | Arrow params stream (stdin) | live window swapchain | frames on screen |
| **`trd-wasm`** | a `<canvas>` element | live canvas surface | frames in the browser |

- **`trd-cli` — headless Arrow filter.** For each input frame it renders to an
  offscreen texture, copies the pixels back (`copy_texture_to_buffer`), and writes
  them as an Arrow image stream. It does **not** encode video itself — piping that
  stream to `scripts/encode.py` (ffmpeg) turns it into a GIF/WebP.
- **`trd-app` — native window.** A background thread reads the params stream from
  stdin; the window plays it at `--fps`, drawing each frame **straight into the
  swapchain surface** and presenting it. No read-back, no file — pixels go on
  screen. With no stdin it shows the identity triangle.
- **`trd-wasm` / `web/` — browser.** `start(canvas)` obtains a wgpu surface from
  the `<canvas>` and draws into it with the same core. Today it renders one static
  frame (the identity triangle) — the browser counterpart that will consume the
  same params stream next. Packaged as the `trd-wasm` npm library; `web/main.ts`
  only calls `init()` then `start(canvas)`.

### Stream protocol 0.0.1

Frame parameters are plain columnar data, so **any** tool that emits the input
columns as an Arrow IPC stream can drive the renderer.

| Direction | Columns | Arrow type |
|---|---|---|
| **Input** (params) | `center`, `size` | `FixedSizeList<f32>[2]` |
| | `theta` | `f32` |
| **Output** (image) | `r`, `g`, `b`, `a` | `fixed_shape_tensor<u8>` `[H, W]` |

The protocol-version metadata is optional, so DuckDB and pyarrow streams are both
accepted as-is.

## Repository layout

| Path | What it is |
|---|---|
| `crates/trd-core` | the unified render core (`render.rs`, `triangle.wgsl`, `stream.rs`) |
| `crates/trd-cli` | headless CLI: Arrow params in → Arrow image out |
| `crates/trd-app` | native interactive window (winit + live wgpu surface) |
| `crates/trd-wasm` | `wasm-bindgen` entry point; packaged as the `trd-wasm` npm library |
| `web/` | bun-managed thin TypeScript wrapper that loads `trd-wasm` |
| `examples/` | `frames.jsonl` demo + `render.sh` / `render.ps1` wrappers |
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

**2. Run the demo** — renders [`examples/frames.jsonl`](examples/frames.jsonl):

```sh
# Linux / macOS / WSL
examples/render.sh            # render → out.gif
examples/render.sh --native   # play live in a window
```

```powershell
# Windows (PowerShell 7)
examples\render.ps1           # render → out.gif
examples\render.ps1 -Native   # play live in a window
```

> On WSL, prefix GPU commands with `WGPU_BACKEND=gl` (otherwise rendering is
> software).

**3. Try the web build:**

```sh
nix run .#web    # build + serve at http://localhost:8080 (open in a WebGPU browser)
```

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
examples/render.sh  [INPUT.jsonl] [OUT.gif|.webp] [WIDTH] [HEIGHT] [FPS]   # Linux/macOS
examples\render.ps1 [-InputPath]  [-Output]       [-Width] [-Height] [-Fps]  # Windows (PS7)
# Defaults: examples/frames.jsonl → out.gif, 256×256 @ 30 fps
```

On Windows the Arrow stages are handed off through a temp dir (Windows DuckDB
can't write to `/dev/stdout` and PowerShell pipelines aren't binary-safe); the
output is identical.

#### The render pipeline

Under the hood it is a fully-piped `JSONL → Arrow → trd → ffmpeg` flow, no
intermediate files:

```sh
# producer → renderer → encoder   (duckdb-free; uses pyarrow)
uv run --with pyarrow scripts/jsonl_to_arrow.py examples/frames.jsonl \
  | cargo run -q -p trd-cli -- --width 256 --height 256 \
  | uv run --with pyarrow --with numpy scripts/encode.py --fps 30 -o out.gif
```

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
    SELECT center::FLOAT[2] AS center, size::FLOAT[2] AS size, theta::FLOAT AS theta
    FROM read_json_auto('examples/frames.jsonl')
  ) TO '/dev/stdout' (FORMAT arrows);" \
  | cargo run -q -p trd-cli -- --width 256 --height 256 \
  | uv run --with pyarrow --with numpy scripts/encode.py --fps 30 -o out.gif
```

`FORMAT arrows` (plural) is the streaming IPC format.

</details>

### Native window

`trd-app` opens a window and plays the *same* params stream live (the desktop
counterpart of the browser target). Use the wrappers, or drive it directly:

```sh
examples/render.sh --native            # Linux/macOS
examples\render.ps1 -Native            # Windows (PowerShell 7)

# …or pipe any producer straight into trd-app:
uv run --with pyarrow scripts/jsonl_to_arrow.py examples/frames.jsonl \
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

The wasm core is a standard, TypeScript-typed npm package (`nix build .#trd-wasm`,
built with `wasm-bindgen-cli` + `wasm-opt`). `web/` imports it like any library:

```ts
import init, { start } from "trd-wasm"; // fully typed
```

For local iteration inside `nix develop`, `web/` uses `wasm-pack` + bun:

```sh
cd web
bun run build      # wasm-pack → pkg, then bun bundles → web/dist
bun run dev        # dev server; open the printed URL in a WebGPU browser
bun run check      # Biome format-check + lint
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
examples\render.ps1         # render examples\frames.jsonl → out.gif
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

### Tests

```sh
cargo fmt --all --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace                 # fast; GPU tests are skipped
cargo test --workspace -- --ignored    # GPU-gated render tests
```
