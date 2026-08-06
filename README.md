# trd

**A tile (relational) oriented renderer, built on Rust + wgpu.**

`trd-core` is the *single* rendering core. The exact same Rust/wgpu code renders
everywhere — a headless CLI, a native window, the browser, and an interactive
viewer — by drawing into whatever render target each front-end provides.
JavaScript/TypeScript is a thin bootstrap only; the WebGPU API is never called
from JS.

## Contents

- [How it fits together](#how-it-fits-together)
- [Quick start](#quick-start)
- [Building with Nix](#building-with-nix)
- [Building directly (no Nix)](#building-directly-linux--windows-no-nix)
- [Stream protocol](#stream-protocol)
- [Material (PBR)](#material-pbr)
- [Documentation](#documentation)
- [Tests](#tests)
- [Verification icons](#verification-icons)

## [How it fits together](docs/architecture.md)

Everything shares **one render function** and **one data format** (a mesh-first
Arrow stream):

```
input-stream ─┬─ trd-cli  → trd-core → offscreen readback → image-stream   (headless)
(mesh-first)  ├─ trd-app  → trd-core → window surface                      (native playback)
              ├─ trd-wasm → trd-core → canvas surface                      (browser)
              └─ trd-gui  → trd-core → offscreen → egui image      (interactive, native + browser)

                  image-stream → scripts/encode.py → ffmpeg → GIF / WebP / MP4
```

| Front-end | Reads | Renders into | Produces |
|---|---|---|---|
| **`trd-cli`** | Arrow stream (stdin) | offscreen texture → read-back | Arrow image stream (stdout) |
| **`trd-app`** | Arrow stream (stdin) | live window swapchain | frames on screen |
| **`trd-wasm`** | Arrow stream (via `loadIpc`) | live canvas (or offscreen texture) | frames in the browser |
| **`trd-gui`** | a mesh + live gestures | offscreen → egui image | an interactive orbit/zoom viewer |

Each front-end is a *thin shell* that only supplies a render target and calls the
core — no per-front-end rendering logic. Primitive dispatch and draw-kind
batching stay in the shared Rust core. See
**[`docs/architecture.md`](docs/architecture.md)** for the internals.

## [Quick start](docs/rendering.md)

**1. Get a dev environment.**

- **Linux / macOS / WSL** — [Nix](https://nixos.org/download) is the build system
  *and* dev shell (pinned Rust, `bun`, wasm tools, `biome`, ffmpeg, Vulkan):

  ```sh
  nix develop
  ```

- **Windows** — no Nix; dot-source the setup script (details in
  [`docs/rendering.md`](docs/rendering.md#windows-setup-without-nix)):

  ```powershell
  . .\scripts\dev-env.ps1
  ```

**2. Run the demo** — the bunny dolly-camera capstone (loads `assets/meshes/bunny.obj`):

```sh
examples/render.sh --cli      # Linux/macOS/WSL — render → output/out.gif
examples/render.sh --native   # play live in a window
```

```powershell
examples\render.ps1 -CLI      # Windows (PowerShell 7)
examples\render.ps1 -Native
```

The wrappers are conveniences around `cargo run`; the same demo, by hand:

```sh
uv run --with pyarrow scripts/obj_to_arrow.py assets/meshes/bunny.obj  > /tmp/stream.arrow
uv run --with pyarrow scripts/jsonl_to_arrow.py examples/frames.bunny_dolly.cg.jsonl >> /tmp/stream.arrow
cat /tmp/stream.arrow \
  | cargo run -q -p trd-cli -- --width 256 --height 256 \
  | uv run --with pyarrow --with numpy scripts/encode.py --fps 30 -o output/out.gif
```

Run either wrapper with no arguments to print its flag guidance. The full
wrapper ⇄ `cargo run` mapping, all flags, PBR/MSAA modes, and the AR demos are in
**[`docs/rendering.md`](docs/rendering.md)**.

**3. Try the web build:**

```sh
examples/render.sh --web   # generate the demo stream, build + serve  (Linux/macOS/WSL)
```

Open the printed URL in a WebGPU browser (Chrome/Edge). On a remote (SSH) host, run
the printed tunnel command first, then browse to <http://localhost:8080>.

> **GPU on WSL / non-NixOS Linux.** On WSL prefix GPU commands with
> `WGPU_BACKEND=gl`; on a native (non-NixOS) Linux GPU box, wrap them with
> [nixGL](https://github.com/nix-community/nixGL). Details:
> [`docs/rendering.md`](docs/rendering.md#gpu-notes) · [`AGENTS.md`](AGENTS.md#gpu).

## Building with Nix

The Nix flake is the build system — reproducible outputs, no manual toolchain:

```sh
nix build .#trd-cli   # native CLI (Arrow stream filter) + Vulkan/GL runtime libs
nix build .#trd-wasm  # wasm-bindgen JS/TS library package
nix build .#web       # bun-bundled, HTTP-servable dist/  (also plain `nix build`)
nix run   .#trd-cli -- --width 256 --height 256   # stream on stdin → images on stdout
nix run   .#web                               # serve dist/  (PORT, default 8080)
nix flake check       # every gate: fmt, clippy (native+wasm32), test, tsc, biome
```

> `nix build` / `nix flake check` only see git-tracked files — `git add` new files
> before building. For fast iteration use plain `cargo` / `bun` inside `nix develop`
> (or, on Windows, after `. .\scripts\dev-env.ps1`).

## Building directly (Linux & Windows, no Nix)

Every front-end also builds with a plain **Rust + bun** toolchain — no Nix
required. You need a Rust toolchain (`rustup`; the **MSVC** host on Windows) and,
for the browser bundle, `bun` + `wasm-pack`; the `render.*` demos additionally need
`uv` + `ffmpeg`. The CLI/GUI dlopen the GPU driver at run time (system Vulkan/GL on
Linux, the vendor driver on Windows).

**🐧 Linux** (a host toolchain, or inside `nix develop`):

```sh
cargo build --workspace                                # shared crates + native delivery apps
cargo run -p trd-cli -- --width 256 --height 256       # headless Arrow filter (stdin → stdout)
cargo run -p trd-gui-app -- --mesh assets/meshes/bunny.obj # interactive viewer window
examples/render.sh --cli                               # end-to-end demo → output/out.gif
( cd web/viewer && bun run dev )                        # build wasm + serve on :8080
( cd web/gui-viewer && bun run dev )                    # interactive GUI viewer on :8082
```

**🪟 Windows** (PowerShell 7; `. .\scripts\dev-env.ps1` puts cargo / MSVC / ffmpeg / uv on PATH):

```powershell
. .\scripts\dev-env.ps1                                # once per shell (must be dot-sourced)
cargo build --workspace
cargo run -p trd-cli -- --width 256 --height 256       # headless Arrow filter (stdin → stdout)
cargo run -p trd-gui-app -- --mesh assets\meshes\bunny.obj # interactive viewer window
examples\render.ps1 -CLI                               # end-to-end demo → output\out.gif
cd web/viewer; bun run dev                             # build wasm + serve on :8080
```

Full setup — Windows `dev-env.ps1`, GPU-driver notes (nixGL / `WGPU_BACKEND=gl`),
and the `wrappers ⇄ cargo run` mapping — is in
[`docs/rendering.md`](docs/rendering.md).

## [Stream protocol](docs/protocol/0.0.6.md)

Frame parameters are plain columnar data, so **any** tool that emits the input
columns as an Arrow IPC stream can drive the renderer. The current — and **only
supported** — version is **0.0.6**:
`[mesh][texture?][frames?][params]`, with every table explicitly tagged by
`trd.table.kind`. It is not backward-compatible; every other or missing version
is hard-rejected.

- a leading **mesh** table (`position`/`color`/`uv`/`index`) — one row per mesh;
- an optional **texture** table (`rgba` tensor) for textured/PBR albedo;
- an optional indexed **frames** resource table (`frame_bytes` Binary or
  `frame_pixels` tensor) for self-contained backgrounds;
- the per-frame **params** table — an optional **camera** (**CV** `k`+`pose`, or
  **CG** `eye`/`target`/`up`+`fovy`…), an optional **draw list** (`draw_mesh` +
  `draw_model`) instancing several meshes, and an optional **background frame**
  (`frame_id` for inline data, or external `frame_path`/`frame_url`) composited
  beneath the scene.

The standard inline e2e packs all 250 native 1920×1080 frames of the Cornell-box
clip as a raw RGBA tensor table and renders only the correctly placed textured bunny; see
[`docs/frame-extraction.md`](docs/frame-extraction.md).

All params columns are optional/additive and drive `clip = P · V · M · (pos, 1)`.
Rendering appearance (filled / wireframe / textured / **PBR**) is a render-time
choice, **not** a wire column, so the same stream renders any way.

**Full column-by-column specification:
[`docs/protocol/0.0.6.md`](docs/protocol/0.0.6.md).**

## [Material (PBR)](docs/pbr.md)

`--pbr` shades meshes with the **Disney principled BRDF** (`disney.wgsl`) — the
bound albedo lit by a virtual key/fill/rim light rig plus an optional HDR
environment probe, with filmic tone mapping. The material + lighting are tunable
from the CLI: `--metallic` / `--roughness` / `--specular` / `--clearcoat`,
`--env <file.hdr>` / `--env-intensity`, `--exposure` / `--ambient`, and
`--tonemap reinhard|aces`. Full parameter reference + material model:
**[`docs/pbr.md`](docs/pbr.md)**.

## [Interactive viewer](docs/rendering.md#interactive-viewer--trd-gui)

`trd-gui` is a live **object editor** — a native window or a WebGPU browser tab.
**Click** an object to select it (its bounding box highlights); the left panel
then edits *that* object. Everything is **per-object**:

- **Transform** — left-drag rotates / right-drag moves / scroll scales the selected
  object (or orbit the camera), with an optional **axis lock** (X/Y/Z) and numeric
  widgets kept in sync with the mouse;
- **Render mode** — Filled / Wireframe / Textured / **PBR**, chosen per object;
- **PBR appearance** — typed Disney material, IBL intensity, and tone-map,
  edited live per object;
- **Diffuse texture** — each object skins its own albedo;
- **Overlays** — smooth, thickness-controlled bounding boxes/grids and
  arrowheaded world/local axes.

The browser loads OBJ or single-primitive GLB objects from **repeatable
`?mesh=`** (+ positional `?texture=`, and `?env=` for PBR). GLB uses its
embedded base-color / metallic-roughness / normal maps and starts in PBR:
`?mesh=a.obj&texture=a.jpg&mesh=b.obj&env=probe.hdr`. Full control reference + URL
params: **[`docs/rendering.md`](docs/rendering.md#interactive-viewer--trd-gui)**.

## Documentation

- [`docs/architecture.md`](docs/architecture.md) — the render core + front-ends,
  and the source layout.
- [`docs/rendering.md`](docs/rendering.md) — running every front-end
  (wrappers ⇄ `cargo run`), all CLI flags, PBR / tone-map / MSAA, camera forms,
  AR demos, the native window, the interactive viewer, web, and Windows setup.
- [`docs/pbr.md`](docs/pbr.md) — the Disney principled-BRDF material model, all PBR
  parameters + defaults, tone mapping, and the HDR environment probe.
- [`docs/protocol/0.0.6.md`](docs/protocol/0.0.6.md) — the full stream-protocol spec.
- [`docs/frame-extraction.md`](docs/frame-extraction.md) — background-frame
  extraction, external references, and inline frames-table authoring.
- [`docs/gui-design.md`](docs/gui-design.md) — the `trd-gui` interactive-viewer design.
- [`AGENTS.md`](AGENTS.md) — contributor/agent guide: build system, GPU-adapter
  selection, testing policy, PR workflow.

## [Tests](AGENTS.md#testing--required-before-a-task-is-done)

```sh
cargo fmt --all --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test  --workspace                 # fast; GPU tests are skipped
cargo test  --workspace -- --ignored    # GPU-gated render tests (need a real GPU)
```

The **golden / snapshot render test** (`crates/trd-core/tests/golden_render.rs`) is
the pixel-level regression net: it runs committed Arrow fixtures through the real
`run_stream` pipeline and pixel-diffs each frame against committed golden PNGs, at
**both** 4× MSAA and MSAA-off, plus PBR tone-map variants (GPU-gated). The full
testing policy (required tiers, per-crate e2e, multi-platform verification +
handoff, golden regeneration) lives in
**[`AGENTS.md`](AGENTS.md#testing--required-before-a-task-is-done)**.

## Verification icons

PRs and issues report their dual-platform checks as a compact, icon-led
**verification matrix** (one row per gate, one column per platform) plus a handoff
checklist — see
[`AGENTS.md`](AGENTS.md#multiple-platform-verification-and-handoff). The fixed
glyph set:

| Kind | Icons |
|------|-------|
| **Status** | ✅ passed · ❌ failed · ⏳ not yet run · 🤝 handed off |
| **Platform** | 🪟 Windows · 🐧 Linux/Nix |
| **Gate** | 🎨 fmt · 📎 clippy · 🕸️ clippy-wasm · 🧪 tests · 🔀 decoder-parity · 🖼️ golden-render · 🌐 tsc/biome |

A reproducible cross-front-end smoke (the coca-cola can PBR scene) is in
[`AGENTS.md`](AGENTS.md#cross-mode-e2e-recipe--coca-cola-can-pbr--aabb--axes).
