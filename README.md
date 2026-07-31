# trd

**A tile (relational) oriented renderer, built on Rust + wgpu.**

`trd-core` is the *single* rendering core. The exact same Rust/wgpu code renders
everywhere — a headless CLI, a native window, the browser, and an interactive
viewer — by drawing into whatever render target each front-end provides.
JavaScript/TypeScript is a thin bootstrap only; the WebGPU API is never called
from JS.

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

| Front-end | Reads | Renders into | Produces |
|---|---|---|---|
| **`trd-cli`** | Arrow stream (stdin) | offscreen texture → read-back | Arrow image stream (stdout) |
| **`trd-app`** | Arrow stream (stdin) | live window swapchain | frames on screen |
| **`trd-wasm`** | Arrow stream (via `loadIpc`) | live canvas (or offscreen texture) | frames in the browser |
| **`trd-gui`** | a mesh + live gestures | offscreen → egui image | an interactive orbit/zoom viewer |

Each front-end is a *thin shell* that only supplies a render target and calls the
core — no per-front-end rendering logic. See
**[`docs/architecture.md`](docs/architecture.md)** for the internals.

## Quick start

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
nix run   .#trd -- --width 256 --height 256   # stream on stdin → images on stdout
nix run   .#web                               # serve dist/  (PORT, default 8080)
nix flake check       # every gate: fmt, clippy (native+wasm32), test, tsc, biome
```

> `nix build` / `nix flake check` only see git-tracked files — `git add` new files
> before building. For fast iteration use plain `cargo` / `bun` inside `nix develop`
> (or, on Windows, after `. .\scripts\dev-env.ps1`).

## Stream protocol

Frame parameters are plain columnar data, so **any** tool that emits the input
columns as an Arrow IPC stream can drive the renderer. The current — and **only
supported** — version is **0.0.5**: **mesh-first** (`[mesh][texture?][params]`)
and **not** backward-compatible with `0.0.1`–`0.0.4` (older streams are
hard-rejected).

- a leading **mesh** table (`position`/`color`/`uv`/`index`) — one row per mesh;
- an optional **texture** table (`rgba` tensor) for textured/PBR albedo;
- the per-frame **params** table — an optional **camera** (**CV** `k`+`pose`, or
  **CG** `eye`/`target`/`up`+`fovy`…), an optional **draw list** (`draw_mesh` +
  `draw_model`) instancing several meshes, and an optional **background frame**
  (`frame_path`/`frame_url`) composited beneath the scene.

All params columns are optional/additive and drive `clip = P · V · M · (pos, 1)`.
Rendering appearance (filled / wireframe / textured / **PBR**) is a render-time
choice, **not** a wire column, so the same stream renders any way.

**PBR.** `--pbr` shades meshes with the **Disney principled BRDF** (`disney.wgsl`)
— the bound albedo lit by a virtual key/fill/rim light rig plus an optional HDR
environment probe, with filmic tone mapping. The material + lighting are tunable
from the CLI: `--metallic` / `--roughness` / `--specular` / `--clearcoat`,
`--env <file.hdr>` / `--env-intensity`, `--exposure` / `--ambient`, and
`--tonemap reinhard|aces`. Full parameter reference: **[`docs/pbr.md`](docs/pbr.md)**.

**Full column-by-column specification:
[`docs/protocol/0.0.5.md`](docs/protocol/0.0.5.md).**

## Documentation

- [`docs/architecture.md`](docs/architecture.md) — the render core + front-ends,
  and the source layout.
- [`docs/rendering.md`](docs/rendering.md) — running every front-end
  (wrappers ⇄ `cargo run`), all CLI flags, PBR / tone-map / MSAA, camera forms,
  AR demos, the native window, the interactive viewer, web, and Windows setup.
- [`docs/pbr.md`](docs/pbr.md) — the Disney principled-BRDF material model, all PBR
  parameters + defaults, tone mapping, and the HDR environment probe.
- [`docs/protocol/0.0.5.md`](docs/protocol/0.0.5.md) — the full stream-protocol spec.
- [`docs/frame-extraction.md`](docs/frame-extraction.md) — background-frame
  extraction + the frame-to-row mapping manifest.
- [`docs/gui-design.md`](docs/gui-design.md) — the `trd-gui` interactive-viewer design.
- [`AGENTS.md`](AGENTS.md) — contributor/agent guide: build system, GPU-adapter
  selection, testing policy, PR workflow.

## Tests

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
