# Testing & verification

How to *run* trd's gates. [`AGENTS.md`](../AGENTS.md) defines **which** gates a
change owes — the L1/L2/L3 floor table — and how to report them; this page is the
operating manual behind that decision.

The §-numbers below are the ones the floor table and the verification matrix cite
(`§4.2/4.3`, `§4.5/4.6`, `§4.7`), so they are stable references, not headings you
may renumber freely.

## Contents

- [GPU](#gpu) — [adapter choice](#always-render-on-the-most-powerful-gpu-available) ·
  [nixGL](#linux-non-nixos-eg-ubuntu-use-nixgl) · [WSL2](#wsl2)
- [The golden render test](#the-golden-render-test-88) —
  [what it covers](#what-it-covers) · [regenerating](#regenerating) ·
  [the companion non-GPU gate](#the-companion-non-gpu-gate--testsdecoder_parityrs)
- [Where a test lives](#where-a-test-lives--by-kind-not-by-size-305)
- [The tiers in full](#the-tiers-in-full) — [§3 end-to-end](#3-end-to-end--linux-and-windows-l3) ·
  [§4 Windows e2e](#4-windows-e2e-manual--l3) ·
  [§4.7 large-file seek](#47--large-file-seek-windows-required-for-any-media-layer-change)
- [Cross-mode e2e recipe — the coca-cola can](#cross-mode-e2e-recipe--coca-cola-can-pbr--aabb--axes)
- [The matrix template](#the-matrix-template)

## GPU

We always work on GPU machines. GPU-dependent tests are marked `#[ignore]` and
run locally; CI skips them.

### Always render on the most powerful GPU available

With multiple adapters, pick the strongest discrete card; never fall back to a
weak display/iGPU (e.g. a Quadro P620) or software (llvmpipe). Preference,
strongest first:

```
RTX PRO 6000 > RTX 5090 > RTX 6000 Ada > RTX 4090 > RTX A6000 > RTX 3090 > others
```

- **List adapters:** `nvidia-smi --query-gpu=index,name,memory.total --format=csv`
- **Confirm trd's choice** from its `trd_core=info` log line
  `using Vulkan adapter "…" (DiscreteGpu)`.
- Selection lives in `render/gpu_context.rs`, whose `GpuRequest` defaults to
  `PowerPreference::HighPerformance`, so Vulkan prefers the discrete card
  (verified: picks the RTX 3090 over a P620).
- **To force one:** Mesa `MESA_VK_DEVICE_SELECT=<vendorId>:<deviceId>`;
  multi-GPU NVIDIA `__NV_PRIME_RENDER_OFFLOAD=1
  __GLX_VENDOR_LIBRARY_NAME=nvidia` (GL). Plain `CUDA_VISIBLE_DEVICES` does
  **not** filter Vulkan physical devices.

### Linux non-NixOS (e.g. Ubuntu): use nixGL

The `nix develop` Vulkan loader can't `dlopen` the host NVIDIA/Mesa driver (fails
with *"No suitable graphics adapter found"*). Wrap GPU commands to inject a
matching host driver:

```sh
# inside `nix develop`; --impure lets nixGL detect the host driver version
NIXPKGS_ALLOW_UNFREE=1 nix run --impure github:nix-community/nixGL#nixGLNvidia -- \
  cargo test -p trd-core -- --ignored          # or: ./result/bin/trd-cli …, render.sh, etc.
```

`NIXPKGS_ALLOW_UNFREE=1` is required for NVIDIA; use `#nixGLIntel` for
Intel/Mesa. NixOS doesn't need this (driver on `/run/opengl-driver`).

### WSL2

NVIDIA ships no native Linux Vulkan ICD, so Vulkan falls back to software
(llvmpipe) and Mesa's `dzn` (Vulkan-on-D3D12) crashes at device creation. Use
`WGPU_BACKEND=gl` for real GPU rendering via Mesa's D3D12 GL driver; the dev
shell auto-configures this on WSL.

## The golden render test (#88)

**The primary pixel-level regression net.** `crates/trd-core/tests/golden_render.rs`
feeds committed Arrow fixtures through the real `run_stream` pipeline and
pixel-diffs the frames against committed golden PNGs.

It is GPU-gated (`#[ignore]`); run it via the nixGL wrapper (Linux) or directly
on a Windows box with a discrete GPU.

### What it covers

Fixtures are `crates/trd-core/tests/golden/stage{1,2}.arrow` (the reduced
two-stage cornellbox placement demo), with goldens in the same dir. Each params
row selects an inline `0.0.6` frames-table resource by `frame_id` (stage 1
encoded Binary, stage 2 raw tensor), composited **under** the scene.

| Variant | Why it exists |
|---|---|
| `stageN_*` — 4× MSAA (`Msaa::X4`, the default anti-aliased mesh pass) | the multisample + resolve path |
| `stageN_noaa_*` — MSAA off (`Msaa::Off`, single-sample) | the raw single-sample path |
| `golden_stage2_pbr_{aces,reinhard}` | PBR tone-map variants |
| `golden_environment_light_syncs_sky_and_reflection` | a hand-built scene (no fixture can draw a sky) pinning that the scene's one `EnvironmentLight.rotation` drives the visible sky **and** the reflections on a near-mirror ball in front of it (#182) |

### Regenerating

Only after an *intended* visual change or a fixture change:

```sh
# 1. rebuild the .arrow fixtures + stills (needs uv + ffmpeg on PATH)
python3 scripts/golden_fixtures.py
# 2. refresh the golden PNGs from the current renderer (GPU box)
TRD_UPDATE_GOLDENS=1 cargo test -p trd-core --test golden_render -- --ignored
```

### The companion non-GPU gate — `tests/decoder_parity.rs`

It decodes the same fixtures through both **public API surfaces** — the native
`InputStream` (`io/input_stream.rs`, a byte transport owning a `Read`) and the
browser's push `InputSession` — and asserts identical *assembled frames*. It runs
in `nix flake check`.

Neither the column decode nor the framing is duplicated: both run the one decoder
in `protocol/arrow_decode.rs` through the one `InputSession`. What this guards is
that the two surfaces a caller assembles a frame through agree —
`prologue`/`next_batch`/`finish` versus a bare `push`, and `InlineFrameCache`
versus `InlineFrame::decode` — the shape of failure the `center` non-nullable bug
had.

## Where a test lives — by kind, not by size (#305)

The rule is one line in [`AGENTS.md`](../AGENTS.md): **unit tests inline
(`#[cfg(test)] mod tests`), however long; integration tests under
`crates/<crate>/tests/`.** The reasoning behind it:

- A **unit test** reaches into the module's own internals (`use super::*`,
  private fields and functions), so it belongs beside the code it pins. **Length
  is irrelevant** — a 1,200-line unit-test module still lives in its module.
- An **integration test** (`golden_render.rs`, `decoder_parity.rs`,
  `gui_render.rs`, `wasm_bindgen_containment.rs`) compiles as a separate crate
  and may only touch the public API, which is what makes it worth isolating.
- There is deliberately **no third form**: a `src/**/tests.rs` is compiled into
  the crate exactly like an inline `mod tests`, so it is a unit test wearing an
  integration test's clothes. A size threshold was tried and dropped (#299 §1,
  #305) — it made the location of a test say nothing about what the test *is*,
  and forced file moves whenever a module crossed a line count.
- Test-only **support** modules are a different thing and stay as they are:
  `render/gpu_tests.rs`, `render/triangle_renderer.rs` and
  `protocol/scene_encode.rs` are not `mod tests` blocks but helper modules that
  happen to be test-gated.

When a module gets too long to read, the fix is to split the *module* — by
responsibility, tests following their code — not to hide its tests in another
file.

## The tiers in full

The contents of the test levels: tiers 1–2 are **L2**, tiers 3–4 are **L3**.

1. **Golden test — MSAA enabled *and* disabled (must — L2).**
   `cargo test -p trd-core --test golden_render -- --ignored` runs both the 4×
   MSAA (`stageN_*`) and single-sample (`stageN_noaa_*`) goldens plus the PBR
   tone-map variants. Mandatory for any render-path change, on every platform
   with a GPU — see [the golden render test](#the-golden-render-test-88).
2. **GPU-gated tests (must — L2).** Every `#[ignore]` test, on a real GPU:
   `cargo test -p trd-core -- --ignored` (golden + `render::gpu_tests`) and
   `cargo test -p trd-gui --test gui_render -- --ignored`

### 3. End-to-end — Linux *and* Windows (L3)

- **trd-core / trd-cli:** stream a real Arrow input through the CLI and read an
  image stream back — `nix run .#trd-cli -- …` / `examples/render.sh` (Linux),
  `examples/render.ps1` (Windows).
- **trd-wasm (web):** build + serve the browser bundle (`nix build .#web` /
  `bun run build:web`), load a stream, and confirm **both** the on-screen
  `CanvasRenderer` and the offscreen `OffscreenRenderer` render with colors
  matching the CLI.
- **trd-gui (wasm + web):** build the gui wasm, serve, and load a mesh
  (`?mesh=…&texture=…`) in the browser.
- **video editor:** serve `web/gui-video-editing`, open the FIBA MP4, and exercise
  quad selection, all three catalog assets, object picking/editing,
  play/pause/seek, and the video-only 222–287 tail. Confirm PBR/IBL and colors
  match the other front-ends. Open **Details** and confirm its displayed frame
  identity does not jump ahead during rapid seek/render, and that the Dragon
  reports its GLB material maps, raw tracking pose delta, zero direct
  light/ambient, and Uffizi IBL. The MP4 stays external/uncommitted.
- **large video over a URL (media-layer gate):** any change to
  `web/gui-video-editing/src/media/` must also be run against a
  *multi-hundred-GiB* MP4 **served over HTTP**, because file size is exactly what
  a ranged reader is for and a local short clip cannot fail the way a 218 GiB one
  does. Serve it with the CORS+range helper
  (`bun web/gui-video-editing/serve-documents.ts <dir> --port 8092 --log`;
  `--log` prints the delivered bytes per request, which is the claim below),
  then drive the probe page — `?url=…&seek=…&frames=…` for one deep seek and
  `?reader=mediabunny&scrub=t1,t2,…` (plus `&overlap=1`, the dragged-scrubber
  shape) for repeated seeks on one reader — and open the editor itself at
  `?document=none&reader=mediabunny&video=<url>`.
  **The probe page is not served by `bun run dev`**, which passes only
  `index.html`; start it explicitly with `bun ./index.html ./probe.html`, and
  note that Bun serves that second entrypoint at the extensionless route
  **`/probe`**, not `/probe.html`. Expect: **opening costs megabytes, not
  gigabytes** (~11 MiB / <2 s for 218 GiB), every seek lands on its exact target,
  overlapping seeks coalesce to the last target, the reader is still usable after
  the run, and Details reports one consistent
  requested/presented/displayed/rendered frame with no pending or in-flight
  frame.
- **native video editor:** run `trd-gui-video-editing --document ... --video ...`;
  verify source validation, streaming RGBA playback, play/pause/seek, timeline
  row identity, and the tracked/video-only transition. ffmpeg and ffprobe are the
  native media adapter; no temporary frame directory is used.

### 4. Windows e2e (manual — L3)

The Linux box is headless, so every path that needs a **display**, a **window
event loop**, or **Windows file/HTTP I/O** is verified here and nowhere else.
Mark N/A on Linux, and put the exact commands in the PR and issue handoff
whenever the current platform cannot run them. Run the ones your change touches;
a render-path or media-layer change runs all of them.

| # | Path | Command | What only Windows can catch |
|---|---|---|---|
| 4.1 | `trd-cli` | `examples\render.ps1 -CLI …` | MSVC toolchain, D3D12/Vulkan adapter choice |
| 4.2 | `trd-app` window | `examples\render.ps1 -Native …` | winit event loop, swapchain resize/occlusion/DPI |
| 4.3 | `trd-gui` window | `cargo run -p trd-gui-app -- --mesh …` | egui interaction on a real surface |
| 4.4 | web renderers | `examples\render.ps1 -Web -CanvasRenderer …` / `-OffscreenRenderer` | Chrome/Edge **on Windows** WebGPU backend (D3D12, not Vulkan) |
| 4.5 | native video editor | `cargo run -p trd-gui-video-editing -- --document … --video …` | ffmpeg/ffprobe **on Windows** — process spawn, path quoting |
| 4.6 | browser video editor | `bun run --cwd gui-video-editing dev` | WebCodecs on the Windows media stack |
| 4.7 | **large-file seek** | see below | 64-bit file offsets and ranged HTTP on Windows |

Use the [coca-cola can recipe](#cross-mode-e2e-recipe--coca-cola-can-pbr--aabb--axes)
for 4.1-4.4 so all four are driven by one scene and their colours can be compared
directly. For 4.2 also confirm playback runs at the stream's declared rate and
loops; for 4.5/4.6 run the editor checks listed under §3 — quad selection, all
three catalog assets, picking/editing, play/pause/seek, the video-only 222-287
tail, and Details' frame identity under rapid seek.

#### 4.7 — large-file seek (Windows, required for any media-layer change)

Linux coverage does **not** substitute: the file APIs, the process spawn, and the
browser's range-request stack are all different here.

This step tests **two separable properties**, and they need different files:

| Property | What proves it | Minimum file |
|---|---|---|
| **64-bit offsets** — the classic Windows failure is a `>4 GiB` offset truncated to 32 bits, showing up as a seek landing in the wrong place or an "unreadable" file rather than a crash | a seek past the 4 GiB mark landing on its exact target | **any MP4 > 4 GiB** |
| **Ranged-read economics** — that opening reads megabytes, not the whole file | `~11 MiB / <2 s` to open | a **multi-hundred-GiB** MP4, where a full read is unmistakable |

A short local clip cannot fail the way a 218 GiB one does, so the
multi-hundred-GiB file remains the standard. But **do not skip the step for want
of one**: a file merely over 4 GiB still exercises the offset-truncation trap,
which is the failure mode most likely to be Windows-only. Run what the device
has, and **name the file and its size in the PR** so a reviewer knows which of
the two properties was actually covered.

Both delivery surfaces must be driven, because they use different readers —
ffmpeg natively, mediabunny in the browser:

```powershell
# native — local path, then the same file over HTTP. `--probe-frame N` is how
# a deep seek is asserted without a window: it reports the frame that came
# back, not the one asked for.
cargo run -p trd-gui-video-editing -- --video <BIG.mp4> --probe-only
cargo run -p trd-gui-video-editing -- --video <BIG.mp4>
cargo run -p trd-gui-video-editing -- --video-url http://localhost:8092/<BIG.mp4> --probe-only --probe-frame <deep>
cargo run -p trd-gui-video-editing -- --video-url http://localhost:8092/<BIG.mp4>

# browser — serve the media, then serve the app *including* the probe page.
# `bun run dev` passes only index.html, so name probe.html explicitly; Bun
# then routes it at /probe (no .html).
bun web\gui-video-editing\serve-documents.ts <dir> --port 8092 --log
cd web\gui-video-editing; bun run build:wasm; $env:BUN_PORT='8085'; bun .\index.html .\probe.html
#   /probe?url=…&seek=<deep>&frames=8                      one deep seek
#   /probe?url=…&reader=mediabunny&scrub=t1,t2,…&overlap=1  dragged scrubber
#   /?document=none&reader=mediabunny&video=<url>          the editor itself
```

Expect, on **both** surfaces:

- **opening costs megabytes, not gigabytes** (~11 MiB / <2 s for 218 GiB) — a
  full read means the ranged path was bypassed;
- a seek **past the 4 GiB mark** lands on its exact target, and so does one near
  the very end (the offset-truncation trap);
- overlapping seeks coalesce to the last target, and the reader is still usable
  afterwards;
- Details reports one consistent requested/presented/displayed/rendered frame,
  with no pending or in-flight frame left over;
- with `?document=none` / no `--document`, the timeline is the **container's** —
  real frame count and rational fps, not an invented 30 fps grid (#264).

The big MP4 stays **external and uncommitted**; name the file and its size in the
PR so a reviewer knows which one was used.

## Cross-mode e2e recipe — coca-cola can (PBR + AABB + axes)

A concrete, reproducible pass that drives **every** front-end with one scene: the
Coca-Cola can (`assets/meshes/can/coke.obj` + `can_around.jpg`) shaded PBR with the
dielectric printed-can preset (**metallic 0.0 / roughness 0.35**, ACES lighting)
over the `qd_beer` dolly camera, with the AABB + world-axes overlays. The **same
params** apply on both platforms; only the wrapper syntax differs.
`render.ps1`/`render.sh` need `uv` + `ffmpeg` on PATH; the web modes need a WebGPU
browser (Chrome/Edge). Shared scene:

```
mesh=assets/meshes/can/coke.obj   texture=assets/meshes/can/can_around.jpg
pbr  metallic=0.0 roughness=0.35 env-intensity=0.90 exposure=0.45 ambient=0.03
specular=0.6 tonemap=aces   env=assets/envmap/uffizi-large.hdr   aabb  axes
input=examples/frames.qd_beer_dolly.cg.jsonl   size=512x768
```

**🪟 Windows** (`examples\render.ps1`; in `trd-gui` the overlays are the side-panel
**"Bounding box"** / **"World axes"** checkboxes):

```powershell
# trd-cli — headless → GIF
examples\render.ps1 -CLI -Mesh assets\meshes\can\coke.obj -Texture assets\meshes\can\can_around.jpg `
  -Pbr -Metallic 0.0 -Roughness 0.35 -EnvIntensity 0.90 -Exposure 0.45 -Ambient 0.03 -Specular 0.6 `
  -Tonemap aces -Env assets\envmap\uffizi-large.hdr -Aabb -Axes `
  -InputPath examples\frames.qd_beer_dolly.cg.jsonl -Output output\coca.gif -Width 512 -Height 768
# trd-app — native window (same flags, -Native, no -Output)
examples\render.ps1 -Native -Mesh assets\meshes\can\coke.obj -Texture assets\meshes\can\can_around.jpg `
  -Pbr -Metallic 0.0 -Roughness 0.35 -EnvIntensity 0.90 -Exposure 0.45 -Ambient 0.03 -Specular 0.6 `
  -Tonemap aces -Env assets\envmap\uffizi-large.hdr -Aabb -Axes `
  -InputPath examples\frames.qd_beer_dolly.cg.jsonl -Width 512 -Height 768
# trd-gui native
cargo run -p trd-gui-app -- --mesh assets\meshes\can\coke.obj `
  --texture assets\meshes\can\can_around.jpg --pbr --env assets\envmap\uffizi-large.hdr `
  --metallic 0.0 --roughness 0.35 --env-intensity 0.90 --exposure 0.45 --ambient 0.03 `
  --specular 0.6 --tonemap aces
# trd-wasm web — build+serve, open http://localhost:8080 (swap -CanvasRenderer ⇄ -OffscreenRenderer)
examples\render.ps1 -Web -CanvasRenderer -Mesh assets\meshes\can\coke.obj -Texture assets\meshes\can\can_around.jpg `
  -Pbr -Metallic 0.0 -Roughness 0.35 -EnvIntensity 0.90 -Exposure 0.45 -Ambient 0.03 -Specular 0.6 `
  -Tonemap aces -Env assets\envmap\uffizi-large.hdr -Aabb -Axes `
  -InputPath examples\frames.qd_beer_dolly.cg.jsonl -Width 512 -Height 768
# trd-gui web — build+serve, then open the ?mesh/?texture/?env URL below
cd web; $env:BUN_PORT='8082'; bun run --cwd gui-viewer dev
#   http://localhost:8082/?mesh=/assets/meshes/can/coke.obj&texture=/assets/meshes/can/can_around.jpg&env=/assets/envmap/uffizi-large.hdr
```

**🐧 Linux/Nix** (inside `nix develop`; on non-NixOS wrap the GPU commands
—`render.sh`, `cargo run -p trd-gui-app`— with nixGL, see [GPU](#gpu); the web URLs are
identical). Same params for **trd-cli**, **trd-wasm/web**, **trd-gui-native** and
**trd-gui-web**:

```sh
# trd-cli — headless → GIF
examples/render.sh --cli --mesh assets/meshes/can/coke.obj --texture assets/meshes/can/can_around.jpg \
  --pbr --metallic 0.0 --roughness 0.35 --env-intensity 0.90 --exposure 0.45 --ambient 0.03 --specular 0.6 \
  --tonemap aces --env assets/envmap/uffizi-large.hdr --aabb --axes \
  examples/frames.qd_beer_dolly.cg.jsonl output/coca.gif 512 768
# trd-wasm web — build+serve, open http://localhost:8080 (swap --canvas-renderer ⇄ --offscreen-renderer)
examples/render.sh --web --canvas-renderer --mesh assets/meshes/can/coke.obj \
  --texture assets/meshes/can/can_around.jpg --pbr --metallic 0.0 --roughness 0.35 --env-intensity 0.90 \
  --exposure 0.45 --ambient 0.03 --specular 0.6 --tonemap aces --env assets/envmap/uffizi-large.hdr \
  --aabb --axes examples/frames.qd_beer_dolly.cg.jsonl 512 768
# trd-gui native
cargo run -p trd-gui-app -- --mesh assets/meshes/can/coke.obj \
  --texture assets/meshes/can/can_around.jpg --pbr --env assets/envmap/uffizi-large.hdr \
  --metallic 0.0 --roughness 0.35 --env-intensity 0.90 --exposure 0.45 --ambient 0.03 \
  --specular 0.6 --tonemap aces
# trd-gui web — build+serve (build:wasm + serve.ts), then open the ?mesh/?texture/?env URL below
# on Linux the RTX box is headless — ALWAYS reach the web viewer via an SSH port-forward:
#   ssh -L 8082:localhost:8082 <host>   then open the localhost URL on your workstation
cd web && BUN_PORT=8082 bun run --cwd gui-viewer dev
#   http://localhost:8082/?mesh=/assets/meshes/can/coke.obj&texture=/assets/meshes/can/can_around.jpg&env=/assets/envmap/uffizi-large.hdr
```

Expect a deep-red, crisp-label can (**not** a washed-out metal) with a green AABB
box and the R/G/B world axes. The `trd-gui` viewers start in PBR from `--env` /
`?env=` (tick the overlay checkboxes; nudge roughness → 0.35). Colors must match
across trd-cli, trd-app, and both web renderers.

> **Linux web access — always SSH port-forward, always the PBR coca-can.** The RTX
> Linux box is headless (no local display), so the browser viewers (`trd-wasm` and
> `trd-gui` web) are **always** reached over an **SSH port-forward**: tunnel the bun
> dev-server port from your workstation (`ssh -L 8082:localhost:8082 <host>`, add
> `-N` to forward only) and open the `http://localhost:8082/?mesh=…&texture=…&env=…`
> URL locally. The **PBR coca-cola can** (`coke.obj` + `can_around.jpg` +
> `uffizi-large.hdr`) is the standard demo scene for these launches.

---

## The matrix template

Copy this into the PR (and mirror it in the issue). The gate rows and their order
are fixed by [`AGENTS.md`](../AGENTS.md#verification-matrix); **every row is
always present**, and one the level excludes reads `n/a (Lx)` rather than being
deleted.

```markdown
**Test level: L2** — touches `crates/trd-core/src/render/` (render path).

## ✅ Verification

| Gate | 🪟 Windows | 🐧 Linux/Nix |
|------|:---:|:---:|
| 🎨 `cargo fmt --check`         | ✅ | 🤝 |
| 📎 clippy native `-D warnings` | ✅ | 🤝 |
| 🕸️ clippy wasm32 (lib)         | ✅ | 🤝 |
| 🧪 `cargo test --lib` (173)    | ✅ | 🤝 |
| 🔀 `decoder_parity` (2)        | ✅ | 🤝 |
| 📚 rustdoc (0 broken links)    | ✅ | 🤝 |
| 🌐 `tsc --noEmit` + Biome      | ✅ | 🤝 |
| 🖼️ `golden_render` (6/6, GPU)  | ✅ | 🤝 |
| 🎮 `gpu_tests` + `gui_render`  | ✅ | 🤝 |
| 🖥️ window e2e (§4.2/4.3)       | n/a (L2) | n/a (L2) |
| 🎬 video-editor e2e (§4.5/4.6) | n/a (L2) | n/a (L2) |
| 📼 large-file seek (§4.7)      | n/a (L2) | n/a (L2) |

## 🤝 Handoff — 🐧 Linux/Nix
- [ ] `nix flake check -L`
- [ ] nixGL-wrapped `cargo test -p trd-core --test golden_render -- --ignored`

> Expected: all green — behaviour-preserving change.
```

Since **there is no CI** — `.github/workflows/ci.yml` is `disabled_manually` —
every one of those cells is filled by a command someone ran on their own machine.
Fill only your platform's column and hand the other off.

---

See also: [`AGENTS.md`](../AGENTS.md) (which gates a change owes),
[`docs/rendering.md`](rendering.md) (every front-end and its flags),
[`docs/architecture.md`](architecture.md) (what the gates are protecting).
