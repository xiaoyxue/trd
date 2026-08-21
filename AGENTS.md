# AGENTS.md

Guidance for agents working in this repository.

## Architecture

- **One rendering core: Rust + wgpu** (`crates/trd-core`). The same code renders
  natively (headless CLI + interactive window) and in the browser (wasm).
  **JS/TS is a thin bootstrap only** — never call the WebGPU API from JS; all
  rendering logic lives in Rust.
- **Typed math** (`crates/trd-core/src/math/`): homogeneous linear algebra
  (`Vector`/`Point`/`Normal`/`Matrix`/`Rotation`/`Transform`/`Aabb`) as zero-cost
  `#[repr(transparent)]` newtypes over glam with **private** fields that enforce
  affine rules glam can't (`point − point → vector`, no `point + point`).
  Conventions live in `math/mod.rs`: column-major, right-handed, clip
  `z ∈ [0, 1]`, `a.then(b) == b * a`, f32 radians. The `render/` MVP transforms
  build on it — keep the GPU `Uniform` byte-identical when touching them. SIMD:
  native SSE2/NEON is automatic; wasm needs `-C target-feature=+simd128`
  (`.cargo/config.toml`).
- **Vertical slicing.** Each increment threads the whole stack and is
  independently end-to-end verifiable.
- **Everything drawn is a `DrawableObject`** (`render/drawable.rs`, #41): a small
  `Copy` struct pairing a `Primitive` — *what* to draw — with the `model` that
  places it (`Primitive::Mesh { mesh_id, mode }` | `AabbBox { mesh_id }` |
  `CoordinateAxes` | `PlaneGrid { plane }` | `QuadOutline { selected }` |
  `QuadFill` | `BlobShadow`) — the single base
  interface for every primitive. **Every drawable is a placed primitive**: it
  names geometry and carries the model that places it, so it can be instanced.
  What a frame draws *behind* them — the HDR environment probe and the
  background frame plane — is not a primitive but a per-frame setting on
  `Scene::background()` (`Background { environment, frame }`, two **independent**
  `Option`s, since both are drawn: environment first, frame plane over it — #204).
  Geometry is owned once (decode-once mesh store + shared
  gizmo buffers); a drawable is a light handle naming *which* primitive + its
  per-frame model. Every front-end hands the same `Scene`
  (objects + background, rebuilt per frame) to `Renderer::render` without
  per-type branching; the render core batches its objects by primitive — a batch
  key *is* a drawable minus its model, so there is one taxonomy, not two (#204),
  and `Primitive::sort_key` spells out the submission order that every
  depth-disabled overlay depends on. A single-object
  frame is the degenerate one-element scene. Add a primitive by adding a variant,
  not by bolting flags onto the renderer. Wireframe (and PBR) is a *mode* of
  `Mesh`, not a separate variant.
- **One render harness; targets are pure data, behaviour is on the renderer.**
  `trd_core::Renderer` (`render/renderer.rs`) owns *GPU context +
  pipelines/materials/mesh store* for **every** front-end, and owns **all** the
  render behaviour too. `render/render_target.rs` is data only — `TextureTarget`
  (texture + padded staging buffer), `SurfaceTarget` (surface + config + sRGB view
  format), and the closed enum `RenderTarget` over the two, which holds *only* the
  discriminant so each variant stays the single source of truth for its own size.
  A target carries **no** `render`/`present`/`acquire`/`read_back` method (#203):
  a swapchain handle knows nothing about pipelines or the mesh store, so
  `Renderer::render(camera, scene, &mut RenderTarget)` is the **one** public entry
  — a single match over private `render_surface` (acquire → encode → submit →
  present) and `render_texture` (encode → submit) — returning
  `Result<Option<SurfaceRepair>, RenderError>` **synchronously** (no `async fn`
  may cross the `wasm_bindgen` boundary). The asymmetric tail is typed, not
  branched: `read_pixels` and the multi-camera
  `draw_layers`/`render_layers`/`render_params` take the concrete
  `&TextureTarget`, so asking a *surface* for pixels is a compile error. Target
  lifecycle lives on the renderer for the same reason (`create_texture_target`,
  `create_surface_target`, `resize_texture_target`, and the *associated*
  `resize_surface`/`reconfigure_surface`/`replace_surface`, which take a
  `&wgpu::Device` because a window is resized and repaired before a mesh has
  arrived to build a `Renderer` from). Texture targets serve `trd-cli`, `trd-gui`
  and the browser `OffscreenRenderer`; surface targets serve `trd-app` and
  `trd-wasm`'s `CanvasRenderer`. Front-ends are **shells, not renderers** — they
  create the surface, own their target, and apply their own
  **surface-recovery policy** to the reported
  `Ok(Some(SurfaceRepair))` / `Err(RenderError::Surface(_))` (native defers to
  the next redraw; the browser repairs in-call and retries once).
- Major input data is columnar (Apache Arrow tables) with simple glue logic.
- **Video editing uses a separate authoring document.**
  `web/gui-video-editing` reads `trd.video_edit.version = 0.2.0` timeline rows
  (video metadata/optional poster + per-frame K/quad/tracked state), from
  **Arrow IPC or Parquet** — the container is sniffed from the bytes, not the
  file name, and both feed one decoder so either produces the same document.
  The table is
  **sparse**: a row exists only for a frame carrying an ad-placement quad, and a
  frame with no row is plain video — so the document is also **optional**, and
  without one the editor is a player whose timeline comes from the container
  (#264). It is deliberately
  independent of render `PROTOCOL_VERSION`; do not add editor-only columns to
  `0.0.6` or bump the render protocol for editor state. Rust maps each presented
  browser `VideoFrame` or native ffmpeg frame to a timeline row, reconstructs
  the quad through `crates/trd-placement`, and derives ordinary
  `DrawableObject`s. Browser and native delivery surfaces host the same
  `VideoEditingApp`, renderer, catalog, placement, picking, and egui panels;
  only their media adapters differ. Current catalog resources are loaded at
  runtime; protocol export remains a separate later slice.
- **Browser demuxing/decoding is [mediabunny](https://mediabunny.dev/), always —
  never a hand-rolled reader.** `MediabunnyReader`
  (`web/gui-video-editing/src/media/mediabunny-reader.ts`) is *the* browser media
  adapter, behind the `FrameReader` seam (#290). Range reads, locating `moov`,
  feeding the demuxer, decoder configuration/reset, key-frame catch-up and the
  end-of-stream drain are the library's job — that layer is where every playback
  fault this package has hit lives (a `flush()` deadlocking on its own output
  pool, a seek offset misread as end of stream, samples arriving after a drain,
  a decoder left `closed` by an overlapping seek), and re-implementing it is how
  they come back. So: **do not extend the hand-written mp4box + `VideoDecoder`
  reader** (`media/mp4-video.ts`, and `VideoPlayer.open`, which still defaults to
  it while `?reader=mediabunny` opts in) — fix things in the mediabunny path,
  make it the default, and delete rather than grow the self-implemented one. The
  only part deliberately kept ours is the raw `moov` box walk (`locateMoov`),
  because Rust needs the frame rate as a **rational**, which mediabunny does not
  surface. New media work goes through `FrameReader` so both delivery surfaces
  and the `?reader=` probe keep measuring the same seam.
- **The input protocol is NOT backward compatible.** Wire format is **mesh-first**
  `[mesh][texture?][frames?][params]`; only the current `PROTOCOL_VERSION`
  (`trd_core::protocol::PROTOCOL_VERSION`, currently `0.0.6`) is accepted — every
  table's `trd.protocol.version` and `trd.table.kind` are checked and any other
  version is
  **hard-rejected** (`UnsupportedVersion`), never silently upgraded. To evolve it,
  **bump `PROTOCOL_VERSION` and migrate all producers + fixtures in the same
  change** (`scripts/{jsonl,obj,texture}_to_arrow.py` stamp the version;
  regenerate the golden `stage{1,2}.arrow` fixtures). Do **not** re-add branches
  for retired versions or params-only (hello-triangle) streams — dropping that
  legacy is deliberate (#82/#90). Every input must begin with a mesh table.

## Toolchain

- **Dev shell:** `nix develop` (pinned Rust via rust-overlay, `bun`,
  `wasm-bindgen-cli`, `biome`, `typescript`, Vulkan). Local `cargo` inside it
  works for fast iteration.
- **The flake is the build system, not just a dev shell.** Prefer the real outputs:
  - `nix build .#trd-cli` — native CLI (`trd-cli`), wrapped with Vulkan/GL libs.
    `nix run .#trd-cli -- --width 256 --height 256` runs the Arrow stream filter
    (frames on stdin → images on stdout).
  - `nix build .#trd-wasm` — the `wasm-bindgen` JS/TS library (built with
    `wasm-bindgen-cli` + `wasm-opt`, replacing `wasm-pack`).
  - `nix build .#web` (also `.#`) — the bun-bundled, HTTP-servable `dist/`.
    `nix run .#web` serves it (`PORT` overridable, defaults to 8080).
  - `nix flake check` — every non-GPU gate: `cargo fmt`, clippy (native +
    wasm32), `cargo test`, `tsc --noEmit`, Biome.
  - `nix fmt` — formats nix files (`nixfmt`).
- **Delivery surfaces are grouped by platform.** Native binaries live in
  `native/{trd-app,trd-gui-app,trd-gui-video-editing}`; reusable Rust stays in `crates/`. Browser apps
  live as sibling packages in `web/{viewer,gui-viewer,gui-video-editing}`.
- **`crates/trd-wasm` is the *only* browser delivery surface.** **Every**
  `#[wasm_bindgen]` export in the repo lives there — the viewer's
  `CanvasRenderer`/`OffscreenRenderer` *and* the GUI's `start` /
  `startVideoEditing` / `VideoEditingHandle` (`src/gui.rs`, `src/gui_web_app.rs`).
  Every other crate, `trd-gui` included, is a plain `rlib` free of
  `wasm-bindgen`, so there is one wasm build and one generated JS package
  (`trd_wasm`) that all three web packages stage into their own `pkg/` (#180).
  Do **not** add a `cdylib` crate-type or a `#[wasm_bindgen]` item anywhere else.
- **`web/` is a Bun workspace** with sibling `viewer/`, `gui-viewer/`, and
  `gui-video-editing/` packages. Each package's lint/format gate is Biome; run all
  packages' checks from the workspace root
  with `bun run check`.
  Each package owns a generated `trd_wasm` package under its own `pkg/` (all
  three stage the same `crates/trd-wasm` build); no delivery surface imports
  another's build output.
  Run `bun run check` / `format` / `typecheck` from `nix develop`, or directly on
  Windows — `@biomejs/biome` + `apache-arrow` are in
  `web/viewer/package.json`. On a clean non-Nix checkout, first run
  `bun run --cwd viewer build:wasm` and
  `bun run --cwd gui-viewer build:wasm` and
  `bun run --cwd gui-video-editing build:wasm` from `web/` to stage all local
  wasm packages, then `bun install --frozen-lockfile`; the workspace
  `check`/`typecheck`/build scripts work without Nix.
- **Nix web deps are installed offline via
  [bun2nix](https://github.com/nix-community/bun2nix).** `web/bun.nix`
  (generated from `web/bun.lock`, hash-pinned) lets
  `nix build .#web` and the `tsc` gate `bun install` reproducibly. **Regenerate
  `web/bun.nix` whenever `web/bun.lock` changes** (add/upgrade an
  npm dep): from `nix develop`, `cd web && bun install` (updates
  `bun.lock`) then
  `nix run github:nix-community/bun2nix -- -l web/bun.lock -o web/bun.nix`, and
  **delete the autogenerated `"trd-wasm" = copyPathToStore ...;` line** — that
  `file:` dep is supplied by the nix-built `trd-wasm`, not fetched. The Biome gate
  runs from nixpkgs' `biome` (version-matched to `package.json`), since bun can't
  materialize biome's large optional platform binary in the sandbox.
- **`nix build`/`nix flake check` only see git-tracked files.** `git add` new
  files before building, or the sandbox won't include them.

### GPU

- We always work on GPU machines. GPU-dependent tests are marked `#[ignore]` and
  run locally; CI skips them.
- **Always render on the most powerful GPU available.** With multiple adapters,
  pick the strongest discrete card; never fall back to a weak display/iGPU (e.g. a
  Quadro P620) or software (llvmpipe). Preference (strongest first):
  `RTX PRO 6000 > RTX 5090 > RTX 6000 Ada > RTX 4090 > RTX A6000 > RTX 3090 > others`.
  List adapters with `nvidia-smi --query-gpu=index,name,memory.total --format=csv`;
  confirm trd's choice from its `trd_core=info` log line
  `using Vulkan adapter "…" (DiscreteGpu)`. Adapter selection lives in
  `render/gpu_context.rs`, whose `GpuRequest` defaults to
  `PowerPreference::HighPerformance`, so Vulkan prefers the discrete card
  (verified: picks the RTX 3090 over a P620). To force one: Mesa
  `MESA_VK_DEVICE_SELECT=<vendorId>:<deviceId>`; multi-GPU NVIDIA
  `__NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia` (GL) — plain
  `CUDA_VISIBLE_DEVICES` does **not** filter Vulkan physical devices.
- **Linux non-NixOS (e.g. Ubuntu): use nixGL.** The `nix develop` Vulkan loader
  can't `dlopen` the host NVIDIA/Mesa driver (fails with *"No suitable graphics
  adapter found"*). Wrap GPU commands to inject a matching host driver:
  ```sh
  # inside `nix develop`; --impure lets nixGL detect the host driver version
  NIXPKGS_ALLOW_UNFREE=1 nix run --impure github:nix-community/nixGL#nixGLNvidia -- \
    cargo test -p trd-core -- --ignored          # or: ./result/bin/trd-cli …, render.sh, etc.
  ```
  `NIXPKGS_ALLOW_UNFREE=1` is required for NVIDIA; use `#nixGLIntel` for
  Intel/Mesa. NixOS doesn't need this (driver on `/run/opengl-driver`); WSL uses
  `WGPU_BACKEND=gl` (below).
- **WSL2:** NVIDIA ships no native Linux Vulkan ICD, so Vulkan falls back to
  software (llvmpipe) and Mesa's `dzn` (Vulkan-on-D3D12) crashes at device
  creation. Use `WGPU_BACKEND=gl` for real GPU rendering via Mesa's D3D12 GL
  driver; the dev shell auto-configures this on WSL.

### Golden render test (#88) — the render-regression gate

`crates/trd-core/tests/golden_render.rs` feeds committed Arrow fixtures
(`crates/trd-core/tests/golden/stage{1,2}.arrow`, the reduced two-stage
cornellbox placement demo) through the real `run_stream` pipeline and pixel-diffs
the frames against committed golden PNGs (same dir). Each params row selects an
inline `0.0.6` frames-table resource by `frame_id` (stage 1 encoded Binary,
stage 2 raw tensor), composited **under** the scene.

Each fixture is rendered **twice — 4× MSAA (`Msaa::X4`, the default anti-aliased
mesh pass) and MSAA-off (`Msaa::Off`, single-sample)** — each pinned to its own
goldens (`stageN_*` vs `stageN_noaa_*`), so both the multisample+resolve path and
the raw single-sample path are covered; plus PBR tone-map variants
(`golden_stage2_pbr_{aces,reinhard}`), plus
`golden_environment_light_syncs_sky_and_reflection` — a hand-built scene (no
fixture can draw a sky) pinning that the scene's one `EnvironmentLight.rotation`
drives the visible sky **and** the reflections on a near-mirror ball in front of
it (#182). It is GPU-gated (`#[ignore]`); run it via
the nixGL wrapper (Linux) or directly on a Windows box with a discrete GPU.

**Mandatory:** any change touching the render path (`crates/trd-core` render code,
PBR/tone-map, shaders, or the golden fixtures) MUST run the golden suite on a real
GPU and land green — on **every** platform where a GPU is available — before the
task is done. After an *intended* visual change or a fixture change, regenerate:
```sh
# 1. rebuild the .arrow fixtures + stills (needs uv + ffmpeg on PATH)
python3 scripts/golden_fixtures.py
# 2. refresh the golden PNGs from the current renderer (GPU box)
TRD_UPDATE_GOLDENS=1 cargo test -p trd-core --test golden_render -- --ignored
```
The companion **non-GPU** `tests/decoder_parity.rs` decodes the same fixtures
through both **public API surfaces** — the native `InputStream`
(`io/input_stream.rs`, a byte transport owning a `Read`) and the browser's push
`InputSession` — and asserts identical *assembled frames*. Neither the column
decode nor the framing is duplicated: both run the one decoder in
`protocol/arrow_decode.rs` through the one `InputSession`. What this guards is
that the two surfaces a caller assembles a frame through agree —
`prologue`/`next_batch`/`finish` versus a bare `push`, and `InlineFrameCache`
versus `InlineFrame::decode` — the shape of failure the `center` non-nullable bug
had. It runs in `nix flake check`.

## PR Workflow

- **pr_first:** push work to a feature branch and open a **draft PR** as early as
  practical; use the PR as the working surface.
- **auto_merge: small** — small, low-risk PRs may be squash-merged once CI is
  green. Risky PRs (public API, schemas, migrations, auth, infra) require human
  review.
- **branch naming:** `feat/<topic>`, `fix/<topic>`, `docs/<topic>`, etc.
  **merge strategy:** squash. PRs that resolve an issue include a `Closes #nn`
  keyword.
- **issue titles:** every issue title **starts with a bracketed category tag**
  naming the kind of work — it is the first token of the title, before the short
  description (`[Tag] <concise summary>`). Use the canonical set:
  - `[Epic]` — umbrella / tracking issue spanning multiple slices.
  - `[Feature]` — a shippable user-facing capability grouping several slices
    (narrower than an `[Epic]`, broader than one `[Slice]`); tracks its slices.
  - `[Design]` — a spec/design settled *before* implementation.
  - `[Plan]` — a roadmap / sequencing plan across issues.
  - `[Slice]` — one vertical slice increment (independently end-to-end verifiable).
  - `[Investigation]` — a research spike / open question to resolve.
  - `[Refactoring]` — a behaviour-preserving restructure (regression net must stay green).
  - `[Risk]` — a risk-register item or hardening task (things likely to break/slip).
  - `[Eval]` — an evaluation / benchmark / quality-measurement task.
  - `[Test]` — a test-only slice (fixtures, goldens, regression nets; no product behaviour change).

  Others in use as needed: `[Ops]`, `[Demo]`, `[Prep]`, `[Integration]`, `[Docs]`.
  Pick the single tag that best fits the issue's primary intent.

### Testing — required before a task is "done"

Run the smallest command that covers the change during iteration, but a task is
not complete until these tiers pass; **record the results on the PR.**

#### Where a test lives — by kind, not by size (#305)

Two kinds of test, and the compiler already tells them apart:

- **Unit test → inline**, in the module it tests: `#[cfg(test)] mod tests { … }`
  at the bottom of that `.rs` file. A unit test reaches into the module's own
  internals (`use super::*`, private fields and functions), so it belongs beside
  the code it pins. **Length is irrelevant** — a 1,200-line unit-test module
  still lives in its module. Do **not** split it out to a sibling `tests.rs`.
- **Integration test → its own file** under `crates/<crate>/tests/`
  (`golden_render.rs`, `decoder_parity.rs`, `gui_render.rs`,
  `wasm_bindgen_containment.rs`). These compile as separate crates and may only
  touch the public API, which is what makes them worth isolating.

There is deliberately **no third form**: a `src/**/tests.rs` is compiled into the
crate exactly like an inline `mod tests`, so it is a unit test wearing an
integration test's clothes. A size threshold was tried and dropped (#299 §1,
#305) — it made the location of a test say nothing about what the test *is*, and
forced file moves whenever a module crossed a line count.

Test-only **support** modules are a different thing and stay as they are:
`render/gpu_tests.rs`, `render/triangle_renderer.rs` and
`protocol/scene_encode.rs` are not `mod tests` blocks but helper modules that
happen to be test-gated.

When a module does get too long to read, the fix is to split the *module* — by
responsibility, tests following their code — not to hide its tests in another
file.

1. **Golden test — MSAA enabled *and* disabled (must).**
   `cargo test -p trd-core --test golden_render -- --ignored` runs both the 4×
   MSAA (`stageN_*`) and single-sample (`stageN_noaa_*`) goldens plus the PBR
   tone-map variants. GPU-gated — see [Golden render test](#golden-render-test-88--the-render-regression-gate).
   Mandatory for any render-path change, on every platform with a GPU.
2. **GPU-gated tests (must).** Every `#[ignore]` test, on a real GPU:
   `cargo test -p trd-core -- --ignored` (golden + `render::gpu_tests`) and
   `cargo test -p trd-gui --test gui_render -- --ignored`
3. **End-to-end — Linux *and* Windows:**
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
     match the other front-ends. Open **Details** and confirm its displayed
     frame identity does not jump ahead during rapid seek/render, and that the
     Dragon reports its GLB material maps, raw tracking pose delta, zero direct
     light/ambient, and Uffizi IBL. The MP4 stays external/uncommitted.
   - **large video over a URL (media-layer gate):** any change to
     `web/gui-video-editing/src/media/` must also be run against a
     *multi-hundred-GiB* MP4 **served over HTTP**, because file size is exactly
     what a ranged reader is for and a local short clip cannot fail the way a
     218 GiB one does. Serve it with the CORS+range helper
     (`bun web/gui-video-editing/serve-documents.ts <dir> --port 8092`), then:
     drive `probe.html` — `?url=…&seek=…&frames=…` for one deep seek and
     `?reader=mediabunny&scrub=t1,t2,…` (plus `&overlap=1`, the dragged-scrubber
     shape) for repeated seeks on one reader — and open the editor itself at
     `?document=none&reader=mediabunny&video=<url>`. Expect: **opening costs
     megabytes, not gigabytes** (~11 MiB / <2 s for 218 GiB), every seek lands on
     its exact target, overlapping seeks coalesce to the last target, the reader
     is still usable after the run, and Details reports one consistent
     requested/presented/displayed/rendered frame with no pending or in-flight
     frame.
   - **native video editor:** run `trd-gui-video-editing --document ... --video
     ...`; verify source validation, streaming RGBA playback, play/pause/seek,
     timeline row identity, and the tracked/video-only transition. ffmpeg and
     ffprobe are the native media adapter; no temporary frame directory is used.
4. **Windows e2e (manual).** The Linux box is headless, so every path that needs
   a **display**, a **window event loop**, or **Windows file/HTTP I/O** is verified
   here and nowhere else. Mark N/A on Linux, and put the exact commands in the PR
   and issue handoff whenever the current platform cannot run them. Run the ones
   your change touches; a render-path or media-layer change runs all of them.

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
   for 4.1-4.4 so all four are driven by one scene and their colours can be
   compared directly. For 4.2 also confirm playback runs at the stream's declared
   rate and loops; for 4.5/4.6 run the editor checks listed under §3 — quad
   selection, all three catalog assets, picking/editing, play/pause/seek, the
   video-only 222-287 tail, and Details' frame identity under rapid seek.

   **4.7 — large-file seek (Windows, required for any media-layer change).**
   A multi-hundred-GiB MP4 is the only thing that exercises **64-bit offsets**;
   a short local clip cannot fail the way a 218 GiB one does, and the classic
   Windows-only failure is a `>4 GiB` offset truncated to 32 bits, which shows up
   as a seek landing at the wrong place or an "unreadable" file rather than as a
   crash. Linux coverage does **not** substitute: the file APIs, the process
   spawn, and the browser's range-request stack are all different here.

   Both delivery surfaces must be driven, because they use different readers —
   ffmpeg natively, mediabunny in the browser:

   ```powershell
   # native — local path, then the same file over HTTP
   cargo run -p trd-gui-video-editing -- --video <BIG.mp4> --probe-only
   cargo run -p trd-gui-video-editing -- --video <BIG.mp4>
   cargo run -p trd-gui-video-editing -- --video-url http://localhost:8092/<BIG.mp4>

   # browser — serve with the CORS+range helper, then drive probe.html and the editor
   bun web\gui-video-editing\serve-documents.ts <dir> --port 8092
   #   probe.html?url=…&seek=<deep>&frames=8                 one deep seek
   #   probe.html?url=…&reader=mediabunny&scrub=t1,t2,…&overlap=1   dragged scrubber
   #   /?document=none&reader=mediabunny&video=<url>          the editor itself
   ```

   Expect, on **both** surfaces:
   - **opening costs megabytes, not gigabytes** (~11 MiB / <2 s for 218 GiB) — a
     full read means the ranged path was bypassed;
   - a seek **past the 4 GiB mark** lands on its exact target, and so does one
     near the very end (the offset-truncation trap);
   - overlapping seeks coalesce to the last target, and the reader is still
     usable afterwards;
   - Details reports one consistent requested/presented/displayed/rendered frame,
     with no pending or in-flight frame left over;
   - with `?document=none` / no `--document`, the timeline is the **container's**
     — real frame count and rational fps, not an invented 30 fps grid (#264).

   The big MP4 stays **external and uncommitted**; name the file and its size in
   the PR so a reviewer knows which one was used.

The non-GPU gates (`nix flake check`: `cargo fmt`, clippy native + wasm32,
`cargo test`, `tsc`, Biome) must pass on both platforms as well.

#### Cross-mode e2e recipe — coca-cola can (PBR + AABB + axes)

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

### Multiple-platform verification and handoff

trd ships a **native Windows** path (`trd-cli`/`trd-app`/`trd-gui`,
`examples/render.ps1`) **and** a **Linux/Nix** path (`nix flake check`, GPU-gated
`#[ignore]` tests on the RTX box, `render.sh`), plus a **browser/wasm** path
shared by both. Every task must be verified on **both** OS platforms before it is
considered done:

- **Linux:** `nix flake check` (fmt, clippy native + wasm32, tests, tsc, biome)
  plus the GPU-gated tests (via nixGL / `WGPU_BACKEND=gl`, see [GPU](#gpu)).
- **Windows (MSVC):** build/run the affected native path (`trd-cli`/`trd-app`/
  `trd-gui`; `examples/render.ps1` for the demo) and confirm it renders. On a box
  with a discrete GPU the golden test is **runnable on Windows too** (wgpu Vulkan
  — verified on a GTX 1080 Ti), so also run
  `cargo test -p trd-core --test golden_render -- --ignored` there; don't defer it
  entirely to Linux.
- **Required render gate:** for any render-path change, the golden suite (MSAA
  on/off + PBR `{aces,reinhard}` variants) MUST pass on a real GPU — or the
  goldens be regenerated for an *intended* visual change — and the result recorded
  on the PR. It is the primary pixel-level regression net.
- **Handoff:** for whichever platform you cannot run yourself, leave an explicit
  note (exact commands + expected result) in **both** the issue and the PR, so the
  other platform's verification can be completed and recorded there.
- **PR presentation — verification matrix & handoff list (required).** Present the
  dual-platform results as a scannable, icon-led **verification matrix** plus an
  explicit **handoff list**, so a reviewer sees at a glance what passed, where, and
  what is still owed. Use this on **every** PR (and mirror it in the issue); never
  report gates as bare prose. Keep to a fixed glyph set so the format stays
  consistent and greppable — **status:** ✅ passed · ❌ failed · ⏳ not yet run · 🤝
  handed off · n/a not applicable; **platform:** 🪟 Windows · 🐧 Linux/Nix;
  **gate:** 🎨 fmt · 📎 clippy · 🕸️ clippy-wasm · 🧪 tests · 🔀 decoder-parity ·
  🖼️ golden-render · 🎮 gpu-tests · 🌐 tsc/biome · 🖥️ window e2e (§4.2/4.3) ·
  🎬 video-editor e2e (§4.5/4.6) · 📼 large-file seek (§4.7).
  One matrix row per gate, one column per platform (cells are status icons, an
  optional count like `(173)`); the handoff list is a 🤝-headed checklist of the
  exact commands the *other* platform still owes, closed by a one-line expected
  result:
  ```markdown
  ## ✅ Verification

  | Gate | 🪟 Windows | 🐧 Linux/Nix |
  |------|:---:|:---:|
  | 🎨 `cargo fmt --check`         | ✅ | 🤝 |
  | 📎 clippy native `-D warnings` | ✅ | 🤝 |
  | 🕸️ clippy wasm32 (lib)         | ✅ | 🤝 |
  | 🧪 `cargo test --lib` (173)    | ✅ | 🤝 |
  | 🔀 `decoder_parity` (2)        | ✅ | 🤝 |
  | 🖼️ `golden_render` (6/6, GPU)  | ✅ | 🤝 |
  | 🖥️ window e2e (§4.2/4.3)       | ✅ | n/a |
  | 📼 large-file seek (§4.7)      | ✅ | n/a |

  ## 🤝 Handoff — 🐧 Linux/Nix
  - [ ] `nix flake check -L`
  - [ ] nixGL-wrapped `cargo test -p trd-core --test golden_render -- --ignored`

  > Expected: all green — behaviour-preserving change.
  ```
- **Re-post the completed matrix after a handoff.** When you finish the
  verification a PR handed off to your platform, don't report it as bare prose —
  **re-post the full verification matrix as a new comment** on the PR (and mirror
  it on the issue) with that platform's column flipped from 🤝 to ✅ (with counts),
  and tick the corresponding items in the 🤝 handoff checklist. The completed
  matrix must be visible as a comment, not only described.

### Documentation

- **`README.md` is a lean entry point; the detail lives in `docs/`.** Keep them in
  sync: whenever a change updates `README.md`, update the affected `docs/` page(s)
  in the **same** PR — and vice-versa. In particular, `docs/architecture.md`
  (crates/render core), `docs/rendering.md` (CLI flags, wrappers ⇄ `cargo run`,
  demos), `docs/pbr.md` (PBR params), `docs/video-editing.md` (editor timeline,
  playback, placement, and catalog), and `docs/protocol/0.0.6.md` (wire format)
  mirror the README's summaries, so a behavior/flag/layout change must be
  reflected in both. Don't let the README and `docs/` drift.

### Worktrees

Keep the git root checkout on `main` at all times, and **update local `main`
before creating a new worktree** so the branch starts from an up-to-date base.
Do all branch/PR work in a git worktree under the root's `.worktree/` folder
(gitignored):
```sh
git switch main && git pull            # refresh local main first
git worktree add .worktree/<topic> -b feat/<topic>
cd .worktree/<topic>
```
Never check out a feature branch in the root itself. Remove the worktree after the
PR merges (`git worktree remove .worktree/<topic>`).

There is only **one** tracked `AGENTS.md` (repo root); a worktree's `AGENTS.md` is
just that file checked out on its branch, not a separate copy to reconcile — edit
it here and it lands with the branch.
