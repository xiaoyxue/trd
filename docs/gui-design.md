# `trd-gui` design — interactive egui front-end

> **Superseded in places.** The `SceneRenderer` trait and the
> `ArrowRoundTripRenderer` backend described below were removed in #180: with
> one concrete backend the trait abstracted nothing, and protocol `0.0.6`
> cannot round-trip a `Scene` losslessly (it has no gizmo/overlay columns), so
> the GUI round-trip was never the external-producer seam it is described as
> here. The wire path is still exercised end-to-end by `run_stream` and pinned
> by the golden suite. Sections below are kept as the design record.


Status: **in progress** (in-process interaction loop implemented; Arrow
round-trip + wasm pending) · Owner: @xiaoyxue · Branch: `feat/trd-gui-design`

## 0. Decisions (locked, 2026-07-24)

1. **Toolkit = egui** (over imgui-rs) — §3.
2. **Integration = Strategy A** (decoupled CPU-RGBA handoff) — §4. Strategy B
   (shared-surface overlay) is a *future* migration, once egui supports wgpu 30.
3. **One cross-platform `trd-gui` library**, not GUI logic spread across
   `trd-app`/`trd-wasm`. The egui UI + interaction logic is written **once**;
   thin platform-owned delivery shells live under `native/trd-gui-app` and
   `web/gui-viewer` — §7.1.

## Status

Work is split **one PR per slice** (§10). **PR #98 = the in-process viewer,
Slices 0–3**; the Arrow round-trip and wasm land as their own follow-up PRs.

- **In-process interaction loop implemented** (`crates/trd-gui`, Slices 0–3 of
  §10): a native eframe/egui window that renders `trd-core`'s headless RGBA into
  a central-panel egui texture and turns pointer/scroll gestures into an updated
  camera/model matrix that is re-rendered — the full
  "input → matrix → render → display" cycle. Modules mirror §7.2: `scene.rs`
  (orbit camera + object transform → `FrameParams`/`Draw`s), `interaction.rs`
  (`InteractionController`: events → scene, unit-tested, egui-free),
  `render_backend.rs` (`InProcRenderer` over
  `trd_core::Renderer`), `app.rs` (egui panels), `cli.rs` (`--mesh` /
  `--texture` / `--width` / `--height`, built-in default cube). Render modes
  Filled / Wireframe / **Textured** (`--texture` binds an albedo, downscaled to
  the renderer's 2048² limit). Deps: `eframe`/`egui` 0.35 (glow), `trd-core`,
  `clap`, `thiserror`, `image`. Native-only (empty `main` on wasm, like
  trd-app); the pure `scene`/`interaction` modules still compile on wasm.
- **Verification:** 19 unit tests (scene/interaction/cli/render_backend, no GPU)
  run in `nix flake check`; a GPU-gated `tests/inproc_render.rs` (`#[ignore]`,
  3 tests) renders the real backend and asserts a non-blank,
  interaction-sensitive, texture-sampling frame (run locally: MSVC on Windows,
  nixGL on Linux).
- **Next (own PRs, §10):** Slice 3's `ArrowRoundTripRenderer` (author a
  `[mesh][params]` stream → `run_stream` → image stream, enabling external
  producers) behind the same `SceneRenderer` trait, then Slice 4 (wasm:
  egui-on-canvas + `trd-core` offscreen). See issue #97 for the per-slice
  checklists.

## 1. Goal

Add an **interactive** desktop/web front-end for trd that

1. **displays** the rendered image stream, and
2. **provides interaction**: the user manipulates the scene (orbit the camera /
   translate–rotate the object / toggle overlays); each interaction is turned
   into a **new model (or camera) matrix**, a **new Arrow scene frame** is
   generated with that matrix, trd-core / trd-cli **re-renders** it, and the
   resulting image is **sent back to the GUI** for display.

The GUI toolkit is **egui** (not imgui-rs) — see §3.

The in-process interaction loop (Slices 1–3) is implemented; the Arrow
round-trip and wasm backends remain (§10). This document is the design of
record — the locked decisions and the full plan live in issue #97.

## 2. Where it fits (current architecture)

| Crate | Role | Render path |
|-------|------|-------------|
| `trd-core` | **single** wgpu rendering core (`wgpu = "30"`) | headless `Renderer::render_frame` → RGBA bytes; live `SceneRenderer::encode` → surface view |
| `trd-cli` | headless filter | Arrow scene stream **in** → Arrow image stream **out** |
| `trd-app` | native winit viewer | plays a stream (resize/close only — **no scene interaction**) |
| `trd-wasm` | browser | `canvas_renderer` (live) + `arrow_renderer` (offscreen), driven by thin TS |

Both render paths build the same value:

```
Scene = Vec<DrawableObject>          // render.rs
DrawableObject::Mesh { mesh_id, model: [f32;16], mode }   // + AabbBox / CoordinateAxes / QuadOutline / FramePlane
Draw { mesh_id, model: [f32;16], mode }                   // wire form, per frame
FrameParams { model?, k?, pose?, eye?, target?, fovy?, ... }  // camera
```

The **model matrix lives per-draw** (`Draw.model`, composed under the mesh's
preview base model). The wire protocol is mesh-first Arrow
`[mesh][texture?][frames?][params]` at `PROTOCOL_VERSION = 0.0.6`. This is exactly the
value the interaction loop needs to recompute.

`trd-gui` is a **new front-end peer** to `trd-app` — it owns UI + interaction,
and delegates *all* rendering to `trd-core`, honoring the AGENTS.md invariant
"trd-core is the single unified rendering core; front-ends are thin".

## 3. Toolkit choice: egui vs imgui-rs

**Decision: egui.** Rationale:

| Criterion | egui | imgui-rs |
|-----------|------|----------|
| **wasm support** | first-class, pure-Rust, runs on canvas (WebGPU/WebGL) | bindings to C++ Dear ImGui; wasm is awkward (needs the C++ lib via emscripten) |
| Language | 100% Rust (no C++ toolchain, matches our pure-Rust + Nix build) | FFI to C++ `cimgui`; extra build-system surface |
| wgpu integration | official `egui-wgpu` companion crate | third-party `imgui-wgpu`, less maintained |
| winit integration | official `egui-winit` | third-party `imgui-winit-support` |
| Retained/immediate | immediate mode (fits a per-frame render loop) | immediate mode |
| License | MIT/Apache-2.0 (matches repo MIT) | Apache-2.0/MIT (Dear ImGui MIT) |

egui aligns with trd's pure-Rust, native+wasm, Nix-built stack; imgui-rs would
drag a C++ toolchain into the flake and complicate the wasm target — the two
platforms we must keep at parity.

## 4. Key constraint — the wgpu version gap ⚠️

This is the decisive architectural fact.

* `trd-core` is on **wgpu 30.0** (`crates/trd-core/Cargo.toml`).
* The **latest** egui stack (egui / egui-wgpu / egui-winit / eframe **0.35.0**)
  depends on **wgpu ^29** (verified on crates.io). egui-wgpu is one major
  version behind.

**Implication:** a *single shared* `wgpu::Device`/`Queue`/`Surface` **cannot**
be handed to both egui-wgpu and trd-core, because they compile against different
`wgpu` majors — the GPU handle types are incompatible across the version
boundary. So the "egui draws chrome directly on top of trd-core's live surface"
(zero-copy overlay) integration is **blocked today**.

Two integration strategies follow from this:

### Strategy A — decoupled CPU-RGBA handoff  ✅ **chosen**

egui/eframe owns **its own** UI renderer, and trd-core renders the scene
**headless** to an RGBA buffer (wgpu 30); the GUI **uploads that buffer as an
egui texture** shown in the central panel. Only **CPU pixels** cross between the
two, so the version gap is irrelevant.

**Realized with eframe's default `glow` (OpenGL) backend** for the UI — since
only CPU RGBA is handed over, egui's renderer is fully independent of trd-core's
`wgpu 30`, and we avoid pulling a *second* wgpu (29) into the build entirely.
(eframe's optional `wgpu` backend is still resolved in `Cargo.lock` but stays
inactive.) On wasm the same holds with eframe's WebGL renderer while trd-core
uses WebGPU.

This is *exactly the loop the request describes* — "generate new arrow with
computed model matrix → render again → send the output to trd-gui" — and it
works today.

Cost: on native, egui runs on OpenGL while trd-core runs on Vulkan/wgpu (two
independent graphics contexts in one process); one GPU→CPU→GPU readback per
updated frame. Both are acceptable (we only re-render on change).

### Strategy B — shared-surface egui overlay  ⏳ future

When egui releases a wgpu-30-compatible version, trd-gui can share one device:
trd-core's `SceneRenderer::encode` paints the scene into the swapchain view, then
`egui-wgpu` paints the UI on top in the same command encoder — zero-copy, live.
Track upstream egui and migrate then. The §5 design keeps the render backend
behind a trait so this is a drop-in later.

## 5. The interaction loop (core of the request)

```
        ┌──────────────────────── trd-gui (egui) ────────────────────────┐
        │                                                                 │
  user input (pointer/wheel/keys/gizmo)                                   │
        │                                                                 │
        ▼                                                                 │
  InteractionController  ──►  new model matrix  (and/or camera FrameParams)
        │                            │                                    │
        │                            ▼                                    │
        │                  author updated Scene / Draw                    │
        │                            │                                    │
        │            ┌───────────────┴────────────────┐                  │
        │            ▼                                 ▼                  │
        │   InProcRenderer                     ArrowRoundTripRenderer     │
        │   (call trd-core directly)           (encode `[mesh][params]`   │
        │            │                          Arrow → trd-cli/run_stream│
        │            │                          → Arrow image stream)     │
        │            └───────────────┬────────────────┘                  │
        │                            ▼                                    │
        │                     RGBA image bytes                            │
        │                            │                                    │
        │                            ▼                                    │
        │              egui texture  ──►  central-panel Image  ───────────┘
        │
        └── side panel: mode (filled/wireframe/textured), aabb/axes, fps, play/pause, reset view
```

### 5.1 `InteractionController` (events → matrix)

A small state machine in `trd-gui` that owns the interaction state and maps egui
input to a transform:

* **Orbit camera** — pointer drag inside the image rect → yaw/pitch around the
  target; wheel → dolly/zoom. Updates the **camera** side of `FrameParams`
  (`eye`/`target`/`fovy`, i.e. the CG form) rather than the object.
* **Manipulate object** — drag/gizmo → a delta rotation/translation composed
  onto the selected draw's `model`: `model' = delta · model` (using the typed
  `trd_core::Transform`/`Matrix4`/`Rotation` math so affine rules hold). This is
  the "computed model matrix".

The controller is **UI-toolkit-agnostic** (takes a normalized
`InteractionEvent`, returns an updated scene state) so it is unit-testable
without egui and reusable by the wasm target.

### 5.2 `SceneRenderer` trait (two backends)

```rust
trait SceneRenderer {
    /// Render the current scene state to an RGBA image (width×height×4).
    fn render(&mut self, state: &SceneState) -> Result<ImageRgba, RenderError>;
}
```

* **`InProcRenderer`** (native default): builds `Draw`/`Scene` in memory and
  calls trd-core directly — reuse `Renderer::render_frame` (headless RGBA)
  or a live `SceneRenderer`. No serialization; lowest latency. Good for smooth
  drag.
* **`ArrowRoundTripRenderer`** (the literal request): serializes the updated
  scene state to a **new Arrow `[mesh][texture?][frames?][params]` stream**, feeds it to
  `trd_core::run_stream` (in-proc) or the `trd-cli` binary (out-of-process),
  reads the Arrow **image** stream back (`output_schema` fixed-shape tensor),
  and decodes it to RGBA. This produces output **pixel-identical to the batch
  pipeline** and lets an **external producer** (Python/ML/CV that consumes the
  event and computes the matrix — e.g. #77's normal-basis / pose estimation)
  sit in the loop. Higher latency (serialize + subprocess), so it re-renders
  on interaction *end* rather than every drag delta.

The GUI renders through `InProcRenderer` only; the Arrow round-trip backend was
removed in #180 (protocol `0.0.6` cannot round-trip a `Scene` losslessly, so it
was never the external-producer seam it was documented as).
third `LiveSurfaceRenderer` later.

### 5.3 New piece of work: a Rust **input**-scene encoder

trd-core today only *decodes* the input scene Arrow (`Mesh::from_arrow_all`,
`decode_frames`, `decode_draws`) and *encodes* the **image output**
(`OutputSession`). The input `[mesh][texture?][frames?][params]` stream is currently
authored only by the Python producers (`scripts/*_to_arrow.py`) and by test
code. `ArrowRoundTripRenderer` needs to author that input stream **in Rust**
(arrow `StreamWriter` + the 0.0.6 schema/metadata). Proposed: add a small
`trd_core::scene_encode` module (mirror of the decoders, reused by tests) so
the encoder is covered by the existing decoder-parity net. `InProcRenderer`
avoids this entirely.

## 6. Reverse channel — the interaction/event protocol

Today the Arrow protocol is **one-directional** (scene in → image out). The GUI
adds a **reverse** flow (events GUI → producer). Design:

* **Native, in-proc:** a typed `InteractionEvent` enum passed directly to the
  controller. No serialization, no protocol change.
* **Out-of-process producer (future):** define a **separate**, small event
  channel (Arrow event schema or JSON lines) so a Python/ML producer can consume
  events and emit the next scene frame. This is a *new, independent* protocol —
  **do not** fold it into the image/scene `PROTOCOL_VERSION` (0.0.6) or bump
  that version for it. Version the event channel on its own.

Recommendation: ship the in-proc enum first; standardize an Arrow event schema
only when an external producer is actually wired in.

## 7. Proposed crate layout

### 7.1 One shared library, platform-owned delivery shells

egui is immediate-mode and **platform-agnostic**: the UI and interaction logic
are plain Rust, written **once**. Only two things differ per platform, and both
are small:

* the **bootstrap** — native event loop/window (winit, via eframe) vs the wasm
  canvas runner; and
* the **render backend** — native in-proc trd-core render vs wasm offscreen
  render + async readback.

`crates/trd-gui` is the **single reusable library that targets both** native and
wasm. Platform ownership is explicit: `native/trd-gui-app` owns the native
eframe runner/CLI, while `web/gui-viewer` owns the browser bootstrap. This keeps
the shared UI, scene, interaction, and renderer integration in one Rust crate
without mixing delivery files into `crates/`. `trd-app`/`trd-wasm` remain the
thin, non-interactive player/renderer peers.

| Concern | Shared (write once) | Native only | Wasm only |
|---------|---------------------|-------------|-----------|
| egui UI (panels, widgets, image) | ✅ `ui.rs` | | |
| `InteractionController` (events → matrix) | ✅ `interaction.rs` | | |
| `SceneState` (models + camera) | ✅ `scene.rs` | | |
| RGBA handoff | ✅ `render_backend::ImageRgba` | synchronous trait | async renderer |
| Application shell | | `native/trd-gui-app/src/app.rs` | `crates/trd-gui/src/web_app.rs` |
| Bootstrap / runner | | `native/trd-gui-app/src/main.rs` | `lib.rs` wasm entry + `web/gui-viewer` |
| Render backend impl | | `InProcRenderer` / `ArrowRoundTripRenderer` | `WebRenderer` |

### 7.2 Files

Reusable library and delivery surfaces:

```
crates/trd-gui/
  Cargo.toml
  src/
    lib.rs             # shared modules + wasm-bindgen entry
    ui.rs              # shared egui panels/layout/image widget
    interaction.rs     # InteractionController: InteractionEvent → SceneState (SHARED, unit-tested, no egui)
    scene.rs           # SceneState: meshes + per-object model + camera FrameParams (SHARED)
    render_backend.rs  # shared RGBA type + native inproc/Arrow backends
    web_app.rs         # browser eframe application
    web_renderer.rs    # browser async offscreen renderer
    error.rs

native/trd-gui-app/
  Cargo.toml
  src/
    main.rs            # native eframe bootstrap
    app.rs             # native eframe application shell
    cli.rs             # native args and filesystem asset loading

web/gui-viewer/
  package.json
  index.html
  serve.ts
  src/main.ts          # thin wasm bootstrap
```

Dependencies (native): `egui`, `egui-winit`, `egui-wgpu` (its own wgpu 29),
`winit 0.30`, `trd-core` (wgpu 30), `arrow`, `clap`, `env_logger`, `log`,
`image` (decode input meshes/textures/frames in the shell, keeping trd-core I/O
free). wasm: `egui`, `eframe`(web) or `egui-wgpu` on canvas, `trd-core`,
`wasm-bindgen` — cfg-gated exactly like the existing crates.

**Invariant:** trd-gui contains **no rendering logic** — only UI, interaction,
scene authoring, and the image-display texture. All pixels come from trd-core.

## 8. egui specifics

* **Central panel**: `egui::Image` fed by an `egui::TextureHandle` that is
  updated (`set`/`load_texture`) whenever a new RGBA frame arrives from the
  `SceneRenderer`. The image rect is the interaction surface.
* **Side panel**: mode toggles (filled/wireframe/textured), overlay toggles
  (aabb / world axes / local axes — already flags in `RenderOptions` /
  `build_scene`), fps, play/pause/step for streamed playback, "reset view".
* **Input mapping**: read `response.dragged()`, `scroll_delta`,
  `hover_pos` relative to the image rect → `InteractionEvent` → controller →
  re-render. Re-render only when state changed (idle otherwise, like trd-app's
  `WaitUntil` pacing).
* **wasm**: egui on the canvas; trd-core (wasm) `arrow_renderer` renders
  offscreen → RGBA → egui texture. Enable wgpu `webgpu` + `webgl` features for
  browser fallback. Same decoupled model as native (Strategy A).

## 9. Toolchain / build integration

* Add `crates/trd-gui` and `native/trd-gui-app` to workspace `members`.
* `flake.nix`: keep the public `trd-gui` package/app output while building its
  native binary from package `trd-gui-app`; build the reusable library separately
  for wasm and stage it into `web/gui-viewer/pkg`.
* Document that, under Strategy A, **both wgpu 29 (egui) and wgpu 30 (trd-core)**
  compile into the graph. That is intentional and safe (separate devices, CPU
  handoff); revisit to unify once egui supports wgpu 30 (Strategy B).
* `git add` new files before `nix build`/`nix flake check` (sandbox sees only
  git-tracked files).
* GPU-dependent behavior stays `#[ignore]`d / run locally via nixGL on Linux and
  the MSVC path on Windows (dual-platform verification per AGENTS.md).

## 10. Phasing (vertical slices — each end-to-end verifiable, one PR per slice)

0. ✅ **Scaffold**: workspace crate + wrapped `nix build .#trd-gui` + an empty
   eframe/egui window. Groundwork for the decoupled toolkit setup. *(PR #98)*
1. ✅ **Display**: `trd-gui` native window shows a `trd-core` render as an egui
   image. Proves the Strategy-A decoupling + the wgpu-gap workaround. *(PR #98)*
2. ✅ **Camera interaction**: orbit/zoom updates the `FrameParams` camera,
   `InProcRenderer` re-renders. Proves the event → render → display loop.
   *(PR #98)*
3. **Object interaction + Arrow round-trip**: ✅ translate/rotate the mesh
   (`Draw.model`) in process, plus Filled/Wireframe/**Textured** render modes
   (`--texture`) *(PR #98)*; ✅ the `ArrowRoundTripRenderer` (`trd_core::scene_encode`
   authors the `[mesh][params]` Arrow → `run_stream` → `read_image_stream`),
   removed in #180.
   render → gui" loop, and the seam for external producers.
4. ✅ **wasm parity**: egui-on-canvas (eframe `WebRunner`) + `trd-core` offscreen
   (async wgpu 30 readback → egui texture, Strategy A). Shared `scene`/
   `interaction`/`ui` compile on both targets; `wasm_renderer` + `web_app` are the
   wasm twins of `render_backend` + the native app. Thin `web/gui-viewer/` bootstrap
   (`start(canvas)`). Compiles + clippy-clean on wasm32; browser render is a
   handoff (WebGPU browser required).
5. ⏳ **(later)** Strategy B live shared-surface overlay once egui ships wgpu 30.

## 11. Open decisions

* Default backend: `InProc` (recommended) vs Arrow round-trip.
* Whether/when to standardize the reverse **event** channel as Arrow (needed
  only for out-of-process producers).
* Whether trd-gui plays *streams* (like trd-app) in addition to single-object
  interactive editing, or focuses on interaction only.
* eframe (batteries-included) vs raw `egui-winit` + `egui-wgpu` (more control
  over the two-device setup). Lean raw for the explicit CPU-RGBA handoff.

## 12. Risks

* **wgpu version gap** (primary) — mitigated by Strategy A; removed later by B.
* GPU→CPU→GPU readback per interaction — mitigated by re-rendering only on
  change; removed by Strategy B.
* Binary size / compile time from two wgpu majors — acceptable, temporary.
* Rust input-scene encoder is new surface — kept small and parity-tested.
* Dual-platform (Windows MSVC + Linux/Nix + wasm) verification burden — follow
  the existing trd-app/trd-wasm cfg-gating pattern.

## 13. Verified facts behind this design

* `trd-core` = `wgpu = "30"`; egui/egui-wgpu/eframe **0.35.0** (latest) depend on
  `wgpu ^29`, egui-winit 0.35 on `winit ^0.30.13` (crates.io, this session).
* Render values (`Scene`, `DrawableObject`, `Draw`, `FrameParams`,
  `build_scene`, `SceneRenderer::encode`, `Renderer::render_frame`,
  `run_stream`, `RenderOptions`, `OutputSession`) — `crates/trd-core/src/{render,stream,output}.rs`.
* Protocol 0.0.6 `[mesh][texture?][frames?][params]`, model per `Draw.model` —
  `docs/protocol/0.0.6.md`, AGENTS.md.
