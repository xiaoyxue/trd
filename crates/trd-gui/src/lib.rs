//! trd-gui: the interactive egui front-end (issue #97).
//!
//! trd-gui is a thin **front-end peer** to `trd-app`: it owns the UI, the
//! interaction loop, and scene authoring, and delegates **all** rendering to
//! `trd-core`, honoring the repo invariant that `trd-core` is the single unified
//! rendering core. The loop is:
//!
//! ```text
//! pointer / wheel gesture
//!   → InteractionController (events → camera / model matrix)   [interaction.rs]
//!   → SceneState (orbit camera + object transform)             [scene.rs]
//!   → GuiRenderer::render (trd-core headless RGBA)              [renderer.rs]
//!   → egui texture in the delivery surface
//! ```
//!
//! It follows **Strategy A** (the decoupled CPU-RGBA handoff): eframe draws the
//! egui UI while `trd-core` renders the scene to an RGBA buffer, so the GUI
//! toolkit stays independent of `trd-core`'s `wgpu 30`. See `docs/gui-design.md`
//! and issue #97 for the full design.
//!
//! ## Module layout
//!
//! `scene`/`interaction`/`ui`/`assets`/`error`/`renderer` are all
//! **platform-agnostic** (the scene + controller are unit-tested without egui or
//! a GPU; `ui` is the shared egui layout). There is **one** renderer,
//! [`GuiRenderer`](renderer::GuiRenderer): native `trd-gui-app` and the browser
//! `web_app` (started via [`start`]) drive the same type, the former blocking on
//! its async API. Only the delivery surfaces are target-split (#180).

pub mod assets;
pub mod error;
pub mod interaction;
pub mod renderer;
pub mod scene;
pub mod ui;
pub mod video_editing;
pub mod video_editing_renderer;
