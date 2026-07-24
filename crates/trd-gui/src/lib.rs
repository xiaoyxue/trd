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
//!   → SceneRenderer::render (trd-core headless RGBA)           [render_backend.rs]
//!   → egui texture in the central panel                        [app.rs]
//! ```
//!
//! It follows **Strategy A** (the decoupled CPU-RGBA handoff): eframe draws the
//! egui UI while `trd-core` renders the scene headless to an RGBA buffer, so the
//! GUI toolkit stays independent of `trd-core`'s `wgpu 30`. See `docs/gui-design.md`
//! and issue #97 for the full design and the remaining slices (the Arrow
//! round-trip backend and the wasm target).
//!
//! ## Module layout
//!
//! The scene model and the interaction controller are **platform-agnostic** and
//! unit-tested without egui or a GPU; the render backend, the egui app, and the
//! CLI are native-only (the browser bootstrap + offscreen backend land later).

pub mod error;
pub mod interaction;
pub mod scene;

#[cfg(not(target_arch = "wasm32"))]
pub mod app;
#[cfg(not(target_arch = "wasm32"))]
pub mod cli;
#[cfg(not(target_arch = "wasm32"))]
pub mod render_backend;
