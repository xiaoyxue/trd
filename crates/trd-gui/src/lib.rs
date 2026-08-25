//! trd-gui: the interactive egui front-end (#97).
//!
//! A thin front-end peer to `trd-app`: it owns the UI, the interaction loop and
//! scene authoring, and delegates **all** rendering to `trd-core`. Everything
//! except the delivery surfaces is platform-agnostic (#180); the pipeline and
//! the Strategy-A CPU-RGBA handoff are in `docs/gui-design.md`.

pub mod assets;
pub mod error;
pub mod fonts;
pub mod interaction;
pub mod model;
mod platform;
pub mod renderer;
pub mod scene;
pub mod ui;
pub mod video_editing;
pub mod video_editing_renderer;
