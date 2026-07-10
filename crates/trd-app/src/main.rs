//! trd-app: the interactive native desktop entry point for the trd core.
//!
//! It opens a window, creates a live wgpu surface from it, and draws the shared
//! [`trd_core::render_triangle`] into that surface — the desktop counterpart of
//! the headless `trd-cli` (offscreen PNG) and the browser `trd-wasm` (canvas
//! surface). All rendering logic still lives in `trd-core`.
//!
//! The windowing stack (winit) is native-only, so on wasm this crate compiles
//! to an empty `main`, keeping workspace-wide wasm builds clean.

#[cfg(not(target_arch = "wasm32"))]
mod app;

#[cfg(not(target_arch = "wasm32"))]
fn main() {
    if let Err(err) = app::run() {
        eprintln!("trd-app: {err}");
        std::process::exit(1);
    }
}

#[cfg(target_arch = "wasm32")]
fn main() {}
