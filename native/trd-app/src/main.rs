//! trd-app: the interactive native desktop entry point for the trd core.
//!
//! It opens a window, creates a live wgpu surface from it, and plays a trd
//! frame-params stream read from stdin — the desktop counterpart of the headless
//! `trd-cli` (Arrow image stream) and the browser `trd-wasm` (canvas surface).
//! Each frame is drawn with the shared [`trd_core::SceneRenderer`], so all
//! rendering logic still lives in `trd-core`.

mod app;
mod cli;
mod error;
mod renderer;
mod stream;

fn main() {
    if let Err(err) = app::run() {
        eprintln!("trd-app: {err}");
        std::process::exit(1);
    }
}
