#![cfg(target_arch = "wasm32")]

//! trd browser (wasm) bindings. The crate root only wires the two
//! `#[wasm_bindgen]` renderers and the small glue they share; each renderer
//! lives in its own module:
//!
//! * [`CanvasRenderer`] (`canvas_renderer`) renders an Arrow stream straight to
//!   an on-screen `<canvas>` surface — the browser twin of `trd-cli`'s live
//!   view (`render.sh --web --canvas-renderer`).
//! * [`ArrowRenderer`] (`arrow_renderer`) renders each frame to an **offscreen**
//!   texture and returns the pixels as an Arrow output stream — the browser twin
//!   of headless `trd-cli` (`render.sh --web --offscreen-renderer`).

use std::fmt::Display;

use wasm_bindgen::prelude::*;

mod arrow_renderer;
mod canvas_renderer;

pub use arrow_renderer::ArrowRenderer;
pub use canvas_renderer::CanvasRenderer;

/// Wraps any `Display` message as a JS `Error` (for a rejected `Promise` or a
/// thrown exception). Shared by both renderers.
pub(crate) fn js_error(message: impl Display) -> JsValue {
    js_sys::Error::new(&message.to_string()).into()
}
