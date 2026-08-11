#![cfg(target_arch = "wasm32")]

//! trd browser (wasm) bindings. The crate root only wires the two
//! `#[wasm_bindgen]` renderers and the small glue they share; each renderer
//! lives in its own module:
//!
//! * [`CanvasRenderer`] (`canvas_renderer`) renders an Arrow stream straight to
//!   an on-screen `<canvas>` surface — the browser twin of `trd-cli`'s live
//!   view (`render.sh --web --canvas-renderer`).
//! * [`OffscreenRenderer`] (`offscreen_renderer`) renders each frame to an
//!   **offscreen** texture and returns the pixels as an Arrow output stream — the
//!   browser twin of headless `trd-cli` (`render.sh --web --offscreen-renderer`).

use std::fmt::Display;

use trd_core::{DisneyMaterial, EnvMapData, ImageBasedLighting, Lighting, ToneMapping};
use wasm_bindgen::prelude::*;

mod canvas_renderer;
mod offscreen_renderer;

pub use canvas_renderer::CanvasRenderer;
pub use offscreen_renderer::OffscreenRenderer;

#[derive(Debug, Clone)]
pub(crate) struct PbrState {
    material: DisneyMaterial,
    lighting: Lighting,
    ibl: ImageBasedLighting,
    tone_mapping: ToneMapping,
}

impl PbrState {
    pub(crate) fn new(
        material: DisneyMaterial,
        lighting: Lighting,
        ibl: ImageBasedLighting,
        tone_mapping: ToneMapping,
    ) -> Self {
        Self {
            material,
            lighting,
            ibl,
            tone_mapping,
        }
    }

    /// Pushes the staged material onto the render harness. Generic over the
    /// render target so the offscreen and canvas surfaces share it — they are the
    /// same `Renderer` now, differing only in where the frame lands (#180).
    pub(crate) fn apply<T: trd_core::RenderTarget>(&self, renderer: &mut trd_core::Renderer<T>) {
        renderer.set_disney_material(self.material.clone());
        renderer.set_lighting(self.lighting);
        renderer.set_image_based_lighting(self.ibl);
        renderer.set_tone_mapping(self.tone_mapping);
    }
}

/// Wraps any `Display` message as a JS `Error` (for a rejected `Promise` or a
/// thrown exception). Shared by both renderers.
pub(crate) fn js_error(message: impl Display) -> JsValue {
    js_sys::Error::new(&message.to_string()).into()
}

/// Decodes an equirectangular Radiance `.hdr` byte buffer into a linear-RGBA f32
/// [`EnvMapData`], downscaled (integer box filter) to the renderer's portable
/// 2048px texture limit. The browser twin of `trd-cli`'s `load_env_map`: the
/// wasm shell decodes the `.hdr` (trd-core does no file/codec I/O), so the
/// Disney PBR environment probe is byte-identical to the native path. Shared by
/// both renderers' `set_env_map_hdr`.
pub(crate) fn decode_env_hdr(bytes: &[u8]) -> Result<EnvMapData, String> {
    let img = image::load_from_memory_with_format(bytes, image::ImageFormat::Hdr)
        .map_err(|error| format!("decode env HDR: {error}"))?
        .to_rgba32f();
    let (width, height) = img.dimensions();
    Ok(EnvMapData::from_rgba32f(
        width,
        height,
        img.into_raw(),
        2048,
    ))
}
