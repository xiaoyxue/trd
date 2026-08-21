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
//! * [`gui`] holds the `trd-gui` entry points — the interactive viewer (`start`)
//!   and the video editor (`startVideoEditing`) — plus the browser shell they run
//!   in (`gui_web_app`).
//! * [`browser_frame`] implements [`trd_core::ExternalFrame`] over a WebCodecs
//!   `VideoFrame`, so the GPU→GPU frame copy lives on the delivery surface and
//!   the shared crates never name a browser type (#302).
//!
//! **Every** `#[wasm_bindgen]` export in the repo lives here (#180): one browser
//! delivery surface, one generated JS package. `trd-gui` is a plain rlib.

use std::fmt::Display;

use trd_core::{DisneyMaterial, EnvMapData, ImageBasedLighting, Lighting, ToneMapping};
use wasm_bindgen::prelude::*;

mod browser_frame;
mod canvas_renderer;
pub mod gui;
mod gui_web_app;
mod offscreen_renderer;

pub use browser_frame::BrowserVideoFrame;
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

    /// The staged light rig, which is a **scene** property rather than renderer
    /// state (#182): the renderers hand it to the `Scene` they build each frame
    /// instead of pushing it through a setter here.
    pub(crate) fn lighting(&self) -> Lighting {
        self.lighting
    }

    /// The staged output transform (exposure + operator). Read by
    /// [`env_background`] so the HDR sky is tone-mapped exactly like the objects
    /// drawn in front of it.
    pub(crate) fn tone_mapping(&self) -> ToneMapping {
        self.tone_mapping
    }

    /// Pushes the staged per-object material onto the render harness. Shared by
    /// the offscreen and canvas renderers — they use the same non-generic
    /// [`trd_core::Renderer`], differing only in which target they pass to their
    /// render calls (#203).
    pub(crate) fn apply(&self, renderer: &mut trd_core::Renderer) {
        renderer.set_disney_material(self.material.clone());
        renderer.set_image_based_lighting(self.ibl);
        renderer.set_tone_mapping(self.tone_mapping);
    }
}

/// The staged HDR **background sky** for both browser renderers: `blur` is what
/// JS asked for (`setEnvBackground`), while the exposure and operator follow the
/// staged PBR tone mapping, so the sky and the objects drawn in front of it
/// cannot be tone-mapped differently. `None` ⇒ no sky.
///
/// A plain [`RenderOptions`](trd_core::RenderOptions) field now (#235 R2), so
/// the browser gets its sky from the same `Scene::from_draws` assembly as every
/// other front-end instead of poking the scene's background itself.
pub(crate) fn env_background(
    blur: Option<f32>,
    pbr: Option<&PbrState>,
) -> Option<trd_core::EnvironmentBackground> {
    let tone = pbr.map(PbrState::tone_mapping).unwrap_or_default();
    blur.map(|blur| trd_core::EnvironmentBackground {
        exposure: tone.exposure,
        blur,
        tonemap: tone.operator,
    })
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
