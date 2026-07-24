//! [`SceneRenderer`] — the trait the GUI renders scenes through, plus the native
//! in-process backend (#97).
//!
//! trd-gui owns **no rendering logic**: it hands a [`SceneState`] to a
//! `SceneRenderer` and displays the RGBA pixels that come back. This realizes
//! Strategy A (the decoupled CPU-RGBA handoff): the egui UI is drawn by eframe's
//! own renderer while `trd-core` renders the scene **headless** to an RGBA
//! buffer, so the two toolkits stay independent of `trd-core`'s `wgpu 30`.
//!
//! [`InProcRenderer`] is the native default — it calls `trd-core`'s
//! [`BatchRenderer`] directly (no serialization, lowest latency). The
//! `ArrowRoundTripRenderer` (author a `[mesh][params]` stream → `run_stream` →
//! image stream) and the wasm offscreen backend are the design's later slices;
//! both slot behind this same trait.

use trd_core::{BatchRenderer, Mesh};

use crate::error::GuiError;
use crate::scene::SceneState;

/// A rendered frame: tightly packed row-major RGBA (`width * height * 4` bytes).
#[derive(Debug, Clone)]
pub struct ImageRgba {
    pub width: u32,
    pub height: u32,
    pub rgba: Vec<u8>,
}

/// Renders a [`SceneState`] to an RGBA image. Implementors own the render target
/// and its dimensions; the GUI only supplies scene state and displays the result.
pub trait SceneRenderer {
    /// Renders the current scene state to an RGBA image.
    fn render(&mut self, state: &SceneState) -> Result<ImageRgba, GuiError>;

    /// The fixed pixel dimensions this backend renders at (`width`, `height`).
    fn size(&self) -> (u32, u32);
}

/// The native in-process backend: builds a `trd-core` [`BatchRenderer`] once at a
/// fixed resolution and re-renders the scene on demand. The GUI displays the
/// output scaled to the panel, so the render resolution is stable (no GPU
/// device churn on window resize) and the interaction maps to the fixed image
/// rect.
pub struct InProcRenderer {
    renderer: BatchRenderer,
    width: u32,
    height: u32,
}

impl InProcRenderer {
    /// Builds the backend for the given meshes (drawn by index; the interactive
    /// scene draws mesh `0`) at a fixed `width` × `height`. The meshes are
    /// centered + scaled to fit by their preview transform inside `trd-core`.
    pub fn new(meshes: &[Mesh], width: u32, height: u32) -> Result<Self, GuiError> {
        let renderer = BatchRenderer::with_meshes(width, height, meshes)?;
        Ok(Self {
            renderer,
            width,
            height,
        })
    }
}

impl SceneRenderer for InProcRenderer {
    fn render(&mut self, state: &SceneState) -> Result<ImageRgba, GuiError> {
        let aspect = self.width as f32 / self.height.max(1) as f32;
        self.renderer.set_mode(state.mode);
        self.renderer.set_show_aabb(state.show_aabb);
        self.renderer.set_show_axes(state.show_axes);
        self.renderer.set_show_local_axes(state.show_local_axes);
        let rgba = self
            .renderer
            .render_frame(state.frame_params(aspect), &state.draws(), None)?;
        Ok(ImageRgba {
            width: self.width,
            height: self.height,
            rgba,
        })
    }

    fn size(&self) -> (u32, u32) {
        (self.width, self.height)
    }
}
