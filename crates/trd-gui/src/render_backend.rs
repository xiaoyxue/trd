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

use trd_core::{
    encode_mesh_stream, encode_params_stream, read_image_stream, run_stream, BatchRenderer, Mesh,
    RenderOptions, Texture,
};

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

    /// Whether a render is costly enough that the UI should re-render only when
    /// an interaction *ends* (on pointer release) rather than every drag frame.
    /// The [`ArrowRoundTripRenderer`] serializes + re-runs the whole pipeline per
    /// frame, so it returns `true`; the in-process backend renders every frame.
    fn defer_expensive(&self) -> bool {
        false
    }
}

/// The render options `trd-core`'s `run_stream` applies globally, derived from
/// the interactive scene state (mode + overlay toggles). Shared by both backends
/// so they stay in agreement.
fn render_options(state: &SceneState) -> RenderOptions {
    RenderOptions {
        mode: state.mode,
        show_aabb: state.show_aabb,
        show_axes: state.show_axes,
        show_local_axes: state.show_local_axes,
    }
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
    /// centered + scaled to fit by their preview transform inside `trd-core`. An
    /// optional `texture` is bound as the albedo sampled by [`RenderMode::Textured`]
    /// meshes; when `None`, Textured mode uses `trd-core`'s 1×1 white default.
    ///
    /// [`RenderMode::Textured`]: trd_core::RenderMode::Textured
    pub fn new(
        meshes: &[Mesh],
        texture: Option<&dyn Texture>,
        width: u32,
        height: u32,
    ) -> Result<Self, GuiError> {
        let mut renderer = BatchRenderer::with_meshes(width, height, meshes)?;
        if let Some(texture) = texture {
            renderer.set_texture(texture);
        }
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

/// The **Arrow round-trip** backend (design §5.2): authors the scene as a
/// `[mesh][params]` Arrow stream (via [`encode_mesh_stream`]/
/// [`encode_params_stream`]), pipes it through `trd-core`'s [`run_stream`]
/// exactly as the headless CLI does, and decodes the resulting image stream back
/// to RGBA ([`read_image_stream`]). This produces output identical to the batch
/// pipeline and is the seam where an **external** producer (Python/ML/CV that
/// consumes the interaction and computes the next matrix) could sit.
///
/// It re-runs the whole pipeline (including GPU device setup) per render, so it
/// reports [`SceneRenderer::defer_expensive`] `= true` and the UI re-renders only
/// when an interaction ends. The leading mesh table is encoded once and cached;
/// only the tiny params stream is re-authored per frame.
pub struct ArrowRoundTripRenderer {
    /// The cached mesh-table IPC bytes (static across frames).
    mesh_stream: Vec<u8>,
    width: u32,
    height: u32,
}

impl ArrowRoundTripRenderer {
    /// Builds the backend, encoding the (static) mesh table once. Textures are
    /// not authored into the round-trip stream yet, so Textured mode renders with
    /// `trd-core`'s default albedo here.
    pub fn new(meshes: &[Mesh], width: u32, height: u32) -> Result<Self, GuiError> {
        let mesh_stream = encode_mesh_stream(meshes)?;
        Ok(Self {
            mesh_stream,
            width,
            height,
        })
    }
}

impl SceneRenderer for ArrowRoundTripRenderer {
    fn render(&mut self, state: &SceneState) -> Result<ImageRgba, GuiError> {
        let aspect = self.width as f32 / self.height.max(1) as f32;
        let params = encode_params_stream(&[state.frame_params(aspect)], Some(&[state.draws()]))?;

        let mut input = self.mesh_stream.clone();
        input.extend(params);

        let mut output = Vec::new();
        run_stream(
            std::io::Cursor::new(input),
            &mut output,
            self.width,
            self.height,
            render_options(state),
            None,
        )?;

        let rgba = read_image_stream(std::io::Cursor::new(output), self.width, self.height)?
            .pop()
            .ok_or(GuiError::NoFrame)?;
        Ok(ImageRgba {
            width: self.width,
            height: self.height,
            rgba,
        })
    }

    fn size(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    fn defer_expensive(&self) -> bool {
        true
    }
}

/// Whether any vertex carries a non-zero UV — i.e. the mesh is UV-mapped.
/// [`RenderMode::Textured`] only maps a bound texture meaningfully on such a
/// mesh; a mesh with no `vt`/`uv` data samples a single texel everywhere, which
/// is the usual cause of a "wrong looking" texture. Front-ends warn on this.
///
/// [`RenderMode::Textured`]: trd_core::RenderMode::Textured
pub fn mesh_has_uvs(mesh: &Mesh) -> bool {
    mesh.vertices.iter().any(|v| v.uv != [0.0, 0.0])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mesh_has_uvs_detects_texcoords() {
        // A plain triangle (no `vt`) has all-zero UVs.
        let plain = Mesh::from_obj("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n").expect("parses");
        assert!(!mesh_has_uvs(&plain));

        // The same triangle with `vt` texture coordinates is UV-mapped.
        let mapped =
            Mesh::from_obj("v 0 0 0\nv 1 0 0\nv 0 1 0\nvt 0 0\nvt 1 0\nvt 0 1\nf 1/1 2/2 3/3\n")
                .expect("parses");
        assert!(mesh_has_uvs(&mapped));
    }
}
