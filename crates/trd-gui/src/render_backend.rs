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

/// A rendered frame: tightly packed row-major RGBA (`width * height * 4` bytes).
/// Shared by the native backends and the wasm offscreen renderer.
#[derive(Debug, Clone)]
pub struct ImageRgba {
    pub width: u32,
    pub height: u32,
    pub rgba: Vec<u8>,
}

// The native `BatchRenderer`-based backends (in-process + Arrow round-trip) and
// the `SceneRenderer` trait are native-only; on wasm the offscreen renderer is
// `crate::web_renderer` (async) instead, so only `ImageRgba` above is shared.
#[cfg(not(target_arch = "wasm32"))]
pub use native::*;

#[cfg(not(target_arch = "wasm32"))]
mod native {
    use super::ImageRgba;
    use trd_core::{
        decode_params_stream, encode_params_stream, read_image_stream, BatchRenderer, EnvMapData,
        Mesh, OutputSession, RenderOptions, Texture,
    };

    use crate::error::GuiError;
    use crate::scene::SceneState;

    /// Renders a [`SceneState`] to an RGBA image. Implementors own the render target
    /// and its dimensions; the GUI only supplies scene state and displays the result.
    pub trait SceneRenderer {
        /// Renders the current scene state to an RGBA image.
        fn render(&mut self, state: &SceneState) -> Result<ImageRgba, GuiError>;

        /// The fixed pixel dimensions this backend renders at (`width`, `height`).
        fn size(&self) -> (u32, u32);

        /// Resolves the object under render-target pixel `(x, y)` via the id-color
        /// picking pass (#141), returning its 0-based index into
        /// [`SceneState::draws`](crate::scene::SceneState::draws), or `None` for
        /// the background. Used by click-to-select.
        fn pick(&mut self, state: &SceneState, x: u32, y: u32) -> Option<u32>;

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
            show_aabb: state.aabb_visible(),
            show_axes: state.show_axes,
            show_local_axes: state.show_local_axes,
            show_local_grid: None,
            show_local_grid_mesh: None,
            pbr: None,
            msaa: trd_core::Msaa::X4,
        }
    }

    /// Maps the scene's XZ grid overlay toggles to the `GridPlane` the
    /// `BatchRenderer` grid setters expect (`Some(Xz)` when on, `None` when off).
    /// Shared by both native backends so the grids appear identically.
    fn apply_grid_overlays(renderer: &mut BatchRenderer, state: &SceneState) {
        let xz = |on: bool| on.then_some(trd_core::GridPlane::Xz);
        renderer.set_show_world_grid(xz(state.show_world_grid));
        renderer.set_show_object_grid(xz(state.show_local_grid));
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
        /// meshes; when `None`, Textured mode uses `trd-core`'s 1×1 white default. An
        /// optional `env` HDR probe is reflected by [`RenderMode::Pbr`] metallic
        /// surfaces (bound once; the interactive material rides on the scene state).
        ///
        /// [`RenderMode::Textured`]: trd_core::RenderMode::Textured
        /// [`RenderMode::Pbr`]: trd_core::RenderMode::Pbr
        pub fn new(
            meshes: &[Mesh],
            texture: Option<&dyn Texture>,
            env: Option<EnvMapData>,
            width: u32,
            height: u32,
        ) -> Result<Self, GuiError> {
            let mut renderer = BatchRenderer::with_meshes(width, height, meshes)?;
            if let Some(texture) = texture {
                renderer.set_texture(texture);
            }
            if let Some(env) = env {
                renderer.set_env_map(env);
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
            self.renderer.set_pbr_material(state.pbr);
            self.renderer.set_show_aabb(state.aabb_visible());
            self.renderer.set_show_axes(state.show_axes);
            self.renderer.set_show_local_axes(state.show_local_axes);
            apply_grid_overlays(&mut self.renderer, state);
            let rgba =
                self.renderer
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

        fn pick(&mut self, state: &SceneState, x: u32, y: u32) -> Option<u32> {
            let aspect = self.width as f32 / self.height.max(1) as f32;
            self.renderer
                .pick(state.frame_params(aspect), &state.draws(), x, y)
        }
    }

    /// The **Arrow round-trip** backend (design §5.2): the seam where an **external**
    /// producer (Python/ML/CV that consumes the interaction and computes the next
    /// matrix) could sit. Each render serializes the per-frame **params** (the
    /// computed camera/model matrix + draw list) to a `0.0.5` Arrow stream, decodes
    /// it back through the real wire decoders ([`decode_params_stream`]), renders it,
    /// then serializes the resulting image to an Arrow stream ([`OutputSession`]) and
    /// decodes it back to RGBA ([`read_image_stream`]) — the exact round-trip an
    /// out-of-process producer would drive.
    ///
    /// Unlike the headless CLI's `run_stream` (which rebuilds the GPU device on every
    /// call), this holds a **persistent** [`BatchRenderer`] built once from the
    /// static mesh/texture (decode-once, matching the real protocol where the mesh is
    /// uploaded once and only the per-frame params cross the wire). That keeps the
    /// full serialize→render→serialize round-trip while rendering at interactive
    /// speed, so it no longer needs to defer to interaction end.
    pub struct ArrowRoundTripRenderer {
        /// The persistent renderer (device + uploaded mesh/texture built once).
        renderer: BatchRenderer,
        width: u32,
        height: u32,
    }

    impl ArrowRoundTripRenderer {
        /// Builds the backend, creating the GPU renderer once from the static meshes
        /// (and optional bound `texture` / `env` probe). When a `texture` is bound,
        /// [`RenderMode::Textured`] samples it; an `env` HDR probe is reflected by
        /// [`RenderMode::Pbr`] metallic surfaces (matching the in-process backend).
        ///
        /// [`RenderMode::Textured`]: trd_core::RenderMode::Textured
        /// [`RenderMode::Pbr`]: trd_core::RenderMode::Pbr
        pub fn new(
            meshes: &[Mesh],
            texture: Option<&dyn Texture>,
            env: Option<EnvMapData>,
            width: u32,
            height: u32,
        ) -> Result<Self, GuiError> {
            let mut renderer = BatchRenderer::with_meshes(width, height, meshes)?;
            if let Some(texture) = texture {
                renderer.set_texture(texture);
            }
            if let Some(env) = env {
                renderer.set_env_map(env);
            }
            Ok(Self {
                renderer,
                width,
                height,
            })
        }
    }

    impl SceneRenderer for ArrowRoundTripRenderer {
        fn render(&mut self, state: &SceneState) -> Result<ImageRgba, GuiError> {
            let aspect = self.width as f32 / self.height.max(1) as f32;

            // 1. Serialize the per-frame params (the computed matrix) to the wire...
            let params_bytes =
                encode_params_stream(&[state.frame_params(aspect)], Some(&[state.draws()]))?;
            // 2. ...and decode it back through the real wire decoder.
            let frame = decode_params_stream(&params_bytes)?
                .into_iter()
                .next()
                .ok_or(GuiError::NoFrame)?;

            // 3. Render on the persistent device.
            let opts = render_options(state);
            self.renderer.set_mode(opts.mode);
            self.renderer.set_pbr_material(state.pbr);
            self.renderer.set_show_aabb(opts.show_aabb);
            self.renderer.set_show_axes(opts.show_axes);
            self.renderer.set_show_local_axes(opts.show_local_axes);
            apply_grid_overlays(&mut self.renderer, state);
            let rgba = self
                .renderer
                .render_frame(frame.params, &frame.resolved_draws(), None)?;

            // 4. Serialize the rendered image to an Arrow stream and decode it back —
            // the output half of the round-trip an external consumer would read.
            let mut session = OutputSession::new(self.width, self.height)?;
            let mut bytes = session.drain_new()?;
            session.write_rgba_batch(&[rgba])?;
            bytes.extend(session.drain_new()?);
            session.finish()?;
            bytes.extend(session.drain_new()?);

            let rgba = read_image_stream(std::io::Cursor::new(bytes), self.width, self.height)?
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

        fn pick(&mut self, state: &SceneState, x: u32, y: u32) -> Option<u32> {
            let aspect = self.width as f32 / self.height.max(1) as f32;
            self.renderer
                .pick(state.frame_params(aspect), &state.draws(), x, y)
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
            let mapped = Mesh::from_obj(
                "v 0 0 0\nv 1 0 0\nv 0 1 0\nvt 0 0\nvt 1 0\nvt 0 1\nf 1/1 2/2 3/3\n",
            )
            .expect("parses");
            assert!(mesh_has_uvs(&mapped));
        }
    }
}
