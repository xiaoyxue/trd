//! The native in-process render backend (#97).
//!
//! trd-gui owns **no rendering logic**: it hands a [`SceneState`]-derived scene
//! to `trd-core`'s [`Renderer`](trd_core::Renderer) and displays the RGBA pixels
//! that come back. This realizes Strategy A (the decoupled CPU-RGBA handoff):
//! eframe draws the egui UI with its own renderer while `trd-core` renders the
//! scene **headless** to an RGBA buffer, so the two toolkits stay independent of
//! `trd-core`'s `wgpu 30`.
//!
//! There is no backend *trait* any more: with the Arrow round-trip gone there is
//! exactly one way the GUI renders, so the abstraction had a single implementor
//! and abstracted nothing (#180).

/// A rendered frame: tightly packed row-major RGBA (`width * height * 4` bytes).
/// Shared by the native backends and the wasm offscreen renderer.
#[derive(Debug, Clone)]
pub struct ImageRgba {
    pub width: u32,
    pub height: u32,
    pub rgba: Vec<u8>,
}

// The native `Renderer`-based backends (in-process + Arrow round-trip) and
// The native backend is native-only; on wasm the offscreen renderer is
// `crate::web_renderer` (async) instead, so only `ImageRgba` above is shared.
#[cfg(not(target_arch = "wasm32"))]
pub use native::*;

#[cfg(not(target_arch = "wasm32"))]
mod native {
    use super::ImageRgba;
    use trd_core::{EnvMapData, Mesh, RenderOptions, Renderer, Texture};

    use crate::error::GuiError;
    use crate::scene::SceneState;

    /// The render options `trd-core`'s `run_stream` applies globally, derived from
    /// the interactive scene state (mode + overlay toggles). Shared by both backends
    /// so they stay in agreement.
    /// The appearance options for `state`: draw mode plus **every** overlay
    /// toggle, so `scene_with_overlays` produces exactly the scene the CLI and
    /// the browser produce from the same inputs. The renderer keeps none of this
    /// (#180).
    fn render_options(state: &SceneState) -> RenderOptions {
        let xz = |on: bool| on.then_some(trd_core::GridPlane::Xz);
        RenderOptions {
            mode: trd_core::RenderMode::Filled, // per-draw Some(mode) overrides; this is only a fallback
            show_aabb: state.show_aabb,
            show_axes: state.show_axes,
            show_local_axes: state.show_local_axes,
            show_local_grid: None,
            show_local_grid_mesh: None,
            show_world_grid: xz(state.show_world_grid),
            show_object_grid: xz(state.show_local_grid),
            selected: state.selected,
            pbr: None,
            msaa: trd_core::Msaa::X4,
        }
    }

    fn apply_pbr(renderer: &mut Renderer, state: &SceneState) {
        renderer.set_lighting(state.lighting);
        for (i, ((material, ibl), tone_mapping)) in state
            .materials
            .iter()
            .zip(&state.image_based_lighting)
            .zip(&state.tone_mappings)
            .enumerate()
        {
            renderer.set_mesh_disney_material(i, material.clone());
            renderer.set_mesh_image_based_lighting(i, *ibl);
            renderer.set_mesh_tone_mapping(i, *tone_mapping);
            renderer.set_mesh_pbr_debug_view(
                i,
                state.pbr_debug_views.get(i).copied().unwrap_or_default(),
            );
        }
    }

    /// The native in-process backend: builds a `trd-core` [`Renderer`] once at a
    /// fixed resolution and re-renders the scene on demand. The GUI displays the
    /// output scaled to the panel, so the render resolution is stable (no GPU
    /// device churn on window resize) and the interaction maps to the fixed image
    /// rect.
    pub struct InProcRenderer {
        renderer: Renderer,
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
            let mut renderer = pollster::block_on(Renderer::with_meshes(width, height, meshes))?;
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

    impl InProcRenderer {
        pub fn render(&mut self, state: &SceneState) -> Result<ImageRgba, GuiError> {
            let aspect = self.width as f32 / self.height.max(1) as f32;
            apply_pbr(&mut self.renderer, state);
            let scene = trd_core::scene_with_overlays(&state.draws(), &render_options(state), None);
            // The native GUI drives rendering from a synchronous eframe frame
            // callback, while the renderer is async because GPU read-back is.
            // Blocking is free natively — the future is already complete when the
            // map poll returns.
            let rgba = pollster::block_on(
                self.renderer
                    .render_scene(state.frame_params(aspect), &scene),
            )?;
            Ok(ImageRgba {
                width: self.width,
                height: self.height,
                rgba,
            })
        }

        pub fn size(&self) -> (u32, u32) {
            (self.width, self.height)
        }

        pub fn pick(&mut self, state: &SceneState, x: u32, y: u32) -> Option<u32> {
            let aspect = self.width as f32 / self.height.max(1) as f32;
            pollster::block_on(
                self.renderer
                    .pick(state.frame_params(aspect), &state.draws(), x, y),
            )
        }
    }

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
