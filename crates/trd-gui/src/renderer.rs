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

use crate::error::GuiError;
use crate::scene::SceneState;

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
/// The appearance options for `state`: draw mode plus **every** overlay toggle,
/// so [`scene_for`] produces exactly the scene the CLI produces from the same
/// inputs. Platform-neutral: native and browser front-ends share it, which is
/// what stops their overlay handling drifting apart again (#180).
pub fn render_options(state: &SceneState) -> trd_core::RenderOptions {
    let xz = |on: bool| on.then_some(trd_core::GridPlane::Xz);
    trd_core::RenderOptions {
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

/// The full per-frame scene for `state`: the shared
/// [`Scene::from_draws`](trd_core::Scene::from_draws) assembly plus the
/// optional HDR environment background, which is a per-frame *background
/// setting* on the scene rather than a drawable or an overlay toggle (#204).
pub fn scene_for(state: &SceneState) -> trd_core::Scene {
    let mut scene = trd_core::Scene::from_draws(&state.draws(), &render_options(state), None)
        // The light rig travels with the frame now, not as sticky renderer
        // state (#182).
        .with_lighting(state.lighting);
    if state.show_environment_background {
        scene.background_mut().environment = Some(trd_core::EnvironmentBackground {
            rotation: state
                .image_based_lighting
                .first()
                .map_or(0.0, |ibl| ibl.rotation),
            exposure: state
                .tone_mappings
                .first()
                .map_or(1.0, |tone_mapping| tone_mapping.exposure),
            blur: state.environment_background_blur,
            tonemap: state
                .tone_mappings
                .first()
                .map_or(trd_core::Tonemap::Reinhard, |t| t.operator),
        });
    }
    scene
}

/// Pushes `state`'s per-object PBR material state onto the renderer.
///
/// Still setters rather than a per-frame argument: these write GPU uniform slots,
/// and threading them through `encode` is the next step (#180). Sharing the loop
/// at least means native and browser cannot disagree about it.
pub fn apply_materials(renderer: &mut trd_core::Renderer, state: &SceneState) {
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
        renderer
            .set_mesh_pbr_debug_view(i, state.pbr_debug_views.get(i).copied().unwrap_or_default());
    }
}

/// The optional PBR maps bound alongside one mesh's albedo — a named type rather
/// than a tuple so the per-mesh binding order can't be transposed by accident.
#[derive(Default, Clone, Copy)]
pub struct MaterialMaps<'a> {
    /// glTF-packed metallic-roughness (roughness in G, metallic in B).
    pub metallic_roughness: Option<&'a dyn trd_core::Texture>,
    /// Tangent-space normal map.
    pub normal: Option<&'a dyn trd_core::Texture>,
}

/// The one GUI renderer: a thin adapter over `trd-core`'s
/// [`Renderer`](trd_core::Renderer) harness that turns the interactive
/// [`SceneState`] into a frame.
///
/// **Platform-neutral.** Native (`trd-gui-app`) and browser (`web_app`) used to
/// have a renderer each — `InProcRenderer` and `WebRenderer` — with byte-identical
/// fields and near-identical bodies, only because the core harness was
/// native-only. It no longer is, so there is one type (#180). The API is `async`
/// because GPU read-back is; natively the future is already complete when the map
/// poll returns, so callers `pollster::block_on` it for free.
///
/// It keeps **no** scene state: what to draw comes from [`scene_for`] every frame,
/// and the GUI displays the output scaled to the panel, so the render resolution
/// stays fixed (no GPU device churn on window resize).
pub struct GuiRenderer {
    renderer: trd_core::Renderer,
    /// The texture target the renderer draws into and reads back from — a
    /// plain field now that the harness no longer owns its render target
    /// (#203). Fixed at construction alongside `width`/`height`, since the GUI
    /// displays the output scaled to the panel rather than resizing the render.
    ///
    /// Held as the concrete [`TextureTarget`](trd_core::TextureTarget), not the
    /// [`RenderTarget`](trd_core::RenderTarget) enum: this front-end always reads
    /// pixels back, and readback is only defined for a texture — keeping the type
    /// concrete makes that a compile-time fact rather than a runtime check.
    target: trd_core::TextureTarget,
    width: u32,
    height: u32,
}

impl GuiRenderer {
    /// Builds the renderer for `meshes` (drawn by index) at a fixed
    /// `width` × `height`; the meshes are centered + scaled to fit by their
    /// preview transform inside `trd-core`.
    ///
    /// `textures` skins each object with its **own** albedo — entry `i` binds to
    /// mesh `i` (#141) — and `material_maps` binds its optional
    /// [`MaterialMaps`] the same way; both may be shorter than
    /// `meshes`, and a `None` entry leaves `trd-core`'s 1×1 defaults in place. An
    /// optional `env` HDR probe is reflected by [`RenderMode::Shaded`] surfaces
    /// (bound once; the interactive material rides on the scene state).
    ///
    /// [`RenderMode::Shaded`]: trd_core::RenderMode::Shaded
    pub async fn new(
        meshes: &[trd_core::Mesh],
        textures: &[Option<&dyn trd_core::Texture>],
        material_maps: &[MaterialMaps<'_>],
        env: Option<trd_core::EnvMapData>,
        width: u32,
        height: u32,
    ) -> Result<Self, GuiError> {
        let (mut renderer, target) = trd_core::Renderer::with_meshes(width, height, meshes).await?;
        for (i, texture) in textures.iter().enumerate() {
            if let Some(texture) = texture {
                renderer.set_mesh_texture(i, *texture);
            }
        }
        for (i, maps) in material_maps.iter().enumerate() {
            if let Some(texture) = maps.metallic_roughness {
                renderer.set_mesh_metallic_roughness_texture(i, texture);
            }
            if let Some(texture) = maps.normal {
                renderer.set_mesh_normal_texture(i, texture);
            }
        }
        if let Some(env) = env {
            renderer.set_env_map(env);
        }
        Ok(Self {
            renderer,
            target,
            width,
            height,
        })
    }

    /// The fixed render dimensions.
    pub fn size(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    /// Renders `state` to an RGBA image.
    pub async fn render(&mut self, state: &SceneState) -> Result<ImageRgba, GuiError> {
        apply_materials(&mut self.renderer, state);
        let scene = scene_for(state);
        let layers = [trd_core::SceneLayer::new(
            state.camera(self.viewport()),
            &scene,
        )];
        let rgba = self.renderer.render_layers(&layers, &self.target).await?;
        Ok(ImageRgba {
            width: self.width,
            height: self.height,
            rgba,
        })
    }

    /// Resolves the object under render-target pixel `(x, y)` via the id-color
    /// picking pass (#141), returning its 0-based index into `state.draws()`, or
    /// `None` for the background.
    pub async fn pick(&mut self, state: &SceneState, x: u32, y: u32) -> Option<u32> {
        let camera = state.camera(self.viewport());
        self.renderer
            .pick(camera, &state.draws(), x, y, self.viewport())
            .await
    }

    fn viewport(&self) -> trd_core::Viewport {
        trd_core::Viewport {
            width: self.width,
            height: self.height,
        }
    }
}
/// Reports whether `mesh` carries real UV coordinates — a mesh without them
/// samples a single texel in Textured mode, so front-ends warn instead of
/// rendering a mysteriously flat surface.
pub fn mesh_has_uvs(mesh: &trd_core::Mesh) -> bool {
    mesh.vertices.iter().any(|v| v.uv != [0.0, 0.0])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mesh_has_uvs_detects_texcoords() {
        // A plain triangle (no `vt`) has all-zero UVs.
        let plain =
            trd_core::Mesh::from_obj("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n").expect("parses");
        assert!(!mesh_has_uvs(&plain));

        // The same triangle with `vt` texture coordinates is UV-mapped.
        let mapped = trd_core::Mesh::from_obj(
            "v 0 0 0\nv 1 0 0\nv 0 1 0\nvt 0 0\nvt 1 0\nvt 0 1\nf 1/1 2/2 3/3\n",
        )
        .expect("parses");
        assert!(mesh_has_uvs(&mapped));
    }

    /// The environment background is a **shared** scene setting, not a
    /// browser-only one.
    ///
    /// It used to be pushed by `WebRenderer` alone, so the side panel's
    /// "Environment background" checkbox silently did nothing in the native
    /// window. Collapsing both renderers onto one `scene_for` fixed that; this
    /// pins it so the two can't drift apart again (#180).
    #[test]
    fn scene_for_sets_the_environment_background_when_enabled() {
        let off = SceneState::default();
        assert!(!off.show_environment_background);
        assert_eq!(scene_for(&off).background().environment, None);

        let on = SceneState {
            show_environment_background: true,
            ..SceneState::default()
        };
        assert!(
            scene_for(&on).background().environment.is_some(),
            "the environment background toggle must reach the scene on every platform"
        );
    }

    /// `render_options` must forward **every** overlay toggle, so the one
    /// `Scene::from_draws` assembly produces what the panel asked for.
    #[test]
    fn render_options_forward_the_overlay_toggles() {
        let state = SceneState {
            show_aabb: true,
            show_axes: true,
            show_local_axes: true,
            show_world_grid: true,
            show_local_grid: true,
            selected: Some(0),
            ..SceneState::default()
        };
        let options = render_options(&state);

        assert!(options.show_aabb);
        assert!(options.show_axes);
        assert!(options.show_local_axes);
        assert_eq!(options.show_world_grid, Some(trd_core::GridPlane::Xz));
        assert_eq!(options.show_object_grid, Some(trd_core::GridPlane::Xz));
        assert_eq!(options.selected, Some(0));
    }
}
