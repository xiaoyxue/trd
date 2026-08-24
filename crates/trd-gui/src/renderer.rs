//! The in-process render backend (#97).
//!
//! trd-gui owns **no rendering logic**: it hands a [`SceneState`]-derived scene
//! to `trd-core`'s [`Renderer`](trd_core::Renderer) and displays the RGBA pixels
//! that come back. There is no backend *trait*: with the Arrow round-trip gone
//! the abstraction had a single implementor (#180).

use crate::error::GuiError;
use crate::scene::SceneState;

/// A rendered frame: tightly packed row-major RGBA (`width * height * 4` bytes).
#[derive(Debug, Clone)]
pub struct ImageRgba {
    pub width: u32,
    pub height: u32,
    pub rgba: Vec<u8>,
}

/// The appearance options for `state`: draw mode plus **every** overlay toggle,
/// so [`scene_for`] produces exactly the scene the CLI produces from the same
/// inputs. Platform-neutral, which is what stops native and browser overlay
/// handling drifting apart again (#180).
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
        // No `rotation`: the probe yaw is a scene-level `EnvironmentLight`, so
        // the sky and the reflections in front of it cannot disagree (#182). The
        // exposure and operator are the background's own, not mesh 0's (#235 S6).
        env_background: state.show_environment_background.then_some(
            trd_core::EnvironmentBackground {
                exposure: state.environment_background_tone_mapping.exposure,
                blur: state.environment_background_blur,
                tonemap: state.environment_background_tone_mapping.operator,
            },
        ),
        msaa: trd_core::Msaa::X4,
    }
}

/// The full per-frame scene for `state`: the shared
/// [`Scene::from_draws`](trd_core::Scene::from_draws) assembly — including the
/// optional HDR environment background, which is a per-frame *background
/// setting* on the scene rather than a drawable or an overlay toggle (#204) —
/// plus the frame's light rig.
pub fn scene_for(state: &SceneState) -> trd_core::Scene {
    trd_core::Scene::from_draws(&state.draws(), &render_options(state), None)
        // The light rig travels with the frame now, not as sticky renderer
        // state (#182).
        .with_lighting(state.lighting)
}

/// Pushes `state`'s per-object PBR material state onto the renderer.
pub fn apply_materials(renderer: &mut trd_core::Renderer, state: &SceneState) {
    for (i, ((material, ibl), tone_mapping)) in state
        .materials
        .iter()
        .zip(&state.image_based_lighting)
        .zip(&state.tone_mappings)
        .enumerate()
    {
        renderer.set_appearance(
            trd_core::MeshTarget::One(i),
            trd_core::MeshAppearance {
                material: material.clone(),
                ibl: *ibl,
                tone_mapping: *tone_mapping,
                debug_view: state.pbr_debug_views.get(i).copied().unwrap_or_default(),
            },
        );
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
/// [`Renderer`](trd_core::Renderer) harness that turns a [`SceneState`] into a
/// frame.
///
/// **Platform-neutral** (#180). The API is `async` because GPU read-back is;
/// natively the future is already complete when the map poll returns, so callers
/// `pollster::block_on` it for free.
///
/// It keeps **no** scene state: [`scene_for`] supplies each frame, at a fixed
/// render resolution scaled to the panel.
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
    /// Whether an HDR probe is bound — see [`has_env`](Self::has_env).
    has_env: bool,
}

impl GuiRenderer {
    /// Builds the renderer for `meshes` (drawn by index) at a fixed
    /// `width` × `height`; meshes are centered + scaled by their preview
    /// transform inside `trd-core`.
    ///
    /// `textures` and `material_maps` bind per object — entry `i` to mesh `i`
    /// (#141); both may be shorter than `meshes`, and a `None` entry keeps
    /// `trd-core`'s 1×1 defaults. Optional `env` HDR is reflected by
    /// [`RenderMode::Shaded`](trd_core::RenderMode::Shaded) surfaces.
    pub async fn new(
        meshes: &[trd_core::Mesh],
        textures: &[Option<&dyn trd_core::Texture>],
        material_maps: &[MaterialMaps<'_>],
        env: Option<trd_core::EnvMapData>,
        width: u32,
        height: u32,
    ) -> Result<Self, GuiError> {
        let (mut renderer, target) = trd_core::Renderer::with_meshes(width, height, meshes).await?;
        let had_env = env.is_some();
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
            has_env: had_env,
        })
    }

    /// The fixed render dimensions.
    pub fn size(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    /// Whether an HDR probe is bound — i.e. whether IBL can light a surface.
    ///
    /// A model loaded at runtime is lit **only** by the probe (#353), so a
    /// caller must bind one through [`set_env`](Self::set_env) first; without it
    /// an IBL-only rig renders black.
    pub fn has_env(&self) -> bool {
        self.has_env
    }

    /// Binds `env` as the HDR probe, replacing any previous one.
    pub fn set_env(&mut self, env: trd_core::EnvMapData) {
        self.renderer.set_env_map(env);
        self.has_env = true;
    }

    /// Uploads `asset`'s mesh as a **new** object and binds its imported material
    /// and glTF maps, returning the new mesh id (#353).
    ///
    /// The id is also the object's row in [`SceneState`](crate::scene::SceneState)'s
    /// parallel vectors, because [`draws`](crate::scene::SceneState::draws) maps
    /// row `i` to `mesh_id: i` — so the caller must register exactly one object
    /// per call, in the same order.
    pub fn add_model(&mut self, asset: &trd_core::GltfAsset) -> usize {
        let mesh_id = self.renderer.add_mesh(&asset.mesh);
        if let Some(texture) = asset.base_color_texture.as_ref() {
            self.renderer.set_mesh_texture(mesh_id, texture);
        }
        if let Some(texture) = asset.metallic_roughness_texture.as_ref() {
            self.renderer
                .set_mesh_metallic_roughness_texture(mesh_id, texture);
        }
        if let Some(texture) = asset.normal_texture.as_ref() {
            self.renderer.set_mesh_normal_texture(mesh_id, texture);
        }
        self.renderer
            .set_disney_material(trd_core::MeshTarget::One(mesh_id), asset.material.clone());
        mesh_id
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

    /// The sky is graded by **its own** tone mapping, not by mesh 0's (#235 S6).
    ///
    /// It used to read `tone_mappings.first()`, so editing the *first* object's
    /// exposure silently re-graded the background while editing any other
    /// object's did nothing to it — "which object's exposure does the sky
    /// follow?" answered by "index 0", the defect #182/P9 removed for the probe
    /// yaw. Per-object tone mapping stays a feature; the two are simply
    /// independent now.
    #[test]
    fn the_sky_is_graded_by_its_own_tone_mapping_not_mesh_zero() {
        let state = SceneState {
            show_environment_background: true,
            environment_background_tone_mapping: trd_core::ToneMapping {
                exposure: 0.25,
                operator: trd_core::Tonemap::Aces,
            },
            // Mesh 0 is deliberately graded differently; the sky must ignore it.
            tone_mappings: vec![trd_core::ToneMapping {
                exposure: 3.5,
                operator: trd_core::Tonemap::Reinhard,
            }],
            ..SceneState::default()
        };

        let sky = scene_for(&state)
            .background()
            .environment
            .expect("the background is enabled");
        assert_eq!(sky.exposure, 0.25, "the sky keeps its own exposure");
        assert_eq!(sky.tonemap, trd_core::Tonemap::Aces, "and its own operator");
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
