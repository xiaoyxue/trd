use std::rc::Rc;

use crate::assets;
use crate::video_editing::CatalogAsset;

/// Adapter facts fixed at device creation. Held behind an [`Rc`] so the
/// per-frame diagnostics clone is a refcount bump instead of three `String`
/// allocations on every rendered frame.
#[derive(Debug)]
pub struct RendererIdentity {
    pub adapter_name: String,
    pub backend: String,
    pub device_type: String,
}

#[derive(Debug, Clone)]
pub struct ImportedAssetDiagnostics {
    pub source_format: &'static str,
    pub aabb_min: [f32; 3],
    pub aabb_max: [f32; 3],
    pub preview_scale: f32,
    pub imported_material: trd_core::DisneyMaterial,
}

#[derive(Debug, Clone)]
pub struct VideoRendererDiagnostics {
    pub identity: Rc<RendererIdentity>,
    pub target_size: (u32, u32),
    pub pick_target_size: Option<(u32, u32)>,
    pub msaa_samples: u32,
    pub asset: Option<ImportedAssetDiagnostics>,
}

pub struct VideoPlacementRenderer {
    /// The shared render harness. This type is a **placement** front-end, not a
    /// renderer: it turns the editor's timeline state into layered scenes and hands
    /// them to `trd-core` like every other front-end (#180).
    renderer: trd_core::Renderer,
    /// The texture target the renderer draws into and reads back from. Owned
    /// here rather than by the harness (#203): the harness has no opinion about
    /// *where* a frame lands, and this front-end resizes its own target on the
    /// editor panel's resize.
    ///
    /// The concrete [`TextureTarget`](trd_core::TextureTarget) rather than the
    /// [`RenderTarget`](trd_core::RenderTarget) enum, because the editor always
    /// reads its frames back — a surface has no pixels to read.
    target: trd_core::TextureTarget,
    default_mode: trd_core::RenderMode,
    default_material: trd_core::DisneyMaterial,
    identity: Rc<RendererIdentity>,
    asset_diagnostics: Option<ImportedAssetDiagnostics>,
}

impl VideoPlacementRenderer {
    pub async fn new_empty(width: u32, height: u32) -> Result<Self, String> {
        let instance = trd_core::create_instance();
        let gpu = trd_core::GpuContext::request(
            &instance,
            &trd_core::GpuRequest {
                label: "trd video editing wasm device",
                ..Default::default()
            },
        )
        .await
        .map_err(|error| error.to_string())?;
        let facts = gpu.adapter_facts();
        let placeholder = assets::default_mesh().map_err(|error| error.to_string())?;
        let (renderer, target) = trd_core::Renderer::with_gpu(
            gpu.clone(),
            width,
            height,
            std::slice::from_ref(&placeholder),
        )
        .map_err(|error| error.to_string())?;
        Ok(Self {
            renderer,
            target,
            default_mode: trd_core::RenderMode::Filled,
            default_material: trd_core::DisneyMaterial::default(),
            identity: Rc::new(RendererIdentity {
                adapter_name: facts.name,
                backend: facts.backend,
                device_type: facts.device_type,
            }),
            asset_diagnostics: None,
        })
    }

    pub async fn new(
        asset: CatalogAsset,
        model_bytes: &[u8],
        texture_bytes: &[u8],
        env_bytes: &[u8],
        width: u32,
        height: u32,
    ) -> Result<Self, String> {
        let imported = match asset {
            CatalogAsset::CocaColaCan | CatalogAsset::BeerCan => {
                let text = std::str::from_utf8(model_bytes)
                    .map_err(|error| format!("OBJ is not UTF-8: {error}"))?;
                let mesh = trd_core::Mesh::from_obj(text).map_err(|error| error.to_string())?;
                let texture =
                    assets::decode_texture(texture_bytes).map_err(|error| error.to_string())?;
                ImportedAsset::Textured { mesh, texture }
            }
            CatalogAsset::Dragon => {
                let glb = trd_core::import_glb(model_bytes).map_err(|error| error.to_string())?;
                ImportedAsset::Pbr(Box::new(glb))
            }
        };
        let instance = trd_core::create_instance();
        let gpu = trd_core::GpuContext::request(
            &instance,
            &trd_core::GpuRequest {
                label: "trd video editing wasm device",
                ..Default::default()
            },
        )
        .await
        .map_err(|error| error.to_string())?;
        let facts = gpu.adapter_facts();
        let asset_diagnostics = imported.diagnostics();
        let mesh = imported.mesh();
        let (mut renderer, target) =
            trd_core::Renderer::with_gpu(gpu.clone(), width, height, std::slice::from_ref(mesh))
                .map_err(|error| error.to_string())?;
        let (default_mode, default_material) = imported.configure(&mut renderer);
        let env = assets::decode_env_hdr(env_bytes).map_err(|error| error.to_string())?;
        renderer.set_env_map(env);
        Ok(Self {
            renderer,
            target,
            default_mode,
            default_material,
            identity: Rc::new(RendererIdentity {
                adapter_name: facts.name,
                backend: facts.backend,
                device_type: facts.device_type,
            }),
            asset_diagnostics: Some(asset_diagnostics),
        })
    }

    pub fn defaults(&self) -> (trd_core::RenderMode, trd_core::DisneyMaterial) {
        (self.default_mode, self.default_material.clone())
    }

    pub fn size(&self) -> (u32, u32) {
        (self.target.width(), self.target.height())
    }

    pub fn diagnostics(&self) -> VideoRendererDiagnostics {
        VideoRendererDiagnostics {
            identity: self.identity.clone(),
            target_size: self.size(),
            pick_target_size: self.renderer.pick_target_size(),
            msaa_samples: 4,
            asset: self.asset_diagnostics.clone(),
        }
    }

    /// Resizes the editor's own render target (#203): the harness owns no target
    /// to resize on its behalf, so this front-end tracks it and asks the renderer
    /// to rebuild it — `TextureTarget::new`'s zero/`max_texture_dimension_2d`
    /// checks are all that guards against a degenerate size here.
    pub fn resize(&mut self, width: u32, height: u32) -> Result<(), String> {
        if self.size() == (width, height) {
            return Ok(());
        }
        self.renderer
            .resize_texture_target(&mut self.target, width, height)
            .map_err(|error| error.to_string())
    }

    pub async fn pick(
        &mut self,
        frame: &trd_core::VideoEditingFrame,
        source_size: (u32, u32),
        model: trd_core::Matrix4,
        point: (u32, u32),
    ) -> Result<Option<u32>, String> {
        let camera = self.frame_camera(frame, source_size)?;
        Ok(self
            .renderer
            .pick(
                camera,
                &[trd_core::Draw {
                    mesh_id: 0,
                    model: model.to_cols_array(),
                    selection: trd_core::DrawSelection::INHERIT,
                }],
                point.0,
                point.1,
                self.viewport(),
            )
            .await)
    }

    #[allow(clippy::too_many_arguments)]
    pub async fn render(
        &mut self,
        rgba: &[u8],
        frame_width: u32,
        frame_height: u32,
        calibration_size: (u32, u32),
        background_frame: &trd_core::VideoEditingFrame,
        quad_model: Option<trd_core::Matrix4>,
        quad_axes: Option<trd_core::Matrix4>,
        selected_quad: bool,
        placement_frame: Option<&trd_core::VideoEditingFrame>,
        model: Option<trd_core::Matrix4>,
        state: &crate::scene::SceneState,
    ) -> Result<Vec<u8>, String> {
        self.renderer
            .update_frame_texture_rgba(rgba, frame_width, frame_height);
        let identity_camera = trd_core::FrameParams::IDENTITY
            .to_camera(self.viewport())
            .map_err(|error| error.to_string())?;
        let background_camera = self
            .frame_camera(background_frame, calibration_size)
            .unwrap_or(identity_camera);
        let foreground_camera = placement_frame
            .map(|frame| self.frame_camera(frame, calibration_size))
            .transpose()?
            .unwrap_or(background_camera);
        if self.renderer.mesh_count() > 0 {
            self.renderer
                .set_mesh_disney_material(0, state.materials[0].clone());
            self.renderer
                .set_mesh_image_based_lighting(0, state.image_based_lighting[0]);
            self.renderer
                .set_mesh_tone_mapping(0, state.tone_mappings[0]);
            self.renderer
                .set_mesh_pbr_debug_view(0, state.pbr_debug_views[0]);
            self.renderer.set_lighting(state.lighting);
        }

        let has_mesh = self.renderer.mesh_count() > 0;
        let (background, foreground, selection_overlay) = placement_scenes(
            quad_model,
            quad_axes,
            selected_quad,
            model.filter(|_| has_mesh),
            state,
        );

        self.renderer
            .render_layers(
                &[
                    trd_core::SceneLayer::new(background_camera, &background),
                    trd_core::SceneLayer::new(foreground_camera, &foreground),
                    trd_core::SceneLayer::new(foreground_camera, &selection_overlay),
                ],
                &self.target,
            )
            .await
            .map_err(|error| error.to_string())
    }

    fn viewport(&self) -> trd_core::Viewport {
        trd_core::Viewport {
            width: self.target.width(),
            height: self.target.height(),
        }
    }

    fn frame_camera(
        &self,
        frame: &trd_core::VideoEditingFrame,
        source_size: (u32, u32),
    ) -> Result<trd_core::Camera, String> {
        let k = frame.k.ok_or("selected video frame has no quad/K")?;
        let (width, height) = self.size();
        let sx = width as f32 / source_size.0 as f32;
        let sy = height as f32 / source_size.1 as f32;
        let params = trd_core::FrameParams {
            k: Some([
                k[0] * sx,
                k[3] * sy,
                k[6],
                k[1] * sx,
                k[4] * sy,
                k[7],
                k[2] * sx,
                k[5] * sy,
                k[8],
            ]),
            ..trd_core::FrameParams::IDENTITY
        };
        params
            .to_camera(self.viewport())
            .map_err(|error| error.to_string())
    }
}

enum ImportedAsset {
    Textured {
        mesh: trd_core::Mesh,
        texture: trd_core::ImageTexture,
    },
    Pbr(Box<trd_core::GltfAsset>),
}

impl ImportedAsset {
    fn mesh(&self) -> &trd_core::Mesh {
        match self {
            Self::Textured { mesh, .. } => mesh,
            Self::Pbr(asset) => &asset.mesh,
        }
    }

    fn diagnostics(&self) -> ImportedAssetDiagnostics {
        let mesh = self.mesh();
        let aabb = mesh.aabb();
        let size = aabb.size();
        let max_extent = size.x().max(size.y()).max(size.z());
        let preview_scale = if max_extent > trd_core::EPSILON {
            trd_core::DEFAULT_PREVIEW_TARGET / max_extent
        } else {
            1.0
        };
        let (source_format, imported_material) = match self {
            Self::Textured { .. } => (
                "OBJ",
                trd_core::DisneyMaterial {
                    metallic: 0.0,
                    roughness: 0.35,
                    auxiliary: trd_core::Auxiliary {
                        textures: trd_core::MaterialTextures {
                            base_color: true,
                            ..Default::default()
                        },
                        ..Default::default()
                    },
                    ..Default::default()
                },
            ),
            Self::Pbr(asset) => ("GLB", asset.material.clone()),
        };
        ImportedAssetDiagnostics {
            source_format,
            aabb_min: aabb.min().to_array(),
            aabb_max: aabb.max().to_array(),
            preview_scale,
            imported_material,
        }
    }

    fn configure(
        self,
        renderer: &mut trd_core::Renderer,
    ) -> (trd_core::RenderMode, trd_core::DisneyMaterial) {
        match self {
            Self::Textured { texture, .. } => {
                renderer.set_mesh_texture(0, &texture);
                let material = trd_core::DisneyMaterial {
                    metallic: 0.0,
                    roughness: 0.35,
                    ..Default::default()
                };
                renderer.set_mesh_disney_material(0, material.clone());
                (trd_core::RenderMode::Shaded, material)
            }
            Self::Pbr(asset) => {
                if let Some(texture) = asset.base_color_texture.as_ref() {
                    renderer.set_mesh_texture(0, texture);
                }
                if let Some(texture) = asset.metallic_roughness_texture.as_ref() {
                    renderer.set_mesh_metallic_roughness_texture(0, texture);
                }
                if let Some(texture) = asset.normal_texture.as_ref() {
                    renderer.set_mesh_normal_texture(0, texture);
                }
                renderer.set_mesh_disney_material(0, asset.material.clone());
                (trd_core::RenderMode::Shaded, asset.material)
            }
        }
    }
}

/// Authors the three layers of an editor frame from the timeline + scene state.
///
/// * **background** — the video plane, plus the placement quad's outline and (when
///   selected) its floor grid and basis axes. Seen through the *background*
///   frame's calibration.
/// * **foreground** — the placed object and its world/local gizmos, seen through
///   the *placement* frame's calibration.
/// * **selection overlay** — the selection AABB, drawn last so it is never
///   occluded by the object it outlines.
///
/// Free function rather than a method: this is placement logic, not rendering, and
/// keeping it out of the renderer makes it testable without a GPU (#180).
pub fn placement_scenes(
    quad_model: Option<trd_core::Matrix4>,
    quad_axes: Option<trd_core::Matrix4>,
    selected_quad: bool,
    model: Option<trd_core::Matrix4>,
    state: &crate::scene::SceneState,
) -> (
    Vec<trd_core::DrawableObject>,
    Vec<trd_core::DrawableObject>,
    Vec<trd_core::DrawableObject>,
) {
    let mut background = vec![trd_core::DrawableObject::FramePlane {
        fit: trd_core::FrameFit::Stretch,
    }];
    if let Some(quad_model) = quad_model {
        background.push(trd_core::DrawableObject::QuadOutline {
            model: quad_model.to_cols_array(),
            selected: selected_quad,
        });
        if selected_quad {
            background.push(trd_core::DrawableObject::PlaneGrid {
                plane: trd_core::GridPlane::Xy,
                model: quad_model.to_cols_array(),
            });
            if let Some(axes) = quad_axes {
                background.push(trd_core::DrawableObject::CoordinateAxes {
                    model: axes.to_cols_array(),
                });
            }
        }
    }

    let mut foreground = Vec::new();
    let mut selection_overlay = Vec::new();
    if let Some(model) = model.map(|model| model.to_cols_array()) {
        foreground.push(trd_core::DrawableObject::Mesh {
            mesh_id: 0,
            model,
            mode: state.modes[0],
        });
        if state.show_aabb || state.selected == Some(0) {
            selection_overlay.push(trd_core::DrawableObject::AabbBox { mesh_id: 0, model });
        }
        if state.show_local_axes {
            foreground.push(trd_core::DrawableObject::CoordinateAxes { model });
        }
        if state.show_axes {
            foreground.push(trd_core::DrawableObject::CoordinateAxes {
                model: trd_core::Matrix4::IDENTITY.to_cols_array(),
            });
        }
        if state.show_local_grid {
            foreground.push(trd_core::DrawableObject::PlaneGrid {
                plane: trd_core::GridPlane::Xz,
                model,
            });
        }
        if state.show_world_grid {
            foreground.push(trd_core::DrawableObject::PlaneGrid {
                plane: trd_core::GridPlane::Xz,
                model: trd_core::Matrix4::IDENTITY.to_cols_array(),
            });
        }
    }
    (background, foreground, selection_overlay)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::SceneState;

    fn is_axes(d: &trd_core::DrawableObject) -> bool {
        matches!(d, trd_core::DrawableObject::CoordinateAxes { .. })
    }

    /// The video plane is always the first background drawable, so everything else
    /// composites over it.
    #[test]
    fn the_video_plane_is_always_drawn_first() {
        let (background, _, _) = placement_scenes(None, None, false, None, &SceneState::default());
        assert!(matches!(
            background.first(),
            Some(trd_core::DrawableObject::FramePlane { .. })
        ));
    }

    /// Selecting the quad reveals its floor grid + basis axes; deselecting hides
    /// them but keeps the outline.
    #[test]
    fn selecting_the_quad_adds_its_grid_and_axes() {
        let quad = trd_core::Matrix4::IDENTITY;
        let state = SceneState::default();

        let (unselected, _, _) = placement_scenes(Some(quad), Some(quad), false, None, &state);
        assert_eq!(unselected.len(), 2, "video plane + quad outline only");

        let (selected, _, _) = placement_scenes(Some(quad), Some(quad), true, None, &state);
        assert_eq!(selected.len(), 4, "+ floor grid + basis axes");
        assert!(selected.iter().any(is_axes));
    }

    /// The selection AABB goes in its own layer, so it is drawn over the object it
    /// outlines rather than z-fighting with it.
    #[test]
    fn the_selection_aabb_is_its_own_layer() {
        let state = SceneState {
            selected: Some(0),
            ..SceneState::default()
        };
        let (_, foreground, overlay) =
            placement_scenes(None, None, false, Some(trd_core::Matrix4::IDENTITY), &state);
        assert!(matches!(
            overlay.as_slice(),
            [trd_core::DrawableObject::AabbBox { mesh_id: 0, .. }]
        ));
        assert!(!foreground
            .iter()
            .any(|d| matches!(d, trd_core::DrawableObject::AabbBox { .. })));
    }

    /// Without a placed object there is no foreground at all — the editor still
    /// shows the video and the quad.
    #[test]
    fn a_video_only_frame_has_an_empty_foreground() {
        let (background, foreground, overlay) =
            placement_scenes(None, None, false, None, &SceneState::default());
        assert_eq!(background.len(), 1);
        assert!(foreground.is_empty());
        assert!(overlay.is_empty());
    }
}
