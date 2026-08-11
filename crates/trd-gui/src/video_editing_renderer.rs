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
    gpu: std::sync::Arc<trd_core::GpuContext>,
    renderer: trd_core::SceneRenderer,
    target: trd_core::OffscreenTarget,
    pick_target: Option<trd_core::PickTarget>,
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
        let renderer = trd_core::SceneRenderer::auto_fit(
            gpu.clone(),
            trd_core::OFFSCREEN_FORMAT,
            std::slice::from_ref(&placeholder),
        );
        let target = trd_core::OffscreenTarget::new(&gpu.device, width, height)
            .map_err(|error| error.to_string())?;
        Ok(Self {
            gpu,
            renderer,
            target,
            pick_target: None,
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
        let mut renderer = trd_core::SceneRenderer::auto_fit(
            gpu.clone(),
            trd_core::OFFSCREEN_FORMAT,
            std::slice::from_ref(mesh),
        );
        let (default_mode, default_material) = imported.configure(&mut renderer);
        let env = assets::decode_env_hdr(env_bytes).map_err(|error| error.to_string())?;
        renderer.set_env_map(env);
        let target = trd_core::OffscreenTarget::new(&gpu.device, width, height)
            .map_err(|error| error.to_string())?;
        Ok(Self {
            gpu,
            renderer,
            target,
            pick_target: None,
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
            pick_target_size: self
                .pick_target
                .as_ref()
                .map(|_| (self.target.width(), self.target.height())),
            msaa_samples: 4,
            asset: self.asset_diagnostics.clone(),
        }
    }

    pub fn resize(&mut self, width: u32, height: u32) -> Result<(), String> {
        if self.size() == (width, height) {
            return Ok(());
        }
        self.target = trd_core::OffscreenTarget::new(&self.gpu.device, width, height)
            .map_err(|error| error.to_string())?;
        self.pick_target = None;
        Ok(())
    }

    pub async fn pick(
        &mut self,
        frame: &trd_core::VideoEditingFrame,
        source_size: (u32, u32),
        model: trd_core::Matrix4,
        point: (u32, u32),
    ) -> Result<Option<u32>, String> {
        let params = self.frame_params(frame, source_size)?;
        let target = self.pick_target.get_or_insert_with(|| {
            trd_core::PickTarget::new(&self.gpu.device, self.target.width(), self.target.height())
        });
        Ok(target
            .pick(
                &self.gpu,
                &mut self.renderer,
                params,
                &[trd_core::Draw {
                    mesh_id: 0,
                    model: model.to_cols_array(),
                    mode: None,
                }],
                point.0,
                point.1,
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
        let background_params = self
            .frame_params(background_frame, calibration_size)
            .unwrap_or(trd_core::FrameParams::IDENTITY);
        let foreground_params = placement_frame
            .map(|frame| self.frame_params(frame, calibration_size))
            .transpose()?
            .unwrap_or(background_params);
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
        let model = model.map(|model| model.to_cols_array());
        if let Some(model) = model.filter(|_| self.renderer.mesh_count() > 0) {
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
        self.target
            .render_three_pass(
                &self.gpu,
                &mut self.renderer,
                background_params,
                foreground_params,
                &background,
                &foreground,
                &selection_overlay,
            )
            .await
            .map_err(|error| error.to_string())
    }

    fn frame_params(
        &self,
        frame: &trd_core::VideoEditingFrame,
        source_size: (u32, u32),
    ) -> Result<trd_core::FrameParams, String> {
        let k = frame.k.ok_or("selected video frame has no quad/K")?;
        let sx = self.target.width() as f32 / source_size.0 as f32;
        let sy = self.target.height() as f32 / source_size.1 as f32;
        Ok(trd_core::FrameParams {
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
        })
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
        renderer: &mut trd_core::SceneRenderer,
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
                (trd_core::RenderMode::Pbr, material)
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
                (trd_core::RenderMode::Pbr, asset.material)
            }
        }
    }
}
