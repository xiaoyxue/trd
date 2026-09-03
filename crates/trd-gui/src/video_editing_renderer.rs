use std::rc::Rc;

use crate::assets;
use crate::video_editing::CatalogAsset;

/// Adapter facts fixed at device creation. Held behind an [`Rc`] to avoid
/// repeated `String` allocations on per-frame diagnostics clones.
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

/// Per-frame CPU↔GPU traffic **on the video frame path**, so the copy count is
/// **observed** rather than asserted.
///
/// Each field is the bytes that crossed the boundary for the most recent frame.
/// Per-frame CPU↔GPU traffic on the video frame path. `0` means no frame-sized
/// buffer crossed; per-frame uniforms and egui geometry are not tracked here.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct TransferCounts {
    /// Video frame CPU→GPU (`queue.write_texture`).
    pub frame_upload: usize,
    /// Rendered pixels GPU→CPU (`read_pixels`).
    pub readback: usize,
    /// Rendered pixels CPU→GPU again, through the UI toolkit.
    pub ui_upload: usize,
}

impl TransferCounts {
    /// How many frame-sized buffers crossed the CPU↔GPU boundary.
    pub fn crossings(self) -> u8 {
        [self.frame_upload, self.readback, self.ui_upload]
            .into_iter()
            .filter(|bytes| *bytes > 0)
            .count() as u8
    }

    /// Total frame-path bytes moved for the last frame.
    pub fn total_bytes(self) -> usize {
        self.frame_upload
            .saturating_add(self.readback)
            .saturating_add(self.ui_upload)
    }
}

#[derive(Debug, Clone)]
pub struct VideoRendererDiagnostics {
    pub identity: Rc<RendererIdentity>,
    pub target_size: (u32, u32),
    pub pick_target_size: Option<(u32, u32)>,
    pub msaa_samples: u32,
    pub asset: Option<ImportedAssetDiagnostics>,
    pub transfers: TransferCounts,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) enum VideoExportAsset {
    Embedded {
        mesh: trd_core::Mesh,
        texture: trd_core::ImageTexture,
    },
    Gltf(trd_core::MeshReference),
}

/// Where a frame's pixels come from. `External` keeps the frame on GPU (#229, #302).
pub enum FrameSource<'a> {
    /// Tightly-packed row-major RGBA8, `width * height * 4` bytes.
    Rgba(&'a [u8]),
    /// A frame the delivery surface kept on the GPU, copied GPU→GPU.
    External(&'a dyn trd_core::ExternalFrame),
}

impl FrameSource<'_> {
    /// Bytes this source moves CPU→GPU; zero for GPU-resident frames.
    fn upload_bytes(&self) -> usize {
        match self {
            Self::Rgba(rgba) => rgba.len(),
            Self::External(_) => 0,
        }
    }
}

pub struct VideoPlacementRenderer {
    /// Shared render harness (#180).
    renderer: trd_core::Renderer,
    /// Texture target owned here, not by the harness (#203).
    target: trd_core::TextureTarget,
    default_mode: trd_core::RenderMode,
    default_material: trd_core::DisneyMaterial,
    identity: Rc<RendererIdentity>,
    asset_diagnostics: Option<ImportedAssetDiagnostics>,
    export_asset: Option<Rc<VideoExportAsset>>,
    replay_lighting: trd_core::Lighting,
    /// Transfer counts written at the transfer sites (#229).
    pub transfers: TransferCounts,
}

impl VideoPlacementRenderer {
    pub async fn new_empty(width: u32, height: u32) -> Result<Self, String> {
        let gpu = Self::own_gpu().await?;
        Self::new_empty_with_gpu(gpu, width, height)
    }

    /// Builds on an already-created GPU context (e.g. `eframe`'s shared device).
    pub fn new_empty_with_gpu(
        gpu: std::sync::Arc<trd_core::GpuContext>,
        width: u32,
        height: u32,
    ) -> Result<Self, String> {
        let facts = gpu.adapter_facts();
        let placeholder = assets::default_mesh().map_err(|error| error.to_string())?;
        let (renderer, target) =
            trd_core::Renderer::with_gpu(gpu, width, height, std::slice::from_ref(&placeholder))
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
            export_asset: None,
            replay_lighting: trd_core::Lighting::default(),
            transfers: TransferCounts::default(),
        })
    }

    pub async fn new_scene(
        assets: &[trd_core::MeshAsset],
        env_bytes: &[u8],
        width: u32,
        height: u32,
    ) -> Result<Self, String> {
        let gpu = Self::own_gpu().await?;
        Self::new_scene_with_gpu(gpu, assets, env_bytes, width, height)
    }

    pub fn new_scene_with_gpu(
        gpu: std::sync::Arc<trd_core::GpuContext>,
        assets: &[trd_core::MeshAsset],
        env_bytes: &[u8],
        width: u32,
        height: u32,
    ) -> Result<Self, String> {
        let facts = gpu.adapter_facts();
        let meshes = assets
            .iter()
            .map(|asset| asset.mesh.clone())
            .collect::<Vec<_>>();
        let (mut renderer, target) = trd_core::Renderer::with_gpu(gpu, width, height, &meshes)
            .map_err(|error| error.to_string())?;
        configure_mesh_assets(&mut renderer, assets);
        renderer.set_env_map(assets::decode_env_hdr(env_bytes).map_err(|error| error.to_string())?);
        let replay_lighting = if assets.iter().any(|asset| {
            asset.metallic_roughness_texture.is_some() || asset.normal_texture.is_some()
        }) {
            trd_core::Lighting {
                ambient: 0.0,
                scale: 0.0,
                ..trd_core::Lighting::default()
            }
        } else {
            trd_core::Lighting::default()
        };
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
            export_asset: None,
            replay_lighting,
            transfers: TransferCounts::default(),
        })
    }

    /// Requests a standalone GPU context for shells with no device to share.
    async fn own_gpu() -> Result<std::sync::Arc<trd_core::GpuContext>, String> {
        let instance = trd_core::create_instance();
        trd_core::GpuContext::request(
            &instance,
            &trd_core::GpuRequest {
                label: "trd video editing device",
                ..Default::default()
            },
        )
        .await
        .map_err(|error| error.to_string())
    }

    pub async fn new(
        asset: CatalogAsset,
        source: trd_core::MeshReference,
        model_bytes: &[u8],
        texture_bytes: &[u8],
        env_bytes: &[u8],
        width: u32,
        height: u32,
    ) -> Result<Self, String> {
        let gpu = Self::own_gpu().await?;
        Self::new_with_gpu(
            gpu,
            asset,
            source,
            model_bytes,
            texture_bytes,
            env_bytes,
            width,
            height,
        )
    }

    /// Like [`new`](Self::new) on a shared GPU context; rebuilt on every asset swap.
    #[allow(clippy::too_many_arguments)]
    pub fn new_with_gpu(
        gpu: std::sync::Arc<trd_core::GpuContext>,
        asset: CatalogAsset,
        source: trd_core::MeshReference,
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
        let facts = gpu.adapter_facts();
        let asset_diagnostics = imported.diagnostics();
        let export_asset = Rc::new(imported.export_asset(source));
        let mesh = imported.mesh();
        let (mut renderer, target) =
            trd_core::Renderer::with_gpu(gpu, width, height, std::slice::from_ref(mesh))
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
            export_asset: Some(export_asset),
            replay_lighting: trd_core::Lighting::default(),
            transfers: TransferCounts::default(),
        })
    }

    pub fn defaults(&self) -> (trd_core::RenderMode, trd_core::DisneyMaterial) {
        (self.default_mode, self.default_material.clone())
    }

    pub(crate) fn export_asset(&self) -> Option<Rc<VideoExportAsset>> {
        self.export_asset.clone()
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
            transfers: self.transfers,
        }
    }

    /// Resizes the render target (#203).
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
                    model,
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
        background_frame: Option<&trd_core::VideoEditingFrame>,
        quad: QuadOverlay,
        placement_frame: Option<&trd_core::VideoEditingFrame>,
        model: Option<trd_core::Matrix4>,
        state: &crate::scene::SceneState,
    ) -> Result<Vec<u8>, String> {
        self.draw(
            FrameSource::Rgba(rgba),
            frame_width,
            frame_height,
            calibration_size,
            background_frame,
            quad,
            placement_frame,
            model,
            state,
        )?;
        let pixels = self
            .renderer
            .read_pixels(&self.target)
            .await
            .map_err(|error| error.to_string())?;
        // GPU→CPU, plus the UI toolkit's own CPU→GPU re-upload of the same bytes.
        self.transfers.readback = pixels.len();
        self.transfers.ui_upload = pixels.len();
        Ok(pixels)
    }

    pub async fn render_scene_frame(
        &mut self,
        rgba: &[u8],
        frame_width: u32,
        frame_height: u32,
        calibration_size: (u32, u32),
        frame: &trd_core::DecodedFrame,
    ) -> Result<Vec<u8>, String> {
        self.draw_scene_frame(
            FrameSource::Rgba(rgba),
            frame_width,
            frame_height,
            calibration_size,
            frame,
        )?;
        let pixels = self
            .renderer
            .read_pixels(&self.target)
            .await
            .map_err(|error| error.to_string())?;
        self.transfers.readback = pixels.len();
        self.transfers.ui_upload = pixels.len();
        Ok(pixels)
    }

    /// Draws the three placement layers without reading them back.
    /// Use [`render`](Self::render) when the shell needs pixels (different device).
    #[allow(clippy::too_many_arguments)]
    pub fn draw(
        &mut self,
        source: FrameSource<'_>,
        frame_width: u32,
        frame_height: u32,
        calibration_size: (u32, u32),
        background_frame: Option<&trd_core::VideoEditingFrame>,
        quad: QuadOverlay,
        placement_frame: Option<&trd_core::VideoEditingFrame>,
        model: Option<trd_core::Matrix4>,
        state: &crate::scene::SceneState,
    ) -> Result<(), String> {
        self.upload_frame(source, frame_width, frame_height);
        let identity_camera = trd_core::FrameParams::IDENTITY
            .to_camera(self.viewport())
            .map_err(|error| error.to_string())?;
        // No annotated row: draw with the identity camera (#264).
        let background_camera = background_frame
            .and_then(|frame| self.frame_camera(frame, calibration_size).ok())
            .unwrap_or(identity_camera);
        let foreground_camera = placement_frame
            .map(|frame| self.frame_camera(frame, calibration_size))
            .transpose()?
            .unwrap_or(background_camera);
        if self.renderer.mesh_count() > 0 {
            self.renderer.set_appearance(
                trd_core::MeshTarget::One(0),
                trd_core::MeshAppearance {
                    material: state.materials[0].clone(),
                    ibl: state.image_based_lighting[0],
                    tone_mapping: state.tone_mappings[0],
                    debug_view: state.pbr_debug_views[0],
                },
            );
        }

        let has_mesh = self.renderer.mesh_count() > 0;
        let (background, foreground, selection_overlay) =
            placement_scenes(quad, model.filter(|_| has_mesh), state);

        self.renderer.draw_layers(
            &[
                trd_core::SceneLayer::new(background_camera, &background),
                trd_core::SceneLayer::new(foreground_camera, &foreground),
                trd_core::SceneLayer::new(foreground_camera, &selection_overlay),
            ],
            &self.target,
        );
        Ok(())
    }

    pub fn draw_scene_frame(
        &mut self,
        source: FrameSource<'_>,
        frame_width: u32,
        frame_height: u32,
        calibration_size: (u32, u32),
        frame: &trd_core::DecodedFrame,
    ) -> Result<(), String> {
        self.upload_frame(source, frame_width, frame_height);
        let camera = self.protocol_camera(&frame.params, calibration_size)?;
        let draws = frame.resolved_draws();
        let (background, foreground) = replay_scenes(&draws, self.replay_lighting);
        self.renderer.draw_layers(
            &[
                trd_core::SceneLayer::new(camera, &background),
                trd_core::SceneLayer::new(camera, &foreground),
            ],
            &self.target,
        );
        Ok(())
    }

    fn upload_frame(&mut self, source: FrameSource<'_>, width: u32, height: u32) {
        self.transfers = TransferCounts {
            frame_upload: source.upload_bytes(),
            readback: 0,
            ui_upload: 0,
        };
        match source {
            FrameSource::Rgba(rgba) => self.renderer.update_frame_texture_rgba(rgba, width, height),
            FrameSource::External(frame) => self.renderer.update_frame_texture_external(frame),
        }
    }

    /// Sampleable view of the rendered target (gamma space).
    pub fn target_view(&self) -> wgpu::TextureView {
        self.target.create_view()
    }

    /// Generation key; changes on resize or asset swap (stale views are invalid).
    pub fn renderer_generation_key(&self) -> usize {
        Rc::as_ptr(&self.identity) as usize
    }

    /// The device the target lives on, needed to register the view with egui.
    pub fn device(&self) -> &wgpu::Device {
        &self.renderer.gpu().device
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
        let params = trd_core::FrameParams {
            k: Some(crate::video_editing::protocol_k_from_row_major(k)),
            ..trd_core::FrameParams::IDENTITY
        };
        self.protocol_camera(&params, source_size)
    }

    fn protocol_camera(
        &self,
        params: &trd_core::FrameParams,
        source_size: (u32, u32),
    ) -> Result<trd_core::Camera, String> {
        let mut params = *params;
        if let Some(k) = params.k {
            let (width, height) = self.size();
            let sx = width as f32 / source_size.0.max(1) as f32;
            let sy = height as f32 / source_size.1.max(1) as f32;
            params.k = Some(scale_protocol_k(k, sx, sy));
        }

        params
            .to_camera(self.viewport())
            .map_err(|error| error.to_string())
    }
}

fn scale_protocol_k(k: [f32; 9], sx: f32, sy: f32) -> [f32; 9] {
    [
        k[0] * sx,
        k[1] * sy,
        k[2],
        k[3] * sx,
        k[4] * sy,
        k[5],
        k[6] * sx,
        k[7] * sy,
        k[8],
    ]
}

fn configure_mesh_assets(renderer: &mut trd_core::Renderer, assets: &[trd_core::MeshAsset]) {
    for (mesh_id, asset) in assets.iter().enumerate() {
        renderer.set_disney_material(trd_core::MeshTarget::One(mesh_id), asset.material.clone());
        if let Some(texture) = asset.base_color_texture.as_ref() {
            renderer.set_mesh_texture(mesh_id, texture);
        }
        if let Some(texture) = asset.metallic_roughness_texture.as_ref() {
            renderer.set_mesh_metallic_roughness_texture(mesh_id, texture);
        }
        if let Some(texture) = asset.normal_texture.as_ref() {
            renderer.set_mesh_normal_texture(mesh_id, texture);
        }
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

    fn export_asset(&self, source: trd_core::MeshReference) -> VideoExportAsset {
        match self {
            Self::Textured { mesh, texture } => VideoExportAsset::Embedded {
                mesh: mesh.clone(),
                texture: texture.clone(),
            },
            Self::Pbr(_) => VideoExportAsset::Gltf(source),
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
                    auxiliary: trd_core::Auxiliary {
                        textures: trd_core::MaterialTextures {
                            base_color: true,
                            ..Default::default()
                        },
                        ..Default::default()
                    },
                    ..Default::default()
                };
                renderer.set_disney_material(trd_core::MeshTarget::One(0), material.clone());
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
                renderer.set_disney_material(trd_core::MeshTarget::One(0), asset.material.clone());
                (trd_core::RenderMode::Shaded, asset.material)
            }
        }
    }
}

/// The tracked quad's draw state: independent outline/gizmo toggles;
/// `hovered`/`selected` wash the quad face.
#[derive(Debug, Clone, Copy, Default)]
pub struct QuadOverlay {
    /// Quad outline, fill, and local grid model. `None` on an untracked row.
    pub model: Option<trd_core::Matrix4>,
    /// Places the local-frame axes. `None` on an untracked row.
    pub axes: Option<trd_core::Matrix4>,
    /// Draw the quad outline.
    pub show_quads: bool,
    /// Draw the local grid and axes.
    pub show_gizmos: bool,
    /// The pointer is over the quad: wash its face.
    pub hovered: bool,
    /// Highlight the outline as the selected object, and wash its face.
    pub selected: bool,
}

/// Authors the three layers — background (video + quad overlay), foreground
/// (object), selection overlay (AABB) drawn last. Free function so it is
/// testable without a GPU (#180).
pub fn placement_scenes(
    quad: QuadOverlay,
    model: Option<trd_core::Matrix4>,
    state: &crate::scene::SceneState,
) -> (trd_core::Scene, trd_core::Scene, trd_core::Scene) {
    let mut background = trd_core::Scene::new().with_background(trd_core::Background {
        environment: None,
        frame: Some(trd_core::FrameFit::Stretch),
    });
    if let Some(quad_model) = quad.model.filter(|_| quad.show_quads) {
        if quad.hovered || quad.selected {
            background.push(trd_core::DrawableObject::quad_fill(quad_model));
        }

        background.push(trd_core::DrawableObject::quad_outline(
            quad_model,
            quad.selected,
        ));
    }
    if quad.show_gizmos {
        if let Some(quad_model) = quad.model {
            background.push(trd_core::DrawableObject::plane_grid(
                trd_core::GridPlane::Xy,
                quad_model,
            ));
        }
        if let Some(axes) = quad.axes {
            background.push(trd_core::DrawableObject::coordinate_axes(axes));
        }
    }

    let mut foreground = trd_core::Scene::new();
    let mut selection_overlay = trd_core::Scene::new();
    if let Some(model) = model {
        foreground.push(trd_core::DrawableObject::mesh(0, model, state.modes[0]));
        if state.show_aabb || state.selected == Some(0) {
            selection_overlay.push(trd_core::DrawableObject::aabb_box(0, model));
        }
        if state.show_local_axes {
            foreground.push(trd_core::DrawableObject::coordinate_axes(model));
        }
        if state.show_axes {
            foreground.push(trd_core::DrawableObject::coordinate_axes(
                trd_core::Matrix4::IDENTITY,
            ));
        }
        if state.show_local_grid {
            foreground.push(trd_core::DrawableObject::plane_grid(
                trd_core::GridPlane::Xz,
                model,
            ));
        }
        if state.show_world_grid {
            foreground.push(trd_core::DrawableObject::plane_grid(
                trd_core::GridPlane::Xz,
                trd_core::Matrix4::IDENTITY,
            ));
        }
    }
    // Each layer carries the full light rig (#182).
    (
        background.with_lighting(state.lighting),
        foreground.with_lighting(state.lighting),
        selection_overlay.with_lighting(state.lighting),
    )
}

fn replay_scenes(
    draws: &[trd_core::Draw],
    lighting: trd_core::Lighting,
) -> (trd_core::Scene, trd_core::Scene) {
    let options = trd_core::RenderOptions::default();
    (
        trd_core::Scene::from_draws(&[], &options, Some(trd_core::FrameFit::Stretch))
            .with_lighting(lighting),
        trd_core::Scene::from_draws(draws, &options, None).with_lighting(lighting),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::SceneState;

    fn is_axes(d: &trd_core::DrawableObject) -> bool {
        matches!(d.primitive(), trd_core::Primitive::CoordinateAxes)
    }

    /// Video plane is a scene setting, not a drawable (#204).
    #[test]
    fn the_video_plane_is_always_the_background() {
        let (background, _, _) =
            placement_scenes(QuadOverlay::default(), None, &SceneState::default());
        assert_eq!(
            background.background().frame,
            Some(trd_core::FrameFit::Stretch)
        );
    }

    #[test]
    fn replay_keeps_video_in_the_background_and_protocol_draws_in_front() {
        let draw = trd_core::Draw {
            mesh_id: 0,
            model: trd_core::Matrix4::IDENTITY,
            selection: trd_core::DrawSelection::Mesh(Some(trd_core::RenderMode::Wireframe)),
        };
        let (background, foreground) = replay_scenes(&[draw], trd_core::Lighting::default());

        assert_eq!(
            background.background().frame,
            Some(trd_core::FrameFit::Stretch)
        );
        assert!(background.objects().is_empty());
        assert_eq!(foreground.objects().len(), 1);
        assert!(matches!(
            foreground.objects()[0].primitive(),
            trd_core::Primitive::Mesh {
                mode: trd_core::RenderMode::Wireframe,
                ..
            }
        ));
    }

    #[test]
    fn authoring_and_protocol_k_paths_share_the_same_transpose_and_scale() {
        let row_major = [1000.0, 0.0, 960.0, 0.0, 900.0, 540.0, 0.0, 0.0, 1.0];
        let protocol = crate::video_editing::protocol_k_from_row_major(row_major);

        assert_eq!(
            scale_protocol_k(protocol, 0.5, 0.25),
            [500.0, 0.0, 0.0, 0.0, 225.0, 0.0, 480.0, 135.0, 1.0]
        );
    }

    /// Quad outline and gizmos are independent toggles.
    #[test]
    fn the_quad_outline_and_the_gizmos_toggle_independently() {
        let matrix = trd_core::Matrix4::IDENTITY;
        let state = SceneState::default();
        let quad = QuadOverlay {
            model: Some(matrix),
            axes: Some(matrix),
            ..QuadOverlay::default()
        };

        let (neither, _, _) = placement_scenes(quad, None, &state);
        assert!(neither.objects().is_empty(), "both toggles off");

        let (outline_only, _, _) = placement_scenes(
            QuadOverlay {
                show_quads: true,
                ..quad
            },
            None,
            &state,
        );
        assert_eq!(outline_only.objects().len(), 1, "quad outline only");
        assert!(!outline_only.objects().iter().any(is_axes));

        let (gizmos_only, _, _) = placement_scenes(
            QuadOverlay {
                show_gizmos: true,
                ..quad
            },
            None,
            &state,
        );
        assert_eq!(gizmos_only.objects().len(), 2, "floor grid + basis axes");
        assert!(gizmos_only.objects().iter().any(is_axes));

        let (both, _, _) = placement_scenes(
            QuadOverlay {
                show_quads: true,
                show_gizmos: true,
                selected: true,
                ..quad
            },
            None,
            &state,
        );
        assert_eq!(
            both.objects().len(),
            4,
            "selection wash + outline + floor grid + basis axes"
        );
    }

    /// Hover washes the quad face; selection yellows the edge.
    #[test]
    fn hover_and_selection_wash_the_quad_face() {
        let matrix = trd_core::Matrix4::IDENTITY;
        let state = SceneState::default();
        let quad = QuadOverlay {
            model: Some(matrix),
            axes: Some(matrix),
            show_quads: true,
            ..QuadOverlay::default()
        };
        let fill = |scene: &trd_core::Scene| {
            scene
                .objects()
                .iter()
                .filter(|d| d.primitive() == trd_core::Primitive::QuadFill)
                .count()
        };
        let outline_selected = |scene: &trd_core::Scene| {
            scene
                .objects()
                .iter()
                .any(|d| d.primitive() == trd_core::Primitive::QuadOutline { selected: true })
        };

        let (idle, _, _) = placement_scenes(quad, None, &state);
        assert_eq!(fill(&idle), 0, "no wash until pointed at");
        assert!(!outline_selected(&idle));

        let (hovered, _, _) = placement_scenes(
            QuadOverlay {
                hovered: true,
                ..quad
            },
            None,
            &state,
        );
        assert_eq!(fill(&hovered), 1, "hover washes the face");
        assert!(!outline_selected(&hovered), "hover keeps the green edge");

        let (selected, _, _) = placement_scenes(
            QuadOverlay {
                selected: true,
                ..quad
            },
            None,
            &state,
        );
        assert_eq!(fill(&selected), 1, "selection keeps the wash");
        assert!(outline_selected(&selected), "selection yellows the edge");

        let (hidden, _, _) = placement_scenes(
            QuadOverlay {
                show_quads: false,
                hovered: true,
                selected: true,
                ..quad
            },
            None,
            &state,
        );
        assert_eq!(fill(&hidden), 0, "no wash with the quads switched off");
    }

    /// Selection AABB in its own layer to avoid z-fighting.
    #[test]
    fn the_selection_aabb_is_its_own_layer() {
        let state = SceneState {
            selected: Some(0),
            ..SceneState::default()
        };
        let (_, foreground, overlay) = placement_scenes(
            QuadOverlay::default(),
            Some(trd_core::Matrix4::IDENTITY),
            &state,
        );
        assert_eq!(
            overlay
                .objects()
                .iter()
                .map(|d| d.primitive())
                .collect::<Vec<_>>(),
            [trd_core::Primitive::AabbBox { mesh_id: 0 }]
        );
        assert!(!foreground
            .objects()
            .iter()
            .any(|d| matches!(d.primitive(), trd_core::Primitive::AabbBox { .. })));
    }

    /// Video-only frame: no foreground.
    #[test]
    fn a_video_only_frame_has_an_empty_foreground() {
        let (background, foreground, overlay) =
            placement_scenes(QuadOverlay::default(), None, &SceneState::default());
        assert!(
            background.objects().is_empty(),
            "video plane only, and it is a setting"
        );
        assert_eq!(
            background.background().frame,
            Some(trd_core::FrameFit::Stretch)
        );
        assert!(foreground.objects().is_empty());
        assert!(overlay.objects().is_empty());
    }
}
