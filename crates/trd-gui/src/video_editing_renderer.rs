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

/// Per-frame CPU↔GPU traffic **on the video frame path**, so the copy count is
/// **observed** rather than asserted.
///
/// Each field is the bytes that crossed the boundary for the most recent frame.
/// A zero means that crossing did not happen at all — which is what makes this a
/// meter rather than a comment: a later change that claims to remove a copy has
/// to show a `0` here, and a silent fall back to the copying path shows up as
/// the old number instead.
///
/// **Scope, so `0` is not over-read.** This counts *full-resolution image data*
/// only. A frame reading `0` still involves small CPU→GPU writes that are not
/// tracked here and never go away:
///
/// * the per-frame uniforms (camera, lighting, the frame-plane fit, instance
///   models) — tens to hundreds of bytes each;
/// * egui's own tessellated UI geometry and font atlas, re-uploaded each frame
///   by `egui-wgpu`, which is far larger than the uniforms though still far
///   smaller than a frame.
///
/// So `0` means *no frame-sized buffer crossed the boundary*, not that the
/// renderer touched the GPU without any CPU writes at all.
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
    /// How many **frame-sized** buffers crossed the CPU↔GPU boundary.
    ///
    /// Derived from the byte counts rather than stored, so it cannot drift from
    /// them: a path that stops reading pixels back reports one crossing fewer
    /// *because* its `readback` is `0`, not because someone remembered to update
    /// a constant.
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

/// Where a frame's pixels come from.
///
/// Native decodes into CPU bytes (an ffmpeg pipe), so it has nothing to avoid.
/// The browser's `VideoDecoder` has already decoded **into GPU memory**, so
/// naming the frame lets the copy stay on the GPU — the difference is a whole
/// frame of traffic at source resolution, ~99 MB for 4K (#229). One `draw`
/// serves both, so the scene assembly cannot drift between them.
pub enum FrameSource<'a> {
    /// Tightly-packed row-major RGBA8, `width * height * 4` bytes.
    Rgba(&'a [u8]),
    /// A decoded browser frame, copied GPU→GPU.
    #[cfg(target_arch = "wasm32")]
    VideoFrame(&'a web_sys::VideoFrame),
}

impl FrameSource<'_> {
    /// Bytes this source moves CPU→GPU — zero when the frame never leaves the
    /// GPU, which is exactly what the transfer meter must report.
    fn upload_bytes(&self) -> usize {
        match self {
            Self::Rgba(rgba) => rgba.len(),
            #[cfg(target_arch = "wasm32")]
            Self::VideoFrame(_) => 0,
        }
    }
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
    /// What the **last** frame actually moved across the CPU↔GPU boundary.
    ///
    /// Written at the transfer sites themselves rather than derived from which
    /// method was called, so a count can only be non-zero if that copy really
    /// ran (#229).
    pub transfers: TransferCounts,
}

impl VideoPlacementRenderer {
    pub async fn new_empty(width: u32, height: u32) -> Result<Self, String> {
        let gpu = Self::own_gpu().await?;
        Self::new_empty_with_gpu(gpu, width, height)
    }

    /// Builds the placement renderer on an **already-created** GPU context —
    /// normally the UI toolkit's own device (`eframe`'s `wgpu_render_state`).
    ///
    /// Sharing one device is what lets the rendered texture be handed to egui
    /// directly; two devices on the same adapter cannot share textures, so a
    /// separate context forces a GPU→CPU→GPU round trip every frame.
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
            transfers: TransferCounts::default(),
        })
    }

    /// Requests a **standalone** context, for a shell with no device to share.
    ///
    /// Keeps the portable path intact: a front-end that does not run on wgpu, or
    /// that has not reached its toolkit's device yet, still gets a renderer — it
    /// just pays the readback.
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
            model_bytes,
            texture_bytes,
            env_bytes,
            width,
            height,
        )
    }

    /// Like [`new`](Self::new), on an already-created (shared) GPU context.
    ///
    /// The catalog renderer is rebuilt on **every** asset swap, so a shell that
    /// misses this one opens a third device without noticing.
    #[allow(clippy::too_many_arguments)]
    pub fn new_with_gpu(
        gpu: std::sync::Arc<trd_core::GpuContext>,
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
        let facts = gpu.adapter_facts();
        let asset_diagnostics = imported.diagnostics();
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
            transfers: TransferCounts::default(),
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
            transfers: self.transfers,
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
        quad_model: Option<trd_core::Matrix4>,
        quad_axes: Option<trd_core::Matrix4>,
        selected_quad: bool,
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
            quad_model,
            quad_axes,
            selected_quad,
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

    /// Draws the three placement layers into the target **without reading them
    /// back**.
    ///
    /// The readback in [`render`](Self::render) exists only so a shell on a
    /// *different* device can re-upload the pixels through its UI toolkit. When
    /// the toolkit shares trd's device, the rendered texture is bound directly
    /// (see [`trd_core::TextureTarget::create_view`]) and those two crossings
    /// disappear. Note the return type: there are **no pixels to return** here.
    /// The readback is not "skipped" on this path; it is absent from it.
    #[allow(clippy::too_many_arguments)]
    pub fn draw(
        &mut self,
        source: FrameSource<'_>,
        frame_width: u32,
        frame_height: u32,
        calibration_size: (u32, u32),
        background_frame: Option<&trd_core::VideoEditingFrame>,
        quad_model: Option<trd_core::Matrix4>,
        quad_axes: Option<trd_core::Matrix4>,
        selected_quad: bool,
        placement_frame: Option<&trd_core::VideoEditingFrame>,
        model: Option<trd_core::Matrix4>,
        state: &crate::scene::SceneState,
    ) -> Result<(), String> {
        // Counted at the transfer site, not inferred from the call: the frame
        // genuinely crosses CPU→GPU here — or, for a `<video>` element, does not.
        // `render` adds the readback pair afterwards, so a `draw`-only frame is
        // left with exactly what it moved.
        self.transfers = TransferCounts {
            frame_upload: source.upload_bytes(),
            readback: 0,
            ui_upload: 0,
        };
        match source {
            FrameSource::Rgba(rgba) => {
                self.renderer
                    .update_frame_texture_rgba(rgba, frame_width, frame_height)
            }
            #[cfg(target_arch = "wasm32")]
            FrameSource::VideoFrame(frame) => {
                self.renderer
                    .update_frame_texture_from_video(frame, frame_width, frame_height)
            }
        }
        let identity_camera = trd_core::FrameParams::IDENTITY
            .to_camera(self.viewport())
            .map_err(|error| error.to_string())?;
        // No row — no document, or a frame the document does not annotate — is
        // the ordinary case for a plain video frame: draw it with the identity
        // camera rather than refusing to draw at all (#264).
        let background_camera = background_frame
            .and_then(|frame| self.frame_camera(frame, calibration_size).ok())
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
        }

        let has_mesh = self.renderer.mesh_count() > 0;
        let (background, foreground, selection_overlay) = placement_scenes(
            quad_model,
            quad_axes,
            selected_quad,
            model.filter(|_| has_mesh),
            state,
        );

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

    /// A sampleable view of the rendered target, for a shell sharing trd's
    /// device. Gamma space — see [`trd_core::TextureTarget::create_view`].
    pub fn target_view(&self) -> wgpu::TextureView {
        self.target.create_view()
    }

    /// Identifies *which* target the current view belongs to, so a host can tell
    /// when its registered texture went stale — the target is recreated on a
    /// resize and on an asset swap, and sampling the freed view is undefined.
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
/// * **background** — the video plane (the scene's
///   [`Background::frame`](trd_core::Background::frame), not a drawable — #204),
///   plus the placement quad's outline and (when selected) its floor grid and
///   basis axes. Seen through the *background* frame's calibration.
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
) -> (trd_core::Scene, trd_core::Scene, trd_core::Scene) {
    let mut background = trd_core::Scene::new().with_background(trd_core::Background {
        environment: None,
        frame: Some(trd_core::FrameFit::Stretch),
    });
    if let Some(quad_model) = quad_model {
        background.push(trd_core::DrawableObject::quad_outline(
            quad_model,
            selected_quad,
        ));
        if selected_quad {
            background.push(trd_core::DrawableObject::plane_grid(
                trd_core::GridPlane::Xy,
                quad_model,
            ));
            if let Some(axes) = quad_axes {
                background.push(trd_core::DrawableObject::coordinate_axes(axes));
            }
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
    // Every layer carries the frame's light rig (#182): only the foreground
    // holds a PBR mesh today, but the rig is a scene property, so each scene
    // states it rather than depending on what a previous encode left behind.
    (
        background.with_lighting(state.lighting),
        foreground.with_lighting(state.lighting),
        selection_overlay.with_lighting(state.lighting),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::SceneState;

    fn is_axes(d: &trd_core::DrawableObject) -> bool {
        matches!(d.primitive(), trd_core::Primitive::CoordinateAxes)
    }

    /// The video plane is the background of the background layer, so everything
    /// else composites over it. It is a scene *setting* now, not a leading
    /// drawable (#204).
    #[test]
    fn the_video_plane_is_always_the_background() {
        let (background, _, _) = placement_scenes(None, None, false, None, &SceneState::default());
        assert_eq!(
            background.background().frame,
            Some(trd_core::FrameFit::Stretch)
        );
    }

    /// Selecting the quad reveals its floor grid + basis axes; deselecting hides
    /// them but keeps the outline. The video plane rides on the background, so it
    /// is not one of the counted objects.
    #[test]
    fn selecting_the_quad_adds_its_grid_and_axes() {
        let quad = trd_core::Matrix4::IDENTITY;
        let state = SceneState::default();

        let (unselected, _, _) = placement_scenes(Some(quad), Some(quad), false, None, &state);
        assert_eq!(unselected.objects().len(), 1, "quad outline only");

        let (selected, _, _) = placement_scenes(Some(quad), Some(quad), true, None, &state);
        assert_eq!(selected.objects().len(), 3, "+ floor grid + basis axes");
        assert!(selected.objects().iter().any(is_axes));
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

    /// Without a placed object there is no foreground at all — the editor still
    /// shows the video and the quad.
    #[test]
    fn a_video_only_frame_has_an_empty_foreground() {
        let (background, foreground, overlay) =
            placement_scenes(None, None, false, None, &SceneState::default());
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
