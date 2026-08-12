//! [`Renderer`] — the persistent offscreen render harness (#134).
//!
//! Owns a [`GpuContext`], a [`SceneRenderer`] and the shared [`OffscreenTarget`]
//! readback plumbing, plus the CLI overlay toggles
//! (`show_aabb`/`show_axes`/`show_local_axes`/`show_local_grid`) and the mesh-pass
//! MSAA sample count, rendering one [`FrameParams`] to tightly-packed row-major
//! RGBA bytes per call.
//!
//! Formerly `BatchRenderer`. "Batch" there meant *batch-mode headless output* and
//! described nothing about the type — instanced batching lives entirely in
//! [`SceneRenderer`] (`batch.rs`) — while colliding with that real meaning. The
//! name now belongs to one concept: grouping draws into instanced commands
//! (#180).
use std::sync::Arc;

use super::{
    FrameParams, GpuContext, Mesh, OffscreenTarget, OnscreenTarget, PickTarget, RenderTarget,
    SceneLayer, SceneRenderer, Viewport, OFFSCREEN_FORMAT,
};
use crate::math::Matrix4;
use crate::visual::{Draw, DrawableObject};
use thiserror::Error;

/// Errors constructing or driving a [`Renderer`].
///
/// Its own type rather than the stream module's: the renderer is
/// platform-neutral, while `stream` is native-only (`std::io` piping), so
/// borrowing that error would have kept the renderer off wasm — which is exactly
/// what blocked the browser from reusing it (#180). `StreamError` converts from
/// this, so `run_stream`'s error surface is unchanged.
#[derive(Debug, Error)]
pub enum RenderError {
    /// The requested render size is zero or would overflow.
    #[error("invalid render dimensions {width}x{height}: {reason}")]
    InvalidDimensions {
        width: u32,
        height: u32,
        reason: &'static str,
    },
    /// GPU device/adapter acquisition or read-back failed.
    #[error("render failed: {0}")]
    Gpu(String),
    /// The offscreen target could not be created or read back.
    #[error(transparent)]
    Offscreen(#[from] super::OffscreenError),
}

/// Validates a render size before anything is allocated for it.
pub(crate) fn check_dimensions(width: u32, height: u32) -> Result<u32, RenderError> {
    const BYTES_PER_PIXEL: u32 = 4;
    let pixels = width
        .checked_mul(height)
        .filter(|&p| p > 0 && p <= i32::MAX as u32)
        .ok_or(RenderError::InvalidDimensions {
            width,
            height,
            reason: "width*height must be non-zero and <= i32::MAX",
        })?;
    width
        .checked_mul(BYTES_PER_PIXEL)
        .ok_or(RenderError::InvalidDimensions {
            width,
            height,
            reason: "row byte size overflows u32",
        })?;
    Ok(pixels)
}

/// The render harness: a persistent GPU context plus a [`SceneRenderer`], drawing
/// one [`FrameParams`] + [`Scene`](crate::Scene) per call into a [`RenderTarget`].
///
/// Generic over **where the frame lands**, because that is the only thing the two
/// kinds of front-end disagree about (#180):
///
/// * `Renderer<OffscreenTarget>` — the default. Renders to a texture and reads it
///   back to RGBA bytes ([`render_scene`](Self::render_scene)): the headless CLI,
///   the GUI's egui texture, the browser's offscreen surface.
/// * `Renderer<OnscreenTarget>` — renders straight into a swapchain texture and
///   presents it ([`present_scene`](Self::present_scene)): the native window and
///   the browser canvas.
///
/// Everything else — uploads, materials, lighting, mesh count, picking — is shared
/// and lives in the `impl<T: RenderTarget>` block.
pub struct Renderer<T = OffscreenTarget> {
    gpu: Arc<GpuContext>,
    renderer: SceneRenderer,
    /// Where the frame lands: an offscreen texture + read-back buffer, or a
    /// surface swapchain.
    target: T,
    /// The object-id picking target (#141), created lazily on the first
    /// [`pick`](Self::pick) call and resized to track the render size. `None`
    /// until a front-end actually picks, so the headless CLI never allocates it.
    pick_target: Option<PickTarget>,
}

impl Renderer<OffscreenTarget> {
    /// Builds the GPU context (instance/adapter/device/pipeline/target/readback)
    /// once for a fixed `width` x `height`, rendering the `meshes` of the stream's
    /// leading mesh table, applying each mesh's [`Mesh::preview_transform`]
    /// (center + uniform scale-to-fit) beneath its per-frame model so an
    /// arbitrary-unit asset renders centered and at a reasonable size. Per-frame
    /// draw lists place instances of these meshes by index. The mesh pass renders
    /// at 4× MSAA; use [`with_meshes_sample_count`](Self::with_meshes_sample_count)
    /// to override (e.g. `1` = no MSAA).
    pub async fn with_meshes(
        width: u32,
        height: u32,
        meshes: &[Mesh],
    ) -> Result<Self, RenderError> {
        Self::with_meshes_sample_count(width, height, meshes, crate::render::MSAA_SAMPLE_COUNT)
            .await
    }

    /// Like [`with_meshes`](Self::with_meshes) but with an explicit mesh-pass MSAA
    /// `sample_count` (`4` = anti-aliased, `1` = single-sampled / no MSAA).
    pub async fn with_meshes_sample_count(
        width: u32,
        height: u32,
        meshes: &[Mesh],
        sample_count: u32,
    ) -> Result<Self, RenderError> {
        let base_models: Vec<Matrix4> = meshes
            .iter()
            .map(|mesh| {
                mesh.preview_transform(crate::DEFAULT_PREVIEW_TARGET)
                    .matrix()
            })
            .collect();
        Self::new_async(width, height, meshes, &base_models, sample_count).await
    }

    async fn new_async(
        width: u32,
        height: u32,
        meshes: &[Mesh],
        base_models: &[Matrix4],
        sample_count: u32,
    ) -> Result<Self, RenderError> {
        // Guard against zero / overflow before allocating (device limits below).
        check_dimensions(width, height)?;

        let instance = crate::create_instance();
        // The headless CLI keeps its historical conservative limits + memory hint
        // (Downlevel 2048 cap, MemoryUsage) so the golden render stays
        // byte-identical; power preference is HighPerformance (the default).
        let gpu = crate::GpuContext::request(
            &instance,
            &crate::GpuRequest {
                limits: super::LimitsPreset::Downlevel,
                memory_hints: wgpu::MemoryHints::MemoryUsage,
                ..Default::default()
            },
        )
        .await
        .map_err(|e| RenderError::Gpu(e.to_string()))?;
        let format = OFFSCREEN_FORMAT;
        let renderer = SceneRenderer::with_sample_count(
            gpu.clone(),
            format,
            meshes,
            base_models,
            sample_count,
        );

        // The shared offscreen harness owns the render target + readback buffer
        // and re-validates the size against the adapter's max dimension.
        let target = OffscreenTarget::new(&gpu.device, width, height)?;

        Ok(Self {
            gpu,
            renderer,
            target,
            pick_target: None,
        })
    }

    /// Builds the harness on an **already-created** [`GpuContext`], for callers
    /// that own the device before they own the meshes.
    ///
    /// A streaming front-end learns its meshes from the wire, long after it must
    /// report whether the GPU is usable at all, so it creates the device eagerly
    /// and the renderer lazily. [`with_meshes`](Self::with_meshes) requests its
    /// own device and is the right constructor everywhere else.
    pub fn with_gpu(
        gpu: Arc<GpuContext>,
        width: u32,
        height: u32,
        meshes: &[Mesh],
    ) -> Result<Self, RenderError> {
        check_dimensions(width, height)?;
        let renderer = SceneRenderer::auto_fit(gpu.clone(), OFFSCREEN_FORMAT, meshes);
        let target = OffscreenTarget::new(&gpu.device, width, height)?;
        Ok(Self {
            gpu,
            renderer,
            target,
            pick_target: None,
        })
    }
    /// Renders `scene` under `params`, returning tightly-packed row-major RGBA
    /// bytes (`width * height * 4`).
    ///
    /// The caller assembles the scene — typically with
    /// [`scene_with_overlays`](crate::scene_with_overlays), which turns a wire
    /// draw list plus [`RenderOptions`](crate::RenderOptions) into exactly the
    /// same `Scene` every other front-end renders. The renderer keeps no
    /// mode/overlay state of its own (#180): what to draw is entirely the scene.
    pub async fn render_scene(
        &mut self,
        params: FrameParams,
        scene: &[DrawableObject],
    ) -> Result<Vec<u8>, RenderError> {
        Ok(self
            .target
            .render(&self.gpu, &mut self.renderer, params, scene)
            .await?)
    }

    /// Renders `layers` back-to-front, returning tightly-packed row-major RGBA.
    ///
    /// The first layer clears; every later one composites over the accumulated
    /// color with depth cleared. Use this when one frame is not one camera's
    /// scene — the video editor draws the video plane through the background
    /// frame's calibration, the placed object through the placement frame's, then
    /// its gizmos on top. A single layer is exactly
    /// [`render_scene`](Self::render_scene).
    pub async fn render_layers(
        &mut self,
        layers: &[SceneLayer<'_>],
    ) -> Result<Vec<u8>, RenderError> {
        Ok(self
            .target
            .render_layers(&self.gpu, &mut self.renderer, layers)
            .await?)
    }
}

/// The parts that do not care **where** the frame lands: uploads, materials,
/// mesh count, sizing, picking. Every target shares them.
impl<T: RenderTarget> Renderer<T> {
    /// Builds the harness around an **existing** device and render target,
    /// building the scene renderer for whatever format that target wants.
    ///
    /// This is the on-screen constructor: a live-surface shell owns its
    /// `wgpu::Surface` (it needs the window/canvas to create one) and wraps it in
    /// an [`OnscreenTarget`](super::OnscreenTarget), then hands it here.
    pub fn with_target(gpu: Arc<GpuContext>, target: T, meshes: &[Mesh]) -> Self {
        let renderer = SceneRenderer::auto_fit(gpu.clone(), target.view_format(), meshes);
        Self {
            gpu,
            renderer,
            target,
            pick_target: None,
        }
    }

    /// The render target this renderer draws into.
    pub fn target(&self) -> &T {
        &self.target
    }

    /// The render target, mutably — the seam a live-surface front-end needs to
    /// reconfigure or replace its swapchain. Surface recovery is a **front-end**
    /// policy (native defers to a redraw, the browser recreates the surface from
    /// its canvas), so the harness exposes the target instead of modelling every
    /// platform's policy (#180).
    pub fn target_mut(&mut self) -> &mut T {
        &mut self.target
    }

    /// Unwraps the harness back into its render target, dropping the scene
    /// renderer's mesh store. A streaming front-end calls this when a new stream's
    /// mesh table arrives and the meshes must be rebuilt around the **same**
    /// surface — recreating the surface instead would lose the swapchain.
    pub fn into_target(self) -> T {
        self.target
    }

    /// The size of the object-id pick target, or `None` if nothing has been
    /// picked yet (it is allocated on the first [`pick`](Self::pick)). Diagnostic
    /// only — front-ends surface it in their debug panels.
    pub fn pick_target_size(&self) -> Option<(u32, u32)> {
        self.pick_target.as_ref().map(|_| {
            let viewport = self.target.viewport();
            (viewport.width, viewport.height)
        })
    }

    /// The current render size in pixels.
    pub fn viewport(&self) -> Viewport {
        self.target.viewport()
    }

    /// Resizes the render target to `width` x `height`. The pick target follows
    /// on the next [`pick`](Self::pick).
    pub fn resize(&mut self, width: u32, height: u32) -> Result<(), RenderError> {
        check_dimensions(width, height)?;
        self.target.resize(&self.gpu, width, height)?;
        Ok(())
    }

    /// The number of loaded meshes; valid [`Draw::mesh_id`]s are `0..mesh_count`.
    pub fn mesh_count(&self) -> usize {
        self.renderer.mesh_count()
    }

    /// Binds `texture` as the source sampled by [`RenderMode::Textured`] meshes
    /// (`0.0.4`). Delegates to [`SceneRenderer::set_texture`]; the image is
    /// (re)uploaded on the next `render`.
    pub fn set_texture(&mut self, texture: &dyn crate::texture::Texture) {
        self.renderer.set_texture(texture);
    }

    /// Binds `texture` as the albedo of mesh `mesh_id` — a **per-object** diffuse
    /// for multi-object scenes (#141). Delegates to
    /// [`SceneRenderer::set_mesh_texture`]; out-of-range ids are ignored.
    pub fn set_mesh_texture(&mut self, mesh_id: usize, texture: &dyn crate::texture::Texture) {
        self.renderer.set_mesh_texture(mesh_id, texture);
    }

    /// Binds mesh `mesh_id`'s **metallic-roughness** map (glTF packing: roughness
    /// in G, metallic in B), sampled by [`RenderMode::Shaded`](crate::RenderMode::Shaded)
    /// in place of the scalar material values. Out-of-range ids are ignored.
    pub fn set_mesh_metallic_roughness_texture(
        &mut self,
        mesh_id: usize,
        texture: &dyn crate::texture::Texture,
    ) {
        self.renderer
            .set_mesh_metallic_roughness_texture(mesh_id, texture);
    }

    /// Binds mesh `mesh_id`'s **tangent-space normal** map, perturbing the shading
    /// normal in [`RenderMode::Shaded`](crate::RenderMode::Shaded). Out-of-range ids are
    /// ignored.
    pub fn set_mesh_normal_texture(
        &mut self,
        mesh_id: usize,
        texture: &dyn crate::texture::Texture,
    ) {
        self.renderer.set_mesh_normal_texture(mesh_id, texture);
    }

    /// Uploads the background image composited beneath the scene by a
    /// [`DrawableObject::FramePlane`](crate::DrawableObject::FramePlane) draw
    /// (#63). `rgba` is tightly packed row-major `width * height * 4`.
    pub fn update_frame_texture_rgba(&mut self, rgba: &[u8], width: u32, height: u32) {
        self.renderer.update_frame_texture_rgba(rgba, width, height);
    }

    /// Sets the [`DisneyMaterial`](crate::DisneyMaterial) applied to every PBR mesh.
    pub fn set_disney_material(&mut self, material: crate::DisneyMaterial) {
        self.renderer.set_disney_material(material);
    }

    /// Sets one mesh's Disney material.
    pub fn set_mesh_disney_material(&mut self, mesh_id: usize, material: crate::DisneyMaterial) {
        self.renderer.set_mesh_disney_material(mesh_id, material);
    }

    /// Sets scene lighting controls shared by every PBR object.
    pub fn set_lighting(&mut self, lighting: crate::Lighting) {
        self.renderer.set_lighting(lighting);
    }

    /// Sets image-based-lighting controls for every PBR object.
    pub fn set_image_based_lighting(&mut self, ibl: crate::ImageBasedLighting) {
        self.renderer.set_image_based_lighting(ibl);
    }

    /// Sets one mesh's image-based-lighting controls.
    pub fn set_mesh_image_based_lighting(
        &mut self,
        mesh_id: usize,
        ibl: crate::ImageBasedLighting,
    ) {
        self.renderer.set_mesh_image_based_lighting(mesh_id, ibl);
    }

    /// Sets the output transform of every PBR object.
    pub fn set_tone_mapping(&mut self, tone_mapping: crate::ToneMapping) {
        self.renderer.set_tone_mapping(tone_mapping);
    }

    /// Sets one mesh's output transform.
    pub fn set_mesh_tone_mapping(&mut self, mesh_id: usize, tone_mapping: crate::ToneMapping) {
        self.renderer.set_mesh_tone_mapping(mesh_id, tone_mapping);
    }

    /// Selects a diagnostic PBR output for one mesh.
    pub fn set_mesh_pbr_debug_view(&mut self, mesh_id: usize, debug_view: crate::PbrDebugView) {
        self.renderer.set_mesh_pbr_debug_view(mesh_id, debug_view);
    }

    /// Binds `env` as the equirectangular HDR environment map reflected by
    /// [`RenderMode::Shaded`] meshes. Delegates to [`SceneRenderer::set_env_map`]; the
    /// probe is (re)uploaded on the next `render`.
    pub fn set_env_map(&mut self, env: crate::EnvMapData) {
        self.renderer.set_env_map(env);
    }

    /// Uploads `image` as the **background frame texture** (#63) sampled by a
    /// [`DrawableObject::FramePlane`]. The GPU texture is reused across frames
    /// (grown only on a resolution change). Call before a
    /// [`render_scene`](Self::render_scene) with a `FramePlane` drawable to composite the
    /// image beneath the mesh scene.
    pub fn update_frame_texture(&mut self, image: &crate::texture::ImageData) {
        self.renderer
            .update_frame_texture_rgba(&image.rgba, image.width, image.height);
    }

    /// **Object-id picking** (#141): renders `draws` through the flat id-color
    /// pass at the current render size and returns the **0-based index into
    /// `draws`** of the object under pixel `(x, y)`, or `None` for the background
    /// (or an out-of-bounds coordinate). The pass is single-sampled and
    /// depth-tested, so the nearest object wins and ids are never blended — the
    /// "color index" method, no ray-marching. The lazily-created pick target
    /// tracks the display size ([`resize`](Self::resize) keeps it in sync).
    pub async fn pick(
        &mut self,
        params: FrameParams,
        draws: &[Draw],
        x: u32,
        y: u32,
    ) -> Option<u32> {
        let Viewport {
            width: w,
            height: h,
        } = self.target.viewport();
        match self.pick_target.as_mut() {
            Some(target) => target.resize(&self.gpu.device, w, h),
            None => self.pick_target = Some(PickTarget::new(&self.gpu.device, w, h)),
        }
        let target = self.pick_target.as_ref()?;
        target
            .pick(&self.gpu, &mut self.renderer, params, draws, x, y)
            .await
    }
}

/// What one on-screen frame did.
///
/// wgpu's surface acquisition can fail in ways only the **front-end** knows how to
/// recover from — the native window defers to the next redraw, while the browser
/// recreates the surface from its canvas — so [`present_scene`] reports what
/// happened instead of baking in a policy (#180).
///
/// [`present_scene`]: Renderer::present_scene
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PresentOutcome {
    /// The frame was drawn and presented.
    Presented,
    /// The frame was drawn and presented, but the surface no longer optimally
    /// matches its configuration — wgpu still hands back a usable texture, so the
    /// frame is **not** lost; reconfigure before the next one.
    PresentedSuboptimal,
    /// Nothing was drawn: the surface configuration is stale (a resize or a
    /// minimise). Reconfigure, then draw again.
    Outdated,
    /// Nothing was drawn: the surface was lost. Recreate it, then draw again.
    Lost,
    /// Nothing was drawn, for a transient reason. Skipping the frame is correct;
    /// the next one is expected to succeed.
    Skipped(SurfaceSkip),
}

/// Why a frame was skipped without being drawn — the transient half of
/// [`PresentOutcome`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SurfaceSkip {
    /// Acquisition timed out.
    Timeout,
    /// The surface is not visible (e.g. an occluded or hidden window).
    Occluded,
    /// The surface failed validation.
    Validation,
}

impl Renderer<OnscreenTarget> {
    /// Acquires the surface's next texture, encodes `scene` under `params` into
    /// it, submits, and presents — reporting the outcome so the caller applies its
    /// own recovery policy.
    ///
    /// Nothing is drawn for [`Outdated`](PresentOutcome::Outdated),
    /// [`Lost`](PresentOutcome::Lost) or [`Skipped`](PresentOutcome::Skipped); the
    /// caller recovers through [`target_mut`](Self::target_mut) and draws again.
    pub fn present_scene(
        &mut self,
        params: FrameParams,
        scene: &[DrawableObject],
    ) -> PresentOutcome {
        let (texture, outcome) = match self.target.acquire() {
            wgpu::CurrentSurfaceTexture::Success(texture) => (texture, PresentOutcome::Presented),
            wgpu::CurrentSurfaceTexture::Suboptimal(texture) => {
                (texture, PresentOutcome::PresentedSuboptimal)
            }
            wgpu::CurrentSurfaceTexture::Outdated => return PresentOutcome::Outdated,
            wgpu::CurrentSurfaceTexture::Lost => return PresentOutcome::Lost,
            wgpu::CurrentSurfaceTexture::Timeout => {
                return PresentOutcome::Skipped(SurfaceSkip::Timeout)
            }
            wgpu::CurrentSurfaceTexture::Occluded => {
                return PresentOutcome::Skipped(SurfaceSkip::Occluded)
            }
            wgpu::CurrentSurfaceTexture::Validation => {
                return PresentOutcome::Skipped(SurfaceSkip::Validation)
            }
        };
        self.target
            .present(&self.gpu, &mut self.renderer, texture, params, scene);
        outcome
    }
}
