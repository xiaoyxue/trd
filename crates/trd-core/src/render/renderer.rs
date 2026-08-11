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
    Draw, DrawableObject, FrameParams, GpuContext, Mesh, OffscreenTarget, PickTarget,
    SceneRenderer, OFFSCREEN_FORMAT,
};
use crate::math::Matrix4;
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

/// A persistent GPU context that renders one [`FrameParams`] to tightly-packed
/// row-major RGBA bytes (`width*height*4`) per call.
pub struct Renderer {
    gpu: Arc<GpuContext>,
    renderer: SceneRenderer,
    /// The shared offscreen render target + readback buffer (#103, Part B).
    target: OffscreenTarget,
    /// The object-id picking target (#141), created lazily on the first
    /// [`pick`](Self::pick) call and resized to track the render size. `None`
    /// until a front-end actually picks, so the headless CLI never allocates it.
    pick_target: Option<PickTarget>,
}

impl Renderer {
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
                limits: crate::LimitsPreset::Downlevel,
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
    /// in G, metallic in B), sampled by [`RenderMode::Pbr`](crate::RenderMode::Pbr)
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
    /// normal in [`RenderMode::Pbr`](crate::RenderMode::Pbr). Out-of-range ids are
    /// ignored.
    pub fn set_mesh_normal_texture(
        &mut self,
        mesh_id: usize,
        texture: &dyn crate::texture::Texture,
    ) {
        self.renderer.set_mesh_normal_texture(mesh_id, texture);
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
    /// [`RenderMode::Pbr`] meshes. Delegates to [`SceneRenderer::set_env_map`]; the
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
        let (w, h) = (self.target.width(), self.target.height());
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
