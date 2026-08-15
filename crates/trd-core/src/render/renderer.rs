//! [`Renderer`] — the persistent render harness (#134, #203).
//!
//! One type owns everything a frame needs: the mesh/gizmo/PBR/picking
//! pipelines, the decode-once [`MeshStore`], materials/lighting/IBL, and the
//! picking pipeline + its lazily-allocated pick target. It used to be split
//! into an outer `Renderer<T: RenderTarget>` — generic over *where* the frame
//! lands, wrapping the render target — around an inner `SceneRenderer` holding
//! everything else. That split had no real seam: every render entry point
//! already lived on a *concrete* `impl Renderer<OffscreenTarget>` /
//! `impl Renderer<OnscreenTarget>` block, so the generic only hid that, and
//! `SceneRenderer`'s outer wrapper had accrued 15 one-line forwarding setters as
//! its only remaining job. So there is one `Renderer`, and the render
//! **target** is a plain argument to the call that needs it (#203).
//!
//! **All the render behaviour is here.** The targets used to carry it —
//! `OffscreenTarget::render`/`draw_layers`/`read_pixels`, `OnscreenTarget::
//! present`/`acquire`/`reconfigure` — with this harness forwarding to them,
//! which had the ownership backwards: a swapchain handle knows nothing about
//! pipelines, materials or the mesh store, and everything it "did" immediately
//! borrowed the renderer straight back. Now a target is pure data
//! ([`RenderTarget`], [`SurfaceTarget`], [`TextureTarget`]), and drawing is:
//!
//! - [`render`](Renderer::render) — the **one** public entry, a match over
//!   [`RenderTarget`]: `render_surface` presents, `render_texture` encodes +
//!   submits. Both are private; the match is the only way in.
//! - [`read_pixels`](Renderer::read_pixels) — the asynchronous tail, taking the
//!   concrete [`TextureTarget`] so asking a *surface* for pixels is a type error
//!   rather than a runtime arm.
//! - [`draw_layers`](Renderer::draw_layers) /
//!   [`render_layers`](Renderer::render_layers) — the multi-camera composited
//!   draw the video editor needs, texture-only for the same reason.
//!
//! Target lifecycle (create / resize / reconfigure / replace) lives here too,
//! for the same reason: it is something *done to* a target, not something a
//! target does. The surface ones are associated functions taking a
//! `&wgpu::Device`, because a window is resized and repaired long before the
//! stream has delivered a mesh to build a `Renderer` from.
//!
//! Internally it is still a composition of a few cohesive parts, each with a
//! single job, so no one struct is a grab-bag of wgpu handles:
//! - [`RenderPipelines`] — the mesh/gizmo pipelines (filled/wireframe/textured/PBR)
//!   and [`SceneUniforms`] — the group-0 uniforms they read (the camera `P·V`,
//!   the gizmo viewport params, the per-mesh PBR slot array). Two types, because
//!   *what draws* and *what it reads* are different questions (#203); both live
//!   in `render_pipelines.rs` because both are `f(format, sample_count)` rather
//!   than a function of any one scene (#221 §2).
//! - [`MeshStore`] — the uploaded [`MeshGpu`]s (each owning its albedo, material
//!   maps, material, IBL, tone-map and debug view), the shared axes gizmo, and
//!   the growable per-instance model buffer; also walks a [`Scene`] into draw
//!   batches.
//! - [`BoundTexture`](super::BoundTexture) — the mesh albedo sampled by textured
//!   draws (#20).
//! - [`FramePlane`] — the background video frame plane (#63).
//! - [`Picking`](super::picking::Picking) — the object-id picking pipeline, its
//!   instance buffer and its `PickTarget` (#141), all three in `picking.rs`
//!   beside the target they create; the target is allocated lazily on the first
//!   [`pick`](Self::pick) call so a headless CLI stream never pays for it. Like
//!   every other target it is **data**: staging, encoding and reading it back are
//!   `Renderer` methods, so no target drives the renderer (#235 R4).
//!
//! **The draw loop is a dispatch, not a state machine** (#204). `encode_pass`
//! walks the batched commands and hands each to the `record` body of its
//! [`Primitive`] — one line per arm — and every body sets its own pipeline and
//! bind groups at entry while restoring nothing at exit, so no body may depend
//! on pass state another one left behind. The bodies and that rule are on the
//! second `impl Renderer` block below.
//!
//! Formerly `BatchRenderer`. "Batch" there meant *batch-mode headless output* and
//! described nothing about the type — instanced batching lives entirely in
//! this module (`draw_command.rs`) — while colliding with that real meaning. The
//! name now belongs to one concept: grouping draws into instanced commands
//! (#180).

use std::ops::Range;
use std::sync::Arc;

use super::bound_material_maps::BoundMaterialMaps;
use super::buffer::{draw_indexed, draw_vertices, InstanceBuffer, VertexGeometry};
use super::draw_command::{build_batches, Batches};
use super::environment::{EnvBackgroundSettings, Environment};
use super::frame_plane::FramePlane;
use super::gizmo::GizmoGeometry;
use super::mesh_store::MeshStore;
use super::picking::{PickTarget, Picking};
use super::*;
use super::{Draw, GridPlane, Primitive, RenderMode, Scene};
use crate::material::DisneyMaterial;
use crate::math::Matrix4;
use crate::output::tightly_pack_rgba;
use crate::texture::Texture;
use crate::Camera;
use futures_channel::oneshot;
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
    /// The mesh set a constructor was handed cannot be rendered: it is empty, or
    /// its base models do not pair with it one-to-one.
    ///
    /// A `Result` rather than the panic it used to be (#235 R8): the mesh set
    /// arrives from the *wire* in every streaming front-end, so "there were no
    /// meshes in that stream" is an input error the shell should report, not a
    /// bug it should abort on.
    #[error("invalid mesh set: {reason}")]
    InvalidMeshSet {
        /// What was wrong with it.
        reason: String,
    },
    /// GPU device/adapter acquisition or read-back failed.
    #[error("render failed: {0}")]
    Gpu(String),
    /// The render target could not be created or read back.
    #[error(transparent)]
    Target(#[from] super::TargetError),
    /// A frame could not be drawn onto a live surface.
    ///
    /// Folded in so [`Renderer::render`] has **one** error type across both
    /// target kinds (#203); a shell matches this variant to reach
    /// [`SurfaceError::repair`] and apply its own recovery policy.
    #[error(transparent)]
    Surface(#[from] SurfaceError),
    /// The frame's camera columns are malformed (CV and CG forms mixed, or an
    /// incomplete CG look-at), so no camera could be resolved.
    #[error(transparent)]
    CameraForm(#[from] super::CameraFormError),
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

/// The render harness: pipelines, decode-once mesh store, materials/lighting,
/// and picking, drawing one frame's camera and [`Scene`](crate::Scene) per call
/// into a caller-supplied render target.
///
/// * [`render`](Self::render) — the one render entry, a match over
///   [`RenderTarget`]: a [`SurfaceTarget`] is drawn into and **presented** (the
///   native window, the browser canvas); a [`TextureTarget`] is drawn into and
///   submitted (the headless CLI, the GUI's egui texture, the browser's
///   offscreen surface), to be collected with [`read_pixels`](Self::read_pixels).
/// * [`render_layers`](Self::render_layers) / [`draw_layers`](Self::draw_layers) /
///   [`render_params`](Self::render_params) — the texture-target compositions:
///   multi-camera layering and wire-driven camera resolution.
///
/// The target is a plain argument rather than a type parameter or an owned field
/// (#203), and it carries **no** behaviour of its own — creating, resizing,
/// drawing into, presenting and reading back a target are all methods here,
/// because they need the pipelines, the GPU context and the mesh store this type
/// owns. Uploads, materials, lighting, mesh count and picking are shared by
/// every caller regardless of target.
pub struct Renderer {
    pipelines: RenderPipelines,
    /// The group-0 uniforms every pass binds: the camera `P·V`, the gizmo
    /// viewport params, and the per-mesh PBR slot array.
    uniforms: SceneUniforms,
    /// The HDR environment subsystem: the probe reflected by
    /// [`RenderMode::Shaded`] draws **and** the pipeline drawing it as the
    /// background sky. One type, like its sibling `FramePlane` (#221 §5).
    environment: Environment,
    /// The caller's uploaded meshes, fixed at construction.
    meshes: MeshStore,
    /// The constant overlay geometry (axes, grids, quad outlines, shadow quad):
    /// a function of nothing, built once.
    gizmos: GizmoGeometry,
    /// The per-frame model matrices every draw kind is instanced through,
    /// rewritten each `encode` and grown on demand (#222).
    instances: InstanceBuffer<InstanceRaw>,
    /// The batcher's scratch, reused across frames (#235 R6): `build_batches`
    /// clears and refills it, so a steady-state frame reuses the capacity of the
    /// previous one instead of allocating three vectors per frame. It is scratch,
    /// not state — nothing outside `encode_pass` reads it between frames.
    batches: Batches,
    frame_plane: FramePlane,
    /// The mesh pass's depth attachment, (re)created lazily in `encode` to match
    /// the viewport. Gives solid (filled/textured) meshes real z-occlusion.
    depth: Option<DepthTarget>,
    /// The mesh pass's color attachment: the format and sample count every
    /// pipeline was built for, plus the multisampled target (re)created lazily
    /// in `encode` to match the viewport. When MSAA is on the pass renders into
    /// that target and resolves into the caller's single-sample `view`, so every
    /// front-end gets multisampled mesh/arrowhead edges transparently; gizmo
    /// lines add analytic AA separately. With MSAA off the pass renders straight
    /// into `view`. One owner rather than three loose fields, so "same format,
    /// same sample count" is structural (#221 §3).
    msaa: MsaaColor,
    /// The shared GPU context. Retained so `encode` can grow GPU resources and
    /// the setters can upload immediately, without the caller threading handles
    /// through every call. Holding the whole context (rather than a bare device)
    /// is what lets the &self.gpu.queue live here too, which is why uploads no longer have
    /// to be deferred to `encode` (#180).
    gpu: Arc<GpuContext>,
    /// Everything the object-id picking pass (#141) needs: its pipeline, its
    /// instance buffer and its lazily-created target.
    picking: Picking,
    /// Whether the per-mesh PBR slot array still matches the meshes' appearance
    /// (#235 R5).
    ///
    /// A slot's contents — material / IBL / tone map / debug view — depend on **no
    /// per-frame value** (the camera and light rig live in the once-per-frame
    /// scene uniform, #182), so re-uploading all of them every frame is redundant
    /// whenever nothing has changed. Appearance is renderer-owned state written
    /// through the setters below, so "changed" is exactly "a setter ran": each
    /// flips this, `prepare_frame` clears it after rewriting the slots.
    slots_dirty: bool,
}

impl Renderer {
    /// Constructs a `Renderer` that derives each mesh's base (preview) model
    /// automatically via [`Mesh::preview_transform`]
    /// ([`crate::DEFAULT_PREVIEW_TARGET`]) — center + uniform scale-to-fit — so an
    /// arbitrary-unit asset renders centered at a reasonable size. A convenience
    /// constructor over [`new`](Self::new); shared by the headless
    /// [`crate::run_stream`] and every on-screen front-end.
    pub fn auto_fit(
        gpu: Arc<GpuContext>,
        format: wgpu::TextureFormat,
        meshes: &[Mesh],
    ) -> Result<Self, RenderError> {
        let base_models: Vec<Matrix4> = meshes
            .iter()
            .map(|mesh| {
                mesh.preview_transform(crate::DEFAULT_PREVIEW_TARGET)
                    .matrix()
            })
            .collect();
        Self::new(gpu, format, meshes, &base_models)
    }

    /// Constructs a `Renderer` over one or more meshes, each paired with an
    /// explicit base (preview) model that is pre-multiplied beneath every
    /// per-frame instance model (`effective = model · base`). This is the primary
    /// constructor; [`auto_fit`](Self::auto_fit) derives the base models for you.
    /// A frame's [`Scene`] references these meshes by id (row index). The mesh
    /// pass renders at [`MSAA_SAMPLE_COUNT`]×; use
    /// [`with_sample_count`](Self::with_sample_count) to override (e.g. `1` = no
    /// MSAA).
    ///
    /// Returns [`RenderError::InvalidMeshSet`] if `meshes` is empty or
    /// `meshes`/`base_models` differ in length.
    pub fn new(
        gpu: Arc<GpuContext>,
        format: wgpu::TextureFormat,
        meshes: &[Mesh],
        base_models: &[Matrix4],
    ) -> Result<Self, RenderError> {
        Self::with_sample_count(gpu, format, meshes, base_models, MSAA_SAMPLE_COUNT)
    }

    /// Like [`new`](Self::new), but with an explicit mesh-pass MSAA
    /// `sample_count`: `4` ([`MSAA_SAMPLE_COUNT`]) for multisampled edges, or `1`
    /// to render single-sampled. Gizmo lines retain their shader-based analytic AA
    /// at `1`; mesh silhouettes and hardware wireframes do not. All pipelines and
    /// the depth/color attachments are built for this count.
    ///
    /// Returns [`RenderError::InvalidMeshSet`] if `meshes` is empty,
    /// `meshes`/`base_models` differ in length, or `sample_count` is 0.
    pub fn with_sample_count(
        gpu: Arc<GpuContext>,
        format: wgpu::TextureFormat,
        meshes: &[Mesh],
        base_models: &[Matrix4],
        sample_count: u32,
    ) -> Result<Self, RenderError> {
        if meshes.is_empty() {
            return Err(RenderError::InvalidMeshSet {
                reason: "a renderer needs at least one mesh".into(),
            });
        }
        if meshes.len() != base_models.len() {
            return Err(RenderError::InvalidMeshSet {
                reason: format!(
                    "{} mesh(es) but {} base model(s) — they pair one-to-one",
                    meshes.len(),
                    base_models.len()
                ),
            });
        }
        if sample_count == 0 {
            return Err(RenderError::InvalidMeshSet {
                reason: "sample_count must be >= 1 (1 = no MSAA)".into(),
            });
        }

        // One shared group-1 albedo layout for the textured/PBR pipelines and
        // every per-mesh [`BoundTexture`] (each object skins with its own diffuse).
        let texture_layout = create_texture_bind_group_layout(&gpu.device);
        let material_maps_layout = BoundMaterialMaps::create_layout(&gpu.device);
        let environment = Environment::new(&gpu, format, sample_count);
        let (pipelines, uniforms) = create_render_pipelines(
            &gpu.device,
            format,
            &texture_layout,
            &material_maps_layout,
            environment.layout(),
            sample_count,
            meshes.len(),
        );
        let store = MeshStore::new(
            &gpu,
            meshes,
            base_models,
            &texture_layout,
            &material_maps_layout,
        );
        let gizmos = GizmoGeometry::new(&gpu);
        let instances =
            InstanceBuffer::new(&gpu.device, "trd mesh instance buffer", meshes.len() as u32);
        let frame_plane = FramePlane::new(&gpu.device, format, sample_count);
        let picking = Picking::new(&gpu.device, meshes.len());

        Ok(Self {
            pipelines,
            uniforms,
            environment,
            meshes: store,
            gizmos,
            instances,
            batches: Batches::default(),
            frame_plane,
            depth: None,
            msaa: MsaaColor::new(format, sample_count),
            gpu,
            picking,
            // Nothing has been uploaded yet, so the first frame writes them all.
            slots_dirty: true,
        })
    }

    /// Builds the harness for `meshes` (drawn by index), requesting its own GPU
    /// device, and a matching [`TextureTarget`] of `width` × `height`. Applies
    /// each mesh's [`Mesh::preview_transform`] (center + uniform scale-to-fit)
    /// beneath its per-frame model so an arbitrary-unit asset renders centered
    /// and at a reasonable size. Per-frame draw lists place instances of these
    /// meshes by index. The mesh pass renders at 4× MSAA; use
    /// [`with_meshes_sample_count`](Self::with_meshes_sample_count) to override
    /// (e.g. `1` = no MSAA).
    ///
    /// Returns the harness **and** the target it was sized for: the target is a
    /// call argument now, not a field the harness owns (#203), so the one
    /// constructor that used to build both hands both back rather than making
    /// every caller build the target separately.
    pub async fn with_meshes(
        width: u32,
        height: u32,
        meshes: &[Mesh],
    ) -> Result<(Self, TextureTarget), RenderError> {
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
    ) -> Result<(Self, TextureTarget), RenderError> {
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
    ) -> Result<(Self, TextureTarget), RenderError> {
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
        let format = TEXTURE_TARGET_FORMAT;
        let renderer =
            Self::with_sample_count(gpu.clone(), format, meshes, base_models, sample_count)?;

        // The target owns the render texture + readback buffer and re-validates
        // the size against the adapter's max dimension.
        let target = TextureTarget::new(&gpu.device, width, height)?;

        Ok((renderer, target))
    }

    /// Builds the harness on an **already-created** [`GpuContext`], and a
    /// matching [`TextureTarget`], for callers that own the device before they
    /// own the meshes.
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
    ) -> Result<(Self, TextureTarget), RenderError> {
        check_dimensions(width, height)?;
        let renderer = Self::auto_fit(gpu.clone(), TEXTURE_TARGET_FORMAT, meshes)?;
        let target = TextureTarget::new(&gpu.device, width, height)?;
        Ok((renderer, target))
    }

    /// The number of meshes this renderer can draw; valid mesh ids in a
    /// [`Primitive::Mesh`]/[`Primitive::AabbBox`] are in
    /// `0..mesh_count()`.
    pub fn mesh_count(&self) -> usize {
        self.meshes.len()
    }

    /// Binds `texture` as the albedo of **mesh 0** — the single-mesh /
    /// wire-protocol default sampled by [`RenderMode::Textured`]/[`RenderMode::Shaded`]
    /// draws (#20). For a multi-object scene, skin each object with
    /// [`set_mesh_texture`](Self::set_mesh_texture). The image is (re)uploaded
    /// lazily on the next [`render`](Self::render); until set it is
    /// 1×1 white.
    pub fn set_texture(&mut self, texture: &dyn Texture) {
        self.set_mesh_texture(0, texture);
    }

    /// Binds `texture` as the albedo of mesh `mesh_id` — so a multi-object scene
    /// skins each object with its **own** diffuse (#141). Out-of-range ids are
    /// ignored. The image uploads lazily on the next
    /// [`render`](Self::render).
    pub fn set_mesh_texture(&mut self, mesh_id: usize, texture: &dyn Texture) {
        if let Some(mesh) = self.meshes.get_mut(mesh_id) {
            mesh.texture.set(&self.gpu, texture);
        }
    }

    /// Binds a glTF metallic-roughness map (G=roughness, B=metallic) for mesh
    /// `mesh_id`, sampled by [`RenderMode::Shaded`] in place of the scalar
    /// material values. Out-of-range ids are ignored.
    pub fn set_mesh_metallic_roughness_texture(&mut self, mesh_id: usize, texture: &dyn Texture) {
        if let Some(mesh) = self.meshes.get_mut(mesh_id) {
            mesh.material_maps
                .set_metallic_roughness(&self.gpu, texture);
        }
    }

    /// Binds mesh `mesh_id`'s tangent-space glTF normal map, perturbing the
    /// shading normal in [`RenderMode::Shaded`]. Out-of-range ids are ignored.
    pub fn set_mesh_normal_texture(&mut self, mesh_id: usize, texture: &dyn Texture) {
        if let Some(mesh) = self.meshes.get_mut(mesh_id) {
            mesh.material_maps.set_normal(&self.gpu, texture);
        }
    }

    /// Sets the [`DisneyMaterial`] of **every** mesh — the single-mesh / global
    /// default. For a multi-object scene, give each object its own material with
    /// [`set_mesh_disney_material`](Self::set_mesh_disney_material). Takes effect
    /// on the next [`render`](Self::render).
    pub fn set_disney_material(&mut self, material: DisneyMaterial) {
        for mesh in self.meshes.iter_mut() {
            mesh.material = material.clone();
        }
        self.slots_dirty = true;
    }

    /// Sets the [`DisneyMaterial`] of mesh `mesh_id` only (#141) — so each
    /// object in a multi-object scene has its own metallic/roughness/base_color.
    /// Out-of-range ids are ignored. Takes effect on the next
    /// [`render`](Self::render).
    pub fn set_mesh_disney_material(&mut self, mesh_id: usize, material: DisneyMaterial) {
        if let Some(mesh) = self.meshes.get_mut(mesh_id) {
            mesh.material = material;
        }
        self.slots_dirty = true;
    }

    /// Sets image-based-lighting controls for every PBR object.
    pub fn set_image_based_lighting(&mut self, ibl: ImageBasedLighting) {
        for mesh in self.meshes.iter_mut() {
            mesh.ibl = ibl;
        }
        self.slots_dirty = true;
    }

    /// Sets image-based-lighting controls for one PBR object.
    pub fn set_mesh_image_based_lighting(&mut self, mesh_id: usize, ibl: ImageBasedLighting) {
        if let Some(mesh) = self.meshes.get_mut(mesh_id) {
            mesh.ibl = ibl;
        }
        self.slots_dirty = true;
    }

    /// Sets the per-object output transform of every PBR object.
    pub fn set_tone_mapping(&mut self, tone_mapping: ToneMapping) {
        for mesh in self.meshes.iter_mut() {
            mesh.tone_mapping = tone_mapping;
        }
        self.slots_dirty = true;
    }

    /// Sets the output transform of one PBR object.
    pub fn set_mesh_tone_mapping(&mut self, mesh_id: usize, tone_mapping: ToneMapping) {
        if let Some(mesh) = self.meshes.get_mut(mesh_id) {
            mesh.tone_mapping = tone_mapping;
        }
        self.slots_dirty = true;
    }

    /// Selects a diagnostic PBR output for one mesh.
    pub fn set_mesh_pbr_debug_view(&mut self, mesh_id: usize, debug_view: PbrDebugView) {
        if let Some(mesh) = self.meshes.get_mut(mesh_id) {
            mesh.debug_view = debug_view;
        }
        self.slots_dirty = true;
    }

    /// Binds `env` as the equirectangular HDR environment map reflected by
    /// [`RenderMode::Shaded`] draws. The probe is (re)uploaded lazily on the next
    /// [`render`](Self::render). Until set, PBR draws use no
    /// environment reflection (a 1×1 black probe keeps the bind group valid).
    pub fn set_env_map(&mut self, env: EnvMapData) {
        self.environment.set(&self.gpu, env);
    }

    /// Uploads `rgba` (tightly-packed, row-major `height`×`width`×4) as the
    /// **background frame texture** (#63) sampled by a scene whose
    /// [`Background::frame`](crate::Background::frame) is set. Delegates to
    /// [`FramePlane::upload_rgba`],
    /// which reuses the GPU texture across same-resolution frames.
    ///
    /// Panics if `rgba.len() != width * height * 4` or either dimension is zero.
    pub fn update_frame_texture_rgba(&mut self, rgba: &[u8], width: u32, height: u32) {
        self.frame_plane.upload_rgba(&self.gpu, rgba, width, height);
    }

    /// Uploads `image` as the **background frame texture** (#63) sampled by a
    /// scene whose [`Background::frame`](crate::Background::frame) is set. The
    /// GPU texture is reused across frames (grown only on a resolution change).
    /// Call before a [`render`](Self::render) of such a scene to composite the
    /// image beneath the mesh scene.
    pub fn update_frame_texture(&mut self, image: &crate::texture::ImageData) {
        self.update_frame_texture_rgba(&image.rgba, image.width, image.height);
    }

    /// Whether a background frame texture is currently bound (so a scene with a
    /// [`Background::frame`](crate::Background::frame) would render one).
    pub fn has_frame_texture(&self) -> bool {
        self.frame_plane.is_bound()
    }

    /// The size of the object-id pick target, or `None` if nothing has been
    /// picked yet (it is allocated on the first [`pick`](Self::pick)). Diagnostic
    /// only — front-ends surface it in their debug panels.
    pub fn pick_target_size(&self) -> Option<(u32, u32)> {
        self.picking.target_size()
    }

    // -----------------------------------------------------------------------
    // Render targets — creation, drawing, readback, lifecycle (#203).
    //
    // All of it lives here rather than on the target types: a target is a place
    // a frame lands, and everything one could "do" needs the pipelines, the mesh
    // store and the GPU context this harness owns.
    // -----------------------------------------------------------------------

    /// Allocates a [`TextureTarget`] of `width` × `height` on this harness's
    /// device.
    ///
    /// A texture target is always [`TEXTURE_TARGET_FORMAT`], so this is only
    /// valid for a harness built for that format (every offscreen front-end); a
    /// harness built for a surface's sRGB view format renders to that surface.
    pub fn create_texture_target(
        &self,
        width: u32,
        height: u32,
    ) -> Result<TextureTarget, RenderError> {
        Ok(TextureTarget::new(&self.gpu.device, width, height)?)
    }

    /// Wraps an already-created `surface` + `config` as a [`SurfaceTarget`],
    /// registering its sRGB view format and configuring it.
    ///
    /// Both live-surface shells create their surface *before* the stream has
    /// delivered a mesh, i.e. before a `Renderer` exists, so
    /// [`SurfaceTarget::new`] stays available for that case; this is the
    /// convenience for a shell that already has a harness.
    pub fn create_surface_target(
        &self,
        surface: wgpu::Surface<'static>,
        config: wgpu::SurfaceConfiguration,
    ) -> SurfaceTarget {
        SurfaceTarget::new(&self.gpu.device, surface, config)
    }

    /// Draws `scene` under `camera` into `target` — **the** render entry point,
    /// and the one place the two kinds of target are told apart (#203).
    ///
    /// Synchronous, because drawing is: only reading pixels back has to await.
    /// A [`SurfaceTarget`] is acquired, drawn into and **presented**; a
    /// [`TextureTarget`] is drawn into and submitted, and its pixels are
    /// collected separately with [`read_pixels`](Self::read_pixels) — which
    /// takes the concrete texture target, so asking a surface for pixels cannot
    /// even be written.
    ///
    /// `Ok(None)` — drawn, nothing to do (always the texture case).
    /// `Ok(Some(repair))` — presented, but repair the surface before the next
    /// frame. `Err(RenderError::Surface(e))` — **nothing was drawn**;
    /// [`SurfaceError::repair`] says what to do, and `None` there means the
    /// cause was transient and skipping this frame is the whole remedy.
    ///
    /// Whether the frame reached the screen and what the surface needs are
    /// deliberately separate axes (#203); returning a `Result` also makes the
    /// outcome `#[must_use]`, and a dropped "outdated" means the window never
    /// repaints again.
    ///
    /// The caller assembles the scene — typically with
    /// [`Scene::from_draws`](crate::Scene::from_draws), which turns a wire draw
    /// list plus [`RenderOptions`](crate::RenderOptions) into exactly the same
    /// `Scene` every other front-end renders. The renderer keeps no
    /// mode/overlay state of its own (#180): what to draw is entirely the scene.
    pub fn render(
        &mut self,
        camera: Camera,
        scene: &Scene,
        target: &mut RenderTarget,
    ) -> Result<Option<SurfaceRepair>, RenderError> {
        match target.kind_mut() {
            RenderTargetType::Surface(surface) => Ok(self.render_surface(camera, scene, surface)?),
            RenderTargetType::Texture(texture) => {
                self.render_texture(camera, scene, texture);
                Ok(None)
            }
        }
    }

    /// Acquires `target`'s next swapchain texture, encodes `scene` through
    /// `camera` onto its **sRGB view**, submits, and presents.
    ///
    /// Rendering through the sRGB view (not the raw surface format, which the
    /// browser usually prefers non-sRGB) is what keeps the window and the canvas
    /// byte-identical with the headless CLI.
    ///
    /// Private: reachable only through [`render`](Self::render), so no shell can
    /// grow its own idea of what presenting means.
    fn render_surface(
        &mut self,
        camera: Camera,
        scene: &Scene,
        target: &mut SurfaceTarget,
    ) -> Result<Option<SurfaceRepair>, SurfaceError> {
        let (texture, repair) = match target.surface().get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(texture) => (texture, None),
            // Presented, but the configuration no longer matches: the frame is on
            // screen, so this is `Ok` with a repair for the next one.
            wgpu::CurrentSurfaceTexture::Suboptimal(texture) => {
                (texture, Some(SurfaceRepair::Reconfigure))
            }
            wgpu::CurrentSurfaceTexture::Outdated => return Err(SurfaceError::Outdated),
            wgpu::CurrentSurfaceTexture::Lost => return Err(SurfaceError::Lost),
            wgpu::CurrentSurfaceTexture::Timeout => return Err(SurfaceError::Timeout),
            wgpu::CurrentSurfaceTexture::Occluded => return Err(SurfaceError::Occluded),
            wgpu::CurrentSurfaceTexture::Validation => return Err(SurfaceError::Validation),
        };
        let gpu = self.gpu.clone();
        let view = texture.texture.create_view(&wgpu::TextureViewDescriptor {
            format: Some(target.view_format()),
            ..Default::default()
        });
        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("trd onscreen frame"),
            });
        self.encode(&mut encoder, &view, camera, scene);
        gpu.queue.submit(Some(encoder.finish()));
        gpu.queue.present(texture);
        Ok(repair)
    }

    /// Encodes `scene` through `camera` into `target`'s texture and submits it,
    /// leaving the pixels on the GPU for [`read_pixels`](Self::read_pixels).
    ///
    /// Private, and the degenerate one-layer [`draw_layers`](Self::draw_layers);
    /// reachable through [`render`](Self::render).
    fn render_texture(&mut self, camera: Camera, scene: &Scene, target: &TextureTarget) {
        self.draw_layers(&[SceneLayer::new(camera, scene)], target);
    }

    /// Encodes and submits `layers` into `target` **without** reading it back.
    ///
    /// The first layer clears; every later one preserves the accumulated color
    /// while clearing depth, so it composites *over* what came before rather
    /// than z-fighting with it — that is what "layer" means here (an overlay
    /// pass), not a general depth-sorted scene split. An empty `layers` draws
    /// nothing.
    ///
    /// Each layer is submitted **separately**: instances are uploaded into one
    /// shared buffer, so the next layer's upload must not race the previous
    /// layer's draw.
    pub fn draw_layers(&mut self, layers: &[SceneLayer<'_>], target: &TextureTarget) {
        let gpu = self.gpu.clone();
        let view = target
            .texture()
            .create_view(&wgpu::TextureViewDescriptor::default());
        for (index, layer) in layers.iter().enumerate() {
            let mut encoder = gpu
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("trd texture target layer"),
                });
            if index == 0 {
                self.encode(&mut encoder, &view, layer.camera, layer.scene);
            } else {
                self.encode_overlay(&mut encoder, &view, layer.camera, layer.scene);
            }
            // Submitted per layer, not once at the end: instances share one
            // buffer, so the next layer's upload must not race this layer's draw.
            gpu.queue.submit(Some(encoder.finish()));
        }
    }

    /// Reads `target`'s **current contents** back as tightly-packed row-major
    /// RGBA (`width * height * 4` bytes).
    ///
    /// Takes the concrete [`TextureTarget`] rather than a [`RenderTarget`]:
    /// a swapchain frame is gone once presented, so "read the pixels of a
    /// surface" is a mistake the type system should catch instead of a runtime
    /// arm returning `None` (#203).
    ///
    /// `async` because the buffer map only resolves while the device is polled —
    /// which this does itself, natively by blocking and on wasm by yielding, so a
    /// caller cannot hang by forgetting to drive it. Reads whatever is in the
    /// texture, so calling it without a preceding draw yields the cleared (or
    /// stale) target rather than failing.
    pub async fn read_pixels(&self, target: &TextureTarget) -> Result<Vec<u8>, RenderError> {
        Ok(self.read_back(target).await?)
    }

    async fn read_back(&self, target: &TextureTarget) -> Result<Vec<u8>, TargetError> {
        let (device, queue) = (&self.gpu.device, &self.gpu.queue);
        let (width, height) = target.size();
        let padded_bytes_per_row = target.padded_bytes_per_row();
        let staging = target.staging();

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("trd texture target readback"),
        });
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: target.texture(),
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: staging,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(padded_bytes_per_row),
                    rows_per_image: Some(height),
                },
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );
        queue.submit(Some(encoder.finish()));

        let slice = staging.slice(..);
        let (sender, receiver) = oneshot::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });
        // Native blocks until the mapping completes; the browser kicks the queue
        // and lets the `.await` below yield. See `platform::poll_for_map`.
        super::platform::poll_for_map(device).map_err(|e| TargetError::Gpu(e.to_string()))?;
        receiver
            .await
            .map_err(|_| TargetError::Gpu("readback callback cancelled".to_owned()))?
            .map_err(|e| TargetError::Gpu(e.to_string()))?;

        let packed = match slice.get_mapped_range() {
            Ok(mapped) => tightly_pack_rgba(&mapped, width, height, padded_bytes_per_row)
                .map_err(TargetError::Output),
            Err(e) => Err(TargetError::Gpu(e.to_string())),
        };
        staging.unmap();
        packed
    }

    /// Renders `layers` back-to-front into `target`, then reads it back as
    /// tightly-packed row-major RGBA — [`draw_layers`](Self::draw_layers)
    /// followed by [`read_pixels`](Self::read_pixels).
    ///
    /// Use this when one frame is not one camera's scene: the video editor draws
    /// the video plane through the background frame's calibration, the placed
    /// object through the placement frame's, then its gizmos on top. A single
    /// layer is exactly what [`render`](Self::render) does to a texture target.
    pub async fn render_layers(
        &mut self,
        layers: &[SceneLayer<'_>],
        target: &TextureTarget,
    ) -> Result<Vec<u8>, RenderError> {
        self.draw_layers(layers, target);
        self.read_pixels(target).await
    }

    /// Renders one wire-decoded frame into `target` and reads it back, resolving
    /// the camera against **`target`'s own size** so the viewport cannot
    /// disagree with the attachments.
    ///
    /// `FrameParams` is a protocol type, so it stays out of the core signature
    /// (#203); this is the convenience for callers that decode a frame and render
    /// it immediately.
    pub async fn render_params(
        &mut self,
        params: FrameParams,
        scene: &Scene,
        target: &TextureTarget,
    ) -> Result<Vec<u8>, RenderError> {
        let camera = params.to_camera(target.viewport())?;
        self.render_layers(&[SceneLayer::new(camera, scene)], target)
            .await
    }

    /// Reallocates `target` at `width` × `height`.
    ///
    /// A texture target is a fixed-size allocation (texture + padded staging
    /// buffer), so resizing means rebuilding it — which re-runs
    /// [`TextureTarget::new`]'s zero / `max_texture_dimension_2d` checks.
    pub fn resize_texture_target(
        &self,
        target: &mut TextureTarget,
        width: u32,
        height: u32,
    ) -> Result<(), RenderError> {
        *target = self.create_texture_target(width, height)?;
        Ok(())
    }

    /// Updates `target`'s configured size and reconfigures its surface. Ignores a
    /// zero width or height (e.g. a minimized window).
    ///
    /// Associated rather than a method: a window is resized long before the
    /// stream has delivered a mesh to build a `Renderer` from, so requiring one
    /// here would silently skip the reconfigure and leave the surface stale
    /// (#203).
    pub fn resize_surface(
        device: &wgpu::Device,
        target: &mut SurfaceTarget,
        width: u32,
        height: u32,
    ) {
        if width > 0 && height > 0 {
            target.set_size(width, height);
            Self::reconfigure_surface(device, target);
        }
    }

    /// Reapplies `target`'s current configuration to its surface — the recovery
    /// step after an outdated/lost/suboptimal acquire
    /// ([`SurfaceRepair::Reconfigure`]).
    pub fn reconfigure_surface(device: &wgpu::Device, target: &SurfaceTarget) {
        target.surface().configure(device, target.config());
    }

    /// Swaps a freshly created surface into `target` and reconfigures it —
    /// [`SurfaceRepair::Recreate`], e.g. after the browser reports the canvas
    /// surface *lost*. The new surface must target the same canvas/window as the
    /// original.
    pub fn replace_surface(
        device: &wgpu::Device,
        target: &mut SurfaceTarget,
        surface: wgpu::Surface<'static>,
    ) {
        target.set_surface(surface);
        Self::reconfigure_surface(device, target);
    }

    /// Encodes one frame's [`Scene`] — an ordered list of [`DrawableObject`]s —
    /// under the shared camera `P·V` uniform.
    ///
    /// The steps read top-to-bottom: set the camera, walk the scene into
    /// per-geometry instance batches, upload them, size the depth buffer, then
    /// record the pass — the background frame plane first (depth-write off) so
    /// the mesh scene z-composites on top, then each batched draw. Instances are
    /// grouped by geometry so each buffer is drawn once over a contiguous range.
    /// Out-of-range `mesh_id`s are skipped (callers should validate first).
    ///
    /// `pub(crate)`: only this module's per-target draw functions call it; a
    /// front-end reaches it through [`render`](Self::render) (or
    /// [`draw_layers`](Self::draw_layers)) instead.
    pub(crate) fn encode(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        camera: Camera,
        scene: &Scene,
    ) {
        self.encode_pass(encoder, view, camera, scene, false);
    }

    /// Encodes a foreground scene while preserving color already rendered into
    /// `view` by an earlier pass in the same command encoder.
    pub(crate) fn encode_overlay(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        camera: Camera,
        scene: &Scene,
    ) {
        self.encode_pass(encoder, view, camera, scene, true);
    }

    /// Everything a frame needs **written and sized** before a pass can start:
    /// the camera and PBR uniforms, the batched instances, the depth/MSAA
    /// attachments, and the two background settings (#235 R8).
    ///
    /// Split out of `encode_pass` so the encode half reads as what it is — a
    /// dispatch over batched commands — instead of ~150 lines where the
    /// preparation and the recording are interleaved. It is also the whole of
    /// the `&mut self` work: everything after it borrows immutably.
    fn prepare_frame(&mut self, camera: Camera, scene: &Scene) {
        // 1. Camera P·V for this frame.
        self.uniforms.write_camera(&self.gpu.queue, camera);
        // 1b. Disney PBR uniform slots for this frame — one per mesh (each carries
        //     the shared camera/lights + that mesh's material, #141). Written
        //     unconditionally so a PBR draw always has a current material slot.
        let viewport = camera.viewport();
        //     The per-mesh **slots** are rewritten only when a setter has changed
        //     one since the last frame (#235 R5): their inputs carry no per-frame
        //     value, so an unchanged scene would upload the same bytes again.
        self.uniforms.write_pbr(
            &self.gpu.queue,
            camera,
            self.meshes.all(),
            scene.lighting(),
            self.environment.has_env(),
            self.slots_dirty,
        );
        self.slots_dirty = false;

        // 2. Walk the scene's objects once into per-geometry instance batches,
        //    then upload the flattened instance models (growing the buffer if
        //    needed). The backgrounds are not objects: they are settings on the
        //    scene, read directly below (#204).
        let background = *scene.background();
        // Split the borrow by field so the walk can fill the renderer's own
        // scratch while reading the mesh store (#235 R6) — the same shape S3 used
        // for the pick target: keep the buffer in `self`, don't move it out.
        let Self {
            batches, meshes, ..
        } = self;
        build_batches(batches, scene.objects(), |mesh_id| {
            meshes.get(mesh_id).map(|mesh| mesh.base_model)
        });
        self.instances.upload(&self.gpu, &self.batches.instances);

        // 3. Match the depth + (when MSAA is on) color attachments to the viewport
        //    (solid meshes z-occlude; the multisampled color, if any, is resolved
        //    into `view`).
        self.ensure_depth(viewport);
        self.msaa
            .ensure(&self.gpu.device, viewport.width, viewport.height);

        // 4. Background frame-plane fit for this viewport (no-op if the scene
        //    asks for no frame plane or no frame texture is bound yet).
        if let Some(fit) = background.frame {
            self.frame_plane.write_fit(&self.gpu.queue, fit, viewport);
        }

        // 5. Bind groups for each mesh's own albedo (#141) and material maps, and
        //    for the HDR environment map. Nothing is uploaded here: now that the
        //    renderer holds the &self.gpu.queue, every setter uploads immediately and the
        //    constructors upload their fallbacks, so `encode` only *reads* bind
        //    groups that are already valid (#180).
        if let Some(settings) = background.environment {
            self.environment.write_background(
                &self.gpu.queue,
                camera,
                EnvBackgroundSettings {
                    // The yaw comes from the scene's environment *light*, not
                    // from the background settings, so the sky and the
                    // reflections cannot disagree (#182).
                    rotation: scene.lighting().environment.rotation,
                    exposure: settings.exposure,
                    blur: settings.blur,
                    tonemap: settings.tonemap,
                },
            );
        }
    }

    /// The mesh pass's color attachment for this frame: with MSAA
    /// (`sample_count > 1`) the pass renders into the multisampled attachment
    /// and **resolves** into the caller's single-sample `view`, so every
    /// front-end (offscreen CLI, native window, wasm canvas) gets multisampled
    /// mesh/arrowhead edges; without MSAA (`sample_count == 1`) there is no MSAA
    /// target and the pass renders straight into `view` (no resolve).
    ///
    /// `load` is what happens to the existing contents — `Load` for a layer
    /// composited over an earlier one, `Clear` for the first.
    fn color_attachment<'a>(
        &'a self,
        view: &'a wgpu::TextureView,
        load: wgpu::LoadOp<wgpu::Color>,
    ) -> wgpu::RenderPassColorAttachment<'a> {
        match self.msaa.target() {
            Some(msaa) => wgpu::RenderPassColorAttachment {
                view: &msaa.view,
                depth_slice: None,
                resolve_target: Some(view),
                ops: wgpu::Operations {
                    load,
                    store: wgpu::StoreOp::Store,
                },
            },
            None => wgpu::RenderPassColorAttachment {
                view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load,
                    store: wgpu::StoreOp::Store,
                },
            },
        }
    }

    /// Draws one frame of `scene` into `view`: prepare, then dispatch the
    /// batched commands. `load_color` keeps the existing contents (a composited
    /// layer) instead of clearing to black (the first pass).
    fn encode_pass(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        camera: Camera,
        scene: &Scene,
        load_color: bool,
    ) {
        self.prepare_frame(camera, scene);

        let background = *scene.background();
        let depth_view = &self
            .depth
            .as_ref()
            .expect("prepare_frame sized the depth attachment")
            .view;
        let color_attachment = self.color_attachment(
            view,
            if load_color {
                wgpu::LoadOp::Load
            } else {
                wgpu::LoadOp::Clear(wgpu::Color::BLACK)
            },
        );
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("trd mesh pass"),
            color_attachments: &[Some(color_attachment)],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: depth_view,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(1.0),
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });

        // The two backgrounds are independent slots drawn in a fixed order (#204):
        // the environment probe first, then the frame plane over it — a scene may
        // carry either, both, or neither.
        if background.environment.is_some() {
            self.environment.draw_background(&mut pass);
        }

        // Draw the background frame plane next (#63): its own pipeline + group-0
        // bind, depth-write off, so the mesh scene composites on top. Only when
        // the scene requested one (and a frame texture is bound).
        if background.frame.is_some() {
            self.frame_plane.draw(&mut pass);
        }

        // 7. Dispatch each batched command to the `record` body of its primitive
        //    (see the `record` block below). One line per arm: the loop decides
        //    *what* is drawn and in which order, never *how*, and it holds no
        //    pass state of its own — every body sets what it needs at entry and
        //    restores nothing at exit (#204).
        for command in &self.batches.commands {
            let range = command.start..command.start + command.count;
            match command.primitive {
                Primitive::Mesh { mesh_id, mode } => {
                    self.record_mesh(&mut pass, mesh_id, mode, range)
                }
                Primitive::AabbBox { mesh_id } => self.record_aabb_box(&mut pass, mesh_id, range),
                Primitive::PlaneGrid { plane } => self.record_plane_grid(&mut pass, plane, range),
                Primitive::QuadOutline { selected } => {
                    self.record_quad_outline(&mut pass, selected, range)
                }
                Primitive::BlobShadow => self.record_blob_shadow(&mut pass, range),
                Primitive::CoordinateAxes => self.record_coordinate_axes(&mut pass, range),
            }
        }
    }

    /// Stages the **object-id picking pass** (#141) for `draws`: writes the frame
    /// camera and uploads one id instance per pickable draw, returning the
    /// `(mesh_id, instance slot)` records [`encode_picking`](Self::encode_picking)
    /// then draws. Out-of-range mesh ids and `Shadow` draws are skipped, but the
    /// index mapping is preserved (a skipped draw's index simply never appears),
    /// so a decoded id maps straight back to `draws[index]`.
    ///
    /// Split from the encode half so the pass's two borrows never overlap: this
    /// is the `&mut self` work (uniform write + instance upload), and encoding
    /// then needs only `&self` — which is what lets [`pick`](Self::pick) render
    /// into a target the renderer still *owns*, instead of moving it out of
    /// `self` and handing it back (#235 R4).
    ///
    /// Private: a front-end reaches it through [`pick`](Self::pick), which owns
    /// the whole sequence — ensure target, prepare, encode, read back (#235 R4).
    ///
    /// **It keeps its own traversal on purpose** (#204). It does *not* batch a
    /// [`Scene`] and does not go through the per-primitive `record` bodies: this
    /// is a different pass with different attachments (single-sampled, flat id
    /// colors, no MSAA resolve) drawing only mesh geometry through the
    /// [`Picking`](super::picking::Picking) pipeline instead of the visual ones,
    /// and it needs an
    /// instance per *object* — never grouped — because the whole point is that
    /// each one carries a distinct id. Sharing the walk would mean threading a
    /// pass-kind through every `record` body to couple two loops that agree on
    /// almost nothing, for little gain.
    fn prepare_picking(&mut self, camera: Camera, draws: &[Draw]) -> Vec<(usize, u32)> {
        // Camera P·V for this frame (writes the shared camera uniform bound by
        // `uniforms.camera`, which is layout-compatible with the pick pipeline).
        self.uniforms.write_camera(&self.gpu.queue, camera);

        // Build one pick instance per drawable object, carrying its index color.
        let mut instances: Vec<PickInstanceRaw> = Vec::with_capacity(draws.len());
        let mut records: Vec<(usize, u32)> = Vec::with_capacity(draws.len());
        // A shadow blob has no mesh geometry to hit-test, so it is not pickable.
        for (index, draw) in draws.iter().enumerate() {
            if !draw.selection.is_mesh() {
                continue;
            }
            let Some(mesh) = self.meshes.get(draw.mesh_id as usize) else {
                continue;
            };
            let effective = draw.model * mesh.base_model;
            let slot = instances.len() as u32;
            instances.push(PickInstanceRaw::new(effective, index as u32));
            records.push((draw.mesh_id as usize, slot));
        }

        // Grow + upload the pick instance buffer.
        self.picking.upload_instances(&self.gpu, &instances);
        records
    }

    /// Encodes the **object-id picking pass** for the records staged by
    /// [`prepare_picking`](Self::prepare_picking): each pickable draw's mesh is
    /// rasterized in a flat color encoding its **index**, single-sampled and
    /// depth-tested into `target`'s id-color attachment (cleared to id `0` =
    /// background) and its depth attachment. No lighting, no texture, no MSAA —
    /// so the pixel under the cursor reads back to an exact id via
    /// [`PickInstanceRaw::decode`].
    ///
    /// Takes `&self`: with the staging already done, the pass borrows the
    /// pipeline, the camera bind group and the mesh geometry immutably — the same
    /// way it borrows the `target`, which is why that target can live in
    /// `self.picking` for the whole call (#235 R4).
    ///
    /// **It keeps its own traversal on purpose** (#204). It does *not* batch a
    /// [`Scene`] and does not go through the per-primitive `record` bodies: this
    /// is a different pass with different attachments (single-sampled, flat id
    /// colors, no MSAA resolve) drawing only mesh geometry through the
    /// [`Picking`](super::picking::Picking) pipeline instead of the visual ones,
    /// and it needs an
    /// instance per *object* — never grouped — because the whole point is that
    /// each one carries a distinct id. Sharing the walk would mean threading a
    /// pass-kind through every `record` body to couple two loops that agree on
    /// almost nothing, for little gain.
    fn encode_picking(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        target: &PickTarget,
        records: &[(usize, u32)],
    ) {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("trd picking pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: target.color_view(),
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    // Clear to id 0 (background).
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: target.depth_view(),
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(1.0),
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });

        self.picking
            .bind(&mut pass, self.uniforms.camera.bind_group());
        for &(mesh_id, slot) in records {
            // `prepare_picking` already dropped out-of-range ids, but the store is
            // the only thing that can prove it — so ask it, don't index it (#235 R7).
            let Some(mesh) = self.meshes.get(mesh_id) else {
                continue;
            };
            draw_indexed(&mut pass, mesh.filled(), slot..slot + 1);
        }
    }

    /// Ensures the depth attachment matches `viewport` (each dimension clamped to
    /// ≥ 1) at the mesh pass's [`sample_count`](MsaaColor::sample_count) (the
    /// depth sample count must match the color attachment), recreating it only
    /// when the target size changes.
    fn ensure_depth(&mut self, viewport: Viewport) {
        let dw = viewport.width.max(1);
        let dh = viewport.height.max(1);
        if self
            .depth
            .as_ref()
            .is_none_or(|d| d.width != dw || d.height != dh)
        {
            self.depth = Some(create_depth_target(
                &self.gpu.device,
                dw,
                dh,
                self.msaa.sample_count(),
            ));
        }
    }

    /// **Object-id picking** (#141): renders `draws` through the flat id-color
    /// pass at `viewport`'s size and returns the **0-based index into `draws`**
    /// of the object under pixel `(x, y)`, or `None` for the background (or an
    /// out-of-bounds coordinate). The pass is single-sampled and depth-tested, so
    /// the nearest object wins and ids are never blended — the "color index"
    /// method, no ray-marching. The lazily-created pick target tracks `viewport`,
    /// which the caller passes on every call (#203): the harness no longer owns a
    /// render target of its own to read a size from, so a shell reports its
    /// current display size the same way it would to `render_params`.
    pub async fn pick(
        &mut self,
        camera: Camera,
        draws: &[Draw],
        x: u32,
        y: u32,
        viewport: Viewport,
    ) -> Option<u32> {
        let gpu = self.gpu.clone();
        let Viewport {
            width: w,
            height: h,
        } = viewport;
        // The target stays owned by `self.picking` for the whole call (#235 R4).
        // The borrows are separated in *time* instead of by moving it out: the
        // `&mut self` staging first, then an all-immutable encode + read-back.
        self.picking.ensure_target(&gpu.device, w, h);
        let records = self.prepare_picking(camera, draws);

        let target = self.picking.target()?;
        if !target.contains(x, y) {
            return None;
        }

        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("trd pick frame"),
            });
        self.encode_picking(&mut encoder, target, &records);
        // Copy just the one texel under the cursor into the staging buffer.
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: target.texture(),
                mip_level: 0,
                origin: wgpu::Origin3d { x, y, z: 0 },
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: target.staging(),
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT),
                    rows_per_image: Some(1),
                },
            },
            wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
        );
        gpu.queue.submit(Some(encoder.finish()));

        let slice = target.staging().slice(..4);
        let (sender, receiver) = futures_channel::oneshot::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });
        // Same wait as the offscreen readback; see `platform::poll_for_map`.
        super::platform::poll_for_map(&gpu.device).ok()?;
        receiver.await.ok()?.ok()?;

        let id = {
            let mapped = slice.get_mapped_range().ok()?;
            let rgba = [mapped[0], mapped[1], mapped[2], mapped[3]];
            PickInstanceRaw::decode(rgba)
        };
        target.staging().unmap();
        id
    }
}

/// **The `record` bodies: one per [`Primitive`], each self-contained** (#204).
///
/// `encode_pass`'s loop is a dispatch — one line per arm — and every case's GPU
/// command sequence lives in its own body here. They are methods on `Renderer`
/// rather than on [`Primitive`] because everything a body binds (pipelines,
/// group-0 uniforms, the mesh store's geometry) is renderer-owned state: a
/// per-primitive type would carry nothing of its own and would only borrow the
/// renderer straight back — and [`Primitive`] is *public*, so hanging GPU code
/// off it would drag `wgpu` into the API of a type whose whole point is to be
/// pure data. (A `DrawDescriptor` value applied by one issuing helper was
/// considered and rejected in #204: it would have to express a mesh's four bind
/// groups plus a dynamic offset, a gizmo's one, and the shadow's vertex-buffer
/// swap at once, degenerating into a union of every case.)
///
/// **The rule has two halves, and the second is load-bearing:**
///
/// > *No `record` may depend on pass state another `record` set* — **and
/// > therefore every `record` sets what it needs at entry.**
///
/// Nothing restores anything at exit. That is what let the trailing
/// `set_bind_group(0, camera)` "restore" lines go: they existed only so the
/// *next* arm's assumptions held, which made the loop a hand-maintained pass
/// state machine (and forced the matching hand-hoisted binds before it). With
/// every body binding its own group 0, there is nothing left to undo.
///
/// Dropping the restores without adding the entry binds would be a wgpu
/// validation error, not a subtle diff:
/// [`PlaneGrid`](Primitive::PlaneGrid) and
/// [`QuadOutline`](Primitive::QuadOutline) swap group 0 to the *gizmo* uniform,
/// and wireframe meshes are submitted **after** them (layers 2/3 before 4 — see
/// [`Primitive::sort_key`]), so a mesh body that assumed group 0 was still the
/// camera binding would hand its pipeline a group-0 layout it was not built for.
///
/// Eliding a redundant state change is allowed only *inside* an issuing helper,
/// where it is provably safe — never by hoisting a bind out to the caller.
/// Within a single body, later commands may of course rely on what that same
/// body set ([`record_coordinate_axes`](Self::record_coordinate_axes) draws
/// twice off one instance binding); the rule is about *cross-body* state.
impl Renderer {
    /// Binds the per-frame instance-model buffer at vertex slot 1 — the one piece
    /// of pass state *every* pipeline in the mesh pass reads.
    ///
    /// It used to be bound once before the loop, which is precisely the coupling
    /// this restructure removes, so each body binds it at entry instead. It is
    /// the same buffer every time and there is one such call per *batch* (a
    /// handful per frame, not per object), so the repetition is free next to the
    /// draw it precedes.
    fn bind_instances(&self, pass: &mut wgpu::RenderPass<'_>) {
        pass.set_vertex_buffer(1, self.instances.slice());
    }

    /// Records one instanced batch of mesh `mesh_id` drawn in `mode` — the one
    /// place a primitive's mode selects a pipeline, because [`Primitive::Mesh`]
    /// is the only variant carrying one (#204).
    ///
    /// Each mode binds its own group 0: the camera `P·V` for the unlit modes, or
    /// this mesh's [`PbrUniform`] slot (a dynamic offset into the slot array) for
    /// [`Shaded`](RenderMode::Shaded).
    fn record_mesh(
        &self,
        pass: &mut wgpu::RenderPass<'_>,
        mesh_id: u32,
        mode: RenderMode,
        range: Range<u32>,
    ) {
        // The batcher already dropped out-of-range ids, but nothing in the types
        // says so — ask the store rather than index it, so a future caller that
        // skips the batcher draws nothing instead of panicking (#235 R7).
        let Some(mesh) = self.meshes.get(mesh_id as usize) else {
            return;
        };
        self.bind_instances(pass);
        match mode {
            RenderMode::Filled => {
                pass.set_pipeline(&self.pipelines.filled);
                pass.set_bind_group(0, self.uniforms.camera.bind_group(), &[]);
                draw_indexed(pass, mesh.filled(), range);
            }
            RenderMode::Textured => {
                pass.set_pipeline(&self.pipelines.textured);
                pass.set_bind_group(0, self.uniforms.camera.bind_group(), &[]);
                pass.set_bind_group(1, mesh.texture.bind_group(), &[]);
                draw_indexed(pass, mesh.filled(), range);
            }
            RenderMode::Shaded => {
                // group 0 = this mesh's PbrUniform slot (selected by a dynamic
                // offset), group 1 = this mesh's albedo, group 2 = the HDR env
                // map, group 3 = its material maps.
                pass.set_pipeline(&self.pipelines.pbr);
                let offset = self.uniforms.pbr.offset(mesh_id as usize);
                pass.set_bind_group(0, self.uniforms.pbr.bind_group(), &[offset]);
                pass.set_bind_group(1, mesh.texture.bind_group(), &[]);
                pass.set_bind_group(2, self.environment.bind_group(), &[]);
                pass.set_bind_group(3, mesh.material_maps.bind_group(), &[]);
                draw_indexed(pass, mesh.pbr(), range);
            }
            RenderMode::Wireframe => {
                pass.set_pipeline(&self.pipelines.wireframe);
                pass.set_bind_group(0, self.uniforms.camera.bind_group(), &[]);
                draw_indexed(pass, mesh.wireframe(), range);
            }
        }
    }

    /// The shared body of every **screen-space-expanded line gizmo**: the
    /// analytic-AA line pipeline plus the viewport-aware gizmo uniform at group 0
    /// (its own layout, *not* the camera one), then `geometry` over `range`.
    ///
    /// The AABB box, the plane grid, the quad outline and the axes' shafts differ
    /// only in which vertex geometry they draw, so they issue through one helper
    /// instead of repeating the same three lines four times (#204).
    fn record_gizmo_lines(
        &self,
        pass: &mut wgpu::RenderPass<'_>,
        geometry: &VertexGeometry,
        range: Range<u32>,
    ) {
        pass.set_pipeline(&self.pipelines.gizmo_line);
        pass.set_bind_group(0, self.uniforms.gizmo.bind_group(), &[]);
        self.bind_instances(pass);
        draw_vertices(pass, geometry, range);
    }

    /// Records the AABB outline of mesh `mesh_id` (#42) from that mesh's own
    /// precomputed corner geometry.
    fn record_aabb_box(&self, pass: &mut wgpu::RenderPass<'_>, mesh_id: u32, range: Range<u32>) {
        // Same as `record_mesh`: the id is checked, not assumed (#235 R7).
        let Some(mesh) = self.meshes.get(mesh_id as usize) else {
            return;
        };
        self.record_gizmo_lines(pass, mesh.aabb(), range);
    }

    /// Records the coordinate-plane grid lattice on `plane`, resolving the plane
    /// to its shared line buffer.
    fn record_plane_grid(
        &self,
        pass: &mut wgpu::RenderPass<'_>,
        plane: GridPlane,
        range: Range<u32>,
    ) {
        self.record_gizmo_lines(pass, &self.gizmos.grid_lines[plane.index()], range);
    }

    /// Records the tracked placement-quad outline; `selected` picks the
    /// highlight-colored line buffer.
    fn record_quad_outline(
        &self,
        pass: &mut wgpu::RenderPass<'_>,
        selected: bool,
        range: Range<u32>,
    ) {
        self.record_gizmo_lines(pass, &self.gizmos.quad_lines[usize::from(selected)], range);
    }

    /// Records the contact / blob grounding shadow: its own alpha-blended,
    /// depth-write-off pipeline over the shared shadow quad at vertex slot 0,
    /// reading the camera `P·V` at group 0.
    fn record_blob_shadow(&self, pass: &mut wgpu::RenderPass<'_>, range: Range<u32>) {
        pass.set_pipeline(&self.pipelines.shadow);
        pass.set_bind_group(0, self.uniforms.camera.bind_group(), &[]);
        pass.set_vertex_buffer(0, self.gizmos.shadow_vertex_buffer.slice(..));
        self.bind_instances(pass);
        pass.draw(0..SHADOW_VERTEX_COUNT, range);
    }

    /// Records the world-orientation gizmo (#42) as two draws over the same
    /// instances: the expanded shafts through the shared gizmo-line body, then
    /// the arrowheads, which are ordinary unlit overlay triangles and so read the
    /// **camera** uniform at group 0 rather than the gizmo one.
    ///
    /// The second draw reuses the instance binding the first made — same body, so
    /// the rule above is not in play.
    fn record_coordinate_axes(&self, pass: &mut wgpu::RenderPass<'_>, range: Range<u32>) {
        self.record_gizmo_lines(pass, &self.gizmos.axes_lines, range.clone());
        pass.set_pipeline(&self.pipelines.gizmo_solid);
        pass.set_bind_group(0, self.uniforms.camera.bind_group(), &[]);
        draw_vertices(pass, &self.gizmos.axes_heads, range);
    }
}

/// What a shell must do to its surface before the next frame.
///
/// wgpu's surface acquisition can fail in ways only the **front-end** knows how
/// to recover from — the native window defers to the next redraw, while the
/// browser recreates the surface from its canvas — so presenting reports what is
/// needed instead of baking in a policy (#180).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SurfaceRepair {
    /// Reapply the current configuration ([`Renderer::reconfigure_surface`]).
    Reconfigure,
    /// Build a new surface for the same window/canvas and swap it in
    /// ([`Renderer::replace_surface`]) — the old one is gone.
    Recreate,
}

/// Why a frame could not be drawn onto a surface.
///
/// Separate from "what repair is needed" ([`SurfaceRepair`]) because they are
/// **independent questions**: a frame can be presented *and* need a repair
/// (`Ok(Some(Reconfigure))`), or be skipped needing none (`Timeout`). Folding
/// both into one enum is what made the old `PresentOutcome` awkward — every
/// consumer immediately re-projected it onto one axis (#203).
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum SurfaceError {
    /// The surface configuration is stale (a resize or a minimise).
    #[error("surface is outdated: reconfigure and draw again")]
    Outdated,
    /// The surface was lost and must be recreated.
    #[error("surface was lost: recreate it and draw again")]
    Lost,
    /// Acquisition timed out — transient, the next frame should succeed.
    #[error("surface acquisition timed out")]
    Timeout,
    /// The surface is not visible (an occluded or hidden window).
    #[error("surface is occluded")]
    Occluded,
    /// The surface failed validation.
    #[error("surface validation failed")]
    Validation,
}

impl SurfaceError {
    /// What the shell must do before the next frame, or `None` when the cause is
    /// transient and skipping this frame is the whole remedy.
    pub fn repair(self) -> Option<SurfaceRepair> {
        match self {
            SurfaceError::Outdated => Some(SurfaceRepair::Reconfigure),
            SurfaceError::Lost => Some(SurfaceRepair::Recreate),
            SurfaceError::Timeout | SurfaceError::Occluded | SurfaceError::Validation => None,
        }
    }
}
