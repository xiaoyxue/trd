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
//! **The render behaviour is all on `Renderer`, but not all in this file.**
//! The targets used to carry it — `OffscreenTarget::render`/`draw_layers`/
//! `read_pixels`, `OnscreenTarget::present`/`acquire`/`reconfigure` — with this
//! harness forwarding to them, which had the ownership backwards: a swapchain
//! handle knows nothing about pipelines, materials or the mesh store, and
//! everything it "did" immediately borrowed the renderer straight back. Now a
//! target is pure data ([`RenderTarget`], [`SurfaceTarget`], [`TextureTarget`]),
//! and drawing is:
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
//! # Where the rest of `impl Renderer` lives (#363)
//!
//! `render/` is organised **by resource**, and almost every remaining method is
//! a *driver of one of those resources* — so each group sits in the file that
//! owns what it touches, rather than 900 lines below the struct. The `Renderer`
//! fields are `pub(super)` for exactly this reason: the impl is spread across
//! `render/`, and nowhere wider. rustdoc still lists every method in one place;
//! this table is for reading the source.
//!
//! | Group | Lives in |
//! |---|---|
//! | `pick`, `prepare_picking`, `encode_picking`, `pick_target_size` | `picking.rs` |
//! | target lifecycle — `create_*_target`, `resize_*`, `reconfigure_surface`, `replace_surface`, `read_pixels`, `read_back` | `render_target.rs` |
//! | the per-[`Primitive`] `record` bodies + `bind_instances` | `draw_command.rs` |
//! | [`MeshTarget`], `add_mesh`, `remove_mesh`, `mesh_count`, the texture setters, `edit_appearance` + the appearance setters | `mesh_store.rs` |
//! | `set_env_map` | `environment.rs` |
//! | `update_frame_texture*`, `has_frame_texture` | `frame_plane.rs` |
//!
//! **What stays here** is what has no other owner: the errors, the struct, the
//! constructors, the render entry points, and the frame loop — `prepare_frame`,
//! `encode`/`encode_overlay` and `encode_pass`.
//!
//! Internally it is still a composition of a few cohesive parts, each with a
//! single job, so no one struct is a grab-bag of wgpu handles:
//! - [`RenderPipelines`] — the mesh/gizmo pipelines (filled/wireframe/textured/PBR)
//!   and [`SceneUniforms`] — the group-0 uniforms they read (the camera `P·V`,
//!   the gizmo viewport params, the per-mesh PBR slot array). Two types, because
//!   *what draws* and *what it reads* are different questions (#203); both live
//!   in `render_pipelines.rs` because both are `f(format, sample_count)` rather
//!   than a function of any one scene (#221 §2).
//! - [`MeshStore`] — the uploaded [`MeshGpu`](super::mesh_store::MeshGpu)s (each
//!   owning its albedo, material maps, material, IBL, tone-map and debug view),
//!   the shared axes gizmo, and the growable per-instance model buffer; also
//!   walks a [`Scene`] into draw batches.
//! - [`BoundTexture`](super::BoundTexture) — the mesh albedo sampled by textured
//!   draws (#20).
//! - [`FramePlane`] — the background video frame plane (#63).
//! - [`MeshAttachments`] — the mesh pass's own depth and (with MSAA) color
//!   attachments, in `attachments.rs` (#363).
//! - [`Picking`](super::picking::Picking) — the object-id picking pipeline, its
//!   instance buffer and its `PickTarget` (#141), all three in `picking.rs`
//!   beside the target they create; the target is allocated lazily on the first
//!   `pick` call so a headless CLI stream never pays for it. Like every other
//!   target it is **data**: staging, encoding and reading it back are `Renderer`
//!   methods, so no target drives the renderer (#235 R4).
//!
//! **The draw loop is a dispatch, not a state machine** (#204). `encode_pass`
//! walks the batched commands and hands each to the `record` body of its
//! [`Primitive`] — one line per arm — and every body sets its own pipeline and
//! bind groups at entry while restoring nothing at exit, so no body may depend
//! on pass state another one left behind. The bodies and that rule are in
//! `draw_command.rs`, beside the batcher they are in lockstep with.
//!
//! Formerly `BatchRenderer`. "Batch" there meant *batch-mode headless output* and
//! described nothing about the type — instanced batching lives entirely in
//! this module (`draw_command.rs`) — while colliding with that real meaning. The
//! name now belongs to one concept: grouping draws into instanced commands
//! (#180).

use std::sync::Arc;

use super::bound_material_maps::BoundMaterialMaps;
use super::buffer::InstanceBuffer;
use super::draw_command::{build_batches, Batches};
use super::environment::{EnvBackgroundSettings, Environment};
use super::frame_plane::FramePlane;
use super::gizmo::GizmoGeometry;
use super::mesh_store::MeshStore;
use super::picking::Picking;
use super::*;
use super::{Primitive, Scene};
use crate::math::Matrix4;
use crate::Camera;
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
    pub(super) pipelines: RenderPipelines,
    /// The group-0 uniforms every pass binds: the camera `P·V`, the gizmo
    /// viewport params, and the per-mesh PBR slot array.
    pub(super) uniforms: SceneUniforms,
    /// The HDR environment subsystem: the probe reflected by
    /// [`RenderMode::Shaded`] draws **and** the pipeline drawing it as the
    /// background sky. One type, like its sibling `FramePlane` (#221 §5).
    pub(super) environment: Environment,
    /// The caller's uploaded meshes, fixed at construction.
    pub(super) meshes: MeshStore,
    /// The constant overlay geometry (axes, grids, quad outlines, shadow quad):
    /// a function of nothing, built once.
    pub(super) gizmos: GizmoGeometry,
    /// The per-frame model matrices every draw kind is instanced through,
    /// rewritten each `encode` and grown on demand (#222).
    pub(super) instances: InstanceBuffer<InstanceRaw>,
    /// The batcher's scratch, reused across frames (#235 R6): `build_batches`
    /// clears and refills it, so a steady-state frame reuses the capacity of the
    /// previous one instead of allocating three vectors per frame. It is scratch,
    /// not state — nothing outside `encode_pass` reads it between frames.
    pub(super) batches: Batches,
    pub(super) frame_plane: FramePlane,
    /// The mesh pass's own attachments: the depth buffer solid meshes z-occlude
    /// through, and — when MSAA is on — the multisampled color the pass renders
    /// into and resolves into the caller's `view`, so every front-end gets
    /// multisampled mesh/arrowhead edges transparently (gizmo lines add analytic
    /// AA separately). Both are sized to the viewport by `encode_pass`.
    pub(super) attachments: MeshAttachments,
    /// The shared GPU context. Retained so `encode` can grow GPU resources and
    /// the setters can upload immediately, without the caller threading handles
    /// through every call. Holding the whole context (rather than a bare device)
    /// is what lets the &self.gpu.queue live here too, which is why uploads no longer have
    /// to be deferred to `encode` (#180).
    pub(super) gpu: Arc<GpuContext>,
    /// Everything the object-id picking pass (#141) needs: its pipeline, its
    /// instance buffer and its lazily-created target.
    pub(super) picking: Picking,
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
            attachments: MeshAttachments::new(format, sample_count),
            gpu,
            picking,
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

    /// The GPU context this harness renders on.
    ///
    /// Exposed so a shell can build further resources — or bind the rendered
    /// texture into its UI toolkit — on the **same** device the frame was drawn
    /// with, rather than opening a second one.
    pub fn gpu(&self) -> &Arc<GpuContext> {
        &self.gpu
    }

    // -----------------------------------------------------------------------
    // Render targets — creation, drawing, readback, lifecycle (#203).
    //
    // All of it lives here rather than on the target types: a target is a place
    // a frame lands, and everything one could "do" needs the pipelines, the mesh
    // store and the GPU context this harness owns.
    // -----------------------------------------------------------------------

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

    /// Everything a frame needs **written** before a pass can start: the camera
    /// and PBR uniforms, the batched instances, and the two background settings
    /// (#235 R8).
    ///
    /// Split out of `encode_pass` so the encode half reads as what it is — a
    /// dispatch over batched commands — instead of ~150 lines where the
    /// preparation and the recording are interleaved. The pass attachments are
    /// deliberately *not* sized here: `encode_pass` sizes them itself, so its
    /// descriptor takes the views straight from the call that allocated them
    /// (#363).
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
        );

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
            meshes.get(mesh_id).map(|mesh| mesh.geometry.base_model)
        });
        self.instances.upload(&self.gpu, &self.batches.instances);

        // 3. Background frame-plane fit for this viewport (no-op if the scene
        //    asks for no frame plane or no frame texture is bound yet).
        if let Some(fit) = background.frame {
            self.frame_plane.write_fit(&self.gpu.queue, fit, viewport);
        }

        // 4. Bind groups for each mesh's own albedo (#141) and material maps, and
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
        // Sizing the attachments here — rather than in `prepare_frame` — is what
        // lets the descriptor take its views from the call that allocated them,
        // so there is no moment at which a depth attachment might be missing
        // (#363). The borrow ends with the block, freeing `self` for the loop.
        let mut pass = {
            let views = self.attachments.resize(&self.gpu.device, camera.viewport());
            let color_attachment = views.color_attachment(
                view,
                if load_color {
                    wgpu::LoadOp::Load
                } else {
                    wgpu::LoadOp::Clear(wgpu::Color::BLACK)
                },
            );
            encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("trd mesh pass"),
                color_attachments: &[Some(color_attachment)],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: views.depth_view(),
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            })
        };

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
                Primitive::QuadFill => self.record_quad_fill(&mut pass, range),
                Primitive::BlobShadow => self.record_blob_shadow(&mut pass, range),
                Primitive::CoordinateAxes => self.record_coordinate_axes(&mut pass, range),
            }
        }
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
