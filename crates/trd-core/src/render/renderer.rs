//! [`Renderer`] — the persistent render harness (#134, #203).
//!
//! One type owns everything a frame needs: the mesh/gizmo/PBR/picking
//! pipelines, the decode-once [`MeshStore`], materials/lighting/IBL, and the
//! picking pipeline + its lazily-allocated [`PickTarget`]. It used to be split
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
//! - [`SceneRenderPipelines`] — the mesh pipelines (filled/wireframe/textured/PBR)
//!   and the camera `P·V` uniform they share.
//! - [`MeshStore`] — the uploaded [`MeshGpu`]s, the shared axes gizmo, and the
//!   growable per-instance model buffer; also walks a [`Scene`] into draw batches.
//! - [`BoundTexture`](super::BoundTexture) — the mesh albedo sampled by textured
//!   draws (#20).
//! - [`FramePlane`] — the background video frame plane (#63).
//! - [`PickTarget`] — the object-id picking target (#141), allocated lazily on
//!   the first [`pick`](Self::pick) call so a headless CLI stream never pays
//!   for it.
//!
//! Formerly `BatchRenderer`. "Batch" there meant *batch-mode headless output* and
//! described nothing about the type — instanced batching lives entirely in
//! this module (`batch.rs`) — while colliding with that real meaning. The
//! name now belongs to one concept: grouping draws into instanced commands
//! (#180).

use std::sync::Arc;

use super::batch::{build_batches, DrawKind};
use super::bound_material_maps::BoundMaterialMaps;
use super::buffer::{draw_indexed, draw_vertices};
use super::env_background::{EnvBackground, EnvBackgroundSettings};
use super::frame_plane::FramePlane;
use super::mesh_store::MeshStore;
use super::pbr::PbrBatchInputs;
use super::*;
use crate::material::DisneyMaterial;
use crate::math::Matrix4;
use crate::output::tightly_pack_rgba;
use crate::texture::Texture;
use crate::visual::{Draw, DrawableObject};
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

/// The mesh and gizmo pipelines plus their camera/material bindings. Filled,
/// wireframe, arrowheads, and textured rendering share the camera layout;
/// expanded gizmo lines use a viewport-aware group-0 uniform.
struct SceneRenderPipelines {
    filled: wgpu::RenderPipeline,
    wireframe: wgpu::RenderPipeline,
    /// Screen-space expanded, alpha-feathered AABB/axes/grid line pipeline.
    gizmo_line: wgpu::RenderPipeline,
    /// Unlit overlay triangles for coordinate-axis arrowheads.
    gizmo_solid: wgpu::RenderPipeline,
    textured: wgpu::RenderPipeline,
    /// The contact / blob grounding-shadow pipeline (alpha-blended, depth-write
    /// off); shares the untextured camera bind-group layout (group 0).
    shadow: wgpu::RenderPipeline,
    /// The Disney PBR pipeline (`pbr.wgsl`): group 0 = [`pbr_uniform`], group 1
    /// = the bound albedo texture, group 2 = the HDR environment map.
    pbr: wgpu::RenderPipeline,
    /// The per-object `PbrUniform` buffer: `mesh_count` [`pbr_stride`]-spaced
    /// slots (each carries the shared camera/lights + that mesh's material),
    /// rewritten each `encode`; a draw binds its slot via a dynamic offset.
    pbr_uniform: wgpu::Buffer,
    pbr_bind_group: wgpu::BindGroup,
    /// The 256-aligned byte stride between adjacent `PbrUniform` slots (the
    /// `min_uniform_buffer_offset_alignment`-rounded `size_of::<PbrUniform>()`),
    /// so `slot i` lives at `i * pbr_stride`.
    pbr_stride: u64,
    camera_uniform: wgpu::Buffer,
    camera_bind_group: wgpu::BindGroup,
    gizmo_uniform: wgpu::Buffer,
    gizmo_bind_group: wgpu::BindGroup,
}

impl SceneRenderPipelines {
    /// Constructs a `SceneRenderPipelines` for `format`, building all pipelines over their
    /// bind-group layouts at `sample_count`× MSAA. `texture_layout` is the albedo
    /// texture's group-1 layout (from [`BoundTexture::layout`]), shared by the
    /// textured and PBR pipelines; `env_layout` is the PBR pipeline's group-2
    /// environment-map layout (from [`BoundEnv::layout`]). Every pipeline in the
    /// pass shares the one `sample_count` (`1` = no MSAA, single-sample).
    fn new(
        device: &wgpu::Device,
        format: wgpu::TextureFormat,
        texture_layout: &wgpu::BindGroupLayout,
        material_maps_layout: &wgpu::BindGroupLayout,
        env_layout: &wgpu::BindGroupLayout,
        sample_count: u32,
        mesh_count: usize,
    ) -> Self {
        // One explicit bind-group layout shared by both untextured pipelines, so
        // the single camera bind group is valid whichever RenderMode is active.
        let camera_layout = create_mesh_bind_group_layout(device);
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("trd mesh pipeline layout"),
            bind_group_layouts: &[Some(&camera_layout)],
            immediate_size: 0,
        });
        let filled = create_mesh_pipeline_with(
            device,
            format,
            &pipeline_layout,
            wgpu::PrimitiveTopology::TriangleList,
            Some(solid_depth_stencil()),
            sample_count,
        );
        let wireframe = create_mesh_pipeline_with(
            device,
            format,
            &pipeline_layout,
            wgpu::PrimitiveTopology::LineList,
            Some(overlay_depth_stencil()),
            sample_count,
        );
        let gizmo_solid = create_mesh_pipeline_with(
            device,
            format,
            &pipeline_layout,
            wgpu::PrimitiveTopology::TriangleList,
            Some(overlay_depth_stencil()),
            sample_count,
        );
        let gizmo_layout = create_gizmo_bind_group_layout(device);
        let gizmo_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("trd gizmo pipeline layout"),
                bind_group_layouts: &[Some(&gizmo_layout)],
                immediate_size: 0,
            });
        let gizmo_line =
            create_gizmo_line_pipeline(device, format, &gizmo_pipeline_layout, sample_count);
        // Contact / blob grounding-shadow pipeline (#110 follow-up): shares the
        // untextured camera layout (group 0), alpha-blended, depth-write off.
        let shadow = create_shadow_pipeline(device, format, &pipeline_layout, sample_count);
        // Textured pipeline (#20): group 0 = the shared camera uniform, group 1 =
        // the bound albedo texture + sampler.
        let textured_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("trd textured pipeline layout"),
                bind_group_layouts: &[Some(&camera_layout), Some(texture_layout)],
                immediate_size: 0,
            });
        let textured =
            create_textured_pipeline(device, format, &textured_pipeline_layout, sample_count);
        // Disney PBR pipeline (#): group 0 = the PbrUniform, group 1 = the shared
        // albedo texture layout, group 2 = the HDR environment map. Its group-0
        // layout differs from the camera layout, so the encode arm restores the
        // camera bind group after each PBR draw.
        let pbr_layout = create_pbr_bind_group_layout(device);
        let pbr_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("trd pbr pipeline layout"),
            bind_group_layouts: &[
                Some(&pbr_layout),
                Some(texture_layout),
                Some(env_layout),
                Some(material_maps_layout),
            ],
            immediate_size: 0,
        });
        let pbr = create_pbr_pipeline(device, format, &pbr_pipeline_layout, sample_count);
        // The per-object PbrUniform buffer: one 256-aligned slot per mesh, each
        // rewritten every frame with the shared camera/lights + that mesh's
        // material; a PBR draw selects its slot via a dynamic offset.
        let align = device.limits().min_uniform_buffer_offset_alignment as u64;
        let pbr_stride = (std::mem::size_of::<PbrUniform>() as u64).next_multiple_of(align);
        let pbr_uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("trd pbr uniform"),
            size: pbr_stride * mesh_count.max(1) as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let pbr_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("trd pbr bind group"),
            layout: &pbr_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                // A single-slot window; the dynamic offset picks which slot.
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &pbr_uniform,
                    offset: 0,
                    size: wgpu::BufferSize::new(std::mem::size_of::<PbrUniform>() as u64),
                }),
            }],
        });
        // A placeholder camera: the buffers only need to exist here, and every
        // frame's `write_camera` overwrites them with the real one.
        let placeholder = FrameParams::IDENTITY
            .to_camera(Viewport {
                width: 1,
                height: 1,
            })
            .expect("the identity params are a valid camera form");
        let (camera_uniform, camera_bind_group) =
            create_view_proj_binding(device, &camera_layout, placeholder);
        let (gizmo_uniform, gizmo_bind_group) =
            create_gizmo_binding(device, &gizmo_layout, placeholder);
        Self {
            filled,
            wireframe,
            gizmo_line,
            gizmo_solid,
            textured,
            shadow,
            pbr,
            pbr_uniform,
            pbr_bind_group,
            pbr_stride,
            camera_uniform,
            camera_bind_group,
            gizmo_uniform,
            gizmo_bind_group,
        }
    }

    /// Rewrites the camera `P·V` uniform for this frame's `camera`.
    fn write_camera(&self, queue: &wgpu::Queue, camera: Camera) {
        write_view_proj(queue, &self.camera_uniform, camera);
        write_gizmo_params(queue, &self.gizmo_uniform, camera);
    }

    /// Rewrites the Disney PBR uniform **slots** for this frame: one slot per
    /// mesh (`materials[i]` → slot `i` at `i * pbr_stride`), each carrying the
    /// shared camera `P·V` + world position + light rig, this mesh's material, and
    /// the env gate. A PBR draw then binds its object's material via a dynamic
    /// offset. `materials` is indexed by mesh id.
    fn write_pbr(
        &self,
        queue: &wgpu::Queue,
        camera: Camera,
        inputs: PbrBatchInputs<'_>,
        use_env: bool,
    ) {
        debug_assert_eq!(inputs.materials.len(), inputs.ibl.len());
        debug_assert_eq!(inputs.materials.len(), inputs.tone_mappings.len());
        debug_assert_eq!(inputs.materials.len(), inputs.debug_views.len());
        let view_proj = camera.view_projection().matrix().to_cols_array();
        let camera_pos = camera.position();
        for (i, (((material, ibl), tone_mapping), debug_view)) in inputs
            .materials
            .iter()
            .zip(inputs.ibl)
            .zip(inputs.tone_mappings)
            .zip(inputs.debug_views)
            .enumerate()
        {
            let uniform = PbrUniform::new(
                view_proj,
                camera_pos,
                PbrUniformInputs {
                    material,
                    lighting: inputs.lighting,
                    ibl: *ibl,
                    tone_mapping: *tone_mapping,
                    debug_view: *debug_view,
                    use_env,
                },
            );
            queue.write_buffer(
                &self.pbr_uniform,
                i as u64 * self.pbr_stride,
                bytemuck::bytes_of(&uniform),
            );
        }
    }
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
    pipelines: SceneRenderPipelines,
    /// The bound HDR environment map reflected by [`RenderMode::Shaded`] draws.
    env: BoundEnv,
    env_background: EnvBackground,
    /// The Disney material of **each** mesh (indexed by mesh id) applied to its
    /// [`RenderMode::Shaded`] draws (#141) — so a multi-object scene can give every
    /// object its own metallic/roughness/base_color.
    pbr_materials: Vec<DisneyMaterial>,
    /// Per-object environment reflection gains, parallel to `pbr_materials`.
    pbr_ibl: Vec<ImageBasedLighting>,
    /// Per-object output transforms, parallel to `pbr_materials`.
    pbr_tone_mappings: Vec<ToneMapping>,
    pbr_debug_views: Vec<PbrDebugView>,
    /// Scene light rig controls shared by every PBR object.
    lighting: Lighting,
    store: MeshStore,
    frame_plane: FramePlane,
    /// The mesh pass's depth attachment, (re)created lazily in `encode` to match
    /// the viewport. Gives solid (filled/textured) meshes real z-occlusion.
    depth: Option<DepthTarget>,
    /// The mesh pass's multisampled color attachment ([`sample_count`](Self::sample_count)×),
    /// (re)created lazily in `encode` to match the viewport. The pass renders into
    /// it and resolves into the caller's single-sample `view`, so every front-end
    /// gets multisampled mesh/arrowhead edges transparently. Gizmo lines add
    /// analytic AA separately. `None` when MSAA is disabled (`sample_count == 1`):
    /// the pass then renders straight into `view`.
    msaa: Option<MsaaColorTarget>,
    /// The mesh pass's MSAA sample count — `4` (the default,
    /// [`MSAA_SAMPLE_COUNT`]) for multisampled edges, or `1` for single-sample
    /// rasterization. Fixed at construction because every pipeline plus the
    /// depth/color attachments must share it.
    sample_count: u32,
    /// The color format the pipelines were built for; the MSAA color target must
    /// be created with the same format.
    format: wgpu::TextureFormat,
    /// The shared GPU context. Retained so `encode` can grow GPU resources and
    /// the setters can upload immediately, without the caller threading handles
    /// through every call. Holding the whole context (rather than a bare device)
    /// is what lets the &self.gpu.queue live here too, which is why uploads no longer have
    /// to be deferred to `encode` (#180).
    gpu: Arc<GpuContext>,
    /// The object-id **picking** pipeline (`picking.wgsl`): renders each drawn
    /// object in a flat id color into a single-sample linear target, reused by
    /// [`encode_picking`](Self::encode_picking). Built once (its own bind-group
    /// layout is structurally the camera layout, so `camera_bind_group` binds it).
    pick_pipeline: wgpu::RenderPipeline,
    /// Per-instance [`PickInstanceRaw`] buffer for the picking pass (model +
    /// id color), grown on demand like the mesh instance buffer.
    pick_instances: wgpu::Buffer,
    pick_instance_capacity: u32,
    /// The object-id picking target (#141), created lazily on the first
    /// [`pick`](Self::pick) call and resized to track whatever `viewport` the
    /// caller passes. `None` until a front-end actually picks, so the headless
    /// CLI never allocates it.
    pick_target: Option<PickTarget>,
}

impl Renderer {
    /// Constructs a `Renderer` that derives each mesh's base (preview) model
    /// automatically via [`Mesh::preview_transform`]
    /// ([`crate::DEFAULT_PREVIEW_TARGET`]) — center + uniform scale-to-fit — so an
    /// arbitrary-unit asset renders centered at a reasonable size. A convenience
    /// constructor over [`new`](Self::new); shared by the headless
    /// [`crate::run_stream`] and every on-screen front-end.
    pub fn auto_fit(gpu: Arc<GpuContext>, format: wgpu::TextureFormat, meshes: &[Mesh]) -> Self {
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
    /// Panics if `meshes` is empty or `meshes`/`base_models` differ in length.
    pub fn new(
        gpu: Arc<GpuContext>,
        format: wgpu::TextureFormat,
        meshes: &[Mesh],
        base_models: &[Matrix4],
    ) -> Self {
        Self::with_sample_count(gpu, format, meshes, base_models, MSAA_SAMPLE_COUNT)
    }

    /// Like [`new`](Self::new), but with an explicit mesh-pass MSAA
    /// `sample_count`: `4` ([`MSAA_SAMPLE_COUNT`]) for multisampled edges, or `1`
    /// to render single-sampled. Gizmo lines retain their shader-based analytic AA
    /// at `1`; mesh silhouettes and hardware wireframes do not. All pipelines and
    /// the depth/color attachments are built for this count.
    ///
    /// Panics if `meshes` is empty, `meshes`/`base_models` differ in length, or
    /// `sample_count` is 0.
    pub fn with_sample_count(
        gpu: Arc<GpuContext>,
        format: wgpu::TextureFormat,
        meshes: &[Mesh],
        base_models: &[Matrix4],
        sample_count: u32,
    ) -> Self {
        assert!(!meshes.is_empty(), "Renderer requires at least one mesh");
        assert_eq!(
            meshes.len(),
            base_models.len(),
            "meshes and base_models must have equal length"
        );
        assert!(sample_count >= 1, "sample_count must be >= 1");

        // One shared group-1 albedo layout for the textured/PBR pipelines and
        // every per-mesh [`BoundTexture`] (each object skins with its own diffuse).
        let texture_layout = create_texture_bind_group_layout(&gpu.device);
        let material_maps_layout = BoundMaterialMaps::create_layout(&gpu.device);
        let env = BoundEnv::new(&gpu);
        let env_background = EnvBackground::new(&gpu.device, format, env.layout(), sample_count);
        let pass = SceneRenderPipelines::new(
            &gpu.device,
            format,
            &texture_layout,
            &material_maps_layout,
            env.layout(),
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
        let frame_plane = FramePlane::new(&gpu.device, format, sample_count);

        // The picking pipeline: a group-0 camera uniform (structurally identical
        // to the mesh camera layout, so `pass.camera_bind_group` binds it) + the
        // per-instance id color, single-sampled into PICK_FORMAT.
        let pick_layout = &gpu
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("trd picking pipeline layout"),
                bind_group_layouts: &[Some(&create_mesh_bind_group_layout(&gpu.device))],
                immediate_size: 0,
            });
        let pick_pipeline = create_picking_pipeline(&gpu.device, pick_layout);
        let pick_instance_capacity = (meshes.len() as u32).max(1);
        let pick_instances = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("trd pick instance buffer"),
            size: pick_instance_capacity as u64 * std::mem::size_of::<PickInstanceRaw>() as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            pipelines: pass,
            env,
            env_background,
            pbr_materials: vec![DisneyMaterial::default(); meshes.len()],
            pbr_ibl: vec![ImageBasedLighting::default(); meshes.len()],
            pbr_tone_mappings: vec![ToneMapping::default(); meshes.len()],
            pbr_debug_views: vec![PbrDebugView::default(); meshes.len()],
            lighting: Lighting::default(),
            store,
            frame_plane,
            depth: None,
            msaa: None,
            sample_count,
            format,
            gpu,
            pick_pipeline,
            pick_instances,
            pick_instance_capacity,
            pick_target: None,
        }
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
            Self::with_sample_count(gpu.clone(), format, meshes, base_models, sample_count);

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
        let renderer = Self::auto_fit(gpu.clone(), TEXTURE_TARGET_FORMAT, meshes);
        let target = TextureTarget::new(&gpu.device, width, height)?;
        Ok((renderer, target))
    }

    /// The number of meshes this renderer can draw; valid mesh ids in a
    /// [`DrawableObject::Mesh`]/[`DrawableObject::AabbBox`] are in
    /// `0..mesh_count()`.
    pub fn mesh_count(&self) -> usize {
        self.store.len()
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
        if let Some(mesh) = self.store.meshes.get_mut(mesh_id) {
            mesh.texture.set(&self.gpu, texture);
        }
    }

    /// Binds a glTF metallic-roughness map (G=roughness, B=metallic) for mesh
    /// `mesh_id`, sampled by [`RenderMode::Shaded`] in place of the scalar
    /// material values. Out-of-range ids are ignored.
    pub fn set_mesh_metallic_roughness_texture(&mut self, mesh_id: usize, texture: &dyn Texture) {
        if let Some(mesh) = self.store.meshes.get_mut(mesh_id) {
            mesh.material_maps
                .set_metallic_roughness(&self.gpu, texture);
        }
    }

    /// Binds mesh `mesh_id`'s tangent-space glTF normal map, perturbing the
    /// shading normal in [`RenderMode::Shaded`]. Out-of-range ids are ignored.
    pub fn set_mesh_normal_texture(&mut self, mesh_id: usize, texture: &dyn Texture) {
        if let Some(mesh) = self.store.meshes.get_mut(mesh_id) {
            mesh.material_maps.set_normal(&self.gpu, texture);
        }
    }

    /// Sets the [`DisneyMaterial`] of **every** mesh — the single-mesh / global
    /// default. For a multi-object scene, give each object its own material with
    /// [`set_mesh_disney_material`](Self::set_mesh_disney_material). Takes effect
    /// on the next [`render`](Self::render).
    pub fn set_disney_material(&mut self, material: DisneyMaterial) {
        for m in &mut self.pbr_materials {
            *m = material.clone();
        }
    }

    /// Sets the [`DisneyMaterial`] of mesh `mesh_id` only (#141) — so each
    /// object in a multi-object scene has its own metallic/roughness/base_color.
    /// Out-of-range ids are ignored. Takes effect on the next
    /// [`render`](Self::render).
    pub fn set_mesh_disney_material(&mut self, mesh_id: usize, material: DisneyMaterial) {
        if let Some(m) = self.pbr_materials.get_mut(mesh_id) {
            *m = material;
        }
    }

    /// Sets scene lighting controls shared by every PBR object.
    pub fn set_lighting(&mut self, lighting: Lighting) {
        self.lighting = lighting;
    }

    /// Sets image-based-lighting controls for every PBR object.
    pub fn set_image_based_lighting(&mut self, ibl: ImageBasedLighting) {
        self.pbr_ibl.fill(ibl);
    }

    /// Sets image-based-lighting controls for one PBR object.
    pub fn set_mesh_image_based_lighting(&mut self, mesh_id: usize, ibl: ImageBasedLighting) {
        if let Some(current) = self.pbr_ibl.get_mut(mesh_id) {
            *current = ibl;
        }
    }

    /// Sets the per-object output transform of every PBR object.
    pub fn set_tone_mapping(&mut self, tone_mapping: ToneMapping) {
        self.pbr_tone_mappings.fill(tone_mapping);
    }

    /// Sets the output transform of one PBR object.
    pub fn set_mesh_tone_mapping(&mut self, mesh_id: usize, tone_mapping: ToneMapping) {
        if let Some(current) = self.pbr_tone_mappings.get_mut(mesh_id) {
            *current = tone_mapping;
        }
    }

    /// Selects a diagnostic PBR output for one mesh.
    pub fn set_mesh_pbr_debug_view(&mut self, mesh_id: usize, debug_view: PbrDebugView) {
        if let Some(current) = self.pbr_debug_views.get_mut(mesh_id) {
            *current = debug_view;
        }
    }

    /// Binds `env` as the equirectangular HDR environment map reflected by
    /// [`RenderMode::Shaded`] draws. The probe is (re)uploaded lazily on the next
    /// [`render`](Self::render). Until set, PBR draws use no
    /// environment reflection (a 1×1 black probe keeps the bind group valid).
    pub fn set_env_map(&mut self, env: EnvMapData) {
        self.env.set(&self.gpu, env);
    }

    /// Uploads `rgba` (tightly-packed, row-major `height`×`width`×4) as the
    /// **background frame texture** (#63) sampled by a
    /// [`DrawableObject::FramePlane`]. Delegates to [`FramePlane::upload_rgba`],
    /// which reuses the GPU texture across same-resolution frames.
    ///
    /// Panics if `rgba.len() != width * height * 4` or either dimension is zero.
    pub fn update_frame_texture_rgba(&mut self, rgba: &[u8], width: u32, height: u32) {
        self.frame_plane.upload_rgba(&self.gpu, rgba, width, height);
    }

    /// Uploads `image` as the **background frame texture** (#63) sampled by a
    /// [`DrawableObject::FramePlane`]. The GPU texture is reused across frames
    /// (grown only on a resolution change). Call before a
    /// [`render`](Self::render) with a `FramePlane` drawable to
    /// composite the image beneath the mesh scene.
    pub fn update_frame_texture(&mut self, image: &crate::texture::ImageData) {
        self.update_frame_texture_rgba(&image.rgba, image.width, image.height);
    }

    /// Whether a background frame texture is currently bound (so a
    /// [`DrawableObject::FramePlane`] would render).
    pub fn has_frame_texture(&self) -> bool {
        self.frame_plane.is_bound()
    }

    /// The size of the object-id pick target, or `None` if nothing has been
    /// picked yet (it is allocated on the first [`pick`](Self::pick)). Diagnostic
    /// only — front-ends surface it in their debug panels.
    pub fn pick_target_size(&self) -> Option<(u32, u32)> {
        self.pick_target.as_ref().map(PickTarget::size)
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
        scene: &[DrawableObject],
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
        scene: &[DrawableObject],
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
    fn render_texture(&mut self, camera: Camera, scene: &[DrawableObject], target: &TextureTarget) {
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
        scene: &[DrawableObject],
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
        scene: &[DrawableObject],
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
        scene: &[DrawableObject],
    ) {
        self.encode_pass(encoder, view, camera, scene, true);
    }

    #[allow(clippy::too_many_arguments)]
    fn encode_pass(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        camera: Camera,
        scene: &[DrawableObject],
        load_color: bool,
    ) {
        // 1. Camera P·V for this frame.
        self.pipelines.write_camera(&self.gpu.queue, camera);
        // 1b. Disney PBR uniform slots for this frame — one per mesh (each carries
        //     the shared camera/lights + that mesh's material, #141). Written
        //     unconditionally so a PBR draw always has a current material slot.
        let viewport = camera.viewport();
        self.pipelines.write_pbr(
            &self.gpu.queue,
            camera,
            PbrBatchInputs {
                materials: &self.pbr_materials,
                ibl: &self.pbr_ibl,
                tone_mappings: &self.pbr_tone_mappings,
                debug_views: &self.pbr_debug_views,
                lighting: self.lighting,
            },
            self.env.has_env(),
        );

        // 2. Walk the scene once into per-geometry instance batches, then upload
        //    the flattened instance models (growing the buffer if needed).
        let batches = build_batches(scene, |mesh_id| {
            self.store.meshes.get(mesh_id).map(|mesh| mesh.base_model)
        });
        self.store.upload_instances(&self.gpu, &batches.instances);

        // 3. Match the depth + (when MSAA is on) color attachments to the viewport
        //    (solid meshes z-occlude; the multisampled color, if any, is resolved
        //    into `view`).
        self.ensure_depth(viewport);
        self.ensure_msaa(viewport);

        // 4. Background frame-plane fit for this viewport (no-op if the scene has
        //    no FramePlane or no frame texture is bound yet).
        if let Some(fit) = batches.frame_fit {
            self.frame_plane.write_fit(&self.gpu.queue, fit, viewport);
        }

        // 5. Bind groups for each mesh's own albedo (#141) and material maps, and
        //    for the HDR environment map. Nothing is uploaded here: now that the
        //    renderer holds the &self.gpu.queue, every setter uploads immediately and the
        //    constructors upload their fallbacks, so `encode` only *reads* bind
        //    groups that are already valid (#180).
        let env_bind_group = self.env.bind_group();
        if let Some(([rotation, exposure, blur], tonemap)) = batches.environment_background {
            self.env_background.write(
                &self.gpu.queue,
                camera,
                EnvBackgroundSettings {
                    rotation,
                    exposure,
                    blur,
                    tonemap,
                },
            );
        }

        // 6. Record the pass. With MSAA (`sample_count > 1`) the mesh pass renders
        //    into the multisampled color attachment and resolves into the caller's
        //    single-sample `view`, so every front-end (offscreen CLI, native
        //    window, wasm canvas) gets multisampled mesh/arrowhead edges.
        //    Without MSAA (`sample_count == 1`) there is no MSAA target — the pass
        //    renders straight into `view` (no resolve).
        let depth_view = &self.depth.as_ref().expect("depth set in step 3").view;
        let color_load = if load_color {
            wgpu::LoadOp::Load
        } else {
            wgpu::LoadOp::Clear(wgpu::Color::BLACK)
        };
        let color_attachment = match self.msaa.as_ref() {
            Some(msaa) => wgpu::RenderPassColorAttachment {
                view: &msaa.view,
                depth_slice: None,
                resolve_target: Some(view),
                ops: wgpu::Operations {
                    load: color_load,
                    store: wgpu::StoreOp::Store,
                },
            },
            None => wgpu::RenderPassColorAttachment {
                view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: color_load,
                    store: wgpu::StoreOp::Store,
                },
            },
        };
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

        if batches.environment_background.is_some() {
            self.env_background.draw(&mut pass, env_bind_group);
        }

        // Draw the background frame plane first (#63): its own pipeline + group-0
        // bind, depth-write off, so the mesh scene composites on top. Only when
        // the scene requested one (and a frame texture is bound).
        if batches.frame_fit.is_some() {
            self.frame_plane.draw(&mut pass);
        }

        // The instance buffer (slot 1) stays bound across every draw. Most
        // commands use the camera bind group; expanded lines briefly swap in the
        // viewport-aware gizmo bind group.
        pass.set_bind_group(0, &self.pipelines.camera_bind_group, &[]);
        pass.set_vertex_buffer(1, self.store.instance_buffer.slice(..));
        for command in &batches.commands {
            let range = command.start..command.start + command.count;
            match command.kind {
                DrawKind::Filled(id) => {
                    let mesh = &self.store.meshes[id];
                    pass.set_pipeline(&self.pipelines.filled);
                    draw_indexed(&mut pass, mesh.filled(), range);
                }
                DrawKind::Textured(id) => {
                    let mesh = &self.store.meshes[id];
                    pass.set_pipeline(&self.pipelines.textured);
                    pass.set_bind_group(1, mesh.texture.bind_group(), &[]);
                    draw_indexed(&mut pass, mesh.filled(), range);
                }
                DrawKind::Shaded(id) => {
                    let mesh = &self.store.meshes[id];
                    pass.set_pipeline(&self.pipelines.pbr);
                    // group 0 = this mesh's PbrUniform slot (selected by a dynamic
                    // offset), group 1 = this mesh's albedo, group 2 = HDR env map.
                    let offset = (id as u64 * self.pipelines.pbr_stride) as u32;
                    pass.set_bind_group(0, &self.pipelines.pbr_bind_group, &[offset]);
                    pass.set_bind_group(1, mesh.texture.bind_group(), &[]);
                    pass.set_bind_group(2, env_bind_group, &[]);
                    pass.set_bind_group(3, mesh.material_maps.bind_group(), &[]);
                    draw_indexed(&mut pass, mesh.pbr(), range);
                    // Restore group 0 = camera for the following non-PBR draws
                    // (their pipelines' group-0 layout is the camera uniform).
                    pass.set_bind_group(0, &self.pipelines.camera_bind_group, &[]);
                }
                DrawKind::Wireframe(id) => {
                    let mesh = &self.store.meshes[id];
                    pass.set_pipeline(&self.pipelines.wireframe);
                    draw_indexed(&mut pass, mesh.wireframe(), range);
                }
                DrawKind::Aabb(id) => {
                    let mesh = &self.store.meshes[id];
                    pass.set_pipeline(&self.pipelines.gizmo_line);
                    pass.set_bind_group(0, &self.pipelines.gizmo_bind_group, &[]);
                    draw_vertices(&mut pass, mesh.aabb(), range);
                    pass.set_bind_group(0, &self.pipelines.camera_bind_group, &[]);
                }
                DrawKind::Grid(plane) => {
                    pass.set_pipeline(&self.pipelines.gizmo_line);
                    pass.set_bind_group(0, &self.pipelines.gizmo_bind_group, &[]);
                    draw_vertices(&mut pass, &self.store.grid_lines[plane], range);
                    pass.set_bind_group(0, &self.pipelines.camera_bind_group, &[]);
                }
                DrawKind::QuadOutline(selected) => {
                    pass.set_pipeline(&self.pipelines.gizmo_line);
                    pass.set_bind_group(0, &self.pipelines.gizmo_bind_group, &[]);
                    draw_vertices(&mut pass, &self.store.quad_lines[selected], range);
                    pass.set_bind_group(0, &self.pipelines.camera_bind_group, &[]);
                }
                DrawKind::Shadow => {
                    pass.set_pipeline(&self.pipelines.shadow);
                    pass.set_vertex_buffer(0, self.store.shadow_vertex_buffer.slice(..));
                    pass.draw(0..SHADOW_VERTEX_COUNT, range);
                }
                DrawKind::Axes => {
                    pass.set_pipeline(&self.pipelines.gizmo_line);
                    pass.set_bind_group(0, &self.pipelines.gizmo_bind_group, &[]);
                    draw_vertices(&mut pass, &self.store.axes_lines, range.clone());
                    pass.set_pipeline(&self.pipelines.gizmo_solid);
                    pass.set_bind_group(0, &self.pipelines.camera_bind_group, &[]);
                    draw_vertices(&mut pass, &self.store.axes_heads, range);
                }
            }
        }
    }

    /// Encodes the **object-id picking pass** (#141): renders each `draws` entry's
    /// mesh in a flat color encoding its **index** (the same 0-based order the
    /// caller placed them), single-sampled and depth-tested into `color_view`
    /// (cleared to id `0` = background) with `depth_view`. No lighting, no
    /// texture, no MSAA — so the pixel under the cursor reads back to an exact id
    /// via [`PickInstanceRaw::decode`]. `color_view` must be a [`PICK_FORMAT`]
    /// (linear) target and `depth_view` a [`DEPTH_FORMAT`] attachment of the same
    /// size. Out-of-range mesh ids and `Shadow` draws are skipped, but the index
    /// mapping is preserved (a skipped draw's index simply never appears).
    ///
    /// `pub(crate)`: only [`PickTarget`] calls this, from its own [`pick`](Self::pick)
    /// per-call setup; a front-end reaches it through [`pick`](Self::pick) instead.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn encode_picking(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        color_view: &wgpu::TextureView,
        depth_view: &wgpu::TextureView,
        camera: Camera,
        draws: &[Draw],
    ) {
        // Camera P·V for this frame (writes the shared camera uniform bound by
        // `camera_bind_group`, which is layout-compatible with the pick pipeline).
        self.pipelines.write_camera(&self.gpu.queue, camera);

        // Build one pick instance per drawable object, carrying its index color.
        // Keep the draw index as the id even when an entry is skipped, so a decoded
        // id maps straight back to `draws[index]`.
        let mut instances: Vec<PickInstanceRaw> = Vec::with_capacity(draws.len());
        let mut records: Vec<(usize, u32)> = Vec::with_capacity(draws.len());
        // A shadow blob has no mesh geometry to hit-test, so it is not pickable.
        for (index, draw) in draws.iter().enumerate() {
            if !draw.selection.is_mesh() {
                continue;
            }
            let Some(mesh) = self.store.meshes.get(draw.mesh_id as usize) else {
                continue;
            };
            let effective = Matrix4::from_cols_array(&draw.model) * mesh.base_model;
            let slot = instances.len() as u32;
            instances.push(PickInstanceRaw::new(
                effective.to_cols_array(),
                index as u32,
            ));
            records.push((draw.mesh_id as usize, slot));
        }

        // Grow + upload the pick instance buffer.
        if instances.len() as u32 > self.pick_instance_capacity {
            self.pick_instance_capacity = (instances.len() as u32).next_power_of_two();
            self.pick_instances = self.gpu.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("trd pick instance buffer"),
                size: self.pick_instance_capacity as u64
                    * std::mem::size_of::<PickInstanceRaw>() as u64,
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
        }
        if !instances.is_empty() {
            self.gpu
                .queue
                .write_buffer(&self.pick_instances, 0, bytemuck::cast_slice(&instances));
        }

        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("trd picking pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: color_view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    // Clear to id 0 (background).
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
            })],
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

        pass.set_pipeline(&self.pick_pipeline);
        pass.set_bind_group(0, &self.pipelines.camera_bind_group, &[]);
        pass.set_vertex_buffer(1, self.pick_instances.slice(..));
        for (mesh_id, slot) in records {
            let mesh = &self.store.meshes[mesh_id];
            draw_indexed(&mut pass, mesh.filled(), slot..slot + 1);
        }
    }

    /// Ensures the depth attachment matches `viewport` (each dimension clamped to
    /// ≥ 1) at the renderer's [`sample_count`](Self::sample_count) (the depth
    /// sample count must match the color attachment), recreating it only when the
    /// target size changes.
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
                self.sample_count,
            ));
        }
    }

    /// Ensures the multisampled color attachment matches `viewport` (each
    /// dimension clamped to ≥ 1) at the renderer's
    /// [`sample_count`](Self::sample_count) and color `format`, recreating it only
    /// when the target size changes. When MSAA is disabled (`sample_count == 1`)
    /// no MSAA target is needed — the pass renders straight into the caller's
    /// single-sample `view` — so this clears it to `None`.
    fn ensure_msaa(&mut self, viewport: Viewport) {
        if self.sample_count <= 1 {
            self.msaa = None;
            return;
        }
        let dw = viewport.width.max(1);
        let dh = viewport.height.max(1);
        if self
            .msaa
            .as_ref()
            .is_none_or(|m| m.width != dw || m.height != dh)
        {
            self.msaa = Some(create_msaa_color_target(
                &self.gpu.device,
                self.format,
                dw,
                dh,
                self.sample_count,
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
        // Move the pick target out of `self` for the duration of the pass: `pick`
        // takes `&mut Renderer` to encode through, and a field already borrowed
        // out of `self` would conflict with re-borrowing all of `self` (#203).
        let pick_target = match self.pick_target.take() {
            Some(mut target) => {
                target.resize(&gpu.device, w, h);
                target
            }
            None => PickTarget::new(&gpu.device, w, h),
        };
        let id = pick_target.pick(&gpu, self, camera, draws, x, y).await;
        self.pick_target = Some(pick_target);
        id
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::visual::{FrameFit, GridPlane, RenderMode};

    fn model(tag: f32) -> [f32; 16] {
        let mut model = Matrix4::IDENTITY.to_cols_array();
        model[12] = tag;
        model
    }

    fn mesh(mesh_id: u32, tag: f32, mode: RenderMode) -> DrawableObject {
        DrawableObject::Mesh {
            mesh_id,
            model: model(tag),
            mode,
        }
    }

    #[test]
    fn batches_in_layer_order_and_preserves_equal_kind_order() {
        let scene = [
            DrawableObject::CoordinateAxes { model: model(80.0) },
            mesh(1, 61.0, RenderMode::Wireframe),
            DrawableObject::FramePlane {
                fit: FrameFit::Stretch,
            },
            mesh(1, 12.0, RenderMode::Filled),
            mesh(0, 30.0, RenderMode::Shaded),
            DrawableObject::BlobShadow { model: model(1.0) },
            DrawableObject::AabbBox {
                mesh_id: 1,
                model: model(71.0),
            },
            DrawableObject::PlaneGrid {
                plane: GridPlane::Yz,
                model: model(52.0),
            },
            mesh(0, 10.0, RenderMode::Filled),
            mesh(0, 20.0, RenderMode::Textured),
            DrawableObject::PlaneGrid {
                plane: GridPlane::Xy,
                model: model(50.0),
            },
            mesh(0, 11.0, RenderMode::Filled),
            DrawableObject::AabbBox {
                mesh_id: 0,
                model: model(70.0),
            },
            mesh(1, 31.0, RenderMode::Shaded),
            mesh(99, 99.0, RenderMode::Filled),
            DrawableObject::FramePlane {
                fit: FrameFit::Cover,
            },
        ];
        let base_models = [Matrix4::IDENTITY, Matrix4::IDENTITY];

        let batches = build_batches(&scene, |mesh_id| base_models.get(mesh_id).copied());
        let commands = batches
            .commands
            .iter()
            .map(|command| (command.kind, command.start, command.count))
            .collect::<Vec<_>>();

        assert_eq!(
            commands,
            [
                (DrawKind::Shadow, 0, 1),
                (DrawKind::Filled(0), 1, 2),
                (DrawKind::Filled(1), 3, 1),
                (DrawKind::Textured(0), 4, 1),
                (DrawKind::Shaded(0), 5, 1),
                (DrawKind::Shaded(1), 6, 1),
                (DrawKind::Grid(0), 7, 1),
                (DrawKind::Grid(2), 8, 1),
                (DrawKind::Wireframe(1), 9, 1),
                (DrawKind::Aabb(0), 10, 1),
                (DrawKind::Aabb(1), 11, 1),
                (DrawKind::Axes, 12, 1),
            ]
        );
        assert_eq!(
            batches
                .instances
                .iter()
                .map(|instance| instance.model[12])
                .collect::<Vec<_>>(),
            [1.0, 10.0, 11.0, 12.0, 20.0, 30.0, 31.0, 50.0, 52.0, 61.0, 70.0, 71.0, 80.0,]
        );
        assert_eq!(batches.frame_fit, Some(FrameFit::Cover));
    }
}
