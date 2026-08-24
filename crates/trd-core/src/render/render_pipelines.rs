//! The render pass's **machinery**: the pipelines every draw kind rasterizes
//! through, and the group-0 uniforms they read.
//!
//! Both are `f(format, sample_count, mesh_count)` — a function of the *device*
//! and the pipeline setup, not of any particular [`Scene`](crate::Scene): two
//! scenes rendered by one [`Renderer`](super::Renderer) share them. They are
//! built together by [`create_render_pipelines`] and live here rather than in
//! `renderer.rs` (#221 §2) or on the scene (which is `Clone + PartialEq` and
//! device-free).

use super::mesh_store::MeshGpu;
use super::*;
use crate::Camera;

/// The mesh, gizmo and PBR **pipelines** — one field per way a draw kind
/// rasterizes, and nothing else.
///
/// It used to be `SceneRenderPipelines`, which also carried the three uniform
/// buffers, their bind groups and the PBR slot stride: a name that promised a
/// pipeline collection over a struct holding three different kinds of thing
/// (#203). The uniforms those pipelines read now live in [`SceneUniforms`], so
/// each type answers one question — *what draws* vs *what it reads*.
///
/// Named for the **renderer**, not for the scene and not for meshes: it is
/// `f(format, sample_count, mesh_count)` — device-level state a
/// [`Renderer`](super::Renderer) builds once and every [`Scene`](crate::Scene)
/// it draws shares (#235 R10) — and of its seven pipelines four (`gizmo_line`,
/// `gizmo_solid`, `shadow`, and the background draws routed beside them) draw no
/// mesh geometry at all. Its sibling [`SceneUniforms`] keeps the *scene* name
/// because that half genuinely is per-frame scene state (camera + light rig,
/// #182), and the one function building both is [`create_render_pipelines`]
/// (#221 §2).
///
/// Filled, wireframe, arrowheads, shadows and textured rendering share the
/// camera group-0 layout; expanded gizmo lines use a viewport-aware one, and PBR
/// a dynamic-offset one — which is why the encode arm restores the camera bind
/// group after switching.
pub(crate) struct RenderPipelines {
    pub(super) filled: wgpu::RenderPipeline,
    pub(super) wireframe: wgpu::RenderPipeline,
    /// Screen-space expanded, alpha-feathered AABB/axes/grid line pipeline.
    pub(super) gizmo_line: wgpu::RenderPipeline,
    /// Unlit overlay triangles for coordinate-axis arrowheads.
    pub(super) gizmo_solid: wgpu::RenderPipeline,
    pub(super) textured: wgpu::RenderPipeline,
    /// The contact / blob grounding-shadow pipeline (alpha-blended, depth-write
    /// off); shares the untextured camera bind-group layout (group 0).
    pub(super) shadow: wgpu::RenderPipeline,
    /// The placement-quad highlight wash (alpha-blended, depth-write off); same
    /// layout and unit-quad geometry as [`shadow`](Self::shadow), flat green
    /// fragment instead of a feathered dark blob.
    pub(super) quad_fill: wgpu::RenderPipeline,
    /// The Disney PBR pipeline (`pbr.wgsl`): group 0 = [`SceneUniforms::pbr`]'s
    /// slot for the drawn mesh, group 1 = the bound albedo texture, group 2 =
    /// the HDR environment map, group 3 = the material maps.
    pub(super) pbr: wgpu::RenderPipeline,
}

/// The group-0 uniforms a scene pass binds, one per binding discipline (#203).
///
/// `camera` and `gizmo` are whole-buffer bindings rewritten once per frame;
/// `pbr` is a slot array indexed per draw. Each is a buffer + the bind group
/// exposing it as a single value ([`BoundUniform`] / [`BoundSceneSlots`]),
/// rather than six loose fields kept in step by naming convention — the same
/// "bound resource" shape as [`BoundTexture`](super::BoundTexture) and
/// [`BoundMaterialMaps`](super::bound_material_maps::BoundMaterialMaps).
pub(crate) struct SceneUniforms {
    /// The camera `P·V` read by every non-gizmo, non-PBR pipeline.
    pub(super) camera: BoundUniform,
    /// The viewport-aware params the expanded-line gizmo pipeline reads.
    pub(super) gizmo: BoundUniform,
    /// Group 0 of the PBR pipeline: the once-per-frame scene uniform (camera +
    /// light rig) at binding 0, and one `PbrUniform` slot per mesh at binding 1,
    /// which a PBR draw selects with a dynamic offset (#182).
    pub(super) pbr: BoundSceneSlots,
}

/// Builds the render pipelines and the uniforms they bind, together.
///
/// One function for both halves because each group-0 bind-group layout is
/// shared by exactly one pipeline and the binding feeding it — the camera layout
/// by the untextured/textured pipelines and `camera`, the gizmo layout by
/// `gizmo_line` and `gizmo`, the `has_dynamic_offset` PBR layout by `pbr` and
/// `pbr` — so creating them apart would either duplicate the layouts or leave
/// that pairing implicit. `texture_layout` is the albedo texture's group-1
/// layout (from [`BoundTexture`](super::BoundTexture)), shared by the textured
/// and PBR pipelines; `env_layout` is the PBR pipeline's group-2
/// environment-map layout (from [`BoundEnv`]). Every pipeline in the pass shares
/// the one `sample_count` (`1` = no MSAA, single-sample), and the PBR slot array
/// is sized for `mesh_count` meshes.
pub(crate) fn create_render_pipelines(
    device: &wgpu::Device,
    format: wgpu::TextureFormat,
    texture_layout: &wgpu::BindGroupLayout,
    material_maps_layout: &wgpu::BindGroupLayout,
    env_layout: &wgpu::BindGroupLayout,
    sample_count: u32,
    mesh_count: usize,
) -> (RenderPipelines, SceneUniforms) {
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
    let gizmo_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("trd gizmo pipeline layout"),
        bind_group_layouts: &[Some(&gizmo_layout)],
        immediate_size: 0,
    });
    let gizmo_line =
        create_gizmo_line_pipeline(device, format, &gizmo_pipeline_layout, sample_count);
    // Contact / blob grounding-shadow pipeline (#110 follow-up): shares the
    // untextured camera layout (group 0), alpha-blended, depth-write off.
    let shadow = create_shadow_pipeline(device, format, &pipeline_layout, sample_count);
    // The quad highlight wash reuses that layout and geometry wholesale — only
    // its fragment differs.
    let quad_fill = create_quad_fill_pipeline(device, format, &pipeline_layout, sample_count);
    // Textured pipeline (#20): group 0 = the shared camera uniform, group 1 =
    // the bound albedo texture + sampler.
    let textured_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
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
    // A placeholder camera: the buffers only need to exist here, and every
    // frame's `write_camera` overwrites them with the real one.
    let placeholder = FrameParams::IDENTITY
        .to_camera(Viewport {
            width: 1,
            height: 1,
        })
        .expect("the identity params are a valid camera form");
    let uniforms = SceneUniforms {
        camera: create_view_proj_binding(device, &camera_layout, placeholder),
        gizmo: create_gizmo_binding(device, &gizmo_layout, placeholder),
        // The per-object PbrUniform slots: one per mesh, each rewritten every
        // frame with the shared camera/lights + that mesh's material; a PBR
        // draw selects its slot via a dynamic offset.
        pbr: BoundSceneSlots::new::<PbrSceneUniform, PbrUniform>(
            device,
            &pbr_layout,
            "trd pbr",
            mesh_count,
        ),
    };
    let pipelines = RenderPipelines {
        filled,
        wireframe,
        gizmo_line,
        gizmo_solid,
        textured,
        shadow,
        quad_fill,
        pbr,
    };
    (pipelines, uniforms)
}

impl SceneUniforms {
    /// Widens the PBR slot array to one slot per mesh after a runtime mesh add
    /// (#353). The layout is a pure function of the device, so it is rebuilt
    /// here rather than retained — [`create_render_pipelines`] drops its own.
    pub(super) fn grow_pbr_slots(&mut self, device: &wgpu::Device, meshes: usize) {
        let layout = create_pbr_bind_group_layout(device);
        self.pbr.grow(device, &layout, meshes);
    }

    /// Rewrites the camera `P·V` uniform for this frame's `camera`.
    pub(super) fn write_camera(&self, queue: &wgpu::Queue, camera: Camera) {
        write_view_proj(queue, &self.camera, camera);
        write_gizmo_params(queue, &self.gizmo, camera);
    }

    /// Rewrites the Disney PBR uniforms for this frame, **split by frequency of
    /// change** (#182): the camera terms + the scene's light rig **once**, then
    /// one slot per mesh (mesh id → slot id) carrying only that mesh's
    /// material/IBL/tone-map/debug view. A PBR draw binds its object's slot via
    /// a dynamic offset.
    ///
    /// The rig used to be re-encoded into every slot, so an N-object scene wrote
    /// N identical copies of the same lights each frame. The per-mesh values are
    /// read straight off the [`MeshGpu`]s that own them (#203): they used to be
    /// four `Vec`s on the renderer, all sized to the mesh count with nothing
    /// enforcing it, joined here by a four-deep `zip`.
    /// `write_slots` skips the per-mesh half when nothing has changed since the
    /// last frame (#235 R5) — the scene half is always written, because the
    /// camera moves every frame.
    pub(super) fn write_pbr(
        &self,
        queue: &wgpu::Queue,
        camera: Camera,
        meshes: &[Option<MeshGpu>],
        lighting: Lighting,
        use_env: bool,
        write_slots: bool,
    ) {
        let scene = PbrSceneUniform::new(
            camera.view_projection().matrix().to_cols_array(),
            camera.position(),
            lighting,
            use_env,
        );
        self.pbr.write_scene(queue, &scene);
        if !write_slots {
            return;
        }
        // A removed mesh leaves a hole whose slot nothing draws; skipping it
        // keeps every surviving mesh on the slot its id names.
        for (slot, mesh) in meshes.iter().enumerate() {
            let Some(mesh) = mesh.as_ref() else {
                continue;
            };
            let appearance = mesh.appearance();
            let uniform = PbrUniform::new(PbrUniformInputs {
                material: &appearance.material,
                ibl: appearance.ibl,
                tone_mapping: appearance.tone_mapping,
                debug_view: appearance.debug_view,
            });
            self.pbr.write_slot(queue, slot, &uniform);
        }
    }
}
