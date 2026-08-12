//! wgpu render-pipeline, bind-group-layout, depth-target, and camera
//! uniform construction helpers.

use super::{
    GizmoLineVertex, GizmoUniform, InstanceRaw, PbrVertex, PickInstanceRaw, Uniform, Vertex,
};
use crate::Camera;

/// The mesh pass's MSAA sample count. 4× multisampling is the WebGPU-guaranteed
/// level for renderable formats (native Vulkan/Metal/DX + the WebGL2 downlevel
/// backend), so it needs no adapter feature check. It smooths mesh silhouettes,
/// hardware wireframes, and solid gizmo arrowheads; expanded AABB/axes/grid lines
/// also apply analytic AA in their shader, including when this count is `1`.
/// Every part of the mesh pass — the color attachment, the depth attachment, and
/// every participating pipeline — must share this count; the multisampled color
/// target is resolved into the caller's single-sample `view`.
pub(crate) const MSAA_SAMPLE_COUNT: u32 = 4;

/// A [`wgpu::MultisampleState`] for `sample_count` (full coverage mask, no
/// alpha-to-coverage). `sample_count == 1` is byte-identical to the wgpu default
/// (the non-MSAA / legacy pipelines).
pub(crate) fn multisample_state(sample_count: u32) -> wgpu::MultisampleState {
    wgpu::MultisampleState {
        count: sample_count,
        mask: !0,
        alpha_to_coverage_enabled: false,
    }
}

/// The explicit bind-group layout shared by every mesh pipeline (group 0,
/// binding 0 = the camera `P·V` uniform, vertex-stage visible). Making it
/// explicit (rather than auto-derived per pipeline) lets the filled and
/// wireframe pipelines share **one** layout, so a single params bind group is
/// valid for both regardless of the active [`RenderMode`].
pub(crate) fn create_mesh_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("trd mesh bind group layout"),
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::VERTEX,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }],
    })
}

/// Group-0 layout for analytic gizmo lines: camera `P·V` plus viewport pixel
/// dimensions in one vertex-stage uniform.
pub(crate) fn create_gizmo_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("trd gizmo bind group layout"),
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::VERTEX,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: wgpu::BufferSize::new(std::mem::size_of::<GizmoUniform>() as u64),
            },
            count: None,
        }],
    })
}

/// The depth buffer format used by the mesh pass. `Depth32Float` is guaranteed
/// by WebGPU (and supported by the GL/WebGL2 downlevel backend), matching the
/// renderer's portability target.
pub(crate) const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

/// Depth state for **solid** geometry (filled + textured): write depth and keep
/// the nearest fragment (`Less`, clip z ∈ [0, 1] with near at 0). This is what
/// makes an opaque mesh occlude its own back faces instead of the last-drawn
/// triangle winning (there is no submission-order z otherwise).
pub(crate) fn solid_depth_stencil() -> wgpu::DepthStencilState {
    wgpu::DepthStencilState {
        format: DEPTH_FORMAT,
        depth_write_enabled: Some(true),
        depth_compare: Some(wgpu::CompareFunction::Less),
        stencil: wgpu::StencilState::default(),
        bias: wgpu::DepthBiasState::default(),
    }
}

/// Depth state for overlays (wireframe, AABB boxes, grids, and coordinate axes):
/// always pass and never write, so they composite on top of the solid meshes in
/// submission order while remaining valid in a pass with a depth attachment.
pub(crate) fn overlay_depth_stencil() -> wgpu::DepthStencilState {
    wgpu::DepthStencilState {
        format: DEPTH_FORMAT,
        depth_write_enabled: Some(false),
        depth_compare: Some(wgpu::CompareFunction::Always),
        stencil: wgpu::StencilState::default(),
        bias: wgpu::DepthBiasState::default(),
    }
}

/// The color format of the object-id **picking** target: a **linear** (non-sRGB)
/// `Rgba8Unorm` so each fragment's flat id color is stored byte-exact and reads
/// back without an sRGB transfer (unlike the sRGB display [`OFFSCREEN_FORMAT`]).
pub(crate) const PICK_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;

/// Builds the object-id picking pipeline: the mesh vertex transform (`clip =
/// P·V·M·p`) over the camera bind-group layout (group 0), a per-instance
/// [`PickInstanceRaw`] carrying the flat id color, single-sampled into
/// [`PICK_FORMAT`] with solid depth testing (so the nearest object wins the
/// pixel). No MSAA — ids must not be averaged at edges.
pub(crate) fn create_picking_pipeline(
    device: &wgpu::Device,
    layout: &wgpu::PipelineLayout,
) -> wgpu::RenderPipeline {
    let shader = device.create_shader_module(wgpu::include_wgsl!("../shader/picking.wgsl"));
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("trd picking pipeline"),
        layout: Some(layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some("vs_main"),
            compilation_options: Default::default(),
            buffers: &[Some(Vertex::layout()), Some(PickInstanceRaw::layout())],
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: Some("fs_main"),
            compilation_options: Default::default(),
            targets: &[Some(PICK_FORMAT.into())],
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            ..Default::default()
        },
        depth_stencil: Some(solid_depth_stencil()),
        multisample: multisample_state(1),
        multiview_mask: None,
        cache: None,
    })
}

/// A depth attachment sized to a render target. The [`SceneRenderer`] owns one
/// and recreates it when the viewport changes, so the mesh pass always has a
/// matching depth buffer for solid occlusion. `sample_count` must match the
/// pass's color attachment (MSAA depth is not resolved — it is discardable).
pub(crate) struct DepthTarget {
    pub(crate) view: wgpu::TextureView,
    pub(crate) width: u32,
    pub(crate) height: u32,
}

/// Creates a [`DEPTH_FORMAT`] depth texture + view of `width`×`height` (each
/// clamped to ≥ 1) and `sample_count` for use as a render-pass depth attachment.
pub(crate) fn create_depth_target(
    device: &wgpu::Device,
    width: u32,
    height: u32,
    sample_count: u32,
) -> DepthTarget {
    let width = width.max(1);
    let height = height.max(1);
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("trd depth texture"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count,
        dimension: wgpu::TextureDimension::D2,
        format: DEPTH_FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    DepthTarget {
        view,
        width,
        height,
    }
}

/// A multisampled color attachment sized to a render target. The [`SceneRenderer`]
/// owns one (mirroring [`DepthTarget`]) and recreates it when the viewport
/// changes; each frame the mesh pass renders into it and resolves it into the
/// caller's single-sample `view`. Only present when `sample_count > 1`.
pub(crate) struct MsaaColorTarget {
    pub(crate) view: wgpu::TextureView,
    pub(crate) width: u32,
    pub(crate) height: u32,
}

/// Creates a `format` multisampled color texture + view of `width`×`height`
/// (each clamped to ≥ 1) and `sample_count`, usable only as a render attachment
/// (its contents are resolved, never copied/sampled).
pub(crate) fn create_msaa_color_target(
    device: &wgpu::Device,
    format: wgpu::TextureFormat,
    width: u32,
    height: u32,
    sample_count: u32,
) -> MsaaColorTarget {
    let width = width.max(1);
    let height = height.max(1);
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("trd msaa color texture"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    MsaaColorTarget {
        view,
        width,
        height,
    }
}

/// Builds an indexed mesh pipeline for `format` and `topology` (filled
/// `TriangleList` or wireframe `LineList`) over the shared explicit `layout`.
/// Both topologies use the same `mesh.wgsl` (the vertex shader only transforms
/// positions; line rasterization needs no extra WebGPU features). `depth_stencil`
/// is `None` for the standalone/legacy pass (no depth attachment) or a state
/// matching the mesh pass's [`DEPTH_FORMAT`] attachment.
pub(crate) fn create_mesh_pipeline_with(
    device: &wgpu::Device,
    format: wgpu::TextureFormat,
    layout: &wgpu::PipelineLayout,
    topology: wgpu::PrimitiveTopology,
    depth_stencil: Option<wgpu::DepthStencilState>,
    sample_count: u32,
) -> wgpu::RenderPipeline {
    let shader = device.create_shader_module(wgpu::include_wgsl!("../shader/mesh.wgsl"));
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("trd mesh pipeline"),
        layout: Some(layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some("vs_main"),
            compilation_options: Default::default(),
            buffers: &[Some(Vertex::layout()), Some(InstanceRaw::layout())],
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: Some("fs_main"),
            compilation_options: Default::default(),
            targets: &[Some(format.into())],
        }),
        primitive: wgpu::PrimitiveState {
            topology,
            ..Default::default()
        },
        depth_stencil,
        multisample: multisample_state(sample_count),
        multiview_mask: None,
        cache: None,
    })
}

/// Builds the analytic-AA gizmo line pipeline. Model-space segments are expanded
/// to triangle quads in screen space, then alpha-feathered in the fragment stage.
pub(crate) fn create_gizmo_line_pipeline(
    device: &wgpu::Device,
    format: wgpu::TextureFormat,
    layout: &wgpu::PipelineLayout,
    sample_count: u32,
) -> wgpu::RenderPipeline {
    let shader = device.create_shader_module(wgpu::include_wgsl!("../shader/gizmo_line.wgsl"));
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("trd gizmo line pipeline"),
        layout: Some(layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some("vs_main"),
            compilation_options: Default::default(),
            buffers: &[Some(GizmoLineVertex::layout()), Some(InstanceRaw::layout())],
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: Some("fs_main"),
            compilation_options: Default::default(),
            targets: &[Some(wgpu::ColorTargetState {
                format,
                blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            ..Default::default()
        },
        depth_stencil: Some(overlay_depth_stencil()),
        multisample: multisample_state(sample_count),
        multiview_mask: None,
        cache: None,
    })
}

/// The group-1 bind-group layout for the textured pipeline (#20): a filterable
/// `texture_2d<f32>` (binding 0) plus a filtering `sampler` (binding 1), both
/// fragment-stage visible.
pub(crate) fn create_texture_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("trd texture bind group layout"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            },
        ],
    })
}

/// Builds the textured `TriangleList` pipeline (#20): `textured.wgsl` over the
/// shared vertex/instance layout, with group 0 = the camera `P·V` uniform and
/// group 1 = the bound texture + sampler.
pub(crate) fn create_textured_pipeline(
    device: &wgpu::Device,
    format: wgpu::TextureFormat,
    layout: &wgpu::PipelineLayout,
    sample_count: u32,
) -> wgpu::RenderPipeline {
    let shader = device.create_shader_module(wgpu::include_wgsl!("../shader/textured.wgsl"));
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("trd textured pipeline"),
        layout: Some(layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some("vs_main"),
            compilation_options: Default::default(),
            buffers: &[Some(Vertex::layout()), Some(InstanceRaw::layout())],
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: Some("fs_main"),
            compilation_options: Default::default(),
            targets: &[Some(format.into())],
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            ..Default::default()
        },
        depth_stencil: Some(solid_depth_stencil()),
        multisample: multisample_state(sample_count),
        multiview_mask: None,
        cache: None,
    })
}

/// Builds the **blob-shadow** pipeline (contact / grounding shadow, #110
/// follow-up): `shadow.wgsl` over the shared vertex/instance layout, group 0 =
/// the camera `P·V` uniform (same untextured layout as the filled/wireframe
/// pipelines). Alpha-blended (`src·α + dst·(1−α)`) so the dark blob darkens the
/// background frame plane, with the depth test on but **depth-write off**
/// ([`overlay_depth_stencil`]) — the shadow is drawn *before* the opaque content
/// mesh and never occludes it, so the mesh composites cleanly on top.
pub(crate) fn create_shadow_pipeline(
    device: &wgpu::Device,
    format: wgpu::TextureFormat,
    layout: &wgpu::PipelineLayout,
    sample_count: u32,
) -> wgpu::RenderPipeline {
    let shader = device.create_shader_module(wgpu::include_wgsl!("../shader/shadow.wgsl"));
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("trd shadow pipeline"),
        layout: Some(layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some("vs_main"),
            compilation_options: Default::default(),
            buffers: &[Some(Vertex::layout()), Some(InstanceRaw::layout())],
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: Some("fs_main"),
            compilation_options: Default::default(),
            targets: &[Some(wgpu::ColorTargetState {
                format,
                blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            ..Default::default()
        },
        depth_stencil: Some(overlay_depth_stencil()),
        multisample: multisample_state(sample_count),
        multiview_mask: None,
        cache: None,
    })
}
/// a filterable `texture_2d<f32>` (binding 0) + a filtering `sampler` (binding 1),
/// both fragment-visible, plus a small **fit** uniform (binding 2, vertex-visible)
/// carrying the centered UV scale. Kept separate from the mesh albedo texture
/// (#62 §D1): same bind pattern, different update rate.
pub(crate) fn create_frame_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("trd frame plane bind group layout"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    })
}

/// Builds the fullscreen **background frame-plane** pipeline (#63):
/// `frame_plane.wgsl` with **no vertex buffers** (a shader-generated fullscreen
/// triangle), drawn first with depth writes disabled + compare `Always`
/// ([`overlay_depth_stencil`]) so the mesh scene composites on top.
pub(crate) fn create_frame_plane_pipeline(
    device: &wgpu::Device,
    format: wgpu::TextureFormat,
    layout: &wgpu::PipelineLayout,
    sample_count: u32,
) -> wgpu::RenderPipeline {
    let shader = device.create_shader_module(wgpu::include_wgsl!("../shader/frame_plane.wgsl"));
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("trd frame plane pipeline"),
        layout: Some(layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some("vs_main"),
            compilation_options: Default::default(),
            buffers: &[],
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: Some("fs_main"),
            compilation_options: Default::default(),
            targets: &[Some(format.into())],
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            ..Default::default()
        },
        depth_stencil: Some(overlay_depth_stencil()),
        multisample: multisample_state(sample_count),
        multiview_mask: None,
        cache: None,
    })
}
/// Creates the camera `P·V` uniform buffer + bind group over an **explicit**
/// bind-group layout (shared by the filled and wireframe mesh pipelines),
/// initialised to `camera`'s view-projection. Used by
/// [`SceneRenderer`], whose two pipelines must share one bind group.
pub(crate) fn create_view_proj_binding(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    camera: Camera,
) -> (wgpu::Buffer, wgpu::BindGroup) {
    use wgpu::util::DeviceExt;
    let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("trd view-proj uniform"),
        contents: bytemuck::bytes_of(&Uniform::view_proj(camera)),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("trd view-proj bind group"),
        layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: buffer.as_entire_binding(),
        }],
    });
    (buffer, bind_group)
}

/// Writes the camera-only `P · V` transform into an existing uniform buffer (the
/// instanced mesh path supplies each model matrix per instance). Lets
/// [`SceneRenderer`] reuse one uniform buffer across frames instead of rebuilding
/// it.
pub(crate) fn write_view_proj(queue: &wgpu::Queue, buffer: &wgpu::Buffer, camera: Camera) {
    queue.write_buffer(buffer, 0, bytemuck::bytes_of(&Uniform::view_proj(camera)));
}

/// Creates the viewport-aware gizmo uniform and bind group.
pub(crate) fn create_gizmo_binding(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    camera: Camera,
) -> (wgpu::Buffer, wgpu::BindGroup) {
    use wgpu::util::DeviceExt;
    let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("trd gizmo uniform"),
        contents: bytemuck::bytes_of(&GizmoUniform::new(camera)),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("trd gizmo bind group"),
        layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: buffer.as_entire_binding(),
        }],
    });
    (buffer, bind_group)
}

/// Updates the gizmo camera + viewport uniform for the current frame.
pub(crate) fn write_gizmo_params(queue: &wgpu::Queue, buffer: &wgpu::Buffer, camera: Camera) {
    queue.write_buffer(buffer, 0, bytemuck::bytes_of(&GizmoUniform::new(camera)));
}

/// The group-0 bind-group layout for the Disney PBR pipeline (#, `pbr.wgsl`):
/// a single `PbrUniform` (binding 0) visible to **both** the vertex stage (the
/// `P·V` transform) and the fragment stage (camera position, material, lights,
/// env/exposure controls).
pub(crate) fn create_pbr_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("trd pbr bind group layout"),
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                // Per-object material (#141): one `PbrUniform` slot per draw,
                // selected at draw time by a dynamic offset into the shared buffer.
                has_dynamic_offset: true,
                min_binding_size: wgpu::BufferSize::new(
                    std::mem::size_of::<super::PbrUniform>() as u64
                ),
            },
            count: None,
        }],
    })
}

/// The group-2 bind-group layout for the PBR pipeline's **environment map**: a
/// filterable `texture_2d<f32>` (binding 0) + a filtering `sampler` (binding 1),
/// both fragment-visible. Mirrors [`create_texture_bind_group_layout`], but the
/// texture is the equirectangular HDR probe (uploaded as `Rgba16Float`, which is
/// filterable on the downlevel target).
pub(crate) fn create_env_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("trd env bind group layout"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 3,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 4,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
        ],
    })
}

/// Builds the Disney **PBR** `TriangleList` pipeline: `pbr.wgsl` over the
/// dedicated [`PbrVertex`] (position + normal + UV) buffer plus the shared
/// [`InstanceRaw`] model buffer, with group 0 = the `PbrUniform`, group 1 = the
/// bound albedo texture + sampler, group 2 = the HDR environment map. Opaque
/// ([`solid_depth_stencil`]), multisampled to match the mesh pass.
pub(crate) fn create_pbr_pipeline(
    device: &wgpu::Device,
    format: wgpu::TextureFormat,
    layout: &wgpu::PipelineLayout,
    sample_count: u32,
) -> wgpu::RenderPipeline {
    let shader = device.create_shader_module(wgpu::include_wgsl!("../shader/pbr.wgsl"));
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("trd pbr pipeline"),
        layout: Some(layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some("vs_main"),
            compilation_options: Default::default(),
            buffers: &[Some(PbrVertex::layout()), Some(InstanceRaw::layout())],
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: Some("fs_main"),
            compilation_options: Default::default(),
            targets: &[Some(format.into())],
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            ..Default::default()
        },
        depth_stencil: Some(solid_depth_stencil()),
        multisample: multisample_state(sample_count),
        multiview_mask: None,
        cache: None,
    })
}
