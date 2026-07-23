//! wgpu render-pipeline, bind-group-layout, depth-target, and camera
//! uniform construction helpers.

use super::{FrameParams, InstanceRaw, Uniform, Vertex, Viewport};

/// Builds the indexed mesh render pipeline for `format` using an auto bind-group
/// layout (group 0, binding 0 = the params uniform), drawn as filled triangles.
pub fn create_mesh_pipeline(
    device: &wgpu::Device,
    format: wgpu::TextureFormat,
) -> wgpu::RenderPipeline {
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("trd mesh pipeline layout"),
        bind_group_layouts: &[Some(&create_mesh_bind_group_layout(device))],
        immediate_size: 0,
    });
    create_mesh_pipeline_with(
        device,
        format,
        &layout,
        wgpu::PrimitiveTopology::TriangleList,
        None,
    )
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

/// Depth state for **line overlays** (wireframe, AABB boxes, coordinate axes):
/// always pass and never write, so they composite on top of the solid meshes in
/// submission order (preserving the pre-depth-buffer overlay behavior) while
/// still being valid in a pass that carries a depth attachment.
pub(crate) fn overlay_depth_stencil() -> wgpu::DepthStencilState {
    wgpu::DepthStencilState {
        format: DEPTH_FORMAT,
        depth_write_enabled: Some(false),
        depth_compare: Some(wgpu::CompareFunction::Always),
        stencil: wgpu::StencilState::default(),
        bias: wgpu::DepthBiasState::default(),
    }
}

/// A depth attachment sized to a render target. The [`MeshRenderer`] owns one
/// and recreates it when the viewport changes, so the mesh pass always has a
/// matching depth buffer for solid occlusion.
pub(crate) struct DepthTarget {
    pub(crate) view: wgpu::TextureView,
    pub(crate) width: u32,
    pub(crate) height: u32,
}

/// Creates a [`DEPTH_FORMAT`] depth texture + view of `width`×`height` (each
/// clamped to ≥ 1) for use as a render-pass depth attachment.
pub(crate) fn create_depth_target(device: &wgpu::Device, width: u32, height: u32) -> DepthTarget {
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
        sample_count: 1,
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
) -> wgpu::RenderPipeline {
    let shader = device.create_shader_module(wgpu::include_wgsl!("../mesh.wgsl"));
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
        multisample: wgpu::MultisampleState::default(),
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
) -> wgpu::RenderPipeline {
    let shader = device.create_shader_module(wgpu::include_wgsl!("../textured.wgsl"));
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
        multisample: wgpu::MultisampleState::default(),
        multiview_mask: None,
        cache: None,
    })
}

/// The group-0 bind-group layout for the background frame-plane pipeline (#63):
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
) -> wgpu::RenderPipeline {
    let shader = device.create_shader_module(wgpu::include_wgsl!("../frame_plane.wgsl"));
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
        multisample: wgpu::MultisampleState::default(),
        multiview_mask: None,
        cache: None,
    })
}
/// Creates the camera `P·V` uniform buffer + bind group over an **explicit**
/// bind-group layout (shared by the filled and wireframe mesh pipelines),
/// initialised to `params`'s view-projection for `viewport`. Used by
/// [`MeshRenderer`], whose two pipelines must share one bind group.
pub(crate) fn create_view_proj_binding(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    params: FrameParams,
    viewport: Viewport,
) -> (wgpu::Buffer, wgpu::BindGroup) {
    use wgpu::util::DeviceExt;
    let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("trd view-proj uniform"),
        contents: bytemuck::bytes_of(&Uniform::view_proj(params, viewport)),
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
/// [`MeshRenderer`] reuse one uniform buffer across frames instead of rebuilding
/// it.
pub(crate) fn write_view_proj(
    queue: &wgpu::Queue,
    buffer: &wgpu::Buffer,
    params: FrameParams,
    viewport: Viewport,
) {
    queue.write_buffer(
        buffer,
        0,
        bytemuck::bytes_of(&Uniform::view_proj(params, viewport)),
    );
}
