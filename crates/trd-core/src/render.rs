//! Shared, platform-agnostic parametric triangle rendering.
//!
//! [`render_triangle`] draws the hello-triangle transformed by [`FrameParams`]
//! into the given texture view. Both the native batch renderer and the wasm
//! entry point build on [`create_triangle_pipeline`].

/// Per-frame transform parameters for the triangle.
///
/// The base triangle vertices `p_i` are transformed as
/// `p' = center + R(theta) * (size ⊙ p_i)` in the vertex shader.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FrameParams {
    /// Triangle centroid in NDC; `(0,0)` is screen center.
    pub center: [f32; 2],
    /// Per-axis scale; `(1,1)` is the base triangle.
    pub size: [f32; 2],
    /// Rotation in radians, counter-clockwise.
    pub theta: f32,
}

impl FrameParams {
    /// The identity transform: centered, unit scale, no rotation.
    pub const IDENTITY: FrameParams = FrameParams {
        center: [0.0, 0.0],
        size: [1.0, 1.0],
        theta: 0.0,
    };
}

/// GPU uniform matching the WGSL `Params` layout (32 bytes: vec2 center,
/// vec2 size, f32 theta, then padding to a 16-byte multiple).
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Uniform {
    center: [f32; 2],
    size: [f32; 2],
    theta: f32,
    _pad: [f32; 3],
}

impl From<FrameParams> for Uniform {
    fn from(p: FrameParams) -> Self {
        Uniform {
            center: p.center,
            size: p.size,
            theta: p.theta,
            _pad: [0.0; 3],
        }
    }
}

/// Builds the triangle render pipeline for `format` using an auto bind-group
/// layout (group 0, binding 0 = the params uniform).
pub fn create_triangle_pipeline(
    device: &wgpu::Device,
    format: wgpu::TextureFormat,
) -> wgpu::RenderPipeline {
    let shader = device.create_shader_module(wgpu::include_wgsl!("triangle.wgsl"));
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("trd triangle pipeline"),
        layout: None,
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
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        multiview_mask: None,
        cache: None,
    })
}

/// Creates the params uniform buffer + bind group for `pipeline`, initialised
/// to `params`.
pub(crate) fn create_params_binding(
    device: &wgpu::Device,
    pipeline: &wgpu::RenderPipeline,
    params: FrameParams,
) -> (wgpu::Buffer, wgpu::BindGroup) {
    use wgpu::util::DeviceExt;
    let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("trd params uniform"),
        contents: bytemuck::bytes_of(&Uniform::from(params)),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("trd params bind group"),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: buffer.as_entire_binding(),
        }],
    });
    (buffer, bind_group)
}

/// Writes `params` into an existing params uniform buffer.
pub(crate) fn write_params(queue: &wgpu::Queue, buffer: &wgpu::Buffer, params: FrameParams) {
    queue.write_buffer(buffer, 0, bytemuck::bytes_of(&Uniform::from(params)));
}

/// Draws the transformed triangle into `view`, clearing to black first.
///
/// Builds a fresh pipeline and uniform each call; intended for one-shot callers
/// (the wasm entry point). The batch renderer reuses a persistent pipeline.
pub fn render_triangle(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    view: &wgpu::TextureView,
    format: wgpu::TextureFormat,
    params: FrameParams,
) {
    let pipeline = create_triangle_pipeline(device, format);
    let (_buffer, bind_group) = create_params_binding(device, &pipeline, params);

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("trd triangle encoder"),
    });
    {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("trd triangle pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.draw(0..3, 0..1);
    }
    queue.submit(Some(encoder.finish()));
}
