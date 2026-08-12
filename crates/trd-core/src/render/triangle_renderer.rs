//! [`TriangleRenderer`]: the minimal canonical wgpu renderer, meant to be read
//! first.
//!
//! **Reference material and test scaffolding — not production code** (#202). It
//! has no consumer outside `render::gpu_tests`, so it is `pub(crate)` rather
//! than part of the crate's public API. It is kept because it is the shortest
//! complete example of the constructs the real renderer uses.
//!
//! It draws one static gradient triangle (three colored vertices) tinted by a
//! uniform, exercising exactly the explicit wgpu constructs that
//! [`SceneRenderer`](super::SceneRenderer) uses at scale — a vertex buffer with a
//! [`wgpu::VertexBufferLayout`], an explicit [`wgpu::BindGroupLayout`] +
//! [`wgpu::PipelineLayout`], a uniform bind group, and a render pass with
//! `set_pipeline`/`set_bind_group`/`set_vertex_buffer`/`draw`. `SceneRenderer` is
//! the same shape generalized to instanced, indexed, multi-pipeline drawing with
//! a depth buffer and per-frame uploads; start here to learn the pattern.

/// A 2D position + RGB color vertex consumed by `triangle.wgsl`.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Vertex2D {
    position: [f32; 2],
    color: [f32; 3],
}

impl Vertex2D {
    const ATTRIBUTES: [wgpu::VertexAttribute; 2] = [
        wgpu::VertexAttribute {
            format: wgpu::VertexFormat::Float32x2,
            offset: 0,
            shader_location: 0,
        },
        wgpu::VertexAttribute {
            format: wgpu::VertexFormat::Float32x3,
            offset: 8,
            shader_location: 1,
        },
    ];

    /// The vertex buffer layout expected by `triangle.wgsl`.
    fn layout() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Self>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &Self::ATTRIBUTES,
        }
    }
}

/// The three vertices of the gradient triangle in normalized device coordinates:
/// red apex on top, green bottom-left, blue bottom-right.
const VERTICES: [Vertex2D; 3] = [
    Vertex2D {
        position: [0.0, 0.5],
        color: [1.0, 0.0, 0.0],
    },
    Vertex2D {
        position: [-0.5, -0.5],
        color: [0.0, 1.0, 0.0],
    },
    Vertex2D {
        position: [0.5, -0.5],
        color: [0.0, 0.0, 1.0],
    },
];

/// The identity tint (opaque white): the fragment color passes through unchanged.
const IDENTITY_TINT: [f32; 4] = [1.0, 1.0, 1.0, 1.0];

/// The minimal wgpu renderer: a gradient triangle tinted by a uniform. Owns the
/// pipeline, the vertex buffer, and the tint bind group — the smallest complete
/// set of GPU objects a draw needs. See the module docs for how it relates to
/// [`SceneRenderer`](super::SceneRenderer).
pub struct TriangleRenderer {
    pipeline: wgpu::RenderPipeline,
    vertex_buffer: wgpu::Buffer,
    tint_bind_group: wgpu::BindGroup,
}

impl TriangleRenderer {
    /// Constructs a `TriangleRenderer` targeting `format`: uploads the three
    /// gradient vertices, builds the tint uniform and its bind group over an
    /// explicit bind-group layout, and compiles the pipeline over an explicit
    /// pipeline layout.
    pub fn new(device: &wgpu::Device, format: wgpu::TextureFormat) -> Self {
        use wgpu::util::DeviceExt;

        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("trd triangle vertex buffer"),
            contents: bytemuck::cast_slice(&VERTICES),
            usage: wgpu::BufferUsages::VERTEX,
        });

        // A group-0 uniform holding the tint color, bound at draw time — the
        // minimal example of the BindGroupLayout -> BindGroup -> set_bind_group
        // path that SceneRenderer uses for its camera and textures.
        let tint_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("trd triangle tint uniform"),
            contents: bytemuck::cast_slice(&IDENTITY_TINT),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let tint_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("trd triangle bind group layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        let tint_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("trd triangle bind group"),
            layout: &tint_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: tint_buffer.as_entire_binding(),
            }],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("trd triangle pipeline layout"),
            bind_group_layouts: &[Some(&tint_layout)],
            immediate_size: 0,
        });
        let shader = device.create_shader_module(wgpu::include_wgsl!("../shader/triangle.wgsl"));
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("trd triangle pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[Some(Vertex2D::layout())],
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
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        Self {
            pipeline,
            vertex_buffer,
            tint_bind_group,
        }
    }

    /// Records a pass into `encoder` that clears `view` to black and draws the
    /// gradient triangle. Unlike [`SceneRenderer::encode`](super::SceneRenderer::encode)
    /// this renderer has no per-frame uploads, so it needs neither a queue nor a
    /// scene.
    pub fn encode(&self, encoder: &mut wgpu::CommandEncoder, view: &wgpu::TextureView) {
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
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.tint_bind_group, &[]);
        pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
        pass.draw(0..3, 0..1);
    }
}
