use super::{overlay_depth_stencil, Tonemap};
use crate::Camera;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct EnvBackgroundUniform {
    inverse_view_proj: [f32; 16],
    camera_pos: [f32; 4],
    params: [f32; 4],
}

pub(super) struct EnvBackground {
    pipeline: wgpu::RenderPipeline,
    uniform: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
}

pub(super) struct EnvBackgroundSettings {
    pub rotation: f32,
    pub exposure: f32,
    pub blur: f32,
    pub tonemap: Tonemap,
}

impl EnvBackground {
    pub(super) fn new(
        device: &wgpu::Device,
        format: wgpu::TextureFormat,
        env_layout: &wgpu::BindGroupLayout,
        sample_count: u32,
    ) -> Self {
        let uniform_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("trd environment background uniform layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: wgpu::BufferSize::new(
                        std::mem::size_of::<EnvBackgroundUniform>() as u64,
                    ),
                },
                count: None,
            }],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("trd environment background pipeline layout"),
            bind_group_layouts: &[Some(&uniform_layout), Some(env_layout)],
            immediate_size: 0,
        });
        let shader =
            device.create_shader_module(wgpu::include_wgsl!("../shader/env_background.wgsl"));
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("trd environment background pipeline"),
            layout: Some(&pipeline_layout),
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
            depth_stencil: Some(overlay_depth_stencil()),
            multisample: wgpu::MultisampleState {
                count: sample_count,
                ..Default::default()
            },
            multiview_mask: None,
            cache: None,
        });
        let uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("trd environment background uniform"),
            size: std::mem::size_of::<EnvBackgroundUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("trd environment background bind group"),
            layout: &uniform_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform.as_entire_binding(),
            }],
        });
        Self {
            pipeline,
            uniform,
            bind_group,
        }
    }

    pub(super) fn write(
        &self,
        queue: &wgpu::Queue,
        camera: Camera,
        settings: EnvBackgroundSettings,
    ) {
        let inverse_view_proj = camera.view_projection().matrix().inverse().to_cols_array();
        let position = camera.position();
        let uniform = EnvBackgroundUniform {
            inverse_view_proj,
            camera_pos: [position[0], position[1], position[2], 1.0],
            params: [
                settings.rotation,
                settings.exposure,
                settings.blur.clamp(0.0, 1.0),
                settings.tonemap.to_uniform(),
            ],
        };
        queue.write_buffer(&self.uniform, 0, bytemuck::bytes_of(&uniform));
    }

    pub(super) fn draw<'pass>(
        &'pass self,
        pass: &mut wgpu::RenderPass<'pass>,
        env: &'pass wgpu::BindGroup,
    ) {
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.bind_group, &[]);
        pass.set_bind_group(1, env, &[]);
        pass.draw(0..3, 0..1);
    }
}
