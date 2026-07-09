//! Shared, platform-agnostic triangle rendering.
//!
//! [`render_triangle`] records and submits a render pass that draws the
//! hello-triangle into the given texture view. Both the native CLI (offscreen
//! texture) and the wasm entry point (canvas surface) call this same function.

/// Draws the hello-triangle into `view`, clearing to black first.
///
/// The pipeline is built on each call; for the hello-triangle this happens once
/// per rendered frame, which is fine. `format` must match the target `view`.
pub fn render_triangle(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    view: &wgpu::TextureView,
    format: wgpu::TextureFormat,
) {
    let shader = device.create_shader_module(wgpu::include_wgsl!("triangle.wgsl"));

    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
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
    });

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
        pass.draw(0..3, 0..1);
    }
    queue.submit(Some(encoder.finish()));
}
