//! Small GPU buffer primitives: a buffer plus its element count.
//!
//! [`IndexBuf`] and [`VertexGeometry`] are twins — same shape, same
//! `create_buffer_init` construction — so they live together rather than being
//! split along "index versus vertex". The two `draw_*` helpers issue the
//! instanced draw for each.

use std::ops::Range;

use super::gpu_types::InstanceRaw;

/// An index buffer plus its element count — one `draw_indexed` range.
pub(super) struct IndexBuf {
    buffer: wgpu::Buffer,
    count: u32,
}

impl IndexBuf {
    pub(super) fn new(device: &wgpu::Device, label: &str, indices: &[u32]) -> Self {
        use wgpu::util::DeviceExt;
        let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(label),
            contents: bytemuck::cast_slice(indices),
            usage: wgpu::BufferUsages::INDEX,
        });
        let count = u32::try_from(indices.len()).expect("index count exceeds u32::MAX");
        Self { buffer, count }
    }
}

/// A self-contained non-indexed draw.
pub(super) struct VertexGeometry {
    vertex_buffer: wgpu::Buffer,
    vertex_count: u32,
}

impl VertexGeometry {
    pub(super) fn new<T: bytemuck::Pod>(
        device: &wgpu::Device,
        label: &str,
        vertices: &[T],
    ) -> Self {
        use wgpu::util::DeviceExt;
        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(label),
            contents: bytemuck::cast_slice(vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let vertex_count = u32::try_from(vertices.len()).expect("vertex count exceeds u32::MAX");
        Self {
            vertex_buffer,
            vertex_count,
        }
    }
}

pub(super) fn create_instance_buffer(device: &wgpu::Device, capacity: u32) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("trd mesh instance buffer"),
        size: capacity as u64 * std::mem::size_of::<InstanceRaw>() as u64,
        usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

/// Binds `vertex_buffer` at slot 0 and `index`, then draws it over `instances`.
/// Pipeline, group bindings and the per-instance model buffer at slot 1 are the
/// caller's responsibility — each `record` body binds its own (#204).
pub(super) fn draw_indexed(
    pass: &mut wgpu::RenderPass,
    (vertex_buffer, index): (&wgpu::Buffer, &IndexBuf),
    instances: Range<u32>,
) {
    pass.set_vertex_buffer(0, vertex_buffer.slice(..));
    pass.set_index_buffer(index.buffer.slice(..), wgpu::IndexFormat::Uint32);
    pass.draw_indexed(0..index.count, 0, instances);
}

pub(super) fn draw_vertices(
    pass: &mut wgpu::RenderPass,
    geometry: &VertexGeometry,
    instances: Range<u32>,
) {
    pass.set_vertex_buffer(0, geometry.vertex_buffer.slice(..));
    pass.draw(0..geometry.vertex_count, instances);
}
