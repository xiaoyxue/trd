//! Small GPU buffer primitives: a buffer plus its element count.
//!
//! [`IndexBuf`] and [`VertexGeometry`] are twins — same shape, same
//! `create_buffer_init` construction — so they live together rather than being
//! split along "index versus vertex". The two `draw_*` helpers issue the
//! instanced draw for each. [`InstanceBuffer`] is the growable third: a buffer
//! plus the capacity it must stay in step with (#222).

use std::marker::PhantomData;
use std::ops::Range;

use super::GpuContext;

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

/// A growable per-instance buffer: the buffer **and** the capacity it must stay
/// in step with, plus the one growth rule they imply (#222).
///
/// The rule — grow to the next power of two, never shrink, and skip the write
/// for an empty frame — used to be written twice, once for the mesh instances
/// and once for the picking pass, each pairing a `wgpu::Buffer` with a loose
/// `u32` that nothing kept correct. `T` is the per-instance record
/// ([`InstanceRaw`](super::InstanceRaw) for the scene pass,
/// [`PickInstanceRaw`](super::PickInstanceRaw) for picking), so the element
/// size is the type's rather than a number the caller has to remember.
pub(super) struct InstanceBuffer<T> {
    buffer: wgpu::Buffer,
    capacity: u32,
    label: &'static str,
    /// `T` is only ever written *through* the buffer, never stored inline.
    instance: PhantomData<fn(T)>,
}

impl<T: bytemuck::Pod> InstanceBuffer<T> {
    /// Allocates a buffer for at least one instance (`capacity` clamped to ≥ 1).
    pub(super) fn new(device: &wgpu::Device, label: &'static str, capacity: u32) -> Self {
        let capacity = capacity.max(1);
        Self {
            buffer: Self::allocate(device, label, capacity),
            capacity,
            label,
            instance: PhantomData,
        }
    }

    fn allocate(device: &wgpu::Device, label: &'static str, capacity: u32) -> wgpu::Buffer {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: capacity as u64 * std::mem::size_of::<T>() as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }

    /// Uploads `instances`, growing the buffer (to the next power of two) when
    /// the frame needs more than it holds. An empty frame writes nothing.
    pub(super) fn upload(&mut self, gpu: &GpuContext, instances: &[T]) {
        if instances.len() as u32 > self.capacity {
            self.capacity = (instances.len() as u32).next_power_of_two();
            self.buffer = Self::allocate(&gpu.device, self.label, self.capacity);
        }
        if !instances.is_empty() {
            gpu.queue
                .write_buffer(&self.buffer, 0, bytemuck::cast_slice(instances));
        }
    }

    /// The whole buffer, for binding at a pass's per-instance vertex slot.
    pub(super) fn slice(&self) -> wgpu::BufferSlice<'_> {
        self.buffer.slice(..)
    }
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
