//! Small GPU buffer primitives: a buffer plus its element count.
//!
//! [`IndexBuffer`] and [`VertexBuffer`] are twins — same shape, same
//! `create_buffer_init` construction — so they live together rather than being
//! split along "index versus vertex". The two `draw_*` helpers issue the
//! instanced draw for each. [`InstanceBuffer`] is the growable third: a buffer
//! plus the capacity it must stay in step with (#222).

use std::marker::PhantomData;
use std::ops::Range;

use super::GpuContext;

/// An index buffer plus its element count — one `draw_indexed` range.
pub(super) struct IndexBuffer {
    buffer: wgpu::Buffer,
    count: u32,
}

impl IndexBuffer {
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

    /// Frees the buffer, whoever else still holds a handle to it.
    pub(super) fn destroy(&self) {
        self.buffer.destroy();
    }
}

/// A vertex buffer plus its element count, typed by the vertex record it holds
/// (#247 R3).
///
/// `T` is the per-vertex record — [`Vertex`](super::Vertex),
/// [`ShadingVertex`](super::ShadingVertex), [`GizmoLineVertex`](super::GizmoLineVertex)
/// — so a buffer names the layout its pipeline expects instead of being an
/// anonymous `wgpu::Buffer` that any pipeline would accept. The mesh store used
/// to hand out three of those bare handles from `filled()` / `pbr()` /
/// `wireframe()`, and swapping two of them compiled cleanly; now it cannot. Same
/// argument as [`InstanceBuffer`]'s: the element size (and here the layout) is
/// the type's, not a number the caller has to remember.
///
/// The `count` is what a **non-indexed** draw ranges over. An **indexed** draw
/// ignores it and ranges over the index buffer instead — wgpu's rule, not ours —
/// which is why one type serves both: the alternative was two near-identical
/// types differing only in whether they carried a `u32` some callers read.
pub(super) struct VertexBuffer<T> {
    buffer: wgpu::Buffer,
    count: u32,
    /// `T` is only ever written *through* the buffer, never stored inline.
    vertex: PhantomData<fn(T)>,
}

impl<T: bytemuck::Pod> VertexBuffer<T> {
    pub(super) fn new(device: &wgpu::Device, label: &str, vertices: &[T]) -> Self {
        use wgpu::util::DeviceExt;
        let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(label),
            contents: bytemuck::cast_slice(vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let count = u32::try_from(vertices.len()).expect("vertex count exceeds u32::MAX");
        Self {
            buffer,
            count,
            vertex: PhantomData,
        }
    }
}

impl<T> VertexBuffer<T> {
    /// The whole buffer, for binding at a pass's vertex slot 0. No `Pod` bound:
    /// binding and drawing never touch `T`, only uploading does.
    pub(super) fn slice(&self) -> wgpu::BufferSlice<'_> {
        self.buffer.slice(..)
    }

    /// Frees the buffer, whoever else still holds a handle to it.
    pub(super) fn destroy(&self) {
        self.buffer.destroy();
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

/// Binds `vertices` at slot 0 and `index`, then draws them over `instances`.
/// Pipeline, group bindings and the per-instance model buffer at slot 1 are the
/// caller's responsibility — each `record` body binds its own (#204).
///
/// The draw ranges over the **index** count; the vertex buffer's own count is
/// not read here (it is what [`draw_vertices`] uses).
pub(super) fn draw_indexed<T>(
    pass: &mut wgpu::RenderPass,
    (vertices, index): (&VertexBuffer<T>, &IndexBuffer),
    instances: Range<u32>,
) {
    pass.set_vertex_buffer(0, vertices.slice());
    pass.set_index_buffer(index.buffer.slice(..), wgpu::IndexFormat::Uint32);
    pass.draw_indexed(0..index.count, 0, instances);
}

/// The non-indexed twin: binds `vertices` at slot 0 and draws its whole span.
pub(super) fn draw_vertices<T>(
    pass: &mut wgpu::RenderPass,
    vertices: &VertexBuffer<T>,
    instances: Range<u32>,
) {
    pass.set_vertex_buffer(0, vertices.slice());
    pass.draw(0..vertices.count, instances);
}
