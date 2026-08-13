//! Uniform buffers paired with the bind group that binds them (#203).
//!
//! The renderer used to spell every uniform out as two loose fields —
//! `camera_uniform`/`camera_bind_group`, `gizmo_uniform`/`gizmo_bind_group`,
//! `pbr_uniform`/`pbr_bind_group` — three copies of one idea, kept in step only
//! by naming discipline. A buffer and the bind group that exposes it are created
//! together, written together and invalidated together, so they are one value.
//! This is the same "bound resource" family as
//! [`BoundTexture`](super::BoundTexture) and
//! [`BoundMaterialMaps`](super::bound_material_maps::BoundMaterialMaps); these
//! are its uniform-buffer members.

/// A **statically** bound uniform: one buffer, bound whole, by a bind group that
/// never takes a dynamic offset (the camera `P·V`, the gizmo viewport params).
///
/// Rewritten in place each frame, so the bind group outlives every value it has
/// ever carried and can stay bound across draws.
pub(crate) struct BoundUniform {
    buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
}

impl BoundUniform {
    pub(crate) fn new(buffer: wgpu::Buffer, bind_group: wgpu::BindGroup) -> Self {
        Self { buffer, bind_group }
    }

    /// The buffer to rewrite for this frame.
    pub(crate) fn buffer(&self) -> &wgpu::Buffer {
        &self.buffer
    }

    /// The bind group to bind with **no** dynamic offsets (`&[]`).
    pub(crate) fn bind_group(&self) -> &wgpu::BindGroup {
        &self.bind_group
    }
}

/// A **dynamic-offset slot array**: one buffer holding `slots` stride-spaced
/// copies of a uniform, plus a single-slot *window* bind group whose offset a
/// draw chooses via [`offset`](Self::offset) (the PBR per-object material, #141).
///
/// A separate type from [`BoundUniform`] rather than the same one with an
/// optional stride, because the stride is not decoration — it is the CPU-side
/// half of the layout's `has_dynamic_offset: true`
/// ([`create_pbr_bind_group_layout`](super::create_pbr_bind_group_layout)).
/// Three things only make sense together: the layout declares a dynamic offset,
/// the buffer is `stride * slots` bytes while the bind group covers just one
/// slot, and every `set_bind_group` **must** pass an offset — binding such a
/// group with `&[]` (or a static one with an offset) is a wgpu validation error,
/// not a subtle mis-render. Making them distinct types means the two binding
/// disciplines can't be confused at a call site, and [`offset`](Self::offset)
/// keeps the slot arithmetic in one place instead of open-coded at each draw.
pub(crate) struct BoundUniformArray {
    buffer: wgpu::Buffer,
    /// A single-slot window over `buffer`; the dynamic offset picks which slot.
    bind_group: wgpu::BindGroup,
    /// The device-aligned byte distance between adjacent slots — `size_of::<T>()`
    /// rounded up to `min_uniform_buffer_offset_alignment`, because a dynamic
    /// offset must satisfy that alignment.
    stride: u64,
}

impl BoundUniformArray {
    /// Allocates `slots` (at least one) uniform slots of `T` over a
    /// `has_dynamic_offset` `layout`. `label` names the pair: `"<label> uniform"`
    /// for the buffer, `"<label> bind group"` for the window.
    ///
    /// The contents are left undefined — every consumer rewrites the slots it
    /// draws before drawing them.
    pub(crate) fn new<T: bytemuck::Pod>(
        device: &wgpu::Device,
        layout: &wgpu::BindGroupLayout,
        label: &str,
        slots: usize,
    ) -> Self {
        let slot_size = std::mem::size_of::<T>() as u64;
        let align = device.limits().min_uniform_buffer_offset_alignment as u64;
        let stride = slot_size.next_multiple_of(align);
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(&format!("{label} uniform")),
            size: stride * slots.max(1) as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(&format!("{label} bind group")),
            layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &buffer,
                    offset: 0,
                    size: wgpu::BufferSize::new(slot_size),
                }),
            }],
        });
        Self {
            buffer,
            bind_group,
            stride,
        }
    }

    /// The dynamic offset selecting `slot`, for
    /// `set_bind_group(group, bind_group(), &[offset(slot)])`. The **only** place
    /// the slot arithmetic lives.
    pub(crate) fn offset(&self, slot: usize) -> u32 {
        self.byte_offset(slot) as u32
    }

    /// The single-slot window bind group. Must be bound **with** an
    /// [`offset`](Self::offset).
    pub(crate) fn bind_group(&self) -> &wgpu::BindGroup {
        &self.bind_group
    }

    /// Rewrites one slot's uniform for this frame.
    pub(crate) fn write_slot<T: bytemuck::Pod>(&self, queue: &wgpu::Queue, slot: usize, value: &T) {
        queue.write_buffer(
            &self.buffer,
            self.byte_offset(slot),
            bytemuck::bytes_of(value),
        );
    }

    fn byte_offset(&self, slot: usize) -> u64 {
        slot as u64 * self.stride
    }
}
