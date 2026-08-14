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

/// The PBR pipeline's **frequency-split group 0**: one statically bound uniform
/// at binding 0 — what the whole frame shares, written once — plus a
/// dynamic-offset slot array at binding 1, one slot per mesh (#182/#141).
///
/// Both live in **one** bind group on purpose: `pbr.wgsl` already occupies all
/// four groups (camera+material, albedo, environment, material maps), and the
/// portable WebGPU baseline guarantees only `max_bind_groups = 4`, which the
/// browser path depends on. So the split had to happen *inside* group 0 rather
/// than by adding a fifth.
///
/// A separate type from [`BoundUniform`] rather than the same one with an
/// optional stride, because the stride is not decoration — it is the CPU-side
/// half of the layout's `has_dynamic_offset: true`
/// ([`create_pbr_bind_group_layout`](super::create_pbr_bind_group_layout)).
/// Three things only make sense together: the layout declares a dynamic offset,
/// the slot buffer is `stride * slots` bytes while the bind group covers just
/// one slot, and every `set_bind_group` **must** pass an offset — binding such a
/// group with `&[]` (or a static one with an offset) is a wgpu validation error,
/// not a subtle mis-render. Making them distinct types means the two binding
/// disciplines can't be confused at a call site, and [`offset`](Self::offset)
/// keeps the slot arithmetic in one place instead of open-coded at each draw.
pub(crate) struct BoundSceneSlots {
    /// Binding 0: the once-per-frame scene uniform.
    scene: wgpu::Buffer,
    /// Binding 1: `slots` stride-spaced per-mesh uniforms.
    slots: wgpu::Buffer,
    /// One bind group covering both — a single-slot *window* over `slots`, whose
    /// offset a draw chooses via [`offset`](Self::offset).
    bind_group: wgpu::BindGroup,
    /// The device-aligned byte distance between adjacent slots — `size_of::<S>()`
    /// rounded up to `min_uniform_buffer_offset_alignment`, because a dynamic
    /// offset must satisfy that alignment.
    stride: u64,
}

impl BoundSceneSlots {
    /// Allocates the scene uniform (`F`) and `slots` (at least one) per-mesh
    /// slots of `S` over a `layout` whose binding 1 is `has_dynamic_offset`.
    /// `label` names the group: `"<label> scene uniform"`, `"<label> uniform"`,
    /// `"<label> bind group"`.
    ///
    /// The contents are left undefined — every consumer rewrites the scene
    /// uniform each frame and the slots it draws before drawing them.
    pub(crate) fn new<F: bytemuck::Pod, S: bytemuck::Pod>(
        device: &wgpu::Device,
        layout: &wgpu::BindGroupLayout,
        label: &str,
        slots: usize,
    ) -> Self {
        let scene_size = std::mem::size_of::<F>() as u64;
        let slot_size = std::mem::size_of::<S>() as u64;
        let align = device.limits().min_uniform_buffer_offset_alignment as u64;
        let stride = slot_size.next_multiple_of(align);
        let scene = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(&format!("{label} scene uniform")),
            size: scene_size,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let slot_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(&format!("{label} uniform")),
            size: stride * slots.max(1) as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(&format!("{label} bind group")),
            layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &scene,
                        offset: 0,
                        size: wgpu::BufferSize::new(scene_size),
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &slot_buffer,
                        offset: 0,
                        size: wgpu::BufferSize::new(slot_size),
                    }),
                },
            ],
        });
        Self {
            scene,
            slots: slot_buffer,
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

    /// The bind group covering both bindings. Must be bound **with** an
    /// [`offset`](Self::offset).
    pub(crate) fn bind_group(&self) -> &wgpu::BindGroup {
        &self.bind_group
    }

    /// Rewrites the scene uniform — once per frame, whatever the object count.
    pub(crate) fn write_scene<F: bytemuck::Pod>(&self, queue: &wgpu::Queue, value: &F) {
        queue.write_buffer(&self.scene, 0, bytemuck::bytes_of(value));
    }

    /// Rewrites one mesh's slot for this frame.
    pub(crate) fn write_slot<S: bytemuck::Pod>(&self, queue: &wgpu::Queue, slot: usize, value: &S) {
        queue.write_buffer(
            &self.slots,
            self.byte_offset(slot),
            bytemuck::bytes_of(value),
        );
    }

    fn byte_offset(&self, slot: usize) -> u64 {
        slot as u64 * self.stride
    }
}
