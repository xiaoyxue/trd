//! [`Picking`] + [`PickTarget`] — the object-id ("color index") picking
//! harness (#141).
//!
//! [`Picking`] is the pass's own state — its pipeline (`picking.wgsl`), its
//! per-instance buffer, and its lazily allocated [`PickTarget`]. The three are
//! used only by [`Renderer::pick`](super::Renderer::pick) and are
//! meaningless apart, so they travel together rather than as three loose
//! renderer fields, and they live here beside the target they create rather
//! than in `renderer.rs` (#221 §4).
//!
//! [`PickTarget`] is the target itself: a single-sample **linear**
//! [`PICK_FORMAT`](super::PICK_FORMAT) color attachment + a depth attachment,
//! into which each drawn object is rasterized in a flat id color. After the
//! pass, the **one** texel under the cursor is copied back and decoded to a
//! 0-based object index (or `None` for the background) — so a click resolves
//! *which* object it hit without ray-marching. Kept separate from the display
//! [`TextureTarget`](super::TextureTarget) because picking must be
//! single-sampled (ids must never be averaged at edges) and use a non-sRGB format
//! (so the id bytes round-trip exactly) — both pinned by
//! [`AttachmentSpec::id_color`].
//!
//! Like every other target, it is **pure data** (#203, #235 R4): the pass that
//! fills it and the read-back that decodes it are [`Renderer`](super::Renderer) methods, because
//! a texture + staging buffer knows nothing about pipelines, uniforms or the
//! mesh store. That also retires the `take_target`/`restore_target` dance the
//! old `PickTarget::pick(&self, renderer: &mut Renderer, …)` needed to dodge the
//! borrow checker — the renderer now splits the borrow in time instead.

use super::GpuContext;

use super::buffer::InstanceBuffer;
use super::{
    create_mesh_bind_group_layout, create_picking_pipeline, Attachment, AttachmentSpec,
    PickInstanceRaw, ViewportAttachment,
};

/// The object-id **picking** pass's own state (#141, grouped in #203).
///
/// Its pipeline (`picking.wgsl`) renders each drawn object in a flat id color
/// into a single-sample linear target; its per-instance buffer carries the
/// model + id color; its target is the surface that is read back.
pub(super) struct Picking {
    /// Built once. Its group-0 bind-group layout is structurally the camera
    /// layout, so `SceneUniforms::camera`'s bind group binds it.
    pipeline: wgpu::RenderPipeline,
    /// Per-instance [`PickInstanceRaw`] buffer (model + id color). The same
    /// [`InstanceBuffer`] the scene pass uses, so the grow rule is written once
    /// (#222) — only the record type differs.
    instances: InstanceBuffer<PickInstanceRaw>,
    /// Created lazily on the first [`pick`](super::Renderer::pick) call and resized to
    /// track whatever `viewport` the caller passes. `None` until a front-end
    /// actually picks, so the headless CLI never allocates it.
    target: Option<PickTarget>,
}

impl Picking {
    /// Builds the picking pipeline and an instance buffer sized for `mesh_count`
    /// objects. The target stays unallocated until something is picked.
    pub(super) fn new(device: &wgpu::Device, mesh_count: usize) -> Self {
        // A group-0 camera uniform (structurally identical to the mesh camera
        // layout, so the scene's camera bind group binds it) + the per-instance
        // id color, single-sampled into PICK_FORMAT.
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("trd picking pipeline layout"),
            bind_group_layouts: &[Some(&create_mesh_bind_group_layout(device))],
            immediate_size: 0,
        });
        Self {
            pipeline: create_picking_pipeline(device, &layout),
            instances: InstanceBuffer::new(device, "trd pick instance buffer", mesh_count as u32),
            target: None,
        }
    }

    /// Uploads this pass's instances through the shared growable buffer.
    pub(super) fn upload_instances(&mut self, gpu: &GpuContext, instances: &[PickInstanceRaw]) {
        self.instances.upload(gpu, instances);
    }

    /// Binds this pass's pipeline, the caller's group-0 camera bind group, and
    /// the id-instance buffer — the whole pass-local setup
    /// `Renderer::encode_picking` needs before its per-object draws, so the
    /// pipeline and instance buffer stay private to this module.
    pub(super) fn bind(&self, pass: &mut wgpu::RenderPass<'_>, camera: &wgpu::BindGroup) {
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, camera, &[]);
        pass.set_vertex_buffer(1, self.instances.slice());
    }

    /// Ensures the pick target exists and matches `width` × `height`, creating it
    /// on first use — the lazy-allocation policy the `target` field documents.
    ///
    /// Returns nothing: the caller reads it back through [`pass`](Self::pass)
    /// *after* the mutable work (writing the camera uniform, uploading the id
    /// instances) is done, which is how [`Renderer::pick`](super::Renderer::pick) encodes into a target
    /// it owns without moving it out of `self` (#235 R4).
    pub(super) fn ensure_target(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        self.target
            .get_or_insert_with(|| PickTarget::new(device))
            .resize(device, width, height);
    }

    /// This pass's sized attachments and read-back buffer, or `None` before the
    /// first [`ensure_target`](Self::ensure_target).
    pub(super) fn pass(&self) -> Option<PickPass<'_>> {
        self.target.as_ref()?.pass()
    }

    /// The current pick-target size, or `None` while nothing has been picked yet.
    pub(super) fn target_size(&self) -> Option<(u32, u32)> {
        self.target.as_ref()?.size()
    }
}

/// A single-sample id-color attachment + depth + a tiny read-back buffer for one
/// pixel. Both attachments track the display size through their shared
/// [`ViewportAttachment`] rule; the buffer is a fixed one-texel row, so it
/// outlives every resize.
pub(super) struct PickTarget {
    id: ViewportAttachment,
    depth: ViewportAttachment,
    /// A `MAP_READ` staging buffer for a single texel's row (padded to the copy
    /// alignment). One row is enough — picking reads exactly one pixel.
    staging: wgpu::Buffer,
}

impl PickTarget {
    /// Allocates the 1-texel read-back buffer; the attachments follow on the
    /// first [`resize`](Self::resize).
    fn new(device: &wgpu::Device) -> Self {
        Self {
            id: ViewportAttachment::new(AttachmentSpec::id_color()),
            depth: ViewportAttachment::new(AttachmentSpec::depth(1)),
            // A single 4-byte RGBA texel, padded to the copy row alignment.
            staging: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("trd pick readback"),
                size: u64::from(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT),
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            }),
        }
    }

    /// Sizes both attachments to `width` × `height`, so the target tracks the
    /// display render size.
    fn resize(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        self.id.ensure(device, width, height);
        self.depth.ensure(device, width, height);
    }

    /// The current dimensions, or `None` before the first
    /// [`resize`](Self::resize).
    fn size(&self) -> Option<(u32, u32)> {
        self.id.current().map(Attachment::size)
    }

    fn pass(&self) -> Option<PickPass<'_>> {
        Some(PickPass {
            id: self.id.current()?,
            depth: self.depth.current()?,
            staging: &self.staging,
        })
    }
}

/// One pick pass's attachments and its read-back buffer, borrowed together.
///
/// Flattening [`PickTarget`]'s options here is what keeps the encode and the
/// read-back free of unwraps: a caller answers "is there a sized target?" once,
/// with the `?` it already had.
pub(super) struct PickPass<'a> {
    /// The id-color attachment this pass renders into, and the texture the
    /// picked texel is copied out of.
    pub(super) id: &'a Attachment,
    /// The single-sample depth attachment, so the nearest object's id wins.
    pub(super) depth: &'a Attachment,
    /// The `MAP_READ` buffer the picked texel is copied into.
    pub(super) staging: &'a wgpu::Buffer,
}
