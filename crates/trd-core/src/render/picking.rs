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
//! [`PICK_FORMAT`] color texture + a depth attachment, into which each drawn
//! object is rasterized in a flat id color. After the pass, the **one** texel
//! under the cursor is copied back and decoded to a 0-based object index (or
//! `None` for the background) — so a click resolves *which* object it hit
//! without ray-marching. Kept separate from the display
//! [`TextureTarget`](super::TextureTarget) because picking must be
//! single-sampled (ids must never be averaged at edges) and use a non-sRGB format
//! (so the id bytes round-trip exactly).
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
    create_depth_target, create_mesh_bind_group_layout, create_picking_pipeline, DepthTarget,
    PickInstanceRaw, PICK_FORMAT,
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
    /// Returns nothing: the caller reads it back through [`target`](Self::target)
    /// *after* the mutable work (writing the camera uniform, uploading the id
    /// instances) is done, which is how [`Renderer::pick`](super::Renderer::pick) encodes into a target
    /// it owns without moving it out of `self` (#235 R4).
    pub(super) fn ensure_target(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        match self.target.as_mut() {
            Some(target) => target.resize(device, width, height),
            None => self.target = Some(PickTarget::new(device, width, height)),
        }
    }

    /// The allocated pick target, or `None` before the first
    /// [`ensure_target`](Self::ensure_target).
    pub(super) fn target(&self) -> Option<&PickTarget> {
        self.target.as_ref()
    }

    /// The current pick-target size, or `None` while nothing has been picked yet.
    pub(super) fn target_size(&self) -> Option<(u32, u32)> {
        self.target.as_ref().map(PickTarget::size)
    }
}

/// A single-sample id-color render target + depth + a tiny read-back buffer for
/// one pixel. Sized to the display; rebuilt when the render size changes.
pub(super) struct PickTarget {
    texture: wgpu::Texture,
    color_view: wgpu::TextureView,
    depth: DepthTarget,
    /// A `MAP_READ` staging buffer for a single texel's row (padded to the copy
    /// alignment). One row is enough — picking reads exactly one pixel.
    staging: wgpu::Buffer,
    width: u32,
    height: u32,
}

impl PickTarget {
    /// Allocates the id-color target + depth + 1-texel read-back buffer for a
    /// fixed `width` × `height`.
    fn new(device: &wgpu::Device, width: u32, height: u32) -> Self {
        let width = width.max(1);
        let height = height.max(1);
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("trd pick target"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: PICK_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let color_view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let depth = create_depth_target(device, width, height, 1);
        // A single 4-byte RGBA texel, padded to the copy row alignment.
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("trd pick readback"),
            size: u64::from(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        Self {
            texture,
            color_view,
            depth,
            staging,
            width,
            height,
        }
    }

    /// The current pick-target dimensions.
    fn size(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    /// Resizes the target to `width` × `height` (no-op when unchanged), so it
    /// tracks the display render size.
    fn resize(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        let width = width.max(1);
        let height = height.max(1);
        if width == self.width && height == self.height {
            return;
        }
        *self = Self::new(device, width, height);
    }

    /// The id-color attachment this pass renders into.
    pub(super) fn color_view(&self) -> &wgpu::TextureView {
        &self.color_view
    }

    /// The single-sample depth attachment, so the nearest object's id wins.
    pub(super) fn depth_view(&self) -> &wgpu::TextureView {
        &self.depth.view
    }

    /// The id-color texture, copied one texel at a time into
    /// [`staging`](Self::staging).
    pub(super) fn texture(&self) -> &wgpu::Texture {
        &self.texture
    }

    /// The `MAP_READ` buffer the picked texel is copied into.
    pub(super) fn staging(&self) -> &wgpu::Buffer {
        &self.staging
    }

    /// Whether `(x, y)` is inside the target (a click outside it hits nothing).
    pub(super) fn contains(&self, x: u32, y: u32) -> bool {
        x < self.width && y < self.height
    }
}
