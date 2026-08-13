//! [`Picking`] + [`PickTarget`] — the object-id ("color index") picking
//! harness (#141).
//!
//! [`Picking`] is the pass's own state — its pipeline (`picking.wgsl`), its
//! per-instance buffer, and its lazily allocated [`PickTarget`]. The three are
//! used only by [`Renderer::pick`] / [`Renderer::encode_picking`] and are
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

use super::GpuContext;
use futures_channel::oneshot;

use super::{
    create_depth_target, create_mesh_bind_group_layout, create_picking_pipeline, DepthTarget,
    PickInstanceRaw, Renderer, PICK_FORMAT,
};
use crate::visual::Draw;
use crate::Camera;

/// The object-id **picking** pass's own state (#141, grouped in #203).
///
/// Its pipeline (`picking.wgsl`) renders each drawn object in a flat id color
/// into a single-sample linear target; its per-instance buffer carries the
/// model + id color; its target is the surface that is read back.
pub(super) struct Picking {
    /// Built once. Its group-0 bind-group layout is structurally the camera
    /// layout, so `SceneUniforms::camera`'s bind group binds it.
    pipeline: wgpu::RenderPipeline,
    /// Per-instance [`PickInstanceRaw`] buffer (model + id color), grown on
    /// demand like the mesh instance buffer.
    instances: wgpu::Buffer,
    instance_capacity: u32,
    /// Created lazily on the first [`pick`](Renderer::pick) call and resized to
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
        let instance_capacity = (mesh_count as u32).max(1);
        Self {
            pipeline: create_picking_pipeline(device, &layout),
            instances: create_pick_instance_buffer(device, instance_capacity),
            instance_capacity,
            target: None,
        }
    }

    /// Uploads this pass's instances, growing the buffer (to the next power of
    /// two) when the frame needs more than it holds — the same policy as
    /// `MeshStore::upload_instances`.
    pub(super) fn upload_instances(&mut self, gpu: &GpuContext, instances: &[PickInstanceRaw]) {
        if instances.len() as u32 > self.instance_capacity {
            self.instance_capacity = (instances.len() as u32).next_power_of_two();
            self.instances = create_pick_instance_buffer(&gpu.device, self.instance_capacity);
        }
        if !instances.is_empty() {
            gpu.queue
                .write_buffer(&self.instances, 0, bytemuck::cast_slice(instances));
        }
    }

    /// Binds this pass's pipeline, the caller's group-0 camera bind group, and
    /// the id-instance buffer — the whole pass-local setup
    /// [`Renderer::encode_picking`] needs before its per-object draws, so the
    /// pipeline and instance buffer stay private to this module.
    pub(super) fn bind(&self, pass: &mut wgpu::RenderPass<'_>, camera: &wgpu::BindGroup) {
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, camera, &[]);
        pass.set_vertex_buffer(1, self.instances.slice(..));
    }

    /// Moves the pick target out for the duration of one pass, creating it on
    /// first use and resizing it to `width` × `height` — the lazy-allocation
    /// policy the `target` field documents. The caller must hand it back through
    /// [`restore_target`](Self::restore_target): [`Renderer::pick`] encodes
    /// through `&mut Renderer`, so a target still borrowed out of `self` would
    /// conflict with re-borrowing all of `self` (#203).
    pub(super) fn take_target(
        &mut self,
        device: &wgpu::Device,
        width: u32,
        height: u32,
    ) -> PickTarget {
        match self.target.take() {
            Some(mut target) => {
                target.resize(device, width, height);
                target
            }
            None => PickTarget::new(device, width, height),
        }
    }

    /// Returns the target taken by [`take_target`](Self::take_target).
    pub(super) fn restore_target(&mut self, target: PickTarget) {
        self.target = Some(target);
    }

    /// The current pick-target size, or `None` while nothing has been picked yet.
    pub(super) fn target_size(&self) -> Option<(u32, u32)> {
        self.target.as_ref().map(PickTarget::size)
    }
}

fn create_pick_instance_buffer(device: &wgpu::Device, capacity: u32) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("trd pick instance buffer"),
        size: capacity as u64 * std::mem::size_of::<PickInstanceRaw>() as u64,
        usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

/// A single-sample id-color render target + depth + a tiny read-back buffer for
/// one pixel. Sized to the display; rebuilt when the render size changes.
pub struct PickTarget {
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
    pub fn new(device: &wgpu::Device, width: u32, height: u32) -> Self {
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
    pub fn size(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    /// Resizes the target to `width` × `height` (no-op when unchanged), so it
    /// tracks the display render size.
    pub fn resize(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        let width = width.max(1);
        let height = height.max(1);
        if width == self.width && height == self.height {
            return;
        }
        *self = Self::new(device, width, height);
    }

    /// Renders the id pass for `draws` under `params` and reads back the object
    /// index at pixel `(x, y)` — `None` for the background or an out-of-bounds
    /// coordinate. `async` so the browser event loop is not blocked during the
    /// read-back; native callers drive it with `pollster::block_on`.
    #[allow(clippy::too_many_arguments)]
    pub async fn pick(
        &self,
        gpu: &GpuContext,
        renderer: &mut Renderer,
        camera: Camera,
        draws: &[Draw],
        x: u32,
        y: u32,
    ) -> Option<u32> {
        let (device, queue) = (&gpu.device, &gpu.queue);
        if x >= self.width || y >= self.height {
            return None;
        }

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("trd pick frame"),
        });
        renderer.encode_picking(
            &mut encoder,
            &self.color_view,
            &self.depth.view,
            camera,
            draws,
        );
        // Copy just the one texel under the cursor into the staging buffer.
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &self.texture,
                mip_level: 0,
                origin: wgpu::Origin3d { x, y, z: 0 },
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &self.staging,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT),
                    rows_per_image: Some(1),
                },
            },
            wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
        );
        queue.submit(Some(encoder.finish()));

        let slice = self.staging.slice(..4);
        let (sender, receiver) = oneshot::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });
        // Same wait as the offscreen readback; see `platform::poll_for_map`.
        super::platform::poll_for_map(device).ok()?;
        receiver.await.ok()?.ok()?;

        let id = {
            let mapped = slice.get_mapped_range().ok()?;
            let rgba = [mapped[0], mapped[1], mapped[2], mapped[3]];
            PickInstanceRaw::decode(rgba)
        };
        self.staging.unmap();
        id
    }
}
