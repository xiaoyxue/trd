//! [`Picking`] + [`PickTarget`] — the object-id ("color index") picking
//! harness (#141).
//!
//! [`Picking`] is the pass's own state — its pipeline (`picking.wgsl`), its
//! per-instance buffer, and its lazily allocated [`PickTarget`]. The three are
//! used only by [`Renderer::pick`](super::Renderer::pick) — which lives here too
//! (#363) — and are meaningless apart, so they travel together rather than as
//! three loose renderer fields (#221 §4).
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
//! Like every other target, it is **pure data** (#203, #235 R4): a texture +
//! staging buffer knows nothing about pipelines, uniforms or the mesh store, so
//! the pass that fills it and the read-back that decodes it are
//! [`Renderer`](super::Renderer) methods — in the `impl Renderer` block at the
//! bottom of this file, beside the state they drive (#363). That also retires
//! the `take_target`/`restore_target` dance the old
//! `PickTarget::pick(&self, renderer: &mut Renderer, …)` needed to dodge the
//! borrow checker — the renderer now splits the borrow in time instead.

use super::buffer::draw_indexed;
use super::{RenderError, Renderer, ResolvedDraw, Viewport};
use crate::{Camera, MeshId, MeshResourceError};

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

/// The object-id picking **pass**, beside the pipeline, instance buffer and
/// target it drives (#363). The state stays in [`Picking`]; the sequence — ensure
/// the target, stage, encode, read back — needs the renderer's uniforms and mesh
/// store, so it is [`Renderer`] behaviour living in the file that owns what it
/// touches.
impl Renderer {
    /// The size of the object-id pick target, or `None` if nothing has been
    /// picked yet (it is allocated on the first [`pick`](Self::pick)). Diagnostic
    /// only — front-ends surface it in their debug panels.
    pub fn pick_target_size(&self) -> Option<(u32, u32)> {
        self.picking.target_size()
    }

    /// Stages the **object-id picking pass** (#141) for `draws`: writes the frame
    /// camera and uploads one id instance per pickable draw, returning the
    /// `(mesh_id, instance slot)` records [`encode_picking`](Self::encode_picking)
    /// then draws. `Shadow` draws are skipped, but the
    /// index mapping is preserved (a shadow's index simply never appears),
    /// so a decoded id maps straight back to `draws[index]`.
    ///
    /// Split from the encode half so the pass's two borrows never overlap: this
    /// is the `&mut self` work (uniform write + instance upload), and encoding
    /// then needs only `&self` — which is what lets [`pick`](Self::pick) render
    /// into a target the renderer still *owns*, instead of moving it out of
    /// `self` and handing it back (#235 R4).
    ///
    /// Private: a front-end reaches it through [`pick`](Self::pick), which owns
    /// the whole sequence — ensure target, prepare, encode, read back (#235 R4).
    ///
    /// **It keeps its own traversal on purpose** (#204). It does *not* batch a
    /// [`Scene`] and does not go through the per-primitive `record` bodies: this
    /// is a different pass with different attachments (single-sampled, flat id
    /// colors, no MSAA resolve) drawing only mesh geometry through the
    /// [`Picking`](super::picking::Picking) pipeline instead of the visual ones,
    /// and it needs an
    /// instance per *object* — never grouped — because the whole point is that
    /// each one carries a distinct id. Sharing the walk would mean threading a
    /// pass-kind through every `record` body to couple two loops that agree on
    /// almost nothing, for little gain.
    fn prepare_picking(
        &mut self,
        camera: Camera,
        draws: &[ResolvedDraw],
    ) -> Result<Vec<(MeshId, u32)>, MeshResourceError> {
        // Camera P·V for this frame (writes the shared camera uniform bound by
        // `uniforms.camera`, which is layout-compatible with the pick pipeline).
        self.uniforms.write_camera(&self.gpu.queue, camera);

        // Build one pick instance per drawable object, carrying its index color.
        let mut instances: Vec<PickInstanceRaw> = Vec::with_capacity(draws.len());
        let mut records: Vec<(MeshId, u32)> = Vec::with_capacity(draws.len());
        // A shadow blob has no mesh geometry to hit-test, so it is not pickable.
        for (index, draw) in draws.iter().enumerate() {
            if !draw.selection.is_mesh() {
                continue;
            }
            let mesh = self.meshes.get(draw.mesh_id)?;
            let effective = draw.model * mesh.geometry.base_model;
            let slot = instances.len() as u32;
            instances.push(PickInstanceRaw::new(effective, index as u32));
            records.push((draw.mesh_id, slot));
        }

        // Grow + upload the pick instance buffer.
        self.picking.upload_instances(&self.gpu, &instances);
        Ok(records)
    }

    /// Encodes the **object-id picking pass** for the records staged by
    /// [`prepare_picking`](Self::prepare_picking): each pickable draw's mesh is
    /// rasterized in a flat color encoding its **index**, single-sampled and
    /// depth-tested into `target`'s id-color attachment (cleared to id `0` =
    /// background) and its depth attachment. No lighting, no texture, no MSAA —
    /// so the pixel under the cursor reads back to an exact id via
    /// [`PickInstanceRaw::decode`].
    ///
    /// Takes `&self`: with the staging already done, the pass borrows the
    /// pipeline, the camera bind group and the mesh geometry immutably — the same
    /// way it borrows the `target`, which is why that target can live in
    /// `self.picking` for the whole call (#235 R4).
    ///
    /// **It keeps its own traversal on purpose** (#204). It does *not* batch a
    /// [`Scene`] and does not go through the per-primitive `record` bodies: this
    /// is a different pass with different attachments (single-sampled, flat id
    /// colors, no MSAA resolve) drawing only mesh geometry through the
    /// [`Picking`](super::picking::Picking) pipeline instead of the visual ones,
    /// and it needs an
    /// instance per *object* — never grouped — because the whole point is that
    /// each one carries a distinct id. Sharing the walk would mean threading a
    /// pass-kind through every `record` body to couple two loops that agree on
    /// almost nothing, for little gain.
    fn encode_picking(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        target: &PickPass<'_>,
        records: &[(MeshId, u32)],
    ) -> Result<(), MeshResourceError> {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("trd picking pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: target.id.view(),
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    // Clear to id 0 (background).
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: target.depth.view(),
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(1.0),
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });

        self.picking
            .bind(&mut pass, self.uniforms.camera.bind_group());
        for &(mesh_id, slot) in records {
            let mesh = self.meshes.get(mesh_id)?;
            draw_indexed(&mut pass, mesh.filled(), slot..slot + 1);
        }
        Ok(())
    }

    /// **Object-id picking** (#141): renders `draws` through the flat id-color
    /// pass at `viewport`'s size and returns the **0-based index into `draws`**
    /// of the object under pixel `(x, y)`, or `None` for the background (or an
    /// out-of-bounds coordinate). The pass is single-sampled and depth-tested, so
    /// the nearest object wins and ids are never blended — the "color index"
    /// method, no ray-marching. The lazily-created pick target tracks `viewport`,
    /// which the caller passes on every call (#203): the harness no longer owns a
    /// render target of its own to read a size from, so a shell reports its
    /// current display size the same way it would to `render_params`.
    pub async fn pick(
        &mut self,
        camera: Camera,
        draws: &[ResolvedDraw],
        x: u32,
        y: u32,
        viewport: Viewport,
    ) -> Result<Option<u32>, RenderError> {
        for draw in draws.iter().filter(|draw| draw.selection.is_mesh()) {
            self.meshes.get(draw.mesh_id)?;
        }
        let gpu = self.gpu.clone();
        let Viewport {
            width: w,
            height: h,
        } = viewport;
        // The target stays owned by `self.picking` for the whole call (#235 R4).
        // The borrows are separated in *time* instead of by moving it out: the
        // `&mut self` staging first, then an all-immutable encode + read-back.
        self.picking.ensure_target(&gpu.device, w, h);
        let records = self.prepare_picking(camera, draws)?;

        let target = self
            .picking
            .pass()
            .ok_or_else(|| RenderError::Gpu("pick target was not allocated".into()))?;
        if !target.id.contains(x, y) {
            return Ok(None);
        }

        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("trd pick frame"),
            });
        self.encode_picking(&mut encoder, &target, &records)?;
        // Copy just the one texel under the cursor into the staging buffer.
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: target.id.texture(),
                mip_level: 0,
                origin: wgpu::Origin3d { x, y, z: 0 },
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: target.staging,
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
        gpu.queue.submit(Some(encoder.finish()));

        let slice = target.staging.slice(..4);
        let (sender, receiver) = futures_channel::oneshot::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });
        // Same wait as the offscreen readback; see `platform::poll_for_map`.
        super::platform::poll_for_map(&gpu.device)
            .map_err(|error| RenderError::Gpu(error.to_string()))?;
        receiver
            .await
            .map_err(|error| RenderError::Gpu(error.to_string()))?
            .map_err(|error| RenderError::Gpu(error.to_string()))?;

        let id = {
            let mapped = slice
                .get_mapped_range()
                .map_err(|error| RenderError::Gpu(error.to_string()))?;
            let rgba = [mapped[0], mapped[1], mapped[2], mapped[3]];
            PickInstanceRaw::decode(rgba)
        };
        target.staging.unmap();
        Ok(id)
    }
}
