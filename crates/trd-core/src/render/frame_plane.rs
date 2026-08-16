//! The background **frame plane** (#63): a fullscreen textured quad drawn
//! beneath the mesh scene and skinned by a per-frame video frame.
//!
//! This is the second, separately-updated texture subsystem (as opposed to the
//! mesh [`BoundTexture`](super::BoundTexture) albedo): the frame image is
//! uploaded at the boundary from `frame_path`/`frame_url`, reused across frames,
//! and sampled when a scene's [`Background::frame`](crate::Background::frame) is
//! set (#204). All
//! of the old `update_frame_texture_rgba` machinery lives here so `Renderer`
//! only has to delegate to it.

use super::GpuContext;
use super::{create_frame_bind_group_layout, create_frame_plane_pipeline, Viewport};
use super::{frame_fit_uv_scale, FrameFit};

/// VRAM the frame ring may spend on decoded frames. Capacity is derived from
/// this and the frame size rather than fixed, because a slot count that is
/// comfortable at 960×540 (2 MB each) is a gigabyte at 4K (33 MB each).
///
/// Sized so 1080p gets [`RING_MAX_SLOTS`]; larger frames fall back to
/// [`RING_MIN_SLOTS`], which is the floor the budget is not allowed to breach.
const RING_BUDGET_BYTES: usize = 64 * 1024 * 1024;

/// Never fewer than this many slots.
///
/// The ring's one hard correctness requirement is that the write cursor must not
/// wrap onto a slot the GPU is still reading. That is a *capacity* problem, not
/// a synchronization one: a barrier would serialize the write against the draw
/// and defeat the pipelining. With at most a couple of frames in flight on the
/// queue, four slots is already slack.
const RING_MIN_SLOTS: u32 = 4;

/// Never more than this, however small the frames.
///
/// **Eight, not sixty-four.** The ring was first budgeted on the assumption that
/// a deep window buys scrub-back hits. It does not: the resident window is a
/// span of *frames*, and a scrub is a span of *seconds*. Even 64 slots hold
/// 2.1 s at 30 fps, and only 0.07 s of 4K60 — so a seek lands outside the window
/// essentially always, whatever the capacity.
///
/// Simulated against four editing workloads, hit rate by capacity:
///
/// | workload | 1 | 8 | 64 |
/// |---|---:|---:|---:|
/// | linear playback | 3.2% | 3.2% | 3.2% |
/// | scrubbing to hunt a shot | 75.6% | 75.8% | 75.9% |
/// | stepping frame by frame | 95.2% | 99.2% | 99.6% |
/// | dragging a gizmo | 100% | 100% | 100% |
///
/// The ring's payoff is the **repeated render of the frame already on screen** —
/// a gizmo drag, an overlay toggle, a selection — which one slot serves. Only
/// frame-stepping wants more, and it saturates by eight. Past that the VRAM is
/// dead: 64 slots at 4K is 2 GiB for the 0.4 points between 8 and 64.
const RING_MAX_SLOTS: u32 = 8;

/// The ring's **bookkeeping**, free of GPU resources: which frame each layer
/// holds and which layer is filled next.
///
/// Split out from [`FrameRing`] so the eviction and lookup rules — the part with
/// actual logic — are unit-testable without a device.
#[derive(Debug)]
struct RingSlots {
    /// Which video frame each layer holds; `None` once invalidated or never
    /// filled. Indexed by layer.
    occupants: Vec<Option<u32>>,
    /// Next layer to fill; wraps.
    write: u32,
}

impl RingSlots {
    fn new(capacity: u32) -> Self {
        Self {
            occupants: vec![None; capacity.max(1) as usize],
            write: 0,
        }
    }

    fn capacity(&self) -> u32 {
        self.occupants.len() as u32
    }

    fn resident(&self) -> u32 {
        self.occupants.iter().filter(|slot| slot.is_some()).count() as u32
    }

    /// The layer holding `frame_index`, if it is still resident.
    fn find(&self, frame_index: u32) -> Option<u32> {
        self.occupants
            .iter()
            .position(|slot| *slot == Some(frame_index))
            .map(|index| index as u32)
    }

    /// Claims the next layer for `frame_index`, evicting whatever it held.
    ///
    /// Any other layer already claiming `frame_index` is released first, so
    /// [`find`](Self::find) can never return a stale copy of a re-uploaded frame.
    fn claim(&mut self, frame_index: Option<u32>) -> u32 {
        let layer = self.write;
        self.write = (self.write + 1) % self.capacity();
        if frame_index.is_some() {
            for slot in self.occupants.iter_mut() {
                if *slot == frame_index {
                    *slot = None;
                }
            }
        }
        self.occupants[layer as usize] = frame_index;
        layer
    }

    /// Drops every resident frame, keeping the allocation.
    fn invalidate(&mut self) {
        self.occupants.iter_mut().for_each(|slot| *slot = None);
        self.write = 0;
    }
}

/// A ring of decoded frames resident on the GPU.
///
/// The decoder fills slots ahead of the renderer, which presents whichever slot
/// holds the frame it wants — the "stream the map in behind you" pattern. It
/// buys two things the single-texture path could not:
///
/// * **Decode/render decoupling.** An upload no longer sits on the render path;
///   a producer that runs ahead simply fills further slots.
/// * **Scrub hits.** A frame still resident is presented by writing its layer
///   index into the fit uniform — no decode, no upload, no rebind. See
///   [`present_resident`](FramePlane::present_resident).
///
/// **One array texture, one bind group.** Slots are array layers, not separate
/// textures: the sampler and the fit uniform are shared by every slot (the fit
/// depends on the frame *size*, not on which frame), so a single bind group
/// serves the whole ring and presenting another frame costs a `vec4` write.
/// `wgpu-core` tracks texture state per subresource (`selector.layers` is a
/// range), so writing one layer while another is being sampled does not
/// serialize.
///
/// **No memory barriers.** wgpu tracks resource state and inserts transitions
/// itself; there is no barrier API to call. The one hazard the ring must avoid —
/// overwriting a slot the GPU is still sampling — is a *capacity* problem, not a
/// synchronization one: a barrier would serialize the write against the draw and
/// defeat the pipelining the ring exists to create. [`RING_MIN_SLOTS`] keeps the
/// write cursor clear of the frames in flight.
struct FrameRing {
    /// `capacity` array layers of `width`×`height`.
    texture: wgpu::Texture,
    /// The one bind group: the whole ring, the sampler, and the fit uniform.
    bind_group: wgpu::BindGroup,
    /// `vec4(uv_scale.xy, layer, _pad)` — rewritten when the fit or the
    /// presented slot changes.
    fit_uniform: wgpu::Buffer,
    /// Which layer holds which frame, and which is filled next.
    slots: RingSlots,
    /// Layer the frame plane draws from, or `None` before the first frame.
    presented: Option<u32>,
    width: u32,
    height: u32,
}

impl FrameRing {
    /// Layers that fit [`RING_BUDGET_BYTES`] at this frame size, clamped to
    /// [`RING_MIN_SLOTS`]..=[`RING_MAX_SLOTS`].
    fn capacity_for(device: &wgpu::Device, width: u32, height: u32) -> u32 {
        let bytes = (width as usize).saturating_mul(height as usize) * 4;
        let fits = (RING_BUDGET_BYTES / bytes.max(1)) as u32;
        fits.clamp(RING_MIN_SLOTS, RING_MAX_SLOTS)
            .min(device.limits().max_texture_array_layers)
            .max(1)
    }

    fn matches(&self, width: u32, height: u32) -> bool {
        self.width == width && self.height == height
    }
}

/// Ring occupancy and reuse, for the diagnostics panel.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct FrameRingStats {
    pub capacity: u32,
    pub resident: u32,
    /// Frames presented straight from the ring, with no upload.
    pub hits: u32,
    /// Frames that had to be uploaded.
    pub misses: u32,
}

/// The background frame-plane subsystem: the fullscreen pipeline, its bind-group
/// layout, the shared sampler, and the ring of decoded frames (empty until the
/// first upload). While nothing is bound every method is a no-op, so a scene
/// asking for a frame plane simply renders nothing.
pub(super) struct FramePlane {
    pipeline: wgpu::RenderPipeline,
    layout: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
    ring: Option<FrameRing>,
    hits: u32,
    misses: u32,
}

impl FramePlane {
    /// Constructs a `FramePlane` with its pipeline and sampler built for `format`
    /// at `sample_count`× (matching the mesh pass it draws within) and no frame
    /// texture yet (the first [`upload_rgba`](Self::upload_rgba) creates it).
    pub(super) fn new(
        device: &wgpu::Device,
        format: wgpu::TextureFormat,
        sample_count: u32,
    ) -> Self {
        let layout = create_frame_bind_group_layout(device);
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("trd frame plane pipeline layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = create_frame_plane_pipeline(device, format, &pipeline_layout, sample_count);
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("trd frame plane sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            ..Default::default()
        });
        Self {
            pipeline,
            layout,
            sampler,
            ring: None,
            hits: 0,
            misses: 0,
        }
    }

    /// Whether a frame is currently presentable (so a scene asking for a frame
    /// plane would render one).
    pub(super) fn is_bound(&self) -> bool {
        self.ring
            .as_ref()
            .is_some_and(|ring| ring.presented.is_some())
    }

    /// Ring occupancy and hit/miss counts, for the diagnostics panel.
    pub(super) fn ring_stats(&self) -> FrameRingStats {
        let (capacity, resident) = self.ring.as_ref().map_or((0, 0), |ring| {
            (ring.slots.capacity(), ring.slots.resident())
        });
        FrameRingStats {
            capacity,
            resident,
            hits: self.hits,
            misses: self.misses,
        }
    }

    /// Presents an already-resident frame, if the ring still holds it.
    ///
    /// This is the ring's payoff for scrubbing: a frame the decoder produced
    /// earlier is shown by rebinding one bind group — no decode, no upload, no
    /// CPU traffic at all.
    pub(super) fn present_resident(&mut self, frame_index: u32) -> bool {
        let Some(ring) = self.ring.as_mut() else {
            return false;
        };
        let Some(slot) = ring.slots.find(frame_index) else {
            return false;
        };
        ring.presented = Some(slot);
        self.hits = self.hits.saturating_add(1);
        true
    }

    /// Drops every resident frame — a seek or source change makes their indices
    /// meaningless. Allocations are kept, so the next frame does not reallocate.
    pub(super) fn invalidate_ring(&mut self) {
        if let Some(ring) = self.ring.as_mut() {
            ring.slots.invalidate();
            ring.presented = None;
        }
    }

    /// Uploads `rgba` (tightly-packed, row-major `height`x`width`x4) into the
    /// ring as `frame_index`, and presents it.
    ///
    /// `frame_index` is what makes a later [`present_resident`] hit possible;
    /// callers with no timeline (a still background) pass `None`, which fills a
    /// slot that can never be found again — correct, since there is nothing to
    /// scrub back to.
    ///
    /// Panics if `rgba.len() != width * height * 4` or either dimension is zero.
    pub(super) fn upload_rgba(
        &mut self,
        gpu: &GpuContext,
        rgba: &[u8],
        width: u32,
        height: u32,
        frame_index: Option<u32>,
    ) {
        assert!(
            width > 0 && height > 0,
            "frame texture dimensions must be non-zero"
        );
        assert_eq!(
            rgba.len(),
            width as usize * height as usize * 4,
            "frame texture rgba length must be width*height*4"
        );
        let layer = self.claim_layer(&gpu.device, width, height, frame_index);
        let ring = self.ring.as_ref().expect("ring created above");
        gpu.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &ring.texture,
                mip_level: 0,
                origin: wgpu::Origin3d {
                    x: 0,
                    y: 0,
                    z: layer,
                },
                aspect: wgpu::TextureAspect::All,
            },
            rgba,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(width * 4),
                rows_per_image: Some(height),
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );
    }

    /// Copies a decoded `VideoFrame`'s pixels straight into a ring slot,
    /// **GPU→GPU**, without them ever entering CPU memory (#229).
    ///
    /// The browser has already decoded the frame, in hardware where it can, into
    /// GPU memory. The `upload_rgba` route drags it back down three times —
    /// `VideoFrame.copyTo` (which also performs the YUV→RGBA conversion), the
    /// wasm-bindgen boundary, and `write_texture` — at *source* resolution:
    /// ~99 MB per frame for 4K.
    ///
    /// Three decisions worth recording:
    ///
    /// * **`VideoFrame`, not `HtmlVideoElement`.** #276 took the element because
    ///   `web_sys::VideoFrame` was believed to need the build-wide
    ///   `web_sys_unstable_apis` rustflag. Checked against web-sys 0.3.103: the
    ///   **type is not gated** — only `rotation`, `flip` and `metadata` are, none
    ///   of which this uses — so the `VideoFrame` feature alone is enough. With
    ///   WebCodecs (#282) there is no element to name anyway, and a frame is the
    ///   thing the decoder actually hands over.
    /// * **The caller still owns the frame.** WebGPU snapshots the source during
    ///   this call, so it may be closed immediately after — and it must be, since
    ///   a `VideoFrame` holds a slot in a small decoder-side pool.
    /// * **Web-only by construction.** `copy_external_image_to_texture` is
    ///   `#[cfg(web)]` in wgpu, and native does not want it: its frames arrive
    ///   from an ffmpeg pipe as CPU bytes already.
    /// * **Zero-copy is the browser's decision, not ours.** The spec does not
    ///   guarantee it; a software-decoded frame starts in CPU memory and a
    ///   YUV→RGB pass may still run. What this guarantees is that *we* no longer
    ///   force the download.
    #[cfg(target_arch = "wasm32")]
    pub(super) fn copy_video_frame(
        &mut self,
        gpu: &GpuContext,
        frame: &web_sys::VideoFrame,
        width: u32,
        height: u32,
        frame_index: Option<u32>,
    ) {
        // The caller validates dimensions before publishing a frame, so a
        // degenerate one here is a bug rather than an input to tolerate.
        assert!(
            width > 0 && height > 0,
            "frame texture dimensions must be non-zero"
        );
        let layer = self.claim_layer(&gpu.device, width, height, frame_index);
        let ring = self.ring.as_ref().expect("ring created above");
        gpu.queue.copy_external_image_to_texture(
            &wgpu::CopyExternalImageSourceInfo {
                // `Clone::clone`, explicitly: `frame.clone()` resolves to
                // WebCodecs' own `clone()`, which duplicates the frame — taking
                // a *second* pool slot that would then have to be closed too.
                // What is wanted here is another handle to the same frame.
                source: wgpu::ExternalImageSource::VideoFrame(Clone::clone(frame)),
                origin: wgpu::Origin2d::ZERO,
                flip_y: false,
            },
            wgpu::CopyExternalImageDestInfo {
                texture: &ring.texture,
                mip_level: 0,
                origin: wgpu::Origin3d {
                    x: 0,
                    y: 0,
                    z: layer,
                },
                aspect: wgpu::TextureAspect::All,
                color_space: wgpu::PredefinedColorSpace::Srgb,
                premultiplied_alpha: false,
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );
    }

    /// Creates the ring if needed, claims the next layer for `frame_index` and
    /// presents it. Shared by both upload paths.
    fn claim_layer(
        &mut self,
        device: &wgpu::Device,
        width: u32,
        height: u32,
        frame_index: Option<u32>,
    ) -> u32 {
        // A resolution change invalidates every resident frame, so the ring is
        // rebuilt rather than resized.
        if !self
            .ring
            .as_ref()
            .is_some_and(|ring| ring.matches(width, height))
        {
            self.ring = Some(self.create_ring(device, width, height));
        }
        let ring = self.ring.as_mut().expect("ring created above");
        let layer = ring.slots.claim(frame_index);
        ring.presented = Some(layer);
        self.misses = self.misses.saturating_add(1);
        layer
    }

    /// Allocates the array texture, its single bind group and the fit uniform.
    fn create_ring(&self, device: &wgpu::Device, width: u32, height: u32) -> FrameRing {
        let capacity = FrameRing::capacity_for(device, width, height);
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("trd frame ring"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: capacity,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            // `RENDER_ATTACHMENT` is required by
            // `copy_external_image_to_texture`, which writes through the render
            // pipeline — omitting it is a validation error, not a slow path.
            usage: wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_DST
                | wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor {
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            ..Default::default()
        });
        let fit_uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("trd frame fit uniform"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("trd frame plane bind group"),
            layout: &self.layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: fit_uniform.as_entire_binding(),
                },
            ],
        });
        log::debug!("frame ring: {capacity} slots of {width}x{height}");
        FrameRing {
            texture,
            bind_group,
            fit_uniform,
            slots: RingSlots::new(capacity),
            presented: None,
            width,
            height,
        }
    }

    /// Computes and uploads the centered UV-fit scale that realizes `fit` on
    /// `viewport`, **together with the presented layer** - the two travel in one
    /// `vec4`, so moving to another resident frame is this single write. No-op
    /// while the ring holds nothing.
    pub(super) fn write_fit(&self, queue: &wgpu::Queue, fit: FrameFit, viewport: Viewport) {
        let Some(ring) = self.ring.as_ref() else {
            return;
        };
        let Some(layer) = ring.presented else {
            return;
        };
        let scale = frame_fit_uv_scale(
            fit,
            ring.width,
            ring.height,
            viewport.width,
            viewport.height,
        );
        let fit_data: [f32; 4] = [scale[0], scale[1], layer as f32, 0.0];
        queue.write_buffer(&ring.fit_uniform, 0, bytemuck::cast_slice(&fit_data));
    }

    /// Records the fullscreen frame-plane draw (its own pipeline + group-0 bind,
    /// depth-write off) so the mesh scene composites on top. No-op while the ring
    /// holds nothing.
    pub(super) fn draw(&self, pass: &mut wgpu::RenderPass) {
        let Some(ring) = self.ring.as_ref() else {
            return;
        };
        if ring.presented.is_none() {
            return;
        }
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &ring.bind_group, &[]);
        pass.draw(0..3, 0..1);
    }
}

#[cfg(test)]
mod tests {
    use super::{RingSlots, RING_BUDGET_BYTES, RING_MAX_SLOTS, RING_MIN_SLOTS};

    /// The same `capacity_for` arithmetic as [`super::FrameRing`], minus the
    /// device limit — that clamp needs a GPU, the budget policy does not.
    fn capacity_for(width: u32, height: u32) -> u32 {
        let bytes = (width as usize).saturating_mul(height as usize) * 4;
        ((RING_BUDGET_BYTES / bytes.max(1)) as u32).clamp(RING_MIN_SLOTS, RING_MAX_SLOTS)
    }

    #[test]
    fn claim_wraps_and_evicts_in_order() {
        let mut slots = RingSlots::new(3);
        assert_eq!(slots.claim(Some(10)), 0);
        assert_eq!(slots.claim(Some(11)), 1);
        assert_eq!(slots.claim(Some(12)), 2);
        assert_eq!(slots.resident(), 3);
        // Wraps onto the oldest slot, evicting frame 10.
        assert_eq!(slots.claim(Some(13)), 0);
        assert_eq!(slots.find(10), None);
        assert_eq!(slots.find(13), Some(0));
        assert_eq!(slots.resident(), 3);
    }

    #[test]
    fn find_locates_resident_frames_only() {
        let mut slots = RingSlots::new(4);
        slots.claim(Some(7));
        assert_eq!(slots.find(7), Some(0));
        assert_eq!(slots.find(8), None);
    }

    #[test]
    fn reuploading_a_frame_releases_its_old_slot() {
        let mut slots = RingSlots::new(4);
        slots.claim(Some(5));
        let second = slots.claim(Some(5));
        assert_eq!(second, 1);
        // Exactly one slot may claim a frame, so `find` cannot return the stale
        // layer whose contents were overwritten.
        assert_eq!(slots.find(5), Some(1));
        assert_eq!(slots.resident(), 1);
    }

    #[test]
    fn unindexed_frames_are_never_found() {
        let mut slots = RingSlots::new(4);
        slots.claim(None);
        assert_eq!(slots.resident(), 0);
        assert_eq!(slots.find(0), None);
    }

    #[test]
    fn invalidate_clears_occupancy_and_rewinds() {
        let mut slots = RingSlots::new(4);
        slots.claim(Some(1));
        slots.claim(Some(2));
        slots.invalidate();
        assert_eq!(slots.resident(), 0);
        assert_eq!(slots.find(1), None);
        assert_eq!(slots.claim(Some(3)), 0);
    }

    #[test]
    fn capacity_scales_with_frame_size_within_bounds() {
        // 4K frames are 31.6 MiB each, so the budget affords only the floor —
        // and the floor wins, because it is a correctness requirement rather
        // than a target.
        assert_eq!(capacity_for(3840, 2160), RING_MIN_SLOTS);
        // 1080p is 7.91 MiB, so the budget affords exactly the cap.
        assert_eq!(capacity_for(1920, 1080), RING_MAX_SLOTS);
        // 960x540 is 2 MiB, so the budget affords far more than the cap allows.
        assert_eq!(capacity_for(960, 540), RING_MAX_SLOTS);
        // A degenerate size must not divide by zero; it saturates at the cap,
        // which is harmless because uploads reject zero dimensions anyway.
        assert_eq!(capacity_for(0, 0), RING_MAX_SLOTS);
    }

    /// The cap is an evidence-based choice, not a round number: a deeper ring
    /// buys almost nothing because a scrub is a span of *seconds* while the ring
    /// is a span of *frames*. Pinning it here so raising it needs new evidence
    /// rather than an intuition.
    #[test]
    fn the_ring_stays_shallow_enough_to_be_worth_its_vram() {
        let vram = |slots: u32, w: usize, h: usize| slots as usize * w * h * 4;
        // At 4K the floor alone already costs 126 MiB; anything deeper would be
        // a gigabyte for a window of well under a tenth of a second at 60 fps.
        assert!(vram(capacity_for(3840, 2160), 3840, 2160) <= 128 * 1024 * 1024);
        // At 1080p the whole ring fits inside the budget.
        assert!(vram(capacity_for(1920, 1080), 1920, 1080) <= RING_BUDGET_BYTES);
    }
}
