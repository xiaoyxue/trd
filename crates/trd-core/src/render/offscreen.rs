//! [`OffscreenTarget`] — the shared **offscreen** render harness (#103, Part B).
//!
//! Every front-end that renders headless to pixels (the native [`BatchRenderer`]
//! behind the CLI + GUI, `trd-wasm`'s browser `OffscreenRenderer`, `trd-gui`'s
//! browser `WebRenderer`) used to own an identical copy of the same GPU
//! plumbing: a `Rgba8UnormSrgb` target texture, a `MAP_READ` staging buffer, and
//! the per-frame *encode → copy-to-buffer → map → readback → unpad* dance. This
//! module owns that once so a renderer is just *device + queue + [`MeshRenderer`]
//! + `OffscreenTarget`*.
//!
//! **One async core, two waits.** [`OffscreenTarget::render`] is `async` because
//! the browser event loop must not be blocked during GPU readback. The only
//! genuinely target-specific bit is *how the map completes*: native blocks the
//! calling thread (`device.poll(wait_indefinitely)`) so the headless CLI/GUI can
//! drive it under `pollster::block_on`, while wasm kicks the queue once
//! (`device.poll(Poll)`) and yields via `.await` to the browser. That is a
//! two-line `cfg` split; everything else is shared.
//!
//! Device/surface creation stays in each shell (native uses
//! `downlevel_defaults`; the browser uses the adapter's real limits), and the
//! on-screen (surface) path is a separate harness.

use futures_channel::oneshot;
use thiserror::Error;

use super::{DrawableObject, FrameParams, MeshRenderer, Viewport};
use crate::tightly_pack_rgba;

/// The fixed offscreen render format. Matches the headless CLI's output target
/// so native and browser renders are byte-identical (hardware sRGB-encode on
/// store).
pub const OFFSCREEN_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8UnormSrgb;

/// Errors creating or driving an [`OffscreenTarget`].
#[derive(Debug, Error)]
pub enum OffscreenError {
    /// A zero width or height was requested.
    #[error("render dimensions {width}x{height} are invalid (must be non-zero)")]
    InvalidDimensions { width: u32, height: u32 },
    /// The requested size exceeds the adapter's `max_texture_dimension_2d`.
    #[error("render dimensions {width}x{height} exceed max_texture_dimension_2d {max}")]
    ExceedsMaxDimension { width: u32, height: u32, max: u32 },
    /// `width * 4` (the unpadded row stride) overflows `u32`.
    #[error("row byte count overflows u32 for width {width}")]
    RowOverflow { width: u32 },
    /// A wgpu error while polling / mapping the readback buffer.
    #[error("GPU readback failed: {0}")]
    Gpu(String),
    /// Unpacking the padded readback rows into tightly-packed RGBA failed.
    #[error(transparent)]
    Output(#[from] crate::OutputError),
}

/// An offscreen render target: an owned [`OFFSCREEN_FORMAT`] texture plus the
/// `MAP_READ` staging buffer used to read it back to tightly-packed RGBA. Built
/// once for a fixed size; [`render`](Self::render) reuses both every frame.
pub struct OffscreenTarget {
    texture: wgpu::Texture,
    staging: wgpu::Buffer,
    width: u32,
    height: u32,
    padded_bytes_per_row: u32,
}

impl OffscreenTarget {
    /// Allocates the render target + readback buffer for a fixed `width` ×
    /// `height`, validating non-zero dimensions and the adapter's
    /// `max_texture_dimension_2d`.
    pub fn new(device: &wgpu::Device, width: u32, height: u32) -> Result<Self, OffscreenError> {
        if width == 0 || height == 0 {
            return Err(OffscreenError::InvalidDimensions { width, height });
        }
        let max = device.limits().max_texture_dimension_2d;
        if width > max || height > max {
            return Err(OffscreenError::ExceedsMaxDimension { width, height, max });
        }

        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("trd offscreen target"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: OFFSCREEN_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });

        let unpadded = width
            .checked_mul(4)
            .ok_or(OffscreenError::RowOverflow { width })?;
        let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        let padded_bytes_per_row = unpadded.div_ceil(align) * align;
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("trd offscreen readback"),
            size: u64::from(padded_bytes_per_row) * u64::from(height),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        Ok(Self {
            texture,
            staging,
            width,
            height,
            padded_bytes_per_row,
        })
    }

    /// The fixed render width in pixels.
    pub fn width(&self) -> u32 {
        self.width
    }

    /// The fixed render height in pixels.
    pub fn height(&self) -> u32 {
        self.height
    }

    /// Encodes `scene` under `params` onto the target with `renderer`, then reads
    /// the target back to tightly-packed row-major RGBA (`width * height * 4`
    /// bytes). `async` so the browser event loop is not blocked during readback;
    /// native callers drive it with `pollster::block_on` (which the
    /// `wait_indefinitely` poll below keeps correct).
    pub async fn render(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        renderer: &mut MeshRenderer,
        params: FrameParams,
        scene: &[DrawableObject],
    ) -> Result<Vec<u8>, OffscreenError> {
        let view = self
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("trd offscreen frame"),
        });
        renderer.encode(
            queue,
            &mut encoder,
            &view,
            params,
            scene,
            Viewport {
                width: self.width,
                height: self.height,
            },
        );
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &self.texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &self.staging,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(self.padded_bytes_per_row),
                    rows_per_image: Some(self.height),
                },
            },
            wgpu::Extent3d {
                width: self.width,
                height: self.height,
                depth_or_array_layers: 1,
            },
        );
        queue.submit(Some(encoder.finish()));

        let slice = self.staging.slice(..);
        let (sender, receiver) = oneshot::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });
        // The single genuinely target-specific line: native blocks the calling
        // thread until the map completes (so `block_on` returns a finished
        // future); wasm can't block the event loop, so it kicks the queue once
        // and lets `.await` yield to the browser.
        #[cfg(not(target_arch = "wasm32"))]
        device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|e| OffscreenError::Gpu(e.to_string()))?;
        #[cfg(target_arch = "wasm32")]
        device
            .poll(wgpu::PollType::Poll)
            .map_err(|e| OffscreenError::Gpu(e.to_string()))?;
        receiver
            .await
            .map_err(|_| OffscreenError::Gpu("readback callback cancelled".to_owned()))?
            .map_err(|e| OffscreenError::Gpu(e.to_string()))?;

        let packed = match slice.get_mapped_range() {
            Ok(mapped) => {
                tightly_pack_rgba(&mapped, self.width, self.height, self.padded_bytes_per_row)
                    .map_err(OffscreenError::Output)
            }
            Err(e) => Err(OffscreenError::Gpu(e.to_string())),
        };
        self.staging.unmap();
        packed
    }
}
