//! Render targets — where a frame's pixels land.
//!
//! Both targets are just somewhere to render **into a `wgpu::TextureView`**: the
//! encoding in between is identical, and only the two ends differ — how the view
//! is acquired (an owned texture, infallibly, vs a surface that can be outdated
//! or lost) and what happens after submission (copy + map + read back the pixels
//! vs present the frame). They live together here so that asymmetry is visible
//! in one place (#180).
//!
//! [`OffscreenTarget`] is the common case: the headless CLI, the golden tests,
//! both browser renderers and the video editors all read pixels back.
//! [`OnscreenTarget`] serves the two live-surface shells (`trd-app`, the browser
//! canvas). Note the picking target is deliberately **not** here: it is a second
//! pass producing ids, not a place a frame is rendered to (see `picking.rs`).

use super::GpuContext;
use futures_channel::oneshot;
use thiserror::Error;

use super::{DrawableObject, FrameParams, SceneRenderer, Viewport};
use crate::tightly_pack_rgba;

/// What every render target has in common.
///
/// Deliberately thin. Both targets are just somewhere to render **into a
/// `wgpu::TextureView`**, and the encoding in between is identical — only the
/// two ends differ: how the view is acquired (an owned texture, infallibly,
/// versus a surface that can be outdated or lost) and what happens after
/// submission (copy + map + read pixels back, versus present). Those tails
/// produce different things (`Vec<u8>` versus nothing), so they stay as inherent
/// methods on each target rather than being forced through this trait; putting
/// them here would mean returning `Option<Vec<u8>>`, where the `None` is real
/// on-screen and therefore belongs to a different layer (I5 in #180).
///
/// `PickTarget` deliberately does **not** implement this: it is a second pass
/// producing ids, not a place a frame is rendered to.
pub trait RenderTarget {
    /// The texture format a renderer's pipelines must be built for. Offscreen
    /// this is [`OFFSCREEN_FORMAT`]; on-screen it is the surface's **sRGB view**
    /// format, which is why it must be read off the target rather than assumed.
    fn view_format(&self) -> wgpu::TextureFormat;

    /// The current render size in pixels.
    fn viewport(&self) -> Viewport;

    /// Resizes the target to `width` x `height`.
    fn resize(&mut self, gpu: &GpuContext, width: u32, height: u32) -> Result<(), OffscreenError>;
}

// ---------------------------------------------------------------------------
// Offscreen
// ---------------------------------------------------------------------------

// [`OffscreenTarget`] — the shared **offscreen** render harness (#103, Part B).
//
// Every front-end that renders headless to pixels (the native [`Renderer`]
// behind the CLI + GUI, `trd-wasm`'s browser `OffscreenRenderer`, `trd-gui`'s
// browser `WebRenderer`) used to own an identical copy of the same GPU
// plumbing: a `Rgba8UnormSrgb` target texture, a `MAP_READ` staging buffer, and
// the per-frame *encode → copy-to-buffer → map → readback → unpad* dance. This
// module owns that once so a renderer is just *device + queue + [`SceneRenderer`]
// + `OffscreenTarget`*.
//
// **One async core, two waits.** [`OffscreenTarget::render`] is `async` because
// the browser event loop must not be blocked during GPU readback. The only
// genuinely target-specific bit is *how the map completes*: native blocks the
// calling thread (`device.poll(wait_indefinitely)`) so the headless CLI/GUI can
// drive it under `pollster::block_on`, while wasm kicks the queue once
// (`device.poll(Poll)`) and yields via `.await` to the browser. That is a
// two-line `cfg` split; everything else is shared.
//
// Device/surface creation stays in each shell (native uses
// `downlevel_defaults`; the browser uses the adapter's real limits), and the
// on-screen (surface) path is a separate harness.

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
        gpu: &GpuContext,
        renderer: &mut SceneRenderer,
        params: FrameParams,
        scene: &[DrawableObject],
    ) -> Result<Vec<u8>, OffscreenError> {
        self.render_passes(gpu, renderer, params, params, None, scene, None)
            .await
    }

    /// Renders a background scene first, then preserves that color while
    /// rendering the foreground scene in a second pass.
    #[allow(clippy::too_many_arguments)]
    pub async fn render_two_pass(
        &self,
        gpu: &GpuContext,
        renderer: &mut SceneRenderer,
        background_params: FrameParams,
        foreground_params: FrameParams,
        background: &[DrawableObject],
        foreground: &[DrawableObject],
    ) -> Result<Vec<u8>, OffscreenError> {
        self.render_passes(
            gpu,
            renderer,
            background_params,
            foreground_params,
            Some(background),
            foreground,
            None,
        )
        .await
    }

    /// Renders background, foreground, and selection/gizmo overlay as three
    /// independently submitted passes.
    #[allow(clippy::too_many_arguments)]
    pub async fn render_three_pass(
        &self,
        gpu: &GpuContext,
        renderer: &mut SceneRenderer,
        background_params: FrameParams,
        foreground_params: FrameParams,
        background: &[DrawableObject],
        foreground: &[DrawableObject],
        overlay: &[DrawableObject],
    ) -> Result<Vec<u8>, OffscreenError> {
        self.render_passes(
            gpu,
            renderer,
            background_params,
            foreground_params,
            Some(background),
            foreground,
            Some(overlay),
        )
        .await
    }

    #[allow(clippy::too_many_arguments)]
    async fn render_passes(
        &self,
        gpu: &GpuContext,
        renderer: &mut SceneRenderer,
        background_params: FrameParams,
        foreground_params: FrameParams,
        background: Option<&[DrawableObject]>,
        foreground: &[DrawableObject],
        overlay: Option<&[DrawableObject]>,
    ) -> Result<Vec<u8>, OffscreenError> {
        let (device, queue) = (&gpu.device, &gpu.queue);
        let view = self
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let viewport = Viewport {
            width: self.width,
            height: self.height,
        };
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("trd offscreen foreground"),
        });
        if let Some(background) = background {
            let mut background_encoder =
                device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("trd offscreen background"),
                });
            renderer.encode(
                &mut background_encoder,
                &view,
                background_params,
                background,
                viewport,
            );
            // Submit before `encode_overlay` uploads foreground instances into
            // SceneRenderer's shared instance buffer.
            queue.submit(Some(background_encoder.finish()));
            renderer.encode_overlay(&mut encoder, &view, foreground_params, foreground, viewport);
        } else {
            renderer.encode(&mut encoder, &view, foreground_params, foreground, viewport);
        }
        if let Some(overlay) = overlay {
            queue.submit(Some(encoder.finish()));
            encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("trd offscreen selection overlay"),
            });
            renderer.encode_overlay(&mut encoder, &view, foreground_params, overlay, viewport);
        }
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
        // Native blocks until the mapping completes; the browser kicks the queue
        // and lets the `.await` below yield. See `platform::poll_for_map`.
        super::platform::poll_for_map(device).map_err(|e| OffscreenError::Gpu(e.to_string()))?;
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

// ---------------------------------------------------------------------------
// Onscreen
// ---------------------------------------------------------------------------

// [`OnscreenTarget`] — the shared **on-screen** (surface) render harness
// (#103, Part B).
//
// The interactive front-ends that present to a live surface — the native
// windowed `trd-app` and `trd-wasm`'s browser `CanvasRenderer` — used to own an
// identical copy of the same per-frame present dance: build an **sRGB view** of
// the acquired surface texture, encode the frame's [`Scene`](crate::Scene) with
// the shared [`SceneRenderer`], submit, and present. This module owns that once,
// so a front-end is just *device + queue + [`SceneRenderer`] + `OnscreenTarget`*
// plus its own surface-acquire recovery policy.
//
// **sRGB, once.** The browser's preferred canvas format is non-sRGB (e.g.
// `Bgra8Unorm`), so a pipeline targeting it writes *linear* values with no
// linear→sRGB encode — darker/muddier than the headless CLI's `Rgba8UnormSrgb`
// target. Native surfaces are usually sRGB already. Rather than each shell
// special-casing this, [`OnscreenTarget`] always renders through the surface's
// **sRGB view** ([`add_srgb_suffix`](wgpu::TextureFormat::add_srgb_suffix),
// registered in `view_formats`), so both platforms match the CLI byte-for-byte.
// Build the front-end's [`SceneRenderer`] with [`OnscreenTarget::view_format`].
//
// **What stays in each shell.** Device/adapter/surface creation (a winit window
// vs a canvas, `downlevel_defaults` vs the adapter's real limits, the
// `present_mode` choice) and the **surface-acquire recovery policy** are
// genuinely target-specific: the native app is driven by a winit event loop, so
// on an outdated/lost surface it reconfigures and defers to the next redraw,
// while the browser renderer is driven imperatively per frame, so it retries
// within the call (recreating the surface from the canvas on loss). The harness
// exposes [`acquire`](Self::acquire), [`reconfigure`](Self::reconfigure), and
// [`replace_surface`](Self::replace_surface) so each shell keeps its policy
// while sharing everything mechanical.

/// A live surface plus its configuration, rendered through an sRGB view so
/// on-screen color matches the headless CLI's `Rgba8UnormSrgb` output. Owns the
/// [`wgpu::Surface`] and its [`wgpu::SurfaceConfiguration`]; the front-end owns
/// the device/queue, the [`SceneRenderer`], and its acquire-recovery policy.
pub struct OnscreenTarget {
    surface: wgpu::Surface<'static>,
    config: wgpu::SurfaceConfiguration,
    /// The sRGB view format the frame is rendered through (the sRGB variant of
    /// `config.format`; equal to `config.format` when it is already sRGB). The
    /// front-end's [`SceneRenderer`] pipeline must target this format.
    view_format: wgpu::TextureFormat,
}

impl OnscreenTarget {
    /// Wraps an already-created surface and its default configuration, registers
    /// the sRGB view format, and configures the surface. `config` is typically
    /// obtained from [`wgpu::Surface::get_default_config`] with the shell's
    /// chosen `present_mode` applied; this harness owns it from here.
    pub fn new(
        device: &wgpu::Device,
        surface: wgpu::Surface<'static>,
        mut config: wgpu::SurfaceConfiguration,
    ) -> Self {
        let view_format = config.format.add_srgb_suffix();
        if view_format != config.format && !config.view_formats.contains(&view_format) {
            config.view_formats.push(view_format);
        }
        surface.configure(device, &config);
        Self {
            surface,
            config,
            view_format,
        }
    }

    /// The sRGB view format the frame is rendered through. Build the front-end's
    /// [`SceneRenderer`] with this so its pipeline target matches the view.
    pub fn view_format(&self) -> wgpu::TextureFormat {
        self.view_format
    }

    /// The current surface width in pixels.
    pub fn width(&self) -> u32 {
        self.config.width
    }

    /// The current surface height in pixels.
    pub fn height(&self) -> u32 {
        self.config.height
    }

    /// Reapplies the current configuration to the surface — the recovery step
    /// after an outdated/lost/suboptimal acquire.
    pub fn reconfigure(&self, device: &wgpu::Device) {
        self.surface.configure(device, &self.config);
    }

    /// Updates the configured size and reconfigures the surface. Ignores a zero
    /// width or height (e.g. a minimized window).
    pub fn resize(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        if width > 0 && height > 0 {
            self.config.width = width;
            self.config.height = height;
            self.reconfigure(device);
        }
    }

    /// Swaps in a freshly created surface (e.g. after the browser reports the
    /// canvas surface *lost*) and reconfigures it. The new surface must target
    /// the same canvas/window as the original.
    pub fn replace_surface(&mut self, device: &wgpu::Device, surface: wgpu::Surface<'static>) {
        self.surface = surface;
        self.reconfigure(device);
    }

    /// Acquires the surface's next texture, returning wgpu's status enum verbatim
    /// so the front-end applies its own recovery policy (native defers to a
    /// redraw; the browser retries in-call, recreating the surface on loss).
    pub fn acquire(&self) -> wgpu::CurrentSurfaceTexture {
        self.surface.get_current_texture()
    }

    /// Encodes `scene` under `params` onto `texture`'s sRGB view with `renderer`,
    /// submits, and presents. The mechanical per-frame block shared by every
    /// on-screen front-end; call it with a texture obtained from
    /// [`acquire`](Self::acquire).
    pub fn present(
        &self,
        gpu: &GpuContext,
        renderer: &mut SceneRenderer,
        texture: wgpu::SurfaceTexture,
        params: FrameParams,
        scene: &[DrawableObject],
    ) {
        let (device, queue) = (&gpu.device, &gpu.queue);
        let view = texture.texture.create_view(&wgpu::TextureViewDescriptor {
            format: Some(self.view_format),
            ..Default::default()
        });
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("trd onscreen frame"),
        });
        renderer.encode(
            &mut encoder,
            &view,
            params,
            scene,
            Viewport {
                width: self.config.width,
                height: self.config.height,
            },
        );
        queue.submit(Some(encoder.finish()));
        queue.present(texture);
    }
}

impl RenderTarget for OffscreenTarget {
    fn view_format(&self) -> wgpu::TextureFormat {
        OFFSCREEN_FORMAT
    }

    fn viewport(&self) -> Viewport {
        Viewport {
            width: self.width,
            height: self.height,
        }
    }

    fn resize(&mut self, gpu: &GpuContext, width: u32, height: u32) -> Result<(), OffscreenError> {
        *self = Self::new(&gpu.device, width, height)?;
        Ok(())
    }
}

impl RenderTarget for OnscreenTarget {
    fn view_format(&self) -> wgpu::TextureFormat {
        self.view_format
    }

    fn viewport(&self) -> Viewport {
        Viewport {
            width: self.config.width,
            height: self.config.height,
        }
    }

    /// Never fails: a surface resize reconfigures in place.
    fn resize(&mut self, gpu: &GpuContext, width: u32, height: u32) -> Result<(), OffscreenError> {
        OnscreenTarget::resize(self, &gpu.device, width, height);
        Ok(())
    }
}
