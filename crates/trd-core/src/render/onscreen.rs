//! [`OnscreenTarget`] — the shared **on-screen** (surface) render harness
//! (#103, Part B).
//!
//! The interactive front-ends that present to a live surface — the native
//! windowed `trd-app` and `trd-wasm`'s browser `CanvasRenderer` — used to own an
//! identical copy of the same per-frame present dance: build an **sRGB view** of
//! the acquired surface texture, encode the frame's [`Scene`](crate::Scene) with
//! the shared [`MeshRenderer`], submit, and present. This module owns that once,
//! so a front-end is just *device + queue + [`MeshRenderer`] + `OnscreenTarget`*
//! plus its own surface-acquire recovery policy.
//!
//! **sRGB, once.** The browser's preferred canvas format is non-sRGB (e.g.
//! `Bgra8Unorm`), so a pipeline targeting it writes *linear* values with no
//! linear→sRGB encode — darker/muddier than the headless CLI's `Rgba8UnormSrgb`
//! target. Native surfaces are usually sRGB already. Rather than each shell
//! special-casing this, [`OnscreenTarget`] always renders through the surface's
//! **sRGB view** ([`add_srgb_suffix`](wgpu::TextureFormat::add_srgb_suffix),
//! registered in `view_formats`), so both platforms match the CLI byte-for-byte.
//! Build the front-end's [`MeshRenderer`] with [`OnscreenTarget::view_format`].
//!
//! **What stays in each shell.** Device/adapter/surface creation (a winit window
//! vs a canvas, `downlevel_defaults` vs the adapter's real limits, the
//! `present_mode` choice) and the **surface-acquire recovery policy** are
//! genuinely target-specific: the native app is driven by a winit event loop, so
//! on an outdated/lost surface it reconfigures and defers to the next redraw,
//! while the browser renderer is driven imperatively per frame, so it retries
//! within the call (recreating the surface from the canvas on loss). The harness
//! exposes [`acquire`](Self::acquire), [`reconfigure`](Self::reconfigure), and
//! [`replace_surface`](Self::replace_surface) so each shell keeps its policy
//! while sharing everything mechanical.

use super::{DrawableObject, FrameParams, MeshRenderer, Viewport};

/// A live surface plus its configuration, rendered through an sRGB view so
/// on-screen color matches the headless CLI's `Rgba8UnormSrgb` output. Owns the
/// [`wgpu::Surface`] and its [`wgpu::SurfaceConfiguration`]; the front-end owns
/// the device/queue, the [`MeshRenderer`], and its acquire-recovery policy.
pub struct OnscreenTarget {
    surface: wgpu::Surface<'static>,
    config: wgpu::SurfaceConfiguration,
    /// The sRGB view format the frame is rendered through (the sRGB variant of
    /// `config.format`; equal to `config.format` when it is already sRGB). The
    /// front-end's [`MeshRenderer`] pipeline must target this format.
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
    /// [`MeshRenderer`] with this so its pipeline target matches the view.
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
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        renderer: &mut MeshRenderer,
        texture: wgpu::SurfaceTexture,
        params: FrameParams,
        scene: &[DrawableObject],
    ) {
        let view = texture.texture.create_view(&wgpu::TextureViewDescriptor {
            format: Some(self.view_format),
            ..Default::default()
        });
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("trd onscreen frame"),
        });
        renderer.encode(
            queue,
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
