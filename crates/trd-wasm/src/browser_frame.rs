//! The browser's [`ExternalFrame`](trd_core::ExternalFrame): a decoded WebCodecs
//! frame, copied GPU→GPU and closed on drop (#229, #302).
//!
//! This is the one place in the repo that names `web_sys::VideoFrame`, and the
//! one place that may: `Queue::copy_external_image_to_texture` is `#[cfg(web)]`
//! in wgpu, so the copy cannot be compiled into a crate that also builds
//! natively. Everything around it — allocating the destination texture, holding
//! its format and usage invariants — stays in `trd-core`.

use trd_core::ExternalFrame;

/// A decoded browser frame, released when the last handle drops.
///
/// **RAII, deliberately.** A `VideoFrame` holds a slot in a small decoder-side
/// pool, so it must be closed or the decoder eventually stalls with nothing to
/// decode into. That used to be spelled out by hand at three separate sites
/// across two crates — the superseded frame, a rejected degenerate size, and a
/// rejected out-of-range index — each of which had to get it right on its own.
/// `Drop` is the same rule stated once, and the editor holds the frame in an
/// `Rc`, so "released when a newer frame replaces it" is now what the types do
/// rather than what a comment asks a caller to remember.
pub struct BrowserVideoFrame(web_sys::VideoFrame);

impl BrowserVideoFrame {
    pub fn new(frame: web_sys::VideoFrame) -> Self {
        Self(frame)
    }
}

impl Drop for BrowserVideoFrame {
    fn drop(&mut self) {
        self.0.close();
    }
}

impl ExternalFrame for BrowserVideoFrame {
    fn size(&self) -> (u32, u32) {
        (self.0.display_width(), self.0.display_height())
    }

    fn copy_into(&self, queue: &wgpu::Queue, texture: &wgpu::Texture) {
        let (width, height) = self.size();
        queue.copy_external_image_to_texture(
            &wgpu::CopyExternalImageSourceInfo {
                // `Clone::clone`, explicitly: `self.0.clone()` resolves to
                // WebCodecs' own `clone()`, which duplicates the frame — taking
                // a *second* pool slot that would then have to be closed too.
                // What is wanted here is another handle to the same frame.
                source: wgpu::ExternalImageSource::VideoFrame(Clone::clone(&self.0)),
                origin: wgpu::Origin2d::ZERO,
                flip_y: false,
            },
            wgpu::CopyExternalImageDestInfo {
                texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
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
}
