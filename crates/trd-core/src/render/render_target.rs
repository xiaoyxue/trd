//! Render targets — **where** a frame's pixels land, as plain data.
//!
//! A target is a *place*, not an actor. Both kinds are somewhere to render into
//! a `wgpu::TextureView`: the encoding in between is identical, and only the two
//! ends differ — how the view is acquired (an owned texture, infallibly, versus
//! a surface that can be outdated or lost) and what happens after submission
//! (copy + map + read the pixels back, versus present the frame).
//!
//! **Nothing in this module renders.** Every `render`/`present`/`read_back`/
//! `acquire` used to hang off the targets while [`Renderer`](super::Renderer)
//! merely forwarded to them, which had the ownership backwards: the harness owns
//! the pipelines, the mesh store and the GPU context, so it — not a swapchain
//! handle — is what knows how to draw. #203 moved all of it onto the renderer as
//! private per-variant functions behind one public
//! [`Renderer::render`](super::Renderer::render) match, and left the target
//! types holding only the resources a frame lands in:
//!
//! - [`TextureTarget`] — an owned [`TEXTURE_TARGET_FORMAT`] texture plus its
//!   `MAP_READ` staging buffer. The common case: the headless CLI, the golden
//!   tests, both browser renderers and the video editors all read pixels back.
//! - [`SurfaceTarget`] — a live [`wgpu::Surface`] plus its configuration and the
//!   sRGB view format it is rendered through. Serves the two live-surface shells
//!   (`trd-app`, the browser canvas).
//! - [`RenderTarget`] — the closed enum over the two, and the argument
//!   [`Renderer::render`](super::Renderer::render) dispatches on. It holds
//!   *only* the discriminant: each variant already stores its own size
//!   (`config.width` versus `texture.width()`), and copying that upward would
//!   create a second source of truth that a resize could put out of date.
//!
//! Fields stay private even though there is no behaviour left. Construction is
//! fallible and GPU-dependent — dimensions are validated against
//! `max_texture_dimension_2d`, the staging buffer is padded to
//! [`wgpu::COPY_BYTES_PER_ROW_ALIGNMENT`], the sRGB view format is registered
//! and the surface configured — and public fields would let a caller assemble a
//! texture without `COPY_SRC` or a mis-padded readback buffer. Property
//! accessors ([`size`](TextureTarget::size), [`view_format`](SurfaceTarget::view_format),
//! …) return data and are fine; they are deliberately *not* unified behind a
//! trait, because two inherent impls of two accessors duplicate nothing.
//!
//! Note the picking target is deliberately **not** here: it is a second pass
//! producing ids, not a place a frame is rendered to (see `picking.rs`).

use thiserror::Error;

use super::Viewport;
use crate::visual::DrawableObject;
use crate::Camera;

/// The fixed texture-target render format. Matches the headless CLI's output
/// target so native and browser renders are byte-identical (hardware sRGB-encode
/// on store).
pub const TEXTURE_TARGET_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8UnormSrgb;

/// Errors creating a render target, or reading one back.
#[derive(Debug, Error)]
pub enum TargetError {
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

/// One pass of a layered render: a scene plus the camera it is seen through.
///
/// Layers exist because a composited frame is not one camera's scene — the video
/// editor draws the video plane through the *background* frame's calibration, then
/// the placed object through the *placement* frame's, then its selection gizmos on
/// top. Each layer keeps the accumulated color and clears depth.
#[derive(Debug, Clone, Copy)]
pub struct SceneLayer<'a> {
    /// The camera this layer is rendered through.
    pub camera: Camera,
    /// What this layer draws.
    pub scene: &'a [DrawableObject],
}

impl<'a> SceneLayer<'a> {
    /// A layer drawing `scene` through `camera`.
    pub fn new(camera: Camera, scene: &'a [DrawableObject]) -> Self {
        Self { camera, scene }
    }
}

// ---------------------------------------------------------------------------
// Texture target
// ---------------------------------------------------------------------------

/// An owned [`TEXTURE_TARGET_FORMAT`] texture plus the `MAP_READ` staging buffer
/// its contents are copied into to be read back as tightly-packed RGBA.
///
/// Allocated once for a fixed size and reused every frame;
/// [`Renderer::render`](super::Renderer::render) draws into it and
/// [`Renderer::read_pixels`](super::Renderer::read_pixels) reads it back.
pub struct TextureTarget {
    texture: wgpu::Texture,
    staging: wgpu::Buffer,
    /// The readback row stride, rounded up to
    /// [`wgpu::COPY_BYTES_PER_ROW_ALIGNMENT`]. Stored because `copy_texture_to_buffer`
    /// needs it and the unpad step has to undo exactly the same padding.
    padded_bytes_per_row: u32,
}

impl TextureTarget {
    /// Allocates the render target + readback buffer for a fixed `width` ×
    /// `height`, validating non-zero dimensions and the adapter's
    /// `max_texture_dimension_2d`.
    ///
    /// Kept as an inherent constructor (rather than only
    /// [`Renderer::create_texture_target`](super::Renderer::create_texture_target))
    /// because a shell may need a target before it has any mesh to build a
    /// renderer from. Either way the invariants above are established here, which
    /// is why the fields are private (#203).
    pub fn new(device: &wgpu::Device, width: u32, height: u32) -> Result<Self, TargetError> {
        if width == 0 || height == 0 {
            return Err(TargetError::InvalidDimensions { width, height });
        }
        let max = device.limits().max_texture_dimension_2d;
        if width > max || height > max {
            return Err(TargetError::ExceedsMaxDimension { width, height, max });
        }

        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("trd texture target"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: TEXTURE_TARGET_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });

        let unpadded = width
            .checked_mul(4)
            .ok_or(TargetError::RowOverflow { width })?;
        let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        let padded_bytes_per_row = unpadded.div_ceil(align) * align;
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("trd texture target readback"),
            size: u64::from(padded_bytes_per_row) * u64::from(height),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        Ok(Self {
            texture,
            staging,
            padded_bytes_per_row,
        })
    }

    /// The texture format a renderer's pipelines must be built for — always
    /// [`TEXTURE_TARGET_FORMAT`].
    pub fn view_format(&self) -> wgpu::TextureFormat {
        TEXTURE_TARGET_FORMAT
    }

    /// The fixed render width in pixels.
    pub fn width(&self) -> u32 {
        self.texture.width()
    }

    /// The fixed render height in pixels.
    pub fn height(&self) -> u32 {
        self.texture.height()
    }

    /// The fixed render size in pixels.
    pub fn size(&self) -> (u32, u32) {
        (self.width(), self.height())
    }

    /// The fixed render size as a [`Viewport`], for resolving a camera against
    /// the target the frame actually lands in.
    pub fn viewport(&self) -> Viewport {
        Viewport {
            width: self.width(),
            height: self.height(),
        }
    }

    /// The texture a frame is drawn into.
    pub(crate) fn texture(&self) -> &wgpu::Texture {
        &self.texture
    }

    /// The `MAP_READ` buffer the texture is copied into for readback.
    pub(crate) fn staging(&self) -> &wgpu::Buffer {
        &self.staging
    }

    /// The alignment-padded readback row stride.
    pub(crate) fn padded_bytes_per_row(&self) -> u32 {
        self.padded_bytes_per_row
    }
}

// ---------------------------------------------------------------------------
// Surface target
// ---------------------------------------------------------------------------

/// A live surface plus its configuration and the **sRGB view format** frames are
/// rendered through.
///
/// The browser's preferred canvas format is non-sRGB (e.g. `Bgra8Unorm`), so a
/// pipeline targeting it writes *linear* values with no linear→sRGB encode —
/// darker and muddier than the headless CLI's `Rgba8UnormSrgb` target. Native
/// surfaces are usually sRGB already. Rather than each shell special-casing
/// this, a surface target always renders through the surface's sRGB view
/// ([`add_srgb_suffix`](wgpu::TextureFormat::add_srgb_suffix), registered in
/// `view_formats`), so both platforms match the CLI byte-for-byte. Build the
/// front-end's renderer with [`view_format`](Self::view_format).
///
/// The shell keeps what is genuinely window-specific: device/adapter/surface
/// creation, the `present_mode` choice, and the **surface-recovery policy** it
/// applies to what [`Renderer::render`](super::Renderer::render) reports.
pub struct SurfaceTarget {
    surface: wgpu::Surface<'static>,
    config: wgpu::SurfaceConfiguration,
    /// The sRGB view format frames are rendered through (the sRGB variant of
    /// `config.format`; equal to `config.format` when it is already sRGB). The
    /// front-end's renderer pipeline must target this format.
    view_format: wgpu::TextureFormat,
}

impl SurfaceTarget {
    /// Wraps an already-created surface and its default configuration, registers
    /// the sRGB view format, and configures the surface. `config` is typically
    /// obtained from [`wgpu::Surface::get_default_config`] with the shell's
    /// chosen `present_mode` applied; the target owns it from here.
    ///
    /// Inherent rather than only
    /// [`Renderer::create_surface_target`](super::Renderer::create_surface_target)
    /// because both live-surface shells create their surface *before* the stream
    /// has delivered a mesh to build a renderer from (#203).
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

    /// The sRGB view format frames are rendered through. Build the front-end's
    /// renderer with this so its pipeline target matches the view.
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

    /// The current surface size in pixels.
    pub fn size(&self) -> (u32, u32) {
        (self.width(), self.height())
    }

    /// The current surface size as a [`Viewport`], for resolving a camera against
    /// the surface the frame actually lands in.
    pub fn viewport(&self) -> Viewport {
        Viewport {
            width: self.config.width,
            height: self.config.height,
        }
    }

    /// The surface a frame is presented on.
    pub(crate) fn surface(&self) -> &wgpu::Surface<'static> {
        &self.surface
    }

    /// The configuration the surface was last configured with.
    pub(crate) fn config(&self) -> &wgpu::SurfaceConfiguration {
        &self.config
    }

    /// Records a new size. Purely data: the caller reconfigures the surface,
    /// because *doing* something to the GPU is the renderer's job (#203).
    pub(crate) fn set_size(&mut self, width: u32, height: u32) {
        self.config.width = width;
        self.config.height = height;
    }

    /// Swaps in a freshly created surface, e.g. after the browser reports the
    /// canvas surface *lost*. The new surface must target the same
    /// canvas/window, so the configuration (and hence the view format) still
    /// applies; the caller reconfigures it.
    pub(crate) fn set_surface(&mut self, surface: wgpu::Surface<'static>) {
        self.surface = surface;
    }
}

// ---------------------------------------------------------------------------
// The closed set
// ---------------------------------------------------------------------------

/// Which kind of place a frame lands in.
///
/// A closed enum, not a trait: there are exactly two ends of the render path and
/// no front-end has ever wanted a third, so the renderer can match on them
/// exhaustively instead of dispatching through a vtable that could only ever
/// hide two arms (#203).
pub enum RenderTargetType {
    /// A live swapchain surface; a frame is presented.
    Surface(SurfaceTarget),
    /// An owned texture; a frame is submitted and can be read back.
    Texture(TextureTarget),
}

/// The render target [`Renderer::render`](super::Renderer::render) takes.
///
/// Deliberately just the discriminant. Size and view format are **not** mirrored
/// here: each variant already stores them in its own form (`config.width` versus
/// `texture.width()`), and a copy on the wrapper would be a second source of
/// truth that a resize could leave disagreeing with the attachments. The
/// accessors below delegate instead (#203).
pub struct RenderTarget {
    kind: RenderTargetType,
}

impl RenderTarget {
    /// Wraps a live surface.
    pub fn surface(target: SurfaceTarget) -> Self {
        Self {
            kind: RenderTargetType::Surface(target),
        }
    }

    /// Wraps an owned texture.
    pub fn texture(target: TextureTarget) -> Self {
        Self {
            kind: RenderTargetType::Texture(target),
        }
    }

    /// Which kind of target this is.
    pub fn kind(&self) -> &RenderTargetType {
        &self.kind
    }

    /// Which kind of target this is, mutably — what
    /// [`Renderer::render`](super::Renderer::render) matches on.
    pub fn kind_mut(&mut self) -> &mut RenderTargetType {
        &mut self.kind
    }

    /// The texture variant, or `None` for a surface. Readback
    /// ([`Renderer::read_pixels`](super::Renderer::read_pixels)) takes the
    /// concrete [`TextureTarget`], so a shell that reads pixels back projects
    /// once here rather than the renderer growing a "not readable" runtime arm.
    pub fn as_texture(&self) -> Option<&TextureTarget> {
        match &self.kind {
            RenderTargetType::Texture(target) => Some(target),
            RenderTargetType::Surface(_) => None,
        }
    }

    /// The surface variant, or `None` for a texture. Surface recovery
    /// (reconfigure / replace / resize) is surface-only, so a live-surface shell
    /// projects once here and keeps its policy typed.
    pub fn as_surface_mut(&mut self) -> Option<&mut SurfaceTarget> {
        match &mut self.kind {
            RenderTargetType::Surface(target) => Some(target),
            RenderTargetType::Texture(_) => None,
        }
    }

    /// The texture format a renderer's pipelines must be built for: the
    /// surface's sRGB view format, or [`TEXTURE_TARGET_FORMAT`].
    pub fn view_format(&self) -> wgpu::TextureFormat {
        match &self.kind {
            RenderTargetType::Surface(target) => target.view_format(),
            RenderTargetType::Texture(target) => target.view_format(),
        }
    }

    /// The current render size in pixels.
    pub fn size(&self) -> (u32, u32) {
        match &self.kind {
            RenderTargetType::Surface(target) => target.size(),
            RenderTargetType::Texture(target) => target.size(),
        }
    }

    /// The current render size as a [`Viewport`].
    pub fn viewport(&self) -> Viewport {
        match &self.kind {
            RenderTargetType::Surface(target) => target.viewport(),
            RenderTargetType::Texture(target) => target.viewport(),
        }
    }
}

impl From<SurfaceTarget> for RenderTarget {
    fn from(target: SurfaceTarget) -> Self {
        Self::surface(target)
    }
}

impl From<TextureTarget> for RenderTarget {
    fn from(target: TextureTarget) -> Self {
        Self::texture(target)
    }
}
