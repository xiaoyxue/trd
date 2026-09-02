//! Pass **attachments** — the textures a render pass draws into and throws
//! away, plus the one rule that keeps them sized to the viewport (#363).
//!
//! Three attachment sets used to open-code "recreate when the size changed":
//! the mesh pass's depth buffer, its multisampled color, and the pick pass's id
//! color + depth. They now share one brick — [`ViewportAttachment`] — over one
//! description of what to allocate, [`AttachmentSpec`], whose kind-named
//! constructors pair format with usage so a fourth call site cannot get that
//! pairing wrong.
//!
//! **`Attachment`, not `*Target`.** A `*Target` here is *a place a frame lands*
//! with a lifecycle a caller drives — [`TextureTarget`](super::TextureTarget),
//! [`PickTarget`](super::picking::PickTarget) (#203). An attachment is what a
//! single pass draws into: the mesh pass's two are `RENDER_ATTACHMENT` only and
//! nothing ever reads them (the color is resolved, the depth discarded), and
//! even the pick id — which does carry `COPY_SRC` for its one-texel read-back —
//! is scratch the pass rebuilds whenever the viewport moves.

use super::{Viewport, DEPTH_FORMAT, PICK_FORMAT};

/// One allocated attachment. Not optional — if you hold one, it exists.
pub(crate) struct Attachment {
    texture: wgpu::Texture,
    view: wgpu::TextureView,
    width: u32,
    height: u32,
}

impl Attachment {
    /// Allocates `spec` at `width` × `height`, which the caller has already
    /// clamped to ≥ 1.
    fn new(device: &wgpu::Device, spec: &AttachmentSpec, width: u32, height: u32) -> Self {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some(spec.label),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: spec.sample_count,
            dimension: wgpu::TextureDimension::D2,
            format: spec.format,
            usage: spec.usage,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        Self {
            texture,
            view,
            width,
            height,
        }
    }

    /// The view a render pass binds this attachment through.
    pub(crate) fn view(&self) -> &wgpu::TextureView {
        &self.view
    }

    /// The texture itself — only a `COPY_SRC` attachment has a reason to want
    /// it, to copy out of.
    pub(crate) fn texture(&self) -> &wgpu::Texture {
        &self.texture
    }

    /// The allocated dimensions (already clamped to ≥ 1).
    pub(crate) fn size(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    /// Whether `(x, y)` is inside the attachment — a click outside it hits
    /// nothing.
    pub(crate) fn contains(&self, x: u32, y: u32) -> bool {
        x < self.width && y < self.height
    }
}

/// How to allocate an [`Attachment`]. The kind-named constructors pair format
/// with usage, so that pairing is stated once instead of at each call site.
pub(crate) struct AttachmentSpec {
    label: &'static str,
    format: wgpu::TextureFormat,
    sample_count: u32,
    usage: wgpu::TextureUsages,
}

impl AttachmentSpec {
    /// The mesh pass's multisampled color: resolved into the caller's
    /// single-sample view and never sampled or copied, hence
    /// `RENDER_ATTACHMENT` alone. Named for the only kind trd owns — with MSAA
    /// off the pass renders straight into the caller's view and allocates
    /// nothing.
    pub(crate) fn msaa_color(format: wgpu::TextureFormat, sample_count: u32) -> Self {
        Self {
            label: "trd msaa color texture",
            format,
            sample_count,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        }
    }

    /// A depth buffer at `sample_count`. A pass requires its depth and color
    /// attachments to agree on that count, which is why [`MeshAttachments`]
    /// builds both specs from one argument.
    pub(crate) fn depth(sample_count: u32) -> Self {
        Self {
            label: "trd depth texture",
            format: DEPTH_FORMAT,
            sample_count,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        }
    }

    /// The pick pass's id color: single-sampled (ids must never be averaged at
    /// an edge), linear [`PICK_FORMAT`] (the bytes round-trip exactly) and
    /// `COPY_SRC`, because one texel is read back.
    pub(crate) fn id_color() -> Self {
        Self {
            label: "trd pick target",
            format: PICK_FORMAT,
            sample_count: 1,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        }
    }
}

/// An [`Attachment`] that tracks a viewport and rebuilds itself — the one copy
/// of a rule three sets used to keep by hand.
pub(crate) struct ViewportAttachment {
    spec: AttachmentSpec,
    current: Option<Attachment>,
}

impl ViewportAttachment {
    /// Nothing is allocated until the first [`ensure`](Self::ensure), so a
    /// front-end that never draws the pass never pays for it.
    pub(crate) fn new(spec: AttachmentSpec) -> Self {
        Self {
            spec,
            current: None,
        }
    }

    /// Matches the attachment to `width` × `height` (each clamped to ≥ 1),
    /// recreating it only when the size changed. Returns a reference, so no
    /// call site unwraps.
    ///
    /// The replacement is built *before* the old one drops — as the assignments
    /// it replaces did — so a failed allocation leaves the previous attachment
    /// in place rather than nothing at all.
    pub(crate) fn ensure(&mut self, device: &wgpu::Device, width: u32, height: u32) -> &Attachment {
        let width = width.max(1);
        let height = height.max(1);
        let current = self
            .current
            .get_or_insert_with(|| Attachment::new(device, &self.spec, width, height));
        if current.width != width || current.height != height {
            *current = Attachment::new(device, &self.spec, width, height);
        }
        current
    }

    /// The allocated attachment, or `None` before the first
    /// [`ensure`](Self::ensure).
    pub(crate) fn current(&self) -> Option<&Attachment> {
        self.current.as_ref()
    }
}

/// The mesh pass's attachment set: the multisampled color it *may* own, and the
/// depth buffer it always has.
///
/// One owner, so the format and sample count are stated once instead of by
/// three doc comments saying they must agree (#221 §3): both specs are built
/// from the one [`new`](Self::new) call and never mutated, so nothing can move
/// the depth attachment off the color attachment's sample count. Matching the
/// *pipelines* is still the caller's job — `Renderer` builds them from the same
/// two arguments.
///
/// The two meanings the old `MsaaColor::target` carried in a single `Option`
/// are now separate: [`color`](Self::color) being `None` means **MSAA is off**,
/// decided at construction from `sample_count`, while "not allocated yet" is the
/// `Option` inside [`ViewportAttachment`].
pub(crate) struct MeshAttachments {
    color: Option<ViewportAttachment>,
    depth: ViewportAttachment,
}

impl MeshAttachments {
    /// The attachment set for pipelines built with `format` and `sample_count`.
    /// At `1` the pass renders straight into the caller's single-sample view, so
    /// there is no color attachment to own.
    pub(crate) fn new(format: wgpu::TextureFormat, sample_count: u32) -> Self {
        Self {
            color: (sample_count > 1)
                .then(|| ViewportAttachment::new(AttachmentSpec::msaa_color(format, sample_count))),
            depth: ViewportAttachment::new(AttachmentSpec::depth(sample_count)),
        }
    }

    /// Sizes both attachments to `viewport` and hands back this frame's views.
    /// Sizing them together is what makes the shared sample count structural.
    pub(crate) fn resize(&mut self, device: &wgpu::Device, viewport: Viewport) -> PassViews<'_> {
        let Viewport { width, height } = viewport;
        PassViews {
            color: self
                .color
                .as_mut()
                .map(|color| color.ensure(device, width, height)),
            depth: self.depth.ensure(device, width, height),
        }
    }
}

/// One frame's mesh-pass attachments, borrowed for the
/// `begin_render_pass` call.
///
/// `depth` is not an `Option`: a pass cannot begin without one, so it is not a
/// runtime question — which is what retires the "`prepare_frame` sized it"
/// `expect` the renderer used to carry.
pub(crate) struct PassViews<'a> {
    color: Option<&'a Attachment>,
    depth: &'a Attachment,
}

impl<'a> PassViews<'a> {
    /// This frame's color attachment. With MSAA the pass renders into the
    /// multisampled attachment and **resolves** into the caller's `view`, so
    /// every front-end gets multisampled mesh/arrowhead edges; without it the
    /// pass renders straight into `view`. `load` is what happens to the existing
    /// contents — `Load` for a composited layer, `Clear` for the first.
    pub(crate) fn color_attachment(
        &self,
        view: &'a wgpu::TextureView,
        load: wgpu::LoadOp<wgpu::Color>,
    ) -> wgpu::RenderPassColorAttachment<'a> {
        let (view, resolve_target) = match self.color {
            Some(color) => (color.view(), Some(view)),
            None => (view, None),
        };
        wgpu::RenderPassColorAttachment {
            view,
            depth_slice: None,
            resolve_target,
            ops: wgpu::Operations {
                load,
                store: wgpu::StoreOp::Store,
            },
        }
    }

    /// The depth attachment's view; its load/store ops are the pass's business,
    /// not the attachment's.
    pub(crate) fn depth_view(&self) -> &'a wgpu::TextureView {
        self.depth.view()
    }
}
