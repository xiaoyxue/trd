//! The mesh **albedo** texture sampled by [`RenderMode::Textured`] draws (#20).
//!
//! Splits the two texture subsystems apart: this is the per-mesh albedo that
//! arrives inside the Arrow scene channel and skins the meshes, as opposed to
//! the background [`FramePlane`](super::FramePlane) frame texture. Both use the
//! same texture+sampler bind pattern but update at different rates, so keeping
//! them as separate types makes which-is-which obvious at a glance.

use super::upload_texture;
use super::GpuContext;
use crate::texture::{ConstantTexture, Texture};

/// The albedo texture bound as group 1 by the textured mesh pipeline (#20).
///
/// Owns the group-1 bind-group layout and the GPU bind group. The bind group is
/// built **eagerly** — at construction with the 1×1 white identity albedo, and
/// again on every [`set`](Self::set) — so `bind_group` is always valid and the
/// renderer never has to defer uploads to `encode` (#180).
pub(super) struct BoundTexture {
    layout: wgpu::BindGroupLayout,
    bind_group: wgpu::BindGroup,
}

impl BoundTexture {
    /// Constructs a `BoundTexture` holding the 1×1 white identity albedo,
    /// uploaded immediately, reusing a **shared** group-1 layout (a cheap wgpu
    /// handle clone) so many per-mesh albedo textures stay compatible with the
    /// one textured/PBR pipeline layout.
    pub(super) fn with_layout(gpu: &GpuContext, layout: wgpu::BindGroupLayout) -> Self {
        // `ConstantTexture::white()` *is* "the identity albedo" — the kind whose
        // documented purpose is this default (#247 T5) — so the 1×1 image is
        // built by the texture abstraction rather than open-coded here.
        let image = ConstantTexture::white().to_image();
        let bind_group = upload_texture(gpu, &layout, &image);
        Self { layout, bind_group }
    }

    /// Replaces the source image, uploading it immediately.
    pub(super) fn set(&mut self, gpu: &GpuContext, texture: &dyn Texture) {
        let image = texture.to_image();
        self.bind_group = upload_texture(gpu, &self.layout, &image);
    }

    /// The group-1 bind group. Always valid: it is uploaded at construction and
    /// replaced on every [`set`](Self::set).
    pub(super) fn bind_group(&self) -> &wgpu::BindGroup {
        &self.bind_group
    }
}
