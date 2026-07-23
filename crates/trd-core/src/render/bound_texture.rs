//! The mesh **albedo** texture sampled by [`RenderMode::Textured`] draws (#20).
//!
//! Splits the two texture subsystems apart: this is the per-mesh albedo that
//! arrives inside the Arrow scene channel and skins the meshes, as opposed to
//! the background [`FramePlane`](super::FramePlane) frame texture. Both use the
//! same texture+sampler bind pattern but update at different rates, so keeping
//! them as separate types makes which-is-which obvious at a glance.

use super::{create_texture_bind_group_layout, upload_texture};
use crate::texture::{ImageData, Texture};

/// The albedo texture bound as group 1 by the textured mesh pipeline (#20).
///
/// Owns the group-1 bind-group layout, the CPU image to upload, and the GPU
/// bind group. The bind group is (re)built **lazily** on the next
/// [`ensure_uploaded`](Self::ensure_uploaded) — the only place a GPU queue is
/// available — so [`set`](Self::set) can swap the image cheaply between frames.
/// Until a real texture is set the image is 1×1 white (the identity albedo).
pub(super) struct BoundTexture {
    layout: wgpu::BindGroupLayout,
    bind_group: Option<wgpu::BindGroup>,
    image: ImageData,
}

impl BoundTexture {
    /// Constructs a `BoundTexture` seeded with the 1×1 white identity albedo. No
    /// GPU upload happens yet; the first [`ensure_uploaded`](Self::ensure_uploaded)
    /// builds the bind group.
    pub(super) fn new(device: &wgpu::Device) -> Self {
        Self {
            layout: create_texture_bind_group_layout(device),
            bind_group: None,
            image: ImageData {
                width: 1,
                height: 1,
                rgba: vec![255, 255, 255, 255],
            },
        }
    }

    /// The group-1 bind-group layout (also fed to the textured pipeline layout so
    /// the pipeline and this bind group agree).
    pub(super) fn layout(&self) -> &wgpu::BindGroupLayout {
        &self.layout
    }

    /// Replaces the source image; the bind group is rebuilt on the next
    /// [`ensure_uploaded`](Self::ensure_uploaded).
    pub(super) fn set(&mut self, texture: &dyn Texture) {
        self.image = texture.to_image();
        self.bind_group = None;
    }

    /// Uploads the current image if it has not been uploaded since the last
    /// [`set`](Self::set), returning the group-1 bind group to bind.
    pub(super) fn ensure_uploaded(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> &wgpu::BindGroup {
        if self.bind_group.is_none() {
            self.bind_group = Some(upload_texture(device, queue, &self.layout, &self.image));
        }
        self.bind_group.as_ref().expect("uploaded above")
    }
}
