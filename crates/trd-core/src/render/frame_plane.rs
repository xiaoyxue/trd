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

/// The reused GPU frame texture plus its bind group and fit uniform. Recreated
/// only when the frame resolution changes, so streaming a fixed-resolution video
/// allocates once and every later frame is a plain `queue.write_texture`.
struct FrameTextureGpu {
    texture: wgpu::Texture,
    bind_group: wgpu::BindGroup,
    /// `vec4` fit uniform (`uv_scale.xy` + padding), rewritten each frame from the
    /// [`FrameFit`] + texture/viewport aspect.
    fit_uniform: wgpu::Buffer,
    width: u32,
    height: u32,
}

/// The background frame-plane subsystem: the fullscreen pipeline, its bind-group
/// layout, the shared sampler, and the currently-bound frame texture (`None`
/// until the first [`upload_rgba`](Self::upload_rgba)). While nothing is bound
/// every method is a no-op, so a scene asking for a frame plane simply renders
/// nothing.
pub(super) struct FramePlane {
    pipeline: wgpu::RenderPipeline,
    layout: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
    texture: Option<FrameTextureGpu>,
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
            texture: None,
        }
    }

    /// Whether a frame texture is currently bound (so a scene asking for a frame
    /// plane would render one).
    pub(super) fn is_bound(&self) -> bool {
        self.texture.is_some()
    }

    /// Uploads `rgba` (tightly-packed, row-major `height`×`width`×4) as the
    /// background frame texture (formerly `SceneRenderer::update_frame_texture_rgba`).
    /// The GPU texture is **reused** across frames — recreated only when the
    /// dimensions change, so streaming a fixed-resolution video allocates once and
    /// every later frame is a plain `queue.write_texture`. The texture is
    /// `Rgba8UnormSrgb` (linearized on sample) and carries **no mipmaps** (a
    /// near-fullscreen background samples ~1:1).
    ///
    /// Panics if `rgba.len() != width * height * 4` or either dimension is zero.
    pub(super) fn upload_rgba(&mut self, gpu: &GpuContext, rgba: &[u8], width: u32, height: u32) {
        let (device, queue) = (&gpu.device, &gpu.queue);
        assert!(
            width > 0 && height > 0,
            "frame texture dimensions must be non-zero"
        );
        assert_eq!(
            rgba.len(),
            width as usize * height as usize * 4,
            "frame texture rgba length must be width*height*4"
        );
        self.ensure_texture(device, width, height);

        let ft = self.texture.as_ref().expect("frame texture set above");
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &ft.texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
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

    /// Copies a frame the delivery surface kept on the GPU **GPU→GPU**, without
    /// its pixels ever entering CPU memory (#229).
    ///
    /// The counterpart of [`upload_rgba`](Self::upload_rgba) for a source that
    /// is already decoded into GPU memory: the RGBA route would drag it back
    /// down at *source* resolution — ~99 MB per frame for 4K — only to push it
    /// straight back up.
    ///
    /// Two decisions worth recording:
    ///
    /// * **The destination stays ours.** This allocates the texture and holds
    ///   the format/usage invariants; the frame supplies only the copy. That
    ///   split is not a preference — `copy_external_image_to_texture` is
    ///   `#[cfg(web)]` in wgpu, so the copy *cannot* be compiled here, while
    ///   everything around it can. See [`ExternalFrame`](crate::ExternalFrame).
    /// * **The frame is borrowed, not consumed.** WebGPU snapshots the source
    ///   during the call, so an implementor *may* release it immediately — but
    ///   trd's browser frame is held until superseded, because any UI change
    ///   repaints and re-renders the same frame.
    pub(super) fn copy_external(&mut self, gpu: &GpuContext, frame: &dyn crate::ExternalFrame) {
        let (width, height) = frame.size();
        assert!(
            width > 0 && height > 0,
            "frame texture dimensions must be non-zero"
        );
        self.ensure_texture(&gpu.device, width, height);
        let ft = self.texture.as_ref().expect("frame texture set above");
        frame.copy_into(&gpu.queue, &ft.texture);
    }

    /// (Re)allocates the frame texture when the resolution changes, and leaves it
    /// alone otherwise — so streaming a fixed-resolution video allocates once.
    fn ensure_texture(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        let needs_new = self
            .texture
            .as_ref()
            .is_none_or(|ft| ft.width != width || ft.height != height);
        if !needs_new {
            return;
        }
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("trd frame texture"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
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
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
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
        self.texture = Some(FrameTextureGpu {
            texture,
            bind_group,
            fit_uniform,
            width,
            height,
        });
    }

    /// Computes and uploads the centered UV-fit scale that realizes `fit` on
    /// `viewport` for the bound frame texture. No-op if no texture is bound.
    pub(super) fn write_fit(&self, queue: &wgpu::Queue, fit: FrameFit, viewport: Viewport) {
        if let Some(ft) = self.texture.as_ref() {
            let scale =
                frame_fit_uv_scale(fit, ft.width, ft.height, viewport.width, viewport.height);
            let fit_data: [f32; 4] = [scale[0], scale[1], 0.0, 0.0];
            queue.write_buffer(&ft.fit_uniform, 0, bytemuck::cast_slice(&fit_data));
        }
    }

    /// Records the fullscreen frame-plane draw (its own pipeline + group-0 bind,
    /// depth-write off) so the mesh scene composites on top. No-op if no texture
    /// is bound.
    pub(super) fn draw(&self, pass: &mut wgpu::RenderPass) {
        if let Some(ft) = self.texture.as_ref() {
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &ft.bind_group, &[]);
            pass.draw(0..3, 0..1);
        }
    }
}
