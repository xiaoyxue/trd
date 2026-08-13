//! The HDR environment subsystem: the bound probe **and** the pipeline that
//! draws it as the background sky (#221 §5).
//!
//! trd has two background subsystems, and this one used to be written
//! asymmetrically to the other: [`FramePlane`](super::frame_plane::FramePlane)
//! bundles its whole stack into one type, while the environment was two
//! renderer fields across two files — `BoundEnv` (the probe, in `ibl.rs`) and
//! `EnvBackground` (the pipeline drawing it, in `env_background.rs`). The
//! second owned **no** texture: it could not be built without the first's
//! bind-group layout, nor draw without its bind group, so `encode` threaded the
//! join by hand. One subsystem, one type, mirroring `FramePlane`.
//!
//! [`bind_group`](Environment::bind_group) stays exposed all the same: PBR
//! draws sample the probe at group 2, so it is not *only* a background resource.
//!
//! The device-free half — the decoded [`EnvMapData`], the per-object
//! [`ImageBasedLighting`], and the CPU precompute this file uploads — is in
//! `env_map.rs`.

use super::env_map::{
    build_irradiance_map, f32_to_f16_bits, fit_environment, integrate_brdf,
    prefilter_environment_level,
};
use super::{create_env_bind_group_layout, overlay_depth_stencil, EnvMapData, GpuContext, Tonemap};
use crate::Camera;

/// The environment background's group-0 uniform: the inverse `P·V` that turns a
/// fullscreen triangle into view rays, the camera position, and the probe's
/// rotation / exposure / blur / tone map.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct EnvBackgroundUniform {
    inverse_view_proj: [f32; 16],
    camera_pos: [f32; 4],
    params: [f32; 4],
}

/// The scene's environment-background settings for one frame, transcribed from
/// [`EnvironmentBackground`](crate::EnvironmentBackground).
///
/// The scene carries the *settings* and the renderer the *resource*: that split
/// is why the probe stays here while `Background::environment` stays on the
/// device-free [`Scene`](crate::Scene).
pub(super) struct EnvBackgroundSettings {
    pub rotation: f32,
    pub exposure: f32,
    pub blur: f32,
    pub tonemap: Tonemap,
}

/// The HDR environment: the bound probe (sampled by `Shaded` draws at group 2)
/// plus the pipeline that draws it as the background sky.
///
/// The probe is uploaded **eagerly**: the constructor binds a 1×1 black
/// fallback and [`set`](Self::set) replaces it immediately, so `encode` never
/// has to upload (#180). [`has_env`](Self::has_env) still reports whether a
/// *real* probe was supplied, which is what the PBR uniform keys reflections
/// off.
pub(super) struct Environment {
    layout: wgpu::BindGroupLayout,
    bind_group: wgpu::BindGroup,
    has_env: bool,
    background_pipeline: wgpu::RenderPipeline,
    background_uniform: wgpu::Buffer,
    background_bind_group: wgpu::BindGroup,
}

impl Environment {
    /// Binds the black fallback probe and builds the background pipeline for
    /// `format`/`sample_count`. The probe's bind-group layout is the pipeline's
    /// group 1, which is why one constructor builds both.
    pub(super) fn new(gpu: &GpuContext, format: wgpu::TextureFormat, sample_count: u32) -> Self {
        let device = &gpu.device;
        let layout = create_env_bind_group_layout(device);
        // 1×1 black: no reflection until a probe is set.
        let fallback = EnvMapData {
            width: 1,
            height: 1,
            rgba: vec![0.0, 0.0, 0.0, 1.0],
        };
        let bind_group = upload_env_texture(gpu, &layout, &fallback);

        let uniform_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("trd environment background uniform layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: wgpu::BufferSize::new(
                        std::mem::size_of::<EnvBackgroundUniform>() as u64,
                    ),
                },
                count: None,
            }],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("trd environment background pipeline layout"),
            bind_group_layouts: &[Some(&uniform_layout), Some(&layout)],
            immediate_size: 0,
        });
        let shader =
            device.create_shader_module(wgpu::include_wgsl!("../shader/env_background.wgsl"));
        let background_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("trd environment background pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(format.into())],
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: Some(overlay_depth_stencil()),
            multisample: wgpu::MultisampleState {
                count: sample_count,
                ..Default::default()
            },
            multiview_mask: None,
            cache: None,
        });
        let background_uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("trd environment background uniform"),
            size: std::mem::size_of::<EnvBackgroundUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let background_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("trd environment background bind group"),
            layout: &uniform_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: background_uniform.as_entire_binding(),
            }],
        });

        Self {
            layout,
            bind_group,
            has_env: false,
            background_pipeline,
            background_uniform,
            background_bind_group,
        }
    }

    /// Replaces the bound probe, uploading `data` immediately.
    pub(super) fn set(&mut self, gpu: &GpuContext, data: EnvMapData) {
        self.bind_group = upload_env_texture(gpu, &self.layout, &data);
        self.has_env = true;
    }

    /// Whether a *real* probe was supplied (the fallback does not count).
    pub(super) fn has_env(&self) -> bool {
        self.has_env
    }

    /// The probe's bind group. Still exposed because PBR draws bind it at
    /// group 2 — the probe is not only a background resource.
    pub(super) fn bind_group(&self) -> &wgpu::BindGroup {
        &self.bind_group
    }

    /// The probe's bind-group layout — the PBR pipeline's group 2, so the scene
    /// pipelines are built against the same layout this type binds.
    pub(super) fn layout(&self) -> &wgpu::BindGroupLayout {
        &self.layout
    }

    /// Rewrites the background uniform for this frame's `camera` and `settings`.
    pub(super) fn write_background(
        &self,
        queue: &wgpu::Queue,
        camera: Camera,
        settings: EnvBackgroundSettings,
    ) {
        let inverse_view_proj = camera.view_projection().matrix().inverse().to_cols_array();
        let position = camera.position();
        let uniform = EnvBackgroundUniform {
            inverse_view_proj,
            camera_pos: [position[0], position[1], position[2], 1.0],
            params: [
                settings.rotation,
                settings.exposure,
                settings.blur.clamp(0.0, 1.0),
                settings.tonemap.to_uniform(),
            ],
        };
        queue.write_buffer(&self.background_uniform, 0, bytemuck::bytes_of(&uniform));
    }

    /// Draws the probe as a fullscreen background triangle. The probe bind
    /// group is bound from this type's own field — the join `encode` used to
    /// thread by hand is now internal.
    pub(super) fn draw_background<'pass>(&'pass self, pass: &mut wgpu::RenderPass<'pass>) {
        pass.set_pipeline(&self.background_pipeline);
        pass.set_bind_group(0, &self.background_bind_group, &[]);
        pass.set_bind_group(1, &self.bind_group, &[]);
        pass.draw(0..3, 0..1);
    }
}
/// Uploads `env` as the probe bind group: a roughness-prefiltered mip chain,
/// the split-sum BRDF LUT, and the diffuse irradiance map — all half-float,
/// all precomputed by `env_map.rs`.
fn upload_env_texture(
    gpu: &GpuContext,
    layout: &wgpu::BindGroupLayout,
    env: &EnvMapData,
) -> wgpu::BindGroup {
    let (device, queue) = (&gpu.device, &gpu.queue);
    let (width, height, base_rgba) = fit_environment(env, 512);
    let size = wgpu::Extent3d {
        width,
        height,
        depth_or_array_layers: 1,
    };
    let mip_level_count = 1 + width.max(height).ilog2();
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("trd env texture"),
        size,
        mip_level_count,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba16Float,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    let mut level_width = width;
    let mut level_height = height;
    for mip_level in 0..mip_level_count {
        let roughness = mip_level as f32 / (mip_level_count - 1).max(1) as f32;
        let level_rgba = if mip_level == 0 {
            base_rgba.clone()
        } else {
            prefilter_environment_level(
                width,
                height,
                &base_rgba,
                level_width,
                level_height,
                roughness,
                32,
            )
        };
        let half: Vec<u16> = level_rgba
            .iter()
            .map(|&c| f32_to_f16_bits(c.clamp(0.0, 65504.0)))
            .collect();
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            bytemuck::cast_slice(&half),
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(level_width * 8),
                rows_per_image: Some(level_height),
            },
            wgpu::Extent3d {
                width: level_width,
                height: level_height,
                depth_or_array_layers: 1,
            },
        );
        if mip_level + 1 < mip_level_count {
            level_width = (level_width / 2).max(1);
            level_height = (level_height / 2).max(1);
        }
    }
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("trd env sampler"),
        address_mode_u: wgpu::AddressMode::Repeat,
        address_mode_v: wgpu::AddressMode::ClampToEdge,
        address_mode_w: wgpu::AddressMode::ClampToEdge,
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        mipmap_filter: wgpu::MipmapFilterMode::Linear,
        ..Default::default()
    });
    let (brdf_view, brdf_sampler) = upload_brdf_lut(gpu);
    let irradiance = build_irradiance_map(width, height, &base_rgba, 64, 32, 32);
    let irradiance_view = upload_rgba16f_view(gpu, "trd diffuse irradiance", 64, 32, &irradiance);
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("trd env bind group"),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&view),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Sampler(&sampler),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::TextureView(&brdf_view),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: wgpu::BindingResource::Sampler(&brdf_sampler),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: wgpu::BindingResource::TextureView(&irradiance_view),
            },
        ],
    })
}

fn upload_rgba16f_view(
    gpu: &GpuContext,
    label: &str,
    width: u32,
    height: u32,
    rgba: &[f32],
) -> wgpu::TextureView {
    let (device, queue) = (&gpu.device, &gpu.queue);
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba16Float,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    let half: Vec<u16> = rgba
        .iter()
        .map(|&value| f32_to_f16_bits(value.clamp(0.0, 65504.0)))
        .collect();
    queue.write_texture(
        texture.as_image_copy(),
        bytemuck::cast_slice(&half),
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(width * 8),
            rows_per_image: Some(height),
        },
        wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
    );
    texture.create_view(&wgpu::TextureViewDescriptor::default())
}

fn upload_brdf_lut(gpu: &GpuContext) -> (wgpu::TextureView, wgpu::Sampler) {
    let (device, queue) = (&gpu.device, &gpu.queue);
    const SIZE: u32 = 128;
    const SAMPLES: u32 = 64;
    let mut rgba = Vec::with_capacity((SIZE * SIZE * 4) as usize);
    for y in 0..SIZE {
        let roughness = (y as f32 + 0.5) / SIZE as f32;
        for x in 0..SIZE {
            let n_dot_v = (x as f32 + 0.5) / SIZE as f32;
            let (a, b) = integrate_brdf(n_dot_v, roughness, SAMPLES);
            rgba.extend([a, b, 0.0, 1.0]);
        }
    }
    let half: Vec<u16> = rgba.into_iter().map(f32_to_f16_bits).collect();
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("trd BRDF integration LUT"),
        size: wgpu::Extent3d {
            width: SIZE,
            height: SIZE,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba16Float,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    queue.write_texture(
        texture.as_image_copy(),
        bytemuck::cast_slice(&half),
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(SIZE * 8),
            rows_per_image: Some(SIZE),
        },
        wgpu::Extent3d {
            width: SIZE,
            height: SIZE,
            depth_or_array_layers: 1,
        },
    );
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("trd BRDF LUT sampler"),
        address_mode_u: wgpu::AddressMode::ClampToEdge,
        address_mode_v: wgpu::AddressMode::ClampToEdge,
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        ..Default::default()
    });
    (view, sampler)
}
