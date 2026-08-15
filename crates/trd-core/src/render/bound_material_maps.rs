use super::GpuContext;
use crate::texture::{ConstantTexture, ImageData, Texture};

/// Linear-data textures used by the glTF metallic-roughness and normal inputs.
pub(super) struct BoundMaterialMaps {
    layout: wgpu::BindGroupLayout,
    /// Always valid: built at construction from the neutral defaults and rebuilt
    /// by each setter, so the renderer never uploads during `encode` (#180).
    bind_group: wgpu::BindGroup,
    metallic_roughness: ImageData,
    normal: ImageData,
}

impl BoundMaterialMaps {
    pub(super) fn create_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("trd PBR material maps layout"),
            entries: &[
                texture_entry(0),
                texture_entry(1),
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        })
    }

    pub(super) fn with_layout(gpu: &GpuContext, layout: wgpu::BindGroupLayout) -> Self {
        // Neutral factors: G roughness=1, B metallic=1; neutral tangent-space
        // normal. Uploaded immediately so `bind_group` is valid from the start.
        //
        // Through `ConstantTexture` — the texture kind whose documented job is
        // exactly this, "a constant map … the default when a mesh is drawn
        // textured but no texture stream is bound" — rather than a private 1×1
        // `ImageData` builder next to it (#247 T5). The sRGB-vs-linear choice
        // still lives at upload, where `upload_linear_view` makes it.
        let metallic_roughness = ConstantTexture::new([0, 255, 255, 255]).to_image();
        let normal = ConstantTexture::new([128, 128, 255, 255]).to_image();
        let bind_group = build_bind_group(gpu, &layout, &metallic_roughness, &normal);
        Self {
            layout,
            bind_group,
            metallic_roughness,
            normal,
        }
    }

    pub(super) fn set_metallic_roughness(&mut self, gpu: &GpuContext, texture: &dyn Texture) {
        self.metallic_roughness = texture.to_image();
        self.upload(gpu);
    }

    pub(super) fn set_normal(&mut self, gpu: &GpuContext, texture: &dyn Texture) {
        self.normal = texture.to_image();
        self.upload(gpu);
    }

    fn upload(&mut self, gpu: &GpuContext) {
        self.bind_group =
            build_bind_group(gpu, &self.layout, &self.metallic_roughness, &self.normal);
    }

    pub(super) fn bind_group(&self) -> &wgpu::BindGroup {
        &self.bind_group
    }
}

/// Uploads both maps and builds the group-3 bind group.
fn build_bind_group(
    gpu: &GpuContext,
    layout: &wgpu::BindGroupLayout,
    mr_image: &ImageData,
    normal_image: &ImageData,
) -> wgpu::BindGroup {
    let device = &gpu.device;
    let metallic_roughness = upload_linear_view(gpu, "trd metallic-roughness", mr_image, false);
    let normal = upload_linear_view(gpu, "trd normal map", normal_image, true);
    let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("trd PBR material map sampler"),
        address_mode_u: wgpu::AddressMode::Repeat,
        address_mode_v: wgpu::AddressMode::Repeat,
        address_mode_w: wgpu::AddressMode::ClampToEdge,
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        mipmap_filter: wgpu::MipmapFilterMode::Linear,
        ..Default::default()
    });
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("trd PBR material maps bind group"),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&metallic_roughness),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(&normal),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::Sampler(&sampler),
            },
        ],
    })
}

fn texture_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable: true },
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false,
        },
        count: None,
    }
}

fn upload_linear_view(
    gpu: &GpuContext,
    label: &str,
    image: &ImageData,
    normal_map: bool,
) -> wgpu::TextureView {
    let (device, queue) = (&gpu.device, &gpu.queue);
    let size = wgpu::Extent3d {
        width: image.width,
        height: image.height,
        depth_or_array_layers: 1,
    };
    let mip_level_count = 1 + image.width.max(image.height).ilog2();
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size,
        mip_level_count,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    let mut width = image.width;
    let mut height = image.height;
    let mut rgba = image.rgba.clone();
    for mip_level in 0..mip_level_count {
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &rgba,
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
        if mip_level + 1 < mip_level_count {
            rgba = downsample_linear(width, height, &rgba, normal_map);
            width = (width / 2).max(1);
            height = (height / 2).max(1);
        }
    }
    texture.create_view(&wgpu::TextureViewDescriptor::default())
}

fn downsample_linear(width: u32, height: u32, rgba: &[u8], normal_map: bool) -> Vec<u8> {
    let next_width = (width / 2).max(1);
    let next_height = (height / 2).max(1);
    let mut out = vec![0; (next_width * next_height * 4) as usize];
    for y in 0..next_height {
        for x in 0..next_width {
            let mut sum = [0.0; 4];
            for (sx, sy) in [
                ((x * 2) % width, (y * 2).min(height - 1)),
                ((x * 2 + 1) % width, (y * 2).min(height - 1)),
                ((x * 2) % width, (y * 2 + 1).min(height - 1)),
                ((x * 2 + 1) % width, (y * 2 + 1).min(height - 1)),
            ] {
                let source = ((sy * width + sx) * 4) as usize;
                for channel in 0..4 {
                    sum[channel] += f32::from(rgba[source + channel]);
                }
            }
            let target = ((y * next_width + x) * 4) as usize;
            if normal_map {
                let mut normal = [
                    sum[0] * (0.5 / 255.0) - 1.0,
                    sum[1] * (0.5 / 255.0) - 1.0,
                    sum[2] * (0.5 / 255.0) - 1.0,
                ];
                let length =
                    (normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2])
                        .sqrt()
                        .max(1e-6);
                for channel in 0..3 {
                    normal[channel] /= length;
                    out[target + channel] = ((normal[channel] * 0.5 + 0.5) * 255.0 + 0.5) as u8;
                }
                out[target + 3] = 255;
            } else {
                for channel in 0..4 {
                    out[target + channel] = (sum[channel] * 0.25 + 0.5) as u8;
                }
            }
        }
    }
    out
}
