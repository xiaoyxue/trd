use super::GpuContext;
use crate::texture::{ConstantTexture, ImageData, Texture};

/// Linear-data textures used by the glTF metallic-roughness and normal inputs.
///
/// Exclusively owns both allocations: bindings may borrow them, but no other
/// material may share them because replacement explicitly destroys the old map.
pub(super) struct BoundMaterialMaps {
    layout: wgpu::BindGroupLayout,
    /// Always valid: built at construction from the neutral defaults and rebuilt
    /// by each setter, so the renderer never uploads during `encode` (#180).
    bind_group: wgpu::BindGroup,
    metallic_roughness: LinearMap,
    normal: LinearMap,
    sampler: wgpu::Sampler,
}

struct LinearMap {
    texture: wgpu::Texture,
    view: wgpu::TextureView,
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
        let metallic_roughness = upload_linear_view(
            gpu,
            "trd metallic-roughness",
            &ConstantTexture::new([0, 255, 255, 255]).to_image(),
            false,
        );
        let normal = upload_linear_view(
            gpu,
            "trd normal map",
            &ConstantTexture::new([128, 128, 255, 255]).to_image(),
            true,
        );
        let sampler = gpu.device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("trd PBR material map sampler"),
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Linear,
            ..Default::default()
        });
        let bind_group = build_bind_group(
            &gpu.device,
            &layout,
            &metallic_roughness.view,
            &normal.view,
            &sampler,
        );
        Self {
            layout,
            bind_group,
            metallic_roughness,
            normal,
            sampler,
        }
    }

    /// Frees both exclusively owned maps, including their bound views.
    pub(super) fn destroy(&self) {
        self.metallic_roughness.texture.destroy();
        self.normal.texture.destroy();
    }

    pub(super) fn set_metallic_roughness(&mut self, gpu: &GpuContext, texture: &dyn Texture) {
        let metallic_roughness =
            upload_linear_view(gpu, "trd metallic-roughness", &texture.to_image(), false);
        let bind_group = build_bind_group(
            &gpu.device,
            &self.layout,
            &metallic_roughness.view,
            &self.normal.view,
            &self.sampler,
        );
        let previous = std::mem::replace(&mut self.metallic_roughness, metallic_roughness);
        self.bind_group = bind_group;
        previous.texture.destroy();
    }

    pub(super) fn set_normal(&mut self, gpu: &GpuContext, texture: &dyn Texture) {
        let normal = upload_linear_view(gpu, "trd normal map", &texture.to_image(), true);
        let bind_group = build_bind_group(
            &gpu.device,
            &self.layout,
            &self.metallic_roughness.view,
            &normal.view,
            &self.sampler,
        );
        let previous = std::mem::replace(&mut self.normal, normal);
        self.bind_group = bind_group;
        previous.texture.destroy();
    }

    pub(super) fn bind_group(&self) -> &wgpu::BindGroup {
        &self.bind_group
    }
}

fn build_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    metallic_roughness: &wgpu::TextureView,
    normal: &wgpu::TextureView,
    sampler: &wgpu::Sampler,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("trd PBR material maps bind group"),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(metallic_roughness),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(normal),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::Sampler(sampler),
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
) -> LinearMap {
    let (device, queue) = (&gpu.device, &gpu.queue);
    let size = wgpu::Extent3d {
        width: image.width,
        height: image.height,
        depth_or_array_layers: 1,
    };
    let mip_level_count = 1 + image.width.max(image.height).ilog2();
    let usage = wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST;
    // Readback proves that replacing one map leaves the other allocation alive.
    #[cfg(all(test, not(target_arch = "wasm32")))]
    let usage = usage | wgpu::TextureUsages::COPY_SRC;
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size,
        mip_level_count,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage,
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
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    LinearMap { texture, view }
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

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use super::*;
    use crate::render::{create_instance, GpuRequest};

    #[test]
    #[ignore = "requires a GPU adapter"]
    fn setters_replace_only_the_changed_map() {
        let instance = create_instance();
        let gpu = pollster::block_on(GpuContext::request(&instance, &GpuRequest::default()))
            .expect("GPU adapter required");
        let mut maps =
            BoundMaterialMaps::with_layout(&gpu, BoundMaterialMaps::create_layout(&gpu.device));

        // Exhaustive destructuring pins the GPU-only ownership record.
        let BoundMaterialMaps {
            layout: _,
            bind_group: _,
            metallic_roughness:
                LinearMap {
                    texture: initial_mr,
                    view: _,
                },
            normal:
                LinearMap {
                    texture: initial_normal,
                    view: _,
                },
            sampler,
        } = &maps;
        let initial_mr = initial_mr.clone();
        let initial_normal = initial_normal.clone();
        let sampler = sampler.clone();
        assert_eq!(read_pixel(&gpu, &initial_mr), [0, 255, 255, 255]);
        assert_eq!(read_pixel(&gpu, &initial_normal), [128, 128, 255, 255]);

        let mr_pixel = [9, 82, 173, 255];
        maps.set_metallic_roughness(&gpu, &ConstantTexture::new(mr_pixel));
        assert_ne!(maps.metallic_roughness.texture, initial_mr);
        assert_eq!(maps.normal.texture, initial_normal);
        assert_eq!(read_pixel(&gpu, &initial_normal), [128, 128, 255, 255]);
        assert_eq!(read_pixel(&gpu, &maps.metallic_roughness.texture), mr_pixel);

        let retained_mr = maps.metallic_roughness.texture.clone();
        let normal_pixel = [153, 102, 245, 255];
        maps.set_normal(&gpu, &ConstantTexture::new(normal_pixel));
        assert_ne!(maps.normal.texture, initial_normal);
        assert_eq!(maps.metallic_roughness.texture, retained_mr);
        assert_eq!(read_pixel(&gpu, &retained_mr), mr_pixel);
        assert_eq!(read_pixel(&gpu, &maps.normal.texture), normal_pixel);

        let retained_normal = maps.normal.texture.clone();
        maps.set_metallic_roughness(&gpu, &ConstantTexture::new([2, 3, 4, 255]));
        assert_ne!(maps.metallic_roughness.texture, retained_mr);
        assert_eq!(maps.normal.texture, retained_normal);
        assert_eq!(read_pixel(&gpu, &retained_normal), normal_pixel);
        assert_eq!(
            read_pixel(&gpu, &maps.metallic_roughness.texture),
            [2, 3, 4, 255]
        );
        assert_eq!(maps.sampler, sampler);
        maps.destroy();
    }

    fn read_pixel(gpu: &GpuContext, texture: &wgpu::Texture) -> [u8; 4] {
        let staging = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("trd material map test readback"),
            size: 4,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        encoder.copy_texture_to_buffer(
            texture.as_image_copy(),
            wgpu::TexelCopyBufferInfo {
                buffer: &staging,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: None,
                    rows_per_image: None,
                },
            },
            wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
        );
        gpu.queue.submit(Some(encoder.finish()));
        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });
        gpu.device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("GPU poll failed");
        rx.recv()
            .expect("map_async callback dropped")
            .expect("GPU readback failed");
        let pixel = {
            let mapped = slice.get_mapped_range().expect("buffer mapped after poll");
            mapped[..4].try_into().expect("one RGBA pixel")
        };
        staging.unmap();
        pixel
    }
}
