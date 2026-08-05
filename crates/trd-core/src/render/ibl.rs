use super::create_env_bind_group_layout;

/// Per-object image-based-lighting controls.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ImageBasedLighting {
    /// Environment-map reflection gain.
    pub intensity: f32,
}

impl Default for ImageBasedLighting {
    fn default() -> Self {
        Self { intensity: 1.0 }
    }
}

/// A decoded, linear-RGB equirectangular HDR environment probe.
#[derive(Debug, Clone)]
pub struct EnvMapData {
    pub width: u32,
    pub height: u32,
    /// Tightly-packed row-major RGBA f32 (`width * height * 4` elements).
    pub rgba: Vec<f32>,
}

impl EnvMapData {
    /// Builds an environment map, box-downscaling to the portable texture limit.
    pub fn from_rgba32f(width: u32, height: u32, rgba: Vec<f32>, max_dim: u32) -> Self {
        let max_dim = max_dim.max(1);
        let factor = (width.max(height).div_ceil(max_dim)).max(1);
        if factor == 1 {
            return Self {
                width,
                height,
                rgba,
            };
        }
        let dw = (width / factor).max(1);
        let dh = (height / factor).max(1);
        let mut out = vec![0.0f32; (dw * dh * 4) as usize];
        for y in 0..dh {
            for x in 0..dw {
                let mut acc = [0.0f32; 4];
                let mut n = 0.0f32;
                for sy in 0..factor {
                    let yy = y * factor + sy;
                    if yy >= height {
                        break;
                    }
                    for sx in 0..factor {
                        let xx = x * factor + sx;
                        if xx >= width {
                            break;
                        }
                        let i = ((yy * width + xx) * 4) as usize;
                        acc[0] += rgba[i];
                        acc[1] += rgba[i + 1];
                        acc[2] += rgba[i + 2];
                        acc[3] += rgba[i + 3];
                        n += 1.0;
                    }
                }
                let di = ((y * dw + x) * 4) as usize;
                for c in 0..4 {
                    out[di + c] = acc[c] / n;
                }
            }
        }
        Self {
            width: dw,
            height: dh,
            rgba: out,
        }
    }
}

/// Lazily uploaded HDR environment-map binding.
pub(crate) struct BoundEnv {
    layout: wgpu::BindGroupLayout,
    bind_group: Option<wgpu::BindGroup>,
    data: Option<EnvMapData>,
}

impl BoundEnv {
    pub(crate) fn new(device: &wgpu::Device) -> Self {
        Self {
            layout: create_env_bind_group_layout(device),
            bind_group: None,
            data: None,
        }
    }

    pub(crate) fn layout(&self) -> &wgpu::BindGroupLayout {
        &self.layout
    }

    pub(crate) fn set(&mut self, data: EnvMapData) {
        self.data = Some(data);
        self.bind_group = None;
    }

    pub(crate) fn has_env(&self) -> bool {
        self.data.is_some()
    }

    pub(crate) fn ensure_uploaded(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> &wgpu::BindGroup {
        if self.bind_group.is_none() {
            let fallback = EnvMapData {
                width: 1,
                height: 1,
                rgba: vec![0.0, 0.0, 0.0, 1.0],
            };
            let data = self.data.as_ref().unwrap_or(&fallback);
            self.bind_group = Some(upload_env_texture(device, queue, &self.layout, data));
        }
        self.bind_group.as_ref().expect("uploaded above")
    }
}

fn upload_env_texture(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    layout: &wgpu::BindGroupLayout,
    env: &EnvMapData,
) -> wgpu::BindGroup {
    let width = env.width.max(1);
    let height = env.height.max(1);
    let size = wgpu::Extent3d {
        width,
        height,
        depth_or_array_layers: 1,
    };
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("trd env texture"),
        size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba16Float,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    let half: Vec<u16> = env
        .rgba
        .iter()
        .map(|&c| f32_to_f16_bits(c.clamp(0.0, 65504.0)))
        .collect();
    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        bytemuck::cast_slice(&half),
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(width * 8),
            rows_per_image: Some(height),
        },
        size,
    );
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("trd env sampler"),
        address_mode_u: wgpu::AddressMode::Repeat,
        address_mode_v: wgpu::AddressMode::ClampToEdge,
        address_mode_w: wgpu::AddressMode::ClampToEdge,
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        mipmap_filter: wgpu::MipmapFilterMode::Nearest,
        ..Default::default()
    });
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
        ],
    })
}

fn f32_to_f16_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xff) as i32 - 127 + 15;
    let mantissa = bits & 0x007f_ffff;

    if exp >= 0x1f {
        if ((bits >> 23) & 0xff) == 0xff && mantissa != 0 {
            return sign | 0x7e00;
        }
        return sign | 0x7c00;
    }
    if exp <= 0 {
        if exp < -10 {
            return sign;
        }
        let mant = mantissa | 0x0080_0000;
        let shift = (14 - exp) as u32;
        let mut half = (mant >> shift) as u16;
        if (mant >> (shift - 1)) & 1 != 0 {
            half += 1;
        }
        return sign | half;
    }
    let mut half = ((exp as u16) << 10) | ((mantissa >> 13) as u16);
    if mantissa & 0x0000_1000 != 0 {
        half += 1;
    }
    sign | half
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f16_conversion_matches_known_values() {
        assert_eq!(f32_to_f16_bits(0.0), 0x0000);
        assert_eq!(f32_to_f16_bits(1.0), 0x3c00);
        assert_eq!(f32_to_f16_bits(2.0), 0x4000);
        assert_eq!(f32_to_f16_bits(0.5), 0x3800);
        assert_eq!(f32_to_f16_bits(1.0e30), 0x7c00);
    }
}
