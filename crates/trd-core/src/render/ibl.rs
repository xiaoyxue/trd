use super::create_env_bind_group_layout;
use super::GpuContext;

/// Per-object image-based-lighting controls.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ImageBasedLighting {
    /// Environment-map reflection gain.
    pub intensity: f32,
    /// Rotation around the environment's Y axis, in radians.
    pub rotation: f32,
}

impl Default for ImageBasedLighting {
    fn default() -> Self {
        Self {
            intensity: 1.0,
            rotation: 0.0,
        }
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

/// The HDR environment-map binding.
///
/// Uploaded **eagerly**: the constructor binds a 1×1 black fallback and
/// [`set`](Self::set) replaces it immediately, so `encode` never has to upload
/// (#180). `has_env` still reports whether a *real* probe was supplied, which is
/// what the PBR uniform keys reflections off.
pub(crate) struct BoundEnv {
    layout: wgpu::BindGroupLayout,
    bind_group: wgpu::BindGroup,
    has_env: bool,
}

impl BoundEnv {
    pub(crate) fn new(gpu: &GpuContext) -> Self {
        let layout = create_env_bind_group_layout(&gpu.device);
        // 1×1 black: no reflection until a probe is set.
        let fallback = EnvMapData {
            width: 1,
            height: 1,
            rgba: vec![0.0, 0.0, 0.0, 1.0],
        };
        let bind_group = upload_env_texture(&gpu.device, &gpu.queue, &layout, &fallback);
        Self {
            layout,
            bind_group,
            has_env: false,
        }
    }

    pub(crate) fn layout(&self) -> &wgpu::BindGroupLayout {
        &self.layout
    }

    pub(crate) fn set(&mut self, gpu: &GpuContext, data: EnvMapData) {
        self.bind_group = upload_env_texture(&gpu.device, &gpu.queue, &self.layout, &data);
        self.has_env = true;
    }

    pub(crate) fn has_env(&self) -> bool {
        self.has_env
    }

    pub(crate) fn bind_group(&self) -> &wgpu::BindGroup {
        &self.bind_group
    }
}

fn upload_env_texture(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    layout: &wgpu::BindGroupLayout,
    env: &EnvMapData,
) -> wgpu::BindGroup {
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
    let (brdf_view, brdf_sampler) = upload_brdf_lut(device, queue);
    let irradiance = build_irradiance_map(width, height, &base_rgba, 64, 32, 32);
    let irradiance_view =
        upload_rgba16f_view(device, queue, "trd diffuse irradiance", 64, 32, &irradiance);
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

fn fit_environment(env: &EnvMapData, max_dim: u32) -> (u32, u32, Vec<f32>) {
    let mut width = env.width.max(1);
    let mut height = env.height.max(1);
    let mut rgba = env.rgba.clone();
    while width.max(height) > max_dim {
        rgba = downsample_environment(width, height, &rgba);
        width = (width / 2).max(1);
        height = (height / 2).max(1);
    }
    (width, height, rgba)
}

fn downsample_environment(width: u32, height: u32, rgba: &[f32]) -> Vec<f32> {
    let next_width = (width / 2).max(1);
    let next_height = (height / 2).max(1);
    let mut out = vec![0.0; (next_width * next_height * 4) as usize];
    for y in 0..next_height {
        for x in 0..next_width {
            let mut sum = [0.0; 4];
            for (sx, sy) in [
                ((x * 2) % width, (y * 2).min(height - 1)),
                ((x * 2 + 1) % width, (y * 2).min(height - 1)),
                ((x * 2) % width, (y * 2 + 1).min(height - 1)),
                ((x * 2 + 1) % width, (y * 2 + 1).min(height - 1)),
            ] {
                let src = ((sy * width + sx) * 4) as usize;
                for channel in 0..4 {
                    sum[channel] += rgba[src + channel];
                }
            }
            let dst = ((y * next_width + x) * 4) as usize;
            for channel in 0..4 {
                out[dst + channel] = sum[channel] * 0.25;
            }
        }
    }
    out
}

fn prefilter_environment_level(
    source_width: u32,
    source_height: u32,
    source: &[f32],
    width: u32,
    height: u32,
    roughness: f32,
    sample_count: u32,
) -> Vec<f32> {
    let mut out = vec![0.0; (width * height * 4) as usize];
    for y in 0..height {
        for x in 0..width {
            let normal = direction_from_uv(
                (x as f32 + 0.5) / width as f32,
                (y as f32 + 0.5) / height as f32,
            );
            let view = normal;
            let mut sum = [0.0; 3];
            let mut weight = 0.0;
            for sample in 0..sample_count {
                let xi = [
                    sample as f32 / sample_count as f32,
                    radical_inverse_vdc(sample),
                ];
                let half = tangent_to_world(importance_sample_ggx(xi, roughness), normal);
                let view_dot_half = dot(view, half).max(0.0);
                let light = normalize([
                    2.0 * view_dot_half * half[0] - view[0],
                    2.0 * view_dot_half * half[1] - view[1],
                    2.0 * view_dot_half * half[2] - view[2],
                ]);
                let n_dot_l = dot(normal, light).max(0.0);
                if n_dot_l > 0.0 {
                    let radiance =
                        sample_equirectangular(source_width, source_height, source, light);
                    for channel in 0..3 {
                        sum[channel] += radiance[channel] * n_dot_l;
                    }
                    weight += n_dot_l;
                }
            }
            let target = ((y * width + x) * 4) as usize;
            for channel in 0..3 {
                out[target + channel] = sum[channel] / weight.max(1e-5);
            }
            out[target + 3] = 1.0;
        }
    }
    out
}

fn build_irradiance_map(
    source_width: u32,
    source_height: u32,
    source: &[f32],
    width: u32,
    height: u32,
    sample_count: u32,
) -> Vec<f32> {
    let mut out = vec![0.0; (width * height * 4) as usize];
    for y in 0..height {
        for x in 0..width {
            let normal = direction_from_uv(
                (x as f32 + 0.5) / width as f32,
                (y as f32 + 0.5) / height as f32,
            );
            let mut sum = [0.0; 3];
            for sample in 0..sample_count {
                let xi = [
                    sample as f32 / sample_count as f32,
                    radical_inverse_vdc(sample),
                ];
                let phi = std::f32::consts::TAU * xi[0];
                let radius = xi[1].sqrt();
                let local = [phi.cos() * radius, phi.sin() * radius, (1.0 - xi[1]).sqrt()];
                let direction = tangent_to_world(local, normal);
                let radiance =
                    sample_equirectangular(source_width, source_height, source, direction);
                for channel in 0..3 {
                    sum[channel] += radiance[channel];
                }
            }
            let target = ((y * width + x) * 4) as usize;
            for channel in 0..3 {
                out[target + channel] = sum[channel] / sample_count as f32;
            }
            out[target + 3] = 1.0;
        }
    }
    out
}

fn direction_from_uv(u: f32, v: f32) -> [f32; 3] {
    let theta = v * std::f32::consts::PI;
    let phi = u * std::f32::consts::TAU;
    let sin_theta = theta.sin();
    [sin_theta * phi.sin(), theta.cos(), sin_theta * phi.cos()]
}

fn tangent_to_world(local: [f32; 3], normal: [f32; 3]) -> [f32; 3] {
    let up = if normal[2].abs() > 0.999 {
        [1.0, 0.0, 0.0]
    } else {
        [0.0, 0.0, 1.0]
    };
    let tangent = normalize(cross(up, normal));
    let bitangent = cross(normal, tangent);
    normalize([
        tangent[0] * local[0] + bitangent[0] * local[1] + normal[0] * local[2],
        tangent[1] * local[0] + bitangent[1] * local[1] + normal[1] * local[2],
        tangent[2] * local[0] + bitangent[2] * local[1] + normal[2] * local[2],
    ])
}

fn sample_equirectangular(width: u32, height: u32, rgba: &[f32], direction: [f32; 3]) -> [f32; 3] {
    let direction = normalize(direction);
    let phi = direction[0]
        .atan2(direction[2])
        .rem_euclid(std::f32::consts::TAU);
    let theta = direction[1].clamp(-1.0, 1.0).acos();
    let x = phi / std::f32::consts::TAU * width as f32 - 0.5;
    let y = theta / std::f32::consts::PI * height as f32 - 0.5;
    let x0 = x.floor() as i32;
    let y0 = y.floor() as i32;
    let tx = x - x.floor();
    let ty = y - y.floor();
    let mut result = [0.0; 3];
    for (dx, wx) in [(0, 1.0 - tx), (1, tx)] {
        for (dy, wy) in [(0, 1.0 - ty), (1, ty)] {
            let sx = (x0 + dx).rem_euclid(width as i32) as u32;
            let sy = (y0 + dy).clamp(0, height as i32 - 1) as u32;
            let source = ((sy * width + sx) * 4) as usize;
            for channel in 0..3 {
                result[channel] += rgba[source + channel] * wx * wy;
            }
        }
    }
    result
}

fn upload_rgba16f_view(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    label: &str,
    width: u32,
    height: u32,
    rgba: &[f32],
) -> wgpu::TextureView {
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

fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn normalize(vector: [f32; 3]) -> [f32; 3] {
    let length = dot(vector, vector).sqrt().max(1e-8);
    [vector[0] / length, vector[1] / length, vector[2] / length]
}

fn upload_brdf_lut(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
) -> (wgpu::TextureView, wgpu::Sampler) {
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

fn integrate_brdf(n_dot_v: f32, roughness: f32, sample_count: u32) -> (f32, f32) {
    let v = [(1.0 - n_dot_v * n_dot_v).sqrt(), 0.0, n_dot_v];
    let mut a = 0.0;
    let mut b = 0.0;
    for i in 0..sample_count {
        let xi = [i as f32 / sample_count as f32, radical_inverse_vdc(i)];
        let h = importance_sample_ggx(xi, roughness);
        let v_dot_h = dot(v, h).max(0.0);
        let l = [
            2.0 * v_dot_h * h[0] - v[0],
            2.0 * v_dot_h * h[1] - v[1],
            2.0 * v_dot_h * h[2] - v[2],
        ];
        let n_dot_l = l[2].max(0.0);
        let n_dot_h = h[2].max(0.0);
        if n_dot_l > 0.0 {
            let geometry = geometry_smith(n_dot_v, n_dot_l, roughness);
            let visibility = geometry * v_dot_h / (n_dot_h * n_dot_v).max(1e-5);
            let fresnel = (1.0 - v_dot_h).powi(5);
            a += (1.0 - fresnel) * visibility;
            b += fresnel * visibility;
        }
    }
    (a / sample_count as f32, b / sample_count as f32)
}

fn radical_inverse_vdc(bits: u32) -> f32 {
    bits.reverse_bits() as f32 * 2.328_306_4e-10
}

fn importance_sample_ggx(xi: [f32; 2], roughness: f32) -> [f32; 3] {
    let alpha = roughness * roughness;
    let phi = std::f32::consts::TAU * xi[0];
    let cos_theta = ((1.0 - xi[1]) / (1.0 + (alpha * alpha - 1.0) * xi[1])).sqrt();
    let sin_theta = (1.0 - cos_theta * cos_theta).max(0.0).sqrt();
    [phi.cos() * sin_theta, phi.sin() * sin_theta, cos_theta]
}

fn geometry_smith(n_dot_v: f32, n_dot_l: f32, roughness: f32) -> f32 {
    let k = roughness * roughness * 0.5;
    let g1 = |n_dot: f32| n_dot / (n_dot * (1.0 - k) + k);
    g1(n_dot_v) * g1(n_dot_l)
}

fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
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
