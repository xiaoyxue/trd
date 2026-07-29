//! Disney principled-BRDF material, its GPU uniform, smooth-normal computation,
//! and the HDR environment-map binding — the CPU side of the `disney.wgsl`
//! physically-based mesh path.
//!
//! This is the data half of the PBR feature (the pipeline/bind-group plumbing
//! lives in [`pipeline`](super::pipeline), the encode arm in
//! [`mesh_renderer`](super::mesh_renderer)): a [`PbrMaterial`] of Disney
//! parameters, the [`PbrUniform`] byte layout matching the shader, a small fixed
//! **virtual light rig**, per-vertex smooth-normal derivation (the assets carry
//! no `vn`), and [`BoundEnv`] — the equirectangular HDR probe (`Rgba16Float`)
//! reflected by metallic surfaces.

use super::{create_env_bind_group_layout, Vertex};
use crate::math::Vector3;

/// The Disney principled-BRDF material parameters plus trd's shading controls
/// (a small ambient fill, light-rig scale, environment intensity, and exposure),
/// applied globally to every [`RenderMode::Pbr`](super::RenderMode) mesh.
///
/// Field ranges mirror Burley 2012 (`ref/DisneyPBR/shader.frag`): every
/// parameter is in `[0, 1]` except `base_color` (a linear RGB tint multiplied
/// onto the sampled albedo), `env_intensity`, `exposure`, `ambient`, and
/// `light_scale` (non-negative gains). [`Default`] is a neutral glossy
/// dielectric; [`PbrMaterial::metal`] is a shiny metal preset (e.g. the coke can).
#[derive(Debug, Clone, Copy)]
pub struct PbrMaterial {
    /// Linear-RGB tint multiplied onto the sampled albedo (identity `[1, 1, 1]`).
    pub base_color: [f32; 3],
    /// 0 = dielectric, 1 = metal (kills the diffuse lobe, tints reflection).
    pub metallic: f32,
    /// Diffuse ↔ subsurface-scattering blend.
    pub subsurface: f32,
    /// Dielectric specular reflectance strength (`0.5` ≈ 4% F0).
    pub specular: f32,
    /// Surface roughness (0 = mirror, 1 = fully rough).
    pub roughness: f32,
    /// Tints the dielectric specular toward the base-color hue.
    pub specular_tint: f32,
    /// Specular anisotropy (0 = isotropic).
    pub anisotropic: f32,
    /// Sheen (grazing retro-reflection, e.g. cloth) strength.
    pub sheen: f32,
    /// Tints the sheen toward the base-color hue.
    pub sheen_tint: f32,
    /// Clearcoat lobe strength (a second, colorless specular layer).
    pub clearcoat: f32,
    /// Clearcoat glossiness (0 = satin, 1 = glossy).
    pub clearcoat_gloss: f32,
    /// Environment-map reflection gain (0 disables the probe reflection).
    pub env_intensity: f32,
    /// Tone-map exposure applied to the linear radiance before the Reinhard curve.
    pub exposure: f32,
    /// Constant ambient fill (× base color) so shadowed regions are not black.
    pub ambient: f32,
    /// Scales every virtual light's contribution (the reference used a fixed 5×).
    pub light_scale: f32,
}

impl Default for PbrMaterial {
    fn default() -> Self {
        Self {
            base_color: [1.0, 1.0, 1.0],
            metallic: 0.0,
            subsurface: 0.0,
            specular: 0.5,
            roughness: 0.5,
            specular_tint: 0.0,
            anisotropic: 0.0,
            sheen: 0.0,
            sheen_tint: 0.5,
            clearcoat: 0.0,
            clearcoat_gloss: 1.0,
            env_intensity: 1.0,
            exposure: 1.2,
            ambient: 0.12,
            light_scale: 2.5,
        }
    }
}

impl PbrMaterial {
    /// A shiny metal preset: fully metallic, moderately smooth, with the
    /// environment probe strongly reflected — the look asked of the coke can.
    pub fn metal() -> Self {
        Self {
            metallic: 1.0,
            roughness: 0.28,
            env_intensity: 1.0,
            ..Self::default()
        }
    }
}

/// The fixed virtual light rig (three directional lights: key, fill, rim). Each
/// `[x, y, z, intensity]` gives the direction the light **travels** (the shader
/// lights along `L = normalize(-xyz)`) and its intensity. World-space, so a
/// spinning turntable object is lit from changing angles, revealing the material.
const DIR_LIGHTS: [[f32; 4]; 3] = [
    // Key: from the upper front-right.
    [-0.5, -0.85, -0.55, 1.0],
    // Fill: softer, from the left.
    [0.8, -0.25, 0.35, 0.4],
    // Rim: from behind to pop the silhouette.
    [0.25, -0.3, 0.9, 0.55],
];

/// Maximum lights per kind, matching `disney.wgsl`'s `MAX_LIGHTS` array size.
const MAX_LIGHTS: usize = 4;

/// GPU byte layout matching `disney.wgsl`'s `PbrUniform` (std140-compatible: all
/// members are 16-byte-aligned `vec4`/`mat4`). 288 bytes, uploaded per frame
/// (the camera terms change; the material + light rig are constant per render).
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct PbrUniform {
    view_proj: [f32; 16],
    camera_pos: [f32; 4],
    mat0: [f32; 4],
    mat1: [f32; 4],
    mat2: [f32; 4],
    mat3: [f32; 4],
    counts: [f32; 4],
    dir_lights: [[f32; 4]; MAX_LIGHTS],
    point_lights: [[f32; 4]; MAX_LIGHTS],
}

impl PbrUniform {
    /// Packs the camera `P·V`, camera world position, the [`PbrMaterial`], and the
    /// fixed light rig into the shader layout. `use_env` gates the environment
    /// reflection (false when no HDR probe is bound).
    pub(crate) fn new(
        view_proj: [f32; 16],
        camera_pos: [f32; 3],
        material: &PbrMaterial,
        use_env: bool,
    ) -> Self {
        let mut dir_lights = [[0.0f32; 4]; MAX_LIGHTS];
        dir_lights[..DIR_LIGHTS.len()].copy_from_slice(&DIR_LIGHTS);
        Self {
            view_proj,
            camera_pos: [camera_pos[0], camera_pos[1], camera_pos[2], 1.0],
            mat0: [
                material.metallic,
                material.subsurface,
                material.specular,
                material.roughness,
            ],
            mat1: [
                material.specular_tint,
                material.anisotropic,
                material.sheen,
                material.sheen_tint,
            ],
            mat2: [
                material.clearcoat,
                material.clearcoat_gloss,
                material.env_intensity,
                material.exposure,
            ],
            mat3: [
                material.base_color[0],
                material.base_color[1],
                material.base_color[2],
                material.ambient,
            ],
            counts: [
                DIR_LIGHTS.len() as f32,
                0.0,
                if use_env { 1.0 } else { 0.0 },
                material.light_scale,
            ],
            dir_lights,
            point_lights: [[0.0; 4]; MAX_LIGHTS],
        }
    }
}

/// Computes area-weighted smooth per-vertex normals from an indexed triangle
/// mesh. The trd assets carry no `vn`, so the Disney path derives shading normals
/// here: each triangle's un-normalized cross product (whose length is ∝ twice its
/// area) is accumulated onto its three vertices, then each vertex normal is
/// normalized. Degenerate (zero-length) accumulations fall back to `+Z`.
pub(crate) fn compute_smooth_normals(vertices: &[Vertex], indices: &[u32]) -> Vec<[f32; 3]> {
    let mut normals = vec![Vector3::ZERO; vertices.len()];
    for tri in indices.chunks_exact(3) {
        let (i0, i1, i2) = (tri[0] as usize, tri[1] as usize, tri[2] as usize);
        let p0 = Vector3::from_array(vertices[i0].position);
        let p1 = Vector3::from_array(vertices[i1].position);
        let p2 = Vector3::from_array(vertices[i2].position);
        // Length ∝ 2× triangle area, so larger faces weight the shared normal
        // more — the standard area-weighted smoothing.
        let face = (p1 - p0).cross(p2 - p0);
        normals[i0] = normals[i0] + face;
        normals[i1] = normals[i1] + face;
        normals[i2] = normals[i2] + face;
    }
    normals
        .into_iter()
        .map(|n| {
            let len = n.length();
            if len > 1e-12 {
                (n / len).to_array()
            } else {
                [0.0, 0.0, 1.0]
            }
        })
        .collect()
}

/// A decoded, linear-RGB equirectangular HDR environment probe (row-major
/// `height`×`width`×4 f32, RGBA). Produced by the shell front-end (trd-cli
/// decodes the `.hdr` file — trd-core does no file/codec I/O) and uploaded as a
/// filterable `Rgba16Float` texture reflected by metallic surfaces.
#[derive(Debug, Clone)]
pub struct EnvMapData {
    pub width: u32,
    pub height: u32,
    /// Tightly-packed row-major RGBA f32 (`width * height * 4` elements).
    pub rgba: Vec<f32>,
}

impl EnvMapData {
    /// Builds an [`EnvMapData`] from a decoded row-major RGBA f32 probe, box-
    /// downscaling by an integer factor so neither dimension exceeds `max_dim`
    /// (the renderer's portable 2048px texture limit). Pure CPU math — the shell
    /// front-ends decode the `.hdr` file (trd-core does no file/codec I/O) then
    /// call this to fit the probe to the device limits.
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

/// The bound HDR environment map (group 2 of the PBR pipeline), mirroring
/// [`BoundTexture`](super::bound_texture): owns the layout + the CPU probe and
/// (re)builds the GPU bind group lazily in `encode` (the only place a queue is
/// available). Until [`set`](Self::set), a 1×1 black probe keeps the bind group
/// valid (the shader gates reads on `use_env`, so it is never actually sampled).
pub(crate) struct BoundEnv {
    layout: wgpu::BindGroupLayout,
    bind_group: Option<wgpu::BindGroup>,
    data: Option<EnvMapData>,
}

impl BoundEnv {
    /// Constructs an unbound `BoundEnv`; the first
    /// [`ensure_uploaded`](Self::ensure_uploaded) builds a 1×1 black probe.
    pub(crate) fn new(device: &wgpu::Device) -> Self {
        Self {
            layout: create_env_bind_group_layout(device),
            bind_group: None,
            data: None,
        }
    }

    /// The group-2 bind-group layout (also fed to the PBR pipeline layout).
    pub(crate) fn layout(&self) -> &wgpu::BindGroupLayout {
        &self.layout
    }

    /// Binds `data` as the environment probe; the bind group rebuilds on the next
    /// [`ensure_uploaded`](Self::ensure_uploaded).
    pub(crate) fn set(&mut self, data: EnvMapData) {
        self.data = Some(data);
        self.bind_group = None;
    }

    /// Whether a real probe is bound (drives the shader's `use_env` flag).
    pub(crate) fn has_env(&self) -> bool {
        self.data.is_some()
    }

    /// Uploads the probe (or a 1×1 black fallback) if not already uploaded since
    /// the last [`set`](Self::set), returning the group-2 bind group.
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

/// Uploads `env` to a filterable `Rgba16Float` texture and builds the group-2
/// bind group (view + a linear sampler that **repeats** horizontally / clamps
/// vertically, matching the equirectangular wrap). `Rgba16Float` (rather than
/// `Rgba32Float`) is filterable on the portable/downlevel target without any
/// adapter feature, so the f32 source is narrowed to half precision on upload.
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
    // Narrow the linear f32 radiance to half precision (clamped to the f16 max so
    // very bright probe pixels don't become +inf, which would blow out filtering).
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
            bytes_per_row: Some(width * 8), // 4 channels × 2 bytes (f16)
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

/// Narrows an `f32` to IEEE-754 half-precision (`f16`) bits (round-to-nearest,
/// with subnormal + overflow handling). Kept as a ~20-line helper so the crate
/// needs no `half` dependency just to upload the HDR probe.
fn f32_to_f16_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xff) as i32 - 127 + 15;
    let mantissa = bits & 0x007f_ffff;

    if exp >= 0x1f {
        // Overflow / inf / NaN.
        if ((bits >> 23) & 0xff) == 0xff && mantissa != 0 {
            return sign | 0x7e00; // NaN
        }
        return sign | 0x7c00; // ±inf
    }
    if exp <= 0 {
        // Subnormal (or underflow to zero).
        if exp < -10 {
            return sign;
        }
        let mant = mantissa | 0x0080_0000;
        let shift = (14 - exp) as u32;
        let mut half = (mant >> shift) as u16;
        // Round to nearest.
        if (mant >> (shift - 1)) & 1 != 0 {
            half += 1;
        }
        return sign | half;
    }
    let mut half = ((exp as u16) << 10) | ((mantissa >> 13) as u16);
    // Round to nearest even (a carry into the exponent is representable).
    if mantissa & 0x0000_1000 != 0 {
        half += 1;
    }
    sign | half
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pbr_uniform_is_288_bytes() {
        assert_eq!(std::mem::size_of::<PbrUniform>(), 288);
        assert_eq!(std::mem::size_of::<PbrUniform>() % 16, 0);
    }

    #[test]
    fn smooth_normals_of_xy_quad_point_up_z() {
        // Two triangles of a unit quad in the z=0 plane, CCW → +Z normals.
        let v = |x: f32, y: f32| Vertex {
            position: [x, y, 0.0],
            color: [0.0; 3],
            uv: [0.0; 2],
        };
        let vertices = vec![v(0.0, 0.0), v(1.0, 0.0), v(1.0, 1.0), v(0.0, 1.0)];
        let indices = vec![0, 1, 2, 0, 2, 3];
        let normals = compute_smooth_normals(&vertices, &indices);
        for n in normals {
            assert!((n[0]).abs() < 1e-6);
            assert!((n[1]).abs() < 1e-6);
            assert!((n[2] - 1.0).abs() < 1e-6);
        }
    }

    #[test]
    fn f16_conversion_matches_known_values() {
        assert_eq!(f32_to_f16_bits(0.0), 0x0000);
        assert_eq!(f32_to_f16_bits(1.0), 0x3c00);
        assert_eq!(f32_to_f16_bits(2.0), 0x4000);
        assert_eq!(f32_to_f16_bits(0.5), 0x3800);
        // Overflow clamps below the f16 max in the uploader, but raw large inputs
        // still saturate to +inf here.
        assert_eq!(f32_to_f16_bits(1.0e30), 0x7c00);
    }

    #[test]
    fn use_env_flag_packs_into_counts() {
        let m = PbrMaterial::default();
        let with = PbrUniform::new([0.0; 16], [0.0; 3], &m, true);
        let without = PbrUniform::new([0.0; 16], [0.0; 3], &m, false);
        assert_eq!(with.counts[2], 1.0);
        assert_eq!(without.counts[2], 0.0);
        assert_eq!(with.counts[0], DIR_LIGHTS.len() as f32);
    }
}
