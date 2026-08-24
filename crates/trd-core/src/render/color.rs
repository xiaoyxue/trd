//! CPU-side color helpers: texture upload with mipmap generation and
//! sRGB (de)linearization.

use super::GpuContext;
use crate::texture::ImageData;

/// Uploads `image` to a fresh `Rgba8UnormSrgb` `wgpu::Texture` and builds the
/// group-1 bind group (texture view + a trilinear, clamp-to-edge sampler) over
/// `layout`. sRGB storage so texels linearize on sample (#20). A full mipmap
/// chain is generated on the CPU (box-filtered in *linear* space, matching the
/// sRGB storage) and uploaded per level, so minified/foreshortened surfaces
/// filter smoothly instead of aliasing the atlas detail into speckle.
pub(crate) fn upload_texture(
    gpu: &GpuContext,
    layout: &wgpu::BindGroupLayout,
    image: &ImageData,
) -> (wgpu::BindGroup, wgpu::Texture) {
    let (device, queue) = (&gpu.device, &gpu.queue);
    let mip_level_count = 1 + image.width.max(image.height).max(1).ilog2();
    let size = wgpu::Extent3d {
        width: image.width,
        height: image.height,
        depth_or_array_layers: 1,
    };
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("trd texture"),
        size,
        mip_level_count,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8UnormSrgb,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    // Upload the base level, then repeatedly box-downsample it to fill the chain.
    let mut level_w = image.width.max(1);
    let mut level_h = image.height.max(1);
    let mut level_rgba = image.rgba.clone();
    for mip in 0..mip_level_count {
        if mip > 0 {
            let (w, h, rgba) = downsample_srgb(level_w, level_h, &level_rgba);
            level_w = w;
            level_h = h;
            level_rgba = rgba;
        }
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: mip,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &level_rgba,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(level_w * 4),
                rows_per_image: Some(level_h),
            },
            wgpu::Extent3d {
                width: level_w,
                height: level_h,
                depth_or_array_layers: 1,
            },
        );
    }
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("trd texture sampler"),
        address_mode_u: wgpu::AddressMode::ClampToEdge,
        address_mode_v: wgpu::AddressMode::ClampToEdge,
        address_mode_w: wgpu::AddressMode::ClampToEdge,
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        mipmap_filter: wgpu::MipmapFilterMode::Linear,
        ..Default::default()
    });
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("trd texture bind group"),
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
    });
    // The texture is handed back so its owner can `destroy()` it. Dropping the
    // bind group is not a release: these are refcounted handles, so it frees
    // only while nothing else holds one (#353).
    (bind_group, texture)
}

/// sRGB byte (`0..=255`) → linear `[0, 1]`.
fn srgb_to_linear(c: u8) -> f32 {
    let c = c as f32 / 255.0;
    if c <= 0.04045 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

/// Linear `[0, 1]` → sRGB byte (`0..=255`), rounded.
fn linear_to_srgb(l: f32) -> u8 {
    let l = l.clamp(0.0, 1.0);
    let s = if l <= 0.0031308 {
        12.92 * l
    } else {
        1.055 * l.powf(1.0 / 2.4) - 0.055
    };
    (s * 255.0 + 0.5) as u8
}

/// Box-downsamples one tightly-packed RGBA8 level to half size (min 1px). Color
/// is averaged in **linear** space (the texture is sRGB) and re-encoded; alpha is
/// averaged linearly. Returns `(width, height, rgba)` of the smaller level.
fn downsample_srgb(w: u32, h: u32, src: &[u8]) -> (u32, u32, Vec<u8>) {
    let w2 = (w / 2).max(1);
    let h2 = (h / 2).max(1);
    let mut dst = vec![0u8; (w2 * h2 * 4) as usize];
    for y in 0..h2 {
        let y0 = (2 * y).min(h - 1);
        let y1 = (2 * y + 1).min(h - 1);
        for x in 0..w2 {
            let x0 = (2 * x).min(w - 1);
            let x1 = (2 * x + 1).min(w - 1);
            let mut lin = [0.0f32; 3];
            let mut a = 0.0f32;
            for (sx, sy) in [(x0, y0), (x1, y0), (x0, y1), (x1, y1)] {
                let i = ((sy * w + sx) * 4) as usize;
                lin[0] += srgb_to_linear(src[i]);
                lin[1] += srgb_to_linear(src[i + 1]);
                lin[2] += srgb_to_linear(src[i + 2]);
                a += src[i + 3] as f32;
            }
            let di = ((y * w2 + x) * 4) as usize;
            dst[di] = linear_to_srgb(lin[0] / 4.0);
            dst[di + 1] = linear_to_srgb(lin[1] / 4.0);
            dst[di + 2] = linear_to_srgb(lin[2] / 4.0);
            dst[di + 3] = (a / 4.0 + 0.5) as u8;
        }
    }
    (w2, h2, dst)
}
