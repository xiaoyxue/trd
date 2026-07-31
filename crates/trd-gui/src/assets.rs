//! Built-in assets shared by the native and wasm entry points (#97): the default
//! mesh, and texture decoding for the optional albedo (native `--texture` /
//! browser `?texture=`).

use trd_core::{EnvMapData, ImageTexture, Mesh, MeshError};

use crate::error::GuiError;

/// Textures are downscaled to fit this square before upload — `trd-core`'s
/// headless renderer uses wgpu's downlevel limits (`max_texture_dimension_2d`
/// 2048) and the demo albedo maps are 3072², so this keeps them within range on
/// every target.
pub const MAX_TEXTURE_DIM: u32 = 2048;

/// HDR environment probes are box-downscaled to fit this dimension before upload,
/// matching the renderer's portable 2048px `max_texture_dimension_2d` limit (the
/// demo `.hdr` maps are larger). Shared by [`decode_env_hdr`].
pub const MAX_ENV_DIM: u32 = 2048;

/// A built-in origin-centered unit cube with per-corner colors, used as the
/// default object when no mesh is supplied (`v x y z r g b` OBJ extension).
pub const DEFAULT_MESH_OBJ: &str = "\
v -0.5 -0.5 -0.5 0.1 0.1 0.9
v  0.5 -0.5 -0.5 0.9 0.1 0.1
v  0.5  0.5 -0.5 0.9 0.9 0.1
v -0.5  0.5 -0.5 0.1 0.9 0.1
v -0.5 -0.5  0.5 0.1 0.9 0.9
v  0.5 -0.5  0.5 0.9 0.1 0.9
v  0.5  0.5  0.5 0.9 0.9 0.9
v -0.5  0.5  0.5 0.2 0.2 0.2
f 1 2 3 4
f 5 6 7 8
f 1 5 8 4
f 2 6 7 3
f 4 8 7 3
f 1 5 6 2
";

/// Parses the built-in default cube into a [`Mesh`].
pub fn default_mesh() -> Result<Mesh, MeshError> {
    Mesh::from_obj(DEFAULT_MESH_OBJ)
}

/// Decodes texture image **bytes** (PNG/JPEG) into an [`ImageTexture`],
/// downscaling to [`MAX_TEXTURE_DIM`] so it fits the renderer's texture-size
/// limit. Shared by the native `--texture` (file bytes) and the browser
/// `?texture=` (fetched bytes) paths; image decoding stays in Rust so trd-core
/// remains I/O-free.
pub fn decode_texture(bytes: &[u8]) -> Result<ImageTexture, GuiError> {
    let image = image::load_from_memory(bytes)?;
    texture_from_image(image)
}

/// Downscales `image` to fit [`MAX_TEXTURE_DIM`] (preserving aspect) and converts
/// it to an [`ImageTexture`]. Split from I/O so it is unit-testable.
pub fn texture_from_image(image: image::DynamicImage) -> Result<ImageTexture, GuiError> {
    let image = if image.width() > MAX_TEXTURE_DIM || image.height() > MAX_TEXTURE_DIM {
        log::info!(
            "downscaling texture {}×{} to fit {MAX_TEXTURE_DIM}²",
            image.width(),
            image.height()
        );
        image.resize(
            MAX_TEXTURE_DIM,
            MAX_TEXTURE_DIM,
            image::imageops::FilterType::Triangle,
        )
    } else {
        image
    };
    let rgba = image.to_rgba8();
    let (width, height) = rgba.dimensions();
    Ok(ImageTexture::from_rgba(width, height, rgba.into_raw())?)
}

/// Decodes an equirectangular Radiance `.hdr` env-map's **bytes** into a
/// linear-RGBA f32 [`EnvMapData`], box-downscaled to fit [`MAX_ENV_DIM`] so it
/// stays within the renderer's portable texture-size limit. Shared by the native
/// `--env` (file bytes) and browser `?env=` (fetched bytes) paths; HDR decoding
/// stays in Rust so trd-core remains I/O-free. The probe is reflected by
/// [`RenderMode::Pbr`](trd_core::RenderMode::Pbr) metallic surfaces.
pub fn decode_env_hdr(bytes: &[u8]) -> Result<EnvMapData, GuiError> {
    let img = image::load_from_memory_with_format(bytes, image::ImageFormat::Hdr)?.to_rgba32f();
    let (width, height) = img.dimensions();
    Ok(EnvMapData::from_rgba32f(
        width,
        height,
        img.into_raw(),
        MAX_ENV_DIM,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_cube_parses() {
        let mesh = default_mesh().expect("built-in cube parses");
        assert!(!mesh.vertices.is_empty() && !mesh.indices.is_empty());
    }

    #[test]
    fn small_texture_is_converted_unchanged() {
        let img = image::DynamicImage::ImageRgba8(image::RgbaImage::from_pixel(
            8,
            8,
            image::Rgba([10, 20, 30, 255]),
        ));
        let tex = texture_from_image(img).expect("small texture converts");
        assert_eq!((tex.width(), tex.height()), (8, 8));
    }

    #[test]
    fn oversized_texture_is_downscaled_within_the_limit() {
        // A thin, over-wide image keeps the test cheap while exercising the
        // downscale branch: the width is clamped to MAX_TEXTURE_DIM.
        let img = image::DynamicImage::ImageRgba8(image::RgbaImage::new(MAX_TEXTURE_DIM + 8, 2));
        let tex = texture_from_image(img).expect("oversized texture converts");
        assert!(tex.width() <= MAX_TEXTURE_DIM && tex.height() <= MAX_TEXTURE_DIM);
        assert_eq!(tex.width(), MAX_TEXTURE_DIM);
    }

    #[test]
    fn env_hdr_round_trips_through_decode() {
        use std::io::Cursor;
        // Encode a tiny Radiance HDR in memory, then decode it back through the
        // shared env-map path. RGBE encoding is lossy, so only the shape (dims +
        // packed RGBA-f32 length) is asserted.
        let src = image::Rgb32FImage::from_pixel(4, 2, image::Rgb([0.5f32, 0.25, 1.0]));
        let mut buf = Cursor::new(Vec::new());
        image::DynamicImage::ImageRgb32F(src)
            .write_to(&mut buf, image::ImageFormat::Hdr)
            .expect("encodes a radiance hdr");
        let env = decode_env_hdr(buf.get_ref()).expect("decodes the hdr bytes");
        assert_eq!((env.width, env.height), (4, 2));
        assert_eq!(env.rgba.len(), 4 * 2 * 4);
    }
}
