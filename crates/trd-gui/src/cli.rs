//! Native command-line arguments for `trd-gui` (#97): render resolution and the
//! input mesh. The mesh is loaded directly into `trd-core`'s canonical [`Mesh`]
//! (OBJ), keeping I/O in the shell so `trd-core` stays I/O-free. When no `--mesh`
//! is given, a small built-in colored cube is used so the viewer runs anywhere
//! without external assets.

use std::path::PathBuf;

use clap::Parser;
use trd_core::{ImageTexture, Mesh};

use crate::error::GuiError;

/// Textures are downscaled to fit this square before upload — `trd-core`'s
/// headless renderer uses wgpu's `downlevel_defaults` limits, whose
/// `max_texture_dimension_2d` is 2048, and the demo albedo maps are 3072².
const MAX_TEXTURE_DIM: u32 = 2048;

/// Which render backend the viewer drives (design §5.2).
#[derive(Copy, Clone, Debug, PartialEq, Eq, Default, clap::ValueEnum)]
pub enum Backend {
    /// Call `trd-core`'s `BatchRenderer` directly (lowest latency; the default).
    #[default]
    Inproc,
    /// Author a `[mesh][params]` Arrow stream → `run_stream` → decode the image
    /// stream back. Identical output to the batch CLI; the seam for external
    /// producers. Higher latency, so it re-renders on interaction end.
    Arrow,
}

/// `trd-gui` — an interactive egui viewer that renders a mesh with `trd-core`
/// and turns orbit/zoom/move gestures into an updated camera/model matrix.
#[derive(Parser, Debug)]
#[command(name = "trd-gui", about, version)]
pub struct Cli {
    /// Render width in pixels (the display scales this to the window).
    #[arg(long, default_value_t = 512)]
    pub width: u32,

    /// Render height in pixels (the display scales this to the window).
    #[arg(long, default_value_t = 512)]
    pub height: u32,

    /// Which render backend to drive.
    #[arg(long, value_enum, default_value_t = Backend::Inproc)]
    pub backend: Backend,

    /// Path to a Wavefront OBJ mesh to view. Defaults to a built-in cube.
    #[arg(long)]
    pub mesh: Option<PathBuf>,

    /// Path to a texture image (PNG/JPEG) bound as the albedo for the
    /// **Textured** render mode. Requires a UV-mapped mesh (e.g.
    /// `assets/meshes/bunny_with_texture/bunny.obj`); without it, Textured mode
    /// samples a flat white default.
    #[arg(long)]
    pub texture: Option<PathBuf>,
}

impl Cli {
    /// Loads the mesh named by `--mesh`, or the built-in default cube.
    pub fn load_mesh(&self) -> Result<Mesh, GuiError> {
        match &self.mesh {
            Some(path) => {
                let text = std::fs::read_to_string(path).map_err(|source| GuiError::MeshIo {
                    path: path.display().to_string(),
                    source,
                })?;
                Ok(Mesh::from_obj(&text)?)
            }
            None => Ok(crate::assets::default_mesh()?),
        }
    }

    /// Loads and decodes the `--texture` image (if any) into an [`ImageTexture`],
    /// downscaling to [`MAX_TEXTURE_DIM`] so it fits the renderer's texture-size
    /// limit. Returns `None` when no texture was requested.
    pub fn load_texture(&self) -> Result<Option<ImageTexture>, GuiError> {
        let Some(path) = &self.texture else {
            return Ok(None);
        };
        let image = image::open(path).map_err(|source| GuiError::TextureIo {
            path: path.display().to_string(),
            source,
        })?;
        Ok(Some(texture_from_image(image)?))
    }
}

/// Downscales `image` to fit [`MAX_TEXTURE_DIM`] (preserving aspect) and converts
/// it to an [`ImageTexture`]. Split out from I/O so it is unit-testable without a
/// file or a GPU.
pub(crate) fn texture_from_image(image: image::DynamicImage) -> Result<ImageTexture, GuiError> {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_mesh_parses() {
        let cli = Cli {
            width: 512,
            height: 512,
            backend: Backend::Inproc,
            mesh: None,
            texture: None,
        };
        let mesh = cli.load_mesh().expect("built-in cube parses");
        assert!(!mesh.vertices.is_empty());
        assert!(!mesh.indices.is_empty());
    }

    #[test]
    fn no_texture_flag_loads_nothing() {
        let cli = Cli {
            width: 256,
            height: 256,
            backend: Backend::Inproc,
            mesh: None,
            texture: None,
        };
        assert!(cli.load_texture().expect("no texture is Ok").is_none());
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
}
