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

/// A built-in origin-centered unit cube with per-corner colors, used as the
/// default object when no `--mesh` is supplied (`v x y z r g b` OBJ extension).
const DEFAULT_MESH_OBJ: &str = "\
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
            None => Ok(Mesh::from_obj(DEFAULT_MESH_OBJ)?),
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
        Ok(Some(ImageTexture::from_rgba(
            width,
            height,
            rgba.into_raw(),
        )?))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_mesh_parses() {
        let cli = Cli {
            width: 512,
            height: 512,
            mesh: None,
            texture: None,
        };
        let mesh = cli.load_mesh().expect("built-in cube parses");
        assert!(!mesh.vertices.is_empty());
        assert!(!mesh.indices.is_empty());
    }
}
