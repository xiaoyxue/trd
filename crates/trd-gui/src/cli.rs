//! Native command-line arguments for `trd-gui` (#97): render resolution and the
//! input mesh. The mesh is loaded directly into `trd-core`'s canonical [`Mesh`]
//! (OBJ), keeping I/O in the shell so `trd-core` stays I/O-free. When no `--mesh`
//! is given, a small built-in colored cube is used so the viewer runs anywhere
//! without external assets.

use std::path::PathBuf;

use clap::Parser;
use trd_core::Mesh;

use crate::error::GuiError;

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
        };
        let mesh = cli.load_mesh().expect("built-in cube parses");
        assert!(!mesh.vertices.is_empty());
        assert!(!mesh.indices.is_empty());
    }
}
