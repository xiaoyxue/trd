//! Native command-line arguments for `trd-gui` (#97): render resolution and the
//! input mesh. The mesh is loaded directly into `trd-core`'s canonical [`Mesh`]
//! (OBJ), keeping I/O in the shell so `trd-core` stays I/O-free. When no `--mesh`
//! is given, a small built-in colored cube is used so the viewer runs anywhere
//! without external assets.

use std::path::PathBuf;

use clap::Parser;
use trd_core::{
    DisneyMaterial, EnvMapData, ImageBasedLighting, Lighting, Mesh, RenderMode, ToneMapping,
};

use trd_gui::error::GuiError;
use trd_gui::scene::{SceneSeed, SceneState};

/// Which render backend the viewer drives (design §5.2).
#[derive(Copy, Clone, Debug, PartialEq, Eq, Default, clap::ValueEnum)]
pub enum Backend {
    /// Call `trd-core`'s `Renderer` directly (lowest latency; the default).
    #[default]
    Inproc,
    /// Author a `[mesh][params]` Arrow stream → `run_stream` → decode the image
    /// stream back. Identical output to the batch CLI; the seam for external
    /// producers. Higher latency, so it re-renders on interaction end.
    Arrow,
}

/// PBR tone-map operator selector, mapped to [`trd_core::Tonemap`]. Kept in the
/// CLI layer so `clap`'s `ValueEnum` derive stays out of `trd-core`.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Default, clap::ValueEnum)]
pub enum TonemapArg {
    /// Per-channel Reinhard `x/(1+x)` (the default).
    #[default]
    Reinhard,
    /// ACES filmic tone map (softer highlight roll-off).
    Aces,
}

impl From<TonemapArg> for trd_core::Tonemap {
    fn from(value: TonemapArg) -> Self {
        match value {
            TonemapArg::Reinhard => trd_core::Tonemap::Reinhard,
            TonemapArg::Aces => trd_core::Tonemap::Aces,
        }
    }
}

/// `trd-gui` — an interactive egui viewer that renders a mesh with `trd-core`
/// and turns orbit/zoom/move gestures into an updated camera/model matrix.
#[derive(Parser, Debug)]
#[command(name = "trd-gui", bin_name = "trd-gui", about, version)]
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

    /// Start in the Disney **PBR** render mode (equivalent to selecting "PBR" in
    /// the UI dropdown). The material is then editable live via the side-panel
    /// sliders; pair with `--env` for environment reflections.
    #[arg(long)]
    pub pbr: bool,

    /// Equirectangular HDR environment map (Radiance `.hdr`) reflected by
    /// metallic PBR surfaces. Decoded here (trd-core does no file I/O) and
    /// downscaled to the renderer's 2048px limit. Bound once; used by PBR mode.
    #[arg(long, value_name = "FILE")]
    pub env: Option<PathBuf>,

    /// Initial PBR metallic parameter (0 = dielectric, 1 = metal). Editable live.
    #[arg(long, default_value_t = 0.0)]
    pub metallic: f32,

    /// Initial PBR surface roughness (0 = mirror, 1 = fully rough). Editable live.
    #[arg(long, default_value_t = 0.35)]
    pub roughness: f32,

    /// Initial PBR dielectric specular reflectance strength (`0.5` ≈ 4% F0).
    #[arg(long, default_value_t = 0.5)]
    pub specular: f32,

    /// Initial PBR clearcoat lobe strength (a second colorless specular layer).
    #[arg(long, default_value_t = 0.0)]
    pub clearcoat: f32,

    /// Initial PBR environment-map reflection gain (0 disables the probe).
    #[arg(long, default_value_t = 1.0)]
    pub env_intensity: f32,

    /// Initial PBR tone-map exposure applied before the tone-map curve.
    #[arg(long, default_value_t = 1.2)]
    pub exposure: f32,

    /// Initial PBR constant ambient fill (× base color) so shadows aren't black.
    #[arg(long, default_value_t = 0.12)]
    pub ambient: f32,

    /// PBR tone-map operator: `reinhard` (per-channel `x/(1+x)`, the default) or
    /// `aces` (filmic — softer highlight roll-off). Editable live.
    #[arg(long, value_enum, default_value_t = TonemapArg::Reinhard)]
    pub tonemap: TonemapArg,
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
            None => Ok(trd_gui::assets::default_mesh()?),
        }
    }

    /// Loads and decodes the `--texture` image (if any) into an [`ImageTexture`]
    /// via [`crate::assets::decode_texture`]. Returns `None` when no texture was
    /// requested.
    pub fn load_texture(&self) -> Result<Option<trd_core::ImageTexture>, GuiError> {
        let Some(path) = &self.texture else {
            return Ok(None);
        };
        let bytes = std::fs::read(path).map_err(|source| GuiError::TextureIo {
            path: path.display().to_string(),
            source,
        })?;
        Ok(Some(trd_gui::assets::decode_texture(&bytes)?))
    }

    /// Loads and decodes the `--env` HDR probe (if any) into an [`EnvMapData`] via
    /// [`crate::assets::decode_env_hdr`]. Returns `None` when no env was requested.
    /// The probe is bound once on the renderer and reflected by PBR metals.
    pub fn load_env(&self) -> Result<Option<EnvMapData>, GuiError> {
        let Some(path) = &self.env else {
            return Ok(None);
        };
        let bytes = std::fs::read(path).map_err(|source| GuiError::EnvIo {
            path: path.display().to_string(),
            source,
        })?;
        Ok(Some(trd_gui::assets::decode_env_hdr(&bytes)?))
    }

    /// The initial Disney PBR material assembled from the material flags
    /// (`--metallic`, `--roughness`, …). The UI edits a live copy from here on.
    pub fn disney_material(&self) -> DisneyMaterial {
        DisneyMaterial {
            metallic: self.metallic,
            roughness: self.roughness,
            specular: self.specular,
            clearcoat: self.clearcoat,
            ..DisneyMaterial::default()
        }
    }

    pub fn image_based_lighting(&self) -> ImageBasedLighting {
        ImageBasedLighting {
            intensity: self.env_intensity,
            ..ImageBasedLighting::default()
        }
    }

    pub fn tone_mapping(&self) -> ToneMapping {
        ToneMapping {
            operator: self.tonemap.into(),
            exposure: self.exposure,
        }
    }

    pub fn lighting(&self) -> Lighting {
        Lighting {
            ambient: self.ambient,
            ..Lighting::default()
        }
    }

    /// The initial [`SceneState`]: the default camera/object with the render mode
    /// set to [`RenderMode::Pbr`] when `--pbr` is given, carrying the material
    /// assembled from the CLI flags (subsequently edited live in the UI).
    pub fn scene_state(&self) -> SceneState {
        // The native CLI authors a single object, so the seed carries exactly one
        // material; `seeded` keeps every per-object vector that length.
        SceneState::seeded(SceneSeed {
            materials: vec![self.disney_material()],
            mode: if self.pbr {
                RenderMode::Pbr
            } else {
                RenderMode::Filled
            },
            image_based_lighting: self.image_based_lighting(),
            tone_mapping: self.tone_mapping(),
            lighting: self.lighting(),
            environment_available: false,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_mesh_parses() {
        let cli = Cli::parse_from(["trd-gui"]);
        let mesh = cli.load_mesh().expect("built-in cube parses");
        assert!(!mesh.vertices.is_empty());
        assert!(!mesh.indices.is_empty());
    }

    #[test]
    fn no_texture_flag_loads_nothing() {
        let cli = Cli::parse_from(["trd-gui"]);
        assert!(cli.load_texture().expect("no texture is Ok").is_none());
        assert!(cli.load_env().expect("no env is Ok").is_none());
    }

    #[test]
    fn pbr_flag_selects_pbr_mode_and_material() {
        let cli = Cli::parse_from(["trd-gui", "--pbr", "--metallic", "1", "--roughness", "0.3"]);
        let state = cli.scene_state();
        assert_eq!(state.modes[0], RenderMode::Pbr);
        assert_eq!(state.materials[0].metallic, 1.0);
        assert_eq!(state.materials[0].roughness, 0.3);
    }

    #[test]
    fn no_pbr_flag_keeps_filled_mode() {
        let cli = Cli::parse_from(["trd-gui"]);
        assert_eq!(cli.scene_state().modes[0], RenderMode::Filled);
    }
}
