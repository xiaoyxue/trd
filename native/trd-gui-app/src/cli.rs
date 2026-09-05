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

/// The probe the viewer falls back to with no `--env`.
///
/// The viewer is lit by image-based lighting alone, so a probe is not optional
/// the way it was when a key/fill/rim rig lit the scene — this is the same
/// Uffizi probe the video editor lights the Dragon with.
pub const DEFAULT_ENV_PATH: &str = "assets/envmap/uffizi-large.hdr";

/// What `--mesh` produced: the geometry, plus the material and maps a glTF
/// binary carries with it (an OBJ carries neither).
pub struct LoadedMesh {
    pub mesh: Mesh,
    /// The GLB's imported material; `None` for an OBJ, which uses the flags.
    pub material: Option<DisneyMaterial>,
    pub base_color: Option<trd_core::ImageTexture>,
    pub metallic_roughness: Option<trd_core::ImageTexture>,
    pub normal: Option<trd_core::ImageTexture>,
}

impl LoadedMesh {
    fn plain(mesh: Mesh) -> Self {
        Self {
            mesh,
            material: None,
            base_color: None,
            metallic_roughness: None,
            normal: None,
        }
    }
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
    /// Defaults to `1.0` for a glTF asset and `1.2` otherwise.
    #[arg(long)]
    pub exposure: Option<f32>,

    /// Initial PBR constant ambient fill (× base color) so shadows aren't black.
    /// Defaults to `0.0` for a glTF asset lit by `--env` — which is lit by the
    /// probe alone — and `0.12` otherwise.
    #[arg(long)]
    pub ambient: Option<f32>,

    /// PBR tone-map operator: `reinhard` (per-channel `x/(1+x)`) or `aces`
    /// (filmic — softer highlight roll-off). Defaults to `aces` for a glTF asset
    /// and `reinhard` otherwise. Editable live.
    #[arg(long, value_enum)]
    pub tonemap: Option<TonemapArg>,
}

impl Cli {
    /// Loads `--mesh`, sniffing GLB's `glTF` magic exactly as the browser's
    /// `?mesh=` does — a glTF binary brings its own material and maps, an OBJ is
    /// parsed as UTF-8 text. Falls back to the built-in cube with no `--mesh`.
    pub fn load_mesh(&self) -> Result<LoadedMesh, GuiError> {
        let Some(path) = &self.mesh else {
            return Ok(LoadedMesh::plain(trd_gui::assets::default_mesh()?));
        };
        let bytes = std::fs::read(path).map_err(|source| GuiError::MeshIo {
            path: path.display().to_string(),
            source,
        })?;
        if bytes.starts_with(b"glTF") {
            let name = path.display().to_string();
            let asset = trd_gui::model::decode_glb(&name, &bytes)?;
            return Ok(LoadedMesh {
                mesh: asset.mesh,
                material: Some(asset.material),
                base_color: asset.base_color_texture,
                metallic_roughness: asset.metallic_roughness_texture,
                normal: asset.normal_texture,
            });
        }
        let text = String::from_utf8(bytes).map_err(|error| GuiError::MeshIo {
            path: path.display().to_string(),
            source: std::io::Error::new(std::io::ErrorKind::InvalidData, error),
        })?;
        Ok(LoadedMesh::plain(Mesh::from_obj(&text)?))
    }

    /// Loads and decodes the `--texture` image (if any) into an [`ImageTexture`](trd_core::ImageTexture)
    /// via [`trd_gui::assets::decode_texture`]. Returns `None` when no texture was
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

    /// Loads the `--env` probe, falling back to the built-in Uffizi one.
    ///
    /// The viewer is lit by the probe alone, so it always needs one: without a
    /// fallback, opening a model with no `--env` would render it black. Returns
    /// `None` only if the built-in probe cannot be read either.
    pub fn load_env(&self) -> Result<Option<EnvMapData>, GuiError> {
        let Some(path) = &self.env else {
            return Ok(match std::fs::read(DEFAULT_ENV_PATH) {
                Ok(bytes) => Some(trd_gui::assets::decode_env_hdr(&bytes)?),
                Err(error) => {
                    log::warn!(
                        "no --env given and the built-in probe {DEFAULT_ENV_PATH} \
                         could not be read ({error}); PBR surfaces will be unlit"
                    );
                    None
                }
            });
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
        }
    }

    /// The output transform, defaulting to the glTF grade (ACES at unit
    /// exposure) for an asset that brought its own material — which is what
    /// `?mesh=<glb>` already seeds in the browser, and what keeps the two
    /// front-ends showing the same colours.
    pub fn tone_mapping(&self, gltf: bool) -> ToneMapping {
        ToneMapping {
            operator: self
                .tonemap
                .unwrap_or(if gltf {
                    TonemapArg::Aces
                } else {
                    TonemapArg::Reinhard
                })
                .into(),
            exposure: self.exposure.unwrap_or(if gltf { 1.0 } else { 1.2 }),
        }
    }

    /// The scene light rig — always **image-based only** in the viewer.
    ///
    /// The key/fill/rim rig double-lights a PBR surface that is already lit by
    /// the probe, which is what washed the Dragon out. `--ambient` still adds a
    /// constant fill for anyone who wants one; nothing adds a virtual *light*.
    pub fn lighting(&self) -> Lighting {
        Lighting {
            ambient: self.ambient.unwrap_or(0.0),
            scale: 0.0,
            ..Lighting::default()
        }
    }

    /// The initial [`SceneState`] for `loaded`: [`RenderMode::Shaded`] when
    /// `--pbr` is given **or** the asset is a glTF binary (which brings a real
    /// material and maps, exactly as `?mesh=<glb>` does in the browser), carrying
    /// the GLB's imported material when there is one and the flag-assembled one
    /// otherwise.
    pub fn scene_state(
        &self,
        loaded: &LoadedMesh,
        has_env: bool,
        mesh_ids: &[trd_core::MeshId],
    ) -> Result<SceneState, GuiError> {
        let imported = loaded.material.clone();
        let gltf = imported.is_some();
        SceneState::seeded(
            mesh_ids,
            SceneSeed {
                materials: vec![imported.unwrap_or_else(|| self.disney_material())],
                mode: if self.pbr || gltf {
                    RenderMode::Shaded
                } else {
                    RenderMode::Filled
                },
                image_based_lighting: self.image_based_lighting(),
                tone_mapping: self.tone_mapping(gltf),
                lighting: self.lighting(),
                // A probe is always bound now (built-in when `--env` is absent), so
                // the side panel's environment controls are always live.
                environment_available: has_env,
                // ...but the sky stays off until asked for: the probe is there to
                // light the model, not to become the backdrop.
                show_environment_background: false,
            },
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scene(cli: &Cli, loaded: &LoadedMesh, has_env: bool) -> SceneState {
        let ids = trd_core::test_support::mesh_ids(1);
        cli.scene_state(loaded, has_env, &ids)
            .expect("matching material")
    }

    #[test]
    fn default_mesh_parses() {
        let cli = Cli::parse_from(["trd-gui"]);
        let loaded = cli.load_mesh().expect("built-in cube parses");
        assert!(!loaded.mesh.vertices.is_empty());
        assert!(!loaded.mesh.indices.is_empty());
        assert!(
            loaded.material.is_none(),
            "an OBJ brings no material of its own"
        );
    }

    #[test]
    fn no_texture_flag_loads_nothing() {
        let cli = Cli::parse_from(["trd-gui"]);
        assert!(cli.load_texture().expect("no texture is Ok").is_none());
    }

    /// The viewer is lit by the probe alone, so "no `--env`" must still produce
    /// one — otherwise every PBR surface would render black.
    #[test]
    fn no_env_flag_falls_back_to_the_built_in_probe() {
        let cli = Cli::parse_from(["trd-gui"]);
        assert!(
            cli.env.is_none(),
            "nothing was asked for on the command line"
        );
        // Resolved relative to the repo root, which is where the viewer runs
        // from; when it is not present the fallback degrades to `None` rather
        // than failing, so only the path is asserted here.
        assert!(
            std::path::Path::new(DEFAULT_ENV_PATH)
                .extension()
                .is_some_and(|e| e == "hdr"),
            "the built-in probe is a Radiance .hdr"
        );
    }

    #[test]
    fn pbr_flag_selects_pbr_mode_and_material() {
        let cli = Cli::parse_from(["trd-gui", "--pbr", "--metallic", "1", "--roughness", "0.3"]);
        let state = scene(
            &cli,
            &cli.load_mesh().expect("the built-in cube parses"),
            true,
        );
        assert_eq!(state.objects[0].mode, RenderMode::Shaded);
        assert_eq!(state.objects[0].appearance.material.metallic, 1.0);
        assert_eq!(state.objects[0].appearance.material.roughness, 0.3);
    }

    #[test]
    fn no_pbr_flag_keeps_filled_mode() {
        let cli = Cli::parse_from(["trd-gui"]);
        assert_eq!(
            scene(
                &cli,
                &cli.load_mesh().expect("the built-in cube parses"),
                true
            )
            .objects[0]
                .mode,
            RenderMode::Filled
        );
    }

    /// A bound probe enables the side panel's environment controls, but the sky
    /// stays off until it is ticked — the viewer always binds one, so coupling
    /// the two would make the backdrop unavoidable.
    #[test]
    fn a_bound_probe_makes_the_environment_background_available() {
        let cli = Cli::parse_from(["trd-gui", "--env", "probe.hdr"]);
        let loaded = cli.load_mesh().expect("the built-in cube parses");
        let state = scene(&cli, &loaded, true);
        assert!(state.environment_available, "the controls are live");
        assert!(
            !state.show_environment_background,
            "but the sky is not drawn until asked for"
        );

        // Only a probe that could not be read at all leaves the controls off.
        let state = scene(&cli, &loaded, false);
        assert!(!state.environment_available);
        assert!(!state.show_environment_background);
    }

    /// Nothing in the viewer adds a virtual light: the rig is the probe alone.
    #[test]
    fn the_light_rig_is_image_based_only() {
        let cli = Cli::parse_from(["trd-gui"]);
        let rig = cli.lighting();
        assert_eq!(rig.scale, 0.0, "no key/fill/rim rig");
        assert_eq!(rig.ambient, 0.0, "and no ambient fill by default");

        // `--ambient` is still an explicit override, and still adds no *light*.
        let cli = Cli::parse_from(["trd-gui", "--ambient", "0.05"]);
        let rig = cli.lighting();
        assert_eq!(rig.ambient, 0.05);
        assert_eq!(rig.scale, 0.0);
    }

    /// A glTF asset is graded like the browser grades `?mesh=<glb>`.
    #[test]
    fn a_gltf_asset_defaults_to_the_aces_grade() {
        let cli = Cli::parse_from(["trd-gui"]);
        let gltf = cli.tone_mapping(true);
        assert_eq!(gltf.operator, trd_core::Tonemap::Aces);
        assert_eq!(gltf.exposure, 1.0);

        let obj = cli.tone_mapping(false);
        assert_eq!(obj.operator, trd_core::Tonemap::Reinhard);
        assert_eq!(obj.exposure, 1.2);

        // An explicit flag still wins over both.
        let cli = Cli::parse_from(["trd-gui", "--exposure", "0.45"]);
        assert_eq!(cli.tone_mapping(true).exposure, 0.45);
    }
}
