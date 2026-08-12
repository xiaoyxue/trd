//! trd-cli: native headless entry point.
//!
//! Reads an Arrow IPC scene stream on stdin and writes an Arrow
//! IPC stream of rendered images on stdout (trd protocol 0.0.6).

use std::io::{self, Write};
use std::path::PathBuf;

use clap::{Parser, ValueEnum};

/// Coordinate plane selector for the `--grid-local` overlay, mapped to
/// [`trd_core::GridPlane`]. Kept in the binary so `trd-core` needn't depend on
/// clap.
#[derive(Clone, Copy, Debug, ValueEnum)]
enum GridPlaneArg {
    /// The local XY plane (Z = 0) — e.g. a placement quad's floor.
    Xy,
    /// The local XZ plane (Y = 0).
    Xz,
    /// The local YZ plane (X = 0).
    Yz,
}

impl From<GridPlaneArg> for trd_core::GridPlane {
    fn from(value: GridPlaneArg) -> Self {
        match value {
            GridPlaneArg::Xy => trd_core::GridPlane::Xy,
            GridPlaneArg::Xz => trd_core::GridPlane::Xz,
            GridPlaneArg::Yz => trd_core::GridPlane::Yz,
        }
    }
}

/// PBR tone-map operator selector, mapped to [`trd_core::Tonemap`]. Kept in the
/// binary so `trd-core` needn't depend on clap.
#[derive(Clone, Copy, Debug, ValueEnum)]
enum TonemapArg {
    /// Per-channel Reinhard `x/(1+x)` — the default.
    Reinhard,
    /// ACES filmic tone map (Narkowicz RRT+ODT fit).
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

/// Streaming Arrow renderer for trd (protocol 0.0.6).
#[derive(Parser)]
#[command(name = "trd", version, about)]
struct Cli {
    /// Output image width in pixels.
    #[arg(long, default_value_t = 256, value_parser = clap::value_parser!(u32).range(1..))]
    width: u32,
    /// Output image height in pixels.
    #[arg(long, default_value_t = 256, value_parser = clap::value_parser!(u32).range(1..))]
    height: u32,
    /// Render meshes as an edge wireframe (line list) instead of filled
    /// triangles.
    #[arg(long)]
    wireframe: bool,
    /// Render meshes textured — sampling the stream's bound texture table at each
    /// vertex UV — instead of the per-vertex color. Requires a stream
    /// carrying a texture table (else the bound texture is 1×1 white).
    #[arg(long, conflicts_with = "wireframe")]
    textured: bool,
    /// Render meshes with the physically-based **Disney principled BRDF**
    /// (`disney.wgsl`): the bound albedo lit by a virtual light rig plus an
    /// optional HDR environment-map reflection (`--env`), with smooth shading
    /// normals. Use `--metallic 1 --roughness 0.3` (or the defaults) for a shiny
    /// metal look. Requires a stream carrying a texture table for the albedo.
    #[arg(long, conflicts_with_all = ["wireframe", "textured"])]
    pbr: bool,
    /// Equirectangular HDR environment map (Radiance `.hdr`) reflected by
    /// metallic PBR surfaces. Decoded here (trd-core does no file I/O) and
    /// downscaled to the renderer's 2048px limit. Only used with `--pbr`.
    #[arg(long, value_name = "FILE")]
    env: Option<PathBuf>,
    /// PBR metallic parameter (0 = dielectric, 1 = metal).
    #[arg(long, default_value_t = 0.0)]
    metallic: f32,
    /// PBR surface roughness (0 = mirror, 1 = fully rough).
    #[arg(long, default_value_t = 0.35)]
    roughness: f32,
    /// PBR dielectric specular reflectance strength (`0.5` ≈ 4% F0).
    #[arg(long, default_value_t = 0.5)]
    specular: f32,
    /// PBR clearcoat lobe strength (a second colorless specular layer).
    #[arg(long, default_value_t = 0.0)]
    clearcoat: f32,
    /// PBR environment-map reflection gain (0 disables the probe reflection).
    #[arg(long, default_value_t = 1.0)]
    env_intensity: f32,
    /// PBR tone-map exposure applied before the Reinhard curve.
    #[arg(long, default_value_t = 1.2)]
    exposure: f32,
    /// PBR constant ambient fill (× base color) so shadows are not pure black.
    #[arg(long, default_value_t = 0.12)]
    ambient: f32,
    /// PBR tone-map operator: `reinhard` (per-channel `x/(1+x)`, the default) or
    /// `aces` (filmic — softer highlight roll-off and better hue retention for
    /// bright albedo).
    #[arg(long, value_enum, default_value_t = TonemapArg::Reinhard)]
    tonemap: TonemapArg,
    /// Overlay each drawn mesh's axis-aligned bounding box as a green
    /// wireframe box.
    #[arg(long)]
    aabb: bool,
    /// Overlay a coordinate-axes gizmo (X=red, Y=green, Z=blue) at the world
    /// origin.
    #[arg(long)]
    axes: bool,
    /// Overlay a coordinate-axes gizmo at *each* drawn object's local frame (its
    /// `model`), i.e. its model-space X/Y/Z axes as placed — e.g. #77's
    /// `(e1,e2,e3)` quad frame the bunny is anchored in.
    #[arg(long)]
    axes_local: bool,
    /// Overlay a coordinate-plane grid lattice on the given plane at each
    /// *wireframe* drawn object's local frame (its `model`) — e.g. `--grid-local
    /// xy` tiles a grid across the placement quad's local floor. Scoped to
    /// wireframe draws, so a filled/textured mesh gets no stray grid. One of
    /// `xy`, `xz`, `yz`.
    #[arg(long, value_enum)]
    grid_local: Option<GridPlaneArg>,
    /// Narrow `--grid-local` to draws of this `mesh_id` only (the placement
    /// quad). Use when a *content* mesh is also drawn wireframe (e.g. a
    /// wireframe-reveal intro) so the floor grid lands only under the quad, not
    /// under every wireframe object. Ignored without `--grid-local`.
    #[arg(long, value_name = "MESH_ID")]
    grid_mesh: Option<u32>,
    /// Base directory for external per-frame background images. When set, a
    /// frame's `frame_path` (relative, e.g. `frames/frame_000000.png`) is joined
    /// to this dir, decoded (PNG/JPEG), and composited beneath the scene as a
    /// background frame plane. Without it, `frame_path` columns are ignored.
    #[arg(long, value_name = "DIR")]
    frames_base: Option<PathBuf>,
    /// Disable 4× MSAA on the mesh pass, rendering single-sampled (aliased
    /// wireframe / gizmo / silhouette edges). By default the mesh pass is
    /// anti-aliased at 4×.
    #[arg(long)]
    no_msaa: bool,
}

fn main() -> Result<(), trd_core::StreamError> {
    env_logger::Builder::from_env(
        env_logger::Env::default().default_filter_or("warn,trd_core=info"),
    )
    .init();

    let cli = Cli::parse();
    let mode = if cli.pbr {
        trd_core::RenderMode::Shaded
    } else if cli.textured {
        trd_core::RenderMode::Textured
    } else if cli.wireframe {
        trd_core::RenderMode::Wireframe
    } else {
        trd_core::RenderMode::Filled
    };

    // Assemble the Disney PBR config (material + optional HDR environment probe)
    // when `--pbr` is set. The `.hdr` file is decoded here so trd-core does no
    // file/codec I/O; it is downscaled to the renderer's portable 2048px limit.
    let pbr = if cli.pbr {
        let material = trd_core::DisneyMaterial {
            metallic: cli.metallic,
            roughness: cli.roughness,
            specular: cli.specular,
            clearcoat: cli.clearcoat,
            ..Default::default()
        };
        let lighting = trd_core::Lighting {
            ambient: cli.ambient,
            ..Default::default()
        };
        let ibl = trd_core::ImageBasedLighting {
            intensity: cli.env_intensity,
            ..trd_core::ImageBasedLighting::default()
        };
        let tone_mapping = trd_core::ToneMapping {
            operator: cli.tonemap.into(),
            exposure: cli.exposure,
        };
        let env_map = match cli.env.as_ref() {
            Some(path) => Some(load_env_map(path)?),
            None => None,
        };
        Some(trd_core::PbrConfig {
            material,
            lighting,
            ibl,
            tone_mapping,
            env_map,
        })
    } else {
        None
    };

    let stdin = io::stdin().lock();
    let stdout = io::stdout().lock();

    // The background frame resolver (#63): decodes a per-frame `frame_path`
    // (relative to `--frames-base`) into RGBA. Kept in the shell so trd-core does
    // no file/image I/O. Only present when `--frames-base` is given.
    let resolver = cli.frames_base.clone().map(|base| {
        move |rel: &str| -> Option<trd_core::ImageData> {
            let path = base.join(rel);
            match image::open(&path) {
                Ok(img) => {
                    let rgba = img.to_rgba8();
                    let (width, height) = rgba.dimensions();
                    Some(trd_core::ImageData {
                        width,
                        height,
                        rgba: rgba.into_raw(),
                    })
                }
                Err(err) => {
                    log::warn!("skipping frame background {}: {err}", path.display());
                    None
                }
            }
        }
    });
    let frame_resolver: Option<trd_core::FrameResolver> = resolver
        .as_ref()
        .map(|r| r as &dyn Fn(&str) -> Option<trd_core::ImageData>);

    trd_core::run_stream(
        stdin,
        stdout,
        cli.width,
        cli.height,
        trd_core::RenderOptions {
            mode,
            show_aabb: cli.aabb,
            show_axes: cli.axes,
            show_local_axes: cli.axes_local,
            show_local_grid: cli.grid_local.map(Into::into),
            show_local_grid_mesh: cli.grid_mesh,
            // The CLI exposes no world/object grid or selection flags; those are
            // interactive-only overlays.
            show_world_grid: None,
            show_object_grid: None,
            selected: None,
            pbr,
            msaa: if cli.no_msaa {
                trd_core::Msaa::Off
            } else {
                trd_core::Msaa::X4
            },
        },
        frame_resolver,
    )?;
    io::stdout().flush()?;
    Ok(())
}

/// Decodes an equirectangular Radiance `.hdr` file into a linear-RGBA f32
/// [`trd_core::EnvMapData`], downscaled (integer box filter) so neither
/// dimension exceeds the renderer's portable 2048px texture limit. Kept in the
/// CLI shell so trd-core does no file/codec I/O.
fn load_env_map(path: &std::path::Path) -> Result<trd_core::EnvMapData, trd_core::StreamError> {
    let img = image::open(path)
        .map_err(|e| {
            trd_core::StreamError::Render(format!("read env map {}: {e}", path.display()))
        })?
        .to_rgba32f();
    let (w, h) = img.dimensions();
    Ok(trd_core::EnvMapData::from_rgba32f(
        w,
        h,
        img.into_raw(),
        2048,
    ))
}
