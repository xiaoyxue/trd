//! trd-cli: native headless entry point.
//!
//! Reads an Arrow IPC scene stream on stdin and writes an Arrow
//! IPC stream of rendered images on stdout (trd protocol 0.0.5).

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

/// Streaming Arrow renderer for trd (protocol 0.0.5).
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
    /// Overlay a coordinate-plane grid lattice on the given plane at *each* drawn
    /// object's local frame (its `model`) — e.g. `--grid-local xy` tiles a grid
    /// across a placement quad's local floor. One of `xy`, `xz`, `yz`.
    #[arg(long, value_enum)]
    grid_local: Option<GridPlaneArg>,
    /// Base directory for per-frame background images (`0.0.5`). When set, a
    /// frame's `frame_path` (relative, e.g. `frames/frame_000000.png`) is joined
    /// to this dir, decoded (PNG/JPEG), and composited beneath the scene as a
    /// background frame plane. Without it, `frame_path` columns are ignored.
    #[arg(long, value_name = "DIR")]
    frames_base: Option<PathBuf>,
}

fn main() -> Result<(), trd_core::StreamError> {
    env_logger::Builder::from_env(
        env_logger::Env::default().default_filter_or("warn,trd_core=info"),
    )
    .init();

    let cli = Cli::parse();
    let mode = if cli.textured {
        trd_core::RenderMode::Textured
    } else if cli.wireframe {
        trd_core::RenderMode::Wireframe
    } else {
        trd_core::RenderMode::Filled
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
        },
        frame_resolver,
    )?;
    io::stdout().flush()?;
    Ok(())
}
