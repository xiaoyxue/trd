//! Command-line arguments for the native stream viewer.

use std::path::PathBuf;

use clap::{Parser, ValueEnum};

/// Coordinate plane selector for the `--grid-local` overlay, mapped to
/// [`trd_core::GridPlane`]. Kept in the binary so `trd-core` needn't depend on
/// clap.
#[derive(Clone, Copy, Debug, ValueEnum)]
pub(crate) enum GridPlaneArg {
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

/// Interactive desktop viewer for a trd scene stream (protocol 0.0.5).
///
/// Reads the Arrow IPC `[mesh][texture?][params]` stream on stdin — a leading
/// mesh table (then an optional texture table) followed by per-frame params +
/// instanced draw lists — and plays it live in a window, e.g.
/// `trd-render.sh --mesh bunny.obj … | trd-app`.
#[derive(Parser)]
#[command(name = "trd-app", version, about)]
pub(crate) struct Cli {
    /// Initial window width in logical pixels.
    #[arg(long, default_value_t = 800, value_parser = clap::value_parser!(u32).range(1..))]
    pub(crate) width: u32,
    /// Initial window height in logical pixels.
    #[arg(long, default_value_t = 600, value_parser = clap::value_parser!(u32).range(1..))]
    pub(crate) height: u32,
    /// Playback frame rate (frames per second): sets both the animation speed
    /// (higher = faster) and the present rate. When omitted, the stream's
    /// declared rate (`trd.stream.frame_rate` metadata, default 30) is used.
    #[arg(long)]
    pub(crate) fps: Option<f64>,
    /// Play the stream once and hold the last frame instead of looping.
    #[arg(long)]
    pub(crate) once: bool,
    /// Lock presentation to the monitor refresh (vsync). By default the app
    /// presents at `--fps` decoupled from the refresh rate (non-vsync).
    #[arg(long)]
    pub(crate) vsync: bool,
    /// Render meshes as an edge wireframe (line list) instead of filled
    /// triangles (#38).
    #[arg(long)]
    pub(crate) wireframe: bool,
    /// Render meshes textured — sampling the stream's bound texture table at
    /// each vertex UV — instead of the per-vertex color (#20). Requires a
    /// stream carrying a texture table (else the bound texture is 1×1
    /// white).
    #[arg(long, conflicts_with = "wireframe")]
    pub(crate) textured: bool,
    /// Overlay each drawn mesh's axis-aligned bounding box as a green wireframe
    /// box (#42).
    #[arg(long)]
    pub(crate) aabb: bool,
    /// Overlay a coordinate-axes gizmo (X=red, Y=green, Z=blue) at the world
    /// origin (#42).
    #[arg(long)]
    pub(crate) axes: bool,
    /// Overlay a coordinate-axes gizmo at *each* drawn object's local frame (its
    /// `model`), i.e. its model-space X/Y/Z axes as placed — e.g. #77's
    /// `(e1,e2,e3)` quad frame the bunny is anchored in.
    #[arg(long)]
    pub(crate) axes_local: bool,
    /// Overlay a coordinate-plane grid lattice on the given plane at each
    /// *wireframe* drawn object's local frame (its `model`) — e.g. `--grid-local
    /// xy` tiles a grid across the placement quad's local floor. Scoped to
    /// wireframe draws, so a filled/textured mesh gets no stray grid. One of
    /// `xy`, `xz`, `yz`.
    #[arg(long, value_enum)]
    pub(crate) grid_local: Option<GridPlaneArg>,
    /// Base directory for per-frame background images (`0.0.5`, #63). When set, a
    /// frame's `frame_path` (relative) is joined to this dir, decoded (PNG/JPEG),
    /// and composited beneath the scene as a background frame plane. Without it,
    /// `frame_path` columns are ignored.
    #[arg(long, value_name = "DIR")]
    pub(crate) frames_base: Option<PathBuf>,
}
