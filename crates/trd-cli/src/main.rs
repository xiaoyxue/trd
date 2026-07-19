//! trd-cli: native headless entry point.
//!
//! Reads an Arrow IPC stream of per-frame params on stdin and writes an Arrow
//! IPC stream of rendered images on stdout (trd protocol 0.0.1).

use std::io::{self, Write};

use clap::Parser;

/// Streaming Arrow renderer for trd (protocol 0.0.1).
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
    /// Overlay each drawn mesh's axis-aligned bounding box as a green
    /// wireframe box.
    #[arg(long)]
    aabb: bool,
    /// Overlay a coordinate-axes gizmo (X=red, Y=green, Z=blue) at the world
    /// origin.
    #[arg(long)]
    axes: bool,
}

fn main() -> Result<(), trd_core::StreamError> {
    env_logger::Builder::from_env(
        env_logger::Env::default().default_filter_or("warn,trd_core=info"),
    )
    .init();

    let cli = Cli::parse();
    let mode = if cli.wireframe {
        trd_core::RenderMode::Wireframe
    } else {
        trd_core::RenderMode::Filled
    };
    let stdin = io::stdin().lock();
    let stdout = io::stdout().lock();
    trd_core::run_stream(
        stdin, stdout, cli.width, cli.height, mode, cli.aabb, cli.axes,
    )?;
    io::stdout().flush()?;
    Ok(())
}
