//! trd-cli: native headless entry point for the trd rendering core.

use std::path::PathBuf;

use clap::Parser;

/// Headless renderer for trd. Renders the hello-triangle to a PNG file.
#[derive(Parser)]
#[command(name = "trd", version, about)]
struct Cli {
    /// Output image width in pixels.
    #[arg(long, default_value_t = 512)]
    width: u32,
    /// Output image height in pixels.
    #[arg(long, default_value_t = 512)]
    height: u32,
    /// Path to write the PNG file to.
    #[arg(long, short, default_value = "triangle.png")]
    output: PathBuf,
}

fn main() -> Result<(), trd_core::RenderError> {
    env_logger::Builder::from_env(
        env_logger::Env::default().default_filter_or("warn,trd_core=info"),
    )
    .init();

    let cli = Cli::parse();
    println!("{}", trd_core::greeting());
    trd_core::render_to_png(cli.width, cli.height, &cli.output)?;
    println!(
        "Rendered {}x{} triangle to {}",
        cli.width,
        cli.height,
        cli.output.display()
    );
    Ok(())
}
