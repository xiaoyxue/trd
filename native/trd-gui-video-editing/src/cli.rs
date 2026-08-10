use std::path::PathBuf;

use clap::Parser;

#[derive(Debug, Parser)]
#[command(name = "trd-gui-video-editing", version, about)]
pub struct Cli {
    /// Versioned `trd.video_edit.version = 0.1.0` Arrow timeline.
    #[arg(long, value_name = "ARROW")]
    pub document: PathBuf,

    /// Local MP4 matching the timeline metadata. Without it, the embedded poster
    /// and timeline details remain available.
    #[arg(long, value_name = "MP4")]
    pub video: Option<PathBuf>,

    /// Width used when ffmpeg scales the streamed native preview frames.
    #[arg(long, default_value_t = 960, value_parser = clap::value_parser!(u32).range(1..=1920))]
    pub preview_width: u32,

    /// Validate the document/video and decode frame 0 without opening a window.
    #[arg(long)]
    pub probe_only: bool,
}
