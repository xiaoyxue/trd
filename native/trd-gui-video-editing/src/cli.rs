use std::path::PathBuf;

use clap::Parser;

#[derive(Debug, Parser)]
#[command(name = "trd-gui-video-editing", version, about)]
pub struct Cli {
    /// Versioned `trd.video_edit.version = 0.2.0` Arrow IPC or Parquet timeline.
    ///
    /// The version is matched exactly, not as a minimum: `trd-core`'s
    /// `VIDEO_EDIT_VERSION` is the only accepted value and any other is
    /// rejected, so this text and that constant must not drift apart. The
    /// container is sniffed from the bytes rather than the file name, so either
    /// format may be passed here.
    ///
    /// **Optional**: without one the editor is a plain player — the timeline
    /// comes from the video container and the placement UI stays inert (#264).
    #[arg(long, value_name = "TIMELINE")]
    pub document: Option<PathBuf>,

    /// Local MP4 matching the timeline metadata. Without a source, the editor
    /// starts with an empty canvas until Open video is used.
    #[arg(long, value_name = "MP4", conflicts_with = "video_url")]
    pub video: Option<PathBuf>,

    /// HTTP(S) MP4 matching the timeline metadata.
    #[arg(
        long,
        value_name = "URL",
        conflicts_with = "video",
        value_parser = parse_http_url
    )]
    pub video_url: Option<String>,

    /// Width used when ffmpeg scales the streamed native preview frames.
    #[arg(long, default_value_t = 960, value_parser = clap::value_parser!(u32).range(1..=1920))]
    pub preview_width: u32,

    /// Validate the document/video and decode frame 0 without opening a window.
    #[arg(long)]
    pub probe_only: bool,
}

fn parse_http_url(value: &str) -> Result<String, String> {
    if value.starts_with("http://") || value.starts_with("https://") {
        Ok(value.to_owned())
    } else {
        Err("video URL must start with http:// or https://".to_owned())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn video_url_requires_http_or_https() {
        assert!(parse_http_url("https://example.com/video.mp4").is_ok());
        assert!(parse_http_url("http://example.com/video.mp4").is_ok());
        assert!(parse_http_url("file:///tmp/video.mp4").is_err());
    }
}
