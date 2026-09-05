use std::path::PathBuf;

use clap::Parser;

#[derive(Debug, Parser)]
#[command(name = "trd-gui-video-editing", version, about)]
pub struct Cli {
    /// Annotation Arrow/Parquet input or an exported protocol `0.0.6` Arrow scene.
    ///
    /// The bytes are sniffed against the exact current video-editing version
    /// first, then the exact current render-protocol version. Other versions are
    /// rejected rather than upgraded.
    ///
    /// **Optional**: without one the editor is a plain player. Annotation input
    /// enables editing; a protocol scene replays over the selected video.
    #[arg(long, value_name = "ARROW_INPUT")]
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

    /// Validate the document/video and decode one frame without opening a
    /// window, reporting the frame that actually came back.
    #[arg(long)]
    pub probe_only: bool,

    /// Which frame `--probe-only` asks for. Defaults to the first.
    ///
    /// The answer is worth having because it is not always the frame requested:
    /// a seek time is computed from the timeline's nominal grid, and a
    /// variable-rate container does not sit on it, so ffmpeg can return the
    /// neighbouring picture (#319). This is the native counterpart of the
    /// browser's probe page — it reports the timestamp the frame carries, not
    /// the one that was asked for.
    #[arg(long, value_name = "N", default_value_t = 0)]
    pub probe_frame: u32,
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
