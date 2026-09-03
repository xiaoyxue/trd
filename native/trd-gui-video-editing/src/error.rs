#[derive(Debug, thiserror::Error)]
pub enum NativeVideoEditingError {
    #[error("failed to read {path}: {source}")]
    Read {
        path: String,
        source: std::io::Error,
    },
    #[error(transparent)]
    Document(#[from] trd_core::VideoEditingError),
    #[error("invalid Arrow input: {0}")]
    Input(String),
    #[error(transparent)]
    Gui(#[from] trd_gui::error::GuiError),
    #[error("video editor renderer failed: {0}")]
    Renderer(String),
    #[error("video source mismatch: {0}")]
    SourceMismatch(String),
    #[error("failed to run {program}: {source}")]
    Spawn {
        program: &'static str,
        source: std::io::Error,
    },
    #[error("{program} failed: {stderr}")]
    Command {
        program: &'static str,
        stderr: String,
    },
    #[error("ffprobe output is missing `{0}`")]
    ProbeField(&'static str),
    #[error("invalid ffprobe value for `{field}`: {value}")]
    ProbeValue { field: &'static str, value: String },
    #[error("decoded frame {index} has {actual} RGBA bytes; expected {expected}")]
    FrameLength {
        index: u32,
        actual: usize,
        expected: usize,
    },
    #[error(transparent)]
    Eframe(#[from] eframe::Error),
}
