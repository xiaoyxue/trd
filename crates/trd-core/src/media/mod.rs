//! Video: what trd knows about a clip, and the two places that knowledge comes
//! from.
//!
//! A video-editing session needs the same handful of facts — size, exact frame
//! rate, frame count, duration — and there are **two** sources for them:
//!
//! * the authoring document ([`video_document`]), `trd.video_edit 0.2.0`, read
//!   from Arrow IPC or Parquet; and
//! * the container itself ([`mp4_probe`]), walked for its `moov` box.
//!
//! The document is **optional** (#264): without one the editor is a player whose
//! timeline comes from the container. So these are not two unrelated parsers but
//! alternative answers to one question, which is why they share
//! [`VideoTiming`](video::VideoTiming) and live in one module — a fallback has no
//! natural home when its two sides cannot see each other.
//!
//! `trd-core` does no codec work: it walks a box and decodes a table. Actual
//! demuxing and decoding belong to the delivery surfaces (mediabunny in the
//! browser, ffmpeg natively).
//!
//! Deliberately **not** under `protocol/`: the editor document is independent of
//! the render `PROTOCOL_VERSION` and must stay that way, so it is not filed
//! beside `0.0.6`.

mod arrow_columns;
pub mod mp4_probe;
pub mod video;
pub mod video_document;

pub use mp4_probe::probe_moov;
pub use video::{UnpresentedTail, UnpresentedTailEvidence, VideoInfo, VideoTiming};
pub use video_document::{
    decode_video_editing_document, Shot, VideoEditingDocument, VideoEditingError,
    VideoEditingFrame, VIDEO_EDIT_TABLE_KIND_KEY, VIDEO_EDIT_TIMELINE_KIND, VIDEO_EDIT_VERSION,
    VIDEO_EDIT_VERSION_KEY,
};
