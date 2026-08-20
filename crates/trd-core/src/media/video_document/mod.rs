//! Versioned Arrow document for the browser video-editing example (#163).

use std::io::Cursor;

use super::arrow_columns::{
    binary_value, describe_foreign_table, fixed_value, metadata, metadata_parse, optional_binary,
    optional_fixed_f32, required_bool, required_i64, required_u32, value_bool, value_i64,
    value_u32,
};
use super::video::VideoInfo;

use arrow::datatypes::{DataType, Schema};
use arrow::ipc::reader::StreamReader;
use arrow::record_batch::RecordBatch;
use bytes::Bytes;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use thiserror::Error;

pub const VIDEO_EDIT_VERSION: &str = "0.2.0";
pub const VIDEO_EDIT_VERSION_KEY: &str = "trd.video_edit.version";
pub const VIDEO_EDIT_TABLE_KIND_KEY: &str = "trd.video_edit.table.kind";
pub const VIDEO_EDIT_TIMELINE_KIND: &str = "timeline";

#[derive(Debug, Clone, PartialEq)]
pub struct VideoEditingFrame {
    /// Zero-based decoded frame index in the local video.
    pub video_frame_index: u32,
    /// Source calibration row index. Equal to `video_frame_index` for FIBA shot 1.
    pub present_index: u32,
    pub timestamp_us: i64,
    pub k: Option<[f32; 9]>,
    pub placement_quad: Option<[[f32; 2]; 4]>,
    pub tracked: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct VideoEditingDocument {
    pub video: VideoInfo,
    pub poster_bytes: Vec<u8>,
    /// The annotated frames, **sparse** and sorted by `video_frame_index`.
    ///
    /// A document names only the frames that carry a placement quad — a few
    /// shots out of a clip that may run to hundreds of thousands of frames — so
    /// this is a lookup table, not a per-frame array (#264). Use
    /// [`frame`](Self::frame) rather than indexing.
    pub frames: Vec<VideoEditingFrame>,
}

/// A maximal run of consecutive annotated frames — what a user calls a *shot*.
///
/// Derived rather than stored: a run boundary is exactly "the next annotated
/// frame is not the next frame", so storing it would be a second source of
/// truth that could disagree with the rows.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Shot {
    /// First annotated frame — what "jump to this shot" seeks to.
    pub start_frame: u32,
    /// Last annotated frame, inclusive.
    pub end_frame: u32,
}

impl Shot {
    pub fn frame_count(self) -> u32 {
        self.end_frame.saturating_sub(self.start_frame) + 1
    }

    pub fn contains(self, frame_index: u32) -> bool {
        (self.start_frame..=self.end_frame).contains(&frame_index)
    }
}

impl VideoEditingDocument {
    /// The row for `frame_index`, or `None` when the frame is not annotated.
    ///
    /// A binary search over the sorted rows: with a sparse document the answer
    /// is usually "no row", and that is the ordinary case, not an error.
    pub fn frame(&self, frame_index: u32) -> Option<&VideoEditingFrame> {
        self.frames
            .binary_search_by_key(&frame_index, |frame| frame.video_frame_index)
            .ok()
            .map(|row| &self.frames[row])
    }

    /// The annotated runs, in play order.
    ///
    /// This is what makes a sparse document navigable: in a 79-minute clip
    /// nothing else tells you *where* the annotated ranges are.
    pub fn shots(&self) -> Vec<Shot> {
        let mut shots: Vec<Shot> = Vec::new();
        for frame in &self.frames {
            match shots.last_mut() {
                Some(shot) if frame.video_frame_index == shot.end_frame + 1 => {
                    shot.end_frame = frame.video_frame_index;
                }
                _ => shots.push(Shot {
                    start_frame: frame.video_frame_index,
                    end_frame: frame.video_frame_index,
                }),
            }
        }
        shots
    }
}

#[derive(Debug, Error)]
pub enum VideoEditingError {
    #[error("Arrow decode failed: {0}")]
    Arrow(#[from] arrow::error::ArrowError),
    #[error("video-editing metadata `{0}` is missing")]
    MissingMetadata(&'static str),
    #[error("not a video-editing document: {0}")]
    NotADocument(String),
    #[error("video-editing metadata `{key}` has invalid value `{value}`")]
    InvalidMetadata { key: &'static str, value: String },
    #[error("video-editing version `{actual}` is unsupported (expected {VIDEO_EDIT_VERSION})")]
    UnsupportedVersion { actual: String },
    #[error("video-editing table kind `{actual}` is unsupported")]
    UnsupportedTableKind { actual: String },
    #[error("video-editing column `{0}` is missing")]
    MissingColumn(&'static str),
    #[error("video-editing column `{column}` has type {actual:?}, expected {expected}")]
    ColumnType {
        column: &'static str,
        actual: DataType,
        expected: &'static str,
    },
    #[error("video-editing column `{column}` is null at row {row}")]
    NullValue { column: &'static str, row: usize },
    #[error("frame {row} has only one of `k` / `placement_quad`")]
    PartialGeometry { row: usize },
    #[error("frame index {actual} is outside the video's {frame_count} frames (row {row})")]
    FrameIndexOutOfRange {
        row: usize,
        actual: u32,
        frame_count: u32,
    },
    #[error("frame indices must increase (row {row} is {actual}, after {previous})")]
    UnsortedFrameIndex {
        row: usize,
        actual: u32,
        previous: u32,
    },
    #[error("poster image must only appear on the first row")]
    ExtraPoster,
    #[error("not an annotation document: expected Arrow IPC or Parquet, got bytes {head:02x?}")]
    UnknownFormat { head: Vec<u8> },
    #[error("Parquet decode failed: {0}")]
    Parquet(#[from] parquet::errors::ParquetError),
}

/// Which container the annotation document arrived in.
///
/// Sniffed from the bytes, never from the file name: a URL need not carry a
/// useful suffix, and a mislabelled file should be read for what it is rather
/// than rejected for what it is called (#264).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DocumentFormat {
    /// Arrow IPC **stream** — what `scripts/` emits and what the editor has
    /// always read.
    ArrowIpc,
    /// Parquet — what tracking and calibration pipelines emit.
    Parquet,
}

impl DocumentFormat {
    /// Parquet brackets the file with this magic, at both ends.
    const PARQUET_MAGIC: &'static [u8] = b"PAR1";
    /// An Arrow IPC stream opens with a continuation marker, followed by the
    /// length of the schema message.
    const ARROW_IPC_CONTINUATION: [u8; 4] = [0xff, 0xff, 0xff, 0xff];

    /// Identifies `bytes`, or `None` if it is neither format.
    pub fn sniff(bytes: &[u8]) -> Option<Self> {
        // Both ends for Parquet, because a bare `PAR1` prefix is also how a
        // *truncated* Parquet file starts, and reading one of those produces a
        // far worse error than saying so here.
        if bytes.len() >= 8
            && bytes.starts_with(Self::PARQUET_MAGIC)
            && bytes.ends_with(Self::PARQUET_MAGIC)
        {
            return Some(Self::Parquet);
        }
        if bytes.starts_with(&Self::ARROW_IPC_CONTINUATION) {
            return Some(Self::ArrowIpc);
        }
        None
    }
}

/// Decodes an annotation document from either supported container.
///
/// The format is sniffed, both readers produce the **same** `RecordBatch`es, and
/// everything after that is shared — so a document cannot mean two different
/// things depending on how it was written.
pub fn decode_video_editing_document(
    bytes: &[u8],
) -> Result<VideoEditingDocument, VideoEditingError> {
    match DocumentFormat::sniff(bytes) {
        Some(DocumentFormat::ArrowIpc) => {
            let mut reader = StreamReader::try_new(Cursor::new(bytes), None)?;
            let schema = reader.schema();
            let batches = (&mut reader).collect::<Result<Vec<_>, _>>()?;
            build_document(&schema, &batches)
        }
        Some(DocumentFormat::Parquet) => {
            let builder = ParquetRecordBatchReaderBuilder::try_new(Bytes::copy_from_slice(bytes))?;
            // Parquet carries the schema's key-value metadata, so the same
            // version / table-kind contract applies untouched. Taken from the
            // builder because building the reader consumes it.
            let schema = builder.schema().clone();
            let batches = builder.build()?.collect::<Result<Vec<_>, _>>()?;
            build_document(&schema, &batches)
        }
        None => Err(VideoEditingError::UnknownFormat {
            head: bytes.iter().take(4).copied().collect(),
        }),
    }
}

/// Everything after the container: the checks and the row walk that both
/// formats share. Having exactly one of these is what makes Arrow and Parquet
/// decode to the same document by construction rather than by agreement.
fn build_document(
    schema: &Schema,
    batches: &[RecordBatch],
) -> Result<VideoEditingDocument, VideoEditingError> {
    validate_schema_metadata(schema)?;
    let video = decode_video_info(schema)?;

    let mut frames = Vec::new();
    let mut poster_bytes = None;
    for batch in batches {
        let video_frame_index = required_u32(batch, "video_frame_index")?;
        let present_index = required_u32(batch, "present_index")?;
        let timestamp_us = required_i64(batch, "timestamp_us")?;
        let tracked = required_bool(batch, "tracked")?;
        let k = optional_fixed_f32(batch, "k", 9)?;
        let quad = optional_fixed_f32(batch, "placement_quad", 8)?;
        let poster = optional_binary(batch, "poster_bytes")?;

        for row in 0..batch.num_rows() {
            let absolute_row = frames.len();
            let frame_index = value_u32(video_frame_index, "video_frame_index", row)?;
            // Sparse, but ordered and in range: rows may skip freely — that is
            // the point — while an index past the video is a document that
            // cannot apply, and an out-of-order one would break the lookup.
            if frame_index >= video.frame_count {
                return Err(VideoEditingError::FrameIndexOutOfRange {
                    row: absolute_row,
                    actual: frame_index,
                    frame_count: video.frame_count,
                });
            }
            if let Some(previous) = frames
                .last()
                .map(|frame: &VideoEditingFrame| frame.video_frame_index)
            {
                if frame_index <= previous {
                    return Err(VideoEditingError::UnsortedFrameIndex {
                        row: absolute_row,
                        actual: frame_index,
                        previous,
                    });
                }
            }
            let k = fixed_value(k, row).map(|values| {
                let mut result = [0.0; 9];
                result.copy_from_slice(values);
                result
            });
            let placement_quad = fixed_value(quad, row).map(|values| {
                [
                    [values[0], values[1]],
                    [values[2], values[3]],
                    [values[4], values[5]],
                    [values[6], values[7]],
                ]
            });
            if k.is_some() != placement_quad.is_some() {
                return Err(VideoEditingError::PartialGeometry { row: absolute_row });
            }
            if let Some(bytes) = binary_value(poster, row) {
                if absolute_row != 0 || poster_bytes.is_some() {
                    return Err(VideoEditingError::ExtraPoster);
                }
                poster_bytes = Some(bytes.to_vec());
            }
            frames.push(VideoEditingFrame {
                video_frame_index: frame_index,
                present_index: value_u32(present_index, "present_index", row)?,
                timestamp_us: value_i64(timestamp_us, "timestamp_us", row)?,
                k,
                placement_quad,
                tracked: value_bool(tracked, "tracked", row)?,
            });
        }
    }
    Ok(VideoEditingDocument {
        video,
        // Optional: the poster exists so an editor can show *something* before
        // the first decode, and a sparse document may not annotate frame 0 at
        // all. The first decoded frame serves just as well (#264).
        poster_bytes: poster_bytes.unwrap_or_default(),
        frames,
    })
}

fn validate_schema_metadata(schema: &Schema) -> Result<(), VideoEditingError> {
    // Absent rather than wrong: the file is some other kind of table, so name
    // what it is instead of the key it happens to lack.
    let Some(version) = schema.metadata().get(VIDEO_EDIT_VERSION_KEY) else {
        return Err(VideoEditingError::NotADocument(describe_foreign_table(
            schema,
        )));
    };
    if version != VIDEO_EDIT_VERSION {
        return Err(VideoEditingError::UnsupportedVersion {
            actual: version.to_owned(),
        });
    }
    let kind = metadata(schema, VIDEO_EDIT_TABLE_KIND_KEY)?;
    if kind != VIDEO_EDIT_TIMELINE_KIND {
        return Err(VideoEditingError::UnsupportedTableKind {
            actual: kind.to_owned(),
        });
    }
    Ok(())
}

fn decode_video_info(schema: &Schema) -> Result<VideoInfo, VideoEditingError> {
    Ok(VideoInfo {
        source_name: metadata(schema, "trd.video.source_name")?.to_owned(),
        mime: metadata(schema, "trd.video.mime")?.to_owned(),
        codec: metadata(schema, "trd.video.codec")?.to_owned(),
        sha256: metadata(schema, "trd.video.sha256")?.to_owned(),
        byte_length: metadata_parse(schema, "trd.video.byte_length")?,
        width: metadata_parse(schema, "trd.video.width")?,
        height: metadata_parse(schema, "trd.video.height")?,
        fps_num: metadata_parse(schema, "trd.video.fps_num")?,
        fps_den: metadata_parse(schema, "trd.video.fps_den")?,
        frame_count: metadata_parse(schema, "trd.video.frame_count")?,
        duration_us: metadata_parse(schema, "trd.video.duration_us")?,
    })
}

#[cfg(test)]
mod tests;
