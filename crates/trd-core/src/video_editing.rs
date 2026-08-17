//! Versioned Arrow document for the browser video-editing example (#163).

use std::io::Cursor;

use arrow::array::{
    Array, BinaryArray, BooleanArray, FixedSizeListArray, Float32Array, Int64Array, UInt32Array,
};
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

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VideoInfo {
    pub source_name: String,
    pub mime: String,
    pub codec: String,
    pub sha256: String,
    pub byte_length: u64,
    pub width: u32,
    pub height: u32,
    pub fps_num: u32,
    pub fps_den: u32,
    pub frame_count: u32,
    pub duration_us: i64,
}

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

/// Says what a table that is *not* a video-editing document appears to be.
///
/// Every trd table is a `.arrow` or `.parquet`, and most of them are **not**
/// documents: render-protocol streams, golden fixtures, raw perception dumps. A
/// file picker can only filter by extension, so reaching for the wrong one is
/// ordinary — and "metadata is missing" named the key the file lacks rather than
/// the thing the user actually did.
fn describe_foreign_table(schema: &Schema) -> String {
    let metadata = schema.metadata();
    if let Some(version) = metadata.get(crate::protocol::PROTOCOL_VERSION_KEY) {
        let kind = metadata
            .get(crate::protocol::TABLE_KIND_KEY)
            .map_or("unknown", String::as_str);
        return format!(
            "this is a trd render-protocol `{kind}` table (version {version}), \
             which describes a scene to render, not frames to annotate"
        );
    }
    if metadata.keys().any(|key| key.starts_with("trd.")) {
        return format!(
            "it carries trd metadata but no `{VIDEO_EDIT_VERSION_KEY}`: {}",
            summarise_columns(schema)
        );
    }
    format!(
        "it has no trd metadata at all — an unrelated Arrow/Parquet table: {}",
        summarise_columns(schema)
    )
}

/// The first few column names, so an unrecognised table is at least identifiable.
fn summarise_columns(schema: &Schema) -> String {
    let names: Vec<&str> = schema
        .fields()
        .iter()
        .take(6)
        .map(|field| field.name().as_str())
        .collect();
    format!(
        "columns [{}{}]",
        names.join(", "),
        if schema.fields().len() > names.len() {
            ", …"
        } else {
            ""
        }
    )
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

fn metadata<'a>(schema: &'a Schema, key: &'static str) -> Result<&'a str, VideoEditingError> {
    schema
        .metadata()
        .get(key)
        .map(String::as_str)
        .ok_or(VideoEditingError::MissingMetadata(key))
}

fn metadata_parse<T>(schema: &Schema, key: &'static str) -> Result<T, VideoEditingError>
where
    T: std::str::FromStr,
{
    let value = metadata(schema, key)?;
    value
        .parse()
        .map_err(|_| VideoEditingError::InvalidMetadata {
            key,
            value: value.to_owned(),
        })
}

fn required_u32<'a>(
    batch: &'a arrow::array::RecordBatch,
    name: &'static str,
) -> Result<&'a UInt32Array, VideoEditingError> {
    downcast(batch, name, "UInt32")
}

fn required_i64<'a>(
    batch: &'a arrow::array::RecordBatch,
    name: &'static str,
) -> Result<&'a Int64Array, VideoEditingError> {
    downcast(batch, name, "Int64")
}

fn required_bool<'a>(
    batch: &'a arrow::array::RecordBatch,
    name: &'static str,
) -> Result<&'a BooleanArray, VideoEditingError> {
    downcast(batch, name, "Boolean")
}

fn optional_binary<'a>(
    batch: &'a arrow::array::RecordBatch,
    name: &'static str,
) -> Result<&'a BinaryArray, VideoEditingError> {
    downcast(batch, name, "Binary")
}

fn downcast<'a, T: 'static>(
    batch: &'a arrow::array::RecordBatch,
    name: &'static str,
    expected: &'static str,
) -> Result<&'a T, VideoEditingError> {
    let column = batch
        .column_by_name(name)
        .ok_or(VideoEditingError::MissingColumn(name))?;
    column
        .as_any()
        .downcast_ref::<T>()
        .ok_or_else(|| VideoEditingError::ColumnType {
            column: name,
            actual: column.data_type().clone(),
            expected,
        })
}

fn optional_fixed_f32<'a>(
    batch: &'a arrow::array::RecordBatch,
    name: &'static str,
    width: i32,
) -> Result<(&'a FixedSizeListArray, &'a Float32Array), VideoEditingError> {
    let list: &FixedSizeListArray = downcast(batch, name, "FixedSizeList<Float32>")?;
    if list.value_length() != width {
        return Err(VideoEditingError::ColumnType {
            column: name,
            actual: list.data_type().clone(),
            expected: "FixedSizeList<Float32> with the declared width",
        });
    }
    let values = list
        .values()
        .as_any()
        .downcast_ref::<Float32Array>()
        .ok_or_else(|| VideoEditingError::ColumnType {
            column: name,
            actual: list.values().data_type().clone(),
            expected: "FixedSizeList<Float32>",
        })?;
    Ok((list, values))
}

fn fixed_value<'a>(
    arrays: (&'a FixedSizeListArray, &'a Float32Array),
    row: usize,
) -> Option<&'a [f32]> {
    let (list, values) = arrays;
    if list.is_null(row) {
        return None;
    }
    let start = list.value_offset(row) as usize;
    let end = start + list.value_length() as usize;
    Some(&values.values()[start..end])
}

fn binary_value(array: &BinaryArray, row: usize) -> Option<&[u8]> {
    (!array.is_null(row)).then(|| array.value(row))
}

fn value_u32(
    array: &UInt32Array,
    column: &'static str,
    row: usize,
) -> Result<u32, VideoEditingError> {
    if array.is_null(row) {
        return Err(VideoEditingError::NullValue { column, row });
    }
    Ok(array.value(row))
}

fn value_i64(
    array: &Int64Array,
    column: &'static str,
    row: usize,
) -> Result<i64, VideoEditingError> {
    if array.is_null(row) {
        return Err(VideoEditingError::NullValue { column, row });
    }
    Ok(array.value(row))
}

fn value_bool(
    array: &BooleanArray,
    column: &'static str,
    row: usize,
) -> Result<bool, VideoEditingError> {
    if array.is_null(row) {
        return Err(VideoEditingError::NullValue { column, row });
    }
    Ok(array.value(row))
}

#[cfg(test)]
mod tests {
    use std::path::{Path, PathBuf};
    use std::sync::Arc;

    use arrow::array::{ArrayRef, BinaryArray, BooleanArray, FixedSizeListArray};
    use arrow::datatypes::Field;
    use arrow::ipc::writer::StreamWriter;

    use super::*;

    /// Reads a fixture that is **generated, not committed** — the FIBA document
    /// is derived from an external MP4, so most checkouts do not have one.
    ///
    /// `cargo test -p trd-core -- --ignored` runs *every* ignored test, so an
    /// absent fixture has to skip rather than fail the suite. A fixture that is
    /// present but unreadable is still an error: that is a broken fixture, not
    /// a missing one.
    fn generated_fixture(path: &Path) -> Option<Vec<u8>> {
        match std::fs::read(path) {
            Ok(bytes) => Some(bytes),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                eprintln!("skipping: {} has not been generated", path.display());
                None
            }
            Err(error) => panic!("{}: {error}", path.display()),
        }
    }

    /// An overridable fixture path, so a document generated elsewhere can be
    /// pointed at without moving it into the tree.
    fn fixture_path(variable: &str, default: impl FnOnce() -> PathBuf) -> PathBuf {
        std::env::var_os(variable).map_or_else(default, PathBuf::from)
    }

    /// The repository root. A test binary runs with the *crate* directory as
    /// its working directory, so a tree-relative default has to be anchored
    /// explicitly — otherwise a fixture sitting exactly where the docs put it
    /// is never found and the test skips instead of asserting.
    fn repository_root() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .ancestors()
            .nth(2)
            .expect("crates/trd-core is two levels below the repository root")
            .to_path_buf()
    }

    /// Where the generated Parquet fixtures are looked for. `std::env::temp_dir`
    /// rather than `TMP`, which only Windows sets.
    fn parquet_fixture_dir() -> PathBuf {
        fixture_path("TRD_DOC_DIR", std::env::temp_dir)
    }

    fn document_batch(version: &str, partial_geometry: bool) -> (Schema, RecordBatch) {
        let metadata = [
            (VIDEO_EDIT_VERSION_KEY.to_owned(), version.to_owned()),
            (
                VIDEO_EDIT_TABLE_KIND_KEY.to_owned(),
                VIDEO_EDIT_TIMELINE_KIND.to_owned(),
            ),
            ("trd.video.source_name".to_owned(), "shot.mp4".to_owned()),
            ("trd.video.mime".to_owned(), "video/mp4".to_owned()),
            ("trd.video.codec".to_owned(), "h264".to_owned()),
            ("trd.video.sha256".to_owned(), "abc".to_owned()),
            ("trd.video.byte_length".to_owned(), "10".to_owned()),
            ("trd.video.width".to_owned(), "1920".to_owned()),
            ("trd.video.height".to_owned(), "1080".to_owned()),
            ("trd.video.fps_num".to_owned(), "24".to_owned()),
            ("trd.video.fps_den".to_owned(), "1".to_owned()),
            ("trd.video.frame_count".to_owned(), "1".to_owned()),
            ("trd.video.duration_us".to_owned(), "41667".to_owned()),
        ]
        .into_iter()
        .collect();
        let f32_field = Arc::new(Field::new("item", DataType::Float32, false));
        let k_values: ArrayRef = Arc::new(Float32Array::from(vec![
            4510.0, 0.0, 960.0, 0.0, 4510.0, 540.0, 0.0, 0.0, 1.0,
        ]));
        let k = FixedSizeListArray::new(f32_field.clone(), 9, k_values, None);
        let quad_values: ArrayRef = Arc::new(Float32Array::from(vec![
            10.0, 20.0, 30.0, 20.0, 30.0, 40.0, 10.0, 40.0,
        ]));
        let quad = FixedSizeListArray::new(
            f32_field,
            8,
            quad_values,
            partial_geometry.then(|| {
                arrow::buffer::NullBuffer::new(arrow::buffer::BooleanBuffer::new_unset(1))
            }),
        );
        let schema = Schema::new(vec![
            Field::new("video_frame_index", DataType::UInt32, false),
            Field::new("present_index", DataType::UInt32, false),
            Field::new("timestamp_us", DataType::Int64, false),
            Field::new("k", k.data_type().clone(), true),
            Field::new("placement_quad", quad.data_type().clone(), true),
            Field::new("tracked", DataType::Boolean, false),
            Field::new("poster_bytes", DataType::Binary, true),
        ])
        .with_metadata(metadata);
        let batch = arrow::array::RecordBatch::try_new(
            Arc::new(schema.clone()),
            vec![
                Arc::new(UInt32Array::from(vec![0])) as ArrayRef,
                Arc::new(UInt32Array::from(vec![0])) as ArrayRef,
                Arc::new(Int64Array::from(vec![0])) as ArrayRef,
                Arc::new(k) as ArrayRef,
                Arc::new(quad) as ArrayRef,
                Arc::new(BooleanArray::from(vec![true])) as ArrayRef,
                Arc::new(BinaryArray::from(vec![Some(b"jpeg".as_slice())])) as ArrayRef,
            ],
        )
        .unwrap();
        (schema, batch)
    }

    /// The same rows as an Arrow IPC stream — what `scripts/` emits.
    fn document_bytes(version: &str, partial_geometry: bool) -> Vec<u8> {
        let (schema, batch) = document_batch(version, partial_geometry);
        let mut bytes = Vec::new();
        let mut writer = StreamWriter::try_new(&mut bytes, &schema).unwrap();
        writer.write(&batch).unwrap();
        writer.finish().unwrap();
        drop(writer);
        bytes
    }

    /// The same rows as Parquet — what tracking and calibration pipelines emit.
    fn document_parquet_bytes(version: &str, partial_geometry: bool) -> Vec<u8> {
        let (schema, batch) = document_batch(version, partial_geometry);
        let mut bytes = Vec::new();
        let mut writer =
            parquet::arrow::ArrowWriter::try_new(&mut bytes, Arc::new(schema), None).unwrap();
        writer.write(&batch).unwrap();
        writer.close().unwrap();
        bytes
    }

    #[test]
    fn decodes_simple_frame_zero_document() {
        let document =
            decode_video_editing_document(&document_bytes(VIDEO_EDIT_VERSION, false)).unwrap();
        assert_eq!(document.frames.len(), 1);
        assert_eq!(document.frames[0].video_frame_index, 0);
        assert_eq!(document.frames[0].present_index, 0);
        assert_eq!(document.poster_bytes, b"jpeg");
        assert!(document.frames[0].k.is_some());
    }

    #[test]
    fn rejects_other_document_version() {
        assert!(matches!(
            decode_video_editing_document(&document_bytes("9.9.9", false)),
            Err(VideoEditingError::UnsupportedVersion { .. })
        ));
    }

    #[test]
    fn rejects_partial_geometry() {
        assert!(matches!(
            decode_video_editing_document(&document_bytes(VIDEO_EDIT_VERSION, true)),
            Err(VideoEditingError::PartialGeometry { row: 0 })
        ));
    }

    /// Picking the wrong `.arrow` is the ordinary mistake — the repository is
    /// full of tables that are not documents and a file picker can only filter
    /// by extension — so the error has to name what the file *is*.
    #[test]
    fn a_table_that_is_not_a_document_says_what_it_is() {
        use arrow::array::{Float32Array, RecordBatch};
        use arrow::datatypes::Field;

        fn encode(metadata: &[(&str, &str)]) -> Vec<u8> {
            let schema = Schema::new(vec![Field::new("x", DataType::Float32, false)])
                .with_metadata(
                    metadata
                        .iter()
                        .map(|(key, value)| ((*key).to_owned(), (*value).to_owned()))
                        .collect(),
                );
            let schema = Arc::new(schema);
            let batch = RecordBatch::try_new(
                Arc::clone(&schema),
                vec![Arc::new(Float32Array::from(vec![1.0]))],
            )
            .unwrap();
            let mut buffer = Vec::new();
            {
                let mut writer = StreamWriter::try_new(&mut buffer, &schema).unwrap();
                writer.write(&batch).unwrap();
                writer.finish().unwrap();
            }
            buffer
        }

        // A render-protocol table: the golden fixtures and every viewer stream.
        let protocol = decode_video_editing_document(&encode(&[
            (crate::protocol::PROTOCOL_VERSION_KEY, "0.0.6"),
            (crate::protocol::TABLE_KIND_KEY, "mesh"),
        ]))
        .unwrap_err()
        .to_string();
        assert!(
            protocol.contains("render-protocol") && protocol.contains("mesh"),
            "names the protocol and the table kind: {protocol}"
        );

        // A raw perception dump: `examples/frames.*.perception.arrow` carries no
        // trd metadata whatsoever.
        let foreign = decode_video_editing_document(&encode(&[]))
            .unwrap_err()
            .to_string();
        assert!(
            foreign.contains("no trd metadata") && foreign.contains("columns [x]"),
            "names the columns so the file is identifiable: {foreign}"
        );

        // A missing *version* must not be reported as a missing *key*.
        assert!(
            !protocol.contains("is missing") && !foreign.contains("is missing"),
            "the old message named the key the file lacks, not what the user picked"
        );
    }

    /// A sparse document: rows name **only** the annotated frames, so most
    /// lookups legitimately find nothing (#264).
    fn sparse_bytes(frame_count: u32, indices: &[u32], poster: bool) -> Vec<u8> {
        let rows = indices.len();
        let metadata = [
            (
                VIDEO_EDIT_VERSION_KEY.to_owned(),
                VIDEO_EDIT_VERSION.to_owned(),
            ),
            (
                VIDEO_EDIT_TABLE_KIND_KEY.to_owned(),
                VIDEO_EDIT_TIMELINE_KIND.to_owned(),
            ),
            ("trd.video.source_name".to_owned(), "shot.mp4".to_owned()),
            ("trd.video.mime".to_owned(), "video/mp4".to_owned()),
            ("trd.video.codec".to_owned(), "h264".to_owned()),
            ("trd.video.sha256".to_owned(), "abc".to_owned()),
            ("trd.video.byte_length".to_owned(), "10".to_owned()),
            ("trd.video.width".to_owned(), "1920".to_owned()),
            ("trd.video.height".to_owned(), "1080".to_owned()),
            ("trd.video.fps_num".to_owned(), "24".to_owned()),
            ("trd.video.fps_den".to_owned(), "1".to_owned()),
            ("trd.video.frame_count".to_owned(), frame_count.to_string()),
            ("trd.video.duration_us".to_owned(), "41667".to_owned()),
        ]
        .into_iter()
        .collect();
        let f32_field = Arc::new(Field::new("item", DataType::Float32, false));
        let k_values: ArrayRef = Arc::new(Float32Array::from(
            (0..rows)
                .flat_map(|_| [4510.0, 0.0, 960.0, 0.0, 4510.0, 540.0, 0.0, 0.0, 1.0])
                .collect::<Vec<f32>>(),
        ));
        let k = FixedSizeListArray::new(f32_field.clone(), 9, k_values, None);
        let quad_values: ArrayRef = Arc::new(Float32Array::from(
            (0..rows)
                .flat_map(|_| [10.0, 20.0, 30.0, 20.0, 30.0, 40.0, 10.0, 40.0])
                .collect::<Vec<f32>>(),
        ));
        let quad = FixedSizeListArray::new(f32_field, 8, quad_values, None);
        let schema = Schema::new(vec![
            Field::new("video_frame_index", DataType::UInt32, false),
            Field::new("present_index", DataType::UInt32, false),
            Field::new("timestamp_us", DataType::Int64, false),
            Field::new("k", k.data_type().clone(), true),
            Field::new("placement_quad", quad.data_type().clone(), true),
            Field::new("tracked", DataType::Boolean, false),
            Field::new("poster_bytes", DataType::Binary, true),
        ])
        .with_metadata(metadata);
        let posters: Vec<Option<&[u8]>> = (0..rows)
            .map(|row| (poster && row == 0).then_some(b"jpeg".as_slice()))
            .collect();
        let batch = arrow::array::RecordBatch::try_new(
            Arc::new(schema.clone()),
            vec![
                Arc::new(UInt32Array::from(indices.to_vec())) as ArrayRef,
                Arc::new(UInt32Array::from(indices.to_vec())) as ArrayRef,
                Arc::new(Int64Array::from(
                    indices
                        .iter()
                        .map(|i| i64::from(*i) * 41_667)
                        .collect::<Vec<_>>(),
                )) as ArrayRef,
                Arc::new(k) as ArrayRef,
                Arc::new(quad) as ArrayRef,
                Arc::new(BooleanArray::from(vec![true; rows])) as ArrayRef,
                Arc::new(BinaryArray::from(posters)) as ArrayRef,
            ],
        )
        .unwrap();
        let mut bytes = Vec::new();
        let mut writer = StreamWriter::try_new(&mut bytes, &schema).unwrap();
        writer.write(&batch).unwrap();
        writer.finish().unwrap();
        drop(writer);
        bytes
    }

    /// The heart of `0.2.0`: rows skip freely, and a frame with no row is an
    /// ordinary un-annotated frame rather than an error.
    #[test]
    fn decodes_sparse_rows_and_looks_them_up() {
        let document =
            decode_video_editing_document(&sparse_bytes(1000, &[10, 11, 12, 40, 41], true))
                .unwrap();

        assert_eq!(document.frames.len(), 5, "only the annotated frames");
        assert_eq!(document.frame(11).map(|f| f.video_frame_index), Some(11));
        assert_eq!(document.frame(40).map(|f| f.video_frame_index), Some(40));
        assert!(document.frame(0).is_none(), "frame 0 is not annotated");
        assert!(document.frame(20).is_none(), "the gap has no row");
        assert!(document.frame(999).is_none());
    }

    /// Shots are what make a sparse document navigable: maximal runs of
    /// consecutive annotated frames, in play order.
    #[test]
    fn derives_shots_from_consecutive_runs() {
        let document =
            decode_video_editing_document(&sparse_bytes(1000, &[10, 11, 12, 40, 41, 900], true))
                .unwrap();

        assert_eq!(
            document.shots(),
            vec![
                Shot {
                    start_frame: 10,
                    end_frame: 12
                },
                Shot {
                    start_frame: 40,
                    end_frame: 41
                },
                Shot {
                    start_frame: 900,
                    end_frame: 900
                },
            ]
        );
        assert_eq!(document.shots()[0].frame_count(), 3);
        assert!(document.shots()[0].contains(11));
        assert!(!document.shots()[0].contains(13));
    }

    /// A poster is an editor convenience, and a sparse document may not annotate
    /// frame 0 at all — so its absence must not be fatal.
    #[test]
    fn poster_is_optional() {
        let document = decode_video_editing_document(&sparse_bytes(100, &[7, 8], false)).unwrap();
        assert!(document.poster_bytes.is_empty());
        assert_eq!(document.frames.len(), 2);
    }

    /// The two things a sparse document still may not do: name a frame the video
    /// does not have, or arrive out of order (which would break the lookup).
    #[test]
    fn rejects_out_of_range_and_unsorted_rows() {
        assert!(matches!(
            decode_video_editing_document(&sparse_bytes(50, &[10, 60], true)),
            Err(VideoEditingError::FrameIndexOutOfRange { actual: 60, .. })
        ));
        assert!(matches!(
            decode_video_editing_document(&sparse_bytes(100, &[10, 9], true)),
            Err(VideoEditingError::UnsortedFrameIndex {
                actual: 9,
                previous: 10,
                ..
            })
        ));
        assert!(
            matches!(
                decode_video_editing_document(&sparse_bytes(100, &[10, 10], true)),
                Err(VideoEditingError::UnsortedFrameIndex { .. })
            ),
            "a duplicated index is not a valid lookup key either"
        );
    }

    /// The slice's contract: the *same rows* written as Arrow IPC and as
    /// Parquet must decode to the *same document*. Both readers feed one
    /// `build_document`, so this pins that the containers really are
    /// interchangeable — including the schema key-value metadata, which is the
    /// part a Parquet round-trip could plausibly drop.
    #[test]
    fn arrow_and_parquet_decode_to_the_same_document() {
        let from_arrow =
            decode_video_editing_document(&document_bytes(VIDEO_EDIT_VERSION, false)).unwrap();
        let from_parquet =
            decode_video_editing_document(&document_parquet_bytes(VIDEO_EDIT_VERSION, false))
                .unwrap();
        assert_eq!(from_arrow, from_parquet);
    }

    /// The version and table-kind checks are the contract, and they have to
    /// survive the Parquet round-trip too — otherwise a Parquet document could
    /// smuggle in a version the Arrow path would reject.
    #[test]
    fn parquet_is_held_to_the_same_version_contract() {
        let error = decode_video_editing_document(&document_parquet_bytes("0.0.1", false))
            .expect_err("an old version must be rejected whatever the container");
        assert!(
            matches!(error, VideoEditingError::UnsupportedVersion { .. }),
            "expected UnsupportedVersion, got {error}"
        );
    }

    /// Format comes from the bytes, not the name — a URL need not carry a
    /// useful suffix, and a `.arrow`-named Parquet file is still Parquet.
    #[test]
    fn format_is_sniffed_from_the_bytes() {
        assert_eq!(
            DocumentFormat::sniff(&document_bytes(VIDEO_EDIT_VERSION, false)),
            Some(DocumentFormat::ArrowIpc)
        );
        assert_eq!(
            DocumentFormat::sniff(&document_parquet_bytes(VIDEO_EDIT_VERSION, false)),
            Some(DocumentFormat::Parquet)
        );
        // Whatever it is called, it decodes as what it is.
        let parquet_named_arrow = document_parquet_bytes(VIDEO_EDIT_VERSION, false);
        assert!(decode_video_editing_document(&parquet_named_arrow).is_ok());
    }

    /// A truncated Parquet file keeps the opening magic but loses the closing
    /// one. Saying so beats handing the bytes to a reader that will fail deep
    /// inside a footer parse.
    #[test]
    fn a_truncated_parquet_file_is_not_mistaken_for_one() {
        let full = document_parquet_bytes(VIDEO_EDIT_VERSION, false);
        let truncated = &full[..full.len() - 8];
        assert_eq!(DocumentFormat::sniff(truncated), None);
        let error =
            decode_video_editing_document(truncated).expect_err("a truncated file must not decode");
        assert!(
            matches!(error, VideoEditingError::UnknownFormat { .. }),
            "expected UnknownFormat, got {error}"
        );
    }

    #[test]
    fn neither_format_is_reported_as_such() {
        for bytes in [
            b"".as_slice(),
            b"PAR1".as_slice(),
            b"not a document".as_slice(),
        ] {
            assert_eq!(DocumentFormat::sniff(bytes), None, "bytes {bytes:02x?}");
        }
    }

    /// The synthetic parity test uses one hand-built row. This one uses the
    /// real FIBA document — 222 sparse rows with poster, K and quads — because
    /// a round-trip bug in a column type or in the sparse-index checks would
    /// only show on real data.
    ///
    /// Ignored by default: it needs the generated document, which is not
    /// committed. Run it after `scripts/fiba_video_editing_bundle.py`;
    /// `TRD_DOC_ARROW` / `TRD_DOC_PARQUET` override where they are looked for.
    #[test]
    #[ignore = "needs generated fixtures: web/gui-video-editing/data/fiba-shot1.{arrow,parquet}"]
    fn the_real_document_decodes_identically_from_both_containers() {
        let arrow = fixture_path("TRD_DOC_ARROW", || {
            repository_root().join("web/gui-video-editing/data/fiba-shot1.arrow")
        });
        let parquet = fixture_path("TRD_DOC_PARQUET", || {
            parquet_fixture_dir().join("fiba-shot1.parquet")
        });
        let (Some(arrow), Some(parquet)) = (generated_fixture(&arrow), generated_fixture(&parquet))
        else {
            return;
        };
        let from_arrow = decode_video_editing_document(&arrow).unwrap();
        let from_parquet = decode_video_editing_document(&parquet).unwrap();
        assert_eq!(from_arrow.frames.len(), 222, "the FIBA document is sparse");
        assert_eq!(from_arrow, from_parquet);
    }

    /// Which Parquet compressions the wasm-safe feature set can actually read.
    ///
    /// `snap` and uncompressed work; `zstd`/`gzip`/`brotli`/`lz4` are excluded
    /// because their C shims do not cross-compile to wasm32. This documents the
    /// trade and pins that an unsupported codec produces parquet's own clear
    /// "Disabled feature at compile time" message rather than something opaque.
    ///
    /// Reads `fiba-<codec>.parquet` from `TRD_DOC_DIR` (default: the platform
    /// temp dir); each codec absent from there is skipped.
    #[test]
    #[ignore = "needs generated fixtures written with several codecs"]
    fn unsupported_compression_says_so_clearly() {
        for (codec, supported) in [
            ("snappy", true),
            ("none", true),
            ("zstd", false),
            ("gzip", false),
        ] {
            let path = parquet_fixture_dir().join(format!("fiba-{codec}.parquet"));
            let Some(bytes) = generated_fixture(&path) else {
                continue;
            };
            assert_eq!(DocumentFormat::sniff(&bytes), Some(DocumentFormat::Parquet));
            match (decode_video_editing_document(&bytes), supported) {
                (Ok(document), true) => assert_eq!(document.frames.len(), 222),
                (Err(error), false) => {
                    let text = error.to_string();
                    assert!(
                        text.contains("Disabled feature at compile time"),
                        "{codec} should name the missing codec, got: {text}"
                    );
                }
                (result, _) => panic!("{codec}: unexpected {result:?}"),
            }
        }
    }
}
