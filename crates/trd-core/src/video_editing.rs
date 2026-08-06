//! Versioned Arrow document for the browser video-editing example (#163).

use std::io::Cursor;

use arrow::array::{
    Array, BinaryArray, BooleanArray, FixedSizeListArray, Float32Array, Int64Array, UInt32Array,
};
use arrow::datatypes::{DataType, Schema};
use arrow::ipc::reader::StreamReader;
use thiserror::Error;

pub const VIDEO_EDIT_VERSION: &str = "0.1.0";
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
    pub frames: Vec<VideoEditingFrame>,
}

#[derive(Debug, Error)]
pub enum VideoEditingError {
    #[error("Arrow decode failed: {0}")]
    Arrow(#[from] arrow::error::ArrowError),
    #[error("video-editing metadata `{0}` is missing")]
    MissingMetadata(&'static str),
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
    #[error("video frame indices must be contiguous from zero (row {row} is {actual})")]
    NonContiguousFrameIndex { row: usize, actual: u32 },
    #[error("poster image is missing from video frame 0")]
    MissingPoster,
    #[error("poster image must only appear on video frame 0")]
    ExtraPoster,
    #[error("document has {actual} rows but metadata declares {expected} frames")]
    FrameCountMismatch { actual: usize, expected: u32 },
}

pub fn decode_video_editing_document(
    bytes: &[u8],
) -> Result<VideoEditingDocument, VideoEditingError> {
    let mut reader = StreamReader::try_new(Cursor::new(bytes), None)?;
    let schema = reader.schema();
    validate_schema_metadata(&schema)?;
    let video = decode_video_info(&schema)?;

    let mut frames = Vec::new();
    let mut poster_bytes = None;
    for batch in &mut reader {
        let batch = batch?;
        let video_frame_index = required_u32(&batch, "video_frame_index")?;
        let present_index = required_u32(&batch, "present_index")?;
        let timestamp_us = required_i64(&batch, "timestamp_us")?;
        let tracked = required_bool(&batch, "tracked")?;
        let k = optional_fixed_f32(&batch, "k", 9)?;
        let quad = optional_fixed_f32(&batch, "placement_quad", 8)?;
        let poster = optional_binary(&batch, "poster_bytes")?;

        for row in 0..batch.num_rows() {
            let absolute_row = frames.len();
            let frame_index = value_u32(video_frame_index, "video_frame_index", row)?;
            if frame_index != absolute_row as u32 {
                return Err(VideoEditingError::NonContiguousFrameIndex {
                    row: absolute_row,
                    actual: frame_index,
                });
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
    if frames.len() != video.frame_count as usize {
        return Err(VideoEditingError::FrameCountMismatch {
            actual: frames.len(),
            expected: video.frame_count,
        });
    }
    Ok(VideoEditingDocument {
        video,
        poster_bytes: poster_bytes.ok_or(VideoEditingError::MissingPoster)?,
        frames,
    })
}

fn validate_schema_metadata(schema: &Schema) -> Result<(), VideoEditingError> {
    let version = metadata(schema, VIDEO_EDIT_VERSION_KEY)?;
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
    use std::sync::Arc;

    use arrow::array::{ArrayRef, BinaryArray, BooleanArray, FixedSizeListArray};
    use arrow::datatypes::Field;
    use arrow::ipc::writer::StreamWriter;

    use super::*;

    fn document_bytes(version: &str, partial_geometry: bool) -> Vec<u8> {
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
        let mut bytes = Vec::new();
        let mut writer = StreamWriter::try_new(&mut bytes, &schema).unwrap();
        writer.write(&batch).unwrap();
        writer.finish().unwrap();
        drop(writer);
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
}
