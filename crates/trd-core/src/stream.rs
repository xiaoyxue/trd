//! Native-only Arrow streaming protocol (trd protocol 0.0.1).
//!
//! Input: one row per frame (`center`, `size`, `theta`). Output: one row per
//! frame, four `fixed_shape_tensor<u8>` channels `r,g,b,a` of shape `[H, W]`.

use std::sync::Arc;

use arrow::array::{Array, FixedSizeListArray, Float32Array, RecordBatch};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::error::ArrowError;

use crate::render::FrameParams;

/// The trd stream protocol version carried in schema metadata.
pub const PROTOCOL_VERSION: &str = "0.0.1";
/// Schema-metadata key for the protocol version.
pub const PROTOCOL_VERSION_KEY: &str = "trd.protocol.version";

/// Errors from decoding, validating, rendering, or encoding a trd stream.
#[derive(Debug, thiserror::Error)]
pub enum StreamError {
    /// An underlying Arrow or IPC error.
    #[error("arrow error: {0}")]
    Arrow(#[from] ArrowError),
    /// I/O error reading or writing the stream.
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    /// A required input column is missing.
    #[error("input is missing required column `{0}`")]
    MissingColumn(&'static str),
    /// A required input column has the wrong Arrow type.
    #[error("input column `{column}` has type {actual:?}, expected {expected}")]
    ColumnType {
        column: &'static str,
        expected: &'static str,
        actual: DataType,
    },
    /// A required input column contains null values (protocol requires non-null).
    #[error("input column `{0}` contains null values")]
    NullValues(&'static str),
    /// The requested image dimensions are invalid or too large.
    #[error("invalid image dimensions {width}x{height}: {reason}")]
    InvalidDimensions {
        width: u32,
        height: u32,
        reason: &'static str,
    },
    /// The stream declares a protocol version this build does not support.
    #[error("unsupported protocol version `{0}` (expected `{PROTOCOL_VERSION}`)")]
    UnsupportedVersion(String),
    /// GPU rendering failed.
    #[error("render error: {0}")]
    Render(String),
}

fn vec2_field(name: &str) -> Field {
    Field::new(
        name,
        DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float32, false)), 2),
        false,
    )
}

/// Builds the protocol 0.0.1 input schema (with version metadata).
pub fn input_schema() -> Schema {
    Schema::new(vec![
        vec2_field("center"),
        vec2_field("size"),
        Field::new("theta", DataType::Float32, false),
    ])
    .with_metadata(
        [(PROTOCOL_VERSION_KEY.to_string(), PROTOCOL_VERSION.to_string())]
            .into_iter()
            .collect(),
    )
}

/// If the schema declares a protocol version, require it to match.
pub fn check_version(schema: &Schema) -> Result<(), StreamError> {
    if let Some(v) = schema.metadata().get(PROTOCOL_VERSION_KEY) {
        if v != PROTOCOL_VERSION {
            return Err(StreamError::UnsupportedVersion(v.clone()));
        }
    }
    Ok(())
}

fn require_vec2<'a>(
    batch: &'a RecordBatch,
    name: &'static str,
) -> Result<&'a FixedSizeListArray, StreamError> {
    let col = batch
        .column_by_name(name)
        .ok_or(StreamError::MissingColumn(name))?;
    match col.data_type() {
        DataType::FixedSizeList(f, 2) if *f.data_type() == DataType::Float32 => Ok(col
            .as_any()
            .downcast_ref::<FixedSizeListArray>()
            .expect("checked FixedSizeList type")),
        other => Err(StreamError::ColumnType {
            column: name,
            expected: "FixedSizeList<Float32>[2]",
            actual: other.clone(),
        }),
    }
}

fn require_f32<'a>(
    batch: &'a RecordBatch,
    name: &'static str,
) -> Result<&'a Float32Array, StreamError> {
    let col = batch
        .column_by_name(name)
        .ok_or(StreamError::MissingColumn(name))?;
    col.as_any()
        .downcast_ref::<Float32Array>()
        .ok_or_else(|| StreamError::ColumnType {
            column: name,
            expected: "Float32",
            actual: col.data_type().clone(),
        })
}

fn read_vec2(list: &FixedSizeListArray, row: usize) -> [f32; 2] {
    let values = list
        .values()
        .as_any()
        .downcast_ref::<Float32Array>()
        .expect("vec2 values are Float32");
    [values.value(row * 2), values.value(row * 2 + 1)]
}

/// Decodes every row of `batch` into [`FrameParams`], validating required
/// columns, types, and non-nullness.
pub fn decode_frames(batch: &RecordBatch) -> Result<Vec<FrameParams>, StreamError> {
    let center = require_vec2(batch, "center")?;
    let size = require_vec2(batch, "size")?;
    let theta = require_f32(batch, "theta")?;
    if center.null_count() > 0 {
        return Err(StreamError::NullValues("center"));
    }
    if size.null_count() > 0 {
        return Err(StreamError::NullValues("size"));
    }
    if theta.null_count() > 0 {
        return Err(StreamError::NullValues("theta"));
    }
    Ok((0..batch.num_rows())
        .map(|i| FrameParams {
            center: read_vec2(center, i),
            size: read_vec2(size, i),
            theta: theta.value(i),
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::ArrayRef;

    fn build_input_batch(frames: &[FrameParams]) -> RecordBatch {
        let schema = Arc::new(input_schema());
        let flat_center: Vec<f32> = frames.iter().flat_map(|f| f.center).collect();
        let flat_size: Vec<f32> = frames.iter().flat_map(|f| f.size).collect();
        let thetas: Vec<f32> = frames.iter().map(|f| f.theta).collect();
        let item = Arc::new(Field::new("item", DataType::Float32, false));
        let center = FixedSizeListArray::new(
            item.clone(),
            2,
            Arc::new(Float32Array::from(flat_center)),
            None,
        );
        let size =
            FixedSizeListArray::new(item, 2, Arc::new(Float32Array::from(flat_size)), None);
        let theta = Float32Array::from(thetas);
        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(center) as ArrayRef,
                Arc::new(size) as ArrayRef,
                Arc::new(theta) as ArrayRef,
            ],
        )
        .unwrap()
    }

    #[test]
    fn decodes_frames_roundtrip() {
        let frames = vec![
            FrameParams::IDENTITY,
            FrameParams { center: [0.1, -0.2], size: [0.5, 0.5], theta: 1.25 },
        ];
        let batch = build_input_batch(&frames);
        let decoded = decode_frames(&batch).unwrap();
        assert_eq!(decoded, frames);
    }

    #[test]
    fn missing_column_is_error() {
        let batch = build_input_batch(&[FrameParams::IDENTITY]);
        let reduced = batch.project(&[0, 1]).unwrap(); // drop `theta`
        assert!(matches!(
            decode_frames(&reduced),
            Err(StreamError::MissingColumn("theta"))
        ));
    }

    #[test]
    fn wrong_type_is_error() {
        use arrow::array::Int32Array;
        let schema = Arc::new(Schema::new(vec![
            super::vec2_field("center"),
            super::vec2_field("size"),
            Field::new("theta", DataType::Int32, false),
        ]));
        let item = Arc::new(Field::new("item", DataType::Float32, false));
        let center = FixedSizeListArray::new(
            item.clone(),
            2,
            Arc::new(Float32Array::from(vec![0.0, 0.0])),
            None,
        );
        let size =
            FixedSizeListArray::new(item, 2, Arc::new(Float32Array::from(vec![1.0, 1.0])), None);
        let theta = Int32Array::from(vec![3]);
        let batch = RecordBatch::try_new(
            schema,
            vec![Arc::new(center) as ArrayRef, Arc::new(size), Arc::new(theta)],
        )
        .unwrap();
        assert!(matches!(
            decode_frames(&batch),
            Err(StreamError::ColumnType { column: "theta", .. })
        ));
    }

    #[test]
    fn version_check_rejects_mismatch() {
        let schema = Schema::empty().with_metadata(
            [(PROTOCOL_VERSION_KEY.to_string(), "9.9.9".to_string())]
                .into_iter()
                .collect(),
        );
        assert!(matches!(
            check_version(&schema),
            Err(StreamError::UnsupportedVersion(v)) if v == "9.9.9"
        ));
    }

    #[test]
    fn version_check_allows_absent_and_matching() {
        assert!(check_version(&Schema::empty()).is_ok());
        assert!(check_version(&input_schema()).is_ok());
    }
}
