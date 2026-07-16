use arrow::array::{Array, FixedSizeListArray, Float32Array, RecordBatch};
use arrow::buffer::Buffer;
use arrow::datatypes::{DataType, Field, Schema};
use arrow::error::ArrowError;
use arrow::ipc::reader::StreamDecoder;

use crate::FrameParams;

pub const PROTOCOL_VERSION: &str = "0.0.2";
pub const PROTOCOL_VERSION_KEY: &str = "trd.protocol.version";

/// Input schema versions this build accepts. `0.0.2` adds the optional `model`,
/// `k`, and `pose` matrix columns; `0.0.1` streams (2D affine only) still decode.
pub const SUPPORTED_INPUT_VERSIONS: &[&str] = &["0.0.1", "0.0.2"];

pub type FrameBatch = Vec<FrameParams>;

#[derive(Debug, thiserror::Error)]
pub enum ProtocolError {
    #[error("arrow IPC error: {0}")]
    Arrow(#[from] ArrowError),
    #[error("input session is finished")]
    SessionFinished,
    #[error("input session previously failed")]
    SessionFailed,
    #[error("input stream ended before a schema was decoded")]
    MissingSchema,
    #[error("decoder made no progress while input bytes remained")]
    NoProgress,
    #[error("input schema is missing required field `{0}`")]
    MissingColumn(&'static str),
    #[error("input field `{0}` must be non-nullable")]
    NullableField(&'static str),
    #[error("input field `{0}` has a nullable FixedSizeList child")]
    NullableChild(&'static str),
    #[error("input column `{column}` has type {actual:?}, expected {expected}")]
    ColumnType {
        column: &'static str,
        expected: &'static str,
        actual: DataType,
    },
    #[error("input column `{0}` contains null values")]
    NullValues(&'static str),
    #[error("unsupported protocol version `{0}` (expected `{PROTOCOL_VERSION}`)")]
    UnsupportedVersion(String),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SessionState {
    Open,
    Finished,
    Failed,
}

pub struct InputSession {
    decoder: StreamDecoder,
    schema_validated: bool,
    state: SessionState,
}

impl InputSession {
    pub fn new() -> Self {
        Self {
            decoder: StreamDecoder::new(),
            schema_validated: false,
            state: SessionState::Open,
        }
    }

    pub fn push(&mut self, chunk: &[u8]) -> Result<Vec<FrameBatch>, ProtocolError> {
        self.require_open()?;
        let result = self.push_open(chunk);
        if result.is_err() {
            self.state = SessionState::Failed;
        }
        result
    }

    pub fn finish(&mut self) -> Result<(), ProtocolError> {
        self.require_open()?;

        let result = (|| {
            self.decoder.finish()?;
            self.validate_schema_if_available()?;
            if !self.schema_validated {
                return Err(ProtocolError::MissingSchema);
            }
            Ok(())
        })();

        match result {
            Ok(()) => {
                self.state = SessionState::Finished;
                Ok(())
            }
            Err(error) => {
                self.state = SessionState::Failed;
                Err(error)
            }
        }
    }

    pub fn has_schema(&self) -> bool {
        self.schema_validated
    }

    fn require_open(&self) -> Result<(), ProtocolError> {
        match self.state {
            SessionState::Open => Ok(()),
            SessionState::Finished => Err(ProtocolError::SessionFinished),
            SessionState::Failed => Err(ProtocolError::SessionFailed),
        }
    }

    fn push_open(&mut self, chunk: &[u8]) -> Result<Vec<FrameBatch>, ProtocolError> {
        let mut bytes = Buffer::from_vec(chunk.to_vec());
        let mut batches = Vec::new();

        while !bytes.is_empty() {
            let before = bytes.len();
            let decoded = self.decoder.decode(&mut bytes)?;
            self.validate_schema_if_available()?;

            if let Some(batch) = decoded {
                batches.push(decode_batch(&batch)?);
            }

            if bytes.len() == before {
                return Err(ProtocolError::NoProgress);
            }
        }

        self.validate_schema_if_available()?;
        Ok(batches)
    }

    fn validate_schema_if_available(&mut self) -> Result<(), ProtocolError> {
        if self.schema_validated {
            return Ok(());
        }

        let Some(schema) = self.decoder.schema() else {
            return Ok(());
        };

        validate_schema(schema.as_ref())?;
        self.schema_validated = true;
        Ok(())
    }
}

impl Default for InputSession {
    fn default() -> Self {
        Self::new()
    }
}

fn validate_schema(schema: &Schema) -> Result<(), ProtocolError> {
    if let Some(version) = schema.metadata().get(PROTOCOL_VERSION_KEY) {
        if !SUPPORTED_INPUT_VERSIONS.contains(&version.as_str()) {
            return Err(ProtocolError::UnsupportedVersion(version.clone()));
        }
    }

    validate_vec2_field(
        schema
            .field_with_name("center")
            .map_err(|_| ProtocolError::MissingColumn("center"))?,
        "center",
    )?;
    validate_vec2_field(
        schema
            .field_with_name("size")
            .map_err(|_| ProtocolError::MissingColumn("size"))?,
        "size",
    )?;
    validate_f32_field(
        schema
            .field_with_name("theta")
            .map_err(|_| ProtocolError::MissingColumn("theta"))?,
        "theta",
    )?;

    // `0.0.2` matrix columns are optional (additive): validate only if present.
    if let Ok(field) = schema.field_with_name("model") {
        validate_fixed_f32_list(field, "model", 16)?;
    }
    if let Ok(field) = schema.field_with_name("k") {
        validate_fixed_f32_list(field, "k", 9)?;
    }
    if let Ok(field) = schema.field_with_name("pose") {
        validate_fixed_f32_list(field, "pose", 16)?;
    }
    Ok(())
}

fn validate_vec2_field(field: &Field, name: &'static str) -> Result<(), ProtocolError> {
    validate_fixed_f32_list(field, name, 2)
}

/// Validates that `field` is a non-nullable `FixedSizeList<Float32>[len]` with a
/// non-nullable child.
fn validate_fixed_f32_list(
    field: &Field,
    name: &'static str,
    len: i32,
) -> Result<(), ProtocolError> {
    if field.is_nullable() {
        return Err(ProtocolError::NullableField(name));
    }

    match field.data_type() {
        DataType::FixedSizeList(item, actual_len)
            if *actual_len == len && item.data_type() == &DataType::Float32 =>
        {
            if item.is_nullable() {
                Err(ProtocolError::NullableChild(name))
            } else {
                Ok(())
            }
        }
        actual => Err(ProtocolError::ColumnType {
            column: name,
            expected: fixed_f32_list_expectation(len),
            actual: actual.clone(),
        }),
    }
}

/// The `expected` label for a `FixedSizeList<Float32>[len]` column error.
fn fixed_f32_list_expectation(len: i32) -> &'static str {
    match len {
        2 => "FixedSizeList<Float32>[2]",
        9 => "FixedSizeList<Float32>[9]",
        16 => "FixedSizeList<Float32>[16]",
        _ => "FixedSizeList<Float32>[N]",
    }
}

fn validate_f32_field(field: &Field, name: &'static str) -> Result<(), ProtocolError> {
    if field.is_nullable() {
        return Err(ProtocolError::NullableField(name));
    }

    if field.data_type() == &DataType::Float32 {
        Ok(())
    } else {
        Err(ProtocolError::ColumnType {
            column: name,
            expected: "Float32",
            actual: field.data_type().clone(),
        })
    }
}

fn decode_batch(batch: &RecordBatch) -> Result<FrameBatch, ProtocolError> {
    let center = require_vec2(batch, "center")?;
    let size = require_vec2(batch, "size")?;
    let theta = require_f32(batch, "theta")?;

    if center.null_count() > 0 || center.values().null_count() > 0 {
        return Err(ProtocolError::NullValues("center"));
    }
    if size.null_count() > 0 || size.values().null_count() > 0 {
        return Err(ProtocolError::NullValues("size"));
    }
    if theta.null_count() > 0 {
        return Err(ProtocolError::NullValues("theta"));
    }

    let center_values = center
        .values()
        .as_any()
        .downcast_ref::<Float32Array>()
        .ok_or_else(|| ProtocolError::ColumnType {
            column: "center",
            expected: "FixedSizeList<Float32>[2]",
            actual: center.values().data_type().clone(),
        })?;
    let size_values = size
        .values()
        .as_any()
        .downcast_ref::<Float32Array>()
        .ok_or_else(|| ProtocolError::ColumnType {
            column: "size",
            expected: "FixedSizeList<Float32>[2]",
            actual: size.values().data_type().clone(),
        })?;

    // Optional `0.0.2` matrix columns (validated + null-checked only if present).
    let model = optional_fixed_list(batch, "model", 16)?;
    let k = optional_fixed_list(batch, "k", 9)?;
    let pose = optional_fixed_list(batch, "pose", 16)?;

    Ok((0..batch.num_rows())
        .map(|row| {
            let center_offset = center.value_offset(row) as usize;
            let size_offset = size.value_offset(row) as usize;
            FrameParams {
                center: [
                    center_values.value(center_offset),
                    center_values.value(center_offset + 1),
                ],
                size: [
                    size_values.value(size_offset),
                    size_values.value(size_offset + 1),
                ],
                theta: theta.value(row),
                model: model.map(|(list, values)| read_fixed::<16>(list, values, row)),
                k: k.map(|(list, values)| read_fixed::<9>(list, values, row)),
                pose: pose.map(|(list, values)| read_fixed::<16>(list, values, row)),
            }
        })
        .collect())
}

/// Looks up an optional `FixedSizeList<Float32>[len]` column, validating its
/// type, list length, and non-nullness. Returns `None` if the column is absent.
fn optional_fixed_list<'a>(
    batch: &'a RecordBatch,
    name: &'static str,
    len: i32,
) -> Result<Option<(&'a FixedSizeListArray, &'a Float32Array)>, ProtocolError> {
    let Some(column) = batch.column_by_name(name) else {
        return Ok(None);
    };
    let list = column
        .as_any()
        .downcast_ref::<FixedSizeListArray>()
        .ok_or_else(|| ProtocolError::ColumnType {
            column: name,
            expected: fixed_f32_list_expectation(len),
            actual: column.data_type().clone(),
        })?;
    if list.value_length() != len {
        return Err(ProtocolError::ColumnType {
            column: name,
            expected: fixed_f32_list_expectation(len),
            actual: list.data_type().clone(),
        });
    }
    if list.null_count() > 0 || list.values().null_count() > 0 {
        return Err(ProtocolError::NullValues(name));
    }
    let values = list
        .values()
        .as_any()
        .downcast_ref::<Float32Array>()
        .ok_or_else(|| ProtocolError::ColumnType {
            column: name,
            expected: fixed_f32_list_expectation(len),
            actual: list.values().data_type().clone(),
        })?;
    Ok(Some((list, values)))
}

/// Reads the `N` `f32` values of a fixed-size-list `row`.
fn read_fixed<const N: usize>(
    list: &FixedSizeListArray,
    values: &Float32Array,
    row: usize,
) -> [f32; N] {
    let offset = list.value_offset(row) as usize;
    std::array::from_fn(|i| values.value(offset + i))
}

fn require_vec2<'a>(
    batch: &'a RecordBatch,
    name: &'static str,
) -> Result<&'a FixedSizeListArray, ProtocolError> {
    let column = batch
        .column_by_name(name)
        .ok_or(ProtocolError::MissingColumn(name))?;
    column
        .as_any()
        .downcast_ref::<FixedSizeListArray>()
        .ok_or_else(|| ProtocolError::ColumnType {
            column: name,
            expected: "FixedSizeList<Float32>[2]",
            actual: column.data_type().clone(),
        })
}

fn require_f32<'a>(
    batch: &'a RecordBatch,
    name: &'static str,
) -> Result<&'a Float32Array, ProtocolError> {
    let column = batch
        .column_by_name(name)
        .ok_or(ProtocolError::MissingColumn(name))?;
    column
        .as_any()
        .downcast_ref::<Float32Array>()
        .ok_or_else(|| ProtocolError::ColumnType {
            column: name,
            expected: "Float32",
            actual: column.data_type().clone(),
        })
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow::array::{ArrayRef, FixedSizeListArray, Float32Array, Int32Array, RecordBatch};
    use arrow::buffer::NullBuffer;
    use arrow::datatypes::{DataType, Field, Schema};
    use arrow::ipc::writer::StreamWriter;

    use super::*;
    use crate::FrameParams;

    fn vec2_field(name: &str, nullable: bool, child_nullable: bool) -> Field {
        Field::new(
            name,
            DataType::FixedSizeList(
                Arc::new(Field::new("item", DataType::Float32, child_nullable)),
                2,
            ),
            nullable,
        )
    }

    fn input_schema_with(
        version: Option<&str>,
        center: Field,
        size: Field,
        theta: Field,
    ) -> Arc<Schema> {
        let mut metadata = std::collections::HashMap::new();
        if let Some(version) = version {
            metadata.insert(PROTOCOL_VERSION_KEY.to_owned(), version.to_owned());
        }
        Arc::new(Schema::new(vec![center, size, theta]).with_metadata(metadata))
    }

    fn valid_schema(version: Option<&str>) -> Arc<Schema> {
        input_schema_with(
            version,
            vec2_field("center", false, false),
            vec2_field("size", false, false),
            Field::new("theta", DataType::Float32, false),
        )
    }

    fn test_batch_with(schema: Arc<Schema>, frames: &[FrameParams]) -> RecordBatch {
        let center_item = match schema.field_with_name("center").unwrap().data_type() {
            DataType::FixedSizeList(item, 2) => item.clone(),
            data_type => panic!("unexpected center test type: {data_type:?}"),
        };
        let size_item = match schema.field_with_name("size").unwrap().data_type() {
            DataType::FixedSizeList(item, 2) => item.clone(),
            data_type => panic!("unexpected size test type: {data_type:?}"),
        };
        let center = FixedSizeListArray::new(
            center_item,
            2,
            Arc::new(Float32Array::from(
                frames
                    .iter()
                    .flat_map(|frame| frame.center)
                    .collect::<Vec<_>>(),
            )),
            None,
        );
        let size = FixedSizeListArray::new(
            size_item,
            2,
            Arc::new(Float32Array::from(
                frames
                    .iter()
                    .flat_map(|frame| frame.size)
                    .collect::<Vec<_>>(),
            )),
            None,
        );
        let theta = Float32Array::from(frames.iter().map(|frame| frame.theta).collect::<Vec<_>>());

        RecordBatch::try_new(
            schema,
            vec![Arc::new(center), Arc::new(size), Arc::new(theta)],
        )
        .unwrap()
    }

    fn test_batch(frames: &[FrameParams]) -> RecordBatch {
        test_batch_with(valid_schema(Some(PROTOCOL_VERSION)), frames)
    }

    fn test_stream(batches: &[RecordBatch]) -> Vec<u8> {
        let schema = batches
            .first()
            .map(RecordBatch::schema)
            .unwrap_or_else(|| valid_schema(Some(PROTOCOL_VERSION)));
        let mut bytes = Vec::new();
        let mut writer = StreamWriter::try_new(&mut bytes, &schema).unwrap();
        for batch in batches {
            writer.write(batch).unwrap();
        }
        writer.finish().unwrap();
        bytes
    }

    fn missing_theta_batch() -> RecordBatch {
        test_batch(&[FrameParams::IDENTITY])
            .project(&[0, 1])
            .unwrap()
    }

    fn wrong_theta_batch() -> RecordBatch {
        let frame = FrameParams::IDENTITY;
        let center = FixedSizeListArray::new(
            Arc::new(Field::new("item", DataType::Float32, false)),
            2,
            Arc::new(Float32Array::from(vec![frame.center[0], frame.center[1]])),
            None,
        );
        let size = FixedSizeListArray::new(
            Arc::new(Field::new("item", DataType::Float32, false)),
            2,
            Arc::new(Float32Array::from(vec![frame.size[0], frame.size[1]])),
            None,
        );
        RecordBatch::try_new(
            Arc::new(Schema::new(vec![
                vec2_field("center", false, false),
                vec2_field("size", false, false),
                Field::new("theta", DataType::Int32, false),
            ])),
            vec![
                Arc::new(center),
                Arc::new(size),
                Arc::new(Int32Array::from(vec![0])),
            ],
        )
        .unwrap()
    }

    fn null_center_batch() -> RecordBatch {
        let center = FixedSizeListArray::new(
            Arc::new(Field::new("item", DataType::Float32, false)),
            2,
            Arc::new(Float32Array::from(vec![0.0, 0.0])),
            Some(NullBuffer::new_null(1)),
        );
        let size = FixedSizeListArray::new(
            Arc::new(Field::new("item", DataType::Float32, false)),
            2,
            Arc::new(Float32Array::from(vec![1.0, 1.0])),
            None,
        );
        let schema = valid_schema(Some(PROTOCOL_VERSION));
        let columns = vec![
            Arc::new(center) as ArrayRef,
            Arc::new(size) as ArrayRef,
            Arc::new(Float32Array::from(vec![0.0])) as ArrayRef,
        ];
        // SAFETY: This fixture intentionally violates non-nullability metadata
        // to verify runtime null rejection after IPC decoding.
        unsafe { RecordBatch::new_unchecked(schema, columns, 1) }
    }

    fn version_stream(version: &str) -> Vec<u8> {
        let frame = test_batch(&[FrameParams::IDENTITY]);
        let changed =
            RecordBatch::try_new(valid_schema(Some(version)), frame.columns().to_vec()).unwrap();
        test_stream(&[changed])
    }

    #[test]
    fn decodes_every_two_part_split() {
        let expected = vec![
            FrameParams::IDENTITY,
            FrameParams {
                center: [0.25, -0.5],
                size: [0.75, 0.5],
                theta: 1.0,
                ..FrameParams::IDENTITY
            },
        ];
        let bytes = test_stream(&[test_batch(&expected)]);

        for split in 0..=bytes.len() {
            let mut session = InputSession::new();
            let mut batches = session.push(&bytes[..split]).unwrap();
            batches.extend(session.push(&bytes[split..]).unwrap());
            session.finish().unwrap();
            assert_eq!(batches, vec![expected.clone()]);
        }
    }

    #[test]
    fn decodes_one_byte_fragments_and_multiple_batches() {
        let first = vec![FrameParams::IDENTITY];
        let second = vec![FrameParams {
            center: [0.5, 0.0],
            size: [0.25, 0.75],
            theta: 0.5,
            ..FrameParams::IDENTITY
        }];
        let bytes = test_stream(&[test_batch(&first), test_batch(&second)]);
        let mut session = InputSession::new();
        let mut batches = Vec::new();

        for byte in bytes {
            batches.extend(session.push(&[byte]).unwrap());
        }

        session.finish().unwrap();
        assert_eq!(batches, vec![first, second]);
    }

    #[test]
    fn rejects_schema_type_nullability_and_runtime_nulls() {
        let nullable_center = test_batch_with(
            input_schema_with(
                Some(PROTOCOL_VERSION),
                vec2_field("center", true, false),
                vec2_field("size", false, false),
                Field::new("theta", DataType::Float32, false),
            ),
            &[FrameParams::IDENTITY],
        );
        let nullable_child = test_batch_with(
            input_schema_with(
                Some(PROTOCOL_VERSION),
                vec2_field("center", false, true),
                vec2_field("size", false, false),
                Field::new("theta", DataType::Float32, false),
            ),
            &[FrameParams::IDENTITY],
        );

        let mut missing = InputSession::new();
        assert!(matches!(
            missing.push(&test_stream(&[missing_theta_batch()])),
            Err(ProtocolError::MissingColumn("theta"))
        ));

        let mut wrong = InputSession::new();
        assert!(matches!(
            wrong.push(&test_stream(&[wrong_theta_batch()])),
            Err(ProtocolError::ColumnType {
                column: "theta",
                ..
            })
        ));

        let mut nullable = InputSession::new();
        assert!(matches!(
            nullable.push(&test_stream(&[nullable_center])),
            Err(ProtocolError::NullableField("center"))
        ));

        let mut child = InputSession::new();
        assert!(matches!(
            child.push(&test_stream(&[nullable_child])),
            Err(ProtocolError::NullableChild("center"))
        ));

        assert!(matches!(
            decode_batch(&null_center_batch()),
            Err(ProtocolError::NullValues("center"))
        ));

        let mut version = InputSession::new();
        assert!(matches!(
            version.push(&version_stream("9.9.9")),
            Err(ProtocolError::UnsupportedVersion(value)) if value == "9.9.9"
        ));

        let without_version = test_batch_with(valid_schema(None), &[FrameParams::IDENTITY]);
        let mut compatible = InputSession::new();
        assert_eq!(
            compatible.push(&test_stream(&[without_version])).unwrap(),
            vec![vec![FrameParams::IDENTITY]]
        );
        compatible.finish().unwrap();
    }

    #[test]
    fn rejects_repeated_schema_truncation_eos_bytes_and_later_calls() {
        let valid = test_stream(&[test_batch(&[FrameParams::IDENTITY])]);

        let mut repeated_bytes = valid.clone();
        repeated_bytes.truncate(repeated_bytes.len() - 8);
        repeated_bytes.extend(valid.clone());
        let mut repeated = InputSession::new();
        assert!(repeated.push(&repeated_bytes).is_err());
        assert!(matches!(
            repeated.push(&[]),
            Err(ProtocolError::SessionFailed)
        ));
        assert!(matches!(
            repeated.finish(),
            Err(ProtocolError::SessionFailed)
        ));

        let mut truncated = InputSession::new();
        truncated.push(&valid[..valid.len() - 1]).unwrap();
        assert!(truncated.finish().is_err());
        assert!(matches!(
            truncated.push(&[]),
            Err(ProtocolError::SessionFailed)
        ));

        let mut after_eos = InputSession::new();
        after_eos.push(&valid).unwrap();
        assert!(after_eos.push(&[0]).is_err());
        assert!(matches!(
            after_eos.finish(),
            Err(ProtocolError::SessionFailed)
        ));

        let mut finished = InputSession::new();
        finished.push(&valid).unwrap();
        finished.finish().unwrap();
        assert!(matches!(
            finished.push(&[]),
            Err(ProtocolError::SessionFinished)
        ));
        assert!(matches!(
            finished.finish(),
            Err(ProtocolError::SessionFinished)
        ));
    }

    // ---- protocol 0.0.2 matrix columns (model / k / pose) ----

    fn fixed_list_field(name: &str, len: i32, nullable: bool, child_nullable: bool) -> Field {
        Field::new(
            name,
            DataType::FixedSizeList(
                Arc::new(Field::new("item", DataType::Float32, child_nullable)),
                len,
            ),
            nullable,
        )
    }

    /// Builds a batch of `rows.len()` identity frames plus one extra fixed-size
    /// list column (`name`) whose row `i` holds `rows[i]`.
    fn batch_with_matrix(
        version: Option<&str>,
        name: &str,
        len: i32,
        rows: &[Vec<f32>],
        nullable: bool,
        child_nullable: bool,
    ) -> RecordBatch {
        let n = rows.len();
        let item = Arc::new(Field::new("item", DataType::Float32, false));
        let center = FixedSizeListArray::new(
            item.clone(),
            2,
            Arc::new(Float32Array::from(vec![0.0_f32; n * 2])),
            None,
        );
        let size = FixedSizeListArray::new(
            item.clone(),
            2,
            Arc::new(Float32Array::from(vec![1.0_f32; n * 2])),
            None,
        );
        let theta = Float32Array::from(vec![0.0_f32; n]);
        let flat: Vec<f32> = rows.iter().flatten().copied().collect();
        let matrix = FixedSizeListArray::new(
            Arc::new(Field::new("item", DataType::Float32, child_nullable)),
            len,
            Arc::new(Float32Array::from(flat)),
            None,
        );

        let mut metadata = std::collections::HashMap::new();
        if let Some(version) = version {
            metadata.insert(PROTOCOL_VERSION_KEY.to_owned(), version.to_owned());
        }
        let schema = Arc::new(
            Schema::new(vec![
                vec2_field("center", false, false),
                vec2_field("size", false, false),
                Field::new("theta", DataType::Float32, false),
                fixed_list_field(name, len, nullable, child_nullable),
            ])
            .with_metadata(metadata),
        );

        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(center),
                Arc::new(size),
                Arc::new(theta),
                Arc::new(matrix),
            ],
        )
        .unwrap()
    }

    #[test]
    fn accepts_supported_versions_and_rejects_unknown() {
        for version in ["0.0.1", "0.0.2"] {
            let mut session = InputSession::new();
            session.push(&version_stream(version)).unwrap();
            session.finish().unwrap();
        }
        let mut session = InputSession::new();
        assert!(matches!(
            session.push(&version_stream("0.0.3")),
            Err(ProtocolError::UnsupportedVersion(v)) if v == "0.0.3"
        ));
    }

    #[test]
    fn decodes_matrix_columns_column_major() {
        // Asymmetric values so a transpose or stride bug can't hide.
        let model_row: Vec<f32> = (1..=16).map(|v| v as f32).collect();
        let k_row: Vec<f32> = (1..=9).map(|v| v as f32 * 0.5).collect();
        let pose_row: Vec<f32> = (1..=16).map(|v| -(v as f32)).collect();

        let model_batch = batch_with_matrix(
            Some("0.0.2"),
            "model",
            16,
            std::slice::from_ref(&model_row),
            false,
            false,
        );
        let k_batch = batch_with_matrix(
            Some("0.0.2"),
            "k",
            9,
            std::slice::from_ref(&k_row),
            false,
            false,
        );
        let pose_batch = batch_with_matrix(
            Some("0.0.2"),
            "pose",
            16,
            std::slice::from_ref(&pose_row),
            false,
            false,
        );

        let model = decode_batch(&model_batch).unwrap();
        assert_eq!(
            model[0].model,
            Some(<[f32; 16]>::try_from(model_row).unwrap())
        );
        assert_eq!(model[0].k, None);

        let k = decode_batch(&k_batch).unwrap();
        assert_eq!(k[0].k, Some(<[f32; 9]>::try_from(k_row).unwrap()));

        let pose = decode_batch(&pose_batch).unwrap();
        assert_eq!(pose[0].pose, Some(<[f32; 16]>::try_from(pose_row).unwrap()));
    }

    #[test]
    fn rejects_wrong_size_matrix_column() {
        // A `model` column declared as length 9 (not 16) is a schema error, and
        // the session becomes terminal.
        let bad = batch_with_matrix(Some("0.0.2"), "model", 9, &[vec![0.0; 9]], false, false);
        let mut session = InputSession::new();
        assert!(matches!(
            session.push(&test_stream(&[bad])),
            Err(ProtocolError::ColumnType {
                column: "model",
                ..
            })
        ));
        assert!(matches!(
            session.push(&[]),
            Err(ProtocolError::SessionFailed)
        ));
    }

    #[test]
    fn validate_if_present_rejects_nullable_matrix_column() {
        let nullable_field =
            batch_with_matrix(Some("0.0.2"), "pose", 16, &[vec![0.0; 16]], true, false);
        let mut session = InputSession::new();
        assert!(matches!(
            session.push(&test_stream(&[nullable_field])),
            Err(ProtocolError::NullableField("pose"))
        ));
    }

    proptest::proptest! {
        #[test]
        fn model_column_roundtrips(values in proptest::collection::vec(-1000.0_f32..1000.0, 16)) {
            let batch = batch_with_matrix(Some("0.0.2"), "model", 16, std::slice::from_ref(&values), false, false);
            let decoded = decode_batch(&batch).unwrap();
            let expected = <[f32; 16]>::try_from(values).unwrap();
            proptest::prop_assert_eq!(decoded[0].model, Some(expected));
        }
    }
}
