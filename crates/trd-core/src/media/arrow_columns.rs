//! Arrow column and metadata accessors for the video-editing document.
//!
//! Plumbing, not video knowledge: fetch a typed column, downcast it, read one
//! row, and turn every miss into a [`VideoEditingError`] that names the column.
//!
//! Kept separate from `protocol/arrow_decode.rs`, which does the same job for
//! the render protocol: the two formats have different nullability and column
//! contracts, so sharing the helpers would couple `0.0.6` to
//! `video_edit 0.2.0`. Co-located per format on purpose.

use arrow::array::{
    Array, BinaryArray, BooleanArray, FixedSizeListArray, Float32Array, Int64Array, UInt32Array,
};
use arrow::datatypes::Schema;

use super::video_document::{VideoEditingError, VIDEO_EDIT_VERSION_KEY};

pub(super) fn metadata<'a>(
    schema: &'a Schema,
    key: &'static str,
) -> Result<&'a str, VideoEditingError> {
    schema
        .metadata()
        .get(key)
        .map(String::as_str)
        .ok_or(VideoEditingError::MissingMetadata(key))
}

pub(super) fn metadata_parse<T>(schema: &Schema, key: &'static str) -> Result<T, VideoEditingError>
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

pub(super) fn required_u32<'a>(
    batch: &'a arrow::array::RecordBatch,
    name: &'static str,
) -> Result<&'a UInt32Array, VideoEditingError> {
    downcast(batch, name, "UInt32")
}

pub(super) fn required_i64<'a>(
    batch: &'a arrow::array::RecordBatch,
    name: &'static str,
) -> Result<&'a Int64Array, VideoEditingError> {
    downcast(batch, name, "Int64")
}

pub(super) fn required_bool<'a>(
    batch: &'a arrow::array::RecordBatch,
    name: &'static str,
) -> Result<&'a BooleanArray, VideoEditingError> {
    downcast(batch, name, "Boolean")
}

pub(super) fn optional_binary<'a>(
    batch: &'a arrow::array::RecordBatch,
    name: &'static str,
) -> Result<&'a BinaryArray, VideoEditingError> {
    downcast(batch, name, "Binary")
}

pub(super) fn downcast<'a, T: 'static>(
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

pub(super) fn optional_fixed_f32<'a>(
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

pub(super) fn fixed_value<'a>(
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

pub(super) fn binary_value(array: &BinaryArray, row: usize) -> Option<&[u8]> {
    (!array.is_null(row)).then(|| array.value(row))
}

pub(super) fn value_u32(
    array: &UInt32Array,
    column: &'static str,
    row: usize,
) -> Result<u32, VideoEditingError> {
    if array.is_null(row) {
        return Err(VideoEditingError::NullValue { column, row });
    }
    Ok(array.value(row))
}

pub(super) fn value_i64(
    array: &Int64Array,
    column: &'static str,
    row: usize,
) -> Result<i64, VideoEditingError> {
    if array.is_null(row) {
        return Err(VideoEditingError::NullValue { column, row });
    }
    Ok(array.value(row))
}

pub(super) fn value_bool(
    array: &BooleanArray,
    column: &'static str,
    row: usize,
) -> Result<bool, VideoEditingError> {
    if array.is_null(row) {
        return Err(VideoEditingError::NullValue { column, row });
    }
    Ok(array.value(row))
}

/// Says what a table that is *not* a video-editing document appears to be.
///
/// Every trd table is a `.arrow`, and most of them are **not** documents:
/// render-protocol streams, golden fixtures, raw perception dumps. A file picker
/// can only filter by extension, so reaching for the wrong one is ordinary — and
/// "metadata is missing" named the key the file lacks rather than the thing the
/// user actually did.
pub(super) fn describe_foreign_table(schema: &Schema) -> String {
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
        "it has no trd metadata at all — an unrelated Arrow table: {}",
        summarise_columns(schema)
    )
}

/// The first few column names, so an unrecognised table is at least identifiable.
pub(super) fn summarise_columns(schema: &Schema) -> String {
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
