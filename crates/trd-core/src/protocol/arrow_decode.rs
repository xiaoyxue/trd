//! IO-free Arrow column decode + schema validation for the trd input protocol.
//!
//! The single source of truth for turning a params [`RecordBatch`] into
//! [`FrameParams`] + per-frame [`Draw`] lists, shared verbatim by the native
//! stream decoder (`stream.rs`) and the wasm push decoder ([`super::InputSession`],
//! `protocol.rs`). It performs **no** I/O and holds **no** state machine: it only
//! validates a schema ([`validate_schema`] / [`check_version`]) and decodes the
//! columns of one already-materialized batch. Keeping this in one place is what
//! prevents "fix the bug in one decoder but not the other" divergence (e.g. the
//! nullable-`center` regression `08c113a`): both paths call these functions, so a
//! decode fix lands once. All errors are surfaced as [`super::ProtocolError`]; the
//! native path maps them onto its `StreamError` at the framing boundary.

use arrow::array::{
    Array, FixedSizeListArray, Float32Array, ListArray, RecordBatch, StringArray, UInt32Array,
    UInt8Array,
};
use arrow::datatypes::{DataType, Field, Schema};

use crate::math::Matrix4;
use crate::render::{Draw, DrawSelection};
use crate::{CameraFormError, FrameParams};

use super::{ProtocolError, PROTOCOL_VERSION_KEY, SUPPORTED_INPUT_VERSIONS};

/// Validates a schema's declared protocol version against
/// [`SUPPORTED_INPUT_VERSIONS`]. The current protocol is deliberately not
/// backward compatible, so missing version metadata is rejected too.
/// The single version check shared by the wasm decoder and the native
/// [`crate::run_stream`] (which maps [`ProtocolError`] to its `StreamError`).
pub(crate) fn check_version(schema: &Schema) -> Result<(), ProtocolError> {
    let version = schema
        .metadata()
        .get(PROTOCOL_VERSION_KEY)
        .ok_or(ProtocolError::MissingMetadata(PROTOCOL_VERSION_KEY))?;
    if !SUPPORTED_INPUT_VERSIONS.contains(&version.as_str()) {
        return Err(ProtocolError::UnsupportedVersion(version.clone()));
    }
    Ok(())
}

/// Decodes the optional per-frame external **background frame reference** columns
/// into one `Option<String>` per row, preferring `frame_path` (native filesystem
/// path) then `frame_url` (browser URL). Returns `None` when neither column is
/// present; per-row nulls or empty strings decode to `None`. The core performs
/// no I/O — it only surfaces the reference for the shell to resolve.
pub(crate) fn decode_frame_refs(
    batch: &RecordBatch,
) -> Result<Option<Vec<Option<String>>>, ProtocolError> {
    let path = optional_string(batch, "frame_path")?;
    let url = optional_string(batch, "frame_url")?;
    if path.is_none() && url.is_none() {
        return Ok(None);
    }
    let refs = (0..batch.num_rows())
        .map(|row| {
            nonempty_string(path, row)
                .or_else(|| nonempty_string(url, row))
                .map(str::to_owned)
        })
        .collect();
    Ok(Some(refs))
}

fn optional_string<'a>(
    batch: &'a RecordBatch,
    name: &'static str,
) -> Result<Option<&'a StringArray>, ProtocolError> {
    let Some(column) = batch.column_by_name(name) else {
        return Ok(None);
    };
    column
        .as_any()
        .downcast_ref::<StringArray>()
        .map(Some)
        .ok_or_else(|| ProtocolError::ColumnType {
            column: name,
            expected: "Utf8",
            actual: column.data_type().clone(),
        })
}

fn nonempty_string(column: Option<&StringArray>, row: usize) -> Option<&str> {
    let column = column?;
    (!column.is_null(row) && !column.value(row).is_empty()).then(|| column.value(row))
}

/// Decodes nullable `frame_id` values and validates that every non-null ID
/// addresses the preceding frames table.
pub(crate) fn decode_frame_ids(
    batch: &RecordBatch,
    frames_table_present: bool,
    frame_count: usize,
) -> Result<Option<Vec<Option<u32>>>, ProtocolError> {
    let Some(column) = batch.column_by_name("frame_id") else {
        return Ok(None);
    };
    let ids = column
        .as_any()
        .downcast_ref::<UInt32Array>()
        .ok_or_else(|| ProtocolError::ColumnType {
            column: "frame_id",
            expected: "UInt32",
            actual: column.data_type().clone(),
        })?;

    let mut decoded = Vec::with_capacity(batch.num_rows());
    for row in 0..batch.num_rows() {
        if ids.is_null(row) {
            decoded.push(None);
            continue;
        }
        let frame_id = ids.value(row);
        if !frames_table_present {
            return Err(ProtocolError::MissingFramesTable { row, frame_id });
        }
        if frame_id as usize >= frame_count {
            return Err(ProtocolError::FrameIdOutOfRange {
                row,
                frame_id,
                frame_count,
            });
        }
        decoded.push(Some(frame_id));
    }
    Ok(Some(decoded))
}

/// Decodes the optional per-frame **instanced draw list** columns `draw_mesh`
/// (`List<UInt32>`) and `draw_model` (`List<FixedSizeList<Float32>[16]>`), plus
/// the optional per-draw `draw_mode` (`List<UInt8>`) render-mode override, into
/// one `Vec<Draw>` per row. Returns `Some(rows)` when both required columns are
/// present, `None` when neither is (legacy single-object streams). Having
/// exactly one of the `draw_mesh`/`draw_model` pair, or a per-row length
/// mismatch, is an error. `draw_mode` bytes decode via
/// [`DrawSelection::from_wire`] (`255` = inherit); an absent column leaves every
/// [`Draw::mode`] `None`. Mirrors the native `stream::decode_draws`.
pub(crate) fn decode_draws(batch: &RecordBatch) -> Result<Option<Vec<Vec<Draw>>>, ProtocolError> {
    let (mesh_col, model_col) = match (
        batch.column_by_name("draw_mesh"),
        batch.column_by_name("draw_model"),
    ) {
        (None, None) => return Ok(None),
        (Some(m), Some(n)) => (m, n),
        (Some(_), None) => return Err(ProtocolError::MissingColumn("draw_model")),
        (None, Some(_)) => return Err(ProtocolError::MissingColumn("draw_mesh")),
    };

    let mesh_list = mesh_col
        .as_any()
        .downcast_ref::<ListArray>()
        .ok_or_else(|| ProtocolError::ColumnType {
            column: "draw_mesh",
            expected: "List<UInt32>",
            actual: mesh_col.data_type().clone(),
        })?;
    let model_list = model_col
        .as_any()
        .downcast_ref::<ListArray>()
        .ok_or_else(|| ProtocolError::ColumnType {
            column: "draw_model",
            expected: "List<FixedSizeList<Float32>[16]>",
            actual: model_col.data_type().clone(),
        })?;
    if mesh_list.null_count() > 0 {
        return Err(ProtocolError::NullValues("draw_mesh"));
    }
    if model_list.null_count() > 0 {
        return Err(ProtocolError::NullValues("draw_model"));
    }

    // Optional per-draw render-mode override (`draw_mode`, `List<UInt8>`).
    let mode_list = match batch.column_by_name("draw_mode") {
        None => None,
        Some(col) => {
            let list = col.as_any().downcast_ref::<ListArray>().ok_or_else(|| {
                ProtocolError::ColumnType {
                    column: "draw_mode",
                    expected: "List<UInt8>",
                    actual: col.data_type().clone(),
                }
            })?;
            if list.null_count() > 0 {
                return Err(ProtocolError::NullValues("draw_mode"));
            }
            Some(list.clone())
        }
    };

    let mut rows = Vec::with_capacity(batch.num_rows());
    for row in 0..batch.num_rows() {
        let ids_ref = mesh_list.value(row);
        let ids = ids_ref
            .as_any()
            .downcast_ref::<UInt32Array>()
            .ok_or_else(|| ProtocolError::ColumnType {
                column: "draw_mesh",
                expected: "List<UInt32>",
                actual: ids_ref.data_type().clone(),
            })?;
        if ids.null_count() > 0 {
            return Err(ProtocolError::NullValues("draw_mesh"));
        }

        let models_ref = model_list.value(row);
        let models = models_ref
            .as_any()
            .downcast_ref::<FixedSizeListArray>()
            .filter(|list| list.value_length() == 16)
            .ok_or_else(|| ProtocolError::ColumnType {
                column: "draw_model",
                expected: "FixedSizeList<Float32>[16]",
                actual: models_ref.data_type().clone(),
            })?;
        if models.null_count() > 0 || models.values().null_count() > 0 {
            return Err(ProtocolError::NullValues("draw_model"));
        }
        if ids.len() != models.len() {
            return Err(ProtocolError::MismatchedDrawLists {
                row,
                mesh_len: ids.len(),
                model_len: models.len(),
            });
        }
        let model_values = models
            .values()
            .as_any()
            .downcast_ref::<Float32Array>()
            .ok_or_else(|| ProtocolError::ColumnType {
                column: "draw_model",
                expected: "FixedSizeList<Float32>[16]",
                actual: models.values().data_type().clone(),
            })?;

        // Per-draw selections for this row (empty ⇒ every draw is a mesh
        // inheriting the global mode).
        let selections: Vec<DrawSelection> = match &mode_list {
            None => Vec::new(),
            Some(mode_list) => {
                let modes_ref = mode_list.value(row);
                let bytes = modes_ref
                    .as_any()
                    .downcast_ref::<UInt8Array>()
                    .ok_or_else(|| ProtocolError::ColumnType {
                        column: "draw_mode",
                        expected: "List<UInt8>",
                        actual: modes_ref.data_type().clone(),
                    })?;
                if bytes.null_count() > 0 {
                    return Err(ProtocolError::NullValues("draw_mode"));
                }
                if bytes.len() != ids.len() {
                    return Err(ProtocolError::MismatchedDrawModes {
                        row,
                        mode_len: bytes.len(),
                        draw_len: ids.len(),
                    });
                }
                (0..bytes.len())
                    .map(|j| {
                        DrawSelection::from_wire(bytes.value(j)).ok_or(
                            ProtocolError::InvalidDrawMode {
                                value: bytes.value(j),
                            },
                        )
                    })
                    .collect::<Result<_, _>>()?
            }
        };

        let draws = (0..ids.len())
            .map(|j| Draw {
                mesh_id: ids.value(j),
                // The wire is raw column-major floats; this is the one place it
                // becomes a typed `Matrix4` (#235 R3).
                model: Matrix4::from_cols_array(&read_fixed::<16>(models, model_values, j)),
                selection: selections.get(j).copied().unwrap_or_default(),
            })
            .collect();
        rows.push(draws);
    }
    Ok(Some(rows))
}

pub(crate) fn validate_schema(schema: &Schema) -> Result<(), ProtocolError> {
    check_version(schema)?;

    // Every params column is optional (additive): validate only if present.
    if let Ok(field) = schema.field_with_name("model") {
        validate_fixed_f32_list(field, "model", 16)?;
    }
    if let Ok(field) = schema.field_with_name("k") {
        validate_fixed_f32_list(field, "k", 9)?;
    }
    if let Ok(field) = schema.field_with_name("pose") {
        validate_fixed_f32_list(field, "pose", 16)?;
    }
    // CG camera columns are optional too.
    for name in ["eye", "target", "direction", "up"] {
        if let Ok(field) = schema.field_with_name(name) {
            validate_fixed_f32_list(field, static_name(name), 3)?;
        }
    }
    for name in ["fovy", "aspect", "znear", "zfar"] {
        if let Ok(field) = schema.field_with_name(name) {
            validate_f32_field(field, static_name(name))?;
        }
    }
    if let Ok(field) = schema.field_with_name("frame_id") {
        if field.data_type() != &DataType::UInt32 {
            return Err(ProtocolError::ColumnType {
                column: "frame_id",
                expected: "UInt32",
                actual: field.data_type().clone(),
            });
        }
    }
    for name in ["frame_path", "frame_url"] {
        if let Ok(field) = schema.field_with_name(name) {
            if field.data_type() != &DataType::Utf8 {
                return Err(ProtocolError::ColumnType {
                    column: if name == "frame_path" {
                        "frame_path"
                    } else {
                        "frame_url"
                    },
                    expected: "Utf8",
                    actual: field.data_type().clone(),
                });
            }
        }
    }
    Ok(())
}

/// Interns a known camera-column name to a `'static str` for error messages
/// (the column names are a fixed, closed set).
fn static_name(name: &str) -> &'static str {
    match name {
        "eye" => "eye",
        "target" => "target",
        "direction" => "direction",
        "up" => "up",
        "fovy" => "fovy",
        "aspect" => "aspect",
        "znear" => "znear",
        "zfar" => "zfar",
        _ => "camera",
    }
}

/// Validates that `field` is a `FixedSizeList<Float32>[len]` column.
///
/// The declared *nullability flags* of the field and its list child are
/// intentionally not rejected here. The native decoder (`stream.rs`) only
/// type-checks columns and rejects null *values* at decode time, and producers
/// (e.g. pyarrow) emit nullable-by-default fields carrying non-null values.
/// Rejecting on the flag alone broke the "same stream renders natively and in
/// the browser" invariant (a stream the CLI rendered failed to load in wasm),
/// so this decoder matches native leniency; null *values* are still rejected in
/// `decode_batch` / `optional_fixed_list`.
fn validate_fixed_f32_list(
    field: &Field,
    name: &'static str,
    len: i32,
) -> Result<(), ProtocolError> {
    match field.data_type() {
        DataType::FixedSizeList(item, actual_len)
            if *actual_len == len && item.data_type() == &DataType::Float32 =>
        {
            Ok(())
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
        3 => "FixedSizeList<Float32>[3]",
        9 => "FixedSizeList<Float32>[9]",
        16 => "FixedSizeList<Float32>[16]",
        _ => "FixedSizeList<Float32>[N]",
    }
}

fn validate_f32_field(field: &Field, name: &'static str) -> Result<(), ProtocolError> {
    // Nullability flag intentionally not rejected (see `validate_fixed_f32_list`);
    // null *values* are still rejected at decode time.
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

pub(crate) fn decode_batch(batch: &RecordBatch) -> Result<Vec<FrameParams>, ProtocolError> {
    // Optional matrix columns (validated + null-checked only if present).
    let model = optional_fixed_list(batch, "model", 16)?;
    let k = optional_fixed_list(batch, "k", 9)?;
    let pose = optional_fixed_list(batch, "pose", 16)?;
    // Optional CG camera columns.
    let eye = optional_fixed_list(batch, "eye", 3)?;
    let target = optional_fixed_list(batch, "target", 3)?;
    let direction = optional_fixed_list(batch, "direction", 3)?;
    let up = optional_fixed_list(batch, "up", 3)?;
    let fovy = optional_f32(batch, "fovy")?;
    let aspect = optional_f32(batch, "aspect")?;
    let znear = optional_f32(batch, "znear")?;
    let zfar = optional_f32(batch, "zfar")?;

    (0..batch.num_rows())
        .map(|row| {
            let frame = FrameParams {
                model: model.map(|(list, values)| read_fixed::<16>(list, values, row)),
                k: k.map(|(list, values)| read_fixed::<9>(list, values, row)),
                pose: pose.map(|(list, values)| read_fixed::<16>(list, values, row)),
                eye: eye.map(|(list, values)| read_fixed::<3>(list, values, row)),
                target: target.map(|(list, values)| read_fixed::<3>(list, values, row)),
                direction: direction.map(|(list, values)| read_fixed::<3>(list, values, row)),
                up: up.map(|(list, values)| read_fixed::<3>(list, values, row)),
                fovy: fovy.map(|a| a.value(row)),
                aspect: aspect.map(|a| a.value(row)),
                znear: znear.map(|a| a.value(row)),
                zfar: zfar.map(|a| a.value(row)),
            };
            frame.check_camera_form().map_err(camera_form_error)?;
            Ok(frame)
        })
        .collect()
}

/// Maps a [`CameraFormError`] onto the protocol error type.
fn camera_form_error(error: CameraFormError) -> ProtocolError {
    match error {
        CameraFormError::Conflicting => ProtocolError::ConflictingCameraForms,
        CameraFormError::Incomplete => ProtocolError::IncompleteCameraForm,
    }
}

/// Looks up an optional non-null `Float32` scalar column, validating its type.
/// Returns `None` if the column is absent (additive `0.0.3` camera columns).
fn optional_f32<'a>(
    batch: &'a RecordBatch,
    name: &'static str,
) -> Result<Option<&'a Float32Array>, ProtocolError> {
    let Some(column) = batch.column_by_name(name) else {
        return Ok(None);
    };
    let array = column
        .as_any()
        .downcast_ref::<Float32Array>()
        .ok_or_else(|| ProtocolError::ColumnType {
            column: name,
            expected: "Float32",
            actual: column.data_type().clone(),
        })?;
    if array.null_count() > 0 {
        return Err(ProtocolError::NullValues(name));
    }
    Ok(Some(array))
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
