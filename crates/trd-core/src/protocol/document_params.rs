use std::sync::Arc;

use arrow::array::{
    Array, FixedSizeBinaryArray, FixedSizeListArray, Float32Array, Int64Array, ListArray,
    RecordBatch, StringArray, StructArray,
};
use arrow::datatypes::{DataType, Field, Schema};
use arrow_schema::extension::Uuid as ArrowUuid;
use uuid::Uuid;

use super::arrow_decode::{decode_batch, decode_draws_for_meshes, validate_schema};
use super::{parse_error, ProtocolError, SceneDocument};
use crate::{DrawSelection, FrameParams, Matrix4, Viewport};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DocumentMesh {
    Id(Uuid),
    Index(u32),
}

#[derive(Debug, Clone, PartialEq)]
pub struct DocumentObject {
    pub mesh: Option<DocumentMesh>,
    pub model: Matrix4,
    pub quad: Option<[[f32; 2]; 4]>,
    pub track_id: Option<String>,
    pub selection: DrawSelection,
}

/// A rendering view of one source row. The original arrays remain in the document.
#[derive(Debug, Clone, PartialEq)]
pub struct DocumentFrame {
    pub params: FrameParams,
    pub source_size: Option<Viewport>,
    pub present_index: Option<i64>,
    pub pts: Option<i64>,
    pub objects: Vec<DocumentObject>,
}

impl SceneDocument {
    /// Decodes just one source row for playback, without rebuilding the full timeline.
    pub fn frame(&self, row: usize) -> Result<DocumentFrame, ProtocolError> {
        let slots = (0..self.meshes().len())
            .map(|slot| u32::try_from(slot).map_err(|_| parse_error("too many meshes")))
            .collect::<Result<Vec<_>, _>>()?;
        let mut local = row;
        for batch in self.batches() {
            if local < batch.num_rows() {
                let slice = batch.slice(local, 1);
                let frames = if is_tracked(&slice) {
                    decode_tracked(&slice)?
                } else {
                    decode_camera_params(&slice, &slots)?
                };
                return frames
                    .into_iter()
                    .next()
                    .ok_or_else(|| parse_error("empty params row"));
            }
            local -= batch.num_rows();
        }
        Err(parse_error(format!("params row {row} is out of range")))
    }

    pub fn frames(&self) -> Result<Vec<DocumentFrame>, ProtocolError> {
        let slots = (0..self.meshes().len())
            .map(|slot| u32::try_from(slot).map_err(|_| parse_error("too many meshes")))
            .collect::<Result<Vec<_>, _>>()?;
        let mut frames = Vec::new();
        let mut identities = std::collections::HashSet::new();
        for batch in self.batches() {
            let mut decoded = if is_tracked(batch) {
                decode_tracked(batch)?
            } else {
                decode_camera_params(batch, &slots)?
            };
            for frame in &decoded {
                if let Some(index) = frame.present_index {
                    if index < 0 || !identities.insert(index) {
                        return Err(parse_error("present_index must be nonnegative and unique"));
                    }
                }
            }
            frames.append(&mut decoded);
        }
        Ok(frames)
    }

    /// Maps a source binding to a renderer-local slot without changing its wire ID.
    pub fn mesh_slot(&self, binding: DocumentMesh) -> Result<usize, ProtocolError> {
        let slot = match binding {
            DocumentMesh::Id(id) => self.meshes().iter().position(|mesh| mesh.id() == id),
            DocumentMesh::Index(index) => {
                ((index as usize) < self.meshes().len()).then_some(index as usize)
            }
        };
        slot.ok_or_else(|| parse_error(format!("mesh binding {binding:?} is unresolved")))
    }
}

pub(super) fn is_tracked(batch: &RecordBatch) -> bool {
    batch.column_by_name("bottom_quads").is_some()
        || batch
            .column_by_name("k")
            .is_some_and(|column| matches!(column.data_type(), DataType::Struct(_)))
}

fn decode_camera_params(
    batch: &RecordBatch,
    mesh_ids: &[u32],
) -> Result<Vec<DocumentFrame>, ProtocolError> {
    validate_schema(&batch.schema())?;
    let params = decode_batch(batch)?;
    let draws = decode_draws_for_meshes(batch, mesh_ids)?;
    let indices = optional_i64(batch, "present_index")?;
    let pts = optional_i64(batch, "pts")?;
    let video_indices = super::decode_video_frame_indices(batch)?;
    params
        .into_iter()
        .enumerate()
        .map(|(row, params)| {
            let objects = draws.as_ref().map_or_else(
                || {
                    vec![DocumentObject {
                        mesh: Some(DocumentMesh::Index(0)),
                        model: params.model_matrix(),
                        quad: None,
                        track_id: None,
                        selection: DrawSelection::INHERIT,
                    }]
                },
                |draws| {
                    draws[row]
                        .iter()
                        .map(|draw| DocumentObject {
                            mesh: Some(DocumentMesh::Index(draw.mesh_id)),
                            model: draw.model,
                            quad: None,
                            track_id: None,
                            selection: draw.selection,
                        })
                        .collect()
                },
            );
            for object in &objects {
                validate_model(object.model)?;
            }
            Ok(DocumentFrame {
                params,
                source_size: None,
                present_index: indices
                    .map(|values| values.value(row))
                    .or_else(|| video_indices.as_ref().map(|values| i64::from(values[row]))),
                pts: pts.map(|values| values.value(row)),
                objects,
            })
        })
        .collect()
}

fn decode_tracked(batch: &RecordBatch) -> Result<Vec<DocumentFrame>, ProtocolError> {
    let quads =
        list_column(batch, "bottom_quads")?.ok_or(ProtocolError::MissingColumn("bottom_quads"))?;
    let k = batch
        .column_by_name("k")
        .ok_or(ProtocolError::MissingColumn("k"))?
        .as_any()
        .downcast_ref::<StructArray>()
        .ok_or_else(|| parse_error("tracked k must be the FHC intrinsics struct"))?;
    require_nonnull(k, "k")?;
    let models = list_column(batch, "model")?;
    let mesh_ids = list_column(batch, "mesh_id")?;
    let tracks = list_column(batch, "track_id")?;
    let indices = optional_i64(batch, "present_index")?;
    let pts = optional_i64(batch, "pts")?;
    if let Some(ids) = mesh_ids {
        let DataType::List(item) = ids.data_type() else {
            unreachable!()
        };
        item.try_extension_type::<ArrowUuid>()?;
    }

    let mut matrices = Vec::with_capacity(batch.num_rows());
    let mut sizes = Vec::with_capacity(batch.num_rows());
    for row in 0..batch.num_rows() {
        let w = component(k, "w", row)?;
        let h = component(k, "h", row)?;
        let fx = component(k, "fx", row)?;
        let fy = component(k, "fy", row)?;
        if w <= 0.0
            || h <= 0.0
            || fx <= 0.0
            || fy <= 0.0
            || w.fract() != 0.0
            || h.fract() != 0.0
            || f64::from(w) > f64::from(u32::MAX)
            || f64::from(h) > f64::from(u32::MAX)
        {
            return Err(parse_error(
                "FHC frame dimensions and focal lengths must be positive",
            ));
        }
        matrices.extend_from_slice(&[
            fx / 2.0,
            0.0,
            0.0,
            component(k, "skew", row)? / 2.0,
            fy / 2.0,
            0.0,
            (component(k, "cx", row)? + w) / 2.0,
            (component(k, "cy", row)? + h) / 2.0,
            1.0,
        ]);
        sizes.push(Viewport {
            width: w as u32,
            height: h as u32,
        });
    }

    // Convert only a temporary camera view; preserve source FHC and model columns.
    let intrinsics = FixedSizeListArray::new(
        Arc::new(Field::new("item", DataType::Float32, false)),
        9,
        Arc::new(Float32Array::from(matrices)),
        None,
    );
    let fields = vec![Field::new("k", intrinsics.data_type().clone(), false)];
    let metadata = [
        (
            super::PROTOCOL_VERSION_KEY.to_owned(),
            super::PROTOCOL_VERSION.to_owned(),
        ),
        (super::TABLE_KIND_KEY.to_owned(), "params".to_owned()),
    ]
    .into_iter()
    .collect();
    let schema = Schema::new_with_metadata(fields, metadata);
    let camera_batch = RecordBatch::try_new(Arc::new(schema), vec![Arc::new(intrinsics)])?;
    validate_schema(&camera_batch.schema())?;
    let params = decode_batch(&camera_batch)?;
    params
        .into_iter()
        .enumerate()
        .map(|(row, params)| {
            let values = quads.value(row);
            let quad_values = values
                .as_any()
                .downcast_ref::<FixedSizeListArray>()
                .filter(|values| values.value_length() == 4)
                .ok_or_else(|| {
                    parse_error("bottom_quads must contain fixed lists of four points")
                })?;
            require_nonnull(quad_values, "bottom_quads")?;
            let count = quad_values.len();
            for (name, column) in [
                ("model", models),
                ("mesh_id", mesh_ids),
                ("track_id", tracks),
            ] {
                if column.is_some_and(|column| column.value_length(row) as usize != count) {
                    return Err(parse_error(format!(
                        "{name} must align with bottom_quads at row {row}"
                    )));
                }
            }
            let row_models = models.map(|models| models.value(row));
            let row_ids = mesh_ids.map(|ids| ids.value(row));
            let row_tracks = tracks.map(|tracks| tracks.value(row));
            let mut unique_tracks = std::collections::HashSet::new();
            let mut objects = Vec::with_capacity(count);
            for index in 0..count {
                let points = quad_values.value(index);
                let points = points
                    .as_any()
                    .downcast_ref::<StructArray>()
                    .ok_or_else(|| parse_error("quad points must be FHC structs"))?;
                require_nonnull(points, "bottom_quads")?;
                let mut quad = [[0.0; 2]; 4];
                for (point, pixel) in quad.iter_mut().enumerate() {
                    let w = component(points, "w", point)?;
                    let h = component(points, "h", point)?;
                    let x = component(points, "x", point)?;
                    let y = component(points, "y", point)?;
                    if w != sizes[row].width as f32
                        || h != sizes[row].height as f32
                        || x.abs() > w
                        || y.abs() > h
                    {
                        return Err(parse_error(
                            "quad points must lie in the camera's FHC frame",
                        ));
                    }
                    *pixel = [(x + w) / 2.0, (y + h) / 2.0];
                }
                validate_quad(quad)?;
                let model = row_models
                    .as_ref()
                    .map_or(Ok(Matrix4::IDENTITY), |models| {
                        read_source_model(models.as_ref(), index)
                    })?;
                let mesh = row_ids
                    .as_ref()
                    .map(|values| {
                        let ids = values
                            .as_any()
                            .downcast_ref::<FixedSizeBinaryArray>()
                            .filter(|ids| ids.value_length() == 16)
                            .ok_or_else(|| {
                                parse_error("mesh_id items must use Arrow UUID storage")
                            })?;
                        require_nonnull(ids, "mesh_id")?;
                        Uuid::from_slice(ids.value(index))
                            .map(DocumentMesh::Id)
                            .map_err(|error| parse_error(format!("mesh_id: {error}")))
                    })
                    .transpose()?;
                let track_id = row_tracks
                    .as_ref()
                    .map(|values| {
                        let ids = values
                            .as_any()
                            .downcast_ref::<StringArray>()
                            .ok_or_else(|| parse_error("track_id items must be strings"))?;
                        require_nonnull(ids, "track_id")?;
                        let id = ids.value(index);
                        if id.is_empty() || !unique_tracks.insert(id.to_owned()) {
                            return Err(parse_error(
                                "track_id must be nonempty and unique within a frame",
                            ));
                        }
                        Ok(id.to_owned())
                    })
                    .transpose()?;
                objects.push(DocumentObject {
                    mesh,
                    model,
                    quad: Some(quad),
                    track_id,
                    selection: DrawSelection::INHERIT,
                });
            }
            Ok(DocumentFrame {
                params,
                source_size: Some(sizes[row]),
                present_index: indices.map(|values| values.value(row)),
                pts: pts.map(|values| values.value(row)),
                objects,
            })
        })
        .collect()
}

pub(super) fn read_source_model(
    values: &dyn Array,
    index: usize,
) -> Result<Matrix4, ProtocolError> {
    let matrices = values
        .as_any()
        .downcast_ref::<FixedSizeListArray>()
        .filter(|values| values.value_length() == 4)
        .ok_or_else(|| parse_error("model items must be row-major nested 4x4 matrices"))?;
    require_nonnull(matrices, "model")?;
    let rows = matrices.value(index);
    let rows = rows
        .as_any()
        .downcast_ref::<FixedSizeListArray>()
        .filter(|rows| rows.value_length() == 4)
        .ok_or_else(|| parse_error("model items must contain four rows of four floats"))?;
    require_nonnull(rows, "model")?;
    let mut columns = [0.0; 16];
    for row in 0..4 {
        let values = rows.value(row);
        let values = values
            .as_any()
            .downcast_ref::<Float32Array>()
            .ok_or_else(|| parse_error("model values must be float32"))?;
        require_nonnull(values, "model")?;
        for col in 0..4 {
            columns[col * 4 + row] = values.value(col);
        }
    }
    let model = Matrix4::from_cols_array(&columns);
    validate_model(model)?;
    Ok(model)
}

pub(super) fn validate_model(model: Matrix4) -> Result<(), ProtocolError> {
    let values = model.to_cols_array();
    if !values.iter().all(|value| value.is_finite())
        || values[3].abs() > 1e-6
        || values[7].abs() > 1e-6
        || values[11].abs() > 1e-6
        || (values[15] - 1.0).abs() > 1e-6
    {
        return Err(parse_error("model must be a finite affine 4x4 transform"));
    }
    Ok(())
}

fn component(values: &StructArray, name: &str, row: usize) -> Result<f32, ProtocolError> {
    let array = values
        .column_by_name(name)
        .and_then(|column| column.as_any().downcast_ref::<Float32Array>())
        .ok_or_else(|| parse_error(format!("FHC field {name} must be float32")))?;
    if array.is_null(row) || !array.value(row).is_finite() {
        return Err(parse_error(format!(
            "FHC field {name} must be finite and non-null"
        )));
    }
    Ok(array.value(row))
}

fn optional_i64<'a>(
    batch: &'a RecordBatch,
    name: &str,
) -> Result<Option<&'a Int64Array>, ProtocolError> {
    batch
        .column_by_name(name)
        .map(|column| {
            let values = column
                .as_any()
                .downcast_ref::<Int64Array>()
                .ok_or_else(|| parse_error(format!("{name} must be int64")))?;
            require_nonnull(values, name)?;
            Ok(values)
        })
        .transpose()
}

pub(super) fn list_column<'a>(
    batch: &'a RecordBatch,
    name: &str,
) -> Result<Option<&'a ListArray>, ProtocolError> {
    batch
        .column_by_name(name)
        .map(|column| {
            let values = column
                .as_any()
                .downcast_ref::<ListArray>()
                .ok_or_else(|| parse_error(format!("{name} must be a list")))?;
            require_nonnull(values, name)?;
            Ok(values)
        })
        .transpose()
}

fn require_nonnull(values: &dyn Array, name: &str) -> Result<(), ProtocolError> {
    if values.null_count() != 0 {
        return Err(parse_error(format!("{name} must not contain nulls")));
    }
    Ok(())
}

fn validate_quad(quad: [[f32; 2]; 4]) -> Result<(), ProtocolError> {
    for index in 0..4 {
        let a = quad[index];
        let b = quad[(index + 1) % 4];
        let c = quad[(index + 2) % 4];
        let cross = (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0]);
        if !cross.is_finite() || cross <= 0.0 {
            return Err(parse_error(
                "bottom_quads must be convex, clockwise and nondegenerate",
            ));
        }
    }
    if quad
        .iter()
        .skip(1)
        .any(|point| point[1] < quad[0][1] || (point[1] == quad[0][1] && point[0] < quad[0][0]))
    {
        return Err(parse_error(
            "bottom_quads must start at minimum y, then minimum x",
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ModelEdit, PROTOCOL_VERSION, PROTOCOL_VERSION_KEY, TABLE_KIND_KEY};
    use arrow::array::{ArrayRef, Int64Array};
    use arrow::buffer::OffsetBuffer;

    fn metadata() -> std::collections::HashMap<String, String> {
        [
            (PROTOCOL_VERSION_KEY.to_owned(), PROTOCOL_VERSION.to_owned()),
            (TABLE_KIND_KEY.to_owned(), "params".to_owned()),
        ]
        .into_iter()
        .collect()
    }

    fn floats(values: Vec<f32>) -> ArrayRef {
        Arc::new(Float32Array::from(values))
    }

    fn fhc_document() -> SceneDocument {
        let k_fields = ["fx", "fy", "cx", "cy", "skew", "w", "h"]
            .map(|name| Arc::new(Field::new(name, DataType::Float32, false)));
        let k = StructArray::new(
            k_fields.to_vec().into(),
            [2000.0, 2000.0, 0.0, 0.0, 0.0, 1920.0, 1080.0]
                .map(|value| floats(vec![value]))
                .to_vec(),
            None,
        );
        let point_fields =
            ["x", "y", "w", "h"].map(|name| Arc::new(Field::new(name, DataType::Float32, false)));
        let points = StructArray::new(
            point_fields.to_vec().into(),
            vec![
                floats(vec![-200.0, 200.0, 200.0, -200.0]),
                floats(vec![-200.0, -200.0, 200.0, 200.0]),
                floats(vec![1920.0; 4]),
                floats(vec![1080.0; 4]),
            ],
            None,
        );
        let quads = FixedSizeListArray::new(
            Arc::new(Field::new("point", points.data_type().clone(), false)),
            4,
            Arc::new(points),
            None,
        );
        let placements = ListArray::new(
            Arc::new(Field::new("quad", quads.data_type().clone(), false)),
            OffsetBuffer::from_lengths([1]),
            Arc::new(quads),
            None,
        );
        let columns: Vec<ArrayRef> = vec![
            Arc::new(placements),
            Arc::new(k),
            Arc::new(Int64Array::from(vec![10])),
            Arc::new(Int64Array::from(vec![-20])),
        ];
        let fields = ["bottom_quads", "k", "present_index", "pts"]
            .into_iter()
            .zip(&columns)
            .map(|(name, array)| Field::new(name, array.data_type().clone(), false))
            .collect::<Vec<_>>();
        let schema = Arc::new(Schema::new(fields).with_metadata(metadata()));
        let batch = RecordBatch::try_new(Arc::clone(&schema), columns).unwrap();
        SceneDocument::from_batches(schema, vec![batch], vec![]).unwrap()
    }

    #[test]
    fn minimal_fhc_fields_decode_without_business_columns() {
        let doc = fhc_document();
        let frames = doc.frames().unwrap();
        assert_eq!(frames[0].present_index, Some(10));
        assert_eq!(frames[0].pts, Some(-20));
        assert_eq!(
            frames[0].source_size,
            Some(Viewport {
                width: 1920,
                height: 1080
            })
        );
        assert_eq!(
            frames[0].params.k,
            Some([1000.0, 0.0, 0.0, 0.0, 1000.0, 0.0, 960.0, 540.0, 1.0,])
        );
        assert_eq!(
            frames[0].objects[0].quad,
            Some([
                [860.0, 440.0],
                [1060.0, 440.0],
                [1060.0, 640.0],
                [860.0, 640.0],
            ])
        );
        assert_eq!(frames[0].objects[0].model, Matrix4::IDENTITY);
        assert_eq!(doc.schema().fields().len(), 4);
    }

    #[test]
    fn existing_cg_camera_and_identity_draw_are_unchanged() {
        let params = FrameParams {
            eye: Some([0.0, 2.0, 6.0]),
            target: Some([0.0, 0.0, 0.0]),
            fovy: Some(0.7),
            ..FrameParams::IDENTITY
        };
        let bytes = super::super::scene_encode::encode_params_stream(&[params], None).unwrap();
        let doc = SceneDocument::read(&bytes).unwrap();
        assert_eq!(doc.frames().unwrap()[0].params, params);
        assert_eq!(doc.frames().unwrap()[0].objects[0].model, Matrix4::IDENTITY);
    }

    #[test]
    fn frame_identity_is_not_an_fps_derived_unsigned_value() {
        let doc = fhc_document();
        let decoded = SceneDocument::read(&doc.write().unwrap()).unwrap();
        assert_eq!(decoded.frames().unwrap()[0].pts, Some(-20));
        assert_eq!(decoded.frames().unwrap()[0].present_index, Some(10));
    }

    #[test]
    fn upstream_subset_does_not_require_trd_metadata() {
        let original = fhc_document();
        let schema = Arc::new(Schema::new(original.schema().fields().clone()));
        let batch = RecordBatch::try_new(
            Arc::clone(&schema),
            original.batches()[0].columns().to_vec(),
        )
        .unwrap();
        let source = SceneDocument::from_batches(Arc::clone(&schema), vec![batch], vec![]).unwrap();
        let bytes = source.write().unwrap();
        assert!(SceneDocument::starts_with_params(&bytes).unwrap());
        let reopened = SceneDocument::read(&bytes).unwrap();
        assert_eq!(reopened.schema(), &schema);
        assert_eq!(reopened.frame(0).unwrap(), original.frame(0).unwrap());
    }

    #[test]
    fn unbound_fhc_edit_fails_atomically_instead_of_inventing_uuid() {
        let mut doc = fhc_document();
        let original = doc.write().unwrap();
        assert!(doc
            .apply_model_edits(&[ModelEdit {
                row: 0,
                object: 0,
                model: Matrix4::IDENTITY
            },])
            .is_err());
        assert_eq!(original, doc.write().unwrap());
    }

    #[test]
    fn fhc_export_adds_row_major_model_and_uuid_without_rewriting_geometry() {
        let mut document = fhc_document();
        let original = document.clone();
        let glb = super::super::glb_mesh::triangle_glb();
        let id = document.bind_glb(&glb).unwrap();
        let mut columns = Matrix4::IDENTITY.to_cols_array();
        columns[12] = 1.25;
        columns[14] = -0.75;
        document
            .apply_model_edits(&[ModelEdit {
                row: 0,
                object: 0,
                model: Matrix4::from_cols_array(&columns),
            }])
            .unwrap();
        for column in 0..4 {
            assert!(Arc::ptr_eq(
                document.batches()[0].column(column),
                original.batches()[0].column(column),
            ));
        }
        let reopened = SceneDocument::read(&document.write().unwrap()).unwrap();
        let frame = reopened.frame(0).unwrap();
        assert_eq!(frame.objects[0].model.to_cols_array(), columns);
        assert_eq!(frame.objects[0].mesh, Some(DocumentMesh::Id(id)));
        assert_eq!(reopened.meshes()[0].bytes(), glb);
        assert_eq!(reopened.schema().field(4).name(), "model");
        assert_eq!(reopened.schema().field(5).name(), "mesh_id");
    }
}
