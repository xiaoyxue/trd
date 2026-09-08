use std::collections::BTreeMap;
use std::sync::Arc;

use arrow::array::{
    ArrayRef, FixedSizeBinaryArray, FixedSizeListArray, Float32Array, ListArray, RecordBatch,
};
use arrow::buffer::OffsetBuffer;
use arrow::datatypes::{DataType, Field, FieldRef, Schema};
use arrow_schema::extension::Uuid as ArrowUuid;
use uuid::Uuid;

use super::document_params::{is_tracked, validate_model};
use super::{parse_error, DocumentFrame, DocumentMesh, ProtocolError, SceneDocument};
use crate::Matrix4;

/// An edit is addressed by source row and instance, not by a shared mesh's GPU ID.
#[derive(Debug, Clone, Copy)]
pub struct ModelEdit {
    pub row: usize,
    pub object: usize,
    pub model: Matrix4,
}

impl SceneDocument {
    /// Captures original local models for one object across the existing rows.
    /// Track IDs survive reordered/missing instances. Without IDs, only a
    /// consistent ordered binding list can establish cross-frame identity.
    pub fn model_track(&self, row: usize, object: usize) -> Result<Vec<ModelEdit>, ProtocolError> {
        let frames = self.frames()?;
        let reference = frames
            .get(row)
            .ok_or_else(|| parse_error(format!("params row {row} is out of range")))?;
        if object >= reference.objects.len() {
            return Err(parse_error("model track instance is out of range"));
        }
        frames
            .iter()
            .enumerate()
            .filter_map(
                |(row, frame)| match Self::track_instance(reference, object, frame) {
                    Ok(Some(object)) => Some(Ok(ModelEdit {
                        row,
                        object,
                        model: frame.objects[object].model,
                    })),
                    Ok(None) => None,
                    Err(error) => Some(Err(error)),
                },
            )
            .collect()
    }

    /// Updates only matrix columns and missing source bindings. Failures are atomic.
    pub fn apply_model_edits(&mut self, edits: &[ModelEdit]) -> Result<(), ProtocolError> {
        if edits.is_empty() {
            return Ok(());
        }
        let frames = self.frames()?;
        let mut selected = BTreeMap::new();
        for edit in edits {
            validate_model(edit.model)?;
            if frames
                .get(edit.row)
                .and_then(|frame| frame.objects.get(edit.object))
                .is_none()
            {
                return Err(parse_error(format!(
                    "model edit ({}, {}) is outside the source rows/objects",
                    edit.row, edit.object
                )));
            }
            if selected
                .insert((edit.row, edit.object), edit.model)
                .is_some()
            {
                return Err(parse_error("the same instance appears twice in one edit"));
            }
        }
        let tracked = self.batches().first().is_some_and(is_tracked);
        let column_name = if tracked { "model" } else { "draw_model" };
        let existing = self.schema().index_of(column_name).ok();
        let binding_missing = tracked && self.schema().index_of("mesh_id").is_err();
        let matrix_field = existing.map_or_else(
            || Arc::new(Field::new(column_name, matrix_type(tracked), false)),
            |index| Arc::clone(&self.schema().fields()[index]),
        );
        let binding_field = Arc::new(Field::new(
            "mesh_id",
            DataType::List(Arc::new(
                Field::new("item", DataType::FixedSizeBinary(16), false)
                    .with_extension_type(ArrowUuid),
            )),
            false,
        ));
        let mut fields = self.schema().fields().to_vec();
        let model_index = existing.unwrap_or(fields.len());
        if existing.is_none() {
            fields.push(Arc::clone(&matrix_field));
        }
        if binding_missing {
            fields.push(Arc::clone(&binding_field));
        }
        let schema = Arc::new(Schema::new_with_metadata(
            fields,
            self.schema().metadata().clone(),
        ));
        let mut batches = Vec::with_capacity(self.batches().len());
        let mut first_row = 0;
        for batch in self.batches() {
            let end = first_row + batch.num_rows();
            let changed = selected.range((first_row, 0)..(end, 0)).next().is_some();
            let rows = &frames[first_row..end];
            let mut columns = batch.columns().to_vec();
            if changed || existing.is_none() {
                let matrices = rows
                    .iter()
                    .enumerate()
                    .map(|(row, frame)| {
                        frame
                            .objects
                            .iter()
                            .enumerate()
                            .map(|(object, value)| {
                                selected
                                    .get(&(first_row + row, object))
                                    .copied()
                                    .unwrap_or(value.model)
                            })
                            .collect::<Vec<_>>()
                    })
                    .collect::<Vec<_>>();
                let values = build_models(&matrices, matrix_field.data_type(), tracked)?;
                if existing.is_some() {
                    columns[model_index] = values;
                } else {
                    columns.push(values);
                }
            }
            if binding_missing {
                let ids = rows
                    .iter()
                    .map(|frame| {
                        (0..frame.objects.len())
                            .map(|object| self.binding_for(frame, object))
                            .collect::<Result<Vec<_>, _>>()
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                columns.push(build_bindings(&ids, &binding_field)?);
            }
            batches.push(RecordBatch::try_new(Arc::clone(&schema), columns)?);
            first_row = end;
        }
        self.replace_batches(schema, batches)
    }

    fn track_instance(
        reference: &DocumentFrame,
        object: usize,
        frame: &DocumentFrame,
    ) -> Result<Option<usize>, ProtocolError> {
        let selected = &reference.objects[object];
        if let Some(id) = selected.track_id.as_deref() {
            if id.is_empty() {
                return Err(parse_error(
                    "an empty track_id cannot identify an edited object",
                ));
            }
            let mut matches = frame
                .objects
                .iter()
                .enumerate()
                .filter(|(_, instance)| instance.track_id.as_deref() == Some(id));
            let found = matches.next().map(|(index, _)| index);
            if matches.next().is_some() {
                return Err(parse_error(format!("ambiguous duplicate track_id {id}")));
            }
            return Ok(found);
        }
        if frame.objects.is_empty() {
            return Ok(None);
        }
        if frame.objects.len() != reference.objects.len()
            || frame
                .objects
                .iter()
                .any(|instance| instance.track_id.is_some())
            || !frame
                .objects
                .iter()
                .map(|instance| instance.mesh)
                .eq(reference.objects.iter().map(|instance| instance.mesh))
        {
            return Err(parse_error(
                "cross-frame editing needs track_id when object counts or ordered mesh bindings change",
            ));
        }
        Ok(Some(object))
    }

    pub fn object_mesh_slot(
        &self,
        frame: &DocumentFrame,
        object: usize,
    ) -> Result<usize, ProtocolError> {
        self.mesh_slot(DocumentMesh::Id(self.binding_for(frame, object)?))
    }

    fn binding_for(&self, frame: &DocumentFrame, object: usize) -> Result<Uuid, ProtocolError> {
        let instance = frame
            .objects
            .get(object)
            .ok_or_else(|| parse_error("instance index is out of range"))?;
        if let Some(binding) = instance.mesh {
            return self.mesh_slot(binding).map(|slot| self.meshes()[slot].id());
        }
        let slot = match self.meshes().len() {
            1 => 0,
            count if count == frame.objects.len() => object,
            _ => return Err(parse_error("missing per-instance mesh binding")),
        };
        self.meshes()
            .get(slot)
            .map(|mesh| mesh.id())
            .ok_or_else(|| parse_error("no GLB asset is bound"))
    }
}

fn matrix_type(tracked: bool) -> DataType {
    let floats = Arc::new(Field::new("item", DataType::Float32, false));
    let matrix = if tracked {
        DataType::FixedSizeList(
            Arc::new(Field::new(
                "item",
                DataType::FixedSizeList(floats, 4),
                false,
            )),
            4,
        )
    } else {
        DataType::FixedSizeList(floats, 16)
    };
    DataType::List(Arc::new(Field::new("item", matrix, false)))
}

fn list_item(data_type: &DataType) -> Result<FieldRef, ProtocolError> {
    match data_type {
        DataType::List(item) => Ok(Arc::clone(item)),
        _ => Err(parse_error("model column must be a list")),
    }
}

fn fixed_item(data_type: &DataType, expected: i32) -> Result<FieldRef, ProtocolError> {
    match data_type {
        DataType::FixedSizeList(item, len) if *len == expected => Ok(Arc::clone(item)),
        _ => Err(parse_error(
            "model column has incompatible matrix dimensions",
        )),
    }
}

fn build_models(
    rows: &[Vec<Matrix4>],
    data_type: &DataType,
    tracked: bool,
) -> Result<ArrayRef, ProtocolError> {
    let matrix_field = list_item(data_type)?;
    let mut values = Vec::new();
    for model in rows.iter().flatten() {
        let columns = model.to_cols_array();
        if tracked {
            values.extend((0..16).map(|i| columns[(i % 4) * 4 + i / 4]));
        } else {
            values.extend_from_slice(&columns);
        }
    }
    let floats: ArrayRef = Arc::new(Float32Array::from(values));
    let matrices: ArrayRef = if tracked {
        let row_field = fixed_item(matrix_field.data_type(), 4)?;
        let float_field = fixed_item(row_field.data_type(), 4)?;
        let four_floats = Arc::new(FixedSizeListArray::try_new(float_field, 4, floats, None)?);
        Arc::new(FixedSizeListArray::try_new(
            row_field,
            4,
            four_floats,
            None,
        )?)
    } else {
        let float_field = fixed_item(matrix_field.data_type(), 16)?;
        Arc::new(FixedSizeListArray::try_new(float_field, 16, floats, None)?)
    };
    Ok(Arc::new(ListArray::try_new(
        matrix_field,
        OffsetBuffer::from_lengths(rows.iter().map(Vec::len)),
        matrices,
        None,
    )?))
}

fn build_bindings(rows: &[Vec<Uuid>], field: &Field) -> Result<ArrayRef, ProtocolError> {
    let ids = if rows.iter().all(Vec::is_empty) {
        FixedSizeBinaryArray::new(16, arrow::buffer::Buffer::from(Vec::<u8>::new()), None)
    } else {
        FixedSizeBinaryArray::try_from_iter(rows.iter().flatten().map(Uuid::as_bytes))?
    };
    Ok(Arc::new(ListArray::try_new(
        list_item(field.data_type())?,
        OffsetBuffer::from_lengths(rows.iter().map(Vec::len)),
        Arc::new(ids),
        None,
    )?))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{FrameParams, PROTOCOL_VERSION, PROTOCOL_VERSION_KEY, TABLE_KIND_KEY};
    use arrow::array::{Int64Array, StringArray};

    fn track_frame(ids: &[Option<&str>]) -> DocumentFrame {
        DocumentFrame {
            params: FrameParams::IDENTITY,
            source_size: None,
            present_index: None,
            pts: None,
            objects: ids
                .iter()
                .map(|id| super::super::DocumentObject {
                    mesh: Some(DocumentMesh::Index(0)),
                    model: Matrix4::IDENTITY,
                    quad: None,
                    track_id: id.map(str::to_owned),
                    selection: crate::DrawSelection::INHERIT,
                })
                .collect(),
        }
    }

    #[test]
    fn track_identity_follows_reordered_instances_and_skips_absent_rows() {
        let original = track_frame(&[Some("dragon"), Some("can")]);
        let reordered = track_frame(&[Some("can"), Some("dragon")]);
        assert_eq!(
            SceneDocument::track_instance(&original, 0, &reordered).unwrap(),
            Some(1)
        );
        assert_eq!(
            SceneDocument::track_instance(&original, 0, &track_frame(&[Some("can")])).unwrap(),
            None
        );
        assert_eq!(
            SceneDocument::track_instance(&original, 0, &track_frame(&[])).unwrap(),
            None
        );
        assert!(SceneDocument::track_instance(
            &original,
            0,
            &track_frame(&[Some("dragon"), Some("dragon")])
        )
        .is_err());
    }

    #[test]
    fn unkeyed_tracks_require_consistent_slots_and_never_edit_by_shared_mesh_alone() {
        let original = track_frame(&[None, None]);
        assert_eq!(
            SceneDocument::track_instance(&original, 1, &original).unwrap(),
            Some(1)
        );
        assert_eq!(
            SceneDocument::track_instance(&original, 1, &track_frame(&[])).unwrap(),
            None
        );
        assert!(SceneDocument::track_instance(&original, 1, &track_frame(&[None])).is_err());
        let mut reordered = original.clone();
        reordered.objects[0].mesh = Some(DocumentMesh::Index(1));
        assert!(SceneDocument::track_instance(&original, 1, &reordered).is_err());
    }

    fn cg_document() -> SceneDocument {
        let matrix = build_models(&[vec![Matrix4::IDENTITY]], &matrix_type(false), false).unwrap();
        let schema = Arc::new(
            Schema::new(vec![
                Field::new("present_index", DataType::Int64, false),
                Field::new("label", DataType::Utf8, true).with_metadata(
                    [("owner".to_owned(), "upstream".to_owned())]
                        .into_iter()
                        .collect(),
                ),
                Field::new("draw_model", matrix.data_type().clone(), false),
            ])
            .with_metadata(
                [
                    (PROTOCOL_VERSION_KEY.to_owned(), PROTOCOL_VERSION.to_owned()),
                    (TABLE_KIND_KEY.to_owned(), "params".to_owned()),
                    ("source".to_owned(), "kept".to_owned()),
                ]
                .into_iter()
                .collect(),
            ),
        );
        let batches = [10, 30]
            .into_iter()
            .map(|index| {
                RecordBatch::try_new(
                    Arc::clone(&schema),
                    vec![
                        Arc::new(Int64Array::from(vec![index])),
                        Arc::new(StringArray::from(vec![Some("not a rendering field")])),
                        Arc::clone(&matrix),
                    ],
                )
                .unwrap()
            })
            .collect();
        SceneDocument::from_batches(schema, batches, vec![]).unwrap()
    }

    #[test]
    fn editing_one_matrix_preserves_other_columns_and_batches() {
        let mut document = cg_document();
        let original = document.clone();
        let mut values = Matrix4::IDENTITY.to_cols_array();
        values[12] = 3.5;
        let edited = Matrix4::from_cols_array(&values);
        document
            .apply_model_edits(&[ModelEdit {
                row: 1,
                object: 0,
                model: edited,
            }])
            .unwrap();
        assert_eq!(document.schema(), original.schema());
        assert_eq!(document.batches().len(), 2);
        for (new, old) in document.batches().iter().zip(original.batches()) {
            assert!(Arc::ptr_eq(new.column(0), old.column(0)));
            assert!(Arc::ptr_eq(new.column(1), old.column(1)));
        }
        assert!(Arc::ptr_eq(
            document.batches()[0].column(2),
            original.batches()[0].column(2)
        ));
        let reopened = SceneDocument::read(&document.write().unwrap()).unwrap();
        let frames = reopened.frames().unwrap();
        assert_eq!(frames[0].objects[0].model, Matrix4::IDENTITY);
        assert_eq!(frames[1].objects[0].model, edited);
        assert_eq!(frames[0].params, FrameParams::IDENTITY);
    }

    #[test]
    fn invalid_edit_does_not_change_source_data() {
        let mut document = cg_document();
        let bytes = document.write().unwrap();
        assert!(document
            .apply_model_edits(&[
                ModelEdit {
                    row: 1,
                    object: 0,
                    model: Matrix4::IDENTITY
                },
                ModelEdit {
                    row: 4,
                    object: 0,
                    model: Matrix4::IDENTITY
                },
            ])
            .is_err());
        assert_eq!(document.write().unwrap(), bytes);
    }

    #[test]
    fn nested_row_major_models_keep_source_field_metadata() {
        let model = Matrix4::from_cols_array(&[
            1.0, 0.0, 0.0, 0.0, 0.25, 2.0, 0.0, 0.0, 0.0, 0.0, 3.0, 0.0, 4.0, 5.0, 6.0, 1.0,
        ]);
        let array = build_models(&[vec![model]], &matrix_type(true), true).unwrap();
        let list = array.as_any().downcast_ref::<ListArray>().unwrap();
        assert_eq!(
            super::super::document_params::read_source_model(list.value(0).as_ref(), 0).unwrap(),
            model
        );
    }

    #[test]
    fn unknown_dictionary_nested_nulls_and_extension_metadata_survive_editing() {
        use arrow::array::{DictionaryArray, FixedSizeBinaryArray};
        use arrow::datatypes::{Int32Type, Int8Type};
        let source = cg_document();
        let dictionary: ArrayRef = Arc::new(
            ["external-label"]
                .into_iter()
                .collect::<DictionaryArray<Int8Type>>(),
        );
        let nested: ArrayRef = Arc::new(ListArray::from_iter_primitive::<Int32Type, _, _>([Some(
            vec![Some(1), None, Some(3)],
        )]));
        let uuid: ArrayRef =
            Arc::new(FixedSizeBinaryArray::try_from_iter([[7u8; 16]].into_iter()).unwrap());
        let mut fields = source.schema().fields().to_vec();
        fields.push(Arc::new(Field::new(
            "classification",
            dictionary.data_type().clone(),
            false,
        )));
        fields.push(Arc::new(Field::new(
            "arbitrary_nested",
            nested.data_type().clone(),
            true,
        )));
        fields.push(Arc::new(
            Field::new("unrelated_uuid", DataType::FixedSizeBinary(16), false)
                .with_extension_type(ArrowUuid),
        ));
        let schema = Arc::new(Schema::new_with_metadata(
            fields,
            source.schema().metadata().clone(),
        ));
        let batches = source
            .batches()
            .iter()
            .map(|batch| {
                let mut arrays = batch.columns().to_vec();
                arrays.extend([dictionary.clone(), nested.clone(), uuid.clone()]);
                RecordBatch::try_new(schema.clone(), arrays).unwrap()
            })
            .collect();
        let mut source = SceneDocument::from_batches(schema.clone(), batches, vec![]).unwrap();
        let original = source.clone();
        source
            .apply_model_edits(&[ModelEdit {
                row: 1,
                object: 0,
                model: Matrix4::IDENTITY,
            }])
            .unwrap();
        let reopened = SceneDocument::read(&source.write().unwrap()).unwrap();
        assert_eq!(reopened.schema(), &schema);
        for (new, old) in reopened.batches().iter().zip(original.batches()) {
            for column in [0, 1, 3, 4, 5] {
                assert_eq!(new.column(column).to_data(), old.column(column).to_data());
            }
        }
    }
}
