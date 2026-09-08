//! Retained source tables for read-modify-write editing, separate from fresh scene encoding.

use std::io::Cursor;
use std::sync::Arc;

use arrow::array::RecordBatch;
use arrow::datatypes::SchemaRef;
use arrow::ipc::reader::StreamReader;
use arrow::ipc::writer::StreamWriter;

use super::glb_mesh::{
    decode_mesh_batch, encode_mesh_batch, validate_mesh_ids, validate_mesh_schema,
};
use super::{check_version, parse_error, GlbMesh, ProtocolError, TABLE_KIND_KEY};

/// The original params columns and immutable GLB payloads in `[params][mesh?]`.
#[derive(Debug, Clone)]
pub struct SceneDocument {
    schema: SchemaRef,
    batches: Vec<RecordBatch>,
    meshes: Vec<GlbMesh>,
    mesh_layout: Option<MeshTableLayout>,
}

/// The immutable GLB bytes already live in `meshes`; retain layout without a
/// second copy of potentially large resource arrays.
#[derive(Debug, Clone)]
struct MeshTableLayout {
    schema: SchemaRef,
    batch_lengths: Vec<usize>,
}

impl SceneDocument {
    pub fn read_from(mut reader: impl std::io::Read) -> Result<Self, ProtocolError> {
        let mut marker = [0u8; 4];
        reader
            .read_exact(&mut marker)
            .map_err(|error| parse_error(error.to_string()))?;
        if marker != [0xff; 4] {
            return Err(parse_error(
                "input must be Arrow IPC, not a video or Parquet file",
            ));
        }
        let mut bytes = marker.to_vec();
        reader
            .read_to_end(&mut bytes)
            .map_err(|error| parse_error(error.to_string()))?;
        Self::read(&bytes)
    }

    pub fn starts_with_params(bytes: &[u8]) -> Result<bool, ProtocolError> {
        let reader = StreamReader::try_new(Cursor::new(bytes), None)?;
        let schema = reader.schema();
        Ok(
            schema.metadata().get(TABLE_KIND_KEY).map(String::as_str) == Some("params")
                || is_source_subset(&schema),
        )
    }

    pub fn read(bytes: &[u8]) -> Result<Self, ProtocolError> {
        if bytes.starts_with(b"PAR1") {
            return Err(parse_error(
                "Parquet input is not supported; provide Arrow IPC",
            ));
        }
        let mut source = Cursor::new(bytes);
        let (schema, batches) = read_table(&mut source, "params")?;
        let mut meshes = Vec::new();
        let mesh_layout = if source.position() < bytes.len() as u64 {
            let (schema, mesh_batches) = read_table(&mut source, "mesh")?;
            validate_mesh_schema(&schema)?;
            for batch in &mesh_batches {
                meshes.extend(decode_mesh_batch(batch)?);
            }
            Some(MeshTableLayout {
                schema,
                batch_lengths: mesh_batches.iter().map(RecordBatch::num_rows).collect(),
            })
        } else {
            None
        };
        if source.position() != bytes.len() as u64 {
            return Err(parse_error(
                "only [params] followed by one optional [mesh] stream is allowed",
            ));
        }
        let mut document = Self::from_batches(schema, batches, meshes)?;
        document.mesh_layout = mesh_layout;
        Ok(document)
    }

    pub fn from_batches(
        schema: SchemaRef,
        batches: Vec<RecordBatch>,
        meshes: Vec<GlbMesh>,
    ) -> Result<Self, ProtocolError> {
        validate_table(&schema, "params")?;
        if batches.iter().any(|batch| batch.schema() != schema) {
            return Err(parse_error(
                "params batches must have the same complete schema",
            ));
        }
        let mut names = std::collections::HashSet::new();
        if schema
            .fields()
            .iter()
            .any(|field| !names.insert(field.name()))
        {
            return Err(parse_error("params column names must be unique"));
        }
        if schema.field_with_name("frame_id").is_ok() {
            return Err(parse_error(
                "frame_id and inline frames tables are not supported",
            ));
        }
        validate_mesh_ids(&meshes)?;
        let document = Self {
            schema,
            batches,
            meshes,
            mesh_layout: None,
        };
        document.validate_render_view()?;
        Ok(document)
    }

    pub fn schema(&self) -> &SchemaRef {
        &self.schema
    }

    pub fn batches(&self) -> &[RecordBatch] {
        &self.batches
    }

    pub fn meshes(&self) -> &[GlbMesh] {
        &self.meshes
    }

    pub fn decoded_assets(&self) -> Result<Vec<crate::MeshAsset>, ProtocolError> {
        if self.meshes.is_empty() {
            return Ok(vec![crate::MeshAsset::embedded(
                crate::Mesh::reference_cube()?,
                crate::DisneyMaterial::default(),
            )]);
        }
        self.meshes
            .iter()
            .enumerate()
            .map(|(slot, mesh)| {
                mesh.decode(
                    u32::try_from(slot).map_err(|_| parse_error("too many mesh resources"))?,
                )
            })
            .collect()
    }

    pub fn row_count(&self) -> usize {
        self.batches.iter().map(RecordBatch::num_rows).sum()
    }

    pub fn frame_ref(&self, row: usize) -> Result<Option<String>, ProtocolError> {
        let mut local = row;
        for batch in &self.batches {
            if local < batch.num_rows() {
                if super::document_params::is_tracked(batch) {
                    return Ok(None);
                }
                let refs = super::decode_frame_refs(&batch.slice(local, 1))?;
                return Ok(refs.and_then(|mut values| values.pop()).flatten());
            }
            local -= batch.num_rows();
        }
        Err(parse_error(format!("params row {row} is out of range")))
    }

    pub fn tonemap_override(&self) -> Result<Option<crate::Tonemap>, ProtocolError> {
        if self
            .batches
            .first()
            .is_some_and(super::document_params::is_tracked)
        {
            return Ok(None);
        }
        let mut observed = None;
        for batch in &self.batches {
            if let Some(operator) = super::decode_tonemap(batch)? {
                if observed.is_some_and(|previous| previous != operator) {
                    return Err(parse_error(
                        "tonemap must be constant across params batches",
                    ));
                }
                observed = Some(operator);
            }
        }
        Ok(observed)
    }

    /// Resources may arrive through a host API instead of the optional mesh stream.
    pub fn bind_meshes(&mut self, meshes: Vec<GlbMesh>) -> Result<(), ProtocolError> {
        validate_mesh_ids(&meshes)?;
        if !self.meshes.is_empty() {
            return Err(parse_error("document already has mesh resources"));
        }
        let mut candidate = self.clone();
        candidate.meshes = meshes;
        candidate.validate_render_view()?;
        self.meshes = candidate.meshes;
        if !self.meshes.is_empty() {
            if let Some(layout) = &mut self.mesh_layout {
                layout.batch_lengths.push(self.meshes.len());
            }
        }
        Ok(())
    }

    /// Uses the source's unique UUID, or creates a binding when the source has none.
    pub fn bind_glb(&mut self, bytes: &[u8]) -> Result<uuid::Uuid, ProtocolError> {
        let mut ids = std::collections::BTreeSet::new();
        for frame in self.frames()? {
            for object in frame.objects {
                if let Some(super::DocumentMesh::Id(id)) = object.mesh {
                    ids.insert(id);
                }
            }
        }
        if ids.len() > 1 {
            return Err(parse_error(
                "source names multiple mesh UUIDs; supply an ID-keyed GLB map",
            ));
        }
        let id = ids.into_iter().next().unwrap_or_else(uuid::Uuid::new_v4);
        let mesh = GlbMesh::new(&id.to_string(), bytes)?;
        self.bind_meshes(vec![mesh])?;
        Ok(id)
    }

    pub fn write(&self) -> Result<Vec<u8>, ProtocolError> {
        let mut bytes = Vec::new();
        {
            let mut writer = StreamWriter::try_new(&mut bytes, &self.schema)?;
            for batch in &self.batches {
                writer.write(batch)?;
            }
            writer.finish()?;
        }
        if let Some(layout) = &self.mesh_layout {
            let mut writer = StreamWriter::try_new(&mut bytes, &layout.schema)?;
            let mut start = 0;
            for &length in &layout.batch_lengths {
                let batch = if length == 0 {
                    RecordBatch::new_empty(Arc::clone(&layout.schema))
                } else {
                    let encoded = encode_mesh_batch(&self.meshes[start..start + length])?;
                    RecordBatch::try_new(Arc::clone(&layout.schema), encoded.columns().to_vec())?
                };
                writer.write(&batch)?;
                start += length;
            }
            writer.finish()?;
        } else if !self.meshes.is_empty() {
            let batch = encode_mesh_batch(&self.meshes)?;
            let mut writer = StreamWriter::try_new(&mut bytes, &batch.schema())?;
            writer.write(&batch)?;
            writer.finish()?;
        }
        Ok(bytes)
    }

    pub(super) fn replace_batches(
        &mut self,
        schema: SchemaRef,
        batches: Vec<RecordBatch>,
    ) -> Result<(), ProtocolError> {
        if batches.len() != self.batches.len()
            || batches
                .iter()
                .zip(&self.batches)
                .any(|(new, old)| new.num_rows() != old.num_rows() || new.schema() != schema)
        {
            return Err(parse_error(
                "an edit must preserve row and batch boundaries",
            ));
        }
        self.schema = schema;
        self.batches = batches;
        Ok(())
    }

    fn validate_render_view(&self) -> Result<(), ProtocolError> {
        self.tonemap_override()?;
        for frame in self.frames()? {
            if !self.meshes.is_empty() {
                for object in 0..frame.objects.len() {
                    self.object_mesh_slot(&frame, object)?;
                }
            }
        }
        Ok(())
    }
}

fn validate_table(schema: &arrow::datatypes::Schema, kind: &str) -> Result<(), ProtocolError> {
    if schema
        .metadata()
        .contains_key(crate::VIDEO_EDIT_VERSION_KEY)
    {
        return Err(parse_error(
            "legacy video-edit annotation input is retired; convert it offline with \
             scripts/timeline_to_params.py and load the params/GLB Arrow document",
        ));
    }
    if kind == "params" && is_source_subset(schema) {
        return Ok(());
    }
    check_version(schema)?;
    if schema.metadata().get(TABLE_KIND_KEY).map(String::as_str) != Some(kind) {
        return Err(parse_error(format!("expected a {kind} table")));
    }
    Ok(())
}

fn is_source_subset(schema: &arrow::datatypes::Schema) -> bool {
    // An upstream tracked table has its own field contract, not trd's wire metadata.
    !schema.metadata().contains_key(super::PROTOCOL_VERSION_KEY)
        && !schema.metadata().contains_key(TABLE_KIND_KEY)
        && schema.field_with_name("bottom_quads").is_ok()
        && schema
            .field_with_name("k")
            .is_ok_and(|field| matches!(field.data_type(), arrow::datatypes::DataType::Struct(_)))
}

fn read_table(
    source: &mut Cursor<&[u8]>,
    kind: &str,
) -> Result<(SchemaRef, Vec<RecordBatch>), ProtocolError> {
    let mut reader = StreamReader::try_new(source, None)?;
    let schema = reader.schema();
    validate_table(&schema, kind)?;
    let batches = reader.by_ref().collect::<Result<Vec<_>, _>>()?;
    Ok((Arc::clone(&schema), batches))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Draw, DrawSelection, FrameParams, Matrix4, ModelEdit};

    fn params() -> Vec<u8> {
        super::super::scene_encode::encode_params_stream(
            &[FrameParams::IDENTITY],
            Some(&[vec![Draw {
                mesh_id: 0,
                model: Matrix4::IDENTITY,
                selection: DrawSelection::INHERIT,
            }]]),
        )
        .unwrap()
    }

    #[test]
    fn edits_preserve_resource_schema_metadata_and_all_batch_boundaries() {
        let meshes = [1, 2].map(|id| {
            GlbMesh::new(
                &format!("00000000-0000-0000-0000-{id:012}"),
                &super::super::triangle_glb(),
            )
            .unwrap()
        });
        let canonical = encode_mesh_batch(&meshes).unwrap();
        let mut metadata = canonical.schema().metadata().clone();
        metadata.insert("upstream.mesh".to_owned(), "retained".to_owned());
        let mut fields = canonical.schema().fields().to_vec();
        fields[1] = Arc::new(
            fields[1].as_ref().clone().with_metadata(
                [("upstream.glb".to_owned(), "original".to_owned())]
                    .into_iter()
                    .collect(),
            ),
        );
        let schema = Arc::new(arrow::datatypes::Schema::new_with_metadata(
            fields, metadata,
        ));
        let source = RecordBatch::try_new(schema.clone(), canonical.columns().to_vec()).unwrap();
        let batches = [source.slice(0, 1), source.slice(1, 0), source.slice(1, 1)];
        let mut bytes = params();
        {
            let mut writer = StreamWriter::try_new(&mut bytes, &schema).unwrap();
            for batch in &batches {
                writer.write(batch).unwrap();
            }
            writer.finish().unwrap();
        }
        let mut document = SceneDocument::read(&bytes).unwrap();
        document
            .apply_model_edits(&[ModelEdit {
                row: 0,
                object: 0,
                model: Matrix4::from_translation(crate::Vector3::new(0.25, 0.0, 0.0)),
            }])
            .unwrap();
        let exported = document.write().unwrap();
        let mut cursor = Cursor::new(exported.as_slice());
        read_table(&mut cursor, "params").unwrap();
        let (actual_schema, actual_batches) = read_table(&mut cursor, "mesh").unwrap();
        assert_eq!(actual_schema, schema);
        assert_eq!(actual_batches, batches);
        assert_eq!(SceneDocument::read(&exported).unwrap().meshes(), meshes);
    }

    #[test]
    fn empty_resource_stream_keeps_its_schema_and_presence() {
        let original = super::super::glb_mesh::mesh_schema();
        let mut metadata = original.metadata().clone();
        metadata.insert("source.empty".to_owned(), "retained".to_owned());
        let schema = Arc::new(original.with_metadata(metadata));
        let mut bytes = params();
        StreamWriter::try_new(&mut bytes, &schema)
            .unwrap()
            .finish()
            .unwrap();
        let mut document = SceneDocument::read(&bytes).unwrap();
        let exported = document.write().unwrap();
        let mut cursor = Cursor::new(exported.as_slice());
        read_table(&mut cursor, "params").unwrap();
        let (actual, batches) = read_table(&mut cursor, "mesh").unwrap();
        assert_eq!(actual, schema);
        assert!(batches.is_empty());
        assert_eq!(cursor.position(), exported.len() as u64);

        document.bind_glb(&super::super::triangle_glb()).unwrap();
        let bound = document.write().unwrap();
        let mut cursor = Cursor::new(bound.as_slice());
        read_table(&mut cursor, "params").unwrap();
        let (actual, batches) = read_table(&mut cursor, "mesh").unwrap();
        assert_eq!(actual, schema);
        assert_eq!(
            batches
                .iter()
                .map(RecordBatch::num_rows)
                .collect::<Vec<_>>(),
            [1]
        );
    }

    #[test]
    fn invalid_empty_mesh_schema_is_rejected() {
        let schema = arrow::datatypes::Schema::empty()
            .with_metadata(super::super::glb_mesh::mesh_schema().metadata().clone());
        let mut bytes = params();
        StreamWriter::try_new(&mut bytes, &schema)
            .unwrap()
            .finish()
            .unwrap();
        assert!(SceneDocument::read(&bytes).is_err());
    }

    #[test]
    fn previous_protocol_version_is_rejected_for_params_and_mesh_schemas() {
        let bytes = params();
        let original = SceneDocument::read(&bytes).unwrap();
        let mut metadata = original.schema().metadata().clone();
        metadata.insert(
            super::super::PROTOCOL_VERSION_KEY.to_owned(),
            "0.0.6".to_owned(),
        );
        let schema = Arc::new(original.schema().as_ref().clone().with_metadata(metadata));
        let batch =
            RecordBatch::try_new(schema.clone(), original.batches()[0].columns().to_vec()).unwrap();
        let mut old_params = Vec::new();
        {
            let mut writer = StreamWriter::try_new(&mut old_params, &schema).unwrap();
            writer.write(&batch).unwrap();
            writer.finish().unwrap();
        }
        assert!(
            matches!(SceneDocument::read(&old_params), Err(ProtocolError::UnsupportedVersion(v)) if v=="0.0.6")
        );
        let mut metadata = super::super::glb_mesh::mesh_schema().metadata().clone();
        metadata.insert(
            super::super::PROTOCOL_VERSION_KEY.to_owned(),
            "0.0.6".to_owned(),
        );
        let schema = super::super::glb_mesh::mesh_schema().with_metadata(metadata);
        let mut old_mesh = bytes;
        StreamWriter::try_new(&mut old_mesh, &schema)
            .unwrap()
            .finish()
            .unwrap();
        assert!(
            matches!(SceneDocument::read(&old_mesh), Err(ProtocolError::UnsupportedVersion(v)) if v=="0.0.6")
        );
    }
}
