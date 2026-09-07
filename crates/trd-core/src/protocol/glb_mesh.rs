use std::collections::HashSet;
use std::sync::Arc;

use arrow::array::{Array, FixedSizeBinaryArray, LargeBinaryArray, RecordBatch};
use arrow::datatypes::{DataType, Field, Schema};
use arrow_schema::extension::Uuid as ArrowUuid;
use uuid::Uuid;

use super::{parse_error, ProtocolError, PROTOCOL_VERSION, PROTOCOL_VERSION_KEY, TABLE_KIND_KEY};
use crate::MeshAsset;

/// An immutable source asset. Its UUID is independent of a renderer's slot.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GlbMesh {
    id: Uuid,
    bytes: Arc<[u8]>,
}

impl GlbMesh {
    pub fn new(id: &str, bytes: &[u8]) -> Result<Self, ProtocolError> {
        let id = Uuid::parse_str(id).map_err(|error| parse_error(format!("mesh_id: {error}")))?;
        Self::from_uuid(id, Arc::from(bytes))
    }

    pub fn id(&self) -> Uuid {
        self.id
    }

    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }

    pub fn decode(&self, slot: u32) -> Result<MeshAsset, ProtocolError> {
        crate::import_glb(&self.bytes)
            .map(|asset| MeshAsset::from_gltf_with_id(slot, asset))
            .map_err(|source| ProtocolError::GltfImport {
                index: slot,
                source,
            })
    }

    fn from_uuid(id: Uuid, bytes: Arc<[u8]>) -> Result<Self, ProtocolError> {
        if bytes.len() < 12 || &bytes[..4] != b"glTF" {
            return Err(parse_error(format!("mesh {id} is not a binary GLB")));
        }
        let version = u32::from_le_bytes(bytes[4..8].try_into().expect("four header bytes"));
        let length = u32::from_le_bytes(bytes[8..12].try_into().expect("four header bytes"));
        if version != 2 || usize::try_from(length).ok() != Some(bytes.len()) {
            return Err(parse_error(format!("mesh {id} has an invalid GLB header")));
        }
        let parsed = gltf_rs::Gltf::from_slice(&bytes)
            .map_err(|error| parse_error(format!("mesh {id}: {error}")))?;
        if parsed.blob.is_none()
            || parsed
                .document
                .buffers()
                .any(|buffer| matches!(buffer.source(), gltf_rs::buffer::Source::Uri(_)))
            || parsed
                .document
                .images()
                .any(|image| matches!(image.source(), gltf_rs::image::Source::Uri { .. }))
        {
            return Err(parse_error(format!(
                "mesh {id} must be a self-contained GLB"
            )));
        }
        Ok(Self { id, bytes })
    }
}

pub(super) fn mesh_schema() -> Schema {
    Schema::new(vec![
        Field::new("mesh_id", DataType::FixedSizeBinary(16), false).with_extension_type(ArrowUuid),
        Field::new("glb", DataType::LargeBinary, false),
    ])
    .with_metadata(
        [
            (PROTOCOL_VERSION_KEY.to_owned(), PROTOCOL_VERSION.to_owned()),
            (TABLE_KIND_KEY.to_owned(), "mesh".to_owned()),
        ]
        .into_iter()
        .collect(),
    )
}

pub(super) fn decode_mesh_batch(batch: &RecordBatch) -> Result<Vec<GlbMesh>, ProtocolError> {
    let schema = batch.schema();
    if schema.fields().len() != 2
        || schema.field(0).name() != "mesh_id"
        || schema.field(1).name() != "glb"
        || schema.fields().iter().any(|field| field.is_nullable())
    {
        return Err(parse_error(
            "mesh table must contain only non-null mesh_id and glb",
        ));
    }
    schema.field(0).try_extension_type::<ArrowUuid>()?;
    let ids = batch
        .column(0)
        .as_any()
        .downcast_ref::<FixedSizeBinaryArray>()
        .filter(|ids| ids.value_length() == 16)
        .ok_or_else(|| parse_error("mesh_id must have Arrow UUID storage"))?;
    let glbs = batch
        .column(1)
        .as_any()
        .downcast_ref::<LargeBinaryArray>()
        .ok_or_else(|| parse_error("glb must be LargeBinary"))?;
    if ids.null_count() != 0 || glbs.null_count() != 0 {
        return Err(parse_error("mesh_id and glb must not contain nulls"));
    }
    (0..batch.num_rows())
        .map(|row| {
            let id = Uuid::from_slice(ids.value(row))
                .map_err(|error| parse_error(format!("mesh_id: {error}")))?;
            GlbMesh::from_uuid(id, Arc::from(glbs.value(row)))
        })
        .collect()
}

pub(super) fn validate_mesh_ids(meshes: &[GlbMesh]) -> Result<(), ProtocolError> {
    let mut ids = HashSet::new();
    for mesh in meshes {
        if !ids.insert(mesh.id) {
            return Err(parse_error(format!("duplicate mesh_id {}", mesh.id)));
        }
    }
    Ok(())
}

pub(super) fn encode_mesh_batch(meshes: &[GlbMesh]) -> Result<RecordBatch, ProtocolError> {
    validate_mesh_ids(meshes)?;
    let ids = FixedSizeBinaryArray::try_from_iter(meshes.iter().map(|mesh| mesh.id.as_bytes()))?;
    let glbs = LargeBinaryArray::from_iter_values(meshes.iter().map(GlbMesh::bytes));
    Ok(RecordBatch::try_new(
        Arc::new(mesh_schema()),
        vec![Arc::new(ids), Arc::new(glbs)],
    )?)
}

#[cfg(test)]
pub(super) fn triangle_glb() -> Vec<u8> {
    let json = serde_json::json!({
        "asset": {"version": "2.0"},
        "buffers": [{"byteLength": 36}],
        "bufferViews": [{"buffer": 0, "byteOffset": 0, "byteLength": 36}],
        "accessors": [{
            "bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3",
            "min": [-0.5, -0.5, 0.0], "max": [0.5, 0.5, 0.0]
        }],
        "meshes": [{"primitives": [{"attributes": {"POSITION": 0}}]}],
        "nodes": [{"mesh": 0}], "scenes": [{"nodes": [0]}], "scene": 0
    });
    let mut json = serde_json::to_vec(&json).unwrap();
    while !json.len().is_multiple_of(4) {
        json.push(b' ');
    }
    let mut bytes = b"glTF".to_vec();
    bytes.extend(2u32.to_le_bytes());
    bytes.extend(
        u32::try_from(12 + 8 + json.len() + 8 + 36)
            .unwrap()
            .to_le_bytes(),
    );
    bytes.extend(u32::try_from(json.len()).unwrap().to_le_bytes());
    bytes.extend(b"JSON");
    bytes.extend(json);
    bytes.extend(36u32.to_le_bytes());
    bytes.extend(b"BIN\0");
    for value in [-0.5f32, -0.5, 0.0, 0.5, -0.5, 0.0, 0.0, 0.5, 0.0] {
        bytes.extend(value.to_le_bytes());
    }
    bytes
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::SceneDocument;

    #[test]
    fn mesh_table_contains_only_uuid_and_original_glb_bytes() {
        let bytes = triangle_glb();
        let mesh = GlbMesh::new("00000000-0000-0000-0000-000000000001", &bytes).unwrap();
        let batch = encode_mesh_batch(std::slice::from_ref(&mesh)).unwrap();
        assert_eq!(batch.num_columns(), 2);
        assert_eq!(batch.schema().field(1).name(), "glb");
        let decoded = decode_mesh_batch(&batch).unwrap();
        assert_eq!(decoded, vec![mesh]);
        assert_eq!(decoded[0].bytes(), bytes);
        assert_eq!(decoded[0].decode(7).unwrap().mesh_id, Some(7));
    }

    #[test]
    fn duplicate_ids_and_non_glb_payloads_are_rejected() {
        let id = "00000000-0000-0000-0000-000000000001";
        assert!(GlbMesh::new(id, b"v 0 0 0\nf 1 2 3").is_err());
        let mesh = GlbMesh::new(id, &triangle_glb()).unwrap();
        assert!(encode_mesh_batch(&[mesh.clone(), mesh]).is_err());
    }

    #[test]
    fn document_round_trip_keeps_glb_bytes_exactly() {
        let params = crate::FrameParams {
            model: Some(crate::Matrix4::IDENTITY.to_cols_array()),
            ..crate::FrameParams::IDENTITY
        };
        let bytes = super::super::scene_encode::encode_params_stream(&[params], None).unwrap();
        let mut document = SceneDocument::read(&bytes).unwrap();
        let glb = triangle_glb();
        document
            .bind_meshes(vec![GlbMesh::new(
                "00000000-0000-0000-0000-000000000001",
                &glb,
            )
            .unwrap()])
            .unwrap();
        let reopened = SceneDocument::read(&document.write().unwrap()).unwrap();
        assert_eq!(reopened.meshes()[0].bytes(), glb);
        assert_eq!(reopened.frames().unwrap()[0].params, params);
    }
}
