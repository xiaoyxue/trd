//! [`InputSession`] — the incremental decoder for the trd input protocol.
//!
//! Push-based and **transport-free** on purpose: it is fed `push(&[u8])` from
//! whatever byte source a platform has, which is what lets the browser drive it
//! from its event loop and the native `InputStream<R: Read>` drive it from a
//! blocking read. It owns the arrow `StreamDecoder`, the `[mesh][texture?]
//! [frames?][params]` framing state machine, and the decoded prologue.
//!
//! It lives in `protocol/` rather than `io/` because a protocol *is* a state
//! machine: the table ordering, the sub-stream boundary recovery and the
//! version/`table_kind` validation it performs are wire-format rules, not
//! transport concerns. Owning no `Read`/`Write` is exactly what makes it a
//! `*Session` rather than a `*Stream`.

use arrow::array::RecordBatch;
use arrow::buffer::Buffer;
use arrow::ipc::reader::StreamDecoder;

use crate::session_state::SessionState;
use crate::{InlineFrame, Mesh, MeshAsset, MeshReference, MeshResource, Tonemap};

use super::{
    decode_frame_batch, decode_tonemap, frame_rate_from_metadata, is_stream_boundary, parse_error,
    table_kind, FrameBatch, ProtocolError, StreamKind,
};
use crate::frame::validate_schema as validate_frames_schema;
use crate::protocol::arrow_decode::validate_schema;

/// Incremental decoder for the trd input protocol, mirroring the native
/// [`crate::run_stream`] multi-stream framing but push-based for wasm. A `0.0.6`
/// stream is `[mesh][texture?][frames?][params]`: a leading **mesh** table
/// (one row = embedded geometry or a glTF reference), an optional embedded-mesh
/// **texture** table, and the resolved assets exposed through
/// [`InputSession::mesh_assets`],
/// then the terminal **params** stream driving per-frame rendering. Like the
/// native decoder this is a pure framing decoder — it accepts a params-only
/// stream; enforcing the mesh-first contract (a scene needs ≥1 mesh) is the
/// renderer's job.
pub struct InputSession {
    decoder: StreamDecoder,
    /// The kind of the sub-stream currently being decoded, or `None` until its
    /// schema is available (or just after a sub-stream boundary, before the next
    /// schema arrives).
    current_kind: Option<StreamKind>,
    mesh_resources: Vec<MeshResource>,
    mesh_assets: Vec<MeshAsset>,
    meshes: Vec<Mesh>,
    frames: Vec<InlineFrame>,
    frames_table_present: bool,
    mesh_table_present: bool,
    texture_table_present: bool,
    /// Whether a **params** schema has been decoded and validated (the terminal
    /// sub-stream). Frames can only be produced once true.
    params_schema_validated: bool,
    tonemap_column_present: bool,
    tonemap_observed: bool,
    tonemap_override: Option<Tonemap>,
    video_frame_indexed: Option<bool>,
    last_video_frame_index: Option<u32>,
    state: SessionState,
}

impl InputSession {
    pub fn new() -> Self {
        Self {
            decoder: StreamDecoder::new(),
            current_kind: None,
            mesh_resources: Vec::new(),
            mesh_assets: Vec::new(),
            meshes: Vec::new(),
            frames: Vec::new(),
            frames_table_present: false,
            mesh_table_present: false,
            texture_table_present: false,
            params_schema_validated: false,
            tonemap_column_present: false,
            tonemap_observed: false,
            tonemap_override: None,
            video_frame_indexed: None,
            last_video_frame_index: None,
            state: SessionState::default(),
        }
    }

    pub fn push(&mut self, chunk: &[u8]) -> Result<Vec<FrameBatch>, ProtocolError> {
        self.require_open()?;
        match self.push_open(chunk) {
            Ok(batches) => Ok(batches),
            Err(error) => self.state.fail(error),
        }
    }

    pub fn finish(&mut self) -> Result<(), ProtocolError> {
        self.require_open()?;

        let result = (|| {
            self.decoder.finish()?;
            self.classify_current_if_ready()?;
            if !self.params_schema_validated {
                return Err(ProtocolError::MissingSchema);
            }
            Ok(())
        })();

        self.state.close(result)
    }

    /// Whether a params schema has been decoded and validated (frames can be
    /// produced). For a stream with leading mesh/texture tables this becomes true
    /// only once the terminal params stream's schema arrives.
    pub fn has_schema(&self) -> bool {
        self.params_schema_validated
    }

    /// The meshes decoded from a stream's (required) leading mesh table, in
    /// stream order (mesh id = index). Non-empty for any accepted stream once its
    /// params schema has been reached.
    pub fn meshes(&self) -> &[Mesh] {
        &self.meshes
    }

    pub fn mesh_assets(&self) -> &[MeshAsset] {
        &self.mesh_assets
    }

    pub fn mesh_resources(&self) -> &[MeshResource] {
        &self.mesh_resources
    }

    pub fn mesh_resource_count(&self) -> usize {
        self.mesh_resources.len()
    }

    pub fn unresolved_mesh_references(&self) -> Vec<(u32, MeshReference)> {
        self.mesh_resources
            .iter()
            .enumerate()
            .filter_map(|(index, resource)| match resource {
                MeshResource::Gltf(reference) => Some((index as u32, reference.clone())),
                MeshResource::Resolved(_) => None,
            })
            .collect()
    }

    pub fn resolve_gltf(&mut self, index: u32, bytes: &[u8]) -> Result<(), ProtocolError> {
        let mesh_count = self.mesh_resources.len();
        let resource = self
            .mesh_resources
            .get_mut(index as usize)
            .ok_or(ProtocolError::MeshReferenceIndex { index, mesh_count })?;
        if !matches!(resource, MeshResource::Gltf(_)) {
            return Err(ProtocolError::MeshReferenceExpected { index });
        }
        let asset = crate::import_glb(bytes)
            .map(|asset| MeshAsset::from_gltf_with_id(index, asset))
            .map_err(|source| ProtocolError::GltfImport { index, source })?;
        *resource = MeshResource::Resolved(Box::new(asset));
        self.refresh_resolved_meshes();
        Ok(())
    }

    /// Whether the stream carried a leading mesh table (required by the protocol).
    pub fn has_meshes(&self) -> bool {
        !self.meshes.is_empty()
    }

    /// Mesh 0's decoded base-color texture, retained for the legacy single-texture
    /// accessor. Multi-model callers use [`InputSession::mesh_assets`].
    pub fn texture(&self) -> Option<&crate::ImageTexture> {
        self.texture_table_present
            .then(|| {
                self.mesh_assets
                    .first()
                    .and_then(|asset| asset.base_color_texture.as_ref())
            })
            .flatten()
    }

    /// Whether the stream carried a leading texture table.
    pub fn has_texture(&self) -> bool {
        self.texture_table_present
    }

    /// Inline background resources in frames-table row order.
    pub fn frames(&self) -> &[InlineFrame] {
        &self.frames
    }

    /// Whether an explicit frames table appeared, including an empty one.
    pub fn has_frames_table(&self) -> bool {
        self.frames_table_present
    }

    /// The stream's declared playback rate (fps) once a schema has been decoded,
    /// or `None` if none has arrived yet. Falls back to [`DEFAULT_FRAME_RATE`](super::DEFAULT_FRAME_RATE)
    /// when the metadata key is absent.
    pub fn frame_rate(&self) -> Option<f64> {
        self.decoder
            .schema()
            .map(|schema| frame_rate_from_metadata(schema.metadata()))
    }

    /// The params stream's explicit tone-map override, if present.
    ///
    /// Absence leaves an explicit consumer setting unchanged; consumers with no
    /// override use their Reinhard default.
    pub fn tonemap_override(&self) -> Option<Tonemap> {
        self.tonemap_override
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub(crate) fn tonemap_ready(&self) -> bool {
        !self.tonemap_column_present || self.tonemap_observed
    }

    fn require_open(&self) -> Result<(), ProtocolError> {
        self.state
            .ensure_open(ProtocolError::SessionFinished, ProtocolError::SessionFailed)
    }

    fn push_open(&mut self, chunk: &[u8]) -> Result<Vec<FrameBatch>, ProtocolError> {
        let mut bytes = Buffer::from_vec(chunk.to_vec());
        let mut batches = Vec::new();

        while !bytes.is_empty() {
            let before = bytes.len();
            match self.decoder.decode(&mut bytes) {
                Ok(decoded) => {
                    self.classify_current_if_ready()?;
                    if let Some(batch) = decoded {
                        match self.current_kind {
                            Some(StreamKind::Mesh) => {
                                let base = self.mesh_resources.len() as u32;
                                let mut resources = Mesh::decode_mesh_resources(&batch)?;
                                for (row, resource) in resources.iter_mut().enumerate() {
                                    if let MeshResource::Resolved(asset) = resource {
                                        asset.mesh_id = Some(base + row as u32);
                                    }
                                }
                                self.mesh_resources.extend(resources);
                                self.refresh_resolved_meshes();
                            }
                            Some(StreamKind::Texture) => self.decode_texture(&batch)?,
                            Some(StreamKind::Frames) => {
                                self.frames.extend(InlineFrame::from_arrow_all(&batch)?)
                            }
                            Some(StreamKind::Params) => {
                                self.observe_tonemap(&batch)?;
                                let frames = decode_frame_batch(
                                    &batch,
                                    self.frames_table_present,
                                    self.frames.len(),
                                )?;
                                self.validate_video_frame_indices(&frames)?;
                                batches.push(frames);
                            }
                            // A batch always implies its schema (classified above)
                            // is available, so `current_kind` is set here.
                            None => return Err(ProtocolError::MissingSchema),
                        }
                    }
                }
                Err(error) => {
                    // A leading mesh/texture sub-stream's end-of-stream marker
                    // leaves the following sub-stream's bytes in `bytes`; the
                    // decoder reports it as an "unexpected EOS". Switch to a fresh
                    // decoder for the next sub-stream and re-drive the remainder.
                    if is_stream_boundary(&error)
                        && matches!(
                            self.current_kind,
                            Some(StreamKind::Mesh)
                                | Some(StreamKind::Texture)
                                | Some(StreamKind::Frames)
                        )
                    {
                        self.decoder = StreamDecoder::new();
                        self.current_kind = None;
                        continue;
                    }
                    return Err(error.into());
                }
            }

            if bytes.len() == before {
                return Err(ProtocolError::NoProgress);
            }
        }

        self.classify_current_if_ready()?;
        Ok(batches)
    }

    /// Classifies the current sub-stream once its schema is available, validating
    /// the version (mesh/texture) or the full params schema (params). Idempotent:
    /// a no-op once `current_kind` is set (until the next sub-stream boundary
    /// resets it).
    fn classify_current_if_ready(&mut self) -> Result<(), ProtocolError> {
        if self.current_kind.is_some() {
            return Ok(());
        }
        let Some(schema) = self.decoder.schema() else {
            return Ok(());
        };
        let kind = table_kind(schema.as_ref())?;
        self.validate_table_order(kind)?;
        match kind {
            StreamKind::Frames => validate_frames_schema(schema.as_ref())?,
            StreamKind::Params => {
                validate_schema(schema.as_ref())?;
                self.params_schema_validated = true;
                self.tonemap_column_present = schema.field_with_name("tonemap").is_ok();
                self.tonemap_observed = !self.tonemap_column_present;
            }
            StreamKind::Mesh | StreamKind::Texture => {}
        }
        self.current_kind = Some(kind);
        Ok(())
    }

    fn validate_table_order(&mut self, kind: StreamKind) -> Result<(), ProtocolError> {
        let valid = match kind {
            StreamKind::Mesh => {
                !self.mesh_table_present
                    && !self.texture_table_present
                    && !self.frames_table_present
                    && !self.params_schema_validated
            }
            StreamKind::Texture => {
                self.mesh_table_present
                    && !self.texture_table_present
                    && !self.frames_table_present
                    && !self.params_schema_validated
            }
            StreamKind::Frames => {
                self.mesh_table_present
                    && !self.frames_table_present
                    && !self.params_schema_validated
            }
            // `decode_params_stream` intentionally uses the same decoder on a
            // standalone params stream. Full render entry points still require
            // the leading mesh table.
            StreamKind::Params => {
                !self.params_schema_validated
                    && (self.mesh_table_present
                        || (!self.texture_table_present && !self.frames_table_present))
            }
        };
        if !valid {
            return Err(ProtocolError::UnexpectedTable {
                actual: kind.as_str(),
                expected: match kind {
                    StreamKind::Mesh => "mesh as the first table",
                    StreamKind::Texture => "texture after mesh and before frames/params",
                    StreamKind::Frames => "frames after mesh/texture and before params",
                    StreamKind::Params => "one terminal params table",
                },
            });
        }
        match kind {
            StreamKind::Mesh => self.mesh_table_present = true,
            StreamKind::Texture => self.texture_table_present = true,
            StreamKind::Frames => self.frames_table_present = true,
            StreamKind::Params => {}
        }
        Ok(())
    }

    /// Decodes base-color textures and binds each row to its `mesh_id`.
    ///
    /// Legacy fixed-shape texture tables decode as one row for mesh 0.
    fn decode_texture(&mut self, batch: &RecordBatch) -> Result<(), ProtocolError> {
        if self.mesh_resources.iter().any(|resource| match resource {
            MeshResource::Gltf(_) => true,
            MeshResource::Resolved(asset) => asset.source == crate::MeshAssetSource::Gltf,
        }) {
            return Err(ProtocolError::TextureWithGltfReference);
        }
        let keyed = [
            crate::TEXTURE_MESH_ID_COLUMN,
            crate::TEXTURE_WIDTH_COLUMN,
            crate::TEXTURE_HEIGHT_COLUMN,
            crate::TEXTURE_RGBA_BYTES_COLUMN,
        ]
        .into_iter()
        .any(|name| batch.column_by_name(name).is_some());
        if !keyed {
            if batch.num_rows() == 0 || self.texture().is_some() {
                return Ok(());
            }
            let texture = crate::ImageTexture::from_arrow(batch)?;
            let mesh_count = self.mesh_resources.len();
            let resource =
                self.mesh_resources
                    .get_mut(0)
                    .ok_or(ProtocolError::MeshReferenceIndex {
                        index: 0,
                        mesh_count,
                    })?;
            let MeshResource::Resolved(asset) = resource else {
                return Err(ProtocolError::TextureWithGltfReference);
            };
            asset.base_color_texture = Some(texture);
            self.refresh_resolved_meshes();
            return Ok(());
        }
        for (mesh_id, texture) in crate::ImageTexture::from_arrow_assets(batch)? {
            let mesh_count = self.mesh_resources.len();
            let resource = self.mesh_resources.get_mut(mesh_id as usize).ok_or(
                ProtocolError::MeshReferenceIndex {
                    index: mesh_id,
                    mesh_count,
                },
            )?;
            let MeshResource::Resolved(asset) = resource else {
                return Err(ProtocolError::TextureWithGltfReference);
            };
            if asset.base_color_texture.is_some() {
                return Err(parse_error(format!(
                    "texture table contains duplicate mesh_id {mesh_id}"
                )));
            }
            asset.base_color_texture = Some(texture);
        }
        self.refresh_resolved_meshes();
        Ok(())
    }

    fn refresh_resolved_meshes(&mut self) {
        if self
            .mesh_resources
            .iter()
            .any(|resource| matches!(resource, MeshResource::Gltf(_)))
        {
            self.mesh_assets.clear();
            self.meshes.clear();
            return;
        }
        self.mesh_assets = self
            .mesh_resources
            .iter()
            .filter_map(|resource| match resource {
                MeshResource::Resolved(asset) => Some(asset.as_ref().clone()),
                MeshResource::Gltf(_) => None,
            })
            .collect();
        self.meshes = self
            .mesh_assets
            .iter()
            .map(|asset| asset.mesh.clone())
            .collect();
    }

    fn validate_video_frame_indices(
        &mut self,
        frames: &[crate::DecodedFrame],
    ) -> Result<(), ProtocolError> {
        if frames.is_empty() {
            return Ok(());
        }
        let indexed = frames
            .first()
            .is_some_and(|frame| frame.video_frame_index.is_some());
        match self.video_frame_indexed {
            Some(previous) if previous != indexed => {
                return Err(ProtocolError::MixedVideoFrameIndexMode);
            }
            None => self.video_frame_indexed = Some(indexed),
            _ => {}
        }
        if !indexed {
            return Ok(());
        }
        for current in frames.iter().filter_map(|frame| frame.video_frame_index) {
            if let Some(previous) = self.last_video_frame_index {
                if current <= previous {
                    return Err(ProtocolError::NonIncreasingVideoFrameIndex { previous, current });
                }
            }
            self.last_video_frame_index = Some(current);
        }
        Ok(())
    }

    fn observe_tonemap(&mut self, batch: &RecordBatch) -> Result<(), ProtocolError> {
        let Some(operator) = decode_tonemap(batch)? else {
            return Ok(());
        };
        self.tonemap_observed = true;
        if let Some(previous) = self.tonemap_override {
            if previous != operator {
                return Err(parse_error(format!(
                    "tonemap must be constant across the params stream (expected {}, got {})",
                    previous.to_wire(),
                    operator.to_wire()
                )));
            }
        } else {
            self.tonemap_override = Some(operator);
        }
        Ok(())
    }
}

impl Default for InputSession {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolved_gltf_still_rejects_external_texture_rows() {
        let gltf = crate::GltfAsset {
            mesh: Mesh::hello_triangle(),
            material: crate::DisneyMaterial::default(),
            base_color_texture: None,
            metallic_roughness_texture: None,
            normal_texture: None,
        };
        let mut session = InputSession::new();
        session.mesh_resources = vec![MeshResource::Resolved(Box::new(
            MeshAsset::from_gltf_with_id(0, gltf),
        ))];
        let texture = crate::ImageTexture::from_rgba(1, 1, vec![255; 4]).unwrap();
        let bytes = crate::encode_texture_assets(&[crate::SceneTexture {
            mesh_id: 0,
            texture: &texture,
        }])
        .unwrap();
        let reader =
            arrow::ipc::reader::StreamReader::try_new(std::io::Cursor::new(bytes), None).unwrap();
        let batch = reader.into_iter().next().unwrap().unwrap();

        assert!(matches!(
            session.decode_texture(&batch),
            Err(ProtocolError::TextureWithGltfReference)
        ));
    }
}
