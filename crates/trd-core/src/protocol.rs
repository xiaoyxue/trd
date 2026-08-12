use std::collections::HashMap;

use arrow::array::RecordBatch;
use arrow::buffer::Buffer;
use arrow::datatypes::{DataType, Schema};
use arrow::error::ArrowError;
use arrow::ipc::reader::StreamDecoder;

use crate::frame::validate_schema as validate_frames_schema;
#[cfg(test)]
use crate::texture::TEXTURE_COLUMN;
use crate::texture::{ImageTexture, TextureError};
use crate::visual::{Draw, DrawSelection};
use crate::{FrameError, FrameParams, InlineFrame, Mesh, MeshError};

mod arrow_decode;
/// The encode half of the wire format, compiled for tests only (#202): it
/// authors the `0.0.6` tables that this module's tests — and its own round-trip
/// tests — feed back through the real decoders.
#[cfg(test)]
mod scene_encode;
pub(crate) use arrow_decode::{
    check_version, decode_batch, decode_draws, decode_frame_ids, decode_frame_refs, validate_schema,
};

pub const PROTOCOL_VERSION: &str = "0.0.6";
pub const PROTOCOL_VERSION_KEY: &str = "trd.protocol.version";
pub const TABLE_KIND_KEY: &str = "trd.table.kind";

pub(crate) const MESH_TABLE_KIND: &str = "mesh";
pub(crate) const TEXTURE_TABLE_KIND: &str = "texture";
pub(crate) const FRAMES_TABLE_KIND: &str = "frames";
pub(crate) const PARAMS_TABLE_KIND: &str = "params";

/// Input schema versions this build accepts. The protocol is **not** backward
/// compatible: only the current [`PROTOCOL_VERSION`] (`0.0.6`) is accepted. A
/// stream is `[mesh][texture?][frames?][params]`; every sub-stream declares its
/// kind through [`TABLE_KIND_KEY`].
pub const SUPPORTED_INPUT_VERSIONS: &[&str] = &[PROTOCOL_VERSION];

/// Schema-metadata key declaring the stream's intended playback rate in frames
/// per second. Optional and version-independent: it defines *animation speed* so
/// that speed is a property of the data, not of a front-end's fps/refresh (see
/// the timing model). Absent ⇒ [`DEFAULT_FRAME_RATE`].
pub const FRAME_RATE_KEY: &str = "trd.stream.frame_rate";

/// Default playback rate (fps) when a stream omits [`FRAME_RATE_KEY`].
pub const DEFAULT_FRAME_RATE: f64 = 30.0;

/// The stream's declared playback rate from schema metadata, or
/// [`DEFAULT_FRAME_RATE`] when absent or unparsable/non-positive.
pub fn frame_rate_from_metadata(metadata: &HashMap<String, String>) -> f64 {
    metadata
        .get(FRAME_RATE_KEY)
        .and_then(|value| value.parse::<f64>().ok())
        .filter(|rate| rate.is_finite() && *rate > 0.0)
        .unwrap_or(DEFAULT_FRAME_RATE)
}

pub type FrameBatch = Vec<DecodedFrame>;

/// One decoded frame: its [`FrameParams`] plus the optional per-frame instanced
/// draw list (`draw_mesh`/`draw_model`) and optional background frame reference.
///
/// `draws` distinguishes two cases the renderer treats differently (via
/// [`DecodedFrame::resolved_draws`]):
///   * `None` — the stream carries **no** `draw_mesh`/`draw_model` columns
///     (legacy single-object stream); the renderer draws one default instance of
///     mesh `0` placed by the frame's own [`FrameParams::model_matrix`].
///   * `Some(list)` — an **explicit** draw list. `Some(vec![…])` places those
///     instances; `Some(vec![])` (an explicit *empty* list) draws **no meshes**,
///     so the frame is just its background plate (e.g. an AR frame where tracking
///     dropped out) rather than a default mesh.
///
/// `frame_ref` is the optional external background frame reference
/// (`frame_path`/`frame_url`) the shell resolves + composites beneath the scene;
/// `frame_id` is an optional index into the preceding inline frames table.
#[derive(Debug, Clone, PartialEq)]
pub struct DecodedFrame {
    pub params: FrameParams,
    pub draws: Option<Vec<Draw>>,
    pub frame_ref: Option<String>,
    pub frame_id: Option<u32>,
}

impl DecodedFrame {
    /// Resolves this frame's [`draws`](Self::draws) to the concrete instance list
    /// the renderer draws. An **explicit** list (`Some`, including an empty one →
    /// background only) is used verbatim; an **absent** list (`None`, legacy
    /// single-object stream) becomes one default instance of mesh `0` placed by
    /// the frame's own [`FrameParams::model_matrix`]. Shared by the native and
    /// wasm render paths so they resolve draws identically.
    pub fn resolved_draws(&self) -> Vec<Draw> {
        match &self.draws {
            Some(draws) => draws.clone(),
            None => vec![Draw {
                mesh_id: 0,
                model: self.params.model_matrix().to_cols_array(),
                selection: DrawSelection::INHERIT,
            }],
        }
    }
}

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
    #[error("input schema metadata is missing required key `{0}`")]
    MissingMetadata(&'static str),
    #[error(
        "input schema has unsupported `{TABLE_KIND_KEY}` value `{0}` \
         (expected mesh, texture, frames, or params)"
    )]
    UnsupportedTableKind(String),
    #[error("input table `{actual}` is out of order; expected {expected}")]
    UnexpectedTable {
        actual: &'static str,
        expected: &'static str,
    },
    #[error("input schema is missing required field `{0}`")]
    MissingColumn(&'static str),
    #[error("input column `{column}` has type {actual:?}, expected {expected}")]
    ColumnType {
        column: &'static str,
        expected: &'static str,
        actual: DataType,
    },
    #[error("input column `{0}` contains null values")]
    NullValues(&'static str),
    #[error(
        "conflicting camera forms: use either CV (`k`/`pose`) or CG \
         (`eye`/`target`/`direction`/`fovy`), not both"
    )]
    ConflictingCameraForms,
    #[error("incomplete CG camera: `eye` requires a look `target`/`direction` (and vice versa)")]
    IncompleteCameraForm,
    #[error("unsupported protocol version `{0}` (expected `{PROTOCOL_VERSION}`)")]
    UnsupportedVersion(String),
    #[error("mesh table decode failed: {0}")]
    Mesh(#[from] MeshError),
    #[error("texture table decode failed: {0}")]
    Texture(#[from] TextureError),
    #[error("frames table decode failed: {0}")]
    Frames(#[from] FrameError),
    #[error("params row {row} references frame_id {frame_id}, but the stream has no frames table")]
    MissingFramesTable { row: usize, frame_id: u32 },
    #[error(
        "params row {row} references frame_id {frame_id}, but the frames table has \
         {frame_count} row(s)"
    )]
    FrameIdOutOfRange {
        row: usize,
        frame_id: u32,
        frame_count: usize,
    },
    #[error("params row {row} specifies both inline frame_id and external frame_path/frame_url")]
    ConflictingFrameSources { row: usize },
    #[error(
        "per-frame draw list length mismatch at row {row}: \
         `draw_mesh` has {mesh_len} entries but `draw_model` has {model_len}"
    )]
    MismatchedDrawLists {
        row: usize,
        mesh_len: usize,
        model_len: usize,
    },
    #[error(
        "per-frame draw mode list length mismatch at row {row}: \
         `draw_mode` has {mode_len} entries but there are {draw_len} draw(s)"
    )]
    MismatchedDrawModes {
        row: usize,
        mode_len: usize,
        draw_len: usize,
    },
    #[error("draw_mode byte {value} is not a valid render mode (0/1/2/255)")]
    InvalidDrawMode { value: u8 },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SessionState {
    Open,
    Finished,
    Failed,
}

/// Which kind of concatenated IPC sub-stream the session is currently decoding.
/// A `0.0.6` stream is `[mesh][texture?][frames?][params]`; the params stream is
/// terminal.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum StreamKind {
    /// The leading **mesh** table (one row = one mesh); accumulated into `meshes`.
    Mesh,
    /// The optional **texture** table (one row = one image); decoded into
    /// `texture` and bound as the sampled albedo for [`RenderMode::Textured`].
    Texture,
    /// Optional inline background-frame resources indexed by params `frame_id`.
    Frames,
    /// The terminal **params** stream (one row = one frame).
    Params,
}

/// Incremental decoder for the trd input protocol, mirroring the native
/// [`crate::run_stream`] multi-stream framing but push-based for wasm. A `0.0.6`
/// stream is `[mesh][texture?][frames?][params]`: a leading **mesh** table
/// (one row = one mesh) decoded via [`Mesh::from_arrow_all`] and exposed through
/// [`InputSession::meshes`], an optional **texture** table decoded via
/// [`ImageTexture::from_arrow`] and exposed through [`InputSession::texture`],
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
    meshes: Vec<Mesh>,
    /// The image decoded from an optional leading **texture** table (`0.0.4`),
    /// bound as the sampled albedo. `None` for streams without a texture table.
    texture: Option<ImageTexture>,
    frames: Vec<InlineFrame>,
    frames_table_present: bool,
    mesh_table_present: bool,
    texture_table_present: bool,
    /// Whether a **params** schema has been decoded and validated (the terminal
    /// sub-stream). Frames can only be produced once true.
    params_schema_validated: bool,
    state: SessionState,
}

impl InputSession {
    pub fn new() -> Self {
        Self {
            decoder: StreamDecoder::new(),
            current_kind: None,
            meshes: Vec::new(),
            texture: None,
            frames: Vec::new(),
            frames_table_present: false,
            mesh_table_present: false,
            texture_table_present: false,
            params_schema_validated: false,
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
            self.classify_current_if_ready()?;
            if !self.params_schema_validated {
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

    /// Whether the stream carried a leading mesh table (required by the protocol).
    pub fn has_meshes(&self) -> bool {
        !self.meshes.is_empty()
    }

    /// The image decoded from an optional leading **texture** table (`0.0.4`),
    /// bound as the sampled albedo for [`crate::RenderMode::Textured`] meshes.
    /// `None` for streams without a texture table.
    pub fn texture(&self) -> Option<&ImageTexture> {
        self.texture.as_ref()
    }

    /// Whether the stream carried a leading texture table (`0.0.4`).
    pub fn has_texture(&self) -> bool {
        self.texture.is_some()
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
    /// or `None` if none has arrived yet. Falls back to [`DEFAULT_FRAME_RATE`]
    /// when the metadata key is absent.
    pub fn frame_rate(&self) -> Option<f64> {
        self.decoder
            .schema()
            .map(|schema| frame_rate_from_metadata(schema.metadata()))
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
            match self.decoder.decode(&mut bytes) {
                Ok(decoded) => {
                    self.classify_current_if_ready()?;
                    if let Some(batch) = decoded {
                        match self.current_kind {
                            Some(StreamKind::Mesh) => {
                                self.meshes.extend(Mesh::from_arrow_all(&batch)?)
                            }
                            Some(StreamKind::Texture) => self.decode_texture(&batch)?,
                            Some(StreamKind::Frames) => {
                                self.frames.extend(InlineFrame::from_arrow_all(&batch)?)
                            }
                            Some(StreamKind::Params) => batches.push(decode_frame_batch(
                                &batch,
                                self.frames_table_present,
                                self.frames.len(),
                            )?),
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

    /// Decodes the texture table's image (first non-empty row wins — one bound
    /// texture per stream). Later rows/batches of the same texture stream are
    /// ignored (a texture table is one row = one image).
    fn decode_texture(&mut self, batch: &RecordBatch) -> Result<(), ProtocolError> {
        if self.texture.is_none() && batch.num_rows() > 0 {
            self.texture = Some(ImageTexture::from_arrow(batch)?);
        }
        Ok(())
    }
}

impl Default for InputSession {
    fn default() -> Self {
        Self::new()
    }
}

impl StreamKind {
    fn as_str(self) -> &'static str {
        match self {
            Self::Mesh => MESH_TABLE_KIND,
            Self::Texture => TEXTURE_TABLE_KIND,
            Self::Frames => FRAMES_TABLE_KIND,
            Self::Params => PARAMS_TABLE_KIND,
        }
    }
}

fn table_kind(schema: &Schema) -> Result<StreamKind, ProtocolError> {
    check_version(schema)?;
    match schema.metadata().get(TABLE_KIND_KEY).map(String::as_str) {
        Some(MESH_TABLE_KIND) => Ok(StreamKind::Mesh),
        Some(TEXTURE_TABLE_KIND) => Ok(StreamKind::Texture),
        Some(FRAMES_TABLE_KIND) => Ok(StreamKind::Frames),
        Some(PARAMS_TABLE_KIND) => Ok(StreamKind::Params),
        Some(other) => Err(ProtocolError::UnsupportedTableKind(other.to_owned())),
        None => Err(ProtocolError::MissingMetadata(TABLE_KIND_KEY)),
    }
}

/// True if `error` is the arrow "Unexpected EOS" signal a [`StreamDecoder`]
/// raises when bytes remain after a stream's end-of-stream marker — i.e. the
/// boundary between a leading mesh stream and the params stream that follows it
/// in the same concatenated byte stream.
fn is_stream_boundary(error: &ArrowError) -> bool {
    matches!(error, ArrowError::IpcError(message) if message == "Unexpected EOS")
}

/// Decodes one params batch into [`DecodedFrame`]s: each row's [`FrameParams`]
/// zipped with its optional per-frame instanced draw list and optional background
/// frame reference. Rows from a stream without `draw_mesh`/`draw_model` columns
/// get `draws: None` (the renderer draws one default instance); when the columns
/// are present each row's list is carried as `Some(list)` (an explicit empty list
/// ⇒ background only). Rows without a frame column get `frame_ref: None`.
fn decode_frame_batch(
    batch: &RecordBatch,
    frames_table_present: bool,
    frame_count: usize,
) -> Result<FrameBatch, ProtocolError> {
    let params = decode_batch(batch)?;
    let draws = decode_draws(batch)?;
    let frame_refs = decode_frame_refs(batch)?;
    let frame_ids = decode_frame_ids(batch, frames_table_present, frame_count)?;
    params
        .into_iter()
        .enumerate()
        .map(|(row, params)| {
            let frame_ref = frame_refs.as_ref().and_then(|rows| rows[row].clone());
            let frame_id = frame_ids.as_ref().and_then(|rows| rows[row]);
            if frame_ref.is_some() && frame_id.is_some() {
                return Err(ProtocolError::ConflictingFrameSources { row });
            }
            Ok(DecodedFrame {
                params,
                draws: draws.as_ref().map(|rows| rows[row].clone()),
                frame_ref,
                frame_id,
            })
        })
        .collect()
}

/// Decodes a standalone **params** Arrow IPC stream (the bytes authored by
/// [`scene_encode::encode_params_stream`]) into one [`DecodedFrame`] per frame,
/// reusing the single [`InputSession`] framing decoder. Unlike a full stream it
/// does **not** require a leading mesh table — it is the params-only counterpart
/// for a producer that already holds the meshes, so it need not re-decode the
/// mesh per frame. Every batch's `trd.protocol.version` metadata is checked (via
/// [`InputSession`]'s schema validation).
///
/// Test-only (#202): its sole caller is the round-trip test in
/// [`scene_encode`], which pins the encoder against this decoder.
#[cfg(test)]
pub(crate) fn decode_params_stream(bytes: &[u8]) -> Result<Vec<DecodedFrame>, ProtocolError> {
    let mut session = InputSession::new();
    let mut frames = Vec::new();
    for batch in session.push(bytes)? {
        frames.extend(batch);
    }
    session.finish()?;
    Ok(frames)
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow::array::{
        Array, ArrayRef, FixedSizeListArray, Float32Array, Int32Array, RecordBatch, StringArray,
        UInt32Array,
    };
    use arrow::buffer::NullBuffer;
    use arrow::datatypes::{DataType, Field, Schema};
    use arrow::ipc::writer::StreamWriter;

    use super::*;
    use crate::FrameParams;

    /// Column-major identity 4×4, the default `model` for the test helpers.
    const IDENTITY_MODEL: [f32; 16] = [
        1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
    ];

    /// The frame a `model`-only params batch decodes to: identity everywhere,
    /// with the `model` column *present* (`Some(IDENTITY_MODEL)`), not absent.
    fn identity_frame() -> FrameParams {
        FrameParams {
            model: Some(IDENTITY_MODEL),
            ..FrameParams::IDENTITY
        }
    }

    /// The `model` column `Field` (`FixedSizeList<Float32>[16]`).
    fn model_field() -> Field {
        fixed_list_field("model", 16, false, false)
    }

    /// A params schema of the given `fields`, optionally tagging the protocol
    /// version in the metadata.
    fn schema_with(version: Option<&str>, fields: Vec<Field>) -> Arc<Schema> {
        let mut metadata = std::collections::HashMap::new();
        if let Some(version) = version {
            metadata.insert(PROTOCOL_VERSION_KEY.to_owned(), version.to_owned());
        }
        metadata.insert(TABLE_KIND_KEY.to_owned(), PARAMS_TABLE_KIND.to_owned());
        Arc::new(Schema::new(fields).with_metadata(metadata))
    }

    /// The minimal valid 0.0.6 params schema: a single `model` column.
    fn valid_schema(version: Option<&str>) -> Arc<Schema> {
        schema_with(version, vec![model_field()])
    }

    /// A params batch whose single `model` column holds one row per frame
    /// (`frame.model` or the identity when absent), reusing `schema`'s `model`
    /// field so nullable-declared variants share the builder.
    fn test_batch_with(schema: Arc<Schema>, frames: &[FrameParams]) -> RecordBatch {
        let model_item = match schema.field_with_name("model").unwrap().data_type() {
            DataType::FixedSizeList(item, 16) => item.clone(),
            data_type => panic!("unexpected model test type: {data_type:?}"),
        };
        let model = FixedSizeListArray::new(
            model_item,
            16,
            Arc::new(Float32Array::from(
                frames
                    .iter()
                    .flat_map(|frame| frame.model.unwrap_or(IDENTITY_MODEL))
                    .collect::<Vec<_>>(),
            )),
            None,
        );
        RecordBatch::try_new(schema, vec![Arc::new(model)]).unwrap()
    }

    fn test_batch(frames: &[FrameParams]) -> RecordBatch {
        test_batch_with(valid_schema(Some(PROTOCOL_VERSION)), frames)
    }

    /// Wraps decoded params as a draws-less [`FrameBatch`] — the shape a legacy
    /// params-only stream (or any frame without a `draw_mesh`/`draw_model` list)
    /// produces — for assertions that predate per-frame instanced draw lists.
    fn plain(frames: Vec<FrameParams>) -> FrameBatch {
        frames
            .into_iter()
            .map(|params| DecodedFrame {
                params,
                draws: None,
                frame_ref: None,
                frame_id: None,
            })
            .collect()
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

    /// A params batch whose `fovy` column is declared with the wrong type
    /// (`Int32` instead of `Float32`) — a schema type error.
    fn wrong_type_batch() -> RecordBatch {
        let schema = schema_with(
            Some(PROTOCOL_VERSION),
            vec![Field::new("fovy", DataType::Int32, false)],
        );
        RecordBatch::try_new(
            schema,
            vec![Arc::new(Int32Array::from(vec![0])) as ArrayRef],
        )
        .unwrap()
    }

    /// A params batch whose declared-non-null `model` column carries a null
    /// value in its single row — a runtime null that must be rejected after
    /// IPC decoding.
    fn null_model_batch() -> RecordBatch {
        let model = FixedSizeListArray::new(
            Arc::new(Field::new("item", DataType::Float32, false)),
            16,
            Arc::new(Float32Array::from(vec![0.0_f32; 16])),
            Some(NullBuffer::new_null(1)),
        );
        let schema = valid_schema(Some(PROTOCOL_VERSION));
        // SAFETY: This fixture intentionally violates non-nullability metadata
        // to verify runtime null rejection after IPC decoding.
        unsafe { RecordBatch::new_unchecked(schema, vec![Arc::new(model) as ArrayRef], 1) }
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
            identity_frame(),
            FrameParams {
                model: Some([
                    0.75, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.25, -0.5, 0.0,
                    1.0,
                ]),
                ..FrameParams::IDENTITY
            },
        ];
        let bytes = test_stream(&[test_batch(&expected)]);

        for split in 0..=bytes.len() {
            let mut session = InputSession::new();
            let mut batches = session.push(&bytes[..split]).unwrap();
            batches.extend(session.push(&bytes[split..]).unwrap());
            session.finish().unwrap();
            assert_eq!(batches, vec![plain(expected.clone())]);
        }
    }

    #[test]
    fn decodes_one_byte_fragments_and_multiple_batches() {
        let first = vec![identity_frame()];
        let second = vec![FrameParams {
            model: Some([
                0.25, 0.0, 0.0, 0.0, 0.0, 0.75, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.5, 0.0, 0.0, 1.0,
            ]),
            ..FrameParams::IDENTITY
        }];
        let bytes = test_stream(&[test_batch(&first), test_batch(&second)]);
        let mut session = InputSession::new();
        let mut batches = Vec::new();

        for byte in bytes {
            batches.extend(session.push(&[byte]).unwrap());
        }

        session.finish().unwrap();
        assert_eq!(batches, vec![plain(first), plain(second)]);
    }

    #[test]
    fn rejects_schema_type_errors_and_runtime_nulls() {
        let mut wrong = InputSession::new();
        assert!(matches!(
            wrong.push(&test_stream(&[wrong_type_batch()])),
            Err(ProtocolError::ColumnType { column: "fovy", .. })
        ));

        assert!(matches!(
            decode_batch(&null_model_batch()),
            Err(ProtocolError::NullValues("model"))
        ));

        let mut version = InputSession::new();
        assert!(matches!(
            version.push(&version_stream("9.9.9")),
            Err(ProtocolError::UnsupportedVersion(value)) if value == "9.9.9"
        ));

        // Absent version metadata is rejected: the protocol has no compatibility
        // mode and every input table declares its exact version.
        let without_version = test_batch_with(valid_schema(None), &[identity_frame()]);
        let mut missing = InputSession::new();
        assert!(matches!(
            missing.push(&test_stream(&[without_version])),
            Err(ProtocolError::MissingMetadata(PROTOCOL_VERSION_KEY))
        ));
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

    /// Builds a batch of `rows.len()` rows carrying a single fixed-size list
    /// column (`name`) whose row `i` holds `rows[i]`. The named column alone
    /// drives the row count (every params column is optional).
    fn batch_with_matrix(
        version: Option<&str>,
        name: &str,
        len: i32,
        rows: &[Vec<f32>],
        nullable: bool,
        child_nullable: bool,
    ) -> RecordBatch {
        let flat: Vec<f32> = rows.iter().flatten().copied().collect();
        let matrix = FixedSizeListArray::new(
            Arc::new(Field::new("item", DataType::Float32, child_nullable)),
            len,
            Arc::new(Float32Array::from(flat)),
            None,
        );
        let schema = schema_with(
            version,
            vec![fixed_list_field(name, len, nullable, child_nullable)],
        );
        RecordBatch::try_new(schema, vec![Arc::new(matrix)]).unwrap()
    }

    #[test]
    fn accepts_only_current_version_and_rejects_others() {
        // 0.0.6 is the only supported version: there is no backward compat for
        // 0.0.1–0.0.5, and future versions are rejected too.
        let mut session = InputSession::new();
        session.push(&version_stream(PROTOCOL_VERSION)).unwrap();
        session.finish().unwrap();

        for version in ["0.0.1", "0.0.2", "0.0.3", "0.0.4", "0.0.5", "0.0.7"] {
            let mut session = InputSession::new();
            assert!(
                matches!(
                    session.push(&version_stream(version)),
                    Err(ProtocolError::UnsupportedVersion(v)) if v == version
                ),
                "version {version} must be rejected"
            );
        }
    }

    #[test]
    fn decodes_matrix_columns_column_major() {
        // Asymmetric values so a transpose or stride bug can't hide.
        let model_row: Vec<f32> = (1..=16).map(|v| v as f32).collect();
        let k_row: Vec<f32> = (1..=9).map(|v| v as f32 * 0.5).collect();
        let pose_row: Vec<f32> = (1..=16).map(|v| -(v as f32)).collect();

        let model_batch = batch_with_matrix(
            Some(PROTOCOL_VERSION),
            "model",
            16,
            std::slice::from_ref(&model_row),
            false,
            false,
        );
        let k_batch = batch_with_matrix(
            Some(PROTOCOL_VERSION),
            "k",
            9,
            std::slice::from_ref(&k_row),
            false,
            false,
        );
        let pose_batch = batch_with_matrix(
            Some(PROTOCOL_VERSION),
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

    /// A non-null `FixedSizeList<Float32>[len]` column array from flat values.
    fn list_col(len: i32, flat: Vec<f32>) -> ArrayRef {
        Arc::new(FixedSizeListArray::new(
            Arc::new(Field::new("item", DataType::Float32, false)),
            len,
            Arc::new(Float32Array::from(flat)),
            None,
        )) as ArrayRef
    }

    /// Builds a one-row `0.0.6` batch of an identity `model` plus the given
    /// extra `(field, column)` pairs.
    fn camera_batch(extra: Vec<(Field, ArrayRef)>) -> RecordBatch {
        let mut fields = vec![model_field()];
        let mut columns: Vec<ArrayRef> = vec![list_col(16, IDENTITY_MODEL.to_vec())];
        for (field, column) in extra {
            fields.push(field);
            columns.push(column);
        }
        let schema = schema_with(Some(PROTOCOL_VERSION), fields);
        RecordBatch::try_new(schema, columns).unwrap()
    }

    #[test]
    fn decodes_cg_camera_columns() {
        let batch = camera_batch(vec![
            (
                fixed_list_field("eye", 3, false, false),
                list_col(3, vec![1.0, 2.0, 3.0]),
            ),
            (
                fixed_list_field("target", 3, false, false),
                list_col(3, vec![0.1, 0.2, 0.3]),
            ),
            (
                Field::new("fovy", DataType::Float32, false),
                Arc::new(Float32Array::from(vec![0.9_f32])) as ArrayRef,
            ),
        ]);
        let frames = decode_batch(&batch).unwrap();
        assert_eq!(frames[0].eye, Some([1.0, 2.0, 3.0]));
        assert_eq!(frames[0].target, Some([0.1, 0.2, 0.3]));
        assert_eq!(frames[0].fovy, Some(0.9));
        assert_eq!(frames[0].k, None);
        assert_eq!(frames[0].pose, None);
    }

    #[test]
    fn decodes_frame_reference_column_prefers_path_and_maps_null_or_empty_to_none() {
        // External background reference: `decode_frame_refs` surfaces one
        // Option<String> per row — the value the browser shell (and CLI/app)
        // resolves + composites beneath the scene, and the wasm renderers expose
        // via `frameRef(i)`. Per-row null/empty ⇒ None (keep the previous
        // background); an absent column ⇒ None for the whole batch.

        // (a) frame_path only: null and empty decode to None; others pass through.
        let batch = RecordBatch::try_new(
            Arc::new(Schema::new(vec![Field::new(
                "frame_path",
                DataType::Utf8,
                true,
            )])),
            vec![Arc::new(StringArray::from(vec![
                Some("frames/frame_000000.jpg"),
                None,
                Some(""),
                Some("frames/frame_000006.jpg"),
            ])) as ArrayRef],
        )
        .unwrap();
        assert_eq!(
            decode_frame_refs(&batch).unwrap(),
            Some(vec![
                Some("frames/frame_000000.jpg".to_owned()),
                None,
                None,
                Some("frames/frame_000006.jpg".to_owned()),
            ])
        );

        // (b) both columns present ⇒ `frame_path` (native) wins over `frame_url`
        // (the documented canonical order; a browser shell serving `frame_path`
        // relative to the page still resolves it, and by default the two strings
        // are identical).
        let batch = RecordBatch::try_new(
            Arc::new(Schema::new(vec![
                Field::new("frame_path", DataType::Utf8, true),
                Field::new("frame_url", DataType::Utf8, true),
            ])),
            vec![
                Arc::new(StringArray::from(vec![Some("local/a.jpg"), None])) as ArrayRef,
                Arc::new(StringArray::from(vec![
                    Some("https://cdn/x.jpg"),
                    Some("https://cdn/fallback.jpg"),
                ])) as ArrayRef,
            ],
        )
        .unwrap();
        assert_eq!(
            decode_frame_refs(&batch).unwrap(),
            Some(vec![
                Some("local/a.jpg".to_owned()),
                Some("https://cdn/fallback.jpg".to_owned()),
            ]),
            "frame_path is preferred per row, then frame_url is the fallback"
        );

        // (c) neither column present ⇒ None for the whole batch (params-only stream).
        let batch = RecordBatch::try_new(
            Arc::new(Schema::new(vec![Field::new(
                "fovy",
                DataType::Float32,
                false,
            )])),
            vec![Arc::new(Float32Array::from(vec![0.0_f32])) as ArrayRef],
        )
        .unwrap();
        assert_eq!(decode_frame_refs(&batch).unwrap(), None);

        // (d) a non-Utf8 frame column is a schema error.
        let batch = RecordBatch::try_new(
            Arc::new(Schema::new(vec![Field::new(
                "frame_path",
                DataType::Int32,
                true,
            )])),
            vec![Arc::new(Int32Array::from(vec![1])) as ArrayRef],
        )
        .unwrap();
        assert!(matches!(
            decode_frame_refs(&batch),
            Err(ProtocolError::ColumnType {
                column: "frame_path",
                ..
            })
        ));
    }

    #[test]
    fn decode_frame_batch_surfaces_frame_reference_on_decoded_frame() {
        // End-to-end through the batch decoder: a params batch carrying a
        // `frame_path` lands on DecodedFrame.frame_ref — what the wasm renderers
        // buffer + expose via frameRef(i), and the CLI/app resolve to composite
        // the background beneath the scene.
        let batch = camera_batch(vec![(
            Field::new("frame_path", DataType::Utf8, true),
            Arc::new(StringArray::from(vec![Some("frames/frame_000000.jpg")])) as ArrayRef,
        )]);
        let frames = decode_frame_batch(&batch, false, 0).unwrap();
        assert_eq!(
            frames[0].frame_ref,
            Some("frames/frame_000000.jpg".to_owned())
        );
    }

    #[test]
    fn rejects_incomplete_and_conflicting_camera_forms() {
        // `eye` alone (no look target/direction) is incomplete.
        let incomplete = camera_batch(vec![(
            fixed_list_field("eye", 3, false, false),
            list_col(3, vec![1.0, 2.0, 3.0]),
        )]);
        assert!(matches!(
            decode_batch(&incomplete),
            Err(ProtocolError::IncompleteCameraForm)
        ));

        // CV `k` mixed with CG `eye` is a conflicting form.
        let conflicting = camera_batch(vec![
            (
                fixed_list_field("k", 9, false, false),
                list_col(9, vec![1.0; 9]),
            ),
            (
                fixed_list_field("eye", 3, false, false),
                list_col(3, vec![1.0, 2.0, 3.0]),
            ),
        ]);
        assert!(matches!(
            decode_batch(&conflicting),
            Err(ProtocolError::ConflictingCameraForms)
        ));
    }

    #[test]
    fn rejects_wrong_size_matrix_column() {
        // A `model` column declared as length 9 (not 16) is a schema error, and
        // the session becomes terminal.
        let bad = batch_with_matrix(
            Some(PROTOCOL_VERSION),
            "model",
            9,
            &[vec![0.0; 9]],
            false,
            false,
        );
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
    fn accepts_nullable_declared_fields_with_non_null_values() {
        // Producers (e.g. pyarrow) emit nullable-by-default fields whose *values*
        // are non-null. The native decoder (`stream.rs`) accepts these, so this
        // cross-platform/wasm decoder must too — otherwise the same stream
        // renders on the CLI but fails to load in the browser. Only null
        // *values* are rejected (see `rejects_schema_type_errors_and_runtime_nulls`).
        let nullable_model = test_batch_with(
            schema_with(
                Some(PROTOCOL_VERSION),
                vec![fixed_list_field("model", 16, true, false)],
            ),
            &[identity_frame()],
        );
        let mut model = InputSession::new();
        assert_eq!(
            model.push(&test_stream(&[nullable_model])).unwrap(),
            vec![plain(vec![identity_frame()])]
        );

        let nullable_child = test_batch_with(
            schema_with(
                Some(PROTOCOL_VERSION),
                vec![fixed_list_field("model", 16, false, true)],
            ),
            &[identity_frame()],
        );
        let mut child = InputSession::new();
        assert_eq!(
            child.push(&test_stream(&[nullable_child])).unwrap(),
            vec![plain(vec![identity_frame()])]
        );

        // A nullable-declared optional matrix column with non-null values also decodes.
        let nullable_pose = batch_with_matrix(
            Some(PROTOCOL_VERSION),
            "pose",
            16,
            &[vec![0.0; 16]],
            true,
            false,
        );
        let mut pose = InputSession::new();
        assert!(pose.push(&test_stream(&[nullable_pose])).is_ok());
    }

    #[test]
    fn frame_rate_metadata_parses_or_defaults() {
        let rate = |value: &str| {
            frame_rate_from_metadata(&std::collections::HashMap::from([(
                FRAME_RATE_KEY.to_string(),
                value.to_string(),
            )]))
        };
        assert_eq!(rate("60"), 60.0);
        assert_eq!(rate("23.976"), 23.976);
        // Absent, unparsable, or non-positive/non-finite fall back to the default.
        assert_eq!(
            frame_rate_from_metadata(&std::collections::HashMap::new()),
            DEFAULT_FRAME_RATE
        );
        assert_eq!(rate("not-a-number"), DEFAULT_FRAME_RATE);
        assert_eq!(rate("0"), DEFAULT_FRAME_RATE);
        assert_eq!(rate("-5"), DEFAULT_FRAME_RATE);
        assert_eq!(rate("inf"), DEFAULT_FRAME_RATE);
    }

    #[test]
    fn input_session_reports_frame_rate_after_schema() {
        let mut session = InputSession::new();
        assert_eq!(session.frame_rate(), None);
        // A stream without frame_rate metadata reports the default once decoded.
        session
            .push(&test_stream(&[test_batch(&[FrameParams::IDENTITY])]))
            .unwrap();
        assert_eq!(session.frame_rate(), Some(DEFAULT_FRAME_RATE));
    }

    proptest::proptest! {
        #[test]
        fn model_column_roundtrips(values in proptest::collection::vec(-1000.0_f32..1000.0, 16)) {
            let batch = batch_with_matrix(Some(PROTOCOL_VERSION), "model", 16, std::slice::from_ref(&values), false, false);
            let decoded = decode_batch(&batch).unwrap();
            let expected = <[f32; 16]>::try_from(values).unwrap();
            proptest::prop_assert_eq!(decoded[0].model, Some(expected));
        }
    }

    /// Serializes a `Mesh` as a `0.0.3` leading **mesh table** Arrow IPC stream
    /// (one row: `position`/`color` `List<FixedSizeList<Float32>[3]>` + `index`
    /// `List<UInt32>`), mirroring the native `stream::write_mesh_stream`.
    fn write_mesh_stream(mesh: &crate::Mesh) -> Vec<u8> {
        use arrow::array::{ListArray, UInt32Array};
        use arrow::buffer::OffsetBuffer;

        let fsl_type =
            DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float32, false)), 3);
        let geometry = |flat: Vec<f32>| -> ArrayRef {
            let fsl = FixedSizeListArray::new(
                Arc::new(Field::new("item", DataType::Float32, false)),
                3,
                Arc::new(Float32Array::from(flat)),
                None,
            );
            let field = Arc::new(Field::new("item", fsl_type.clone(), false));
            Arc::new(ListArray::new(
                field,
                OffsetBuffer::from_lengths([fsl.len()]),
                Arc::new(fsl),
                None,
            ))
        };
        let positions: Vec<f32> = mesh.vertices.iter().flat_map(|v| v.position).collect();
        let colors: Vec<f32> = mesh.vertices.iter().flat_map(|v| v.color).collect();
        let idx_values = UInt32Array::from(mesh.indices.clone());
        let index: ArrayRef = Arc::new(ListArray::new(
            Arc::new(Field::new("item", DataType::UInt32, false)),
            OffsetBuffer::from_lengths([idx_values.len()]),
            Arc::new(idx_values),
            None,
        ));

        let list_of_fsl = DataType::List(Arc::new(Field::new("item", fsl_type.clone(), false)));
        let schema = Schema::new(vec![
            Field::new("position", list_of_fsl.clone(), false),
            Field::new("color", list_of_fsl, false),
            Field::new(
                "index",
                DataType::List(Arc::new(Field::new("item", DataType::UInt32, false))),
                false,
            ),
        ])
        .with_metadata(
            [
                (
                    PROTOCOL_VERSION_KEY.to_string(),
                    PROTOCOL_VERSION.to_string(),
                ),
                (TABLE_KIND_KEY.to_string(), MESH_TABLE_KIND.to_string()),
            ]
            .into_iter()
            .collect(),
        );
        let batch = RecordBatch::try_new(
            Arc::new(schema.clone()),
            vec![geometry(positions), geometry(colors), index],
        )
        .unwrap();
        let mut buf = Vec::new();
        let mut wr = StreamWriter::try_new(&mut buf, &schema).unwrap();
        wr.write(&batch).unwrap();
        wr.finish().unwrap();
        buf
    }

    /// A one-row params stream whose frame carries a `draw_mesh`/`draw_model`
    /// instanced draw list of the given `(mesh_id, model)` pairs.
    fn params_stream_with_draws(draws: &[(u32, [f32; 16])]) -> Vec<u8> {
        use arrow::array::{ListArray, UInt32Array};
        use arrow::buffer::OffsetBuffer;

        let mut fields = vec![model_field()];
        let mut columns: Vec<ArrayRef> = vec![list_col(16, IDENTITY_MODEL.to_vec())];

        let mesh_ids = UInt32Array::from(draws.iter().map(|(id, _)| *id).collect::<Vec<_>>());
        let draw_mesh: ArrayRef = Arc::new(ListArray::new(
            Arc::new(Field::new("item", DataType::UInt32, false)),
            OffsetBuffer::from_lengths([draws.len()]),
            Arc::new(mesh_ids),
            None,
        ));
        let mat_item =
            DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float32, false)), 16);
        let models = FixedSizeListArray::new(
            Arc::new(Field::new("item", DataType::Float32, false)),
            16,
            Arc::new(Float32Array::from(
                draws.iter().flat_map(|(_, m)| *m).collect::<Vec<_>>(),
            )),
            None,
        );
        let draw_model: ArrayRef = Arc::new(ListArray::new(
            Arc::new(Field::new("item", mat_item, false)),
            OffsetBuffer::from_lengths([draws.len()]),
            Arc::new(models),
            None,
        ));

        fields.push(Field::new(
            "draw_mesh",
            DataType::List(Arc::new(Field::new("item", DataType::UInt32, false))),
            false,
        ));
        fields.push(Field::new(
            "draw_model",
            DataType::List(Arc::new(Field::new(
                "item",
                DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float32, false)), 16),
                false,
            ))),
            false,
        ));
        columns.push(draw_mesh);
        columns.push(draw_model);

        let mut metadata = std::collections::HashMap::new();
        metadata.insert(PROTOCOL_VERSION_KEY.to_owned(), PROTOCOL_VERSION.to_owned());
        metadata.insert(TABLE_KIND_KEY.to_owned(), PARAMS_TABLE_KIND.to_owned());
        let schema = Arc::new(Schema::new(fields).with_metadata(metadata));
        let batch = RecordBatch::try_new(schema.clone(), columns).unwrap();
        let mut bytes = Vec::new();
        let mut writer = StreamWriter::try_new(&mut bytes, &schema).unwrap();
        writer.write(&batch).unwrap();
        writer.finish().unwrap();
        bytes
    }

    #[test]
    fn decodes_leading_mesh_table_then_params() {
        let mesh = crate::Mesh::hello_triangle();
        let frames = vec![
            identity_frame(),
            FrameParams {
                model: Some([
                    1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.3, 0.0, 0.0, 1.0,
                ]),
                ..FrameParams::IDENTITY
            },
        ];
        let mut bytes = write_mesh_stream(&mesh);
        bytes.extend(test_stream(&[test_batch(&frames)]));

        let mut session = InputSession::new();
        let batches = session.push(&bytes).unwrap();
        session.finish().unwrap();

        assert!(session.has_meshes());
        assert_eq!(session.meshes(), &[mesh]);
        assert_eq!(batches, vec![plain(frames)]);
    }

    #[test]
    fn decodes_mesh_then_params_across_every_split() {
        // The mesh→params boundary (an end-of-stream marker) must be recovered no
        // matter which byte the chunk boundary falls on, including inside the EOS
        // marker itself and after a clean mesh-only chunk.
        let mesh = crate::Mesh::hello_triangle();
        let frames = vec![identity_frame()];
        let mut bytes = write_mesh_stream(&mesh);
        bytes.extend(test_stream(&[test_batch(&frames)]));

        for split in 0..=bytes.len() {
            let mut session = InputSession::new();
            let mut batches = session.push(&bytes[..split]).unwrap();
            batches.extend(session.push(&bytes[split..]).unwrap());
            session.finish().unwrap();
            assert_eq!(
                session.meshes(),
                std::slice::from_ref(&mesh),
                "split at {split}"
            );
            assert_eq!(batches, vec![plain(frames.clone())], "split at {split}");
        }
    }

    #[test]
    fn resolved_draws_distinguishes_absent_from_explicit_empty() {
        // Absent draw list (`None`, legacy single-object stream) ⇒ one default
        // instance of mesh 0 placed by the frame's own model.
        let absent = DecodedFrame {
            params: identity_frame(),
            draws: None,
            frame_ref: None,
            frame_id: None,
        };
        let resolved = absent.resolved_draws();
        assert_eq!(resolved.len(), 1);
        assert_eq!(resolved[0].mesh_id, 0);

        // Explicit *empty* draw list ⇒ no meshes: the frame is just its
        // background plate (e.g. an AR frame where tracking dropped out).
        let empty = DecodedFrame {
            params: identity_frame(),
            draws: Some(Vec::new()),
            frame_ref: None,
            frame_id: None,
        };
        assert!(empty.resolved_draws().is_empty());

        // Explicit non-empty list ⇒ used verbatim.
        let one = Draw {
            mesh_id: 3,
            model: [1.0_f32; 16],
            selection: DrawSelection::INHERIT,
        };
        let explicit = DecodedFrame {
            params: identity_frame(),
            draws: Some(vec![one]),
            frame_ref: None,
            frame_id: None,
        };
        assert_eq!(explicit.resolved_draws(), vec![one]);
    }

    #[test]
    fn decodes_per_frame_instanced_draw_lists() {
        let a = [1.0_f32; 16];
        let b = [2.0_f32; 16];
        let mesh = crate::Mesh::hello_triangle();
        let mut bytes = write_mesh_stream(&mesh);
        bytes.extend(params_stream_with_draws(&[(0, a), (1, b)]));

        let mut session = InputSession::new();
        let batches = session.push(&bytes).unwrap();
        session.finish().unwrap();

        assert_eq!(batches.len(), 1);
        assert_eq!(batches[0].len(), 1);
        assert_eq!(
            batches[0][0].draws,
            Some(vec![
                Draw {
                    mesh_id: 0,
                    model: a,
                    selection: DrawSelection::INHERIT
                },
                Draw {
                    mesh_id: 1,
                    model: b,
                    selection: DrawSelection::INHERIT
                },
            ])
        );
    }

    #[test]
    fn params_only_stream_has_no_meshes() {
        // The push decoder itself accepts a params-only stream (mesh-first is
        // enforced by the renderer, not the decoder).
        let frames = vec![identity_frame()];
        let mut session = InputSession::new();
        let batches = session.push(&test_stream(&[test_batch(&frames)])).unwrap();
        session.finish().unwrap();
        assert!(!session.has_meshes());
        assert!(session.meshes().is_empty());
        assert_eq!(batches, vec![plain(frames)]);
    }

    /// Serializes a `[height, width, 4]` RGBA image as a `0.0.4` **texture table**
    /// Arrow IPC stream (one row: `rgba` `FixedSizeList<UInt8>[H*W*4]` carrying the
    /// `arrow.fixed_shape_tensor` extension), mirroring `texture::from_arrow`'s
    /// expected wire form.
    fn write_texture_stream(width: usize, height: usize, rgba: Vec<u8>) -> Vec<u8> {
        use arrow::array::UInt8Array;
        use arrow_schema::extension::FixedShapeTensor;

        let list_size = (width * height * 4) as i32;
        let storage = DataType::FixedSizeList(
            Arc::new(Field::new("item", DataType::UInt8, false)),
            list_size,
        );
        let extension = FixedShapeTensor::try_new(
            DataType::UInt8,
            vec![height, width, 4],
            Some(vec![
                "height".to_string(),
                "width".to_string(),
                "channel".to_string(),
            ]),
            None,
        )
        .unwrap();
        let field = Field::new(TEXTURE_COLUMN, storage, false).with_extension_type(extension);
        let array = FixedSizeListArray::new(
            Arc::new(Field::new("item", DataType::UInt8, false)),
            list_size,
            Arc::new(UInt8Array::from(rgba)),
            None,
        );
        let schema = Arc::new(
            Schema::new(vec![field]).with_metadata(
                [
                    (
                        PROTOCOL_VERSION_KEY.to_string(),
                        PROTOCOL_VERSION.to_string(),
                    ),
                    (TABLE_KIND_KEY.to_string(), TEXTURE_TABLE_KIND.to_string()),
                ]
                .into_iter()
                .collect(),
            ),
        );
        let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(array)]).unwrap();
        let mut bytes = Vec::new();
        let mut writer = StreamWriter::try_new(&mut bytes, &schema).unwrap();
        writer.write(&batch).unwrap();
        writer.finish().unwrap();
        bytes
    }

    #[test]
    fn decodes_mesh_then_texture_then_params() {
        // A full `[mesh][texture][params]` stream: the mesh table, a 2x2
        // checker texture, then a params frame all decode, with the texture bound.
        let mesh = crate::Mesh::hello_triangle();
        let rgba = vec![
            255, 255, 255, 255, 255, 0, 0, 255, // white, red
            0, 255, 0, 255, 0, 0, 255, 255, // green, blue
        ];
        let frames = vec![identity_frame()];
        let mut bytes = write_mesh_stream(&mesh);
        bytes.extend(write_texture_stream(2, 2, rgba.clone()));
        bytes.extend(test_stream(&[test_batch(&frames)]));

        let mut session = InputSession::new();
        let batches = session.push(&bytes).unwrap();
        session.finish().unwrap();

        assert!(session.has_meshes());
        assert_eq!(session.meshes(), std::slice::from_ref(&mesh));
        assert!(session.has_texture());
        let texture = session.texture().expect("texture bound");
        assert_eq!((texture.width(), texture.height()), (2, 2));
        assert_eq!(texture.rgba(), rgba.as_slice());
        assert_eq!(batches, vec![plain(frames)]);
    }

    #[test]
    fn decodes_mesh_then_texture_then_params_across_every_split() {
        // Every sub-stream boundary (two EOS markers) must be recovered no matter
        // which byte the chunk boundary lands on.
        let mesh = crate::Mesh::hello_triangle();
        let rgba = vec![9u8; 2 * 2 * 4];
        let frames = vec![identity_frame()];
        let mut bytes = write_mesh_stream(&mesh);
        bytes.extend(write_texture_stream(2, 2, rgba.clone()));
        bytes.extend(test_stream(&[test_batch(&frames)]));

        for split in 0..=bytes.len() {
            let mut session = InputSession::new();
            let mut batches = session.push(&bytes[..split]).unwrap();
            batches.extend(session.push(&bytes[split..]).unwrap());
            session.finish().unwrap();
            assert_eq!(
                session.meshes(),
                std::slice::from_ref(&mesh),
                "split {split}"
            );
            assert!(session.has_texture(), "split {split}");
            assert_eq!(batches, vec![plain(frames.clone())], "split {split}");
        }
    }

    #[test]
    fn mesh_then_params_has_no_texture() {
        // A `[mesh][params]` stream (no texture table) binds no texture.
        let mesh = crate::Mesh::hello_triangle();
        let frames = vec![identity_frame()];
        let mut bytes = write_mesh_stream(&mesh);
        bytes.extend(test_stream(&[test_batch(&frames)]));

        let mut session = InputSession::new();
        session.push(&bytes).unwrap();
        session.finish().unwrap();

        assert!(session.has_meshes());
        assert!(!session.has_texture());
        assert!(session.texture().is_none());
    }

    fn inline_pixel(value: u8) -> InlineFrame {
        InlineFrame::Pixels(crate::ImageData {
            width: 1,
            height: 1,
            rgba: vec![value, value + 1, value + 2, 255],
        })
    }

    #[test]
    fn decodes_frames_table_and_reusable_frame_ids() {
        let mesh = crate::Mesh::hello_triangle();
        let resources = vec![inline_pixel(10), inline_pixel(20)];
        let params = vec![
            FrameParams::IDENTITY,
            FrameParams::IDENTITY,
            FrameParams::IDENTITY,
        ];
        let ids = [Some(0), Some(1), Some(0)];
        let bytes = scene_encode::encode_scene_with_frames(
            std::slice::from_ref(&mesh),
            &resources,
            &params,
            None,
            &ids,
        )
        .unwrap();

        let mut session = InputSession::new();
        let decoded: Vec<_> = session
            .push(&bytes)
            .unwrap()
            .into_iter()
            .flatten()
            .collect();
        session.finish().unwrap();

        assert_eq!(session.meshes(), &[mesh]);
        assert!(session.has_frames_table());
        assert_eq!(session.frames(), resources);
        assert_eq!(
            decoded
                .iter()
                .map(|frame| frame.frame_id)
                .collect::<Vec<_>>(),
            ids
        );
        assert!(decoded.iter().all(|frame| frame.frame_ref.is_none()));
    }

    #[test]
    fn mesh_texture_frames_params_decodes_across_every_split() {
        let mesh = crate::Mesh::hello_triangle();
        let texture = crate::ImageTexture::from_rgba(1, 1, vec![100, 110, 120, 255]).unwrap();
        let resources = vec![inline_pixel(30)];
        let mut bytes = scene_encode::encode_mesh_stream(std::slice::from_ref(&mesh)).unwrap();
        bytes.extend(scene_encode::encode_texture_stream(&texture).unwrap());
        bytes.extend(scene_encode::encode_frames_stream(&resources).unwrap());
        bytes.extend(
            scene_encode::encode_params_stream_with_frame_ids(
                &[FrameParams::IDENTITY],
                None,
                Some(&[Some(0)]),
            )
            .unwrap(),
        );

        for split in 0..=bytes.len() {
            let mut session = InputSession::new();
            let mut batches = session.push(&bytes[..split]).unwrap();
            batches.extend(session.push(&bytes[split..]).unwrap());
            session.finish().unwrap();
            assert_eq!(
                session.meshes(),
                std::slice::from_ref(&mesh),
                "split {split}"
            );
            assert_eq!(session.frames(), resources, "split {split}");
            assert!(session.has_texture(), "split {split}");
            assert_eq!(batches.len(), 1, "split {split}");
            assert_eq!(batches[0][0].frame_id, Some(0), "split {split}");
        }
    }

    #[test]
    fn frame_id_requires_table_and_valid_index() {
        let missing = camera_batch(vec![(
            Field::new("frame_id", DataType::UInt32, true),
            Arc::new(UInt32Array::from(vec![Some(0)])) as ArrayRef,
        )]);
        assert!(matches!(
            decode_frame_batch(&missing, false, 0),
            Err(ProtocolError::MissingFramesTable {
                row: 0,
                frame_id: 0
            })
        ));

        let out_of_range = camera_batch(vec![(
            Field::new("frame_id", DataType::UInt32, true),
            Arc::new(UInt32Array::from(vec![Some(2)])) as ArrayRef,
        )]);
        assert!(matches!(
            decode_frame_batch(&out_of_range, true, 2),
            Err(ProtocolError::FrameIdOutOfRange {
                row: 0,
                frame_id: 2,
                frame_count: 2
            })
        ));

        let null = camera_batch(vec![(
            Field::new("frame_id", DataType::UInt32, true),
            Arc::new(UInt32Array::from(vec![None])) as ArrayRef,
        )]);
        let decoded = decode_frame_batch(&null, false, 0).unwrap();
        assert_eq!(decoded[0].frame_id, None);
    }

    #[test]
    fn inline_and_external_frame_sources_conflict() {
        let batch = camera_batch(vec![
            (
                Field::new("frame_id", DataType::UInt32, true),
                Arc::new(UInt32Array::from(vec![Some(0)])) as ArrayRef,
            ),
            (
                Field::new("frame_path", DataType::Utf8, true),
                Arc::new(StringArray::from(vec![Some("frames/a.png")])) as ArrayRef,
            ),
        ]);
        assert!(matches!(
            decode_frame_batch(&batch, true, 1),
            Err(ProtocolError::ConflictingFrameSources { row: 0 })
        ));
    }

    #[test]
    fn explicit_table_kind_is_required_and_validated() {
        let schema = valid_schema(Some(PROTOCOL_VERSION));
        let mut metadata = schema.metadata().clone();
        metadata.remove(TABLE_KIND_KEY);
        let missing_kind = test_batch_with(
            Arc::new(Schema::new(schema.fields().clone()).with_metadata(metadata)),
            &[identity_frame()],
        );
        let mut session = InputSession::new();
        assert!(matches!(
            session.push(&test_stream(&[missing_kind])),
            Err(ProtocolError::MissingMetadata(TABLE_KIND_KEY))
        ));

        let schema = valid_schema(Some(PROTOCOL_VERSION));
        let mut metadata = schema.metadata().clone();
        metadata.insert(TABLE_KIND_KEY.to_owned(), "mystery".to_owned());
        let unknown_kind = test_batch_with(
            Arc::new(Schema::new(schema.fields().clone()).with_metadata(metadata)),
            &[identity_frame()],
        );
        let mut session = InputSession::new();
        assert!(matches!(
            session.push(&test_stream(&[unknown_kind])),
            Err(ProtocolError::UnsupportedTableKind(kind)) if kind == "mystery"
        ));
    }

    #[test]
    fn schema_only_frames_table_is_validated_during_classification() {
        let mesh = crate::Mesh::hello_triangle();
        let schema = Schema::new(vec![Field::new("other", DataType::UInt8, false)]).with_metadata(
            [
                (
                    PROTOCOL_VERSION_KEY.to_string(),
                    PROTOCOL_VERSION.to_string(),
                ),
                (TABLE_KIND_KEY.to_string(), FRAMES_TABLE_KIND.to_string()),
            ]
            .into_iter()
            .collect(),
        );
        let mut frames_stream = Vec::new();
        let mut writer = StreamWriter::try_new(&mut frames_stream, &schema).unwrap();
        writer.finish().unwrap();

        let mut bytes = write_mesh_stream(&mesh);
        bytes.extend(frames_stream);
        let mut session = InputSession::new();
        assert!(matches!(
            session.push(&bytes),
            Err(ProtocolError::Frames(FrameError::MissingPayloadColumns))
        ));
    }

    #[test]
    fn rejects_frames_table_before_mesh_and_duplicate_resource_tables() {
        let mesh = crate::Mesh::hello_triangle();
        let frames = scene_encode::encode_frames_stream(&[inline_pixel(1)]).unwrap();
        let params = scene_encode::encode_params_stream_with_frame_ids(
            &[FrameParams::IDENTITY],
            None,
            Some(&[Some(0)]),
        )
        .unwrap();

        let mut before_mesh = frames.clone();
        before_mesh.extend(scene_encode::encode_mesh_stream(std::slice::from_ref(&mesh)).unwrap());
        before_mesh.extend(params.clone());
        let mut session = InputSession::new();
        assert!(matches!(
            session.push(&before_mesh),
            Err(ProtocolError::UnexpectedTable {
                actual: FRAMES_TABLE_KIND,
                ..
            })
        ));

        let mut duplicate = scene_encode::encode_mesh_stream(std::slice::from_ref(&mesh)).unwrap();
        duplicate.extend(frames.clone());
        duplicate.extend(frames);
        duplicate.extend(params);
        let mut session = InputSession::new();
        assert!(matches!(
            session.push(&duplicate),
            Err(ProtocolError::UnexpectedTable {
                actual: FRAMES_TABLE_KIND,
                ..
            })
        ));
    }
}
