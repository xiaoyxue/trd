//! Native-only Arrow streaming protocol (trd protocol 0.0.6).
//!
//! The protocol is **not backward compatible**: only `0.0.6` is accepted (see
//! `AGENTS.md`). Input is a `[mesh][texture?][frames?][params]` byte stream of
//! concatenated Arrow IPC streams on stdin:
//! a **required** leading **mesh** table (one row = one mesh, all rows decoded
//! by [`Mesh::from_arrow_all`]), an optional **texture** table (one row = one
//! `fixed_shape_tensor<u8>[H,W,4]` image, decoded by [`ImageTexture::from_arrow`]
//! and bound as the sampled albedo), then the **params** stream (one row per
//! frame: optional camera columns `model`/`k`/`pose`/`eye`/`target`/`direction`/
//! `up`/`fovy`/`aspect`/`znear`/`zfar`, an optional per-frame instanced draw list
//! `draw_mesh` (`List<UInt32>`) / `draw_model`
//! (`List<FixedSizeList<Float32>[16]>`) placing instances of the loaded meshes,
//! and an optional per-frame background `frame_path` reference). When the draw
//! list is absent, one instance of mesh 0 is placed by the frame's own `model`
//! (identity when absent). A params stream with no leading mesh table is an error
//! ([`StreamError::MissingMeshStream`]). Output: one row per frame, four
//! `fixed_shape_tensor<u8>` channels `r,g,b,a` of shape `[H, W]`.

use arrow::array::RecordBatch;
use arrow::datatypes::DataType;
use arrow::error::ArrowError;
use std::io::{Read, Write};
use std::sync::Arc;

// `Schema` is only referenced by the `#[cfg(test)]` decode wrappers + unit tests.
#[cfg(test)]
use arrow::datatypes::Schema;

// `Matrix4` is referenced only by the `#[cfg(test)]` unit tests (imported there).
use crate::protocol::{ProtocolError, PROTOCOL_VERSION};
use crate::render::{
    check_dimensions, FrameParams, Mesh, RenderOptions, Renderer, TargetError, TextureTarget,
};
use crate::render::{Draw, FrameFit};
use crate::texture::ImageTexture;
use crate::OutputSession;

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
    /// The stream mixes the CV (`k`/`pose`) and CG (`eye`/`target`/`direction`/
    /// `fovy`) camera forms; exactly one must be used.
    #[error(
        "conflicting camera forms: use either CV (`k`/`pose`) or CG \
         (`eye`/`target`/`direction`/`fovy`), not both"
    )]
    ConflictingCameraForms,
    /// The CG camera form is incomplete (an `eye` without a look
    /// `target`/`direction`, or vice versa).
    #[error("incomplete CG camera: `eye` requires a look `target`/`direction` (and vice versa)")]
    IncompleteCameraForm,
    /// A frame's per-instance `draw_mesh` and `draw_model` lists differ in
    /// length; each drawn instance needs exactly one mesh id and one model.
    #[error(
        "frame {row}: draw_mesh has {mesh_len} entries but draw_model has {model_len} \
         (each instance needs one mesh id and one model)"
    )]
    MismatchedDrawLists {
        row: usize,
        mesh_len: usize,
        model_len: usize,
    },
    /// A frame's per-instance `draw_mode` list differs in length from its
    /// `draw_mesh`/`draw_model` lists; each drawn instance needs exactly one
    /// mode byte when the column is present.
    #[error(
        "frame {row}: draw_mode has {mode_len} entries but there are {draw_len} \
         draw(s) (each instance needs one mode byte)"
    )]
    MismatchedDrawModes {
        row: usize,
        mode_len: usize,
        draw_len: usize,
    },
    /// A `draw_mode` byte is not a recognized [`crate::RenderMode`] encoding
    /// (`0`=filled, `1`=wireframe, `2`=textured, `255`=inherit global).
    #[error("draw_mode byte {value} is not a valid render mode (0/1/2/255)")]
    InvalidDrawMode { value: u8 },
    /// A draw references a mesh index outside the uploaded mesh set.
    #[error("draw references mesh index {mesh_id} but only {mesh_count} mesh(es) are loaded")]
    MeshIndexOutOfRange { mesh_id: u32, mesh_count: usize },
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
    /// The input is not mesh-first: the protocol requires a leading mesh table
    /// before the params stream (`[mesh][texture?][frames?][params]`). Params-only
    /// streams are no longer accepted.
    #[error("input is missing the required leading mesh table (protocol is mesh-first)")]
    MissingMeshStream,
    /// GPU rendering failed.
    #[error("render error: {0}")]
    Render(String),
    /// The leading mesh table could not be decoded into a [`Mesh`].
    #[error("mesh decode error: {0}")]
    Mesh(#[from] crate::MeshError),
    /// The optional leading texture table could not be decoded.
    #[error("texture decode error: {0}")]
    Texture(#[from] crate::TextureError),
    /// The optional inline frames table or a selected encoded frame failed to decode.
    #[error("inline frame decode error: {0}")]
    Frames(#[from] crate::FrameError),
    #[error(transparent)]
    Output(#[from] crate::OutputError),
}

/// Maps the shared [`ProtocolError`] (from the single decoder in
/// [`crate::protocol`]) onto this module's [`StreamError`], so the native
/// `run_stream` path keeps its flat error surface while the per-batch decode
/// logic lives in exactly one place.
/// The renderer has its own platform-neutral error (it must compile for wasm,
/// which the native-only stream module cannot); this keeps `run_stream`'s error
/// surface unchanged.
impl From<crate::render::RenderError> for StreamError {
    fn from(error: crate::render::RenderError) -> Self {
        match error {
            crate::render::RenderError::InvalidDimensions {
                width,
                height,
                reason,
            } => StreamError::InvalidDimensions {
                width,
                height,
                reason,
            },
            other => StreamError::Render(other.to_string()),
        }
    }
}

impl From<ProtocolError> for StreamError {
    fn from(error: ProtocolError) -> Self {
        match error {
            ProtocolError::Arrow(e) => StreamError::Arrow(e),
            ProtocolError::MissingColumn(c) => StreamError::MissingColumn(c),
            ProtocolError::ColumnType {
                column,
                expected,
                actual,
            } => StreamError::ColumnType {
                column,
                expected,
                actual,
            },
            ProtocolError::NullValues(c) => StreamError::NullValues(c),
            ProtocolError::ConflictingCameraForms => StreamError::ConflictingCameraForms,
            ProtocolError::IncompleteCameraForm => StreamError::IncompleteCameraForm,
            ProtocolError::UnsupportedVersion(v) => StreamError::UnsupportedVersion(v),
            ProtocolError::Mesh(e) => StreamError::Mesh(e),
            ProtocolError::Texture(e) => StreamError::Texture(e),
            ProtocolError::Frames(e) => StreamError::Frames(e),
            ProtocolError::MismatchedDrawLists {
                row,
                mesh_len,
                model_len,
            } => StreamError::MismatchedDrawLists {
                row,
                mesh_len,
                model_len,
            },
            ProtocolError::MismatchedDrawModes {
                row,
                mode_len,
                draw_len,
            } => StreamError::MismatchedDrawModes {
                row,
                mode_len,
                draw_len,
            },
            ProtocolError::InvalidDrawMode { value } => StreamError::InvalidDrawMode { value },
            // The session-framing errors can't arise from the per-batch decoders
            // used by `run_stream`; surface them as a generic render error if they
            // ever do.
            other @ (ProtocolError::SessionFinished
            | ProtocolError::SessionFailed
            | ProtocolError::MissingSchema
            | ProtocolError::NoProgress
            | ProtocolError::MissingMetadata(_)
            | ProtocolError::UnsupportedTableKind(_)
            | ProtocolError::UnexpectedTable { .. }
            | ProtocolError::MissingFramesTable { .. }
            | ProtocolError::FrameIdOutOfRange { .. }
            | ProtocolError::ConflictingFrameSources { .. }) => {
                StreamError::Render(other.to_string())
            }
        }
    }
}

/// Maps a render target's [`TargetError`] onto this module's [`StreamError`],
/// so [`Renderer`] keeps its flat error surface while the target's allocation
/// invariants live in [`crate::render::TextureTarget`].
impl From<TargetError> for StreamError {
    fn from(error: TargetError) -> Self {
        match error {
            TargetError::InvalidDimensions { width, height } => StreamError::InvalidDimensions {
                width,
                height,
                reason: "dimensions must be non-zero",
            },
            TargetError::ExceedsMaxDimension { width, height, .. } => {
                StreamError::InvalidDimensions {
                    width,
                    height,
                    reason: "exceeds adapter max_texture_dimension_2d",
                }
            }
            TargetError::RowOverflow { .. } | TargetError::Gpu(_) => {
                StreamError::Render(error.to_string())
            }
            TargetError::Output(e) => StreamError::Output(e),
        }
    }
}

/// If the schema declares a protocol version, require it to be supported.
/// Delegates to the shared [`crate::protocol::check_version`]. A test-only
/// [`StreamError`]-typed wrapper: `run_stream` now validates the version inside
/// the shared [`InputSession`](crate::InputSession), so this only exists to
/// exercise the [`ProtocolError`] → [`StreamError`] mapping in unit tests.
#[cfg(test)]
pub fn check_version(schema: &Schema) -> Result<(), StreamError> {
    Ok(crate::protocol::check_version(schema)?)
}

/// Decodes every row of `batch` into [`FrameParams`]. Delegates to the single
/// shared per-batch decoder [`crate::protocol::decode_batch`] (the source of
/// truth for both the native and wasm paths).
pub fn decode_frames(batch: &RecordBatch) -> Result<Vec<FrameParams>, StreamError> {
    Ok(crate::protocol::decode_batch(batch)?)
}

/// Decodes the optional per-frame **instanced draw list** columns into one
/// `Vec<Draw>` per row. A test-only [`StreamError`]-typed wrapper over
/// [`crate::protocol::decode_draws`] — the native/wasm paths use the shared
/// [`InputSession`](crate::InputSession) decoder directly; this only exercises
/// the [`ProtocolError`] → [`StreamError`] mapping in unit tests.
#[cfg(test)]
fn decode_draws(batch: &RecordBatch) -> Result<Option<Vec<Vec<Draw>>>, StreamError> {
    Ok(crate::protocol::decode_draws(batch)?)
}

/// Decodes the optional per-frame external **background frame reference** columns
/// into one `Option<String>` per row. A test-only [`StreamError`]-typed wrapper
/// over [`crate::protocol::decode_frame_refs`], exercising the [`ProtocolError`]
/// → [`StreamError`] mapping in unit tests.
#[cfg(test)]
fn decode_frame_refs(batch: &RecordBatch) -> Result<Option<Vec<Option<String>>>, StreamError> {
    Ok(crate::protocol::decode_frame_refs(batch)?)
}

/// Resolves one decoded frame's instanced draw list: its wire `draws` when
/// present (an explicit empty list ⇒ no meshes, so just the background plate),
/// else one instance of mesh 0 placed by the frame's own model (legacy
/// single-object behavior) — see [`DecodedFrame::resolved_draws`]. Every
/// referenced `mesh_id` is validated against `mesh_count`. Shared by the headless
/// [`run_stream`] path and the live [`read_scene_stream_with_meta`] front-end so
/// both resolve draws identically.
fn resolve_frame_draws(
    frame: &crate::DecodedFrame,
    mesh_count: usize,
) -> Result<Vec<Draw>, StreamError> {
    let draws = frame.resolved_draws();
    for draw in &draws {
        if draw.mesh_id as usize >= mesh_count {
            return Err(StreamError::MeshIndexOutOfRange {
                mesh_id: draw.mesh_id,
                mesh_count,
            });
        }
    }
    Ok(draws)
}

/// Reads a trd input stream **mesh-aware** — the same
/// `[mesh][texture?][frames?][params]`
/// framing [`run_stream`] uses — for a live front-end (e.g. the windowed
/// `trd-app`) that owns its own render target and encodes each frame's
/// [`Scene`](crate::Scene) itself, rather than the headless byte-stream path
/// [`run_stream`] drives.
///
/// Invokes `on_meshes` **once** with the decoded (required) leading mesh table,
/// then `on_texture` **once** with the optional bound texture (`Some` only when
/// the stream carries a texture table), then `on_meta` with the stream's declared
/// playback rate, then `on_frame` for each frame's `(FrameParams, draws)` in
/// order. A frame carrying no wire draw list defaults to one instance of mesh 0
/// placed by the frame's own model — matching [`run_stream`]. The mesh table's
/// rows are referenced by 0-based index; out-of-range `mesh_id`s are an error. A
/// params-only stream with no leading mesh table is a
/// [`StreamError::MissingMeshStream`].
pub fn read_scene_stream_with_meta<R: Read>(
    mut input: R,
    on_meshes: impl FnOnce(Vec<Mesh>),
    on_texture: impl FnOnce(Option<ImageTexture>),
    on_meta: impl FnOnce(f64),
    mut on_frame: impl FnMut(FrameParams, Vec<Draw>, Option<String>, Option<Arc<crate::ImageData>>),
) -> Result<(), StreamError> {
    let mut session = crate::InputSession::new();
    // FnOnce callbacks fired exactly once, when the params schema is first
    // reached (meshes + texture + fps complete); `Option::take` moves each out on
    // that single iteration so the borrow checker accepts calling them in a loop.
    let mut on_meshes = Some(on_meshes);
    let mut on_texture = Some(on_texture);
    let mut on_meta = Some(on_meta);
    let mut mesh_count = 0usize;
    let mut ready = false;
    let mut background_state = FrameBackgroundState::default();

    let mut buf = [0u8; 64 * 1024];
    loop {
        let n = input.read(&mut buf)?;
        if n == 0 {
            break;
        }
        let batches = session.push(&buf[..n])?;

        if !ready && session.has_schema() {
            if session.meshes().is_empty() {
                // The protocol is mesh-first; a params-only stream is rejected.
                return Err(StreamError::MissingMeshStream);
            }
            mesh_count = session.meshes().len();
            if let Some(cb) = on_meshes.take() {
                cb(session.meshes().to_vec());
            }
            if let Some(cb) = on_texture.take() {
                cb(session.texture().cloned());
            }
            if let Some(cb) = on_meta.take() {
                cb(session.frame_rate().unwrap_or(crate::DEFAULT_FRAME_RATE));
            }
            ready = true;
        }

        for batch in batches {
            for frame in batch {
                let draws = resolve_frame_draws(&frame, mesh_count)?;
                let inline =
                    resolve_inline_frame(frame.frame_id, session.frames(), &mut background_state)?
                        .map(|(image, _changed)| image);
                on_frame(frame.params, draws, frame.frame_ref, inline);
            }
        }
    }
    session.finish()?;

    if !ready {
        // No params schema was ever reached (empty input) — mesh-first unmet.
        return Err(StreamError::MissingMeshStream);
    }
    Ok(())
}

/// A shell-provided closure that resolves a per-frame background frame reference
/// (a `frame_path`/`frame_url` string) into decoded RGBA pixels. Kept
/// out of `trd-core` so the core performs no file/network I/O: the native CLI
/// supplies one backed by the `image` crate + a `--frames-base` dir; a stream
/// without background frames (or a shell that doesn't load them) passes `None`.
/// Returning `None` for a given reference renders that frame without a
/// background plane (the shell decides how to report the miss).
pub type FrameResolver<'a> = &'a dyn Fn(&str) -> Option<crate::texture::ImageData>;

#[derive(Default)]
struct FrameBackgroundState {
    last_ref: Option<String>,
    last_inline_id: Option<u32>,
    last_inline_image: Option<Arc<crate::ImageData>>,
}

fn resolve_inline_frame(
    frame_id: Option<u32>,
    frames: &[crate::InlineFrame],
    state: &mut FrameBackgroundState,
) -> Result<Option<(Arc<crate::ImageData>, bool)>, StreamError> {
    let Some(frame_id) = frame_id else {
        state.last_inline_id = None;
        state.last_inline_image = None;
        return Ok(None);
    };
    if state.last_inline_id == Some(frame_id) {
        return Ok(state.last_inline_image.clone().map(|image| (image, false)));
    }
    let resource = frames.get(frame_id as usize).ok_or_else(|| {
        StreamError::Render(format!(
            "frame_id {frame_id} escaped protocol range validation"
        ))
    })?;
    let image = Arc::new(resource.decode()?);
    state.last_inline_id = Some(frame_id);
    state.last_inline_image = Some(image.clone());
    Ok(Some((image, true)))
}

/// Renders one decoded [`FrameBatch`](crate::FrameBatch) and writes its output
/// batch, mirroring one Arrow output batch per input record batch. When
/// `frame_resolver` is `Some`, a frame carrying a `frame_path`/`frame_url`
/// reference has its background image resolved + uploaded and composited
/// beneath the scene via the scene's [`Background::frame`](crate::Background::frame).
/// `last_frame_ref` tracks the currently uploaded background so consecutive
/// frames sharing it skip the decode + re-upload.
#[allow(clippy::too_many_arguments)]
fn render_and_write_batch<W: Write>(
    renderer: &mut Renderer,
    target: &TextureTarget,
    options: &RenderOptions,
    output_session: &mut OutputSession,
    batch: &crate::FrameBatch,
    inline_frames: &[crate::InlineFrame],
    frame_resolver: Option<FrameResolver>,
    background_state: &mut FrameBackgroundState,
    output: &mut W,
) -> Result<(), StreamError> {
    let mesh_count = renderer.mesh_count();
    let mut planes: Vec<Vec<u8>> = Vec::with_capacity(batch.len());
    for frame in batch {
        let draws = resolve_frame_draws(frame, mesh_count)?;
        let mut frame_fit = None;
        if let Some((image, changed)) =
            resolve_inline_frame(frame.frame_id, inline_frames, background_state)?
        {
            if changed {
                renderer.update_frame_texture(&image);
            }
            background_state.last_ref = None;
            frame_fit = Some(FrameFit::Stretch);
        } else if let (Some(path), Some(resolve)) = (frame.frame_ref.as_deref(), frame_resolver) {
            if background_state.last_ref.as_deref() != Some(path) {
                if let Some(image) = resolve(path) {
                    renderer.update_frame_texture(&image);
                    background_state.last_ref = Some(path.to_owned());
                    frame_fit = Some(FrameFit::Stretch);
                }
            } else {
                frame_fit = Some(FrameFit::Stretch);
            }
        } else {
            background_state.last_ref = None;
        }
        // The scene is assembled here, from the wire draw list plus the CLI's
        // appearance options — the same `scene_with_overlays` every other
        // front-end uses, so they cannot drift apart (#180).
        let scene = crate::render::Scene::from_draws(&draws, options, frame_fit);
        // `run_stream` is a synchronous `Read`/`Write` filter, while the renderer
        // is async because GPU read-back is (the browser must not block its event
        // loop). Natively blocking here is free: the future is already complete
        // when `poll_for_map` returns. This is the only bridge between the two.
        planes.push(pollster::block_on(renderer.render_params(
            frame.params,
            &scene,
            target,
        ))?);
    }
    output_session.write_rgba_batch(&planes)?;
    output.write_all(&output_session.drain_new()?)?;
    Ok(())
}

/// Reads a trd input stream, renders each frame, and writes an Arrow IPC stream
/// of `fixed_shape_tensor` images to `output`. Output batch boundaries mirror
/// input batches (one batch in flight).
///
/// The protocol is `[mesh][texture?][frames?][params]`: the **required**
/// leading mesh table is decoded once (via [`Mesh::from_arrow_all`]) and
/// uploaded, then an optional texture table is uploaded as the bound albedo,
/// then the following params stream drives per-frame rendering. A params-only
/// stream with no leading mesh table is a [`StreamError::MissingMeshStream`].
///
/// Framing is driven by the single shared [`InputSession`](crate::InputSession)
/// (also used by the wasm renderers): input bytes are read in chunks and pushed
/// through it, so all the mesh-first sub-stream sniffing + boundary handling
/// lives in exactly one place. The only native-specific bit is the blocking
/// [`Read`] byte source.
pub fn run_stream<R: Read, W: Write>(
    mut input: R,
    mut output: W,
    width: u32,
    height: u32,
    options: RenderOptions,
    frame_resolver: Option<FrameResolver>,
) -> Result<(), StreamError> {
    // Validate dimensions up front so schema construction (which multiplies
    // width*height) can't overflow before Renderer's guard runs.
    check_dimensions(width, height)?;

    let mut session = crate::InputSession::new();
    // Built once the params schema is reached (meshes + texture + fps known).
    // The renderer and its texture target are a matched pair (#203): the
    // target is a call argument now, not a field the renderer owns, so the
    // stream holds both and threads the target through each render call.
    let mut renderer: Option<Renderer> = None;
    let mut target: Option<TextureTarget> = None;
    let mut output_session: Option<OutputSession> = None;
    // The background currently uploaded, so consecutive frames sharing it skip
    // the decode + re-upload.
    let mut background_state = FrameBackgroundState::default();

    let mut buf = [0u8; 64 * 1024];
    loop {
        let n = input.read(&mut buf)?;
        if n == 0 {
            break;
        }
        let batches = session.push(&buf[..n])?;

        // The mesh-first protocol delivers meshes + optional texture before the
        // params schema, so `has_schema()` flips true only once they're complete.
        if renderer.is_none() && session.has_schema() {
            if session.meshes().is_empty() {
                return Err(StreamError::MissingMeshStream);
            }
            let (mut built, built_target) =
                pollster::block_on(Renderer::with_meshes_sample_count(
                    width,
                    height,
                    session.meshes(),
                    options.msaa.sample_count(),
                ))?;
            if let Some(pbr) = &options.pbr {
                built.set_disney_material(pbr.material.clone());
                built.set_image_based_lighting(pbr.ibl);
                built.set_tone_mapping(pbr.tone_mapping);
                if let Some(env) = &pbr.env_map {
                    built.set_env_map(env.clone());
                }
            }
            if let Some(texture) = session.texture() {
                built.set_texture(texture);
            }
            renderer = Some(built);
            target = Some(built_target);

            let frame_rate = session.frame_rate().unwrap_or(crate::DEFAULT_FRAME_RATE);
            let mut session_out = OutputSession::with_frame_rate(width, height, Some(frame_rate))?;
            output.write_all(&session_out.drain_new()?)?;
            output_session = Some(session_out);
        }

        if let (Some(renderer), Some(target), Some(output_session)) =
            (renderer.as_mut(), target.as_ref(), output_session.as_mut())
        {
            for batch in &batches {
                render_and_write_batch(
                    renderer,
                    target,
                    &options,
                    output_session,
                    batch,
                    session.frames(),
                    frame_resolver,
                    &mut background_state,
                    &mut output,
                )?;
            }
        }
    }
    session.finish()?;

    // A stream that never reached a params schema (empty input) — the mesh-first
    // contract wasn't satisfied.
    let mut output_session = output_session.ok_or(StreamError::MissingMeshStream)?;
    output_session.finish()?;
    output.write_all(&output_session.drain_new()?)?;
    Ok(())
}

#[cfg(test)]
mod tests;
