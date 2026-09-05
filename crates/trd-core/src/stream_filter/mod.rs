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
use std::io::{Read, Write};

// `Matrix4` is referenced only by the `#[cfg(test)]` unit tests (imported there).
use crate::protocol::ProtocolError;
use crate::render::FrameFit;
use crate::render::{
    check_dimensions, FrameParams, RenderOptions, Renderer, TargetError, TextureTarget,
};
use crate::OutputStream;

/// Errors from decoding, validating, rendering, or encoding a trd stream.
///
/// Each layer keeps its own error and is wrapped **transparently**, so a message
/// is identical whether it surfaces here, in `trd-wasm` (which reports
/// [`ProtocolError`] directly) or from the renderer. Only the mesh-first
/// requirement is a stream-specific error.
#[derive(Debug, thiserror::Error)]
pub enum StreamError {
    /// Decoding or validating the input protocol failed.
    #[error(transparent)]
    Protocol(#[from] ProtocolError),
    /// Encoding the rendered image stream failed.
    #[error(transparent)]
    Output(#[from] crate::OutputError),
    /// Rendering failed, including invalid dimensions and render-target
    /// allocation ([`TargetError`] arrives through
    /// [`RenderError::Target`](crate::render::RenderError::Target)).
    #[error(transparent)]
    Render(#[from] crate::render::RenderError),
    /// I/O error reading or writing the stream.
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    /// Assembling a frame's scene failed — today, a draw naming a mesh the
    /// stream never sent.
    #[error(transparent)]
    Scene(#[from] crate::SceneError),
    /// CPU mesh registration or a mesh-addressed renderer edit failed.
    #[error(transparent)]
    MeshResource(#[from] crate::MeshResourceError),
    /// The input is not mesh-first: the protocol requires a leading mesh table
    /// before the params stream (`[mesh][texture?][frames?][params]`). Params-only
    /// streams are no longer accepted.
    #[error("input is missing the required leading mesh table (protocol is mesh-first)")]
    MissingMeshStream,
}

/// [`TargetError`] reaches [`StreamError`] through
/// [`RenderError`](crate::render::RenderError), which already wraps it — this
/// only spares call sites an explicit hop.
impl From<TargetError> for StreamError {
    fn from(error: TargetError) -> Self {
        StreamError::Render(error.into())
    }
}

/// [`FrameError`](crate::FrameError) likewise reaches [`StreamError`] through
/// [`ProtocolError::Frames`], so decoding an inline background needs no
/// hand-written mapping at the call site.
impl From<crate::FrameError> for StreamError {
    fn from(error: crate::FrameError) -> Self {
        StreamError::Protocol(error.into())
    }
}

/// Decodes every row of `batch` into [`FrameParams`]. Delegates to the single
/// shared per-batch decoder [`crate::protocol::decode_batch`] (the source of
/// truth for both the native and wasm paths).
pub fn decode_frames(batch: &RecordBatch) -> Result<Vec<FrameParams>, StreamError> {
    Ok(crate::protocol::decode_batch(batch)?)
}

/// A shell-provided closure that resolves a per-frame background frame reference
/// (a `frame_path`/`frame_url` string) into decoded RGBA pixels. Kept
/// out of `trd-core` so the core performs no file/network I/O: the native CLI
/// supplies one backed by the `image` crate + a `--frames-base` dir; a stream
/// without background frames (or a shell that doesn't load them) passes `None`.
/// Returning `None` for a given reference renders that frame without a
/// background plane (the shell decides how to report the miss).
pub type FrameResolver<'a> = &'a dyn Fn(&str) -> Option<crate::texture::ImageData>;

/// The **external** background reference currently uploaded, so consecutive
/// frames naming the same `frame_path`/`frame_url` skip the resolver + upload.
/// Its inline-`frame_id` counterpart is the shared
/// [`InlineFrameCache`](crate::InlineFrameCache).
#[derive(Default)]
struct FrameBackgroundState {
    last_ref: Option<String>,
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
    output: &mut OutputStream<W>,
    batch: &crate::FrameBatch,
    inline_frames: &[crate::InlineFrame],
    frame_resolver: Option<FrameResolver>,
    background_state: &mut FrameBackgroundState,
    inline_cache: &mut crate::InlineFrameCache,
) -> Result<(), StreamError> {
    let mut planes: Vec<Vec<u8>> = Vec::with_capacity(batch.len());
    for frame in batch {
        let mut frame_fit = None;
        if let Some((image, changed)) = inline_cache.resolve(frame.frame_id, inline_frames)? {
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
        // CPU registrations, not the live GPU mesh count, define wire rows.
        let scene =
            crate::render::Scene::try_from_frame(frame, renderer.mesh_table(), options, frame_fit)?;
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
    // `OutputStream` owns the sink, so encoding *is* writing — no drain + hand
    // -off pair at the call site.
    output.write_rgba_batch(&planes)?;
    Ok(())
}

/// Reads a trd input stream, renders each frame, and writes an Arrow IPC stream
/// of `fixed_shape_tensor` images to `output`. Output batch boundaries mirror
/// input batches (one batch in flight).
///
/// The protocol is `[mesh][texture?][frames?][params]`: the **required**
/// leading mesh table is decoded once (via [`Mesh::from_arrow_all`](crate::Mesh::from_arrow_all)) and
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
    input: R,
    output: W,
    width: u32,
    height: u32,
    options: RenderOptions,
    frame_resolver: Option<FrameResolver>,
) -> Result<(), StreamError> {
    // Validate dimensions up front so schema construction (which multiplies
    // width*height) can't overflow before Renderer's guard runs.
    check_dimensions(width, height)?;

    let mut input = crate::InputStream::new(input);
    // The mesh-first prologue is complete here, so the renderer can be built
    // from it eagerly rather than lazily inside the frame loop. The renderer and
    // its texture target are a matched pair (#203): the target is a call
    // argument, not a field, so both are held here.
    let prologue = input.prologue()?;
    let frame_rate = prologue.frame_rate;
    let (mut renderer, target) = pollster::block_on(Renderer::with_meshes_sample_count(
        width,
        height,
        prologue.meshes,
        options.msaa.sample_count(),
    ))?;
    if let Some(pbr) = &options.pbr {
        renderer.set_appearance(
            crate::MeshTarget::All,
            crate::MeshAppearance {
                material: pbr.material.clone(),
                ibl: pbr.ibl,
                tone_mapping: pbr.tone_mapping,
                ..Default::default()
            },
        )?;
        if let Some(env) = &pbr.env_map {
            renderer.set_env_map(env.clone());
        }
    }
    if let Some(texture) = prologue.texture {
        renderer.set_texture(texture)?;
    }

    // Opening the stream writes its IPC header straight into `output`.
    let mut output = OutputStream::new(output, width, height, Some(frame_rate))?;
    // The background currently uploaded, so consecutive frames sharing it skip
    // the decode + re-upload.
    let mut background_state = FrameBackgroundState::default();
    let mut inline_cache = crate::InlineFrameCache::default();

    // `next_batch` rather than `for batch in &mut input`: the loop body needs
    // `input.frames()` too, which a `for` loop's borrow would forbid.
    while let Some(batch) = input.next_batch() {
        render_and_write_batch(
            &mut renderer,
            &target,
            &options,
            &mut output,
            &batch?,
            input.frames(),
            frame_resolver,
            &mut background_state,
            &mut inline_cache,
        )?;
    }
    input.finish()?;
    output.finish()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use crate::math::Matrix4;
    use crate::protocol::{
        check_version, decode_draws, decode_frame_refs, ProtocolError, PROTOCOL_VERSION,
    };
    use crate::protocol::{
        MESH_TABLE_KIND, PARAMS_TABLE_KIND, PROTOCOL_VERSION_KEY, TABLE_KIND_KEY,
    };
    use crate::render::{Draw, DrawSelection, RenderMode};
    use crate::stream_filter::*;
    use crate::Mesh;
    use crate::MeshTableIndex;
    use arrow::array::{
        Array, ArrayRef, FixedSizeListArray, FixedSizeListArray as U8List, Float32Array, ListArray,
        StringArray, UInt32Array, UInt8Array,
    };
    use arrow::datatypes::Field;
    use arrow::datatypes::{DataType, Schema};
    use arrow::ipc::reader::StreamReader;
    use arrow::ipc::writer::StreamWriter;
    use std::sync::Arc;

    fn build_input_batch(frames: &[FrameParams]) -> RecordBatch {
        // A minimal 0.0.6 params batch carries a single `model` column; every
        // params column is optional, and `model` alone drives the row count.
        let flat: Vec<f32> = frames
            .iter()
            .flat_map(|f| f.model.unwrap_or(IDENTITY_MODEL))
            .collect();
        let schema = Arc::new(
            Schema::new(vec![model_field()]).with_metadata(
                [
                    (
                        PROTOCOL_VERSION_KEY.to_string(),
                        PROTOCOL_VERSION.to_string(),
                    ),
                    (TABLE_KIND_KEY.to_string(), PARAMS_TABLE_KIND.to_string()),
                ]
                .into_iter()
                .collect(),
            ),
        );
        RecordBatch::try_new(schema, vec![list_col(16, flat)]).unwrap()
    }

    /// Column-major identity 4×4, the default `model` for the test helpers.
    const IDENTITY_MODEL: [f32; 16] = [
        1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
    ];

    /// The `model` column `Field` (`FixedSizeList<Float32>[16]`).
    fn model_field() -> Field {
        Field::new(
            "model",
            DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float32, false)), 16),
            false,
        )
    }

    /// A `model` column of `n` identity matrices.
    fn model_col(n: usize) -> ArrayRef {
        list_col(16, (0..n).flat_map(|_| IDENTITY_MODEL).collect())
    }

    #[test]
    fn decodes_frames_roundtrip() {
        let frames = vec![
            FrameParams {
                model: Some(IDENTITY_MODEL),
                ..FrameParams::IDENTITY
            },
            FrameParams {
                model: Some([
                    0.5, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.1, -0.2, 0.0, 1.0,
                ]),
                ..FrameParams::IDENTITY
            },
        ];
        let batch = build_input_batch(&frames);
        let decoded = decode_frames(&batch).unwrap();
        assert_eq!(decoded, frames);
    }

    /// A non-null `FixedSizeList<Float32>[len]` column from flat values.
    fn list_col(len: i32, flat: Vec<f32>) -> ArrayRef {
        Arc::new(FixedSizeListArray::new(
            Arc::new(Field::new("item", DataType::Float32, false)),
            len,
            Arc::new(Float32Array::from(flat)),
            None,
        )) as ArrayRef
    }

    /// Builds a one-row batch of an identity `model` plus the given extra
    /// `(field, column)` pairs.
    fn camera_batch(extra: Vec<(Field, ArrayRef)>) -> RecordBatch {
        let mut fields = vec![model_field()];
        let mut columns: Vec<ArrayRef> = vec![model_col(1)];
        for (field, column) in extra {
            fields.push(field);
            columns.push(column);
        }
        let schema = Arc::new(Schema::new(fields));
        RecordBatch::try_new(schema, columns).unwrap()
    }

    #[test]
    fn decodes_cg_camera_columns() {
        let list3 = |name| {
            Field::new(
                name,
                DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float32, false)), 3),
                false,
            )
        };
        let batch = camera_batch(vec![
            (list3("eye"), list_col(3, vec![1.0, 2.0, 3.0])),
            (list3("target"), list_col(3, vec![0.1, 0.2, 0.3])),
            (
                Field::new("fovy", DataType::Float32, false),
                Arc::new(Float32Array::from(vec![0.9_f32])) as ArrayRef,
            ),
        ]);
        let frames = decode_frames(&batch).unwrap();
        assert_eq!(frames[0].eye, Some([1.0, 2.0, 3.0]));
        assert_eq!(frames[0].target, Some([0.1, 0.2, 0.3]));
        assert_eq!(frames[0].fovy, Some(0.9));
    }

    #[test]
    fn rejects_incomplete_and_conflicting_camera_forms() {
        let list_field = |name, len| {
            Field::new(
                name,
                DataType::FixedSizeList(
                    Arc::new(Field::new("item", DataType::Float32, false)),
                    len,
                ),
                false,
            )
        };
        // `eye` alone is incomplete.
        let incomplete = camera_batch(vec![(
            list_field("eye", 3),
            list_col(3, vec![1.0, 2.0, 3.0]),
        )]);
        assert!(matches!(
            decode_frames(&incomplete),
            Err(StreamError::Protocol(ProtocolError::IncompleteCameraForm))
        ));
        // CV `k` mixed with CG `eye` is conflicting.
        let conflicting = camera_batch(vec![
            (list_field("k", 9), list_col(9, vec![1.0; 9])),
            (list_field("eye", 3), list_col(3, vec![1.0, 2.0, 3.0])),
        ]);
        assert!(matches!(
            decode_frames(&conflicting),
            Err(StreamError::Protocol(ProtocolError::ConflictingCameraForms))
        ));
    }

    use arrow::buffer::OffsetBuffer;

    /// A `List<UInt32>` column with the given per-row id lists.
    fn draw_mesh_col(rows: &[Vec<u32>]) -> ArrayRef {
        let field = Arc::new(Field::new("item", DataType::UInt32, false));
        let flat: Vec<u32> = rows.iter().flatten().copied().collect();
        let offsets = OffsetBuffer::from_lengths(rows.iter().map(Vec::len));
        Arc::new(ListArray::new(
            field,
            offsets,
            Arc::new(UInt32Array::from(flat)),
            None,
        )) as ArrayRef
    }

    /// A `List<FixedSizeList<Float32>[16]>` column with the given per-row model
    /// lists (each model is 16 flat column-major floats).
    fn draw_model_col(rows: &[Vec<[f32; 16]>]) -> ArrayRef {
        let item = Arc::new(Field::new("item", DataType::Float32, false));
        let flat: Vec<f32> = rows.iter().flatten().flatten().copied().collect();
        let fsl = FixedSizeListArray::new(item, 16, Arc::new(Float32Array::from(flat)), None);
        let field = Arc::new(Field::new(
            "item",
            DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float32, false)), 16),
            false,
        ));
        let offsets = OffsetBuffer::from_lengths(rows.iter().map(Vec::len));
        Arc::new(ListArray::new(field, offsets, Arc::new(fsl), None)) as ArrayRef
    }

    /// The `Field` for a `draw_mesh` / `draw_model` column.
    fn draw_field(name: &str, item: DataType) -> Field {
        Field::new(
            name,
            DataType::List(Arc::new(Field::new("item", item, false))),
            false,
        )
    }

    fn draw_batch(mesh_rows: &[Vec<u32>], model_rows: &[Vec<[f32; 16]>]) -> RecordBatch {
        let n = mesh_rows.len();
        let schema = Arc::new(Schema::new(vec![
            model_field(),
            draw_field("draw_mesh", DataType::UInt32),
            draw_field(
                "draw_model",
                DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float32, false)), 16),
            ),
        ]));
        RecordBatch::try_new(
            schema,
            vec![
                model_col(n),
                draw_mesh_col(mesh_rows),
                draw_model_col(model_rows),
            ],
        )
        .unwrap()
    }

    #[test]
    fn decode_draws_absent_returns_none() {
        let batch = build_input_batch(&[FrameParams::IDENTITY]);
        assert!(decode_draws(&batch).unwrap().is_none());
    }

    /// A `Utf8` column of `frame_path`/`frame_url` references from optional strings.
    fn frame_ref_batch(name: &str, refs: &[Option<&str>]) -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![Field::new(name, DataType::Utf8, true)]));
        let col = StringArray::from(refs.to_vec());
        RecordBatch::try_new(schema, vec![Arc::new(col) as ArrayRef]).unwrap()
    }

    #[test]
    fn decode_frame_refs_absent_returns_none() {
        // A stream with no background-frame column decodes to `None` (soft skip).
        let batch = build_input_batch(&[FrameParams::IDENTITY]);
        assert!(decode_frame_refs(&batch).unwrap().is_none());
    }

    #[test]
    fn decode_frame_refs_reads_paths_nulls_and_empty() {
        // Native prefers `frame_path`; per-row null or empty ⇒ `None` (no
        // background for that frame), a non-empty string ⇒ the reference.
        let batch = frame_ref_batch(
            "frame_path",
            &[Some("frames/frame_000000.png"), None, Some("")],
        );
        let refs = decode_frame_refs(&batch).unwrap().unwrap();
        assert_eq!(
            refs,
            vec![Some("frames/frame_000000.png".to_owned()), None, None]
        );
    }

    #[test]
    fn decode_frame_refs_falls_back_to_frame_url() {
        // With no `frame_path`, the `frame_url` column (browser) is used instead.
        let batch = frame_ref_batch("frame_url", &[Some("https://host/a.png"), None]);
        let refs = decode_frame_refs(&batch).unwrap().unwrap();
        assert_eq!(refs, vec![Some("https://host/a.png".to_owned()), None]);
    }

    #[test]
    fn decode_frame_refs_prefers_frame_path_over_url() {
        // Both columns present ⇒ native path wins.
        let schema = Arc::new(Schema::new(vec![
            Field::new("frame_path", DataType::Utf8, true),
            Field::new("frame_url", DataType::Utf8, true),
        ]));
        let path = StringArray::from(vec![Some("local/a.png")]);
        let url = StringArray::from(vec![Some("https://host/a.png")]);
        let batch = RecordBatch::try_new(
            schema,
            vec![Arc::new(path) as ArrayRef, Arc::new(url) as ArrayRef],
        )
        .unwrap();
        let refs = decode_frame_refs(&batch).unwrap().unwrap();
        assert_eq!(refs, vec![Some("local/a.png".to_owned())]);
    }

    #[test]
    fn decodes_variable_length_draw_lists() {
        let a = [
            1.0f32, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        ];
        let b = [
            2.0f32, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 5.0, 6.0, 7.0, 1.0,
        ];
        // Row 0 draws two instances (meshes 0 and 1); row 1 draws one (mesh 1).
        let batch = draw_batch(&[vec![0, 1], vec![1]], &[vec![a, b], vec![b]]);
        let rows = decode_draws(&batch).unwrap().unwrap();
        assert_eq!(rows.len(), 2);
        assert_eq!(
            rows[0],
            vec![
                Draw {
                    mesh_id: MeshTableIndex::new(0),
                    model: Matrix4::from_cols_array(&a),
                    selection: DrawSelection::INHERIT
                },
                Draw {
                    mesh_id: MeshTableIndex::new(1),
                    model: Matrix4::from_cols_array(&b),
                    selection: DrawSelection::INHERIT
                },
            ]
        );
        assert_eq!(
            rows[1],
            vec![Draw {
                mesh_id: MeshTableIndex::new(1),
                model: Matrix4::from_cols_array(&b),
                selection: DrawSelection::INHERIT
            }]
        );
    }

    #[test]
    fn rejects_mismatched_draw_lists() {
        let m = [0.0f32; 16];
        // Row 0: two mesh ids but only one model.
        let batch = draw_batch(&[vec![0, 1]], &[vec![m]]);
        assert!(matches!(
            decode_draws(&batch),
            Err(ProtocolError::MismatchedDrawLists {
                row: 0,
                mesh_len: 2,
                model_len: 1,
            })
        ));
    }

    // Build a `[draw_mesh, draw_model, draw_mode]` batch (`draw_mode` optional).
    fn draw_batch_with_modes(
        mesh_rows: &[Vec<u32>],
        model_rows: &[Vec<[f32; 16]>],
        mode_rows: Option<&[Vec<u8>]>,
    ) -> RecordBatch {
        let n = mesh_rows.len();
        let mut fields = vec![
            model_field(),
            draw_field("draw_mesh", DataType::UInt32),
            draw_field(
                "draw_model",
                DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float32, false)), 16),
            ),
        ];
        let mut cols: Vec<ArrayRef> = vec![
            model_col(n),
            draw_mesh_col(mesh_rows),
            draw_model_col(model_rows),
        ];
        if let Some(mode_rows) = mode_rows {
            fields.push(draw_field("draw_mode", DataType::UInt8));
            let flat: Vec<u8> = mode_rows.iter().flatten().copied().collect();
            let offsets = OffsetBuffer::from_lengths(mode_rows.iter().map(Vec::len));
            cols.push(Arc::new(ListArray::new(
                Arc::new(Field::new("item", DataType::UInt8, false)),
                offsets,
                Arc::new(UInt8Array::from(flat)),
                None,
            )) as ArrayRef);
        }
        RecordBatch::try_new(Arc::new(Schema::new(fields)), cols).unwrap()
    }

    #[test]
    fn decodes_per_draw_render_modes() {
        let m = [0.0f32; 16];
        // Row 0 mixes a global-inheriting draw (255) with an explicit wireframe
        // (1); row 1's textured (2) and filled (0). Absent column ⇒ all None.
        let batch = draw_batch_with_modes(
            &[vec![0, 1], vec![2]],
            &[vec![m, m], vec![m]],
            Some(&[vec![255, 1], vec![2]]),
        );
        let rows = decode_draws(&batch).unwrap().unwrap();
        assert_eq!(rows[0][0].selection, DrawSelection::INHERIT);
        assert_eq!(
            rows[0][1].selection,
            DrawSelection::Mesh(Some(RenderMode::Wireframe))
        );
        assert_eq!(
            rows[1][0].selection,
            DrawSelection::Mesh(Some(RenderMode::Textured))
        );

        // Absent `draw_mode` column ⇒ every draw inherits (None).
        let plain = draw_batch_with_modes(&[vec![0, 1]], &[vec![m, m]], None);
        let plain_rows = decode_draws(&plain).unwrap().unwrap();
        assert!(plain_rows[0]
            .iter()
            .all(|d| d.selection == DrawSelection::INHERIT));
    }

    #[test]
    fn rejects_invalid_and_mismatched_draw_modes() {
        let m = [0.0f32; 16];
        // A byte outside {0,1,2,255} is rejected.
        let bad = draw_batch_with_modes(&[vec![0]], &[vec![m]], Some(&[vec![7]]));
        assert!(matches!(
            decode_draws(&bad),
            Err(ProtocolError::InvalidDrawMode { value: 7 })
        ));
        // A `draw_mode` list shorter than the draw list is rejected.
        let short = draw_batch_with_modes(&[vec![0, 1]], &[vec![m, m]], Some(&[vec![0]]));
        assert!(matches!(
            decode_draws(&short),
            Err(ProtocolError::MismatchedDrawModes {
                row: 0,
                mode_len: 1,
                draw_len: 2,
            })
        ));
    }

    #[test]
    fn draw_columns_must_come_as_a_pair() {
        let batch = build_input_batch(&[FrameParams::IDENTITY]);
        let schema = Arc::new(Schema::new(vec![
            batch.schema().field(0).clone(),
            draw_field("draw_mesh", DataType::UInt32),
        ]));
        let with_mesh_only = RecordBatch::try_new(
            schema,
            vec![batch.column(0).clone(), draw_mesh_col(&[vec![0]])],
        )
        .unwrap();
        assert!(matches!(
            decode_draws(&with_mesh_only),
            Err(ProtocolError::MissingColumn("draw_model"))
        ));
    }

    #[test]
    fn child_null_in_camera_list_is_error() {
        // A non-null camera-list row whose child float is null must be rejected.
        let item = Arc::new(Field::new("item", DataType::Float32, true));
        let eye_values = Float32Array::from(vec![Some(0.0), Some(0.0), None]);
        let eye = FixedSizeListArray::new(item, 3, Arc::new(eye_values), None);
        let schema = Arc::new(Schema::new(vec![Field::new(
            "eye",
            eye.data_type().clone(),
            false,
        )]));
        let batch = RecordBatch::try_new(schema, vec![Arc::new(eye) as ArrayRef]).unwrap();
        assert!(matches!(
            decode_frames(&batch),
            Err(StreamError::Protocol(ProtocolError::NullValues("eye")))
        ));
    }

    #[test]
    fn wrong_type_is_error() {
        use arrow::array::Int32Array;
        // A camera scalar column of the wrong Arrow type must be rejected.
        let schema = Arc::new(Schema::new(vec![Field::new(
            "fovy",
            DataType::Int32,
            false,
        )]));
        let fovy = Int32Array::from(vec![3]);
        let batch = RecordBatch::try_new(schema, vec![Arc::new(fovy) as ArrayRef]).unwrap();
        assert!(matches!(
            decode_frames(&batch),
            Err(StreamError::Protocol(ProtocolError::ColumnType {
                column: "fovy",
                ..
            }))
        ));
    }

    #[test]
    fn version_check_rejects_mismatch() {
        let schema = Schema::empty().with_metadata(
            [(PROTOCOL_VERSION_KEY.to_string(), "9.9.9".to_string())]
                .into_iter()
                .collect(),
        );
        assert!(matches!(
            check_version(&schema),
            Err(ProtocolError::UnsupportedVersion(v)) if v == "9.9.9"
        ));
    }

    #[test]
    fn version_check_rejects_absent_and_allows_matching() {
        assert!(matches!(
            check_version(&Schema::empty()),
            Err(ProtocolError::MissingMetadata(key)) if key == PROTOCOL_VERSION_KEY
        ));
        let versioned = Schema::empty().with_metadata(
            [(
                PROTOCOL_VERSION_KEY.to_string(),
                PROTOCOL_VERSION.to_string(),
            )]
            .into_iter()
            .collect(),
        );
        assert!(check_version(&versioned).is_ok());
    }

    #[test]
    fn check_dimensions_rejects_zero_and_overflow() {
        use crate::render::RenderError;
        assert!(check_dimensions(4, 3).is_ok());
        assert!(matches!(
            check_dimensions(0, 3),
            Err(RenderError::InvalidDimensions { .. })
        ));
        // width*height overflows u32 / exceeds i32::MAX.
        assert!(matches!(
            check_dimensions(100_000, 100_000),
            Err(RenderError::InvalidDimensions { .. })
        ));
        // ...and still surfaces as the stream's own error for CLI callers, wrapped
        // transparently rather than re-declared.
        assert!(matches!(
            StreamError::from(check_dimensions(0, 3).unwrap_err()),
            StreamError::Render(RenderError::InvalidDimensions { .. })
        ));
    }

    // ---- two-stream [mesh][params] framing ----

    /// Serializes a mesh as a one-row Arrow IPC **mesh stream** (nested list
    /// columns: `position`/`color` `List<FixedSizeList<Float32>[3]>`, `index`
    /// `List<UInt32>`), tagged with the 0.0.3 protocol version.
    fn write_mesh_stream(buf: &mut Vec<u8>, mesh: &Mesh) {
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
            let offsets = OffsetBuffer::from_lengths([fsl.len()]);
            Arc::new(ListArray::new(field, offsets, Arc::new(fsl), None))
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
        let mut wr = StreamWriter::try_new(buf, &schema).unwrap();
        wr.write(&batch).unwrap();
        wr.finish().unwrap();
    }

    /// Serializes frames as an Arrow IPC **params stream**.
    fn write_params_stream(buf: &mut Vec<u8>, frames: &[FrameParams]) {
        let batch = build_input_batch(frames);
        let mut wr = StreamWriter::try_new(buf, batch.schema().as_ref()).unwrap();
        wr.write(&batch).unwrap();
        wr.finish().unwrap();
    }

    #[test]
    fn two_stream_mesh_then_params_split_and_decode() {
        // Build a concatenated [mesh][params] byte stream in memory.
        let mesh = Mesh::hello_triangle();
        let frames = vec![
            FrameParams {
                model: Some(IDENTITY_MODEL),
                ..FrameParams::IDENTITY
            },
            FrameParams {
                model: Some([
                    1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.3, 0.0, 0.0, 1.0,
                ]),
                ..FrameParams::IDENTITY
            },
        ];
        let mut bytes = Vec::new();
        write_mesh_stream(&mut bytes, &mesh);
        write_params_stream(&mut bytes, &frames);

        // The single shared `InputSession` framing driver must recover the mesh,
        // then the params that follow it in the same byte stream (the mesh
        // sub-stream boundary must not swallow the params).
        let mut session = crate::InputSession::new();
        let mut decoded = Vec::new();
        for batch in session.push(&bytes).unwrap() {
            for frame in batch {
                decoded.push(frame.params);
            }
        }
        session.finish().unwrap();
        assert_eq!(session.meshes(), &[mesh]);
        assert_eq!(decoded, frames);
    }

    #[test]
    #[ignore = "requires a GPU adapter"]
    fn run_stream_renders_mesh_first_stream() {
        let (w, h) = (32u32, 32u32);
        // A full-screen quad as the leading mesh; two params frames follow.
        let mesh =
            Mesh::from_obj("v -1 -1 0\nv 1 -1 0\nv 1 1 0\nv -1 1 0\nf 1 2 3\nf 1 3 4\n").unwrap();
        let frames = vec![
            FrameParams::IDENTITY,
            FrameParams {
                model: Some([
                    1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.2, 0.0, 0.0, 1.0,
                ]),
                ..FrameParams::IDENTITY
            },
        ];
        let mut input_bytes = Vec::new();
        write_mesh_stream(&mut input_bytes, &mesh);
        write_params_stream(&mut input_bytes, &frames);

        let mut output_bytes = Vec::new();
        run_stream(
            &input_bytes[..],
            &mut output_bytes,
            w,
            h,
            RenderOptions::default(),
            None,
        )
        .unwrap();

        let reader = StreamReader::try_new(&output_bytes[..], None).unwrap();
        let batches = reader.collect::<Result<Vec<_>, _>>().unwrap();
        let total_rows: usize = batches.iter().map(|b| b.num_rows()).sum();
        assert_eq!(total_rows, frames.len());

        // The white quad covers the frame, so the center pixel must be lit.
        let get = |batch: &RecordBatch, name: &str| -> U8List {
            batch
                .column_by_name(name)
                .unwrap()
                .as_any()
                .downcast_ref::<U8List>()
                .unwrap()
                .clone()
        };
        let r = get(&batches[0], "r");
        let center = (h as usize / 2) * w as usize + w as usize / 2;
        let value = r
            .value(0)
            .as_any()
            .downcast_ref::<UInt8Array>()
            .unwrap()
            .value(center);
        assert!(value > 0, "mesh quad should cover the center pixel");
    }

    /// A reader that hands out at most `chunk` bytes per `read`, so a test can force
    /// the transport to need several reads per decoded batch.
    struct Trickle<'a> {
        bytes: &'a [u8],
        chunk: usize,
    }

    impl std::io::Read for Trickle<'_> {
        fn read(&mut self, out: &mut [u8]) -> std::io::Result<usize> {
            let n = self.bytes.len().min(self.chunk).min(out.len());
            out[..n].copy_from_slice(&self.bytes[..n]);
            self.bytes = &self.bytes[n..];
            Ok(n)
        }
    }

    #[test]
    fn input_stream_yields_the_same_frames_however_the_bytes_arrive() {
        let mesh = Mesh::hello_triangle();
        let frames = vec![
            FrameParams {
                model: Some(IDENTITY_MODEL),
                ..FrameParams::IDENTITY
            },
            FrameParams {
                model: Some([
                    1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.3, 0.0, 0.0, 1.0,
                ]),
                ..FrameParams::IDENTITY
            },
        ];
        let mut bytes = Vec::new();
        write_mesh_stream(&mut bytes, &mesh);
        write_params_stream(&mut bytes, &frames);

        // One byte at a time is the worst case: most reads decode nothing, so the
        // read loop must keep pumping rather than report end of stream.
        for chunk in [1, 7, 4096, usize::MAX] {
            let mut stream = crate::InputStream::new(Trickle {
                bytes: &bytes,
                chunk,
            });
            let prologue = stream.prologue().expect("prologue");
            assert_eq!(prologue.meshes, std::slice::from_ref(&mesh));
            assert!(prologue.texture.is_none());

            let decoded: Vec<FrameParams> = stream
                .by_ref()
                .flat_map(|batch| batch.expect("batch"))
                .map(|frame| frame.params)
                .collect();
            assert_eq!(decoded, frames, "chunk size {chunk}");
            stream.finish().expect("finish");
        }
    }

    #[test]
    fn input_stream_rejects_a_stream_that_is_not_mesh_first() {
        // Params with no leading mesh table...
        let mut params_only = Vec::new();
        write_params_stream(&mut params_only, &[FrameParams::IDENTITY]);
        assert!(matches!(
            crate::InputStream::new(&params_only[..]).prologue(),
            Err(StreamError::MissingMeshStream)
        ));
        // ...and a stream that ends before any schema arrives, which the session's
        // own end-of-stream check catches first (unchanged from the pre-refactor
        // path, where `session.finish()` also ran before the mesh-first check).
        assert!(matches!(
            crate::InputStream::new(&[][..]).prologue(),
            Err(StreamError::Protocol(ProtocolError::MissingSchema))
        ));
    }
}
