//! Native Arrow scene-document rendering.
//!
//! Input is `[params][mesh?]`: complete Arrow IPC streams decoded through
//! [`crate::SceneDocument`], with optional UUID-keyed original GLB resources.
//! External background references are resolved by the shell. Explicit object
//! transforms render directly; quad placement requires a caller-supplied
//! [`SceneBuilder`] so placement policy stays outside the core.
//!
//! Output is one row per frame, four `fixed_shape_tensor<u8>` channels
//! `r,g,b,a` of shape `[H, W]`, preserving the params batch boundaries.

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
/// [`ProtocolError`] directly) or from the renderer. The legacy mesh-first
/// transport helpers still use their existing error variants during migration.
#[derive(Debug, thiserror::Error)]
pub enum StreamError {
    #[error("scene assembly failed: {0}")]
    SceneBuild(String),
    #[error("background frame `{0}` could not be resolved")]
    FrameResolve(String),
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
    /// Legacy `InputStream` diagnostic. The document-based renderer accepts
    /// params-only input and does not emit this mesh-first transport error.
    #[error("input is missing the required leading mesh table (protocol is mesh-first)")]
    MissingMeshStream,
    #[error("mesh row {index} reference `{reference}` has no resolver")]
    UnresolvedMeshReference { index: u32, reference: String },
    #[error("mesh row {index} reference `{reference}` failed to load: {message}")]
    MeshResolve {
        index: u32,
        reference: String,
        message: String,
    },
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
/// Returning `None` for a referenced image is a [`StreamError::FrameResolve`],
/// not a silent omission of the background.
pub type FrameResolver<'a> = &'a dyn Fn(&str) -> Option<crate::texture::ImageData>;
pub type MeshResolver<'a> = &'a dyn Fn(&crate::MeshReference) -> Result<Vec<u8>, String>;

/// The caller owns placement policy; core renders the resulting domain values.
pub type SceneBuilder = fn(
    &crate::SceneDocument,
    &crate::DocumentFrame,
    crate::Viewport,
    &RenderOptions,
    Option<FrameFit>,
) -> Result<(crate::Camera, crate::Scene), String>;

/// The **external** background reference currently uploaded, so consecutive
/// frames naming the same `frame_path`/`frame_url` skip the resolver + upload.
#[derive(Default)]
struct FrameBackgroundState {
    last_ref: Option<String>,
}

/// Renders a retained params row range and writes its output batch, mirroring
/// one Arrow output batch per input record batch. When
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
    document: &crate::SceneDocument,
    rows: std::ops::Range<usize>,
    frame_resolver: Option<FrameResolver>,
    background_state: &mut FrameBackgroundState,
    build_scene: SceneBuilder,
) -> Result<(), StreamError> {
    let mut planes: Vec<Vec<u8>> = Vec::with_capacity(rows.len());
    for row in rows {
        let frame = document.frame(row)?;
        let mut frame_fit = None;
        if let Some(path) = document.frame_ref(row)? {
            if background_state.last_ref.as_deref() != Some(&path) {
                let image = frame_resolver
                    .and_then(|resolve| resolve(&path))
                    .ok_or_else(|| StreamError::FrameResolve(path.clone()))?;
                renderer.update_frame_texture(&image);
                background_state.last_ref = Some(path);
            }
            frame_fit = Some(FrameFit::Stretch);
        } else {
            background_state.last_ref = None;
        }
        // The scene is assembled here, from the wire draw list plus the CLI's
        // appearance options — the same `scene_with_overlays` every other
        // front-end uses, so they cannot drift apart (#180).
        let (camera, scene) = build_scene(document, &frame, target.viewport(), options, frame_fit)
            .map_err(StreamError::SceneBuild)?;
        // `run_stream` is a synchronous `Read`/`Write` filter, while the renderer
        // is async because GPU read-back is (the browser must not block its event
        // loop). Natively blocking here is free: the future is already complete
        // when `poll_for_map` returns. This is the only bridge between the two.
        renderer.draw_layers(&[crate::SceneLayer::new(camera, &scene)], target);
        planes.push(pollster::block_on(renderer.read_pixels(target))?);
    }
    // `OutputStream` owns the sink, so encoding *is* writing — no drain + hand
    // -off pair at the call site.
    output.write_rgba_batch(&planes)?;
    Ok(())
}

/// Reads a trd input stream, renders each frame, and writes an Arrow IPC stream
/// of `fixed_shape_tensor` images to `output`. Output batch boundaries mirror
/// input batches (one output batch in flight).
///
/// [`crate::SceneDocument::read_from`] reads `[params][mesh?]` through the same
/// document decoder as the browser's complete-buffer API. Original GLBs are
/// decoded once before rendering. Params-only input produces reference
/// geometry; quad-bound input uses [`run_stream_with_scene_builder`].
pub fn run_stream<R: Read, W: Write>(
    input: R,
    output: W,
    width: u32,
    height: u32,
    options: RenderOptions,
    frame_resolver: Option<FrameResolver>,
) -> Result<(), StreamError> {
    run_stream_with_mesh_resolver(input, output, width, height, options, frame_resolver, None)
}

pub fn run_stream_with_mesh_resolver<R: Read, W: Write>(
    input: R,
    output: W,
    width: u32,
    height: u32,
    options: RenderOptions,
    frame_resolver: Option<FrameResolver>,
    _mesh_resolver: Option<MeshResolver>,
) -> Result<(), StreamError> {
    run_stream_with_scene_builder(
        input,
        output,
        width,
        height,
        options,
        frame_resolver,
        explicit_scene,
    )
}

pub fn run_stream_with_scene_builder<R: Read, W: Write>(
    input: R,
    output: W,
    width: u32,
    height: u32,
    mut options: RenderOptions,
    frame_resolver: Option<FrameResolver>,
    build_scene: SceneBuilder,
) -> Result<(), StreamError> {
    // Validate dimensions up front so schema construction (which multiplies
    // width*height) can't overflow before Renderer's guard runs.
    check_dimensions(width, height)?;

    let document = crate::SceneDocument::read_from(input)?;
    // The document's trailing resources are complete here. The renderer and
    // its texture target are a matched pair (#203): the target is a call
    // argument, not a field, so both are held here.
    let assets = document.decoded_assets()?;
    let frame_rate = crate::frame_rate_from_metadata(document.schema().metadata());
    let instance = crate::create_instance();
    let gpu = pollster::block_on(crate::GpuContext::request(
        &instance,
        &crate::GpuRequest {
            limits: crate::render::LimitsPreset::Downlevel,
            memory_hints: wgpu::MemoryHints::MemoryUsage,
            ..Default::default()
        },
    ))
    .map_err(|error| crate::RenderError::Gpu(error.to_string()))?;
    let mut renderer = Renderer::with_assets_sample_count(
        gpu,
        crate::TEXTURE_TARGET_FORMAT,
        &assets,
        options.msaa.sample_count(),
    )?;
    let target = renderer.create_texture_target(width, height)?;
    if document.meshes().is_empty() {
        renderer.set_mesh_aabb_color(0, crate::Mesh::REFERENCE_CUBE_COLOR)?;
    }
    if let Some(pbr) = &options.pbr {
        if options.mode == crate::RenderMode::Shaded {
            renderer.set_appearance(
                crate::MeshTarget::All,
                crate::MeshAppearance {
                    material: pbr.material.clone(),
                    ibl: pbr.ibl,
                    tone_mapping: pbr.tone_mapping,
                    ..Default::default()
                },
            );
        }
        if let Some(env) = &pbr.env_map {
            renderer.set_env_map(env.clone());
        }
    }
    if let Some(operator) = document.tonemap_override()? {
        renderer.set_tonemap_operator(crate::MeshTarget::All, operator);
        if let Some(background) = options.env_background.as_mut() {
            background.tonemap = operator;
        }
    }
    // Opening the stream writes its IPC header straight into `output`.
    let mut output = OutputStream::new(output, width, height, Some(frame_rate))?;
    // The background currently uploaded, so consecutive frames sharing it skip
    // the decode + re-upload.
    let mut background_state = FrameBackgroundState::default();

    let mut row = 0;
    for batch in document.batches() {
        let end = row + batch.num_rows();
        render_and_write_batch(
            &mut renderer,
            &target,
            &options,
            &mut output,
            &document,
            row..end,
            frame_resolver,
            &mut background_state,
            build_scene,
        )?;
        row = end;
    }
    output.finish()?;
    Ok(())
}

fn explicit_scene(
    document: &crate::SceneDocument,
    frame: &crate::DocumentFrame,
    viewport: crate::Viewport,
    options: &RenderOptions,
    fit: Option<FrameFit>,
) -> Result<(crate::Camera, crate::Scene), String> {
    if frame.objects.iter().any(|object| object.quad.is_some()) {
        return Err(
            "quad placement requires a caller-provided scene builder from trd-placement".to_owned(),
        );
    }
    let camera = frame
        .params
        .to_camera(viewport)
        .map_err(|error| error.to_string())?;
    let mut draws = Vec::new();
    if !document.meshes().is_empty() {
        for (index, object) in frame.objects.iter().enumerate() {
            draws.push(crate::Draw {
                mesh_id: u32::try_from(
                    document
                        .object_mesh_slot(frame, index)
                        .map_err(|error| error.to_string())?,
                )
                .map_err(|_| "too many meshes".to_owned())?,
                model: object.model,
                selection: object.selection,
            });
        }
    }
    let mut scene = crate::Scene::from_draws(&draws, options, fit);
    if document.meshes().is_empty() && !frame.objects.is_empty() {
        scene.push(crate::DrawableObject::aabb_box(0, crate::Matrix4::IDENTITY));
        scene.push(crate::DrawableObject::coordinate_axes(
            crate::Matrix4::IDENTITY,
        ));
    }
    Ok((camera, scene))
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
                    mesh_id: 0,
                    model: Matrix4::from_cols_array(&a),
                    selection: DrawSelection::INHERIT
                },
                Draw {
                    mesh_id: 1,
                    model: Matrix4::from_cols_array(&b),
                    selection: DrawSelection::INHERIT
                },
            ]
        );
        assert_eq!(
            rows[1],
            vec![Draw {
                mesh_id: 1,
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
    fn run_stream_rejects_mesh_first_documents() {
        let mut input_bytes = Vec::new();
        write_mesh_stream(&mut input_bytes, &Mesh::hello_triangle());
        write_params_stream(&mut input_bytes, &[FrameParams::IDENTITY]);
        let error = run_stream(
            &input_bytes[..],
            &mut Vec::new(),
            32,
            32,
            RenderOptions::default(),
            None,
        )
        .unwrap_err();
        assert!(matches!(
            error,
            StreamError::Protocol(ProtocolError::Arrow(arrow::error::ArrowError::ParseError(message)))
                if message == "expected a params table"
        ));
    }

    #[test]
    #[ignore = "requires a GPU adapter"]
    fn run_stream_renders_params_first_glb_document() {
        let (w, h) = (32u32, 32u32);
        let frames = vec![
            FrameParams::IDENTITY,
            FrameParams {
                model: Some([
                    1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.2, 0.0, 0.0, 1.0,
                ]),
                ..FrameParams::IDENTITY
            },
        ];
        let batch = build_input_batch(&frames);
        let document = crate::SceneDocument::from_batches(
            batch.schema(),
            vec![batch],
            vec![crate::GlbMesh::new(
                "00000000-0000-0000-0000-000000000001",
                &crate::protocol::triangle_glb(),
            )
            .unwrap()],
        )
        .unwrap();
        let input_bytes = document.write().unwrap();
        assert!(crate::SceneDocument::starts_with_params(&input_bytes).unwrap());

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
        assert_eq!(batches.len(), 1, "preserve the params batch boundary");
        assert_eq!(total_rows, frames.len());

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
        assert!(value > 0, "the GLB triangle should cover the center pixel");
        assert_ne!(
            r.value(0).to_data(),
            r.value(1).to_data(),
            "the second params row must visibly translate the GLB"
        );
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
            assert!(prologue.mesh_assets[0].base_color_texture.is_none());

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
