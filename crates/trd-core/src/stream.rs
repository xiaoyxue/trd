//! Native-only Arrow streaming protocol (trd protocol 0.0.5).
//!
//! Input is one to three concatenated Arrow IPC streams on stdin. A `0.0.4`
//! stream is `[mesh][texture?][params]`: a leading **mesh** table (one row = one
//! mesh, all rows decoded by [`Mesh::from_arrow_all`]), an optional **texture**
//! table (one row = one `fixed_shape_tensor<u8>[H,W,4]` image, decoded by
//! [`ImageTexture::from_arrow`] and bound as the sampled albedo), then the
//! **params** stream (one row per frame: `center`, `size`, `theta`, + optional
//! camera columns `model`/`k`/`pose`/`eye`/`target`/`direction`/`up`/`fovy`/
//! `aspect`/`znear`/`zfar`, + an optional per-frame instanced draw list
//! `draw_mesh` (`List<UInt32>`) / `draw_model`
//! (`List<FixedSizeList<Float32>[16]>`) placing instances of the loaded meshes).
//! When the draw list is absent, one instance of mesh 0 is placed by the frame's
//! own model. A legacy `0.0.1`/`0.0.2` stream is just the params stream and
//! renders the built-in hello-triangle. Output: one row per frame, four
//! `fixed_shape_tensor<u8>` channels `r,g,b,a` of shape `[H, W]`.

use std::sync::Arc;

use arrow::array::{
    Array, FixedSizeListArray, Float32Array, ListArray, RecordBatch, StringArray, UInt32Array,
    UInt8Array,
};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::error::ArrowError;
use arrow::ipc::reader::StreamReader;
use std::io::{Read, Write};

use crate::math::Matrix4;
use crate::protocol::{
    frame_rate_from_metadata, is_mesh_schema, is_texture_schema, PROTOCOL_VERSION,
    PROTOCOL_VERSION_KEY, SUPPORTED_INPUT_VERSIONS,
};
use crate::render::{
    CameraFormError, Draw, DrawableObject, FrameFit, FrameParams, Mesh, MeshRenderer, RenderMode,
};
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
    /// GPU rendering failed.
    #[error("render error: {0}")]
    Render(String),
    /// The leading mesh table could not be decoded into a [`Mesh`].
    #[error("mesh decode error: {0}")]
    Mesh(#[from] crate::MeshError),
    /// The optional leading texture table could not be decoded.
    #[error("texture decode error: {0}")]
    Texture(#[from] crate::TextureError),
    #[error(transparent)]
    Output(#[from] crate::OutputError),
}

fn vec2_field(name: &str) -> Field {
    Field::new(
        name,
        DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float32, false)), 2),
        false,
    )
}

/// Builds the protocol 0.0.1 input schema (with version metadata).
pub fn input_schema() -> Schema {
    Schema::new(vec![
        vec2_field("center"),
        vec2_field("size"),
        Field::new("theta", DataType::Float32, false),
    ])
    .with_metadata(
        [(
            PROTOCOL_VERSION_KEY.to_string(),
            PROTOCOL_VERSION.to_string(),
        )]
        .into_iter()
        .collect(),
    )
}

/// If the schema declares a protocol version, require it to be supported.
pub fn check_version(schema: &Schema) -> Result<(), StreamError> {
    if let Some(v) = schema.metadata().get(PROTOCOL_VERSION_KEY) {
        if !SUPPORTED_INPUT_VERSIONS.contains(&v.as_str()) {
            return Err(StreamError::UnsupportedVersion(v.clone()));
        }
    }
    Ok(())
}

fn require_vec2<'a>(
    batch: &'a RecordBatch,
    name: &'static str,
) -> Result<&'a FixedSizeListArray, StreamError> {
    let col = batch
        .column_by_name(name)
        .ok_or(StreamError::MissingColumn(name))?;
    match col.data_type() {
        DataType::FixedSizeList(f, 2) if *f.data_type() == DataType::Float32 => Ok(col
            .as_any()
            .downcast_ref::<FixedSizeListArray>()
            .expect("checked FixedSizeList type")),
        other => Err(StreamError::ColumnType {
            column: name,
            expected: "FixedSizeList<Float32>[2]",
            actual: other.clone(),
        }),
    }
}

fn require_f32<'a>(
    batch: &'a RecordBatch,
    name: &'static str,
) -> Result<&'a Float32Array, StreamError> {
    let col = batch
        .column_by_name(name)
        .ok_or(StreamError::MissingColumn(name))?;
    col.as_any()
        .downcast_ref::<Float32Array>()
        .ok_or_else(|| StreamError::ColumnType {
            column: name,
            expected: "Float32",
            actual: col.data_type().clone(),
        })
}

fn read_vec2(list: &FixedSizeListArray, row: usize) -> [f32; 2] {
    let values = list
        .values()
        .as_any()
        .downcast_ref::<Float32Array>()
        .expect("vec2 values are Float32");
    [values.value(row * 2), values.value(row * 2 + 1)]
}

/// Looks up an optional `FixedSizeList<Float32>[len]` column, validating type,
/// length, and non-nullness. Returns `None` if the column is absent (additive
/// `0.0.2` columns are optional).
fn optional_fixed_list<'a>(
    batch: &'a RecordBatch,
    name: &'static str,
    len: i32,
) -> Result<Option<(&'a FixedSizeListArray, &'a Float32Array)>, StreamError> {
    let Some(column) = batch.column_by_name(name) else {
        return Ok(None);
    };
    let list = column
        .as_any()
        .downcast_ref::<FixedSizeListArray>()
        .filter(|list| list.value_length() == len)
        .ok_or_else(|| StreamError::ColumnType {
            column: name,
            expected: "FixedSizeList<Float32>[N]",
            actual: column.data_type().clone(),
        })?;
    if list.null_count() > 0 || list.values().null_count() > 0 {
        return Err(StreamError::NullValues(name));
    }
    let values = list
        .values()
        .as_any()
        .downcast_ref::<Float32Array>()
        .ok_or_else(|| StreamError::ColumnType {
            column: name,
            expected: "FixedSizeList<Float32>[N]",
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

/// Looks up an optional non-null `Float32` scalar column, validating its type.
/// Returns `None` if the column is absent (additive `0.0.3` camera columns).
fn optional_f32<'a>(
    batch: &'a RecordBatch,
    name: &'static str,
) -> Result<Option<&'a Float32Array>, StreamError> {
    let Some(column) = batch.column_by_name(name) else {
        return Ok(None);
    };
    let array = column
        .as_any()
        .downcast_ref::<Float32Array>()
        .ok_or_else(|| StreamError::ColumnType {
            column: name,
            expected: "Float32",
            actual: column.data_type().clone(),
        })?;
    if array.null_count() > 0 {
        return Err(StreamError::NullValues(name));
    }
    Ok(Some(array))
}

/// Maps a [`CameraFormError`] onto the stream error type.
fn camera_form_error(error: CameraFormError) -> StreamError {
    match error {
        CameraFormError::Conflicting => StreamError::ConflictingCameraForms,
        CameraFormError::Incomplete => StreamError::IncompleteCameraForm,
    }
}

/// Decodes every row of `batch` into [`FrameParams`], validating required
/// columns, types, and non-nullness (including the fixed-size-list children).
/// The optional `0.0.2` `model`/`k`/`pose` matrix columns and the `0.0.3` CG
/// camera columns (`eye`/`target`/`direction`/`up`/`fovy`/`aspect`/`znear`/
/// `zfar`) are decoded if present.
pub fn decode_frames(batch: &RecordBatch) -> Result<Vec<FrameParams>, StreamError> {
    let center = require_vec2(batch, "center")?;
    let size = require_vec2(batch, "size")?;
    let theta = require_f32(batch, "theta")?;
    // Reject nulls at both the list level and the child-element level: a
    // non-null list row may still contain null float components.
    if center.null_count() > 0 || center.values().null_count() > 0 {
        return Err(StreamError::NullValues("center"));
    }
    if size.null_count() > 0 || size.values().null_count() > 0 {
        return Err(StreamError::NullValues("size"));
    }
    if theta.null_count() > 0 {
        return Err(StreamError::NullValues("theta"));
    }
    let model = optional_fixed_list(batch, "model", 16)?;
    let k = optional_fixed_list(batch, "k", 9)?;
    let pose = optional_fixed_list(batch, "pose", 16)?;
    let eye = optional_fixed_list(batch, "eye", 3)?;
    let target = optional_fixed_list(batch, "target", 3)?;
    let direction = optional_fixed_list(batch, "direction", 3)?;
    let up = optional_fixed_list(batch, "up", 3)?;
    let fovy = optional_f32(batch, "fovy")?;
    let aspect = optional_f32(batch, "aspect")?;
    let znear = optional_f32(batch, "znear")?;
    let zfar = optional_f32(batch, "zfar")?;
    (0..batch.num_rows())
        .map(|i| {
            let frame = FrameParams {
                center: read_vec2(center, i),
                size: read_vec2(size, i),
                theta: theta.value(i),
                model: model.map(|(list, values)| read_fixed::<16>(list, values, i)),
                k: k.map(|(list, values)| read_fixed::<9>(list, values, i)),
                pose: pose.map(|(list, values)| read_fixed::<16>(list, values, i)),
                eye: eye.map(|(list, values)| read_fixed::<3>(list, values, i)),
                target: target.map(|(list, values)| read_fixed::<3>(list, values, i)),
                direction: direction.map(|(list, values)| read_fixed::<3>(list, values, i)),
                up: up.map(|(list, values)| read_fixed::<3>(list, values, i)),
                fovy: fovy.map(|a| a.value(i)),
                aspect: aspect.map(|a| a.value(i)),
                znear: znear.map(|a| a.value(i)),
                zfar: zfar.map(|a| a.value(i)),
            };
            frame.check_camera_form().map_err(camera_form_error)?;
            Ok(frame)
        })
        .collect()
}

/// Decodes the optional per-frame **instanced draw list** columns `draw_mesh`
/// (`List<UInt32>`) and `draw_model` (`List<FixedSizeList<Float32>[16]>`), plus
/// the optional per-draw `draw_mode` (`List<UInt8>`) render-mode override, into
/// one `Vec<Draw>` per row. Returns `Some(rows)` when both required columns are
/// present, or `None` when neither is (legacy single-object streams). Having
/// exactly one of the `draw_mesh`/`draw_model` pair is an error, as is a per-row
/// length mismatch between any of the present lists. `draw_mode` bytes are
/// decoded via [`RenderMode::from_wire`] (`255` = inherit the global mode); an
/// absent `draw_mode` column leaves every [`Draw::mode`] as `None` (inherit).
fn decode_draws(batch: &RecordBatch) -> Result<Option<Vec<Vec<Draw>>>, StreamError> {
    let mesh_col = batch.column_by_name("draw_mesh");
    let model_col = batch.column_by_name("draw_model");
    let (mesh_col, model_col) = match (mesh_col, model_col) {
        (None, None) => return Ok(None),
        (Some(m), Some(n)) => (m, n),
        (Some(_), None) => return Err(StreamError::MissingColumn("draw_model")),
        (None, Some(_)) => return Err(StreamError::MissingColumn("draw_mesh")),
    };

    let mesh_list = mesh_col
        .as_any()
        .downcast_ref::<ListArray>()
        .ok_or_else(|| StreamError::ColumnType {
            column: "draw_mesh",
            expected: "List<UInt32>",
            actual: mesh_col.data_type().clone(),
        })?;
    let model_list = model_col
        .as_any()
        .downcast_ref::<ListArray>()
        .ok_or_else(|| StreamError::ColumnType {
            column: "draw_model",
            expected: "List<FixedSizeList<Float32>[16]>",
            actual: model_col.data_type().clone(),
        })?;
    if mesh_list.null_count() > 0 {
        return Err(StreamError::NullValues("draw_mesh"));
    }
    if model_list.null_count() > 0 {
        return Err(StreamError::NullValues("draw_model"));
    }

    // Optional per-draw render-mode override (`draw_mode`, `List<UInt8>`).
    let mode_list = match batch.column_by_name("draw_mode") {
        None => None,
        Some(col) => {
            let list = col.as_any().downcast_ref::<ListArray>().ok_or_else(|| {
                StreamError::ColumnType {
                    column: "draw_mode",
                    expected: "List<UInt8>",
                    actual: col.data_type().clone(),
                }
            })?;
            if list.null_count() > 0 {
                return Err(StreamError::NullValues("draw_mode"));
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
            .ok_or_else(|| StreamError::ColumnType {
                column: "draw_mesh",
                expected: "List<UInt32>",
                actual: ids_ref.data_type().clone(),
            })?;
        if ids.null_count() > 0 {
            return Err(StreamError::NullValues("draw_mesh"));
        }

        let models_ref = model_list.value(row);
        let models = models_ref
            .as_any()
            .downcast_ref::<FixedSizeListArray>()
            .filter(|list| list.value_length() == 16)
            .ok_or_else(|| StreamError::ColumnType {
                column: "draw_model",
                expected: "FixedSizeList<Float32>[16]",
                actual: models_ref.data_type().clone(),
            })?;
        if models.null_count() > 0 || models.values().null_count() > 0 {
            return Err(StreamError::NullValues("draw_model"));
        }
        if ids.len() != models.len() {
            return Err(StreamError::MismatchedDrawLists {
                row,
                mesh_len: ids.len(),
                model_len: models.len(),
            });
        }
        let model_values = models
            .values()
            .as_any()
            .downcast_ref::<Float32Array>()
            .ok_or_else(|| StreamError::ColumnType {
                column: "draw_model",
                expected: "FixedSizeList<Float32>[16]",
                actual: models.values().data_type().clone(),
            })?;

        // Per-draw modes for this row (empty ⇒ every draw inherits the global).
        let modes: Vec<Option<RenderMode>> = match &mode_list {
            None => Vec::new(),
            Some(mode_list) => {
                let modes_ref = mode_list.value(row);
                let bytes = modes_ref
                    .as_any()
                    .downcast_ref::<UInt8Array>()
                    .ok_or_else(|| StreamError::ColumnType {
                        column: "draw_mode",
                        expected: "List<UInt8>",
                        actual: modes_ref.data_type().clone(),
                    })?;
                if bytes.null_count() > 0 {
                    return Err(StreamError::NullValues("draw_mode"));
                }
                if bytes.len() != ids.len() {
                    return Err(StreamError::MismatchedDrawModes {
                        row,
                        mode_len: bytes.len(),
                        draw_len: ids.len(),
                    });
                }
                (0..bytes.len())
                    .map(|j| {
                        RenderMode::from_wire(bytes.value(j)).ok_or(StreamError::InvalidDrawMode {
                            value: bytes.value(j),
                        })
                    })
                    .collect::<Result<_, _>>()?
            }
        };

        let draws = (0..ids.len())
            .map(|j| Draw {
                mesh_id: ids.value(j),
                model: read_fixed::<16>(models, model_values, j),
                mode: modes.get(j).copied().flatten(),
            })
            .collect();
        rows.push(draws);
    }
    Ok(Some(rows))
}

/// Decodes the optional per-frame **background frame reference** column (`0.0.5`)
/// into one `Option<String>` per row. The column names a per-frame image the
/// shell loads at the boundary and composites beneath the scene via a
/// [`DrawableObject::FramePlane`]: `frame_path` (a filesystem path, native) is
/// preferred, else `frame_url` (a URL, browser). Returns `None` when neither
/// column is present (a stream without background frames); per-row nulls decode
/// to `None` (that frame has no background). The core never performs the I/O —
/// it only surfaces the reference string for the shell to resolve.
fn decode_frame_refs(batch: &RecordBatch) -> Result<Option<Vec<Option<String>>>, StreamError> {
    let (name, col) = match batch
        .column_by_name("frame_path")
        .map(|c| ("frame_path", c))
        .or_else(|| batch.column_by_name("frame_url").map(|c| ("frame_url", c)))
    {
        Some(pair) => pair,
        None => return Ok(None),
    };
    let strings =
        col.as_any()
            .downcast_ref::<StringArray>()
            .ok_or_else(|| StreamError::ColumnType {
                column: name,
                expected: "Utf8",
                actual: col.data_type().clone(),
            })?;
    let refs = (0..batch.num_rows())
        .map(|row| {
            if strings.is_null(row) {
                None
            } else {
                let s = strings.value(row);
                if s.is_empty() {
                    None
                } else {
                    Some(s.to_owned())
                }
            }
        })
        .collect();
    Ok(Some(refs))
}

/// each decoded [`FrameParams`] in stream order.
///
/// Convenience wrapper over [`read_frame_stream_with_meta`] that ignores the
/// stream's declared playback rate.
pub fn read_frame_stream<R: Read>(
    input: R,
    on_frame: impl FnMut(FrameParams),
) -> Result<(), StreamError> {
    read_frame_stream_with_meta(input, |_rate| {}, on_frame)
}

/// Like [`read_frame_stream`], but first invokes `on_meta` with the stream's
/// declared playback rate (fps, [`crate::DEFAULT_FRAME_RATE`] when absent) as
/// soon as the schema is known — before any frames — so a live player can pace
/// playback by wall-clock time. Rendering logic still lives in [`decode_frames`].
pub fn read_frame_stream_with_meta<R: Read>(
    input: R,
    on_meta: impl FnOnce(f64),
    mut on_frame: impl FnMut(FrameParams),
) -> Result<(), StreamError> {
    let reader = StreamReader::try_new(input, None)?;
    check_version(reader.schema().as_ref())?;
    on_meta(frame_rate_from_metadata(reader.schema().metadata()));
    for batch in reader {
        let batch = batch?;
        for frame in decode_frames(&batch)? {
            on_frame(frame);
        }
    }
    Ok(())
}

const BYTES_PER_PIXEL: u32 = 4;

/// Validates image dimensions against zero and `u32` overflow, returning the
/// pixel count. Does not check device limits (that needs an adapter). Called
/// before any `width * height` / `width * 4` arithmetic so absurd dimensions
/// produce a clean [`StreamError::InvalidDimensions`] instead of an overflow.
fn check_dimensions(width: u32, height: u32) -> Result<u32, StreamError> {
    let pixels = width
        .checked_mul(height)
        .filter(|&p| p > 0 && p <= i32::MAX as u32)
        .ok_or(StreamError::InvalidDimensions {
            width,
            height,
            reason: "width*height must be non-zero and <= i32::MAX",
        })?;
    width
        .checked_mul(BYTES_PER_PIXEL)
        .ok_or(StreamError::InvalidDimensions {
            width,
            height,
            reason: "row byte size overflows u32",
        })?;
    Ok(pixels)
}

/// A persistent GPU context that renders one [`FrameParams`] to tightly-packed
/// row-major RGBA bytes (`width*height*4`) per call.
pub struct BatchRenderer {
    device: wgpu::Device,
    queue: wgpu::Queue,
    renderer: MeshRenderer,
    texture: wgpu::Texture,
    staging: wgpu::Buffer,
    width: u32,
    height: u32,
    padded_bytes_per_row: u32,
    /// Render mode (filled/wireframe) applied to every mesh drawable this
    /// renderer builds into its per-frame [`Scene`](crate::Scene).
    mode: RenderMode,
    /// Whether to add a [`DrawableObject::AabbBox`] gizmo per drawn instance.
    show_aabb: bool,
    /// Whether to add a single origin [`DrawableObject::CoordinateAxes`] gizmo.
    show_axes: bool,
    /// Whether to add a [`DrawableObject::CoordinateAxes`] at *each* drawn
    /// instance's own `model` — the object's local coordinate frame.
    show_local_axes: bool,
}

impl BatchRenderer {
    /// Builds the GPU context (instance/adapter/device/pipeline/target/readback)
    /// once for a fixed `width` x `height`, rendering the built-in hello-triangle.
    pub fn new(width: u32, height: u32) -> Result<Self, StreamError> {
        pollster::block_on(Self::new_async(
            width,
            height,
            &[Mesh::hello_triangle()],
            &[Matrix4::IDENTITY],
        ))
    }

    /// Like [`BatchRenderer::new`] but renders the `meshes` of a `0.0.3` stream's
    /// leading mesh table, applying each mesh's [`Mesh::preview_transform`]
    /// (center + uniform scale-to-fit) beneath its per-frame model so an
    /// arbitrary-unit asset renders centered and at a reasonable size. Per-frame
    /// draw lists place instances of these meshes by index.
    pub fn with_meshes(width: u32, height: u32, meshes: &[Mesh]) -> Result<Self, StreamError> {
        let base_models: Vec<Matrix4> = meshes
            .iter()
            .map(|mesh| {
                mesh.preview_transform(crate::DEFAULT_PREVIEW_TARGET)
                    .matrix()
            })
            .collect();
        pollster::block_on(Self::new_async(width, height, meshes, &base_models))
    }

    async fn new_async(
        width: u32,
        height: u32,
        meshes: &[Mesh],
        base_models: &[Matrix4],
    ) -> Result<Self, StreamError> {
        // Guard against zero / overflow before allocating (device limits below).
        check_dimensions(width, height)?;

        let instance =
            wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions::default())
            .await
            .map_err(|e| StreamError::Render(e.to_string()))?;
        let info = adapter.get_info();
        log::info!(
            "using {:?} adapter \"{}\" ({:?})",
            info.backend,
            info.name,
            info.device_type
        );
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("trd device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::downlevel_defaults(),
                experimental_features: wgpu::ExperimentalFeatures::disabled(),
                memory_hints: wgpu::MemoryHints::MemoryUsage,
                trace: wgpu::Trace::Off,
            })
            .await
            .map_err(|e| StreamError::Render(e.to_string()))?;

        let max_dim = device.limits().max_texture_dimension_2d;
        if width > max_dim || height > max_dim {
            return Err(StreamError::InvalidDimensions {
                width,
                height,
                reason: "exceeds adapter max_texture_dimension_2d",
            });
        }

        let format = wgpu::TextureFormat::Rgba8UnormSrgb;
        let renderer = MeshRenderer::with_meshes(&device, format, meshes, base_models);

        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("trd render target"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });

        let unpadded = width * BYTES_PER_PIXEL;
        let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        let padded_bytes_per_row = unpadded.div_ceil(align) * align;
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("trd readback buffer"),
            size: u64::from(padded_bytes_per_row) * u64::from(height),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        Ok(Self {
            device,
            queue,
            renderer,
            texture,
            staging,
            width,
            height,
            padded_bytes_per_row,
            mode: RenderMode::Filled,
            show_aabb: false,
            show_axes: false,
            show_local_axes: false,
        })
    }

    /// The number of loaded meshes; valid [`Draw::mesh_id`]s are `0..mesh_count`.
    pub fn mesh_count(&self) -> usize {
        self.renderer.mesh_count()
    }

    /// Sets the [`RenderMode`] (filled or wireframe) applied to later `render`s.
    pub fn set_mode(&mut self, mode: RenderMode) {
        self.mode = mode;
    }

    /// Binds `texture` as the source sampled by [`RenderMode::Textured`] meshes
    /// (`0.0.4`). Delegates to [`MeshRenderer::set_texture`]; the image is
    /// (re)uploaded on the next `render`.
    pub fn set_texture(&mut self, texture: &dyn crate::texture::Texture) {
        self.renderer.set_texture(texture);
    }

    /// Enables/disables the per-instance AABB overlay box: when on, each drawn
    /// instance also contributes a [`DrawableObject::AabbBox`] to the scene.
    pub fn set_show_aabb(&mut self, show: bool) {
        self.show_aabb = show;
    }

    /// Enables/disables the origin coordinate-axes overlay gizmo: when on, the
    /// scene gains a single [`DrawableObject::CoordinateAxes`] at the world
    /// origin.
    pub fn set_show_axes(&mut self, show: bool) {
        self.show_axes = show;
    }

    /// Enables/disables the per-instance *local* coordinate-axes overlay: when
    /// on, each drawn instance also gains a [`DrawableObject::CoordinateAxes`]
    /// placed by its own `model`, visualizing that object's local frame (e.g.
    /// #77's `(e1,e2,e3)` quad placement).
    pub fn set_show_local_axes(&mut self, show: bool) {
        self.show_local_axes = show;
    }

    /// Uploads `image` as the **background frame texture** (#63) sampled by a
    /// [`DrawableObject::FramePlane`]. The GPU texture is reused across frames
    /// (grown only on a resolution change). Call before a
    /// [`render_frame`](Self::render_frame) with a `Some(fit)` to composite the
    /// image beneath the mesh scene.
    pub fn update_frame_texture(&mut self, image: &crate::texture::ImageData) {
        self.renderer.update_frame_texture_rgba(
            &self.queue,
            &image.rgba,
            image.width,
            image.height,
        );
    }

    /// Builds the per-frame [`Scene`](crate::Scene) from a wire `draws` list and
    /// this renderer's mode/overlay flags (delegates to [`build_scene`]). A
    /// `Some(fit)` prepends a background [`DrawableObject::FramePlane`] (#63).
    fn build_scene(&self, draws: &[Draw], frame: Option<FrameFit>) -> Vec<DrawableObject> {
        crate::render::build_scene(
            draws,
            self.mode,
            self.show_aabb,
            self.show_axes,
            self.show_local_axes,
            frame,
        )
    }

    /// Renders `params` with the given per-frame instance `draws`, compositing a
    /// background [`DrawableObject::FramePlane`] (#63) beneath the scene when
    /// `frame` is `Some(fit)` and a frame texture has been uploaded via
    /// [`update_frame_texture`](Self::update_frame_texture). Returns
    /// tightly-packed row-major RGBA bytes (`width*height*4`).
    pub fn render_frame(
        &mut self,
        params: FrameParams,
        draws: &[Draw],
        frame: Option<FrameFit>,
    ) -> Result<Vec<u8>, StreamError> {
        pollster::block_on(self.render_async(params, draws, frame))
    }

    async fn render_async(
        &mut self,
        params: FrameParams,
        draws: &[Draw],
        frame: Option<FrameFit>,
    ) -> Result<Vec<u8>, StreamError> {
        let scene = self.build_scene(draws, frame);
        let view = self
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("trd frame"),
            });
        self.renderer.encode(
            &self.queue,
            &mut encoder,
            &view,
            params,
            &scene,
            crate::render::Viewport {
                width: self.width,
                height: self.height,
            },
        );
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &self.texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &self.staging,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(self.padded_bytes_per_row),
                    rows_per_image: Some(self.height),
                },
            },
            wgpu::Extent3d {
                width: self.width,
                height: self.height,
                depth_or_array_layers: 1,
            },
        );
        self.queue.submit(Some(encoder.finish()));

        let slice = self.staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|e| StreamError::Render(e.to_string()))?;
        rx.recv()
            .expect("map_async callback dropped")
            .map_err(|e| StreamError::Render(e.to_string()))?;

        let pixels = {
            let mapped = slice.get_mapped_range().expect("buffer mapped after poll");
            crate::tightly_pack_rgba(&mapped, self.width, self.height, self.padded_bytes_per_row)?
        };
        self.staging.unmap();
        Ok(pixels)
    }
}

/// Reads the leading mesh stream, decoding **every** row of its batches into a
/// `Vec<Mesh>` (one mesh per row, in order), then **drains the rest of the
/// stream through its end-of-stream marker** so the underlying reader is
/// positioned at the start of the following params stream. The reader must be
/// unbuffered (as [`StreamReader::try_new`] produces) so it does not over-read
/// past the mesh stream's EOS into the params stream.
fn read_meshes<R: Read>(reader: &mut StreamReader<R>) -> Result<Vec<Mesh>, StreamError> {
    let mut meshes = Vec::new();
    for batch in reader.by_ref() {
        let batch = batch?;
        if batch.num_rows() > 0 {
            meshes.extend(Mesh::from_arrow_all(&batch)?);
        }
    }
    if meshes.is_empty() {
        return Err(StreamError::Mesh(crate::MeshError::Empty));
    }
    Ok(meshes)
}

/// Reads an optional leading **texture** sub-stream (`0.0.4`), decoding its first
/// row into an [`ImageTexture`], then **drains the rest of the stream through its
/// end-of-stream marker** so the underlying reader is positioned at the start of
/// the following params stream. Like [`read_meshes`], the reader must be
/// unbuffered so it does not over-read past this stream's EOS. Returns the
/// decoded image (a texture table is one row = one image).
fn read_texture<R: Read>(reader: &mut StreamReader<R>) -> Result<ImageTexture, StreamError> {
    let mut texture: Option<ImageTexture> = None;
    for batch in reader.by_ref() {
        let batch = batch?;
        if texture.is_none() && batch.num_rows() > 0 {
            texture = Some(ImageTexture::from_arrow(&batch)?);
        }
    }
    texture.ok_or(StreamError::Texture(crate::TextureError::Empty))
}

/// Resolves the `i`-th frame's instanced draw list: the wire `draw_lists` row
/// when present, else one instance of mesh 0 placed by the frame's own model
/// (legacy single-object behavior). Every referenced `mesh_id` is validated
/// against `mesh_count`. Shared by the headless [`run_stream`] path and the live
/// [`read_scene_stream_with_meta`] front-end so both resolve draws identically.
fn resolve_draws(
    params: &FrameParams,
    draw_lists: &Option<Vec<Vec<Draw>>>,
    i: usize,
    mesh_count: usize,
) -> Result<Vec<Draw>, StreamError> {
    let draws = match draw_lists {
        Some(rows) => rows[i].clone(),
        None => vec![Draw {
            mesh_id: 0,
            model: params.model_matrix().to_cols_array(),
            mode: None,
        }],
    };
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

/// Reads a trd input stream **mesh-aware** — the same `[mesh][texture?][params]`
/// framing [`run_stream`] uses — for a live front-end (e.g. the windowed
/// `trd-app`) that owns its own render target and encodes each frame's
/// [`Scene`](crate::Scene) itself, rather than the headless byte-stream path
/// [`run_stream`] drives.
///
/// Invokes `on_meshes` **once** with the decoded mesh table (a `0.0.3`+
/// mesh-first stream) or the built-in hello-triangle (a legacy `0.0.1`/`0.0.2`
/// params-only stream), then `on_texture` **once** with the optional bound
/// texture (`Some` only for a `0.0.4` stream carrying a texture table), then
/// `on_meta` with the stream's declared playback rate, then `on_frame` for each
/// frame's `(FrameParams, draws)` in order. A frame carrying no wire draw list
/// defaults to one instance of mesh 0 placed by the frame's own model — matching
/// [`run_stream`]. The mesh table's rows are referenced by 0-based index;
/// out-of-range `mesh_id`s are an error.
pub fn read_scene_stream_with_meta<R: Read>(
    input: R,
    on_meshes: impl FnOnce(Vec<Mesh>),
    on_texture: impl FnOnce(Option<ImageTexture>),
    on_meta: impl FnOnce(f64),
    mut on_frame: impl FnMut(FrameParams, Vec<Draw>, Option<String>),
) -> Result<(), StreamError> {
    let mut first = StreamReader::try_new(input, None)?;
    check_version(first.schema().as_ref())?;

    // Decodes one params batch into `(FrameParams, draws, frame_ref)` callbacks,
    // validating draw mesh ids against `mesh_count`.
    let mut emit = |batch: &RecordBatch, mesh_count: usize| -> Result<(), StreamError> {
        let frames = decode_frames(batch)?;
        let draw_lists = decode_draws(batch)?;
        let frame_refs = decode_frame_refs(batch)?;
        for (i, params) in frames.iter().enumerate() {
            let draws = resolve_draws(params, &draw_lists, i, mesh_count)?;
            let frame_ref = frame_refs.as_ref().and_then(|r| r[i].clone());
            on_frame(*params, draws, frame_ref);
        }
        Ok(())
    };

    if is_mesh_schema(first.schema().as_ref()) {
        // 0.0.3+ mesh-first: decode the leading mesh table, then the optional
        // texture table, then the params stream — all in the same byte stream.
        let meshes = read_meshes(&mut first)?;
        let mesh_count = meshes.len();
        on_meshes(meshes);

        // The stream after the mesh table is either a 0.0.4 texture table or the
        // params stream; sniff its schema to decide.
        let mut next = StreamReader::try_new(first.get_mut(), None)?;
        check_version(next.schema().as_ref())?;
        if is_texture_schema(next.schema().as_ref()) {
            on_texture(Some(read_texture(&mut next)?));
            let params = StreamReader::try_new(next.get_mut(), None)?;
            check_version(params.schema().as_ref())?;
            on_meta(frame_rate_from_metadata(params.schema().metadata()));
            for batch in params {
                emit(&batch?, mesh_count)?;
            }
        } else {
            on_texture(None);
            on_meta(frame_rate_from_metadata(next.schema().metadata()));
            for batch in next {
                emit(&batch?, mesh_count)?;
            }
        }
    } else {
        // Legacy params-only stream → the built-in hello-triangle, so the live
        // renderer draws the same demo the headless CLI does.
        on_meshes(vec![Mesh::hello_triangle()]);
        on_texture(None);
        on_meta(frame_rate_from_metadata(first.schema().metadata()));
        for batch in first {
            emit(&batch?, 1)?;
        }
    }
    Ok(())
}

/// A shell-provided closure that resolves a per-frame background frame reference
/// (a `frame_path`/`frame_url` string, `0.0.5`) into decoded RGBA pixels. Kept
/// out of `trd-core` so the core performs no file/network I/O: the native CLI
/// supplies one backed by the `image` crate + a `--frames-base` dir; a stream
/// without background frames (or a shell that doesn't load them) passes `None`.
/// Returning `None` for a given reference renders that frame without a
/// background plane (the shell decides how to report the miss).
pub type FrameResolver<'a> = &'a dyn Fn(&str) -> Option<crate::texture::ImageData>;

/// Renders every frame of the `params` batch stream to `output`, one Arrow
/// output batch per input batch. Shared by both the mesh-first and legacy paths.
/// When `frame_resolver` is `Some`, a frame carrying a `frame_path`/`frame_url`
/// reference (`0.0.5`) has its background image resolved + uploaded and composited
/// beneath the scene via a [`DrawableObject`](crate::render::DrawableObject)`::FramePlane`.
fn render_params<I, W>(
    params: I,
    mut renderer: BatchRenderer,
    frame_rate: f64,
    width: u32,
    height: u32,
    frame_resolver: Option<FrameResolver>,
    mut output: W,
) -> Result<(), StreamError>
where
    I: Iterator<Item = Result<RecordBatch, ArrowError>>,
    W: Write,
{
    let mut output_session = OutputSession::with_frame_rate(width, height, Some(frame_rate))?;
    output.write_all(&output_session.drain_new()?)?;
    let mesh_count = renderer.mesh_count();
    // The path of the frame texture currently uploaded, so consecutive frames
    // sharing a background image skip the decode + re-upload.
    let mut last_frame_ref: Option<String> = None;
    for batch in params {
        let batch = batch?;
        let frames = decode_frames(&batch)?;
        // Optional per-frame instanced draw list; absent ⇒ one instance of mesh 0
        // placed by the frame's own model (legacy single-object behavior).
        let draw_lists = decode_draws(&batch)?;
        // Optional per-frame background frame reference (0.0.5).
        let frame_refs = decode_frame_refs(&batch)?;
        let mut planes: Vec<Vec<u8>> = Vec::with_capacity(frames.len());
        for (i, params) in frames.iter().enumerate() {
            let draws = resolve_draws(params, &draw_lists, i, mesh_count)?;
            let frame_ref = frame_refs.as_ref().and_then(|r| r[i].as_deref());
            let mut frame_fit = None;
            if let (Some(path), Some(resolve)) = (frame_ref, frame_resolver) {
                if last_frame_ref.as_deref() != Some(path) {
                    if let Some(image) = resolve(path) {
                        renderer.update_frame_texture(&image);
                        last_frame_ref = Some(path.to_owned());
                        frame_fit = Some(FrameFit::Stretch);
                    }
                } else {
                    frame_fit = Some(FrameFit::Stretch);
                }
            }
            planes.push(renderer.render_frame(*params, &draws, frame_fit)?);
        }
        output_session.write_rgba_batch(&planes)?;
        output.write_all(&output_session.drain_new()?)?;
    }
    output_session.finish()?;
    output.write_all(&output_session.drain_new()?)?;
    Ok(())
}

/// Appearance options for [`run_stream`]: the mesh draw [`RenderMode`] plus the
/// optional AABB / coordinate-axes gizmo overlays. Bundled into one value so the
/// entry point threads a single struct instead of three positional flags (and
/// stays within clippy's argument budget). [`Default`] is filled, no overlays.
#[derive(Debug, Clone, Copy, Default)]
pub struct RenderOptions {
    /// How meshes are drawn (filled / wireframe / textured).
    pub mode: RenderMode,
    /// Overlay each drawn mesh instance's axis-aligned bounding box (#42).
    pub show_aabb: bool,
    /// Overlay a world-origin coordinate-axes gizmo (#42).
    pub show_axes: bool,
    /// Overlay a coordinate-axes gizmo at *each* drawn object's local (model)
    /// frame — its model-space X/Y/Z axes as placed (e.g. #77's `(e1,e2,e3)`).
    pub show_local_axes: bool,
}

/// Reads a trd input stream, renders each frame, and writes an Arrow IPC stream
/// of `fixed_shape_tensor` images to `output`. Output batch boundaries mirror
/// input batches (one batch in flight).
///
/// A `0.0.4` stream is `[mesh][texture?][params]`: the leading mesh table is
/// decoded once (via [`Mesh::from_arrow_all`]) and uploaded, then an optional
/// texture table is uploaded as the bound albedo, then the following params
/// stream drives per-frame rendering. A legacy `0.0.1`/`0.0.2` params-only stream
/// renders the built-in hello-triangle. The sub-streams are told apart by
/// sniffing each schema ([`is_mesh_schema`] / [`is_texture_schema`]).
pub fn run_stream<R: Read, W: Write>(
    input: R,
    output: W,
    width: u32,
    height: u32,
    options: RenderOptions,
    frame_resolver: Option<FrameResolver>,
) -> Result<(), StreamError> {
    let RenderOptions {
        mode,
        show_aabb,
        show_axes,
        show_local_axes,
    } = options;
    // Validate dimensions up front so schema construction (which multiplies
    // width*height) can't overflow before BatchRenderer's guard runs.
    check_dimensions(width, height)?;

    let mut first = StreamReader::try_new(input, None)?;
    check_version(first.schema().as_ref())?;

    if is_mesh_schema(first.schema().as_ref()) {
        // 0.0.3+ mesh-first: decode + upload the mesh table (one mesh per row),
        // then the optional texture table, then render the params stream that
        // follows them in the same byte stream.
        let meshes = read_meshes(&mut first)?;
        let mut renderer = BatchRenderer::with_meshes(width, height, &meshes)?;
        renderer.set_mode(mode);
        renderer.set_show_aabb(show_aabb);
        renderer.set_show_axes(show_axes);
        renderer.set_show_local_axes(show_local_axes);

        // The stream after the mesh table is either a 0.0.4 texture table or the
        // params stream; sniff its schema to decide.
        let mut next = StreamReader::try_new(first.get_mut(), None)?;
        check_version(next.schema().as_ref())?;
        if is_texture_schema(next.schema().as_ref()) {
            let texture = read_texture(&mut next)?;
            renderer.set_texture(&texture);
            let params = StreamReader::try_new(next.get_mut(), None)?;
            check_version(params.schema().as_ref())?;
            let frame_rate = frame_rate_from_metadata(params.schema().metadata());
            render_params(
                params,
                renderer,
                frame_rate,
                width,
                height,
                frame_resolver,
                output,
            )
        } else {
            let frame_rate = frame_rate_from_metadata(next.schema().metadata());
            render_params(
                next,
                renderer,
                frame_rate,
                width,
                height,
                frame_resolver,
                output,
            )
        }
    } else {
        // Legacy params-only stream → built-in hello-triangle.
        let mut renderer = BatchRenderer::new(width, height)?;
        renderer.set_mode(mode);
        renderer.set_show_aabb(show_aabb);
        renderer.set_show_axes(show_axes);
        renderer.set_show_local_axes(show_local_axes);
        let frame_rate = frame_rate_from_metadata(first.schema().metadata());
        render_params(
            first,
            renderer,
            frame_rate,
            width,
            height,
            frame_resolver,
            output,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::render::build_scene;
    use arrow::array::{ArrayRef, FixedSizeListArray as U8List, UInt8Array};
    use arrow::ipc::writer::StreamWriter;

    fn build_input_batch(frames: &[FrameParams]) -> RecordBatch {
        let schema = Arc::new(input_schema());
        let flat_center: Vec<f32> = frames.iter().flat_map(|f| f.center).collect();
        let flat_size: Vec<f32> = frames.iter().flat_map(|f| f.size).collect();
        let thetas: Vec<f32> = frames.iter().map(|f| f.theta).collect();
        let item = Arc::new(Field::new("item", DataType::Float32, false));
        let center = FixedSizeListArray::new(
            item.clone(),
            2,
            Arc::new(Float32Array::from(flat_center)),
            None,
        );
        let size = FixedSizeListArray::new(item, 2, Arc::new(Float32Array::from(flat_size)), None);
        let theta = Float32Array::from(thetas);
        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(center) as ArrayRef,
                Arc::new(size) as ArrayRef,
                Arc::new(theta) as ArrayRef,
            ],
        )
        .unwrap()
    }

    #[test]
    fn decodes_frames_roundtrip() {
        let frames = vec![
            FrameParams::IDENTITY,
            FrameParams {
                center: [0.1, -0.2],
                size: [0.5, 0.5],
                theta: 1.25,
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

    /// Builds a one-row batch of identity center/size/theta plus the given extra
    /// `(field, column)` pairs.
    fn camera_batch(extra: Vec<(Field, ArrayRef)>) -> RecordBatch {
        let vec2 = |name| {
            Field::new(
                name,
                DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float32, false)), 2),
                false,
            )
        };
        let mut fields = vec![
            vec2("center"),
            vec2("size"),
            Field::new("theta", DataType::Float32, false),
        ];
        let mut columns: Vec<ArrayRef> = vec![
            list_col(2, vec![0.0, 0.0]),
            list_col(2, vec![1.0, 1.0]),
            Arc::new(Float32Array::from(vec![0.0_f32])) as ArrayRef,
        ];
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
            Err(StreamError::IncompleteCameraForm)
        ));
        // CV `k` mixed with CG `eye` is conflicting.
        let conflicting = camera_batch(vec![
            (list_field("k", 9), list_col(9, vec![1.0; 9])),
            (list_field("eye", 3), list_col(3, vec![1.0, 2.0, 3.0])),
        ]);
        assert!(matches!(
            decode_frames(&conflicting),
            Err(StreamError::ConflictingCameraForms)
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
        let vec2 = |name| {
            Field::new(
                name,
                DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float32, false)), 2),
                false,
            )
        };
        let n = mesh_rows.len();
        let schema = Arc::new(Schema::new(vec![
            vec2("center"),
            vec2("size"),
            Field::new("theta", DataType::Float32, false),
            draw_field("draw_mesh", DataType::UInt32),
            draw_field(
                "draw_model",
                DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float32, false)), 16),
            ),
        ]));
        RecordBatch::try_new(
            schema,
            vec![
                list_col(2, vec![0.0; n * 2]),
                list_col(2, vec![1.0; n * 2]),
                Arc::new(Float32Array::from(vec![0.0_f32; n])) as ArrayRef,
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
                    model: a,
                    mode: None
                },
                Draw {
                    mesh_id: 1,
                    model: b,
                    mode: None
                },
            ]
        );
        assert_eq!(
            rows[1],
            vec![Draw {
                mesh_id: 1,
                model: b,
                mode: None
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
            Err(StreamError::MismatchedDrawLists {
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
            Field::new(
                "center",
                DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float32, false)), 2),
                false,
            ),
            Field::new(
                "size",
                DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float32, false)), 2),
                false,
            ),
            Field::new("theta", DataType::Float32, false),
            draw_field("draw_mesh", DataType::UInt32),
            draw_field(
                "draw_model",
                DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float32, false)), 16),
            ),
        ];
        let mut cols: Vec<ArrayRef> = vec![
            list_col(2, vec![0.0; n * 2]),
            list_col(2, vec![1.0; n * 2]),
            Arc::new(Float32Array::from(vec![0.0_f32; n])) as ArrayRef,
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
        assert_eq!(rows[0][0].mode, None);
        assert_eq!(rows[0][1].mode, Some(RenderMode::Wireframe));
        assert_eq!(rows[1][0].mode, Some(RenderMode::Textured));

        // Absent `draw_mode` column ⇒ every draw inherits (None).
        let plain = draw_batch_with_modes(&[vec![0, 1]], &[vec![m, m]], None);
        let plain_rows = decode_draws(&plain).unwrap().unwrap();
        assert!(plain_rows[0].iter().all(|d| d.mode.is_none()));
    }

    #[test]
    fn rejects_invalid_and_mismatched_draw_modes() {
        let m = [0.0f32; 16];
        // A byte outside {0,1,2,255} is rejected.
        let bad = draw_batch_with_modes(&[vec![0]], &[vec![m]], Some(&[vec![7]]));
        assert!(matches!(
            decode_draws(&bad),
            Err(StreamError::InvalidDrawMode { value: 7 })
        ));
        // A `draw_mode` list shorter than the draw list is rejected.
        let short = draw_batch_with_modes(&[vec![0, 1]], &[vec![m, m]], Some(&[vec![0]]));
        assert!(matches!(
            decode_draws(&short),
            Err(StreamError::MismatchedDrawModes {
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
            batch.schema().field(1).clone(),
            batch.schema().field(2).clone(),
            draw_field("draw_mesh", DataType::UInt32),
        ]));
        let with_mesh_only = RecordBatch::try_new(
            schema,
            vec![
                batch.column(0).clone(),
                batch.column(1).clone(),
                batch.column(2).clone(),
                draw_mesh_col(&[vec![0]]),
            ],
        )
        .unwrap();
        assert!(matches!(
            decode_draws(&with_mesh_only),
            Err(StreamError::MissingColumn("draw_model"))
        ));
    }

    #[test]
    fn build_scene_maps_draws_and_overlays_in_bucket_order() {
        // #41: the draw list + mode/overlay flags become an ordered `Scene` of
        // `DrawableObject`s — one Mesh per draw (in `mode`), then one AabbBox per
        // draw when enabled, then a single origin CoordinateAxes when enabled.
        let a = [1.0f32; 16];
        let b = [2.0f32; 16];
        let draws = [
            Draw {
                mesh_id: 0,
                model: a,
                mode: None,
            },
            Draw {
                mesh_id: 1,
                model: b,
                mode: None,
            },
        ];

        // Plain filled: exactly one Mesh drawable per draw, no gizmos.
        assert_eq!(
            build_scene(&draws, RenderMode::Filled, false, false, false, None),
            vec![
                DrawableObject::Mesh {
                    mesh_id: 0,
                    model: a,
                    mode: RenderMode::Filled,
                },
                DrawableObject::Mesh {
                    mesh_id: 1,
                    model: b,
                    mode: RenderMode::Filled,
                },
            ]
        );

        // Wireframe propagates the mode to every mesh drawable.
        assert_eq!(
            build_scene(&draws, RenderMode::Wireframe, false, false, false, None),
            vec![
                DrawableObject::Mesh {
                    mesh_id: 0,
                    model: a,
                    mode: RenderMode::Wireframe,
                },
                DrawableObject::Mesh {
                    mesh_id: 1,
                    model: b,
                    mode: RenderMode::Wireframe,
                },
            ]
        );

        // Both overlays: meshes, then a tracking box per draw, then one gizmo.
        assert_eq!(
            build_scene(&draws, RenderMode::Filled, true, true, false, None),
            vec![
                DrawableObject::Mesh {
                    mesh_id: 0,
                    model: a,
                    mode: RenderMode::Filled,
                },
                DrawableObject::Mesh {
                    mesh_id: 1,
                    model: b,
                    mode: RenderMode::Filled,
                },
                DrawableObject::AabbBox {
                    mesh_id: 0,
                    model: a,
                },
                DrawableObject::AabbBox {
                    mesh_id: 1,
                    model: b,
                },
                DrawableObject::CoordinateAxes {
                    model: Matrix4::IDENTITY.to_cols_array(),
                },
            ]
        );

        // Local axes: one CoordinateAxes per draw at its own model (in the mesh
        // bucket order, before the world-origin gizmo), each tracking its draw.
        assert_eq!(
            build_scene(&draws, RenderMode::Filled, false, false, true, None),
            vec![
                DrawableObject::Mesh {
                    mesh_id: 0,
                    model: a,
                    mode: RenderMode::Filled,
                },
                DrawableObject::Mesh {
                    mesh_id: 1,
                    model: b,
                    mode: RenderMode::Filled,
                },
                DrawableObject::CoordinateAxes { model: a },
                DrawableObject::CoordinateAxes { model: b },
            ]
        );

        // Per-draw mode override: a draw's own `mode` wins over the global one,
        // so one frame can mix (e.g.) a textured mesh with a wireframe overlay.
        let mixed = [
            Draw {
                mesh_id: 0,
                model: a,
                mode: None,
            },
            Draw {
                mesh_id: 1,
                model: b,
                mode: Some(RenderMode::Wireframe),
            },
        ];
        assert_eq!(
            build_scene(&mixed, RenderMode::Textured, false, false, false, None),
            vec![
                DrawableObject::Mesh {
                    mesh_id: 0,
                    model: a,
                    mode: RenderMode::Textured,
                },
                DrawableObject::Mesh {
                    mesh_id: 1,
                    model: b,
                    mode: RenderMode::Wireframe,
                },
            ]
        );
    }

    #[test]
    fn read_frame_stream_roundtrip() {
        use arrow::ipc::writer::StreamWriter;

        let frames = vec![
            FrameParams::IDENTITY,
            FrameParams {
                center: [0.2, -0.1],
                size: [0.5, 0.5],
                theta: 1.0,
                ..FrameParams::IDENTITY
            },
        ];
        let batch = build_input_batch(&frames);

        // Encode the batch as an Arrow IPC stream, then read it back through the
        // public streaming decoder used by the live viewer.
        let mut buf: Vec<u8> = Vec::new();
        {
            let mut writer = StreamWriter::try_new(&mut buf, batch.schema().as_ref()).unwrap();
            writer.write(&batch).unwrap();
            writer.finish().unwrap();
        }

        let mut got = Vec::new();
        read_frame_stream(buf.as_slice(), |p| got.push(p)).unwrap();
        assert_eq!(got, frames);
    }

    #[test]
    fn missing_column_is_error() {
        let batch = build_input_batch(&[FrameParams::IDENTITY]);
        let reduced = batch.project(&[0, 1]).unwrap(); // drop `theta`
        assert!(matches!(
            decode_frames(&reduced),
            Err(StreamError::MissingColumn("theta"))
        ));
    }

    #[test]
    fn child_null_in_vec2_is_error() {
        // A non-null list row whose child float is null must be rejected.
        let item = Arc::new(Field::new("item", DataType::Float32, true));
        let center_values = Float32Array::from(vec![Some(0.0), None]);
        let center = FixedSizeListArray::new(item.clone(), 2, Arc::new(center_values), None);
        let size = FixedSizeListArray::new(
            Arc::new(Field::new("item", DataType::Float32, false)),
            2,
            Arc::new(Float32Array::from(vec![1.0, 1.0])),
            None,
        );
        let theta = Float32Array::from(vec![0.0]);
        let schema = Arc::new(Schema::new(vec![
            Field::new("center", center.data_type().clone(), false),
            Field::new("size", size.data_type().clone(), false),
            Field::new("theta", DataType::Float32, false),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(center) as ArrayRef,
                Arc::new(size),
                Arc::new(theta),
            ],
        )
        .unwrap();
        assert!(matches!(
            decode_frames(&batch),
            Err(StreamError::NullValues("center"))
        ));
    }

    #[test]
    fn wrong_type_is_error() {
        use arrow::array::Int32Array;
        let schema = Arc::new(Schema::new(vec![
            super::vec2_field("center"),
            super::vec2_field("size"),
            Field::new("theta", DataType::Int32, false),
        ]));
        let item = Arc::new(Field::new("item", DataType::Float32, false));
        let center = FixedSizeListArray::new(
            item.clone(),
            2,
            Arc::new(Float32Array::from(vec![0.0, 0.0])),
            None,
        );
        let size =
            FixedSizeListArray::new(item, 2, Arc::new(Float32Array::from(vec![1.0, 1.0])), None);
        let theta = Int32Array::from(vec![3]);
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(center) as ArrayRef,
                Arc::new(size),
                Arc::new(theta),
            ],
        )
        .unwrap();
        assert!(matches!(
            decode_frames(&batch),
            Err(StreamError::ColumnType {
                column: "theta",
                ..
            })
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
            Err(StreamError::UnsupportedVersion(v)) if v == "9.9.9"
        ));
    }

    #[test]
    fn version_check_allows_absent_and_matching() {
        assert!(check_version(&Schema::empty()).is_ok());
        assert!(check_version(&input_schema()).is_ok());
    }

    #[test]
    fn check_dimensions_rejects_zero_and_overflow() {
        assert!(check_dimensions(4, 3).is_ok());
        assert!(matches!(
            check_dimensions(0, 3),
            Err(StreamError::InvalidDimensions { .. })
        ));
        // width*height overflows u32 / exceeds i32::MAX.
        assert!(matches!(
            check_dimensions(100_000, 100_000),
            Err(StreamError::InvalidDimensions { .. })
        ));
    }

    #[test]
    #[ignore = "requires a GPU adapter"]
    fn run_stream_renders_rotating_triangle() {
        let (w, h) = (32u32, 32u32);
        let frames: Vec<FrameParams> = (0..6)
            .map(|i| FrameParams {
                center: [0.0, 0.0],
                size: [1.0, 1.0],
                theta: i as f32 * std::f32::consts::PI / 6.0,
                ..FrameParams::IDENTITY
            })
            .collect();

        // Encode two input batches to an in-memory Arrow IPC stream.
        let first = build_input_batch(&frames[..1]);
        let second = build_input_batch(&frames[1..]);
        let mut input_bytes = Vec::new();
        {
            let schema = Arc::new(input_schema());
            let mut wr = StreamWriter::try_new(&mut input_bytes, &schema).unwrap();
            wr.write(&first).unwrap();
            wr.write(&second).unwrap();
            wr.finish().unwrap();
        }

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

        // Decode output and assert per-frame invariants.
        let reader = StreamReader::try_new(&output_bytes[..], None).unwrap();
        let batches = reader.collect::<Result<Vec<_>, _>>().unwrap();
        assert_eq!(
            batches
                .iter()
                .map(RecordBatch::num_rows)
                .collect::<Vec<_>>(),
            vec![1, frames.len() - 1]
        );
        let total_rows: usize = batches.iter().map(|b| b.num_rows()).sum();
        assert_eq!(total_rows, frames.len());

        let pixels = (w * h) as usize;
        let get = |batch: &RecordBatch, name: &str| -> U8List {
            batch
                .column_by_name(name)
                .unwrap()
                .as_any()
                .downcast_ref::<U8List>()
                .unwrap()
                .clone()
        };
        let (r, g, b, a) = (
            get(&batches[0], "r"),
            get(&batches[0], "g"),
            get(&batches[0], "b"),
            get(&batches[0], "a"),
        );
        assert_eq!(r.value_length(), pixels as i32);

        let row0 = |ch: &U8List, row: usize, i: usize| -> u8 {
            ch.value(row)
                .as_any()
                .downcast_ref::<UInt8Array>()
                .unwrap()
                .value(i)
        };
        // Top-left corner is background black; alpha opaque everywhere.
        let corner = 0usize;
        assert_eq!(row0(&r, 0, corner), 0);
        assert_eq!(row0(&g, 0, corner), 0);
        assert_eq!(row0(&b, 0, corner), 0);
        assert_eq!(row0(&a, 0, corner), 255);
        // Center pixel is inside the triangle (non-black).
        let center = (h as usize / 2) * w as usize + w as usize / 2;
        let cbright =
            row0(&r, 0, center) as u32 + row0(&g, 0, center) as u32 + row0(&b, 0, center) as u32;
        assert!(cbright > 0, "center should be inside the triangle");
        // Rotation changes the image between the first and a later frame.
        let last = get(&batches[1], "r");
        let last_row = frames.len() - 2;
        let differs = (0..pixels).any(|i| row0(&r, 0, i) != row0(&last, last_row, i));
        assert!(differs, "rotation should change pixels across frames");
    }

    // ---- 0.0.3 two-stream [mesh][params] framing ----

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
            [(
                PROTOCOL_VERSION_KEY.to_string(),
                PROTOCOL_VERSION.to_string(),
            )]
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
        let mut wr = StreamWriter::try_new(buf, &input_schema()).unwrap();
        wr.write(&batch).unwrap();
        wr.finish().unwrap();
    }

    #[test]
    fn two_stream_mesh_then_params_split_and_decode() {
        use std::io::Cursor;

        // Build a concatenated [mesh][params] byte stream in memory.
        let mesh = Mesh::hello_triangle();
        let frames = vec![
            FrameParams::IDENTITY,
            FrameParams {
                theta: 1.0,
                ..FrameParams::IDENTITY
            },
        ];
        let mut bytes = Vec::new();
        write_mesh_stream(&mut bytes, &mesh);
        write_params_stream(&mut bytes, &frames);

        // The framing helpers must recover the mesh, then the params that follow
        // it in the same byte stream (the mesh reader must not over-read).
        let mut first = StreamReader::try_new(Cursor::new(bytes), None).unwrap();
        assert!(is_mesh_schema(first.schema().as_ref()));
        let decoded_meshes = read_meshes(&mut first).unwrap();
        assert_eq!(decoded_meshes, vec![mesh]);

        let params = StreamReader::try_new(first.get_mut(), None).unwrap();
        assert!(!is_mesh_schema(params.schema().as_ref()));
        let mut decoded = Vec::new();
        for batch in params {
            decoded.extend(decode_frames(&batch.unwrap()).unwrap());
        }
        assert_eq!(decoded, frames);
    }

    #[test]
    fn legacy_params_only_stream_has_no_mesh_schema() {
        let mut bytes = Vec::new();
        write_params_stream(&mut bytes, &[FrameParams::IDENTITY]);
        let reader = StreamReader::try_new(bytes.as_slice(), None).unwrap();
        assert!(!is_mesh_schema(reader.schema().as_ref()));
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
                theta: 0.5,
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
}
