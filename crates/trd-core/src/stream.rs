//! Native-only Arrow streaming protocol (trd protocol 0.0.3).
//!
//! Input is one or two concatenated Arrow IPC streams on stdin. A `0.0.3` stream
//! is `[mesh][params]`: a leading **mesh** table (one row = one mesh, decoded by
//! [`Mesh::from_arrow`]) followed by the **params** stream (one row per frame:
//! `center`, `size`, `theta`, + optional `model`/`k`/`pose`). A legacy
//! `0.0.1`/`0.0.2` stream is just the params stream and renders the built-in
//! hello-triangle. Output: one row per frame, four `fixed_shape_tensor<u8>`
//! channels `r,g,b,a` of shape `[H, W]`.

use std::sync::Arc;

use arrow::array::{Array, FixedSizeListArray, Float32Array, RecordBatch};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::error::ArrowError;
use arrow::ipc::reader::StreamReader;
use std::io::{Read, Write};

use crate::math::Matrix4;
use crate::protocol::{
    frame_rate_from_metadata, PROTOCOL_VERSION, PROTOCOL_VERSION_KEY, SUPPORTED_INPUT_VERSIONS,
};
use crate::render::{CameraFormError, FrameParams, Mesh, MeshRenderer};
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

/// Reads an Arrow IPC frame-params stream from `input`, invoking `on_frame` for
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
}

impl BatchRenderer {
    /// Builds the GPU context (instance/adapter/device/pipeline/target/readback)
    /// once for a fixed `width` x `height`, rendering the built-in hello-triangle.
    pub fn new(width: u32, height: u32) -> Result<Self, StreamError> {
        pollster::block_on(Self::new_async(
            width,
            height,
            &Mesh::hello_triangle(),
            Matrix4::IDENTITY,
        ))
    }

    /// Like [`BatchRenderer::new`] but renders `mesh` (the leading mesh table of
    /// a `0.0.3` stream), applying its [`Mesh::preview_transform`] (center +
    /// uniform scale-to-fit) beneath the per-frame model so an arbitrary-unit
    /// asset renders centered and at a reasonable size.
    pub fn with_mesh(width: u32, height: u32, mesh: &Mesh) -> Result<Self, StreamError> {
        let base_model = mesh
            .preview_transform(crate::DEFAULT_PREVIEW_TARGET)
            .matrix();
        pollster::block_on(Self::new_async(width, height, mesh, base_model))
    }

    async fn new_async(
        width: u32,
        height: u32,
        mesh: &Mesh,
        base_model: Matrix4,
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
        let renderer = MeshRenderer::with_base_model(&device, format, mesh, base_model);

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
        })
    }

    /// Renders `params` and returns tightly-packed row-major RGBA bytes
    /// (`width*height*4`).
    pub fn render(&mut self, params: FrameParams) -> Result<Vec<u8>, StreamError> {
        pollster::block_on(self.render_async(params))
    }

    async fn render_async(&mut self, params: FrameParams) -> Result<Vec<u8>, StreamError> {
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
            self.width,
            self.height,
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

/// True if `schema` is a **mesh table** (has a `position` column) — used to tell
/// a leading `0.0.3` mesh stream apart from the params stream that follows it (or
/// a legacy params-only stream).
fn is_mesh_schema(schema: &Schema) -> bool {
    schema.fields().iter().any(|f| f.name() == "position")
}

/// Reads the leading mesh stream, decoding its first row into a [`Mesh`], then
/// **drains the rest of the stream through its end-of-stream marker** so the
/// underlying reader is positioned at the start of the following params stream.
/// The reader must be unbuffered (as [`StreamReader::try_new`] produces) so it
/// does not over-read past the mesh stream's EOS into the params stream.
fn read_single_mesh<R: Read>(reader: &mut StreamReader<R>) -> Result<Mesh, StreamError> {
    let mut mesh = None;
    for batch in reader.by_ref() {
        let batch = batch?;
        if mesh.is_none() {
            mesh = Some(Mesh::from_arrow(&batch)?);
        }
    }
    mesh.ok_or(StreamError::Mesh(crate::MeshError::Empty))
}

/// Renders every frame of the `params` batch stream to `output`, one Arrow
/// output batch per input batch. Shared by both the mesh-first and legacy paths.
fn render_params<I, W>(
    params: I,
    mut renderer: BatchRenderer,
    frame_rate: f64,
    width: u32,
    height: u32,
    mut output: W,
) -> Result<(), StreamError>
where
    I: Iterator<Item = Result<RecordBatch, ArrowError>>,
    W: Write,
{
    let mut output_session = OutputSession::with_frame_rate(width, height, Some(frame_rate))?;
    output.write_all(&output_session.drain_new()?)?;
    for batch in params {
        let batch = batch?;
        let frames = decode_frames(&batch)?;
        let planes: Vec<Vec<u8>> = frames
            .iter()
            .map(|p| renderer.render(*p))
            .collect::<Result<_, _>>()?;
        output_session.write_rgba_batch(&planes)?;
        output.write_all(&output_session.drain_new()?)?;
    }
    output_session.finish()?;
    output.write_all(&output_session.drain_new()?)?;
    Ok(())
}

/// Reads a trd input stream, renders each frame, and writes an Arrow IPC stream
/// of `fixed_shape_tensor` images to `output`. Output batch boundaries mirror
/// input batches (one batch in flight).
///
/// A `0.0.3` stream is `[mesh][params]`: the leading mesh table is decoded once
/// (via [`Mesh::from_arrow`]) and uploaded, then the following params stream
/// drives per-frame rendering. A legacy `0.0.1`/`0.0.2` params-only stream
/// renders the built-in hello-triangle. The two are told apart by sniffing the
/// first stream's schema ([`is_mesh_schema`]).
pub fn run_stream<R: Read, W: Write>(
    input: R,
    output: W,
    width: u32,
    height: u32,
) -> Result<(), StreamError> {
    // Validate dimensions up front so schema construction (which multiplies
    // width*height) can't overflow before BatchRenderer's guard runs.
    check_dimensions(width, height)?;

    let mut first = StreamReader::try_new(input, None)?;
    check_version(first.schema().as_ref())?;

    if is_mesh_schema(first.schema().as_ref()) {
        // 0.0.3 mesh-first: decode + upload the mesh, then render the params
        // stream that follows it in the same byte stream.
        let mesh = read_single_mesh(&mut first)?;
        let renderer = BatchRenderer::with_mesh(width, height, &mesh)?;
        let params = StreamReader::try_new(first.get_mut(), None)?;
        check_version(params.schema().as_ref())?;
        let frame_rate = frame_rate_from_metadata(params.schema().metadata());
        render_params(params, renderer, frame_rate, width, height, output)
    } else {
        // Legacy params-only stream → built-in hello-triangle.
        let renderer = BatchRenderer::new(width, height)?;
        let frame_rate = frame_rate_from_metadata(first.schema().metadata());
        render_params(first, renderer, frame_rate, width, height, output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
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
        run_stream(&input_bytes[..], &mut output_bytes, w, h).unwrap();

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
        let decoded_mesh = read_single_mesh(&mut first).unwrap();
        assert_eq!(decoded_mesh, mesh);

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
        run_stream(&input_bytes[..], &mut output_bytes, w, h).unwrap();

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
