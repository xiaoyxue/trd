//! Native-only Arrow streaming protocol (trd protocol 0.0.1).
//!
//! Input: one row per frame (`center`, `size`, `theta`). Output: one row per
//! frame, four `fixed_shape_tensor<u8>` channels `r,g,b,a` of shape `[H, W]`.

use std::sync::Arc;

use arrow::array::{Array, FixedSizeListArray, Float32Array, RecordBatch};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::error::ArrowError;
use arrow::ipc::reader::StreamReader;
use std::io::{Read, Write};

use crate::protocol::{PROTOCOL_VERSION, PROTOCOL_VERSION_KEY, SUPPORTED_INPUT_VERSIONS};
use crate::render::FrameParams;
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

/// Decodes every row of `batch` into [`FrameParams`], validating required
/// columns, types, and non-nullness (including the fixed-size-list children).
/// The optional `0.0.2` `model`/`k`/`pose` matrix columns are decoded if present.
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
    Ok((0..batch.num_rows())
        .map(|i| FrameParams {
            center: read_vec2(center, i),
            size: read_vec2(size, i),
            theta: theta.value(i),
            model: model.map(|(list, values)| read_fixed::<16>(list, values, i)),
            k: k.map(|(list, values)| read_fixed::<9>(list, values, i)),
            pose: pose.map(|(list, values)| read_fixed::<16>(list, values, i)),
        })
        .collect())
}

/// Reads an Arrow IPC frame-params stream from `input`, invoking `on_frame` for
/// each decoded [`FrameParams`] in stream order.
///
/// This is the input-side counterpart to [`run_stream`]: instead of rendering
/// each row to an output image stream, it hands every frame to the caller — for
/// example, a live window that renders it to a surface. It validates the
/// protocol version and each batch's columns, types, and non-nullness via
/// [`decode_frames`]. Returns once the stream is exhausted.
pub fn read_frame_stream<R: Read>(
    input: R,
    mut on_frame: impl FnMut(FrameParams),
) -> Result<(), StreamError> {
    let reader = StreamReader::try_new(input, None)?;
    check_version(reader.schema().as_ref())?;
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
    pipeline: wgpu::RenderPipeline,
    uniform: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    texture: wgpu::Texture,
    staging: wgpu::Buffer,
    width: u32,
    height: u32,
    padded_bytes_per_row: u32,
}

impl BatchRenderer {
    /// Builds the GPU context (instance/adapter/device/pipeline/target/readback)
    /// once for a fixed `width` x `height`.
    pub fn new(width: u32, height: u32) -> Result<Self, StreamError> {
        pollster::block_on(Self::new_async(width, height))
    }

    async fn new_async(width: u32, height: u32) -> Result<Self, StreamError> {
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
        let pipeline = crate::render::create_triangle_pipeline(&device, format);
        let (uniform, bind_group) = crate::render::create_params_binding(
            &device,
            &pipeline,
            FrameParams::IDENTITY,
            crate::render::Viewport { width, height },
        );

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
            pipeline,
            uniform,
            bind_group,
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
        crate::render::write_params(
            &self.queue,
            &self.uniform,
            params,
            crate::render::Viewport {
                width: self.width,
                height: self.height,
            },
        );
        let view = self
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("trd frame"),
            });
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("trd frame pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);
            pass.draw(0..3, 0..1);
        }
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

/// Reads an Arrow IPC stream of frame params from `input`, renders each row,
/// and writes an Arrow IPC stream of `fixed_shape_tensor` images to `output`.
/// Output batch boundaries mirror input batches (one batch in flight).
pub fn run_stream<R: Read, W: Write>(
    input: R,
    mut output: W,
    width: u32,
    height: u32,
) -> Result<(), StreamError> {
    // Validate dimensions up front so schema construction (which multiplies
    // width*height) can't overflow before BatchRenderer's guard runs.
    check_dimensions(width, height)?;

    let reader = StreamReader::try_new(input, None)?;
    check_version(reader.schema().as_ref())?;

    let mut renderer = BatchRenderer::new(width, height)?;
    let mut output_session = OutputSession::new(width, height)?;
    output.write_all(&output_session.drain_new()?)?;

    for batch in reader {
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
}
