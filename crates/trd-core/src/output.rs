use std::cell::RefCell;
use std::io::{Read, Write};
use std::rc::Rc;
use std::sync::Arc;

use arrow::array::{Array, ArrayRef, FixedSizeListArray, RecordBatch, UInt8Array};
use arrow::datatypes::{DataType, Field, Fields, Schema};
use arrow::error::ArrowError;
use arrow::ipc::reader::StreamReader;
use arrow::ipc::writer::StreamWriter;
use arrow_schema::extension::FixedShapeTensor;
use thiserror::Error;

use crate::protocol::{FRAME_RATE_KEY, PROTOCOL_VERSION, PROTOCOL_VERSION_KEY};

#[derive(Debug, Error)]
pub enum OutputError {
    #[error("Arrow output error: {0}")]
    Arrow(#[from] ArrowError),

    #[error("invalid output dimensions {width}x{height}")]
    InvalidDimensions { width: u32, height: u32 },

    #[error("pixel count {pixels} does not fit usize")]
    PixelCountDoesNotFitUsize { pixels: u32 },

    #[error("pixel count {pixels} does not fit Arrow fixed-size-list length")]
    PixelCountDoesNotFitI32 { pixels: u32 },

    #[error("dimension {dimension} does not fit usize")]
    DimensionDoesNotFitUsize { dimension: u32 },

    #[error("RGBA byte count overflows usize for {pixels} pixels")]
    RgbaByteCountOverflow { pixels: usize },

    #[error("channel value count overflows usize")]
    ChannelValueCountOverflow,

    #[error("RGBA frame has {actual} bytes; expected {expected}")]
    InvalidRgbaFrameLength { actual: usize, expected: usize },

    #[error("image output column `{0}` is missing or has an unexpected type")]
    MalformedImage(&'static str),

    #[error("row stride {stride} does not fit usize")]
    RowStrideDoesNotFitUsize { stride: u32 },

    #[error("padded row stride {padded} is smaller than unpadded row stride {unpadded}")]
    InvalidPaddedRowStride { padded: usize, unpadded: usize },

    #[error("mapped readback byte count overflows usize")]
    MappedReadbackLengthOverflow,

    #[error("mapped readback has {actual} bytes; expected at least {expected}")]
    MappedReadbackTooShort { actual: usize, expected: usize },

    #[error("output session is already finished")]
    OutputSessionFinished,

    #[error("output session previously failed")]
    OutputSessionFailed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum OutputState {
    Open,
    Finished,
    Failed,
}

#[derive(Clone, Default)]
struct OutputBytes(Rc<RefCell<Vec<u8>>>);

impl Write for OutputBytes {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        self.0.borrow_mut().extend_from_slice(bytes);
        Ok(bytes.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

#[derive(Clone, Copy)]
struct OutputLayout {
    pixels: usize,
    rgba_bytes: usize,
    list_size: i32,
    width: usize,
    height: usize,
}

fn output_layout(width: u32, height: u32) -> Result<OutputLayout, OutputError> {
    let pixels_u32 = width
        .checked_mul(height)
        .filter(|pixels| *pixels > 0)
        .ok_or(OutputError::InvalidDimensions { width, height })?;

    let pixels = usize::try_from(pixels_u32)
        .map_err(|_| OutputError::PixelCountDoesNotFitUsize { pixels: pixels_u32 })?;
    let list_size = i32::try_from(pixels_u32)
        .map_err(|_| OutputError::PixelCountDoesNotFitI32 { pixels: pixels_u32 })?;
    let width = usize::try_from(width)
        .map_err(|_| OutputError::DimensionDoesNotFitUsize { dimension: width })?;
    let height = usize::try_from(height)
        .map_err(|_| OutputError::DimensionDoesNotFitUsize { dimension: height })?;
    let rgba_bytes = pixels
        .checked_mul(4)
        .ok_or(OutputError::RgbaByteCountOverflow { pixels })?;

    Ok(OutputLayout {
        pixels,
        rgba_bytes,
        list_size,
        width,
        height,
    })
}

fn tensor_field(name: &str, layout: OutputLayout) -> Result<Field, OutputError> {
    let storage = DataType::FixedSizeList(
        Arc::new(Field::new("item", DataType::UInt8, false)),
        layout.list_size,
    );
    let extension = FixedShapeTensor::try_new(
        DataType::UInt8,
        vec![layout.height, layout.width],
        Some(vec!["height".to_string(), "width".to_string()]),
        None,
    )?;

    Ok(Field::new(name, storage, false).with_extension_type(extension))
}

pub fn output_schema(width: u32, height: u32) -> Result<Schema, OutputError> {
    output_schema_with_frame_rate(width, height, None)
}

/// Like [`output_schema`], but also stamps the stream's playback rate
/// (`trd.stream.frame_rate`) when `frame_rate` is `Some`, so the rendered image
/// stream carries the same speed as the input for downstream encoders.
pub fn output_schema_with_frame_rate(
    width: u32,
    height: u32,
    frame_rate: Option<f64>,
) -> Result<Schema, OutputError> {
    let layout = output_layout(width, height)?;
    let fields: Fields = ["r", "g", "b", "a"]
        .into_iter()
        .map(|name| tensor_field(name, layout))
        .collect::<Result<Vec<_>, _>>()?
        .into();

    let mut metadata = std::collections::HashMap::from([(
        PROTOCOL_VERSION_KEY.to_string(),
        PROTOCOL_VERSION.to_string(),
    )]);
    if let Some(rate) = frame_rate {
        metadata.insert(FRAME_RATE_KEY.to_string(), rate.to_string());
    }

    Ok(Schema::new(fields).with_metadata(metadata))
}

fn channel_column(
    frames: &[Vec<u8>],
    channel: usize,
    layout: OutputLayout,
) -> Result<Arc<FixedSizeListArray>, OutputError> {
    let capacity = frames
        .len()
        .checked_mul(layout.pixels)
        .ok_or(OutputError::ChannelValueCountOverflow)?;
    let mut values = Vec::with_capacity(capacity);

    for frame in frames {
        for pixel in 0..layout.pixels {
            values.push(frame[pixel * 4 + channel]);
        }
    }

    Ok(Arc::new(FixedSizeListArray::new(
        Arc::new(Field::new("item", DataType::UInt8, false)),
        layout.list_size,
        Arc::new(UInt8Array::from(values)),
        None,
    )))
}

pub fn output_batch(
    schema: Arc<Schema>,
    frames: &[Vec<u8>],
    width: u32,
    height: u32,
) -> Result<RecordBatch, OutputError> {
    let layout = output_layout(width, height)?;

    for frame in frames {
        if frame.len() != layout.rgba_bytes {
            return Err(OutputError::InvalidRgbaFrameLength {
                actual: frame.len(),
                expected: layout.rgba_bytes,
            });
        }
    }

    let columns: Vec<ArrayRef> = (0..4)
        .map(|channel| Ok(channel_column(frames, channel, layout)? as ArrayRef))
        .collect::<Result<_, OutputError>>()?;

    Ok(RecordBatch::try_new(schema, columns)?)
}

pub struct OutputSession {
    schema: Arc<Schema>,
    sink: OutputBytes,
    writer: StreamWriter<OutputBytes>,
    state: OutputState,
    width: u32,
    height: u32,
}

impl OutputSession {
    pub fn new(width: u32, height: u32) -> Result<Self, OutputError> {
        Self::with_frame_rate(width, height, None)
    }

    /// Builds an output session whose schema also carries the stream playback
    /// rate (`trd.stream.frame_rate`) when `frame_rate` is `Some`.
    pub fn with_frame_rate(
        width: u32,
        height: u32,
        frame_rate: Option<f64>,
    ) -> Result<Self, OutputError> {
        let schema = Arc::new(output_schema_with_frame_rate(width, height, frame_rate)?);
        let sink = OutputBytes::default();
        let writer = StreamWriter::try_new(sink.clone(), &schema)?;

        Ok(Self {
            schema,
            sink,
            writer,
            state: OutputState::Open,
            width,
            height,
        })
    }

    fn ensure_open(&self) -> Result<(), OutputError> {
        match self.state {
            OutputState::Open => Ok(()),
            OutputState::Finished => Err(OutputError::OutputSessionFinished),
            OutputState::Failed => Err(OutputError::OutputSessionFailed),
        }
    }

    fn fail<T>(&mut self, error: OutputError) -> Result<T, OutputError> {
        self.state = OutputState::Failed;
        Err(error)
    }

    pub fn write_rgba_batch(&mut self, frames: &[Vec<u8>]) -> Result<(), OutputError> {
        self.ensure_open()?;

        let batch = match output_batch(self.schema.clone(), frames, self.width, self.height) {
            Ok(batch) => batch,
            Err(error) => return self.fail(error),
        };

        if let Err(error) = self.writer.write(&batch) {
            return self.fail(OutputError::Arrow(error));
        }

        Ok(())
    }

    pub fn finish(&mut self) -> Result<(), OutputError> {
        self.ensure_open()?;

        if let Err(error) = self.writer.finish() {
            return self.fail(OutputError::Arrow(error));
        }

        self.state = OutputState::Finished;
        Ok(())
    }

    /// Removes and returns all output bytes produced since the last drain.
    ///
    /// The `StreamWriter` appends to an in-memory sink; each call hands back the
    /// bytes accumulated so far (schema, record batches, or EOS) so the caller
    /// can forward them incrementally. Fails if the session previously failed,
    /// so a partially-written batch is never handed back as success-shaped bytes.
    pub fn drain_new(&mut self) -> Result<Vec<u8>, OutputError> {
        if self.state == OutputState::Failed {
            return Err(OutputError::OutputSessionFailed);
        }

        Ok(std::mem::take(&mut *self.sink.0.borrow_mut()))
    }
}

pub fn tightly_pack_rgba(
    mapped: &[u8],
    width: u32,
    height: u32,
    padded_bytes_per_row: u32,
) -> Result<Vec<u8>, OutputError> {
    let layout = output_layout(width, height)?;
    let padded = usize::try_from(padded_bytes_per_row).map_err(|_| {
        OutputError::RowStrideDoesNotFitUsize {
            stride: padded_bytes_per_row,
        }
    })?;
    let unpadded = layout
        .width
        .checked_mul(4)
        .ok_or(OutputError::RgbaByteCountOverflow {
            pixels: layout.width,
        })?;

    if padded < unpadded {
        return Err(OutputError::InvalidPaddedRowStride { padded, unpadded });
    }

    let required = padded
        .checked_mul(layout.height)
        .ok_or(OutputError::MappedReadbackLengthOverflow)?;

    if mapped.len() < required {
        return Err(OutputError::MappedReadbackTooShort {
            actual: mapped.len(),
            expected: required,
        });
    }

    let mut rgba = Vec::with_capacity(layout.rgba_bytes);
    for row in 0..layout.height {
        let start = row * padded;
        rgba.extend_from_slice(&mapped[start..start + unpadded]);
    }

    Ok(rgba)
}

/// Decodes a rendered **image** Arrow IPC stream (the bytes written by
/// [`OutputSession`] / [`crate::run_stream`]) back into one tightly-packed
/// row-major RGBA frame per row (`width * height * 4` bytes each). The inverse of
/// [`output_batch`]: it reads the four `r`/`g`/`b`/`a`
/// `FixedSizeList<UInt8>[width*height]` channel columns and interleaves them.
///
/// Lets a Rust consumer (e.g. the GUI's `ArrowRoundTripRenderer`) round-trip a
/// scene through `run_stream` in-process and read the frames back without going
/// out to an external image encoder.
pub fn read_image_stream<R: Read>(
    reader: R,
    width: u32,
    height: u32,
) -> Result<Vec<Vec<u8>>, OutputError> {
    let layout = output_layout(width, height)?;
    let stream = StreamReader::try_new(reader, None)?;
    let mut frames = Vec::new();
    for batch in stream {
        let batch = batch?;
        let channels: Vec<&FixedSizeListArray> = ["r", "g", "b", "a"]
            .into_iter()
            .map(|name| {
                batch
                    .column_by_name(name)
                    .and_then(|c| c.as_any().downcast_ref::<FixedSizeListArray>())
                    .filter(|c| c.value_length() == layout.list_size)
                    .ok_or(OutputError::MalformedImage(match name {
                        "r" => "r",
                        "g" => "g",
                        "b" => "b",
                        _ => "a",
                    }))
            })
            .collect::<Result<_, _>>()?;

        for row in 0..batch.num_rows() {
            let mut rgba = vec![0u8; layout.rgba_bytes];
            for (channel, list) in channels.iter().enumerate() {
                let values_ref = list.value(row);
                let values = values_ref
                    .as_any()
                    .downcast_ref::<UInt8Array>()
                    .ok_or(OutputError::MalformedImage("channel"))?;
                for pixel in 0..layout.pixels {
                    rgba[pixel * 4 + channel] = values.value(pixel);
                }
            }
            frames.push(rgba);
        }
    }
    Ok(frames)
}

#[cfg(test)]
mod tests {
    use super::*;

    use arrow::array::RecordBatch;
    use arrow::datatypes::DataType;
    use arrow::ipc::reader::StreamReader;

    use crate::{PROTOCOL_VERSION, PROTOCOL_VERSION_KEY};

    #[test]
    fn output_schema_is_fixed_shape_tensor() {
        let schema = output_schema(4, 3).unwrap();
        assert_eq!(
            schema
                .metadata()
                .get(PROTOCOL_VERSION_KEY)
                .map(String::as_str),
            Some(PROTOCOL_VERSION)
        );
        for name in ["r", "g", "b", "a"] {
            let field = schema.field_with_name(name).unwrap();
            assert_eq!(
                field
                    .metadata()
                    .get("ARROW:extension:name")
                    .map(String::as_str),
                Some("arrow.fixed_shape_tensor")
            );
            match field.data_type() {
                DataType::FixedSizeList(_, 12) => {}
                other => panic!("unexpected storage type: {other:?}"),
            }
        }
    }

    #[test]
    fn output_session_drains_schema_batches_and_eos_once() {
        let mut output = OutputSession::new(2, 1).unwrap();

        let schema = output.drain_new().unwrap();
        assert!(!schema.is_empty());
        assert!(output.drain_new().unwrap().is_empty());

        output
            .write_rgba_batch(&[vec![1, 2, 3, 255, 4, 5, 6, 255]])
            .unwrap();
        let first = output.drain_new().unwrap();

        output
            .write_rgba_batch(&[vec![7, 8, 9, 255, 10, 11, 12, 255]])
            .unwrap();
        let second = output.drain_new().unwrap();

        output.finish().unwrap();
        let eos = output.drain_new().unwrap();

        let bytes = [schema, first, second, eos].concat();
        let reader = StreamReader::try_new(bytes.as_slice(), None).unwrap();

        assert_eq!(
            reader
                .schema()
                .metadata()
                .get(PROTOCOL_VERSION_KEY)
                .map(String::as_str),
            Some(PROTOCOL_VERSION)
        );

        let batches = reader.collect::<Result<Vec<_>, _>>().unwrap();
        assert_eq!(
            batches
                .iter()
                .map(RecordBatch::num_rows)
                .collect::<Vec<_>>(),
            vec![1, 1]
        );
    }

    #[test]
    fn read_image_stream_roundtrips_written_frames() {
        let (w, h) = (2u32, 2u32);
        let frame0: Vec<u8> = (0..(w * h * 4) as u8).collect();
        let frame1: Vec<u8> = (0..(w * h * 4) as u8).map(|b| 255 - b).collect();

        let mut output = OutputSession::new(w, h).unwrap();
        let mut bytes = output.drain_new().unwrap();
        output
            .write_rgba_batch(&[frame0.clone(), frame1.clone()])
            .unwrap();
        bytes.extend(output.drain_new().unwrap());
        output.finish().unwrap();
        bytes.extend(output.drain_new().unwrap());

        let frames = read_image_stream(bytes.as_slice(), w, h).unwrap();
        assert_eq!(frames, vec![frame0, frame1]);
    }

    #[test]
    fn output_session_drain_releases_drained_bytes() {
        let mut output = OutputSession::new(2, 1).unwrap();

        assert!(!output.drain_new().unwrap().is_empty());
        assert!(output.sink.0.borrow().is_empty());

        output
            .write_rgba_batch(&[vec![1, 2, 3, 255, 4, 5, 6, 255]])
            .unwrap();
        assert!(!output.drain_new().unwrap().is_empty());
        assert!(output.sink.0.borrow().is_empty());
    }

    #[test]
    fn output_session_finish_without_batches_emits_eos() {
        let mut output = OutputSession::new(2, 1).unwrap();

        let schema = output.drain_new().unwrap();
        output.finish().unwrap();
        let eos = output.drain_new().unwrap();

        let bytes = [schema, eos].concat();
        let reader = StreamReader::try_new(bytes.as_slice(), None).unwrap();
        assert!(reader.collect::<Result<Vec<_>, _>>().unwrap().is_empty());

        assert!(matches!(
            output.write_rgba_batch(&[vec![0; 8]]),
            Err(OutputError::OutputSessionFinished)
        ));
        assert!(matches!(
            output.finish(),
            Err(OutputError::OutputSessionFinished)
        ));
    }

    #[test]
    fn output_session_is_terminal_after_batch_failure() {
        let mut output = OutputSession::new(2, 1).unwrap();

        assert!(matches!(
            output.write_rgba_batch(&[vec![0; 7]]),
            Err(OutputError::InvalidRgbaFrameLength {
                actual: 7,
                expected: 8
            })
        ));
        assert!(matches!(
            output.drain_new(),
            Err(OutputError::OutputSessionFailed)
        ));
        assert!(matches!(
            output.finish(),
            Err(OutputError::OutputSessionFailed)
        ));
    }

    #[test]
    fn tightly_packed_rgba_removes_row_padding() {
        let mapped = [
            1, 2, 3, 4, 5, 6, 7, 8, 0, 0, 0, 0, 9, 10, 11, 12, 13, 14, 15, 16, 0, 0, 0, 0,
        ];

        assert_eq!(
            tightly_pack_rgba(&mapped, 2, 2, 12).unwrap(),
            vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
        );
    }
}
