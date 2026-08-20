//! The **output** wire format: rendered RGBA frames -> the `r`/`g`/`b`/`a`
//! planar `fixed_shape_tensor<u8>[H, W]` schema of protocol `0.0.6`.
//!
//! Pure encoding maths — the schema, the interleaved-to-planar channel split,
//! the readback row-stride unpad — plus the reader that turns the result back
//! into frames. No transport: writing the bytes somewhere is
//! [`OutputStream`](crate::OutputStream)'s job.

use std::io::Read;
use std::sync::Arc;

use arrow::array::{Array, ArrayRef, FixedSizeListArray, RecordBatch, UInt8Array};
use arrow::datatypes::{DataType, Field, Fields, Schema};
use arrow::error::ArrowError;
use arrow::ipc::reader::StreamReader;
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

/// The **semantic** half of the output protocol: the schema, and RGBA frames →
/// one [`RecordBatch`]. Owns no transport, so it is a plain non-generic type
/// usable on any platform (its `*Stream` counterpart carries the `W`).
pub(crate) fn tightly_pack_rgba(
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
/// Lets a Rust consumer round-trip a
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

    use arrow::datatypes::DataType;

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
