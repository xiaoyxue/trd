//! Inline background-frame resources for protocol `0.0.6`.
//!
//! A `frames` table precedes the terminal params table. Each row stores one
//! encoded PNG/JPEG or one raw `arrow.fixed_shape_tensor<u8>[H,W,C]` image;
//! params rows reference resources by `frame_id`.

use arrow::array::{Array, BinaryArray, FixedSizeListArray, RecordBatch, UInt8Array};
use arrow::datatypes::DataType;
use thiserror::Error;

use crate::texture::{
    parse_tensor_shape, ImageData, EXTENSION_METADATA_KEY, EXTENSION_NAME_KEY, FIXED_SHAPE_TENSOR,
};

pub const FRAME_BYTES_COLUMN: &str = "frame_bytes";
pub const FRAME_PIXELS_COLUMN: &str = "frame_pixels";

/// One frames-table resource. Encoded images stay compressed until selected;
/// raw pixels are normalized to tightly packed RGBA during Arrow decode.
#[derive(Debug, Clone, PartialEq)]
pub enum InlineFrame {
    Encoded(Vec<u8>),
    Pixels(ImageData),
}

impl InlineFrame {
    /// Decodes this resource to tightly packed row-major RGBA8.
    pub fn decode(&self) -> Result<ImageData, FrameError> {
        match self {
            Self::Pixels(image) => Ok(image.clone()),
            Self::Encoded(bytes) => {
                let format = image::guess_format(bytes)?;
                if !matches!(format, image::ImageFormat::Png | image::ImageFormat::Jpeg) {
                    return Err(FrameError::UnsupportedEncoding(format!("{format:?}")));
                }
                let rgba = image::load_from_memory_with_format(bytes, format)?.to_rgba8();
                let (width, height) = rgba.dimensions();
                Ok(ImageData {
                    width,
                    height,
                    rgba: rgba.into_raw(),
                })
            }
        }
    }

    /// Decodes every row of a frames-table record batch.
    pub(crate) fn from_arrow_all(batch: &RecordBatch) -> Result<Vec<Self>, FrameError> {
        let encoded = optional_binary(batch)?;
        let pixels = optional_pixels(batch)?;
        if encoded.is_none() && pixels.is_none() {
            return Err(FrameError::MissingPayloadColumns);
        }

        let mut frames = Vec::with_capacity(batch.num_rows());
        for row in 0..batch.num_rows() {
            let bytes =
                encoded.and_then(|array| (!array.is_null(row)).then(|| array.value(row).to_vec()));
            let image = pixels
                .as_ref()
                .and_then(|column| (!column.array.is_null(row)).then(|| column.decode_row(row)))
                .transpose()?;

            match (bytes, image) {
                (Some(bytes), None) if bytes.is_empty() => {
                    return Err(FrameError::EmptyEncoded { row });
                }
                (Some(bytes), None) => frames.push(Self::Encoded(bytes)),
                (None, Some(image)) => frames.push(Self::Pixels(image)),
                (None, None) => {
                    return Err(FrameError::PayloadCount { row, actual: 0 });
                }
                (Some(_), Some(_)) => {
                    return Err(FrameError::PayloadCount { row, actual: 2 });
                }
            }
        }
        Ok(frames)
    }
}

#[derive(Debug, Error)]
pub enum FrameError {
    #[error("frames table must contain `{FRAME_BYTES_COLUMN}` and/or `{FRAME_PIXELS_COLUMN}`")]
    MissingPayloadColumns,
    #[error("frames column `{column}` has type {actual:?}, expected {expected}")]
    ColumnType {
        column: &'static str,
        expected: &'static str,
        actual: DataType,
    },
    #[error("frames column `{FRAME_PIXELS_COLUMN}` is not a fixed_shape_tensor: {0}")]
    NotTensor(String),
    #[error(
        "frames tensor shape {shape:?} is not [H, W, C] with non-zero H/W and C equal to 3 or 4"
    )]
    Shape { shape: Vec<usize> },
    #[error("frames column `{0}` contains null child values")]
    NullValues(&'static str),
    #[error("frames row {row} has {actual} payloads; expected exactly one")]
    PayloadCount { row: usize, actual: usize },
    #[error("frames row {row} has an empty encoded-image payload")]
    EmptyEncoded { row: usize },
    #[error("frames tensor byte length {actual} != {width}x{height}x{channels} = {expected}")]
    ByteLength {
        actual: usize,
        expected: usize,
        width: u32,
        height: u32,
        channels: usize,
    },
    #[error("unsupported inline-frame encoding `{0}` (expected PNG or JPEG)")]
    UnsupportedEncoding(String),
    #[error("inline-frame image decode failed: {0}")]
    Image(#[from] image::ImageError),
}

fn optional_binary(batch: &RecordBatch) -> Result<Option<&BinaryArray>, FrameError> {
    let Some(column) = batch.column_by_name(FRAME_BYTES_COLUMN) else {
        return Ok(None);
    };
    column
        .as_any()
        .downcast_ref::<BinaryArray>()
        .map(Some)
        .ok_or_else(|| FrameError::ColumnType {
            column: FRAME_BYTES_COLUMN,
            expected: "Binary",
            actual: column.data_type().clone(),
        })
}

struct PixelColumn<'a> {
    array: &'a FixedSizeListArray,
    width: u32,
    height: u32,
    channels: usize,
    expected: usize,
}

impl PixelColumn<'_> {
    fn decode_row(&self, row: usize) -> Result<ImageData, FrameError> {
        let value = self.array.value(row);
        let bytes =
            value
                .as_any()
                .downcast_ref::<UInt8Array>()
                .ok_or_else(|| FrameError::ColumnType {
                    column: FRAME_PIXELS_COLUMN,
                    expected: "FixedSizeList<UInt8>",
                    actual: value.data_type().clone(),
                })?;
        if bytes.null_count() > 0 {
            return Err(FrameError::NullValues(FRAME_PIXELS_COLUMN));
        }
        if bytes.len() != self.expected {
            return Err(FrameError::ByteLength {
                actual: bytes.len(),
                expected: self.expected,
                width: self.width,
                height: self.height,
                channels: self.channels,
            });
        }

        let rgba = if self.channels == 4 {
            bytes.values().to_vec()
        } else {
            let mut rgba = Vec::with_capacity(self.width as usize * self.height as usize * 4);
            for rgb in bytes.values().chunks_exact(3) {
                rgba.extend_from_slice(rgb);
                rgba.push(255);
            }
            rgba
        };
        Ok(ImageData {
            width: self.width,
            height: self.height,
            rgba,
        })
    }
}

fn optional_pixels(batch: &RecordBatch) -> Result<Option<PixelColumn<'_>>, FrameError> {
    let Ok(index) = batch.schema().index_of(FRAME_PIXELS_COLUMN) else {
        return Ok(None);
    };
    let schema = batch.schema();
    let field = schema.field(index);
    let metadata = field.metadata();
    if metadata.get(EXTENSION_NAME_KEY).map(String::as_str) != Some(FIXED_SHAPE_TENSOR) {
        return Err(FrameError::NotTensor(format!(
            "column `{FRAME_PIXELS_COLUMN}` is not `{FIXED_SHAPE_TENSOR}`"
        )));
    }
    let shape = metadata
        .get(EXTENSION_METADATA_KEY)
        .and_then(|json| parse_tensor_shape(json))
        .ok_or_else(|| {
            FrameError::NotTensor("missing/invalid fixed_shape_tensor shape metadata".to_string())
        })?;
    let (height, width, channels) = match shape.as_slice() {
        [height, width, channels] if *height > 0 && *width > 0 && matches!(*channels, 3 | 4) => {
            (*height, *width, *channels)
        }
        _ => return Err(FrameError::Shape { shape }),
    };
    let width = u32::try_from(width).map_err(|_| FrameError::Shape {
        shape: shape.clone(),
    })?;
    let height = u32::try_from(height).map_err(|_| FrameError::Shape {
        shape: shape.clone(),
    })?;
    let expected = (width as usize)
        .checked_mul(height as usize)
        .and_then(|pixels| pixels.checked_mul(channels))
        .ok_or_else(|| FrameError::Shape {
            shape: shape.clone(),
        })?;

    let column = batch.column(index);
    let array = column
        .as_any()
        .downcast_ref::<FixedSizeListArray>()
        .ok_or_else(|| FrameError::ColumnType {
            column: FRAME_PIXELS_COLUMN,
            expected: "FixedSizeList<UInt8>",
            actual: column.data_type().clone(),
        })?;
    if array.value_length() as usize != expected {
        return Err(FrameError::ByteLength {
            actual: array.value_length() as usize,
            expected,
            width,
            height,
            channels,
        });
    }
    if array.values().null_count() > 0 {
        return Err(FrameError::NullValues(FRAME_PIXELS_COLUMN));
    }
    Ok(Some(PixelColumn {
        array,
        width,
        height,
        channels,
        expected,
    }))
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;
    use std::sync::Arc;

    use arrow::array::{ArrayRef, BinaryArray};
    use arrow::buffer::{BooleanBuffer, NullBuffer};
    use arrow::datatypes::{Field, Schema};
    use arrow_schema::extension::FixedShapeTensor;
    use image::{DynamicImage, ImageBuffer, ImageFormat, Rgb};

    use super::*;

    fn encoded_image(format: ImageFormat) -> Vec<u8> {
        let image = match format {
            ImageFormat::Jpeg => {
                DynamicImage::ImageRgb8(ImageBuffer::from_pixel(2, 1, Rgb([40, 120, 220])))
            }
            _ => DynamicImage::ImageRgba8(
                ImageBuffer::from_raw(2, 1, vec![1, 2, 3, 4, 10, 20, 30, 40]).unwrap(),
            ),
        };
        let mut output = Cursor::new(Vec::new());
        image.write_to(&mut output, format).unwrap();
        output.into_inner()
    }

    fn pixel_field(shape: Vec<usize>, list_size: i32) -> Field {
        let storage = DataType::FixedSizeList(
            Arc::new(Field::new("item", DataType::UInt8, false)),
            list_size,
        );
        let extension = FixedShapeTensor::try_new(DataType::UInt8, shape, None, None).unwrap();
        Field::new(FRAME_PIXELS_COLUMN, storage, true).with_extension_type(extension)
    }

    fn unchecked_pixel_field(shape: &[usize], list_size: i32, child_nullable: bool) -> Field {
        let storage = DataType::FixedSizeList(
            Arc::new(Field::new("item", DataType::UInt8, child_nullable)),
            list_size,
        );
        let metadata = [
            (
                EXTENSION_NAME_KEY.to_string(),
                FIXED_SHAPE_TENSOR.to_string(),
            ),
            (
                EXTENSION_METADATA_KEY.to_string(),
                format!(
                    r#"{{"shape":[{}]}}"#,
                    shape
                        .iter()
                        .map(usize::to_string)
                        .collect::<Vec<_>>()
                        .join(",")
                ),
            ),
        ]
        .into_iter()
        .collect();
        Field::new(FRAME_PIXELS_COLUMN, storage, true).with_metadata(metadata)
    }

    fn pixels_batch(shape: Vec<usize>, rows: Vec<Option<Vec<u8>>>) -> RecordBatch {
        let list_size = rows
            .iter()
            .find_map(|row| row.as_ref().map(Vec::len))
            .unwrap_or_else(|| shape.iter().product());
        let mut flat = Vec::with_capacity(rows.len() * list_size);
        let mut valid = Vec::with_capacity(rows.len());
        for row in rows {
            valid.push(row.is_some());
            flat.extend(row.unwrap_or_else(|| vec![0; list_size]));
        }
        let array = FixedSizeListArray::new(
            Arc::new(Field::new("item", DataType::UInt8, false)),
            list_size as i32,
            Arc::new(UInt8Array::from(flat)),
            Some(NullBuffer::new(BooleanBuffer::from(valid))),
        );
        let schema = Arc::new(Schema::new(vec![pixel_field(shape, list_size as i32)]));
        RecordBatch::try_new(schema, vec![Arc::new(array)]).unwrap()
    }

    fn mixed_batch(
        encoded: Vec<Option<Vec<u8>>>,
        pixels: Vec<Option<Vec<u8>>>,
        shape: Vec<usize>,
    ) -> RecordBatch {
        assert_eq!(encoded.len(), pixels.len());
        let list_size: usize = shape.iter().product();
        let bytes = BinaryArray::from_iter(encoded.iter().map(Option::as_deref));
        let mut flat = Vec::with_capacity(pixels.len() * list_size);
        let mut valid = Vec::with_capacity(pixels.len());
        for row in pixels {
            valid.push(row.is_some());
            flat.extend(row.unwrap_or_else(|| vec![0; list_size]));
        }
        let pixel_array = FixedSizeListArray::new(
            Arc::new(Field::new("item", DataType::UInt8, false)),
            list_size as i32,
            Arc::new(UInt8Array::from(flat)),
            Some(NullBuffer::new(BooleanBuffer::from(valid))),
        );
        let schema = Arc::new(Schema::new(vec![
            Field::new(FRAME_BYTES_COLUMN, DataType::Binary, true),
            pixel_field(shape, list_size as i32),
        ]));
        RecordBatch::try_new(
            schema,
            vec![Arc::new(bytes) as ArrayRef, Arc::new(pixel_array)],
        )
        .unwrap()
    }

    #[test]
    fn encoded_png_decodes_exact_rgba() {
        let bytes = encoded_image(ImageFormat::Png);
        let image = InlineFrame::Encoded(bytes).decode().unwrap();
        assert_eq!((image.width, image.height), (2, 1));
        assert_eq!(image.rgba, vec![1, 2, 3, 4, 10, 20, 30, 40]);
    }

    #[test]
    fn encoded_jpeg_decodes_to_rgba() {
        let bytes = encoded_image(ImageFormat::Jpeg);
        let image = InlineFrame::Encoded(bytes).decode().unwrap();
        assert_eq!((image.width, image.height), (2, 1));
        assert_eq!(image.rgba.len(), 8);
        assert_eq!(image.rgba[3], 255);
        assert_eq!(image.rgba[7], 255);
    }

    #[test]
    fn raw_rgb_expands_opaque_alpha() {
        let batch = pixels_batch(vec![1, 2, 3], vec![Some(vec![1, 2, 3, 10, 20, 30])]);
        let frames = InlineFrame::from_arrow_all(&batch).unwrap();
        assert_eq!(
            frames,
            vec![InlineFrame::Pixels(ImageData {
                width: 2,
                height: 1,
                rgba: vec![1, 2, 3, 255, 10, 20, 30, 255],
            })]
        );
    }

    #[test]
    fn raw_rgba_is_byte_exact() {
        let rgba = vec![1, 2, 3, 4, 10, 20, 30, 40];
        let batch = pixels_batch(vec![1, 2, 4], vec![Some(rgba.clone())]);
        let frames = InlineFrame::from_arrow_all(&batch).unwrap();
        assert_eq!(
            frames,
            vec![InlineFrame::Pixels(ImageData {
                width: 2,
                height: 1,
                rgba,
            })]
        );
    }

    #[test]
    fn mixed_binary_and_tensor_rows_decode() {
        let png = encoded_image(ImageFormat::Png);
        let batch = mixed_batch(
            vec![Some(png), None],
            vec![None, Some(vec![9, 8, 7, 6, 5, 4, 3, 2])],
            vec![1, 2, 4],
        );
        let frames = InlineFrame::from_arrow_all(&batch).unwrap();
        assert!(matches!(frames[0], InlineFrame::Encoded(_)));
        assert_eq!(
            frames[1],
            InlineFrame::Pixels(ImageData {
                width: 2,
                height: 1,
                rgba: vec![9, 8, 7, 6, 5, 4, 3, 2],
            })
        );
    }

    #[test]
    fn missing_payload_columns_is_rejected() {
        let schema = Arc::new(Schema::new(vec![Field::new(
            "other",
            DataType::UInt8,
            false,
        )]));
        let batch =
            RecordBatch::try_new(schema, vec![Arc::new(UInt8Array::from(vec![1]))]).unwrap();
        assert!(matches!(
            InlineFrame::from_arrow_all(&batch),
            Err(FrameError::MissingPayloadColumns)
        ));
    }

    #[test]
    fn wrong_binary_type_is_rejected() {
        let schema = Arc::new(Schema::new(vec![Field::new(
            FRAME_BYTES_COLUMN,
            DataType::UInt8,
            false,
        )]));
        let batch =
            RecordBatch::try_new(schema, vec![Arc::new(UInt8Array::from(vec![1]))]).unwrap();
        assert!(matches!(
            InlineFrame::from_arrow_all(&batch),
            Err(FrameError::ColumnType {
                column: FRAME_BYTES_COLUMN,
                ..
            })
        ));
    }

    #[test]
    fn missing_tensor_metadata_is_rejected() {
        let array = FixedSizeListArray::new(
            Arc::new(Field::new("item", DataType::UInt8, false)),
            4,
            Arc::new(UInt8Array::from(vec![1, 2, 3, 4])),
            None,
        );
        let schema = Arc::new(Schema::new(vec![Field::new(
            FRAME_PIXELS_COLUMN,
            array.data_type().clone(),
            false,
        )]));
        let batch = RecordBatch::try_new(schema, vec![Arc::new(array)]).unwrap();
        assert!(matches!(
            InlineFrame::from_arrow_all(&batch),
            Err(FrameError::NotTensor(_))
        ));
    }

    #[test]
    fn invalid_tensor_shape_is_rejected() {
        let batch = pixels_batch(vec![1, 1, 2], vec![Some(vec![1, 2])]);
        assert!(matches!(
            InlineFrame::from_arrow_all(&batch),
            Err(FrameError::Shape { .. })
        ));
    }

    #[test]
    fn tensor_shape_and_storage_length_must_match() {
        let array = FixedSizeListArray::new(
            Arc::new(Field::new("item", DataType::UInt8, false)),
            8,
            Arc::new(UInt8Array::from(vec![0; 8])),
            None,
        );
        let schema = Arc::new(Schema::new(vec![unchecked_pixel_field(
            &[1, 1, 4],
            8,
            false,
        )]));
        let batch = RecordBatch::try_new(schema, vec![Arc::new(array)]).unwrap();
        assert!(matches!(
            InlineFrame::from_arrow_all(&batch),
            Err(FrameError::ByteLength {
                actual: 8,
                expected: 4,
                ..
            })
        ));
    }

    #[test]
    fn each_row_requires_exactly_one_payload() {
        let png = encoded_image(ImageFormat::Png);
        let both = mixed_batch(
            vec![Some(png.clone())],
            vec![Some(vec![0; 4])],
            vec![1, 1, 4],
        );
        assert!(matches!(
            InlineFrame::from_arrow_all(&both),
            Err(FrameError::PayloadCount { row: 0, actual: 2 })
        ));

        let neither = mixed_batch(vec![None], vec![None], vec![1, 1, 4]);
        assert!(matches!(
            InlineFrame::from_arrow_all(&neither),
            Err(FrameError::PayloadCount { row: 0, actual: 0 })
        ));
    }

    #[test]
    fn empty_and_malformed_encoded_payloads_are_rejected() {
        let empty = BinaryArray::from(vec![Some(&[][..])]);
        let schema = Arc::new(Schema::new(vec![Field::new(
            FRAME_BYTES_COLUMN,
            DataType::Binary,
            true,
        )]));
        let batch = RecordBatch::try_new(schema, vec![Arc::new(empty)]).unwrap();
        assert!(matches!(
            InlineFrame::from_arrow_all(&batch),
            Err(FrameError::EmptyEncoded { row: 0 })
        ));

        let malformed_png = InlineFrame::Encoded(vec![137, 80, 78, 71, 13, 10, 26, 10]);
        assert!(matches!(malformed_png.decode(), Err(FrameError::Image(_))));
        let gif = InlineFrame::Encoded(b"GIF89a".to_vec());
        assert!(matches!(
            gif.decode(),
            Err(FrameError::UnsupportedEncoding(_))
        ));
    }

    #[test]
    fn tensor_child_nulls_are_rejected() {
        let values = UInt8Array::from(vec![Some(1), Some(2), None, Some(4)]);
        let array = FixedSizeListArray::new(
            Arc::new(Field::new("item", DataType::UInt8, true)),
            4,
            Arc::new(values),
            None,
        );
        let schema = Arc::new(Schema::new(vec![unchecked_pixel_field(
            &[1, 1, 4],
            4,
            true,
        )]));
        let batch = RecordBatch::try_new(schema, vec![Arc::new(array)]).unwrap();
        assert!(matches!(
            InlineFrame::from_arrow_all(&batch),
            Err(FrameError::NullValues(FRAME_PIXELS_COLUMN))
        ));
    }
}
