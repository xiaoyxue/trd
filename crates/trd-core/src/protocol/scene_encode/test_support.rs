use super::*;
use crate::protocol::FRAMES_TABLE_KIND;
use crate::{InlineFrame, FRAME_BYTES_COLUMN, FRAME_PIXELS_COLUMN};
use arrow::array::{BinaryArray, FixedSizeListArray, UInt8Array};
use arrow::buffer::{BooleanBuffer, NullBuffer};

/// Authors an embedded-geometry mesh stream with default materials.
pub(crate) fn encode_mesh_stream(meshes: &[Mesh]) -> Result<Vec<u8>, SceneEncodeError> {
    let materials = vec![DisneyMaterial::default(); meshes.len()];
    let resources = meshes
        .iter()
        .zip(&materials)
        .map(|(mesh, material)| SceneMesh::Embedded { mesh, material })
        .collect::<Vec<_>>();
    encode_mesh_resources(&resources)
}

/// Authors params without frame-resource or sparse-video references.
pub(crate) fn encode_params_stream(
    frames: &[FrameParams],
    draws: Option<&[Vec<Draw>]>,
) -> Result<Vec<u8>, SceneEncodeError> {
    encode_params_stream_with_frame_ids_and_rate(frames, draws, None, None, None, None)
}

/// Authors params with optional nullable references into a preceding frames table.
pub(crate) fn encode_params_stream_with_frame_ids(
    frames: &[FrameParams],
    draws: Option<&[Vec<Draw>]>,
    frame_ids: Option<&[Option<u32>]>,
) -> Result<Vec<u8>, SceneEncodeError> {
    encode_params_stream_with_frame_ids_and_rate(frames, draws, frame_ids, None, None, None)
}

/// Authors an inline background `frames` resource table.
pub(crate) fn encode_frames_stream(frames: &[InlineFrame]) -> Result<Vec<u8>, SceneEncodeError> {
    if frames.is_empty() {
        return Err(SceneEncodeError::EmptyFrames);
    }

    let has_encoded = frames
        .iter()
        .any(|frame| matches!(frame, InlineFrame::Encoded(_)));
    let pixel_shape = frames.iter().find_map(|frame| match frame {
        InlineFrame::Pixels(image) => Some((image.width, image.height)),
        InlineFrame::Encoded(_) => None,
    });
    let mut fields = Vec::new();
    let mut columns: Vec<ArrayRef> = Vec::new();

    if has_encoded {
        let rows: Vec<Option<&[u8]>> = frames
            .iter()
            .enumerate()
            .map(|(row, frame)| match frame {
                InlineFrame::Encoded(bytes) if bytes.is_empty() => {
                    Err(SceneEncodeError::EmptyEncodedFrame { row })
                }
                InlineFrame::Encoded(bytes) => Ok(Some(bytes.as_slice())),
                InlineFrame::Pixels(_) => Ok(None),
            })
            .collect::<Result<_, _>>()?;
        fields.push(Field::new(FRAME_BYTES_COLUMN, DataType::Binary, true));
        columns.push(Arc::new(BinaryArray::from_iter(rows)));
    }

    if let Some((width, height)) = pixel_shape {
        if width == 0 || height == 0 {
            let row = frames
                .iter()
                .position(|frame| matches!(frame, InlineFrame::Pixels(_)))
                .expect("pixel_shape came from a pixel frame");
            return Err(SceneEncodeError::InvalidFrameDimensions { row, width, height });
        }
        let expected = (width as usize)
            .checked_mul(height as usize)
            .and_then(|pixels| pixels.checked_mul(4))
            .ok_or(SceneEncodeError::InvalidFramePixels {
                row: 0,
                actual: 0,
                expected: 0,
                width,
                height,
            })?;
        let list_size =
            i32::try_from(expected).map_err(|_| SceneEncodeError::InvalidFramePixels {
                row: 0,
                actual: expected,
                expected,
                width,
                height,
            })?;
        let capacity =
            expected
                .checked_mul(frames.len())
                .ok_or(SceneEncodeError::InvalidFramePixels {
                    row: 0,
                    actual: expected,
                    expected,
                    width,
                    height,
                })?;
        let mut flat = Vec::with_capacity(capacity);
        let mut valid = Vec::with_capacity(frames.len());
        for (row, frame) in frames.iter().enumerate() {
            match frame {
                InlineFrame::Encoded(_) => {
                    valid.push(false);
                    flat.resize(flat.len() + expected, 0);
                }
                InlineFrame::Pixels(image) => {
                    if (image.width, image.height) != (width, height) {
                        return Err(SceneEncodeError::MixedFrameDimensions {
                            row,
                            width: image.width,
                            height: image.height,
                            expected_width: width,
                            expected_height: height,
                        });
                    }
                    if image.rgba.len() != expected {
                        return Err(SceneEncodeError::InvalidFramePixels {
                            row,
                            actual: image.rgba.len(),
                            expected,
                            width,
                            height,
                        });
                    }
                    valid.push(true);
                    flat.extend_from_slice(&image.rgba);
                }
            }
        }

        let storage = DataType::FixedSizeList(
            Arc::new(Field::new("item", DataType::UInt8, false)),
            list_size,
        );
        let extension = arrow_schema::extension::FixedShapeTensor::try_new(
            DataType::UInt8,
            vec![height as usize, width as usize, 4],
            Some(vec![
                "height".to_string(),
                "width".to_string(),
                "channel".to_string(),
            ]),
            None,
        )?;
        fields.push(Field::new(FRAME_PIXELS_COLUMN, storage, true).with_extension_type(extension));
        columns.push(Arc::new(FixedSizeListArray::new(
            Arc::new(Field::new("item", DataType::UInt8, false)),
            list_size,
            Arc::new(UInt8Array::from(flat)),
            Some(NullBuffer::new(BooleanBuffer::from(valid))),
        )));
    }

    let schema = Schema::new(fields).with_metadata(table_metadata(FRAMES_TABLE_KIND));
    let batch = RecordBatch::try_new(Arc::new(schema.clone()), columns)?;
    write_ipc(&schema, &batch)
}

/// Authors a complete scene with inline frame resources.
pub(crate) fn encode_scene_with_frames(
    meshes: &[Mesh],
    inline_frames: &[InlineFrame],
    frames: &[FrameParams],
    draws: Option<&[Vec<Draw>]>,
    frame_ids: &[Option<u32>],
) -> Result<Vec<u8>, SceneEncodeError> {
    if let Some(frame_id) = frame_ids
        .iter()
        .flatten()
        .copied()
        .find(|frame_id| *frame_id as usize >= inline_frames.len())
    {
        return Err(SceneEncodeError::FrameIdOutOfRange {
            frame_id,
            frame_count: inline_frames.len(),
        });
    }
    let mut bytes = encode_mesh_stream(meshes)?;
    bytes.extend(encode_frames_stream(inline_frames)?);
    bytes.extend(encode_params_stream_with_frame_ids(
        frames,
        draws,
        Some(frame_ids),
    )?);
    Ok(bytes)
}
