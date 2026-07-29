//! Authoring the trd **input** stream in Rust (#97, Slice 3b).
//!
//! `trd-core` decodes the mesh-first `[mesh][texture?][params]` input stream
//! (`Mesh::from_arrow_all`, [`crate::decode_frames`], the wasm
//! [`crate::InputSession`]) and encodes the **image output** ([`crate::OutputSession`]).
//! The **input** stream is normally authored by the Python producers
//! (`scripts/*_to_arrow.py`); this module authors the same
//! `0.0.5` stream **in Rust** so an in-process front-end (the GUI's
//! `ArrowRoundTripRenderer`) — or any Rust producer — can drive
//! [`crate::run_stream`] without shelling out to Python.
//!
//! It mirrors the decoders exactly (column names, nested Arrow types, and the
//! `trd.protocol.version` metadata), and is covered by a round-trip test that
//! feeds the bytes back through the real decoders, so it can't silently drift
//! from the wire format.
//!
//! The stream is **mesh-first**: [`encode_mesh_stream`] writes the leading mesh
//! table (its own complete Arrow IPC stream) and [`encode_params_stream`] writes
//! the following params stream; [`encode_scene`] concatenates them (the framing
//! [`run_stream`](crate::run_stream) expects). Texture tables are not authored
//! here yet (the GUI's Textured mode uses the in-process backend).

use std::sync::Arc;

use arrow::array::{
    ArrayRef, FixedSizeListArray, Float32Array, ListArray, RecordBatch, UInt32Array, UInt8Array,
};
use arrow::buffer::OffsetBuffer;
use arrow::datatypes::{DataType, Field, Schema};
use arrow::ipc::writer::StreamWriter;
use thiserror::Error;

use crate::protocol::{PROTOCOL_VERSION, PROTOCOL_VERSION_KEY};
use crate::render::{Draw, FrameParams, Mesh, RenderMode};
use crate::texture::{Texture, TEXTURE_COLUMN};

/// A failure authoring an input Arrow stream.
#[derive(Debug, Error)]
pub enum SceneEncodeError {
    /// The underlying Arrow builder/writer failed.
    #[error("arrow encode error: {0}")]
    Arrow(#[from] arrow::error::ArrowError),
    /// A params stream would have no columns (no camera/model fields and no
    /// draws), so its row count is undefined.
    #[error("params batch has no columns")]
    EmptyParams,
    /// The `draws` list length disagrees with the frame count.
    #[error("draws has {draws} rows but there are {frames} frames")]
    DrawsLengthMismatch { draws: usize, frames: usize },
}

/// The `FixedSizeList<Float32>[stride]` element type of a geometry column.
fn fsl_type(stride: i32) -> DataType {
    DataType::FixedSizeList(
        Arc::new(Field::new("item", DataType::Float32, false)),
        stride,
    )
}

/// A `List<FixedSizeList<Float32>[stride]>` column, one list per row, from each
/// row's flat row-major values (length must be a multiple of `stride`).
fn list_of_fixed_column(per_row_flat: &[Vec<f32>], stride: i32) -> ArrayRef {
    let flat: Vec<f32> = per_row_flat.iter().flatten().copied().collect();
    let fsl = FixedSizeListArray::new(
        Arc::new(Field::new("item", DataType::Float32, false)),
        stride,
        Arc::new(Float32Array::from(flat)),
        None,
    );
    let field = Arc::new(Field::new("item", fsl_type(stride), false));
    let lengths = per_row_flat
        .iter()
        .map(|row| row.len() / stride as usize)
        .collect::<Vec<_>>();
    let offsets = OffsetBuffer::from_lengths(lengths);
    Arc::new(ListArray::new(field, offsets, Arc::new(fsl), None))
}

/// A `List<UInt32>` column, one list per row.
fn list_of_u32_column(per_row: &[Vec<u32>]) -> ArrayRef {
    let flat: Vec<u32> = per_row.iter().flatten().copied().collect();
    let offsets = OffsetBuffer::from_lengths(per_row.iter().map(Vec::len));
    Arc::new(ListArray::new(
        Arc::new(Field::new("item", DataType::UInt32, false)),
        offsets,
        Arc::new(UInt32Array::from(flat)),
        None,
    ))
}

/// A non-null `FixedSizeList<Float32>[len]` column from flat values.
fn fixed_list_column(len: i32, flat: Vec<f32>) -> ArrayRef {
    Arc::new(FixedSizeListArray::new(
        Arc::new(Field::new("item", DataType::Float32, false)),
        len,
        Arc::new(Float32Array::from(flat)),
        None,
    ))
}

/// The `trd.protocol.version` schema metadata every sub-stream carries.
fn version_metadata() -> std::collections::HashMap<String, String> {
    [(
        PROTOCOL_VERSION_KEY.to_string(),
        PROTOCOL_VERSION.to_string(),
    )]
    .into_iter()
    .collect()
}

/// Writes `batch` as a complete single-batch Arrow IPC stream into `buf`.
fn write_ipc(schema: &Schema, batch: &RecordBatch) -> Result<Vec<u8>, SceneEncodeError> {
    let mut buf = Vec::new();
    let mut writer = StreamWriter::try_new(&mut buf, schema)?;
    writer.write(batch)?;
    writer.finish()?;
    Ok(buf)
}

/// Authors the leading **mesh table** IPC stream: one row per mesh with
/// `position`/`color` `List<FixedSizeList<Float32>[3]>`, `uv`
/// `List<FixedSizeList<Float32>[2]>`, and `index` `List<UInt32>` columns, tagged
/// with the `0.0.5` protocol version. Decodes back via `Mesh::from_arrow_all`.
pub fn encode_mesh_stream(meshes: &[Mesh]) -> Result<Vec<u8>, SceneEncodeError> {
    let positions: Vec<Vec<f32>> = meshes
        .iter()
        .map(|m| m.vertices.iter().flat_map(|v| v.position).collect())
        .collect();
    let colors: Vec<Vec<f32>> = meshes
        .iter()
        .map(|m| m.vertices.iter().flat_map(|v| v.color).collect())
        .collect();
    let uvs: Vec<Vec<f32>> = meshes
        .iter()
        .map(|m| m.vertices.iter().flat_map(|v| v.uv).collect())
        .collect();
    let indices: Vec<Vec<u32>> = meshes.iter().map(|m| m.indices.clone()).collect();

    let list_of_fsl =
        |stride: i32| DataType::List(Arc::new(Field::new("item", fsl_type(stride), false)));
    let schema = Schema::new(vec![
        Field::new("position", list_of_fsl(3), false),
        Field::new("color", list_of_fsl(3), false),
        Field::new("uv", list_of_fsl(2), false),
        Field::new(
            "index",
            DataType::List(Arc::new(Field::new("item", DataType::UInt32, false))),
            false,
        ),
    ])
    .with_metadata(version_metadata());

    let batch = RecordBatch::try_new(
        Arc::new(schema.clone()),
        vec![
            list_of_fixed_column(&positions, 3),
            list_of_fixed_column(&colors, 3),
            list_of_fixed_column(&uvs, 2),
            list_of_u32_column(&indices),
        ],
    )?;
    write_ipc(&schema, &batch)
}

/// Authors an optional **texture table** IPC stream: a one-row `rgba`
/// `FixedSizeList<UInt8>[H*W*4]` column bearing the `arrow.fixed_shape_tensor`
/// extension with shape `[H, W, 4]` (interleaved RGBA), tagged with the `0.0.5`
/// protocol version. Placed between the mesh and params sub-streams
/// (`[mesh][texture][params]`), it binds the albedo `run_stream` samples in
/// [`RenderMode::Textured`]. Decodes back via `ImageTexture::from_arrow`.
pub fn encode_texture_stream(texture: &dyn Texture) -> Result<Vec<u8>, SceneEncodeError> {
    let image = texture.to_image();
    let (width, height) = (image.width as usize, image.height as usize);
    let list_size = (width * height * 4) as i32;

    let storage = DataType::FixedSizeList(
        Arc::new(Field::new("item", DataType::UInt8, false)),
        list_size,
    );
    let extension = arrow_schema::extension::FixedShapeTensor::try_new(
        DataType::UInt8,
        vec![height, width, 4],
        Some(vec![
            "height".to_string(),
            "width".to_string(),
            "channel".to_string(),
        ]),
        None,
    )?;
    let field = Field::new(TEXTURE_COLUMN, storage, false).with_extension_type(extension);
    let array = FixedSizeListArray::new(
        Arc::new(Field::new("item", DataType::UInt8, false)),
        list_size,
        Arc::new(UInt8Array::from(image.rgba)),
        None,
    );

    let schema = Schema::new(vec![field]).with_metadata(version_metadata());
    let batch = RecordBatch::try_new(Arc::new(schema.clone()), vec![Arc::new(array) as ArrayRef])?;
    write_ipc(&schema, &batch)
}

/// The `draw_mode` wire byte for a per-draw render-mode override (`255` = inherit
/// the global mode). Inverse of [`RenderMode::from_wire`].
fn mode_to_wire(mode: Option<RenderMode>) -> u8 {
    match mode {
        None => crate::render::DRAW_MODE_INHERIT,
        Some(RenderMode::Filled) => 0,
        Some(RenderMode::Wireframe) => 1,
        Some(RenderMode::Textured) => 2,
        Some(RenderMode::Shadow) => 3,
    }
}

/// Collects a fixed-size-list column's flat values iff **every** frame has the
/// field set (a non-null Arrow column can't skip rows); returns `None` otherwise.
fn all_or_none<const N: usize>(
    frames: &[FrameParams],
    get: impl Fn(&FrameParams) -> Option<[f32; N]>,
) -> Option<Vec<f32>> {
    let mut flat = Vec::with_capacity(frames.len() * N);
    for frame in frames {
        flat.extend(get(frame)?);
    }
    Some(flat)
}

/// Collects a scalar `Float32` column iff every frame has the field set.
fn all_or_none_f32(
    frames: &[FrameParams],
    get: impl Fn(&FrameParams) -> Option<f32>,
) -> Option<Vec<f32>> {
    frames.iter().map(get).collect()
}

/// Authors the **params** IPC stream from per-frame [`FrameParams`] and an
/// optional per-frame [`Draw`] list. A camera/model column is emitted only when
/// **all** frames set that field (Arrow columns are non-null); `draws`, when
/// given, emits the `draw_mesh`/`draw_model` (+ `draw_mode` when any override is
/// present) instanced-draw columns. Decodes back via [`crate::decode_frames`]
/// and the draw decoder. Tagged with the `0.0.5` protocol version.
pub fn encode_params_stream(
    frames: &[FrameParams],
    draws: Option<&[Vec<Draw>]>,
) -> Result<Vec<u8>, SceneEncodeError> {
    if let Some(draws) = draws {
        if draws.len() != frames.len() {
            return Err(SceneEncodeError::DrawsLengthMismatch {
                draws: draws.len(),
                frames: frames.len(),
            });
        }
    }

    let mut fields: Vec<Field> = Vec::new();
    let mut columns: Vec<ArrayRef> = Vec::new();

    let mut push_fixed = |name: &str, len: i32, flat: Option<Vec<f32>>| {
        if let Some(flat) = flat {
            fields.push(Field::new(name, fsl_type(len), false));
            columns.push(fixed_list_column(len, flat));
        }
    };
    push_fixed("model", 16, all_or_none::<16>(frames, |f| f.model));
    push_fixed("k", 9, all_or_none::<9>(frames, |f| f.k));
    push_fixed("pose", 16, all_or_none::<16>(frames, |f| f.pose));
    push_fixed("eye", 3, all_or_none::<3>(frames, |f| f.eye));
    push_fixed("target", 3, all_or_none::<3>(frames, |f| f.target));
    push_fixed("direction", 3, all_or_none::<3>(frames, |f| f.direction));
    push_fixed("up", 3, all_or_none::<3>(frames, |f| f.up));

    let mut push_scalar = |name: &str, values: Option<Vec<f32>>| {
        if let Some(values) = values {
            fields.push(Field::new(name, DataType::Float32, false));
            columns.push(Arc::new(Float32Array::from(values)) as ArrayRef);
        }
    };
    push_scalar("fovy", all_or_none_f32(frames, |f| f.fovy));
    push_scalar("aspect", all_or_none_f32(frames, |f| f.aspect));
    push_scalar("znear", all_or_none_f32(frames, |f| f.znear));
    push_scalar("zfar", all_or_none_f32(frames, |f| f.zfar));

    if let Some(draws) = draws {
        let mesh_rows: Vec<Vec<u32>> = draws
            .iter()
            .map(|row| row.iter().map(|d| d.mesh_id).collect())
            .collect();
        let model_rows: Vec<Vec<f32>> = draws
            .iter()
            .map(|row| row.iter().flat_map(|d| d.model).collect())
            .collect();
        fields.push(Field::new(
            "draw_mesh",
            DataType::List(Arc::new(Field::new("item", DataType::UInt32, false))),
            false,
        ));
        columns.push(list_of_u32_column(&mesh_rows));
        fields.push(Field::new(
            "draw_model",
            DataType::List(Arc::new(Field::new("item", fsl_type(16), false))),
            false,
        ));
        columns.push(list_of_fixed_column(&model_rows, 16));

        // Emit the per-draw mode override only when a draw actually overrides the
        // global mode; otherwise every draw inherits (the common case).
        if draws.iter().flatten().any(|d| d.mode.is_some()) {
            let mode_rows: Vec<u8> = draws
                .iter()
                .flat_map(|row| row.iter().map(|d| mode_to_wire(d.mode)))
                .collect();
            let offsets = OffsetBuffer::from_lengths(draws.iter().map(Vec::len));
            fields.push(Field::new(
                "draw_mode",
                DataType::List(Arc::new(Field::new("item", DataType::UInt8, false))),
                false,
            ));
            columns.push(Arc::new(ListArray::new(
                Arc::new(Field::new("item", DataType::UInt8, false)),
                offsets,
                Arc::new(UInt8Array::from(mode_rows)),
                None,
            )) as ArrayRef);
        }
    }

    if columns.is_empty() {
        return Err(SceneEncodeError::EmptyParams);
    }

    let schema = Schema::new(fields).with_metadata(version_metadata());
    let batch = RecordBatch::try_new(Arc::new(schema.clone()), columns)?;
    write_ipc(&schema, &batch)
}

/// Authors a complete mesh-first input stream: the mesh table followed by the
/// params stream, concatenated as [`run_stream`](crate::run_stream) expects.
pub fn encode_scene(
    meshes: &[Mesh],
    frames: &[FrameParams],
    draws: Option<&[Vec<Draw>]>,
) -> Result<Vec<u8>, SceneEncodeError> {
    let mut bytes = encode_mesh_stream(meshes)?;
    bytes.extend(encode_params_stream(frames, draws)?);
    Ok(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::InputSession;
    use crate::render::Vertex;

    fn tri_mesh() -> Mesh {
        Mesh {
            vertices: vec![
                Vertex {
                    position: [0.0, 0.5, 0.0],
                    color: [1.0, 0.0, 0.0],
                    uv: [0.1, 0.2],
                },
                Vertex {
                    position: [-0.5, -0.5, 0.0],
                    color: [0.0, 1.0, 0.0],
                    uv: [0.3, 0.4],
                },
                Vertex {
                    position: [0.5, -0.5, 0.0],
                    color: [0.0, 0.0, 1.0],
                    uv: [0.5, 0.6],
                },
            ],
            indices: vec![0, 1, 2],
        }
    }

    #[test]
    fn mesh_stream_roundtrips_through_the_decoder() {
        let meshes = vec![tri_mesh()];
        let bytes = encode_mesh_stream(&meshes).unwrap();
        let reader =
            arrow::ipc::reader::StreamReader::try_new(std::io::Cursor::new(bytes), None).unwrap();
        let batch = reader.into_iter().next().unwrap().unwrap();
        let decoded = Mesh::from_arrow_all(&batch).unwrap();
        assert_eq!(decoded, meshes);
    }

    #[test]
    fn scene_roundtrips_through_the_input_session() {
        let meshes = vec![tri_mesh()];
        let frame = FrameParams {
            eye: Some([0.0, 0.0, 4.0]),
            target: Some([0.0, 0.0, 0.0]),
            up: Some([0.0, 1.0, 0.0]),
            fovy: Some(0.8),
            aspect: Some(1.5),
            ..FrameParams::IDENTITY
        };
        let draws = vec![vec![Draw {
            mesh_id: 0,
            model: [
                1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.2, -0.1, 0.0, 1.0,
            ],
            mode: None,
        }]];

        let bytes = encode_scene(&meshes, &[frame], Some(&draws)).unwrap();

        let mut session = InputSession::new();
        let batches = session.push(&bytes).unwrap();
        assert_eq!(session.meshes(), meshes.as_slice());

        let decoded: Vec<_> = batches.into_iter().flatten().collect();
        assert_eq!(decoded.len(), 1);
        assert_eq!(decoded[0].params, frame);
        assert_eq!(decoded[0].draws, Some(draws[0].clone()));
    }

    #[test]
    fn texture_stream_roundtrips_through_the_decoder() {
        use crate::texture::ImageTexture;
        // 2×2 RGBA checker: white, red / green, blue.
        let rgba = vec![
            255, 255, 255, 255, 255, 0, 0, 255, //
            0, 255, 0, 255, 0, 0, 255, 255,
        ];
        let texture = ImageTexture::from_rgba(2, 2, rgba.clone()).unwrap();
        let bytes = encode_texture_stream(&texture).unwrap();
        let reader =
            arrow::ipc::reader::StreamReader::try_new(std::io::Cursor::new(bytes), None).unwrap();
        let batch = reader.into_iter().next().unwrap().unwrap();
        let decoded = ImageTexture::from_arrow(&batch).unwrap();
        assert_eq!((decoded.width(), decoded.height()), (2, 2));
        assert_eq!(decoded.rgba(), rgba.as_slice());
    }

    #[test]
    fn params_without_columns_is_rejected() {
        let err = encode_params_stream(&[FrameParams::IDENTITY], None).unwrap_err();
        assert!(matches!(err, SceneEncodeError::EmptyParams));
    }

    #[test]
    fn params_roundtrip_through_decode_params_stream() {
        let frames = vec![
            FrameParams {
                eye: Some([0.0, 0.0, 4.0]),
                target: Some([0.0, 0.0, 0.0]),
                up: Some([0.0, 1.0, 0.0]),
                fovy: Some(0.8),
                aspect: Some(1.5),
                ..FrameParams::IDENTITY
            },
            FrameParams {
                eye: Some([1.0, 0.0, 3.0]),
                target: Some([0.0, 0.0, 0.0]),
                up: Some([0.0, 1.0, 0.0]),
                fovy: Some(0.8),
                aspect: Some(1.5),
                ..FrameParams::IDENTITY
            },
        ];
        let draws = vec![
            vec![Draw {
                mesh_id: 0,
                model: [
                    1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.2, -0.1, 0.0, 1.0,
                ],
                mode: None,
            }],
            vec![Draw {
                mesh_id: 0,
                model: [
                    0.5, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 1.0,
                ],
                mode: Some(RenderMode::Wireframe),
            }],
        ];

        let bytes = encode_params_stream(&frames, Some(&draws)).unwrap();
        let decoded = crate::decode_params_stream(&bytes).unwrap();

        assert_eq!(decoded.len(), 2);
        for (i, frame) in decoded.iter().enumerate() {
            assert_eq!(frame.params, frames[i]);
            assert_eq!(frame.draws, Some(draws[i].clone()));
        }
    }

    #[test]
    fn explicit_empty_draw_list_roundtrips_as_background_only() {
        // A frame with an **explicit empty** draw list must decode to
        // `Some(vec![])` (no meshes → background plate only), distinct from a
        // frame with a real draw (`Some(vec![draw])`). Both go through the wire
        // encode/decode, so the empty-list row proves the draw columns stay
        // present and the empty list survives (rather than collapsing to "absent
        // ⇒ default instance"). Guards the FIBA tail (untracked frames).
        let frames = vec![FrameParams::IDENTITY, FrameParams::IDENTITY];
        let placed = Draw {
            mesh_id: 0,
            model: [
                1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.2, -0.1, 0.0, 1.0,
            ],
            mode: None,
        };
        let draws = vec![vec![placed], Vec::new()];

        let bytes = encode_params_stream(&frames, Some(&draws)).unwrap();
        let decoded = crate::decode_params_stream(&bytes).unwrap();

        assert_eq!(decoded.len(), 2);
        // Frame 0: the placed draw survives verbatim.
        assert_eq!(decoded[0].draws, Some(vec![placed]));
        assert_eq!(decoded[0].resolved_draws(), vec![placed]);
        // Frame 1: explicit empty ⇒ no meshes (not a default instance).
        assert_eq!(decoded[1].draws, Some(Vec::new()));
        assert!(decoded[1].resolved_draws().is_empty());
    }

    #[test]
    fn draws_length_must_match_frames() {
        let frame = FrameParams {
            fovy: Some(0.8),
            ..FrameParams::IDENTITY
        };
        let err = encode_params_stream(&[frame], Some(&[])).unwrap_err();
        assert!(matches!(err, SceneEncodeError::DrawsLengthMismatch { .. }));
    }
}
