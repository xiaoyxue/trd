//! Authoring the trd **input** stream in Rust.
//!
//! `trd-core` decodes the `[mesh][texture?][frames?][params]` input stream
//! (`Mesh::from_arrow_all`, [`crate::decode_frames`], the wasm
//! [`super::InputSession`]) and encodes the **image output**
//! ([`crate::OutputSession`]). This module authors the same `0.0.6` input format
//! for Rust producers such as the video editor. Its round-trip tests feed the
//! result through the real decoder so the two halves cannot drift.
//!
//! It sits beside [`super::arrow_decode`] as the encode counterpart of the same
//! wire format: `arrow_decode` turns a `RecordBatch` into [`FrameParams`], this
//! turns values back into the tables.
//!
//! The stream is **mesh-first**: [`encode_mesh_stream`] writes the leading mesh
//! table (its own complete Arrow IPC stream) and [`encode_params_stream`] writes
//! the following params stream; [`encode_scene`] concatenates them (the framing
//! [`run_stream`](crate::run_stream) expects). Texture tables are authored by
//! [`encode_texture_stream`].

use std::sync::Arc;

#[cfg(test)]
use arrow::array::BinaryArray;
use arrow::array::{
    ArrayRef, FixedSizeListArray, Float32Array, ListArray, RecordBatch, StringArray, UInt32Array,
    UInt8Array,
};
use arrow::buffer::{BooleanBuffer, NullBuffer, OffsetBuffer};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::ipc::writer::StreamWriter;
use thiserror::Error;

#[cfg(test)]
use super::FRAMES_TABLE_KIND;
use super::{
    FRAME_RATE_KEY, MESH_TABLE_KIND, PARAMS_TABLE_KIND, PROTOCOL_VERSION, PROTOCOL_VERSION_KEY,
    TABLE_KIND_KEY, TEXTURE_TABLE_KIND,
};
#[cfg(test)]
use crate::math::Matrix4;
use crate::render::{Draw, DrawSelection, RenderMode};
use crate::render::{FrameParams, Mesh};
use crate::texture::{Texture, TEXTURE_COLUMN};
use crate::{DisneyMaterial, MeshReference};
#[cfg(test)]
use crate::{InlineFrame, FRAME_BYTES_COLUMN, FRAME_PIXELS_COLUMN};

/// A failure authoring an input Arrow stream.
#[derive(Debug, Error)]
pub enum SceneEncodeError {
    /// The underlying Arrow builder/writer failed.
    #[error("arrow encode error: {0}")]
    Arrow(#[from] arrow::error::ArrowError),
    /// The protocol requires at least one leading mesh row.
    #[error("scene has no meshes")]
    EmptyMeshes,
    /// Playback rate metadata must name a finite positive rate.
    #[error("frame rate must be finite and positive, got {0}")]
    InvalidFrameRate(f64),
    /// A Disney material could not be represented as JSON.
    #[error("material JSON encode failed: {0}")]
    Material(#[from] serde_json::Error),
    /// A reference-only mesh must name at least one path or URL.
    #[error("glTF mesh row {0} has neither a path nor a URL")]
    MissingGltfReference(usize),
    /// glTF owns its textures; a separate texture table would duplicate state.
    #[error("a scene containing a glTF reference cannot carry a texture table")]
    TextureWithGltfReference,
    /// A params stream would have no columns (no camera/model fields and no
    /// draws), so its row count is undefined.
    #[error("params batch has no columns")]
    EmptyParams,
    /// The `draws` list length disagrees with the frame count.
    #[error("draws has {draws} rows but there are {frames} frames")]
    DrawsLengthMismatch { draws: usize, frames: usize },
    /// The `frame_id` list length disagrees with the params row count.
    #[error("frame_ids has {frame_ids} rows but there are {frames} frames")]
    FrameIdsLengthMismatch { frame_ids: usize, frames: usize },
    /// Sparse sidecar-video frame keys must parallel the params rows.
    #[error("video_frame_indices has {indices} rows but there are {frames} frames")]
    VideoFrameIndicesLengthMismatch { indices: usize, frames: usize },
    /// Sparse sidecar-video frame keys must be strictly increasing.
    #[error("video_frame_indices must be strictly increasing: {current} follows {previous}")]
    NonIncreasingVideoFrameIndex { previous: u32, current: u32 },
    /// An explicit frames table must contain at least one resource.
    #[error("frames table is empty")]
    EmptyFrames,
    /// An encoded frame payload cannot be empty.
    #[error("encoded frame resource {row} is empty")]
    EmptyEncodedFrame { row: usize },
    /// Raw frame dimensions must be non-zero.
    #[error("raw frame resource {row} has invalid dimensions {width}x{height}")]
    InvalidFrameDimensions { row: usize, width: u32, height: u32 },
    /// Raw frame pixels do not match their declared dimensions.
    #[error(
        "raw frame resource {row} has {actual} RGBA bytes; expected {expected} for \
         {width}x{height}"
    )]
    InvalidFramePixels {
        row: usize,
        actual: usize,
        expected: usize,
        width: u32,
        height: u32,
    },
    /// One fixed-shape tensor column cannot carry several image dimensions.
    #[error(
        "raw frame resource {row} is {width}x{height}, but the frames tensor is \
         {expected_width}x{expected_height}"
    )]
    MixedFrameDimensions {
        row: usize,
        width: u32,
        height: u32,
        expected_width: u32,
        expected_height: u32,
    },
    /// A full scene references a missing inline frame resource.
    #[error("frame_id {frame_id} is out of range for {frame_count} frame resource(s)")]
    FrameIdOutOfRange { frame_id: u32, frame_count: usize },
}

#[derive(Debug, Clone, Copy)]
pub enum SceneMesh<'a> {
    Embedded {
        mesh: &'a Mesh,
        material: &'a DisneyMaterial,
    },
    Gltf(&'a MeshReference),
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

fn nullable_list_of_fixed_column(per_row_flat: &[Option<Vec<f32>>], stride: i32) -> ArrayRef {
    let flat: Vec<f32> = per_row_flat
        .iter()
        .filter_map(Option::as_ref)
        .flatten()
        .copied()
        .collect();
    let fsl = FixedSizeListArray::new(
        Arc::new(Field::new("item", DataType::Float32, false)),
        stride,
        Arc::new(Float32Array::from(flat)),
        None,
    );
    let offsets = OffsetBuffer::from_lengths(
        per_row_flat
            .iter()
            .map(|row| row.as_ref().map_or(0, |row| row.len() / stride as usize)),
    );
    let valid = per_row_flat.iter().map(Option::is_some).collect::<Vec<_>>();
    Arc::new(ListArray::new(
        Arc::new(Field::new("item", fsl_type(stride), false)),
        offsets,
        Arc::new(fsl),
        Some(NullBuffer::new(BooleanBuffer::from(valid))),
    ))
}

fn nullable_list_of_u32_column(per_row: &[Option<Vec<u32>>]) -> ArrayRef {
    let flat: Vec<u32> = per_row
        .iter()
        .filter_map(Option::as_ref)
        .flatten()
        .copied()
        .collect();
    let offsets =
        OffsetBuffer::from_lengths(per_row.iter().map(|row| row.as_ref().map_or(0, Vec::len)));
    let valid = per_row.iter().map(Option::is_some).collect::<Vec<_>>();
    Arc::new(ListArray::new(
        Arc::new(Field::new("item", DataType::UInt32, false)),
        offsets,
        Arc::new(UInt32Array::from(flat)),
        Some(NullBuffer::new(BooleanBuffer::from(valid))),
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

/// Required schema metadata every input sub-stream carries.
fn table_metadata(kind: &'static str) -> std::collections::HashMap<String, String> {
    [
        (
            PROTOCOL_VERSION_KEY.to_string(),
            PROTOCOL_VERSION.to_string(),
        ),
        (TABLE_KIND_KEY.to_string(), kind.to_string()),
    ]
    .into_iter()
    .collect()
}

fn params_metadata(
    frame_rate: Option<f64>,
) -> Result<std::collections::HashMap<String, String>, SceneEncodeError> {
    let mut metadata = table_metadata(PARAMS_TABLE_KIND);
    if let Some(frame_rate) = frame_rate {
        if !frame_rate.is_finite() || frame_rate <= 0.0 {
            return Err(SceneEncodeError::InvalidFrameRate(frame_rate));
        }
        metadata.insert(FRAME_RATE_KEY.to_owned(), frame_rate.to_string());
    }
    Ok(metadata)
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
/// with the `0.0.6` protocol version. Decodes back via `Mesh::from_arrow_all`.
#[cfg(test)]
pub fn encode_mesh_stream(meshes: &[Mesh]) -> Result<Vec<u8>, SceneEncodeError> {
    let materials = vec![DisneyMaterial::default(); meshes.len()];
    let resources = meshes
        .iter()
        .zip(&materials)
        .map(|(mesh, material)| SceneMesh::Embedded { mesh, material })
        .collect::<Vec<_>>();
    encode_mesh_resources(&resources)
}

pub fn encode_mesh_resources(meshes: &[SceneMesh<'_>]) -> Result<Vec<u8>, SceneEncodeError> {
    if meshes.is_empty() {
        return Err(SceneEncodeError::EmptyMeshes);
    }
    if let Some(index) = meshes.iter().position(|resource| {
        matches!(
            resource,
            SceneMesh::Gltf(MeshReference {
                path: None,
                url: None
            })
        )
    }) {
        return Err(SceneEncodeError::MissingGltfReference(index));
    }
    let positions = meshes
        .iter()
        .map(|resource| match resource {
            SceneMesh::Embedded { mesh, .. } => Some(
                mesh.vertices
                    .iter()
                    .flat_map(|vertex| vertex.position)
                    .collect(),
            ),
            SceneMesh::Gltf(_) => None,
        })
        .collect::<Vec<_>>();
    let colors = meshes
        .iter()
        .map(|resource| match resource {
            SceneMesh::Embedded { mesh, .. } => Some(
                mesh.vertices
                    .iter()
                    .flat_map(|vertex| vertex.color)
                    .collect(),
            ),
            SceneMesh::Gltf(_) => None,
        })
        .collect::<Vec<_>>();
    let uvs = meshes
        .iter()
        .map(|resource| match resource {
            SceneMesh::Embedded { mesh, .. } => {
                Some(mesh.vertices.iter().flat_map(|vertex| vertex.uv).collect())
            }
            SceneMesh::Gltf(_) => None,
        })
        .collect::<Vec<_>>();
    let indices = meshes
        .iter()
        .map(|resource| match resource {
            SceneMesh::Embedded { mesh, .. } => Some(mesh.indices.clone()),
            SceneMesh::Gltf(_) => None,
        })
        .collect::<Vec<_>>();
    let paths = meshes
        .iter()
        .map(|resource| match resource {
            SceneMesh::Embedded { .. } => None,
            SceneMesh::Gltf(reference) => reference.path.as_deref(),
        })
        .collect::<Vec<_>>();
    let urls = meshes
        .iter()
        .map(|resource| match resource {
            SceneMesh::Embedded { .. } => None,
            SceneMesh::Gltf(reference) => reference.url.as_deref(),
        })
        .collect::<Vec<_>>();
    let materials = meshes
        .iter()
        .map(|resource| match resource {
            SceneMesh::Embedded { material, .. } => serde_json::to_string(material).map(Some),
            SceneMesh::Gltf(_) => Ok(None),
        })
        .collect::<Result<Vec<_>, serde_json::Error>>()?;

    let list_of_fsl =
        |stride: i32| DataType::List(Arc::new(Field::new("item", fsl_type(stride), false)));
    let schema = Schema::new(vec![
        Field::new("position", list_of_fsl(3), true),
        Field::new("color", list_of_fsl(3), true),
        Field::new("uv", list_of_fsl(2), true),
        Field::new(
            "index",
            DataType::List(Arc::new(Field::new("item", DataType::UInt32, false))),
            true,
        ),
        Field::new(crate::mesh::GLTF_PATH_COLUMN, DataType::Utf8, true),
        Field::new(crate::mesh::GLTF_URL_COLUMN, DataType::Utf8, true),
        Field::new(crate::mesh::MATERIAL_COLUMN, DataType::Utf8, true),
    ])
    .with_metadata(table_metadata(MESH_TABLE_KIND));

    let batch = RecordBatch::try_new(
        Arc::new(schema.clone()),
        vec![
            nullable_list_of_fixed_column(&positions, 3),
            nullable_list_of_fixed_column(&colors, 3),
            nullable_list_of_fixed_column(&uvs, 2),
            nullable_list_of_u32_column(&indices),
            Arc::new(StringArray::from(paths)) as ArrayRef,
            Arc::new(StringArray::from(urls)) as ArrayRef,
            Arc::new(StringArray::from(
                materials.iter().map(Option::as_deref).collect::<Vec<_>>(),
            )) as ArrayRef,
        ],
    )?;
    write_ipc(&schema, &batch)
}

/// Authors an optional **texture table** IPC stream: a one-row `rgba`
/// `FixedSizeList<UInt8>[H*W*4]` column bearing the `arrow.fixed_shape_tensor`
/// extension with shape `[H, W, 4]` (interleaved RGBA), tagged with the `0.0.6`
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

    let schema = Schema::new(vec![field]).with_metadata(table_metadata(TEXTURE_TABLE_KIND));
    let batch = RecordBatch::try_new(Arc::new(schema.clone()), vec![Arc::new(array) as ArrayRef])?;
    write_ipc(&schema, &batch)
}

/// The `draw_mode` wire byte for a per-draw selection (`255` = a mesh inheriting
/// the global mode). Inverse of [`DrawSelection::from_wire`], and pinned against
/// it by the round-trip tests below.
fn selection_to_wire(selection: DrawSelection) -> u8 {
    match selection {
        DrawSelection::Mesh(None) => crate::render::DRAW_MODE_INHERIT,
        DrawSelection::Mesh(Some(RenderMode::Filled)) => 0,
        DrawSelection::Mesh(Some(RenderMode::Wireframe)) => 1,
        DrawSelection::Mesh(Some(RenderMode::Textured)) => 2,
        DrawSelection::Shadow => 3,
        DrawSelection::Mesh(Some(RenderMode::Shaded)) => 4,
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
/// and the draw decoder. Tagged with the `0.0.6` protocol version.
#[cfg(test)]
pub fn encode_params_stream(
    frames: &[FrameParams],
    draws: Option<&[Vec<Draw>]>,
) -> Result<Vec<u8>, SceneEncodeError> {
    encode_params_stream_with_frame_ids_and_rate(frames, draws, None, None, None)
}

/// Authors params with optional nullable `frame_id` references into a preceding
/// frames table.
#[cfg(test)]
pub fn encode_params_stream_with_frame_ids(
    frames: &[FrameParams],
    draws: Option<&[Vec<Draw>]>,
    frame_ids: Option<&[Option<u32>]>,
) -> Result<Vec<u8>, SceneEncodeError> {
    encode_params_stream_with_frame_ids_and_rate(frames, draws, frame_ids, None, None)
}

fn encode_params_stream_with_frame_ids_and_rate(
    frames: &[FrameParams],
    draws: Option<&[Vec<Draw>]>,
    frame_ids: Option<&[Option<u32>]>,
    video_frame_indices: Option<&[u32]>,
    frame_rate: Option<f64>,
) -> Result<Vec<u8>, SceneEncodeError> {
    if let Some(draws) = draws {
        if draws.len() != frames.len() {
            return Err(SceneEncodeError::DrawsLengthMismatch {
                draws: draws.len(),
                frames: frames.len(),
            });
        }
    }
    if let Some(frame_ids) = frame_ids {
        if frame_ids.len() != frames.len() {
            return Err(SceneEncodeError::FrameIdsLengthMismatch {
                frame_ids: frame_ids.len(),
                frames: frames.len(),
            });
        }
    }
    if let Some(indices) = video_frame_indices {
        if indices.len() != frames.len() {
            return Err(SceneEncodeError::VideoFrameIndicesLengthMismatch {
                indices: indices.len(),
                frames: frames.len(),
            });
        }
        if let Some(pair) = indices.windows(2).find(|pair| pair[1] <= pair[0]) {
            return Err(SceneEncodeError::NonIncreasingVideoFrameIndex {
                previous: pair[0],
                current: pair[1],
            });
        }
    }

    let mut fields: Vec<Field> = Vec::new();
    let mut columns: Vec<ArrayRef> = Vec::new();
    if let Some(indices) = video_frame_indices {
        fields.push(Field::new("video_frame_index", DataType::UInt32, false));
        columns.push(Arc::new(UInt32Array::from(indices.to_vec())));
    }

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
            .map(|row| row.iter().flat_map(|d| d.model.to_cols_array()).collect())
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
        if draws
            .iter()
            .flatten()
            .any(|d| d.selection != DrawSelection::INHERIT)
        {
            let mode_rows: Vec<u8> = draws
                .iter()
                .flat_map(|row| row.iter().map(|d| selection_to_wire(d.selection)))
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
    if let Some(frame_ids) = frame_ids {
        fields.push(Field::new("frame_id", DataType::UInt32, true));
        columns.push(Arc::new(UInt32Array::from(frame_ids.to_vec())));
    }

    if columns.is_empty() {
        return Err(SceneEncodeError::EmptyParams);
    }

    let schema = Schema::new(fields).with_metadata(params_metadata(frame_rate)?);
    let batch = RecordBatch::try_new(Arc::new(schema.clone()), columns)?;
    write_ipc(&schema, &batch)
}

/// Authors an inline background `frames` resource table. Encoded and raw rows
/// may coexist through two nullable columns; raw rows share one RGBA tensor
/// shape, as required by Arrow's field-level fixed-shape metadata.
#[cfg(test)]
pub fn encode_frames_stream(frames: &[InlineFrame]) -> Result<Vec<u8>, SceneEncodeError> {
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

/// Authors a complete mesh-first input stream with an optional albedo texture.
///
/// The returned bytes are concatenated Arrow IPC streams in protocol order:
/// `[mesh][texture?][params]`.
pub fn encode_scene(
    meshes: &[Mesh],
    texture: Option<&dyn Texture>,
    frames: &[FrameParams],
    draws: Option<&[Vec<Draw>]>,
    frame_rate: Option<f64>,
) -> Result<Vec<u8>, SceneEncodeError> {
    let materials = vec![DisneyMaterial::default(); meshes.len()];
    let resources = meshes
        .iter()
        .zip(&materials)
        .map(|(mesh, material)| SceneMesh::Embedded { mesh, material })
        .collect::<Vec<_>>();
    encode_scene_resources(&resources, texture, frames, draws, frame_rate)
}

pub fn encode_scene_resources(
    meshes: &[SceneMesh<'_>],
    texture: Option<&dyn Texture>,
    frames: &[FrameParams],
    draws: Option<&[Vec<Draw>]>,
    frame_rate: Option<f64>,
) -> Result<Vec<u8>, SceneEncodeError> {
    reject_gltf_texture(meshes, texture)?;
    let mut bytes = encode_mesh_resources(meshes)?;
    if let Some(texture) = texture {
        bytes.extend(encode_texture_stream(texture)?);
    }
    bytes.extend(encode_params_stream_with_frame_ids_and_rate(
        frames, draws, None, None, frame_rate,
    )?);
    Ok(bytes)
}

pub fn encode_scene_resources_with_frame_indices(
    meshes: &[SceneMesh<'_>],
    texture: Option<&dyn Texture>,
    video_frame_indices: &[u32],
    frames: &[FrameParams],
    draws: Option<&[Vec<Draw>]>,
    frame_rate: Option<f64>,
) -> Result<Vec<u8>, SceneEncodeError> {
    reject_gltf_texture(meshes, texture)?;
    let mut bytes = encode_mesh_resources(meshes)?;
    if let Some(texture) = texture {
        bytes.extend(encode_texture_stream(texture)?);
    }
    bytes.extend(encode_params_stream_with_frame_ids_and_rate(
        frames,
        draws,
        None,
        Some(video_frame_indices),
        frame_rate,
    )?);
    Ok(bytes)
}

fn reject_gltf_texture(
    meshes: &[SceneMesh<'_>],
    texture: Option<&dyn Texture>,
) -> Result<(), SceneEncodeError> {
    if texture.is_some() && meshes.iter().any(|mesh| matches!(mesh, SceneMesh::Gltf(_))) {
        return Err(SceneEncodeError::TextureWithGltfReference);
    }
    Ok(())
}

/// Authors a complete scene with inline frame resources.
#[cfg(test)]
pub fn encode_scene_with_frames(
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::{decode_params_stream, InputSession};
    use crate::render::Vertex;
    use arrow::array::Array;

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
            shading: None,
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
    fn embedded_mesh_roundtrips_every_disney_material_field() {
        let mesh = tri_mesh();
        let mut material = DisneyMaterial {
            name: Some("edited can".to_owned()),
            base_color: [0.8, 0.7, 0.6],
            metallic: 0.25,
            subsurface: 0.1,
            specular: 0.65,
            roughness: 0.42,
            specular_tint: 0.2,
            anisotropic: 0.3,
            sheen: 0.15,
            sheen_tint: 0.75,
            clearcoat: 0.4,
            clearcoat_gloss: 0.9,
            auxiliary: crate::Auxiliary {
                opacity: 0.95,
                alpha_mode: crate::AlphaMode::Mask,
                alpha_cutoff: Some(0.45),
                double_sided: true,
                emissive: [0.1, 0.2, 0.3],
                emissive_strength: 2.0,
                ior: 1.45,
                transmission: 0.2,
                textures: crate::MaterialTextures {
                    base_color: true,
                    metallic_roughness: false,
                    normal: false,
                    occlusion: false,
                    emissive: false,
                },
            },
            ..DisneyMaterial::default()
        };
        material.sources.insert("roughness".into(), "editor".into());
        let bytes = encode_mesh_resources(&[SceneMesh::Embedded {
            mesh: &mesh,
            material: &material,
        }])
        .unwrap();
        let reader =
            arrow::ipc::reader::StreamReader::try_new(std::io::Cursor::new(bytes), None).unwrap();
        let batch = reader.into_iter().next().unwrap().unwrap();
        let decoded = Mesh::decode_mesh_resources(&batch).unwrap();

        assert_eq!(
            decoded,
            vec![crate::MeshResource::Resolved(Box::new(
                crate::MeshAsset::embedded(mesh, material)
            ))]
        );
    }

    #[test]
    fn gltf_mesh_row_contains_only_its_reference() {
        let reference = crate::MeshReference::new(
            Some("assets/meshes/glb/dragon.glb".to_owned()),
            Some("https://example.com/dragon.glb".to_owned()),
        )
        .unwrap();
        let bytes = encode_mesh_resources(&[SceneMesh::Gltf(&reference)]).unwrap();
        let reader =
            arrow::ipc::reader::StreamReader::try_new(std::io::Cursor::new(bytes), None).unwrap();
        let batch = reader.into_iter().next().unwrap().unwrap();

        assert!(batch.column_by_name("position").unwrap().is_null(0));
        assert!(batch.column_by_name("material").unwrap().is_null(0));
        assert_eq!(
            Mesh::decode_mesh_resources(&batch).unwrap(),
            vec![crate::MeshResource::Gltf(reference)]
        );
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
            model: Matrix4::from_cols_array(&[
                1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.2, -0.1, 0.0, 1.0,
            ]),
            selection: DrawSelection::INHERIT,
        }]];

        let bytes = encode_scene(&meshes, None, &[frame], Some(&draws), Some(24.0)).unwrap();

        let mut session = InputSession::new();
        let batches = session.push(&bytes).unwrap();
        assert_eq!(session.meshes(), meshes.as_slice());
        assert_eq!(session.frame_rate(), Some(24.0));

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
    fn complete_scene_carries_its_texture_and_frame_rate() {
        use crate::texture::ImageTexture;

        let meshes = vec![tri_mesh()];
        let texture = ImageTexture::from_rgba(1, 1, vec![10, 20, 30, 255]).unwrap();
        let frames = [FrameParams {
            model: Some(Matrix4::IDENTITY.to_cols_array()),
            ..FrameParams::IDENTITY
        }];

        let bytes = encode_scene(
            &meshes,
            Some(&texture),
            &frames,
            None,
            Some(30000.0 / 1001.0),
        )
        .unwrap();
        let mut session = InputSession::new();
        let batches = session.push(&bytes).unwrap();
        session.finish().unwrap();

        assert_eq!(session.meshes(), meshes.as_slice());
        assert_eq!(session.texture(), Some(&texture));
        assert_eq!(session.frame_rate(), Some(30000.0 / 1001.0));
        assert_eq!(batches.into_iter().flatten().count(), 1);
    }

    #[test]
    fn complete_scene_rejects_missing_meshes_and_invalid_rate() {
        let frame = FrameParams {
            model: Some(Matrix4::IDENTITY.to_cols_array()),
            ..FrameParams::IDENTITY
        };
        assert!(matches!(
            encode_scene(&[], None, &[frame], None, Some(24.0)),
            Err(SceneEncodeError::EmptyMeshes)
        ));
        assert!(matches!(
            encode_scene(&[tri_mesh()], None, &[frame], None, Some(0.0)),
            Err(SceneEncodeError::InvalidFrameRate(0.0))
        ));
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
                model: Matrix4::from_cols_array(&[
                    1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.2, -0.1, 0.0, 1.0,
                ]),
                selection: DrawSelection::INHERIT,
            }],
            vec![Draw {
                mesh_id: 0,
                model: Matrix4::from_cols_array(&[
                    0.5, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 1.0,
                ]),
                selection: DrawSelection::Mesh(Some(RenderMode::Wireframe)),
            }],
        ];

        let bytes = encode_params_stream(&frames, Some(&draws)).unwrap();
        let decoded = decode_params_stream(&bytes).unwrap();

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
            model: Matrix4::from_cols_array(&[
                1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.2, -0.1, 0.0, 1.0,
            ]),
            selection: DrawSelection::INHERIT,
        };
        let draws = vec![vec![placed], Vec::new()];

        let bytes = encode_params_stream(&frames, Some(&draws)).unwrap();
        let decoded = decode_params_stream(&bytes).unwrap();

        assert_eq!(decoded.len(), 2);
        // Frame 0: the placed draw survives verbatim.
        assert_eq!(decoded[0].draws, Some(vec![placed]));
        assert_eq!(decoded[0].resolved_draws(), vec![placed]);
        // Frame 1: explicit empty ⇒ no meshes (not a default instance).
        assert_eq!(decoded[1].draws, Some(Vec::new()));
        assert!(decoded[1].resolved_draws().is_empty());
    }

    #[test]
    fn sparse_video_frame_indices_roundtrip_without_placeholder_rows() {
        let mesh = tri_mesh();
        let indices = [3, 10];
        let frames = [
            FrameParams {
                k: Some([1.0; 9]),
                ..FrameParams::IDENTITY
            },
            FrameParams {
                k: Some([2.0; 9]),
                ..FrameParams::IDENTITY
            },
        ];
        let draws = vec![
            vec![Draw {
                mesh_id: 0,
                model: Matrix4::IDENTITY,
                selection: DrawSelection::INHERIT,
            }],
            vec![Draw {
                mesh_id: 0,
                model: Matrix4::IDENTITY,
                selection: DrawSelection::INHERIT,
            }],
        ];
        let material = DisneyMaterial::default();
        let scene_mesh = SceneMesh::Embedded {
            mesh: &mesh,
            material: &material,
        };
        let bytes = encode_scene_resources_with_frame_indices(
            &[scene_mesh],
            None,
            &indices,
            &frames,
            Some(&draws),
            Some(24.0),
        )
        .unwrap();
        let mut session = InputSession::new();
        let decoded: Vec<_> = session
            .push(&bytes)
            .unwrap()
            .into_iter()
            .flatten()
            .collect();
        session.finish().unwrap();

        assert_eq!(decoded.len(), 2);
        assert_eq!(decoded[0].video_frame_index, Some(3));
        assert_eq!(decoded[1].video_frame_index, Some(10));
    }

    #[test]
    fn sparse_video_frame_indices_must_increase() {
        let mesh = tri_mesh();
        let material = DisneyMaterial::default();
        let error = encode_scene_resources_with_frame_indices(
            &[SceneMesh::Embedded {
                mesh: &mesh,
                material: &material,
            }],
            None,
            &[10, 3],
            &[FrameParams::IDENTITY, FrameParams::IDENTITY],
            Some(&[Vec::new(), Vec::new()]),
            Some(24.0),
        )
        .unwrap_err();

        assert!(matches!(
            error,
            SceneEncodeError::NonIncreasingVideoFrameIndex {
                previous: 10,
                current: 3
            }
        ));
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

    #[test]
    fn frames_encoder_rejects_zero_dimensions() {
        let frame = InlineFrame::Pixels(crate::ImageData {
            width: 0,
            height: 1,
            rgba: Vec::new(),
        });
        assert!(matches!(
            encode_frames_stream(&[frame]),
            Err(SceneEncodeError::InvalidFrameDimensions {
                row: 0,
                width: 0,
                height: 1
            })
        ));
    }
}
