//! Decoding the columnar **Arrow mesh table** the stream protocol carries into
//! the canonical [`Mesh`] (#37).
//!
//! Because Arrow requires every column in a record batch to have the same
//! length — while a mesh has a different number of vertices and indices — each
//! row of the table is one embedded mesh or one glTF reference. Embedded
//! per-vertex and per-index data uses `position`
//! `List<FixedSizeList<Float32>[3]>`, an optional `color`
//! `List<FixedSizeList<Float32>[3]>`, an optional `uv`
//! `List<FixedSizeList<Float32>[2]>`, and an optional `index` `List<UInt32>`
//! (absent ⇒ a non-indexed triangle list), plus optional material JSON.
//! Reference rows use `gltf_path`/`gltf_url` and are resolved by the shell.

use arrow::array::{
    Array, FixedSizeListArray, Float32Array, ListArray, RecordBatch, StringArray, UInt32Array,
};
use arrow::datatypes::DataType;

use super::{Mesh, MeshAsset, MeshError, MeshReference, MeshResource, DEFAULT_COLOR};
use crate::render::Vertex;
use crate::DisneyMaterial;

pub const GLTF_PATH_COLUMN: &str = "gltf_path";
pub const GLTF_URL_COLUMN: &str = "gltf_url";
pub const MATERIAL_COLUMN: &str = "material";

impl Mesh {
    /// Decodes a columnar **Arrow mesh table** into the canonical [`Mesh`] (#37).
    ///
    /// Embedded rows use a required `position`
    /// `List<FixedSizeList<Float32>[3]>` column, an optional `color`
    /// `List<FixedSizeList<Float32>[3]>` (defaults to white), and an optional
    /// `index` `List<UInt32>` (absent ⇒ the vertices are a non-indexed triangle
    /// list, so their count must be a multiple of three). The **first row** is
    /// decoded; use [`Mesh::from_arrow_all`] to decode every row. Produces the
    /// same [`Mesh`] as [`Mesh::from_obj`] for equivalent geometry. Reference
    /// rows return [`MeshError::ExternalReference`].
    pub fn from_arrow(batch: &RecordBatch) -> Result<Mesh, MeshError> {
        match Self::decode_mesh_resources(batch)?.into_iter().next() {
            Some(MeshResource::Resolved(asset)) => Ok(asset.mesh),
            Some(MeshResource::Gltf(_)) => Err(MeshError::ExternalReference { row: 0 }),
            None => Err(MeshError::Empty),
        }
    }

    /// Decodes **every** row of an Arrow mesh table into one [`Mesh`] each,
    /// preserving row order so a stream's draw list can reference meshes by row
    /// index. Returns [`MeshError::Empty`] for a zero-row table.
    pub fn from_arrow_all(batch: &RecordBatch) -> Result<Vec<Mesh>, MeshError> {
        Self::decode_mesh_resources(batch)?
            .into_iter()
            .enumerate()
            .map(|(row, resource)| match resource {
                MeshResource::Resolved(asset) => Ok(asset.mesh),
                MeshResource::Gltf(_) => Err(MeshError::ExternalReference { row }),
            })
            .collect()
    }

    pub(crate) fn decode_mesh_resources(
        batch: &RecordBatch,
    ) -> Result<Vec<MeshResource>, MeshError> {
        if batch.num_rows() == 0 {
            return Err(MeshError::Empty);
        }
        let path = optional_string(batch, GLTF_PATH_COLUMN)?;
        let url = optional_string(batch, GLTF_URL_COLUMN)?;
        let material = optional_string(batch, MATERIAL_COLUMN)?;
        let geometry = ["position", "color", "uv", "index"]
            .into_iter()
            .map(|name| {
                batch
                    .column_by_name(name)
                    .map(|column| as_list(column, name))
                    .transpose()
                    .map(|column| (name, column))
            })
            .collect::<Result<Vec<_>, _>>()?;

        (0..batch.num_rows())
            .map(|row| {
                let reference = MeshReference::new(
                    nonempty_string(path, row).map(str::to_owned),
                    nonempty_string(url, row).map(str::to_owned),
                );
                let has_geometry = geometry
                    .iter()
                    .filter_map(|(_, column)| *column)
                    .any(|column| !column.is_null(row));
                let material_json = nonempty_string(material, row);
                match (has_geometry, reference) {
                    (true, None) => {
                        let material = material_json
                            .map(|json| {
                                serde_json::from_str::<DisneyMaterial>(json).map_err(|error| {
                                    MeshError::InvalidMaterial {
                                        row,
                                        message: error.to_string(),
                                    }
                                })
                            })
                            .transpose()?
                            .unwrap_or_default();
                        Ok(MeshResource::Resolved(Box::new(MeshAsset::embedded(
                            Mesh::from_arrow_row(batch, row)?,
                            material,
                        ))))
                    }
                    (false, Some(reference)) if material_json.is_none() => {
                        Ok(MeshResource::Gltf(reference))
                    }
                    _ => Err(MeshError::InvalidSource { row }),
                }
            })
            .collect()
    }

    /// Decodes a single `row` of an Arrow mesh table into a [`Mesh`]. Shared by
    /// [`Mesh::from_arrow`] (row 0) and [`Mesh::from_arrow_all`] (all rows).
    fn from_arrow_row(batch: &RecordBatch, row: usize) -> Result<Mesh, MeshError> {
        let position_list = require_list(batch, "position")?;
        if position_list.is_null(row) {
            return Err(MeshError::NullValues("position"));
        }
        let position_ref = position_list.value(row);
        let position = fixed_f32_list(&position_ref, "position", 3)?;
        if position.null_count() > 0 || position.values().null_count() > 0 {
            return Err(MeshError::NullValues("position"));
        }
        let vertex_count = position.len();
        if vertex_count == 0 {
            return Err(MeshError::Empty);
        }
        let position_values = fixed_list_f32_values(position, "position")?;
        let position_base = position.value_offset(0) as usize;

        let color_ref = match batch.column_by_name("color") {
            Some(column) => {
                let list = as_list(column, "color")?;
                if list.is_null(row) {
                    None
                } else {
                    Some(list.value(row))
                }
            }
            None => None,
        };
        let color = match &color_ref {
            Some(color_ref) => {
                let color = fixed_f32_list(color_ref, "color", 3)?;
                if color.len() != vertex_count {
                    return Err(MeshError::ColumnType {
                        column: "color",
                        expected: "one color per vertex",
                        actual: color.data_type().clone(),
                    });
                }
                if color.null_count() > 0 || color.values().null_count() > 0 {
                    return Err(MeshError::NullValues("color"));
                }
                let values = fixed_list_f32_values(color, "color")?;
                Some((color.value_offset(0) as usize, values))
            }
            None => None,
        };

        let uv_ref = match batch.column_by_name("uv") {
            Some(column) => {
                let list = as_list(column, "uv")?;
                if list.is_null(row) {
                    None
                } else {
                    Some(list.value(row))
                }
            }
            None => None,
        };
        let uv = match &uv_ref {
            Some(uv_ref) => {
                let uv = fixed_f32_list(uv_ref, "uv", 2)?;
                if uv.len() != vertex_count {
                    return Err(MeshError::ColumnType {
                        column: "uv",
                        expected: "one uv per vertex",
                        actual: uv.data_type().clone(),
                    });
                }
                if uv.null_count() > 0 || uv.values().null_count() > 0 {
                    return Err(MeshError::NullValues("uv"));
                }
                let values = fixed_list_f32_values(uv, "uv")?;
                Some((uv.value_offset(0) as usize, values))
            }
            None => None,
        };

        let vertices: Vec<Vertex> = (0..vertex_count)
            .map(|i| {
                let po = position_base + i * 3;
                let position = [
                    position_values.value(po),
                    position_values.value(po + 1),
                    position_values.value(po + 2),
                ];
                let color = match color {
                    Some((base, values)) => {
                        let co = base + i * 3;
                        [values.value(co), values.value(co + 1), values.value(co + 2)]
                    }
                    None => DEFAULT_COLOR,
                };
                let uv = match uv {
                    Some((base, values)) => {
                        let uo = base + i * 2;
                        [values.value(uo), values.value(uo + 1)]
                    }
                    None => [0.0, 0.0],
                };
                Vertex {
                    position,
                    color,
                    uv,
                }
            })
            .collect();

        let indices = decode_indices(batch, row, vertex_count)?;
        if indices.is_empty() {
            return Err(MeshError::Empty);
        }
        Ok(Mesh {
            vertices,
            indices,
            shading: None,
        })
    }
}

/// Decodes the optional `index` `List<UInt32>` column at `row`, or synthesizes a
/// non-indexed triangle list `[0, 1, …, vertex_count)` when absent/null.
/// Validates every index is in range and (for the non-indexed case) that the
/// vertex count is a multiple of three.
fn decode_indices(
    batch: &RecordBatch,
    row: usize,
    vertex_count: usize,
) -> Result<Vec<u32>, MeshError> {
    let list = match batch.column_by_name("index") {
        Some(column) => as_list(column, "index")?,
        None => return synthesize_triangle_list(vertex_count),
    };
    if list.is_null(row) {
        return synthesize_triangle_list(vertex_count);
    }
    let values_ref = list.value(row);
    let array = values_ref
        .as_any()
        .downcast_ref::<UInt32Array>()
        .ok_or_else(|| MeshError::ColumnType {
            column: "index",
            expected: "List<UInt32>",
            actual: list.data_type().clone(),
        })?;
    if array.null_count() > 0 {
        return Err(MeshError::NullValues("index"));
    }
    let indices: Vec<u32> = array.values().to_vec();
    for &index in &indices {
        if index as usize >= vertex_count {
            return Err(MeshError::IndexOutOfRange {
                index,
                vertex_count,
            });
        }
    }
    Ok(indices)
}

/// A non-indexed triangle list `[0, 1, …, vertex_count)`, valid only when the
/// vertex count is a multiple of three.
fn synthesize_triangle_list(vertex_count: usize) -> Result<Vec<u32>, MeshError> {
    if !vertex_count.is_multiple_of(3) {
        return Err(MeshError::NonTriangleList { vertex_count });
    }
    Ok((0..vertex_count as u32).collect())
}

/// Looks up a required `List<…>` column.
fn require_list<'a>(
    batch: &'a RecordBatch,
    name: &'static str,
) -> Result<&'a ListArray, MeshError> {
    let column = batch
        .column_by_name(name)
        .ok_or(MeshError::MissingColumn(name))?;
    as_list(column, name)
}

fn optional_string<'a>(
    batch: &'a RecordBatch,
    name: &'static str,
) -> Result<Option<&'a StringArray>, MeshError> {
    let Some(column) = batch.column_by_name(name) else {
        return Ok(None);
    };
    column
        .as_any()
        .downcast_ref::<StringArray>()
        .map(Some)
        .ok_or_else(|| MeshError::ColumnType {
            column: name,
            expected: "Utf8",
            actual: column.data_type().clone(),
        })
}

fn nonempty_string(column: Option<&StringArray>, row: usize) -> Option<&str> {
    let column = column?;
    (!column.is_null(row) && !column.value(row).is_empty()).then(|| column.value(row))
}

/// Downcasts `column` to a [`ListArray`].
fn as_list<'a>(
    column: &'a arrow::array::ArrayRef,
    name: &'static str,
) -> Result<&'a ListArray, MeshError> {
    column
        .as_any()
        .downcast_ref::<ListArray>()
        .ok_or_else(|| MeshError::ColumnType {
            column: name,
            expected: "List<…>",
            actual: column.data_type().clone(),
        })
}

/// Downcasts a list row's values to a `FixedSizeList<Float32>[len]`, erroring on
/// a type or list-length mismatch.
fn fixed_f32_list<'a>(
    values: &'a arrow::array::ArrayRef,
    name: &'static str,
    len: i32,
) -> Result<&'a FixedSizeListArray, MeshError> {
    let expected = if len == 3 {
        "List<FixedSizeList<Float32>[3]>"
    } else {
        "List<FixedSizeList<Float32>[N]>"
    };
    let list = values
        .as_any()
        .downcast_ref::<FixedSizeListArray>()
        .ok_or_else(|| MeshError::ColumnType {
            column: name,
            expected,
            actual: values.data_type().clone(),
        })?;
    if list.value_length() != len || list.values().data_type() != &DataType::Float32 {
        return Err(MeshError::ColumnType {
            column: name,
            expected,
            actual: list.data_type().clone(),
        });
    }
    Ok(list)
}

/// Downcasts a validated `FixedSizeList<Float32>` array's child to a
/// [`Float32Array`].
fn fixed_list_f32_values<'a>(
    list: &'a FixedSizeListArray,
    name: &'static str,
) -> Result<&'a Float32Array, MeshError> {
    list.values()
        .as_any()
        .downcast_ref::<Float32Array>()
        .ok_or_else(|| MeshError::ColumnType {
            column: name,
            expected: "FixedSizeList<Float32>[N]",
            actual: list.values().data_type().clone(),
        })
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow::array::ArrayRef;
    use arrow::buffer::OffsetBuffer;
    use arrow::datatypes::{Field, Schema};

    use super::*;
    use crate::mesh::QUAD_OBJ;

    /// The `FixedSizeList<Float32>[stride]` element type of a geometry column.
    fn fsl_type(stride: i32) -> DataType {
        DataType::FixedSizeList(
            Arc::new(Field::new("item", DataType::Float32, false)),
            stride,
        )
    }

    /// A single-row `List<FixedSizeList<Float32>[stride]>` array holding one
    /// mesh's flat, row-major values.
    fn geometry_column(values: Vec<f32>, stride: i32) -> ArrayRef {
        let child = Arc::new(Field::new("item", DataType::Float32, false));
        let fsl =
            FixedSizeListArray::new(child, stride, Arc::new(Float32Array::from(values)), None);
        let field = Arc::new(Field::new("item", fsl_type(stride), false));
        let offsets = OffsetBuffer::from_lengths([fsl.len()]);
        Arc::new(ListArray::new(field, offsets, Arc::new(fsl), None))
    }

    /// A single-row `List<UInt32>` array holding one mesh's indices.
    fn index_column(indices: Vec<u32>) -> ArrayRef {
        let field = Arc::new(Field::new("item", DataType::UInt32, false));
        let values = UInt32Array::from(indices);
        let offsets = OffsetBuffer::from_lengths([values.len()]);
        Arc::new(ListArray::new(field, offsets, Arc::new(values), None))
    }

    /// Builds a one-row mesh `RecordBatch` from flat positions, optional flat
    /// colors, and optional indices.
    fn mesh_batch(
        positions: Vec<f32>,
        colors: Option<Vec<f32>>,
        indices: Option<Vec<u32>>,
    ) -> RecordBatch {
        let list_of_fsl =
            |stride: i32| DataType::List(Arc::new(Field::new("item", fsl_type(stride), false)));
        let mut fields: Vec<Field> = vec![Field::new("position", list_of_fsl(3), false)];
        let mut columns: Vec<ArrayRef> = vec![geometry_column(positions, 3)];
        if let Some(colors) = colors {
            fields.push(Field::new("color", list_of_fsl(3), false));
            columns.push(geometry_column(colors, 3));
        }
        if let Some(indices) = indices {
            fields.push(Field::new(
                "index",
                DataType::List(Arc::new(Field::new("item", DataType::UInt32, false))),
                false,
            ));
            columns.push(index_column(indices));
        }
        RecordBatch::try_new(Arc::new(Schema::new(fields)), columns).unwrap()
    }

    #[test]
    fn arrow_uv_column_is_decoded() {
        // 3 vertices with an explicit `uv` column (FixedSizeList<f32>[2]).
        let list_of_fsl =
            |stride: i32| DataType::List(Arc::new(Field::new("item", fsl_type(stride), false)));
        let fields = vec![
            Field::new("position", list_of_fsl(3), false),
            Field::new("uv", list_of_fsl(2), false),
        ];
        let columns = vec![
            geometry_column(vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0], 3),
            geometry_column(vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6], 2),
        ];
        let batch = RecordBatch::try_new(Arc::new(Schema::new(fields)), columns).unwrap();
        let mesh = Mesh::from_arrow(&batch).unwrap();
        assert_eq!(mesh.vertices[0].uv, [0.1, 0.2]);
        assert_eq!(mesh.vertices[1].uv, [0.3, 0.4]);
        assert_eq!(mesh.vertices[2].uv, [0.5, 0.6]);
        // Color defaults to white when absent (uv is independent).
        assert_eq!(mesh.vertices[0].color, DEFAULT_COLOR);
    }

    #[test]
    fn arrow_without_uv_defaults_to_zero() {
        let batch = mesh_batch(
            vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            None,
            None,
        );
        let mesh = Mesh::from_arrow(&batch).unwrap();
        assert!(mesh.vertices.iter().all(|v| v.uv == [0.0, 0.0]));
    }

    #[test]
    fn arrow_quad_matches_obj_quad() {
        // The same geometry via Arrow and OBJ must yield an identical Mesh.
        let batch = mesh_batch(
            vec![
                -0.5, -0.5, 0.0, // v0
                0.5, -0.5, 0.0, // v1
                0.5, 0.5, 0.0, // v2
                -0.5, 0.5, 0.0, // v3
            ],
            None,
            Some(vec![0, 1, 2, 0, 2, 3]),
        );
        let arrow_mesh = Mesh::from_arrow(&batch).unwrap();
        assert_eq!(arrow_mesh, Mesh::from_obj(QUAD_OBJ).unwrap());
    }

    #[test]
    fn arrow_colors_are_decoded() {
        let batch = mesh_batch(
            vec![0.0, 0.5, 0.0, -0.5, -0.5, 0.0, 0.5, -0.5, 0.0],
            Some(vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]),
            Some(vec![0, 1, 2]),
        );
        let mesh = Mesh::from_arrow(&batch).unwrap();
        assert_eq!(mesh, Mesh::hello_triangle());
    }

    #[test]
    fn arrow_without_color_defaults_white() {
        let batch = mesh_batch(
            vec![0.0, 0.5, 0.0, -0.5, -0.5, 0.0, 0.5, -0.5, 0.0],
            None,
            Some(vec![0, 1, 2]),
        );
        let mesh = Mesh::from_arrow(&batch).unwrap();
        assert!(mesh.vertices.iter().all(|v| v.color == DEFAULT_COLOR));
    }

    #[test]
    fn arrow_without_index_is_non_indexed_triangle_list() {
        // 3 vertices, no index column ⇒ implicit [0, 1, 2].
        let batch = mesh_batch(
            vec![0.0, 0.5, 0.0, -0.5, -0.5, 0.0, 0.5, -0.5, 0.0],
            None,
            None,
        );
        let mesh = Mesh::from_arrow(&batch).unwrap();
        assert_eq!(mesh.indices, vec![0, 1, 2]);
    }

    #[test]
    fn arrow_non_indexed_needs_multiple_of_three() {
        let batch = mesh_batch(vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0], None, None);
        assert!(matches!(
            Mesh::from_arrow(&batch),
            Err(MeshError::NonTriangleList { vertex_count: 2 })
        ));
    }

    #[test]
    fn arrow_index_out_of_range_errors() {
        let batch = mesh_batch(
            vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            None,
            Some(vec![0, 1, 3]),
        );
        assert!(matches!(
            Mesh::from_arrow(&batch),
            Err(MeshError::IndexOutOfRange {
                index: 3,
                vertex_count: 3
            })
        ));
    }

    #[test]
    fn arrow_missing_position_errors() {
        let batch = RecordBatch::try_new(
            Arc::new(Schema::new(vec![Field::new(
                "index",
                DataType::List(Arc::new(Field::new("item", DataType::UInt32, false))),
                false,
            )])),
            vec![index_column(vec![0, 1, 2])],
        )
        .unwrap();
        assert!(matches!(
            Mesh::from_arrow(&batch),
            Err(MeshError::MissingColumn("position"))
        ));
    }

    #[test]
    fn arrow_wrong_position_type_errors() {
        // position as a list of `[2]` lists, not `[3]`.
        let batch = RecordBatch::try_new(
            Arc::new(Schema::new(vec![Field::new(
                "position",
                DataType::List(Arc::new(Field::new("item", fsl_type(2), false))),
                false,
            )])),
            vec![geometry_column(vec![0.0, 0.0, 1.0, 0.0], 2)],
        )
        .unwrap();
        assert!(matches!(
            Mesh::from_arrow(&batch),
            Err(MeshError::ColumnType {
                column: "position",
                ..
            })
        ));
    }

    #[test]
    fn arrow_wrong_index_type_errors() {
        // index as a list of Float32, not UInt32.
        let position = DataType::List(Arc::new(Field::new("item", fsl_type(3), false)));
        let float_list = DataType::List(Arc::new(Field::new("item", DataType::Float32, false)));
        let idx_field = Arc::new(Field::new("item", DataType::Float32, false));
        let idx_values = Float32Array::from(vec![0.0, 1.0, 2.0]);
        let idx = ListArray::new(
            idx_field,
            OffsetBuffer::from_lengths([idx_values.len()]),
            Arc::new(idx_values),
            None,
        );
        let batch = RecordBatch::try_new(
            Arc::new(Schema::new(vec![
                Field::new("position", position, false),
                Field::new("index", float_list, false),
            ])),
            vec![
                geometry_column(vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0], 3),
                Arc::new(idx),
            ],
        )
        .unwrap();
        assert!(matches!(
            Mesh::from_arrow(&batch),
            Err(MeshError::ColumnType {
                column: "index",
                ..
            })
        ));
    }

    #[test]
    fn arrow_empty_is_empty_error() {
        // A batch with zero rows (no meshes) is empty.
        let batch = RecordBatch::try_new(
            Arc::new(Schema::new(vec![Field::new(
                "position",
                DataType::List(Arc::new(Field::new("item", fsl_type(3), false))),
                false,
            )])),
            vec![Arc::new(ListArray::new_null(
                Arc::new(Field::new("item", fsl_type(3), false)),
                0,
            ))],
        )
        .unwrap();
        assert!(matches!(Mesh::from_arrow(&batch), Err(MeshError::Empty)));
    }
}
