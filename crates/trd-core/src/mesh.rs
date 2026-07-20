//! Loading meshes into the canonical [`Mesh`] — from Wavefront `.obj` text
//! (#36) and from a columnar Arrow mesh table (#37) — plus mesh bounds and the
//! preview model transform.
//!
//! OBJ is trd's **default** authorable mesh format (#36). Rather than hand-roll
//! a parser, we normalize OBJ into the canonical [`Mesh`] via the popular,
//! pure-Rust [`tobj`] loader (wasm-compatible) — following the repo convention
//! that each mesh format is decoded by its own established loader crate. The
//! same [`Mesh`] is produced by the GPU path (#35), the OBJ loader, and the
//! Arrow decoder, so all mesh sources converge on one in-memory type.
//!
//! For OBJ we load with triangulation + a single index buffer, and read the
//! common `v x y z r g b` **vertex-color extension** (via
//! [`tobj::Mesh::vertex_color`]) so colored OBJ vertices survive; uncolored
//! meshes default to white. Normals and texture coordinates are ignored for now
//! (reserved for shading / #20). Parsing is from an in-memory buffer (no file
//! I/O), so it works natively and on wasm.
//!
//! [`Mesh::from_arrow`] decodes the (typically static) **mesh Arrow table** the
//! stream protocol carries (#37). Because Arrow requires every column in a
//! record batch to have the same length — while a mesh has a different number of
//! vertices and indices — each row of the table is **one whole mesh**, with the
//! per-vertex and per-index data nested inside list columns: `position`
//! `List<FixedSizeList<Float32>[3]>` (required), an optional `color`
//! `List<FixedSizeList<Float32>[3]>`, and an optional `index` `List<UInt32>`
//! (absent ⇒ a non-indexed triangle list). [`Mesh::from_arrow_all`] decodes
//! every row (one mesh each) for multi-mesh scenes; [`Mesh::from_arrow`] decodes
//! just the first row. It yields the same canonical [`Mesh`] as the OBJ path, so
//! both authoring routes agree.

use crate::math::{Aabb3, Point3, Transform, Vector3, EPSILON};
use crate::render::{Mesh, Vertex};
use arrow::array::{Array, FixedSizeListArray, Float32Array, ListArray, RecordBatch, UInt32Array};
use arrow::datatypes::DataType;
use std::io::Cursor;

/// Vertex color used when a mesh source carries no per-vertex color.
const DEFAULT_COLOR: [f32; 3] = [1.0, 1.0, 1.0];

/// Default target size (world units, largest AABB extent) for
/// [`Mesh::preview_transform`] — "a reasonable size" for arbitrary-unit assets.
pub const DEFAULT_PREVIEW_TARGET: f32 = 2.0;

/// Errors produced while loading a mesh into the canonical [`Mesh`].
#[derive(Debug, thiserror::Error)]
pub enum MeshError {
    /// The underlying `tobj` loader rejected the OBJ (bad index, malformed
    /// coordinate, unreadable buffer, …).
    #[error("failed to parse OBJ: {0}")]
    Obj(#[from] tobj::LoadError),
    /// The mesh parsed but produced no geometry (no vertices, or no faces).
    #[error("mesh is empty (no vertices or no faces)")]
    Empty,
    /// An Arrow mesh table is missing a required column.
    #[error("mesh table is missing required column `{0}`")]
    MissingColumn(&'static str),
    /// An Arrow mesh column has an unexpected Arrow type.
    #[error("mesh column `{column}` has type {actual:?}, expected {expected}")]
    ColumnType {
        column: &'static str,
        expected: &'static str,
        actual: DataType,
    },
    /// An Arrow mesh column that must be non-null contains null values.
    #[error("mesh column `{0}` contains null values")]
    NullValues(&'static str),
    /// An `index` entry references a vertex beyond the `position` column.
    #[error("mesh index {index} is out of range (only {vertex_count} vertices)")]
    IndexOutOfRange { index: u32, vertex_count: usize },
    /// A non-indexed mesh table's vertex count is not a multiple of three, so it
    /// cannot be a triangle list.
    #[error("non-indexed mesh has {vertex_count} vertices, not a multiple of 3")]
    NonTriangleList { vertex_count: usize },
}

impl Mesh {
    /// Loads Wavefront OBJ text into an indexed [`Mesh`].
    ///
    /// Faces are triangulated and de-indexed into a single vertex/index buffer.
    /// Per-vertex colors are taken from the `v x y z r g b` extension when
    /// present, else default to white. Multiple `o`/`g` objects are merged into
    /// one mesh. Returns [`MeshError`] on malformed input or empty geometry.
    pub fn from_obj(text: &str) -> Result<Mesh, MeshError> {
        let mut reader = Cursor::new(text.as_bytes());
        let (models, _materials) = tobj::load_obj_buf(
            &mut reader,
            &tobj::LoadOptions {
                triangulate: true,
                single_index: true,
                ..tobj::LoadOptions::default()
            },
            // trd derives color from geometry / the vertex-color extension, so
            // external `.mtl` materials are never needed.
            |_| Ok((Vec::new(), Default::default())),
        )?;

        let mut vertices: Vec<Vertex> = Vec::new();
        let mut indices: Vec<u32> = Vec::new();
        for model in &models {
            let mesh = &model.mesh;
            let base = vertices.len() as u32;
            // `vertex_color` is either empty or parallel to `positions`.
            let has_color = mesh.vertex_color.len() == mesh.positions.len();
            // `texcoords` (u, v) are two-per-vertex when present (single_index).
            let has_uv = mesh.texcoords.len() == mesh.positions.len() / 3 * 2;
            for v in 0..mesh.positions.len() / 3 {
                let position = [
                    mesh.positions[3 * v],
                    mesh.positions[3 * v + 1],
                    mesh.positions[3 * v + 2],
                ];
                let color = if has_color {
                    [
                        mesh.vertex_color[3 * v],
                        mesh.vertex_color[3 * v + 1],
                        mesh.vertex_color[3 * v + 2],
                    ]
                } else {
                    DEFAULT_COLOR
                };
                // OBJ `vt` v runs bottom-up; flip to the top-left texel origin
                // used by the uploaded texture / wgpu sampler (#20).
                let uv = if has_uv {
                    [mesh.texcoords[2 * v], 1.0 - mesh.texcoords[2 * v + 1]]
                } else {
                    [0.0, 0.0]
                };
                vertices.push(Vertex {
                    position,
                    color,
                    uv,
                });
            }
            indices.extend(mesh.indices.iter().map(|&i| i + base));
        }

        if vertices.is_empty() || indices.is_empty() {
            return Err(MeshError::Empty);
        }
        Ok(Mesh { vertices, indices })
    }

    /// The axis-aligned bounding box of the mesh's vertex positions.
    ///
    /// Reuses the shared [`Aabb3`]; use [`Aabb3::center`] / [`Aabb3::size`] for
    /// preview centering + uniform scale-to-fit (#37) and camera framing (#43).
    /// Assumes a non-empty mesh (as produced by [`Mesh::from_obj`]).
    pub fn aabb(&self) -> Aabb3 {
        Aabb3::from_points(self.vertices.iter().map(|v| Point3::from_array(v.position)))
    }

    /// The center of the mesh's [`Mesh::aabb`] — the point translated to the
    /// world origin when previewing a loaded mesh (#37).
    pub fn center(&self) -> Point3 {
        self.aabb().center()
    }

    /// A uniform **preview model transform** that centers the mesh's AABB at the
    /// world origin and scales it to fit `target` world units along its largest
    /// extent: `scale(s) · translate(−center)`, `s = target / max_extent`.
    ///
    /// Applied as the default model matrix (#41) so an arbitrary-unit asset
    /// (e.g. `bunny.obj`) renders **centered and at a reasonable size** rather
    /// than off-screen or tiny (#37/#44). The scale is uniform (isotropic); a
    /// degenerate (zero-extent) mesh keeps unit scale. Compose a per-frame
    /// rotation on the left for a turntable (`rotate.then(...)` is wrong; use
    /// `preview.then(rotate)` — rotation applied after centering+scaling).
    pub fn preview_transform(&self, target: f32) -> Transform {
        let aabb = self.aabb();
        let center = aabb.center().to_array();
        let size = aabb.size();
        let max_extent = size.x().max(size.y()).max(size.z());
        let scale = if max_extent > EPSILON {
            target / max_extent
        } else {
            1.0
        };
        Transform::from_translation(Vector3::new(-center[0], -center[1], -center[2]))
            .then(Transform::from_scale(Vector3::new(scale, scale, scale)))
    }

    /// Derives a deduplicated **edge** index buffer for wireframe rendering
    /// (#38): each triangle `(a, b, c)` of the index buffer contributes its
    /// three undirected edges `(a,b)`, `(b,c)`, `(c,a)`, normalized to
    /// `(min, max)` and deduped so an edge shared by two triangles is emitted
    /// once. The result is flattened as **two indices per line** — a
    /// `PrimitiveTopology::LineList` index buffer over the same vertex buffer.
    ///
    /// A `LineList` edge buffer is preferred over `PolygonMode::Line`, which
    /// needs `Features::POLYGON_MODE_LINE` that WebGPU does not guarantee, so
    /// this stays portable across native and wasm. Pure (GPU-free) and
    /// unit-testable. Assumes a triangle list (`indices.len()` a multiple of 3,
    /// as every [`Mesh`] constructor guarantees); a trailing partial triangle,
    /// if any, is ignored.
    pub fn edge_indices(&self) -> Vec<u32> {
        use std::collections::HashSet;
        let mut seen: HashSet<(u32, u32)> = HashSet::new();
        let mut edges: Vec<u32> = Vec::new();
        for tri in self.indices.chunks_exact(3) {
            for &(a, b) in &[(tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])] {
                let key = if a <= b { (a, b) } else { (b, a) };
                if seen.insert(key) {
                    edges.push(key.0);
                    edges.push(key.1);
                }
            }
        }
        edges
    }

    /// Decodes a columnar **Arrow mesh table** into the canonical [`Mesh`] (#37).
    ///
    /// Each row of the table is one mesh (see the module docs): a required
    /// `position` `List<FixedSizeList<Float32>[3]>` column, an optional `color`
    /// `List<FixedSizeList<Float32>[3]>` (defaults to white), and an optional
    /// `index` `List<UInt32>` (absent ⇒ the vertices are a non-indexed triangle
    /// list, so their count must be a multiple of three). The **first row** is
    /// decoded; use [`Mesh::from_arrow_all`] to decode every row. Produces the
    /// same [`Mesh`] as [`Mesh::from_obj`] for equivalent geometry.
    pub fn from_arrow(batch: &RecordBatch) -> Result<Mesh, MeshError> {
        if batch.num_rows() == 0 {
            return Err(MeshError::Empty);
        }
        Self::from_arrow_row(batch, 0)
    }

    /// Decodes **every** row of an Arrow mesh table into one [`Mesh`] each,
    /// preserving row order so a stream's draw list can reference meshes by row
    /// index. Returns [`MeshError::Empty`] for a zero-row table.
    pub fn from_arrow_all(batch: &RecordBatch) -> Result<Vec<Mesh>, MeshError> {
        if batch.num_rows() == 0 {
            return Err(MeshError::Empty);
        }
        (0..batch.num_rows())
            .map(|row| Self::from_arrow_row(batch, row))
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
        Ok(Mesh { vertices, indices })
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
    use super::*;

    // The trd hello-triangle as OBJ, using the `v x y z r g b` color extension
    // so it reproduces `Mesh::hello_triangle()` exactly. Mirrors the shipped
    // `examples/triangle.obj` (kept inline so the crate build does not depend on
    // asset files the nix source filter drops).
    const TRIANGLE_OBJ: &str = "\
v 0.0 0.5 0.0 1.0 0.0 0.0
v -0.5 -0.5 0.0 0.0 1.0 0.0
v 0.5 -0.5 0.0 0.0 0.0 1.0
f 1 2 3
";
    // A unit quad (two triangles), white. Mirrors `examples/quad.obj`.
    const QUAD_OBJ: &str = "\
v -0.5 -0.5 0.0
v 0.5 -0.5 0.0
v 0.5 0.5 0.0
v -0.5 0.5 0.0
f 1 2 3 4
";

    #[test]
    fn triangle_obj_matches_hello_triangle() {
        let mesh = Mesh::from_obj(TRIANGLE_OBJ).expect("triangle parses");
        assert_eq!(mesh, Mesh::hello_triangle());
    }

    #[test]
    fn quad_polygon_fan_triangulates_to_two_triangles() {
        let mesh = Mesh::from_obj(QUAD_OBJ).expect("quad parses");
        assert_eq!(mesh.vertices.len(), 4);
        assert_eq!(mesh.indices, vec![0, 1, 2, 0, 2, 3]);
        // No color extension → all white.
        assert!(mesh.vertices.iter().all(|v| v.color == DEFAULT_COLOR));
    }

    /// Collects an edge index buffer into a set of undirected `(min, max)` pairs.
    fn edge_set(edges: &[u32]) -> std::collections::BTreeSet<(u32, u32)> {
        assert_eq!(edges.len() % 2, 0, "edge buffer must be pairs of indices");
        edges
            .chunks_exact(2)
            .map(|e| (e[0].min(e[1]), e[0].max(e[1])))
            .collect()
    }

    #[test]
    fn edge_indices_single_triangle_yields_three_edges() {
        let mesh = Mesh::hello_triangle();
        let edges = mesh.edge_indices();
        assert_eq!(edges.len(), 6, "3 edges × 2 indices");
        assert_eq!(
            edge_set(&edges),
            [(0, 1), (1, 2), (0, 2)].into_iter().collect()
        );
    }

    #[test]
    fn edge_indices_quad_dedups_shared_diagonal() {
        // Two triangles [0,1,2] + [0,2,3] share the diagonal edge (0,2), so the
        // quad has 5 unique edges (4 sides + 1 diagonal), not 6.
        let mesh = Mesh::from_obj(QUAD_OBJ).expect("quad parses");
        let edges = mesh.edge_indices();
        assert_eq!(edges.len(), 10, "5 unique edges × 2 indices");
        assert_eq!(
            edge_set(&edges),
            [(0, 1), (1, 2), (0, 2), (2, 3), (0, 3)]
                .into_iter()
                .collect()
        );
    }

    #[test]
    fn face_index_forms_use_position_only() {
        // `v/vt/vn` form must resolve to the position index buffer [0,1,2].
        let obj = "\
v 0 0 0
v 1 0 0
v 0 1 0
vt 0 0
vt 1 0
vt 0 1
vn 0 0 1
f 1/1/1 2/2/1 3/3/1
";
        let mesh = Mesh::from_obj(obj).unwrap();
        assert_eq!(mesh.indices, vec![0, 1, 2]);
        assert_eq!(mesh.vertices.len(), 3);
    }

    #[test]
    fn negative_indices_are_relative() {
        let obj = "\
v 0 0 0
v 1 0 0
v 0 1 0
f -3 -2 -1
";
        let mesh = Mesh::from_obj(obj).unwrap();
        assert_eq!(mesh.indices, vec![0, 1, 2]);
    }

    #[test]
    fn all_vertex_colors_are_parsed() {
        let obj = "\
v 0 0 0 0.25 0.5 0.75
v 1 0 0 0.1 0.2 0.3
v 0 1 0 0.9 0.8 0.7
f 1 2 3
";
        let mesh = Mesh::from_obj(obj).unwrap();
        assert_eq!(mesh.vertices[0].color, [0.25, 0.5, 0.75]);
        assert_eq!(mesh.vertices[1].color, [0.1, 0.2, 0.3]);
    }

    #[test]
    fn multiple_objects_are_merged_with_offset_indices() {
        let obj = "\
o first
v 0 0 0
v 1 0 0
v 0 1 0
f 1 2 3
o second
v 2 0 0
v 3 0 0
v 2 1 0
f 4 5 6
";
        let mesh = Mesh::from_obj(obj).unwrap();
        assert_eq!(mesh.vertices.len(), 6);
        assert_eq!(mesh.indices, vec![0, 1, 2, 3, 4, 5]);
    }

    #[test]
    fn comments_and_blank_lines_are_ignored() {
        let obj = "\
# a comment

v 0 0 0
v 1 0 0
v 0 1 0
f 1 2 3
";
        let mesh = Mesh::from_obj(obj).unwrap();
        assert_eq!(mesh.vertices.len(), 3);
        assert_eq!(mesh.indices, vec![0, 1, 2]);
    }

    #[test]
    fn out_of_range_index_errors() {
        let err = Mesh::from_obj("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 4\n");
        assert!(err.is_err(), "out-of-range face index should error");
    }

    #[test]
    fn empty_input_is_empty_error() {
        assert!(matches!(
            Mesh::from_obj("# nothing here\n"),
            Err(MeshError::Empty)
        ));
    }

    #[test]
    fn vertices_without_faces_is_empty_error() {
        assert!(matches!(
            Mesh::from_obj("v 0 0 0\nv 1 0 0\nv 0 1 0\n"),
            Err(MeshError::Empty)
        ));
    }

    #[test]
    fn aabb_and_center_of_quad() {
        let mesh = Mesh::from_obj(QUAD_OBJ).unwrap();
        let aabb = mesh.aabb();
        assert_eq!(aabb.min(), Point3::new(-0.5, -0.5, 0.0));
        assert_eq!(aabb.max(), Point3::new(0.5, 0.5, 0.0));
        assert_eq!(mesh.center(), Point3::new(0.0, 0.0, 0.0));
    }

    // ---- Arrow mesh table (#37) ----

    use arrow::array::{
        ArrayRef, FixedSizeListArray, Float32Array, ListArray, RecordBatch, UInt32Array,
    };
    use arrow::buffer::OffsetBuffer;
    use arrow::datatypes::{DataType, Field, Schema};
    use std::sync::Arc;

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

    // ---- preview transform (#37) ----

    #[test]
    fn preview_transform_centers_and_scales_to_target() {
        // A 4-unit-wide box centered at (10, 0, 0): after the preview transform
        // its center is at the origin and its largest extent spans `target`.
        let mesh = Mesh {
            vertices: vec![
                Vertex {
                    position: [8.0, -1.0, -0.5],
                    color: DEFAULT_COLOR,
                    uv: [0.0, 0.0],
                },
                Vertex {
                    position: [12.0, 1.0, 0.5],
                    color: DEFAULT_COLOR,
                    uv: [0.0, 0.0],
                },
            ],
            indices: vec![0, 1, 0],
        };
        let target = 2.0;
        let t = mesh.preview_transform(target);
        // Center maps to origin.
        let c = t.transform_point(Point3::new(10.0, 0.0, 0.0));
        assert!(
            c.to_array().iter().all(|v| v.abs() < 1e-5),
            "center = {c:?}"
        );
        // The transformed AABB's largest extent equals `target`.
        let out = t.transform_aabb(mesh.aabb());
        let size = out.size();
        let max_extent = size.x().max(size.y()).max(size.z());
        assert!(
            (max_extent - target).abs() < 1e-5,
            "max_extent = {max_extent}"
        );
    }

    #[test]
    fn preview_transform_degenerate_mesh_keeps_unit_scale() {
        let mesh = Mesh {
            vertices: vec![Vertex {
                position: [3.0, 3.0, 3.0],
                color: DEFAULT_COLOR,
                uv: [0.0, 0.0],
            }],
            indices: vec![0],
        };
        let t = mesh.preview_transform(2.0);
        // A single point has zero extent → unit scale, only the centering shift.
        let moved = t.transform_point(Point3::new(3.0, 3.0, 3.0));
        assert!(moved.to_array().iter().all(|v| v.abs() < 1e-5));
    }
}
