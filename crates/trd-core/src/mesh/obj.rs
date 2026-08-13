//! Wavefront OBJ loading into the canonical [`Mesh`] (#36).
//!
//! OBJ is trd's **default** authorable mesh format. Rather than hand-roll a
//! parser, we normalize OBJ into the canonical [`Mesh`] via the popular,
//! pure-Rust [`tobj`] loader (wasm-compatible) — following the repo convention
//! that each mesh format is decoded by its own established loader crate.
//!
//! We load with triangulation + a single index buffer, and read the common
//! `v x y z r g b` **vertex-color extension** (via [`tobj::Mesh::vertex_color`])
//! so colored OBJ vertices survive; uncolored meshes default to white. Normals
//! are ignored for now (reserved for shading / #20). Parsing is from an
//! in-memory buffer (no file I/O), so it works natively and on wasm.

use std::io::Cursor;

use super::{Mesh, MeshError, DEFAULT_COLOR};
use crate::render::Vertex;

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
        Ok(Mesh {
            vertices,
            indices,
            shading: None,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::{QUAD_OBJ, TRIANGLE_OBJ};

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
}
