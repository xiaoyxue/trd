//! The canonical indexed mesh container every loader converges on.
//!
//! [`Mesh`] owns heap buffers and is therefore **not** a GPU byte layout: its
//! vertices are `cast_slice`d into a buffer at upload time, but the struct
//! itself is never uploaded. Its dependence on [`Vertex`] is real rather than
//! accidental — trd's mesh is *the mesh this renderer eats*, and the shader
//! defines that vertex layout — so [`Vertex`] stays in `render/gpu_types.rs`
//! with the other `repr(C)` + `Pod` types while the mesh lives here (#221).

use crate::render::Vertex;

/// Per-vertex shading attributes carried alongside a [`Mesh`]'s vertices.
#[derive(Debug, Clone, PartialEq)]
pub struct MeshShading {
    pub normals: Vec<[f32; 3]>,
    /// Optional authored tangents. Empty means derive them from positions/UVs.
    pub tangents: Vec<[f32; 4]>,
}

/// Canonical indexed mesh container.
#[derive(Debug, Clone, PartialEq)]
pub struct Mesh {
    pub vertices: Vec<Vertex>,
    pub indices: Vec<u32>,
    pub shading: Option<MeshShading>,
}

impl Mesh {
    /// The legacy hello-triangle expressed as a 3-vertex indexed mesh.
    pub fn hello_triangle() -> Self {
        Self {
            vertices: vec![
                Vertex {
                    position: [0.0, 0.5, 0.0],
                    color: [1.0, 0.0, 0.0],
                    uv: [0.0, 0.0],
                },
                Vertex {
                    position: [-0.5, -0.5, 0.0],
                    color: [0.0, 1.0, 0.0],
                    uv: [0.0, 0.0],
                },
                Vertex {
                    position: [0.5, -0.5, 0.0],
                    color: [0.0, 0.0, 1.0],
                    uv: [0.0, 0.0],
                },
            ],
            indices: vec![0, 1, 2],
            shading: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hello_triangle_mesh_matches_shader_constants() {
        let mesh = Mesh::hello_triangle();
        assert_eq!(
            mesh.vertices,
            vec![
                Vertex {
                    position: [0.0, 0.5, 0.0],
                    color: [1.0, 0.0, 0.0],
                    uv: [0.0, 0.0],
                },
                Vertex {
                    position: [-0.5, -0.5, 0.0],
                    color: [0.0, 1.0, 0.0],
                    uv: [0.0, 0.0],
                },
                Vertex {
                    position: [0.5, -0.5, 0.0],
                    color: [0.0, 0.0, 1.0],
                    uv: [0.0, 0.0],
                },
            ]
        );
        assert_eq!(mesh.indices, vec![0, 1, 2]);
    }
}
