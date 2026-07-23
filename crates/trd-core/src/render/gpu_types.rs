//! GPU byte-layout types: the camera uniform, the vertex/instance
//! buffer layouts, and the indexed mesh container.

use super::{FrameParams, Viewport};

/// GPU uniform matching the WGSL `Params` layout: a single column-major 4×4
/// matrix (64 bytes) storing the camera-only `P · V` (each instance supplies
/// its own model matrix).
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct Uniform {
    transform: [f32; 16],
}

impl Uniform {
    pub(crate) fn view_proj(params: FrameParams, viewport: Viewport) -> Self {
        Uniform {
            transform: params.view_proj_matrix(viewport).to_cols_array(),
        }
    }
}

/// A mesh vertex consumed by `mesh.wgsl` / `textured.wgsl`.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Vertex {
    pub position: [f32; 3],
    pub color: [f32; 3],
    /// Texture coordinate (#20). `[0, 0]` for untextured/gizmo geometry; the
    /// textured pipeline samples the bound texture at this UV.
    pub uv: [f32; 2],
}

impl Vertex {
    const ATTRIBUTES: [wgpu::VertexAttribute; 3] = [
        wgpu::VertexAttribute {
            format: wgpu::VertexFormat::Float32x3,
            offset: 0,
            shader_location: 0,
        },
        wgpu::VertexAttribute {
            format: wgpu::VertexFormat::Float32x3,
            offset: 12,
            shader_location: 1,
        },
        wgpu::VertexAttribute {
            format: wgpu::VertexFormat::Float32x2,
            offset: 24,
            shader_location: 2,
        },
    ];

    /// Returns the vertex buffer layout expected by `mesh.wgsl`.
    pub const fn layout() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Self>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &Self::ATTRIBUTES,
        }
    }
}

/// Per-instance model matrix fed to `mesh.wgsl` as four `vec4` instance
/// attributes (shader locations 2-5, column-major, 64-byte stride).
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct InstanceRaw {
    pub(crate) model: [f32; 16],
}

impl InstanceRaw {
    const ATTRIBUTES: [wgpu::VertexAttribute; 4] = [
        wgpu::VertexAttribute {
            format: wgpu::VertexFormat::Float32x4,
            offset: 0,
            shader_location: 3,
        },
        wgpu::VertexAttribute {
            format: wgpu::VertexFormat::Float32x4,
            offset: 16,
            shader_location: 4,
        },
        wgpu::VertexAttribute {
            format: wgpu::VertexFormat::Float32x4,
            offset: 32,
            shader_location: 5,
        },
        wgpu::VertexAttribute {
            format: wgpu::VertexFormat::Float32x4,
            offset: 48,
            shader_location: 6,
        },
    ];

    /// Returns the per-instance buffer layout expected by `mesh.wgsl`.
    pub(crate) const fn layout() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Self>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &Self::ATTRIBUTES,
        }
    }
}

/// Canonical indexed mesh container.
#[derive(Debug, Clone, PartialEq)]
pub struct Mesh {
    pub vertices: Vec<Vertex>,
    pub indices: Vec<u32>,
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
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::Matrix4;

    #[test]
    fn uniform_layout_matches_wgsl_params() {
        // One column-major 4x4 f32 matrix = 64 bytes.
        assert_eq!(std::mem::size_of::<Uniform>(), 64);
        let viewport = Viewport {
            width: 8,
            height: 4,
        };
        assert_eq!(
            Uniform::view_proj(FrameParams::IDENTITY, viewport).transform,
            Matrix4::IDENTITY.to_cols_array()
        );
    }

    #[test]
    fn vertex_layout_matches_wgsl_inputs() {
        assert_eq!(std::mem::size_of::<Vertex>(), 32);
        assert_eq!(std::mem::align_of::<Vertex>(), 4);

        let layout = Vertex::layout();
        assert_eq!(layout.array_stride, 32);
        assert_eq!(layout.step_mode, wgpu::VertexStepMode::Vertex);
        assert_eq!(layout.attributes.len(), 3);
        assert_eq!(layout.attributes[0].offset, 0);
        assert_eq!(layout.attributes[0].shader_location, 0);
        assert_eq!(layout.attributes[0].format, wgpu::VertexFormat::Float32x3);
        assert_eq!(layout.attributes[1].offset, 12);
        assert_eq!(layout.attributes[1].shader_location, 1);
        assert_eq!(layout.attributes[1].format, wgpu::VertexFormat::Float32x3);
        assert_eq!(layout.attributes[2].offset, 24);
        assert_eq!(layout.attributes[2].shader_location, 2);
        assert_eq!(layout.attributes[2].format, wgpu::VertexFormat::Float32x2);
    }

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
