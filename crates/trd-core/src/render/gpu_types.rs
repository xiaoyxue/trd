//! GPU byte-layout types: the camera uniforms and the vertex/instance buffer
//! layouts.
//!
//! Every type here is `#[repr(C)]` + [`bytemuck::Pod`], so it can be
//! `cast_slice`d straight into a GPU buffer. The heap-owning [`Mesh`] that
//! *supplies* those bytes is not one of them and lives at the crate root
//! (`crate::mesh`, #221).

use crate::math::Matrix4;
use crate::Camera;

/// GPU uniform matching the WGSL `Params` layout: a single column-major 4×4
/// matrix (64 bytes) storing the camera-only `P · V` (each instance supplies
/// its own model matrix).
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct Uniform {
    transform: [f32; 16],
}

impl Uniform {
    pub(crate) fn view_proj(camera: Camera) -> Self {
        Uniform {
            transform: camera.view_projection().matrix().to_cols_array(),
        }
    }
}

/// GPU uniform for screen-space gizmo lines. The camera matrix stays byte-for-byte
/// identical to [`Uniform`]; the extra vec4 carries viewport size in `xy` and
/// clip-space pixel scale (`2 / size`) in `zw`.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct GizmoUniform {
    view_proj: [f32; 16],
    viewport: [f32; 4],
}

impl GizmoUniform {
    pub(crate) fn new(camera: Camera) -> Self {
        let viewport = camera.viewport();
        let width = viewport.width.max(1) as f32;
        let height = viewport.height.max(1) as f32;
        Self {
            view_proj: camera.view_projection().matrix().to_cols_array(),
            viewport: [width, height, 2.0 / width, 2.0 / height],
        }
    }
}

/// A mesh vertex consumed by `mesh.wgsl` / `textured.wgsl` — and the **authored**
/// vertex record: it is [`Mesh::vertices`](crate::Mesh)' element, decoded from
/// OBJ, glTF and the `0.0.6` wire, so it is `pub` and compared for equality.
///
/// **Why this and [`ShadingVertex`] are two types** (#247): they answer
/// different questions. This one is what an *asset* carries; `ShadingVertex` is
/// what the Disney path *derives* at upload (smooth normals + tangents the OBJ
/// assets do not have), bound beside it at a second vertex slot. Merging them — one record with an "optional" tangent — is not
/// expressible: a vertex buffer has a fixed `array_stride`, so "optional" can
/// only mean "a field that is sometimes zero", paid on every vertex of every
/// mesh in all six pipelines, and it would push derived data into a public,
/// wire-decoded type every producer would then have to fill. The optionality
/// that *is* real lives one level up, on
/// [`MeshShading`](crate::MeshShading), whose tangents may legitimately be
/// absent. An enum over the two is the same idea in a worse place: a
/// discriminant per vertex, not `Pod`, and unusable as a GPU record — which
/// layout a buffer holds is a property of the **buffer**, and `VertexBuffer<T>`
/// already says it statically (#247 S2).
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

/// One vertex of a screen-space-expanded gizmo line quad. Every six vertices
/// describe one segment: `start`/`end` are shared, while `extrusion` stores the
/// endpoint selector, side (`-1`/`+1`), and full line width in pixels.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct GizmoLineVertex {
    pub(crate) start: [f32; 3],
    pub(crate) end: [f32; 3],
    pub(crate) color: [f32; 3],
    pub(crate) extrusion: [f32; 3],
}

impl GizmoLineVertex {
    const ATTRIBUTES: [wgpu::VertexAttribute; 4] = [
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
            format: wgpu::VertexFormat::Float32x3,
            offset: 24,
            shader_location: 2,
        },
        // Locations 3-6 remain the shared per-instance model matrix.
        wgpu::VertexAttribute {
            format: wgpu::VertexFormat::Float32x3,
            offset: 36,
            shader_location: 7,
        },
    ];

    pub(crate) const fn layout() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Self>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &Self::ATTRIBUTES,
        }
    }
}

/// The **derived** half of a shaded mesh vertex: the smooth shading normal
/// (loc 8) and tangent (loc 9), bound at vertex slot **2** by `pbr.wgsl` beside
/// the ordinary [`Vertex`] buffer every other pipeline reads (#247 S7).
///
/// It is a separate record rather than fields on [`Vertex`] because it is
/// *derived*, not authored: the OBJ assets carry no `vn`, so
/// [`compute_smooth_normals`](super::pbr::compute_smooth_normals) and
/// [`compute_tangents`](super::pbr::compute_tangents) fill it at upload unless
/// the asset supplies its own through [`MeshShading`](crate::MeshShading).
/// Putting it on `Vertex` would push derived data into a public, wire-decoded
/// type and make every other pipeline fetch 28 bytes it never reads.
///
/// It is a separate *buffer* rather than a wider PBR vertex — which is what it
/// used to be, `PbrVertex { position, normal, uv, tangent }` — because that
/// stored `position` and `uv` a second time: 20 of its 48 bytes duplicated the
/// mesh's own vertex buffer, 80 B/vertex across the two. Splitting the derived
/// half off costs one extra `set_vertex_buffer` per shaded draw and stores
/// 60 B/vertex instead.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct ShadingVertex {
    pub(crate) normal: [f32; 3],
    /// xyz = tangent, w = handedness used to reconstruct the bitangent.
    pub(crate) tangent: [f32; 4],
}

impl ShadingVertex {
    // 0-2 are `Vertex`, 3-6 the instance model, 7 the gizmo line's extrusion.
    const ATTRIBUTES: [wgpu::VertexAttribute; 2] = [
        wgpu::VertexAttribute {
            format: wgpu::VertexFormat::Float32x3,
            offset: 0,
            shader_location: 8,
        },
        wgpu::VertexAttribute {
            format: wgpu::VertexFormat::Float32x4,
            offset: 12,
            shader_location: 9,
        },
    ];

    /// The slot-2 vertex buffer layout expected by `pbr.wgsl`.
    pub(crate) const fn layout() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Self>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &Self::ATTRIBUTES,
        }
    }
}

/// Per-instance model matrix fed to `mesh.wgsl` as four `vec4` instance/// attributes (shader locations 2-5, column-major, 64-byte stride).
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct InstanceRaw {
    /// Typed, but byte-identical to the old `[f32; 16]`: `Matrix4` is
    /// `#[repr(transparent)]` over `glam::Mat4` and already `Pod`, so the record
    /// keeps its 64-byte stride and column-major layout (#235 R3).
    pub(crate) model: Matrix4,
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

/// Per-instance data for the object-id picking pass (`picking.wgsl`): a model
/// matrix (four `vec4` attributes, locations 3-6) plus a flat `id_color` (the
/// object id encoded as RGBA, location 7). Rendered single-sampled so each id
/// color reads back exactly.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct PickInstanceRaw {
    pub(crate) model: Matrix4,
    pub(crate) id_color: [f32; 4],
}

impl PickInstanceRaw {
    const ATTRIBUTES: [wgpu::VertexAttribute; 5] = [
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
        wgpu::VertexAttribute {
            format: wgpu::VertexFormat::Float32x4,
            offset: 64,
            shader_location: 7,
        },
    ];

    /// Encodes a 0-based object index into a `PickInstanceRaw` id color. The
    /// stored id is `index + 1` so `0` is reserved for the cleared background;
    /// the 24-bit RGB packing round-trips through the linear `Rgba8Unorm` pick
    /// target (each byte `/ 255.0`).
    pub(crate) fn new(model: Matrix4, index: u32) -> Self {
        let id = index + 1;
        let id_color = [
            (id & 0xFF) as f32 / 255.0,
            ((id >> 8) & 0xFF) as f32 / 255.0,
            ((id >> 16) & 0xFF) as f32 / 255.0,
            1.0,
        ];
        Self { model, id_color }
    }

    /// Decodes a read-back RGBA pixel from the pick target into a 0-based object
    /// index, or `None` for the background (id `0`). Inverse of [`Self::new`].
    pub(crate) fn decode(rgba: [u8; 4]) -> Option<u32> {
        let id = rgba[0] as u32 | ((rgba[1] as u32) << 8) | ((rgba[2] as u32) << 16);
        id.checked_sub(1)
    }

    /// Returns the per-instance buffer layout expected by `picking.wgsl`.
    pub(crate) const fn layout() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Self>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &Self::ATTRIBUTES,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::Matrix4;
    use crate::render::{FrameParams, Viewport};

    #[test]
    fn pick_instance_id_round_trips_through_rgba() {
        // index 0 -> id 1, encoded low byte, decodes back to 0.
        for index in [0u32, 1, 2, 42, 255, 256, 300, 65_535, 70_000] {
            let inst = PickInstanceRaw::new(Matrix4::IDENTITY, index);
            // The id color bytes are the pick target's stored RGBA (× 255, exact).
            let rgba = [
                (inst.id_color[0] * 255.0).round() as u8,
                (inst.id_color[1] * 255.0).round() as u8,
                (inst.id_color[2] * 255.0).round() as u8,
                (inst.id_color[3] * 255.0).round() as u8,
            ];
            assert_eq!(PickInstanceRaw::decode(rgba), Some(index), "index {index}");
        }
    }

    #[test]
    fn pick_decode_treats_zero_as_background() {
        // A cleared (black) pixel is id 0 → no object.
        assert_eq!(PickInstanceRaw::decode([0, 0, 0, 255]), None);
        assert_eq!(PickInstanceRaw::decode([0, 0, 0, 0]), None);
    }

    #[test]
    fn pick_instance_layout_stride_and_id_offset() {
        // model (64) + id_color (16) = 80 bytes; id color attribute at offset 64.
        assert_eq!(std::mem::size_of::<PickInstanceRaw>(), 80);
        let attrs = PickInstanceRaw::ATTRIBUTES;
        assert_eq!(attrs[4].shader_location, 7);
        assert_eq!(attrs[4].offset, 64);
    }

    #[test]
    fn uniform_layout_matches_wgsl_params() {
        // One column-major 4x4 f32 matrix = 64 bytes.
        assert_eq!(std::mem::size_of::<Uniform>(), 64);
        let viewport = Viewport {
            width: 8,
            height: 4,
        };
        assert_eq!(
            Uniform::view_proj(FrameParams::IDENTITY.to_camera(viewport).unwrap()).transform,
            Matrix4::IDENTITY.to_cols_array()
        );
    }

    #[test]
    fn gizmo_uniform_layout_matches_wgsl_params() {
        assert_eq!(std::mem::size_of::<GizmoUniform>(), 80);
        let viewport = Viewport {
            width: 8,
            height: 4,
        };
        let uniform = GizmoUniform::new(FrameParams::IDENTITY.to_camera(viewport).unwrap());
        assert_eq!(uniform.view_proj, Matrix4::IDENTITY.to_cols_array());
        assert_eq!(uniform.viewport, [8.0, 4.0, 0.25, 0.5]);
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
    fn gizmo_line_vertex_layout_matches_wgsl_inputs() {
        assert_eq!(std::mem::size_of::<GizmoLineVertex>(), 48);
        let layout = GizmoLineVertex::layout();
        assert_eq!(layout.array_stride, 48);
        assert_eq!(layout.step_mode, wgpu::VertexStepMode::Vertex);
        assert_eq!(
            layout
                .attributes
                .iter()
                .map(|attribute| attribute.shader_location)
                .collect::<Vec<_>>(),
            [0, 1, 2, 7]
        );
    }
}
