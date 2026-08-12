//! GPU-side mesh storage: the decode-once mesh store.
//!
//! [`MeshGpu`] is one uploaded mesh (vertex/index buffers, its bound albedo and
//! material maps, its AABB gizmo geometry and base model); [`MeshStore`] owns
//! every mesh plus the geometry shared by all gizmo drawables (axes, plane
//! grids, quad outlines, the shadow quad) and the one instance buffer every
//! draw kind writes through.
//!
//! Named `mesh_store` rather than `mesh` because `crate::mesh` already holds the
//! CPU-side [`Mesh`]; this is its GPU face.

use super::bound_material_maps::BoundMaterialMaps;
use super::bound_texture::BoundTexture;
use super::buffer::{create_instance_buffer, IndexBuf, VertexGeometry};
use super::*;
use crate::math::Matrix4;
use crate::scene::GridPlane;

/// A mesh uploaded to the GPU. Its `vertex_buffer` feeds both the filled
/// `triangles` and the deduped wireframe `edges` (#38); the `aabb` overlay (#42)
/// is a standalone box of 12 screen-space-expanded edge quads. `base_model` is
/// the base (preview) transform pre-multiplied beneath every per-frame instance
/// model (`effective = model · base`).
pub(super) struct MeshGpu {
    pub(super) vertex_buffer: wgpu::Buffer,
    /// Parallel vertex buffer for the Disney PBR path (`pbr.wgsl`): the same
    /// positions + UVs as `vertex_buffer`, but with a derived smooth shading
    /// **normal** in place of the vertex color. Reuses the `triangles` index
    /// buffer. Built once per mesh; only bound by [`RenderMode::Pbr`] draws.
    pub(super) pbr_vertex_buffer: wgpu::Buffer,
    pub(super) triangles: IndexBuf,
    pub(super) edges: IndexBuf,
    pub(super) aabb: VertexGeometry,
    pub(super) base_model: Matrix4,
    /// This mesh's **own** albedo texture (group 1), so a multi-object scene skins
    /// each object with its own diffuse (#141). Defaults to 1×1 white (identity
    /// albedo) until [`set`](BoundTexture::set) via `set_mesh_texture`.
    pub(super) texture: BoundTexture,
    pub(super) material_maps: BoundMaterialMaps,
}

impl MeshGpu {
    pub(super) fn filled(&self) -> (&wgpu::Buffer, &IndexBuf) {
        (&self.vertex_buffer, &self.triangles)
    }

    pub(super) fn pbr(&self) -> (&wgpu::Buffer, &IndexBuf) {
        (&self.pbr_vertex_buffer, &self.triangles)
    }

    pub(super) fn wireframe(&self) -> (&wgpu::Buffer, &IndexBuf) {
        (&self.vertex_buffer, &self.edges)
    }

    pub(super) fn aabb(&self) -> &VertexGeometry {
        &self.aabb
    }
}

pub(super) fn upload_mesh(
    gpu: &GpuContext,
    mesh: &Mesh,
    base_model: Matrix4,
    texture_layout: &wgpu::BindGroupLayout,
    material_maps_layout: &wgpu::BindGroupLayout,
) -> MeshGpu {
    use wgpu::util::DeviceExt;

    let vertex_buffer = gpu
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("trd mesh vertex buffer"),
            contents: bytemuck::cast_slice(&mesh.vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });
    let triangles = IndexBuf::new(&gpu.device, "trd mesh index buffer", &mesh.indices);
    let edges = mesh.edge_indices();
    let edges = IndexBuf::new(&gpu.device, "trd mesh edge buffer", &edges);

    // PBR vertex buffer (#): derive area-weighted smooth normals (the assets have
    // no `vn`) and pack position + normal + UV for `pbr.wgsl`, reusing the
    // triangle index buffer above.
    let normals = mesh
        .shading
        .as_ref()
        .filter(|shading| shading.normals.len() == mesh.vertices.len())
        .map(|shading| shading.normals.clone())
        .unwrap_or_else(|| compute_smooth_normals(&mesh.vertices, &mesh.indices));
    let tangents = mesh
        .shading
        .as_ref()
        .filter(|shading| shading.tangents.len() == mesh.vertices.len())
        .map(|shading| shading.tangents.clone())
        .unwrap_or_else(|| compute_tangents(&mesh.vertices, &mesh.indices, &normals));
    let pbr_vertices: Vec<PbrVertex> = mesh
        .vertices
        .iter()
        .zip(&normals)
        .zip(&tangents)
        .map(|((v, &normal), &tangent)| PbrVertex {
            position: v.position,
            normal,
            uv: v.uv,
            tangent,
        })
        .collect();
    let pbr_vertex_buffer = gpu
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("trd mesh pbr vertex buffer"),
            contents: bytemuck::cast_slice(&pbr_vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });

    // AABB overlay box: the mesh's own bounding box (mesh-local coords) as 8
    // colored corner vertices + a 12-edge line list. Built once per mesh; drawn
    // only when the scene contains an `AabbBox` for this mesh.
    let aabb_corners = mesh.aabb().corners().map(|corner| corner.to_array());
    let aabb_vertices = aabb_line_vertices(&aabb_corners);

    MeshGpu {
        vertex_buffer,
        pbr_vertex_buffer,
        triangles,
        edges,
        aabb: VertexGeometry::new(&gpu.device, "trd mesh aabb line buffer", &aabb_vertices),
        base_model,
        texture: BoundTexture::with_layout(gpu, texture_layout.clone()),
        material_maps: BoundMaterialMaps::with_layout(gpu, material_maps_layout.clone()),
    }
}

/// The decode-once geometry store: the uploaded [`MeshGpu`]s (referenced by a
/// scene's mesh ids), the shared coordinate-axes gizmo vertices, and the
/// growable per-instance model-matrix buffer.
pub(super) struct MeshStore {
    pub(super) meshes: Vec<MeshGpu>,
    /// The coordinate-axis shafts and cone arrowheads.
    pub(super) axes_lines: VertexGeometry,
    pub(super) axes_heads: VertexGeometry,
    /// The coordinate-plane grid geometry, one expanded-line buffer per
    /// [`GridPlane`] (indexed by [`GridPlane::index`]): XY, XZ, YZ. Each
    /// [`DrawableObject::PlaneGrid`] draws the buffer for its plane under its own
    /// model, supplied through the shared instance buffer.
    pub(super) grid_lines: [VertexGeometry; 3],
    pub(super) quad_lines: [VertexGeometry; 2],
    /// The contact / blob **grounding-shadow** quad geometry (six `TriangleList`
    /// vertices, a unit XY quad); each [`DrawableObject::BlobShadow`] draws it
    /// under its own model through the shared instance buffer, alpha-blended.
    pub(super) shadow_vertex_buffer: wgpu::Buffer,
    pub(super) instance_buffer: wgpu::Buffer,
    pub(super) instance_capacity: u32,
}

impl MeshStore {
    /// Constructs a `MeshStore`, uploading each mesh with its base (preview)
    /// model and sizing the instance buffer to at least one instance.
    pub(super) fn new(
        gpu: &GpuContext,
        meshes: &[Mesh],
        base_models: &[Matrix4],
        texture_layout: &wgpu::BindGroupLayout,
        material_maps_layout: &wgpu::BindGroupLayout,
    ) -> Self {
        use wgpu::util::DeviceExt;

        let gpu_meshes = meshes
            .iter()
            .zip(base_models)
            .map(|(mesh, &base)| upload_mesh(gpu, mesh, base, texture_layout, material_maps_layout))
            .collect();
        let instance_capacity = (meshes.len() as u32).max(1);
        let instance_buffer = create_instance_buffer(&gpu.device, instance_capacity);

        let axes_lines =
            VertexGeometry::new(&gpu.device, "trd axes line buffer", &axes_line_vertices());
        let axes_heads =
            VertexGeometry::new(&gpu.device, "trd axes arrow buffer", &axes_arrow_vertices());

        // Coordinate-plane grids: one expanded-line vertex buffer per plane
        // (XY/XZ/YZ), drawn under each PlaneGrid object's model.
        let grid_lines = [
            VertexGeometry::new(
                &gpu.device,
                "trd xy grid line buffer",
                &grid_line_vertices(GridPlane::Xy),
            ),
            VertexGeometry::new(
                &gpu.device,
                "trd xz grid line buffer",
                &grid_line_vertices(GridPlane::Xz),
            ),
            VertexGeometry::new(
                &gpu.device,
                "trd yz grid line buffer",
                &grid_line_vertices(GridPlane::Yz),
            ),
        ];
        let quad_lines = [
            VertexGeometry::new(
                &gpu.device,
                "trd placement quad line buffer",
                &quad_outline_vertices(false),
            ),
            VertexGeometry::new(
                &gpu.device,
                "trd selected placement quad line buffer",
                &quad_outline_vertices(true),
            ),
        ];

        // Contact / blob grounding-shadow quad: six TriangleList vertices (a unit
        // XY quad). Each BlobShadow drawable draws them under its own model via
        // the shared instance buffer, alpha-blended over the frame plane.
        let shadow_vertex_buffer =
            gpu.device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("trd shadow vertex buffer"),
                    contents: bytemuck::cast_slice(&blob_shadow_vertices()),
                    usage: wgpu::BufferUsages::VERTEX,
                });

        Self {
            meshes: gpu_meshes,
            axes_lines,
            axes_heads,
            grid_lines,
            quad_lines,
            shadow_vertex_buffer,
            instance_buffer,
            instance_capacity,
        }
    }

    pub(super) fn len(&self) -> usize {
        self.meshes.len()
    }

    /// Uploads the flattened instance models, growing the buffer (to the next
    /// power of two) when the frame needs more instances than it holds.
    pub(super) fn upload_instances(&mut self, gpu: &GpuContext, instances: &[InstanceRaw]) {
        let (device, queue) = (&gpu.device, &gpu.queue);
        if instances.len() as u32 > self.instance_capacity {
            self.instance_capacity = (instances.len() as u32).next_power_of_two();
            self.instance_buffer = create_instance_buffer(device, self.instance_capacity);
        }
        if !instances.is_empty() {
            queue.write_buffer(&self.instance_buffer, 0, bytemuck::cast_slice(instances));
        }
    }
}
