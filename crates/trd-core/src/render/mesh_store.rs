//! GPU-side mesh storage: the decode-once mesh store.
//!
//! [`MeshGpu`] is one uploaded mesh (vertex/index buffers, its bound albedo and
//! material maps, its AABB gizmo geometry and base model); [`MeshStore`] owns
//! every mesh a scene's ids can name, and **only** those — the constant gizmo
//! geometry and the per-frame instance buffer moved to their own owners in
//! `gizmo.rs` and `buffer.rs` (#222).
//!
//! Named `mesh_store` rather than `mesh` because `crate::mesh` already holds the
//! CPU-side [`Mesh`]; this is its GPU face.

use super::bound_material_maps::BoundMaterialMaps;
use super::bound_texture::BoundTexture;
use super::buffer::{IndexBuffer, VertexGeometry};
use super::*;
use crate::material::DisneyMaterial;
use crate::math::Matrix4;

/// A mesh uploaded to the GPU. Its `vertex_buffer` feeds both the filled
/// `triangles` and the deduped wireframe `edges` (#38); the `aabb` overlay (#42)
/// is a standalone box of 12 screen-space-expanded edge quads. `base_model` is
/// the base (preview) transform pre-multiplied beneath every per-frame instance
/// model (`effective = model · base`).
///
/// It owns **all** of one mesh's per-object state, shading included: the
/// material, IBL gain, tone map and debug view used to sit on the renderer as
/// four `Vec`s parallel to the mesh store, each allocated to the mesh count with
/// nothing keeping them that length — one concept in two storage schemes, of
/// which only the `Vec`s could fall out of sync (#203). They are scalars here
/// because the `Vec` already exists one level up, in
/// [`MeshStore::meshes`](super::mesh_store::MeshStore).
pub(super) struct MeshGpu {
    pub(super) vertex_buffer: wgpu::Buffer,
    /// Parallel vertex buffer for the Disney PBR path (`pbr.wgsl`): the same
    /// positions + UVs as `vertex_buffer`, but with a derived smooth shading
    /// **normal** in place of the vertex color. Reuses the `triangles` index
    /// buffer. Built once per mesh; only bound by [`RenderMode::Shaded`] draws.
    pub(super) pbr_vertex_buffer: wgpu::Buffer,
    pub(super) triangles: IndexBuffer,
    pub(super) edges: IndexBuffer,
    pub(super) aabb: VertexGeometry,
    pub(super) base_model: Matrix4,
    /// This mesh's **own** albedo texture (group 1), so a multi-object scene skins
    /// each object with its own diffuse (#141). Defaults to 1×1 white (identity
    /// albedo) until [`set`](BoundTexture::set) via `set_mesh_texture`.
    pub(super) texture: BoundTexture,
    pub(super) material_maps: BoundMaterialMaps,
    /// The Disney material applied to this mesh's [`RenderMode::Shaded`] draws
    /// (#141) — so a multi-object scene gives every object its own
    /// metallic/roughness/base_color.
    pub(super) material: DisneyMaterial,
    /// This mesh's environment reflection gain.
    pub(super) ibl: ImageBasedLighting,
    /// This mesh's output transform (exposure + tone-map curve).
    pub(super) tone_mapping: ToneMapping,
    /// Which PBR input this mesh renders diagnostically (`Shaded` = the real
    /// shading).
    pub(super) debug_view: PbrDebugView,
}

impl MeshGpu {
    pub(super) fn filled(&self) -> (&wgpu::Buffer, &IndexBuffer) {
        (&self.vertex_buffer, &self.triangles)
    }

    pub(super) fn pbr(&self) -> (&wgpu::Buffer, &IndexBuffer) {
        (&self.pbr_vertex_buffer, &self.triangles)
    }

    pub(super) fn wireframe(&self) -> (&wgpu::Buffer, &IndexBuffer) {
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
    let triangles = IndexBuffer::new(&gpu.device, "trd mesh index buffer", &mesh.indices);
    let edges = mesh.edge_indices();
    let edges = IndexBuffer::new(&gpu.device, "trd mesh edge buffer", &edges);

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
        material: DisneyMaterial::default(),
        ibl: ImageBasedLighting::default(),
        tone_mapping: ToneMapping::default(),
        debug_view: PbrDebugView::default(),
    }
}

/// The **decode-once mesh store**: the uploaded [`MeshGpu`]s a scene's mesh ids
/// index into, and nothing else (#222).
///
/// It used to also own the gizmo geometry and the per-frame instance buffer —
/// three unrelated lifetimes in one struct, of which the name described one.
/// Those are now [`GizmoGeometry`](super::gizmo::GizmoGeometry) (constant) and
/// [`InstanceBuffer`](super::buffer::InstanceBuffer) (per frame), leaving this
/// type holding exactly what it is called: the caller's meshes, fixed at
/// construction.
pub(super) struct MeshStore {
    meshes: Vec<MeshGpu>,
}

impl MeshStore {
    /// Uploads each mesh with its base (preview) model.
    pub(super) fn new(
        gpu: &GpuContext,
        meshes: &[Mesh],
        base_models: &[Matrix4],
        texture_layout: &wgpu::BindGroupLayout,
        material_maps_layout: &wgpu::BindGroupLayout,
    ) -> Self {
        let meshes = meshes
            .iter()
            .zip(base_models)
            .map(|(mesh, &base)| upload_mesh(gpu, mesh, base, texture_layout, material_maps_layout))
            .collect();
        Self { meshes }
    }

    pub(super) fn len(&self) -> usize {
        self.meshes.len()
    }

    /// Every uploaded mesh, in id order — what
    /// [`SceneUniforms::write_pbr`](super::SceneUniforms::write_pbr) walks to
    /// fill one PBR slot per mesh.
    pub(super) fn all(&self) -> &[MeshGpu] {
        &self.meshes
    }

    /// The mesh for `id`, or `None` when a draw names one that was never
    /// uploaded (out-of-range ids are skipped, not an error).
    pub(super) fn get(&self, id: usize) -> Option<&MeshGpu> {
        self.meshes.get(id)
    }

    pub(super) fn get_mut(&mut self, id: usize) -> Option<&mut MeshGpu> {
        self.meshes.get_mut(id)
    }

    /// Every mesh, mutably — for the setters that apply one value to all of them.
    pub(super) fn iter_mut(&mut self) -> impl Iterator<Item = &mut MeshGpu> {
        self.meshes.iter_mut()
    }
}

/// Indexing is for ids a batch already resolved through [`MeshStore::get`];
/// anything reading straight from a scene must use `get` instead.
impl std::ops::Index<usize> for MeshStore {
    type Output = MeshGpu;

    fn index(&self, id: usize) -> &MeshGpu {
        &self.meshes[id]
    }
}
