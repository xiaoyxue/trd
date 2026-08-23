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
use super::buffer::{IndexBuffer, VertexBuffer};
use super::*;
use crate::material::DisneyMaterial;
use crate::math::Matrix4;
use crate::texture::Texture;

/// A mesh uploaded to the GPU, in three parts by how they change: `geometry` is
/// fixed at upload, `textures` are bind groups uploaded the moment they are set,
/// and `appearance` is the PBR slot's input, written only through
/// `Renderer::edit_appearance`.
pub(super) struct MeshGpu {
    geometry: MeshGeometry,
    textures: MeshTextures,
    appearance: MeshAppearance,
}

/// The buffers and base transform, fixed when the mesh is uploaded. `vertices`
/// feeds both the filled `triangles` and the deduped wireframe `edges` (#38);
/// `aabb` is a standalone box of 12 screen-space-expanded edge quads (#42).
struct MeshGeometry {
    vertices: VertexBuffer<Vertex>,
    /// Smooth normal + tangent for `pbr.wgsl`, bound at vertex slot 2 *beside*
    /// `vertices` rather than duplicating its positions and UVs.
    shading: VertexBuffer<ShadingVertex>,
    triangles: IndexBuffer,
    edges: IndexBuffer,
    aabb: VertexBuffer<GizmoLineVertex>,
    /// The preview transform pre-multiplied beneath every per-frame instance
    /// model (`effective = model · base`).
    base_model: Matrix4,
}

/// This mesh's own bind groups (group 1 and 3), so a multi-object scene skins
/// each object separately (#141). Uploaded when set, and **not** part of a PBR
/// slot — which is why setting one must not mark the slots dirty.
struct MeshTextures {
    /// Defaults to 1×1 white — an identity albedo — until set.
    albedo: BoundTexture,
    maps: BoundMaterialMaps,
}

/// **The entire input of one mesh's PBR uniform slot** — which is what makes
/// this a type rather than four loose fields: it is exactly what
/// [`SceneUniforms::write_pbr`](super::SceneUniforms::write_pbr) reads, so
/// writing it is exactly when the slot array goes stale.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct MeshAppearance {
    pub material: DisneyMaterial,
    /// Environment reflection gain.
    pub ibl: ImageBasedLighting,
    /// Exposure + tone-map curve.
    pub tone_mapping: ToneMapping,
    /// Which PBR input to render diagnostically (`Shaded` = the real shading).
    pub debug_view: PbrDebugView,
}

impl MeshGpu {
    pub(super) fn filled(&self) -> (&VertexBuffer<Vertex>, &IndexBuffer) {
        (&self.geometry.vertices, &self.geometry.triangles)
    }

    /// What a shaded draw binds **in addition** to [`filled`](Self::filled) —
    /// the geometry is the same buffer, so there is no second copy of it.
    pub(super) fn shading(&self) -> &VertexBuffer<ShadingVertex> {
        &self.geometry.shading
    }

    pub(super) fn wireframe(&self) -> (&VertexBuffer<Vertex>, &IndexBuffer) {
        (&self.geometry.vertices, &self.geometry.edges)
    }

    pub(super) fn aabb(&self) -> &VertexBuffer<GizmoLineVertex> {
        &self.geometry.aabb
    }

    pub(super) fn base_model(&self) -> Matrix4 {
        self.geometry.base_model
    }

    pub(super) fn albedo_bind_group(&self) -> &wgpu::BindGroup {
        self.textures.albedo.bind_group()
    }

    pub(super) fn material_maps_bind_group(&self) -> &wgpu::BindGroup {
        self.textures.maps.bind_group()
    }

    pub(super) fn appearance(&self) -> &MeshAppearance {
        &self.appearance
    }

    /// The only way to mutate appearance, so `Renderer::edit_appearance` stays
    /// the single place `slots_dirty` is set.
    pub(super) fn appearance_mut(&mut self) -> &mut MeshAppearance {
        &mut self.appearance
    }

    pub(super) fn set_albedo(&mut self, gpu: &GpuContext, texture: &dyn Texture) {
        self.textures.albedo.set(gpu, texture);
    }

    pub(super) fn set_metallic_roughness(&mut self, gpu: &GpuContext, texture: &dyn Texture) {
        self.textures.maps.set_metallic_roughness(gpu, texture);
    }

    pub(super) fn set_normal_map(&mut self, gpu: &GpuContext, texture: &dyn Texture) {
        self.textures.maps.set_normal(gpu, texture);
    }
}

pub(super) fn upload_mesh(
    gpu: &GpuContext,
    mesh: &Mesh,
    base_model: Matrix4,
    texture_layout: &wgpu::BindGroupLayout,
    material_maps_layout: &wgpu::BindGroupLayout,
) -> MeshGpu {
    let vertices = VertexBuffer::new(&gpu.device, "trd mesh vertex buffer", &mesh.vertices);
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
    let shading_vertices: Vec<ShadingVertex> = normals
        .iter()
        .zip(&tangents)
        .map(|(&normal, &tangent)| ShadingVertex { normal, tangent })
        .collect();
    let shading = VertexBuffer::new(&gpu.device, "trd mesh shading buffer", &shading_vertices);

    // AABB overlay box: the mesh's own bounding box (mesh-local coords) as 8
    // colored corner vertices + a 12-edge line list. Built once per mesh; drawn
    // only when the scene contains an `AabbBox` for this mesh.
    let aabb_corners = mesh.aabb().corners().map(|corner| corner.to_array());
    let aabb_vertices = aabb_line_vertices(&aabb_corners);

    MeshGpu {
        geometry: MeshGeometry {
            vertices,
            shading,
            triangles,
            edges,
            aabb: VertexBuffer::new(&gpu.device, "trd mesh aabb line buffer", &aabb_vertices),
            base_model,
        },
        textures: MeshTextures {
            albedo: BoundTexture::with_layout(gpu, texture_layout.clone()),
            maps: BoundMaterialMaps::with_layout(gpu, material_maps_layout.clone()),
        },
        appearance: MeshAppearance::default(),
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
