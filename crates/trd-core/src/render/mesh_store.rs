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

/// A mesh uploaded to the GPU, in three parts by how they change: `geometry` is
/// fixed at upload, `textures` are bind groups uploaded the moment they are set,
/// and `appearance` is the PBR slot's input.
///
/// Only `appearance` is private, and that is the point: privacy here means
/// "there is an invariant", so the one private field is the one
/// `Renderer::edit_appearance` must be the sole writer of (#347).
pub(super) struct MeshGpu {
    pub(super) geometry: MeshGeometry,
    pub(super) textures: MeshTextures,
    appearance: MeshAppearance,
}

/// The buffers and base transform, fixed when the mesh is uploaded. `vertices`
/// feeds both the filled `triangles` and the deduped wireframe `edges` (#38);
/// `aabb` is a standalone box of 12 screen-space-expanded edge quads (#42).
pub(super) struct MeshGeometry {
    pub(super) vertices: VertexBuffer<Vertex>,
    /// Smooth normal + tangent for `pbr.wgsl`, bound at vertex slot 2 *beside*
    /// `vertices` rather than duplicating its positions and UVs.
    pub(super) shading: VertexBuffer<ShadingVertex>,
    pub(super) triangles: IndexBuffer,
    pub(super) edges: IndexBuffer,
    pub(super) aabb: VertexBuffer<GizmoLineVertex>,
    /// The preview transform pre-multiplied beneath every per-frame instance
    /// model (`effective = model · base`).
    pub(super) base_model: Matrix4,
}

/// This mesh's own bind groups (group 1 and 3), so a multi-object scene skins
/// each object separately (#141). Uploaded when set, and **not** part of a PBR
/// slot — which is why setting one must not mark the slots dirty.
pub(super) struct MeshTextures {
    /// Defaults to 1×1 white — an identity albedo — until set.
    pub(super) albedo: BoundTexture,
    pub(super) maps: BoundMaterialMaps,
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
    /// Composes the pair a draw binds — `vertices` is shared with
    /// [`wireframe`](Self::wireframe), so there is no second copy of it.
    pub(super) fn filled(&self) -> (&VertexBuffer<Vertex>, &IndexBuffer) {
        (&self.geometry.vertices, &self.geometry.triangles)
    }

    pub(super) fn wireframe(&self) -> (&VertexBuffer<Vertex>, &IndexBuffer) {
        (&self.geometry.vertices, &self.geometry.edges)
    }

    pub(super) fn appearance(&self) -> &MeshAppearance {
        &self.appearance
    }

    /// The only way to mutate appearance, so `Renderer::edit_appearance` stays
    /// the single place `slots_dirty` is set.
    pub(super) fn appearance_mut(&mut self) -> &mut MeshAppearance {
        &mut self.appearance
    }

    /// Frees every GPU resource this mesh owns.
    ///
    /// Explicitly, not by dropping: `wgpu::Buffer`/`Texture` are refcounted
    /// handles, so dropping ours frees the memory only while nothing else holds
    /// one — measured, a second handle keeps a 256 MiB buffer resident through a
    /// drop **and** a flush. Nothing enforces that no one else holds one, and
    /// the failure is silent, which is exactly how "delete freed nothing" looks
    /// from the outside (#353).
    pub(super) fn destroy(&self) {
        self.geometry.vertices.destroy();
        self.geometry.shading.destroy();
        self.geometry.triangles.destroy();
        self.geometry.edges.destroy();
        self.geometry.aabb.destroy();
        self.textures.albedo.destroy();
        self.textures.maps.destroy();
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
/// type holding exactly what it is called: the caller's meshes.
///
/// Slots are `Option` because a mesh can be **removed** at runtime (#353) and
/// its GPU memory has to go back — a 138 MiB GLB holds vertex/index buffers and
/// three 2048² textures. Compacting the `Vec` instead would renumber every mesh
/// after the hole, silently repointing scenes that hold ids; a tombstone keeps
/// every surviving id valid and lets the next upload reuse the slot.
pub(super) struct MeshStore {
    meshes: Vec<Option<MeshGpu>>,
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
            .map(|(mesh, &base)| {
                Some(upload_mesh(
                    gpu,
                    mesh,
                    base,
                    texture_layout,
                    material_maps_layout,
                ))
            })
            .collect();
        Self { meshes }
    }

    /// Adds an uploaded mesh, reusing a removed mesh's slot when there is one,
    /// and returns its id.
    ///
    /// The store is still decode-once *per mesh* — this adds one the caller did
    /// not have at construction (a runtime model load, #353), it does not
    /// re-upload an existing one.
    pub(super) fn push(&mut self, mesh: MeshGpu) -> usize {
        match self.meshes.iter().position(Option::is_none) {
            Some(slot) => {
                self.meshes[slot] = Some(mesh);
                slot
            }
            None => {
                self.meshes.push(Some(mesh));
                self.meshes.len() - 1
            }
        }
    }

    /// Drops mesh `id`, freeing its GPU memory, and reports whether one was
    /// there. The slot is left as a hole so no other mesh is renumbered.
    ///
    /// The release is explicit — see [`MeshGpu::destroy`].
    pub(super) fn remove(&mut self, id: usize) -> bool {
        let Some(slot) = self.meshes.get_mut(id) else {
            return false;
        };
        let Some(mesh) = slot.take() else {
            return false;
        };
        mesh.destroy();
        true
    }

    /// The number of **slots**, live or removed — the span valid ids come from,
    /// and the size the PBR slot array must cover.
    pub(super) fn len(&self) -> usize {
        self.meshes.len()
    }

    /// Every slot, in id order — what
    /// [`SceneUniforms::write_pbr`](super::SceneUniforms::write_pbr) walks to
    /// fill one PBR slot per mesh. A `None` is a removed mesh, whose slot keeps
    /// its index so the surviving ids still address their own.
    pub(super) fn all(&self) -> &[Option<MeshGpu>] {
        &self.meshes
    }

    /// The mesh for `id`, or `None` when a draw names one that was never
    /// uploaded or has been removed (such ids are skipped, not an error).
    pub(super) fn get(&self, id: usize) -> Option<&MeshGpu> {
        self.meshes.get(id)?.as_ref()
    }

    pub(super) fn get_mut(&mut self, id: usize) -> Option<&mut MeshGpu> {
        self.meshes.get_mut(id)?.as_mut()
    }

    /// Every live mesh, mutably — for the setters that apply one value to all.
    pub(super) fn iter_mut(&mut self) -> impl Iterator<Item = &mut MeshGpu> {
        self.meshes.iter_mut().filter_map(Option::as_mut)
    }
}
