//! Addressable asset residency: identity, upload, mutation and destruction.
//!
//! [`GpuResourceManager`] owns the live mesh records, not CPU assets or frame
//! scratch. Mesh-exclusive buffers/textures keep their existing thin wrappers;
//! shared texture/material registries are not introduced without real consumers.

use super::bound_material_maps::BoundMaterialMaps;
use super::bound_texture::BoundTexture;
use super::buffer::{IndexBuffer, VertexBuffer};
use super::*;
use crate::material::DisneyMaterial;
use crate::math::Matrix4;
use crate::texture::Texture;
use crate::{MeshId, MeshResourceError};
use std::collections::HashMap;

/// A mesh uploaded to the GPU, in three parts by how they change: `geometry` is
/// fixed at upload, `textures` are bind groups uploaded the moment they are set,
/// and `appearance` is the PBR slot's input.
///
/// Only `appearance` is private, and that is the point: privacy here means
/// "there is an invariant", so the one private field is the one
/// the manager's appearance-edit path must be the sole writer of (#347).
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

    /// Used only by the manager; Renderer coordinates successful edits with
    /// the uniform dirty flag.
    pub(super) fn appearance_mut(&mut self) -> &mut MeshAppearance {
        &mut self.appearance
    }

    /// Frees every GPU resource this mesh owns.
    ///
    /// Explicitly, not by dropping: `wgpu::Buffer`/`Texture` are refcounted
    /// handles, so dropping ours frees the memory only while nothing else holds
    /// one — measured, a second handle keeps a 256 MiB buffer resident through a
    /// drop **and** a flush, while `destroy()` frees it regardless.
    ///
    /// Today the bind group holds the last reference, so dropping would work and
    /// **no test fails without this** — verified by disabling it. It is here
    /// because nothing enforces that property: the first cached view or bind
    /// group added anywhere makes delete silently stop freeing, which is exactly
    /// the failure this file previously shipped (#353, #357).
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

fn upload_mesh(
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

/// A private storage/PBR address, never a scene's resource identity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(super) struct MeshSlot(usize);

impl MeshSlot {
    pub(super) fn index(self) -> usize {
        self.0
    }
}

struct ResidentMesh {
    id: MeshId,
    gpu: MeshGpu,
}

/// Caller-supplied mesh residency, with identity independent of reusable slots.
pub(super) struct GpuResourceManager {
    slots: Vec<Option<ResidentMesh>>,
    by_id: HashMap<MeshId, MeshSlot>,
    initial_ids: Vec<MeshId>,
    texture_layout: wgpu::BindGroupLayout,
    material_maps_layout: wgpu::BindGroupLayout,
}

/// A successful lookup supplies both the resource and its private PBR slot.
pub(super) struct ResolvedMesh<'a> {
    pub(super) slot: MeshSlot,
    pub(super) gpu: &'a MeshGpu,
}

impl GpuResourceManager {
    pub(super) fn new(
        gpu: &GpuContext,
        meshes: &[Mesh],
        base_models: &[Matrix4],
        texture_layout: wgpu::BindGroupLayout,
        material_maps_layout: wgpu::BindGroupLayout,
    ) -> Result<Self, MeshResourceError> {
        let mut resources = Self {
            slots: Vec::with_capacity(meshes.len()),
            by_id: HashMap::with_capacity(meshes.len()),
            initial_ids: Vec::with_capacity(meshes.len()),
            texture_layout,
            material_maps_layout,
        };
        for (mesh, &base) in meshes.iter().zip(base_models) {
            let id = resources.upload(gpu, mesh, base)?;
            resources.initial_ids.push(id);
        }
        Ok(resources)
    }

    fn upload(
        &mut self,
        gpu: &GpuContext,
        mesh: &Mesh,
        base_model: Matrix4,
    ) -> Result<MeshId, MeshResourceError> {
        let id = MeshId::fresh()?;
        let uploaded = upload_mesh(
            gpu,
            mesh,
            base_model,
            &self.texture_layout,
            &self.material_maps_layout,
        );
        self.insert(id, uploaded);
        Ok(id)
    }

    fn insert(&mut self, id: MeshId, gpu: MeshGpu) {
        let resident = ResidentMesh { id, gpu };
        let slot = match self.slots.iter().position(Option::is_none) {
            Some(slot) => {
                self.slots[slot] = Some(resident);
                slot
            }
            None => {
                self.slots.push(Some(resident));
                self.slots.len() - 1
            }
        };
        let previous = self.by_id.insert(id, MeshSlot(slot));
        assert!(previous.is_none(), "registration identities are unique");
    }

    fn remove(&mut self, gpu: &GpuContext, id: MeshId) -> Result<(), MeshResourceError> {
        let slot = self
            .by_id
            .remove(&id)
            .ok_or(MeshResourceError::NotResident { mesh: id })?;
        let resident = self.slots[slot.index()]
            .take()
            .expect("the identity lookup contains only occupied slots");
        debug_assert_eq!(resident.id, id);
        resident.gpu.destroy();
        gpu.queue.submit([]);
        Ok(())
    }

    fn set_texture(
        &mut self,
        gpu: &GpuContext,
        id: MeshId,
        texture: &dyn Texture,
    ) -> Result<(), MeshResourceError> {
        self.get_mut(id)?.textures.albedo.set(gpu, texture);
        Ok(())
    }

    fn set_metallic_roughness(
        &mut self,
        gpu: &GpuContext,
        id: MeshId,
        texture: &dyn Texture,
    ) -> Result<(), MeshResourceError> {
        self.get_mut(id)?
            .textures
            .maps
            .set_metallic_roughness(gpu, texture);
        Ok(())
    }

    fn set_normal(
        &mut self,
        gpu: &GpuContext,
        id: MeshId,
        texture: &dyn Texture,
    ) -> Result<(), MeshResourceError> {
        self.get_mut(id)?.textures.maps.set_normal(gpu, texture);
        Ok(())
    }

    fn edit_appearance(
        &mut self,
        target: MeshTarget,
        edit: impl Fn(&mut MeshAppearance),
    ) -> Result<(), MeshResourceError> {
        match target {
            MeshTarget::All => self.iter_mut().for_each(|mesh| edit(mesh.appearance_mut())),
            MeshTarget::One(id) => edit(self.get_mut(id)?.appearance_mut()),
        }
        Ok(())
    }

    /// Live resources, not the span covered by PBR allocation.
    pub(super) fn len(&self) -> usize {
        self.by_id.len()
    }

    pub(super) fn slot_count(&self) -> usize {
        self.slots.len()
    }

    pub(super) fn all(&self) -> impl Iterator<Item = (MeshSlot, &MeshGpu)> {
        self.slots
            .iter()
            .enumerate()
            .filter_map(|(slot, resident)| {
                resident
                    .as_ref()
                    .map(|resident| (MeshSlot(slot), &resident.gpu))
            })
    }

    pub(super) fn resolve(&self, id: MeshId) -> Result<ResolvedMesh<'_>, MeshResourceError> {
        let slot = *self
            .by_id
            .get(&id)
            .ok_or(MeshResourceError::NotResident { mesh: id })?;
        let resident = self.slots[slot.index()]
            .as_ref()
            .expect("the identity lookup contains only occupied slots");
        debug_assert_eq!(resident.id, id);
        Ok(ResolvedMesh {
            slot,
            gpu: &resident.gpu,
        })
    }

    pub(super) fn get(&self, id: MeshId) -> Result<&MeshGpu, MeshResourceError> {
        self.resolve(id).map(|resident| resident.gpu)
    }

    fn get_mut(&mut self, id: MeshId) -> Result<&mut MeshGpu, MeshResourceError> {
        let slot = *self
            .by_id
            .get(&id)
            .ok_or(MeshResourceError::NotResident { mesh: id })?;
        let resident = self.slots[slot.index()]
            .as_mut()
            .expect("the identity lookup contains only occupied slots");
        debug_assert_eq!(resident.id, id);
        Ok(&mut resident.gpu)
    }

    fn iter_mut(&mut self) -> impl Iterator<Item = &mut MeshGpu> {
        self.slots
            .iter_mut()
            .filter_map(|resident| resident.as_mut().map(|resident| &mut resident.gpu))
    }
}

/// Which resident meshes one appearance edit applies to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MeshTarget {
    /// Every live mesh, including the valid empty set.
    All,
    /// One registered identity; a nonresident identity is an error.
    One(MeshId),
}

impl Renderer {
    /// Immutable initial wire-row bindings; runtime additions do not renumber them.
    pub fn initial_mesh_ids(&self) -> &[MeshId] {
        &self.resources.initial_ids
    }

    /// The live mesh count, never a resource-ID validation bound.
    pub fn mesh_count(&self) -> usize {
        self.resources.len()
    }

    /// Uploads a preview-fitted mesh under a fresh identity, even on slot reuse.
    pub fn add_mesh(&mut self, mesh: &Mesh) -> Result<MeshId, MeshResourceError> {
        let mesh_id = self.resources.upload(
            &self.gpu,
            mesh,
            mesh.preview_transform(crate::DEFAULT_PREVIEW_TARGET)
                .matrix(),
        )?;
        self.uniforms
            .grow_pbr_slots(&self.gpu.device, self.resources.slot_count());
        // Reuse also invalidates the old occupant's uniform bytes.
        self.uniforms.mark_slots_dirty();
        Ok(mesh_id)
    }

    /// Invalidates `mesh_id` and destroys its GPU resources. Repeated removal
    /// fails explicitly. In-flight submissions may defer physical reclamation.
    pub fn remove_mesh(&mut self, mesh_id: MeshId) -> Result<(), MeshResourceError> {
        self.resources.remove(&self.gpu, mesh_id)?;
        self.uniforms.mark_slots_dirty();
        Ok(())
    }

    /// Sets the initial row-zero mesh's albedo, never a replacement in its slot.
    pub fn set_texture(&mut self, texture: &dyn Texture) -> Result<(), MeshResourceError> {
        let id = self
            .initial_mesh_ids()
            .first()
            .copied()
            .expect("renderer construction requires a nonempty mesh table");
        self.set_mesh_texture(id, texture)
    }

    /// Uploads the resident mesh's albedo immediately.
    pub fn set_mesh_texture(
        &mut self,
        mesh_id: MeshId,
        texture: &dyn Texture,
    ) -> Result<(), MeshResourceError> {
        self.resources.set_texture(&self.gpu, mesh_id, texture)
    }

    /// Uploads a glTF metallic-roughness map (G=roughness, B=metallic).
    pub fn set_mesh_metallic_roughness_texture(
        &mut self,
        mesh_id: MeshId,
        texture: &dyn Texture,
    ) -> Result<(), MeshResourceError> {
        self.resources
            .set_metallic_roughness(&self.gpu, mesh_id, texture)
    }

    /// Uploads the resident mesh's tangent-space glTF normal map.
    pub fn set_mesh_normal_texture(
        &mut self,
        mesh_id: MeshId,
        texture: &dyn Texture,
    ) -> Result<(), MeshResourceError> {
        self.resources.set_normal(&self.gpu, mesh_id, texture)
    }

    fn edit_appearance(
        &mut self,
        target: MeshTarget,
        edit: impl Fn(&mut MeshAppearance),
    ) -> Result<(), MeshResourceError> {
        self.resources.edit_appearance(target, edit)?;
        self.uniforms.mark_slots_dirty();
        Ok(())
    }

    /// Replaces `target`'s whole [`MeshAppearance`] — what most callers want,
    /// since material, IBL, tone map and debug view are set together.
    pub fn set_appearance(
        &mut self,
        target: MeshTarget,
        appearance: MeshAppearance,
    ) -> Result<(), MeshResourceError> {
        self.edit_appearance(target, |current| *current = appearance.clone())
    }

    /// The current appearance, or an explicit nonresident-resource error.
    pub fn mesh_appearance(&self, mesh_id: MeshId) -> Result<&MeshAppearance, MeshResourceError> {
        self.resources.get(mesh_id).map(MeshGpu::appearance)
    }

    /// Sets the [`DisneyMaterial`] of `target`. Takes effect on the next
    /// [`render`](Self::render).
    pub fn set_disney_material(
        &mut self,
        target: MeshTarget,
        material: DisneyMaterial,
    ) -> Result<(), MeshResourceError> {
        self.edit_appearance(target, |appearance| {
            appearance.material = material.clone();
        })
    }

    /// Sets the environment-reflection gain of `target`.
    pub fn set_image_based_lighting(
        &mut self,
        target: MeshTarget,
        ibl: ImageBasedLighting,
    ) -> Result<(), MeshResourceError> {
        self.edit_appearance(target, |appearance| appearance.ibl = ibl)
    }

    /// Sets the output transform (exposure + curve) of `target`.
    pub fn set_tone_mapping(
        &mut self,
        target: MeshTarget,
        tone_mapping: ToneMapping,
    ) -> Result<(), MeshResourceError> {
        self.edit_appearance(target, |appearance| appearance.tone_mapping = tone_mapping)
    }

    /// Selects a diagnostic PBR output for `target`.
    pub fn set_pbr_debug_view(
        &mut self,
        target: MeshTarget,
        debug_view: PbrDebugView,
    ) -> Result<(), MeshResourceError> {
        self.edit_appearance(target, |appearance| appearance.debug_view = debug_view)
    }
}
