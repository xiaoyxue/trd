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
//!
//! [`MeshTarget`] and the renderer-level surface over the store — uploads,
//! removals, texture binding, appearance edits — are at the bottom of this file
//! (#363). They stay [`Renderer`] methods because each needs the GPU context,
//! but they read beside the state they mutate.

use super::bound_material_maps::BoundMaterialMaps;
use super::bound_texture::BoundTexture;
use super::buffer::{IndexBuffer, VertexBuffer};
use super::*;
use crate::material::DisneyMaterial;
use crate::math::Matrix4;
use crate::texture::Texture;

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
    /// the one place an appearance edit marks the PBR slots stale.
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

/// Which meshes one appearance edit applies to.
///
/// A value rather than a pair of setters per field: "which meshes" and "what to
/// change" are independent questions, and keeping them apart is what lets
/// [`Renderer::edit_appearance`] be the one place an *appearance* edit marks the
/// PBR slots stale.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MeshTarget {
    /// Every uploaded mesh — the single-mesh and wire-protocol default.
    All,
    /// One mesh by id. Out-of-range ids change nothing.
    One(usize),
}

/// The mesh store's renderer-level surface (#363): uploads, removals, texture
/// binding and appearance edits.
///
/// Every one of these forwards into the [`MeshStore`] this module owns, so they
/// read beside the state they mutate; they stay [`Renderer`] methods because
/// each needs the GPU context, and the appearance edits additionally mark the
/// PBR slots stale.
impl Renderer {
    /// The number of meshes this renderer can draw; valid mesh ids in a
    /// [`Primitive::Mesh`]/[`Primitive::AabbBox`] are in
    /// `0..mesh_count()`.
    pub fn mesh_count(&self) -> usize {
        self.meshes.len()
    }

    /// Uploads `mesh` as a new drawable and returns its mesh id — the runtime
    /// twin of the constructor's mesh set (#353).
    ///
    /// The renderer's mesh set used to be fixed at construction, so loading a
    /// model meant rebuilding the whole renderer and losing the scene. This
    /// grows the store **and** the PBR slot array together, which is the part
    /// that cannot be a plain `Vec` push: a slot is chosen by a dynamic offset
    /// validated against the slot buffer, so the buffer and its bind group are
    /// reallocated. Every existing mesh keeps its appearance — the slots are
    /// re-uploaded from the meshes that own them on the next frame.
    ///
    /// The new mesh gets [`Mesh::preview_transform`]
    /// ([`crate::DEFAULT_PREVIEW_TARGET`]) as its base model, exactly like
    /// [`auto_fit`](Self::auto_fit), and starts with the 1×1 white albedo and a
    /// default appearance; bind its textures and material through the setters
    /// above.
    pub fn add_mesh(&mut self, mesh: &Mesh) -> usize {
        let texture_layout = create_texture_bind_group_layout(&self.gpu.device);
        let material_maps_layout = BoundMaterialMaps::create_layout(&self.gpu.device);
        let uploaded = upload_mesh(
            &self.gpu,
            mesh,
            mesh.preview_transform(crate::DEFAULT_PREVIEW_TARGET)
                .matrix(),
            &texture_layout,
            &material_maps_layout,
        );
        let mesh_id = self.meshes.push(uploaded);
        self.uniforms
            .grow_pbr_slots(&self.gpu.device, self.meshes.len());
        // Unconditional: growing reallocates and discards every slot, and when
        // it does *not* grow the id is a reused hole still holding the previous
        // occupant's material.
        self.uniforms.mark_slots_dirty();
        mesh_id
    }

    /// Removes mesh `mesh_id`, freeing its GPU memory, and reports whether one
    /// was there (#353).
    ///
    /// Surviving meshes **keep their ids**: the slot becomes a hole rather than
    /// the `Vec` compacting, because compacting would renumber every mesh after
    /// it and silently repoint any scene holding an id. A later
    /// [`add_mesh`](Self::add_mesh) reuses the hole. Drawing a removed id is
    /// skipped like any other unknown id, not an error.
    ///
    /// **What the release does and does not guarantee.** The mesh's buffers and
    /// textures are destroyed explicitly and the queue is flushed, so no later
    /// render is needed to get the memory back — which is the bug this replaced,
    /// where a delete freed nothing until something else happened to draw. It is
    /// not a synchronous free: wgpu defers the physical deallocation of anything
    /// still referenced by an in-flight submission until that submission
    /// completes, so loading another large model immediately afterwards can
    /// briefly hold both.
    pub fn remove_mesh(&mut self, mesh_id: usize) -> bool {
        let removed = self.meshes.remove(mesh_id);
        if removed {
            // `MeshStore::remove` destroyed the resources; this hands wgpu a
            // submission to service that destruction on. Dropping alone frees
            // nothing here on two counts: the handles are refcounted, and
            // reclamation happens while servicing a submission rather than on
            // drop or on poll.
            self.gpu.queue.submit([]);
            // The freed slot keeps its stale contents; the next frame rewrites
            // every live one.
            self.uniforms.mark_slots_dirty();
        }
        removed
    }

    /// Binds `texture` as the albedo of **mesh 0** — the single-mesh /
    /// wire-protocol default sampled by [`RenderMode::Textured`]/[`RenderMode::Shaded`]
    /// draws (#20). For a multi-object scene, skin each object with
    /// [`set_mesh_texture`](Self::set_mesh_texture). The image is (re)uploaded
    /// lazily on the next [`render`](Self::render); until set it is
    /// 1×1 white.
    pub fn set_texture(&mut self, texture: &dyn Texture) {
        self.set_mesh_texture(0, texture);
    }

    /// Binds `texture` as the albedo of mesh `mesh_id` — so a multi-object scene
    /// skins each object with its **own** diffuse (#141). Out-of-range ids are
    /// ignored. The image uploads lazily on the next
    /// [`render`](Self::render).
    pub fn set_mesh_texture(&mut self, mesh_id: usize, texture: &dyn Texture) {
        if let Some(mesh) = self.meshes.get_mut(mesh_id) {
            mesh.textures.albedo.set(&self.gpu, texture);
        }
    }

    /// Binds a glTF metallic-roughness map (G=roughness, B=metallic) for mesh
    /// `mesh_id`, sampled by [`RenderMode::Shaded`] in place of the scalar
    /// material values. Out-of-range ids are ignored.
    pub fn set_mesh_metallic_roughness_texture(&mut self, mesh_id: usize, texture: &dyn Texture) {
        if let Some(mesh) = self.meshes.get_mut(mesh_id) {
            mesh.textures
                .maps
                .set_metallic_roughness(&self.gpu, texture);
        }
    }

    /// Binds mesh `mesh_id`'s tangent-space glTF normal map, perturbing the
    /// shading normal in [`RenderMode::Shaded`]. Out-of-range ids are ignored.
    pub fn set_mesh_normal_texture(&mut self, mesh_id: usize, texture: &dyn Texture) {
        if let Some(mesh) = self.meshes.get_mut(mesh_id) {
            mesh.textures.maps.set_normal(&self.gpu, texture);
        }
    }

    /// **The one path that mutates per-mesh appearance, and therefore the one
    /// place the PBR slots are marked stale.** An out-of-range
    /// [`MeshTarget::One`] edits nothing and leaves the slots clean.
    ///
    /// The texture setters above deliberately do *not* come through here: they
    /// upload a bind group immediately and feed no PBR slot.
    fn edit_appearance(&mut self, target: MeshTarget, edit: impl Fn(&mut MeshAppearance)) {
        match target {
            MeshTarget::All => self
                .meshes
                .iter_mut()
                .for_each(|mesh| edit(mesh.appearance_mut())),
            MeshTarget::One(mesh_id) => {
                let Some(mesh) = self.meshes.get_mut(mesh_id) else {
                    return;
                };
                edit(mesh.appearance_mut());
            }
        }
        self.uniforms.mark_slots_dirty();
    }

    /// Replaces `target`'s whole [`MeshAppearance`] — what most callers want,
    /// since material, IBL, tone map and debug view are set together.
    pub fn set_appearance(&mut self, target: MeshTarget, appearance: MeshAppearance) {
        self.edit_appearance(target, |current| *current = appearance.clone());
    }

    /// The current appearance of mesh `mesh_id`, or `None` if out of range.
    pub fn mesh_appearance(&self, mesh_id: usize) -> Option<&MeshAppearance> {
        self.meshes.get(mesh_id).map(MeshGpu::appearance)
    }

    /// Sets the [`DisneyMaterial`] of `target`. Takes effect on the next
    /// [`render`](Self::render).
    pub fn set_disney_material(&mut self, target: MeshTarget, material: DisneyMaterial) {
        self.edit_appearance(target, |appearance| {
            appearance.material = material.clone();
        });
    }

    /// Sets the environment-reflection gain of `target`.
    pub fn set_image_based_lighting(&mut self, target: MeshTarget, ibl: ImageBasedLighting) {
        self.edit_appearance(target, |appearance| appearance.ibl = ibl);
    }

    /// Sets the output transform (exposure + curve) of `target`.
    pub fn set_tone_mapping(&mut self, target: MeshTarget, tone_mapping: ToneMapping) {
        self.edit_appearance(target, |appearance| appearance.tone_mapping = tone_mapping);
    }

    /// Selects a diagnostic PBR output for `target`.
    pub fn set_pbr_debug_view(&mut self, target: MeshTarget, debug_view: PbrDebugView) {
        self.edit_appearance(target, |appearance| appearance.debug_view = debug_view);
    }
}
