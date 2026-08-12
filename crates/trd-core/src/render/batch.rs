//! Per-frame batching: walking a [`Scene`] into instanced draw commands.
//!
//! This is what "batch" means in trd: grouping draws that share GPU state into
//! one instanced command. (The former `BatchRenderer` used the word for
//! *batch-mode headless output* and owned none of this; it is now `Renderer`,
//! so the term is unambiguous — #180.)
//!
//! Pure data and a pure function — no `wgpu::Device`, no GPU state — so this is
//! unit-testable on its own. [`build_batches`] flattens every
//! [`DrawableObject`] into a `(DrawKind, model)` pair, stable-sorts by kind, and
//! groups equal runs into instanced [`DrawCommand`]s.
//!
//! **[`DrawKind`]'s variant order is the draw order**: its derived [`Ord`] is
//! what sorts the frame, so declaring `Shadow` first and `Axes` last is what
//! layers the frame correctly. Sorting therefore does state batching and layer
//! ordering in one pass.

use super::*;
use crate::math::Matrix4;
use crate::scene::{DrawableObject, FrameFit, RenderMode};

/// Which geometry a [`DrawCommand`] binds. The `usize` is a mesh id (index into
/// [`MeshStore::meshes`](super::mesh_store::MeshStore) for the mesh kinds, or a
/// [`GridPlane::index`] for `Grid`; `Axes` uses the shared gizmo geometry.
/// Variants are declared in their **layered draw order**; the derived ordering
/// is the batching order.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(super) enum DrawKind {
    /// A contact / blob **grounding shadow** (the shared shadow quad geometry,
    /// non-indexed triangle draw, alpha-blended over the frame plane).
    Shadow,
    /// Filled triangles of a mesh (its triangle index buffer + filled pipeline).
    Filled(usize),
    /// Textured triangles of a mesh (triangle index buffer + textured pipeline,
    /// sampling the bound texture at each vertex UV) (#20).
    Textured(usize),
    /// **Shaded** triangles of a mesh: the Disney PBR path (its dedicated
    /// position+normal+UV vertex buffer + `pbr.wgsl` pipeline, lit by the virtual
    /// light rig and the bound HDR environment map). Reuses the triangle index
    /// buffer.
    Shaded(usize),
    /// A coordinate-plane grid (the shared per-plane grid vertex buffer indexed
    /// by [`GridPlane::index`], non-indexed line draw).
    Grid(usize),
    /// Green/yellow placement-quad outline.
    QuadOutline(usize),
    /// Edge lines of a mesh (its deduped edge index buffer + line pipeline).
    Wireframe(usize),
    /// A mesh's AABB box (its precomputed corner geometry + line pipeline).
    Aabb(usize),
    /// The coordinate-axes gizmo (shared vertex buffer, non-indexed line draw).
    Axes,
}

/// One instanced draw recorded while walking a [`Scene`]: the geometry to bind
/// ([`DrawKind`]) and the contiguous instance-buffer range to draw it over.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct DrawCommand {
    pub(super) kind: DrawKind,
    pub(super) start: u32,
    pub(super) count: u32,
}

/// The result of walking a [`Scene`] once: the flattened per-instance models,
/// the [`DrawCommand`]s over them (already in draw order), and the singleton
/// background frame-plane fit (if any).
pub(super) struct Batches {
    pub(super) instances: Vec<InstanceRaw>,
    pub(super) commands: Vec<DrawCommand>,
    pub(super) frame_fit: Option<FrameFit>,
    pub(super) environment_background: Option<([f32; 3], Tonemap)>,
}

/// Walks `scene` once into a flat draw list, stable-sorts by [`DrawKind`], then
/// groups equal runs into instanced commands. Out-of-range mesh ids are skipped;
/// the last background frame plane wins.
pub(super) fn build_batches(
    scene: &[DrawableObject],
    mut mesh_base_model: impl FnMut(usize) -> Option<Matrix4>,
) -> Batches {
    let mut draws: Vec<(DrawKind, InstanceRaw)> = Vec::with_capacity(scene.len());
    let mut frame_fit = None;
    let mut environment_background = None;

    for object in scene {
        let (kind, model) = match *object {
            DrawableObject::Mesh {
                mesh_id,
                model,
                mode,
            } => {
                let mesh_id = mesh_id as usize;
                let Some(base_model) = mesh_base_model(mesh_id) else {
                    continue;
                };
                let kind = match mode {
                    RenderMode::Filled => DrawKind::Filled(mesh_id),
                    RenderMode::Textured => DrawKind::Textured(mesh_id),
                    RenderMode::Shaded => DrawKind::Shaded(mesh_id),
                    RenderMode::Wireframe => DrawKind::Wireframe(mesh_id),
                    RenderMode::Shadow => continue,
                };
                let effective = Matrix4::from_cols_array(&model) * base_model;
                (kind, effective.to_cols_array())
            }
            DrawableObject::AabbBox { mesh_id, model } => {
                let mesh_id = mesh_id as usize;
                let Some(base_model) = mesh_base_model(mesh_id) else {
                    continue;
                };
                let effective = Matrix4::from_cols_array(&model) * base_model;
                (DrawKind::Aabb(mesh_id), effective.to_cols_array())
            }
            DrawableObject::CoordinateAxes { model } => (DrawKind::Axes, model),
            DrawableObject::PlaneGrid { plane, model } => (DrawKind::Grid(plane.index()), model),
            DrawableObject::QuadOutline { model, selected } => {
                (DrawKind::QuadOutline(usize::from(selected)), model)
            }
            DrawableObject::BlobShadow { model } => (DrawKind::Shadow, model),
            DrawableObject::EnvironmentBackground {
                rotation,
                exposure,
                blur,
                tonemap,
            } => {
                environment_background = Some(([rotation, exposure, blur], tonemap));
                continue;
            }
            DrawableObject::FramePlane { fit } => {
                frame_fit = Some(fit);
                continue;
            }
        };
        draws.push((kind, InstanceRaw { model }));
    }

    draws.sort_by_key(|(kind, _)| *kind);

    let mut instances = Vec::with_capacity(draws.len());
    let mut commands = Vec::new();
    for run in draws.chunk_by(|a, b| a.0 == b.0) {
        let start = instances.len() as u32;
        instances.extend(run.iter().map(|(_, instance)| *instance));
        commands.push(DrawCommand {
            kind: run[0].0,
            start,
            count: run.len() as u32,
        });
    }

    Batches {
        instances,
        commands,
        frame_fit,
        environment_background,
    }
}
