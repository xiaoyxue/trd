//! Per-frame batching: walking a [`Scene`](crate::Scene)'s objects into
//! instanced [`DrawCommand`]s.
//!
//! This is what "batch" means in trd: grouping draws that share GPU state into
//! one instanced command. (The former `BatchRenderer` used the word for
//! *batch-mode headless output* and owned none of this; it is now `Renderer`,
//! so the term is unambiguous — #180. The module is named for what it produces
//! rather than for the overloaded verb.)
//!
//! Pure data and a pure function — no `wgpu::Device`, no GPU state — so this is
//! unit-testable on its own. [`build_batches`] splits every [`DrawableObject`]
//! into its `(primitive, model)` halves, stable-sorts by
//! [`Primitive::sort_key`], and groups equal runs into instanced
//! [`DrawCommand`]s. It sees a scene's *objects* only: the frame's backgrounds
//! are settings on [`Scene::background`](crate::Scene::background), not
//! primitives to batch (#204).
//!
//! **A batch key is a drawable minus its model** (#204). There is no separate
//! batch-key taxonomy to translate into: sorting and grouping run on the very
//! [`Primitive`] the front-end named, so a new primitive is one variant in one
//! place. **Submission order is the frame's z-order** — every overlay pipeline
//! disables depth — and it is [`Primitive::sort_key`] that spells it out, not
//! the declaration order of an enum.

use super::InstanceRaw;
use super::{DrawableObject, Primitive};
use crate::math::Matrix4;

/// One instanced draw recorded while walking a scene: the primitive whose
/// geometry and pipeline to bind, and the contiguous instance-buffer range to
/// draw it over.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct DrawCommand {
    pub(super) primitive: Primitive,
    pub(super) start: u32,
    pub(super) count: u32,
}

/// The result of walking a scene's objects once: the flattened per-instance
/// models and the [`DrawCommand`]s over them (already in draw order).
///
/// Instancing only — the frame's backgrounds are *settings* on
/// [`Scene::background`](crate::Scene::background), read straight by the encoder
/// (#204). They used to be drawables the batcher had to recognise and `continue`
/// past, and were carried out again here as two singleton fields, which is
/// exactly the round-trip that moving them onto the scene removes.
pub(super) struct Batches {
    pub(super) instances: Vec<InstanceRaw>,
    pub(super) commands: Vec<DrawCommand>,
}

/// Walks `objects` once into a flat draw list, stable-sorts by
/// [`Primitive::sort_key`], then groups equal primitives into instanced
/// commands. Out-of-range mesh ids are skipped.
///
/// The only thing this has to *decide* is the model: mesh-backed primitives
/// compose the mesh's base (preview) model beneath the drawable's, everything
/// else is placed by its own model alone. Choosing the geometry is no longer
/// part of batching — the primitive already is the batch key (#204).
///
/// Takes the objects alone: every one of them is a placed primitive that becomes
/// an instance, so there is no longer a non-instanced member to filter out
/// (#204).
pub(super) fn build_batches(
    objects: &[DrawableObject],
    mut mesh_base_model: impl FnMut(usize) -> Option<Matrix4>,
) -> Batches {
    let mut draws: Vec<(Primitive, InstanceRaw)> = Vec::with_capacity(objects.len());

    for object in objects {
        let primitive = object.primitive();
        let model = match primitive {
            // Mesh-backed primitives ride on the mesh's base (preview) model;
            // gizmos are tied to no mesh and are placed by their own model alone.
            // Listed exhaustively so a new primitive has to answer the question.
            Primitive::Mesh { mesh_id, .. } | Primitive::AabbBox { mesh_id } => {
                let Some(base_model) = mesh_base_model(mesh_id as usize) else {
                    continue;
                };
                let effective = Matrix4::from_cols_array(&object.model()) * base_model;
                effective.to_cols_array()
            }
            Primitive::PlaneGrid { .. }
            | Primitive::QuadOutline { .. }
            | Primitive::CoordinateAxes
            | Primitive::BlobShadow => object.model(),
        };
        draws.push((primitive, InstanceRaw { model }));
    }

    draws.sort_by_key(|(primitive, _)| primitive.sort_key());

    let mut instances = Vec::with_capacity(draws.len());
    let mut commands = Vec::new();
    for run in draws.chunk_by(|a, b| a.0 == b.0) {
        let start = instances.len() as u32;
        instances.extend(run.iter().map(|(_, instance)| *instance));
        commands.push(DrawCommand {
            primitive: run[0].0,
            start,
            count: run.len() as u32,
        });
    }

    Batches {
        instances,
        commands,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::render::{GridPlane, RenderMode};

    fn model(tag: f32) -> [f32; 16] {
        let mut model = Matrix4::IDENTITY.to_cols_array();
        model[12] = tag;
        model
    }

    fn mesh(mesh_id: u32, tag: f32, mode: RenderMode) -> DrawableObject {
        DrawableObject::mesh(mesh_id, model(tag), mode)
    }

    /// Pins the exact submitted sequence for a scene holding **every** primitive
    /// kind, with mesh modes mixed across several mesh ids.
    ///
    /// Both halves matter, because every overlay pipeline is depth-disabled and
    /// alpha-blended, so submission order is the frame's z-order (#204):
    ///
    /// 1. the **layer** order — shadow, solid meshes, grid, quad outline,
    ///    wireframe meshes, AABB boxes, axes — with the wireframe meshes *after*
    ///    the grid and quad outline they overlay, not beside their solid
    ///    siblings;
    /// 2. within the solid layer, the **mode before the mesh id** — all filled
    ///    draws, then all textured, then all shaded — which keeps one pipeline
    ///    switch per mode instead of one per mesh.
    ///
    /// The instance tags then show the sort is stable (equal primitives keep
    /// their scene order) and that skipped, out-of-range mesh ids leave no trace.
    #[test]
    fn batches_in_layer_order_and_preserves_equal_primitive_order() {
        let scene = [
            DrawableObject::coordinate_axes(model(80.0)),
            mesh(1, 61.0, RenderMode::Wireframe),
            mesh(1, 12.0, RenderMode::Filled),
            mesh(0, 30.0, RenderMode::Shaded),
            DrawableObject::blob_shadow(model(1.0)),
            DrawableObject::aabb_box(1, model(71.0)),
            DrawableObject::plane_grid(GridPlane::Yz, model(52.0)),
            mesh(0, 10.0, RenderMode::Filled),
            mesh(0, 20.0, RenderMode::Textured),
            DrawableObject::quad_outline(model(40.0), false),
            DrawableObject::plane_grid(GridPlane::Xy, model(50.0)),
            mesh(0, 11.0, RenderMode::Filled),
            DrawableObject::quad_outline(model(41.0), true),
            DrawableObject::aabb_box(0, model(70.0)),
            mesh(1, 21.0, RenderMode::Textured),
            mesh(1, 31.0, RenderMode::Shaded),
            mesh(0, 60.0, RenderMode::Wireframe),
            mesh(99, 99.0, RenderMode::Filled),
        ];
        let base_models = [Matrix4::IDENTITY, Matrix4::IDENTITY];

        let batches = build_batches(&scene, |mesh_id| base_models.get(mesh_id).copied());
        let commands = batches
            .commands
            .iter()
            .map(|command| (command.primitive, command.start, command.count))
            .collect::<Vec<_>>();

        let solid = |mesh_id, mode| Primitive::Mesh { mesh_id, mode };
        assert_eq!(
            commands,
            [
                (Primitive::BlobShadow, 0, 1),
                (solid(0, RenderMode::Filled), 1, 2),
                (solid(1, RenderMode::Filled), 3, 1),
                (solid(0, RenderMode::Textured), 4, 1),
                (solid(1, RenderMode::Textured), 5, 1),
                (solid(0, RenderMode::Shaded), 6, 1),
                (solid(1, RenderMode::Shaded), 7, 1),
                (
                    Primitive::PlaneGrid {
                        plane: GridPlane::Xy
                    },
                    8,
                    1
                ),
                (
                    Primitive::PlaneGrid {
                        plane: GridPlane::Yz
                    },
                    9,
                    1
                ),
                (Primitive::QuadOutline { selected: false }, 10, 1),
                (Primitive::QuadOutline { selected: true }, 11, 1),
                (solid(0, RenderMode::Wireframe), 12, 1),
                (solid(1, RenderMode::Wireframe), 13, 1),
                (Primitive::AabbBox { mesh_id: 0 }, 14, 1),
                (Primitive::AabbBox { mesh_id: 1 }, 15, 1),
                (Primitive::CoordinateAxes, 16, 1),
            ]
        );
        assert_eq!(
            batches
                .instances
                .iter()
                .map(|instance| instance.model[12])
                .collect::<Vec<_>>(),
            [
                1.0, 10.0, 11.0, 12.0, 20.0, 21.0, 30.0, 31.0, 50.0, 52.0, 40.0, 41.0, 60.0, 61.0,
                70.0, 71.0, 80.0,
            ]
        );
    }
}
