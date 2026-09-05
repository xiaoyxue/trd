//! [`Primitive`] — *what* the renderer can draw — and [`DrawableObject`], one
//! primitive *placed* by a model: the single base interface for everything a
//! frame draws (#41).
//!
//! Adding a primitive means adding a [`Primitive`] variant; the renderer batches
//! a scene by primitive without special-casing any concrete type. Geometry is
//! owned once by the renderer's decode-once store, so a drawable is a light
//! `Copy` handle naming *which* primitive plus its per-frame model.
//!
//! **One taxonomy, not two** (#204). A [`DrawableObject`] is exactly
//! `primitive + model`, and a batch is exactly "the drawables that share a
//! primitive". Strip the model and what is left *is* the batch key, so the
//! render backend keys its instanced draw commands on this very enum instead of
//! restating the same list as a parallel batch-key enum — a second taxonomy that
//! had to be kept in step by hand and that flattened `GridPlane`/`bool` into an
//! opaque `usize` payload purely to give its variants a uniform shape.

use super::{GridPlane, RenderMode};
use crate::math::Matrix4;
use crate::MeshId;

/// **What** the renderer draws: the closed list of primitives it knows, with the
/// per-primitive configuration that selects the geometry and pipeline — but
/// *not* the model that places it, which is [`DrawableObject`]'s half.
///
/// This is also the renderer's **batch key**: two [`DrawableObject`]s with equal
/// primitives bind identical GPU state and are drawn as one instanced command,
/// so equality here is what defines the batches (#204).
///
/// Wireframe is a render *mode* of [`Primitive::Mesh`] (not a separate variant);
/// the coordinate axes and the AABB box are genuinely distinct gizmo primitives
/// rendered with screen-space-expanded line geometry.
///
/// **Deliberately not `Ord`.** Submission order is the frame's layer order (every
/// overlay pipeline disables depth — see [`Primitive::layer`]), and a derived
/// `Ord` would tie that to the order the variants happen to be declared in *and*
/// would compare `Mesh`'s `mesh_id` before its `mode`. The order is spelled out
/// in [`sort_key`](Self::sort_key) instead.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Primitive {
    /// A registered mesh identity drawn in `mode`
    /// (filled, textured, shaded, or wireframe). The renderer pre-multiplies the
    /// mesh's base (preview) model beneath the drawable's model
    /// (`effective = model · base`).
    ///
    /// The **only** variant carrying a mode: every other primitive has exactly
    /// one way of being drawn.
    Mesh { mesh_id: MeshId, mode: RenderMode },
    /// The axis-aligned bounding-box outline of mesh `mesh_id` (#42), placed by
    /// the same model as the mesh instance it boxes (the renderer applies that
    /// mesh's base model beneath it too), so the box tracks the mesh exactly.
    /// Reuses the mesh's precomputed corner geometry.
    AabbBox { mesh_id: MeshId },
    /// A **coordinate-plane grid** lattice on `plane` (X/Y, X/Z, or Y/Z),
    /// spanning the model-space square `[-1, 1]²`. Like
    /// [`CoordinateAxes`](Self::CoordinateAxes) it is a screen-space-expanded
    /// line gizmo tied to no mesh (no base model); with a #77 placement-quad
    /// model the `Xy` grid lays exactly over the reconstructed quad in its local
    /// frame.
    PlaneGrid { plane: GridPlane },
    /// The tracked placement-quad outline, rendered by the shared analytic-AA
    /// gizmo line pipeline at 1.5 px. `selected` picks the highlight color (the
    /// renderer keeps one line buffer per state).
    QuadOutline { selected: bool },
    /// A translucent wash over the placement quad's whole face — the hover /
    /// selection highlight that the [outline](Self::QuadOutline) alone cannot
    /// give, since pointing at a quad has to tint the *area* an object would be
    /// placed on. Shares the blob shadow's unit-quad geometry and differs only in
    /// its fragment shader; alpha-blended under the outline so the coloured edge
    /// still reads on top.
    QuadFill,
    /// The world-orientation coordinate gizmo (#42): three anti-aliased shafts
    /// with cone arrowheads from the origin along +X/+Y/+Z, colored
    /// red/green/blue. Placed by the drawable's model (identity marks the world
    /// origin); not tied to any mesh, so no base model is applied.
    CoordinateAxes,
    /// A **contact / blob grounding shadow** (#110 follow-up): a soft dark radial
    /// blob laid on a placed mesh's ground plane (a flat quad on the plane, sized
    /// to the mesh footprint), so the mesh reads as *sitting on* the
    /// reconstructed surface rather than floating over the composited video
    /// plate. A [`DrawSelection::Shadow`](super::DrawSelection) draw becomes this
    /// primitive. Tied to no mesh (no base model); alpha-blended over the
    /// background frame plane ([`Background::frame`](super::Background::frame))
    /// and drawn *before* the opaque content meshes (depth-write off) so they
    /// composite on top while the surrounding rim darkens the floor.
    BlobShadow,
}

impl Primitive {
    pub(crate) fn mesh_id(self) -> Option<MeshId> {
        match self {
            Self::Mesh { mesh_id, .. } | Self::AabbBox { mesh_id } => Some(mesh_id),
            Self::PlaneGrid { .. }
            | Self::QuadOutline { .. }
            | Self::QuadFill
            | Self::CoordinateAxes
            | Self::BlobShadow => None,
        }
    }

    /// The frame's **layer order**: lower layers are submitted first, so later
    /// layers paint over them.
    ///
    /// Explicit, reviewable and testable rather than an artefact of the order the
    /// variants happen to be declared in (#204). It has to be, because **every
    /// overlay pipeline disables depth** (`overlay_depth_stencil` in
    /// `render/pipeline.rs` sets `depth_write_enabled: false` and
    /// `depth_compare: Always` for the wireframe, gizmo-line and shadow
    /// pipelines): grids, quad outlines, wireframes, AABB boxes and axes have no
    /// z-ordering at all, so **submission order *is* the z-order** — and since
    /// they are alpha-blended, an overlap changes the blended *result*, not
    /// merely which one wins.
    ///
    /// The layering, bottom to top:
    ///
    /// | layer | primitive | why |
    /// |---|---|---|
    /// | 0 | [`BlobShadow`](Self::BlobShadow) | a floor decal — everything sits *on* it |
    /// | 1 | [`Mesh`](Self::Mesh), solid modes | the opaque content, z-tested against itself |
    /// | 2 | [`PlaneGrid`](Self::PlaneGrid) | floor lattice, under the other line gizmos |
    /// | 3 | [`QuadFill`](Self::QuadFill) | the quad's highlight wash, under its own edge |
    /// | 4 | [`QuadOutline`](Self::QuadOutline) | the placement quad over its own grid and wash |
    /// | 5 | [`Mesh`](Self::Mesh) in [`Wireframe`](super::RenderMode::Wireframe) | mesh edges over the grid/quad they stand on |
    /// | 6 | [`AabbBox`](Self::AabbBox) | the tracking/selection box over its mesh |
    /// | 7 | [`CoordinateAxes`](Self::CoordinateAxes) | the topmost reference gizmo |
    ///
    /// Note that a **wireframe mesh is not on the solid mesh layer**: it is an
    /// overlay like the gizmos, and it composites *over* the grid and the quad
    /// outline rather than under them.
    fn layer(self) -> u8 {
        match self {
            Primitive::BlobShadow => 0,
            Primitive::Mesh { mode, .. } => match mode {
                RenderMode::Filled | RenderMode::Textured | RenderMode::Shaded => 1,
                RenderMode::Wireframe => 5,
            },
            Primitive::PlaneGrid { .. } => 2,
            Primitive::QuadFill => 3,
            Primitive::QuadOutline { .. } => 4,
            Primitive::AabbBox { .. } => 6,
            Primitive::CoordinateAxes => 7,
        }
    }

    /// The total order the renderer submits primitives in:
    /// `(layer, variation, private geometry slot, complete identity)`.
    ///
    /// - `layer` is [`layer`](Self::layer) — the frame's z-order, since every
    ///   overlay pipeline is depth-disabled and submission order is all there is.
    /// - `variation` orders *within* a layer by what switches pipeline or bind
    ///   groups, so draws sharing GPU state stay adjacent: the solid mesh layer
    ///   runs filled → textured → shaded, one pipeline switch each. Every other
    ///   layer has a single pipeline and leaves it `0`.
    /// - `geometry` orders by the buffers bound: the resolved mesh slot, the grid
    ///   plane, or the quad-outline state.
    ///
    /// Ranking `variation` **above** `geometry` is what keeps mesh draws grouped
    /// by mode rather than by mesh id — the opposite of what a derived `Ord` on
    /// the struct-like variants would do, and worth a pipeline switch per mesh if
    /// it were reversed.
    ///
    /// Slot ordering preserves overlay order after a hole is reused; complete
    /// identity still participates so equal keys mean equal primitives.
    pub(crate) fn sort_key(self, mesh_slot: usize) -> (u8, u8, usize, Option<MeshId>) {
        let (variation, geometry) = match self {
            Primitive::Mesh { mode, .. } => {
                let variation = match mode {
                    RenderMode::Filled => 0,
                    RenderMode::Textured => 1,
                    RenderMode::Shaded => 2,
                    // Alone on layer 5, so its rank within the layer is free.
                    RenderMode::Wireframe => 0,
                };
                (variation, mesh_slot)
            }
            Primitive::AabbBox { .. } => (0, mesh_slot),
            Primitive::PlaneGrid { plane } => (0, plane.index()),
            Primitive::QuadOutline { selected } => (0, usize::from(selected)),
            Primitive::CoordinateAxes | Primitive::BlobShadow | Primitive::QuadFill => (0, 0),
        };
        (self.layer(), variation, geometry, self.mesh_id())
    }
}

/// The base interface for every primitive the renderer can draw (#41): a
/// [`Primitive`] **placed** by a model. A `DrawableObject` is a light, `Copy`
/// handle — geometry (GPU buffers) is owned once by the renderer's decode-once
/// store (meshes keyed by id, plus the shared gizmo geometry), and the drawable
/// carries only *which* primitive to draw and *where*. The renderer and
/// [`Scene`](super::Scene) only ever see `DrawableObject`s and never special-case
/// a concrete primitive type.
///
/// **Every drawable is a placed primitive**: it names geometry and carries the
/// model that places it, so it can be instanced. The two members that did not —
/// the environment background and the background frame plane — are per-frame
/// settings, not primitives, and now live on
/// [`Scene::background`](super::Scene::background) (#204).
///
/// The two fields are private and reached through the named constructors so the
/// split stays honest: `primitive + model` is precisely the decomposition the
/// batcher relies on — strip the model and the rest *is* the batch key — and
/// nothing can bolt a third, un-batched component onto a drawable without going
/// through [`Primitive`] (#204).
///
/// **Why the model lives here and not on the mesh.** `mesh_id` names geometry
/// that is decoded *once* and shared; the model is per-object and per-frame, so
/// a mesh owning it could be placed only once. It would also destroy the batch
/// key — equal [`Primitive`]s are drawn as one instanced command, and a key
/// containing a model makes every object its own batch (and asks `f32` to be
/// `Eq`). The model is per-*instance* GPU data and is fed through the instance
/// vertex buffer, exactly where it belongs. Four of the six primitives have no
/// mesh at all yet still need placing. The matrix that genuinely *is* a property
/// of the asset does live on the mesh — `MeshGpu::base_model`, the decode-time
/// normalization — and the renderer composes the two as
/// `effective = model · base`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DrawableObject {
    primitive: Primitive,
    model: Matrix4,
}

impl DrawableObject {
    /// `primitive` placed by `model` (column-major 4×4).
    pub fn new(primitive: Primitive, model: Matrix4) -> Self {
        Self { primitive, model }
    }

    /// Mesh `mesh_id` placed by `model` and drawn in `mode`.
    pub fn mesh(mesh_id: MeshId, model: Matrix4, mode: RenderMode) -> Self {
        Self::new(Primitive::Mesh { mesh_id, mode }, model)
    }

    /// The AABB outline of mesh `mesh_id`, placed by the same `model` as the mesh
    /// instance it boxes.
    pub fn aabb_box(mesh_id: MeshId, model: Matrix4) -> Self {
        Self::new(Primitive::AabbBox { mesh_id }, model)
    }

    /// A coordinate-plane grid on `plane`, placed by `model`.
    pub fn plane_grid(plane: GridPlane, model: Matrix4) -> Self {
        Self::new(Primitive::PlaneGrid { plane }, model)
    }

    /// The placement-quad outline placed by `model`, in its `selected` or
    /// unselected color.
    pub fn quad_outline(model: Matrix4, selected: bool) -> Self {
        Self::new(Primitive::QuadOutline { selected }, model)
    }

    /// The placement quad's translucent highlight wash, placed by the same
    /// `model` as its outline.
    pub fn quad_fill(model: Matrix4) -> Self {
        Self::new(Primitive::QuadFill, model)
    }

    /// The coordinate-axes gizmo placed by `model` (identity marks the world
    /// origin).
    pub fn coordinate_axes(model: Matrix4) -> Self {
        Self::new(Primitive::CoordinateAxes, model)
    }

    /// A contact/blob grounding shadow placed by `model`.
    pub fn blob_shadow(model: Matrix4) -> Self {
        Self::new(Primitive::BlobShadow, model)
    }

    /// Which primitive this draws — also its batch key.
    pub fn primitive(&self) -> Primitive {
        self.primitive
    }

    /// The typed column-major 4×4 model that places it this frame.
    pub fn model(&self) -> Matrix4 {
        self.model
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// [`Primitive::layer`]'s table in one assertion, so a new variant cannot
    /// quietly land in the wrong layer: overlays are depth-disabled, so this
    /// sequence *is* the frame's z-order (#204).
    ///
    /// The wireframe mesh sitting **after** the grid and the quad outline — and
    /// apart from its solid siblings — is the subtle one: it is an overlay that
    /// composites over them, and alpha blending makes the order visible.
    #[test]
    fn layers_run_shadow_solid_grid_quad_wireframe_aabb_axes() {
        let id = MeshId::fresh().unwrap();
        let layers = [
            Primitive::BlobShadow,
            Primitive::Mesh {
                mesh_id: id,
                mode: RenderMode::Filled,
            },
            Primitive::Mesh {
                mesh_id: id,
                mode: RenderMode::Textured,
            },
            Primitive::Mesh {
                mesh_id: id,
                mode: RenderMode::Shaded,
            },
            Primitive::PlaneGrid {
                plane: GridPlane::Xy,
            },
            Primitive::QuadFill,
            Primitive::QuadOutline { selected: false },
            Primitive::Mesh {
                mesh_id: id,
                mode: RenderMode::Wireframe,
            },
            Primitive::AabbBox { mesh_id: id },
            Primitive::CoordinateAxes,
        ]
        .map(Primitive::layer);

        assert_eq!(layers, [0, 1, 1, 1, 2, 3, 4, 5, 6, 7]);
    }

    /// Within the solid mesh layer the render *mode* outranks the mesh id, so
    /// draws group by pipeline (one switch per mode) instead of by mesh.
    #[test]
    fn mode_outranks_mesh_id_within_the_solid_layer() {
        let filled_last_mesh = Primitive::Mesh {
            mesh_id: MeshId::fresh().unwrap(),
            mode: RenderMode::Filled,
        };
        let textured_first_mesh = Primitive::Mesh {
            mesh_id: MeshId::fresh().unwrap(),
            mode: RenderMode::Textured,
        };
        assert!(filled_last_mesh.sort_key(9) < textured_first_mesh.sort_key(0));
    }
}
