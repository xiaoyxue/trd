//! [`DrawableObject`] — the single base interface for every primitive the
//! renderer can draw (#41), and the [`Scene`] it forms.
//!
//! Adding a primitive means adding a variant here; the renderer batches a scene
//! by primitive without special-casing any concrete type. Geometry is owned once
//! by the renderer's decode-once store, so a drawable is a light `Copy` handle
//! naming *which* primitive plus its per-frame model.

use super::{GridPlane, RenderMode};

/// The base interface for every primitive the renderer can draw (#41). A
/// `DrawableObject` is a light, `Copy` handle: geometry (GPU buffers) is owned
/// once by the renderer's decode-once store (meshes keyed by id, plus the shared
/// gizmo geometry), and each variant carries only *which* primitive to draw and
/// its per-frame model. The renderer and [`Scene`] only ever see
/// `DrawableObject`s and never special-case a concrete primitive type.
///
/// Wireframe is a render *mode* of the [`DrawableObject::Mesh`] primitive (not a
/// separate variant); the coordinate axes and the AABB box are genuinely
/// distinct gizmo primitives rendered with screen-space-expanded line geometry.
///
/// **Every variant is a placed primitive**: it names geometry and carries the
/// model that places it, so it can be instanced. The two members that did not —
/// the environment background and the background frame plane — are per-frame
/// settings, not primitives, and now live on
/// [`Scene::background`](super::Scene::background) (#204).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DrawableObject {
    /// A decoded mesh (id = row index in the leading mesh table) placed by
    /// `model` and drawn in `mode` (filled or wireframe). `model` is the
    /// per-frame draw model; the renderer pre-multiplies the mesh's base
    /// (preview) model beneath it (`effective = model · base`).
    Mesh {
        mesh_id: u32,
        model: [f32; 16],
        mode: RenderMode,
    },
    /// The axis-aligned bounding-box outline of mesh `mesh_id` (#42), placed by
    /// the same `model` as the mesh instance it boxes (the renderer applies that
    /// mesh's base model beneath `model` too), so the box tracks the mesh
    /// exactly. Reuses the mesh's precomputed corner geometry.
    AabbBox { mesh_id: u32, model: [f32; 16] },
    /// The world-orientation coordinate gizmo (#42): three anti-aliased shafts
    /// with cone arrowheads from the origin along +X/+Y/+Z, colored
    /// red/green/blue. Placed by `model` (identity marks the world origin); not
    /// tied to any mesh, so no base model is applied.
    CoordinateAxes { model: [f32; 16] },
    /// A **coordinate-plane grid** lattice on `plane` (X/Y, X/Z, or Y/Z),
    /// spanning the model-space square `[-1, 1]²`, placed by `model`. Like
    /// [`CoordinateAxes`](Self::CoordinateAxes) it is a screen-space-expanded
    /// line gizmo tied to no mesh (no base model); with a #77 placement-quad
    /// `model` the `Xy` grid lays exactly over the reconstructed quad in its
    /// local frame.
    PlaneGrid { plane: GridPlane, model: [f32; 16] },
    /// The tracked placement-quad outline, rendered by the shared analytic-AA
    /// gizmo line pipeline at 1.5 px.
    QuadOutline { model: [f32; 16], selected: bool },
    /// A **contact / blob grounding shadow** (#110 follow-up): a soft dark radial
    /// blob laid on a placed mesh's ground plane, placed by `model` (a flat quad
    /// on the plane, sized to the mesh footprint), so the mesh reads as *sitting
    /// on* the reconstructed surface rather than floating over the composited
    /// video plate. A [`DrawSelection::Shadow`](super::DrawSelection) draw becomes this variant. Tied to no
    /// mesh (no base model); alpha-blended over the background frame plane
    /// ([`Background::frame`](super::Background::frame)) and drawn *before* the
    /// opaque content mesh (depth-write off) so the mesh composites on top while
    /// the surrounding rim darkens the floor.
    BlobShadow { model: [f32; 16] },
}
