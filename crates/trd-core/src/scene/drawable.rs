//! [`DrawableObject`] — the single base interface for every primitive the
//! renderer can draw (#41), and the [`Scene`] it forms.
//!
//! Adding a primitive means adding a variant here; the renderer batches a scene
//! by primitive without special-casing any concrete type. Geometry is owned once
//! by the renderer's decode-once store, so a drawable is a light `Copy` handle
//! naming *which* primitive plus its per-frame model.

use super::{FrameFit, GridPlane, RenderMode};
use crate::render::Tonemap;

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
    /// video plate. A [`RenderMode::Shadow`] draw becomes this variant. Tied to no
    /// mesh (no base model); alpha-blended over the [`FramePlane`](Self::FramePlane)
    /// and drawn *before* the opaque content mesh (depth-write off) so the mesh
    /// composites on top while the surrounding rim darkens the floor.
    BlobShadow { model: [f32; 16] },
    /// Camera-centered spherical HDR environment drawn behind the scene.
    EnvironmentBackground {
        rotation: f32,
        exposure: f32,
        blur: f32,
        tonemap: Tonemap,
    },
    /// A screen-aligned **background frame plane** (#63): a fullscreen quad that
    /// samples the renderer's bound background frame texture (set via
    /// [`SceneRenderer::update_frame_texture_rgba`]), composited **under** the
    /// mesh scene. `fit` selects how the image maps to the viewport. Carries no
    /// model — it is authored directly in clip space and ignores the camera.
    /// Drawn only when a background texture is bound (else skipped), so an absent
    /// `frame_path`/`frame_url` renders with no background (back-compat).
    FramePlane { fit: FrameFit },
}

/// A frame's ordered list of [`DrawableObject`]s the renderer walks and encodes
/// under the one shared camera `P·V` uniform. The wire authors the mesh draws
/// (the protocol 0.0.3 draw list); the core adds gizmo drawables (axes, AABB
/// boxes). A single-mesh frame is the degenerate one-element scene — the
/// renderer always iterates a `Scene`, with no single-object special case.
pub type Scene = Vec<DrawableObject>;
