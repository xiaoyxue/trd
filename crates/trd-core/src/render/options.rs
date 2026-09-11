//! Appearance options shared by every front-end (#180).
//!
//! These used to live in `stream.rs` (since split into `io/` + `stream_filter/`
//! by #296), which is `#[cfg(not(target_arch =
//! "wasm32"))]`, so the browser renderers could not even *name* `RenderOptions`
//! and each kept its own `show_aabb` / `show_axes` booleans — which is why
//! native and web overlay assembly drifted apart. They are plain configuration
//! with no I/O, so they belong beside the scene they describe, available on both
//! platforms.

use super::{GridPlane, RenderMode};

/// The typed Disney PBR configuration threaded through [`RenderOptions`].
#[derive(Debug, Clone, Default)]
pub struct PbrConfig {
    /// The Disney material applied to every PBR mesh.
    pub material: crate::DisneyMaterial,
    /// Scene light-rig controls.
    pub lighting: crate::Lighting,
    /// Image-based-lighting controls applied to every PBR mesh.
    pub ibl: crate::ImageBasedLighting,
    /// Per-object output transform seeded onto every PBR mesh.
    pub tone_mapping: crate::ToneMapping,
    /// The HDR environment probe reflected by metallic surfaces (`None` ⇒ no
    /// environment reflection).
    pub env_map: Option<crate::EnvMapData>,
}

/// The mesh-pass multisample anti-aliasing setting threaded through
/// [`RenderOptions`]. [`Msaa::X4`] (the default) renders the 4×-multisampled mesh
/// pass — smooth wireframe / gizmo / AABB / silhouette edges; [`Msaa::Off`]
/// renders single-sampled (aliased edges, the raw rasterized coverage). Both are
/// covered by the golden-render test.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Msaa {
    /// 4× multisampling — the default anti-aliased mesh pass.
    #[default]
    X4,
    /// No multisampling: render the mesh pass single-sampled (aliased edges).
    Off,
}

impl Msaa {
    /// The wgpu sample count for this setting (`4` for [`Msaa::X4`], `1` for
    /// [`Msaa::Off`]).
    pub fn sample_count(self) -> u32 {
        match self {
            Msaa::X4 => super::MSAA_SAMPLE_COUNT,
            Msaa::Off => 1,
        }
    }
}

/// Which draws an [`ObjectGrid`] is laid under.
///
/// The two scopes are the two overlay fields this enum merged (#376): `AllMeshes`
/// is the interactive "local grid" — a floor under every object, whatever its
/// render mode — and `Wireframe` is the #77 placement-quad grid, scoped so a
/// filled/textured content mesh whose local `Xy` is a vertical plane gets no
/// stray grid "wall".
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum GridScope {
    /// Every drawn mesh instance gets the grid, ungated by render mode.
    #[default]
    AllMeshes,
    /// Only **wireframe** draws get the grid, optionally narrowed further to
    /// draws of a single `mesh`.
    ///
    /// `mesh` exists because the *content* mesh is sometimes drawn wireframe too
    /// (a wireframe-reveal intro over a placement quad), and the grid would then
    /// land under every wireframe object; pin it to the quad's `mesh_id` so
    /// exactly one floor grid is laid (#114). `None` keeps every wireframe draw.
    Wireframe {
        /// The only `mesh_id` that gets a grid, or `None` for all of them.
        mesh: Option<u32>,
    },
}

impl GridScope {
    /// Whether a draw of `mesh_id` drawn in `mode` gets the grid.
    pub(crate) fn covers(self, mesh_id: u32, mode: RenderMode) -> bool {
        match self {
            GridScope::AllMeshes => true,
            GridScope::Wireframe { mesh } => {
                mode == RenderMode::Wireframe && mesh.is_none_or(|id| id == mesh_id)
            }
        }
    }
}

/// A coordinate-plane grid lattice laid at *each* drawn object's local (model)
/// frame — e.g. `Xy` tiles a grid across a placement quad's local floor, `Xz`
/// lays a floor that follows an object as it is moved.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ObjectGrid {
    /// The model-space plane the lattice spans.
    pub plane: GridPlane,
    /// Which draws it is laid under.
    pub scope: GridScope,
}

/// The gizmo and grid overlays drawn over a scene's meshes.
///
/// One value rather than seven fields on [`RenderOptions`] (#376): assembly
/// consumes them as a unit, and every front-end that holds overlay state now
/// holds *this* — so `trd-gui` no longer keeps a parallel set of `bool`s that a
/// hand-written translator (and a test guarding it) had to keep in step. That
/// drift is the same one `options.rs` exists to prevent (#180).
///
/// [`Default`] is no overlays.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Overlays {
    /// Overlay each drawn mesh instance's axis-aligned bounding box (#42).
    pub aabb: bool,
    /// Overlay a world-origin coordinate-axes gizmo (#42).
    pub axes: bool,
    /// Overlay a coordinate-axes gizmo at *each* drawn object's local (model)
    /// frame — its model-space X/Y/Z axes as placed (e.g. #77's `(e1,e2,e3)`).
    pub local_axes: bool,
    /// If `Some`, lay a plane grid at *each* drawn object's own model, gated by
    /// the grid's own [`GridScope`].
    pub object_grid: Option<ObjectGrid>,
    /// If `Some(plane)`, add one **world-origin** plane grid (a floor at the
    /// world origin, analogous to [`axes`](Self::axes)), ungated by render mode.
    pub world_grid: Option<GridPlane>,
}

/// Appearance options for [`run_stream`](crate::run_stream): the mesh draw [`RenderMode`] plus the
/// [`Overlays`] drawn over it. Bundled into one value so the
/// entry point threads a single struct instead of many positional flags (and
/// stays within clippy's argument budget). [`Default`] is filled, no overlays,
/// 4× MSAA.
#[derive(Debug, Clone, Default)]
pub struct RenderOptions {
    /// How meshes are drawn (filled / wireframe / textured / PBR).
    pub mode: RenderMode,
    /// The gizmo and grid overlays drawn over those meshes.
    pub overlays: Overlays,
    /// If `Some(index)`, highlight that draw's AABB — the **selected** object
    /// (#141) — regardless of [`Overlays::aabb`].
    ///
    /// Not an [`Overlays`] field: it is an index into the frame's `draws`, not a
    /// toggle the front-end holds across frames.
    pub selected: Option<u32>,
    /// Disney PBR material + environment map, applied when `mode` is
    /// [`RenderMode::Shaded`] (also honoured for any per-draw PBR-mode draws).
    pub pbr: Option<PbrConfig>,
    /// Draw the bound HDR environment probe as the frame's **background sky**,
    /// behind every primitive. `None` (the default) ⇒ no sky.
    ///
    /// A top-level option rather than a [`PbrConfig`] field (#235 R2): the sky is
    /// a [`Background`](crate::Background) setting on the assembled scene, not a
    /// surface material, and the front-end that draws it today (`trd-gui`) passes
    /// `pbr: None`. Requires an environment probe to be bound (`--env` /
    /// [`PbrConfig::env_map`] / `Renderer::set_env_map`); with no probe the
    /// placeholder 1×1 black one is drawn.
    ///
    /// It carries **no yaw**: the probe's rotation is the scene-level
    /// [`EnvironmentLight`](crate::EnvironmentLight), so the sky and the
    /// reflections on the objects in front of it cannot disagree (#182).
    pub env_background: Option<crate::EnvironmentBackground>,
    /// Mesh-pass multisample anti-aliasing (default [`Msaa::X4`]).
    pub msaa: Msaa,
}
