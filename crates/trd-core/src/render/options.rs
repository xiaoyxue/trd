//! Appearance options shared by every front-end (#180).
//!
//! These used to live in `stream.rs`, which is `#[cfg(not(target_arch =
//! "wasm32"))]`, so the browser renderers could not even *name* `RenderOptions`
//! and each kept its own `show_aabb` / `show_axes` booleans — which is why
//! native and web overlay assembly drifted apart. They are plain configuration
//! with no I/O, so they belong beside the scene they describe, available on both
//! platforms.

use crate::scene::{GridPlane, RenderMode};

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

/// Appearance options for [`run_stream`]: the mesh draw [`RenderMode`] plus the
/// optional AABB / coordinate-axes gizmo overlays. Bundled into one value so the
/// entry point threads a single struct instead of many positional flags (and
/// stays within clippy's argument budget). [`Default`] is filled, no overlays,
/// 4× MSAA.
#[derive(Debug, Clone, Default)]
pub struct RenderOptions {
    /// How meshes are drawn (filled / wireframe / textured / PBR).
    pub mode: RenderMode,
    /// Overlay each drawn mesh instance's axis-aligned bounding box (#42).
    pub show_aabb: bool,
    /// Overlay a world-origin coordinate-axes gizmo (#42).
    pub show_axes: bool,
    /// Overlay a coordinate-axes gizmo at *each* drawn object's local (model)
    /// frame — its model-space X/Y/Z axes as placed (e.g. #77's `(e1,e2,e3)`).
    pub show_local_axes: bool,
    /// Overlay a coordinate-plane grid lattice on the given plane at *each*
    /// drawn object's local (model) frame — e.g. `Some(GridPlane::Xy)` tiles a
    /// grid across a placement quad's local floor. `None` disables it.
    pub show_local_grid: Option<GridPlane>,
    /// Narrows [`show_local_grid`](Self::show_local_grid) to draws of a single
    /// `mesh_id` (the placement quad), so a wireframe *content* mesh doesn't also
    /// pick up a floor grid. `None` keeps the grid on every wireframe draw (#114).
    pub show_local_grid_mesh: Option<u32>,
    /// If `Some(plane)`, add one **world-origin** plane grid (a floor at the
    /// world origin), ungated by render mode.
    pub show_world_grid: Option<GridPlane>,
    /// If `Some(plane)`, add a plane grid at *each* drawn instance's own model,
    /// ungated by render mode — unlike [`show_local_grid`](Self::show_local_grid),
    /// which is wireframe-scoped.
    pub show_object_grid: Option<GridPlane>,
    /// If `Some(index)`, highlight that draw's AABB — the **selected** object
    /// (#141) — regardless of [`show_aabb`](Self::show_aabb).
    pub selected: Option<u32>,
    /// Disney PBR material + environment map, applied when `mode` is
    /// [`RenderMode::Shaded`] (also honoured for any per-draw PBR-mode draws).
    pub pbr: Option<PbrConfig>,
    /// Mesh-pass multisample anti-aliasing (default [`Msaa::X4`]).
    pub msaa: Msaa,
}
