//! [`Scene`] — a frame's ordered list of primitives plus the [`Background`] they
//! composite over, and the one place it is assembled from a wire draw list.
//!
//! [`Scene::try_from_frame`] resolves wire rows, while [`Scene::from_draws`]
//! accepts registered identities. Both share the same assembly; the overlay
//! builders it composes are private, so a front-end cannot assemble half a
//! scene. That is what keeps native and browser rendering the same frame from
//! the same inputs (#180).
//!
//! Objects and background are separate because they *are* separate things
//! (#204): an object is a placed, instanceable primitive; a background is a
//! per-frame setting with no model, no instance and no place in the draw list.

use super::{Draw, DrawableObject, FrameFit, GridPlane, RenderMode, ResolvedDraw};
use crate::math::Matrix4;
use crate::render::Tonemap;
use crate::{DecodedFrame, Lighting, MeshId, MeshTable, MeshTableIndex, RenderOptions};

/// Errors from assembling a [`Scene`] out of a decoded frame.
///
/// Separate from `RenderError` because assembly touches no GPU: it is a pure
/// function of the wire frame plus the caller's appearance options, so every
/// front-end can validate a frame before it has a device.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum SceneError {
    /// A wire draw or mesh-selector option names a row outside the CPU table.
    #[error("draw references mesh index {mesh_id} but only {mesh_count} mesh(es) are loaded")]
    MeshIndexOutOfRange {
        mesh_id: MeshTableIndex,
        mesh_count: usize,
    },
}

/// The camera-centered spherical HDR **environment background**: the bound
/// environment map drawn behind everything else, as seen through this frame's
/// camera.
///
/// Settings, not a primitive (#204): it carries no model and never enters the
/// instance buffer, so it lives on [`Background`] rather than in the drawable
/// list. `exposure` scales its radiance, `blur` (0…1) fades it toward its
/// blurred mips, and `tonemap` is the operator applied on the way to display.
///
/// It carries **no yaw**: the probe's rotation is a scene-level
/// [`EnvironmentLight`](crate::EnvironmentLight), so the sky drawn here and the
/// reflections on the objects in front of it cannot disagree (#182).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EnvironmentBackground {
    /// Linear radiance multiplier.
    pub exposure: f32,
    /// Blur amount, `0.0` (sharp) … `1.0` (fully blurred).
    pub blur: f32,
    /// The tone-map operator applied to the background.
    pub tonemap: Tonemap,
}

/// What a frame draws **behind** its primitives: the environment probe and/or
/// the background video/still frame plane.
///
/// Both used to be `DrawableObject` variants, which was a category error (#204):
/// they are the only members carrying no model, the only ones the batcher had to
/// `continue` past, and their singleton-ness had no type-level guarantee — two
/// frame planes in one list simply meant the last one silently won. On `Scene`
/// they are per-frame *settings*, set once, with no ordering to get wrong.
///
/// **Two independent `Option`s, deliberately — not one enum.** The environment
/// and the frame plane are *not* alternatives: the renderer draws the
/// environment first and the frame plane over it, in the same pass, and a scene
/// may legitimately have both, either, or neither. A single
/// `Option<BackgroundKind>` would make them mutually exclusive and silently drop
/// one of the two for any frame that uses both.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Background {
    /// The HDR environment probe drawn first, behind everything.
    pub environment: Option<EnvironmentBackground>,
    /// How the renderer's bound background frame texture (see
    /// [`Renderer::update_frame_texture_rgba`](crate::Renderer::update_frame_texture_rgba))
    /// maps onto the viewport (#63). A fullscreen quad authored directly in clip
    /// space, so it ignores the camera; drawn *over* the environment and *under*
    /// the mesh scene. `Some(fit)` with no bound texture draws nothing.
    pub frame: Option<FrameFit>,
}

/// A frame's ordered list of [`DrawableObject`]s the renderer walks and encodes
/// under the one shared camera `P·V` uniform, plus the [`Background`] they
/// composite over. The wire authors the mesh draws (the protocol draw list); the
/// core adds gizmo drawables (axes, AABB boxes). A single-mesh frame is the
/// degenerate one-element scene — the renderer always iterates a `Scene`, with
/// no single-object special case.
///
/// Shared assembly behind [`Scene::from_draws`] and [`Scene::try_from_frame`]
/// keeps every front-end rendering the same scene from the same inputs.
/// Primitives are exposed through [`objects`](Self::objects), and background
/// settings through [`background`](Self::background).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Scene {
    background: Background,
    objects: Vec<DrawableObject>,
    /// The light rig every PBR object in this frame is lit by.
    ///
    /// Scene-level like `objects` and `background`, so it arrives **with** the
    /// frame instead of being sticky renderer state a front-end pokes through a
    /// setter — two scenes rendered by one renderer used to share it silently
    /// (#182). `Default` keeps the previous behaviour for a scene that never
    /// sets it.
    lighting: Lighting,
}

impl Scene {
    /// An empty scene with no background.
    pub fn new() -> Self {
        Self::default()
    }

    /// An empty scene with room for `capacity` objects and no background.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            objects: Vec::with_capacity(capacity),
            ..Self::default()
        }
    }

    /// This scene with `background` behind it.
    #[must_use]
    pub fn with_background(mut self, background: Background) -> Self {
        self.background = background;
        self
    }

    /// This scene lit by `lighting`.
    #[must_use]
    pub fn with_lighting(mut self, lighting: Lighting) -> Self {
        self.lighting = lighting;
        self
    }

    /// The light rig every PBR object in this frame is lit by.
    pub fn lighting(&self) -> Lighting {
        self.lighting
    }

    /// The light rig, for a front-end that adjusts it after assembly (a GUI
    /// slider) rather than at build time.
    pub fn lighting_mut(&mut self) -> &mut Lighting {
        &mut self.lighting
    }

    /// What this frame draws behind its primitives.
    pub fn background(&self) -> &Background {
        &self.background
    }

    /// The background, for a front-end that turns one of its slots on or off
    /// (e.g. an "environment background" toggle) after assembly.
    pub fn background_mut(&mut self) -> &mut Background {
        &mut self.background
    }

    /// Appends one primitive. Order is draw order within a primitive, so callers
    /// push in the order the renderer's buckets expect.
    pub fn push(&mut self, object: DrawableObject) {
        self.objects.push(object);
    }

    /// Appends every primitive of `objects`.
    pub fn extend(&mut self, objects: impl IntoIterator<Item = DrawableObject>) {
        self.objects.extend(objects);
    }

    /// The primitives, in draw order.
    pub fn objects(&self) -> &[DrawableObject] {
        &self.objects
    }

    /// Assembles a frame's scene from registered mesh placements plus
    /// appearance options.
    ///
    /// Every front-end used to do this itself — the headless `Renderer` from nine
    /// retained overlay flags, `trd-gui` through setters, the browser renderers
    /// from their own booleans — which is how native and web overlay assembly
    /// drifted apart. Assembling here means all of them get the same scene from
    /// the same inputs (#180), and owning it on `Scene` means the overlay
    /// builders need not be public at all (#203).
    ///
    /// `frame` sets the [`Background::frame`] fit (#63) when the caller has a
    /// frame texture bound; the sky *behind* it comes from
    /// [`RenderOptions::env_background`], so every front-end that binds an HDR
    /// probe can draw it as a background from the same shared assembly instead of
    /// reaching around this function to set `background_mut()` (#235 R2).
    pub fn from_draws(
        draws: &[ResolvedDraw],
        options: &RenderOptions<MeshId>,
        frame: Option<FrameFit>,
    ) -> Self {
        Self::assemble(draws, options, frame, options.show_local_grid_mesh)
    }

    fn assemble<M>(
        draws: &[ResolvedDraw],
        options: &RenderOptions<M>,
        frame: Option<FrameFit>,
        grid_mesh: Option<MeshId>,
    ) -> Self {
        let mut scene = build_scene(draws, options, frame, grid_mesh);
        // World / object plane grids (#140) are ungated by render mode, so a
        // filled or shaded object still gets a floor. `encode` buckets by
        // primitive type, so appending here still draws them in the grid pass.
        scene.extend(plane_grid_overlays(
            draws,
            options.show_world_grid,
            options.show_object_grid,
        ));
        // Selection highlight (#141): drawn even when show-all-AABBs is off.
        scene.extend(selection_aabb_overlay(draws, options.selected));
        // The light rig is part of the frame, not sticky renderer state (#182),
        // so assembly carries it over from the same options every front-end
        // already passes.
        scene.lighting = options
            .pbr
            .as_ref()
            .map(|pbr| pbr.lighting)
            .unwrap_or_default();
        scene
    }

    /// Assembles a decoded wire frame into its [`Scene`], validating that every
    /// draw names a mesh the stream actually sent.
    ///
    /// The wire draw list is resolved through
    /// [`DecodedFrame::resolved_draws`] — an explicit list is drawn verbatim (an
    /// empty one leaves just the background), an absent one places a single
    /// instance of row zero by the frame's own model. Every row, including a
    /// shadow draw's row and the local-grid selector, resolves through `meshes`.
    ///
    /// This is the whole of what the CLI, `CanvasRenderer` and
    /// `OffscreenRenderer` used to spell out separately, in three different
    /// error types. Pure: no GPU, no render target, so it is testable without a
    /// device and identical on every platform.
    pub fn try_from_frame(
        frame: &DecodedFrame,
        meshes: &MeshTable,
        options: &RenderOptions,
        frame_fit: Option<FrameFit>,
    ) -> Result<Self, SceneError> {
        let draws = Self::resolve_draws(&frame.resolved_draws(), meshes)?;
        let grid_mesh = options
            .show_local_grid_mesh
            .map(|row| resolve_mesh(row, meshes))
            .transpose()?;
        Ok(Self::assemble(&draws, options, frame_fit, grid_mesh))
    }

    /// Resolves every wire row against one CPU registration, before any GPU upload.
    ///
    /// Shadow rows are validated too. Draw order and selection indices are unchanged.
    pub fn resolve_draws(
        draws: &[Draw],
        meshes: &MeshTable,
    ) -> Result<Vec<ResolvedDraw>, SceneError> {
        draws
            .iter()
            .map(|draw| {
                Ok(ResolvedDraw {
                    mesh_id: resolve_mesh(draw.mesh_id, meshes)?,
                    model: draw.model,
                    selection: draw.selection,
                })
            })
            .collect()
    }
}

fn resolve_mesh(row: MeshTableIndex, meshes: &MeshTable) -> Result<MeshId, SceneError> {
    meshes.id(row).ok_or(SceneError::MeshIndexOutOfRange {
        mesh_id: row,
        mesh_count: meshes.len(),
    })
}

impl FromIterator<DrawableObject> for Scene {
    fn from_iter<T: IntoIterator<Item = DrawableObject>>(iter: T) -> Self {
        Self {
            objects: iter.into_iter().collect(),
            ..Self::default()
        }
    }
}

impl From<Vec<DrawableObject>> for Scene {
    fn from(objects: Vec<DrawableObject>) -> Self {
        Self {
            objects,
            ..Self::default()
        }
    }
}

/// Builds a per-frame [`Scene`] from a wire `draws` list plus the appearance
/// `options` every front-end already holds ([`RenderOptions::mode`] and the
/// overlay flags; the grid/selection/PBR fields are applied by
/// [`Scene::from_draws`] around this core). When `frame` is `Some`, the scene's
/// [`Background::frame`] fit is set so the mesh scene composites on top of the
/// bound frame texture (#63); the background is a scene-level setting rather
/// than a leading drawable, since it carries no model and cannot be instanced
/// (#204). Each mesh [`ResolvedDraw`] becomes one [`Primitive::Mesh`](super::Primitive::Mesh)
/// in the draw's own [`ResolvedDraw::selection`] mode when set, else [`RenderOptions::mode`]; with
/// [`show_aabb`](RenderOptions::show_aabb), each also emits a tracking
/// [`Primitive::AabbBox`](super::Primitive::AabbBox); with
/// [`show_local_grid`](RenderOptions::show_local_grid) `= Some(plane)`, each
/// **wireframe-mode** draw emits a [`Primitive::PlaneGrid`](super::Primitive::PlaneGrid) on `plane` at
/// **its own `model`** (a coordinate-plane lattice in that object's local frame —
/// e.g. the `Xy` grid on a #77 placement quad's surface; scoped to wireframe
/// draws so a filled/textured content mesh whose local `Xy` is vertical gets no
/// stray grid wall).
/// [`show_local_grid_mesh`](RenderOptions::show_local_grid_mesh) `= Some(id)`
/// narrows the grid further to draws of
/// **that** mesh only (#110 follow-up): when the *content* mesh is also drawn
/// wireframe (e.g. a wireframe-reveal intro over a placement quad), the grid
/// would otherwise land on every wireframe object; pin it to the placement
/// quad's `mesh_id` so exactly one floor grid is laid. `None` keeps
/// the "all wireframe draws" behaviour. With
/// [`show_axes`](RenderOptions::show_axes), one **world-origin**
/// [`Primitive::CoordinateAxes`](super::Primitive::CoordinateAxes) is appended; with
/// [`show_local_axes`](RenderOptions::show_local_axes), each
/// draw also emits a [`Primitive::CoordinateAxes`](super::Primitive::CoordinateAxes) at **its own `model`** —
/// i.e. that object's *local* coordinate frame (its model-space X/Y/Z axes as
/// placed, e.g. #77's `(e1,e2,e3)` quad frame). The order (all meshes, then all
/// boxes, then per-draw grids, then per-draw local axes, then the world-origin
/// axes — all of it over the background) matches the renderer's draw buckets so
/// output is pixel-identical to the pre-scene, flag-driven path.
///
/// Takes the whole `&RenderOptions` rather than one positional flag per field
/// (#235 R0): every argument but `frame` was already a field of the options its
/// only caller holds, exploded one by one and immediately re-assembled — which
/// is how an 8-parameter private function acquires a ninth.
///
/// Private: [`Scene::from_draws`] is the entry point, so a front-end cannot
/// assemble half a scene.
fn build_scene<M>(
    draws: &[ResolvedDraw],
    options: &RenderOptions<M>,
    frame: Option<FrameFit>,
    grid_mesh: Option<MeshId>,
) -> Scene {
    let RenderOptions {
        mode,
        show_aabb,
        show_axes,
        show_local_axes,
        show_local_grid: local_grid,
        env_background,
        ..
    } = *options;
    let mut scene = Scene::with_capacity(
        draws.len()
            * (1 + usize::from(show_aabb)
                + usize::from(show_local_axes)
                + usize::from(local_grid.is_some()))
            + usize::from(show_axes),
    )
    .with_background(Background {
        // Both slots are filled here, and independently: the environment sky is
        // drawn first, the frame plane over it (#204).
        environment: env_background,
        frame,
    });
    // The one place a `DrawSelection` is resolved: past here the scene holds
    // primitives, and nothing downstream needs to know a shadow ever existed.
    for draw in draws {
        scene.push(match draw.selection.mesh_mode(mode) {
            Some(mode) => DrawableObject::mesh(draw.mesh_id, draw.model, mode),
            None => DrawableObject::blob_shadow(draw.model),
        });
    }
    if show_aabb {
        for draw in draws.iter().filter(|d| d.selection.is_mesh()) {
            scene.push(DrawableObject::aabb_box(draw.mesh_id, draw.model));
        }
    }
    if let Some(plane) = local_grid {
        for draw in draws {
            // Scope the grid to wireframe draws only — the #77 placement quad is
            // always an outline (its local Xy *is* the placement surface), while a
            // filled/textured content mesh's local Xy may be a vertical plane, so a
            // per-mesh grid there would draw a stray grid "wall". When the content
            // mesh is *also* wireframe (e.g. a wireframe-reveal intro), `grid_mesh`
            // narrows the grid to the placement quad's `mesh_id` so exactly one
            // floor grid is laid — not one under every wireframe object (#114).
            let is_wireframe = draw.selection.mesh_mode(mode) == Some(RenderMode::Wireframe);
            let mesh_selected = grid_mesh.is_none_or(|id| draw.mesh_id == id);
            if is_wireframe && mesh_selected {
                scene.push(DrawableObject::plane_grid(plane, draw.model));
            }
        }
    }
    if show_local_axes {
        // A shadow blob is a floor decal, not a placed object whose local frame
        // warrants an axes gizmo.
        for draw in draws.iter().filter(|d| d.selection.is_mesh()) {
            scene.push(DrawableObject::coordinate_axes(draw.model));
        }
    }
    if show_axes {
        scene.push(DrawableObject::coordinate_axes(Matrix4::IDENTITY));
    }
    scene
}

/// Builds **plane-grid overlay** drawables independent of [`build_scene`]'s
/// wireframe-scoped [`show_local_grid`](RenderOptions::show_local_grid) (#114): a `world_grid` lays one
/// [`Primitive::PlaneGrid`](super::Primitive::PlaneGrid) at the **world origin** (identity model — the
/// world floor, analogous to `show_axes`), and an `object_grid` lays a
/// `PlaneGrid` at **each drawn object's own model** frame (analogous to
/// `show_local_axes`), ungated by render mode. Shadow draws are skipped (a blob
/// decal has no frame to grid). Appended to a scene by front-ends that want a
/// grid under a *filled/textured/PBR* object (e.g. the interactive `trd-gui`
/// overlays) without the #77 wireframe-quad gating; `None`/`None` yields an empty
/// list, so callers that don't opt in are byte-identical.
pub(crate) fn plane_grid_overlays(
    draws: &[ResolvedDraw],
    world_grid: Option<GridPlane>,
    object_grid: Option<GridPlane>,
) -> Vec<DrawableObject> {
    let mut grids = Vec::new();
    if let Some(plane) = world_grid {
        grids.push(DrawableObject::plane_grid(plane, Matrix4::IDENTITY));
    }
    if let Some(plane) = object_grid {
        for draw in draws.iter().filter(|d| d.selection.is_mesh()) {
            grids.push(DrawableObject::plane_grid(plane, draw.model));
        }
    }
    grids
}

/// A **selection-highlight** overlay (#141): the [`Primitive::AabbBox`](super::Primitive::AabbBox) of a
/// single object — the `selected` 0-based index into `draws` — so *only* that
/// object's bounding box is drawn (unlike the global "show all AABBs" toggle).
/// `None`, an out-of-range index, or a `Shadow` draw yields an empty list, so a
/// caller that doesn't opt in is byte-identical. Appended to the scene by
/// front-ends that highlight a clicked object.
pub(crate) fn selection_aabb_overlay(
    draws: &[ResolvedDraw],
    selected: Option<u32>,
) -> Vec<DrawableObject> {
    let Some(draw) = selected.and_then(|i| draws.get(i as usize)) else {
        return Vec::new();
    };
    if !draw.selection.is_mesh() {
        return Vec::new();
    }
    vec![DrawableObject::aabb_box(draw.mesh_id, draw.model)]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::render::DrawSelection;
    use crate::render::Primitive;

    fn mesh_id(row: u32) -> MeshId {
        static TABLE: std::sync::OnceLock<MeshTable> = std::sync::OnceLock::new();
        TABLE
            .get_or_init(|| MeshTable::new(vec![crate::Mesh::hello_triangle(); 8]).unwrap())
            .id(MeshTableIndex::new(row))
            .unwrap()
    }

    fn build_scene(
        draws: &[ResolvedDraw],
        options: &RenderOptions<MeshId>,
        frame: Option<FrameFit>,
    ) -> Scene {
        super::build_scene(draws, options, frame, options.show_local_grid_mesh)
    }

    #[test]
    fn plane_grid_overlays_place_world_and_object_grids() {
        let draws = [
            ResolvedDraw {
                mesh_id: mesh_id(0),
                model: Matrix4::from_translation(crate::math::Vector3::new(1.0, 0.0, 0.0)),
                selection: DrawSelection::INHERIT,
            },
            ResolvedDraw {
                mesh_id: mesh_id(1),
                model: Matrix4::IDENTITY,
                selection: DrawSelection::Shadow,
            },
        ];
        // Neither grid ⇒ empty (opt-in only, byte-identical for non-users).
        assert!(plane_grid_overlays(&draws, None, None).is_empty());

        // World grid ⇒ exactly one identity-model grid on the requested plane.
        let world = plane_grid_overlays(&draws, Some(GridPlane::Xz), None);
        assert_eq!(world.len(), 1);
        assert_eq!(
            world[0],
            DrawableObject::plane_grid(GridPlane::Xz, Matrix4::IDENTITY)
        );

        // Object grid ⇒ one grid per *non-shadow* draw, at that draw's model.
        let object = plane_grid_overlays(&draws, None, Some(GridPlane::Xz));
        assert_eq!(object.len(), 1);
        assert_eq!(
            object[0],
            DrawableObject::plane_grid(GridPlane::Xz, draws[0].model)
        );
    }

    #[test]
    fn build_scene_puts_the_frame_on_the_background_not_in_the_objects() {
        let draws = [ResolvedDraw {
            mesh_id: mesh_id(0),
            model: Matrix4::IDENTITY,
            selection: DrawSelection::INHERIT,
        }];

        // No frame ⇒ no background frame (byte-identical to the pre-0.0.5 scene).
        let scene = build_scene(
            &draws,
            &RenderOptions {
                mode: RenderMode::Filled,
                show_aabb: true,
                show_axes: true,
                ..Default::default()
            },
            None,
        );
        assert_eq!(scene.background().frame, None, "no frame ⇒ no frame plane");

        // Some(fit) ⇒ the fit lands on the background, which the renderer draws
        // under every primitive — so it is *not* an object in the draw list
        // (#204).
        let scene = build_scene(
            &draws,
            &RenderOptions {
                mode: RenderMode::Filled,
                show_aabb: true,
                show_axes: true,
                ..Default::default()
            },
            Some(FrameFit::Cover),
        );
        assert_eq!(scene.background().frame, Some(FrameFit::Cover));
        assert_eq!(
            scene.background().environment,
            None,
            "the frame slot must not touch the environment slot"
        );
        // The objects keep the mesh → aabb → axes order, with nothing prepended.
        assert!(matches!(
            scene.objects()[0].primitive(),
            Primitive::Mesh { .. }
        ));
        assert!(matches!(
            scene.objects()[1].primitive(),
            Primitive::AabbBox { .. }
        ));
        assert!(matches!(
            scene.objects()[2].primitive(),
            Primitive::CoordinateAxes
        ));
    }

    /// The two background slots are **independent** (#204): a scene may carry an
    /// environment probe *and* a frame plane, and the renderer draws both (the
    /// environment first, the frame plane over it). This is the regression the
    /// two-`Option` shape exists to prevent — a single `Option<enum>` would
    /// silently drop one of them.
    #[test]
    fn both_backgrounds_can_be_set_at_once() {
        let environment = EnvironmentBackground {
            exposure: 1.5,
            blur: 0.25,
            tonemap: Tonemap::Aces,
        };
        let draws = [ResolvedDraw {
            mesh_id: mesh_id(0),
            model: Matrix4::IDENTITY,
            selection: DrawSelection::INHERIT,
        }];
        let options = crate::RenderOptions {
            mode: RenderMode::Filled,
            env_background: Some(environment),
            ..Default::default()
        };

        // Both slots come from the *shared* assembly now (#235 R2): the sky from
        // the options, the frame fit from the argument.
        let scene = Scene::from_draws(&draws, &options, Some(FrameFit::Stretch));

        assert_eq!(scene.background().environment, Some(environment));
        assert_eq!(scene.background().frame, Some(FrameFit::Stretch));
        // Setting one slot must not disturb the objects either.
        assert_eq!(scene.objects().len(), 1);

        // The same holds for the builder form and a plain `Background`.
        let built = Scene::new().with_background(Background {
            environment: Some(environment),
            frame: Some(FrameFit::Cover),
        });
        assert_eq!(built.background().environment, Some(environment));
        assert_eq!(built.background().frame, Some(FrameFit::Cover));
    }

    /// The environment sky is assembled from the options like every other
    /// appearance setting (#235 R2), so the CLI and both browser renderers — none
    /// of which ever touched `background_mut()` — can draw it. Absent the option
    /// the slot stays empty, which is what keeps existing streams byte-identical.
    #[test]
    fn from_draws_carries_the_environment_background_from_the_options() {
        let draws = [ResolvedDraw {
            mesh_id: mesh_id(0),
            model: Matrix4::IDENTITY,
            selection: DrawSelection::INHERIT,
        }];
        let sky = EnvironmentBackground {
            exposure: 0.75,
            blur: 0.5,
            tonemap: Tonemap::Aces,
        };

        let scene = Scene::from_draws(
            &draws,
            &crate::RenderOptions {
                env_background: Some(sky),
                ..Default::default()
            },
            None,
        );
        assert_eq!(scene.background().environment, Some(sky));
        assert_eq!(scene.background().frame, None);

        // Default ⇒ no sky: every stream that never asks for one is unchanged.
        let scene = Scene::from_draws(&draws, &crate::RenderOptions::default(), None);
        assert_eq!(scene.background().environment, None);
    }

    #[test]
    fn build_scene_per_draw_mode_overrides_default() {
        // Per-draw draw_mode (#79 slice): a draw with an explicit `mode` renders
        // in THAT mode regardless of the scene default, while `mode: None`
        // inherits the default. This lets one frame mix, e.g., a textured mesh
        // and a wireframe placement quad (the two-stage cornellbox scene).
        let model = Matrix4::IDENTITY;
        let draws = [
            ResolvedDraw {
                mesh_id: mesh_id(0),
                model,
                selection: DrawSelection::INHERIT,
            }, // inherits the scene default
            ResolvedDraw {
                mesh_id: mesh_id(1),
                model,
                selection: DrawSelection::Mesh(Some(RenderMode::Wireframe)),
            }, // overrides
            ResolvedDraw {
                mesh_id: mesh_id(2),
                model,
                selection: DrawSelection::Mesh(Some(RenderMode::Textured)),
            }, // overrides
        ];

        let mesh_modes = |scene: &[DrawableObject]| -> Vec<RenderMode> {
            scene
                .iter()
                .filter_map(|o| match o.primitive() {
                    Primitive::Mesh { mode, .. } => Some(mode),
                    _ => None,
                })
                .collect()
        };

        // Filled default: only the `None` draw is Filled; the overrides stand.
        let scene = build_scene(
            &draws,
            &RenderOptions {
                mode: RenderMode::Filled,
                ..Default::default()
            },
            None,
        );
        assert_eq!(
            mesh_modes(scene.objects()),
            vec![
                RenderMode::Filled,
                RenderMode::Wireframe,
                RenderMode::Textured
            ],
            "None inherits the scene default (Filled); Some(..) overrides it"
        );

        // Wireframe default: only the `None` draw changes (now Wireframe); the two
        // explicit overrides are unaffected — the override is per draw, not global.
        let scene = build_scene(
            &draws,
            &RenderOptions {
                mode: RenderMode::Wireframe,
                ..Default::default()
            },
            None,
        );
        assert_eq!(
            mesh_modes(scene.objects()),
            vec![
                RenderMode::Wireframe,
                RenderMode::Wireframe,
                RenderMode::Textured
            ],
        );
    }

    #[test]
    fn build_scene_local_axes_one_gizmo_per_draw_at_its_own_model() {
        // --axes-local (#77 slice) overlays a coordinate gizmo at EACH drawn
        // object's own local frame (its `model`), distinct from the single
        // world-origin gizmo of --axes. With two draws (e.g. a placed bunny and
        // its placement quad) the scene carries TWO local gizmos (one per draw
        // model) plus, when --axes is also set, ONE world gizmo at the identity —
        // the "two axes" a placed-mesh-on-quad frame shows.
        let model_a = Matrix4::from_translation(crate::math::Vector3::new(1.0, 0.0, 0.0)); // distinct translation (col-major tx)
        let model_b = Matrix4::from_translation(crate::math::Vector3::new(5.0, 0.0, 0.0));
        let draws = [
            ResolvedDraw {
                mesh_id: mesh_id(0),
                model: model_a,
                selection: DrawSelection::INHERIT,
            },
            ResolvedDraw {
                mesh_id: mesh_id(1),
                model: model_b,
                selection: DrawSelection::INHERIT,
            },
        ];

        let axes_models = |scene: &[DrawableObject]| -> Vec<Matrix4> {
            scene
                .iter()
                .filter_map(|o| match o.primitive() {
                    Primitive::CoordinateAxes => Some(o.model()),
                    _ => None,
                })
                .collect()
        };

        // Everything on: frame + aabb + world axes + local axes.
        let scene = build_scene(
            &draws,
            &RenderOptions {
                mode: RenderMode::Filled,
                show_aabb: true,
                show_axes: true,
                show_local_axes: true,
                ..Default::default()
            },
            Some(FrameFit::Cover), // background frame plane
        );

        // Counts: 2 meshes + 2 aabb + 2 local axes + 1 world axis = 7 objects; the
        // frame plane is a background setting, not an object (#204).
        assert_eq!(scene.objects().len(), 7, "scene = {scene:?}");
        assert_eq!(scene.background().frame, Some(FrameFit::Cover));
        // Order: Mesh×2, AabbBox×2, CoordinateAxes(local)×2, CoordinateAxes(world).
        assert!(matches!(
            scene.objects()[0].primitive(),
            Primitive::Mesh { .. }
        ));
        assert!(matches!(
            scene.objects()[1].primitive(),
            Primitive::Mesh { .. }
        ));
        assert!(matches!(
            scene.objects()[2].primitive(),
            Primitive::AabbBox { .. }
        ));
        assert!(matches!(
            scene.objects()[3].primitive(),
            Primitive::AabbBox { .. }
        ));

        // The local gizmos carry each draw's own model (in draw order); the world
        // gizmo is last, at the identity (origin).
        assert_eq!(
            axes_models(scene.objects()),
            vec![model_a, model_b, Matrix4::IDENTITY],
            "two local gizmos (per draw model) then one world gizmo at identity"
        );

        // --axes-local WITHOUT --axes ⇒ only the per-draw local gizmos, no world
        // one (both draw models are non-identity, so this is unambiguous).
        let scene = build_scene(
            &draws,
            &RenderOptions {
                mode: RenderMode::Filled,
                show_local_axes: true,
                ..Default::default()
            },
            None,
        );
        assert_eq!(
            axes_models(scene.objects()),
            vec![model_a, model_b],
            "local gizmos only; no extra world-origin gizmo"
        );
    }

    #[test]
    fn build_scene_local_grid_one_per_wireframe_draw_at_its_own_model() {
        // --grid-local (PlaneGrid slice) overlays a coordinate-plane grid at each
        // *wireframe* drawn object's own frame (its `model`), on the requested
        // plane — the lattice twin of --axes-local, but scoped to wireframe draws
        // (the placement quad) so a filled/textured content mesh gets no grid. For
        // the FIBA quad-only scene this is one Xy grid over the quad's surface.
        let model_a = Matrix4::from_translation(crate::math::Vector3::new(2.0, 0.0, 0.0)); // distinct translations (col-major tx)
        let model_b = Matrix4::from_translation(crate::math::Vector3::new(7.0, 0.0, 0.0));
        let draws = [
            ResolvedDraw {
                mesh_id: mesh_id(0),
                model: model_a,
                selection: DrawSelection::INHERIT,
            },
            ResolvedDraw {
                mesh_id: mesh_id(1),
                model: model_b,
                selection: DrawSelection::INHERIT,
            },
        ];

        let grids = |scene: &[DrawableObject]| -> Vec<(GridPlane, Matrix4)> {
            scene
                .iter()
                .filter_map(|o| match o.primitive() {
                    Primitive::PlaneGrid { plane } => Some((plane, o.model())),
                    _ => None,
                })
                .collect()
        };

        // None ⇒ no grid at all (byte-identical to the pre-grid scene).
        let scene = build_scene(
            &draws,
            &RenderOptions {
                mode: RenderMode::Wireframe,
                ..Default::default()
            },
            None,
        );
        assert!(
            grids(scene.objects()).is_empty(),
            "no grid when local_grid is None"
        );

        // Global wireframe mode ⇒ both draws are wireframe ⇒ one PlaneGrid per
        // draw, on that plane, at the draw's model.
        let scene = build_scene(
            &draws,
            &RenderOptions {
                mode: RenderMode::Wireframe,
                show_local_grid: Some(GridPlane::Xy),
                ..Default::default()
            },
            None,
        );
        assert_eq!(
            grids(scene.objects()),
            vec![(GridPlane::Xy, model_a), (GridPlane::Xy, model_b)],
            "one Xy grid per wireframe draw at its own model"
        );

        // The plane is honored (Yz here) and grids sit after the meshes.
        let scene = build_scene(
            &draws,
            &RenderOptions {
                mode: RenderMode::Wireframe,
                show_local_grid: Some(GridPlane::Yz),
                ..Default::default()
            },
            None,
        );
        assert!(matches!(
            scene.objects()[0].primitive(),
            Primitive::Mesh { .. }
        ));
        assert!(matches!(
            scene.objects()[1].primitive(),
            Primitive::Mesh { .. }
        ));
        assert_eq!(
            grids(scene.objects()),
            vec![(GridPlane::Yz, model_a), (GridPlane::Yz, model_b)],
        );

        // Mixed scene (bunny + quad): only the wireframe quad (draw b) gets the
        // grid; the filled/textured content mesh (draw a) does not.
        let mixed = [
            ResolvedDraw {
                mesh_id: mesh_id(0),
                model: model_a,
                selection: DrawSelection::Mesh(Some(RenderMode::Textured)),
            },
            ResolvedDraw {
                mesh_id: mesh_id(1),
                model: model_b,
                selection: DrawSelection::Mesh(Some(RenderMode::Wireframe)),
            },
        ];
        let scene = build_scene(
            &mixed,
            &RenderOptions {
                mode: RenderMode::Filled,
                show_local_grid: Some(GridPlane::Xy),
                ..Default::default()
            },
            None,
        );
        assert_eq!(
            grids(scene.objects()),
            vec![(GridPlane::Xy, model_b)],
            "only the wireframe quad draw gets a grid, not the textured mesh"
        );
    }

    #[test]
    fn build_scene_grid_mesh_scopes_grid_to_the_placement_quad_only() {
        // `grid_mesh = Some(id)` narrows the --grid-local overlay to draws of that
        // mesh only. This is the wireframe-reveal case (#114): a *content* mesh
        // (the can, mesh 0) is drawn wireframe alongside the placement quad (mesh
        // 1), so the plain "all wireframe draws" scoping would lay a stray floor
        // grid under every can. Pinning `grid_mesh = Some(1)` keeps exactly one
        // grid — under the quad — while the cans stay wireframe with no grid.
        let model_can = Matrix4::from_translation(crate::math::Vector3::new(3.0, 0.0, 0.0));
        let model_quad = Matrix4::from_translation(crate::math::Vector3::new(9.0, 0.0, 0.0));
        // Two cans (mesh 0) + one placement quad (mesh 1), all wireframe.
        let draws = [
            ResolvedDraw {
                mesh_id: mesh_id(0),
                model: model_can,
                selection: DrawSelection::Mesh(Some(RenderMode::Wireframe)),
            },
            ResolvedDraw {
                mesh_id: mesh_id(0),
                model: Matrix4::IDENTITY,
                selection: DrawSelection::Mesh(Some(RenderMode::Wireframe)),
            },
            ResolvedDraw {
                mesh_id: mesh_id(1),
                model: model_quad,
                selection: DrawSelection::Mesh(Some(RenderMode::Wireframe)),
            },
        ];

        let grids = |scene: &[DrawableObject]| -> Vec<(GridPlane, Matrix4)> {
            scene
                .iter()
                .filter_map(|o| match o.primitive() {
                    Primitive::PlaneGrid { plane } => Some((plane, o.model())),
                    _ => None,
                })
                .collect()
        };

        // Without a mesh filter, every wireframe draw gets a grid (3 here) — the
        // very over-emission #110 fixes.
        let scene = build_scene(
            &draws,
            &RenderOptions {
                mode: RenderMode::Filled,
                show_local_grid: Some(GridPlane::Xy),
                ..Default::default()
            },
            None,
        );
        assert_eq!(
            grids(scene.objects()).len(),
            3,
            "unscoped grid lands on every wireframe draw"
        );

        // Scoped to the placement quad's mesh (id 1) ⇒ exactly one grid, at the
        // quad's model — no grid under either can.
        let scene = build_scene(
            &draws,
            &RenderOptions {
                mode: RenderMode::Filled,
                show_local_grid: Some(GridPlane::Xy),
                show_local_grid_mesh: Some(mesh_id(1)),
                ..Default::default()
            },
            None,
        );
        assert_eq!(
            grids(scene.objects()),
            vec![(GridPlane::Xy, model_quad)],
            "grid_mesh = Some(1) lays exactly one grid, under the placement quad only"
        );

        // A mesh filter with no matching draw ⇒ no grid at all.
        let scene = build_scene(
            &draws,
            &RenderOptions {
                mode: RenderMode::Filled,
                show_local_grid: Some(GridPlane::Xy),
                show_local_grid_mesh: Some(mesh_id(7)),
                ..Default::default()
            },
            None,
        );
        assert!(
            grids(scene.objects()).is_empty(),
            "grid_mesh naming an absent mesh yields no grid"
        );
    }

    #[test]
    fn build_scene_shadow_draw_becomes_blob_shadow_not_mesh() {
        // A per-draw mode "shadow" lifts that draw's model into a BlobShadow
        // grounding blob (not a Mesh), and it carries no AABB / axes gizmo even
        // when those overlays are on. A mixed FIBA-style scene [shadow, bunny,
        // quad] must yield exactly one BlobShadow at the shadow draw's model.
        let shadow_m = Matrix4::from_translation(crate::math::Vector3::new(3.0, 0.0, 0.0)); // distinct col-major tx
        let bunny_m = Matrix4::from_translation(crate::math::Vector3::new(4.0, 0.0, 0.0));
        let quad_m = Matrix4::from_translation(crate::math::Vector3::new(5.0, 0.0, 0.0));
        let draws = [
            ResolvedDraw {
                mesh_id: mesh_id(0),
                model: shadow_m,
                selection: DrawSelection::Shadow,
            },
            ResolvedDraw {
                mesh_id: mesh_id(0),
                model: bunny_m,
                selection: DrawSelection::Mesh(Some(RenderMode::Textured)),
            },
            ResolvedDraw {
                mesh_id: mesh_id(1),
                model: quad_m,
                selection: DrawSelection::Mesh(Some(RenderMode::Wireframe)),
            },
        ];

        // aabb + local axes on: the shadow draw contributes a BlobShadow but no
        // Mesh / AabbBox / CoordinateAxes.
        let scene = build_scene(
            &draws,
            &RenderOptions {
                mode: RenderMode::Filled,
                show_aabb: true,
                show_local_axes: true,
                ..Default::default()
            },
            None,
        );

        let blobs: Vec<Matrix4> = scene
            .objects()
            .iter()
            .filter_map(|o| match o.primitive() {
                Primitive::BlobShadow => Some(o.model()),
                _ => None,
            })
            .collect();
        assert_eq!(
            blobs,
            vec![shadow_m],
            "exactly one BlobShadow, at the shadow draw's model"
        );

        // The shadow draw must NOT produce a Mesh, and only the two non-shadow
        // draws get AABB boxes / local axes gizmos.
        let meshes = scene
            .objects()
            .iter()
            .filter(|o| matches!(o.primitive(), Primitive::Mesh { .. }))
            .count();
        let aabbs = scene
            .objects()
            .iter()
            .filter(|o| matches!(o.primitive(), Primitive::AabbBox { .. }))
            .count();
        let axes = scene
            .objects()
            .iter()
            .filter(|o| matches!(o.primitive(), Primitive::CoordinateAxes))
            .count();
        assert_eq!(meshes, 2, "shadow draw is not a Mesh; bunny + quad are");
        assert_eq!(aabbs, 2, "no AABB for the shadow draw");
        assert_eq!(axes, 2, "no local-axes gizmo for the shadow draw");
    }

    #[test]
    fn build_scene_maps_draws_and_overlays_in_bucket_order() {
        // #41: the draw list + mode/overlay flags become an ordered `Scene` of
        // `DrawableObject`s — one Mesh per draw (in `mode`), then one AabbBox per
        // draw when enabled, then a single origin CoordinateAxes when enabled.
        let a = Matrix4::from_cols_array(&[1.0f32; 16]);
        let b = Matrix4::from_cols_array(&[2.0f32; 16]);
        let draws = [
            ResolvedDraw {
                mesh_id: mesh_id(0),
                model: a,
                selection: DrawSelection::INHERIT,
            },
            ResolvedDraw {
                mesh_id: mesh_id(1),
                model: b,
                selection: DrawSelection::INHERIT,
            },
        ];

        // Plain filled: exactly one Mesh drawable per draw, no gizmos.
        assert_eq!(
            build_scene(
                &draws,
                &RenderOptions {
                    mode: RenderMode::Filled,
                    ..Default::default()
                },
                None
            )
            .objects(),
            [
                DrawableObject::mesh(mesh_id(0), a, RenderMode::Filled),
                DrawableObject::mesh(mesh_id(1), b, RenderMode::Filled),
            ]
        );

        // Wireframe propagates the mode to every mesh drawable.
        assert_eq!(
            build_scene(
                &draws,
                &RenderOptions {
                    mode: RenderMode::Wireframe,
                    ..Default::default()
                },
                None
            )
            .objects(),
            [
                DrawableObject::mesh(mesh_id(0), a, RenderMode::Wireframe),
                DrawableObject::mesh(mesh_id(1), b, RenderMode::Wireframe),
            ]
        );

        // Both overlays: meshes, then a tracking box per draw, then one gizmo.
        assert_eq!(
            build_scene(
                &draws,
                &RenderOptions {
                    mode: RenderMode::Filled,
                    show_aabb: true,
                    show_axes: true,
                    ..Default::default()
                },
                None
            )
            .objects(),
            [
                DrawableObject::mesh(mesh_id(0), a, RenderMode::Filled),
                DrawableObject::mesh(mesh_id(1), b, RenderMode::Filled),
                DrawableObject::aabb_box(mesh_id(0), a),
                DrawableObject::aabb_box(mesh_id(1), b),
                DrawableObject::coordinate_axes(Matrix4::IDENTITY),
            ]
        );

        // Local axes: one CoordinateAxes per draw at its own model (in the mesh
        // bucket order, before the world-origin gizmo), each tracking its draw.
        assert_eq!(
            build_scene(
                &draws,
                &RenderOptions {
                    mode: RenderMode::Filled,
                    show_local_axes: true,
                    ..Default::default()
                },
                None
            )
            .objects(),
            [
                DrawableObject::mesh(mesh_id(0), a, RenderMode::Filled),
                DrawableObject::mesh(mesh_id(1), b, RenderMode::Filled),
                DrawableObject::coordinate_axes(a),
                DrawableObject::coordinate_axes(b),
            ]
        );

        // Per-draw mode override: a draw's own `mode` wins over the global one,
        // so one frame can mix (e.g.) a textured mesh with a wireframe overlay.
        let mixed = [
            ResolvedDraw {
                mesh_id: mesh_id(0),
                model: a,
                selection: DrawSelection::INHERIT,
            },
            ResolvedDraw {
                mesh_id: mesh_id(1),
                model: b,
                selection: DrawSelection::Mesh(Some(RenderMode::Wireframe)),
            },
        ];
        assert_eq!(
            build_scene(
                &mixed,
                &RenderOptions {
                    mode: RenderMode::Textured,
                    ..Default::default()
                },
                None
            )
            .objects(),
            [
                DrawableObject::mesh(mesh_id(0), a, RenderMode::Textured),
                DrawableObject::mesh(mesh_id(1), b, RenderMode::Wireframe),
            ]
        );
    }

    #[test]
    fn try_from_frame_rejects_a_draw_naming_a_mesh_the_stream_never_sent() {
        let table = MeshTable::new(vec![crate::Mesh::hello_triangle(); 2]).unwrap();
        let frame = crate::DecodedFrame {
            params: crate::FrameParams::IDENTITY,
            draws: Some(vec![Draw {
                mesh_id: MeshTableIndex::new(3),
                model: Matrix4::IDENTITY,
                selection: crate::DrawSelection::INHERIT,
            }]),
            frame_ref: None,
            frame_id: None,
        };
        assert_eq!(
            Scene::try_from_frame(&frame, &table, &RenderOptions::default(), None),
            Err(SceneError::MeshIndexOutOfRange {
                mesh_id: MeshTableIndex::new(3),
                mesh_count: 2
            })
        );
    }

    #[test]
    fn try_from_frame_defaults_an_absent_draw_list_to_mesh_zero() {
        let table = MeshTable::new(vec![crate::Mesh::hello_triangle()]).unwrap();
        // An absent wire draw list is the legacy single-object stream: one
        // instance of mesh 0 placed by the frame's own model.
        let frame = crate::DecodedFrame {
            params: crate::FrameParams {
                model: Some(
                    Matrix4::from_translation(crate::math::Vector3::new(2.0, 3.0, 4.0))
                        .to_cols_array(),
                ),
                ..crate::FrameParams::IDENTITY
            },
            draws: None,
            frame_ref: None,
            frame_id: None,
        };
        let scene = Scene::try_from_frame(&frame, &table, &RenderOptions::default(), None).unwrap();
        assert_eq!(
            scene.objects(),
            [DrawableObject::mesh(
                table.id(MeshTableIndex::new(0)).unwrap(),
                frame.params.model_matrix(),
                RenderMode::Filled,
            )],
        );
        // An explicit empty list draws no meshes at all — the background plate only.
        let background_only = crate::DecodedFrame {
            draws: Some(Vec::new()),
            ..frame
        };
        let scene =
            Scene::try_from_frame(&background_only, &table, &RenderOptions::default(), None)
                .unwrap();
        assert!(scene.objects().is_empty());
    }

    #[test]
    fn resolve_draws_preserves_registration_order_models_and_selections() {
        let table = MeshTable::new(vec![crate::Mesh::hello_triangle(); 2]).unwrap();
        let draws = [
            Draw {
                mesh_id: MeshTableIndex::new(1),
                model: Matrix4::from_translation(crate::math::Vector3::new(2.0, 0.0, 0.0)),
                selection: DrawSelection::Shadow,
            },
            Draw {
                mesh_id: MeshTableIndex::new(0),
                model: Matrix4::IDENTITY,
                selection: DrawSelection::Mesh(Some(RenderMode::Wireframe)),
            },
            Draw {
                mesh_id: MeshTableIndex::new(1),
                model: Matrix4::IDENTITY,
                selection: DrawSelection::INHERIT,
            },
        ];
        let resolved = Scene::resolve_draws(&draws, &table).unwrap();
        for (wire, registered) in draws.iter().zip(&resolved) {
            assert_eq!(Some(registered.mesh_id), table.id(wire.mesh_id));
            assert_eq!(registered.model, wire.model);
            assert_eq!(registered.selection, wire.selection);
        }
        assert_eq!(resolved[0].mesh_id, resolved[2].mesh_id);
        assert_eq!(
            Scene::resolve_draws(&draws, &table.clone()).unwrap(),
            resolved,
        );

        let independent = MeshTable::new(vec![crate::Mesh::hello_triangle(); 2]).unwrap();
        let other = Scene::resolve_draws(&draws, &independent).unwrap();
        for (first, second) in resolved.iter().zip(&other) {
            assert_ne!(first.mesh_id, second.mesh_id);
            assert_eq!(first.model, second.model);
            assert_eq!(first.selection, second.selection);
        }
    }

    #[test]
    fn wire_shadow_rows_are_validated_before_becoming_shared_geometry() {
        let table = MeshTable::new(vec![crate::Mesh::hello_triangle()]).unwrap();
        let mut frame = crate::DecodedFrame {
            params: crate::FrameParams::IDENTITY,
            draws: Some(vec![Draw {
                mesh_id: MeshTableIndex::new(1),
                model: Matrix4::IDENTITY,
                selection: DrawSelection::Shadow,
            }]),
            frame_ref: None,
            frame_id: None,
        };
        assert_eq!(
            Scene::try_from_frame(&frame, &table, &RenderOptions::default(), None),
            Err(SceneError::MeshIndexOutOfRange {
                mesh_id: MeshTableIndex::new(1),
                mesh_count: 1,
            }),
        );
        frame.draws.as_mut().unwrap()[0].mesh_id = MeshTableIndex::new(0);
        let scene = Scene::try_from_frame(&frame, &table, &RenderOptions::default(), None).unwrap();
        assert_eq!(
            scene.objects(),
            [DrawableObject::blob_shadow(Matrix4::IDENTITY)]
        );
    }

    #[test]
    fn empty_registration_rejects_implicit_zero_but_accepts_explicit_empty() {
        let table = MeshTable::new(Vec::new()).unwrap();
        let mut frame = crate::DecodedFrame {
            params: crate::FrameParams::IDENTITY,
            draws: None,
            frame_ref: None,
            frame_id: None,
        };
        assert_eq!(
            Scene::try_from_frame(&frame, &table, &RenderOptions::default(), None),
            Err(SceneError::MeshIndexOutOfRange {
                mesh_id: MeshTableIndex::new(0),
                mesh_count: 0,
            }),
        );
        frame.draws = Some(Vec::new());
        let scene = Scene::try_from_frame(
            &frame,
            &table,
            &RenderOptions::default(),
            Some(FrameFit::Stretch),
        )
        .unwrap();
        assert!(scene.objects().is_empty());
        assert_eq!(scene.background().frame, Some(FrameFit::Stretch));
    }

    #[test]
    fn wire_grid_selector_resolves_against_the_same_registration_as_draws() {
        let table = MeshTable::new(vec![crate::Mesh::hello_triangle(); 2]).unwrap();
        let model = Matrix4::from_translation(crate::math::Vector3::new(3.0, 0.0, 0.0));
        let frame = crate::DecodedFrame {
            params: crate::FrameParams::IDENTITY,
            draws: Some(vec![
                Draw {
                    mesh_id: MeshTableIndex::new(0),
                    model: Matrix4::IDENTITY,
                    selection: DrawSelection::Shadow,
                },
                Draw {
                    mesh_id: MeshTableIndex::new(1),
                    model,
                    selection: DrawSelection::INHERIT,
                },
                Draw {
                    mesh_id: MeshTableIndex::new(0),
                    model: Matrix4::IDENTITY,
                    selection: DrawSelection::INHERIT,
                },
            ]),
            frame_ref: None,
            frame_id: None,
        };
        let options = RenderOptions {
            mode: RenderMode::Wireframe,
            show_aabb: true,
            show_axes: true,
            show_local_axes: true,
            show_local_grid: Some(GridPlane::Xy),
            show_local_grid_mesh: Some(MeshTableIndex::new(1)),
            show_world_grid: Some(GridPlane::Xz),
            show_object_grid: Some(GridPlane::Yz),
            selected: Some(1),
            pbr: Some(crate::PbrConfig::default()),
            ..Default::default()
        };
        let scene = Scene::try_from_frame(&frame, &table, &options, Some(FrameFit::Cover)).unwrap();
        let row_zero = table.id(MeshTableIndex::new(0)).unwrap();
        let row_one = table.id(MeshTableIndex::new(1)).unwrap();
        assert_eq!(
            scene.objects(),
            [
                DrawableObject::blob_shadow(Matrix4::IDENTITY),
                DrawableObject::mesh(row_one, model, RenderMode::Wireframe),
                DrawableObject::mesh(row_zero, Matrix4::IDENTITY, RenderMode::Wireframe),
                DrawableObject::aabb_box(row_one, model),
                DrawableObject::aabb_box(row_zero, Matrix4::IDENTITY),
                DrawableObject::plane_grid(GridPlane::Xy, model),
                DrawableObject::coordinate_axes(model),
                DrawableObject::coordinate_axes(Matrix4::IDENTITY),
                DrawableObject::coordinate_axes(Matrix4::IDENTITY),
                DrawableObject::plane_grid(GridPlane::Xz, Matrix4::IDENTITY),
                DrawableObject::plane_grid(GridPlane::Yz, model),
                DrawableObject::plane_grid(GridPlane::Yz, Matrix4::IDENTITY),
                DrawableObject::aabb_box(row_one, model),
            ],
        );
        assert_eq!(scene.background().frame, Some(FrameFit::Cover));
        assert_eq!(scene.lighting(), options.pbr.as_ref().unwrap().lighting);
        assert!(selection_aabb_overlay(
            &Scene::resolve_draws(&frame.resolved_draws(), &table).unwrap(),
            Some(0),
        )
        .is_empty());
    }

    #[test]
    fn wire_grid_selector_rejects_bad_rows_even_without_draws_or_a_grid() {
        let table = MeshTable::new(vec![crate::Mesh::hello_triangle()]).unwrap();
        let frame = crate::DecodedFrame {
            params: crate::FrameParams::IDENTITY,
            draws: Some(Vec::new()),
            frame_ref: None,
            frame_id: None,
        };
        let options = RenderOptions {
            show_local_grid_mesh: Some(MeshTableIndex::new(u32::MAX)),
            ..Default::default()
        };
        assert_eq!(
            Scene::try_from_frame(&frame, &table, &options, None),
            Err(SceneError::MeshIndexOutOfRange {
                mesh_id: MeshTableIndex::new(u32::MAX),
                mesh_count: 1,
            }),
        );
    }
}
