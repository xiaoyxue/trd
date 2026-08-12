//! [`Scene`] — a frame's ordered list of primitives, and the one place it is
//! assembled from a wire draw list.
//!
//! [`Scene::from_draws`] is the entry point every front-end uses; the overlay
//! builders it composes are private, so a front-end cannot assemble half a
//! scene. That is what keeps native and browser rendering the same frame from
//! the same inputs (#180).

use super::{Draw, DrawableObject, FrameFit, GridPlane, RenderMode};
use crate::math::Matrix4;

/// A frame's ordered list of [`DrawableObject`]s the renderer walks and encodes
/// under the one shared camera `P·V` uniform. The wire authors the mesh draws
/// (the protocol draw list); the core adds gizmo drawables (axes, AABB boxes).
/// A single-mesh frame is the degenerate one-element scene — the renderer always
/// iterates a `Scene`, with no single-object special case.
///
/// A struct rather than a `Vec` alias (#203) so assembly can live on it:
/// [`Scene::from_draws`] is the one entry point every front-end uses, which is
/// what keeps them all rendering the same scene from the same inputs. It
/// [`Deref`]s to `[DrawableObject]`, so it reads like the slice it wraps.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Scene {
    objects: Vec<DrawableObject>,
}

impl Scene {
    /// An empty scene.
    pub fn new() -> Self {
        Self::default()
    }

    /// An empty scene with room for `capacity` objects.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            objects: Vec::with_capacity(capacity),
        }
    }

    /// Appends one primitive. Order is draw order within a kind, so callers push
    /// in the order the renderer's buckets expect.
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

    /// The **one** place a frame's scene is assembled from a wire draw list plus
    /// appearance options.
    ///
    /// Every front-end used to do this itself — the headless `Renderer` from nine
    /// retained overlay flags, `trd-gui` through setters, the browser renderers
    /// from their own booleans — which is how native and web overlay assembly
    /// drifted apart. Assembling here means all of them get the same scene from
    /// the same inputs (#180), and owning it on `Scene` means the overlay
    /// builders need not be public at all (#203).
    ///
    /// `frame` prepends a background [`DrawableObject::FramePlane`] (#63) when
    /// the caller has a frame texture bound.
    pub fn from_draws(
        draws: &[Draw],
        options: &crate::RenderOptions,
        frame: Option<FrameFit>,
    ) -> Self {
        let mut scene = build_scene(
            draws,
            options.mode,
            options.show_aabb,
            options.show_axes,
            options.show_local_axes,
            options.show_local_grid,
            options.show_local_grid_mesh,
            frame,
        );
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
        scene
    }
}

impl std::ops::Deref for Scene {
    type Target = [DrawableObject];

    fn deref(&self) -> &Self::Target {
        &self.objects
    }
}

impl FromIterator<DrawableObject> for Scene {
    fn from_iter<T: IntoIterator<Item = DrawableObject>>(iter: T) -> Self {
        Self {
            objects: iter.into_iter().collect(),
        }
    }
}

impl From<Vec<DrawableObject>> for Scene {
    fn from(objects: Vec<DrawableObject>) -> Self {
        Self { objects }
    }
}

/// Builds a per-frame [`Scene`] from a wire `draws` list plus the render `mode`
/// and overlay flags. When `frame` is `Some`, a background
/// [`DrawableObject::FramePlane`] is pushed **first** so the mesh scene
/// composites on top of it. Each [`Draw`] becomes one [`DrawableObject::Mesh`]
/// in the draw's own [`Draw::mode`] when set, else the passed `mode`; with
/// `show_aabb`, each also emits a tracking
/// [`DrawableObject::AabbBox`]; with `local_grid = Some(plane)`, each
/// **wireframe-mode** draw emits a [`DrawableObject::PlaneGrid`] on `plane` at
/// **its own `model`** (a coordinate-plane lattice in that object's local frame —
/// e.g. the `Xy` grid on a #77 placement quad's surface; scoped to wireframe
/// draws so a filled/textured content mesh whose local `Xy` is vertical gets no
/// stray grid wall). `grid_mesh = Some(id)` narrows the grid further to draws of
/// **that** mesh only (#110 follow-up): when the *content* mesh is also drawn
/// wireframe (e.g. a wireframe-reveal intro over a placement quad), the grid
/// would otherwise land on every wireframe object; pin it to the placement
/// quad's `mesh_id` so exactly one floor grid is laid. `grid_mesh = None` keeps
/// the "all wireframe draws" behaviour. With `show_axes`, one **world-origin**
/// [`DrawableObject::CoordinateAxes`] is appended; with `show_local_axes`, each
/// draw also emits a [`DrawableObject::CoordinateAxes`] at **its own `model`** —
/// i.e. that object's *local* coordinate frame (its model-space X/Y/Z axes as
/// placed, e.g. #77's `(e1,e2,e3)` quad frame). The order (frame plane, then all
/// meshes, then all boxes, then per-draw grids, then per-draw local axes, then
/// the world-origin axes) matches the renderer's draw buckets so output is
/// pixel-identical to the pre-scene, flag-driven path.
///
/// Private: [`Scene::from_draws`] is the entry point, so a front-end cannot
/// assemble half a scene.
#[allow(clippy::too_many_arguments)]
fn build_scene(
    draws: &[Draw],
    mode: RenderMode,
    show_aabb: bool,
    show_axes: bool,
    show_local_axes: bool,
    local_grid: Option<GridPlane>,
    grid_mesh: Option<u32>,
    frame: Option<FrameFit>,
) -> Scene {
    let mut scene = Scene::with_capacity(
        draws.len()
            * (1 + usize::from(show_aabb)
                + usize::from(show_local_axes)
                + usize::from(local_grid.is_some()))
            + usize::from(show_axes)
            + usize::from(frame.is_some()),
    );
    if let Some(fit) = frame {
        scene.push(DrawableObject::FramePlane { fit });
    }
    // The one place a `DrawSelection` is resolved: past here the scene holds
    // primitives, and nothing downstream needs to know a shadow ever existed.
    for draw in draws {
        scene.push(match draw.selection.mesh_mode(mode) {
            Some(mode) => DrawableObject::Mesh {
                mesh_id: draw.mesh_id,
                model: draw.model,
                mode,
            },
            None => DrawableObject::BlobShadow { model: draw.model },
        });
    }
    if show_aabb {
        for draw in draws.iter().filter(|d| d.selection.is_mesh()) {
            scene.push(DrawableObject::AabbBox {
                mesh_id: draw.mesh_id,
                model: draw.model,
            });
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
                scene.push(DrawableObject::PlaneGrid {
                    plane,
                    model: draw.model,
                });
            }
        }
    }
    if show_local_axes {
        // A shadow blob is a floor decal, not a placed object whose local frame
        // warrants an axes gizmo.
        for draw in draws.iter().filter(|d| d.selection.is_mesh()) {
            scene.push(DrawableObject::CoordinateAxes { model: draw.model });
        }
    }
    if show_axes {
        scene.push(DrawableObject::CoordinateAxes {
            model: Matrix4::IDENTITY.to_cols_array(),
        });
    }
    scene
}

/// Builds **plane-grid overlay** drawables independent of [`build_scene`]'s
/// wireframe-scoped `local_grid` (#114): a `world_grid` lays one
/// [`DrawableObject::PlaneGrid`] at the **world origin** (identity model — the
/// world floor, analogous to `show_axes`), and an `object_grid` lays a
/// `PlaneGrid` at **each drawn object's own model** frame (analogous to
/// `show_local_axes`), ungated by render mode. Shadow draws are skipped (a blob
/// decal has no frame to grid). Appended to a scene by front-ends that want a
/// grid under a *filled/textured/PBR* object (e.g. the interactive `trd-gui`
/// overlays) without the #77 wireframe-quad gating; `None`/`None` yields an empty
/// list, so callers that don't opt in are byte-identical.
pub(crate) fn plane_grid_overlays(
    draws: &[Draw],
    world_grid: Option<GridPlane>,
    object_grid: Option<GridPlane>,
) -> Vec<DrawableObject> {
    let mut grids = Vec::new();
    if let Some(plane) = world_grid {
        grids.push(DrawableObject::PlaneGrid {
            plane,
            model: Matrix4::IDENTITY.to_cols_array(),
        });
    }
    if let Some(plane) = object_grid {
        for draw in draws.iter().filter(|d| d.selection.is_mesh()) {
            grids.push(DrawableObject::PlaneGrid {
                plane,
                model: draw.model,
            });
        }
    }
    grids
}

/// A **selection-highlight** overlay (#141): the [`DrawableObject::AabbBox`] of a
/// single object — the `selected` 0-based index into `draws` — so *only* that
/// object's bounding box is drawn (unlike the global "show all AABBs" toggle).
/// `None`, an out-of-range index, or a `Shadow` draw yields an empty list, so a
/// caller that doesn't opt in is byte-identical. Appended to the scene by
/// front-ends that highlight a clicked object.
pub(crate) fn selection_aabb_overlay(draws: &[Draw], selected: Option<u32>) -> Vec<DrawableObject> {
    let Some(draw) = selected.and_then(|i| draws.get(i as usize)) else {
        return Vec::new();
    };
    if !draw.selection.is_mesh() {
        return Vec::new();
    }
    vec![DrawableObject::AabbBox {
        mesh_id: draw.mesh_id,
        model: draw.model,
    }]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::visual::DrawSelection;

    #[test]
    fn plane_grid_overlays_place_world_and_object_grids() {
        let draws = [
            Draw {
                mesh_id: 0,
                model: Matrix4::from_translation(crate::math::Vector3::new(1.0, 0.0, 0.0))
                    .to_cols_array(),
                selection: DrawSelection::INHERIT,
            },
            Draw {
                mesh_id: 1,
                model: Matrix4::IDENTITY.to_cols_array(),
                selection: DrawSelection::Shadow,
            },
        ];
        // Neither grid ⇒ empty (opt-in only, byte-identical for non-users).
        assert!(plane_grid_overlays(&draws, None, None).is_empty());

        // World grid ⇒ exactly one identity-model grid on the requested plane.
        let world = plane_grid_overlays(&draws, Some(GridPlane::Xz), None);
        assert_eq!(world.len(), 1);
        assert!(matches!(
            world[0],
            DrawableObject::PlaneGrid {
                plane: GridPlane::Xz,
                model,
            } if model == Matrix4::IDENTITY.to_cols_array()
        ));

        // Object grid ⇒ one grid per *non-shadow* draw, at that draw's model.
        let object = plane_grid_overlays(&draws, None, Some(GridPlane::Xz));
        assert_eq!(object.len(), 1);
        assert!(matches!(
            object[0],
            DrawableObject::PlaneGrid {
                plane: GridPlane::Xz,
                model,
            } if model == draws[0].model
        ));
    }

    #[test]
    fn build_scene_prepends_frame_plane_first() {
        let draws = [Draw {
            mesh_id: 0,
            model: Matrix4::IDENTITY.to_cols_array(),
            selection: DrawSelection::INHERIT,
        }];

        // No frame ⇒ no FramePlane in the scene (byte-identical to the pre-0.0.5
        // scene).
        let scene = build_scene(
            &draws,
            RenderMode::Filled,
            true,
            true,
            false,
            None,
            None,
            None,
        );
        assert!(
            !scene
                .iter()
                .any(|o| matches!(o, DrawableObject::FramePlane { .. })),
            "no frame ⇒ no FramePlane, got {scene:?}"
        );

        // Some(fit) ⇒ exactly one FramePlane, pushed FIRST (before every mesh /
        // aabb / axes), so it composites under the scene.
        let scene = build_scene(
            &draws,
            RenderMode::Filled,
            true,
            true,
            false,
            None,
            None,
            Some(FrameFit::Cover),
        );
        assert!(
            matches!(
                scene[0],
                DrawableObject::FramePlane {
                    fit: FrameFit::Cover
                }
            ),
            "FramePlane must be first, got {:?}",
            scene[0]
        );
        assert_eq!(
            scene
                .iter()
                .filter(|o| matches!(o, DrawableObject::FramePlane { .. }))
                .count(),
            1,
            "exactly one FramePlane"
        );
        // The remainder keeps the mesh → aabb → axes order.
        assert!(matches!(scene[1], DrawableObject::Mesh { .. }));
        assert!(matches!(scene[2], DrawableObject::AabbBox { .. }));
        assert!(matches!(scene[3], DrawableObject::CoordinateAxes { .. }));
    }

    #[test]
    fn build_scene_per_draw_mode_overrides_default() {
        // Per-draw draw_mode (#79 slice): a draw with an explicit `mode` renders
        // in THAT mode regardless of the scene default, while `mode: None`
        // inherits the default. This lets one frame mix, e.g., a textured mesh
        // and a wireframe placement quad (the two-stage cornellbox scene).
        let model = Matrix4::IDENTITY.to_cols_array();
        let draws = [
            Draw {
                mesh_id: 0,
                model,
                selection: DrawSelection::INHERIT,
            }, // inherits the scene default
            Draw {
                mesh_id: 1,
                model,
                selection: DrawSelection::Mesh(Some(RenderMode::Wireframe)),
            }, // overrides
            Draw {
                mesh_id: 2,
                model,
                selection: DrawSelection::Mesh(Some(RenderMode::Textured)),
            }, // overrides
        ];

        let mesh_modes = |scene: &[DrawableObject]| -> Vec<RenderMode> {
            scene
                .iter()
                .filter_map(|o| match o {
                    DrawableObject::Mesh { mode, .. } => Some(*mode),
                    _ => None,
                })
                .collect()
        };

        // Filled default: only the `None` draw is Filled; the overrides stand.
        let scene = build_scene(
            &draws,
            RenderMode::Filled,
            false,
            false,
            false,
            None,
            None,
            None,
        );
        assert_eq!(
            mesh_modes(&scene),
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
            RenderMode::Wireframe,
            false,
            false,
            false,
            None,
            None,
            None,
        );
        assert_eq!(
            mesh_modes(&scene),
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
        let mut model_a = Matrix4::IDENTITY.to_cols_array();
        model_a[12] = 1.0; // distinct translation (col-major tx)
        let mut model_b = Matrix4::IDENTITY.to_cols_array();
        model_b[12] = 5.0;
        let draws = [
            Draw {
                mesh_id: 0,
                model: model_a,
                selection: DrawSelection::INHERIT,
            },
            Draw {
                mesh_id: 1,
                model: model_b,
                selection: DrawSelection::INHERIT,
            },
        ];

        let axes_models = |scene: &[DrawableObject]| -> Vec<[f32; 16]> {
            scene
                .iter()
                .filter_map(|o| match o {
                    DrawableObject::CoordinateAxes { model } => Some(*model),
                    _ => None,
                })
                .collect()
        };

        // Everything on: frame + aabb + world axes + local axes.
        let scene = build_scene(
            &draws,
            RenderMode::Filled,
            true,                  // show_aabb
            true,                  // show_axes (world)
            true,                  // show_local_axes
            None,                  // local_grid
            None,                  // grid_mesh
            Some(FrameFit::Cover), // background frame plane
        );

        // Counts: 1 frame + 2 meshes + 2 aabb + 2 local axes + 1 world axis = 8.
        assert_eq!(scene.len(), 8, "scene = {scene:?}");
        // Order: FramePlane, Mesh×2, AabbBox×2, CoordinateAxes(local)×2, CoordinateAxes(world).
        assert!(matches!(scene[0], DrawableObject::FramePlane { .. }));
        assert!(matches!(scene[1], DrawableObject::Mesh { .. }));
        assert!(matches!(scene[2], DrawableObject::Mesh { .. }));
        assert!(matches!(scene[3], DrawableObject::AabbBox { .. }));
        assert!(matches!(scene[4], DrawableObject::AabbBox { .. }));

        // The local gizmos carry each draw's own model (in draw order); the world
        // gizmo is last, at the identity (origin).
        assert_eq!(
            axes_models(&scene),
            vec![model_a, model_b, Matrix4::IDENTITY.to_cols_array()],
            "two local gizmos (per draw model) then one world gizmo at identity"
        );

        // --axes-local WITHOUT --axes ⇒ only the per-draw local gizmos, no world
        // one (both draw models are non-identity, so this is unambiguous).
        let scene = build_scene(
            &draws,
            RenderMode::Filled,
            false,
            false,
            true,
            None,
            None,
            None,
        );
        assert_eq!(
            axes_models(&scene),
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
        let mut model_a = Matrix4::IDENTITY.to_cols_array();
        model_a[12] = 2.0; // distinct translations (col-major tx)
        let mut model_b = Matrix4::IDENTITY.to_cols_array();
        model_b[12] = 7.0;
        let draws = [
            Draw {
                mesh_id: 0,
                model: model_a,
                selection: DrawSelection::INHERIT,
            },
            Draw {
                mesh_id: 1,
                model: model_b,
                selection: DrawSelection::INHERIT,
            },
        ];

        let grids = |scene: &[DrawableObject]| -> Vec<(GridPlane, [f32; 16])> {
            scene
                .iter()
                .filter_map(|o| match o {
                    DrawableObject::PlaneGrid { plane, model } => Some((*plane, *model)),
                    _ => None,
                })
                .collect()
        };

        // None ⇒ no grid at all (byte-identical to the pre-grid scene).
        let scene = build_scene(
            &draws,
            RenderMode::Wireframe,
            false,
            false,
            false,
            None,
            None,
            None,
        );
        assert!(grids(&scene).is_empty(), "no grid when local_grid is None");

        // Global wireframe mode ⇒ both draws are wireframe ⇒ one PlaneGrid per
        // draw, on that plane, at the draw's model.
        let scene = build_scene(
            &draws,
            RenderMode::Wireframe,
            false,
            false,
            false,
            Some(GridPlane::Xy),
            None,
            None,
        );
        assert_eq!(
            grids(&scene),
            vec![(GridPlane::Xy, model_a), (GridPlane::Xy, model_b)],
            "one Xy grid per wireframe draw at its own model"
        );

        // The plane is honored (Yz here) and grids sit after the meshes.
        let scene = build_scene(
            &draws,
            RenderMode::Wireframe,
            false,
            false,
            false,
            Some(GridPlane::Yz),
            None,
            None,
        );
        assert!(matches!(scene[0], DrawableObject::Mesh { .. }));
        assert!(matches!(scene[1], DrawableObject::Mesh { .. }));
        assert_eq!(
            grids(&scene),
            vec![(GridPlane::Yz, model_a), (GridPlane::Yz, model_b)],
        );

        // Mixed scene (bunny + quad): only the wireframe quad (draw b) gets the
        // grid; the filled/textured content mesh (draw a) does not.
        let mixed = [
            Draw {
                mesh_id: 0,
                model: model_a,
                selection: DrawSelection::Mesh(Some(RenderMode::Textured)),
            },
            Draw {
                mesh_id: 1,
                model: model_b,
                selection: DrawSelection::Mesh(Some(RenderMode::Wireframe)),
            },
        ];
        let scene = build_scene(
            &mixed,
            RenderMode::Filled,
            false,
            false,
            false,
            Some(GridPlane::Xy),
            None,
            None,
        );
        assert_eq!(
            grids(&scene),
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
        let mut model_can = Matrix4::IDENTITY.to_cols_array();
        model_can[12] = 3.0;
        let mut model_quad = Matrix4::IDENTITY.to_cols_array();
        model_quad[12] = 9.0;
        // Two cans (mesh 0) + one placement quad (mesh 1), all wireframe.
        let draws = [
            Draw {
                mesh_id: 0,
                model: model_can,
                selection: DrawSelection::Mesh(Some(RenderMode::Wireframe)),
            },
            Draw {
                mesh_id: 0,
                model: Matrix4::IDENTITY.to_cols_array(),
                selection: DrawSelection::Mesh(Some(RenderMode::Wireframe)),
            },
            Draw {
                mesh_id: 1,
                model: model_quad,
                selection: DrawSelection::Mesh(Some(RenderMode::Wireframe)),
            },
        ];

        let grids = |scene: &[DrawableObject]| -> Vec<(GridPlane, [f32; 16])> {
            scene
                .iter()
                .filter_map(|o| match o {
                    DrawableObject::PlaneGrid { plane, model } => Some((*plane, *model)),
                    _ => None,
                })
                .collect()
        };

        // Without a mesh filter, every wireframe draw gets a grid (3 here) — the
        // very over-emission #110 fixes.
        let scene = build_scene(
            &draws,
            RenderMode::Filled,
            false,
            false,
            false,
            Some(GridPlane::Xy),
            None,
            None,
        );
        assert_eq!(
            grids(&scene).len(),
            3,
            "unscoped grid lands on every wireframe draw"
        );

        // Scoped to the placement quad's mesh (id 1) ⇒ exactly one grid, at the
        // quad's model — no grid under either can.
        let scene = build_scene(
            &draws,
            RenderMode::Filled,
            false,
            false,
            false,
            Some(GridPlane::Xy),
            Some(1),
            None,
        );
        assert_eq!(
            grids(&scene),
            vec![(GridPlane::Xy, model_quad)],
            "grid_mesh = Some(1) lays exactly one grid, under the placement quad only"
        );

        // A mesh filter with no matching draw ⇒ no grid at all.
        let scene = build_scene(
            &draws,
            RenderMode::Filled,
            false,
            false,
            false,
            Some(GridPlane::Xy),
            Some(7),
            None,
        );
        assert!(
            grids(&scene).is_empty(),
            "grid_mesh naming an absent mesh yields no grid"
        );
    }

    #[test]
    fn build_scene_shadow_draw_becomes_blob_shadow_not_mesh() {
        // A per-draw mode "shadow" lifts that draw's model into a BlobShadow
        // grounding blob (not a Mesh), and it carries no AABB / axes gizmo even
        // when those overlays are on. A mixed FIBA-style scene [shadow, bunny,
        // quad] must yield exactly one BlobShadow at the shadow draw's model.
        let mut shadow_m = Matrix4::IDENTITY.to_cols_array();
        shadow_m[12] = 3.0; // distinct col-major tx
        let mut bunny_m = Matrix4::IDENTITY.to_cols_array();
        bunny_m[12] = 4.0;
        let mut quad_m = Matrix4::IDENTITY.to_cols_array();
        quad_m[12] = 5.0;
        let draws = [
            Draw {
                mesh_id: 0,
                model: shadow_m,
                selection: DrawSelection::Shadow,
            },
            Draw {
                mesh_id: 0,
                model: bunny_m,
                selection: DrawSelection::Mesh(Some(RenderMode::Textured)),
            },
            Draw {
                mesh_id: 1,
                model: quad_m,
                selection: DrawSelection::Mesh(Some(RenderMode::Wireframe)),
            },
        ];

        // aabb + local axes on: the shadow draw contributes a BlobShadow but no
        // Mesh / AabbBox / CoordinateAxes.
        let scene = build_scene(
            &draws,
            RenderMode::Filled,
            true,
            false,
            true,
            None,
            None,
            None,
        );

        let blobs: Vec<[f32; 16]> = scene
            .iter()
            .filter_map(|o| match o {
                DrawableObject::BlobShadow { model } => Some(*model),
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
            .iter()
            .filter(|o| matches!(o, DrawableObject::Mesh { .. }))
            .count();
        let aabbs = scene
            .iter()
            .filter(|o| matches!(o, DrawableObject::AabbBox { .. }))
            .count();
        let axes = scene
            .iter()
            .filter(|o| matches!(o, DrawableObject::CoordinateAxes { .. }))
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
        let a = [1.0f32; 16];
        let b = [2.0f32; 16];
        let draws = [
            Draw {
                mesh_id: 0,
                model: a,
                selection: DrawSelection::INHERIT,
            },
            Draw {
                mesh_id: 1,
                model: b,
                selection: DrawSelection::INHERIT,
            },
        ];

        // Plain filled: exactly one Mesh drawable per draw, no gizmos.
        assert_eq!(
            *build_scene(
                &draws,
                RenderMode::Filled,
                false,
                false,
                false,
                None,
                None,
                None
            ),
            [
                DrawableObject::Mesh {
                    mesh_id: 0,
                    model: a,
                    mode: RenderMode::Filled,
                },
                DrawableObject::Mesh {
                    mesh_id: 1,
                    model: b,
                    mode: RenderMode::Filled,
                },
            ]
        );

        // Wireframe propagates the mode to every mesh drawable.
        assert_eq!(
            *build_scene(
                &draws,
                RenderMode::Wireframe,
                false,
                false,
                false,
                None,
                None,
                None
            ),
            [
                DrawableObject::Mesh {
                    mesh_id: 0,
                    model: a,
                    mode: RenderMode::Wireframe,
                },
                DrawableObject::Mesh {
                    mesh_id: 1,
                    model: b,
                    mode: RenderMode::Wireframe,
                },
            ]
        );

        // Both overlays: meshes, then a tracking box per draw, then one gizmo.
        assert_eq!(
            *build_scene(
                &draws,
                RenderMode::Filled,
                true,
                true,
                false,
                None,
                None,
                None
            ),
            [
                DrawableObject::Mesh {
                    mesh_id: 0,
                    model: a,
                    mode: RenderMode::Filled,
                },
                DrawableObject::Mesh {
                    mesh_id: 1,
                    model: b,
                    mode: RenderMode::Filled,
                },
                DrawableObject::AabbBox {
                    mesh_id: 0,
                    model: a,
                },
                DrawableObject::AabbBox {
                    mesh_id: 1,
                    model: b,
                },
                DrawableObject::CoordinateAxes {
                    model: Matrix4::IDENTITY.to_cols_array(),
                },
            ]
        );

        // Local axes: one CoordinateAxes per draw at its own model (in the mesh
        // bucket order, before the world-origin gizmo), each tracking its draw.
        assert_eq!(
            *build_scene(
                &draws,
                RenderMode::Filled,
                false,
                false,
                true,
                None,
                None,
                None
            ),
            [
                DrawableObject::Mesh {
                    mesh_id: 0,
                    model: a,
                    mode: RenderMode::Filled,
                },
                DrawableObject::Mesh {
                    mesh_id: 1,
                    model: b,
                    mode: RenderMode::Filled,
                },
                DrawableObject::CoordinateAxes { model: a },
                DrawableObject::CoordinateAxes { model: b },
            ]
        );

        // Per-draw mode override: a draw's own `mode` wins over the global one,
        // so one frame can mix (e.g.) a textured mesh with a wireframe overlay.
        let mixed = [
            Draw {
                mesh_id: 0,
                model: a,
                selection: DrawSelection::INHERIT,
            },
            Draw {
                mesh_id: 1,
                model: b,
                selection: DrawSelection::Mesh(Some(RenderMode::Wireframe)),
            },
        ];
        assert_eq!(
            *build_scene(
                &mixed,
                RenderMode::Textured,
                false,
                false,
                false,
                None,
                None,
                None
            ),
            [
                DrawableObject::Mesh {
                    mesh_id: 0,
                    model: a,
                    mode: RenderMode::Textured,
                },
                DrawableObject::Mesh {
                    mesh_id: 1,
                    model: b,
                    mode: RenderMode::Wireframe,
                },
            ]
        );
    }
}
