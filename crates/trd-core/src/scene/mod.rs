//! The **scene model**: what a frame draws, independent of how it is rendered.
//!
//! A [`Scene`] is an ordered list of [`DrawableObject`]s — light `Copy` handles
//! naming *which* primitive to draw and its per-frame model. Geometry and GPU
//! state are owned by the renderer; nothing here touches wgpu, which is why this
//! sits at the crate root beside `mesh`/`camera`/`material` rather than inside
//! the render backend (the same reasoning that moved materials out in #180).
//!
//! Split by concern (#203):
//!
//! | module | owns |
//! |---|---|
//! | [`drawable`] | [`DrawableObject`] — the base interface for every primitive — and [`Scene`] |
//! | [`draw`] | [`Draw`], the *wire* instance record, and its render-mode byte codec |
//! | [`draw_config`] | [`RenderMode`], [`FrameFit`], [`GridPlane`] — the per-drawable configuration a front-end selects |
//! | this module | assembly: [`scene_with_overlays`], [`build_scene`], and the overlay builders |
//!
//! Assembly is the **one** place a wire [`Draw`] becomes a [`DrawableObject`],
//! which is what keeps every front-end rendering the same scene from the same
//! inputs (#180).

mod draw;
mod draw_config;
mod drawable;

pub use draw::Draw;
#[cfg(test)]
pub(crate) use draw::DRAW_MODE_INHERIT;
pub(crate) use draw_config::frame_fit_uv_scale;
pub use draw_config::{FrameFit, GridPlane, RenderMode};
pub use drawable::{DrawableObject, Scene};

use crate::math::Matrix4;

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
/// Shared by the native ([`crate::run_stream`]) and wasm front-ends so neither
/// branches per primitive type: both author the same ordered `Scene` and hand
/// it to [`SceneRenderer::encode`].
/// The **one** place a frame's [`Scene`] is assembled from a wire draw list plus
/// appearance options.
///
/// Every front-end used to do this itself — the headless `Renderer` from nine
/// retained overlay flags, `trd-gui` through setters, the browser renderers from
/// their own booleans — which is how native and web overlay assembly drifted
/// apart. Composing [`build_scene`], [`plane_grid_overlays`] and
/// [`selection_aabb_overlay`] here means all of them get the same scene from the
/// same inputs (#180).
///
/// `frame` prepends a background [`DrawableObject::FramePlane`] (#63) when the
/// caller has a frame texture bound.
pub fn scene_with_overlays(
    draws: &[Draw],
    options: &super::RenderOptions,
    frame: Option<FrameFit>,
) -> Scene {
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
    // World / object plane grids (#140) are ungated by render mode, so a filled or
    // PBR object still gets a floor. `encode` buckets by primitive type, so
    // appending here still draws them in the grid pass.
    scene.extend(plane_grid_overlays(
        draws,
        options.show_world_grid,
        options.show_object_grid,
    ));
    // Selection highlight (#141): drawn even when the show-all-AABBs toggle is off.
    scene.extend(selection_aabb_overlay(draws, options.selected));
    scene
}

#[allow(clippy::too_many_arguments)]
pub fn build_scene(
    draws: &[Draw],
    mode: RenderMode,
    show_aabb: bool,
    show_axes: bool,
    show_local_axes: bool,
    local_grid: Option<GridPlane>,
    grid_mesh: Option<u32>,
    frame: Option<FrameFit>,
) -> Scene {
    let mut scene = Vec::with_capacity(
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
    for draw in draws {
        let resolved = draw.mode.unwrap_or(mode);
        // A `Shadow` draw is not a mesh rasterization: lift its model into a
        // BlobShadow grounding blob on the placed mesh's plane (#110 follow-up).
        if resolved == RenderMode::Shadow {
            scene.push(DrawableObject::BlobShadow { model: draw.model });
            continue;
        }
        scene.push(DrawableObject::Mesh {
            mesh_id: draw.mesh_id,
            model: draw.model,
            mode: resolved,
        });
    }
    if show_aabb {
        for draw in draws {
            // Skip shadow draws — they carry no mesh geometry to box.
            if draw.mode.unwrap_or(mode) == RenderMode::Shadow {
                continue;
            }
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
            let is_wireframe = draw.mode.unwrap_or(mode) == RenderMode::Wireframe;
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
        for draw in draws {
            // Skip shadow draws — the blob is a floor decal, not a placed object
            // whose local frame warrants an axes gizmo.
            if draw.mode.unwrap_or(mode) == RenderMode::Shadow {
                continue;
            }
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
        for draw in draws {
            if draw.mode == Some(RenderMode::Shadow) {
                continue;
            }
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
    if draw.mode == Some(RenderMode::Shadow) {
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

    #[test]
    fn plane_grid_overlays_place_world_and_object_grids() {
        let draws = [
            Draw {
                mesh_id: 0,
                model: Matrix4::from_translation(crate::math::Vector3::new(1.0, 0.0, 0.0))
                    .to_cols_array(),
                mode: None,
            },
            Draw {
                mesh_id: 1,
                model: Matrix4::IDENTITY.to_cols_array(),
                mode: Some(RenderMode::Shadow),
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
            mode: None,
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
                mode: None,
            }, // inherits the scene default
            Draw {
                mesh_id: 1,
                model,
                mode: Some(RenderMode::Wireframe),
            }, // overrides
            Draw {
                mesh_id: 2,
                model,
                mode: Some(RenderMode::Textured),
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
                mode: None,
            },
            Draw {
                mesh_id: 1,
                model: model_b,
                mode: None,
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
                mode: None,
            },
            Draw {
                mesh_id: 1,
                model: model_b,
                mode: None,
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
                mode: Some(RenderMode::Textured),
            },
            Draw {
                mesh_id: 1,
                model: model_b,
                mode: Some(RenderMode::Wireframe),
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
                mode: Some(RenderMode::Wireframe),
            },
            Draw {
                mesh_id: 0,
                model: Matrix4::IDENTITY.to_cols_array(),
                mode: Some(RenderMode::Wireframe),
            },
            Draw {
                mesh_id: 1,
                model: model_quad,
                mode: Some(RenderMode::Wireframe),
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
                mode: Some(RenderMode::Shadow),
            },
            Draw {
                mesh_id: 0,
                model: bunny_m,
                mode: Some(RenderMode::Textured),
            },
            Draw {
                mesh_id: 1,
                model: quad_m,
                mode: Some(RenderMode::Wireframe),
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
}
