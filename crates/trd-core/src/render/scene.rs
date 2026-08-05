//! The `DrawableObject` scene model: render modes, frame fit, draws, and
//! per-frame scene assembly.

use crate::math::Matrix4;

/// How a [`MeshRenderer`] rasterizes its meshes: solid filled triangles, or an
/// edge **wireframe** (`LineList` over the derived [`crate::Mesh::edge_indices`]
/// buffer). Default is [`RenderMode::Filled`]; wireframe (#38) is opt-in via
/// [`BatchRenderer::set_mode`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RenderMode {
    /// Draw triangles filled with the per-vertex color (the mesh's triangle
    /// index buffer).
    #[default]
    Filled,
    /// Draw only triangle edges as lines (the deduped edge index buffer).
    Wireframe,
    /// Draw triangles filled, sampling the renderer's bound texture at each
    /// vertex UV instead of the vertex color (#20).
    Textured,
    /// Physically-based **Disney principled BRDF** shading (`disney.wgsl`): the
    /// bound albedo lit by a small virtual light rig plus an optional
    /// equirectangular HDR environment-map reflection, with smooth shading
    /// normals derived at upload. Metallic materials read as shiny reflective
    /// metal (e.g. the coke can). Configured globally via the renderer's
    /// [`DisneyMaterial`](crate::DisneyMaterial) + bound environment map.
    Pbr,
    /// Not a mesh rasterization at all: draw a **contact / blob grounding
    /// shadow** ([`DrawableObject::BlobShadow`]) instead of the mesh. A per-draw
    /// `mode: "shadow"` in the stream lifts that draw's `model` into a soft dark
    /// blob on the placed mesh's ground plane (#110 follow-up), so the placed mesh
    /// reads as sitting on the reconstructed surface. The draw's `mesh` id is
    /// ignored (the shadow uses shared gizmo geometry).
    Shadow,
}

/// Wire byte meaning "inherit the renderer's global mode" in the optional
/// per-draw `draw_mode` (`List<UInt8>`) protocol column (see
/// [`RenderMode::from_wire`]). A draw carrying this value defers to the `mode`
/// argument of [`build_scene`], so a stream can override only *some* draws
/// (e.g. draw a wireframe overlay quad while every other draw follows the
/// front-end's global mode).
pub const DRAW_MODE_INHERIT: u8 = 255;

impl RenderMode {
    /// Decodes an optional per-draw `draw_mode` wire byte into a [`Draw::mode`]
    /// override: `0`→`Filled`, `1`→`Wireframe`, `2`→`Textured`, `3`→`Shadow`,
    /// `4`→`Pbr`, and [`DRAW_MODE_INHERIT`]→`None` (inherit the global mode).
    /// Returns `None` for an unrecognized byte so callers can raise a decode error.
    pub fn from_wire(byte: u8) -> Option<Option<RenderMode>> {
        match byte {
            0 => Some(Some(RenderMode::Filled)),
            1 => Some(Some(RenderMode::Wireframe)),
            2 => Some(Some(RenderMode::Textured)),
            3 => Some(Some(RenderMode::Shadow)),
            4 => Some(Some(RenderMode::Pbr)),
            DRAW_MODE_INHERIT => Some(None),
            _ => None,
        }
    }
}

/// How a [`DrawableObject::FramePlane`] maps its background image onto the
/// viewport (#63). Both modes fill the whole viewport (no letterbox bars); they
/// differ only in how a mismatched image/viewport aspect is handled.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum FrameFit {
    /// Stretch the image to exactly fill the viewport, ignoring aspect ratio.
    /// The natural choice when the frame image already matches the render aspect
    /// (e.g. a 16:9 video rendered at 16:9).
    #[default]
    Stretch,
    /// Scale the image to cover the viewport preserving aspect, center-cropping
    /// the overflowing axis (no bars, some content cropped).
    Cover,
}

/// The centered UV scale that realizes `fit` for an image of `tex_w`×`tex_h`
/// displayed on a `view_w`×`view_h` viewport. Applied in `frame_plane.wgsl` as
/// `uv' = (uv − 0.5)·scale + 0.5`, so `1.0` fills and `< 1.0` crops (zooms in).
/// [`FrameFit::Stretch`] is always `(1, 1)`; [`FrameFit::Cover`] shrinks the UV
/// range on the longer axis so the shorter one fills.
pub(crate) fn frame_fit_uv_scale(
    fit: FrameFit,
    tex_w: u32,
    tex_h: u32,
    view_w: u32,
    view_h: u32,
) -> [f32; 2] {
    match fit {
        FrameFit::Stretch => [1.0, 1.0],
        FrameFit::Cover => {
            let tex_aspect = tex_w.max(1) as f32 / tex_h.max(1) as f32;
            let view_aspect = view_w.max(1) as f32 / view_h.max(1) as f32;
            if tex_aspect > view_aspect {
                // Image wider than the viewport: crop its width (sample a
                // narrower horizontal UV range).
                [view_aspect / tex_aspect, 1.0]
            } else {
                // Image taller than the viewport: crop its height.
                [1.0, tex_aspect / view_aspect]
            }
        }
    }
}

/// Which coordinate plane a [`DrawableObject::PlaneGrid`] lattices, i.e. the two
/// model-space axes it spans (the third is held at 0): `Xy` → the X/Y plane,
/// `Xz` → X/Z, `Yz` → Y/Z. For a #77 placement quad (whose local Z is the plane
/// normal), `Xy` is the quad's own plane — a grid on the reconstructed surface.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GridPlane {
    /// The model-space X/Y plane (Z = 0). The placement-quad's own surface.
    Xy,
    /// The model-space X/Z plane (Y = 0).
    Xz,
    /// The model-space Y/Z plane (X = 0).
    Yz,
}

impl GridPlane {
    /// A stable `0..3` index (`Xy`→0, `Xz`→1, `Yz`→2) used to key the renderer's
    /// per-plane grid vertex buffers.
    pub(crate) fn index(self) -> usize {
        match self {
            GridPlane::Xy => 0,
            GridPlane::Xz => 1,
            GridPlane::Yz => 2,
        }
    }
}

impl std::str::FromStr for GridPlane {
    type Err = String;

    /// Parses `xy` / `xz` / `yz` (case-insensitive) into a [`GridPlane`], so
    /// front-ends can accept the plane as a plain flag value.
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_ascii_lowercase().as_str() {
            "xy" => Ok(GridPlane::Xy),
            "xz" => Ok(GridPlane::Xz),
            "yz" => Ok(GridPlane::Yz),
            other => Err(format!(
                "unknown grid plane {other:?} (expected xy, xz, or yz)"
            )),
        }
    }
}

/// A single instance placement decoded from a frame's protocol draw list
/// (`draw_mesh` / `draw_model`): which mesh to draw (index into the leading mesh
/// table) and the per-instance model matrix (column-major), applied beneath that
/// mesh's base (preview) model. This is the *wire* representation; the renderer
/// composes it (plus core gizmos) into a [`Scene`] of [`DrawableObject`]s.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Draw {
    pub mesh_id: u32,
    pub model: [f32; 16],
    /// Optional per-draw [`RenderMode`] override (protocol `draw_mode` column):
    /// `Some(mode)` draws this instance in `mode` regardless of the front-end's
    /// global mode; `None` inherits the global `mode` passed to [`build_scene`].
    /// Lets one frame mix e.g. a textured mesh with a wireframe overlay quad.
    pub mode: Option<RenderMode>,
}

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
    /// A **contact / blob grounding shadow** (#110 follow-up): a soft dark radial
    /// blob laid on a placed mesh's ground plane, placed by `model` (a flat quad
    /// on the plane, sized to the mesh footprint), so the mesh reads as *sitting
    /// on* the reconstructed surface rather than floating over the composited
    /// video plate. A [`RenderMode::Shadow`] draw becomes this variant. Tied to no
    /// mesh (no base model); alpha-blended over the [`FramePlane`](Self::FramePlane)
    /// and drawn *before* the opaque content mesh (depth-write off) so the mesh
    /// composites on top while the surrounding rim darkens the floor.
    BlobShadow { model: [f32; 16] },
    /// A screen-aligned **background frame plane** (#63): a fullscreen quad that
    /// samples the renderer's bound background frame texture (set via
    /// [`MeshRenderer::update_frame_texture_rgba`]), composited **under** the
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
/// it to [`MeshRenderer::encode`].
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
pub fn plane_grid_overlays(
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
pub fn selection_aabb_overlay(draws: &[Draw], selected: Option<u32>) -> Vec<DrawableObject> {
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

    #[test]
    fn grid_plane_from_str_roundtrip() {
        use std::str::FromStr;
        assert_eq!(GridPlane::from_str("xy"), Ok(GridPlane::Xy));
        assert_eq!(GridPlane::from_str("XZ"), Ok(GridPlane::Xz));
        assert_eq!(GridPlane::from_str("yz"), Ok(GridPlane::Yz));
        assert!(GridPlane::from_str("zz").is_err());
    }

    #[test]
    fn frame_fit_uv_scale_stretch_and_cover() {
        // Stretch always fills exactly (no crop), regardless of aspect mismatch.
        assert_eq!(
            frame_fit_uv_scale(FrameFit::Stretch, 200, 100, 100, 100),
            [1.0, 1.0]
        );

        // Cover a 2:1 image on a 1:1 viewport: crop width (sample a narrower
        // horizontal UV range), full height.
        let s = frame_fit_uv_scale(FrameFit::Cover, 200, 100, 100, 100);
        assert!(
            (s[0] - 0.5).abs() < 1e-6 && (s[1] - 1.0).abs() < 1e-6,
            "wide image over square viewport crops width, got {s:?}"
        );

        // Cover a 1:2 image on a 1:1 viewport: crop height, full width.
        let s = frame_fit_uv_scale(FrameFit::Cover, 100, 200, 100, 100);
        assert!(
            (s[0] - 1.0).abs() < 1e-6 && (s[1] - 0.5).abs() < 1e-6,
            "tall image over square viewport crops height, got {s:?}"
        );

        // Matching aspect ⇒ no crop either way.
        assert_eq!(
            frame_fit_uv_scale(FrameFit::Cover, 160, 90, 320, 180),
            [1.0, 1.0]
        );
    }
}
