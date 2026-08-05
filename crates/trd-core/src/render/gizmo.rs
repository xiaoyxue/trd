//! Overlay gizmo geometry: screen-space-expanded line segments for AABBs, axes,
//! and grids, plus the coordinate-axis arrowhead triangles.

use super::{GizmoLineVertex, GridPlane, Vertex};

/// RGB color of the optional AABB overlay box (bright green), chosen to stand
/// out against the default white mesh.
pub(crate) const AABB_COLOR: [f32; 3] = [0.0, 1.0, 0.0];

/// Pixel width of AABB edges.
pub(crate) const AABB_LINE_WIDTH_PX: f32 = 1.5;

/// The 12 edges of an axis-aligned box, indexing
/// the 8 corners in the order produced by [`crate::math::Aabb3::corners`]
/// (bit 0 = x, bit 1 = y, bit 2 = z of `(lo, hi)`): 4 bottom (`z=lo`) edges, 4
/// top (`z=hi`) edges, then the 4 vertical edges.
const AABB_EDGE_INDICES: [usize; 24] = [
    0, 1, 1, 3, 3, 2, 2, 0, // bottom face (z = lo)
    4, 5, 5, 7, 7, 6, 6, 4, // top face (z = hi)
    0, 4, 1, 5, 2, 6, 3, 7, // vertical edges
];

/// RGB colors of the coordinate-axes overlay gizmo (#42): X = red, Y = green,
/// Z = blue — the conventional right-handed axis coloring.
pub(crate) const AXES_COLORS: [[f32; 3]; 3] = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

/// World-space length of each coordinate axis in the overlay gizmo. The mesh
/// preview transform ([`crate::Mesh::preview_transform`]) fits a mesh's largest
/// extent to [`crate::mesh::DEFAULT_PREVIEW_TARGET`] world units (so a centered
/// mesh spans about `[-1, 1]` on its largest axis); a length of `1.5` reaches
/// from the world origin out past that half-extent, keeping the axis tips
/// visible just outside the silhouette.
pub(crate) const AXES_LENGTH: f32 = 1.5;

/// Pixel width of the RGB axis shafts.
pub(crate) const AXES_LINE_WIDTH_PX: f32 = 3.0;

/// Arrowhead dimensions scale with [`AXES_LENGTH`] and keep each tip at the old
/// line endpoint, so the gizmo's extent and meaning stay unchanged.
pub(crate) const AXES_ARROW_LENGTH: f32 = AXES_LENGTH * 0.09;
pub(crate) const AXES_ARROW_RADIUS: f32 = AXES_LENGTH * 0.035;
const AXES_ARROW_SIDES: usize = 12;

/// RGB color of the coordinate-plane grid overlay (#PlaneGrid): a light gray —
/// bright enough to read clearly over the court, but softened off pure white so
/// it doesn't compete with the red/green/blue axes gizmo drawn over it.
pub(crate) const GRID_COLOR: [f32; 3] = [0.65, 0.65, 0.65];

/// Number of cells per side of the coordinate-plane grid. The grid spans the
/// model-space square `[-GRID_HALF, GRID_HALF]²`; at `GRID_HALF = 3` (three
/// times the #77 placement-quad extent), `30` cells keep the classic `0.2`
/// model-unit spacing so the lattice extends well beyond the reconstructed quad
/// — enough of the floor to read the recovered plane — without thinning out.
pub(crate) const GRID_DIVISIONS: u32 = 30;

/// Pixel width of coordinate-plane grid lines.
pub(crate) const GRID_LINE_WIDTH_PX: f32 = 1.5;

/// Half-extent of the coordinate-plane grid in model space (the grid spans
/// `[-GRID_HALF, GRID_HALF]` on each in-plane axis). `3.0` reaches three times
/// past the unit placement-quad edge, so the grid carpets a large patch of the
/// recovered plane around the quad (the quad still occupies the central
/// `[-1, 1]²`) — making the found floor plane easy to eyeball.
const GRID_HALF: f32 = 3.0;

const LINE_QUAD_CORNERS: [(f32, f32); 6] = [
    (0.0, -1.0),
    (1.0, -1.0),
    (1.0, 1.0),
    (0.0, -1.0),
    (1.0, 1.0),
    (0.0, 1.0),
];

fn push_line(
    vertices: &mut Vec<GizmoLineVertex>,
    start: [f32; 3],
    end: [f32; 3],
    color: [f32; 3],
    width_px: f32,
) {
    vertices.extend(LINE_QUAD_CORNERS.map(|(endpoint, side)| GizmoLineVertex {
        start,
        end,
        color,
        extrusion: [endpoint, side, width_px],
    }));
}

/// Six triangle-list vertices per AABB edge, expanded to
/// [`AABB_LINE_WIDTH_PX`] in the gizmo shader.
pub(crate) fn aabb_line_vertices(corners: &[[f32; 3]; 8]) -> Vec<GizmoLineVertex> {
    let mut vertices = Vec::with_capacity(12 * LINE_QUAD_CORNERS.len());
    for edge in AABB_EDGE_INDICES.chunks_exact(2) {
        push_line(
            &mut vertices,
            corners[edge[0]],
            corners[edge[1]],
            AABB_COLOR,
            AABB_LINE_WIDTH_PX,
        );
    }
    vertices
}

/// Three anti-aliased axis shafts. Each ends at the base of its arrowhead while
/// the corresponding cone reaches the original [`AXES_LENGTH`] endpoint.
pub(crate) fn axes_line_vertices() -> Vec<GizmoLineVertex> {
    let shaft_length = AXES_LENGTH - AXES_ARROW_LENGTH;
    let mut vertices = Vec::with_capacity(3 * LINE_QUAD_CORNERS.len());
    for (axis, color) in [
        ([shaft_length, 0.0, 0.0], AXES_COLORS[0]),
        ([0.0, shaft_length, 0.0], AXES_COLORS[1]),
        ([0.0, 0.0, shaft_length], AXES_COLORS[2]),
    ] {
        push_line(
            &mut vertices,
            [0.0, 0.0, 0.0],
            axis,
            color,
            AXES_LINE_WIDTH_PX,
        );
    }
    vertices
}

/// RGB cone arrowheads for +X/+Y/+Z. The unlit overlay-triangle pipeline keeps
/// their colors exact and makes the tips readable over mesh surfaces.
pub(crate) fn axes_arrow_vertices() -> Vec<Vertex> {
    let mut vertices = Vec::with_capacity(3 * AXES_ARROW_SIDES * 6);
    let frames = [
        (
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            AXES_COLORS[0],
        ),
        (
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
            AXES_COLORS[1],
        ),
        (
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            AXES_COLORS[2],
        ),
    ];
    let vertex = |position: [f32; 3], color: [f32; 3]| Vertex {
        position,
        color,
        uv: [0.0, 0.0],
    };

    for (axis, u, v, color) in frames {
        let tip = axis.map(|component| component * AXES_LENGTH);
        let base = axis.map(|component| component * (AXES_LENGTH - AXES_ARROW_LENGTH));
        for side in 0..AXES_ARROW_SIDES {
            let angle0 = std::f32::consts::TAU * side as f32 / AXES_ARROW_SIDES as f32;
            let angle1 = std::f32::consts::TAU * (side + 1) as f32 / AXES_ARROW_SIDES as f32;
            let ring = |angle: f32| {
                let (sin, cos) = angle.sin_cos();
                [
                    base[0] + AXES_ARROW_RADIUS * (cos * u[0] + sin * v[0]),
                    base[1] + AXES_ARROW_RADIUS * (cos * u[1] + sin * v[1]),
                    base[2] + AXES_ARROW_RADIUS * (cos * u[2] + sin * v[2]),
                ]
            };
            let p0 = ring(angle0);
            let p1 = ring(angle1);
            vertices.extend([
                vertex(tip, color),
                vertex(p0, color),
                vertex(p1, color),
                vertex(base, color),
                vertex(p1, color),
                vertex(p0, color),
            ]);
        }
    }
    vertices
}

/// Triangle-list line quads for a coordinate-plane grid on `plane`, spanning
/// `[-GRID_HALF, GRID_HALF]²` in model space.
pub(crate) fn grid_line_vertices(plane: GridPlane) -> Vec<GizmoLineVertex> {
    // Lift a 2D in-plane coordinate `(u, w)` into the 3D plane (0 on the third
    // axis): XY → (u, w, 0), XZ → (u, 0, w), YZ → (0, u, w).
    let lift = |u: f32, w: f32| -> [f32; 3] {
        match plane {
            GridPlane::Xy => [u, w, 0.0],
            GridPlane::Xz => [u, 0.0, w],
            GridPlane::Yz => [0.0, u, w],
        }
    };
    let n = GRID_DIVISIONS;
    let step = 2.0 * GRID_HALF / n as f32;
    let mut vertices =
        Vec::with_capacity(2 * (GRID_DIVISIONS + 1) as usize * LINE_QUAD_CORNERS.len());
    for i in 0..=n {
        let t = -GRID_HALF + step * i as f32;
        push_line(
            &mut vertices,
            lift(t, -GRID_HALF),
            lift(t, GRID_HALF),
            GRID_COLOR,
            GRID_LINE_WIDTH_PX,
        );
        push_line(
            &mut vertices,
            lift(-GRID_HALF, t),
            lift(GRID_HALF, t),
            GRID_COLOR,
            GRID_LINE_WIDTH_PX,
        );
    }
    vertices
}

/// Number of `TriangleList` vertices in the **contact/blob-shadow** quad: two
/// triangles → `6`.
pub(crate) const SHADOW_VERTEX_COUNT: u32 = 6;

/// The `TriangleList` vertices of the **contact/blob-shadow** quad: a unit square
/// in the model-space XY plane (`[-1, 1]²` at `z = 0`). `shadow.wgsl` reads each
/// vertex's local `position.xy` as the radial coordinate and feathers a soft dark
/// alpha (darkest under the placed mesh, fading to `0` at the rim), so a
/// [`DrawableObject::BlobShadow`](super::DrawableObject::BlobShadow) lays a
/// grounding shadow on the plane beneath a placed mesh via its per-instance
/// model. Drawn non-indexed (`draw(0..SHADOW_VERTEX_COUNT, ..)`), alpha-blended
/// over the background frame plane.
pub(crate) fn blob_shadow_vertices() -> [Vertex; SHADOW_VERTEX_COUNT as usize] {
    let v = |x: f32, y: f32| Vertex {
        position: [x, y, 0.0],
        color: [0.0, 0.0, 0.0],
        uv: [0.0, 0.0],
    };
    [
        v(-1.0, -1.0),
        v(1.0, -1.0),
        v(1.0, 1.0),
        v(-1.0, -1.0),
        v(1.0, 1.0),
        v(-1.0, 1.0),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn line_geometry_expands_every_segment_to_two_triangles() {
        let axes = axes_line_vertices();
        let aabb = aabb_line_vertices(&[
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
        ]);
        let grid = grid_line_vertices(GridPlane::Xy);

        assert_eq!(axes.len(), 3 * 6);
        assert_eq!(aabb.len(), 12 * 6);
        assert_eq!(grid.len(), 2 * (GRID_DIVISIONS + 1) as usize * 6);
        assert!(axes
            .iter()
            .all(|vertex| vertex.extrusion[2] == AXES_LINE_WIDTH_PX));
        assert!(aabb
            .iter()
            .all(|vertex| vertex.extrusion[2] == AABB_LINE_WIDTH_PX));
        assert!(grid
            .iter()
            .all(|vertex| vertex.extrusion[2] == GRID_LINE_WIDTH_PX));
        assert_eq!(aabb[0].extrusion[2], grid[0].extrusion[2]);
        assert!(aabb[0].extrusion[2] < axes[0].extrusion[2]);
    }

    #[test]
    fn arrowheads_reach_the_original_axis_tips() {
        let vertices = axes_arrow_vertices();
        assert_eq!(vertices.len(), 3 * AXES_ARROW_SIDES * 6);
        for (tip, color) in [
            ([AXES_LENGTH, 0.0, 0.0], AXES_COLORS[0]),
            ([0.0, AXES_LENGTH, 0.0], AXES_COLORS[1]),
            ([0.0, 0.0, AXES_LENGTH], AXES_COLORS[2]),
        ] {
            assert!(vertices
                .iter()
                .any(|vertex| vertex.position == tip && vertex.color == color));
        }
    }
}
