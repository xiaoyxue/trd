//! Overlay gizmo geometry: the AABB box edge indices, the coordinate-axes line
//! vertices, and the coordinate-plane grid line vertices.

use super::{GridPlane, Vertex};

/// RGB color of the optional AABB overlay box (bright green), chosen to stand
/// out against the default white mesh. See [`MeshRenderer::set_show_aabb`].
pub(crate) const AABB_COLOR: [f32; 3] = [0.0, 1.0, 0.0];

/// The 12 edges of an axis-aligned box as a `LineList` index buffer, indexing
/// the 8 corners in the order produced by [`crate::math::Aabb3::corners`]
/// (bit 0 = x, bit 1 = y, bit 2 = z of `(lo, hi)`): 4 bottom (`z=lo`) edges, 4
/// top (`z=hi`) edges, then the 4 vertical edges.
pub(crate) const AABB_EDGE_INDICES: [u32; 24] = [
    0, 1, 1, 3, 3, 2, 2, 0, // bottom face (z = lo)
    4, 5, 5, 7, 7, 6, 6, 4, // top face (z = hi)
    0, 4, 1, 5, 2, 6, 3, 7, // vertical edges
];

/// RGB colors of the coordinate-axes overlay gizmo (#42): X = red, Y = green,
/// Z = blue — the conventional right-handed axis coloring. See
/// [`MeshRenderer::set_show_axes`].
pub(crate) const AXES_COLORS: [[f32; 3]; 3] = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

/// World-space length of each coordinate axis in the overlay gizmo. The mesh
/// preview transform ([`crate::Mesh::preview_transform`]) fits a mesh's largest
/// extent to [`crate::mesh::DEFAULT_PREVIEW_TARGET`] world units (so a centered
/// mesh spans about `[-1, 1]` on its largest axis); a length of `1.5` reaches
/// from the world origin out past that half-extent, keeping the axis tips
/// visible just outside the silhouette.
pub(crate) const AXES_LENGTH: f32 = 1.5;

/// Number of `LineList` vertices in the coordinate-axes gizmo (three lines →
/// six vertices), drawn non-indexed. See [`axes_vertices`].
pub(crate) const AXES_VERTEX_COUNT: u32 = 6;

/// The six vertices of the coordinate-axes gizmo as a `LineList`: three lines
/// from the world origin along +X, +Y, +Z, each colored per [`AXES_COLORS`].
/// Drawn non-indexed (`draw(0..6, ..)`) under the camera `P·V` with an identity
/// per-instance model, so the gizmo marks the world origin/frame.
pub(crate) const fn axes_vertices() -> [Vertex; 6] {
    [
        Vertex {
            position: [0.0, 0.0, 0.0],
            color: AXES_COLORS[0],
            uv: [0.0, 0.0],
        },
        Vertex {
            position: [AXES_LENGTH, 0.0, 0.0],
            color: AXES_COLORS[0],
            uv: [0.0, 0.0],
        },
        Vertex {
            position: [0.0, 0.0, 0.0],
            color: AXES_COLORS[1],
            uv: [0.0, 0.0],
        },
        Vertex {
            position: [0.0, AXES_LENGTH, 0.0],
            color: AXES_COLORS[1],
            uv: [0.0, 0.0],
        },
        Vertex {
            position: [0.0, 0.0, 0.0],
            color: AXES_COLORS[2],
            uv: [0.0, 0.0],
        },
        Vertex {
            position: [0.0, 0.0, AXES_LENGTH],
            color: AXES_COLORS[2],
            uv: [0.0, 0.0],
        },
    ]
}

/// RGB color of the coordinate-plane grid overlay (#PlaneGrid): a neutral light
/// gray so the grid reads as a reference lattice without competing with the
/// red/green/blue axes gizmo drawn over it.
pub(crate) const GRID_COLOR: [f32; 3] = [0.75, 0.75, 0.75];

/// Number of cells per side of the coordinate-plane grid. The grid spans the
/// model-space square `[-GRID_HALF, GRID_HALF]²`; at `GRID_HALF = 3` (three
/// times the #77 placement-quad extent), `30` cells keep the classic `0.2`
/// model-unit spacing so the lattice extends well beyond the reconstructed quad
/// — enough of the floor to read the recovered plane — without thinning out.
pub(crate) const GRID_DIVISIONS: u32 = 30;

/// Half-extent of the coordinate-plane grid in model space (the grid spans
/// `[-GRID_HALF, GRID_HALF]` on each in-plane axis). `3.0` reaches three times
/// past the unit placement-quad edge, so the grid carpets a large patch of the
/// recovered plane around the quad (the quad still occupies the central
/// `[-1, 1]²`) — making the found floor plane easy to eyeball.
const GRID_HALF: f32 = 3.0;

/// Number of `LineList` vertices in the coordinate-plane grid: `GRID_DIVISIONS +
/// 1` lines in each of the two in-plane directions, two vertices per line →
/// `4 · (GRID_DIVISIONS + 1)`. Drawn non-indexed like the axes gizmo.
pub(crate) const GRID_VERTEX_COUNT: u32 = 4 * (GRID_DIVISIONS + 1);

/// The `LineList` vertices of a coordinate-plane grid on `plane`, spanning the
/// model-space square `[-GRID_HALF, GRID_HALF]²` at `0` on the third axis:
/// `GRID_DIVISIONS + 1` lines along each in-plane axis, all colored
/// [`GRID_COLOR`]. Drawn non-indexed
/// (`draw(0..GRID_VERTEX_COUNT, ..)`) under the camera `P·V` with a per-instance
/// model, so a [`DrawableObject::PlaneGrid`](super::DrawableObject::PlaneGrid)
/// lays the grid in that object's local frame (e.g. #77's quad plane).
pub(crate) fn grid_vertices(plane: GridPlane) -> Vec<Vertex> {
    // Lift a 2D in-plane coordinate `(u, w)` into the 3D plane (0 on the third
    // axis): XY → (u, w, 0), XZ → (u, 0, w), YZ → (0, u, w).
    let lift = |u: f32, w: f32| -> [f32; 3] {
        match plane {
            GridPlane::Xy => [u, w, 0.0],
            GridPlane::Xz => [u, 0.0, w],
            GridPlane::Yz => [0.0, u, w],
        }
    };
    let vert = |p: [f32; 3]| Vertex {
        position: p,
        color: GRID_COLOR,
        uv: [0.0, 0.0],
    };
    let n = GRID_DIVISIONS;
    let step = 2.0 * GRID_HALF / n as f32;
    let mut verts = Vec::with_capacity(GRID_VERTEX_COUNT as usize);
    for i in 0..=n {
        let t = -GRID_HALF + step * i as f32;
        // Line at fixed first axis = t, spanning the second axis.
        verts.push(vert(lift(t, -GRID_HALF)));
        verts.push(vert(lift(t, GRID_HALF)));
        // Line at fixed second axis = t, spanning the first axis.
        verts.push(vert(lift(-GRID_HALF, t)));
        verts.push(vert(lift(GRID_HALF, t)));
    }
    verts
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
