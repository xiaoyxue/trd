//! Overlay gizmo geometry: the AABB box edge indices and the
//! coordinate-axes line vertices.

use super::Vertex;

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
