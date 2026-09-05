//! The CPU-side mesh: trd's canonical [`Mesh`], the loaders that produce it,
//! and the geometry derived from it.
//!
//! Under the crate's module boundary (#224) the root holds the universal domain
//! vocabulary — what any renderer would have — so the mesh and its three
//! loaders live here, device-free, while their GPU residency stays in
//! `render/mesh_store.rs`:
//!
//! - [`mesh`] — [`Mesh`] / [`MeshShading`], the container every source converges on
//! - [`obj`] — Wavefront OBJ text, trd's default authorable format (#36)
//! - [`arrow`] — the columnar Arrow mesh table the stream protocol carries (#37)
//! - [`gltf`] — binary glTF (GLB) primitives + their Disney materials
//!
//! What stays in this module is the geometry every source shares: the mesh's
//! bounds ([`Mesh::aabb`] / [`Mesh::center`]), the [`Mesh::preview_transform`]
//! that frames an arbitrary-unit asset, and the wireframe [`Mesh::edge_indices`].

mod arrow;
mod gltf;
mod identity;
// `mesh::mesh` holds the type this module is named for; the loaders around it
// are named for their formats, so the inner module keeps the type's own name.
#[allow(clippy::module_inception)]
mod mesh;
mod obj;

pub use gltf::{import_glb, import_gltf_materials, GltfAsset, GltfImportError};
pub use identity::{MeshId, MeshResourceError, MeshTable, MeshTableIndex};
pub use mesh::{Mesh, MeshShading};

// `::arrow` (leading `::`) is the external Arrow crate, not this module's
// sibling `arrow` submodule.
use ::arrow::datatypes::DataType;

use crate::math::{Aabb3, Point3, Transform, Vector3, EPSILON};

/// Vertex color used when a mesh source carries no per-vertex color.
const DEFAULT_COLOR: [f32; 3] = [1.0, 1.0, 1.0];

/// Default target size (world units, largest AABB extent) for
/// [`Mesh::preview_transform`] — "a reasonable size" for arbitrary-unit assets.
pub const DEFAULT_PREVIEW_TARGET: f32 = 2.0;

/// Errors produced while loading a mesh into the canonical [`Mesh`].
#[derive(Debug, thiserror::Error)]
pub enum MeshError {
    /// The underlying `tobj` loader rejected the OBJ (bad index, malformed
    /// coordinate, unreadable buffer, …).
    #[error("failed to parse OBJ: {0}")]
    Obj(#[from] tobj::LoadError),
    /// The mesh parsed but produced no geometry (no vertices, or no faces).
    #[error("mesh is empty (no vertices or no faces)")]
    Empty,
    /// An Arrow mesh table is missing a required column.
    #[error("mesh table is missing required column `{0}`")]
    MissingColumn(&'static str),
    /// An Arrow mesh column has an unexpected Arrow type.
    #[error("mesh column `{column}` has type {actual:?}, expected {expected}")]
    ColumnType {
        column: &'static str,
        expected: &'static str,
        actual: DataType,
    },
    /// An Arrow mesh column that must be non-null contains null values.
    #[error("mesh column `{0}` contains null values")]
    NullValues(&'static str),
    /// An `index` entry references a vertex beyond the `position` column.
    #[error("mesh index {index} is out of range (only {vertex_count} vertices)")]
    IndexOutOfRange { index: u32, vertex_count: usize },
    /// A non-indexed mesh table's vertex count is not a multiple of three, so it
    /// cannot be a triangle list.
    #[error("non-indexed mesh has {vertex_count} vertices, not a multiple of 3")]
    NonTriangleList { vertex_count: usize },
}

impl Mesh {
    /// The axis-aligned bounding box of the mesh's vertex positions.
    ///
    /// Reuses the shared [`Aabb3`]; use [`Aabb3::center`] / [`Aabb3::size`] for
    /// preview centering + uniform scale-to-fit (#37) and camera framing (#43).
    /// Assumes a non-empty mesh (as produced by [`Mesh::from_obj`]).
    pub fn aabb(&self) -> Aabb3 {
        Aabb3::from_points(self.vertices.iter().map(|v| Point3::from_array(v.position)))
    }

    /// The center of the mesh's [`Mesh::aabb`] — the point translated to the
    /// world origin when previewing a loaded mesh (#37).
    pub fn center(&self) -> Point3 {
        self.aabb().center()
    }

    /// A uniform **preview model transform** that centers the mesh's AABB at the
    /// world origin and scales it to fit `target` world units along its largest
    /// extent: `scale(s) · translate(−center)`, `s = target / max_extent`.
    ///
    /// Applied as the default model matrix (#41) so an arbitrary-unit asset
    /// (e.g. `bunny.obj`) renders **centered and at a reasonable size** rather
    /// than off-screen or tiny (#37/#44). The scale is uniform (isotropic); a
    /// degenerate (zero-extent) mesh keeps unit scale. Compose a per-frame
    /// rotation on the left for a turntable (`rotate.then(...)` is wrong; use
    /// `preview.then(rotate)` — rotation applied after centering+scaling).
    pub fn preview_transform(&self, target: f32) -> Transform {
        let aabb = self.aabb();
        let center = aabb.center().to_array();
        let size = aabb.size();
        let max_extent = size.x().max(size.y()).max(size.z());
        let scale = if max_extent > EPSILON {
            target / max_extent
        } else {
            1.0
        };
        Transform::from_translation(Vector3::new(-center[0], -center[1], -center[2]))
            .then(Transform::from_scale(Vector3::new(scale, scale, scale)))
    }

    /// Derives a deduplicated **edge** index buffer for wireframe rendering
    /// (#38): each triangle `(a, b, c)` of the index buffer contributes its
    /// three undirected edges `(a,b)`, `(b,c)`, `(c,a)`, normalized to
    /// `(min, max)` and deduped so an edge shared by two triangles is emitted
    /// once. The result is flattened as **two indices per line** — a
    /// `PrimitiveTopology::LineList` index buffer over the same vertex buffer.
    ///
    /// A `LineList` edge buffer is preferred over `PolygonMode::Line`, which
    /// needs `Features::POLYGON_MODE_LINE` that WebGPU does not guarantee, so
    /// this stays portable across native and wasm. Pure (GPU-free) and
    /// unit-testable. Assumes a triangle list (`indices.len()` a multiple of 3,
    /// as every [`Mesh`] constructor guarantees); a trailing partial triangle,
    /// if any, is ignored.
    pub fn edge_indices(&self) -> Vec<u32> {
        use std::collections::HashSet;
        let mut seen: HashSet<(u32, u32)> = HashSet::new();
        let mut edges: Vec<u32> = Vec::new();
        for tri in self.indices.chunks_exact(3) {
            for &(a, b) in &[(tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])] {
                let key = if a <= b { (a, b) } else { (b, a) };
                if seen.insert(key) {
                    edges.push(key.0);
                    edges.push(key.1);
                }
            }
        }
        edges
    }
}

/// The trd hello-triangle as OBJ, using the `v x y z r g b` color extension so
/// it reproduces `Mesh::hello_triangle()` exactly. Mirrors the shipped
/// `examples/triangle.obj` (kept inline so the crate build does not depend on
/// asset files the nix source filter drops). Shared by the loader submodules'
/// tests so both authoring routes are compared against one fixture.
#[cfg(test)]
const TRIANGLE_OBJ: &str = "\
v 0.0 0.5 0.0 1.0 0.0 0.0
v -0.5 -0.5 0.0 0.0 1.0 0.0
v 0.5 -0.5 0.0 0.0 0.0 1.0
f 1 2 3
";

/// A unit quad (two triangles), white. Mirrors `examples/quad.obj`.
#[cfg(test)]
const QUAD_OBJ: &str = "\
v -0.5 -0.5 0.0
v 0.5 -0.5 0.0
v 0.5 0.5 0.0
v -0.5 0.5 0.0
f 1 2 3 4
";

#[cfg(test)]
mod tests {
    use super::*;
    use crate::render::Vertex;

    /// Collects an edge index buffer into a set of undirected `(min, max)` pairs.
    fn edge_set(edges: &[u32]) -> std::collections::BTreeSet<(u32, u32)> {
        assert_eq!(edges.len() % 2, 0, "edge buffer must be pairs of indices");
        edges
            .chunks_exact(2)
            .map(|e| (e[0].min(e[1]), e[0].max(e[1])))
            .collect()
    }

    #[test]
    fn edge_indices_single_triangle_yields_three_edges() {
        let mesh = Mesh::hello_triangle();
        let edges = mesh.edge_indices();
        assert_eq!(edges.len(), 6, "3 edges × 2 indices");
        assert_eq!(
            edge_set(&edges),
            [(0, 1), (1, 2), (0, 2)].into_iter().collect()
        );
    }

    #[test]
    fn edge_indices_quad_dedups_shared_diagonal() {
        // Two triangles [0,1,2] + [0,2,3] share the diagonal edge (0,2), so the
        // quad has 5 unique edges (4 sides + 1 diagonal), not 6.
        let mesh = Mesh::from_obj(QUAD_OBJ).expect("quad parses");
        let edges = mesh.edge_indices();
        assert_eq!(edges.len(), 10, "5 unique edges × 2 indices");
        assert_eq!(
            edge_set(&edges),
            [(0, 1), (1, 2), (0, 2), (2, 3), (0, 3)]
                .into_iter()
                .collect()
        );
    }

    #[test]
    fn aabb_and_center_of_quad() {
        let mesh = Mesh::from_obj(QUAD_OBJ).unwrap();
        let aabb = mesh.aabb();
        assert_eq!(aabb.min(), Point3::new(-0.5, -0.5, 0.0));
        assert_eq!(aabb.max(), Point3::new(0.5, 0.5, 0.0));
        assert_eq!(mesh.center(), Point3::new(0.0, 0.0, 0.0));
    }

    #[test]
    fn preview_transform_centers_and_scales_to_target() {
        // A 4-unit-wide box centered at (10, 0, 0): after the preview transform
        // its center is at the origin and its largest extent spans `target`.
        let mesh = Mesh {
            vertices: vec![
                Vertex {
                    position: [8.0, -1.0, -0.5],
                    color: DEFAULT_COLOR,
                    uv: [0.0, 0.0],
                },
                Vertex {
                    position: [12.0, 1.0, 0.5],
                    color: DEFAULT_COLOR,
                    uv: [0.0, 0.0],
                },
            ],
            indices: vec![0, 1, 0],
            shading: None,
        };
        let target = 2.0;
        let t = mesh.preview_transform(target);
        // Center maps to origin.
        let c = t.transform_point(Point3::new(10.0, 0.0, 0.0));
        assert!(
            c.to_array().iter().all(|v| v.abs() < 1e-5),
            "center = {c:?}"
        );
        // The transformed AABB's largest extent equals `target`.
        let out = t.transform_aabb(mesh.aabb());
        let size = out.size();
        let max_extent = size.x().max(size.y()).max(size.z());
        assert!(
            (max_extent - target).abs() < 1e-5,
            "max_extent = {max_extent}"
        );
    }

    #[test]
    fn preview_transform_degenerate_mesh_keeps_unit_scale() {
        let mesh = Mesh {
            vertices: vec![Vertex {
                position: [3.0, 3.0, 3.0],
                color: DEFAULT_COLOR,
                uv: [0.0, 0.0],
            }],
            indices: vec![0],
            shading: None,
        };
        let t = mesh.preview_transform(2.0);
        // A single point has zero extent → unit scale, only the centering shift.
        let moved = t.transform_point(Point3::new(3.0, 3.0, 3.0));
        assert!(moved.to_array().iter().all(|v| v.abs() < 1e-5));
    }
}
