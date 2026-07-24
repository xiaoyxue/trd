//! Built-in assets shared by the native and wasm entry points (#97): the default
//! mesh shown when no external mesh is supplied.

use trd_core::{Mesh, MeshError};

/// A built-in origin-centered unit cube with per-corner colors, used as the
/// default object when no mesh is supplied (`v x y z r g b` OBJ extension).
pub const DEFAULT_MESH_OBJ: &str = "\
v -0.5 -0.5 -0.5 0.1 0.1 0.9
v  0.5 -0.5 -0.5 0.9 0.1 0.1
v  0.5  0.5 -0.5 0.9 0.9 0.1
v -0.5  0.5 -0.5 0.1 0.9 0.1
v -0.5 -0.5  0.5 0.1 0.9 0.9
v  0.5 -0.5  0.5 0.9 0.1 0.9
v  0.5  0.5  0.5 0.9 0.9 0.9
v -0.5  0.5  0.5 0.2 0.2 0.2
f 1 2 3 4
f 5 6 7 8
f 1 5 8 4
f 2 6 7 3
f 4 8 7 3
f 1 5 6 2
";

/// Parses the built-in default cube into a [`Mesh`].
pub fn default_mesh() -> Result<Mesh, MeshError> {
    Mesh::from_obj(DEFAULT_MESH_OBJ)
}
