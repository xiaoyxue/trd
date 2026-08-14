//! The **wire** instance record and its `draw_mode` codec.
//!
//! [`Draw`] is what the protocol's `draw_mesh` / `draw_model` / `draw_mode`
//! columns decode into: which mesh to place, where, and what to draw there. It
//! is deliberately separate from [`DrawableObject`](super::DrawableObject), the
//! *renderer's* primitive — scene assembly
//! ([`build_scene`](super::build_scene)) is the one place that turns the former
//! into the latter.
//!
//! The codec lives here rather than beside [`RenderMode`](super::RenderMode)
//! (`draw_config.rs`) because it is wire knowledge: the byte values are
//! protocol, not configuration.

use super::RenderMode;

/// A single instance placement decoded from a frame's protocol draw list
/// (`draw_mesh` / `draw_model`): which mesh to draw (index into the leading mesh
/// table) and the per-instance model matrix (column-major), applied beneath that
/// mesh's base (preview) model. This is the *wire* representation; the renderer
/// composes it (plus core gizmos) into a [`Scene`](super::Scene) of
/// [`DrawableObject`](super::DrawableObject)s.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Draw {
    pub mesh_id: u32,
    pub model: [f32; 16],
    /// What this draw places: a mesh (optionally overriding the global render
    /// mode) or a grounding shadow. Decoded from the optional `draw_mode` column.
    pub selection: DrawSelection,
}

/// What a [`Draw`] selects — the answer to *"what is drawn here?"*, which is a
/// different question from [`RenderMode`]'s *"how is this mesh rasterized?"*.
///
/// `Shadow` used to be a `RenderMode` variant, but it is not a way of
/// rasterizing a mesh: it means *"do not draw this mesh at all; lay a blob decal
/// on its ground plane instead"*. As a mode it forced every consumer to special-
/// case it back out — six `continue`s across scene assembly, the batcher and the
/// picker — because none of them can act on a draw that has no mesh geometry.
/// As a selection it is resolved **once**, in [`build_scene`](super::build_scene),
/// and nothing downstream needs to know a shadow ever existed (#203).
///
/// The wire is unchanged: byte `3` still means shadow.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DrawSelection {
    /// Draw the mesh. `Some(mode)` overrides the front-end's global render mode
    /// for this instance; `None` inherits it — which lets one frame mix e.g. a
    /// textured mesh with a wireframe overlay quad.
    Mesh(Option<RenderMode>),
    /// Draw a **contact / blob grounding shadow** on the placed mesh's ground
    /// plane instead of the mesh itself (#110 follow-up). Becomes a
    /// [`Primitive::BlobShadow`](super::Primitive::BlobShadow); the
    /// draw's `mesh_id` is ignored, since the blob uses shared gizmo geometry.
    Shadow,
}

impl Default for DrawSelection {
    /// A mesh draw inheriting the front-end's global render mode — what an
    /// absent `draw_mode` column decodes to.
    fn default() -> Self {
        Self::INHERIT
    }
}

impl DrawSelection {
    /// A mesh draw inheriting the global render mode.
    pub const INHERIT: Self = Self::Mesh(None);

    /// This draw's render mode resolved against the front-end's `global` mode, or
    /// `None` when the draw is not a mesh at all.
    ///
    /// The one place the "override else inherit" rule is applied, so a caller
    /// cannot forget that a shadow has no render mode.
    pub fn mesh_mode(self, global: RenderMode) -> Option<RenderMode> {
        match self {
            DrawSelection::Mesh(mode) => Some(mode.unwrap_or(global)),
            DrawSelection::Shadow => None,
        }
    }

    /// Whether this draw places actual mesh geometry — false for a shadow, whose
    /// blob has no mesh to box, grid, or hit-test.
    pub fn is_mesh(self) -> bool {
        matches!(self, DrawSelection::Mesh(_))
    }
}

/// Wire byte meaning "inherit the renderer's global mode" in the optional
/// per-draw `draw_mode` (`List<UInt8>`) protocol column (see
/// [`DrawSelection::from_wire`]). A draw carrying this value defers to the `mode`
/// argument of [`build_scene`](super::build_scene), so a stream can override only
/// *some* draws (e.g. draw a wireframe overlay quad while every other draw
/// follows the front-end's global mode).
pub const DRAW_MODE_INHERIT: u8 = 255;

impl DrawSelection {
    /// Decodes a per-draw `draw_mode` wire byte: `0`→`Filled`, `1`→`Wireframe`,
    /// `2`→`Textured`, `3`→[`Shadow`](Self::Shadow), `4`→`Shaded`, and
    /// [`DRAW_MODE_INHERIT`]→[`INHERIT`](Self::INHERIT). Returns `None` for an
    /// unrecognized byte so callers can raise a decode error.
    ///
    /// **The byte values are protocol and never change**: `3` is still shadow and
    /// `4` is still what the producers and `viewer.ts` spell `"pbr"`; #203 moved
    /// `Shadow` out of `RenderMode` and renamed `Pbr` to `Shaded` in Rust only.
    pub fn from_wire(byte: u8) -> Option<Self> {
        match byte {
            0 => Some(Self::Mesh(Some(RenderMode::Filled))),
            1 => Some(Self::Mesh(Some(RenderMode::Wireframe))),
            2 => Some(Self::Mesh(Some(RenderMode::Textured))),
            3 => Some(Self::Shadow),
            4 => Some(Self::Mesh(Some(RenderMode::Shaded))),
            DRAW_MODE_INHERIT => Some(Self::INHERIT),
            _ => None,
        }
    }
}
