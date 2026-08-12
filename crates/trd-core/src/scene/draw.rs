//! The **wire** instance record and its render-mode codec.
//!
//! [`Draw`] is what the protocol's `draw_mesh` / `draw_model` / `draw_mode`
//! columns decode into: which mesh to place, where, and optionally how to draw
//! it. It is deliberately separate from [`DrawableObject`](super::DrawableObject),
//! the *renderer's* primitive — scene assembly
//! ([`build_scene`](super::build_scene)) is the one place that turns the former
//! into the latter.
//!
//! The mode codec lives here rather than beside
//! [`RenderMode`](super::RenderMode) (`draw_config.rs`) because it is wire
//! knowledge: the byte values are protocol, not configuration.

use super::RenderMode;

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
    /// `4`→`Shaded`, and [`DRAW_MODE_INHERIT`]→`None` (inherit the global mode).
    /// Returns `None` for an unrecognized byte so callers can raise a decode error.
    ///
    /// **The byte values are protocol and never change** — `4` still means what
    /// the producers and `viewer.ts` spell `"pbr"`; only the Rust variant was
    /// renamed to `Shaded` (#203).
    pub fn from_wire(byte: u8) -> Option<Option<RenderMode>> {
        match byte {
            0 => Some(Some(RenderMode::Filled)),
            1 => Some(Some(RenderMode::Wireframe)),
            2 => Some(Some(RenderMode::Textured)),
            3 => Some(Some(RenderMode::Shadow)),
            4 => Some(Some(RenderMode::Shaded)),
            DRAW_MODE_INHERIT => Some(None),
            _ => None,
        }
    }
}
