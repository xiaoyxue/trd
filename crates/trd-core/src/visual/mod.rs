//! The **visual model**: what a frame draws, independent of how it is rendered.
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
//! | [`scene`] | [`Scene`] and its assembly, [`Scene::from_draws`] |
//! | [`drawable`] | [`DrawableObject`] — the base interface for every primitive |
//! | [`draw`] | [`Draw`] + [`DrawSelection`], the *wire* instance record and its byte codec |
//! | [`draw_config`] | [`RenderMode`], [`FrameFit`], [`GridPlane`] — the per-drawable configuration a front-end selects |
//!
//! Assembly ([`Scene::from_draws`]) is the **one** place a wire [`Draw`] becomes
//! a [`DrawableObject`], which is what keeps every front-end rendering the same
//! scene from the same inputs (#180).

mod draw;
mod draw_config;
mod drawable;
mod scene;

#[cfg(test)]
pub(crate) use draw::DRAW_MODE_INHERIT;
pub use draw::{Draw, DrawSelection};
pub(crate) use draw_config::frame_fit_uv_scale;
pub use draw_config::{FrameFit, GridPlane, RenderMode};
pub use drawable::DrawableObject;
pub use scene::Scene;
